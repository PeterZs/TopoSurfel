#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from utils.mesh_guided_densify_utils import (
    knn_1nn,
    compute_face_coverage_mask,
    visible_uncovered_face_ids,
    build_orthonormal_frame_from_normals,
)

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        self.setup_functions()
        self.use_app = False
        self.view_dir_accumulation = None
        self.mesh_guidance_cached_normals = None  # Mesh-guidance normal matched to each Gaussian.
        self.mesh_guidance_cached_reliable = None  # Nearest-neighbor reliability mask for each Gaussian.
        self.mesh_guidance_face_centroids = None
        self.mesh_guidance_face_visible = None
        self.mesh_guidance_face_normals = None

    @torch.no_grad()
    def set_mesh_guidance(self, face_centroids, face_visible=None, face_normals=None):
        self.mesh_guidance_face_centroids = None if face_centroids is None else face_centroids.detach()
        if face_visible is not None:
            self.mesh_guidance_face_visible = face_visible.detach().bool()
        self.mesh_guidance_face_normals = None if face_normals is None else face_normals.detach()

    @torch.no_grad()
    def clear_mesh_guidance(self):
        self.mesh_guidance_face_centroids = None
        self.mesh_guidance_face_visible = None
        self.mesh_guidance_face_normals = None

    def decay_view_dir_stats(self, decay: float = 0.8):
        """EMA-style forgetting for view direction stats.

        decay in (0, 1]. Smaller -> faster forgetting (more like a short sliding window).
        """
        if self.view_dir_accumulation is not None:
            self.view_dir_accumulation.mul_(float(decay))

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        scales = torch.log(dist)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_mesh(
        self,
        mesh_path: str,
        spatial_lr_scale: float,
        flatten_ratio: float = 0.12,
        tangent_scale: float = 0.55,
        opacity_init: float = 0.10,
        min_scale_ratio: float = 1e-4,
        max_scale_ratio: float = 0.05,
    ):
        """Initialize one Gaussian per mesh face using face centroids and normals."""
        self.spatial_lr_scale = spatial_lr_scale

        ply = PlyData.read(mesh_path)
        if "face" not in ply:
            raise ValueError(f"Mesh file has no face element: {mesh_path}")

        verts = np.stack(
            [
                np.asarray(ply["vertex"]["x"], dtype=np.float32),
                np.asarray(ply["vertex"]["y"], dtype=np.float32),
                np.asarray(ply["vertex"]["z"], dtype=np.float32),
            ],
            axis=1,
        )
        faces_raw = np.asarray(ply["face"].data["vertex_indices"])
        faces = np.stack([np.asarray(f, dtype=np.int64) for f in faces_raw], axis=0)
        if faces.shape[1] != 3:
            raise ValueError(f"Only triangular faces are supported, got face size {faces.shape[1]}")

        has_color = all(c in ply["vertex"].data.dtype.names for c in ["red", "green", "blue"])
        if has_color:
            vcols = np.stack(
                [
                    np.asarray(ply["vertex"]["red"], dtype=np.float32),
                    np.asarray(ply["vertex"]["green"], dtype=np.float32),
                    np.asarray(ply["vertex"]["blue"], dtype=np.float32),
                ],
                axis=1,
            ) / 255.0
        else:
            vcols = np.full((verts.shape[0], 3), 0.5, dtype=np.float32)

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0

        e01 = v1 - v0
        e02 = v2 - v0
        n = np.cross(e01, e02)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        valid = n_norm[:, 0] > 1e-12
        if valid.sum() == 0:
            raise ValueError(f"All mesh faces are degenerate in {mesh_path}")

        n = n[valid] / np.clip(n_norm[valid], 1e-12, None)
        centroids = centroids[valid]
        faces_valid = faces[valid]

        t1 = e01[valid]
        t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
        fallback_mask = t1_norm[:, 0] <= 1e-12
        t1 = t1 / np.clip(t1_norm, 1e-12, None)
        if np.any(fallback_mask):
            t1[fallback_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        t2 = np.cross(n, t1)
        t2 = t2 / np.clip(np.linalg.norm(t2, axis=1, keepdims=True), 1e-12, None)

        # Re-orthogonalize tangent basis for numerical stability.
        t1 = np.cross(t2, n)
        t1 = t1 / np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12, None)

        p0 = v0[valid] - centroids
        p1 = v1[valid] - centroids
        p2 = v2[valid] - centroids

        u0 = np.abs(np.sum(p0 * t1, axis=1))
        u1 = np.abs(np.sum(p1 * t1, axis=1))
        u2 = np.abs(np.sum(p2 * t1, axis=1))
        v0p = np.abs(np.sum(p0 * t2, axis=1))
        v1p = np.abs(np.sum(p1 * t2, axis=1))
        v2p = np.abs(np.sum(p2 * t2, axis=1))

        span_u = np.maximum(np.maximum(u0, u1), u2)
        span_v = np.maximum(np.maximum(v0p, v1p), v2p)

        scene_scale = max(float(spatial_lr_scale), 1e-6)
        min_scale = scene_scale * float(min_scale_ratio)
        max_scale = scene_scale * float(max_scale_ratio)

        su = np.clip(float(tangent_scale) * span_u, min_scale, max_scale)
        sv = np.clip(float(tangent_scale) * span_v, min_scale, max_scale)
        sn = np.clip(float(flatten_ratio) * np.minimum(su, sv), min_scale * 0.5, max_scale)

        rot_mats = np.stack([t1, t2, n], axis=-1).astype(np.float32)  # [N, 3, 3]
        rotations = matrix_to_quaternion(torch.from_numpy(rot_mats).cuda())

        face_colors = (
            vcols[faces_valid[:, 0]] + vcols[faces_valid[:, 1]] + vcols[faces_valid[:, 2]]
        ) / 3.0

        fused_point_cloud = torch.tensor(centroids, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(face_colors, dtype=torch.float32, device="cuda"))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color

        scales_xyz = np.stack([su, sv, sn], axis=1).astype(np.float32)
        scales = torch.log(torch.tensor(scales_xyz, dtype=torch.float32, device="cuda"))

        opacity_init = float(np.clip(opacity_init, 1e-4, 0.999))
        opacities = inverse_sigmoid(opacity_init * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda"))

        print(f"Number of points from mesh initialisation : {fused_point_cloud.shape[0]}")

        knn_f = torch.randn((fused_point_cloud.shape[0], 6), dtype=torch.float32, device="cuda")
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._knn_f = nn.Parameter(torch.randn((xyz.shape[0], 6), dtype=torch.float32, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    @torch.no_grad()
    def append_gaussians_from_model(self, source_gaussians, mask, opacity_cull_threshold: float = 0.0):
        if mask is None:
            mask = torch.ones((source_gaussians.get_xyz.shape[0],), dtype=torch.bool, device=source_gaussians.get_xyz.device)
        else:
            mask = mask.to(device=source_gaussians.get_xyz.device, dtype=torch.bool)

        opacity_mask = source_gaussians.get_opacity.squeeze(-1) >= (10.0 * float(opacity_cull_threshold))
        mask = mask & opacity_mask

        if mask.numel() == 0 or not bool(mask.any()):
            return 0

        if source_gaussians._knn_f.numel() == 0 or source_gaussians._knn_f.shape[0] != source_gaussians.get_xyz.shape[0]:
            knn_width = self._knn_f.shape[1] if self._knn_f.numel() > 0 else 6
            source_knn_f = torch.randn((source_gaussians.get_xyz.shape[0], knn_width), dtype=torch.float32, device=source_gaussians.get_xyz.device)
        else:
            source_knn_f = source_gaussians._knn_f

        self.densification_postfix(
            source_gaussians._xyz[mask],
            source_knn_f[mask],
            source_gaussians._features_dc[mask],
            source_gaussians._features_rest[mask],
            source_gaussians._opacity[mask],
            source_gaussians._scaling[mask],
            source_gaussians._rotation[mask],
        )
        return int(mask.sum().item())

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]
        if self.view_dir_accumulation is not None:
            self.view_dir_accumulation = self.view_dir_accumulation[valid_points_mask]
        if self.mesh_guidance_cached_normals is not None:
            self.mesh_guidance_cached_normals = self.mesh_guidance_cached_normals[valid_points_mask]
        if self.mesh_guidance_cached_reliable is not None:
            self.mesh_guidance_cached_reliable = self.mesh_guidance_cached_reliable[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "knn_f": new_knn_f,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # Newly generated points have no view statistics yet; initialize them to zero.
        new_dirs = torch.zeros((new_xyz.shape[0], 3), device="cuda")
        if self.view_dir_accumulation is None:
            self.view_dir_accumulation = torch.zeros((self._xyz.shape[0] - new_xyz.shape[0], 3), device="cuda")
        self.view_dir_accumulation = torch.cat((self.view_dir_accumulation, new_dirs), dim=0)
        if self.mesh_guidance_cached_normals is not None:
            new_guidance_normals = torch.zeros((new_xyz.shape[0], 3), device="cuda")
            self.mesh_guidance_cached_normals = torch.cat((self.mesh_guidance_cached_normals, new_guidance_normals), dim=0)
        if self.mesh_guidance_cached_reliable is not None:
            new_guidance_reliable = torch.zeros((new_xyz.shape[0],), dtype=torch.bool, device="cuda")
            self.mesh_guidance_cached_reliable = torch.cat((self.mesh_guidance_cached_reliable, new_guidance_reliable), dim=0)

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_mesh_guided(self, grads, grads_abs, grad_threshold, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        """Densify from visible mesh faces that are not covered by existing Gaussians.

        The procedure only appends Gaussians: it identifies faces not selected as a
        nearest neighbor by any Gaussian, intersects them with visible faces, and
        creates a Gaussian at each selected face centroid. Its shortest axis aligns
        with the face normal and its color is inherited from the nearest Gaussian.
        """
        _ = (grads, grads_abs, grad_threshold, grad_abs_threshold, scene_extent, max_radii2D, N)
        if (
            self.mesh_guidance_face_centroids is None
            or self.mesh_guidance_face_visible is None
            or self.mesh_guidance_face_normals is None
        ):
            return

        face_centroids = self.mesh_guidance_face_centroids.to(device=self.get_xyz.device)
        face_visible = self.mesh_guidance_face_visible.to(device=self.get_xyz.device)
        face_normals = self.mesh_guidance_face_normals.to(device=self.get_xyz.device)
        if face_centroids.numel() == 0 or face_visible.numel() == 0 or face_normals.numel() == 0:
            return

        current_points = self.get_xyz
        if current_points.numel() == 0:
            return

        # Faces not covered by any Gaussian are candidate holes.
        covered_face_mask, gs_to_mesh_dist, gs_to_mesh_idx = compute_face_coverage_mask(face_centroids, current_points)
        candidate_face_ids = visible_uncovered_face_ids(face_visible, covered_face_mask)
        if candidate_face_ids.numel() == 0:
            return

        available_slots = int(self.max_all_points - current_points.shape[0])
        if available_slots <= 0:
            return

        # Limit candidates so the point cloud does not exceed its global capacity.
        candidate_centroids = face_centroids[candidate_face_ids]
        nn_dist, nn_idx = knn_1nn(candidate_centroids, current_points)

        # Keep candidates farther than scene_extent / 500 from existing points.
        distance_threshold = scene_extent / 500
        dist_mask = nn_dist > distance_threshold

        # Retain candidates satisfying the distance criterion.
        candidate_face_ids = candidate_face_ids[dist_mask]
        nn_dist = nn_dist[dist_mask]
        nn_idx = nn_idx[dist_mask]
        # Return if filtering removed all candidates.
        if candidate_face_ids.numel() == 0:
            return
        
        if candidate_face_ids.shape[0] > available_slots:
            keep_order = torch.argsort(nn_dist)[:available_slots]
            candidate_face_ids = candidate_face_ids[keep_order]
            nn_idx = nn_idx[keep_order]

        if candidate_face_ids.numel() == 0:
            return

        n_new = candidate_face_ids.shape[0]
        
        # Use face centroids as Gaussian centers and face normals as shortest-axis directions.
        selected_centroids = face_centroids[candidate_face_ids]
        selected_normals = face_normals[candidate_face_ids]
        selected_rotation = matrix_to_quaternion(build_orthonormal_frame_from_normals(selected_normals))

        # Use the nearest Gaussian color as the appearance prior instead of mesh color.
        nearest_dc = self._features_dc[nn_idx].squeeze(1)
        nearest_rgb = torch.clamp(SH2RGB(nearest_dc), 0.0, 1.0)
        new_features_dc = RGB2SH(nearest_rgb).unsqueeze(1)
        new_features_rest = torch.zeros(
            (candidate_face_ids.shape[0], self._features_rest.shape[1], self._features_rest.shape[2]),
            dtype=self._features_rest.dtype,
            device=self._features_rest.device,
        )
        new_opacity = self._opacity[nn_idx]
        # Keep new Gaussians thin along face normals by placing the smallest scale on local z.
        nearest_scaling = self.get_scaling[nn_idx]
        sorted_scaling, _ = torch.sort(nearest_scaling, dim=1)
        normal_scale = torch.clamp_min(sorted_scaling[:, 0] * 0.5, 1e-6)
        new_scaling = self.scaling_inverse_activation(
            torch.stack(
                [sorted_scaling[:, 2], sorted_scaling[:, 1], normal_scale],
                dim=1,
            )
        )
        new_knn_f = self._knn_f[nn_idx]

        self.densification_postfix(
            selected_centroids,
            new_knn_f,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            selected_rotation,
        )
        
        # Mark original Gaussians farther than scene_extent / 10 for pruning.
        prune_dist_threshold = scene_extent / 10
        prune_mask_original = gs_to_mesh_dist > prune_dist_threshold
        # Keep all newly added Gaussians.
        prune_mask_new = torch.zeros(n_new, dtype=torch.bool, device=prune_mask_original.device)
        # Concatenate masks for original and newly added Gaussians.
        final_prune_mask = torch.cat([prune_mask_original, prune_mask_new])
        #if final_prune_mask.any():
            #self.prune_points(final_prune_mask)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size, mesh_guide=False):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)
        if mesh_guide:
            self.densify_mesh_guided(grads, grads_abs, max_grad, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts

    def _knn_face_centroids(self, points):
        if self.mesh_guidance_face_centroids is None or self.mesh_guidance_face_centroids.shape[0] == 0:
            return None, None

        return knn_1nn(points, self.mesh_guidance_face_centroids)

    @torch.no_grad()
    def update_mesh_normal_cache(self, distance_ratio=3.0, normal_cos_thresh=0.2):
        """Refresh per-Gaussian mesh-guidance cache and reliability mask."""
        # Find the nearest initial-mesh face centroid for every Gaussian center.
        # Reliability requires both a sufficiently close in-plane distance and
        # abs(dot(raw_normal, guidance_normal)) >= normal_cos_thresh.
        if self.mesh_guidance_face_centroids is None or self.mesh_guidance_face_normals is None or self._xyz.numel() == 0:
            self.mesh_guidance_cached_normals = None
            self.mesh_guidance_cached_reliable = None
            return

        points = self._xyz.detach()
        raw_normal = self.get_smallest_axis().detach()
        nn_dist, nn_idx = self._knn_face_centroids(points)
        if nn_idx is None:
            self.mesh_guidance_cached_normals = None
            self.mesh_guidance_cached_reliable = None
            return

        guidance_normal = self.mesh_guidance_face_normals[nn_idx]
        guidance_normal = torch.nn.functional.normalize(guidance_normal, dim=1)

        sorted_scales, _ = torch.sort(self.get_scaling.detach(), dim=1)
        in_plane_scale = torch.sqrt(torch.clamp_min(sorted_scales[:, 1] * sorted_scales[:, 2], 1e-12))
        dist_thresh = torch.clamp(in_plane_scale * float(distance_ratio), min=1e-5)
        cos_abs = torch.abs(torch.sum(raw_normal * guidance_normal, dim=1))
        reliable = (nn_dist <= dist_thresh) & (cos_abs >= float(normal_cos_thresh))

        self.mesh_guidance_cached_normals = guidance_normal
        self.mesh_guidance_cached_reliable = reliable
    
    
    # Update viewing-direction statistics during training.
    def add_view_dir_stat(self, cam_center, visibility_filter):
        """
        Call once per iteration.
        cam_center: center of the current camera, shape [3]
        visibility_filter: visible Gaussians in the current view, shape [N] (bool)
        """
        if self.view_dir_accumulation is None:
            # Handle models saved before view statistics were introduced.
            self.view_dir_accumulation = torch.zeros((self._xyz.shape[0], 3), device="cuda")
        
        # visible_xyz = self._xyz[visibility_filter]
        # dirs = cam_center - visible_xyz
        # dirs = torch.nn.functional.normalize(dirs, dim=1)
        
        # Update only visible Gaussians.
        visible_indices = torch.where(visibility_filter)[0]
        if len(visible_indices) == 0: return

        visible_points = self._xyz[visible_indices].detach()
        dirs = cam_center - visible_points
        # Normalize directions before accumulation.
        dirs = torch.nn.functional.normalize(dirs, dim=1)
        
        # Accumulate viewing directions.
        self.view_dir_accumulation[visible_indices] += dirs
    
    def get_oriented_normal(self):
        """
        Return normals oriented using mesh guidance or viewing-direction statistics.
        """
        raw_normal = self.get_smallest_axis()
        n = raw_normal.shape[0]
        sign = torch.ones((n, 1), dtype=raw_normal.dtype, device=raw_normal.device)

        guidance_ready = (
            self.mesh_guidance_cached_normals is not None
            and self.mesh_guidance_cached_reliable is not None
            and self.mesh_guidance_cached_normals.shape[0] == n
            and self.mesh_guidance_cached_reliable.shape[0] == n
        )
        # guidance_ready = False  # Disable mesh-guided normal orientation for ablations.

        if guidance_ready:
            reliable = self.mesh_guidance_cached_reliable
            if bool(torch.any(reliable)):
                dot_mesh = torch.sum(
                    raw_normal[reliable] * self.mesh_guidance_cached_normals[reliable],
                    dim=1,
                    keepdim=True,
                )
                sign_mesh = torch.sign(dot_mesh)
                sign_mesh[sign_mesh == 0] = 1.0
                sign[reliable] = sign_mesh
            need_view = ~reliable
        else:
            need_view = torch.ones((n,), dtype=torch.bool, device=raw_normal.device)

        if self.view_dir_accumulation is not None and bool(torch.any(need_view)):
            dot_view = torch.sum(
                raw_normal[need_view] * self.view_dir_accumulation[need_view],
                dim=1,
                keepdim=True,
            )
            sign_view = torch.sign(dot_view)
            sign_view[sign_view == 0] = 1.0
            sign[need_view] = sign_view

        return raw_normal * sign
    
    def extract_oriented_pointcloud(self, opacity_threshold=0.0, normalize_weight=True, visibility_mask=None, log_weight=False, use_cut_points=False):
        """
        Export an oriented point cloud from Gaussian centers.
        - points: Gaussian centers
        - normals: normals oriented by viewing-direction statistics
        - weights: opacity * area (product of the two largest axes)

        Returns:
            points [M, 3], normals [M, 3], weights [M], mask [N]
        """
        points = self.get_xyz
        normals = self.get_oriented_normal()

        opacity = self.get_opacity.squeeze(-1)
        scales = self.get_scaling
        sorted_scales, _ = torch.sort(scales, dim=-1)
        area = sorted_scales[:, 1] * sorted_scales[:, 2]
        weights = opacity

        mask = opacity > opacity_threshold

        if visibility_mask is not None:
            mask = mask & visibility_mask.bool()

        points = points[mask]
        normals = normals[mask]
        weights = weights[mask]

        '''if log_weight and weights.numel() > 0:
            weights = torch.log1p(weights)'''
        if normalize_weight and weights.numel() > 0:
            weights = weights / (weights.max() + 1e-8)

        if use_cut_points and points.numel() > 0:
            rotation_matrices = self.get_rotation_matrix()[mask]
            scales = self.get_scaling[mask]
            sorted_scales, sorted_idx = torch.sort(scales, dim=-1)

            def gather_axis(mats, axis_idx):
                axis_idx = axis_idx[..., None, None].expand(-1, 3, 1)
                return mats.gather(2, axis_idx).squeeze(2)

            tangent_axis_0 = gather_axis(rotation_matrices, sorted_idx[:, 1])
            tangent_axis_1 = gather_axis(rotation_matrices, sorted_idx[:, 2])

            offset_0 = tangent_axis_0 * (sorted_scales[:, 1:2] * 0.5)
            offset_1 = tangent_axis_1 * (sorted_scales[:, 2:3] * 0.5)

            cut_points = torch.stack(
                [
                    points + offset_0 + offset_1,
                    points + offset_0 - offset_1,
                    points - offset_0 + offset_1,
                    points - offset_0 - offset_1,
                ],
                dim=1,
            ).reshape(-1, 3)
            cut_normals = normals.repeat_interleave(4, dim=0)
            cut_weights = (weights * 0.5).repeat_interleave(4, dim=0)

            points = torch.cat([points, cut_points], dim=0)
            normals = torch.cat([normals, cut_normals], dim=0)
            weights = torch.cat([weights, cut_weights], dim=0)

        return points, normals, weights, mask

    def save_oriented_pointcloud_ply(
        self,
        path,
        opacity_threshold=0.0,
        normalize_weight=True,
        visibility_mask=None,
        log_weight=False,
        use_cut_points=False,
        color_mode: int = 1,
    ):
        """
        Save an oriented point cloud to PLY:
        - x, y, z
        - nx, ny, nz
        - weight (opacity * area)

        color_mode:
            1 -> color by weight (default behavior)
            2 -> color by normal direction, mapping [-1, 1] to [0, 255]
        """
        mkdir_p(os.path.dirname(path))

        points, normals, weights, _ = self.extract_oriented_pointcloud(
            opacity_threshold=opacity_threshold,
            normalize_weight=normalize_weight,
            visibility_mask=visibility_mask,
            log_weight=log_weight,
            use_cut_points=use_cut_points,
        )

        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()

        if color_mode == 1:
            w = weights_np.astype(np.float64)
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w_norm = (w - w_min) / (w_max - w_min)
            else:
                w_norm = np.zeros_like(w)

            red = (w_norm * 255).astype(np.uint8)
            green = np.zeros_like(red, dtype=np.uint8)
            blue = np.full_like(red, 255, dtype=np.uint8)
        elif color_mode == 2:
            normals_vis = np.clip((normals_np.astype(np.float64) + 1.0) * 0.5, 0.0, 1.0)
            red = (normals_vis[:, 0] * 255).astype(np.uint8)
            green = (normals_vis[:, 1] * 255).astype(np.uint8)
            blue = (normals_vis[:, 2] * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported color_mode: {color_mode}. Expected 1 or 2.")

        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('weight', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        elements = np.empty(points_np.shape[0], dtype=dtype_full)
        elements['x'] = points_np[:, 0]
        elements['y'] = points_np[:, 1]
        elements['z'] = points_np[:, 2]
        elements['nx'] = normals_np[:, 0]
        elements['ny'] = normals_np[:, 1]
        elements['nz'] = normals_np[:, 2]
        elements['weight'] = weights_np
        elements['red'] = red
        elements['green'] = green
        elements['blue'] = blue

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"DEBUG: Oriented point cloud saved to {path} (N={points_np.shape[0]})")
