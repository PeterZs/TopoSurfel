import os

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
data_base_path='../workdir/TNT'
out_base_path='init_tnt105'
out_name='test'
gpu_id=6

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)
    
    common_args = f"--quiet -r2 --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)
    
    if scene == 'Barn':
        common_args = f"--data_device {data_devices[id]} --cluster_keep_range 2 --use_depth_filter --voxel_size 0.008"
    elif scene == 'Caterpillar':
        common_args = f"--data_device {data_devices[id]} --cluster_keep_range 3 --use_depth_filter --voxel_size 0.008"
    else:
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter --voxel_size 0.008"
    cmd = f'OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{scene}/{out_name}/mesh'
    print(cmd)
    os.system(cmd)
    
    
    cmd = f'cp {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply {data_base_path}/{scene}/mesh_init.ply'
    print(cmd)
    os.system(cmd)
    pretrained_dir = f"{data_base_path}/{scene}/PGSR_pretrained"
    cmd = f"mkdir -p {pretrained_dir}"
    print(cmd); os.system(cmd)
    cmd = f"cp {out_base_path}/{scene}/{out_name}/point_cloud/iteration_15000/point_cloud.ply {pretrained_dir}/point_cloud.ply"
    print(cmd); os.system(cmd)
    cmd = f"cp {out_base_path}/{scene}/{out_name}/cfg_args {pretrained_dir}/cfg_args"
    print(cmd); os.system(cmd)