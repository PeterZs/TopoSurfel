import os
    
scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
data_base_path='../workdir/mipnerf360'
out_base_path='init_mip360_105'
out_name='test'
gpu_id=3

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = f"--quiet -r{factors[id]} --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --eval"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    if scene == 'garden':
        common_args = f"--data_device {data_devices[id]} --num_cluster 2 --quiet --use_depth_filter --voxel_size 0.008"
    elif scene == 'room':
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --quiet --use_depth_filter --voxel_size 0.012"
    else:
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --quiet --use_depth_filter --voxel_size 0.008"
    cmd = f'OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}' 
    print(cmd)
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
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