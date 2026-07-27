import os
    
scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
data_base_path='workdir/mipnerf360'
out_base_path='output_mip360'
out_name='test'
gpu_id=4

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = f"--quiet -r{factors[id]} --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --eval"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    if scene == "garden":
        common_args = f"--data_device {data_devices[id]} --num_cluster 2 --quiet --use_depth_filter"
    else: 
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --quiet --use_depth_filter"
    cmd = f'OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}' 
    print(cmd)
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
    print(cmd)
    os.system(cmd)
