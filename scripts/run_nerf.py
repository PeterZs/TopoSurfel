import os

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
data_base_path='workdir/nerf_synthetic'
out_base_path='output_nerf_tmp'
out_name='test'
gpu_id=5

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5 --eval"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    if scene == "materials":
        common_args = "--quiet --num_cluster 12 --voxel_size 0.002 --max_depth 5.0 -w --eval"
    elif scene == "ship" or scene == "ficus" or scene == "drums":
        common_args = "--quiet --num_cluster 100 --voxel_size 0.002 --max_depth 5.0 --eval"
    else:
        common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --eval"
    cmd = f'OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
    print(cmd)
    os.system(cmd)
    