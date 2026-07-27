import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='../workdir/DTU'
out_base_path='init_dtu_105'
eval_path='../workdir/DTU'
out_name='test'
gpu_id=7

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.005 --max_depth 5.0"
    cmd = f'OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)
    
    cmd = f'cp {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply {data_base_path}/scan{scene}/mesh_init.ply'
    print(cmd)
    os.system(cmd)