cfg_name=voxel_rcnn

## kitti mamba
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=29964 train.py  --tcp_port 29964  --launcher pytorch  \
--cfg_file ./cfgs/models/nuscenes/$cfg_name.yaml \
--extra_tag $cfg_name \
--batch_size 2 --epochs 20 --max_ckpt_save_num 5 --workers 0 --sync_bn