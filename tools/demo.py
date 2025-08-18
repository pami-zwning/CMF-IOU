import os
import datetime
import argparse
import glob
from pathlib import Path
from test import repeat_eval_ckpt
import torch.distributed as dist
import torch.nn as nn
import tqdm
import time
import cv2
import pickle
from tensorboardX import SummaryWriter

# try:
import open3d
from visual_utils import open3d_vis_utils
OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import warnings
warnings.filterwarnings("ignore")

def calculate_parameters(model, only_grad = False):
    total_params = 0
    for param in model.parameters():
        if only_grad == False:
            total_params += param.numel() 
        else:
            total_params += param.numel() if param.requires_grad else 0
    return total_params

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="cfgs/models/kitti/VirConv-T.yaml", help='specify the config for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=20017, help='tcp port for distrbuted training')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--sample_nums', type=int, default=55, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    # for key, val in vars(args).items():
    #     logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.cuda()

    # load checkpoint if it is possible
    cur_epoch_id = int(args.ckpt.split('.')[-2].split('_')[-1]) #TODO

    assert args.ckpt is not None, 'the ckpt shounld be redefined'
    model.load_params_from_file(args.ckpt, to_cpu=dist, logger=logger)

    model.eval()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])#,find_unused_parameters=True
    # logger.info(model)
    
    eval_output_dir = output_dir / 'eval' / ('eval_with_train')
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = cur_epoch_id  # Only evaluate the last 10 epochs

    model = model.module if dist_train else model
    dist_test = dist_train

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))

    #cal the model size
    print('#'*10 + '\n' + f'model total params is :{calculate_parameters(model, only_grad=True)} \n'+'#'*10 )

    # start evaluation
    cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    
    epoch_id = cur_epoch_id
    result_dir = cur_result_dir
    save_to_file=args.save_to_file
    dataloader = test_loader

    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        if i>= args.sample_nums:
            break
        #begin = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model(batch_dict)
        
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )

        cur_img_file = '../data/kitti/training/image_2/' + batch_dict['frame_id'][0] + '.png'
        cur_image = cv2.imread(cur_img_file)
        calib = batch_dict['calib'][0]

        images_dir = final_output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        image_file_path = str(images_dir) + '/' + batch_dict['frame_id'][0]+'.png'

        velodyne_dir = final_output_dir / 'velodyne'
        velodyne_dir.mkdir(parents=True, exist_ok=True)
        velodyne_file_path = str(velodyne_dir) + '/' + batch_dict['frame_id'][0]

        gt_boxes = batch_dict['gt_boxes'][0][:,:7].cpu().numpy()
        ref_boxes = annos[0]['boxes_lidar']

        gt_boxes_2d = common_utils.corners_3d_to_2d(calib, gt_boxes)
        ref_boxes_2d = common_utils.corners_3d_to_2d(calib, ref_boxes)

        draw_image = common_utils.draw_boxes_2d(cur_image, gt_boxes_2d, color = 'r')
        # draw_image = common_utils.draw_boxes_2d(cur_image, ref_boxes_2d, color = 'g')

        cv2.imwrite(image_file_path, draw_image)

        open3d_vis_utils.save_scenes(
            points=batch_dict['raw_points'][:, 1:4], filename=velodyne_file_path, gt_boxes=gt_boxes ,ref_boxes=ref_boxes,
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    print('*'*10 + 'DEMO finished' + '*'*10)
    

if __name__ == '__main__':
    main()

