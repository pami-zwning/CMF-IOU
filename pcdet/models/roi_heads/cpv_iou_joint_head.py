import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.spconv_utils import spconv
from torch.autograd import Variable
from ...utils import box_coder_utils, common_utils, loss_utils, box_utils
import torch.nn.functional as F
import numpy as np
from functools import partial
import pickle
import copy
import time
import cv2

from pcdet.datasets.augmentor.X_transform import X_TRANS
from .refinement_head_utils import CPConvs, IMG_CNN, CrossAttention

def gen_sample_grid(rois, grid_size=7, grid_offsets=(0, 0), spatial_scale=1.):
    faked_features = rois.new_ones((grid_size, grid_size))
    N = rois.shape[0]
    dense_idx = faked_features.nonzero()  # (N, 2) [x_idx, y_idx]
    dense_idx = dense_idx.repeat(N, 1, 1).float()  # (B, 7 * 7, 2)

    local_roi_size = rois.view(N, -1)[:, 3:5]
    local_roi_grid_points = (dense_idx ) / (grid_size-1) * local_roi_size.unsqueeze(dim=1) \
                      - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 7 * 7, 2)

    ones = torch.ones_like(local_roi_grid_points[..., 0:1])
    local_roi_grid_points = torch.cat([local_roi_grid_points, ones], -1)

    global_roi_grid_points = common_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)

    x = global_roi_grid_points[..., 0:1]
    y = global_roi_grid_points[..., 1:2]

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(grid_size**2, -1), y.view(grid_size**2, -1)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W        C,K,1,2   C,K,1,1

    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)# 49,K,1,1
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / W)  # normalize to between  0 and 1

    samples[:, :, :, 1] = (samples[:, :, :, 1] / H)  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1  # 49,K,1,2

    #B,C,H,W
    #B,H,W,2
    #B,C,H,W

    return torch.nn.functional.grid_sample(image, samples, align_corners=False)
   
class CPVIOUHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range=None, voxel_size=None, num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.pool_cfg_mm = model_cfg.ROI_GRID_POOL_MM
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        LAYER_cfg_mm = self.pool_cfg_mm.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.rot_num = model_cfg.ROT_NUM
        self.iou_generate = model_cfg.IOU_GENERATE
        self.grid_aware = model_cfg.ROI_GRID_POOL.get('ENABLE', True)
        self.grid_aware_mm = model_cfg.ROI_GRID_POOL_MM.get('ENABLE', True)
        self.point_aware = model_cfg.ROI_POINT_POOL.get('ENABLE', False)
        self.point_aware_mm = model_cfg.ROI_POINT_POOL_MM.get('ENABLE',False)

        self.x_trans_train = X_TRANS()

        if self.grid_aware:
            # RoI Grid Pool
            c_out = 0
            self.roi_grid_pool_layers = nn.ModuleList()
            for src_name in self.pool_cfg.FEATURES_SOURCE:
                mlps = LAYER_cfg[src_name].MLPS
                for k in range(len(mlps)):
                    mlps[k] = [input_channels[src_name]] + mlps[k]
                pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                    query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                    nsamples=LAYER_cfg[src_name].NSAMPLE,
                    radii=LAYER_cfg[src_name].POOL_RADIUS,
                    mlps=mlps,
                    pool_method=LAYER_cfg[src_name].POOL_METHOD,
                )

                self.roi_grid_pool_layers.append(pool_layer)

                c_out += sum([x[-1] for x in mlps])

        if self.grid_aware_mm:
            # RoI MM Grid Pool
            c_out_mm = 0
            self.roi_grid_pool_layers_mm = nn.ModuleList()
            feat = self.pool_cfg_mm.get('FEAT_NUM', 1)
            for src_name in self.pool_cfg_mm.FEATURES_SOURCE:
                mlps = LAYER_cfg_mm[src_name].MLPS
                for k in range(len(mlps)):
                    mlps[k] = [input_channels[src_name]*feat] + mlps[k]
                pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                    query_ranges=LAYER_cfg_mm[src_name].QUERY_RANGES,
                    nsamples=LAYER_cfg_mm[src_name].NSAMPLE,
                    radii=LAYER_cfg_mm[src_name].POOL_RADIUS,
                    mlps=mlps,
                    pool_method=LAYER_cfg_mm[src_name].POOL_METHOD,
                )

                self.roi_grid_pool_layers_mm.append(pool_layer)

                c_out_mm += sum([x[-1] for x in mlps])

        # RoI Point Pool
        self.coords_3x3        = torch.tensor([ [0, 0], 
        [-1,  0], [-1,  1], 
        [ 0,  1], [ 1,  1], 
        [ 1,  0], [ 1, -1], 
        [ 0, -1], [-1, -1]]).long()
        self.coords_5x5_dilate = torch.tensor([ [0, 0], 
        [-2,  0], [-2,  2], 
        [ 0,  2], [ 2,  2], 
        [ 2,  0], [ 2, -2], 
        [ 0, -2], [-2, -2]]).long()
        self.coords_9x9_dilate = torch.tensor([ [0, 0], 
        [-2,  0], [-2,  2], 
        [ 0,  2], [ 2,  2], 
        [ 2,  0], [ 2, -2], 
        [ 0, -2], [-2, -2],
        [-4, -2], [-4,  0], [-4,  2], [-4,  4],
        [-2,  4], [ 0,  4], [ 2,  4], [ 4,  4],
        [ 4,  2], [ 4,  0], [ 4, -2], [ 4, -4],
        [ 2, -4], [ 0, -4], [-2, -4], [-4, -4]]).long()
        self.pointnet_kernel_size = {'coords_3x3':3, 'coords_5x5_dilate':5, 'coords_9x9_dilate':9}

        shared_pre_channel = 0

        if self.grid_aware:
            GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layer_og = nn.Sequential(*shared_fc_list)
            shared_pre_channel += self.model_cfg.SHARED_FC[-1] * 2
        
        if self.grid_aware_mm:
            GRID_SIZE = self.model_cfg.ROI_GRID_POOL_MM.GRID_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_mm
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layer_pg = nn.Sequential(*shared_fc_list)
            shared_pre_channel += self.model_cfg.SHARED_FC[-1] * 2
            
        if self.point_aware: # RoI point for raw
            self.cpconvs_layer = CPConvs()
            # RoI Voxel Pool
            self.roiaware_pool3d_layer = roiaware_pool3d_utils.PostPointNet(
                num_points = self.model_cfg.ROI_POINT_POOL.NUM_POINTS,
                output_dim=self.model_cfg.ROI_POINT_POOL.OUTPUT_DIM,
                mid_dim = self.model_cfg.ROI_POINT_POOL.CHANNEL_MID
            )
            block = self.post_act_block
            c0 = self.model_cfg.ROI_POINT_POOL.NUM_FEATURES_RAW # 90
            c1 = self.model_cfg.ROI_POINT_POOL.NUM_FEATURES # 128
            shared_pre_channel += self.model_cfg.SHARED_FC[-1] * 2

        if self.point_aware_mm: # RoI point for mm
            self.cpconvs_layer_mm = CPConvs()
            # RoI Voxel Pool
            # self.roiaware_pool3d_layer = roiaware_pool3d_utils.PostPointNet(
            #     num_points = self.model_cfg.ROI_POINT_POOL_MM.NUM_POINTS,
            #     output_dim=self.model_cfg.ROI_POINT_POOL_MM.OUTPUT_DIM,
            #     mid_dim = self.model_cfg.ROI_POINT_POOL_MM.CHANNEL_MID
            # )
            self.roiaware_pool3d_layer_mm = roiaware_pool3d_utils.RoIAwarePool3d(
                out_size=self.model_cfg.ROI_POINT_POOL_MM.POOL_SIZE,
                max_pts_each_voxel=self.model_cfg.ROI_POINT_POOL_MM.MAX_POINTS_PER_VOXEL
            )
            block = self.post_act_block
            c0 = self.model_cfg.ROI_POINT_POOL_MM.NUM_FEATURES_RAW # 90
            c1 = self.model_cfg.ROI_POINT_POOL_MM.NUM_FEATURES # 128

            GRID_SIZE = self.model_cfg.ROI_POINT_POOL_MM.POOL_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c0
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k]*2, bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]*2),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k] * 2

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.fc_pseudo_pp = nn.Sequential(*shared_fc_list)

            shared_pre_channel += self.model_cfg.SHARED_FC[-1] * 2

        if self.iou_generate:
            self.shared_sampling_fc_layers = nn.ModuleList()
            for i in range(self.rot_num):
                GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
                pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
                shared_fc_list = []
                for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                    shared_fc_list.extend([
                        nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                        nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                        nn.ReLU(inplace=True)
                    ])
                    pre_channel = self.model_cfg.SHARED_FC[k]

                    if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                        shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
                self.shared_sampling_fc_layers.append(nn.Sequential(*shared_fc_list))
                break


            self.shared_sampling_fc_layers_mm = nn.ModuleList()

            for i in range(self.rot_num):
                GRID_SIZE = self.model_cfg.ROI_GRID_POOL_MM.GRID_SIZE
                pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_mm
                shared_fc_list = []
                for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                    shared_fc_list.extend([
                        nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                        nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                        nn.ReLU(inplace=True)
                    ])
                    pre_channel = self.model_cfg.SHARED_FC[k]

                    if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                        shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
                self.shared_sampling_fc_layers_mm.append(nn.Sequential(*shared_fc_list))
                break

        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = shared_pre_channel
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers.append(cls_fc_layers)

            pre_channel = shared_pre_channel
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers.append(reg_fc_layers)
        
        if self.iou_generate:
            self.sampling_iou_layers = nn.ModuleList()
            self.sampling_reg_layers = nn.ModuleList()

            for i in range(self.rot_num):
                iou_fc_list = []
                iou_pre_channel_raw = self.model_cfg.SHARED_FC[-1] * 2 
                for k in range(0, self.model_cfg.IOU_FC.__len__()):
                    iou_fc_list.extend([
                        nn.Linear(iou_pre_channel_raw, self.model_cfg.IOU_FC[k], bias=False),
                        nn.BatchNorm1d(self.model_cfg.IOU_FC[k]),
                        nn.ReLU()
                    ])
                    iou_pre_channel_raw = self.model_cfg.IOU_FC[k]

                    if k != self.model_cfg.IOU_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                        iou_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
                iou_fc_list.append(nn.Linear(iou_pre_channel_raw, self.num_class, bias=True))
                iou_fc_layers = nn.Sequential(*iou_fc_list)
                self.sampling_iou_layers.append(iou_fc_layers)

                reg_fc_list = []
                reg_pre_channel_raw = self.model_cfg.SHARED_FC[-1] * 2 
                for k in range(0, self.model_cfg.IOU_FC.__len__()):
                    reg_fc_list.extend([
                        nn.Linear(reg_pre_channel_raw, self.model_cfg.IOU_FC[k], bias=False),
                        nn.BatchNorm1d(self.model_cfg.IOU_FC[k]),
                        nn.ReLU()
                    ])
                    reg_pre_channel_raw = self.model_cfg.IOU_FC[k]

                    if k != self.model_cfg.IOU_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                        reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
                reg_fc_list.append(nn.Linear(reg_pre_channel_raw, self.box_coder.code_size * self.num_class, bias=True))
                reg_fc_layers = nn.Sequential(*reg_fc_list)
                self.sampling_reg_layers.append(reg_fc_layers)

                break 

        if self.grid_aware:

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]
                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_list.append(nn.Sigmoid())
            self.cls_fc_og = nn.Sequential(*cls_fc_list)


            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]
                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            self.reg_fc_og = nn.Sequential(*reg_fc_list)

        if self.grid_aware_mm:

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_list.append(nn.Sigmoid())
            self.cls_fc_pg = nn.Sequential(*cls_fc_list)


            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]
                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            self.reg_fc_pg = nn.Sequential(*reg_fc_list)

        if self.point_aware:

            cls_fc_list_origin = []
            cls_pre_channel_origin = self.model_cfg.SHARED_FC[-1] * 2
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list_origin.extend([
                    nn.Linear(cls_pre_channel_origin, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                cls_pre_channel_origin = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list_origin.append(nn.Dropout(self.model_cfg.DP_RATIO))
            cls_fc_list_origin.append(nn.Linear(cls_pre_channel_origin, self.num_class, bias=True))
            cls_fc_list.append(nn.Sigmoid())
            self.cls_fc_op = nn.Sequential(*cls_fc_list_origin)

            reg_fc_list_origin = []
            reg_pre_channel_origin = self.model_cfg.SHARED_FC[-1] * 2
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list_origin.extend([
                    nn.Linear(reg_pre_channel_origin, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                reg_pre_channel_origin = self.model_cfg.REG_FC[k]
                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list_origin.append(nn.Dropout(self.model_cfg.DP_RATIO))
            reg_fc_list_origin.append(nn.Linear(reg_pre_channel_origin, self.box_coder.code_size * self.num_class, bias=True))
            self.reg_fc_op = nn.Sequential(*reg_fc_list_origin)

        if self.point_aware_mm:

            cls_fc_list_pseudo = []
            cls_pre_channel_pseudo = self.model_cfg.SHARED_FC[-1] * 2
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list_pseudo.extend([
                    nn.Linear(cls_pre_channel_pseudo, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                cls_pre_channel_pseudo = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list_pseudo.append(nn.Dropout(self.model_cfg.DP_RATIO))
            cls_fc_list_pseudo.append(nn.Linear(cls_pre_channel_pseudo, self.num_class, bias=True))
            cls_fc_list.append(nn.Sigmoid())
            self.cls_fc_pp = nn.Sequential(*cls_fc_list_pseudo)

            reg_fc_list_pseudo = []
            reg_pre_channel_pseudo = self.model_cfg.SHARED_FC[-1] * 2
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list_pseudo.extend([
                    nn.Linear(reg_pre_channel_pseudo, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                reg_pre_channel_pseudo = self.model_cfg.REG_FC[k]
                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list_pseudo.append(nn.Dropout(self.model_cfg.DP_RATIO))
            reg_fc_list_pseudo.append(nn.Linear(reg_pre_channel_pseudo, self.box_coder.code_size * self.num_class, bias=True))
            self.reg_fc_pp = nn.Sequential(*reg_fc_list_pseudo)

        if self.model_cfg.get('PART', False):
            self.grid_offsets = self.model_cfg.PART.GRID_OFFSETS
            self.featmap_stride = self.model_cfg.PART.FEATMAP_STRIDE
            part_inchannel = self.model_cfg.PART.IN_CHANNEL
            self.num_parts = self.model_cfg.PART.SIZE ** 2

            self.conv_part = nn.Sequential(
                nn.Conv2d(part_inchannel, part_inchannel, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(part_inchannel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(part_inchannel, self.num_parts, 1, 1, padding=0, bias=False),
            )
            self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets,
                                   spatial_scale=1 / self.featmap_stride)

        if self.model_cfg.get('CROSS_ATTN', False):
            if self.grid_aware:
                self.cross_attention_layer_og = CrossAttention(self.model_cfg.SHARED_FC[-1])
            if self.grid_aware_mm:
                self.cross_attention_layer_pg = CrossAttention(self.model_cfg.SHARED_FC[-1])
            # if self.point_aware:
            #     self.cross_attention_layer_op = CrossAttention(self.model_cfg.SHARED_FC[-1])
            # if self.point_aware_mm:
            #     self.cross_attention_layer_pp = CrossAttention(self.model_cfg.SHARED_FC[-1])

        self.init_weights()
        self.ious = {0: [], 1: [], 2: [], 3: []}

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.cls_layers, self.reg_layers]:
            for trans_module in module_list:
                for m in trans_module.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        if self.grid_aware:
            for module_list in [ self.shared_fc_layer_og, self.cls_fc_og, self.reg_fc_og]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        if self.grid_aware_mm:
            for module_list in [ self.shared_fc_layer_pg, self.cls_fc_pg, self.reg_fc_pg]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        if self.point_aware:
            for module_list in [ self.cls_fc_op, self.reg_fc_op]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        if self.point_aware_mm:
            for module_list in [ self.cls_fc_pp, self.reg_fc_pp]:
                for m in module_list.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes)
                out = bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1, 1)
                confi.append(x)

        confi = torch.cat(confi)

        return confi

    def roi_part_pool(self, batch_dict, parts_feat, enable_sampling=False):
        if enable_sampling == False:
            rois = batch_dict['rois_score'].clone()
        else:
            rois = batch_dict['sampling_rois'].clone()
        confi_preds = self.obtain_conf_preds(parts_feat, rois)

        return confi_preds

    def roi_grid_pool(self, batch_dict, i, enable_sampling=False):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        if i==0:
            rot_num_id = ''
        else:
            rot_num_id = str(i)

        if enable_sampling == False:
            rois = batch_dict['rois'].clone()
        else:
            rois = batch_dict['sampling_rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                j=i
                while 'multi_scale_3d_features'+rot_num_id not in batch_dict:
                    j-=1
                    rot_num_id = str(j)

                cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]

                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features
    
    def roi_grid_pool_mm(self, batch_dict, i, enable_sampling=False):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        if i==0:
            rot_num_id = ''
        else:
            rot_num_id = str(i)

        if enable_sampling == False:
            rois = batch_dict['rois'].clone()
        else:
            rois = batch_dict['sampling_rois'].clone()
        #rois[:, 3:5] = rois[:, 3:5]*0.5

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg_mm.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg_mm.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers_mm[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    if 'multi_scale_3d_features_mm'+rot_num_id in batch_dict:
                        cur_sp_tensors = batch_dict['multi_scale_3d_features_mm'+rot_num_id][src_name]
                    else:
                        cur_sp_tensors = batch_dict['multi_scale_3d_features' + rot_num_id][src_name]
                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg_mm.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points


    def roi_x_trans(self, rois, trans_i, transform_param):
        while trans_i>=len(transform_param[0]):
            trans_i-=1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_trans_param = bt_transform_param[trans_i-1]
            current_trans_param = bt_transform_param[trans_i]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': previous_trans_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': current_trans_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)
    
    def roi_x_reverse_trans(self, rois, trans_i, transform_param):
        while trans_i<0:
            trans_i += 1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_trans_param = bt_transform_param[trans_i+1]
            current_trans_param = bt_transform_param[trans_i]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': previous_trans_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': current_trans_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def roi_score_trans(self, rois, trans_i, transform_param):
        while trans_i>=len(transform_param[0]):
            trans_i-=1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_trans_param = bt_transform_param[0]
            current_trans_param = bt_transform_param[trans_i]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': current_trans_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': previous_trans_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def pred_x_trans(self, preds, trans_i, transform_param):
        while trans_i>=len(transform_param[0]):
            trans_i-=1

        batch_size = len(preds)
        preds = preds.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = preds[bt_i]
            bt_transform_param = transform_param[bt_i]
            current_trans_param = bt_transform_param[trans_i]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': current_trans_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m
    
    def roiaware_pool(self, batch_dict, i, points_roi_idx, points_tag= 'origin'):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """

        batch_size     = batch_dict['batch_size']
        rois           = batch_dict['rois']

        if points_tag == 'pseudo':
            batch_idx        = batch_dict['points_pseudo_features'][:, 0]
            point_coords   = batch_dict['points_pseudo_features'][:, 1:4]
            point_features = batch_dict['points_pseudo_features'][:, 4:] 

        elif points_tag == 'origin':
            batch_idx        = batch_dict['points_origin_features'][:, 0]
            point_coords   = batch_dict['points_origin_features'][:, 1:4]
            point_features = batch_dict['points_origin_features'][:, 4:] 

        pooled_points_list = []

        for bs_idx in range(batch_size):
            bs_mask = (batch_idx == bs_idx)
            if bs_mask.sum() == 0:
                cur_point_coords = point_coords.new_zeros((1,point_coords.shape[-1]))
                cur_pseudo_features = point_features.new_zeros((1,point_features.shape[-1]))
            else:
                cur_point_coords = point_coords[bs_mask]
                cur_pseudo_features = point_features[bs_mask]

            cur_roi = rois[bs_idx][:, 0:7].contiguous()  # (N, 7)

            cur_points_roi_idx = points_roi_idx[bs_mask]

            if points_tag == 'origin':
                pooled_points = self.roiaware_pool3d_layer.forward(
                    cur_roi.shape[0], cur_points_roi_idx, cur_pseudo_features, cur_point_coords
                )  # (N, C)

            elif points_tag == 'pseudo':
                pooled_points = self.roiaware_pool3d_layer_mm.forward(
                    cur_roi, cur_point_coords, cur_pseudo_features, pool_method=self.model_cfg.ROI_POINT_POOL_MM.POOL_METHOD
                )  # (N, out_x, out_y, out_z, C)

            pooled_points_list.append(pooled_points)
        pooled_points_feat = torch.cat(pooled_points_list, dim=0)  # (B * N, out_x, out_y, out_z, C) / (B*N, C)

        return  pooled_points_feat

    @staticmethod
    def fake_sparse_idx(sparse_idx, batch_size_rcnn):
        print('Warning: Sparse_Idx_Shape(%s) \r' % (str(sparse_idx.shape)), end='', flush=True)
        # at most one sample is non-empty, then fake the first voxels of each sample(BN needs at least
        # two values each channel) as non-empty for the below calculation
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def roicrop3d_gpu(self, batch_dict, pool_extra_width, i, point_tag = 'pseudo'):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        if i==0:
            idx_str = ''
        else:
            idx_str = str(i)
        batch_size     = batch_dict['batch_size']
        rois           = batch_dict['rois']
        num_rois       = rois.shape[1]

        enlarged_rois = box_utils.enlarge_box3d(rois.view(-1, 7).clone().detach(), pool_extra_width).view(batch_size, -1, 7) 
        
        if point_tag == 'pseudo':
            if 'points_mm' + idx_str not in batch_dict:
                idx_str = ''
            points_str = 'points_mm'+idx_str
            points_infos = batch_dict[points_str]
            coords = getattr(self, self.model_cfg.ROI_POINT_POOL.KERNEL_TYPE)

        else:
            if 'points' + idx_str not in batch_dict:
                idx_str = ''
            points_str = 'points'+idx_str
            points_infos = batch_dict[points_str]
            coords = getattr(self, self.model_cfg.ROI_POINT_POOL_MM.KERNEL_TYPE)

        batch_idx      = points_infos[:, 0]
        point_coords   = points_infos[:, 1:4]
        point_features = points_infos[:, 1:]    # N, 9 {x,y,z,i,r,g,b,u,v}                    

        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_CROP.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_features, point_depths[:, None]]
        point_features = torch.cat(point_features_list, dim=1)
        w, h = 1400, 400

        with torch.no_grad():
            total_pts_roi_index = []
            total_pts_batch_index = []
            total_pts_features = []
            for bs_idx in range(batch_size):
                bs_mask          = (batch_idx == bs_idx)
                cur_point_coords = point_coords[bs_mask]
                cur_features     = point_features[bs_mask]
                cur_roi          = enlarged_rois[bs_idx][:, 0:7].contiguous()

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(       
                    cur_point_coords.unsqueeze(0), cur_roi.unsqueeze(0)
                )      
                cur_box_idxs_of_pts = box_idxs_of_pts[0]

                points_in_rois = cur_box_idxs_of_pts != -1

                in_box_roi = cur_point_coords[points_in_rois] # the points is no problem
                # common_utils.save_point_cloud_to_ply(in_box_roi.cpu().numpy(), 'in_box_roi.ply')
                # common_utils.save_point_cloud_to_ply(cur_point_coords.cpu().numpy(), 'total_scene.ply')
                cur_box_idxs_of_pts = cur_box_idxs_of_pts[points_in_rois] + num_rois * bs_idx

                cur_pts_batch_index = cur_box_idxs_of_pts.new_zeros((cur_box_idxs_of_pts.shape[0]))
                cur_pts_batch_index[:] = bs_idx

                cur_features        = cur_features[points_in_rois]

                total_pts_roi_index.append(cur_box_idxs_of_pts)
                total_pts_batch_index.append(cur_pts_batch_index)
                total_pts_features.append(cur_features)

            total_pts_roi_index     =  torch.cat(total_pts_roi_index, dim=0)
            total_pts_batch_index =  torch.cat(total_pts_batch_index, dim=0)
            total_pts_features      =  torch.cat(total_pts_features, dim=0)
            total_pts_features_xyz_src = total_pts_features.clone()[...,:3]
            total_pts_rois = torch.index_select(rois.view(-1,7), 0, total_pts_roi_index.long())

            total_pts_features[:, 0:3] -= total_pts_rois[:, 0:3]
            total_pts_features[:, 0:3] = common_utils.rotate_points_along_z(
                total_pts_features[:, 0:3].unsqueeze(dim=1), -total_pts_rois[:, 6]
            ).squeeze(dim=1)      
            total_pts_features_raw = total_pts_features.clone()
            

        if point_tag == 'pseudo':

            with torch.no_grad():
                global_dv = total_pts_roi_index * h 
                total_pts_features[:, -2] += global_dv

            image = total_pts_features.new_zeros((batch_size*num_rois*h, w)).long()
            global_index = torch.arange(1, total_pts_features.shape[0]+1)

            total_pts_features[:,-3] = torch.clamp(total_pts_features[:,-3], min = 3, max= w-4)
            total_pts_features[:,-2] = torch.clamp(total_pts_features[:,-2], min = 3, max= batch_size*num_rois*h-4)

            image[total_pts_features[:,-2].long(), total_pts_features[:,-3].long()] = global_index.to(device=total_pts_features.device)
            
            # cv2.imwrite('img_view.png',image[:1500,:].cpu().numpy())

            points_list = []
            for circle_i in range(len(coords)):
                dx, dy = coords[circle_i]
                points_cur = image[total_pts_features[:,-2].long() + dx, total_pts_features[:,-3].long() + dy]
                points_list.append(points_cur)

        elif point_tag == 'origin':

            with torch.no_grad():
                bw, bh = 400, 400
                global_dv = total_pts_roi_index * bh
                assert torch.abs(total_pts_features[:,:2]).all() < 5, "the size is over"

                bev_coords_w = total_pts_features[:,0] * bw/10 + bw/2
                bev_coords_h = total_pts_features[:,1] * bh/10 + bh/2 + global_dv

                bev_coords_w = torch.clamp(bev_coords_w, min = 3, max= bw-4)
                bev_coords_h = torch.clamp(bev_coords_h, min = 3, max= batch_size*num_rois*bh-4)

            image = total_pts_features.new_zeros((batch_size*num_rois*bh, bw)).long()
            global_index = torch.arange(1, total_pts_features.shape[0]+1)
            image[bev_coords_h.int().long(), bev_coords_w.int().long()] = global_index.to(device=total_pts_features.device)

            tmp_img = (image!=0)*255
            cv2.imwrite('bev_view.png',tmp_img[:2000,:].cpu().numpy())

            points_list = []
            for circle_i in range(len(coords)):
                dx, dy = coords[circle_i]
                points_cur = image[bev_coords_h.int().long() + dx, bev_coords_w.int().long() + dy]
                points_list.append(points_cur)

        total_pts_neighbor = torch.stack(points_list,dim=0).transpose(0,1).contiguous()

        zero_features = total_pts_features.new_zeros((1,total_pts_features.shape[-1]))
        total_pts_features = torch.cat([zero_features,total_pts_features],dim=0)
        zero_neighbor = total_pts_neighbor.new_zeros((1,total_pts_neighbor.shape[-1]))
        total_pts_neighbor = torch.cat([zero_neighbor,total_pts_neighbor],dim=0)
        total_pts_batch_index = total_pts_batch_index.float().unsqueeze(dim=-1)

        return total_pts_features, total_pts_neighbor, total_pts_batch_index, total_pts_roi_index, total_pts_features_xyz_src

    def multi_grid_pool_aggregation(self, batch_dict, targets_dict):

        if self.model_cfg.get('PART', False):
            feat_2d = batch_dict['st_features_2d']
            parts_feat = self.conv_part(feat_2d)

        all_preds = []
        all_scores = []
        all_ious = []

        all_shared_features = []
        all_shared_features_mm = []

        for i in range(self.rot_num):

            rot_num_id = str(i)

            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois'] = self.roi_x_trans(batch_dict['rois'], i, batch_dict['transform_param'])

            if self.training:
                targets_dict = self.assign_targets(batch_dict, i)
                
                targets_dict['aug_param'] = batch_dict['aug_param']
                targets_dict['image_shape'] = batch_dict['image_shape']
                targets_dict['calib'] = batch_dict['calib']

                if self.iou_generate:
                    sampling_targets_dict = self.sampling_targets(batch_dict, i) #hebing
                    batch_dict['sampling_rois'] = sampling_targets_dict['rois']
                    batch_dict['sampling_roi_labels'] = sampling_targets_dict['roi_labels']

                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois_score'] = self.roi_score_trans(batch_dict['rois'], i, batch_dict['transform_param'])
            else:
                batch_dict['rois_score'] = batch_dict['rois']

            if not self.training and self.iou_generate:
                batch_dict['sampling_rois'] = batch_dict['rois']

            if self.model_cfg.get('PART', False):
                part_scores = self.roi_part_pool(batch_dict, parts_feat)

            if 'transform_param' in batch_dict:
                pooled_features = self.roi_grid_pool(batch_dict, i)
                if self.grid_aware_mm:
                    pooled_features_mm = self.roi_grid_pool_mm(batch_dict, i)
            else:
                pooled_features = self.roi_grid_pool(batch_dict, 0)
                if self.grid_aware_mm:
                    pooled_features_mm = self.roi_grid_pool_mm(batch_dict, 0)

            if self.grid_aware:
                pooled_features = pooled_features.view(pooled_features.size(0), -1)
                shared_features = self.shared_fc_layer_og(pooled_features)
                shared_features = shared_features.unsqueeze(0)  # 1,B,C
                all_shared_features.append(shared_features)
                pre_feat = torch.cat(all_shared_features, 0)
                cur_feat = self.cross_attention_layer_og(pre_feat, shared_features)
                cur_feat = torch.cat([cur_feat, shared_features], -1)
                cur_feat = cur_feat.squeeze(0)  # B, C*2

                final_feat = cur_feat

            if self.grid_aware_mm:
                pooled_features_mm = pooled_features_mm.view(pooled_features_mm.size(0), -1)
                shared_features_mm = self.shared_fc_layer_pg(pooled_features_mm)
                shared_features_mm = shared_features_mm.unsqueeze(0)  # 1,B,C
                all_shared_features_mm.append(shared_features_mm)
                pre_feat_mm = torch.cat(all_shared_features_mm, 0)
                cur_feat_mm = self.cross_attention_layer_pg(pre_feat_mm, shared_features_mm)
                cur_feat_mm = torch.cat([cur_feat_mm, shared_features_mm], -1)
                cur_feat_mm = cur_feat_mm.squeeze(0)  # B, C*2

                final_feat = torch.cat([final_feat, cur_feat_mm],-1)

            if self.point_aware:
                # RoI Point Pool
                B, N, _ = batch_dict['rois'].shape
                points_features_tmp, points_neighbor, points_batch, points_roi, points_coords_src = \
                    self.roicrop3d_gpu(batch_dict, self.model_cfg.ROI_POINT_CROP.POOL_EXTRA_WIDTH, i, point_tag= 'origin') 
                points_features = points_features_tmp.clone()
                points_features_expand = self.cpconvs_layer(points_features_tmp, points_neighbor)[1:]
                batch_dict['points_origin_features'] = torch.cat([points_batch, points_coords_src, points_features_expand],dim=-1)
                # RoI Voxel Pool
                cur_feat_op = self.roiaware_pool(batch_dict, i,  points_roi, points_tag='origin')# 320 * 512
               
                final_feat = torch.cat([final_feat, cur_feat_op],-1)

            if self.point_aware_mm:
                # RoI Point Pool
                B, N, _ = batch_dict['rois'].shape
                points_features_tmp, points_neighbor, points_batch, points_roi, points_coords_src = \
                    self.roicrop3d_gpu(batch_dict, self.model_cfg.ROI_POINT_CROP.POOL_EXTRA_WIDTH, i, point_tag = 'pseudo') 
                points_features = points_features_tmp.clone()
                points_features_expand = self.cpconvs_layer_mm(points_features_tmp, points_neighbor)[1:]
                batch_dict['points_pseudo_features'] = torch.cat([points_batch, points_coords_src, points_features_expand],dim=-1)
                # RoI Voxel Pool
                cur_feat_grid_pp = self.roiaware_pool(batch_dict, i,  points_roi, points_tag='pseudo')# 320 * 6 * 6 * 6 * 90
                cur_feat_grid_pp = cur_feat_grid_pp.view(B*N, -1)

                cur_feat_pp = self.fc_pseudo_pp(cur_feat_grid_pp)
               
                final_feat = torch.cat([final_feat, cur_feat_pp],-1)

            rcnn_cls = self.cls_layers[i](final_feat)
            rcnn_reg = self.reg_layers[i](final_feat)

            if self.grid_aware:
                rcnn_cls_og = self.cls_fc_og(cur_feat)
                rcnn_reg_og = self.reg_fc_og(cur_feat)

            if self.grid_aware_mm:
                rcnn_cls_pg = self.cls_fc_pg(cur_feat_mm)
                rcnn_reg_pg = self.reg_fc_pg(cur_feat_mm)

            if self.point_aware:
                rcnn_cls_op = self.cls_fc_op(cur_feat_op)
                rcnn_reg_op = self.reg_fc_op(cur_feat_op)

            if self.point_aware_mm:
                rcnn_cls_pp = self.cls_fc_pp(cur_feat_pp)
                rcnn_reg_pp = self.reg_fc_pp(cur_feat_pp)

            if self.model_cfg.get('PART', False):
                rcnn_cls = rcnn_cls+part_scores


            if self.iou_generate:
                if self.training:
                    if 'transform_param' in batch_dict:
                        pooled_features = self.roi_grid_pool(batch_dict, i, enable_sampling=True)
                        pooled_features_mm = self.roi_grid_pool_mm(batch_dict, i, enable_sampling=True)
                    else:
                        pooled_features = self.roi_grid_pool(batch_dict, 0, enable_sampling=True)
                        pooled_features_mm = self.roi_grid_pool_mm(batch_dict, 0, enable_sampling=True)

                    pooled_features = pooled_features.view(pooled_features.size(0), -1)
                    pooled_features_mm = pooled_features_mm.view(pooled_features_mm.size(0), -1)

                    if self.model_cfg.get('PART', False):
                        if i >= 1 and 'transform_param' in batch_dict:
                            batch_dict['sampling_rois_score'] = self.roi_score_trans(batch_dict['sampling_rois'], i, batch_dict['transform_param'])
                        else:
                            batch_dict['sampling_rois_score'] = batch_dict['sampling_rois']

                sampling_features = self.shared_sampling_fc_layers[0](pooled_features)
                sampling_features_mm = self.shared_sampling_fc_layers_mm[0](pooled_features_mm)

                final_sampling_feat = torch.cat([sampling_features_mm, sampling_features],-1)
                rcnn_sampling_reg = self.sampling_reg_layers[0](final_sampling_feat)
                rcnn_sampling_iou = self.sampling_iou_layers[0](final_sampling_feat)

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            outs = batch_box_preds.clone()
            all_scores.append(batch_cls_preds)

            if self.iou_generate:
                batch_iou_preds, batch_sampling_box_preds = self.generate_predicted_boxes_ious(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['sampling_rois'] , iou_preds=rcnn_sampling_iou, box_preds=rcnn_sampling_reg)
                sampling_outs = batch_sampling_box_preds.clone()
                all_ious.append(batch_iou_preds)
            
            if 'transform_param' in batch_dict:
                outs = self.pred_x_trans(outs, i, batch_dict['transform_param'])
                if self.iou_generate:
                    sampling_outs = self.pred_x_trans(sampling_outs,i, batch_dict['transform_param'])

            if self.iou_generate and not self.training:
                all_preds.append(torch.mean(torch.stack([outs,sampling_outs]), 0))
            else:  
                all_preds.append(outs)
            
            if self.training:
                if self.grid_aware:
                    targets_dict_og = copy.deepcopy(targets_dict)
                    targets_dict_og['rcnn_cls'] = rcnn_cls_og
                    targets_dict_og['rcnn_reg'] = rcnn_reg_og
                    self.forward_ret_dict['targets_dict_og' + rot_num_id] = targets_dict_og

                if self.grid_aware_mm:
                    targets_dict_pg = copy.deepcopy(targets_dict)
                    targets_dict_pg['rcnn_cls'] = rcnn_cls_pg
                    targets_dict_pg['rcnn_reg'] = rcnn_reg_pg
                    self.forward_ret_dict['targets_dict_pg' + rot_num_id] = targets_dict_pg

                if self.point_aware:
                    targets_dict_op = copy.deepcopy(targets_dict)
                    targets_dict_op['rcnn_cls'] = rcnn_cls_op
                    targets_dict_op['rcnn_reg'] = rcnn_reg_op
                    self.forward_ret_dict['targets_dict_op' + rot_num_id] = targets_dict_op

                if self.point_aware_mm:
                    targets_dict_pp = copy.deepcopy(targets_dict)
                    targets_dict_pp['rcnn_cls'] = rcnn_cls_pp
                    targets_dict_pp['rcnn_reg'] = rcnn_reg_pp
                    self.forward_ret_dict['targets_dict_pp' + rot_num_id] = targets_dict_pp

                if self.iou_generate:
                    targets_dict_sampling = copy.deepcopy(sampling_targets_dict)
                    targets_dict_sampling['rcnn_reg'] = rcnn_sampling_reg
                    targets_dict_sampling['rcnn_iou'] = rcnn_sampling_iou
                    self.forward_ret_dict['target_dict_sampling' + rot_num_id] = targets_dict_sampling
                targets_dict['rcnn_cls'] = rcnn_cls
                targets_dict['rcnn_reg'] = rcnn_reg
                self.forward_ret_dict['targets_dict' + rot_num_id] = targets_dict

            batch_dict['rois'] = batch_box_preds
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

        # return torch.mean(torch.stack(all_preds), 0), torch.mean(torch.stack(all_scores), 0), torch.mean(torch.stack(all_ious),0) if self.iou_generate else None
        return torch.mean(torch.stack(all_preds), 0), torch.mean(torch.stack(all_scores), 0), all_ious[-1] if self.iou_generate else None

    def voxel_grid_iou_pool(self, batch_dict, rois, targets_dict):

        all_shared_features = []
        all_shared_features_mm = []

        for i in range(self.rot_num-1,-1,-1):
            if i < self.rot_num-1 and 'transform_param' in batch_dict:
                batch_dict['rois'] = self.roi_x_reverse_trans(batch_dict['rois'], i, batch_dict['transform_param'])

        if self.training:
            targets_dict = self.assign_targets(batch_dict, 0, enable_dif=True) # 512 sampling to 160
            targets_dict['aug_param'] = batch_dict['aug_param']
            targets_dict['image_shape'] = batch_dict['image_shape']
            targets_dict['calib'] = batch_dict['calib']
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['rois_score'] = batch_dict['rois']

        pooled_features = self.roi_grid_pool(batch_dict, 0)
        pooled_features_mm = self.roi_grid_pool_mm(batch_dict, 0)

        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer_og(pooled_features)
        shared_features = shared_features.unsqueeze(0)  # 1,B,C
        all_shared_features.append(shared_features)
        pre_feat = torch.cat(all_shared_features, 0)
        cur_feat = self.cross_attention_layer_og(pre_feat, shared_features)
        cur_feat = torch.cat([cur_feat, shared_features], -1)
        cur_feat = cur_feat.squeeze(0)  # B, C*2

        pooled_features_mm = pooled_features_mm.view(pooled_features_mm.size(0), -1)
        shared_features_mm = self.shared_fc_layer_pg(pooled_features_mm)
        shared_features_mm = shared_features_mm.unsqueeze(0)  # 1,B,C
        all_shared_features_mm.append(shared_features_mm)
        pre_feat_mm = torch.cat(all_shared_features_mm, 0)
        cur_feat_mm = self.cross_attention_layer_pg(pre_feat_mm, shared_features_mm)
        cur_feat_mm = torch.cat([cur_feat_mm, shared_features_mm], -1)
        cur_feat_mm = cur_feat_mm.squeeze(0)  # B, C*2

        final_feat = torch.cat([cur_feat_mm, cur_feat],-1)
        rcnn_iou = self.iou_layers[0](final_feat)

        if self.training:
            targets_dict_raw_iou = copy.deepcopy(targets_dict)
            targets_dict_raw_iou['rcnn_iou'] = rcnn_iou
            self.forward_ret_dict['targets_dict_iou_final'] = targets_dict_raw_iou
        # rcnn_iou= rcnn_iou.view(batch_dict['batch_size'],-1,1).contiguous()

        return rcnn_iou

    def forward(self, batch_dict):

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            self.rot_num = trans_param.shape[1]

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        boxes, scores, ious1 = self.multi_grid_pool_aggregation(batch_dict, targets_dict)

        # ious2 = self.voxel_grid_iou_pool(batch_dict,boxes, targets_dict)

        if not self.training:
            batch_dict['batch_box_preds'] = boxes
            batch_dict['batch_cls_preds'] = scores
            if self.iou_generate:
                batch_dict['batch_iou_preds_s0'] = ious1

        return batch_dict
