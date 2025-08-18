from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn
import numpy as np
import torch
from pcdet.datasets.augmentor.X_transform import X_TRANS
import cv2

def index2uv(indices, stride):
    """
    convert the 3D voxel indices to image pixel indices.
    """

    new_uv[:, 0] = indices[:,0]
    new_uv[:,1:3] = indices[:,1:3] // (8//stride)
    height = indices[:, 3] * voxel_size[-1] * stride - 2
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max=200 - 1)
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=200 - 1)

    return new_uv, height

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m

def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

class ResConvBlock(nn.Module):# utilized for the pseudo points
    """
    convolve the voxel features in both 3D and 2D space.
    """

    def __init__(self, input_c=16, output_c=16, stride=1, padding=1, indice_key='vir1', conv_depth=False):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        block = post_act_block
        block2d = post_act_block2d
        norm_fn = norm_fn_1d
        self.conv_depth = conv_depth

        if self.stride > 1:
            self.down_layer = block(input_c,
                                    output_c,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=stride,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        c1 = input_c

        if self.stride > 1:
            c1 = output_c
        if self.conv_depth:
            c1 += 4

        c2 = output_c

        self.d3_conv1 = block(c1,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.d2_conv1 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm3' + indice_key))

        self.d3_conv2 = block(c2 // 2,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))
        self.d2_conv2 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm4' + indice_key))

    def forward(self, sp_tensor, img_stride):
        batch_size = sp_tensor.batch_size

        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)

        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)

        uv_coords, depth = index2uv(d3_feat2.indices, img_stride)
        # N*3,N*1
        d2_sp_tensor1 = spconv.SparseConvTensor(
            features=d3_feat2.features,
            indices=uv_coords.int(),
            spatial_shape=[200, 176],
            batch_size=batch_size
        )

        d2_feat1 = self.d2_conv1(d2_sp_tensor1)
        d2_feat2 = self.d2_conv2(d2_feat1)

        d3_feat3 = replace_feature(d3_feat2, torch.cat([d3_feat2.features, d2_feat2.features], -1))

        return d3_feat3

class SparseBasicBlock3D(spconv.SparseModule):

    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d):
        super(SparseBasicBlock3D, self).__init__()

        self.conv1 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(dim)

        self.conv2 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out

class RAWSDBlock(spconv.SparseModule): # utilized for the raw points

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, down_filters: list, indice_key = 'rsd'):
        super().__init__()

        block = SparseBasicBlock3D

        self.encoder = nn.ModuleList(
            [spconv.SparseSequential(
                *[block(dim, indice_key=f"{indice_key}_0") for _ in range(down_filters[0])])]
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [ # once spconv + twice subm conv
                post_act_block(
                    dim, dim, down_kernel_size[idx], f'spconv_{indice_key}_{idx}', down_stride[idx], down_kernel_size[idx] // 2, 
                    conv_type='spconv', norm_fn=norm_fn_1d),
                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(down_filters[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers)) 

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append( # once inverse spconv
                post_act_block(
                    dim, dim, down_kernel_size[idx], f'spconv_{indice_key}_{idx}', conv_type='inverseconv', norm_fn = norm_fn_1d))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)

        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x

class VoxelBackBone8xBIL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = norm_fn_1d

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.s2d_enable = model_cfg.get('SD_BRANCH', False)

        if self.s2d_enable:
            self.s2d_layers = nn.ModuleList()
            s2d_layer_num = model_cfg.S2D_LAYERS_NUM
            for i in range(s2d_layer_num):
                layer = RAWSDBlock(64, model_cfg.DOWN_KERNEL_SIZE, model_cfg.DOWN_STRIDE, model_cfg.DOWN_FILTERS, indice_key=f's2d_{i}')
                self.s2d_layers.append(layer)

        self.resvc_enable = model_cfg.get('RESVC_BRANCH',False)

        if self.resvc_enable:
            self.vir_conv1 = ResConvBlock(16, 16, stride=1, indice_key='vir1')
            self.vir_conv2 = ResConvBlock(16, 32, stride=2, stride=2, indice_key='vir2')
            self.vir_conv3 = ResConvBlock(32, 64, stride=2, stride=2, indice_key='vir3')
            self.vir_conv4 = ResConvBlock(64, 64, stride=2, stride=2, padding=(0, 1, 1), indice_key='vir4')

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.vir_conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        if self.s2d_enable:
            for layer in self.s2d_layers:
                x_conv4 = layer(x_conv4)
        
        if self.resvc_enable:
            x_vir_conv1 = self.vir_conv1(x, 1)
            x_vir_conv2 = self.vir_conv2(x_vir_conv1, 2)
            x_vir_conv3 = self.vir_conv3(x_vir_conv2, 4)
            x_vir_conv4 = self.vir_conv4(x_vir_conv3, 8)


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        vir_out = self.vir_conv_out(x_vir_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_vir': vir_out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            },
            'multi_scale_3d_features_vir': {
                'x_vir_conv1': x_vir_conv1,
                'x_vir_conv2': x_vir_conv2,
                'x_vir_conv3': x_vir_conv3,
                'x_vir_conv4': x_vir_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = norm_fn_1d

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict





