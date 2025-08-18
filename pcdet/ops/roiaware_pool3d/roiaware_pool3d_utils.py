import torch
import torch.nn as nn
from torch.autograd import Function

from ...utils import common_utils
from ..pointnet2.pointnet2_batch.pointnet2_utils import furthest_point_sample 
from . import roiaware_pool3d_cuda

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetFeatureExtractor(nn.Module):
    def __init__(self, num_points = 50, mid_dim = [512, 256]):
        super(PointNetFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(90 * num_points, mid_dim[1])
        self.fc2 = nn.Linear(mid_dim[1], mid_dim[1])
        self.fc3 = nn.Linear(mid_dim[1], mid_dim[0])
        self.bn1 = nn.BatchNorm1d(mid_dim[1])
        self.bn2 = nn.BatchNorm1d(mid_dim[1])
        self.bn3 = nn.BatchNorm1d(mid_dim[0])
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x

class PostPointNet(nn.Module):
    def __init__(self, num_points=50, output_dim=512, mid_dim= [512, 256]):
        super(PostPointNet, self).__init__()
        self.num_points = num_points
        self.feature_extractor = PointNetFeatureExtractor(num_points, mid_dim)
        self.fc1 = nn.Linear(mid_dim[0], mid_dim[1])
        self.fc2 = nn.Linear(mid_dim[1], output_dim)
        self.bn1 = nn.BatchNorm1d(mid_dim[1])
        self.bn2 = nn.BatchNorm1d(output_dim)
    
    def forward(self, cur_roi_nums, cur_point_rois_idx, cur_points_features, cur_points_coords):
        
        roi_features_list = []
        
        for i in range(cur_roi_nums):
            mask = cur_point_rois_idx == i
                   
            roi_coords = cur_points_coords[mask]
            roi_features = cur_points_features[mask]

            if roi_coords.shape[0]==0:
                sampled_features = cur_points_features.new_zeros((1,cur_points_features.shape[-1] * self.num_points))
                roi_features_list.append(sampled_features)
                continue
                
            elif roi_coords.shape[0] <= self.num_points:
                sampled_points_idx = torch.randint(0,roi_coords.shape[0],(self.num_points,))

            else:
                numbers = torch.arange(roi_coords.shape[0])
                sampled_points_idx = furthest_point_sample(roi_coords.unsqueeze(0), self.num_points).squeeze(0)

            sampled_features = roi_features[sampled_points_idx.long()]
            sampled_features = sampled_features.view(1, -1)  # [1, 90, num_points]
            roi_features_list.append(sampled_features)
        
        roi_features = torch.cat(roi_features_list, dim=0)
        roi_features = self.feature_extractor(roi_features)
        x = F.relu(self.bn1(self.fc1(roi_features)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        return x  # 输出特征维度为 [M, 512]

def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices

def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    return box_idxs_of_pts

class RoIAwarePool3d(nn.Module):
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)

class RoIAwarePool3dFunction(Function):
    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        Args:
            ctx:
            rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
            pts: (npoints, 3)
            pts_feature: (npoints, C)
            out_size: int or tuple, like 7 or (7, 7, 7)
            max_pts_each_voxel:
            pool_method: 'max' or 'avg'

        Returns:
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)

        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward

        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)

        return None, None, grad_in, None, None, None

if __name__ == '__main__':
    pass
