import logging
import os
import pickle
import random
import shutil
import subprocess
import open3d as o3d

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
from pcdet.datasets.augmentor.X_transform import X_TRANS


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info["name"]) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (
        (points[:, 0] >= limit_range[0])
        & (points[:, 0] <= limit_range[3])
        & (points[:, 1] >= limit_range[1])
        & (points[:, 1] <= limit_range[4])
    )
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = (
        torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    )
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else "ERROR")
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else "ERROR")
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else "ERROR")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend="nccl"):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = str(tcp_port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["RANK"] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend="nccl"):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:%d" % tcp_port,
        rank=local_rank,
        world_size=num_gpus,
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def save_point_cloud_to_ply(points, filename):
    """
    保存点云为PLY格式文件

    参数:
    points (numpy.ndarray): N*3形状的点云数据
    filename (str): 输出文件的路径
    """
    # 检查点云数据是否为N*3的形状
    if points.shape[1] != 3:
        raise ValueError("点云数据必须为N*3的形状")

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 将点云数据转换为Open3D格式
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 保存为PLY文件
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"点云数据已保存为 {filename}")


def get_box_corners(box):
    """
    计算一个gt_box的8个角点坐标
    参数:
    box (numpy.ndarray): 形状为(7,)的数组，包含中心点 (x, y, z)，长宽高 (l, w, h)，以及朝向角度 (yaw)
    返回:
    corners (numpy.ndarray): 形状为(8, 3)的数组,表示8个角点的坐标
    """
    x, y, z, l, w, h, yaw = box

    # 立方体的角点相对于中心的偏移量
    corners = np.array(
        [
            [l / 2, w / 2, h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, -h / 2],
        ]
    )

    # 旋转角点
    rotation_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    corners = corners @ rotation_matrix.T

    # 平移角点到中心位置
    corners += np.array([x, y, z])

    return corners


def corners_3d_to_2d(calib, boxes_3d):
    boxes_2d = []
    # for box in annos[0]['boxes_lidar']:
    for box in boxes_3d:
        corners_3d = get_box_corners(box)

        # 使用标定矩阵将3D角点转换为2D
        corners_2d = []

        rect_coords = calib.lidar_to_rect(corners_3d)
        img_coords, _ = calib.rect_to_img(rect_coords)

        corners_2d = img_coords.astype(np.int32)

        boxes_2d.append(corners_2d)

    return boxes_2d


def draw_boxes_2d(image, boxes_2d, color="g", line_width=1):
    if color == "g":
        color_val = (0, 255, 0)
    elif color == "r":
        color_val = (0, 0, 255)
    elif color == "b":
        color_val = (255, 0, 0)
    else:
        print("the input of color is not defined")
    for box in boxes_2d:
        for i in range(4):
            pt1 = (int(box[i][0]), int(box[i][1]))
            pt2 = (int(box[i + 4][0]), int(box[i + 4][1]))
            cv2.line(image, pt1, pt2, color_val, line_width)

        pt0 = (int(box[0][0]), int(box[0][1]))
        pt1 = (int(box[1][0]), int(box[1][1]))
        pt2 = (int(box[2][0]), int(box[2][1]))
        pt3 = (int(box[3][0]), int(box[3][1]))

        pt4 = (int(box[4][0]), int(box[4][1]))
        pt5 = (int(box[5][0]), int(box[5][1]))
        pt6 = (int(box[6][0]), int(box[6][1]))
        pt7 = (int(box[7][0]), int(box[7][1]))

        cv2.line(image, pt0, pt2, color_val, line_width)
        cv2.line(image, pt0, pt1, color_val, line_width)
        cv2.line(image, pt3, pt1, color_val, line_width)
        cv2.line(image, pt2, pt3, color_val, line_width)

        cv2.line(image, pt4, pt6, color_val, line_width)
        cv2.line(image, pt4, pt5, color_val, line_width)
        cv2.line(image, pt7, pt5, color_val, line_width)
        cv2.line(image, pt6, pt7, color_val, line_width)
    return image


def save_gt_boxes_to_ply(gt_boxes, filename):
    """
    将gt_boxes的角点保存为PLY文件
    参数:
    gt_boxes (numpy.ndarray): 形状为(N, 7)的gt_boxes数据
    filename (str): 输出文件的路径
    """
    all_corners = []

    for box in gt_boxes:
        corners = get_box_corners(box)
        all_corners.append(corners)

    # 将所有角点整合到一个数组中
    all_corners = np.vstack(all_corners)

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 将点云数据转换为Open3D格式
    point_cloud.points = o3d.utility.Vector3dVector(all_corners)

    # 保存为PLY文件
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"点云数据已保存为 {filename}")

def project_to_image(P, calib, aug_param=None, tag = 'projected'):
    pts_3d = P[:,:3]
    x_trans_train = X_TRANS()

    if aug_param is not None:
        transed = x_trans_train.backward_with_param({'points': pts_3d[:, :3],
                                                        'transform_param': aug_param})

        pts_3d = transed['points']  # .cpu().numpy()

    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.cpu().numpy()
    
    rect_coords = calib.lidar_to_rect(pts_3d)
    img_coords, _ = calib.rect_to_img(rect_coords)

    corners_2d = img_coords.astype(np.int32)
    corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, 1300-1)
    corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, 400-1)

    image = np.zeros((400, 1300), dtype=np.uint8)
    image[corners_2d[:, 1], corners_2d[:, 0]] = 255

    cv2.imwrite(f'./image/{tag}.png', image)

def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret
    

def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor

    
def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(
        result_part, open(os.path.join(tmpdir, "result_part_{}.pkl".format(rank)), "wb")
    )
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, "result_part_{}.pkl".format(i))
        part_list.append(pickle.load(open(part_file, "rb")))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

