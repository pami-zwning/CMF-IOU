"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np


box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def create_lines_from_box(box, color):
    # box format: [x, y, z, dx, dy, dz, yaw]
    x, y, z, dx, dy, dz, yaw = box
    
    # Create box corners
    corners = np.array([
        [dx / 2, dy / 2, dz / 2],
        [dx / 2, -dy / 2, dz / 2],
        [-dx / 2, -dy / 2, dz / 2],
        [-dx / 2, dy / 2, dz / 2],
        [dx / 2, dy / 2, -dz / 2],
        [dx / 2, -dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, -dz / 2],
        [-dx / 2, dy / 2, -dz / 2]
    ])
    
    # Rotation matrix around z-axis
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotate and translate corners
    corners = np.dot(corners, R.T)
    corners += np.array([x, y, z])
    
    # Define box lines
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
    ]
    
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(corners)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def save_scenes(points, filename, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    # 将 torch.Tensor 转换为 numpy 数组
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # 创建点云对象
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    if point_colors is not None:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    else:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))

    # 保存点云信息到 PLY 文件
    open3d.io.write_point_cloud(filename + "_points.ply", pts)

    # 保存 gt_boxes 到 PLY 文件
    # if len(gt_boxes)!=0:
    if gt_boxes is not None:
        gt_boxes_lines = open3d.geometry.LineSet()
        for box in gt_boxes:
            box_lines = create_lines_from_box(box, [1.0, 0.0, 0.0])
            gt_boxes_lines += box_lines
        open3d.io.write_line_set(filename + "_gt_boxes.ply", gt_boxes_lines)

    # 保存 ref_boxes 到 PLY 文件
    # if len(ref_boxes)!=0:
    if ref_boxes is not None:
        ref_boxes_lines = open3d.geometry.LineSet()
        for box in ref_boxes:
            box_lines = create_lines_from_box(box, [0.0, 1.0, 0.0])
            ref_boxes_lines += box_lines
        open3d.io.write_line_set(filename + "_ref_boxes.ply", ref_boxes_lines)


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
