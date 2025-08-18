from .roi_head_template import RoIHeadTemplate

from .ted_head import TEDMHead
from .iou_joint_head import TEDIOUHead
from .cpv_iou_joint_head import CPVIOUHead

from .voxelrcnn_head import VoxelRCNNHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'TEDMHead': TEDMHead,
    'TEDIOUHead': TEDIOUHead,
    'CPVIOUHead': CPVIOUHead,
    'VoxelRCNNHead': VoxelRCNNHead

}
