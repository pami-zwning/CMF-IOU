
from .spconv_backbone import VoxelBackBone8x, VoxelBackBone8xBIL
from .vc_enhance_backbone import VCEConv8x
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt,VoxelResBackBone8xVoxelNeXtTED
__all__ = {
    'VCEConv8x':VCEConv8x,
    'VoxelBackBone8x': VoxelBackBone8x,
    'VoxelBackBone8xBIL': VoxelBackBone8xBIL,
    'VoxelResBackBone8xVoxelNeXt':VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXtTED':VoxelResBackBone8xVoxelNeXtTED
}
