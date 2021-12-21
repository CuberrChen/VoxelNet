import numpy as np

from voxelnet.protos import box_coder_pb2
from voxelnet.pypaddle.core.box_coders import (BevBoxCoderPaddle,
                                               GroundBox3dCoderPaddle)


def build(box_coder_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    box_coder_type = box_coder_config.WhichOneof('box_coder')
    if box_coder_type == 'ground_box3d_coder':
        cfg = box_coder_config.ground_box3d_coder
        return GroundBox3dCoderPaddle(cfg.linear_dim, cfg.encode_angle_vector)
    elif box_coder_type == 'bev_box_coder':
        cfg = box_coder_config.bev_box_coder
        return BevBoxCoderPaddle(cfg.linear_dim, cfg.encode_angle_vector, cfg.z_fixed, cfg.h_fixed)
    else:
        raise ValueError("unknown box_coder type")
