import numpy as np

from voxelnet.core.target_assigner import TargetAssigner
from voxelnet.builder import similarity_calculator_builder
from voxelnet.builder import anchor_generator_builder

def build(target_assigner_config, bv_range, box_coder):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    anchor_cfg = target_assigner_config.anchor_generators
    anchor_generators = []
    anchor_generator = anchor_generator_builder.build(anchor_cfg)
    anchor_generators.append(anchor_generator)
    similarity_calc = similarity_calculator_builder.build(
        target_assigner_config.region_similarity_calculator)
    positive_fraction = target_assigner_config.sample_positive_fraction
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config.sample_size)
    return target_assigner

