from mmcv.utils import Registry


OBJECTIVE = Registry('objective')


def build_attack(cfg):
    """Build adversarial loss function."""
    return OBJECTIVE.build(cfg)