from mmcv.utils import Registry
from mmdet.core.bbox.builder import BBOX_ASSIGNERS

ATTACKER = Registry('attack')



def build_attack(cfg):
    """Build attacker."""
    assert cfg.assigner is not None, \
        "Should specify an assigner in attack"

    ## a workaround to build assigner in attacker
    ## there may be some better way to do this
    cfg.assigner = BBOX_ASSIGNERS.build(cfg.assigner)
    return ATTACKER.build(cfg)