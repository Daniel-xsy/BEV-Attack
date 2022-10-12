from mmcv.utils import Registry
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES

ATTACKER = Registry('attack')



def build_attack(cfg):
    """Build attacker."""
    assert cfg.assigner is not None, \
        "Should specify an assigner in attacker"
    assert cfg.loss_fn is not None, \
        "Should specify an loss_fn in attacker"

    ## a workaround to build class or function in attacker
    ## there may be some better ways to do this
    cfg.assigner = BBOX_ASSIGNERS.build(cfg.assigner)
    cfg.loss_fn = LOSSES.build(cfg.loss_fn)
    return ATTACKER.build(cfg)