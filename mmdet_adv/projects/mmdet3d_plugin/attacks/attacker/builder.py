from mmcv.utils import Registry
from copy import deepcopy

ATTACKER = Registry('attack')



def build_attack(cfg):
    """Build attacker."""

    cfg = deepcopy(cfg)

    assert cfg.assigner is not None, \
        "Should specify an assigner in attacker"
    assert cfg.loss_fn is not None, \
        "Should specify an loss_fn in attacker"

    return ATTACKER.build(cfg)