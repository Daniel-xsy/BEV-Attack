from mmcv.utils import Registry

ATTACKER = Registry('attack')



def build_attack(cfg):
    """Build attacker."""
    assert cfg.assigner is not None, \
        "Should specify an assigner in attacker"
    assert cfg.loss_fn is not None, \
        "Should specify an loss_fn in attacker"

    return ATTACKER.build(cfg)