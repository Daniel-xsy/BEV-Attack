from mmcv.utils import Registry


ATTACKER = Registry('attack')


def build_attack(cfg):
    """Build attacker."""
    return ATTACKER.build(cfg)