# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.necks.fpn import FPN
from .cp_fpn import CPFPN

#####   depreciate  #####
# from .view_transformer import ViewTransformerLiftSplatShoot, \
#     ViewTransformerLSSBEVDepth
# from .fpn import FPNForBEVDet
# from .lss_fpn import FPN_LSS

__all__ = ['FPN', 'CPFPN']
