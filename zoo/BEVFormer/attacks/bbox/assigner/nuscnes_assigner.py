from mmdet.core.bbox.builder import BBOX_ASSIGNERS

@BBOX_ASSIGNERS.register_module()
class NuScenesAssigner:
    def __init__(self, dis_thresh):
        pass

    def assign(self):
        pass