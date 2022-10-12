import torch.nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class ClassficationObjective(nn.Module):
    def __init__(self, activate=False):
        """Classification adversarial objective, try to modified the classification result
        of detection model

        Args:
             activate: whether the input is logits and probability distribution
        """
        super().__init__()
        if activate == True:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, pred_logits, gt_label, pred_bboxes=None, pred_scores=None, gt_bbox=None):
        cls_loss = self.loss(pred_logits, gt_label)
        return cls_loss