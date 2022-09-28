import torch.nn as nn

from .builder import OBJECTIVE


@OBJECTIVE.register_module()
class ClassficationObjective(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self):
        pass