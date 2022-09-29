from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def initMetrics(annFile, anntype='bbox'):
    metrics = {}
    metrics['Gt'] = COCO(annFile)
    

def updateMetrics(metrics):
    pass

