import numpy as np

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', '[*] street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '[*] hat', 'backpack', 'umbrella', '[*] shoe', '[*] eye glasses', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '[*] plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', '[*] mirror', 'dining table', '[*] window', '[*] desk', 'toilet', 
    '[*] door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
    'sink', 'refrigerator', '[*] blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 
    '[*] hair brush']

COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))

CONFIDENCE = 0.7

def encode_detection_results(detections, labels, factors=(1, 1)):
    """Arrange results to match COCO specs in http://cocodataset.org/#format
    """
    # print(detections)
    image_id = labels[0]['image_id'] #.cpu().numpy()
    class_ids = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    resize_factor = factors
    boxes = detections['boxes'].cpu() #.numpy()
    

    # If no results, return an empty list
    if boxes is None:
        return []

    results = []
    
    # Loop through detections
    for i in range(boxes.shape[0]):
        class_id = class_ids[i]
        score = scores[i]
        bbox = np.around(boxes[i], 1)
        # mask = masks[:, :, i]

        result = {
            "image_id": int(image_id),
            "category_id": int(class_id), #dataset.get_source_class_id(class_id, "coco"),
            # "bbox": [bbox[1] / resize_factor[1], bbox[0] / resize_factor[0], (bbox[3] - bbox[1]) / resize_factor[1], (bbox[2] - bbox[0]) / resize_factor[0]],
            "bbox": [bbox[0] / resize_factor[1], bbox[1] / resize_factor[0], (bbox[2] - bbox[0]) / resize_factor[1], (bbox[3] - bbox[1]) / resize_factor[0]],
            "score": score,
            # "segmentation": maskUtils.encode(np.asfortranarray(mask))
        }
        results.append(result)
    return results