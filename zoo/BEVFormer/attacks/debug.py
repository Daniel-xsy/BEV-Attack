import mmcv

file_path = '/home/cihangxie/shaoyuan/BEV-Attack/test.pkl'
data = mmcv.load(file_path)

outputs = data['outputs']
gt_bboxes_3d = data['gt_bboxes_3d']
gt_labels_3d = data['gt_labels_3d']

a = 1