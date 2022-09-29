import numpy as np
import torch




def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                            352, left_margin:left_margin + 1216]            

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif args.dataset == 'nyudepthv2':
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[45:471, 41:601] = 1
    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]

class runningScore(object):
    
    def __init__(self, ):
        self.metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']
        self.result_metrics = {}
        for name in self.metric_name:
            self.result_metrics[name] = 0.0

        self.num_images = 0

    def get_scores(self):
        return {
            'd1': self.result_metrics['d1'] / self.num_images, 
            'd2': self.result_metrics['d2'] / self.num_images, 
            'd3': self.result_metrics['d3'] / self.num_images, 
            'abs_rel': self.result_metrics['abs_rel'] / self.num_images,
            'sq_rel': self.result_metrics['sq_rel'] / self.num_images, 
            'rmse':self.result_metrics['rmse'] / self.num_images, 
            'rmse_log': self.result_metrics['rmse_log'] / self.num_images, 
            'log10': self.result_metrics['log10'] / self.num_images, 
            'silog': self.result_metrics['silog'] / self.num_images
        }

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            eval_results = self.eval_depth(lp, lt)
            for name in self.metric_name:
                self.result_metrics[name] += eval_results[name]
            self.num_images += 1

    def eval_depth(self, pred, target):
        # print(pred.shape, target.shape)
        assert pred.shape == target.shape
        pred = pred[torch.where(target > 0)]
        target = target[torch.where(target > 0)]

        thresh = torch.max((target / pred), (pred / target))

        d1 = torch.sum(thresh < 1.25).float() / len(thresh)
        d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
        d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

        diff = pred - target
        diff_log = torch.log(pred) - torch.log(target)

        abs_rel = torch.mean(torch.abs(diff) / target)
        sq_rel = torch.mean(torch.pow(diff, 2) / target)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

        log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

        return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
                'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
                'log10':log10.item(), 'silog':silog.item()}

    # def reset(self):
    #     self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count