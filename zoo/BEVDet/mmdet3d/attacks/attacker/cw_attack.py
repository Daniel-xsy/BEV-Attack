import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from .base import BaseAttacker
from .builder import ATTACKER
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES


@ATTACKER.register_module()
class CWAttack(BaseAttacker):
    def __init__(self,
                 max_iterations,
                 learning_rate,
                 initial_const,
                 loss_fn,
                 assigner,
                 single_camera=False,
                 mono_model=False,
                 lambda_fixed=True,
                 *args, 
                 **kwargs):
        """ C&W pixel attack
        Args:
            max_iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimization
            initial_const (float): Initial trade-off constant c
            loss_fn (class): adversarial objective function
            assigner (class): assign prediction bbox to ground truth bbox
            single_camera (bool): only attack random choose single camera
            lambda_fixed (bool): whether to use a fixed lambda for all models
        """
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.initial_const = initial_const
        self.assigner = BBOX_ASSIGNERS.build(assigner)
        self.loss_fn = LOSSES.build(loss_fn)
        self.single_camera = single_camera
        self.mono_model = mono_model
        self.lambda_fixed = lambda_fixed

        if self.mono_model:
            self.size = (1, 3, 1, 1) # do not have stereo camera information
        else:
            self.size = (1, 1, 3, 1, 1)

    def run(self, model, img_inputs, img_metas, gt_bboxes_3d, gt_labels_3d):
        model.eval()

        img_ = img_inputs[0][0].clone()
        B = img_.size(0)
        assert B == 1, f"Batchsize should set to 1 in attack, but now is {B}"

        # Initialize adversarial image
        x_adv = img_.detach()
        x_adv.requires_grad_()

        # Setting up the optimizer
        optimizer = optim.Adam([x_adv], lr=self.learning_rate)

        for iteration in range(self.max_iterations):
            
            # print('Iteration:', iteration, end='\r')
            
            optimizer.zero_grad()

            # Forward pass
            img_inputs[0][0] = x_adv
            inputs = {'img_inputs': img_inputs, 'img_metas': img_metas}
            # try:
                # outputs = model(return_loss=False, rescale=True, adv_mode=True, **inputs)
            # except:
            outputs = model(return_loss=False, rescale=True, **inputs)

            # Assign results and compute adversarial loss
            assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
            if assign_results is None:
                break  # End if no assignment results are found

            loss_adv = self.loss_fn(**assign_results)

            # Adding the L2 distance term to the loss
            l2_loss = nn.MSELoss()(x_adv, img_)
            loss = loss_adv + self.initial_const * l2_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Projecting the perturbed image to be within the valid range
            x_adv.data = torch.clamp(x_adv.data, self.lower.view(self.size), self.upper.view(self.size))

            torch.cuda.empty_cache()

        img_inputs[0][0] = x_adv.detach()
        return {'img_inputs': img_inputs, 'img_metas': img_metas}

