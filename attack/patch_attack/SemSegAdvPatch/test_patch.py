import yaml
import torch
import argparse
import timeit, time
import numpy as np
from collections import OrderedDict

from torch.utils import data
import os


import patch_utils as patch_utils
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.utils import get_model_state


torch.backends.cudnn.benchmark = True


#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------
def test_patch( cfg, 
                loader, 
                n_classes, 
                patch = None, 
                patch_params = None,
                output_file = None, 
                use_transformations=False):

    running_metrics = runningScore(n_classes)

    ex_clear_image, ex_adv_image      =  None, None
    ex_clear_out, ex_adv_out          =  None, None

    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)
#     model_dict = {"arch": model_name}
    model_dict = {"arch": cfg["model"]["arch"]}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)    
    patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    ex_index = 0
    iteration = 0

    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(loader):

        with torch.no_grad():

            images = images.to(device)
            if isinstance(labels, list):
                labels, extrinsic, intrinsic = labels
                extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)

            #-------------------------------------------------------------------
            # Compute an example image for visualiation
            #-------------------------------------------------------------------
            if ex_clear_image is None:
                ex_clear_image = images[ex_index].clone().reshape(1,images.shape[1], images.shape[2], images.shape[3])
                ex_clear_out = model(ex_clear_image)

                ex_adv_image = patch_utils.add_patch_to_batch(
                    ex_clear_image.clone(), 
                    patch = model.patch, 
                    device = device, 
                    use_transformations=use_transformations, 
                    patch_params=patch_params,
                    int_filtering=True)[0]

                ex_adv_out = model(ex_adv_image)
            #-------------------------------------------------------------------
            images, patch_mask = patch_utils.add_patch_to_batch(
                    images, 
                    patch = model.patch, 
                    device = device, 
                    use_transformations=use_transformations, 
                    patch_params=patch_params,
                    int_filtering=True)

            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
            
            gt_woPatch = patch_utils.remove_mask(labels, patch_mask.detach())
            running_metrics.update(gt_woPatch.numpy(), pred)
#             gt = labels.numpy()
#             running_metrics.update(gt, pred)
            
            # comment to validate on the entire dataset

        

    score, class_iou = running_metrics.get_scores()


    return score, class_iou, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out  


#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------
def test_rw_patch( cfg, 
                loader, 
                n_classes, 
                patch = None, 
                patch_params = None,
                output_file = None, 
                use_transformations=False):

    running_metrics = runningScore(n_classes)

    ex_clear_image, ex_adv_image      =  None, None
    ex_clear_out, ex_adv_out          =  None, None

    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)
#     model_dict = {"arch": model_name}
    model_dict = {"arch": cfg["model"]["arch"]}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)    
    patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    ex_index = 2
    iteration = 0

    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(loader):

        with torch.no_grad():

            images = images.to(device)
            if isinstance(labels, list):
                labels, extrinsic, intrinsic = labels
                extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)
            pred_outputs = model(images).data.max(1)[1].cpu().numpy()
            #-------------------------------------------------------------------
            # Compute an example image for visualiation
            #-------------------------------------------------------------------
            if ex_clear_image is None:
                ex_clear_image = images[ex_index].clone().reshape(1,images.shape[1], images.shape[2], images.shape[3])
                ex_clear_out = model(ex_clear_image)

                ex_adv_image = patch_utils.add_patch_to_batch(
                    ex_clear_image.clone(), 
                    patch = model.patch, 
                    device = device, 
                    use_transformations=use_transformations, 
                    patch_params=patch_params,
                    int_filtering=True)[0]

                ex_adv_out = model(ex_adv_image)
            #-------------------------------------------------------------------

            images = patch_utils.add_patch_to_batch(
                    images, 
                    patch = model.patch, 
                    device = device, 
                    use_transformations=use_transformations, 
                    patch_params=patch_params,
                    int_filtering=True)[0]

            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()

#             gt = labels.numpy()
            running_metrics.update(pred_outputs, pred)

        

    score, class_iou = running_metrics.get_scores()


    return score, class_iou, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out  


#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------
def test_specific_patch( cfg, 
                        loader, 
                        n_classes, 
                        patch, 
                        patch_params,
                        output_file = None, 
                        use_transformations=False, 
                        ):

    running_metrics = runningScore(n_classes)

#     ex_clear_image, ex_adv_image      =  None, None
#     ex_clear_out, ex_adv_out          =  None, None

    device = torch.device("cuda")
#     torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup Model and patch
    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)
#     model_dict = {"arch": model_name}
    model_dict = {"arch": cfg['model']['arch']}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)    
#     model = torch.nn.DataParallel(model, device_ids=[2, 3, 4])
    patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    ex_index = 1
    iteration = 0

    model.eval()
    model.to(device)
    
    p_w, real_width, offset = cfg['adv_patch']['attr']['width'], cfg['adv_patch']['attr']['world_width'], cfg['adv_patch']['attr']['offset']
    block_width, rescale = cfg['adv_patch']['attr']['block_width'], cfg['adv_patch']['attr']['rescale']
    pixel_width = real_width / p_w
    
    for i, (images, labels) in enumerate(loader):
        t = time.time()
        with torch.no_grad():
            images = images.to(device)
            labels, extrinsic, intrinsic = labels
            extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)
            
            adv_images = patch_utils.project_patch_batch(images.clone(), model.patch, extrinsic, intrinsic, 
                                                                   pixel_dim=pixel_width, offset=offset, 
                                                                   rescale=rescale, device=device, patch_params=patch_params)[0] #mean=loader.dataset.mean, std=loader.dataset.std)
            
#             adv_images = patch_utils.project_patch_blocks_batch(images, model.patch, extrinsic, intrinsic, pixel_width=pixel_width, 
#                                                   block_width=block_width, offset=offset, rescale=rescale, device=device)

            #-------------------------------------------------------------------
            
            outputs = model(adv_images)
            pred = outputs.data.max(1)[1].cpu().numpy()

            gt = labels.numpy()
            running_metrics.update(gt, pred)
        batch_time = time.time() - t
        print("Performed test on %d/%d batches... ETA: %.3f seconds" % (i+1, len(loader), (len(loader) - (i+1)) * batch_time), end='\r')
            
    score, class_iou = running_metrics.get_scores()

    return score, class_iou




#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def load_model(cfg, loader, n_classes, device='cuda'):
    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
#     model_dict = {"arch": model_name}
    model_dict = {"arch": cfg['model']['arch']}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)    
    return model 




#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def get_examples(cfg, 
                loader, 
                set_loader,
                n_classes, 
                patch = None, 
                patch_params = None,
                output_file = None, 
                num_batch = 5,
                use_transformations=False):

    device = torch.device("cuda")
#     torch.cuda.set_device(cfg["device"]["gpu"])

    model = load_model(cfg, loader, n_classes, device=device)

    if patch is not None:
        patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    model.eval()
    model.to(device)

    clear_images, adv_images, clear_outputs, adv_outputs  = None, None, None, None

    for i, (images, labels) in enumerate(loader):

        #if set_loader.bottom_crop > 0:
        #       images, labels = set_loader.crop_image(images, labels)

        with torch.no_grad():
            images = images.to(device)

            outputs = model(images)

            if patch is not None:
                patched_images = patch_utils.add_patch_to_batch(
                    images.clone(), 
                    model.patch.clone(), 
                    patch_params = patch_params,
                    device = device, 
                    use_transformations=use_transformations, 
                    int_filtering=True)[0]
            else:
                patched_images = images.clone()

            patched_outputs = model(patched_images)

            images = images.detach().cpu()
            outputs = outputs.detach().cpu()
            patched_images = patched_images.detach().cpu()
            patched_outputs = patched_outputs.detach().cpu()

            
            if clear_images is None:
                clear_images, clear_outputs = images.clone(), outputs.clone()
                adv_images, adv_outputs = patched_images.clone(), patched_outputs.clone()
                in_size = (-1, images.size(1), images.size(2), images.size(3))
                out_size = (-1, outputs.size(1), outputs.size(2), outputs.size(3))
            else:
                clear_images = torch.cat((clear_images, images))
                adv_images = torch.cat((adv_images, patched_images))
                clear_outputs = torch.cat((clear_outputs, outputs))
                adv_outputs = torch.cat((adv_outputs, patched_outputs))
   
        if(i >= num_batch -1):
            break
    


    return clear_images, adv_images, clear_outputs, adv_outputs




#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def get_specific_examples( cfg, 
                        loader, 
                        n_classes, 
                        patch_params,
                        patch = None, 
                        output_file = None, 
                        num_batch = 5):

    device = torch.device("cuda")
#     torch.cuda.set_device(cfg["device"]["gpu"])

    model = load_model(cfg, loader, n_classes)

    if patch is not None:
        patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    model.eval()
    model.to(device)

    clear_images, adv_images, clear_outptus, adv_outputs  = None, None, None, None
    p_w, real_width, offset = cfg['adv_patch']['attr']['width'], cfg['adv_patch']['attr']['world_width'], cfg['adv_patch']['attr']['offset']
    block_width, rescale = cfg['adv_patch']['attr']['block_width'], cfg['adv_patch']['attr']['rescale']
    pixel_width = real_width / p_w
    for i, (images, labels) in enumerate(loader):

        with torch.no_grad():
            images = images.to(device)
            if isinstance(labels, list):
                labels, extrinsic, intrinsic = labels
                extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)
            
            outputs = model(images)

            if patch is not None:
                patched_images = patch_utils.project_patch_batch(images.clone(), model.patch, extrinsic, intrinsic, 
                                                                   pixel_dim=pixel_width, offset=offset, 
                                                                   rescale=rescale, device=device, patch_params=patch_params)[0]# mean=loader.dataset.mean, std=loader.dataset.std)
#                 patched_images = patch_utils.project_patch_blocks_batch(images, model.patch, extrinsic, intrinsic, pixel_width=pixel_width, 
#                                                   block_width=block_width, offset=offset, rescale=rescale, device=device)
#                
            else:
                patched_images = images.clone()

            patched_outputs = model(patched_images)

            images = images.detach().cpu()
            outputs = outputs.detach().cpu()
            patched_images = patched_images.detach().cpu()
            patched_outputs = patched_outputs.detach().cpu()

            
            if clear_images is None:
                clear_images, clear_outputs = images.clone(), outputs.clone()
                adv_images, adv_outputs = patched_images.clone(), patched_outputs.clone()
                in_size = (-1, images.size(1), images.size(2), images.size(3))
                out_size = (-1, outputs.size(1), outputs.size(2), outputs.size(3))
            else:
                clear_images = torch.cat((clear_images, images))
                adv_images = torch.cat((adv_images, patched_images))
                clear_outputs = torch.cat((clear_outputs, outputs))
                adv_outputs = torch.cat((adv_outputs, patched_outputs))
   
        if(i >= num_batch - 1):
            break
    


    return clear_images, adv_images, clear_outputs, adv_outputs




#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def get_score_validation( cfg, 
                loader, 
                n_classes, 
                patch, 
                patch_params,
                use_transformations=False):
                
    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    running_metrics = runningScore(n_classes)

    model = load_model(cfg, loader, n_classes)

    if patch is not None:
        patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(loader):

        with torch.no_grad():

            images = images.to(device)
            if isinstance(labels, list):
                labels, extrinsic, intrinsic = labels
                extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)

            if patch is not None:
                images = patch_utils.add_patch_to_batch(
                    images.clone(), 
                    model.patch.clone(), 
                    patch_params = patch_params,
                    device = device, 
                    use_transformations=use_transformations, 
                    int_filtering=True)[0]

            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()

            gt = labels.numpy()
            running_metrics.update(gt, pred)


    score, class_iou = running_metrics.get_scores()
    return score, class_iou




#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def get_score_validation_specifc( cfg, 
                loader, 
                n_classes, 
                patch, 
                patch_params,
                use_transformations=False):
                
    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    running_metrics = runningScore(n_classes)

    model = load_model(cfg, loader, n_classes)

    if patch is not None:
        patch_utils.init_model_patch(model = model, mode = "test", seed_patch = patch)

    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(loader):

        with torch.no_grad():

            images = images.to(device)

            if patch is not None:
                images = patch_utils.add_patch_to_batch(
                    images.clone(), 
                    model.patch.clone(), 
                    patch_params = patch_params,
                    device = device, 
                    use_transformations=use_transformations, 
                    int_filtering=True)[0]

            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()

            gt = labels.numpy()
            running_metrics.update(gt, pred)


    score, class_iou = running_metrics.get_scores()
    return score, class_iou
