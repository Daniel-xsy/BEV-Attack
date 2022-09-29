#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Patch utils file
#---------------------------------------------------------------------
#---------------------------------------------------------------------

import torch
import torch.nn as nn
import pickle
import torchvision
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import math
import kornia
import yaml

import scipy.misc as misc
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os


from PIL import Image
import imageio
# import scipy.misc 
# import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import time

#from ptod.coco_utils import encode_detection_results


#---------------------------------------------------------------------
# patch clipping class
#---------------------------------------------------------------------
class PatchConstraints(object):
    def __init__(self, set_loader):
        self.max_val = set_loader.max_val
        self.min_val = set_loader.min_val
        print("patch constraints (min-max): " + str(self.min_val) + " - " + str(self.max_val))
        return
    
    def __call__(self,module):
        if hasattr(module,'patch'):
            w=module.patch.data # NCWH
            w[:,0,:,:] = w[:,0,:,:].clamp(self.min_val[0], self.max_val[0])
            w[:,1,:,:] = w[:,1,:,:].clamp(self.min_val[1], self.max_val[1])
            w[:,2,:,:] = w[:,2,:,:].clamp(self.min_val[2], self.max_val[2])
            module.patch.data=w
    


#---------------------------------------------------------------------
# patches clipping class
#---------------------------------------------------------------------
class PatchesConstraints(object):
    def __init__(self, set_loader):
        self.max_val = set_loader.max_val
        self.min_val = set_loader.min_val
        print("patch constraints (min-max): " + str(self.min_val) + " - " + str(self.max_val))
        return
    
    def __call__(self,module):
        if hasattr(module,'patches'):
            for i,patch in enumerate(module.patches):
                w = patch.data #NCWH
                w[:,0,:,:] = w[:,0,:,:].clamp(self.min_val[0], self.max_val[0])
                w[:,1,:,:] = w[:,1,:,:].clamp(self.min_val[1], self.max_val[1])
                w[:,2,:,:] = w[:,2,:,:].clamp(self.min_val[2], self.max_val[2])
                module.patches[i].data=w


#---------------------------------------------------------------------
# patch_params
#---------------------------------------------------------------------
class patch_params(object):
    def __init__(self, 
        x_default = 0, 
        y_default = 0,
        noise_magn_percent = 0.05, 
        eps_x_translation = 1.0, 
        eps_y_translation = 1.0,
        max_scaling = 1.2, 
        min_scaling = 0.8,
        set_loader = None,
        rw_transformations = False):

            self.x_default = x_default
            self.y_default = y_default
            self.eps_x_translation = eps_x_translation
            self.eps_y_translation = eps_y_translation

            self.rw_transformations = rw_transformations

            self.set_loader = set_loader

            self.noise_magn_percent = noise_magn_percent
            self.noise_magn = np.max(np.abs(self.set_loader.max_val - self.set_loader.min_val)) * \
                self.noise_magn_percent
            print("noise mangitude: " + str(self.noise_magn))

            self.max_scaling =  max_scaling
            self.min_scaling =  min_scaling


#---------------------------------------------------------------------
# export the patch as numpy
#---------------------------------------------------------------------
def create_experiment_folders(cfg):
    exp_folder = cfg['adv_patch']['path']["out_dir"]
    exp_name = cfg['adv_patch']['path']["exp_name"]
    exp_root = os.path.join(exp_folder, exp_name)
    patches_folder = os.path.join(exp_root, "patches")
    if exp_name in os.listdir(exp_folder):
        #input("The folder %s already exists. If you want to overwrite it, press Enter. Otherwise ctrl-c to exit" % exp_root)
        #input("Are you sure? Do you really want to overwrite folder %s?" % exp_root)
        os.popen("rm -r %s" % exp_root)
        print("Folder %s deleted!" % exp_root)
        time.sleep(1) # required to complete rm request
    os.mkdir(exp_root)
    
    if "patches" not in os.listdir(exp_root):
        os.mkdir(patches_folder)
        
    with open(os.path.join(exp_root, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    



#---------------------------------------------------------------------
# export the patch as numpy
#---------------------------------------------------------------------
def save_patch_numpy(patch, path):
    patch_np = patch.detach().cpu().numpy()
    with open(path, 'wb') as f:
        pickle.dump(patch_np, f)




#---------------------------------------------------------------------
# export an obj into a pkl file
#---------------------------------------------------------------------
def save_obj(path, obj = None):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)




#---------------------------------------------------------------------
# Import a new patch from a png value
#---------------------------------------------------------------------
def get_patch_from_img(path,set_loader):
    patch = imageio.imread(path)
    patch = set_loader.image_transform(patch, resize=False)
    patch = np.expand_dims(patch, 0)
    patch = torch.from_numpy(patch).float()
    return patch



#---------------------------------------------------------------------
# Import N new patches from a png value
#---------------------------------------------------------------------
def get_N_patches_from_img(path,set_loader, N=2):
    patches = []
    for i in range(N):
        patch = get_patch_from_img(path=path + '_%d.png' % i, set_loader =set_loader)
        patches.append(patch)
    return patches



#---------------------------------------------------------------------
# Create a random patch 
#---------------------------------------------------------------------
def get_random_patch(cfg_patch, set_loader):
    patch = torch.rand(1,3, cfg_patch['height'], cfg_patch['width'])
    if set_loader.img_norm == False:
        patch *= 255.0  
    patch[:,0,:,:] -= set_loader.mean[0]
    patch[:,1,:,:] -= set_loader.mean[1]
    patch[:,2,:,:] -= set_loader.mean[2]
    patch[:,0,:,:] /= set_loader.std[0]
    patch[:,1,:,:] /= set_loader.std[1]
    patch[:,2,:,:] /= set_loader.std[2]
    return patch



#---------------------------------------------------------------------
# Create a N-random patches
#---------------------------------------------------------------------
def get_N_random_patches(cfg_patch, set_loader, N=2, resize=True):
    patches = []
    # compute the width to keep the same side ratio and overall area of a single big patch
    if resize:
        height = int(cfg_patch['height'] / math.sqrt(N))
        width = 2 * height
    else:
        height = int(cfg_patch['height'])
        width = int(cfg_patch['width'])
    print("Patch size: " + str(height))
    patch_size = {'height': height, 'width': width}
    for i in range(N):
        patch = get_random_patch(patch_size, set_loader)
        patches.append(patch)
    return patches





#---------------------------------------------------------------------
# Import the patch from a numpy file
#---------------------------------------------------------------------
def get_patch_from_numpy(path):
    print("retrieving patch from: " + cfg_patch['path'])
    with open(path, 'rb') as f:
        patch = torch.from_numpy(pickle.load(f))
    return patch



#---------------------------------------------------------------------
# Remove mask from a batch of images
#---------------------------------------------------------------------
def remove_mask (images,mask):
    mask = F.interpolate(mask, size=images.shape[1:],mode='bilinear', align_corners=True)
    images[mask.squeeze(1)==1] = 255.0
    return images




#---------------------------------------------------------------------
# Add the patch_obj as a new model parameter
#---------------------------------------------------------------------
def init_model_patch(model, mode = "train", seed_patch = None):
    # add new attribute into the model class
    setattr(model, "patch", None)
    # patch initialization
    if mode =='train':                    
        model.patch = nn.Parameter(seed_patch, requires_grad=True)
    # load an already trained patch for testing
    elif mode =='test':
        model.patch = nn.Parameter(seed_patch, requires_grad=False)
    



#---------------------------------------------------------------------
# Add two patches as new model parameters
#---------------------------------------------------------------------
def init_model_N_patches(model, mode = "train", N = 2, seed_patches = None):
    # add new attribute into the model class
    list_patches = []
    if mode =='train':       
        list_patches = [ nn.Parameter(seed_patches[i], requires_grad=True) for i in range(N)]           
    elif mode =='test':
        list_patches = [ nn.Parameter(seed_patches[i], requires_grad=False) for i in range(N)] 
    setattr(model, "patches", None)  
    model.patches = nn.ParameterList(list_patches)   



#---------------------------------------------------------------------
# Set multiple output during the eval mode
# The new attribute specify in the corresponding network model to return 
# also the auxiliary outputs which are common usually used during the 
# train mode
#---------------------------------------------------------------------
def set_multiple_output(model):
    # add new attribute into the model class
    setattr(model, "multiple_outputs", True)




#---------------------------------------------------------------------
# add the patch using add_patch() to each tensor image in the mini-batch 
#---------------------------------------------------------------------
def add_patch_to_batch(
    images, 
    patch,
    patch_params,
    device='cuda', 
    use_transformations=True, 
    int_filtering=False):

    patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])

    for k in range(images.size(0)):
        images[k], patch_mask[k]= add_patch(
            image=images[k], 
            patch=patch, 
            patch_params = patch_params,
            use_transformations = use_transformations,
            int_filtering=int_filtering,
            device=device)

    return images, patch_mask





#---------------------------------------------------------------------
# add patches using add_patch() to each tensor image in the mini-batch 
#---------------------------------------------------------------------
def add_N_patches_to_batch(
    images, 
    patches, 
    patch_params_array, 
    device = 'cuda', 
    use_transformations = True,
    int_filtering=False):
    
    patches_mask = torch.zeros([images.shape[0], 1, images.shape[2], images.shape[3]])

    for i, patch in enumerate(patches):
        patch_params = patch_params_array[i]

        # mask corresponding to a single mask
        patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
        for k in range(images.size(0)):
            images[k], patch_mask[k] = add_patch(
                image=images[k], 
                patch=patch, 
                patch_params = patch_params,
                use_transformations = use_transformations,
                int_filtering=int_filtering,
                device=device)

        # add multiple patch together
        patches_mask += patch_mask 
    
    # TODO - check for collision between multiple mask
    return images, patches_mask




#---------------------------------------------------------------------
# given a single tensor_image, this function creates a patched_image as a
# composition of the original input image and the patch using masks for
# keep everything differentable
#---------------------------------------------------------------------
def add_patch(image, 
    patch, 
    patch_params,
    device='cuda', 
    use_transformations=True, 
    int_filtering=False):

    applied_patch, patch_mask, img_mask, x_location, y_location = mask_generation(
            mask_type='rectangle', 
            patch=patch, 
            patch_params = patch_params,
            image_size=image.shape[:], 
            use_transformations = use_transformations,
            int_filtering=int_filtering,
            device=device)

    patch_mask = Variable(patch_mask, requires_grad=False).to(device)
    img_mask = Variable(img_mask, requires_grad=False).to(device)

    perturbated_image = torch.mul(applied_patch.type(torch.FloatTensor), patch_mask.type(torch.FloatTensor)) + \
        torch.mul(img_mask.type(torch.FloatTensor), image.type(torch.FloatTensor))
    
    return perturbated_image, patch_mask[0,:,:]






#---------------------------------------------------------------------
# TRANSFORMATION : Rotation
# the actual rotation angle is rotation_angle * 90 on all the 3 channels
# TODO: reimplement from scratch. 
#---------------------------------------------------------------------
def rotate_patch(in_patch):
    rotation_angle = np.random.choice(4)
    for i in range(0, rotation_angle):
        in_patch = in_patch.transpose(2,3).flip(3)
    return in_patch





#---------------------------------------------------------------------
# TRANSFORMATION: patch scaling
#---------------------------------------------------------------------
def random_scale_patch(patch, patch_params):
    scaling_factor = np.random.uniform(low=patch_params.min_scaling, high=patch_params.max_scaling)
    new_size_y = int(scaling_factor * patch.shape[2])
    new_size_x = int(scaling_factor * patch.shape[3])
    patch = F.interpolate(patch, size=(new_size_y, new_size_x), mode="bilinear", align_corners=True)
    return patch





#---------------------------------------------------------------------
# TRANSFORMATION: translation
# scale the patch (define the methodologies)
#---------------------------------------------------------------------
def random_pos(patch, image_size):
    x_location, y_location = int(image_size[2]) , int(image_size[1])
    x_location = np.random.randint(low=0, high=x_location - patch.shape[3])
    y_location = np.random.randint(low=0, high=y_location - patch.shape[2])
    return x_location, y_location





#---------------------------------------------------------------------
# TRANSFORMATION: translation
# scale the patch (define the methodologies)
#---------------------------------------------------------------------
def random_pos_local(patch, x_pos, y_pos, patch_params):
    eps_x = patch_params.eps_x_translation
    eps_y= patch_params.eps_y_translation
    x_location = np.random.randint(low= x_pos - eps_x, high=x_pos + eps_x)
    y_location = np.random.randint(low= y_pos - eps_y, high=y_pos + eps_y)
    return x_location, y_location




#---------------------------------------------------------------------
# TRANSFORMATION: uniform noise
#---------------------------------------------------------------------
def unif_noise(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255, device='cuda'):
    magn = patch_params.noise_magn
    noise = (-magnitude*2)* torch.rand(patch.size(), requires_grad=False).to(device) + magnitude
    patch_noise = (torch.clamp(((patch + noise) * std + mean), 0, max_val) - mean) / std
    return patch_noise

#---------------------------------------------------------------------
# TRANSFORMATION: gaussian noise
#---------------------------------------------------------------------
def gaussian_noise(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255, device='cuda'):
    noise = magnitude * torch.randn(patch.size(), requires_grad=False).to(device)
    patch_noise = (torch.clamp(((patch + noise) * std + mean), 0, max_val) - mean) / std
    return patch_noise

'''
#---------------------------------------------------------------------
# TRANSFORMATION: int filtering
#---------------------------------------------------------------------
def get_integer_patch(patch, patch_params, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    
    int_patch = patch.clone()
    if patch_params.set_loader.img_norm is False:
        int_patch = (torch.clamp(torch.round(int_patch * std + mean), 0, max_val) - mean) / std
#         int_patch = torch.round(int_patch)
    else:
        int_patch = (torch.clamp(torch.round(255 * (int_patch * std + mean)), 0, max_val) - mean) / std
#         int_patch *= 255.0 
#         int_patch = torch.round(int_patch)
#         int_patch /= 255.0
    return int_patch
'''


#---------------------------------------------------------------------
# TRANSFORMATION: int filtering
#---------------------------------------------------------------------
def get_integer_patch(patch, patch_param, device='cuda'):
    
    mean = torch.Tensor(patch_param.set_loader.mean.reshape((1, 3, 1, 1))).to(device) #
    std = torch.Tensor(patch_param.set_loader.std.reshape((1, 3, 1, 1))).to(device) #
    int_patch = patch.clone()
    if patch_param.set_loader.img_norm is False:
        int_patch = (torch.clamp(torch.round(patch * std + mean), 0, 255.) - mean) / std
    else:
        int_patch = (torch.clamp(torch.round(255 * (patch * std + mean)), 0, 255.)/255.0 - mean) / std
#         int_patch *= 255.0 
#         int_patch = torch.round(int_patch)
#         int_patch /= 255.0
    return int_patch

#---------------------------------------------------------------------
# TRANSFORMATION: contrast change
#---------------------------------------------------------------------
def contrast_change(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    contr_delta = 1 + magnitude * torch.randn(1).numpy()[0]
    patch = (torch.clamp((patch * std + mean) * contr_delta, 0, max_val) - mean) / std
    return patch


#---------------------------------------------------------------------
# TRANSFORMATION: brightness change
#---------------------------------------------------------------------
def brightness_change(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    bright_delta = magnitude * torch.randn(1).numpy()[0]
    patch = (torch.clamp((patch * std + mean) + bright_delta, 0, max_val) - mean) / std
#     for c in range(3):
#         patch[:, c, :, :] = torch.clamp(patch[:, c, :, :] + bright_delta, -mean[0, c, 0, 0].cpu().numpy(), 255-mean[0, c, 0, 0].cpu().numpy())
    return patch



#---------------------------------------------------------------------
# util for patch projection
#---------------------------------------------------------------------
def get_dest_corners(patch, extrinsic, intrinsic, pixel_dim=0.2, offset=[0, 0, 0], device='cuda'):
    # Define corners of each pixel of the patch (sign reference frame)
    p_h, p_w = patch.shape[2:]
    x, y, z = offset
    patch_corners = torch.Tensor([[[x, y, z, 1], 
                         [x, y - p_h*pixel_dim, z, 1],
                         [x, y - p_h*pixel_dim, -p_w*pixel_dim + z, 1],
                         [x, y, -p_w*pixel_dim + z, 1]]]).to(device)
    p = torch.transpose(patch_corners, 1, 2)
    # print(patch_corners)
    
    # Transform to camera reference frame
    corners_points_homogeneous = extrinsic @ p
    corners_points_3d = corners_points_homogeneous[:, :-1, :] / corners_points_homogeneous[:, -1:, :]

    # Project onto image
    corner_pixels_homogeneous = intrinsic @ corners_points_3d
    corner_pixels = corner_pixels_homogeneous[:, :-1, :] / corner_pixels_homogeneous[:, -1:, :]

    return torch.transpose(corner_pixels, 1, 2)




#---------------------------------------------------------------------
# patch projection for specific attack
#---------------------------------------------------------------------
def project_patch(im, patch, extrinsic, intrinsic, patch_params, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'):
    use_transformations, int_filtering = True, False
    mean = torch.Tensor(patch_params.set_loader.mean.reshape((1, 3, 1, 1))).to(device) #
    std = torch.Tensor(patch_params.set_loader.std.reshape((1, 3, 1, 1))).to(device) #
    max_val = 255
    if patch_params.set_loader.img_norm:
        max_val = 1
    # print(mean, std, max_val)
    p_h, p_w = patch.shape[2:]
    h, w = im.shape[-2:]
    
    if use_transformations is True:
        patch = gaussian_noise(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val, device=device)
        patch = brightness_change(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
        patch = contrast_change(patch, magnitude=0.1, mean=mean, std=std, max_val=max_val)
        # Blur
        window_size = np.random.randint(3, 7)
        patch = kornia.filters.box_blur(patch, (window_size, window_size))
    if int_filtering is True:
        patch = get_integer_patch(patch, patch_params, mean=mean, std=std, max_val=max_val, device=device)
        
    
    # if p_h != p_w:
    #     im_p = im
    #     for i in range(2):
    #         points_src = torch.Tensor([[
    #             [0, 0], [p_h, 0.], [p_h, p_w//2], [0., p_w//2],
    #         ]]).to(device)
    #         x_off, y_off, z_off = offset
        
    #         points_dst = get_dest_corners(patch[:, :, :, i*p_w//2:(i+1)*p_w//2], extrinsic, intrinsic, pixel_dim=pixel_dim, offset=[x_off, y_off-i * p_w/2 * pixel_dim, z_off], device=device)

    #         # compute perspective transform
    #         M: torch.Tensor = kornia.geometry.get_perspective_transform(points_src, points_dst).to(device)
    #         # warp the original image by the found transform
    #         data_warp: torch.Tensor = kornia.geometry.warp_perspective((patch[:, :, :, i*p_w//2:(i+1)*p_w//2].float() * std) + mean, M, dsize=(h, w))

    #         mask = torch.zeros_like(data_warp[0], device=device)
    #         mask[data_warp[0] > 0] = 1
    #         data_warp = ((data_warp - mean)/std)[0]

    #         mask_img = torch.ones((h, w), device=device) - mask

    #         im_p = im_p * mask_img  + data_warp * mask
        
        
    # else:
        
    points_src = torch.Tensor([[
        [0, 0], [p_h, 0.], [p_h, p_w], [0., p_w],
    ]]).to(device)

    points_dst = get_dest_corners(patch, extrinsic, intrinsic, pixel_dim=pixel_dim, offset=offset, device=device)

    # compute perspective transform
    M: torch.Tensor = kornia.geometry.get_perspective_transform(points_src, points_dst).to(device)
    # warp the original image by the found transform
    data_warp: torch.Tensor = kornia.geometry.warp_perspective((patch.float() * std) + mean, M, dsize=(h, w))

    mask = torch.zeros_like(data_warp[0], device=device)
    mask[data_warp[0] > 0] = 1
    data_warp = ((data_warp - mean)/std)[0]

    mask_img = torch.ones((h, w), device=device) - mask

    im_p = im * mask_img  + data_warp * mask
    
    if torch.sum(torch.isnan(im_p)) > 0:
        return im, torch.zeros_like(data_warp[0], device=device)
    return im_p, mask[0,:,:]


#---------------------------------------------------------------------
# REPROJECT PATCH ONTO IMAGE (BATCH VERSION)
#---------------------------------------------------------------------
def project_patch_batch(images, patch, extrinsic, intrinsic, patch_params, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'):# mean=[0, 0, 0], std=[1, 1, 1]):   
    patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
    for j in range(images.shape[0]):
        images[j], patch_mask[j] = project_patch(images[j], patch, extrinsic[j], intrinsic[j], 
                                      pixel_dim=pixel_dim, offset=offset,rescale=rescale, device=device, patch_params=patch_params)
    return images, patch_mask


#---------------------------------------------------------------------
# TODO add patches using add_patch() to each tensor image in the mini-batch 
#---------------------------------------------------------------------
def project_N_patches_batch(
    images, 
    patches,
    extrinsics, 
    intrinsics,
    patch_params_array, 
    device = 'cuda', 
    pixel_dim=0.2, 
    offset=[0, 0, 0], 
    rescale=None):
    
    patches_mask = torch.zeros([images.shape[0], 1, images.shape[2], images.shape[3]])

    for i, patch in enumerate(patches):
        patch_params = patch_params_array[i]

        # mask corresponding to a single mask
        patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
        for k in range(images.size(0)):
#             print(extrinsics)
#             print(intrinsics)
            images[k], patches_mask[k] = project_patch(images[k], patch, extrinsics[i][k:k+1, :], intrinsics[i][k:k+1, :], 
                                      pixel_dim=pixel_dim, offset=offset,rescale=rescale, device=device, patch_params=patch_params)
            
#             images[k], patch_mask[k] = add_patch(
#                 image=images[k], 
#                 patch=patch, 
#                 patch_params = patch_params,
#                 use_transformations = use_transformations,
#                 int_filtering=int_filtering)

        # add multiple patch together
        patches_mask += patch_mask 
    
    # TODO - check for collision between multiple mask

    return images, patches_mask


#---------------------------------------------------------------------
# Apply transformation to the patch and generate masks
#---------------------------------------------------------------------
def mask_generation(
    patch,
    patch_params,
    mask_type='rectangle', 
    image_size=(3, 224, 224), 
    use_transformations = True,
    int_filtering=False,
    device='cuda'):

    mean = torch.Tensor(patch_params.set_loader.mean.reshape((1, 3, 1, 1))).to(device) #
    std = torch.Tensor(patch_params.set_loader.std.reshape((1, 3, 1, 1))).to(device) #
    max_val = 255.0
    if patch_params.set_loader.img_norm:
        max_val = 1.0
    
    x_location = patch_params.x_default
    y_location = patch_params.y_default
    applied_patch = torch.zeros(image_size, requires_grad=False).to(device)

    if use_transformations is True:
        patch = random_scale_patch(patch, patch_params)
        patch = gaussian_noise(patch, patch_params.noise_magn, mean=mean, std=std, max_val=max_val, device=device)

        if patch_params.rw_transformations is True:
            patch = brightness_change(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
            patch = contrast_change(patch, magnitude=0.1, mean=mean, std=std, max_val=max_val)

        x_location, y_location = random_pos_local(patch, x_pos = x_location, y_pos = y_location, patch_params=patch_params)
        #patch = rotate_patch(patch)


    if int_filtering is True:
        patch = get_integer_patch(patch, patch_params, device=device)
    applied_patch[:,  y_location:y_location + patch.shape[2], x_location:x_location + patch.shape[3]] = patch[0]
    applied_patch_mask = torch.zeros_like(applied_patch).to(device)
    applied_patch_mask[:,  y_location:y_location + patch.shape[2], x_location:x_location + patch.shape[3]] = 1.0
    
    # patch_mask = applied_patch.clone()

    applied_patch_mask[applied_patch_mask != 0.0] = 1.0
    img_mask = torch.ones([3,image_size[1], image_size[2]]).to(device) - applied_patch_mask
    
    return applied_patch, applied_patch_mask, img_mask, x_location, y_location



#---------------------------------------------------------------------
# export a tensor as png (for visualization)
# similar to save_patch_png
#---------------------------------------------------------------------
#def save_tensor_png(im_tensor, path, bgr=True, img_norm=False, mean = 0.0):
def save_tensor_png(im_tensor, path, set_loader):
    im_data = im_tensor.clone().reshape(im_tensor.shape[1],im_tensor.shape[2],im_tensor.shape[3])
    im_data = im_data.detach().cpu().numpy()
    im_data = set_loader.to_image_transform(im_data)
    im_data = im_data.astype('uint8')
    data_img = Image.fromarray(im_data)
    print("save patch as img ", path)
    data_img.save(path)
    del im_data


#---------------------------------------------------------------------
# convert a tensor to png 
#---------------------------------------------------------------------
#def convert_tensor_image(im_tensor, bgr=True, img_norm=False, mean = 0.0):
def convert_tensor_image(im_tensor, set_loader):
    im_data = im_tensor.clone().reshape(im_tensor.shape[1],im_tensor.shape[2],im_tensor.shape[3])
    im_data = im_data.detach().cpu().numpy()
    im_data = set_loader.to_image_transform(im_data)
    im_data = im_data.astype('uint8')
    im_data = Image.fromarray(im_data)
    return im_data


#---------------------------------------------------------------------
# convert a tensor semantic segmentation to png 
#---------------------------------------------------------------------
def convert_tensor_SS_image(im_tensor, model_name = None, orig_size = None, set_loader = None):
    im_data = im_tensor.clone().reshape(im_tensor.shape[1],im_tensor.shape[2],im_tensor.shape[3])
    p_out = np.squeeze(im_tensor.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN", "deeplabv2"]:
        p_out = p_out.astype(np.float32)
         # float32 with F mode, resize back to orig_size
        p_out = misc.imresize(p_out, orig_size, "nearest", mode="F")
    
    decoded_p_out = set_loader.decode_segmap(p_out)
    return decoded_p_out




#---------------------------------------------------------------------
# export the patch as png (for visualization)
#---------------------------------------------------------------------
#def save_patch_png(patch, path, bgr=True, img_norm=False, mean = 0.0):
def save_patch_png(patch, path, set_loader):
    np_patch = patch.clone()

    #  (NCHW -> CHW -> HWC) 
    # print(np_patch.shape)
    np_patch = np_patch[0].detach().cpu().numpy()
    # print(np_patch.shape)
    np_patch = set_loader.to_image_transform(np_patch)
    np_patch = np_patch.astype('uint8')
    # print(np_patch.shape)
    patch_img = Image.fromarray(np_patch)
    print("save patch as img ", path)
    patch_img.save(path)
    del np_patch

    



#-------------------------------------------------------------------
# plot a subfigure for visualizing the adversarial patch effect
#-------------------------------------------------------------------
#def save_summary_img(tensor_list, path, model_name, orig_size, loader,  bgr=True, img_norm=False, count=0, imm_num=0):
def save_summary_img(tensor_list, path, model_name, orig_size, set_loader, count=0, img_num=0):
    p_image = tensor_list[0]
    c_image = tensor_list[1]
    p_out = tensor_list[2]
    c_out = tensor_list[3]

    #  (NCHW -> CHW -> HWC) 
    p_image = p_image.detach().cpu().numpy()
    c_image = c_image.detach().cpu().numpy()

    p_image = set_loader.to_image_transform(p_image)
    c_image = set_loader.to_image_transform(c_image)
        
    p_image = p_image.astype('uint8')
    c_image = c_image.astype('uint8')
    
    p_out = np.squeeze(p_out.data.max(1)[1].cpu().numpy(), axis=0)
    c_out = np.squeeze(c_out.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        p_out = p_out.astype(np.float32)
        c_out = c_out.astype(np.float32)
         # float32 with F mode, resize back to orig_size
        p_out = misc.imresize(p_out, orig_size, "nearest", mode="F")
        c_out = misc.imresize(c_out, orig_size, "nearest", mode="F")
    
    decoded_p_out = set_loader.decode_segmap(p_out)
    decoded_c_out = set_loader.decode_segmap(c_out)

    # clear and adversarial images and predictions
    fig, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(c_image)
    axarr[0,0].title.set_text('original image')
    axarr[0,1].imshow(decoded_c_out)
    axarr[0,1].title.set_text('original prediction')
    axarr[1,0].imshow(p_image)
    axarr[1,0].title.set_text('adversarial image')
    axarr[1,1].imshow(decoded_p_out)
    axarr[1,1].title.set_text('adversarial prediction')
    for ax in axarr.reshape(-1) : ax.set_axis_off()
    figname = os.path.join(path, "summary_patch%d_%d.png" % (count, img_num))
    fig.savefig(figname, bbox_inches='tight', dpi = 500) #high-quality
        
    print("summary_patch" + str(count) + "_" + str(img_num) + ".png" + " saved ")



def patch_mask2bbox(patch_mask):
    mask_2d = patch_mask[0]
    y_idx, x_idx = torch.where(mask_2d)[-2:]
    x_start, x_end = int(torch.min(x_idx)), int(torch.max(x_idx))
    y_start, y_end = int(torch.min(y_idx)), int(torch.max(y_idx))

    return [x_start, y_start, x_end - x_start, y_end - y_start]




def draw_bbox(ax, detections, factors=(1, 1), true_labels=False, debug=False):
    from ptod.coco_utils import CLASSES, COLORS, CONFIDENCE
    # print(len(detections), detections)
    for i in range(len(detections)):
        startX = detections[i]["bbox"][0] * factors[1]
        w = detections[i]["bbox"][2] * factors[1]
        startY = detections[i]["bbox"][1] * factors[0]
        h = detections[i]["bbox"][3] * factors[0]

        confidence = 1
        if not true_labels:
            confidence = detections[i]["score"]
        
        if confidence < CONFIDENCE:
            continue

        try:
            idx = int(detections[i]['category_id'].detach().cpu().numpy()) - 1
        except:
            idx = int(detections[i]['category_id']) - 1

        text_label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        
        rect = patches.Rectangle((startX, startY), w, h, linewidth=1, edgecolor=COLORS[idx], facecolor='none')
        y = startY - 5 if startY - 5 > 5 else startY + 5

        ax.add_patch(rect)
        ax.text(startX, y, text_label, c=COLORS[idx], fontsize=3) #'xx-small')
        if debug:
            print("[INFO] {}".format(text_label))
    return ax
        
def save_summary_img_od(clean_img, adv_img, clean_detections, adv_detections, true_label, path, model_name, orig_size, set_loader, count=0, img_num=0, factors=(1, 1)):
    from ptod.coco_utils import encode_detection_results
    #  (NCHW -> CHW -> HWC) 
    clean_img = clean_img.detach().cpu().numpy()
    adv_img = adv_img.detach().cpu().numpy()
    
    # clean_img = clean_img.detach().cpu().numpy().transpose((1, 2, 0))
    # adv_img = adv_img.detach().cpu().numpy().transpose((1, 2, 0))
        
    clean_img = set_loader.to_image_transform(clean_img).astype('uint8')
    adv_img = set_loader.to_image_transform(adv_img).astype('uint8')

    clean_res = encode_detection_results(clean_detections[0], true_label, factors=(1, 1))
    adv_res = encode_detection_results(adv_detections[0], true_label, factors=(1, 1))
    
    # clear and adversarial images and predictions
    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(clean_img)
    axarr[0].title.set_text('original image')
    axarr[0] = draw_bbox(axarr[0], true_label, factors=factors, true_labels=True)
    

    axarr[1].imshow(clean_img)
    axarr[1].title.set_text('original prediction')
    axarr[0] = draw_bbox(axarr[1], clean_res, factors=(1, 1), true_labels=False)

    axarr[2].imshow(adv_img)
    axarr[2].title.set_text('adversarial prediction')
    axarr[0] = draw_bbox(axarr[2], adv_res, factors=(1, 1), true_labels=False)

    """ idx = int(detections["labels"][i]) - 1
    #         print(idx)
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            rect = patches.Rectangle((startX, startY), endX-startX, endY-startY, linewidth=1, edgecolor=COLORS[idx], facecolor='none')
            ax.add_patch(rect)
    #         cv2.rectangle(orig, (startX, startY), (endX, endY),
    #             COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            ax.text(startX, y, label, c=COLORS[idx])

    
    idx = int(detections["labels"][i]) - 1
    #         print(idx)
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            rect = patches.Rectangle((startX, startY), endX-startX, endY-startY, linewidth=1, edgecolor=COLORS[idx], facecolor='none')
            ax.add_patch(rect)
    #         cv2.rectangle(orig, (startX, startY), (endX, endY),
    #             COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            ax.text(startX, y, label, c=COLORS[idx]) """
    # axarr[1,1].imshow(decoded_p_out)
    # axarr[1,1].title.set_text('adversarial prediction')
    for ax in axarr.reshape(-1) : ax.set_axis_off()
    figname = os.path.join(path, "summary_patch%d_%d.png" % (count, img_num))
    fig.savefig(figname, bbox_inches='tight', dpi = 500) #high-quality
        
    print("summary_patch" + str(count) + "_" + str(img_num) + ".png" + " saved ")



# Save summary image for depth estimation.
def save_summary_img_depth(tensor_list, path, model_name, orig_size, set_loader, count=0, img_num=0, patch_mask=None, max_depth=80):
    vmax = max_depth

    if patch_mask is not None:
        tensor_list[2][torch.where(patch_mask==1)] = 0
        tensor_list[3][torch.where(patch_mask==1)] = 0
        tensor_list[4][torch.where(patch_mask==1)[-2:]] = 0

    p_image = tensor_list[0].detach().cpu().numpy().transpose((1, 2, 0))
    c_image = tensor_list[1].detach().cpu().numpy().transpose((1, 2, 0))
    p_out = tensor_list[2][0].detach().cpu().numpy()
    c_out = tensor_list[3][0].detach().cpu().numpy()
    label = tensor_list[4].detach().cpu().numpy()

    def compute_error_map(predicted_depth_img, gt_depth_img):
        error_map = np.fabs(np.subtract(predicted_depth_img,gt_depth_img.astype(np.float32)))

        valid_values = gt_depth_img == 0
        error_map[valid_values] = 0

        num_valid_pixels = np.count_nonzero(error_map)
        mean_error_depth = np.sum(error_map) / num_valid_pixels
        # print("Mean error depth: {} meters | num valid pixels: {}".format(mean_error_depth, num_valid_pixels))

        return error_map, mean_error_depth

    # TODO include error maps.
    error_map, mean_error = compute_error_map(p_out, label)
    error_map_adv, mean_error_adv = compute_error_map(c_out, label)

    
    fig = plt.figure(figsize=(20, 8))

    # RGB orig
    fig.add_subplot(321)
    ii = plt.imshow(p_image)
    fig.colorbar(ii, fraction=0.02)
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # Original prediction
    fig.add_subplot(322)
    ii = plt.imshow(c_image, interpolation='nearest') #, vmax=70)
    fig.colorbar(ii, fraction=0.02)
    plt.title('Adversarial Image')
    plt.xticks([])
    plt.yticks([])

    # Original error map
    fig.add_subplot(323)
    ii = plt.imshow(p_out, vmax=vmax) #, vmax=70)
    fig.colorbar(ii, fraction=0.02)
    plt.title('Original Prediction')
    plt.xticks([])
    plt.yticks([])

    # Adv image
    fig.add_subplot(324)
    ii = plt.imshow(c_out, vmax=vmax)
    fig.colorbar(ii, fraction=0.02)
    plt.title('Adversarial Prediction')
    plt.xticks([])
    plt.yticks([])

    # Adversarial prediction
    fig.add_subplot(325)
    ii = plt.imshow(error_map, interpolation='nearest', vmax=vmax)
    fig.colorbar(ii, fraction=0.02)
    plt.title('Original Depth Error')
    plt.xticks([])
    plt.yticks([])

    # Adversarial Error Map
    fig.add_subplot(326)
    ii = plt.imshow(error_map_adv, interpolation='nearest', vmax=vmax) #, vmax=70)
    fig.colorbar(ii, fraction=0.02)
    plt.title('Adversarial Depth Error')
    plt.xticks([])
    plt.yticks([])

    # error map
    # fig.add_subplot(224)
    # ii = plt.imshow(error_map, interpolation='nearest',)
    # fig.colorbar(ii, fraction=0.02)
    # plt.title('Error map')
    # plt.xlabel("Mean error: {} meters". format(str(mean_error)))
    # plt.xticks([])
    # plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    plt.close()
    figname = os.path.join(path, "summary_patch%d_%d.png" % (count, img_num))
    fig.savefig(figname, bbox_inches='tight', dpi = 500) #high-quality
        
    print("summary_patch" + str(count) + "_" + str(img_num) + ".png" + " saved ")


def save_summary_img_3dod(
        clean_img, adv_img, path, model_name, orig_size, set_loader, count=0, img_num=0
    ):
    img_left, img_right, score = clean_img
    img_left_adv, img_right_adv, score_adv = adv_img

    im2show = np.block([[[img_left], [img_right]], [[img_left_adv], [img_right_adv]]])
    plt.imshow(im2show)
    plt.xticks([])
    plt.yticks([])
    figname = os.path.join(path, "summary_patch%d_%d.png" % (count, img_num))
    plt.savefig(figname, bbox_inches='tight', dpi = 500) #high-quality
    plt.close()  
    print("summary_patch" + str(count) + "_" + str(img_num) + ".png" + " saved ")

    
#---------------------------------------------------------------------
# Basic implementation of the neighrest neighbors labels 
# substitution 
# params:
# - in_label: original target to modify
# - targets: list of target classes to be removed in the output label
#---------------------------------------------------------------------
def remove_target_class(label, attacked, target, scale=1, maxd=250):
    if target == -1:
        # nearest neighbor
        label = nearest_neighbor(label, attacked, scale=scale, maxd=maxd)
    elif target == -2:
        label = untargeted_labeling(label, attacked)
    else:
#         print("Number of pixels labeled as class %d: %d" % (attacked, (label==attacked).sum()))
        label[label == attacked] = target
#         print("Number of pixels labeled as class %d: %d" % (attacked, (label==attacked).sum()))
    return label.long()


# Save rgb, gt, pred
def save_images_list(images_list, set_loader, task, path):
    rgb, gt, pred = images_list
    
    # Save gt according to task
    if 'depth' in task:
        rgb = rgb[0].detach().cpu().numpy().transpose((1, 2, 0))
        gt = gt[0].detach().cpu().numpy()
        pred = pred[0, 0].detach().cpu().numpy()  

        # Save raw rgb
        fig, ax = plt.subplots(1, sharex=True, sharey=True)
        ax.imshow(rgb)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'rgb.png'), bbox_inches='tight', dpi=300)
        plt.close()
                      
        np.save(os.path.join(path, 'gt.npy'), gt)
        # fig, ax = plt.subplots(1, sharex=True, sharey=True)
        # gt = np.log(gt)
        
        # ax.imshow(gt/gt.max(),cmap='hot', vmax=80)
        # ax.set_axis_off()
        # plt.savefig(os.path.join(path, 'gt.png'), bbox_inches='tight', dpi=300)
        # plt.close()
        np.save(os.path.join(path, 'pred.npy'), pred)
        # fig, ax = plt.subplots(1, sharex=True, sharey=True) 
        
        # ax.imshow(pred,cmap='hot', vmax=80)
        # ax.set_axis_off()
        # plt.savefig(os.path.join(path, 'pred.png'), bbox_inches='tight', dpi=300)
        # plt.close()

    # Save pred according to task
    elif 'ss' in task:
        rgb = rgb[0].detach().cpu().numpy()
        rgb = set_loader.to_image_transform(rgb)   

        # Save raw rgb
        fig, ax = plt.subplots(1, sharex=True, sharey=True) 
        ax.imshow(rgb/255)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'rgb.png'), bbox_inches='tight', dpi=300)
        plt.close()

        pred = np.squeeze(pred.data.max(1)[1].cpu().numpy(), axis=0)
        pred = set_loader.decode_segmap(pred)
        # gt = np.squeeze(gt.data.max(1)[1].cpu().numpy(), axis=0)
        gt = set_loader.decode_segmap(gt.cpu().numpy())

        fig, ax = plt.subplots(1, sharex=True, sharey=True)
        ax.imshow(gt)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'gt.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(1, sharex=True, sharey=True)
        ax.imshow(pred)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'pred.png'), bbox_inches='tight', dpi=300)
        plt.close()

    elif '2dod' in task:
        from ptod.coco_utils import encode_detection_results
        #  (NCHW -> CHW -> HWC) 
        rgb = rgb[0].detach().cpu().numpy()
        rgb = set_loader.to_image_transform(rgb).astype('uint8')
        
        # Save raw rgb
        fig, ax = plt.subplots(1, sharex=True, sharey=True) 
        ax.imshow(rgb/255)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'rgb.png'), bbox_inches='tight', dpi=300)
        plt.close()

        pred = encode_detection_results(pred[0], gt, factors=(1, 1))
        
        fig, ax = plt.subplots(1, sharex=True, sharey=True)
        ax.imshow(rgb)
        ax = draw_bbox(ax, pred, factors=(1, 1), true_labels=False)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'pred.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(1, sharex=True, sharey=True)
        ax.imshow(rgb)
        ax = draw_bbox(ax, gt, factors=(1, 1), true_labels=True)
        ax.set_axis_off()
        plt.savefig(os.path.join(path, 'gt.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
        

        
    elif '3dod' in task:
        img_left, img_right, score = rgb
        img_left_adv, img_right_adv, score_adv = pred
        img_left = np.uint8(set_loader.imdb.to_image_transform(img_left[0].cpu().numpy(), img_resize=True))

        # im2show = np.block([[[img_left], [img_right]], [[img_left_adv], [img_right_adv]]])
        plt.imshow(img_left)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(path, 'rgb.png'), bbox_inches='tight', dpi=300)
        plt.close()  

        plt.imshow(img_left_adv)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(path, 'pred.png'), bbox_inches='tight', dpi=300)
        plt.close()  

        plt.imshow(gt)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(path, 'gt.png'), bbox_inches='tight', dpi=300)
        plt.close() 


#---------------------------------------------------------------------
# Remove mask from a batch of images
#---------------------------------------------------------------------
def remove_mask (images,mask):
    mask = F.interpolate(mask, size=images.shape[1:],mode='bilinear', align_corners=True)
    images[mask.squeeze(1)==1] = 250.0
    return images



def nearest_neighbor(label, attacked, maxd=250, scale=1):
    device = label.device
    
#     label = F.interpolate(label.float(), scale_factor=scale, mode='nearest')
    N, H, W = label.shape
    attacked_mask = (label == attacked)
    index = attacked_mask.nonzero(as_tuple=False)
    index_tuple = attacked_mask.nonzero(as_tuple=True)
    
#     nearest = torch.zeros((index.shape[0]), device=device, dtype=torch.long, requires_grad=False)
#     count_pixel = 0
    for n in range(N):
        pixels_in_image = (index_tuple[0] == n).sum()
        pixel_index = (index_tuple[0] == n).nonzero(as_tuple=True)[0]
        print("Find %d pixels in image %d" % (pixels_in_image, n))
#         print(pixels_in_image.detach().cpu().numpy())
        for i in np.random.permutation(int(pixels_in_image.detach().cpu().numpy())):
#             print("Finding %d-th nearest neighbor" % i)
            pixel_center = index[pixel_index[i], 1:]
#             print("_________________")
#             print(pixel_center)
#             print(maxd)
#             print(pixel_center[0] - maxd//2)
#             print(pixel_center[0] + maxd//2)
            # Consider limited area around pixel center
            min_i, max_i = pixel_center[0] - maxd//2, pixel_center[0] + maxd//2
            min_j, max_j = pixel_center[1] - maxd//2, pixel_center[1] + maxd//2
    
            corners_i = (torch.clip(min_i, 0, H), torch.clip(max_i, 0, H))
            corners_j = (torch.clip(min_j, 0, W), torch.clip(max_j, 0, W))
#             print(pixel_center)
#             print(max_i)
#             print(corners_i)
            
#             elim_min_i, elim_max_i = min_i - corners_i[0], max_i - corners_i[1]
#             elim_min_j, elim_max_j = min_j - corners_j[0], max_j - corners_j[1]
#             print(corners_i)
#             print(corners_j)
            h_, w_ = corners_i[1] - corners_i[0], corners_j[1] - corners_j[0]
#             print(h_)
#             print(w_)
            I = torch.tensor(range(corners_i[0], corners_i[1]), device=device, dtype=torch.int, requires_grad=False).reshape((h_, 1)).repeat_interleave(w_, axis=1) - pixel_center[0]
#             print(I)
            J = torch.tensor(range(corners_j[0], corners_j[1]), device=device, dtype=torch.int, requires_grad=False).reshape((1, w_)).repeat_interleave(h_, axis=0) - pixel_center[1]
#             print(J)
            D = I**2 + J**2 + maxd**2 * torch.where(label==attacked, 1, 0)[n, corners_i[0]:corners_i[1], corners_j[0]: corners_j[1]] + maxd**2 * torch.where(label>18, 1, 0)[n, corners_i[0]:corners_i[1], corners_j[0]: corners_j[1]]
            
            nearest_pix = (D==torch.min(D)).nonzero()[0] #[torch.argmin(D, axis=1)[0], torch.argmin(D, axis=0)[0]]
            nearest = [corners_i[0] + nearest_pix[0], corners_j[0] + nearest_pix[1]]
#             print(torch.argmin(D, axis=))
#             print(nearest)
#             print("_________________")
#             print(torch.argmin(D, axis=1))
#             print(torch.argmin(D, axis=0))
#             print(D)
#             print(nearest_pix)
#             print(nearest)
            label[n, pixel_center[0], pixel_center[1]] = label[n, nearest[0], nearest[1]]
            
            
#             labels_mod = torch.where(labels_mod==attacked, )
#             labels_considered = labels[corners_i[0]:corners_i[1], corners_j[0]:corners_j[1]]
            
            
        
            
            
            """
            found_nearest = False
            old_considered = pixel_center.clone()
            while d < 250 or not found_nearest:
                corners_i = (torch.clip(pixel_center[0] - d, 0, H), torch.clip(pixel_center[0] + d, 0, H))
                corners_j = (torch.clip(pixel_center[1] - d, 0, W), torch.clip(pixel_center[1] + d, 0, W))
                
                pixels_considered = label[n, corners_i[0]:corners_i[1]+1, corners_j[0]:corners_j[1]+1]
#                 pixels_considered = torch.cat((label[n, corners_i[0]:corners_i[1] + 1, corners_j[0]].reshape(-1), 
#                                               label[n, corners_i[0]:corners_i[1] + 1, corners_j[1]].reshape(-1)))
                print(pixels_considered.shape)
#                  = torch.cat((
#                     label[n, corners_i[0]:corners_i[1], corners_j[0]].reshape(-1)
#                     label[n, corners_i[0]:corners_i[1], corners_j[1]].reshape(-1)
#                     label[n, corners_i[0], corners_j[0] + 1: corners_j[1] - 1].reshape(-1)
#                     label[n, corners_i[1], corners_j[0] + 1: corners_j[1] - 1].reshape(-1)
#                 ))
                
                d = d + 1 + (d // 10)
                found_nearest = (pixels_considered != attacked).any()
                if found_nearest:
                    print("Found nearest")
                    print((pixels_considered != attacked).nonzero())
                    ll = label[n, (pixels_considered != attacked).nonzero()[0]]
                    print(ll)
                    label[n, pixel_center] = ll
           """ 
            
            
    return label


def untargeted_labeling(label, attacked):
    torch.where(label != attacked, label, 255)
#     label[label != attacked] = 255
    
    return label

    

'''
#-------------------------------------------------------------------
#
#-------------------------------------------------------------------
def change_labels_to_static(labels, static_label):
    print(labels)
    print(static_label)
    for i in range(labels.shape[0]):
        labels[i] = static_label
'''


'''
TO CHECK 
make a class for all the possible transformation

# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total

'''





'''
# ------------------------------------------------------------------
def create_img_mask(in_features, patch_mask):
    mask = torch.ones([3,in_features.size(1), in_features.size(2)])
    img_mask = mask - patch_mask
    return img_mask




# ------------------------------------------------------------------

def create_patch_mask(in_features, my_patch, cfg_ctrs):
    patch_size = 40 #TODO JUST NOW, TO BE CHANGED
    width = in_features.size(1)
    height = in_features.size(2)
    patch_mask = torch.zeros([3, width,height])

    p_w = patch_size + cfg.patch_x
    p_h = patch_size + cfg.patch_y
    patch_mask[:, int(cfg.patch_x):int(p_w), int(cfg.patch_y):int(p_h)]= 1

    return patch_mask
'''


'''
TO CHECK 
make a class for all the possible transformation

def transform_patch(width, x_shift, y_shift, im_scale, rot_in_degree):
    """
      If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
      then it maps the output point (x, y) to a transformed input point 
      (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
      where k = c0 x + c1 y + 1. 
      The transforms are inverted compared to the transform mapping input points to output points.
     """
    rot = float(rot_in_degree) /90. *(math.pi/2)
     
    # rotation matrix
    rot_matrix = np.array( 
                [[math.cos(rot), -math.sin(rot)],
                 [math.sin(rot), math.cos(rot)]] )

    # scale it
'''  



'''
TO CHECK 
'''
'''
def create_patch_mask_bbox(im_data, bbox, advpatch):
    width = im_data.size(1)
    height = im_data.size(2)
    patch_mask = torch.zeros([3,width,height])

    p_w = bbox[2]-bbox[0]
    p_h = bbox[3]-bbox[1]
    patch_mask[:, 0:p_w,0:p_h]=1
    return patch_mask
'''


'''
def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        # noise in the initial size (why???)
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch
'''


'''
#---------------------------------------------------------------------
# Basic implementation of the neighrest neighbors labels 
# substitution 
# params:
# - in_label: original target to modify
# - targets: list of target classes to be removed in the output label
#---------------------------------------------------------------------
def remove_target_class_with_NN(label, targets):
    i_t = 0
    for v1 in label:
        j_t = 0
        for e1 in v1:
            if(label[i_t][j_t] in target):
                # the element [i_t][j_t] correponds to a target class
                min_dist = 99999
                s_x, s_y = -1,-1
                    
                i = 0
                for vec in label:
                    j = 0
                    for elem in vec:
                        if(elem not in target):
                            distance = (i - i_t)**2 + (j - j_t)**2 
                            if(distance < min_dist):
                                min_dist = distance
                                s_x, s_y = j,i
                        j += 1
                    i += 1
                    
                label[i_t][j_t] = label[s_y][s_x]
                   
            j_t += 1
        i_t += 1
    return label
'''