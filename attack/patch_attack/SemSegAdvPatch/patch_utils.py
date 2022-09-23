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


import scipy.misc as misc
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os


from PIL import Image
import imageio


import scipy.misc 
import cv2
import matplotlib.pyplot as plt


import matplotlib
import matplotlib.pyplot as plt


import time


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
            int_filtering=int_filtering)

    return images, patch_mask







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
            int_filtering=int_filtering)

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
def unif_noise(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    magn = patch_params.noise_magn
    noise = (-magnitude*2)* torch.rand(patch.size(), requires_grad=False).to('cuda') + magnitude
    patch_noise = (torch.clamp(((patch + noise) * std + mean), 0, max_val) - mean) / std
    return patch_noise

#---------------------------------------------------------------------
# TRANSFORMATION: gaussian noise
#---------------------------------------------------------------------
def gaussian_noise(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    noise = magnitude * torch.randn(patch.size(), requires_grad=False).to('cuda')
    patch_noise = (torch.clamp(((patch + noise) * std + mean), 0, max_val) - mean) / std
    return patch_noise



#---------------------------------------------------------------------
# TRANSFORMATION: int filtering
#---------------------------------------------------------------------
def get_integer_patch(patch, patch_param):
    
    mean = torch.Tensor(patch_param.set_loader.mean.reshape((1, 3, 1, 1))).to('cuda') #
    std = torch.Tensor(patch_param.set_loader.std.reshape((1, 3, 1, 1))).to('cuda') #
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
                         [x, y - p_w*pixel_dim, z, 1],
                         [x, y - p_w*pixel_dim, -p_h*pixel_dim + z, 1],
                         [x, y, -p_h*pixel_dim + z, 1]]]).to(device)
    p = torch.transpose(patch_corners, 1, 2)
    
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
        

    # Define corners of each pixel of the patch (sign reference frame)
    p_h, p_w = patch.shape[2:]
    h, w = im.shape[-2:]
    
    if use_transformations is True:
        patch = gaussian_noise(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
        patch = brightness_change(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
        patch = contrast_change(patch, magnitude=0.1, mean=mean, std=std, max_val=max_val)
    if int_filtering is True:
        patch = get_integer_patch(patch, patch_params, mean=mean, std=std, max_val=max_val)
        
    
    if p_h != p_w:
        im_p = im
        # project in blocks (required for KORNIA bug)
        for i in range(2):
#                 points_src = torch.Tensor([[
#                     [0, i * p_w//2], [p_h, i * p_w//2], [p_h, (i + 1) * p_w//2], [0., (i + 1) * p_w//2],
#                 ]]).to(device)
            points_src = torch.Tensor([[
                [0, 0], [p_h, 0.], [p_h, p_w//2], [0., p_w//2],
            ]]).to(device)
            x_off, y_off, z_off = offset
  
            points_dst = get_dest_corners(patch[:, :, :, i*p_w//2:(i+1)*p_w//2], extrinsic, intrinsic, pixel_dim=pixel_dim, offset=[x_off, y_off-i * p_w/2 * pixel_dim, z_off], device=device)

            # compute perspective transform
            M: torch.Tensor = kornia.get_perspective_transform(points_src, points_dst).to(device)
            # warp the original image by the found transform
            data_warp: torch.Tensor = kornia.warp_perspective((patch[:, :, :, i*p_w//2:(i+1)*p_w//2].float() * std) + mean, M, dsize=(h, w))

            mask = torch.zeros_like(data_warp[0], device=device)
            mask[data_warp[0] > 0] = 1
            data_warp = ((data_warp - mean)/std)[0]

            mask_img = torch.ones((h, w), device=device) - mask

            im_p = im_p * mask_img  + data_warp * mask
        
        
    else:
        
        points_src = torch.Tensor([[
            [0, 0], [p_h, 0.], [p_h, p_w], [0., p_w],
        ]]).to(device)

        points_dst = get_dest_corners(patch, extrinsic, intrinsic, pixel_dim=pixel_dim, offset=offset, device=device)

        # compute perspective transform
        M: torch.Tensor = kornia.get_perspective_transform(points_src, points_dst).to(device)
        # warp the original image by the found transform
        data_warp: torch.Tensor = kornia.warp_perspective((patch.float() * std) + mean, M, dsize=(h, w))

        mask = torch.zeros_like(data_warp[0], device=device)
        mask[data_warp[0] > 0] = 1
        data_warp = ((data_warp - mean)/std)[0]

        mask_img = torch.ones((h, w), device=device) - mask

        im_p = im * mask_img  + data_warp * mask
        
    if torch.sum(torch.isnan(im_p)) > 0:
        return im
    return im_p, mask[0,:,:]


#---------------------------------------------------------------------
# REPROJECT PATCH ONTO IMAGE (BATCH VERSION)
#---------------------------------------------------------------------
def project_patch_batch(images, patch, extrinsic, intrinsic, patch_params, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'):
    
    patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
    for j in range(images.shape[0]):
        images[j], patch_mask[j] = project_patch(images[j], patch, extrinsic[j], intrinsic[j], 
                                      pixel_dim=pixel_dim, offset=offset,rescale=rescale, device=device, patch_params=patch_params)
    return images, patch_mask




#---------------------------------------------------------------------
# Apply transformation to the patch and generate masks
#---------------------------------------------------------------------
def mask_generation(
    patch,
    patch_params,
    mask_type='rectangle', 
    image_size=(3, 224, 224), 
    use_transformations = True,
    int_filtering=False):

    mean = torch.Tensor(patch_params.set_loader.mean.reshape((1, 3, 1, 1))).to('cuda') #
    std = torch.Tensor(patch_params.set_loader.std.reshape((1, 3, 1, 1))).to('cuda') #
    max_val = 255.0
    if patch_params.set_loader.img_norm:
        max_val = 1.0
    
    x_location = patch_params.x_default
    y_location = patch_params.y_default
    applied_patch = torch.zeros(image_size, requires_grad=False).to('cuda')

    if use_transformations is True:
        patch = random_scale_patch(patch, patch_params)
        patch = gaussian_noise(patch, patch_params.noise_magn, mean=mean, std=std, max_val=max_val)

        if patch_params.rw_transformations is True:
            patch = brightness_change(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
            patch = contrast_change(patch, magnitude=0.1, mean=mean, std=std, max_val=max_val)

        x_location, y_location = random_pos_local(patch, x_pos = x_location, y_pos = y_location, patch_params=patch_params)
        #patch = rotate_patch(patch)


    if int_filtering is True:
        patch = get_integer_patch(patch, patch_params)
    applied_patch[:,  y_location:y_location + patch.shape[2], x_location:x_location + patch.shape[3]] = patch[0]

    patch_mask = applied_patch.clone()

    patch_mask[patch_mask != 0.0] = 1.0
    img_mask = torch.ones([3,image_size[1], image_size[2]]).to('cuda') - patch_mask

    return applied_patch, patch_mask, img_mask, x_location, y_location



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
    if model_name in ["pspnet", "icnet", "icnetBN"]:
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
    np_patch = np_patch[0].detach().cpu().numpy()
    np_patch = set_loader.to_image_transform(np_patch)
    np_patch = np_patch.astype('uint8')
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
    
    for n in range(N):
        pixels_in_image = (index_tuple[0] == n).sum()
        pixel_index = (index_tuple[0] == n).nonzero(as_tuple=True)[0]
        print("Find %d pixels in image %d" % (pixels_in_image, n))
#         print(pixels_in_image.detach().cpu().numpy())
        for i in np.random.permutation(int(pixels_in_image.detach().cpu().numpy())):
#             print("Finding %d-th nearest neighbor" % i)
            pixel_center = index[pixel_index[i], 1:]
            min_i, max_i = pixel_center[0] - maxd//2, pixel_center[0] + maxd//2
            min_j, max_j = pixel_center[1] - maxd//2, pixel_center[1] + maxd//2
    
            corners_i = (torch.clip(min_i, 0, H), torch.clip(max_i, 0, H))
            corners_j = (torch.clip(min_j, 0, W), torch.clip(max_j, 0, W))

            h_, w_ = corners_i[1] - corners_i[0], corners_j[1] - corners_j[0]

            I = torch.tensor(range(corners_i[0], corners_i[1]), device=device, dtype=torch.int, requires_grad=False).reshape((h_, 1)).repeat_interleave(w_, axis=1) - pixel_center[0]

            J = torch.tensor(range(corners_j[0], corners_j[1]), device=device, dtype=torch.int, requires_grad=False).reshape((1, w_)).repeat_interleave(h_, axis=0) - pixel_center[1]

            D = I**2 + J**2 + maxd**2 * torch.where(label==attacked, 1, 0)[n, corners_i[0]:corners_i[1], corners_j[0]: corners_j[1]] + maxd**2 * torch.where(label>18, 1, 0)[n, corners_i[0]:corners_i[1], corners_j[0]: corners_j[1]]
            
            nearest_pix = (D==torch.min(D)).nonzero()[0] #[torch.argmin(D, axis=1)[0], torch.argmin(D, axis=0)[0]]
            nearest = [corners_i[0] + nearest_pix[0], corners_j[0] + nearest_pix[1]]

            label[n, pixel_center[0], pixel_center[1]] = label[n, nearest[0], nearest[1]]
            
    return label



def untargeted_labeling(label, attacked):
    torch.where(label != attacked, label, 255)    
    return label

    