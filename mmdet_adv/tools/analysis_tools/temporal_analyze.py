import pickle
import torch
import torch.nn.functional as F

clean_file = '/root/autodl-tmp/models/BEV-Attack/mmdet_adv/clean_prev_bev.pkl'
adv_file = '/root/autodl-tmp/models/BEV-Attack/mmdet_adv/adv_prev_bev.pkl'
clean_adv_file = '/root/autodl-tmp/models/BEV-Attack/mmdet_adv/clean_adv_prev_bev.pkl'

with open(clean_file, 'rb') as f:
    clean_data = pickle.load(f)
    
with open(adv_file, 'rb') as f:
    adv_data = pickle.load(f)
    
with open(clean_adv_file, 'rb') as f:
    clean_adv_data = pickle.load(f)
    

adv_mse_loss = []
clean_adv_mse_loss = []
for i in range(len(clean_adv_data)):
    clean_single = clean_data[i]
    adv_single = adv_data[i]
    clean_adv_single = clean_adv_data[i]
    adv_mse_loss.append(F.mse_loss(clean_single, adv_single, reduction='none').mean().item())
    clean_adv_mse_loss.append(F.mse_loss(clean_single, clean_adv_single, reduction='none').mean().item())
    
a = 1