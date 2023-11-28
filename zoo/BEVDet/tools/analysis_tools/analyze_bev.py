import pickle
import torch
import torch.nn.functional as F

with open('/root/autodl-tmp/models/BEV-Attack/zoo/BEVDet/clean_bev_feat.pkl', 'rb') as f:
    clean_bev = pickle.load(f)

with open('/root/autodl-tmp/models/BEV-Attack/zoo/BEVDet/adv_bev_feat.pkl', 'rb') as f:
    adv_bev = pickle.load(f)
    
with open('/root/autodl-tmp/models/BEV-Attack/zoo/BEVDet/clean_depth_feat.pkl', 'rb') as f:
    clean_depth = pickle.load(f)
    
with open('/root/autodl-tmp/models/BEV-Attack/zoo/BEVDet/adv_depth_feat.pkl', 'rb') as f:
    adv_depth = pickle.load(f)

with open('/root/autodl-tmp/models/BEV-Attack/zoo/BEVDet/clean_img_feat.pkl', 'rb') as f:
    clean_feat = pickle.load(f)
    
with open('/root/autodl-tmp/models/BEV-Attack/zoo/BEVDet/adv_img_feat.pkl', 'rb') as f:
    adv_feat = pickle.load(f)

clean_bev = torch.cat(clean_bev, dim=0)[:21]
adv_bev = torch.cat(adv_bev, dim=0)[:21]
clean_depth = torch.cat(clean_depth, dim=0)[:21]
adv_depth = torch.cat(adv_depth, dim=0)[:21]
clean_feat = torch.cat(clean_feat, dim=0)[:21]
adv_feat = torch.cat(adv_feat, dim=0)[:21]

eps = 1e-8

bev_error = torch.mean(((adv_bev - clean_bev) / (clean_bev + eps)).abs_())
depth_error = torch.mean(((adv_depth - clean_depth) / (clean_depth + eps)).abs_())
img_error = torch.mean(((adv_feat - clean_feat) / (clean_feat + eps)).abs_())


a = 1