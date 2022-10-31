import torch
import os
from glob import glob

model_folder = './models'
folder = os.listdir(model_folder)

for i in range(len(folder)):
    filenames = glob(os.path.join(model_folder, folder[i], '*.pth'))
    for j in range(len(filenames)):
        model = torch.load(filenames[j], map_location='cpu')
        backbone = 0
        neck = 0
        head = 0
        all = 0
        for key in list(model['state_dict'].keys()):
            if 'backbone' in key:
            # if key.startswith('img_backbone'):
                backbone += model['state_dict'][key].nelement()
            elif 'neck' in key:
                neck += model['state_dict'][key].nelement()
            elif 'head' in key:
                head += model['state_dict'][key].nelement()

            all += model['state_dict'][key].nelement()
        print(filenames[j])
        print(f"Backbone param: {backbone / 1e6}M")
        print(f"Neck param: {neck / 1e6}M")
        print(f"Head param: {head / 1e6}M")
        print(f"Total param: {all / 1e6}M")

# smaller 63374123
# v4 69140395
