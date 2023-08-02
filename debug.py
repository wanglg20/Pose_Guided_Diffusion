
from utils.dataset import *
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

test_ds = RealEstate_dataset(info_root = 'dataset/RealEstate10K', data_root = 'dataset/RealEstate_data', mode = 'test')
loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
example = None
for idx, data in enumerate(loader):
    example = data 
    break

img_s, img_t, K_s, R_rel, t_rel = example

from models.epipolar import *

W_Mat = batch_epipolar_weight_Mat((64, 64), K_s, R_rel, t_rel)
print(W_Mat.shape)