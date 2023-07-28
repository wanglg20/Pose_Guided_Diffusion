import torchvision.datasets as datasets
import torch
from torch.utils.data.dataset import Dataset
import cv2
import random
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from dataset.utils import *


###############################
#   Training dataset          #
###############################

class RealEstate_dataset(Dataset):
    """
    dataset for RealEstate10k dataset
    Args:
        info_root: root of the dataset info, txt file
        data_root: root of images of the dataset
        mode: 'train' or 'test'
    """
    def __init__(self, info_root, data_root, mode = 'train',
                 img_size = (128, 128),
                 trans = transforms.Compose([
                    transforms.CenterCrop(128),      
                    transforms.ToTensor(),           
                    ])):
        super().__init__()  
        
        try:
            assert mode in ['train', 'test']
        except:
            print('mode must be train or test')
        
        self.info_root = info_root + '/' + mode + '/'
        self.data_root = data_root
        self.transform = trans
        
        self.img_size = img_size
        self.width = img_size[0]
        self.height = img_size[1]
        
        self.seq_set = os.listdir(self.data_root)

        self.total_len = self.seq_set.__len__()
        #for seq in self.seq_set:
        #    seq_path = self.info_root + '/' + seq
        #    with open(seq_path, 'r') as file:
        #        lines = file.readlines()
        #        self.total_len += len(lines) - 1
        #        file.close()
    
    def read_line(self, line, file_name):
        """
        given a line with 19 columns, return intrinsics, extrinsics, and image
        
        Args:
            line: a line of the txt file from RealEstate10k
            file_name: the name of the folder of the image
        return: image, K, R, t
        """
        values = list(line.strip().split())
        image_path = self.data_root + '/' + file_name + '/'  + values[0] + '.png'
        image = Image.open(image_path)
        image = self.transform(image)
        intrinsics = list(map(float, values[1:5]))
        value_pose1 = list(map(float, values[7:]))
        
        R = np.array([[value_pose1[0], value_pose1[1], value_pose1[2]],
                      [value_pose1[4], value_pose1[5], value_pose1[6]],
                      [value_pose1[8], value_pose1[9], value_pose1[10]]])
        
        t = np.array([value_pose1[3], value_pose1[7], value_pose1[11]])
        
        K = np.array([[intrinsics[0]*self.width, 0, intrinsics[2]*self.width],
                      [0, intrinsics[1]*self.height, intrinsics[3]*self.height],
                      [0, 0, 1]])
        return image, K, R, t
        
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        """
        for each seq, randomly choose a source frame adn a target frame as the input
        return: 
            img_s: source image
            img_t: target image
        """
        
        
        seq_path = self.info_root + '/' + self.seq_set[index] + '.txt'
        print(seq_path)
        with open(seq_path, 'r') as file:
            lines = file.readlines()
            lines = lines[1:]
        seq_len = len(lines)
        source_index = random.randint(0, seq_len - 2)
        target_index = random.randint(source_index + 1, seq_len - 1)
        
        source_line = lines[source_index + 1]
        target_iine = lines[target_index + 1]
        
        file_name = self.seq_set[index]
        
        (img_s, K_s, R_s, t_s) = self.read_line(source_line, file_name)
        (img_t, K_t, R_t, t_t) = self.read_line(target_iine, file_name)
        
        R_rel, t_rel = Calculate_Rel_Mat(R_s, t_s, R_t, t_t)
        R_rel = torch.from_numpy(R_rel).float()
        t_rel = torch.from_numpy(t_rel).float()
        K_s = torch.from_numpy(K_s).float()

        return img_s, img_t, K_s, R_rel, t_rel