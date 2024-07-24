from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image

class BaseDataset_new(torch.utils.data.Dataset):
    def __init__(self, path_x,path_y, mode, transform = None, autoencoderType = None):
        self.path_x = path_x
        self.path_y = path_y
        self.mode = mode
        self.transform = transform
        self.autoencoderType = autoencoderType
        self.xs, self.ys, self.I = [], [], []

    def nb_classes(self):
        print(set(self.ys),set(self.classes))
        assert set(np.array(self.ys)) == set(self.classes)
        return len(set(self.classes))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        # print(index,len(self.xs))
        if(self.transform):

            x = np.load(self.path_x)[self.I[index]]
            # x = self.xs[index]
            # print(x.shape, self.I[index])
            # print(np.transpose(x, (0,2, 1)).shape)
            x = torch.Tensor(np.transpose(x, (1,0)))
            # x = x.unsqueeze(0)
            
            mean = torch.mean(x, dim = 1).unsqueeze(-1)
            std = torch.std(x, dim = 1).unsqueeze(-1)
            
            x = torch.concat((mean,std, torch.corrcoef(x)), dim = 1)
            
            y = self.ys[index]
        else:
            x = np.load(self.path_x)[self.I[index]]
            x = torch.Tensor(np.transpose(x, (1, 0)))
            y = self.ys[index]

            if(self.autoencoderType):
                if(self.autoencoderType == 'denoising'):
                    noise_factor = 0.5
                    noisy_x = torch.FloatTensor(noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape))
                    return noisy_x, y, x, index
        
            
   
        return x, y, index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        # self.im_paths = [self.im_paths[i] for i in I]