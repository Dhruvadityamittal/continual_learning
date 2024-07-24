from .base_new import *
import torch
from torchvision import transforms, datasets
from sklearn.preprocessing import LabelEncoder

class ADL_10s(BaseDataset_new):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        self.path_train_o_x = self.root + 'x_train_initial.npy'   
        self.path_train_o_y = self.root + 'y_train_initial.npy'

        self.path_val_o_x = self.root + 'x_val_initial.npy'
        self.path_val_o_y = self.root + 'y_val_initial.npy'

        self.path_train_n_1_x = self.root + 'x_train_later.npy'
        self.path_train_n_1_y = self.root + 'y_train_later.npy'

        self.path_val_n_1_x = self.root + 'x_val_later.npy'
        self.path_val_n_1_y = self.root + 'y_val_later.npy'

        if self.mode == 'train_0':
            self.classes = range(0, 3)
            self.path_x = self.path_train_o_x
            self.path_y = self.path_train_o_y

        elif self.mode == 'train_1':
            # self.classes = range(0, 200)
            self.path_x = self.path_train_n_1_x
            self.path_y = self.path_train_n_1_y

        elif self.mode == 'eval_0':
            self.classes = range(0, 3)
            self.path_x = self.path_val_o_x
            self.path_y = self.path_val_o_y

        elif self.mode == 'eval_1':
            self.classes = range(0, 5)
            self.path_x = self.path_val_n_1_x
            self.path_y = self.path_val_n_1_y
        
        BaseDataset_new.__init__(self, self.path_x,self.path_y, self.mode, self.transform)

        
        
        le = LabelEncoder()
        le.fit(np.load(self.path_y))
        index = 0
        for x,y in zip(np.load(self.path_x),np.load(self.path_y)):            
            # self.xs += [x]
            self.ys += [le.transform([y])[0]]
            self.I += [index]
            index += 1
        