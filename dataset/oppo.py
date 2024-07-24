from .base_new import *
import torch
from torchvision import transforms, datasets
from sklearn.preprocessing import LabelEncoder

class oppo(BaseDataset_new):
    def __init__(self, root, mode, windowlen,transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        self.path_train_o_x = self.root + 'data_initial_step_scenario_1_x_train_windowLen_' + str(windowlen)+'.npy'   
        self.path_train_o_y = self.root + 'data_initial_step_scenario_1_y_train_windowLen_' + str(windowlen)+'.npy'

        self.path_val_o_x = self.root + 'data_initial_step_scenario_1_x_val_windowLen_' + str(windowlen)+'.npy'
        self.path_val_o_y = self.root + 'data_initial_step_scenario_1_y_val_windowLen_' + str(windowlen)+'.npy'

        self.path_train_n_1_x = self.root + 'data_increment_step_scenario_1_x_train_windowLen_' + str(windowlen)+'.npy'
        self.path_train_n_1_y = self.root + 'data_increment_step_scenario_1_y_train_windowLen_' + str(windowlen)+'.npy'

        self.path_val_n_1_x = self.root + 'data_increment_step_scenario_1_x_val_windowLen_' + str(windowlen)+'.npy'
        self.path_val_n_1_y = self.root + 'data_increment_step_scenario_1_y_val_windowLen_' + str(windowlen)+'.npy'

        self.path_test_x = self.root + 'data_increment_step_scenario_1_x_test_windowLen_' + str(windowlen)+'.npy'
        self.path_test_y = self.root + 'data_increment_step_scenario_1_y_test_windowLen_' + str(windowlen)+'.npy'

        if self.mode == 'train_0':
            self.classes = range(0, 13)
            self.path_x = self.path_train_o_x
            self.path_y = self.path_train_o_y

        elif self.mode == 'train_1':
            self.classes = range(0, 17)
            self.path_x = self.path_train_n_1_x
            self.path_y = self.path_train_n_1_y

        elif self.mode == 'eval_0':
            self.classes = range(0, 13)
            self.path_x = self.path_val_o_x
            self.path_y = self.path_val_o_y

        elif self.mode == 'eval_1':
            self.classes = range(0, 17)
            self.path_x = self.path_val_n_1_x
            self.path_y = self.path_val_n_1_y
        
        elif self.mode == 'test':
            self.classes = range(0, 8)
            self.path_x = self.path_test_x
            self.path_y = self.path_test_y
        
        BaseDataset_new.__init__(self, self.path_x,self.path_y, self.mode, self.transform)

        
        
        le = LabelEncoder()
        le.fit(np.load(self.path_y))
        index = 0
        for x,y in zip(np.load(self.path_x),np.load(self.path_y)):            
            # self.xs += [x]
            self.ys += [le.transform([y])[0]]
            self.I += [index]
            index += 1
        