from .base_new import *
import torch
from torchvision import transforms, datasets
from sklearn.preprocessing import LabelEncoder

class realworld(BaseDataset_new):
    def __init__(self, root, mode, windowlen,transform=None, autoencoderType = None,  standardize = None, fold = None):
        self.root = root
        self.mode = mode
        self.transform = transform

        if(standardize):
            wln = str(windowlen) + '_standardized_True'
        else:
            wln = str(windowlen)

        self.path_train_o_x = self.root + 'data_initial_step_scenario_1_x_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'     
        self.path_train_o_y = self.root + 'data_initial_step_scenario_1_y_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        self.path_val_o_x = self.root + 'data_initial_step_scenario_1_x_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_val_o_y = self.root + 'data_initial_step_scenario_1_y_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        self.path_test_o_x = self.root + 'data_initial_step_scenario_1_x_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_test_o_y = self.root + 'data_initial_step_scenario_1_y_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        self.path_train_n_1_x = self.root + 'data_incremental_step_scenario_1_x_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_train_n_1_y = self.root + 'data_incremental_step_scenario_1_y_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'    

        self.path_val_n_1_x = self.root + 'data_incremental_step_scenario_1_x_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_val_n_1_y = self.root + 'data_incremental_step_scenario_1_y_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        self.path_test_x = self.root + 'data_incremental_step_scenario_1_x_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_test_y = self.root + 'data_incremental_step_scenario_1_y_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        if self.mode == 'train_0':
            self.classes = range(0, 6)  # Version 0
            # self.classes = range(0, 5)  # Version 1
            
            self.path_x = self.path_train_o_x
            self.path_y = self.path_train_o_y

        elif self.mode == 'train_1':
            self.classes = range(0, 8)  # Version 0
            # self.classes = range(0, 5)  # Version 1
            self.path_x = self.path_train_n_1_x
            self.path_y = self.path_train_n_1_y

        elif self.mode == 'eval_0':
            self.classes = range(0, 6)  # Version 0
            # self.classes = range(0, 5)  # Version 1
            self.path_x = self.path_val_o_x
            self.path_y = self.path_val_o_y

        elif self.mode == 'eval_1':
            self.classes = range(0, 8)  # Version 1
            # self.classes = range(0, 5)  # Version 1
            self.path_x = self.path_val_n_1_x
            self.path_y = self.path_val_n_1_y

        elif self.mode == 'test_0':
            self.classes = range(0,6)
            self.path_x = self.path_test_o_x
            self.path_y = self.path_test_o_y

        
        elif self.mode == 'test_1':
            self.classes = range(0, 8)
            self.path_x = self.path_test_x
            self.path_y = self.path_test_y
            print(len(np.load(self.path_x)))
            print(self.path_x)
            print(self.path_y)
        
        BaseDataset_new.__init__(self, self.path_x,self.path_y, self.mode, self.transform)

        
        
        # le = LabelEncoder()
        # # le.fit(np.load(self.path_y))

        # le.fit(range(0, 8))

        index = 0
        for x,y in zip(np.load(self.path_x),np.load(self.path_y)):            
            # self.xs += [x]
            self.ys += [y]
            self.xs += [x]
            
            self.I += [index]
            index += 1
        
        