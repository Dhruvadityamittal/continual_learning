import argparse, os, copy, random, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from functools import partial

from tqdm import *

import dataset, utils_CGCD, losses, net
from net.resnet import *

from models.modelgen import ModelGen, ModelGen_new
from torchsummary import summary
np.set_printoptions(threshold=sys.maxsize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.metrics import accuracy_score
from autoencoder import Autoencoder
torch.manual_seed(1)


def generate_dataset(dataset, index, index_target=None, target=None):
    dataset_ = copy.deepcopy(dataset)

    if target is not None:
        for i, v in enumerate(index_target):
            dataset_.ys[v] = target[i]

    for i, v in enumerate(index):
        # print("Index ",index, "Dataset.I",dataset_.I)
        j = v - i    # We seperate i because as we pop a element outside the array moves towards left and its size decreases
        dataset_.I.pop(j)
        dataset_.ys.pop(j)
        # dataset_.im_paths.pop(j)
    return dataset_


def merge_dataset(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    # if len(dataset_n.classes) > len(dataset_.classes):
    #     dataset_.classes = dataset_n.classes
    dataset_.I.extend(dataset_n.I)
    # dataset_.I  = list(set(dataset_n.I))
    # dataset_.im_paths.extend(dataset_n.im_paths)
    dataset_.ys.extend(dataset_n.ys)

    return dataset_


class MyDict(dict):
    def __init__(self, args):
        self.args

class BlankClass:
    pass

args = BlankClass()


args.sz_batch  = 256
args.nb_epochs = 60
args.nb_workers = 0
args.loss = "Proxy_Anchor"
args.optimizer = "adamw"
args.lr = 1e-2 
args.weight_decay = 1e-7
args.lr_decay_step = 5
args.lr_decay_gamma =  0.5
args.alpha = 32
args.mrg = 0.1
args.warm = 5
args.bn_freeze = False 
args.l2_norm =  True
args.use_split_modlue =  True
args.use_GM_clustering =  True
args.exp = '0'



# Hard Coded Variables.

args.dataset = 'pamap'   # Dataset 
only_test = False            # Just to test the data on Train_1
args.model = 'resnet18'        # Model Name
window_len = 200
args.use_split_modlue = True
args.sz_embedding = 256


pth_rst = 'CGCD-main/src/result/' + args.dataset
os.makedirs(pth_rst, exist_ok=True)
pth_rst_exp = 'AutoEncoders/' + args.dataset + '/' #+ args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
os.makedirs(pth_rst_exp, exist_ok=True)

print("Dataset :", args.dataset)
####

pth_dataset = '../datasets'
if args.dataset == 'cub':
    pth_dataset += '/CUB200'
elif args.dataset == 'mit':
    pth_dataset += '/MIT67'
elif args.dataset == 'dog':
    pth_dataset += '/DOG120'
elif args.dataset == 'air':
    pth_dataset += '/AIR100'
elif args.dataset == 'adl':
    
    pth_dataset = 'netscratch/martelleto/iswc/ssl-data/downstream/adl_30hz_clean/'
elif args.dataset =='wisdm':
    pth_dataset = 'CGCD-main/src/HAR_data/Wisdm/'
elif args.dataset =='realworld':
    pth_dataset = 'CGCD-main/src/HAR_data/realworld/'
    nb_classes_now = 8
elif args.dataset =='oppo':
    pth_dataset = 'CGCD-main/src/HAR_data/oppo/'
elif args.dataset =='pamap':
    pth_dataset = 'CGCD-main/src/HAR_data/pamap/'
    nb_classes_now = 12
    
from autoencoder import Autoencoder_FC


autoencoderType = 'normal'   # 'normal'


dset_tr_0_AE = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, transform= True, autoencoderType = autoencoderType)
# criterion = nn.MSELoss().to(device)
criterion = torch.nn.L1Loss().to(device) 



# dlod_tr_0_AE = torch.utils.data.DataLoader(dset_tr_0_AE, shuffle=False, num_workers=args.nb_workers)
transform = False  # Instead of Using Raw input we take transformed information like mean, std, correlation coeff.
dataloaders = []
trained_FC_autoencoders = []
load_pretrained = False

if(transform == True and autoencoderType == 'denoising'):
    raise AssertionError("Autoencoder Cannot be Denoising if we are Transforming the Data. Change Transform to True or Change AutoEncoder to Normal")
    

# for clas in range(0, nb_classes_now):
for clas in [9,10,11]:

    dset_tr_0_AE = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, transform= transform, autoencoderType = autoencoderType)
    y_temp = torch.tensor(dset_tr_0_AE.ys)

    data_beloning_to_a_specific_class =  np.nonzero(torch.where(y_temp == clas, 1, 0))
    
    dset_tr_0_AE.ys = np.concatenate(list(np.array(dset_tr_0_AE.ys)[data_beloning_to_a_specific_class])).tolist()
    dset_tr_0_AE.I = np.concatenate(list(np.array(dset_tr_0_AE.I)[data_beloning_to_a_specific_class])).tolist()
    
    # print(dset_tr_0_AE.I)W
    
    
    print("Class {}  No of Data {}".format(clas, len(dset_tr_0_AE.ys)))
    dl_specific_class = torch.utils.data.DataLoader(dset_tr_0_AE, batch_size=16, shuffle=True, num_workers=args.nb_workers)
    

    autoencoder_FC = Autoencoder_FC(next(iter(dl_specific_class))[0].shape).to(device)
    print("Shape of X  :", next(iter(dl_specific_class))[0].shape)
    
   
    
    optimizer = torch.optim.Adam(autoencoder_FC.parameters(), lr=0.0001)

    if(load_pretrained):
        print("Loading a Pretrained Model.")
        ckpt = torch.load('{}AutoEncoder_Class_{}_{}_{}_best_windowlen_{}_embedding_size_{}_transform_{}_AEType_{}.pth'.format(pth_rst_exp, clas, args.dataset, args.model, window_len, args.sz_embedding, transform, autoencoderType))['model_pa_state_dict']
        autoencoder_FC.load_state_dict(ckpt)

    for epoch in range(0,50):
        pbar = tqdm(enumerate(dl_specific_class))
        total_loss = 0
        total_inputs = 0 
        autoencoder_FC.train()

        # Denoising AE
        if(autoencoderType == 'denoising'):
            for batch_idx, (x_noise, _, x_clean,_) in pbar:
                feats = autoencoder_FC(x_noise.to(device))
                loss = criterion(feats, x_clean.to(device))
                total_loss += loss.item()*x_noise.shape[0]
                total_inputs += x_noise.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dl_specific_class), 100. * batch_idx / len(dl_specific_class), loss.item()))
        else:
            for batch_idx, (x, _, _) in pbar: # Simple AutoEncoder.

                # if(batch_idx>1):
                #     break
                feats = autoencoder_FC(x.to(device))
                loss = criterion(feats, x.to(device))
                total_loss += loss.item()*x.shape[0]
                total_inputs += x.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dl_specific_class), 100. * batch_idx / len(dl_specific_class), loss.item()))
        print("Class {}  Epoch {}  Loss {}".format(clas, epoch, total_loss/total_inputs))
    
    trained_FC_autoencoders.append(autoencoder_FC)
    torch.save({'model_pa_state_dict': autoencoder_FC.state_dict()}, '{}AutoEncoder_Class_{}_{}_{}_best_windowlen_{}_embedding_size_{}_transform_{}_AEType_{}.pth'.format(pth_rst_exp, clas, args.dataset, args.model, window_len, args.sz_embedding, transform, autoencoderType))
    del optimizer
    del autoencoder_FC
    
    # dataloaders.append(torch.utils.data.DataLoader(dset_tr_0_AE, batch_size= args.batch_size, shuffle=False,  num_workers=args.nb_workers))

