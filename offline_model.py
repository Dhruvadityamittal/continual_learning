import argparse, os, copy, random, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import json
import wandb
import pandas as pd

torch.manual_seed(1)
# np.set_printoptions(threshold=sys.maxsize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'




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

import itertools
lrs = [1e-4]
kd_weights = [5] #[1,5,10, 15, 20]
pa_wights = [20] #[1, 5, 10, 15, 20]
dstes = ['pamap', 'wisdm', 'realworld', 'mhealth']
iterations = [1,2,3,4,5]

results_df = pd.DataFrame(columns = ['Dataset', 'Iteration', 'Initial Acc', 'Initial F1', 'Incremental Acc Seen1',  'Incremental F1 Seen1', 'Incremental Acc Seen2',  'Incremental F1 Seen2', 'Incremental Acc Unseen', 'Incremental F1 Unseen', 'Incremental Acc All', 'Incremental F1 All' ])

if __name__ == '__main__':
    for lr, kd_weight, pa_weight, dset, iter in itertools.product(lrs, kd_weights, pa_wights, dstes, iterations):

        
        parser = argparse.ArgumentParser(description=
                                        'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                        + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`')
        # export directory, training and val datasets, test datasets
        parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
        parser.add_argument('--dataset', default='wisdm', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
        parser.add_argument('--embedding-size', default=128, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
        parser.add_argument('--batch-size', default=512, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
        parser.add_argument('--epochs', default=100, type=int, dest='nb_epochs', help='Number of training epochs.')
        parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
        parser.add_argument('--workers', default=8, type=int, dest='nb_workers', help='Number of workers for dataloader.')
        parser.add_argument('--model', default='resnet18', help='Model for training')  # resnet50 #resnet18  VIT
        parser.add_argument('--loss', default='Proxy_Anchor', help='Criterion for training') #Proxy_Anchor #Contrastive
        parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
        parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate setting')  #1e-4
        parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
        parser.add_argument('--lr-decay-step', default=5, type=int, help='Learning decay step setting')  #
        parser.add_argument('--lr-decay-gamma', default=0.5, type=float, help='Learning decay gamma setting')
        parser.add_argument('--alpha', default=16, type=float, help='Scaling Parameter setting')   # 32
        parser.add_argument('--mrg', default=0.4, type=float, help='Margin parameter setting')    # 0.1
        parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
        parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
        parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization') 
        parser.add_argument('--exp', type=str, default='0')
        parser.add_argument('--kd_weight', default = 10, type=float)
        parser.add_argument('--pa_weight', default = 1 , type=float)
        parser.add_argument('--processes', default = 1 , type=int)
        parser.add_argument('--threads', default = 32 , type=int)

        
        ####
        args = parser.parse_args()
        args.dataset = dset
        if(iter == 1 and  dset!='mhealth'):
            only_test_step1 = True            # Just to test the data on Train_1
        else:
            only_test_step1 = False
        
        args.lr = lr
        args.weight_decay =  args.lr


        print("Dataset :", args.dataset)

        pth_rst_exp = 'Saved_Models/Initial/Offline/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
        os.makedirs(pth_rst_exp, exist_ok=True)

        print("Dataset :", args.dataset)
        ####
        
        pth_dataset = '../datasets'

        if args.dataset =='wisdm':
            pth_dataset = 'HAR_data/Wisdm/'
            window_len = 40
            n_channels = 3
        elif args.dataset =='realworld':
            pth_dataset = 'HAR_data/realworld/'
            nb_classes_now = 8
            window_len = 100
            n_channels = 3
    
        elif args.dataset =='oppo':
            pth_dataset = 'CGCD-main/src/HAR_data/oppo/'
        elif args.dataset =='pamap':
            pth_dataset = 'HAR_data/pamap/'
            nb_classes_now = 12
            window_len = 200
            n_channels = 9
        elif args.dataset =='mhealth':
            pth_dataset = '/home/dmittal/Desktop/CGCD-main/src/HAR_data/mhealth/'
            nb_classes_now = 6
            window_len = 100
            n_channels = 9
        
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="CGCD-HAR-Supervised",
            name='specific-run-name',
            resume='allow',
            # track hyperparameters and run metadata
            config={
            "learning_rate_step_1": args.lr,
            "sz_embedding" : args.sz_embedding,
            "window_len": window_len,
            "batch-size" : args.sz_batch,
            "loss" : args.loss,
            "nb_workers": args.nb_workers,
            "architecture": args.model,
            "dataset": args.dataset,
            "epochs": args.nb_epochs,
            "optimizer" : args.optimizer, 
            "weight-decay" :args.weight_decay,
            "lr-decay-step" : args.lr_decay_step,
            "lr-decay-gamma" : args.lr_decay_gamma,
            "bn-freeze" : args.bn_freeze,
            "l2-norm" : args.l2_norm,
            "exp" : args.exp,
            "KD_weight" : args.kd_weight,
            "pa_weight" : args.pa_weight,

            }
        ) 
        wandb.log({"Method": "Offline"})
                 
        # Initial Step Dataloader ..............
        dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, autoencoderType= None)
        dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

        dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen=window_len, autoencoderType= None)
        dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen=window_len, autoencoderType= None)
        dload_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

        dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode='train_1',windowlen= window_len, autoencoderType= None)
        dload_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

        dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_1',windowlen= window_len, autoencoderType= None)
        dset_test = dataset.load(name=args.dataset, root=pth_dataset, mode='test_1', windowlen= window_len, autoencoderType= None)

        
        from torch.utils.data import ConcatDataset, DataLoader
        concatenated_dataset = ConcatDataset([dlod_tr_0.dataset, dload_tr_now.dataset])
        dlod_tr_merged = torch.utils.data.DataLoader(concatenated_dataset, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)
        
       
        # dlod_tr_merged = merge_dataset(dset_tr_0, dset_tr_now)
        



        # dlod_tr_merged = torch.utils.data.DataLoader(dset_merged_train, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)
        dlod_val = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_test = torch.utils.data.DataLoader(dset_test, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

        nb_classes_before = len(np.unique(dset_tr_0.ys))
        nb_classes_now = len(np.unique(dset_tr_now.ys))

        print("nb_classes Before:", nb_classes_before)
        print("nb_classes Merged:", nb_classes_now)
        bottleneck_dim = 128

        

        # Configuration for the Model  -> Here We have Botleneck later
        if(args.model == 'resnet18'):
            cfg = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
                'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': bottleneck_dim,
                    'output_size':nb_classes_now,'weight_norm_dim': 0 ,  'n_channels' :n_channels, 'window_len': window_len}
            model = ModelGen_new(cfg).create_model().to(device)
        
        if(args.model == 'harnet'):
            repo = 'OxWearables/ssl-wearables'
            model = torch.hub.load(repo, 'harnet5', class_num=nb_classes_now, pretrained=True).to(device)
            del model.classifier
            model.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels,args.sz_embedding)).to(device)


        opt_pa = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
        scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
        loss_baseline = nn.CrossEntropyLoss()


        es_count = 1          # Early Stopping Count
        best_acc = 0
        losses_list = []
        step = 1 
        version = 1
        for epoch in range(0,args.nb_epochs):
            
            if(only_test_step1): continue
            
            losses_per_epoch = []
            pbar = tqdm(enumerate(dlod_tr_merged))
            model.train()
            for batch_idx, (x, y, z) in pbar:
                # if(batch_idx>2): break
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor)

        
                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                opt_pa.zero_grad()
                loss_pa.backward()

                torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                
               
                opt_pa.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_tr_merged), 100. * batch_idx / len(dlod_tr_merged), loss_pa.item(), 0, 0))
                
            
            
            pbar_eval = tqdm(enumerate(dlod_val))
            eval_loss = 0

            print("Evaluating ------------------->")
            total_batch_size = 0
            acc_total = 0
            model.eval()
            for batch_idx, (x, y, z) in pbar_eval:
                
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor).to(device)
                pred_out = torch.argmax(feats, dim=1)
                acc = accuracy_score(y.to('cpu').detach().numpy(),pred_out.to('cpu').detach().numpy())
                acc_total +=acc*x.shape[0]  # tOTAL CORRECT PREDICTIONS
                total_batch_size += x.shape[0]

                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                eval_loss = eval_loss+ loss_pa.item()

                pbar_eval.set_description('Eval Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_val), 100. * batch_idx / len(dlod_val), loss_pa.item(), 0, 0))

            if(acc_total/total_batch_size > best_acc):  #  eval_loss.item()/(batch_idx+1) < best_eval
                print("Saving New Model")
                print("Intitial Acc {}, Current Acc: {}".format(best_acc,acc_total/total_batch_size))
                best_eval = eval_loss/(batch_idx+1)
                # Just saving the model without botteleneck layer
                torch.save(model.state_dict(), '{}{}_{}_best_windowlen_{}_bottleneck_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, bottleneck_dim, version, step)) 
                print("Best Accuracy changed from {} to {}".format(best_acc,acc_total/total_batch_size) )
                best_acc = acc_total/total_batch_size
                es_count = 1
            else:
                es_count +=1
                print("Early Stopping ", es_count)
            
            if(es_count==5):
                print("Performing Early Stopping")
                break
        
        if(args.model == 'resnet18'):
            cfg = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
                'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': bottleneck_dim,
                    'output_size':nb_classes_now,'weight_norm_dim': 0 ,  'n_channels' :n_channels, 'window_len': window_len}
            test_model = ModelGen_new(cfg).create_model().to(device)

        print("Evaluating on Test Dataset in Initial Step..")
        pth_pth_pretrained = '{}{}_{}_best_windowlen_{}_bottleneck_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, bottleneck_dim, version, step)
        checkpoint_pretrained = torch.load(pth_pth_pretrained, map_location=torch.device(device))
        test_model.load_state_dict(checkpoint_pretrained)

        test_model.eval()
        predicted_ys = []
        actual_ys = []
        for x, y, z in dload_test_0:
            
            feats = test_model(x.squeeze().to(device))
            y = y.type(torch.LongTensor).to('cpu').detach().numpy()
            pred_out = torch.argmax(feats, dim=1).to('cpu').detach().numpy()

            if(len(predicted_ys) == 0):
                predicted_ys = pred_out 
            else:
                predicted_ys = np.concatenate((predicted_ys, pred_out))
            
            if(len(actual_ys) == 0):
                actual_ys = y 
            else:
                actual_ys = np.concatenate((actual_ys, y))

        acc_step_1 = accuracy_score(predicted_ys, actual_ys)
        f1_step_1 = f1_score(predicted_ys, actual_ys, average= 'macro')
        wandb.log({"M1-TA1_Old": acc_step_1})   # Logging Initial Step Accuracy.
        wandb.log({"M1-TF1_Old": f1_step_1})  # Logging Initial Step F1 Score.
        print("Stage 1 Test Dataset Seen Classes Acc {} F1 {}".format(acc_step_1, f1_step_1))
        
        test_model.eval()
        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test:
            
            feats = test_model(x.squeeze().to(device))
            y = y.type(torch.LongTensor).to('cpu').detach().numpy()
            pred_out = torch.argmax(feats, dim=1).to('cpu').detach().numpy()

            if(len(predicted_ys) == 0):
                predicted_ys = pred_out 
            else:
                predicted_ys = np.concatenate((predicted_ys, pred_out))
            
            if(len(actual_ys) == 0):
                actual_ys = y 
            else:
                actual_ys = np.concatenate((actual_ys, y))
            

        acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes_before], actual_ys[actual_ys<nb_classes_before])
        acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes_before], actual_ys[actual_ys>=nb_classes_before])
        acc_a = accuracy_score(predicted_ys, actual_ys)

        f1_o = f1_score(predicted_ys[actual_ys<nb_classes_before], actual_ys[actual_ys<nb_classes_before], average='macro')
        f1_n = f1_score(predicted_ys[actual_ys>=nb_classes_before], actual_ys[actual_ys>=nb_classes_before], average= 'macro')
        f1_a = f1_score(predicted_ys, actual_ys, average= 'macro')
        

        wandb.log({"M2-TA2_Old": acc_o})   # Logging Initial Step Accuracy.
        wandb.log({"M2-TF2_Old": f1_o})  # Logging Initial Step F1 Score.
        wandb.log({"M2-TA2_New": acc_n})   # Logging Initial Step Accuracy.
        wandb.log({"M2-TF2_New": f1_n})  # Logging Initial Step F1 Score.
        wandb.log({"M2-TA2_All": acc_a})   # Logging Initial Step Accuracy.
        wandb.log({"M2-TF2_All": f1_a})  # Logging Initial Step F1 Score.

        print("Stage 2 Test Dataset Seen Classes Acc {} F1 {}".format(acc_o, f1_o))
        print("Stage 2 Test Dataset Unseen Classes Acc {} F1 {}".format(acc_n, f1_n))
        print("Stage 2 Test Dataset All Classes Acc {} F1 {}".format(acc_a, f1_a))
        print("\n\n")

        new_row = {'Dataset':dset, 'Iteration':iter, 
               'Initial Acc': acc_step_1, 'Initial F1': f1_step_1, 'Incremental Acc Seen1':0,
                 'Incremental F1 Seen1':0, 'Incremental Acc Seen2' :acc_o,
                   'Incremental F1 Seen2':f1_o, 'Incremental Acc Unseen' :acc_n,
                     'Incremental F1 Unseen': f1_n, 'Incremental Acc All':acc_a, 
                     'Incremental F1 All' :f1_a}
        
        
        results_df = results_df.append(new_row, ignore_index = True)
        
        wandb.finish()

results_df.to_csv('results_offline_model.csv', index=False)