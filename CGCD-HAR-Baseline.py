import argparse, os, copy, random, sys
import numpy as np

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
from sklearn.metrics import accuracy_score , f1_score
import json
torch.manual_seed(1)
np.set_printoptions(threshold=sys.maxsize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("BASELINE MODEL")

print("Using Device", device)

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

import pandas as pd
import itertools
dstes = ['realworld', 'mhealth']
iterations = [1,2,3,4,5] #[1,2,3,4,5]
results_df = pd.DataFrame(columns = ['Dataset', 'Iteration', 'Initial Acc', 'Initial F1', 'Incremental Acc Seen1',  'Incremental F1 Seen1', 'Incremental Acc Seen2',  'Incremental F1 Seen2', 'Incremental Acc Unseen', 'Incremental F1 Unseen', 'Incremental Acc All', 'Incremental F1 All', "Forget Acc", 'Forget F1' ])

if __name__ == '__main__':
    for dset, iter in itertools.product(dstes, iterations):
        parser = argparse.ArgumentParser(description=
                                        'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                        + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`')
        # export directory, training and val datasets, test datasets
        parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
        parser.add_argument('--dataset', default='realworld', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
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
        parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
        parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
        parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization')
        parser.add_argument('--remark', default='', help='Any reamrk')
        parser.add_argument('--use_split_modlue', type=bool, default=True)
        parser.add_argument('--use_GM_clustering', type=bool, default=True) # False
        parser.add_argument('--exp', type=str, default='0')
        parser.add_argument('--processes', default = 1 , type=int)
        parser.add_argument('--threads', default = 32 , type=int)

        
    
        args = parser.parse_args()
        args.dataset = dset

        if(iter ==1):
            only_test_step1 = True            # Just to test the data on Train_1
            only_test_step2 = True
        else:
            only_test_step1 = False            # Just to test the data on Train_1
            only_test_step2 = False


        pth_rst = 'CGCD-main/src/result/' + args.dataset
        os.makedirs(pth_rst, exist_ok=True)
        pth_rst_exp = 'AutoEncoders/' + args.dataset + '/' #+ args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
        os.makedirs(pth_rst_exp, exist_ok=True)

        print("Dataset :", args.dataset)

        pth_rst = 'CGCD-main/src/result/' + args.dataset
        os.makedirs(pth_rst, exist_ok=True)
        pth_rst_exp = '/home/dmittal/Desktop/CGCD-main/src/Saved_Models/Initial/Baseline/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
        os.makedirs(pth_rst_exp, exist_ok=True)

        print("\n\nDataset :", args.dataset)
        ####
        
        pth_dataset = '../datasets'

        if args.dataset =='wisdm':
            pth_dataset = '/home/dmittal/Desktop/CGCD-main/src/HAR_data/Wisdm/'
            nb_classes_now = 14
            window_len = 40
            n_channels = 3
        elif args.dataset =='realworld':
            pth_dataset = '/home/dmittal/Desktop/CGCD-main/src/HAR_data/realworld/'
            nb_classes_now = 8
            window_len = 100
            n_channels = 3
        elif args.dataset =='oppo':
            pth_dataset = 'CGCD-main/src/HAR_data/oppo/'
        elif args.dataset =='pamap':
            pth_dataset = '/home/dmittal/Desktop/CGCD-main/src/HAR_data/pamap/'
            nb_classes_now = 12
            window_len = 200
            n_channels = 9
        elif args.dataset =='mhealth':
            pth_dataset = '/home/dmittal/Desktop/CGCD-main/src/HAR_data/mhealth/'
            nb_classes_now = 6
            window_len = 100
            n_channels = 9
        
        bottleneck_dim = 128

        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="CGCD-HAR-Supervised",
            name='specific-run-name',
            resume='allow',
            # track hyperparameters and run metadata
            config={
            "learning_rate_step_1": args.lr,
            "learning_rate_step_2": args.lr,
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
            "warm" : args.warm,
            "bn-freeze" : args.bn_freeze,
            }
            )
        wandb.log({"Method": "Baseline"})

        # Initial Step Dataloader ..............
        dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, autoencoderType= None)
        dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)

        dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen= window_len, autoencoderType= None)
        dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

        dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen= window_len, autoencoderType= None)
        dlod_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)


        nb_classes = dset_tr_0.nb_classes()
        print("Number of Classes :",nb_classes)

        # Configuration for the Model  -> Here We have Botleneck later
        if(args.model == 'resnet18'):
            cfg = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
                'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': bottleneck_dim,
                    'output_size':nb_classes,'weight_norm_dim': 0 ,  'n_channels' :n_channels, 'window_len': window_len}
            model = ModelGen_new(cfg).create_model().to(device)
        
        if(args.model == 'harnet'):
            repo = 'OxWearables/ssl-wearables'
            model = torch.hub.load(repo, 'harnet5', class_num=nb_classes, pretrained=True).to(device)
            del model.classifier
            model.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels,args.sz_embedding)).to(device)

                                        # All the Above code is similar to the G-Baseline Code
                                        #  Starting to just train on Seen Dataset. 

        best_eval = 9999999

        # Training Parameters
        opt_pa = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
        loss_baseline = nn.CrossEntropyLoss()
        scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
        
        es_count = 1          # Early Stopping Count
        best_acc = 0
        losses_list = []
        step = 1 
        version = 1
        for epoch in range(0,args.nb_epochs):
            if(only_test_step1): continue
            model.train()
            losses_per_epoch = []
            pbar = tqdm(enumerate(dlod_tr_0))

            for batch_idx, (x, y, z) in pbar:
                
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor)
        
                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                opt_pa.zero_grad()
                loss_pa.backward()

                torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                
                losses_per_epoch.append(loss_pa.data.cpu().numpy())
                opt_pa.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), loss_pa.item(), 0, 0))
                
            
            losses_list.append(np.mean(losses_per_epoch))
            scheduler_pa.step()
            pbar_eval = tqdm(enumerate(dlod_ev))
            eval_loss = 0

            print("Evaluating ------------------->")
            total_batche_size = 0
            acc_total = 0
            model.eval()
            for batch_idx, (x, y, z) in pbar_eval:
                feats = model(x.squeeze().to(device))
                y = y.type(torch.LongTensor).to(device)
                pred_out = torch.argmax(feats, dim=1)
                acc = accuracy_score(y.to('cpu').detach().numpy(),pred_out.to('cpu').detach().numpy())
                acc_total +=acc*x.shape[0]  # tOTAL CORRECT PREDICTIONS
                total_batche_size += x.shape[0]

                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                eval_loss = eval_loss+ loss_pa

                pbar_eval.set_description('Eval Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_ev), 100. * batch_idx / len(dlod_ev), loss_pa.item(), 0, 0))

            if(acc_total/total_batche_size > best_acc):  #  eval_loss.item()/(batch_idx+1) < best_eval
                print("Saving New Model")
                print("Intitial Acc {}, Current Acc: {}".format(best_acc,acc_total/total_batche_size))
                best_eval = eval_loss.item()/(batch_idx+1)
                # Just saving the model without botteleneck layer
                torch.save(model.state_dict(), '{}{}_{}_best_windowlen_{}_bottleneck_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, bottleneck_dim, version, step)) 
                print("Best Accuracy changed from {} to {}".format(best_acc,acc_total/total_batche_size) )
                best_acc = acc_total/total_batche_size
                es_count = 1
            else:
                es_count +=1
                print("Early Stopping ", es_count)
            
            if(es_count==5):
                print("Performing Early Stopping")
                break

            # if (epoch >= 0):
            #     with torch.no_grad():
            #         print('Evaluating..')
            #         Recalls = utils_CGCD.evaluate_cos(model, dlod_ev, epoch)
            #     #### Best model save
            #     if best_recall[0] < Recalls[0]:
            #         best_recall = Recalls
            #         best_epoch = epoch
            #         # torch.save({'model_pa_state_dict': model.state_dict(), 'proxies_param': criterion_pa.proxies}, '{}/{}_{}_best_step_0.pth'.format(pth_rst_exp, args.dataset, args.model))
            #         # with open('{}/{}_{}_best_results.txt'.format(pth_rst_exp, args.dataset, model_name), 'w') as f:
            #         #     f.write('Best Epoch: {}\tBest Recall@{}: {:.4f}\n'.format(best_epoch, 1, best_recall[0] * 100))

        

    ######################################################################################################
    # Incremental Step
        print("\n Incremental Step \n")

        dset_tr_1 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_1', windowlen= window_len, autoencoderType= None)
        dlod_tr_1 = torch.utils.data.DataLoader(dset_tr_1, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)

        
        dset_ev_1 = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_1', windowlen= window_len, autoencoderType= None)
        dlod_ev_1 = torch.utils.data.DataLoader(dset_ev_1, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

        dset_test = dataset.load(name=args.dataset, root=pth_dataset, mode='test_1', windowlen= window_len, autoencoderType= None)
        dlod_test = torch.utils.data.DataLoader(dset_test, batch_size=32, shuffle=False, num_workers=args.nb_workers)

        


        nb_classes_1 = dset_tr_1.nb_classes()
        print("Classes in the initial Step: {}, Incremental Step: {}".format(nb_classes,nb_classes_1))

        # cfg_new = {'weights_path': pth_rst_exp +'/Baseline_Initial_resnet_'+args.dataset+'.mdl' , "use_ssl_weights" : True, 'conv_freeze': False, 'load_finetuned_mtl': False,
        #      'checkpoint_name' :'', 'epoch_len': 10, 'output_size': '', 'embedding_dim': 0, 'bottleneck_dim': bottleneck_dim,
        #         'output_size':nb_classes,'weight_norm_dim': 0}
        

        cfg = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': bottleneck_dim,
                'output_size':nb_classes,'weight_norm_dim': 0, 'n_channels' :n_channels, 'window_len': window_len}


        # Just Testing the Initital Model
        print("Loading Saved Model")
        initial_model = ModelGen_new(cfg).create_model().to(device)

        pth_pth = '{}{}_{}_best_windowlen_{}_bottleneck_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, bottleneck_dim, version, step)
        checkpoint = torch.load(pth_pth, map_location=torch.device(device))
        initial_model.load_state_dict(checkpoint)

        print('==>Evaluation of Stage 1 Dataset on Stage 1 Model')
        
        test_loss,total_batche_size, acc_total, epoch = 0, 0, 0, 0
        initial_model.eval()

        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test_0:
            
            feats = initial_model(x.squeeze().to(device))
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
            

        acc_o_base = accuracy_score(predicted_ys,actual_ys)
        f1_o_base = f1_score(predicted_ys, actual_ys, average= 'macro')

        wandb.log({"M1-TA1_Old": acc_o_base})   # Logging Initial Step Accuracy.
        wandb.log({"M1-TF1_Old": f1_o_base})  # Logging Initial Step F1 Score.

        print("Stage 1 Test Dataset on Stage 1 trained model ",acc_o_base)

        
        # We only have seen and unseen classes, but model is only trained on seen classes to unseen accuracy should be zero
        print('==>Evaluation of Stage 2 Dataset on Stage 1 Model')


        initial_model.eval()
        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test:
            feats = initial_model(x.squeeze().to(device))
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
    
        acc_o1 = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
        acc_n1 = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])

        f1_o1 = f1_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes], average='macro')
        f1_n1 = f1_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes], average= 'macro')

        print("Stage 2 Test Dataset on Stage 1 trained model Accuracy, Seen {}  Unseen {}".
            format(acc_o1, acc_n1))
        print("Stage 2 Test Dataset on Stage 1 trained model F1 Score , Seen  {} Unseen {}".
            format(f1_o1, f1_n1))

        wandb.log({"M1-TA2_Old": acc_o1})   # Logging Initial Step Accuracy.
        wandb.log({"M1-TF2_Old": f1_o1})  # Logging Initial Step F1 Score.
        

        opt_pa_new = torch.optim.AdamW(initial_model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay)
        scheduler_pa_new = torch.optim.lr_scheduler.StepLR(opt_pa_new, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
        loss_baseline = nn.CrossEntropyLoss()
        
        
        last_layers = initial_model.fc.weight[0:nb_classes]  # Taking initial model weights
        initial_model.fc = nn.Linear(bottleneck_dim, nb_classes_1).to(device)    # Changing Classifier outputs

        with torch.no_grad(): # Copying the learned seen classes layers to the extended model.
            initial_model.fc.weight[:nb_classes].copy_(last_layers)


        best_weighted_acc =0 
        es_count =1
        pth_rst_exp = '/home/dmittal/Desktop/CGCD-main/src/Saved_Models/Incremental/Baseline/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
        os.makedirs(pth_rst_exp, exist_ok=True)
        step = 2

        load_pretrained = False
        
        if(load_pretrained):
            print("Loading Pre-Trained Model ...")
            step = 1
            pth_pth_pretrained = '{}{}_{}_model_last_windowlen_{}_sz_bottleneck_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len,bottleneck_dim , str(step))
            checkpoint_pretrained = torch.load(pth_pth_pretrained,map_location=torch.device(device))
            initial_model.load_state_dict(checkpoint_pretrained['model_pa_state_dict'])
        
        
        for epoch in range(0,args.nb_epochs):
            if(only_test_step2): continue 
            losses_per_epoch = []
            pbar = tqdm(enumerate(dlod_tr_1))
            total_correct_seen = 0
            total_seen = 0
            total_correct_unseen = 0
            total_unseen = 0
            initial_model.train()
            for batch_idx, (x, y, z) in pbar:
                ####
                feats = initial_model(x.squeeze().to(device))
                y = y.type(torch.LongTensor)

                loss_pa = loss_baseline(feats, y.squeeze().to(device)).to(device)
                opt_pa_new.zero_grad()
                loss_pa.backward()

                losses_per_epoch.append(loss_pa.data.cpu().numpy())
                opt_pa_new.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: Overall {:.4f}'.format(
                    epoch, batch_idx + 1, len(dlod_tr_1), 100. * batch_idx / len(dlod_tr_1), loss_pa.item(), 0,0))
            
            losses_list.append(np.mean(losses_per_epoch))
            scheduler_pa_new.step()

            eval_loss, total_correct_seen, total_seen, total_correct_unseen, total_unseen, total_data, overall_correct  = 0, 0, 0, 0, 0, 0, 0

            print("Evaluating ------------------->")
            initial_model.eval()

            predicted_ys = []
            actual_ys = []
            for x, y, z in dlod_ev_1:
                feats = initial_model(x.squeeze().to(device))
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
        
            acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
            acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
            acc_a = accuracy_score(predicted_ys, actual_ys)
                    
                

            print("Valid Accuracies Seen {}, Unseen {}, Overall {}".format(acc_o, acc_n, acc_a))
            

            
            if((acc_o + acc_n + acc_a)/3 > best_weighted_acc):
                best_seen = acc_o
                best_unseen = acc_n
                best_overall = acc_a
                best_weighted_acc = (best_seen + best_unseen + best_overall)/3
                best_epoch = epoch
                es_count = 0
                print("Got Better Model with  Seen {},  Unseen Accuracies {},  Overalll {} and  Average as {} ".format( best_seen,best_unseen, best_overall,best_weighted_acc))
                torch.save({'model_pa_state_dict': initial_model.state_dict()}, '{}{}_{}_model_last_windowlen_{}_sz_bottleneck_{}_step_{}_epoch_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len, bottleneck_dim, str(step), epoch))
            else:
                es_count +=1
                print("Early Stopping Count ", es_count)
                
            if(epoch< args.warm):
                es_count =0    # If we are having warmup then no early stopping
            
            if(es_count ==10):
                
                
                print("Best  Valid Accuracies Seen {}, and Unseen {},  Overall {}, Average {} ".format( best_seen, best_unseen, best_overall, best_weighted_acc))
                print("Early Stopping")
                break
        if(only_test_step2 == False):
            with open('{}{}_{}_BestEpoch__model_last_windowlen_{}_sz_bottleneck_{}_step_{}.json'.format(pth_rst_exp, 
                                                                                                                    args.dataset,
                                                                                                                    args.model,window_len,bottleneck_dim, str(step)), 'w') as f:
                        json.dump(best_epoch, f)
        
        
        ###########################################  Testing On Test Dataset ...  #####################################################
        
        print("Testing On Test Dataset ...")
        pth_rst_exp_test = 'CGCD-main/src/Saved_Models/Incremental/Baseline/' + args.dataset + '/'
        step =2
        
        
        
        with open('{}{}_{}_BestEpoch__model_last_windowlen_{}_sz_bottleneck_{}_step_{}.json'.format(pth_rst_exp, 
                                                                                                                    args.dataset,
                                                                                                                    args.model,window_len,bottleneck_dim, str(step)), 'r') as f:
            best_epoch = json.load(f)

        cfg_test = {'weights_path': 'Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': None, 'bottleneck_dim': bottleneck_dim,
                'output_size':nb_classes_1,'weight_norm_dim': 0, 'n_channels' :n_channels, 'window_len': window_len}
        
        pth_pth_test = '{}{}_{}_model_last_windowlen_{}_sz_bottleneck_{}_step_{}_epoch_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len,bottleneck_dim, str(step), best_epoch)

        model_test = ModelGen_new(cfg_test).create_model().to(device)
        model_test.fc = nn.Linear(bottleneck_dim, nb_classes_1).to(device)
        model_test = model_test.to(device)

        checkpoint_test = torch.load(pth_pth_test,map_location=torch.device(device))
        model_test.load_state_dict(checkpoint_test['model_pa_state_dict'])
        
        model_test.eval()
        predicted_ys = []
        actual_ys = []
        for x, y, z in dlod_test:
            feats = model_test(x.squeeze().to(device))
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

        acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
        acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
        acc_all = accuracy_score(predicted_ys, actual_ys)

        f1_o = f1_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes], average='macro')
        f1_n = f1_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes], average= 'macro')
        f1_all = f1_score(predicted_ys, actual_ys, average= 'macro')

        wandb.log({"M2-TA2_Old": acc_o})   
        wandb.log({"M2-TF2_Old": f1_o})  
        wandb.log({"M2-TA2_New": acc_n})   
        wandb.log({"M2-TF2_New": f1_n})  
        wandb.log({"M2-TA2_All": acc_all})   
        wandb.log({"M2-TF2_All": f1_all})  
        wandb.log({"M2-TFA":  acc_o1 - acc_o})  
        wandb.log({"M2-TFF1": f1_o1 - f1_n})  
        wandb.finish()
        
        new_row = {'Dataset':dset, 'Iteration':iter, 
        'Initial Acc': acc_o_base, 'Initial F1': f1_o_base, 'Incremental Acc Seen1':acc_o1,
            'Incremental F1 Seen1':f1_o_base, 'Incremental Acc Seen2' :acc_o,
            'Incremental F1 Seen2':f1_o1, 'Incremental Acc Unseen' :acc_n,
                'Incremental F1 Unseen': f1_n, 'Incremental Acc All':acc_all, 
                'Incremental F1 All' :f1_all, "Forget Acc": acc_o1 - acc_o, "Forget F1": f1_o1 - f1_n}
        
        
        results_df = results_df.append(new_row, ignore_index = True)

        print("Testing Accuracies : Seen {} Unseen {} Overall {}".format(acc_o, acc_n, acc_all))
        print("Testing F1 Scores  : Seen {} Unseen {} Overall {}".format(f1_o, f1_n, f1_all))

results_df.to_csv('results_online_finetuing_{}.csv'.format('_'.join(dstes)), index=False)