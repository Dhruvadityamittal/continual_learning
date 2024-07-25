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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
torch.manual_seed(1)
np.set_printoptions(threshold=sys.maxsize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'




def calculate_accuracy(predicted_classes, true_labels):

    # Ensure both tensors have the same shape
    
    if predicted_classes.shape != true_labels.shape:
        raise ValueError("Shapes of predicted classes and true labels do not match.")
    
    # Calculate accuracy
    correct_predictions = (predicted_classes == true_labels).sum().item()
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions

    return accuracy

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                     + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`')
    # export directory, training and val datasets, test datasets
    parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
    parser.add_argument('--dataset', default='realworld', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
    parser.add_argument('--embedding-size', default=128, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
    parser.add_argument('--batch-size', default=512, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
    parser.add_argument('--epochs', default=60, type=int, dest='nb_epochs', help='Number of training epochs.')
    parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
    parser.add_argument('--workers', default=0, type=int, dest='nb_workers', help='Number of workers for dataloader.')
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
    parser.add_argument('--remark', default='', help='Any reamrk')
    parser.add_argument('--use_split_modlue', type=bool, default=True)
    parser.add_argument('--use_GM_clustering', type=bool, default=True) # False
    parser.add_argument('--exp', type=str, default='0')


    ####
    args = parser.parse_args()


    only_test = False            # Just to test the data on Train_1


    pth_rst_exp = 'AutoEncoders/' + args.dataset + '/' #+ args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
    os.makedirs(pth_rst_exp, exist_ok=True)

    print("Dataset :", args.dataset)


    
    pth_rst = 'result/' + args.dataset
    os.makedirs(pth_rst, exist_ok=True)
    pth_rst_exp = 'Saved_Models/Initial/G-Baseline/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
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
        pth_dataset = 'HAR_data/Wisdm/' 
        window_len = 40
        n_channels = 3
    elif args.dataset =='realworld':
        pth_dataset = 'HAR_data/realworld/'
        nb_classes_now = 8
        window_len = 100
        n_channels = 3

    elif args.dataset =='oppo':
        pth_dataset = 'HAR_data/oppo/'
    elif args.dataset =='pamap':
        pth_dataset = 'HAR_data/pamap/'
        nb_classes_now = 12
        window_len = 200
        n_channels = 9
    
    
    # Initial Step Dataloader ..............
    dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, autoencoderType= None)
    dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)

    dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen=window_len, autoencoderType= None)
    dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen=window_len, autoencoderType= None)
    dlod_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    # nb_classes = dset_test_0.nb_classes()
    nb_classes = dset_tr_0.nb_classes()

    # Configuration for the Model
    if(args.model == 'resnet18'):
        cfg = {'weights_path': 'CGCD-main/src/Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
                'output_size':nb_classes,'weight_norm_dim': '', 'n_channels' :n_channels, 'window_len': window_len}

        model = ModelGen_new(cfg).create_model().to(device)
    
    if(args.model == 'harnet'):
        repo = 'OxWearables/ssl-wearables'
        model = torch.hub.load(repo, 'harnet5', class_num=nb_classes, pretrained=True).to(device)
        del model.classifier
        model.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels,args.sz_embedding)).to(device)

                                               # Starting with CGCD
    
    #### DML Losses
    # Initial Step
    criterion_pa = losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).to(device)

    #### Train Parameters
    gpu_id =0
    param_groups = [
        {'params': list(set(model.parameters()).difference(set(model.embedding.parameters()))) if gpu_id != -1 else list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {'params': model.embedding.parameters() if gpu_id != -1 else model.embedding.parameters(), 'lr': float(args.lr) * 1},]
    # param_groups = [{'params': model.parameters()}]
    param_groups.append({'params': criterion_pa.parameters(), 'lr': float(args.lr)*100 })
    
    
    #### Optimizer
    opt_pa = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # print('Training parameters: {}'.format(vars(args)))
    print('Training for {} epochs'.format(args.nb_epochs))
    losses_list = []
    best_recall = [0]
    best_epoch = 0
    best_acc, es_count = 0,0

    version = 1
    step = 0

    for epoch in range(0,1):  #args.nb_epochs
        model.train()

        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model.modules() if gpu_id != -1 else model.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []

        #### Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            if gpu_id != -1:
                unfreeze_model_param = list(model.embedding.parameters()) + list(criterion_pa.parameters())
            else:
                unfreeze_model_param = list(model.embedding.parameters()) + list(criterion_pa.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        total, correct = 0, 0
        pbar = tqdm(enumerate(dlod_tr_0))    
        for batch_idx, (x, y, z) in pbar:
            ####
            feats = model(x.squeeze().to(device))
            # print("X Shape {}, Y Shape {}".format(x.shape, y.shape))
           
      
            loss_pa = criterion_pa(feats, y.squeeze().to(device)).to(device)
            opt_pa.zero_grad()
            loss_pa.backward()

            
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == 'Proxy_Anchor':
                torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

            losses_per_epoch.append(loss_pa.data.cpu().numpy())
            opt_pa.step()

            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), loss_pa.item(), 0, 0))
            
        
        losses_list.append(np.mean(losses_per_epoch))
        # scheduler_pa.step()
        
        if (epoch >= 0):
            # with torch.no_grad():
            #     print('Evaluating..')
            #     Recalls = utils_CGCD.evaluate_cos(model, dlod_ev, epoch)
            # #### Best model save
            # if best_recall[0] < Recalls[0]:
            #     best_recall = Recalls
            #     best_epoch = epoch
            #     torch.save({'model_pa_state_dict': model.state_dict(), 'proxies_param': criterion_pa.proxies}, '{}{}_{}_best_windowlen_{}_embedding_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, version, step))
            #     with open('{}/{}_{}_best_results.txt'.format(pth_rst_exp, args.dataset, args.model), 'w') as f:
            #         f.write('Best Epoch: {}\tBest Recall@{}: {:.4f}\n'.format(best_epoch, 1, best_recall[0] * 100))

            with torch.no_grad():
                feats, _ = utils_CGCD.evaluate_cos_(model, dlod_ev)
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
                _, preds_lb = torch.max(cos_sim, dim=1)
                preds = preds_lb.detach().cpu().numpy()
                acc_0, _ = utils_CGCD._hungarian_match_(np.array(dlod_ev.dataset.ys), preds)
                print('Valid Epoch: {} Acc: {:.4f}'.format(str(-1), acc_0))  
            
            # Saving the Best Model According to Accuracy
            if(best_acc< acc_0):
                print("Got better model with from accuracy {} to {}".format(best_acc,acc_0))
                best_acc = acc_0
                torch.save({'model_pa_state_dict': model.state_dict(), 'proxies_param': criterion_pa.proxies}, '{}{}_{}_best_windowlen_{}_embedding_size_{}_alpha{}_mrg_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, args.alpha, args.mrg, version, step))

            else:                      # if We don't get better model
                es_count +=1
                if(epoch> args.warm):
                    print("Early stopping count {}".format(es_count))
                            
            if(epoch< args.warm):
                es_count = 0

        if(es_count == 8):
            print("Early Stopping with Count {} and Best Accuracy {}".format(es_count, best_acc))
            break

    
    #################################################################Proxy Anchor are now trained. ############################################
    

    ## Load checkpoint.
    print('==> Resuming from checkpoint..')
    
    pth_pth = '{}{}_{}_best_windowlen_{}_embedding_size_{}_alpha{}_mrg_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, args.alpha, args.mrg, version, step)
    checkpoint = torch.load(pth_pth)
    model.load_state_dict(checkpoint['model_pa_state_dict'])
    criterion_pa.proxies = checkpoint['proxies_param']
    model = model.to(device)
    model.eval()
    
    
    ####
    print('==>Evaluation of Stage 1 Dataset on Stage 1 Model')
    model.eval()
    with torch.no_grad():
        feats, _ = utils_CGCD.evaluate_cos_(model, dlod_test_0)
    
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))  # Cosine Similarity is between -1 to 1
        _, preds_lb = torch.max(cos_sim, dim=1)  # Returns values and the class
        # print(_.detach().cpu().numpy())
        preds = preds_lb.detach().cpu().numpy()
        acc_0, _ = utils_CGCD._hungarian_match_(np.array(dlod_test_0.dataset.ys), preds)
        print(calculate_accuracy(np.array(dlod_test_0.dataset.ys), preds))
        print("F1 Score", f1_score(np.array(dlod_test_0.dataset.ys), preds, average='micro'))
        

    print("Stage 1 Test Dataset on  Stage 1 trained Model,",acc_0 )
    
    ####
    
    # Incremental Step ###########################################################
    
    # for i in range(args.steps):
    #     print('==> Step {} th..'.format(i + 1))
    # dlod_tr_prv = torch.utils.data.DataLoader(dset_tr_now_m, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    # dlod_tr_prv.dataset.ys = dlod_tr_now_m.dataset.ys
    dlod_tr_prv = dlod_tr_0
    dset_tr_now_md = 'train_1' # 'train_2'
    dset_ev_now_md = 'eval_1' # 'eval_2'
    nb_classes_prv = nb_classes
    nb_classes_evn = nb_classes # nb_classes_evn + nb_classes_
    # model = model_now
    # model.eval()
    # criterion_pa = criterion_pa_now
    
    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md,windowlen= window_len, autoencoderType= None)
    dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md,windowlen= window_len, autoencoderType= None)
    dset_test = dataset.load(name=args.dataset, root=pth_dataset, mode='test_1', windowlen= window_len, autoencoderType= None)

    dlod_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)
    dlod_ev_now = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    
    dlod_test = torch.utils.data.DataLoader(dset_test, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    print('==>Evaluation of Stage 2 Dataset on Stage 1 Model')
    model.eval()
    with torch.no_grad():
        feats, _ = utils_CGCD.evaluate_cos_(model, dlod_test)
    
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))  # Cosine Similarity is between -1 to 1
        _, preds_lb = torch.max(cos_sim, dim=1)  # Returns values and the class
        # print(_.detach().cpu().numpy())

        y_temp = torch.tensor(dlod_test.dataset.ys)
        old_classes = np.nonzero( torch.where(y_temp < nb_classes, 1, 0))
        new_classes =  np.nonzero(torch.where(y_temp >= nb_classes, 1, 0))

        preds = preds_lb.detach().cpu().numpy()

        # print(np.array(dlod_test.dataset.ys)[new_classes], preds[new_classes])
        acc_0 =  accuracy_score(np.array(dlod_test.dataset.ys)[old_classes], preds[old_classes])
        acc_1 =  accuracy_score(np.array(dlod_test.dataset.ys)[new_classes], preds[new_classes])
        
        # acc_0, _ = utils_CGCD._hungarian_match_(np.array(dlod_test.dataset.ys)[old_classes], preds[old_classes])
        # acc_1, _ = utils_CGCD._hungarian_match_(np.array(dlod_test.dataset.ys)[new_classes], preds[new_classes])

    print("Stage 2 Test Dataset on Stage 1 trained model, Seen Accuracy {}  Unseen Accuracy  {}".
          format(acc_0, acc_1))
    
    ####
    if(only_test != True):
        load_exempler = False
        if(load_exempler):
            print("Loading Saved Exempler ..")
            expler_s = torch.load(pth_rst_exp + 'expler_s_tensor_model_{}_{}_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}.pt'.format(args.dataset,args.model,window_len, args.sz_embedding, args.alpha, args.mrg))
        else:
            print('==> Calc. proxy mean and sigma for exemplar..')
            with torch.no_grad():
                feats, _ = utils_CGCD.evaluate_cos_(model, dlod_tr_prv)
                feats = losses.l2_norm(feats)
                expler_s = feats.std(dim=0).to(device)
                torch.save(expler_s, pth_rst_exp + 'expler_s_tensor_model_{}_{}_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}.pt'.format(args.dataset,args.model,window_len, args.sz_embedding, args.alpha, args.mrg))

        # print("Exmpler {} \n".format(expler_s))

    nb_classes_now = dset_tr_now.nb_classes()
    criterion_pa_now = losses.Proxy_Anchor(nb_classes=nb_classes_now, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).to(device)
    criterion_pa_now.proxies.data[:nb_classes_prv] = criterion_pa.proxies.data  # Reusing proxies from Initial Step
    
    print("NB Classes Now :", nb_classes_now)

    bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0., 0., 0., 0., 0.
    bst_epoch_a, bst_epoch_o, bst_epoch_n = 0., 0., 0.

    model_now = copy.deepcopy(model)
    model_now = model_now.to(device)



    param_groups = [
        {'params': list(set(model_now.parameters()).difference(set(model_now.embedding.parameters()))) if args.gpu_id != -1 else list(set(model_now.module.parameters()).difference(set(model_now.module.model.embedding.parameters())))},
        {'params': model_now.embedding.parameters() if args.gpu_id != -1 else model_now.embedding.parameters(), 'lr': float(args.lr) * 1},]
    # param_groups= [{'params': model_now.parameters()}]
    
    param_groups.append({'params': criterion_pa_now.parameters(), 'lr': float(args.lr)* 100 })  #*100

    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    print("Classes in the initial Step: {}, Incremental Step: {}".format(nb_classes,nb_classes_now))
    # Training the Incremental Step..........
    pth_rst_exp = 'CGCD-main/src/Saved_Models/Incremental/G-Baseline/' + args.dataset + '/'
    pth_rst_exp_log = pth_rst_exp  + "result_G-Baseline.txt"
    os.makedirs(pth_rst_exp, exist_ok= True)

    epoch = 0
    best_weighted_acc = 0
    es_count = 0

    load_pretrained = False
    if(load_pretrained):
        print("Loading Pre-Trained Model ...")
        step = 1
        epoch = 5
        # pth_pth_pretrained = '{}{}_{}_model_last_windowlen_{}_sz_embedding_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len, args.sz_embedding, str(step))
        pth_pth_pretrained = '{}{}_{}_model_last_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}_step_{}_epoch_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len, args.sz_embedding, args.alpha, args.mrg, str(step), epoch)
        
        checkpoint_pretrained = torch.load(pth_pth_pretrained)
        model_now.load_state_dict(checkpoint_pretrained['model_pa_state_dict'])
        criterion_pa_now.proxies = checkpoint_pretrained['proxies_param']
        args.warm  = -1
        
    
    
    ep =100
    if(only_test!=True):
        for epoch in range(0,ep):  #args.nb_epochs
            model_now.train()
            ####
            bn_freeze = args.bn_freeze
            if bn_freeze:
                modules = model_now.modules() if args.gpu_id != -1 else model_now.module.model.modules()
                for m in modules:
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            if args.warm > 0:
                # Early Stopping to remain one during the warnmp
                if args.gpu_id != -1:
                    unfreeze_model_param = list(model_now.embedding.parameters()) + list(criterion_pa_now.parameters())
                else:
                    unfreeze_model_param = list(model_now.module.model.embedding.parameters()) + list(criterion_pa_now.parameters())

                if epoch == 0:
                    for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = False
                if epoch == args.warm:
                    for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = True

            pbar = tqdm(enumerate(dlod_tr_now))
            # print(next(iter(dlod_tr_now_m)))
            total_train_loss, total_kd_loss,total_pa_loss  = 0.0, 0.0, 0.0
            total_data = 0
            for batch_idx, (x, y, z) in pbar:
                feats = model_now(x.squeeze().to(device))
                
                #### Exampler
                y_n = torch.where(y >= nb_classes_prv, 1, 0)  # New Classes
                y_o = y.size(0) - y_n.sum()                  # Number of Old Classes
                # y_o = y.size(0)
                if y_o > 0:    # if number of old classes in the batch are greater then zero
                    y_sp = torch.randint(nb_classes_prv, (y_o,))   # Select y_o random integers from 0 to nb_classes_prv  
                                            # mean                      std
                    feats_sp = torch.normal(criterion_pa.proxies[y_sp], expler_s).to(device)   # Selecting random old proxies given mean and standard deviation
                    y = torch.cat((y, y_sp), dim=0)
                    feats = torch.cat((feats, feats_sp), dim=0)
                loss_pa = criterion_pa_now(feats, y.squeeze().to(device))

                #### KD
                y_o_msk = torch.nonzero(y_n)  # Provides index of all new classes
                
                if y_o_msk.size(0) > 1:
                    y_o_msk = torch.nonzero(y_n).squeeze()
                    x_o = torch.unsqueeze(x[y_o_msk[0]], dim=0)  # Just first element

                    feats_n = torch.unsqueeze(feats[y_o_msk[0]], dim=0)
                    for kd_idx in range(1, y_o_msk.size(0)): # After 1st Index

                        x_o_ = torch.unsqueeze(x[y_o_msk[kd_idx]], dim=0)
                        x_o = torch.cat((x_o, x_o_), dim=0)
                        feats_n_ = torch.unsqueeze(feats[y_o_msk[kd_idx]], dim=0)
                        feats_n = torch.cat((feats_n, feats_n_), dim=0)
                    with torch.no_grad():
                        feats_o = model(x_o.squeeze().to(device))
                    feats_n = feats_n.to(device)
                    # FRoST
                    loss_kd = torch.dist(F.normalize(feats_o.view(feats_o.size(0) * feats_o.size(1), 1), dim=0).detach(), F.normalize(feats_n.view(feats_o.size(0) * feats_o.size(1), 1), dim=0))
                else:
                    loss_kd = torch.tensor(0.).to(device)

                loss = loss_pa * 10.0 + loss_kd * 1.0  

                total_train_loss += loss.item()*x.size(0)
                total_kd_loss += loss_kd.item()*x.size(0)
                total_pa_loss += loss_pa.item()*x.size(0)

                total_data += x.size(0)

                opt.zero_grad()
                loss.backward()
                opt.step()

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}/{:.6f}/{:.6f}'.format(epoch, batch_idx + 1, len(dlod_tr_now), 100. * batch_idx / len(dlod_tr_now), loss.item(), loss_pa.item(), loss_kd.item()))
            
            print("Epoch {} Train Total Loss {:.4f}, PA Loss {:.4f}, KD Loss {:.4f}".format(epoch, total_train_loss/total_data,total_pa_loss/total_data, total_kd_loss/total_data  ))
            # scheduler.step()  # Sungho asked to remove scheduler

            ####
            print('==> Evaluation..')
            model_now.eval()
            with torch.no_grad():
                feats, _ = utils_CGCD.evaluate_cos_(model_now, dlod_ev_now)
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa_now.proxies))
                _, preds_lb = torch.max(cos_sim, dim=1)
                # preds_lb = preds_lb.detach().cpu().numpy()

                y = torch.tensor(dlod_ev_now.dataset.ys).type(torch.LongTensor)
                
                seen_classes = torch.where(y < nb_classes, 1, 0)
                seen_classes_idx = torch.nonzero(seen_classes)
                unseen_classes = torch.where(y >= nb_classes, 1, 0)
                unseen_classes_idx = torch.nonzero(unseen_classes)

                if(seen_classes.sum().item()>0):
                    acc_o = calculate_accuracy(y[seen_classes_idx].to(device), preds_lb[seen_classes_idx].to(device))
                else:
                    acc_o =0
                if(unseen_classes.sum().item()>0):
                    acc_n = calculate_accuracy(y[unseen_classes_idx].to(device), preds_lb[unseen_classes_idx].to(device))
                else:
                    acc_n =0
                            
                acc_a = calculate_accuracy(y.to(device), preds_lb.to(device))
                
                
            # if acc_a > bst_acc_a:  # Overall Accuracy
            #     bst_acc_a = acc_a
            #     bst_epoch_a = epoch

            # if acc_o > bst_acc_oo:
            #     bst_acc_on = acc_n
            #     bst_acc_oo = acc_o
            #     bst_epoch_o = epoch
            # if acc_n > bst_acc_nn:
            #     bst_acc_nn = acc_n
            #     bst_acc_no = acc_o
            #     bst_epoch_n = epoch

            # print('Valid Epoch: {} Acc: {:.4f}/{:.4f}/{:.4f} Best result: {}/{}/{} {:.4f}/{:.4f}/{:.4f}'.format(epoch,
            #                                                                                                     acc_a, acc_o, acc_n,
            #                                                                                                     bst_epoch_a, bst_epoch_o, bst_epoch_n,
            #                                                                                                     bst_acc_a, bst_acc_oo, bst_acc_nn))
            print("Valid Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".format(acc_o, acc_n, acc_a))
            
            with open(pth_rst_exp_log, "a+") as fval:
                fval.write('Epoch {:.2f}, Valid Accuracies Seen {:.2f}, Unseen {:.2f}, Overall {:.2f} \n'.format(epoch, acc_o, acc_n, acc_a))
            
            step = 2
            if((acc_o + acc_n + acc_a)/3 > best_weighted_acc):
                best_seen = acc_o
                best_unseen = acc_n
                best_overall = acc_a
                best_weighted_acc = (best_seen + best_unseen + best_overall)/3
                es_count = 0
                print("Got Better Model with  Seen {},  Unseen Accuracies {},  Overalll {} and  Average as {} ".format( best_seen,best_unseen, best_overall,best_weighted_acc))
                torch.save({'model_pa_state_dict': model_now.state_dict(), 'proxies_param': criterion_pa_now.proxies}, '{}{}_{}_model_last_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}_step_{}_epoch_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len, args.sz_embedding, args.alpha, args.mrg, str(step), epoch))
            else:                                                                                                     
                es_count +=1 
                print("Early Stopping Count ", es_count)
                
            if(epoch< args.warm):
                es_count =0    # If we are having warmup then no early stopping
            
            if(es_count ==10):
                with open(pth_rst_exp_log, "a+") as fval:
                    fval.write('Best Valid Accuracies Seen {}, Unseen {}, Overall {}, Average \n'.format( best_seen, best_unseen, best_overall, best_weighted_acc))
                
                print("Best  Valid Accuracies Seen {}, and Unseen {},  Overall {}, Average {} ".format( best_seen, best_unseen, best_overall, best_weighted_acc))
                print("Early Stopping")
                break
        
        with open(pth_rst_exp_log, "a+") as fval:
            fval.write('Best Valid Accuracies Seen {}, Unseen {}, Overall {}, Average {}\n'.format( best_seen, best_unseen, best_overall, best_weighted_acc)) 


    # ************************Testing Code Starts Here*************************

    print("Testing On Test Dataset ...")
    step = 2
    epoch = 28
    # pth_pth_test ='CGCD-main/src/Saved_Models/Incremental/G-Baseline/'+args.dataset +'/'+ args.dataset +'_resnet18_model_last_step_1.pth'
    pth_pth_test = '{}{}_{}_model_last_windowlen_{}_sz_embedding_{}_alpha{}_mrg_{}_step_{}_epoch_{}.pth'.format(pth_rst_exp, args.dataset, args.model,window_len, args.sz_embedding, args.alpha, args.mrg, str(step), epoch)

    cfg_test = {'weights_path': 'CGCD-main/src/Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
         'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
            'output_size':nb_classes_now,'weight_norm_dim': ''}

    criterion_pa_test = losses.Proxy_Anchor(nb_classes=nb_classes_now, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).to(device)

    model_test = ModelGen_new(cfg_test).create_model().to(device)
    model_test = model_test.to(device)
    
    checkpoint_test = torch.load(pth_pth_test)
    model_test.load_state_dict(checkpoint_test['model_pa_state_dict'])
    criterion_pa_test.proxies = checkpoint_test['proxies_param']



    model_test.eval()
    with torch.no_grad():
        feats, _ = utils_CGCD.evaluate_cos_(model_test, dlod_test)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa_test.proxies))
        _, preds_lb = torch.max(cos_sim, dim=1)
        # preds_lb = preds_lb.detach().cpu().numpy()

        y = torch.tensor(dlod_test.dataset.ys).type(torch.LongTensor)
        
        seen_classes = torch.where(y < nb_classes, 1, 0)
        seen_classes_idx = torch.nonzero(seen_classes)
        unseen_classes = torch.where(y >= nb_classes, 1, 0)
        unseen_classes_idx = torch.nonzero(unseen_classes)

        if(seen_classes.sum().item()>0):
            acc_o = calculate_accuracy(y[seen_classes_idx].to(device), preds_lb[seen_classes_idx].to(device))
        else:
            acc_o =0
        if(unseen_classes.sum().item()>0):
            acc_n = calculate_accuracy(y[unseen_classes_idx].to(device), preds_lb[unseen_classes_idx].to(device))
        else:
            acc_n =0
                        
        acc_a = calculate_accuracy(y.to(device), preds_lb.to(device))
        
            
        print("Test Accuracies Seen {}, Unseen {}, Overall {}".format(acc_o, acc_n, acc_a))