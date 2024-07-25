import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse,  copy, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import *
import dataset, utils_CGCD, losses, net
from net.resnet import *
from models.modelgen import ModelGen, ModelGen_new
from torchsummary import summary
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import tensorflow as tf
from covmet import ConvNet
from edison_functions import compute_euclidean, contrastive_loss, extract_sample
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyDict(dict):
    def __init__(self, args):
        self.args

class BlankClass:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                    'Implementaion of Edison Paper.')
    # export directory, training and val datasets, test datasets

    parser.add_argument('--dataset', default='pamap', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
    parser.add_argument('--embedding-size', default=128, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
    parser.add_argument('--batch-size', default=512, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
    parser.add_argument('--epochs', default=100, type=int, dest='nb_epochs', help='Number of training epochs.')
    parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
    parser.add_argument('--workers', default=32, type=int, dest='nb_workers', help='Number of workers for dataloader.')
    parser.add_argument('--model', default='resnet18', help='Model for training')  # resnet50 #resnet18  VIT
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate setting')  #1e-4
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
    parser.add_argument('--lr-decay-step', default=5, type=int, help='Learning decay step setting')  #
    parser.add_argument('--lr-decay-gamma', default=0.5, type=float, help='Learning decay gamma setting')
    parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
    parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
    parser.add_argument('--processes', default = 1 , type=int)
    parser.add_argument('--threads', default = 32 , type=int)
    parser.add_argument('--loss', default='Prototypes', help='Criterion for training')
    parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')

    args = parser.parse_args()


    only_test_step1 = True            # Just to test the data on Train_1
    args.model = 'resnet18'        # Model Name
    window_len = 200
    args.use_split_modlue = True
    args.sz_embedding = 128


    pth_rst = 'CGCD-main/src/result/' + args.dataset
    os.makedirs(pth_rst, exist_ok=True)
    pth_rst_exp = 'Saved_Models/Initial/Edison/' + args.dataset + '/' # + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
    os.makedirs(pth_rst_exp, exist_ok=True)

    print("Dataset :", args.dataset)
    ####

    pth_dataset = '../datasets'

    if args.dataset =='wisdm':
        pth_dataset = 'HAR_data/Wisdm/'
    elif args.dataset =='realworld':
        pth_dataset = 'HAR_data/realworld/'
        nb_classes_now = 8
    elif args.dataset =='oppo':
        pth_dataset = 'HAR_data/oppo/'
    elif args.dataset =='pamap':
        pth_dataset = 'HAR_data/pamap/'

    

    dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', windowlen= window_len, autoencoderType= None)
    dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers, drop_last=False)


    dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', windowlen=window_len, autoencoderType= None)
    dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False)

    dset_test_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='test_0', windowlen=window_len, autoencoderType= None)
    dlod_test_0 = torch.utils.data.DataLoader(dset_test_0, batch_size=args.sz_batch, shuffle=False)

    # nb_classes = dset_test_0.nb_classes()
    nb_classes = dset_tr_0.nb_classes()

    # Configuration for the Model
    if(args.model == 'resnet18'):
        cfg = {'weights_path': 'CGCD-main/src/Saved_Models/UK_BioBank_pretrained/mtl_best.mdl', "use_ssl_weights" : False, 'conv_freeze': False, 'load_finetuned_mtl': False,
            'checkpoint_name' :'', 'epoch_len': 2, 'output_size': '', 'embedding_dim': args.sz_embedding, 'bottleneck_dim': None,
                'output_size':nb_classes,'weight_norm_dim': ''}

        model = ModelGen_new(cfg).create_model().to(device)

    if(args.model == 'harnet'):
        repo = 'OxWearables/ssl-wearables'
        model = torch.hub.load(repo, 'harnet5', class_num=nb_classes, pretrained=True).to(device)
        del model.classifier
        model.embedding = nn.Sequential(nn.Linear(model.feature_extractor.layer5[0].out_channels,args.sz_embedding)).to(device)



    gpu_id =0
    param_groups = [
        {'params': list(set(model.parameters()).difference(set(model.embedding.parameters()))) if gpu_id != -1 else list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {'params': model.embedding.parameters() if gpu_id != -1 else model.embedding.parameters(), 'lr': float(args.lr) * 1},]

    # param_groups.append({'params': criterion_pa.parameters(), 'lr': float(args.lr)*100 })
    opt_pa = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)


    # model = ConvNet().to(device)   # Justing testing on a simple model.
    # opt_pa = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    print("Training on Model ->",model.name)
    # print('Training parameters: {}'.format(vars(args)))
    print('Training for {} epochs'.format(args.nb_epochs))
    losses_list = []
    best_recall = [0]
    best_epoch = 0
    best_acc, es_count = 0,0

    version = 1
    step = 0


    loss_1 = nn.CrossEntropyLoss()
    sttmax = nn.Softmax(1)


    for epoch in range(0,args.nb_epochs):  #args.nb_epochs
        if(only_test_step1):
            continue
        model.train()

        if(model.name=='resnet18'):
            if args.bn_freeze:
                modules = model.modules() if gpu_id != -1 else model.module.model.modules()
                for m in modules:
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            if args.warm > 0:
                if gpu_id != -1:
                    unfreeze_model_param = list(model.embedding.parameters())
                else:
                    unfreeze_model_param = list(model.embedding.parameters()) 

                if epoch == 0:
                    for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = False
                if epoch == args.warm:
                    for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                        param.requires_grad = True

        losses_per_epoch = []

        total, correct = 0, 0
        pbar = tqdm(enumerate(dlod_tr_0)) 
        n_support = 10
        total_loss_per_epoch = 0
        nsteps = 0
        
        total_ce_loss = 0
        total_contrastive_losses = 0 
        total_accuracy = 0
        for batch_idx, (x, y, z) in pbar:
            ####
            # Extracting Support and Query points.
            x_support, y_support, x_query, y_query = extract_sample(len(np.unique(y)), n_support, n_support, x, y, seed = epoch,shuffle=True)
            

            #y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
            y_query = torch.from_numpy(y_query).long().to(device)
            y_query_temp = y_query
        
        
            x_support_embeddings = model(x_support.to(device))
            x_query_embeddings = model(x_query.to(device))

            # Get prototypes
            prototypes = dict()
            counters = dict()
            for clas in sorted(np.unique(y_support)):
                p_mean = np.mean(x_support_embeddings.data.cpu().numpy()[y_support==clas],axis=0)
                prototypes[clas] = p_mean
                # counters[clas] = len(x_support_embeddings[y_support==clas])

            # Merging all the prototypes.
            z_proto = torch.from_numpy(np.array(list(prototypes.values()))).float().to(device)
            dists = compute_euclidean(x_query_embeddings,z_proto)
            # log_p = F.softmax(-dists,dim=1)
            log_p = sttmax(-dists)

            for i in range(nb_classes):
                if(np.array_equal(np.array(prototypes[0]), np.array(z_proto[0].cpu().numpy())) == False):
                    print("Key order and prototype order not matching")
            
            key2idx = torch.empty(nb_classes,dtype=torch.long).to(device)
            proto_keys = list(prototypes.keys())
            
            key2idx[proto_keys] = torch.arange(len(proto_keys)).to(device)

            y_query = key2idx[y_query].view(-1,1)
            y_query = tf.keras.utils.to_categorical(y_query.cpu().numpy(), num_classes=len(proto_keys))
            y_query = torch.from_numpy(y_query).float().to(device)
            
            # ce_loss = loss_1(dists, y_query_temp)
            # ce_loss = nn.BCELoss(log_p, y_query)
            ce_loss = loss_1(log_p, y_query)
            # ce_loss = F.binary_cross_entropy(dists, y_query)
            contrastive_losses  = contrastive_loss(x_query_embeddings, y_query,prototypes, balance=True)

            loss = 10*ce_loss + contrastive_losses
            
            
            opt_pa.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_pa.step()

            
            
            total_ce_loss += ce_loss.item()
            total_contrastive_losses += contrastive_losses.item()        
            accuracy = accuracy_score(torch.argmax(log_p, dim = 1).to('cpu').detach().numpy(), y_query_temp.to('cpu').detach().numpy())
            total_accuracy += accuracy
            total_loss_per_epoch +=  loss.item()
            
            nsteps += 1

            # pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss {}: {:.4f}/{:.4f} '.format(
            #         epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), loss.item(), 0))
                
            # print("X Shape {}, Y Shape {}".format(x.shape, y.shape))
            
        print("Epoch {} Total  {} Contrastive  {} CE  {} Acc {:.4f}".format(epoch,total_loss_per_epoch/nsteps, total_contrastive_losses/nsteps, total_ce_loss/nsteps, total_accuracy/nsteps))

        model.eval()

        total_accuracy_val = 0
        n_steps = 0
        with torch.no_grad():
            for x, y, z in dlod_ev:
                embeddings_val = model(x.to(device))
                dists_val = compute_euclidean(embeddings_val,z_proto)
                log_p_val = F.softmax(-dists_val,dim=1)
                val_accuracy = accuracy_score(torch.argmax(log_p_val, dim = 1).to('cpu').detach().numpy(), y.to('cpu').detach().numpy())
                total_accuracy_val += val_accuracy
                n_steps +=1

        print("Epoch {} Val Accuracy {:.4f}".format(epoch, total_accuracy_val/n_steps))
        val_accuracy = total_accuracy_val/n_steps



        if(best_acc<val_accuracy):
            best_acc = val_accuracy
            best_prototypes = prototypes
            z_proto_best = z_proto
            best_model = copy.deepcopy(model)
            print("Got Better Model with Accuracy {:.4f}".format(best_acc))
            tuple_list = [(key, value) for key, value in best_prototypes.items()]
            np.save('{}{}_{}_prototypes_windowlen_{}_embedding_size_{}_version_{}_step_{}.npy'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, version, step), tuple_list)
            torch.save({'model_pa_state_dict': model.state_dict()}, '{}{}_{}_best_windowlen_{}_embedding_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, version, step))

        else:
            es_count +=1
        
        if(es_count==10):
            print("Early Stopping.")
            print("Best Accuracy {}".format(best_acc))
            break



            
    print("Testing Model Accuracy on Test Data..")
    
    # Loading the saved model.
    pth_pth = '{}{}_{}_best_windowlen_{}_embedding_size_{}_version_{}_step_{}.pth'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, version, step)
    checkpoint = torch.load(pth_pth)
    model.load_state_dict(checkpoint['model_pa_state_dict'])  

    # Loading the saved prototypes
    loaded_prototypes = np.load('{}{}_{}_prototypes_windowlen_{}_embedding_size_{}_version_{}_step_{}.npy'.format(pth_rst_exp, args.dataset, args.model, window_len, args.sz_embedding, version, step), allow_pickle= True)
    trained_prototypes_dict = dict()
    for loaded_prototype in loaded_prototypes:
        trained_prototypes_dict[int(loaded_prototype[0])] = loaded_prototype[1] 

    # Merging all the saved prototypes
    z_proto_trained = torch.from_numpy(np.array(list(trained_prototypes_dict.values()))).float().to(device)
    
    # calculating accuracy.
    with torch.no_grad():    
        total_accuracy_test, n_steps = 0, 0
        for x, y, z in dlod_test_0:
            embeddings_test = model(x.to(device))
            dists_test = compute_euclidean(embeddings_test,z_proto_trained)
            log_p_test = F.softmax(-dists_test,dim=1)
            test_accuracy = accuracy_score(torch.argmax(log_p_test, dim = 1).to('cpu').detach().numpy(), y.to('cpu').detach().numpy())
            total_accuracy_test += test_accuracy
            n_steps +=1

        print("Test Accuracy {:.2f} %".format(total_accuracy_test/n_steps))
        


        
