import torch
import numpy as np
import copy
from sklearn import utils
from collections import Counter
from itertools import combinations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
np.seterr(all="ignore")
import warnings
warnings.filterwarnings('ignore')


def extract_sample(n_classes, n_support, n_query, inputs, labels, seed, shuffle=False):
    
    support = []
    y_support = []
    query = []
    y_query = []
    np.random.seed(seed)
    #print(Counter(labels.data.cpu().numpy()))
    K = np.random.choice(np.unique(labels), n_classes, replace=False)
    
    for cls in K:
        datax_cls = copy.deepcopy(inputs[labels == cls])
        perm = utils.shuffle(datax_cls.data.cpu().numpy())
        #print(perm)
        #perm = np.random.permutation(datax_cls)
        # if len(perm) < n_support:
        #     change = n_support - len(perm)
        support_cls = copy.deepcopy(perm[:n_support])
        #print(np.shape(support_cls))
        support.extend(support_cls)
        #print(support)
        y_support.extend([cls]*len(support_cls))
        query_cls = copy.deepcopy(perm[n_support:])
        query.extend(query_cls)
        y_query.extend([cls]*len(query_cls))

        #print(np.shape(support_cls), np.shape(query_cls),np.shape(perm))

    if len(y_query) < 1:
        
        y_query = copy.deepcopy(y_support)
        query = copy.deepcopy(support)
    elif len(np.unique(y_query)) < len(np.unique(y_support)):
        size = int(np.mean(list(Counter(y_query).values())))
        for cls in np.setdiff1d(list(np.unique(y_support)), list(np.unique(y_query))):
            datax_cls = np.where(y_support == cls)[0]
            #print(size, datax_cls)
            idx = np.random.choice(datax_cls,min(len(datax_cls),size),replace=False)
           # print(idx)
            y_query.extend(list(np.array(y_support)[idx]))
            query.extend(list(np.array(support)[idx]))

    support = np.array(support)
    query = np.array(query)
    y_support = np.array(y_support)
    y_query = np.array(y_query)

    if shuffle:
        support, y_support = utils.shuffle(support,y_support,random_state=seed)
        query, y_query = utils.shuffle(query,y_query,random_state=seed)

    support = torch.from_numpy(support).float()
    query = torch.from_numpy(query).float()
    return support, y_support, query, y_query


def compute_euclidean(query, proto):
    #print(np.shape(query), np.shape(proto))
    # query_n = torch.linalg.norm(query, dim=1, keepdims=True)
    # proto_n = torch.linalg.norm(proto, dim=1, keepdims=True)
    #import pdb; pdb.set_trace()
    x = query.unsqueeze(1).expand(query.size(0),proto.size(0),query.size(1))
    y = proto.unsqueeze(0).expand(query.size(0),proto.size(0),query.size(1))
    #print(np.shape(x),np.shape(y))
    return torch.pow(x-y,2).sum(2)


def contrastive_loss(embeddings, labels, prototypes, balance):
    labels = torch.argmax(labels, dim=1)
    prototypes_values = torch.from_numpy(np.array(list(prototypes.values()))).float().to(device)
    prototypes_keys = torch.from_numpy(np.array(list(prototypes.keys()))).float().to(device)
    
    embeddings = torch.cat((embeddings,prototypes_values),0)
    labels = torch.cat((labels,prototypes_keys),0)

    labels_numpy = labels.cpu().data.numpy()
    all_pairs = np.array(list(combinations(range(len(labels_numpy)),2)))
    all_pairs = torch.LongTensor(all_pairs)

    positive_pairs = all_pairs[(labels_numpy[all_pairs[:,0]] == labels_numpy[all_pairs[:,1]]).nonzero()]
    negative_pairs = all_pairs[(labels_numpy[all_pairs[:,0]] != labels_numpy[all_pairs[:,1]]).nonzero()]
		#print(np.shape(positive_pairs), np.shape(negative_pairs))
    if balance:
        negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

    positive_loss = (embeddings[positive_pairs[:,0]] - embeddings[positive_pairs[:,1]]).pow(2).sum(1)
    margin = 1.0
    negative_loss = F.relu(margin - (embeddings[negative_pairs[:,0]] - embeddings[negative_pairs[:,1]]).pow(2).sum(1).sqrt()).pow(2)

    loss = torch.cat([positive_loss, negative_loss], dim = 0)
    return loss.mean()

def get_accuracy_old_new(model, dload_test, nb_classes, nb_classes_now, trained_prototypes_dict):
    model.eval()
    predicted_ys, actual_ys = [], []
    with torch.no_grad():  
        for x, y, z in dload_test:
            embeddings_test = model(x.to(device))

            dists_test = torch.ones((len(embeddings_test),nb_classes_now))*float('inf')
            dists_test = dists_test.float().to(device)        
            
            for c in trained_prototypes_dict.keys():

                z_proto_test = torch.from_numpy(trained_prototypes_dict[c][None,:]).float().to(device)# Adding None just increases the shape from (128,) -> (1,128)
                dist_test = compute_euclidean(embeddings_test,z_proto_test)
                #print(np.shape(dist))
                dists_test[:,c] = torch.squeeze(dist_test)

            log_p_test = F.softmax(-dists_test,dim=1)
            if(len(predicted_ys) ==0):
                predicted_ys =  torch.argmax(log_p_test, dim = 1).to('cpu').detach().numpy()
            else:
                predicted_ys = np.concatenate((predicted_ys, torch.argmax(log_p_test, dim = 1).to('cpu').detach().numpy()),axis=0)
                
            if(len(actual_ys) ==0):
                actual_ys =  y.to('cpu').detach().numpy()
            else:
                actual_ys = np.concatenate((actual_ys, y.to('cpu').detach().numpy()),axis=0)

            acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
            f1_o = f1_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes], average= 'macro')
            acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
            f1_n = f1_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes], average= 'macro')
            acc_all = accuracy_score(predicted_ys, actual_ys)
            f1_all = f1_score(predicted_ys, actual_ys, average= 'macro')
        
        print("Test Dataset, Accuracy : Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".
            format(acc_o, acc_n, acc_all))
        print("Test Dataset, F1 Score : Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".
            format(f1_o, f1_n, f1_all))
            

        return acc_o, f1_o, acc_n, f1_n, acc_all, f1_all
    

def get_accuracy_old_new_LSTM(model, dload_test, nb_classes, nb_classes_now, trained_prototypes_dict):
    model.eval()
    predicted_ys, actual_ys = [], []
    with torch.no_grad():  
        for x, y, z in dload_test:
            
            h = model.init_hidden(len(x))
            h = tuple([each.data for each in h])
            _, _, embeddings_test = model(x.to(device), h, len(x))

            dists_test = torch.ones((len(embeddings_test),nb_classes_now))*float('inf')
            dists_test = dists_test.float().to(device)        
            
            for c in trained_prototypes_dict.keys():

                z_proto_test = torch.from_numpy(trained_prototypes_dict[c][None,:]).float().to(device)# Adding None just increases the shape from (128,) -> (1,128)
                dist_test = compute_euclidean(embeddings_test,z_proto_test)
                #print(np.shape(dist))
                dists_test[:,c] = torch.squeeze(dist_test)

            log_p_test = F.softmax(-dists_test,dim=1)
            if(len(predicted_ys) ==0):
                predicted_ys =  torch.argmax(log_p_test, dim = 1).to('cpu').detach().numpy()
            else:
                predicted_ys = np.concatenate((predicted_ys, torch.argmax(log_p_test, dim = 1).to('cpu').detach().numpy()),axis=0)
                
            if(len(actual_ys) ==0):
                actual_ys =  y.to('cpu').detach().numpy()
            else:
                actual_ys = np.concatenate((actual_ys, y.to('cpu').detach().numpy()),axis=0)

            acc_o = accuracy_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes])
            f1_o = f1_score(predicted_ys[actual_ys<nb_classes], actual_ys[actual_ys<nb_classes], average= 'macro')
            acc_n = accuracy_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes])
            f1_n = f1_score(predicted_ys[actual_ys>=nb_classes], actual_ys[actual_ys>=nb_classes], average= 'macro')
            acc_all = accuracy_score(predicted_ys, actual_ys)
            f1_all = f1_score(predicted_ys, actual_ys, average= 'macro')
        
        print("Test Dataset, Accuracy : Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".
            format(acc_o, acc_n, acc_all))
        print("Test Dataset, F1 Score : Seen {:.2f}, Unseen {:.2f}, Overall {:.2f}".
            format(f1_o, f1_n, f1_all))
            

        return acc_o, f1_o, acc_n, f1_n, acc_all, f1_all