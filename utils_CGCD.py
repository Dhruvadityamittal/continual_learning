import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import logging
import losses
import json
from tqdm import tqdm

import math
import os
import sys

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, Birch, AffinityPropagation, MeanShift, OPTICS, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import combinations
# import hdbscan
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy


def cluster_pred_2_gt(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size  
    D = max(y_pred.max(), y_true.max()) + 1   
    w = np.zeros((D, D), dtype=np.int64) 
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1    ## Similar to a Confusion matrix
    _, col_idx = linear_sum_assignment(w.max() - w)
    return col_idx

def pred_2_gt_proj_acc(proj, y_true, y_pred):
    proj_pred = proj[y_pred]
    return accuracy_score(y_true, proj_pred)

def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return

def _hungarian_match_(y_pred, y_true):
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # ind = linear_sum_assignment(w.max() - w)
    # acc = 0.
    # for i in range(D):
    #     acc += w[ind[0][i], ind[1][i]]
    # acc = acc * 1. / y_pred.size
    # return acc

    ind_arr, jnd_arr = linear_sum_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output




def predict_batchwise(model, dataloader):
    # device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.to(device)) 
                    # J, _ = model(J.cuda())

                for j in J:    # For each example in the Batch
                    A[i].append(j)
            
        
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, _ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean



def evaluate_cos_ev(model, dataloader, proxies_new):
    nb_classes = dataloader.dataset.nb_classes()

    # acc, _ = _hungarian_match_(clustering.labels_, np.array(dlod_tr_n.dataset.ys)) #pred, true

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_(model, dataloader):
    # nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    # X = torch.softmax(X, dim=1)

    # cos_sim = F.linear(X, X)  # 2158x2158
    # v, i = cos_sim.topk(1 + 5)
    # T1 = T[i[:, 1]]
    # V = v[:, 1].float().cpu()

    # return X[i[:, 1]], T, T1
    # return X, T, T1
    return X, T

    # clustering = AffinityPropagation(damping=0.5).fit(X.cpu().numpy())  ###
    # u, c = np.unique(clustering.labels_, return_counts=True)
    # print(u, c)

    # get predictions by assigning nearest 8 neighbors with cosine

    xs = []

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall






def saveImage(strPath, input):
    normalize_un = transforms.Compose([transforms.Normalize(mean=[-0.4914/0.2471, -0.4824/0.2435, -0.4413/0.2616], std=[1/0.2471, 1/0.2435, 1/0.2616])])

    sqinput = input.squeeze()
    unnorminput = normalize_un(sqinput)
    npinput = unnorminput.cpu().numpy()
    npinput = np.transpose(npinput, (1,2,0))
    npinput = np.clip(npinput, 0.0, 1.0)

    plt.imsave(strPath, npinput)


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def evaluate_cos(model, dataloader, epoch):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)

    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X).to(device)
    T = T.to(device)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float()

    recall = []
    r_at_k = calc_recall_at_k(T, Y, 1)
    recall.append(r_at_k)
    print("R@{} : {:.3f}".format(1, 100 * r_at_k))
    return recall

# m - model, y - true labels, v - cosine similarity
def show_OnN(m, y, v, nb_classes, pth_result, thres=0., is_hist=False, iter=0):
    oo_i, on_i, no_i, nn_i = 0, 0, 0, 0
    o, n = [], []

    for j in range(m.size(0)):
        
        if y[j] < nb_classes:
            o.append(v[j].cpu().numpy())   # Appending Cosine Similarity
            if v[j] >= thres:  # if cosine similarity is greater than the threshold
                oo_i += 1      # Old data close to old proxies
            else:
                on_i += 1     # Old data not close to old proxies
        else:
            n.append(v[j].cpu().numpy())
            if v[j] >= thres:
                no_i += 1    # New data close to old proxies
            else:
                nn_i += 1   # New data away from old proxies

    if is_hist is True:

        plt.hist((o, n), histtype='bar', bins=100)
        plt.legend(['Old','New'])
        plt.savefig(pth_result + '/' + 'Init_Split_' + str(iter) + '.png')
        
        plt.close()
        # plt.clf()

    print('Init. Split result(0.)\t oo: {}\t on: {}\t no: {}\t nn: {}'.format(oo_i, on_i, no_i, nn_i))


def visualize_proxy_anchors(model, dloader, proxies, dataset, embedding_size, classes, step, method):


    model.eval()
    with torch.no_grad():
        feats, ys = evaluate_cos_(model, dloader)
    
    if(method == 'pca'):
        pca = PCA(n_components=2, random_state=42)
        pca.fit(feats)
        feats_transformed = pca.transform(feats)

        embedded_data = pca.transform(losses.l2_norm(proxies).to('cpu').detach().numpy())
    elif(method =='tsne'):
        tsne = TSNE(n_components=2, random_state=42)
        
        # fits_transformed = tsne.transform(feats)
        orignal_len_feats = len(feats)
       
        feats = np.concatenate((feats,losses.l2_norm(proxies).to('cpu').detach().numpy() ))
        
        feats_transformed = tsne.fit_transform(feats)
        embedded_data = feats_transformed[orignal_len_feats:]  # Proxy Anchors
        feats_transformed = feats_transformed[:orignal_len_feats]


    # clst_a = AffinityPropagation(damping =0.75).fit(feats_transformed) # 0.75
    # p, c = np.unique(clst_a.labels_, return_counts=True)  
    # nb_classes_k = len(p)   # Number of Determined Unique Clusters
    # print("Number of Clusters determined for New Classes.", nb_classes_k)
    # print(p,c)

    from sklearn.cluster import KMeans
    from collections import Counter


    kmeans = KMeans(n_clusters=classes, random_state=0)
    # ari = adjusted_rand_score(true_labels, clusters)
    # nmi = normalized_mutual_info_score(true_labels, clusters)
    
    
    clusters = kmeans.fit_predict(feats_transformed)
    
    def mapped_clusters(clusters, ys):
        mapped_cl = [-1]*len(clusters)
        for clas in np.unique(ys):
            specific_class_indices = np.where(ys == clas)
        
            most_matching_clusters = Counter(clusters[specific_class_indices])
            most_matching_cluster_label = most_matching_clusters.most_common(1)[0][0]
            
            for j in range(len(ys)):
                if(clusters[j]==most_matching_cluster_label):
                    mapped_cl[j] = clas
            
        return mapped_cl
    
    # print(ys[0:50], mapped_clusters(clusters, ys)[0:50])
    print("Clustering Accuracy Score: {:.2f} %".format(accuracy_score(ys, mapped_clusters(clusters, ys))))

    plt.figure(figsize=(15, 15))
    colors = plt.cm.tab20(np.linspace(0, 1, classes))
    Mapping = {}

    if dataset == 'realworld':
        Mapping[0], Mapping[1], Mapping[2], Mapping[3] = 'Climbing Down', 'Climbing Up', 'Jumping', 'Lying'
        Mapping[4], Mapping[5], Mapping[6], Mapping[7] = 'Running', 'Sitting', 'Standing', 'Walking'
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'magenta', 'teal']

    handles = []
    labels = []

    for class_idx in range(classes):
        class_data = feats_transformed[ys == class_idx]

        plt.scatter(class_data[:, 0], class_data[:, 1], alpha=0.3, color=colors[class_idx], s=30)
        # plt.scatter(embedded_data[class_idx][0], embedded_data[class_idx][1], s=200, color="black", marker='x', alpha=1)
        
        if len(list(Mapping.keys())) == 0:
            label = f'Class {class_idx}'
            
        else:
            label = Mapping[int(class_idx)]
            
        # plt.annotate(label, (embedded_data[class_idx][0], embedded_data[class_idx][1]), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=20)
        # Collect handles and labels for legend
        handles.append(plt.scatter([], [], color=colors[class_idx], label=label))
        labels.append(label)

    # Plot legend
    handles.append(plt.scatter([], [], color='black', marker ='x', label='Anchor'))
    labels.append("Anchor")

    plt.legend(handles, labels, fontsize=18)

    plt.xlabel('Dimension 1', fontsize=20)
    plt.ylabel('Dimension 2', fontsize=20)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(False)  # Remove grid lines
    plt.show()
        # Save the figure
    plt.savefig('proxy_visualization_{}_embedding_size_{}_nb_classes_{}_method_{}_step_{}.pdf'.format(dataset,embedding_size,classes, method, step), format='pdf', dpi=300)
    plt.savefig('proxy_visualization_{}_embedding_size_{}_nb_classes_{}_method_{}_step_{}.png'.format(dataset,embedding_size,classes, method, step))

def contrastive_loss(embeddings, labels, proxies, balance):
    # labels = torch.argmax(labels, dim=1)
    # prototypes_values = torch.from_numpy(np.array(list(prototypes.values()))).float().to(device)
    # prototypes_keys = torch.from_numpy(np.array(list(prototypes.keys()))).float().to(device)
    
    proxy_labels = torch.from_numpy(np.arange(0,len(proxies)))
    
    embeddings = torch.cat((embeddings,proxies),0)
    labels = torch.cat((labels,proxy_labels),0)

    labels_numpy = labels.cpu().data.numpy()
    all_pairs = np.array(list(combinations(range(len(labels_numpy)),2)))
    all_pairs = torch.LongTensor(all_pairs)

    print(all_pairs.shape)

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