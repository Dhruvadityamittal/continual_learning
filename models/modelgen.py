from models.accNet import Resnet, Resnet_new
import torch
import torch.nn as nn
import copy
import freeze



class ModelGen_new:
    def __init__(self, cfg):
        self.cfg = cfg
        
    
    def create_model(self, is_mtl=False):
        cfg = self.cfg

        model = Resnet_new(cfg, is_mtl=is_mtl)
        # .cuda()
        

        if self.cfg["use_ssl_weights"]:   # self.cfg.model_src.use_ssl_weights
            # note that this makes no difference for target training, since we always load the source weights (and thus overwrite the ssl ones)
            self.load_weights(model, cfg["weights_path"])
            # print("using ssl weights")
        
        if self.cfg["conv_freeze"]:   #self.cfg.model_src.conv_freeze
            print("!!!!! Freezing encoder weights")
            freeze.freeze_weights(model)

        if self.cfg["load_finetuned_mtl"]:  # self.cfg.model_src.load_finetuned_mtl
            print("!!!!! Loading finetuned MTL weights")
            mtl_name = cfg["checkpoint_name"].replace("Msrc_", "Mmtl_")   # cfg.checkpoint_name.replace("Msrc_", "Mmtl_")
            self.load_weights(model, f"/netscratch/martelleto/iswc/{mtl_name}")

        return model
    
    def load_weights_from_dict(self, weights_dict, model):
        # only need to change weights name when
        # the model is trained in a distributed manner

        pretrained_dict = weights_dict
        pretrained_dict_v2 = copy.deepcopy(pretrained_dict)  # v2 has the right para names

        # distributed pretraining can be inferred from the keys' module. prefix
        head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
        if head == 'module':
            # remove module. prefix from dict keys
            pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}

        model_dict = model.state_dict()

        # 1. filter out unnecessary keys such as the final linear layers
        #    we don't want linear layer weights either
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict_v2.items()
            if k in model_dict and k.split(".")[0] != "classifier"
        }

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("%d Weights loaded by" % len(pretrained_dict))

    def load_weights(self, model, weights_path):
        # only need to change weights name when
        # the model is trained in a distributed manner

        pretrained_dict = torch.load(weights_path, map_location="cpu")
        self.load_weights_from_dict(pretrained_dict, model)

class ModelGen:
    def __init__(self, cfg):
        self.cfg = cfg
        
    
    def create_model(self, is_mtl=False):
        cfg = self.cfg

        model = Resnet(cfg, is_mtl=is_mtl)
        # .cuda()
        

        if self.cfg.model_src.use_ssl_weights:
            # note that this makes no difference for target training, since we always load the source weights (and thus overwrite the ssl ones)
            self.load_weights(model, cfg.weights_path)
            # print("using ssl weights")
        
        if self.cfg.model_src.conv_freeze:
            print("!!!!! Freezing encoder weights")
            freeze.freeze_weights(model)

        if self.cfg.model_src.load_finetuned_mtl:
            print("!!!!! Loading finetuned MTL weights")
            mtl_name = cfg.checkpoint_name.replace("Msrc_", "Mmtl_")
            self.load_weights(model, f"/netscratch/martelleto/iswc/{mtl_name}")

        return model
    

    
    def load_weights_from_dict(self, weights_dict, model):
        # only need to change weights name when
        # the model is trained in a distributed manner

        pretrained_dict = weights_dict
        pretrained_dict_v2 = copy.deepcopy(pretrained_dict)  # v2 has the right para names

        # distributed pretraining can be inferred from the keys' module. prefix
        head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
        if head == 'module':
            # remove module. prefix from dict keys
            pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}

        model_dict = model.state_dict()

        # 1. filter out unnecessary keys such as the final linear layers
        #    we don't want linear layer weights either
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict_v2.items()
            if k in model_dict and k.split(".")[0] != "classifier"
        }

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("%d Weights loaded by" % len(pretrained_dict))

    def load_weights(self, model, weights_path):
        # only need to change weights name when
        # the model is trained in a distributed manner

        pretrained_dict = torch.load(weights_path, map_location="cpu")
        self.load_weights_from_dict(pretrained_dict, model)
