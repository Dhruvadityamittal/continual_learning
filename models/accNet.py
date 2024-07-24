import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, List, Dict, Any, cast
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# class Classifier(nn.Module):
#     def __init__(self, input_size=1024, output_size=2):
#         super(Classifier, self).__init__()
#         self.linear1 = torch.nn.Linear(input_size, output_size)

#     def forward(self, x):
#         y_pred = self.linear1(x)
#         return y_pred


# class EvaClassifier(nn.Module):
#     def __init__(self, input_size=1024, nn_size=512, output_size=2):
#         super(EvaClassifier, self).__init__()
#         self.linear1 = torch.nn.Linear(input_size, nn_size)
#         self.linear2 = torch.nn.Linear(nn_size, output_size)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         return x

# class EvaTransformerClassifier(nn.Module):
#     def __init__(self, output_size, downfactor, downorder):
#         super(EvaTransformerClassifier, self).__init__()
#         d_model = 192 # embedding dim size
#         # init transformer with input size and no masking
#         self.transformer = nn.Transformer(d_model=d_model,
#                                           nhead=8,
#                                           num_encoder_layers=6,
#                                           num_decoder_layers=6,
#                                           dim_feedforward=2048,
#                                           dropout=0.4)
#         self.downsample = Downsample(1024, downfactor, downorder)
#         self.linear = torch.nn.Linear(1024, output_size)

#     def forward(self, x):
#         # output from transformer is (seq_len, batch, d_model)
#         # we want (batch, seq_len, d_model)

#         x = self.transformer(x, x)
#         x = x.permute(1, 0, 2)
#         x = x.reshape(x.shape[0], 1024, 3)
#         x = self.downsample(x)
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         return x

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        # print(x.shape)
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)


        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if(len(x.shape) ==2):
            x = torch.unsqueeze(x, 0)
            # print("X Shape ", x.shape)
        identity = x
        # x = self.relu(self.bn1(x))
        x = self.conv1(x)
        # x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x

class Resnet_new(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(self, cfg, is_mtl=False):
        super(Resnet_new, self).__init__()

        self.cfg = cfg
        self.is_mtl = is_mtl
        epoch_len = cfg["epoch_len"]

        n_channels = cfg["n_channels"]
        window_len = cfg["window_len"]
        resnet_version = 1
        self.name = 'resnet18'
        
        

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            
            if epoch_len == 2:
                
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    
                ]

                # cgf = [
                #     (8, 5, 2, 5, 2, 2),
                #     # (128, 21, 2, 5, 2, 2),
                #     # (64, 21, 2, 5, 3, 1),
                #     # (32, 21, 2, 5, 3, 1),
                    
                # ]

            elif epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params

            
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

        # Just checking the expected ouput demention with a dummy variable
        expected_ouput_from_feature_extractor = self.feature_extractor(torch.FloatTensor(np.random.rand(64,n_channels,window_len)))
        ouput_after_flattening = expected_ouput_from_feature_extractor.shape[1]*expected_ouput_from_feature_extractor.shape[2]
        # print("outout of FE", self.feature_extractor(torch.FloatTensor(np.random.rand(64,n_channels,200))).shape)
        

        
        # Classifier input size = last out_channels in previous layer
        in_feats = out_channels # Should be 1024
        
        # print("In Feats", in_feats)

        # No bottleneck means that the encoder is just the resnet encoder
        if self.cfg["bottleneck_dim"]!= None and not self.cfg["bottleneck_dim"] > 0:
            self.feat_dim = in_feats
            self.bottleneck = None  # No bottlenect
            print("No bottleneck")
        elif self.cfg['embedding_dim'] != None and self.cfg["embedding_dim"] > 0:
            print("With Embedding")
            
            self.embedding_size = self.cfg["embedding_dim"]
            
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.gmp = nn.AdaptiveMaxPool1d(1)
            self.embedding = nn.Linear(ouput_after_flattening, self.embedding_size)
            self.feat_dim = self.embedding_size
            self.bottleneck = None
        else:
            # With a bottleneck, we include a fc layer and the bn layer in the encoder
            
            # bottleneck_fc = nn.Linear(in_feats, cfg["bottleneck_dim"]) # Orignal Code
            bottleneck_fc = nn.Linear(ouput_after_flattening, cfg["bottleneck_dim"]) # Modifief Code.
            bn = nn.BatchNorm1d(cfg["bottleneck_dim"])
            self.bottleneck = nn.Sequential(bottleneck_fc, bn)
            self.feat_dim = cfg["bottleneck_dim"]
            self.embedding = None

        # Ensures we have an output
        num_classes = cfg["output_size"]     # cfg.data.output_size
        print("No of classes : ->",num_classes)
        self.num_classes = num_classes
        
        if(not self.embedding):  # If we donot have embedding go inside the statement
            if not self.is_mtl:
                self.fc = nn.Linear(self.feat_dim, num_classes)
                if self.cfg["weight_norm_dim"] >= 0:
                    self.fc = nn.utils.weight_norm(self.fc, dim=cfg["weight_norm_dim"])
            else:
                self.aot_h = nn.Linear(self.feat_dim, 2)
                self.scale_h = nn.Linear(self.feat_dim, 2)
                self.permute_h = nn.Linear(self.feat_dim, 2)
                self.time_w_h = nn.Linear(self.feat_dim, 2)

        weight_init(self)

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.cfg["bottleneck_dim"] > 0:
            backbone_params.extend(self.encoder.parameters())
            backbone_params.extend(self.fc.parameters())
        # case 2)
        else:
            # The last layer of the encoder is the bottleneck fc, so we exclude it
            for module in self.feature_extractor.children():
                backbone_params.extend(module.parameters())
            
            # Extra params: self.bottleneck (includes bn) + classifier fc
            extra_params.extend(self.bottleneck.parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=True,                  # False
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        # modules.append(nn.ReLU(True))  # Removing ReLU
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x, return_feats=False):
        
        # print("X Shape :", x.shape)
        feats = self.feature_extractor(x)
        # print(feats.shape)
        feats = feats.view(feats.shape[0], -1)
        
        
        if self.bottleneck != None:
            feats = self.bottleneck(feats)
        
        if self.embedding:
            # print("Using Embedding")
            # avg_x = self.gap(feats)
            # max_x = self.gmp(feats)
            # x = max_x + avg_x
            # print(x.shape)
            
            feats = self.embedding(feats)

            return feats
        
        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h

        y = self.fc(feats)

        if return_feats:
            return feats, y
        
        return y
    
    



class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(self, cfg, is_mtl=False):
        super(Resnet, self).__init__()

        self.cfg = cfg
        self.is_mtl = is_mtl
        epoch_len = cfg.epoch_len

        n_channels = 3
        resnet_version = 1

        

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (64, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

        # Classifier input size = last out_channels in previous layer
        in_feats = out_channels # Should be 1024

        # No bottleneck means that the encoder is just the resnet encoder
       

        if not self.cfg.bottleneck_dim > 0:
            self.feat_dim = in_feats
            print("No bottleneck")
        elif self.cfg.embedding_dim > 0:
            print("With Embedding")
            self.embedding_size = self.cfg.embedding_dim
            
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.gmp = nn.AdaptiveMaxPool1d(1)
            self.embedding = nn.Linear(in_feats, self.embedding_size)
            self.feat_dim = self.embedding_size
        else:
            # With a bottleneck, we include a fc layer and the bn layer in the encoder
            bottleneck_fc = nn.Linear(in_feats, cfg.bottleneck_dim)
            bn = nn.BatchNorm1d(cfg.bottleneck_dim)
            self.bottleneck = nn.Sequential(bottleneck_fc, bn)
            self.feat_dim = cfg.bottleneck_dim

        # Ensures we have an output
        num_classes = cfg.data.output_size
        print("No of classes : ->",num_classes)
        self.num_classes = num_classes
        
        if(not self.embedding):  # If we donot have embedding go inside the statement
            if not self.is_mtl:
                self.fc = nn.Linear(self.feat_dim, num_classes)
                if self.cfg.weight_norm_dim >= 0:
                    self.fc = nn.utils.weight_norm(self.fc, dim=cfg.weight_norm_dim)
            else:
                self.aot_h = nn.Linear(self.feat_dim, 2)
                self.scale_h = nn.Linear(self.feat_dim, 2)
                self.permute_h = nn.Linear(self.feat_dim, 2)
                self.time_w_h = nn.Linear(self.feat_dim, 2)

        weight_init(self)

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.cfg.bottleneck_dim > 0:
            backbone_params.extend(self.encoder.parameters())
            backbone_params.extend(self.fc.parameters())
        # case 2)
        else:
            # The last layer of the encoder is the bottleneck fc, so we exclude it
            for module in self.feature_extractor.children():
                backbone_params.extend(module.parameters())
            
            # Extra params: self.bottleneck (includes bn) + classifier fc
            extra_params.extend(self.bottleneck.parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x, return_feats=False):
        
        feats = self.feature_extractor(x)
        feats = feats.view(x.shape[0], -1)
        
        # if self.bottleneck is not None:
        #     feats = self.bottleneck(feats)
        
        if self.embedding:
            # print("Using Embedding")
            # avg_x = self.gap(feats)
            # max_x = self.gmp(feats)
            # x = max_x + avg_x
            # print(x.shape)
            feats = self.embedding(feats)

            return feats
        
        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h

        y = self.fc(feats)

        if return_feats:
            return feats, y
        
        return y


def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)