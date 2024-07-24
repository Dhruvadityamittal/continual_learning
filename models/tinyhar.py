from models.TinyHAR import TinyHAR_Model
import torch
import torch.nn as nn

class tinyhar(nn.Module):
    def __init__(self, in_size, win_size, embd_size):
        super(tinyhar, self).__init__()
        # B F L C
        self.name = "Tiny HAR"
        self.model = TinyHAR_Model(
            input_shape = (64, 1, win_size, in_size),
            number_class = embd_size,
            filter_num = 100,
            cross_channel_interaction_type = "attn",
            cross_channel_aggregation_type = "FC",
            temporal_info_interaction_type = "lstm",
            temporal_info_aggregation_type = "tnaive"
        )


    def forward(self, x):

        # B, C, L -> B, 1, C, L
        x = torch.unsqueeze(x, 1)

        # print(x.shape)
        
        # B, 1, C, L -> B, 1, L, C
        x = torch.permute(x, (0, 1, 3, 2) )

        # print(x.shape)
        # B, 1, L, C -> B, n_classes
        x = self.model(x)

        # BCE loss
        if len(x.shape) == 1:
            return x.squeeze()
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r = self.forward(x)
        # BCE
        if len(r.squeeze().shape)==1:
            return (torch.sigmoid(r)>0.5).long()
        return r.argmax(dim=1, keepdim=False)