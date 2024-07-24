import torch.nn as nn
import torch
import torch.nn.functional as F

class DeepConvLSTM(nn.Module):
    def __init__(self, n_classes, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, n_hidden=128, n_layers=1, n_filters=64, 
                filter_size=5, drop_prob=0.5, ):
        super(DeepConvLSTM, self).__init__()
        self.name = "DeepConvLSTM"
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.NB_SENSOR_CHANNELS = NB_SENSOR_CHANNELS
        self.SLIDING_WINDOW_LENGTH = SLIDING_WINDOW_LENGTH
             
        self.conv1 = nn.Conv1d(self.NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)  #n_filters
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.lstm1)
        init_layer(self.lstm2)
        #init_layer(self.lstmAcc)
        #init_layer(self.lstmGyr)

        # init_bn(self.bn1)
        # init_bn(self.bn2)
        # init_bn(self.bn3)
        # init_bn(self.bn4)

        #init_layer(self.dense)
        init_layer(self.fc)        

    def forward(self, x, hidden, batch_size):
        # print("Iniital :",x.shape)
        # x = x.view(-1, self.NB_SENSOR_CHANNELS, self.SLIDING_WINDOW_LENGTH)
        
        #x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(x.shape[-1], -1, self.n_filters)
        #print(x.shape)

        #print(np.shape(x), np.shape(hidden))
        x = self.dropout(x)
        # print(x.shape)
        x, hidden = self.lstm1(x, hidden)
        #print(x.shape)

        x, hidden = self.lstm2(x, hidden)
        #print(x.shape)

        #print(np.shape(x))

        x = x.contiguous().view(-1, self.n_hidden)
        embeddings = x.contiguous().view(batch_size,-1,self.n_hidden)[:,-1,:]
        x = torch.sigmoid(self.fc(x))
        #print(np.shape(x))
        temp = x.view(batch_size, -1, self.n_classes)
        #print(np.shape(temp))
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
        return out, hidden, embeddings
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        device = 'cuda'
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
    
def init_layer(layer):

    if type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
 
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)