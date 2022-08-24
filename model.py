import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from efficientnet_pytorch import EfficientNet

class CNN_LSTM(nn.Module):
    def __init__(self, lstm_input, lstm_hidden, lstm_layers, bidirectional=True, dropout=True):
        super(CNN_LSTM, self).__init__()
        self.lstm_input = lstm_input
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.cnn = EfficientNet.from_pretrained('efficientnet-b0') ## try b7
        self.cnn._fc = nn.Linear(in_features=self.cnn._fc.in_features, 
                                 out_features=self.lstm_input, bias=True)
        
        self.lstm = nn.LSTM(input_size=self.lstm_input,
                            hidden_size=self.lstm_hidden,
                            num_layers=self.lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        
        if self.bidirectional:
            self.linear = nn.Linear(2*self.lstm_hidden, 1)
        else:
            self.linear = nn.Linear(self.lstm_hidden, 1)
            
        if self.dropout:
            self.drop = nn.Dropout(0.2)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, outfit, outfit_lens, max_outfit=6):
        batch_size, _, C, H, W = outfit.size()
        self.hidden = self.init_hidden(batch_size)

        cnn_in = outfit.view(batch_size * max_outfit, C, H, W)
        cnn_out = self.cnn(cnn_in)
        if self.dropout:
            cnn_out = self.drop(cnn_out)

        rnn_in = cnn_out.view(batch_size, max_outfit, self.lstm_input)
        
        packed_input = pack_padded_sequence(rnn_in, outfit_lens, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, self.hidden)
        rnn_out, states = pad_packed_sequence(packed_output, batch_first=True, total_length=max_outfit)
        
        if self.bidirectional:
            out_forward = rnn_out[range(len(rnn_out)), max_outfit - 1, : self.lstm_hidden]
            out_reverse = rnn_out[:, 0, self.lstm_hidden :]
            out_reduced = torch.cat((out_forward, out_reverse), 1)
        
        if self.dropout:
            outfit_features = self.drop(out_reduced)
            outfit_features = self.linear(outfit_features)
        else:
            outfit_features = self.linear(out_reduced)
            
        outfit_features = torch.squeeze(outfit_features, 1)
        
        outfit_output = torch.sigmoid(outfit_features)
        
        return outfit_output