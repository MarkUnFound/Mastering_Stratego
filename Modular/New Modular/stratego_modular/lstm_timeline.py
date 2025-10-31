import torch
import torch.nn as nn

class LSTMTimeline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMTimeline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        return out

if __name__ == '__main__':
    input_dim = 10
    hidden_dim = 20
    n_layers = 1
    batch_size = 5
    seq_len = 15

    model = LSTMTimeline(input_dim, hidden_dim, n_layers)

    input_data = torch.randn(batch_size, seq_len, input_dim)

    timeline = model(input_data)

    print("Input shape:", input_data.shape)
    print("Output (timeline) shape:", timeline.shape)