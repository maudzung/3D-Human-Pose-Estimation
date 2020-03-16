import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torch


class simple_model(nn.Module):
    def __init__(self, n_keypoints=17, hidden_neurons=1024):
        super(simple_model, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(n_keypoints * 2, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(hidden_neurons, (n_keypoints -1) * 3)
        )

    def forward(self, x):
        out_1 = self.fc1(x)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        # in_4 = torch.cat((out_1, out_3), 1)
        in_4 = out_1 + out_3
        out_4 = self.fc4(in_4)
        out_5 = self.fc5(out_4)

        # print('out_4: {}'.format(out_4.size()))
        # in_6 = torch.cat((x, out_4), 1)
        in_6 = in_4 + out_5
        out_6 = self.fc6(in_6)

        return out_6


if __name__ == '__main__':
    model = simple_model().cuda()
    inp_sample = torch.rand((8,34)).cuda()
    # summary(model.cuda(), (1,34))
    # print(model)
    out_sample = model(inp_sample)
