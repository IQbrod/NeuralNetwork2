import torch.nn as nn
import torch.nn.functional as F

#Creation du réseau
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.conv2 = nn.Conv2d(5, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 14, 3)
        self.conv4 = nn.Conv2d(14, 20, 3)
        self.fc1 = nn.Linear(20 * 5 * 5, 250)
        self.fc2 = nn.Linear(250, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(self.conv1(x))))
        x = self.pool(F.relu(self.conv4(self.conv3(x))))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def op_counter(self,data,display):
        return 0