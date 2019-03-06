import torch.nn as nn
import torch.nn.functional as F

#Creation du réseau
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def op_counter(self,data,display):
        inpt = data[0][0]
        par = list(self.parameters())
        nb_conn_oneimg = 0
        nb_red_oneimg = 0

        nBatch = len(data)
        sBatch = inpt.size()[0]

        # -- Operation in conv1 --
        xIn = inpt.size()[1]
        yIn = inpt.size()[2]
        zIn = inpt.size()[0]
        zOut = par[0].size()[0]
        #zIn = par[0].size()[1]
        xFil = par[0].size()[2]
        yFil = par[0].size()[3]

        xOut = xIn - (xFil-1) #Conv2d Padding
        yOut = yIn - (yFil-1) #Conv2d Padding

        nb_conn_oneimg += xOut * yOut * zOut * xFil * yFil * zIn

        # -- Operation in pool1
        pool_size = 2
        xOut = (xOut/pool_size)
        yOut = (yOut/pool_size)
        nb_red_oneimg += xOut * yOut * zOut

        # -- Operation in conv2
        xIn = xOut
        yIn = yOut
        zIn = zOut

        xFil = par[2].size()[2]
        xFil = par[2].size()[3]

        xOut = xIn - (xFil-1) #Conv2d Padding
        yOut = yIn - (yFil-1) #Conv2d Padding
        zOut = par[2].size()[0]

        nb_conn_oneimg += xOut * yOut * zOut * xFil * yFil * zIn

        # -- Operation in pool2
        xOut = (xOut/pool_size)
        yOut = (yOut/pool_size)

        nb_red_oneimg += xOut * yOut * zOut

        # -- Operation in F1
        sIn = xOut * yOut * zOut
        #sIn = par[4].size()[1]
        sOut = par[4].size()[0]

        nb_conn_oneimg += sIn * sOut

        # -- operation in F2
        sIn = sOut
        #sIn = par[6].size()[1]
        sOut = par[6].size()[0]
        
        nb_conn_oneimg += sIn * sOut

        # -- operation in F3
        sIn = sOut
        #sIn = par[8].size()[1]
        sOut = par[8].size()[0]
        
        nb_conn_oneimg += sIn * sOut
        
        if (display):
            nb_conn = int(nb_conn_oneimg * sBatch * nBatch)
            nb_red = int(nb_red_oneimg * sBatch * nBatch)
            total = 2*nb_conn + 3*nb_red
            print("Operations per epoch on the entire dataset")
            print("Addition:",nb_conn)
            print("Multiplication:", nb_conn)
            print("Maximum:", nb_red * 3)
            print("Total:", total)

        return total