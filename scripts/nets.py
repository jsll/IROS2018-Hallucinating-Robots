# Network definition and initialization (All convolutional auto-encoders)
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable

class AutoEnc(nn.Module):
    def __init__(self, parameters, gamma=1.):
        super(AutoEnc, self).__init__()
        assert len(parameters) > 0
        assert gamma > 0
        
        self.gamma = gamma
        last_p = parameters[0]
        e_convs = [nn.Conv1d(1, last_p, 5, stride=2, padding=3)]
        e_bns = [nn.BatchNorm1d(last_p)]
        d_convs = [nn.ConvTranspose1d(last_p, 1, 5, stride=2, padding=3, output_padding=1)]
        d_bns = [nn.BatchNorm1d(1)]
        if len(parameters) > 1:
            for p in parameters[1:]:
                e_convs.append(nn.Conv1d(last_p, p, 5, stride=2, padding=2))
                e_bns.append(nn.BatchNorm1d(p))
                d_convs.append(nn.ConvTranspose1d(p, last_p, 5, stride=2, padding=2, output_padding=1))
                d_bns.append(nn.BatchNorm1d(last_p))
                last_p = p
        
        d_convs.reverse()
        d_bns.reverse()
        self.convs = nn.ModuleList(e_convs + d_convs)
        self.bns = nn.ModuleList(e_bns + d_bns)
    
    def forward(self, x):
        for i in range(len(self.convs)):
            #print(x.size())
            x = F.relu(self.bns[i](self.convs[i](x)))
            
        return tc.clamp(x, max=1) ** self.gamma

class AutoEncSkipAdd(AutoEnc):
    def forward(self, x):
        l = len(self.convs)
        skip_num = l / 2 - 1
        skip_x = [None] * skip_num
        for i in range(l):
            x = F.relu(self.bns[i](self.convs[i](x)))
            if i < skip_num:
                #print("saving %d" % (i))
                skip_x[i] = x
            elif i > skip_num and i < l-1:
                #print("adding %d to %d" % (l-2-i, i))
                x = x + skip_x[l-2-i]
            
        return tc.clamp(x, max=1) ** self.gamma
    
class AutoEncSmall(AutoEnc):
    def __init__(self, gamma=1):
        super(AutoEncSmall, self).__init__([16,32,64], gamma)
        
class AutoEncBig(AutoEnc):
    def __init__(self, gamma=1):
        super(AutoEncSmall, self).__init__([64,128,256], gamma)

# old implementations
class AutoEncSmall_(nn.Module):
    def __init__(self):
        super(AutoEncSmall, self).__init__()
        self.e_conv1 = nn.Conv1d(1, 16, 5, stride=2, padding=16)
        self.e_bnorm1 = nn.BatchNorm1d(16)
        self.e_conv2 = nn.Conv1d(16, 32, 5, stride=2, padding=2)
        self.e_bnorm2 = nn.BatchNorm1d(32)
        self.e_conv3 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.e_bnorm3 = nn.BatchNorm1d(64)
        
        self.d_conv1 = nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1)
        self.d_bnorm1 = nn.BatchNorm1d(32)
        self.d_conv2 = nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1)
        self.d_bnorm2 = nn.BatchNorm1d(16)
        self.d_conv3 = nn.ConvTranspose1d(16, 1, 5, stride=2, padding=16, output_padding=1)
        self.d_bnorm3 = nn.BatchNorm1d(1)

    def forward(self, x):
        #print("in", x.size())
        x = F.relu(self.e_bnorm1(self.e_conv1(x)))
        #print("c1", x.size())
        x = F.relu(self.e_bnorm2(self.e_conv2(x)))
        #print("c2", x.size())
        x = F.relu(self.e_bnorm3(self.e_conv3(x)))
        #print("c3", x.size())
        #x = x.view(-1, 1, 36)
        #print("dec", x.size())
        x = F.relu(self.d_bnorm1(self.d_conv1(x)))
        #print("c1", x.size())
        x = F.relu(self.d_bnorm2(self.d_conv2(x)))
        #print("c2", x.size())
        x = F.relu(self.d_bnorm3(self.d_conv3(x)))
        #print("c3", x.size())
        
        return x
    

class AutoEncBig_(nn.Module):
    def __init__(self):
        super(AutoEncBig, self).__init__()
        self.e_conv1 = nn.Conv1d(1, 64, 5, stride=2, padding=16)
        self.e_bnorm1 = nn.BatchNorm1d(64)
        self.e_conv2 = nn.Conv1d(64, 128, 5, stride=2, padding=2)
        self.e_bnorm2 = nn.BatchNorm1d(128)
        self.e_conv3 = nn.Conv1d(128, 256, 5, stride=2, padding=2)
        self.e_bnorm3 = nn.BatchNorm1d(256)
        
        self.d_conv1 = nn.ConvTranspose1d(256, 128, 5, stride=2, padding=2, output_padding=1)
        self.d_bnorm1 = nn.BatchNorm1d(128)
        self.d_conv2 = nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.d_bnorm2 = nn.BatchNorm1d(64)
        self.d_conv3 = nn.ConvTranspose1d(64, 1, 5, stride=2, padding=16, output_padding=1)
        self.d_bnorm3 = nn.BatchNorm1d(1)

    def forward(self, x):
        #print("in", x.size())
        x = F.relu(self.e_bnorm1(self.e_conv1(x)))
        #print("c1", x.size())
        x = F.relu(self.e_bnorm2(self.e_conv2(x)))
        #print("c2", x.size())
        x = F.relu(self.e_bnorm3(self.e_conv3(x)))
        #print("c3", x.size())
        #x = x.view(-1, 1, 36)
        #print("dec", x.size())
        x = F.relu(self.d_bnorm1(self.d_conv1(x)))
        #print("c1", x.size())
        x = F.relu(self.d_bnorm2(self.d_conv2(x)))
        #print("c2", x.size())
        x = F.relu(self.d_bnorm3(self.d_conv3(x)))
        #print("c3", x.size())
        
        return x

class SkewedMSE(Module):
    def __init__(self, alpha):
        super(SkewedMSE, self).__init__()
        self.a = alpha
    
    def forward(self, prediction, target):
        x = prediction - target
        
        loss = (1./self.a) ** 2 * F.mse_loss(prediction, target, reduce=False) * tc.pow(tc.sign(x)+self.a, 2)
        return tc.mean(loss)
    
class RMSLE(Module):    
    def forward(self, prediction, target):
        return tc.sqrt(F.mse_loss(tc.log(prediction+1), tc.log(target+1))) #todo: replace with tc.log1p

def train(net, dataloader, criterion, optimizer, n_of_epochs, augmentations=None):
    net.train()
    
    for epoch in range(n_of_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            #augment the data (if necessary)
            if augmentations:
                data = augmentations(data)
            
            # get the inputs
            inputs = data['laser'].unsqueeze_(1).float()
            truths = data['truth'].unsqueeze_(1).float()
            
            # wrap them in Variable
            inputs, truths = to_variable(inputs), to_variable(truths)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, truths)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 400 == 399:    # print every 400 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
    
    net.eval()

def test(net, dataloader, criterion, augmentations=None):
    total_loss = 0
    i = 0
    for data in dataloader:
        #augment the data (if necessary)
        if augmentations:
            data = augmentations(data)
        
        # get the inputs
        inputs = data['laser'].unsqueeze_(1).float()
        truths = data['truth'].unsqueeze_(1).float()

        # wrap them in Variable
        inputs, truths = to_variable(inputs), to_variable(truths)

        #network pass
        outputs = net(inputs)
        loss = criterion(outputs, truths)
        total_loss += loss.data[0]
        i += 1

    return total_loss / i

def to_cuda(net, use_cuda=True):
    if use_cuda:
        if tc.cuda.device_count() > 1:
            print("Let's use", tc.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)

        if tc.cuda.is_available():
            print("loading network on CUDA")
            net.cuda()
        else:
            print("CUDA not available")
    return net
            
def to_variable(tensor, use_cuda=True):
    if use_cuda and tc.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)