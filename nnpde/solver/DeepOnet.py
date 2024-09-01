import torch
import numpy as np
import torch.nn as nn
import bfgs
import time
from data import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
dtype = torch.float32

fname_y = "data_y.npy"
fname_u = "data_u.npy"
fname_x = "data_x.npy"
'''
fname_y = "C://Users//2001213226//Desktop//graduation//deeponet//data_y.npy"
fname_u = "C://Users//2001213226//Desktop//graduation//deeponet//data_u.npy"
fname_x = "C://Users//2001213226//Desktop//graduation//deeponet//data_x.npy"
'''
data_y = np.load(fname_y)
data_u = np.load(fname_u)
data_x = np.load(fname_x)

class Solvernet(torch.nn.Module):
    def __init__(self, data_size,input_dim,output_dim,layers, dtype):#注意这里的layers只是hidden
        super(Solvernet, self).__init__()
        
        
        self.dtype = dtype
        #------------------------
        self.kernel_size = [10,6,10]
        self.stride = [1,2,2]
        cov = []
        out_dim = [3,1,1]
        in_dim = 1

        for i in range(len(self.kernel_size)):
            cov.append(nn.Conv1d(in_channels = in_dim,out_channels = out_dim[i],kernel_size = self.kernel_size[i],stride = self.stride[i]))
            data_size = (data_size - self.kernel_size[i])//self.stride[i] + 1
            in_dim = out_dim[i]
        self.cov = torch.nn.Sequential(*cov)
        for i in range(len(self.kernel_size)):
            self.cov[i].weight.data = self.cov[i].weight.data.type(dtype)
            self.cov[i].bias.data = self.cov[i].bias.data.type(dtype)
        #------------------------
        #-------branch DNN
        self.bra_layers = [data_size] + layers + [output_dim]
        
        self.bra_layers_hid_num = len(self.bra_layers)-2
        bra_fc = []
        for i in range(self.bra_layers_hid_num+1):
            bra_fc.append(torch.nn.Linear(self.bra_layers[i],self.bra_layers[i+1]))
            bra_fc.append(torch.nn.Linear(self.bra_layers[i + 1],self.bra_layers[i+1]))
        bra_fc.append(torch.nn.Linear(self.bra_layers[-2],self.bra_layers[-1]))
        self.bra_fc = torch.nn.Sequential(*bra_fc)
        for i in range(self.bra_layers_hid_num+1):
            self.bra_fc[2*i].weight.data = self.bra_fc[2*i].weight.data.type(dtype)
            self.bra_fc[2*i].bias.data = self.bra_fc[2*i].bias.data.type(dtype)
            self.bra_fc[2*i + 1].weight.data = self.bra_fc[2*i + 1].weight.data.type(dtype)
            self.bra_fc[2*i + 1].bias.data = self.bra_fc[2*i + 1].bias.data.type(dtype)
        self.bra_fc[-1].weight.data = self.bra_fc[-1].weight.data.type(dtype)
        self.bra_fc[-1].bias.data = self.bra_fc[-1].bias.data.type(dtype)
        #------------------------
        self.layers = [input_dim] + layers + [output_dim]
        self.layers_hid_num = len(self.layers)-2
        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)
        for i in range(self.layers_hid_num):
            self.fc[2*i].weight.data = self.fc[2*i].weight.data.type(dtype)
            self.fc[2*i].bias.data = self.fc[2*i].bias.data.type(dtype)
            self.fc[2*i + 1].weight.data = self.fc[2*i + 1].weight.data.type(dtype)
            self.fc[2*i + 1].bias.data = self.fc[2*i + 1].bias.data.type(dtype)
        self.fc[-1].weight.data = self.fc[-1].weight.data.type(dtype)
        self.fc[-1].bias.data = self.fc[-1].bias.data.type(dtype)
        
    def CNN(self,x):
        h = x.reshape(x.shape[0],1,x.shape[1])
        for i in range(len(self.cov)):
            h = self.cov[i](h)
            h = torch.sin(h)
        #print(h.shape)
        return h.squeeze(dim = 1)
    def branch(self,data_u):
        
        x = self.CNN(data_u)
        dev = x.device
        for i in range(self.bra_layers_hid_num):
            h = torch.sin(self.bra_fc[2*i](x))
            
            h = torch.sin(self.bra_fc[2*i+1](h))
            
            temp = torch.eye(x.shape[-1],self.bra_layers[i+1],dtype = self.dtype,device = dev)
            x = h + x@temp
        
        return self.bra_fc[-1](x)
    def trunk(self, x):
        dev = x.device
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            temp = torch.eye(x.shape[-1],self.layers[i+1],dtype = self.dtype,device = dev)
            x = h + x@temp
        
        return self.fc[-1](x)
    
    def forward(self,data_u,x):
        bra = self.branch(data_u)
        tru = self.trunk(x)
        
        #print(bra.shape,tru.shape,(bra@tru.t()).shape)
        return bra@tru.t()
        
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])  

def pred_y(solvernet,data_u,x):
    return solvernet.forward(data_u,x)
def Loss(solvernet,data_u,x,data_y):
    y = pred_y(solvernet,data_u,x)
    err = ((y - data_y)**2).sum()
    return torch.sqrt(err)
def train_yp(solvernet,data_u,x,data_y,optim,epoch):
    print('Train y&p Neural Network')
    loss = Loss(solvernet,data_u,x,data_y)
    print('epoch:%d,loss:%.2e, time: %.2f'
          %(0, loss.item(), 0.00))
    for it in range(epoch):
        st = time.time()
        def closure():
            loss = Loss(solvernet,data_u,x,data_y)
            optim.zero_grad()
            loss.backward()
            return loss
        optim.step(closure) 
        loss = Loss(solvernet,data_u,x,data_y)
        ela = time.time() - st
        print('epoch:%d,loss:%.2e, time: %.2f'
              %((it+1), loss.item(), ela))
def predata(data_u,data_y,data_x,solvernet,dev):
    data_u = torch.tensor(data_u).type(dtype)
    data_y = torch.tensor(data_y).type(dtype)
    data_x = torch.tensor(data_x).type(dtype)
    data_u = data_u.to(dev)
    data_y = data_y.to(dev)
    data_x = data_x.to(dev)
    solvernet = solvernet.to(dev)

data_size = data_y.shape[1]
input_dim = 2
output_dim = 9
layers = [15,15]
solvernet = Solvernet(data_size,input_dim,output_dim,layers, dtype)

fname1 = "lay%d-ynet-var.pt"%(output_dim)
#fname1 = ".//deeponet//lay%d-ynet-var.pt"%(output_dim)
#predata(data_u,data_y,data_x,solvernet,device)
data_u = torch.tensor(data_u).type(dtype)
data_y = torch.tensor(data_y).type(dtype)
data_x = torch.tensor(data_x).type(dtype)
data_u = data_u.to(device)
data_y = data_y.to(device)
data_x = data_x.to(device)
print(data_u.shape,data_y.shape,data_x.shape)
solvernet = solvernet.to(device)
lr = 1e-1
optim = bfgs.BFGS(solvernet.parameters(),
                      lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
epoch = 10
#rint(type(data_y),type(data_u),type(data_x))
y = solvernet.forward(data_u,data_x)
train_yp(solvernet,data_u,data_x,data_y,optim,epoch)
torch.save(solvernet,fname1)
solvernet = torch.load(fname1)
#-----test
class Testdata():
    def __init__(self,data_x,data_size,fun_size,layers,dtype,seeds):
        self.dim = 2
        tmp = self.quasi_samples(data_size)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.x = torch.tensor(tmp).type(dtype)
        self.data_y = torch.zeros(fun_size,data_size)
        self.data_u = torch.zeros(fun_size,data_x.shape[0])
        for i in range(fun_size):
            np.random.seed(seeds + i)
            torch.manual_seed(seeds + i)
            net_data = Net(layers,dtype)
            
            self.data_y[i:i + 1,:] = function_y(net_data,len_data,self.x).t()
            self.data_u[i:i + 1,:] = function_u(net_data,len_data,data_x).t()
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample

test_size = 32*64
fun_size = 3
lay_wid = 20
testseeds = data_size
layers = [2,lay_wid,lay_wid,1]

test = Testdata(data_x.to('cpu'),test_size,fun_size,layers,dtype,testseeds)
solvernet = solvernet.to('cpu')
y = solvernet.forward(test.data_u,test.x)
y_acc = test .data_y
err = (y - y_acc).reshape(-1,1)
L1 = max(abs(err))
L2 = (err**2).sum()
print('test error:L1:%.2e,L2:%.2e'%(L1,L2))



