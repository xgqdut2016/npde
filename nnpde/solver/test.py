import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import qmc
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.float32
'''
fname_y = "data_y.npy"
fname_x = "data_x.npy"
'''
fname_y = "C://Users//2001213226//Desktop//graduation//deeponet//data_y.npy"

fname_x = "C://Users//2001213226//Desktop//graduation//deeponet//data_x.npy"

data_y = np.load(fname_y)
data_x = np.load(fname_x)

mu1 = 2
hr = 1e-2
bound = np.array([0 + hr,1 + mu1 - hr,0 + hr,1 - hr]).reshape(2,2)
class Net(torch.nn.Module):
    def __init__(self, layers, dtype):
        super(Net, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        
        self.dtype = dtype
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
    def forward(self, x):
        dev = x.device
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            temp = torch.eye(x.shape[-1],self.layers[i+1],dtype = self.dtype,device = dev)
            x = h + x@temp
        
        return self.fc[-1](x)
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])  
class LEN():
    def __init__(self):
        pass
    def forward(self,x):
        
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        L = (x1 - bound[0,0])*(bound[0,1] - x1)*(x2 - bound[1,0])*(bound[1,1] - x2)
        return L.reshape(-1,1)
len_data = LEN()
def function_y(net_data,len_data,x):
    if x.requires_grad == False:
        x.requires_grad = True
    return net_data.forward(x)*len_data.forward(x) + 1
def function_u(net_data,len_data,x):
    if x.requires_grad == False:
        x.requires_grad = True
    state = function_y(net_data,len_data,x)
    state_x, = torch.autograd.grad(state,x, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones_like(x[:,0:1]))
    state_xx, = torch.autograd.grad(state_x[:,0:1],x, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones_like(x[:,0:1])) 
    state_yy, = torch.autograd.grad(state_x[:,1:2],x, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones_like(x[:,0:1]))     
    state_lap = state_xx[:,0:1] + state_yy[:,1:2]
    x.requires_grad = False
    return -state_lap.data     
                                                                       


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
def pred_y(solvernet,x):#x = [fun_size,sample,3]
    return solvernet.forward(x)
#-----test
class Testdata():
    def __init__(self,fun_size,sample_size,layers,dtype,seeds):
        self.dim = 2
        tmp = self.quasi_samples(sample_size)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.x = torch.tensor(tmp).type(dtype)
        self.data_y = torch.zeros(fun_size,sample_size)
        self.data_u = torch.zeros(fun_size,sample_size)
        for i in range(fun_size):
            np.random.seed(seeds + i)
            torch.manual_seed(seeds + i)
            net_data = Net(layers,dtype)
            
            self.data_y[i:i + 1,:] = function_y(net_data,len_data,self.x).t()
            self.data_u[i:i + 1,:] = function_u(net_data,len_data,self.x).t()
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
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

#fname1 = "lay%d-ynet-var.pt"%(output_dim)
fname1 = ".//deeponet//lay%d-ynet-var.pt"%(output_dim)
#predata(data_u,data_y,data_x,solvernet,device)

solvernet = solvernet.to(device)
lr = 1e-1

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
data_x = torch.tensor(data_x).type(dtype)
test = Testdata(data_x,test_size,fun_size,layers,dtype,testseeds)
solvernet = solvernet.to('cpu')
print(solvernet)
y = solvernet.forward(test.data_u,test.x)
y_acc = test.data_y
err = (y - y_acc).reshape(-1,1)
L1 = max(abs(err))
L2 = (err**2).sum()
print('test error:L1:%.2e,L2:%.2e'%(L1,L2))

