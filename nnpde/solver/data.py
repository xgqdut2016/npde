import torch
import numpy as np
import torch.nn as nn
from scipy.stats import qmc

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
                                                                       
class INdata():
    def __init__(self,data_size,fun_size,layers,dtype,seeds):
        self.dim = 2
        tmp = self.quasi_samples(data_size)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.x = torch.tensor(tmp).type(dtype)
        self.data_y = torch.zeros(fun_size,data_size)
        self.data_u = torch.zeros(fun_size,data_size)
        for i in range(fun_size):
            np.random.seed(seeds + 10*i)
            torch.manual_seed(seeds + 10*i)
            net_data = Net(layers,dtype)
            
            self.data_y[i:i + 1,:] = function_y(net_data,len_data,self.x).t()
            self.data_u[i:i + 1,:] = function_u(net_data,len_data,self.x).t()
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample

data_size = 32*32
fun_size = 20
lay_wid = 20
layers = [2,lay_wid,lay_wid,lay_wid,1]
dtype = torch.float32
seeds = 0
indata = INdata(data_size,fun_size,layers,dtype,seeds)
'''
np.save('data_y.npy',indata.data_y.detach().numpy())
np.save('data_u.npy',indata.data_u.detach().numpy())
np.save('data_x.npy',indata.x.detach().numpy())

'''
np.save('.//deeponet//data_y.npy',indata.data_y.detach().numpy())
np.save('.//deeponet//data_u.npy',indata.data_u.detach().numpy())
np.save('.//deeponet//data_x.npy',indata.x.detach().numpy())


