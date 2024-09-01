import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
from scipy.stats import qmc
import bfgs
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

def UU(X,prob):
    tmp = 0
    if prob == 1:
        for i in range(X.shape[1]):
            tmp += X[:,i]**2
        return tmp.reshape(-1,1)
    if prob == 2:
        for i in range(X.shape[1]):
            tmp += torch.cos(np.pi*X[:,i])
        return tmp.reshape(-1,1)
def FF(X,prob):
    tmp = 0
    if prob == 1:
        return -2*X.shape[1]*torch.ones(X.shape[0],1)
    if prob == 2:
        for i in range(X.shape[1]):
            tmp += np.pi*np.pi*torch.cos(np.pi*X[:,i])
            #tmp += np.pi**torch.sin(np.pi*X[:,i])
        return -tmp.reshape(-1,1)
class INSET():
    def __init__(self,size_tr,bound,dtype,prob,device):
        self.dim = bound.shape[0]
        self.size_tr = size_tr
        self.bound = bound
        self.dtype = dtype
        self.prob = prob
        self.device = device
        self.quasi_samples()
        
    def quasi_samples(self):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=self.size_tr)
        
        for i in range(dim):
            sample[:,i] = sample[:,i]*(self.bound[i,1] - self.bound[i,0]) + self.bound[i,0]
        
        self.X = torch.tensor(sample,dtype=self.dtype).to(self.device)
        self.X.requires_grad = True
        self.u_acc = UU(self.X,self.prob)
        self.ff = FF(self.X,self.prob)
        self.u_acc = self.u_acc.to(self.device).data
        self.ff = self.ff.to(self.device).data
        self.weight_grad = torch.ones_like(self.X[:,0:1]).to(self.device)
    
class BDSET():
    def __init__(self,bound,dtype,prob,device):
        self.dim = bound.shape[0]
        
        self.bound = bound
        self.dtype = dtype
        self.prob = prob
        self.device = device
        self.rock = 60
        self.ro = 20
        self.size = self.rock*self.dim
        self.quasi_samples()
    def quasi_samples(self):
        sampler = qmc.Sobol(d=self.dim)
        tmp = sampler.random(n=self.size)
        
        for i in range(self.dim):
            tmp[:,i] = tmp[:,i]*(self.bound[i,1] - self.bound[i,0]) + self.bound[i,0]
        for i in range(self.dim):
            tmp[i*self.rock:i*self.rock + self.ro,i] = self.bound[i,0]
            tmp[i*self.rock + self.ro:(i + 1)*self.rock,i] = self.bound[i,1]
        self.X = torch.tensor(tmp).type(self.dtype).to(self.device)
        self.u_acc = UU(self.X,self.prob)
        self.u_acc = self.u_acc.to(self.device)
np.random.seed(1234)
torch.manual_seed(1234)
class Net(torch.nn.Module):
    def __init__(self, layers, dtype):
        super(Net, self).__init__()
        self.dtype = dtype
        self.layers = layers
        self.layers_hid_num = len(self.layers)-2
        fc = []
        for i in range(self.layers_hid_num+1):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
        self.fc = torch.nn.Sequential(*fc)
        for i in range(self.layers_hid_num+1):
            self.fc[i].weight.data = self.fc[i].weight.data.type(dtype)
            self.fc[i].bias.data = self.fc[i].bias.data.type(dtype)
    def forward(self, x):
        dev = x.device
        
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[i](x))
            temp = torch.eye(x.shape[-1],self.layers[i+1],dtype = self.dtype,device = dev)
            x = h + x@temp
        return self.fc[-1](x) 
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])

class LEN():
    def __init__(self,bound):
        self.bound = bound
        self.dim = bound.shape[0]
        self.hx = bound[:,1] - bound[:,0]
        self.mu = self.dim
    def forward(self,X):
        L = 1.0
        for i in range(self.dim):
            tmp1 = (X[:,i] - self.bound[i,0])/self.hx[i]
            tmp2 = (self.bound[i,1] - X[:,i])/self.hx[i]
            L = L*(1 - (1 - tmp1)**self.mu)*(1 - (1 - tmp2)**self.mu)
        return L.reshape(-1,1)
def L1_error(u_pred,u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    return max(abs(u_pred - u_acc))
def pred_u(netg,netf,lenth,X):
    return netf.forward(X)*lenth.forward(X) + netg.forward(X)
def Loss_bd(netg,bdset):
    bdset.res_u = (netg.forward(bdset.X) - bdset.u_acc)**2
    return torch.sqrt(bdset.res_u.mean())
def Loss_in(netf,inset):
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    insetF = netf.forward(inset.X)
    insetFx, = torch.autograd.grad(insetF,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = inset.weight_grad)
    inset.u = insetF*inset.L + inset.G
    inset.ux = insetFx*inset.L + insetF*inset.Lx + inset.Gx                            
   
    inset.res_u = 0.5*((inset.ux**2).sum(1).reshape(-1,1) - 2*inset.ff*inset.u).mean() 
                      
    return inset.res_u
def Traing(netg,bdset,optimtype,optimg,epochg):
    print('train neural network g')
    t0 = time.time()
    lossoptimal = Loss_bd(netg,bdset)
    for it in range(epochg):
        if it%200 == 0:
            bdset.quasi_samples()
        st = time.time()
        if optimtype == 'LBFGS' or optimtype == 'BFGS':
            def closure():
                optimg.zero_grad()
                loss = Loss_bd(netg,bdset)
                loss.backward()
                return loss
            optimg.step(closure) 
            
        else:
            for j in range(100):
                optimg.zero_grad()
                loss = Loss_bd(netg,bdset)
                loss.backward()
                optimg.step()
        loss = Loss_bd(netg,bdset)
        if loss < lossoptimal:
            fnameg = 'netg.pt'
            torch.save(netg,fnameg) 
        ela = time.time() - st
        
        print('epoch:%d,lossg:%.3e,time:%.2f'%(it,loss.item(),ela))
def Trainf(netg,netf,inset,optimtype,optimf,epochf):
    print('train neural network f')
    t0 = time.time()
    lossoptimal = Loss_in(netf,inset)
    for it in range(epochf):
        inset.quasi_samples()
        inset.L = lenth.forward(inset.X)
        inset.Lx, = torch.autograd.grad(inset.L,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = inset.weight_grad)
        inset.L = inset.L.data; inset.Lx = inset.Lx.data
        
        
        inset.G = netg.forward(inset.X)
        inset.Gx, = torch.autograd.grad(inset.G,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = inset.weight_grad)
        inset.G = inset.G.data; inset.Gx = inset.Gx.data
        st = time.time()
        if optimtype == 'LBFGS' or optimtype == 'BFGS':
            def closure():
                optimf.zero_grad()
                loss = Loss_in(netf,inset)
                loss.backward()
                return loss
            optimf.step(closure) 
            
        else:
            for j in range(100):
                optimf.zero_grad()
                loss = Loss_in(netf,inset)
                loss.backward()
                optimf.step()
        loss = Loss_in(netf,inset)
        if loss < lossoptimal:
            fnamef = 'netf.pt'
            torch.save(netf,fnamef)
        ela = time.time() - st
        err = L1_error(inset.u,inset.u_acc)
        print('epoch:%d,lossf:%.2e,error:%.3e,time:%.2f'%
            (it,loss.item(),err,ela))
def Train(netg,netf,lenth,inset,bdset,optimtype,optimg,optimf,epochg,epochf):
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.L = lenth.forward(inset.X)
    inset.Lx, = torch.autograd.grad(inset.L,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = inset.weight_grad)
    inset.L = inset.L.data; inset.Lx = inset.Lx.data
    Traing(netg,bdset,optimtype,optimg,epochg)
    netg = torch.load('netg.pt')
    inset.G = netg.forward(inset.X)
    inset.Gx, = torch.autograd.grad(inset.G,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = inset.weight_grad)
    inset.G = inset.G.data; inset.Gx = inset.Gx.data             
    Trainf(netg,netf,inset,optimtype,optimf,epochf)  
    u = pred_u(netg,netf,lenth,inset.X)
    inset.Ux, = torch.autograd.grad(u,inset.X,create_graph = True,retain_graph = True,
                                  grad_outputs = inset.weight_grad)
    er = (inset.Ux - inset.ux).reshape(-1,1)
    print(max(abs(er)),max(abs(bdset.u_acc - pred_u(netg,netf,lenth,bdset.X))))
parser = argparse.ArgumentParser(description='PFNN Neural Network Method')
parser.add_argument('--tr', type=int, default=5000,
                    help='train size')  
parser.add_argument('--prob', type=int, default=1,
                    help='problem idx')   
parser.add_argument('--wid', type=int, default=15,
                    help='layers width') 
parser.add_argument('--iter', type=int, default=1,
                    help='max_iter') 
parser.add_argument('--epochg', type=int, default=50,
                    help='netg epoch')  
parser.add_argument('--epochf', type=int, default=20,
                    help='netf epoch')                                                            
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate') 
parser.add_argument('--dim', type=int, default=2,
                    help='dimension')                    
                                     
dtype = torch.float64
args = parser.parse_args()
dim = args.dim
bound = np.zeros([dim,2])
for i in range(dim):
    bound[i,0] = -1
    bound[i,1] = 1  
size_tr = args.tr
prob = args.prob
lay_wid = args.wid
max_iters = args.iter
epochg = args.epochg
epochf = args.epochf
lr = args.lr

bdset = BDSET(bound,dtype,prob,device)
inset = INSET(size_tr,bound,dtype,prob,device)
lenth = LEN(bound)

layerg = [dim,lay_wid,lay_wid,lay_wid,1];netg = Net(layerg,dtype).to(device)
layerf = [dim,lay_wid,lay_wid,lay_wid,1];netf = Net(layerf,dtype).to(device)

#optimtype = 'BFGS'
#optimtype = 'LBFGS'
optimtype = 'adam'
if optimtype == 'BFGS':
    optimg = bfgs.BFGS(netg.parameters(),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
    optimf = bfgs.BFGS(netf.parameters(),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')                  
elif optimtype == 'LBFGS':
    optimg = torch.optim.LBFGS(netg.parameters(),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
    optimf = torch.optim.LBFGS(netf.parameters(),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')                  
else:
    optimg = torch.optim.Adam(netg.parameters(),lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimf = torch.optim.Adam(netf.parameters(),lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 
st = time.time()
for i in range(max_iters):
    Train(netg,netf,lenth,inset,bdset,optimtype,optimg,optimf,epochg,epochf)
ela = time.time() - st
print('finish,time:%.2f'%(ela))
#print(pred_u(netg,netf,lenth,bdset.X) - bdset.u_acc)

print(inset.X.shape,inset.L.shape,inset.Lx.shape,inset.G.shape,inset.Gx.shape,inset.ux.shape)



