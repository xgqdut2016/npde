import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
import itertools
import bfgs
from scipy.stats import qmc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

bounds = np.array([0,1,0,1]).reshape(2,2)
Re = 1
Pr = 0.72
Gr = 1e4

class INSET():#边界点取值
    def __init__(self,nx,mode):
        self.dim = 2
        #self.area = (bound[0,1] - bound[0,0])*(bound[1,1] - bound[1,0])
        self.hx = [(bounds[0,1] - bounds[0,0])/nx[0],(bounds[1,1] - bounds[1,0])/nx[1]]
        self.size = nx[0]*nx[1]
        self.X = torch.zeros(self.size,self.dim)#储存内点
        if mode == 'uniform':
            for i in range(nx[0]):
                for j in range(nx[1]):
                    self.X[i*nx[1] + j,0] = bounds[0,0] + (i + 0.5)*self.hx[0]
                    self.X[i*nx[1] + j,1] = bounds[1,0] + (j + 0.5)*self.hx[1]
        elif mode == 'random':
            tmp = torch.rand(self.size,2)
            self.X[:,0] = bounds[0,0] + self.hx[0] + (bounds[0,1] - bounds[0,0] - 2*self.hx[0])*tmp[:,0]
            self.X[:,1] = bounds[1,0] + self.hx[1] + (bounds[1,1] - bounds[1,0] - 2*self.hx[1])*tmp[:,1]
        else:
            tmp = torch.tensor(self.quasi_samples(self.size))
            self.X[:,0] = bounds[0,0] + self.hx[0] + (bounds[0,1] - bounds[0,0] - 2*self.hx[0])*tmp[:,0]
            self.X[:,1] = bounds[1,0] + self.hx[1] + (bounds[1,1] - bounds[1,0] - 2*self.hx[1])*tmp[:,1]
        
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
class BDSET():#边界点取值
    def __init__(self,nx):
        self.dim = 2
        self.hx = [(bounds[0,1] - bounds[0,0])/nx[0],(bounds[1,1] - bounds[1,0])/nx[1]]
        self.size = 2*(nx[0] + nx[1])
        #----------------------------------
        self.u1 = torch.zeros(nx[0] + 1,1)
        self.v1 = torch.zeros(nx[0] + 1,1)
        self.x1 = torch.zeros(nx[0] + 1,2)
        for i in range(nx[0] + 1):
            self.x1[i,0] = bounds[0,0] + i*self.hx[0]
            self.x1[i,1] = bounds[1,0]
        self.T1 = torch.ones(self.x1.shape[0],1)
        #--------------------------------    
        self.u2 = torch.zeros(2*nx[1],1)
        self.v2 = torch.zeros(2*nx[1],1)
        self.x2 = torch.zeros(2*nx[1],2)
        self.dir2 = torch.zeros(2*nx[1],2)
        for i in range(nx[1]):
            self.x2[i,0] = bounds[0,0]
            self.x2[i,1] = bounds[1,0] + (i + 0.5)*self.hx[1]
            self.dir2[i,0] = -1;self.dir2[i,1] = 0
            self.x2[i + nx[1],0] = bounds[0,1]
            self.x2[i + nx[1],1] = bounds[1,0] + (i + 0.5)*self.hx[1]
            self.dir2[i + nx[1],0] = 1;self.dir2[i + nx[1],1] = 0
        self.g2 = torch.zeros_like(self.u2)
        #--------------------------
        self.u3 = torch.zeros(nx[0],1)
        self.v3 = torch.zeros(nx[0],1)
        self.x3 = torch.zeros(nx[0],2)
        
        for i in range(1,nx[0] - 1):
            self.x3[i,0] = 1/3 + i/(3*nx[0])
            self.x3[i,1] = 1.0
        self.x3[0,0] = 1/3;self.x3[0,1] = 1
        self.x3[-1,0] = 2/3;self.x3[-1,1] = 1
        for i in range(nx[0]):
            self.v3[i,0] = -4*(self.x3[i,0] - 1/3)*(2/3 - self.x3[i,0])
        self.T3 = torch.zeros(self.x3.shape[0],1)
        #------------------------------------
        self.x4 = torch.zeros(nx[0],2)
        self.u4 = torch.zeros(nx[0],1)
        self.v4 = torch.zeros(nx[0],1)
        self.dir4 = torch.zeros(nx[0],2)
        for i in range(1,nx[0]):
            self.x4[i,0] = i/(3*nx[0])
            self.x4[i,1] = 1.0
        self.x4[0,0] = 0;self.x4[0,1] = 1.0
        for i in range(nx[0]):
            self.v4[i,0] = 2*self.x4[i,0]*(1/3 - self.x4[i,0])
        self.dir4[:,0] = torch.zeros(nx[0]);self.dir4[:,1] = torch.ones(nx[0])
        #------------------------------------
        self.x5 = torch.zeros(nx[0],2)
        self.u5 = torch.zeros(nx[0],1)
        self.v5 = torch.zeros(nx[0],1)
        self.dir5 = torch.zeros(nx[0],2)
        for i in range(nx[0] - 1):
            self.x5[i,0] = 2/3 + (i + 0.5)/(3*nx[0])
            self.x5[i,1] = 1.0
        self.x5[-1,0] = 1;self.x5[-1,1] = 1.0
        for i in range(nx[0]):
            self.v5[i,0] = 2*(self.x5[i,0] - 2/3)*(1 - self.x5[i,0])

class LENU():
    def __init__(self):
        pass
    def forward(self,X):
        x = X[:,0]
        y = X[:,1]
        L = x*(1 -x)*y*(1 - y)
        return L.reshape(-1,1)
class LENV():
    def __init__(self):
        pass
    def forward(self,X):
        x = X[:,0]
        y = X[:,1]
        L = x*(1 -x)*y
        return L.reshape(-1,1)
        
np.random.seed(1234)
torch.manual_seed(1234)
class Net(torch.nn.Module):
    def __init__(self, layers, dtype):
        super(Net, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        self.device = device
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

def pred_u(netu,lenthu,X):
    return netu.forward(X)*lenthu.forward(X)
def pred_v(netv,lenthv,X):
    return netv.forward(X)*lenthv.forward(X)
def pred_w(netw,X):
    return netw.forward(X)
def pred_T(netT,X):
    return netT.forward(X)*X[:,1:2] + 1
def loadtype(inset,bdset,dtype):
    inset.X = inset.X.type(dtype)
    
    #-----------------------------
    bdset.x1 = bdset.x1.type(dtype)
    bdset.u1 = bdset.u1.type(dtype)
    bdset.v1 = bdset.v1.type(dtype)
    bdset.T1 = bdset.T1.type(dtype)
    #-----------------------------
    bdset.x2 = bdset.x2.type(dtype)
    bdset.u2 = bdset.u2.type(dtype)
    bdset.v2 = bdset.v2.type(dtype)
    bdset.g2 = bdset.g2.type(dtype)
    bdset.dir2 = bdset.dir2.type(dtype)
    #-----------------------------
    bdset.x3 = bdset.x3.type(dtype)
    bdset.u3 = bdset.u3.type(dtype)
    bdset.v3 = bdset.v3.type(dtype)
    bdset.T3 = bdset.T3.type(dtype)
    #-----------------------------
    bdset.x4 = bdset.x4.type(dtype)
    bdset.u4 = bdset.u4.type(dtype)
    bdset.v4 = bdset.v4.type(dtype)
    bdset.dir4 = bdset.dir4.type(dtype)
    #-----------------------------
    bdset.x5 = bdset.x5.type(dtype)
    bdset.u5 = bdset.u5.type(dtype)
    bdset.v5 = bdset.v5.type(dtype)
    bdset.dir5 = bdset.dir5.type(dtype)
    
def loadcuda(netu,netv,netw,netT,inset,bdset):    
    netu = netu.to(device)
    netv = netv.to(device)
    netw = netw.to(device)
    netT = netT.to(device)
    
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to(device)
    
    #-----------------------------
    bdset.x1 = bdset.x1.to(device)
    bdset.u1 = bdset.u1.to(device)
    bdset.v1 = bdset.v1.to(device)
    bdset.T1 = bdset.T1.to(device)
    #-----------------------------
    if bdset.x2.requires_grad == False:
        bdset.x2.requires_grad = True
    bdset.x2 = bdset.x2.to(device)
    bdset.u2 = bdset.u2.to(device)
    bdset.v2 = bdset.v2.to(device)
    bdset.g2 = bdset.g2.to(device)
    bdset.dir2 = bdset.dir2.to(device)
    #-----------------------------
    bdset.x3 = bdset.x3.to(device)
    bdset.u3 = bdset.u3.to(device)
    bdset.v3 = bdset.v3.to(device)
    bdset.T3 = bdset.T3.to(device)
    #-----------------------------
    if bdset.x4.requires_grad == False:
        bdset.x4.requires_grad = True
    bdset.x4 = bdset.x4.to(device)
    bdset.u4 = bdset.u4.to(device)
    bdset.v4 = bdset.v4.to(device)
    bdset.dir4 = bdset.dir4.to(device)
    #-----------------------------
    if bdset.x5.requires_grad == False:
        bdset.x5.requires_grad = True
    bdset.x5 = bdset.x5.to(device)
    bdset.u5 = bdset.u5.to(device)
    bdset.v5 = bdset.v5.to(device)
    bdset.dir5 = bdset.dir5.to(device)

def loadcpu(netu,netv,netw,netT,inset,bdset):    
    netu = netu.to('cpu')
    netv = netv.to('cpu')
    netw = netw.to('cpu')
    netT = netT.to('cpu')
    
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to('cpu')
    
    #-----------------------------
    bdset.x1 = bdset.x1.to('cpu')
    bdset.u1 = bdset.u1.to('cpu')
    bdset.v1 = bdset.v1.to('cpu')
    bdset.T1 = bdset.T1.to('cpu')
    #-----------------------------
    if bdset.x2.requires_grad == False:
        bdset.x2.requires_grad = True
    bdset.x2 = bdset.x2.to('cpu')
    bdset.u2 = bdset.u2.to('cpu')
    bdset.v2 = bdset.v2.to('cpu')
    bdset.g2 = bdset.g2.to('cpu')
    bdset.dir2 = bdset.dir2.to('cpu')
    #-----------------------------
    bdset.x3 = bdset.x3.to('cpu')
    bdset.u3 = bdset.u3.to('cpu')
    bdset.v3 = bdset.v3.to('cpu')
    bdset.T3 = bdset.T3.to('cpu')
    #-----------------------------
    if bdset.x4.requires_grad == False:
        bdset.x4.requires_grad = True
    bdset.x4 = bdset.x4.to('cpu')
    bdset.u4 = bdset.u4.to('cpu')
    bdset.v4 = bdset.v4.to('cpu')
    bdset.dir4 = bdset.dir4.to('cpu')
    #-----------------------------
    if bdset.x5.requires_grad == False:
        bdset.x5.requires_grad = True
    bdset.x5 = bdset.x5.to('cpu')
    bdset.u5 = bdset.u5.to('cpu')
    bdset.v5 = bdset.v5.to('cpu')
    bdset.dir5 = bdset.dir5.to('cpu')
def Loss_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd):

    inset.u = pred_u(netu,lenthu,inset.X)
    inset.v = pred_v(netv,lenthv,inset.X)
    inset.w = pred_w(netw,inset.X)
    inset.T = pred_T(netT,inset.X)
    u_x, = torch.autograd.grad(inset.u, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    u_xx, = torch.autograd.grad(u_x[:,0:1], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    u_yy, = torch.autograd.grad(u_x[:,1:2], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))  
    u_lap = u_xx[:,0:1] + u_yy[:,1:2]

    v_x, = torch.autograd.grad(inset.v, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    v_xx, = torch.autograd.grad(v_x[:,0:1], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    v_yy, = torch.autograd.grad(v_x[:,1:2], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))  
    v_lap = v_xx[:,0:1] + v_yy[:,1:2]

    w_x, = torch.autograd.grad(inset.w, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    w_xx, = torch.autograd.grad(w_x[:,0:1], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    w_yy, = torch.autograd.grad(w_x[:,1:2], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))  
    w_lap = w_xx[:,0:1] + w_yy[:,1:2]

    T_x, = torch.autograd.grad(inset.T, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    T_xx, = torch.autograd.grad(T_x[:,0:1], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    T_yy, = torch.autograd.grad(T_x[:,1:2], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))  
    T_lap = T_xx[:,0:1] + T_yy[:,1:2]

    inset.res_u = (u_lap + w_x[:,1:2])**2
    inset.res_v = (-v_lap + w_x[:,0:1])**2
    inset.res_w = (-w_lap/Re + inset.u*w_x[:,0:1] + inset.v*w_x[:,1:2] - T_x[:,0:1]*Gr/(Re**2))**2
    inset.res_T = (-T_lap/(Re*Pr) + inset.u*T_x[:,0:1] + inset.v*T_x[:,1:2])**2
    inset.var_w = (inset.w + u_x[:,1:2] - v_x[:,0:1])**2
    inset.loss_u = torch.sqrt(inset.res_u.mean())
    inset.loss_v = torch.sqrt(inset.res_v.mean())
    inset.loss_w = torch.sqrt(inset.res_w.mean())
    inset.loss_T = torch.sqrt(inset.res_T.mean())
    inset.loss_var = torch.sqrt(inset.var_w.mean())
    inset.loss_in = penalty_in[0]*inset.loss_u + penalty_in[1]*inset.loss_v + \
        penalty_in[2]*inset.loss_w + penalty_in[3]*inset.loss_T + penalty_in[4]*inset.loss_var
    #--------------------------------------
    
    #2
    bdset.T2 = pred_T(netT,bdset.x2)
    Tbd_x2, = torch.autograd.grad(bdset.T2, bdset.x2, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(bdset.x2.shape[0],1).to(device))
    Tbd_n2 = ((Tbd_x2*bdset.dir2).sum(1)).reshape(-1,1)
    
    bdset.loss2 = torch.sqrt((Tbd_n2 + bdset.T2)**2).mean()
    #3
    tmpbd3 = (pred_v(netv,lenthv,bdset.x3) - bdset.v3)**2 + (pred_T(netT,bdset.x3))**2
    bdset.loss3 = torch.sqrt(tmpbd3.mean())
    #4
    Tbd_x4, = torch.autograd.grad(pred_T(netT,bdset.x4), bdset.x4, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(bdset.x4.shape[0],1).to(device))
    Tbd_n4 = ((Tbd_x4*bdset.dir4).sum(1)).reshape(-1,1)
    tmpbd4 = (pred_v(netv,lenthv,bdset.x4) - bdset.v4)**2 + (Tbd_n4)**2
    bdset.loss4 = torch.sqrt(tmpbd4.mean())
    #5
    Tbd_x5, = torch.autograd.grad(pred_T(netT,bdset.x5), bdset.x5, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(bdset.x5.shape[0],1).to(device))
    Tbd_n5 = ((Tbd_x5*bdset.dir5).sum(1)).reshape(-1,1)
    tmpbd5 = (pred_v(netv,lenthv,bdset.x5) - bdset.v5)**2 + (Tbd_n5)**2
    bdset.loss5 = torch.sqrt(tmpbd5.mean())
    inset.loss_bd = penalty_bd[0]*bdset.loss2 + penalty_bd[1]*bdset.loss3 +\
        penalty_bd[2]*bdset.loss4 + penalty_bd[3]*bdset.loss5
    return inset.loss_in + inset.loss_bd
    
def train_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd,optim,epoch,error_history,optimtype):
    print('Train y&p Neural Network')
    lossin_history,var_history,lossbd_history = error_history
    loss = Loss_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd)
    print('epoch_yp: %d, Lu:%.3e,Lv:%.3e,Lw:%.3e,LT:%.3e,Lvar:%.3e, time: %.2f\n'
          %(0,inset.loss_u.item(),inset.loss_v.item(),inset.loss_w.item(),inset.loss_T.item(),inset.loss_var.item(), 0.00))
    print('bd2:%.3e,bd3:%.3e,bd4:%.4e,bd5:%.3e\n'%
    (bdset.loss2.item(),bdset.loss3.item(),bdset.loss4.item(),bdset.loss5.item()))
    for it in range(epoch):
        st = time.time()
        if optimtype == 'Adam':
            for j in range(100):
                optim.zero_grad()
                loss = Loss_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd)
                loss.backward()
                optim.step()
        if optimtype == 'BFGS' or optimtype == 'LBFGS':
            def closure():
                optim.zero_grad()
                loss = Loss_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd)
                loss.backward()
                return loss
            optim.step(closure) 
        loss = Loss_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd)
        ela = time.time() - st
        print('------------------------------')
        print('loss:%.3e,bd2:%.3e,bd3:%.3e,bd4:%.4e,bd5:%.3e\n'%
        (loss.item(),bdset.loss2.item(),bdset.loss3.item(),bdset.loss4.item(),bdset.loss5.item()))
        print('epoch_yp: %d, Lu:%.3e,Lv:%.3e,Lw:%.3e,LT:%.3e,Lvar:%.3e, time: %.2f\n'
          %(it + 1,inset.loss_u.item(),inset.loss_v.item(),inset.loss_w.item(),inset.loss_T.item(),inset.loss_var.item(), ela))
        
        loss_u = inset.loss_u.item()
        loss_v = inset.loss_v.item()
        loss_w = inset.loss_w.item()
        loss_T = inset.loss_T.item()
        loss_var = inset.loss_var.item()
        
        bd2 = bdset.loss2.item()
        bd3 = bdset.loss3.item()
        bd4 = bdset.loss4.item()
        bd5 = bdset.loss5.item()
        lossin_history.append([loss.item(),loss_u,loss_v,loss_w,loss_T,loss_var])
        var_history.append(loss_var)
        lossbd_history.append([bd2,bd3,bd4,bd5])
        error_history = [lossin_history,var_history,lossbd_history]
    return error_history

nx = [64,64]
nx_bd = [100,100]

mu = 1
lr = 1e0
#mode = 'random'
mode = 'uniform'
#mode = 'qmc'
inset = INSET(nx,mode)
bdset = BDSET(nx_bd)
lenthu = LENU()
lenthv = LENV()

dtype = torch.float64
wid = 25
layer_u = [2,wid,wid,wid,1];netu = Net(layer_u,dtype)
layer_v = [2,wid,wid,wid,1];netv = Net(layer_v,dtype)
layer_w = [2,wid,wid,wid,1];netw = Net(layer_w,dtype)
layer_T = [2,wid,wid,wid,1];netT = Net(layer_T,dtype)

fname1 = "unobasic-CVDlay%dvar.pt"%(wid)
fname2 = "vnobasic-CVDlay%dvar.pt"%(wid)
fname3 = "wnobasic-CVDlay%dvar.pt"%(wid)
fname4 = "Tnobasic-CVDlay%dvar.pt"%(wid)

#netu = torch.load(fname1)
#netv = torch.load(fname2)
#netw = torch.load(fname3)
#netT = torch.load(fname4)

loadtype(inset,bdset,dtype)
loadcuda(netu,netv,netw,netT,inset,bdset)

loss_history = [[],[],[]]
epoch = 10
start_time = time.time()
penalty_in = [1e0,1e0,1e0,1e0,1e0]
penalty_bd = [1e0,1e0,1e0,1e0,1e0]
lr = 1e0
max_iter = 1
#optimtype = 'Adam'
optimtype = 'BFGS'
#optimtype = 'LBFGS'
for i in range(max_iter):
    if optimtype == 'BFGS':
        optim = bfgs.BFGS(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netw.parameters(),
                                        netT.parameters()),
                        lr=lr, max_iter=100,
                        tolerance_grad=1e-16, tolerance_change=1e-16,
                        line_search_fn='strong_wolfe')
    if optimtype == 'LBFGS':
        optim = torch.optim.LBFGS(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netw.parameters(),
                                        netT.parameters()),
                        lr=lr, max_iter=100,
                        history_size=100,
                        line_search_fn='strong_wolfe')
    if optimtype == 'Adam':
        optim = torch.optim.Adam(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netw.parameters(),
                                        netT.parameters()),
                        lr=lr)
    loss_history = \
        train_yp(netu,netv,netw,netT,lenthu,lenthv,inset,bdset,penalty_in,penalty_bd,optim,epoch,loss_history,optimtype)
    if (i + 1)%2 == 0:
        lr *= 0.985
elapsed = time.time() - start_time
print('Finishied! train time: %.2f\n' %(elapsed)) 

loadcpu(netu,netv,netw,netT,inset,bdset)
torch.cuda.empty_cache()

torch.save(netu, fname1)
torch.save(netv, fname2)
torch.save(netw, fname3)
torch.save(netT, fname4)

lossin_history,var_history,lossbd_history = loss_history
np.save('lay%d-epoch%d-lossin.npy'%(wid,epoch),lossin_history)
np.save('lay%d-epoch%d-lossvd.npy'%(wid,epoch),lossbd_history)
np.save('lay%d-epoch%d-lossvar.npy'%(wid,epoch),var_history)
fig, ax = plt.subplots(1,2,figsize=(12,3.5))

ax[0].semilogy(np.array(lossin_history))
ax[0].legend(['loss','loss_u', 'loss_v', 'loss_w', 'loss_T','loss_var'])
ax[0].set_xlabel('iters') 

ax[1].plot(np.array(lossbd_history))
ax[1].legend(['bd2','bd3','bd4','bd5'])
ax[1].set_xlabel('iters') 
plt.yscale('log')
fig.tight_layout()
plt.show()


plt.scatter(inset.X.detach().numpy()[:,0],inset.X.detach().numpy()[:,1])
plt.show()
#%%

fig, ax = plt.subplots(2,2,figsize=(8,8))
for i in range(2):
    for j in range(2):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,1])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')
num_line = 100
nx = [21,11]
x = np.linspace(bounds[0,0],bounds[0,1],nx[0])
y = np.linspace(bounds[1,0],bounds[1,1],nx[1])
hx = [(bounds[0,1] - bounds[0,0])/(nx[0] - 1),(bounds[1,1] - bounds[1,0])/(nx[1] - 1)]
inp = torch.zeros(nx[0]*nx[1],2).type(dtype)
for i in range(nx[0]):
    for j in range(nx[1]):
        inp[i*nx[1] + j,0] = bounds[0,0] + i*hx[0]
        inp[i*nx[1] + j,1] = bounds[1,0] + j*hx[1]
u = pred_u(netu,lenthu,inp).detach().numpy().reshape(nx[0],nx[1]).T
v = pred_u(netv,lenthv,inp).detach().numpy().reshape(nx[0],nx[1]).T
w = pred_w(netw,inp).detach().numpy().reshape(nx[0],nx[1]).T
T = pred_T(netT,inp).detach().numpy().reshape(nx[0],nx[1]).T
X,Y = np.meshgrid(x,y)

s = ax[0,0].contourf(X,Y, u, num_line, cmap='rainbow')
ax[0,0].contour(s, linewidths=0.6, colors='black')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax[0,0].set_title('NN:u',fontsize=15)    
fig.colorbar(s,ax=ax[0,0],fraction = 0.045)

s = ax[0,1].contourf(X,Y, v, num_line, cmap='rainbow')
ax[0,1].contour(s, linewidths=0.6, colors='black')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax[0,1].set_title('NN:v',fontsize=15)    
fig.colorbar(s,ax=ax[0,1],fraction = 0.045)

s = ax[1,0].contourf(X,Y, w, num_line, cmap='rainbow')
ax[1,0].contour(s, linewidths=0.6, colors='black')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax[1,0].set_title('NN:w',fontsize=15)    
fig.colorbar(s,ax=ax[1,0],fraction = 0.045)

s = ax[1,1].contourf(X,Y, T, num_line, cmap='rainbow')
ax[1,1].contour(s, linewidths=0.6, colors='black')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax[1,1].set_title('NN:T',fontsize=15)    
fig.colorbar(s,ax=ax[1,1],fraction = 0.045)

fig.tight_layout()
plt.show()

#%%
plt.contourf(x,y,T,40,cmap = 'rainbow')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('the temperture T')
plt.show()

