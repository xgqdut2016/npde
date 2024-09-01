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

nu = 1e-2
d = 0.1
def region(x):
    x1 = x[:,0:1]
    x2 = x[:,1:2]
    box = (x1>1-d)&(x1<1+d)&(x2>0.5-d)&(x2<0.5+d)
    s = (x1>0)&(x1<2)&(x2>0)&(x2<1)&(np.invert(box))
    return s.flatten()
hr = 1e-3
bounds = np.array([0 - hr,2 + hr,0 - hr,1 + hr]).reshape(2,2)

Lx = 2.0
Ly = 1.0
def y_in(x):
    x2 = x[:,1:2]
    s = 0*x
    s[:,0:1] = 4*x2*(1-x2)/(Ly**2)
    return s

def y_d(x):
    return y_in(x)

class Geo():
    def __init__(self,region,bounds):
        self.region = region
        self.bounds = bounds
        self.dim = self.bounds.shape[0]
        
    def samples(self,N):
        x = np.zeros((N,self.dim))
        m=0
        while (m<N):
            pt = np.random.uniform(0,1,self.dim).reshape(1,-1)
            pt = pt*(self.bounds[:,1]-self.bounds[:,0])+self.bounds[:,0]
            if self.region(pt).all():
                x[m,:] = pt.ravel()
                m += 1
        return x          

    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=2*N)
        sample = sample*(self.bounds[:,1]-self.bounds[:,0])+self.bounds[:,0]
        sample = sample[self.region(sample),:][:N,:]
        return sample   

Omega = Geo(region, bounds)
class INSET():
    def __init__(self,n_tr,mode):
        if mode == 'uniform':
            x = np.linspace(bounds[0,0],bounds[0,1],2*n_tr)
            y = np.linspace(bounds[1,0],bounds[1,1],n_tr)
            xx0, xx1 = np.meshgrid(x,y)

            inp_s = (np.hstack([xx0.reshape(-1,1),xx1.reshape(-1,1)]))
            x = []
            for i in range(inp_s.shape[0]):
                ind = (inp_s[i,0] > 1 - d)&(inp_s[i,0] < 1 + d)&(inp_s[i,1] > 0.5 - d)&(inp_s[i,1] < 0.5 + d)
                if ~ind:
                    x.append(inp_s[i,:])
            x = np.array(x)
            self.X = torch.tensor(x)
        else:
            self.X = torch.from_numpy(Omega.quasi_samples(n_tr*n_tr))
        self.size = self.X.shape[0]
        self.area = 2-d**2
               
        self.dim = 2
        self.yd = y_d(self.X)
class BDSET():#边界点取值
    def __init__(self,nx):
        self.dim = 2
        self.hx = [(bounds[0,1] - bounds[0,0])/nx[0],(bounds[1,1] - bounds[1,0])/nx[1]]
        self.x_in = torch.zeros(nx[1],self.dim)
        self.dir_in = torch.zeros(nx[1],self.dim)
        for i in range(nx[1]):
            self.x_in[i,0] = bounds[0,0]
            self.x_in[i,1] = bounds[1,0] + (i + 0.5)*self.hx[1]
            self.dir_in[i,0] = -1
            self.dir_in[i,1] = 0
        self.rig_in = torch.zeros_like(self.x_in)
        self.rig_in = y_in(self.x_in)

        self.x_out = torch.zeros(nx[1],self.dim)
        self.dir_out = torch.zeros(nx[1],self.dim)
        for i in range(nx[1]):
            self.x_out[i,0] = bounds[0,1]
            self.x_out[i,1] = bounds[1,0] + (i + 0.5)*self.hx[1]
            self.dir_out[i,0] = 1
            self.dir_out[i,1] = 0

        self.x_w = torch.zeros(4*nx[0] + 2*nx[1],self.dim)
        self.dir_w = torch.zeros(4*nx[0] + 2*nx[1],self.dim)
        m = 0
        for i in range(nx[0]):
            self.x_w[m,0] = bounds[0,0] + (i + 0.5)*self.hx[0]
            self.x_w[m,1] = bounds[1,0]
            self.dir_w[m,0] = 0
            self.dir_w[m,1] = -1
            m = m + 1
        for i in range(nx[0]):
            self.x_w[m,0] = bounds[0,0] + (i + 0.5)*self.hx[0]
            self.x_w[m,1] = bounds[1,1]
            self.dir_w[m,0] = 0
            self.dir_w[m,1] = 1
            m = m + 1
        for i in range(nx[0]):
            self.x_w[m,0] = 1 - d + (i + 0.5)*(2*d/nx[0])
            self.x_w[m,1] = 0.5 - d
            self.dir_w[m,0] = 0
            self.dir_w[m,1] = -1
            m = m + 1
        for i in range(nx[0]):
            self.x_w[m,0] = 1 - d + (i + 0.5)*(2*d/nx[0])
            self.x_w[m,1] = 0.5 + d
            self.dir_w[m,0] = 0
            self.dir_w[m,1] = 1
            m = m + 1
        for i in range(nx[1]):
            self.x_w[m,0] = 1 - d
            self.x_w[m,1] = 0.5 - d + (i + 0.5)*(2*d/nx[1])
            self.dir_w[m,0] = -1
            self.dir_w[m,1] = 0
            m = m + 1
        for i in range(nx[1]):
            self.x_w[m,0] = 1 + d
            self.x_w[m,1] = 0.5 - d + (i + 0.5)*(2*d/nx[1])
            self.dir_w[m,0] = 1
            self.dir_w[m,1] = 0
            m = m + 1
        self.x_oc = torch.cat([self.x_in,self.x_w],0)
        self.dir_oc = torch.cat([self.dir_in,self.dir_w],0)
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
        for i in range(self.layers_hid_num - 1):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[self.layers_hid_num],self.layers[self.layers_hid_num]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)
        for i in range(self.layers_hid_num - 1):
            self.fc[2*i].weight.data = self.fc[2*i].weight.data.type(dtype)
            self.fc[2*i].bias.data = self.fc[2*i].bias.data.type(dtype)
            self.fc[2*i + 1].weight.data = self.fc[2*i + 1].weight.data.type(dtype)
            self.fc[2*i + 1].bias.data = self.fc[2*i + 1].bias.data.type(dtype)
        self.fc[2*(self.layers_hid_num - 1)].weight.data = self.fc[2*(self.layers_hid_num - 1)].weight.data.type(dtype)
        self.fc[2*(self.layers_hid_num - 1)].bias.data = self.fc[2*(self.layers_hid_num - 1)].bias.data.type(dtype)
        self.fc[-1].weight.data = self.fc[-1].weight.data.type(dtype)
        self.fc[-1].bias.data = self.fc[-1].bias.data.type(dtype)
    
    def forward(self, x):
        dev = x.device
        h0 = torch.sin(self.fc[2*0](x))
        temp = torch.eye(x.shape[-1],self.layers[1],dtype = self.dtype,device = dev)
        h0 = torch.sin(self.fc[2*0 + 1](h0)) + x@temp
        tmp = h0.to(x.device)
        for i in range(1,self.layers_hid_num - 1):
            h = torch.sin(self.fc[2*i](h0))
            h = torch.sin(self.fc[2*i+1](h))
            temp = torch.eye(h0.shape[-1],self.layers[i+1],dtype = self.dtype,device = dev)
            h0 = h + h0@temp
        h0 = h0.mean(0,keepdim = True)
        #h0 = h0.max(0,keepdim = True)[0]
        h0 = (torch.ones(x.shape[0],1,dtype = self.dtype,device = dev)@h0)
        h0 = torch.cat([tmp,h0],1)
        temp = torch.eye(x.shape[-1],self.layers[-2],dtype = self.dtype,device = dev)
        h = torch.sin(self.fc[2*(self.layers_hid_num - 1)](h0)) + x@temp
        
        return self.fc[-1](h) 
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])  
def length(X):
    return (X[:,1:2] - bounds[1,0])*(bounds[1,1] - X[:,1:2])
def pred_u(netu,X):
    return netu.forward(X)*(X[:,0:1] - bounds[0,0]) + 4*X[:,1:2]*(1 - X[:,1:2])
def pred_v(netv,X):
    return netv.forward(X)*(X[:,0:1] - bounds[0,0])*length(X)
def pred_p(netp,X):
    return netp.forward(X)*(X[:,0:1] - bounds[0,1])
def loadtype(inset,bdset,dtype):
    inset.X = inset.X.type(dtype)
    inset.yd = inset.yd.type(dtype)
    
    #-----------------------------
    bdset.x_in = bdset.x_in.type(dtype)
    bdset.rig_in = bdset.rig_in.type(dtype)
    bdset.x_out = bdset.x_out.type(dtype)
    bdset.x_w = bdset.x_w.type(dtype)
    bdset.x_oc = bdset.x_oc.type(dtype)
    bdset.dir_in = bdset.dir_in.type(dtype)
    bdset.dir_out = bdset.dir_out.type(dtype)
    bdset.dir_w = bdset.dir_w.type(dtype)
    bdset.dir_oc = bdset.dir_oc.type(dtype)
def loadcuda(netu,netv,netp,inset,bdset):    
    netu = netu.to(device)
    netv = netv.to(device)
    netp = netp.to(device)
    
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to(device)
    inset.yd = inset.yd.to(device)
    bdset.rig_in = bdset.rig_in.to(device)
    bdset.x_in = bdset.x_in.to(device)
    if bdset.x_out.requires_grad == False:
        bdset.x_out.requires_grad = True
    bdset.x_out = bdset.x_out.to(device)
    bdset.x_w = bdset.x_w.to(device)
    if bdset.x_oc.requires_grad == False:
        bdset.x_oc.requires_grad = True
    bdset.x_oc = bdset.x_oc.to(device)
    bdset.dir_in = bdset.dir_in.to(device)
    bdset.dir_out = bdset.dir_out.to(device)
    bdset.dir_w = bdset.dir_w.to(device)
    bdset.dir_oc = bdset.dir_oc.to(device)
def loadcpu(netu,netv,netp,inset,bdset):    
    netu = netu.to('cpu')
    netv = netv.to('cpu')
    netp = netp.to('cpu')
    
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to('cpu')
    inset.yd = inset.yd.to('cpu')
    bdset.rig_in = bdset.rig_in.to('cpu')
    bdset.x_in = bdset.x_in.to('cpu')
    bdset.x_out = bdset.x_out.to('cpu')
    bdset.x_w = bdset.x_w.to('cpu')
    bdset.x_oc = bdset.x_oc.to('cpu')
    bdset.dir_in = bdset.dir_in.to('cpu')
    bdset.dir_out = bdset.dir_out.to('cpu')
    bdset.dir_w = bdset.dir_w.to('cpu')
    bdset.dir_oc = bdset.dir_oc.to('cpu')
def Loss_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd):
    inset.u = pred_u(netu,inset.X)
    inset.v = pred_v(netv,inset.X)
    inset.p = pred_p(netp,inset.X)
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
    p_x, = torch.autograd.grad(inset.p, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=torch.ones(inset.size,1).to(device))
    inset.res_u = (-nu*u_lap + inset.u*u_x[:,0:1] + inset.v*u_x[:,1:2] + p_x[:,0:1])**2
    inset.res_v = (-nu*v_lap + inset.u*v_x[:,0:1] + inset.v*v_x[:,1:2] + p_x[:,1:2])**2
    inset.res_div = (u_x[:,0:1] + v_x[:,1:2])**2

    inset.loss_u = torch.sqrt(inset.res_u.mean())
    inset.loss_v = torch.sqrt(inset.res_v.mean())
    inset.loss_div = torch.sqrt(inset.res_div.mean())
    inset.loss_in = penalty_in[0]*inset.loss_u + penalty_in[1]*inset.loss_v + \
        penalty_in[2]*inset.loss_div


    u_out = pred_u(netu,bdset.x_out)
    u_out_x, = torch.autograd.grad(u_out, bdset.x_out, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones(bdset.x_out.shape[0],1).to(device))
    u_out_n = ((u_out_x*bdset.dir_out).sum(1)).reshape(-1,1)
    bdset.loss1 = torch.sqrt(u_out_n**2).mean()

    v_out = pred_v(netv,bdset.x_out)
    v_out_x, = torch.autograd.grad(v_out, bdset.x_out, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones(bdset.x_out.shape[0],1).to(device))
    v_out_n = ((v_out_x*bdset.dir_out).sum(1)).reshape(-1,1)
    bdset.loss2 = torch.sqrt(v_out_n**2).mean()

    u_w = pred_u(netu,bdset.x_w)
    bdset.loss3 = torch.sqrt(u_w**2).mean()
    v_w = pred_v(netv,bdset.x_w)
    bdset.loss4 = torch.sqrt(v_w**2).mean()

    p_oc = pred_p(netp,bdset.x_oc)
    p_oc_x, = torch.autograd.grad(p_oc, bdset.x_oc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones(bdset.x_oc.shape[0],1).to(device))
    p_oc_n = ((p_oc_x*bdset.dir_oc).sum(1)).reshape(-1,1)
    bdset.loss5 = torch.sqrt(p_oc_n**2).mean()

    inset.loss_bd = penalty_bd[0]*bdset.loss1 + penalty_bd[1]*bdset.loss2 + penalty_bd[2]*bdset.loss3 +\
        penalty_bd[3]*bdset.loss4 + penalty_bd[4]*bdset.loss5
    return inset.loss_in + inset.loss_bd
def train_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd,optim,epoch,error_history,optimtype):
    print('Train y&p Neural Network')
    lossin_history,div_history,lossbd_history,lo_histroy = error_history
    loss = Loss_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd)
    print('epoch_yp: %d, Lu:%.3e,Lv:%.3e,div:%.3e, time: %.2f\n'
          %(0,inset.loss_u.item(),inset.loss_v.item(),inset.loss_div.item(), 0.00))
    print('bd1:%.3e,bd2:%.3e,bd3:%.3e,bd4:%.4e,bd5:%.3e\n'%
    (bdset.loss1.item(),bdset.loss2.item(),bdset.loss3.item(),bdset.loss4.item(),bdset.loss5.item()))
    for it in range(epoch):
        st = time.time()
        if optimtype == 'Adam':
            for j in range(100):
                optim.zero_grad()
                loss = Loss_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd)
                loss.backward()
                optim.step()
        if optimtype == 'BFGS' or optimtype == 'LBFGS':
            def closure():
                loss = Loss_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd)
                optim.zero_grad()
                loss.backward()
                return loss
            optim.step(closure) 
        loss = Loss_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd)
        ela = time.time() - st
        print('------------------------------')
        print('epoch_yp: %d, Loss:%.3e,Lu:%.3e,Lv:%.3e,div:%.3e, time: %.2f\n'
          %(it + 1,loss.item(),inset.loss_u.item(),inset.loss_v.item(),inset.loss_div.item(), ela))
        print('bd1:%.3e,bd2:%.3e,bd3:%.3e,bd4:%.4e,bd5:%.3e\n'%
        (bdset.loss1.item(),bdset.loss2.item(),bdset.loss3.item(),bdset.loss4.item(),bdset.loss5.item()))
        
        loss_u = inset.loss_u.item()
        loss_v = inset.loss_v.item()
        loss_div = inset.loss_div.item()
        bd1 = bdset.loss1.item()
        bd2 = bdset.loss2.item()
        bd3 = bdset.loss3.item()
        bd4 = bdset.loss4.item()
        bd5 = bdset.loss5.item()
        
        lossin_history.append([loss.item(),loss_u,loss_v,loss_div])
        div_history.append(loss_div)
        lossbd_history.append([bd1,bd2,bd3,bd4,bd5])
        lo_histroy.append(loss.item())
        error_history = [lossin_history,div_history,lossbd_history,lo_histroy]
    return error_history

n_tr = 64
nx_bd = [100,100]
mode = 'uniform'
#mode = 'qmc'
inset = INSET(n_tr,mode)

bdset = BDSET(nx_bd)


dtype = torch.float64
wid = 25
wid = 25
layer_u = [2,15,10,wid,1];netu = Net(layer_u,dtype)
layer_v = [2,15,10,wid,1];netv = Net(layer_v,dtype)
layer_p = [2,15,10,wid,1];netp = Net(layer_p,dtype)


fname1 = "upoint-NSlay%dvar.pt"%(wid)
fname2 = "vpoint-NSlay%dvar.pt"%(wid)
fname3 = "ppoint-NSlay%dvar.pt"%(wid)


#netu = torch.load(fname1)
#netv = torch.load(fname2)
#netp = torch.load(fname3)


loadtype(inset,bdset,dtype)
loadcuda(netu,netv,netp,inset,bdset)

loss_history = [[],[],[],[]]
epoch = 30
start_time = time.time()
penalty_in = [1e0,1e0,1e0]
penalty_bd = [1e0,1e0,1e0,1e0,1e0]
lr = 1e0
max_iter = 1
#optimtype = 'Adam'
optimtype = 'BFGS'
#optimtype = 'LBFGS'
number = 0
for i in range(max_iter):
    if optimtype == 'BFGS':
        optim = bfgs.BFGS(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),
                        lr=lr, max_iter=100,
                        tolerance_grad=1e-16, tolerance_change=1e-16,
                        line_search_fn='strong_wolfe')
    if optimtype == 'LBFGS':
        optim = torch.optim.LBFGS(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),
                        lr=lr, max_iter=100,
                        history_size=100,
                        line_search_fn='strong_wolfe')
    if optimtype == 'Adam':
        optim = torch.optim.Adam(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),
                        lr=lr)
    loss_history = \
        train_yp(netu,netv,netp,inset,bdset,penalty_in,penalty_bd,optim,epoch,loss_history,optimtype)
    if (i + 1)%5 == 0:
        number += 1
        
        lr *= 0.985
elapsed = time.time() - start_time
print('Finishied! train time: %.2f\n' %(elapsed)) 
loadcpu(netu,netv,netp,inset,bdset)
torch.cuda.empty_cache()

torch.save(netu, fname1)
torch.save(netv, fname2)
torch.save(netp, fname3)


lossin_history,div_history,lossbd_history,lo_history = loss_history
np.save('lay%d-epoch%d-lossin.npy'%(wid,epoch),lossin_history)
np.save('lay%d-epoch%d-lossbd.npy'%(wid,epoch),lossbd_history)
np.save('lay%d-epoch%d-lossdiv.npy'%(wid,epoch),div_history)
np.save('lay%d-epoch%d-losslo.npy'%(wid,epoch),lo_history)
#%%
fig, ax = plt.subplots(1,2,figsize=(12,3.5))

ax[0].semilogy(np.array(lossin_history))
ax[0].legend(['loss','loss_u', 'loss_v', 'loss_div'])
ax[0].set_xlabel('iters') 

ax[1].plot(np.array(lossbd_history))
ax[1].legend(['bd1','bd2','bd3','bd4','bd5'])
ax[1].set_xlabel('iters') 
plt.yscale('log')
fig.tight_layout()
plt.show()


plt.scatter(inset.X.detach().numpy()[:,0],inset.X.detach().numpy()[:,1])
plt.show()
#%%
nx_te = [40,40]
te_bd_set = BDSET(nx_te)
hx = [(bounds[0,1] - bounds[0,0])/nx_te[0],(bounds[1,1] - bounds[1,0])/nx_te[1]]
fig, ax = plt.subplots(1,3, figsize=(13,4))
x1s = np.linspace(bounds[1,0] + 0.5*hx[1],bounds[1,1] - hx[1],nx_te[1])
te_bd_set.x_in = te_bd_set.x_in.type(dtype)
te_bd_set.x_out = te_bd_set.x_out.type(dtype)
u_in = pred_u(netu,te_bd_set.x_in).detach().numpy()
v_in = pred_v(netv,te_bd_set.x_in).detach().numpy()

u_out = pred_u(netu,te_bd_set.x_out).detach().numpy()
v_out = pred_v(netv,te_bd_set.x_out).detach().numpy()

u_o_p = x1s*(1-x1s)*4/(Ly**2)

ax[1].plot(x1s,u_out); ax[1].plot(x1s,u_in)
ax[1].legend(['outlet','inlet'])
ax[1].set_xlabel('y'); ax[1].set_ylabel('u')
ax[1].set_title('PointNet: u')

ax[2].plot(x1s,v_out); ax[2].plot(x1s,v_in)
ax[2].legend(['outlet','inlet'])
ax[2].set_xlabel('y'); ax[2].set_ylabel('v')
ax[2].set_title('PointNet: v')


ax[0].semilogy(np.array(lo_history))
ax[0].legend(['pde_loss'])
ax[0].set_xlabel('iters') 
plt.show()
#%%
n=n_tr
xx0 = np.linspace(bounds[0,0],bounds[0,1],2*n)
xx1 = np.linspace(bounds[1,0],bounds[1,1],n)
xx0, xx1 = np.meshgrid(xx0,xx1)
ind = (xx0>1-d)&(xx0<1+d)&(xx1>0.5-d)&(xx1<0.5+d)

inp_s = torch.from_numpy(np.hstack([xx0.reshape(-1,1),xx1.reshape(-1,1)]))
u_pred = torch.zeros(inp_s.shape[0],1)
v_pred = torch.zeros(inp_s.shape[0],1)
p_pred = torch.zeros(inp_s.shape[0],1)
'''
for i in range(inp_s.shape[0]):
    u_pred[i,0] = pred_u(netu,inp_s[i,:])
    v_pred[i,0] = pred_v(netv,inp_s[i,:])
    p_pred[i,0] = pred_p(netp,inp_s[i,:])
'''
u_pred = pred_u(netu,inp_s)
v_pred = pred_v(netv,inp_s)
p_pred = pred_p(netp,inp_s)

u = u_pred.detach().numpy()
v = v_pred.detach().numpy()
p = p_pred.detach().numpy()


y1_s = u.reshape(n,2*n); y1 = u.reshape(n,2*n);
y2_s = v.reshape(n,2*n); y2 = v.reshape(n,2*n);
p = p.reshape(n,2*n)

y1_s[ind] = np.nan  # for streamplot mask!
y2_s[ind] = np.nan
n_ind = np.invert(ind)
y1 = y1[n_ind].flatten()
y2 = y2[n_ind].flatten()
p = p[n_ind].flatten()

speed_s = np.sqrt(y1_s**2+y2_s**2)
x0 = xx0[n_ind].flatten()
x1 = xx1[n_ind].flatten()

num_line = 100

fig, ax = plt.subplots(2,2, figsize=(12,6))

for i in range(2): 
    for j in range(2):
            ax[i,j].axis('equal')
            ax[i,j].set_xlim([0.,2.0])
            ax[i,j].set_ylim([0.,1.])
            ax[i,j].axis('off')
            ax[i,j].add_artist(plt.Rectangle((1-d,0.5-d),2*d,2*d, fill=True, color='white'))


ax00 = ax[0,0].tricontourf(x0, x1, y1, num_line, cmap='rainbow')
fig.colorbar(ax00,ax=ax[0,0],fraction = 0.024,pad = 0.04)
#ax[0,0].contour(ax00, linewidths=0.6, colors='black')
                   #, cmap='jet', color=speed_s, linewidth=0.5)
ax01 = ax[0,1].tricontourf(x0, x1, y2, num_line, cmap='rainbow')
fig.colorbar(ax01,ax=ax[0,1],fraction = 0.024,pad = 0.04)
ax[0,0].set_title('PointNet: u')
ax[0,1].set_title('PointNet: v')


ax10 = ax[1,0].tricontourf(x0, x1, (y1**2+y2**2)**0.5, num_line, cmap='rainbow')
ax[1,0].streamplot(xx0, xx1, y1_s, y2_s, density = 1.5)
fig.colorbar(ax10,ax=ax[1,0],fraction = 0.024,pad = 0.04)
ax[1,0].set_title('PointNet: stream')


ax11 = ax[1,1].tricontourf(x0, x1, p, num_line, cmap='rainbow')
#ax[1,0].contour(ax10, linewidths=0.6, colors='black')
#ax11 = ax[1,1].tricontourf(x0, x1, p, num_line, cmap='rainbow')
ax[1,1].set_title('PointNet: p')
#ax[1,1].set_title('PINN: p')
fig.colorbar(ax11,ax=ax[1,1],fraction = 0.024,pad = 0.04)
#fig.colorbar(ax11,ax=ax[1,1])
plt.show()

