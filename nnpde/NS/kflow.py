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

bound = np.array([-0.5,1,-0.5,1.5]).reshape(2,2)
Re = 40;nu = 1/Re
lam = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2)
def UU(X,order,prob):
    if prob == 1:
        eta = 2*np.pi
        if order == [0,0]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = 1 - torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*lam/(eta)
            return tmp
        if order == [1,0]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = - torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*lam
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*(lam**2)/(eta)
            return tmp
        if order == [0,1]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*(eta)
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*lam
            return tmp
        if order == [2,0]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = - torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*lam*lam
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*(lam**3)/(eta)
            return tmp
        if order == [0,2]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*(eta)**2
            tmp[:,1] = -torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*lam*(eta)
            return tmp
    
def Delta(X,prob):
    return UU(X,[2,0],prob) + UU(X,[0,2],prob)
def PP(X,order,prob):
    if prob == 1:
        if order == [0,0]:
            return 0.5*(1 - torch.exp(2*lam*X[:,0]))
        if order == [1,0]:
            return - lam*torch.exp(2*lam*X[:,0])
        if order == [0,1]:
            return 0*X[:,0]
def FF(X,prob):#mu = 0.5
    tmp = torch.zeros(X.shape[0],2)
    tmp[:,0] = -nu*Delta(X,prob)[:,0] + (UU(X,[0,0],prob)[:,0])*(UU(X,[1,0],prob)[:,0]) + \
    (UU(X,[0,0],prob)[:,1])*(UU(X,[0,1],prob)[:,0]) + PP(X,[1,0],prob)
    tmp[:,1] = -nu*Delta(X,prob)[:,1] + (UU(X,[0,0],prob)[:,0])*(UU(X,[1,0],prob)[:,1]) + \
    (UU(X,[0,0],prob)[:,1])*(UU(X,[0,1],prob)[:,1]) + PP(X,[0,1],prob)
    return tmp
class INSET():#边界点取值
    def __init__(self,bound,nx,prob,mode):
        self.dim = 2
        #self.area = (bound[0,1] - bound[0,0])*(bound[1,1] - bound[1,0])
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        self.size = nx[0]*nx[1]
        self.X = torch.zeros(self.size,self.dim)#储存内点
        if mode == 'uniform':
            for i in range(nx[0]):
                for j in range(nx[1]):
                    self.X[i*nx[1] + j,0] = bound[0,0] + (i + 0.5)*self.hx[0]
                    self.X[i*nx[1] + j,1] = bound[1,0] + (j + 0.5)*self.hx[1]
        elif mode == 'random':
            tmp = torch.rand(self.size,2)
            self.X[:,0] = bound[0,0] + self.hx[0] + (bound[0,1] - bound[0,0] - 2*self.hx[0])*tmp[:,0]
            self.X[:,1] = bound[1,0] + self.hx[1] + (bound[1,1] - bound[1,0] - 2*self.hx[1])*tmp[:,1]
        else:
            tmp = torch.tensor(self.quasi_samples(self.size))
            self.X[:,0] = bound[0,0] + self.hx[0] + (bound[0,1] - bound[0,0] - 2*self.hx[0])*tmp[:,0]
            self.X[:,1] = bound[1,0] + self.hx[1] + (bound[1,1] - bound[1,0] - 2*self.hx[1])*tmp[:,1]
        self.uu = UU(self.X,[0,0],prob)[:,0:1] 
        self.vv = UU(self.X,[0,0],prob)[:,1:2] 
        self.ff = FF(self.X,prob)
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
class BDSET():#边界点取值
    def __init__(self,bound,nx,prob):
        self.dim = 2
        self.area = (bound[0,1] - bound[0,0])*(bound[1,1] - bound[1,0])
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        self.size = 2*(nx[0] + nx[1])
        self.X = torch.zeros(self.size,self.dim)#储存内点
        m = 0
        for i in range(nx[0]):
            self.X[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
            self.X[m,1] = bound[1,0] 
            m = m + 1
        for j in range(nx[1]):
            self.X[m,0] = bound[0,1]
            self.X[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
            m = m + 1
        for i in range(nx[0]):
            self.X[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
            self.X[m,1] = bound[1,1] 
            m = m + 1
        for j in range(nx[1]):
            self.X[m,0] = bound[0,0]
            self.X[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
            m = m + 1
        #plt.scatter(self.X[:,0],self.X[:,1])
        self.uu = UU(self.X,[0,0],prob)[:,0:1] 
        self.vv = UU(self.X,[0,0],prob)[:,1:2] 
class TESET():
    def __init__(self, bound, nx,prob):
        self.bound = bound
        self.nx = nx
        self.hx = [(self.bound[0,1]-self.bound[0,0])/self.nx[0],
                   (self.bound[1,1]-self.bound[1,0])/self.nx[1]]
        self.prob = prob
        self.size = (self.nx[0] + 1)*(self.nx[1] + 1)
        self.X = torch.zeros(self.size,2)
        m = 0
        for i in range(self.nx[0] + 1):
            for j in range(self.nx[1] + 1):
                self.X[m,0] = self.bound[0,0] + i*self.hx[0]
                self.X[m,1] = self.bound[1,0] + j*self.hx[1]
                m = m + 1
        #plt.scatter(self.X[:,0],self.X[:,1])
        self.uu = UU(self.X,[0,0],prob)[:,0:1] 
        self.vv = UU(self.X,[0,0],prob)[:,1:2] 
        self.pp = PP(self.X,[0,0],prob) 
class LEN():
    def __init__(self,bound,mu):
        self.mu = mu
        self.bound = bound
        self.hx = bound[:,1] - bound[:,0]
    def forward(self,X):
        L = 1.0
        for i in range(2):
            L = L*(1 - (1 - (X[:,i] - self.bound[i,0])/self.hx[i])**self.mu)
            L = L*(1 - (1 - (self.bound[i,1] - X[:,i])/self.hx[i])**self.mu)
        return L.view(-1,1)
        
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
def pred_u(netu,X):
    return netu.forward(X)
def pred_v(netv,X):
    return netv.forward(X)
def pred_p(netp,X):
    return netp.forward(X)
def loadtype(inset,teset,bdset,dtype):
    inset.X = inset.X.type(dtype)
    inset.uu = inset.uu.type(dtype)
    inset.vv = inset.vv.type(dtype)
    inset.ff = inset.ff.type(dtype)

    bdset.X = bdset.X.type(dtype)
    bdset.uu = bdset.uu.type(dtype)
    bdset.vv = bdset.vv.type(dtype)

    teset.X = teset.X.type(dtype)
    teset.uu = teset.uu.type(dtype)
    teset.vv = teset.vv.type(dtype)
    
def loadcuda(netu,netv,netp,inset,teset,bdset):    
    netu = netu.to(device)
    netv = netv.to(device)
    netp = netp.to(device)
    
    
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to(device)
    inset.ff = inset.ff.to(device)
    bdset.X = bdset.X.to(device)
    bdset.uu = bdset.uu.to(device)
    bdset.vv = bdset.vv.to(device)
    teset.X = teset.X.to(device)
def loadcpu(netu,netv,netp,inset,teset,bdset):    
    netu = netu.to('cpu')
    netv = netv.to('cpu')
    netp = netp.to('cpu')
    

    inset.X = inset.X.to('cpu')
    inset.ff = inset.ff.to('cpu')
    bdset.X = bdset.X.to('cpu')
    bdset.uu = bdset.uu.to('cpu')
    bdset.vv = bdset.vv.to('cpu')
    teset.X = teset.X.to('cpu')

def error(u_pred,u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    tmp = (u_pred - u_acc)
    return max(abs(tmp))

#----------------------
def Lossyp(netu,netv,netp,inset,bdset,penalty_in):
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
    
    inset.res_u = (-nu*u_lap + inset.u*u_x[:,0:1] + inset.v*u_x[:,1:2] + p_x[:,0:1] - inset.ff[:,0:1])**2
    inset.res_v = (-nu*v_lap + inset.u*v_x[:,0:1] + inset.v*v_x[:,1:2] + p_x[:,1:2] - inset.ff[:,1:2])**2
    inset.res_div = (u_x[:,0:1] + v_x[:,1:2])**2
    inset.loss_u = torch.sqrt(inset.res_u.mean())
    inset.loss_v = torch.sqrt(inset.res_v.mean())
    inset.loss_div = torch.sqrt(inset.res_div.mean())

    inset.loss = penalty_in[0]*inset.loss_u + penalty_in[1]*inset.loss_v + \
        penalty_in[2]*inset.loss_div
    bdset.res_u = (pred_u(netu,bdset.X) - bdset.uu)**2
    bdset.res_v = (pred_v(netv,bdset.X) - bdset.vv)**2
    bdset.loss = torch.sqrt(bdset.res_u).mean() + torch.sqrt(bdset.res_v).mean()
    return inset.loss + bdset.loss
    
def Trainyp(netu,netv,netp, inset,bdset,penalty_in,optim, epoch,error_history,optimtype):
    print('train neural network yp,optim:%s'%(optimtype))
    loss_history,lossin_history,lossbd_history = error_history
    loss = Lossyp(netu,netv,netp,inset,bdset,penalty_in)
    print('epoch_yp: %d, Lu:%.3e,Lv:%.3e,Ldiv:%.3e, bloss:%.3e,time: %.2f\n'
          %(0,inset.loss_u.item(),inset.loss_v.item(),inset.loss_div.item(), bdset.loss.item(),0.00))
    
    for it in range(epoch):
        st = time.time()
        if optimtype == 'Adam':
            for j in range(100):
                optim.zero_grad()
                loss = Lossyp(netu,netv,netp,inset,bdset,penalty_in)
                loss.backward()
                optim.step()
        if optimtype == 'BFGS' or optimtype == 'LBFGS':
            def closure():
                optim.zero_grad()
                loss = Lossyp(netu,netv,netp,inset,bdset,penalty_in)
                loss.backward()
                return loss
            optim.step(closure) 
        loss = Lossyp(netu,netv,netp,inset,bdset,penalty_in)
        ela = time.time() - st
        print('epoch_yp: %d, loss:%.3e,Lu:%.3e,Lv:%.3e,Ldiv:%.3e, bloss:%.3e,time: %.2f\n'
          %(it + 1,loss.item(),inset.loss_u.item(),inset.loss_v.item(),inset.loss_div.item(), bdset.loss.item(),ela))
        if (it + 1)%4 == 0:
            u_L1err = error(pred_u(netu,inset.X),inset.uu)
            v_L1err = error(pred_v(netv,inset.X),inset.vv)
            print('epoch_yp:%d,u_L1err:%.3e,v_L1err:%.3e'%(it + 1,u_L1err,v_L1err))
        loss_u = inset.loss_u.item()
        loss_v = inset.loss_v.item()
        loss_div = inset.loss_div.item()
        loss_history.append(loss.item())
        lossin_history.append([inset.loss.item(),loss_u,loss_v,loss_div])
        
        lossbd_history.append(bdset.loss.item())
        error_history = [loss_history,lossin_history,lossbd_history]
    return error_history



nx = [64,64]
nx_te = [10,10]
prob = 1
mu = 1
lr = 1e0
#mode = 'random'
#mode = 'uniform'
mode = 'qmc'
inset = INSET(bound,nx,prob,mode)
bdset = BDSET(bound,nx,prob)
teset = TESET(bound,nx_te,prob)

lenth = LEN(bound,mu)


dtype = torch.float64
wid = 25
layer_u = [2,wid,wid,1];netu = Net(layer_u,dtype)
layer_v = [2,wid,wid,1];netv = Net(layer_v,dtype)
layer_p = [2,wid,wid,1];netp = Net(layer_p,dtype)


fname1 = "u-CVDlay%dvar.pt"%(wid)
fname2 = "v-CVDlay%dvar.pt"%(wid)
fname3 = "p-CVDlay%dvar.pt"%(wid)

#netu = torch.load(fname1)
#netv = torch.load(fname2)
#netp = torch.load(fname3)

loadtype(inset,teset,bdset,dtype)
loadcuda(netu,netv,netp,inset,teset,bdset)

error_history = [[],[],[]]
epoch = 20
start_time = time.time()
penalty_in = [1e0,1e0,1e0]

lr = 1e0
max_iter = 1
#optimtype = 'Adam'
optimtype = 'BFGS'
#optimtype = 'LBFGS'
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
                        tolerance_grad=1e-16, tolerance_change=1e-16,
                        history_size=2500,
                        line_search_fn='strong_wolfe')
    if optimtype == 'Adam':
        optim = torch.optim.Adam(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),
                        lr=lr)
    error_history = \
        Trainyp(netu,netv,netp, inset,bdset,penalty_in,optim, epoch,error_history,optimtype)
    if (i + 1)%5 == 0:
        lr *= 0.985
elapsed = time.time() - start_time
print('Finishied! train time: %.2f\n' %(elapsed)) 

loadcpu(netu,netv,netp,inset,teset,bdset)
torch.cuda.empty_cache()

torch.save(netu, fname1)
torch.save(netv, fname2)
torch.save(netp, fname3)

loss_history,lossin_history,lossbd_history = error_history
np.save('lay%d-epoch%d-lossin.npy'%(wid,epoch),lossin_history)
np.save('lay%d-epoch%d-lossvd.npy'%(wid,epoch),lossbd_history)
np.save('lay%d-epoch%d-loss.npy'%(wid,epoch),loss_history)
#%%
fig, ax = plt.subplots(1,2,figsize=(12,3.5))

ax[0].semilogy(np.array(lossin_history))
ax[0].legend(['loss', 'loss_u', 'loss_v', 'loss_div'])
ax[0].set_xlabel('iters') 

ax[1].plot(np.array(lossbd_history))
ax[1].legend(['boundary loss'])
ax[1].set_xlabel('iters') 

fig.tight_layout()
plt.show()

plt.scatter(inset.X.detach().numpy()[:,0],inset.X.detach().numpy()[:,1])
plt.show()
#%%
nx_te_in = [64,64]
x_train = np.linspace(bound[0,0],bound[0,1],nx_te_in[0])
y_train = np.linspace(bound[1,0],bound[1,1],nx_te_in[1])

x0, x1= np.meshgrid(x_train,y_train)

xx = np.hstack((x0.reshape(-1,1), x1.reshape(-1,1)))
xx = torch.from_numpy(xx).type(dtype)
u_pred = pred_u(netu,xx).detach().numpy()
v_pred = pred_v(netv,xx).detach().numpy()
u_acc = UU(xx,[0,0],prob).numpy()[:,0:1]

v_acc = UU(xx,[0,0],prob).numpy()[:,1:2]
fig, ax = plt.subplots(2,3,figsize=(12,6))
for i in range(1):
    for j in range(3):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([bound[0,0],bound[0,1]])
        ax[i,j].set_ylim([bound[1,0],bound[1,1]])
        ax[i,j].axis('off')
        
num_line = 10
x0 = x0.flatten()
x1 = x1.flatten()
u_pred = u_pred.reshape(-1,1).flatten()
v_pred = v_pred.reshape(-1,1).flatten()
u_acc = u_acc.reshape(-1,1).flatten()
v_acc = v_acc.reshape(-1,1).flatten()

    
ax00 = ax[0,0].tricontourf(x0, x1, u_pred, num_line, cmap='rainbow')
fig.colorbar(ax00,ax=ax[0,0],fraction = 0.05,pad = 0.04)
ax[0,0].set_title('PINN: u')

ax10 = ax[1,0].tricontourf(x0, x1, v_pred, num_line, alpha=1, cmap='rainbow')
fig.colorbar(ax10,ax=ax[1,0],fraction = 0.05,pad = 0.04)
ax[1,0].set_title('PINN: v')

ax01 = ax[0,1].tricontourf(x0, x1, u_acc, num_line, cmap='rainbow')
fig.colorbar(ax01,ax=ax[0,1],fraction = 0.05,pad = 0.04)
ax[0,1].set_title('exact: u')

ax11 = ax[1,1].tricontourf(x0, x1, v_acc, num_line, alpha=1, cmap='rainbow')
fig.colorbar(ax11,ax=ax[1,1],fraction = 0.05,pad = 0.04)
ax[1,1].set_title('exact: v')

ax02 = ax[0,2].tricontourf(x0, x1, u_pred - u_acc, num_line, cmap='rainbow')
fig.colorbar(ax02,ax=ax[0,2],fraction = 0.05,pad = 0.04)
ax[0,2].set_title('difference: u')


ax12 = ax[1,2].tricontourf(x0, x1, v_pred - v_acc, num_line, alpha=1, cmap='rainbow')
fig.colorbar(ax12,ax=ax[1,2],fraction = 0.05,pad = 0.04)
ax[1,2].set_title('difference: v')

plt.suptitle('PINN solve K flow',fontsize=20)

fig.tight_layout()
plt.show()



