import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import bfgs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
a_min = 0
a_max = 1
b_min = 0
b_max = 1
L_min = 0.1
L_max = 1
bounds = torch.tensor([0,1,0,1,a_min,a_max,b_min,b_max,L_min,L_max]).reshape(5,2)
hr = 1e-3
def UU(X, order,prob):
    x = X[:,0]
    y = X[:,1]
    a = X[:,2]
    b = X[:,3]
    L = X[:,4]
    tmp = torch.exp((-(x - a)**2 - (y - b)**2)/(2*L**2))
    if prob==1:
        if order[0]==0 and order[1]==0:
            return tmp
        if order[0]==1 and order[1]==0:
            return -(x - a)*tmp/(L**2)
        if order[0]==0 and order[1]==1:
            return -(y - b)*tmp/(L**2)
        if order[0]==2 and order[1]==0:
            return -tmp/(L**2) + tmp*(x - a)**2/(L**4)
        if order[0]==0 and order[1]==2:
            return -tmp/(L**2) + tmp*(y - b)**2/(L**4)
def FF(X,prob):
    return -(UU(X,[2,0],prob) + UU(X,[0,2],prob))
class INSET():
    def __init__(self,bounds,size_tr,prob):
        self.dim = 5
        tmp = torch.rand(size_tr,self.dim)
        self.X = bounds[:,0] + hr + (bounds[:,1] - bounds[:,0] - 2*hr)*tmp
        self.right = FF(self.X,prob).reshape(-1,1)
        self.size = self.X.shape[0]
        self.u_acc = UU(self.X,[0,0],prob).reshape(-1,1)
class BDSET():#边界点取值
    def __init__(self,bounds,nx,prob):
        self.dim = 5
        self.DS = 2*(nx[0] + nx[1])
        tmp = torch.rand(self.DS,self.dim)
        self.X = bounds[:,0] + hr + (bounds[:,1] - bounds[:,0] - 2*hr)*tmp
        
        self.hx = [(bounds[0,1] - bounds[0,0])/nx[0],(bounds[1,1] - bounds[1,0])/nx[1]]
        
        tmp = torch.zeros(self.DS,2)#储存内点
        m = 0
        for i in range(nx[0]):
            tmp[m,0] = bounds[0,0] + (i + 0.5)*self.hx[0]
            tmp[m,1] = bounds[1,0] 
            m = m + 1
        for j in range(nx[1]):
            tmp[m,0] = bounds[0,1]
            tmp[m,1] = bounds[1,0] + (j + 0.5)*self.hx[1]
            m = m + 1
        for i in range(nx[0]):
            tmp[m,0] = bounds[0,0] + (i + 0.5)*self.hx[0]
            tmp[m,1] = bounds[1,1] 
            m = m + 1
        for j in range(nx[1]):
            tmp[m,0] = bounds[0,0]
            tmp[m,1] = bounds[1,0] + (j + 0.5)*self.hx[1]
            m = m + 1
        self.X[:,0:2] = tmp
        self.Dright = UU(self.X,[0,0],prob).view(-1,1)
class TESET():
    def __init__(self,bounds,size_te,prob):
        self.dim = 5
        tmp = torch.rand(size_te,self.dim)
        self.X = bounds[:,0] + hr + (bounds[:,1] - bounds[:,0] - 2*hr)*tmp
        self.right = FF(self.X,prob).reshape(-1,1)
        self.size = self.X.shape[0]
        self.u_acc = UU(self.X,[0,0],prob).reshape(-1,1)

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
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])

def pred(netf,X):
    return netf.forward(X)

def error(u_pred, u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    #return (((u_pred-u_acc)**2).sum()/(u_acc**2).sum()) ** (0.5)
    #return (((u_pred-u_acc)**2).mean()) ** (0.5)
    return max(abs(u_pred - u_acc))
# ----------------------------------------------------------------------------------------------------
def loadtype(inset,bdset,teset,dtype):
    inset.X = inset.X.type(dtype)
    inset.right = inset.right.type(dtype)
    inset.u_acc = inset.u_acc.type(dtype)
    bdset.X = bdset.X.type(dtype)
    bdset.Dright = bdset.Dright.type(dtype)
    teset.X = teset.X.type(dtype)
    teset.u_acc = teset.u_acc.type(dtype)
    
def loadcuda(inset,bdset,teset,netf):    
    netf = netf.to(device)
    
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to(device)
    bdset.X = bdset.X.to(device)
    inset.right = inset.right.to(device)
    bdset.Dright = bdset.Dright.to(device)
    teset.X = teset.X.to(device)
    teset.u_acc = teset.u_acc.to(device)
    
def loadcpu(inset,bdset,teset,netf):    
    netf = netf.to('cpu')
    
    inset.X = inset.X.to('cpu')
    bdset.X = bdset.X.to('cpu')
    inset.right = inset.right.to('cpu')
    bdset.Dright = bdset.Dright.to('cpu')
    
    teset.X = teset.X.to('cpu')
    teset.u_acc = teset.u_acc.to('cpu')
    

def Lossf(netf,inset,bdset):
    if inset.X.requires_grad is not True:
        inset.X.requires_grad = True
    
    u = netf.forward(inset.X)
    insetFx, = torch.autograd.grad(u, inset.X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(inset.size,1).to(device))
    
    
    tmp_xx, = torch.autograd.grad(insetFx[:,0:1], inset.X, create_graph=True, retain_graph=True,
                                  grad_outputs=torch.ones(inset.size,1).to(device))
    tmp_yy, = torch.autograd.grad(insetFx[:,1:2], inset.X, create_graph=True, retain_graph=True,
                                  grad_outputs=torch.ones(inset.size,1).to(device))
    out_in = ((tmp_xx[:,0:1] + tmp_yy[:,1:2] + inset.right)**2).mean()
    
    beta = 1e0
    ub = netf.forward(bdset.X)
    out_b = ((ub - bdset.Dright)**2).mean()
    
    res = out_in + beta*out_b
    return torch.sqrt(res)

def Trainf(netf,inset,bdset,optimf, epochf):
    print('train neural network f')
    ERROR,BUZHOU = [],[]
    
    lossf = Lossf(netf,inset,bdset)
    lossoptimal = lossf
    trainerror = error(pred(netf,inset.X), inset.u_acc)
    print('epoch: %d, loss: %.3e, trainerror: %.3e'
          %(0, lossf.item(), trainerror.item()))
    torch.save(netf.state_dict(),'best_netf.pkl')
    cycle = 100
    for i in range(epochf):
        st = time.time()
        '''
        for j in range(cycle):
            optimf.zero_grad()
            lossf = Lossf(netf,inset)
            lossf.backward()
            optimf.step()
        '''
        def closure():
            optimf.zero_grad()
            lossf = Lossf(netf,inset,bdset)
            lossf.backward()
            return lossf
        optimf.step(closure)
        
        lossf = Lossf(netf,inset,bdset)
        
        if lossf < lossoptimal:
            lossoptimal = lossf
            torch.save(netf.state_dict(),'best_netf.pkl')
        ela = time.time() - st
        trainerror = error(pred(netf,inset.X), inset.u_acc)
        ERROR.append(trainerror.item())
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,lossf:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,lossf.item(),trainerror,ela))
    return ERROR,BUZHOU
size_tr = 1000
size_te = 1000
prob = 1
nx = [20,20]
inset = INSET(bounds,size_tr,prob)
bdset = BDSET(bounds,nx,prob)
teset = TESET(bounds,size_te,prob)

epochf = 10

lr_f = 1e-1
tests_num = 1
#dtype = torch.float32
dtype = torch.float64
    # ------------------------------------------------------------------------------------------------
testerror = torch.zeros(tests_num)
loadtype(inset,bdset,teset,dtype)

layf = [5,20,20,1];netf = Net(layf,dtype)

loadcuda(inset,bdset,teset,netf)

for it in range(tests_num):

    
    optimf = bfgs.BFGS(netf.parameters(), 
                      lr=lr_f, max_iter = 100,
                      tolerance_grad=1e-15, tolerance_change=1e-15,
                      line_search_fn='strong_wolfe')


    start_time = time.time()
    ERROR,BUZHOU = Trainf(netf,inset,bdset,optimf, epochf)
    netf.load_state_dict(torch.load('best_netf.pkl'))
    elapsed = time.time() - start_time
    print('Train time: %.2f' %(elapsed))

    
    
    te_U = pred(netf,teset.X)
    testerror[it] = error(te_U, teset.u_acc)
    print('testerror = %.3e\n' %(testerror[it].item()))
    
    print(testerror.data)
    testerror_mean = testerror.mean()
    testerror_std = testerror.std()
    print('testerror_mean = %.3e, testerror_std = %.3e'
      %(testerror_mean.item(),testerror_std.item()))

loadcpu(inset,bdset,teset,netf)
torch.cuda.empty_cache()
nx_te_in = [64,32]
x0 = np.linspace(bounds[0,0],bounds[0,1],nx_te_in[0])
x1 = np.linspace(bounds[1,0],bounds[1,1],nx_te_in[1])
number = 8
L = np.linspace(L_min,L_max,number)
a = 0.5
b = 0.5
xx = np.zeros([nx_te_in[0]*nx_te_in[1]*number,5])
m = 0
for i in range(nx_te_in[0]):
    for j in range(nx_te_in[1]):
        for k in range(number):
            xx[m,0] = x0[i]
            xx[m,1] = x1[j]
            xx[m,4] = L[k]
            xx[m,2] = a
            xx[m,3] = b
            m = m + 1

x0, x1, L = np.meshgrid(x0,x1,L)

xx = torch.from_numpy(xx)
U = pred(netf, xx).reshape(-1,number); U = U.detach().numpy()
Ua = UU(xx,[0,0],prob).reshape(-1,number); Ua = Ua.detach().numpy()

fig, ax = plt.subplots(3,number,figsize=(27,9))
for i in range(3):
    for j in range(number):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,1])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')
        
num_line = 100
x0 = np.linspace(bounds[0,0],bounds[0,1],nx_te_in[0])
x1 = np.linspace(bounds[1,0],bounds[1,1],nx_te_in[1])
x0, x1 = np.meshgrid(x0,x1)

for i in range(number):
    u = U[:,i].reshape(nx_te_in[0],nx_te_in[1]).T
    ua = Ua[:,i].reshape(nx_te_in[0],nx_te_in[1]).T
    
    s0 = ax[0,i].contourf(x0, x1, u, num_line, cmap='rainbow')
    ax[0,i].contour(s0, linewidths=0.6, colors='black')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax[0,i].set_title('a=%.2f,b=%.2f,L = %.2f'%(a,b,L[0,0,i]),fontsize=15)    
    fig.colorbar(s0,ax=ax[0,i],fraction = 0.045)

    s1 = ax[1,i].contourf(x0, x1, ua, num_line, cmap='rainbow')
    ax[1,i].contour(s1, linewidths=0.6, colors='black')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.colorbar(s1,ax=ax[1,i],fraction = 0.045)

    s2 = ax[2,i].contourf(x0, x1, u-ua, num_line, cmap='rainbow')
    ax[2,i].contour(s2, linewidths=0.6, colors='black')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.colorbar(s2,ax=ax[2,i],fraction = 0.045)

ax[0,0].text(-0.5,0.5,'NN:',fontsize=15)
ax[1,0].text(-0.5,0.5,'Exact:',fontsize=15)
ax[2,0].text(-0.6,0.5,'err:',fontsize=15)
plt.savefig('gt.png')
fig.tight_layout()
plt.show()

print(netf.total_para())


