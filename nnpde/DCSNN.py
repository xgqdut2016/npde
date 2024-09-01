import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
from scipy.stats import qmc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
beta_in = 1
beta_out = 1e-3


def surface(X):
    return (X[:,0]/0.2)**2 + (X[:,1]/0.5)**2 - 1

def UU(X, order,prob):
    ind = (surface(X) < 0)
    out = torch.zeros_like(X[:,0:1])
    if prob == 1:
        if order == [0,0]:
            out[ind,:] = torch.exp(X[ind,0:1])*torch.exp(X[ind,1:2])
            out[ind == 0,:] = torch.sin(X[ind == 0,0:1])*torch.sin(X[ind == 0,1:2])
            return out
        if order == [2,0]:
            out[ind,:] = torch.exp(X[ind,0:1])*torch.exp(X[ind,1:2])
            out[ind == 0,:] = -torch.sin(X[ind == 0,0:1])*torch.sin(X[ind == 0,1:2])
            return out
        if order == [0,2]:
            out[ind,:] = torch.exp(X[ind,0:1])*torch.exp(X[ind,1:2])
            out[ind == 0,:] = -torch.sin(X[ind == 0,0:1])*torch.sin(X[ind == 0,1:2])
            return out
    if prob == 2:
        temp1 = 10*(X[ind,0]+X[ind,1])**2 + (X[ind,0]-X[ind,1])**2 + 0.5
        temp2 = (X[ind == 0,0]**3 - X[ind == 0,0]) * 0.5*(torch.exp(2*X[ind == 0,1]) + torch.exp(-2*X[ind == 0,1]))
        if order[0]==0 and order[1]==0:
            out[ind,0] = torch.log(temp1)
            out[ind == 0,0] = temp2
            return out
        if order == [2,0]:
            out[ind,0] = - temp1**(-2) * (20*(X[ind,0]+X[ind,1])+2*(X[ind,0]-X[ind,1])) ** 2 \
                   + temp1**(-1) * (22)
            out[ind == 0,0] = (6*X[ind == 0,0]) * \
                   0.5*(torch.exp(2*X[ind == 0,1])+torch.exp(-2*X[ind == 0,1]))
            return out
        if order == [0,2]:
            out[ind,0] = - temp1**(-2) * (20*(X[ind,0]+X[ind,1])-2*(X[ind,0]-X[ind,1])) ** 2 \
                   + temp1**(-1) * (22)
            out[ind == 0,0] = (X[ind == 0,0]**3-X[ind == 0,0]) * \
                   2*(torch.exp(2*X[ind == 0,1])+torch.exp(-2*X[ind == 0,1]))
            return out
def FF(X,prob):
    return -UU(X,[2,0],prob) - UU(X,[0,2],prob)
def VV(X,prob):
    if prob == 1:
        return torch.sin(X[:,0:1])*torch.sin(X[:,1:2]) - torch.exp(X[:,0:1])*torch.exp(X[:,1:2])
    if prob == 2:
        temp1 = 10*(X[:,0:1]+X[:,1:2])**2 + (X[:,0:1]-X[:,1:2])**2 + 0.5
        temp2 =  (X[:,0:1]**3 -X[:,0:1]) * \
                   0.5*(torch.exp(2*X[:,1:2])+torch.exp(-2*X[:,1:2]))
        return temp2 - torch.log(temp1) 
def WW(X,prob):
    dir = torch.zeros_like(X)
    dir[:,0] = 2*X[:,0]/(0.2**2)
    dir[:,1] = 2*X[:,1]/(0.5**2)
    tmp = dir/torch.sqrt(dir[:,0:1]**2 + dir[:,1:2]**2)
    if prob == 1:
        if X.requires_grad == False:
            X.requires_grad = True
        phi_in = torch.exp(X[:,0:1])*torch.exp(X[:,1:2])
        phi_out = torch.sin(X[:,0:1])*torch.sin(X[:,1:2])
        phi_in_x, = torch.autograd.grad(phi_in, X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(X.shape[0],1).to(X.device))
        phi_out_x, = torch.autograd.grad(phi_out, X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(X.shape[0],1).to(X.device))                           
        phi_in_n = (phi_in_x*tmp).sum(1,keepdims=True)
        phi_out_n = (phi_out_x*tmp).sum(1,keepdims=True)
        X.requires_grad = False
        return (beta_out*phi_out_n - beta_in*phi_in_n).data
    if prob == 2:
        
        if X.requires_grad == False:
            X.requires_grad = True
        temp1 = 10*(X[:,0:1]+X[:,1:2])**2 + (X[:,0:1]-X[:,1:2])**2 + 0.5
        temp2 =  (X[:,0:1]**3 - X[:,0:1]) * \
                   0.5*(torch.exp(2*X[:,1:2])+torch.exp(-2*X[:,1:2]))
        
        phi_in = torch.log(temp1)
        phi_out = temp2
        phi_in_x, = torch.autograd.grad(phi_in, X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(X.shape[0],1).to(X.device))
        phi_out_x, = torch.autograd.grad(phi_out, X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(X.shape[0],1).to(X.device))                           
        phi_in_n = (phi_in_x*tmp).sum(1,keepdims=True)
        phi_out_n = (phi_out_x*tmp).sum(1,keepdims=True)
        X.requires_grad = False
        return (beta_out*phi_out_n - beta_in*phi_in_n).data
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
        self.uu = UU(self.X,[0,0],prob).reshape(-1,1) 
        self.ind = (self.X[:,0]/0.2)**2 + (self.X[:,1]/0.5)**2 < 1
        self.ff = FF(self.X,prob).reshape(-1,1)
        
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
        self.uu = UU(self.X,[0,0],prob).reshape(-1,1)
        
class Interface():
    def __init__(self,size):
        self.X = torch.zeros(size,2)
        self.size = size
        hx = torch.tensor(2*np.pi/size)
        for i in range(size):
            theta = (i + 0.5)*hx
            self.X[i,0] = 0.2*torch.cos(theta)
            self.X[i,1] = 0.5*torch.sin(theta)
        self.vv = VV(self.X,prob).reshape(-1,1)
        self.ww = WW(self.X,prob).reshape(-1,1)
        tmp = torch.zeros_like(self.X)
        tmp[:,0] = 2*self.X[:,0]/(0.2**2)
        tmp[:,1] = 2*self.X[:,1]/(0.5**2)
        self.dir = tmp/torch.sqrt(tmp[:,0:1]**2 + tmp[:,1:2]**2)
        self.z = torch.ones_like(self.X[:,0:1])
        self.X_in = torch.cat([self.X,-self.z],1)
        self.X_out = torch.cat([self.X,self.z],1)

def predata(inset,bdset):
    ind = surface(inset.X) < 0
    inset.z = torch.ones_like(inset.X[:,0:1])
    inset.z[ind,:] = -1
    
    inset.X = torch.cat([inset.X,inset.z],1)
    bdset.z = torch.ones_like(bdset.X[:,0:1])
    bdset.X = torch.cat([bdset.X,bdset.z],1)
    
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
def loadtype(inset,bdset,interface,dtype):
    inset.X = inset.X.type(dtype)
    inset.uu = inset.uu.type(dtype)
    inset.ff = inset.ff.type(dtype)

    bdset.X = bdset.X.type(dtype)
    bdset.uu = bdset.uu.type(dtype)

    interface.X = interface.X.type(dtype)
    interface.X_in = interface.X_in.type(dtype)
    interface.X_out = interface.X_out.type(dtype)
    interface.dir = interface.dir.type(dtype)
    interface.vv = interface.vv.type(dtype)
    interface.ww = interface.ww.type(dtype)
def loadcuda(netu,inset,bdset,interface):    
    netu = netu.to(device)
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.X = inset.X.to(device)
    inset.ff = inset.ff.to(device)
    bdset.X = bdset.X.to(device)
    bdset.uu = bdset.uu.to(device)

    interface.X = interface.X.to(device)
    if interface.X_in.requires_grad == False:
        interface.X_in.requires_grad = True
    if interface.X_out.requires_grad == False:
        interface.X_out.requires_grad = True
    interface.X_in = interface.X_in.to(device)
    interface.X_out = interface.X_out.to(device)
    interface.dir = interface.dir.to(device)
    interface.vv = interface.vv.to(device)
    interface.ww = interface.ww.to(device)
def loadcpu(netu,inset,bdset,interface):    
    netu = netu.to('cpu')
    inset.X = inset.X.to('cpu')
    inset.ff = inset.ff.to('cpu')
    bdset.X = bdset.X.to('cpu')
    bdset.uu = bdset.uu.to('cpu')
    interface.dir = interface.dir.to('cpu')
    interface.X = interface.X.to('cpu')
    interface.X_in = interface.X_in.to('cpu')
    interface.X_out = interface.X_out.to('cpu')
    interface.vv = interface.vv.to('cpu')
    interface.ww = interface.ww.to('cpu')
    

def error(u_pred,u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    tmp = (u_pred - u_acc)
    return max(abs(tmp))
def Loss_u(netu,inset,bdset,interface):
    if inset.X.requires_grad is not True:
        inset.X.requires_grad = True
    
    inset.u = pred_u(netu,inset.X)
    u_x, = torch.autograd.grad(inset.u, inset.X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(inset.size,1).to(device))
    
    
    u_xx, = torch.autograd.grad(u_x[:,0:1], inset.X, create_graph=True, retain_graph=True,
                                  grad_outputs=torch.ones(inset.size,1).to(device))
    u_yy, = torch.autograd.grad(u_x[:,1:2], inset.X, create_graph=True, retain_graph=True,
                                  grad_outputs=torch.ones(inset.size,1).to(device))
    u_lap = u_xx[:,0:1] + u_yy[:,1:2] 
    inset.res = (u_lap + inset.ff)**2
    
    bdset.u = pred_u(netu,bdset.X)
    bdset.res = (bdset.u - bdset.uu)**2

    interface.u_out = pred_u(netu,interface.X_out)
    interface.u_in = pred_u(netu,interface.X_in)
    interface.res_d = (interface.u_out - interface.u_in - interface.vv)**2
    interin_x, = torch.autograd.grad(interface.u_in, interface.X_in, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(interface.X_in.shape[0],1).to(device))
    interin_n = (interin_x[:,0:2]*interface.dir).sum(1,keepdims = True)

    interout_x, = torch.autograd.grad(interface.u_out, interface.X_out, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(interface.X_out.shape[0],1).to(device))                               
    interout_n = (interout_x[:,0:2]*interface.dir).sum(1,keepdims = True)
    interface.res_n = (beta_out*interout_n - beta_in*interin_n - interface.ww)**2
    inset.loss = torch.sqrt(inset.res.mean())
    bdset.loss = torch.sqrt(bdset.res.mean())
    interface.loss_d = torch.sqrt(interface.res_d.mean())
    interface.loss_n = torch.sqrt(interface.res_n.mean())
    loss = inset.loss + bdset.loss + interface.loss_d + interface.loss_n
    return loss
def Train_u(netu,inset,bdset,optimu, epochu,interface):
    print('train neural network u')
    ERROR,BUZHOU = [],[]
    
    loss = Loss_u(netu,inset,bdset,interface)
    lossoptimal = loss
    trainerror = error(pred_u(netu,inset.X), inset.uu)
    print('epoch: %d, loss: %.3e, trainerror: %.3e'
          %(0, loss.item(), trainerror.item()))
    
    cycle = 100
    for i in range(epochu):
        st = time.time()
        def closure():
            optimu.zero_grad()
            lossu = Loss_u(netu,inset,bdset,interface)
            lossu.backward()
            return lossu
        optimu.step(closure)
        
        loss = Loss_u(netu,inset,bdset,interface)
        
        ela = time.time() - st
        trainerror = error(pred_u(netu,inset.X), inset.uu)
        ERROR.append(trainerror.item())
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,loss:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,loss.item(),trainerror,ela))
    return ERROR,BUZHOU
nx = [32,32]
nx_bd = [16,16]
bound = np.array([-1,1,-1,1]).reshape(2,2)
prob = 1

lr = 1e-2
epochu = 10
#mode = 'random'
#mode = 'uniform'
mode = 'qmc'
inset = INSET(bound,nx,prob,mode)
bdset = BDSET(bound,nx_bd,prob)
size = 40
interface = Interface(size)
predata(inset,bdset)
dtype = torch.float64
loadtype(inset,bdset,interface,dtype)
wid = 10
lay = [3,wid,wid,1];netu = Net(lay,dtype)
fname1 = "u-lay%dvar.pt"%(wid)
#netu = torch.load(fname1)
loadcuda(netu,inset,bdset,interface)
test_num = 1
for it in range(test_num):

        
    optimu = torch.optim.LBFGS(netu.parameters(), lr=lr,max_iter = 100,history_size=2500,
                                line_search_fn = 'strong_wolfe')


    start_time = time.time()
    ERROR,BUZHOU = Train_u(netu,inset,bdset,optimu, epochu,interface)
    
    elapsed = time.time() - start_time
    print('Train time: %.2f' %(elapsed))

torch.save(netu, fname1)
loadcpu(netu,inset,bdset,interface)
torch.cuda.empty_cache()
#%%
nx_te_in = [64,64]
x_train = np.linspace(bound[0,0],bound[0,1],nx_te_in[0])
y_train = np.linspace(bound[1,0],bound[1,1],nx_te_in[1])

X,Y= np.meshgrid(x_train,y_train)

xx = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
xx = torch.from_numpy(xx).type(dtype)
u_acc = UU(xx,[0,0],prob).numpy().reshape(nx_te_in[0],nx_te_in[1])
#xiugai xx
z = torch.ones_like(xx[:,0:1])
ind = surface(xx) < 0
z[ind,:] = -1
xx = torch.cat([xx,z],1)
u_pred = pred_u(netu,xx).detach().numpy().reshape(nx_te_in[0],nx_te_in[1])

err = u_acc - u_pred
print('the test error:%.3e'%(max(abs(err.reshape(-1,1)))))
fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.set_title('u_acc of prob = %d:test grid:%d'%(prob,nx_te_in[0]))
surf = ax.plot_surface(X, Y, u_acc, cmap='turbo', linewidth=1, antialiased=False)
plt.colorbar(surf,ax=ax,fraction=0.03)
ax.view_init(20, -120)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.set_title('u_pred of prob = %d:test grid:%d'%(prob,nx_te_in[0]))
surf = ax.plot_surface(X, Y, u_pred, cmap='jet', linewidth=1, antialiased=False)
plt.colorbar(surf,ax=ax,fraction=0.03)
ax.view_init(20, -120)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.set_title('err of prob = %d:test grid:%d'%(prob,nx_te_in[0]))
surf = ax.plot_surface(X, Y, err, cmap='jet', linewidth=1, antialiased=False)
plt.colorbar(surf,ax=ax,fraction=0.03)
ax.view_init(20, -120) 
plt.show()
#plt.suptitle(r'$\beta=0.032$')
plt.savefig('3D.png')



