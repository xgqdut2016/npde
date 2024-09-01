import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def UU(X, order,prob):
    if prob==1:
        temp = 10*(X[:,0]+X[:,1])**2 + (X[:,0]-X[:,1])**2 + 0.5
        if order[0]==0 and order[1]==0:
            return torch.log(temp)
        if order[0]==1 and order[1]==0:
            return temp**(-1) * (20*(X[:,0]+X[:,1]) + 2*(X[:,0]-X[:,1]))
        if order[0]==0 and order[1]==1:
            return temp**(-1) * (20*(X[:,0]+X[:,1]) - 2*(X[:,0]-X[:,1]))
        if order[0]==2 and order[1]==0:
            return - temp**(-2) * (20*(X[:,0]+X[:,1])+2*(X[:,0]-X[:,1])) ** 2 \
                   + temp**(-1) * (22)
        if order[0]==1 and order[1]==1:
            return - temp**(-2) * (20*(X[:,0]+X[:,1])+2*(X[:,0]-X[:,1])) \
                   * (20*(X[:,0]+X[:,1])-2*(X[:,0]-X[:,1])) \
                   + temp**(-1) * (18)
        if order[0]==0 and order[1]==2:
            return - temp**(-2) * (20*(X[:,0]+X[:,1])-2*(X[:,0]-X[:,1])) ** 2 \
                   + temp**(-1) * (22)

    if prob==2:
        if order[0]==0 and order[1]==0:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   0.5*(torch.exp(2*X[:,1])+torch.exp(-2*X[:,1]))
        if order[0]==1 and order[1]==0:
            return (3*X[:,0]*X[:,0]-1) * \
                   0.5*(torch.exp(2*X[:,1])+torch.exp(-2*X[:,1]))
        if order[0]==0 and order[1]==1:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   (torch.exp(2*X[:,1])-torch.exp(-2*X[:,1]))
        if order[0]==2 and order[1]==0:
            return (6*X[:,0]) * \
                   0.5*(torch.exp(2*X[:,1])+torch.exp(-2*X[:,1]))
        if order[0]==1 and order[1]==1:
            return (3*X[:,0]*X[:,0]-1) * \
                   (torch.exp(2*X[:,1])-torch.exp(-2*X[:,1]))
        if order[0]==0 and order[1]==2:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   2*(torch.exp(2*X[:,1])+torch.exp(-2*X[:,1]))

    if prob==3:
        temp1 = X[:,0]*X[:,0] - X[:,1]*X[:,1]
        temp2 = X[:,0]*X[:,0] + X[:,1]*X[:,1] + 0.1
        if order[0]==0 and order[1]==0:
            return temp1 * temp2**(-1)
        if order[0]==1 and order[1]==0:
            return (2*X[:,0]) * temp2**(-1) + \
                   temp1 * (-1)*temp2**(-2) * (2*X[:,0])
        if order[0]==0 and order[1]==1:
            return (-2*X[:,1]) * temp2**(-1) + \
                   temp1 * (-1)*temp2**(-2) * (2*X[:,1])
        if order[0]==2 and order[1]==0:
            return (2) * temp2**(-1) + \
                   2 * (2*X[:,0]) * (-1)*temp2**(-2) * (2*X[:,0]) + \
                   temp1 * (2)*temp2**(-3) * (2*X[:,0])**2 + \
                   temp1 * (-1)*temp2**(-2) * (2)
        if order[0]==1 and order[1]==1:
            return (2*X[:,0]) * (-1)*temp2**(-2) * (2*X[:,1]) + \
                   (-2*X[:,1]) * (-1)*temp2**(-2) * (2*X[:,0]) + \
                   temp1 * (2)*temp2**(-3) * (2*X[:,0]) * (2*X[:,1])
        if order[0]==0 and order[1]==2:
            return (-2) * temp2**(-1) + \
                   2 * (-2*X[:,1]) * (-1)*temp2**(-2) * (2*X[:,1]) + \
                   temp1 * (2)*temp2**(-3) * (2*X[:,1])**2 + \
                   temp1 * (-1)*temp2**(-2) * (2)

    if prob==4:
        temp = torch.exp(-4*X[:,1]*X[:,1])
        if order[0]==0 and order[1]==0:
            ind = (X[:,0]<=0).float()
            return ind * ((X[:,0]+1)**4-1) * temp + \
                   (1-ind) * (-(-X[:,0]+1)**4+1) * temp
        if order[0]==1 and order[1]==0:
            ind = (X[:,0]<=0).float()
            return ind * (4*(X[:,0]+1)**3) * temp + \
                   (1-ind) * (4*(-X[:,0]+1)**3) * temp
        if order[0]==0 and order[1]==1:
            ind = (X[:,0]<=0).float()
            return ind * ((X[:,0]+1)**4-1) * (temp*(-8*X[:,1])) + \
                   (1-ind) * (-(-X[:,0]+1)**4+1) * (temp*(-8*X[:,1]))
        if order[0]==2 and order[1]==0:
            ind = (X[:,0]<=0).float()
            return ind * (12*(X[:,0]+1)**2) * temp + \
                   (1-ind) * (-12*(-X[:,0]+1)**2) * temp
        if order[0]==1 and order[1]==1:
            ind = (X[:,0]<=0).float()
            return ind * (4*(X[:,0]+1)**3) * (temp*(-8*X[:,1])) + \
                   (1-ind) * (4*(-X[:,0]+1)**3) * (temp*(-8*X[:,1]))
        if order[0]==0 and order[1]==2:
            ind = (X[:,0]<=0).float()
            return ind * ((X[:,0]+1)**4-1) * (temp*(64*X[:,1]*X[:,1]-8)) + \
                   (1-ind) * (-(-X[:,0]+1)**4+1) * (temp*(64*X[:,1]*X[:,1]-8))
def FF(X,prob):
    return -UU(X,[2,0],prob) - UU(X,[0,2],prob)


#函数定义修改完成
class INSET():
    def __init__(self,bound,nx,prob):
        self.dim = 2
        self.area = (bound[0,1] - bound[0,0])*(bound[1,1] - bound[1,0])
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        self.size = nx[0]*nx[1]
        self.X = torch.zeros(self.size,self.dim)#储存内点
        for j in range(nx[1]):
            for i in range(nx[0]):
                self.X[j*nx[0] + i,0] = bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[j*nx[0] + i,1] = bound[1,0] + (j + 0.5)*self.hx[1]
        self.u_acc = UU(self.X,[0,0],prob).view(-1,1)#储存内点精确解
        self.right = FF(self.X,prob).view(-1,1)# - nabla A \nabla u  = -c
        

class BDSET():#边界点取值
    def __init__(self,bound,nx,prob):
        self.dim = 2
        self.DS = 2*(nx[0] + nx[1])
        self.Dlenth = 2*(bound[0,1] - bound[0,0]) + 2*(bound[1,1] - bound[1,0])
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        
        self.X = torch.zeros(self.DS,self.dim)#储存内点
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
        self.Dright = UU(self.X,[0,0],prob).view(-1,1)
class TESET():
    def __init__(self,bound,nx,prob):
        self.dim = 2
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        M,N = nx[0] + 1,nx[1] + 1
        self.size = M*N
        self.X = torch.zeros(self.size,self.dim)#储存求解区域所有网格点，包括边界点
        for j in range(N):
            for i in range(M):
                self.X[j*M + i,0] = bound[0,0] + i*self.hx[0]
                self.X[j*M + i,1] = bound[1,0] + j*self.hx[1]
        self.u_acc = UU(self.X,[0,0],prob).view(-1,1)#储存求解区域网格点对应精确解
#数据集修改完成
np.random.seed(1234)
torch.manual_seed(1234)


class Net(torch.nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        

        fc = []
        for i in range(self.layers_hid_num+1):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
        self.fc = torch.nn.Sequential(*fc)

        for i in range(self.layers_hid_num+1):
            
            #self.fc[i].weight.data = self.fc[i].weight.data.clone().detach().requires_grad_(True)
            self.fc[i].weight.data = self.fc[i].weight.data
            #self.fc[i].bias.data = self.fc[i].bias.data.clone().detach().requires_grad_(True)
            self.fc[i].bias.data = self.fc[i].bias.data
    def forward(self, x):
        #x = x.to(device)
        for i in range(self.layers_hid_num):
            x = torch.sin(self.fc[i](x))#.to(device)
        return self.fc[-1](x)
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])   

def pred(netf,X):
    return netf.forward(X)

def error(u_pred, u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    #return (((u_pred-u_acc)**2).sum()/(u_acc**2).sum()) ** (0.5)
    return (((u_pred-u_acc)**2).mean()) ** (0.5)
# ----------------------------------------------------------------------------------------------------
def loadcuda(inset,bdset,netf):    
    netf = netf.to(device)
    
    #inset.X.requires_grad = True
    inset.X = inset.X.to(device)
    bdset.X = bdset.X.to(device)
    inset.right = inset.right.to(device)
    bdset.Dright = bdset.Dright.to(device)
    bdset.Dlenth = bdset.Dlenth.to(device)
    return inset,bdset,netf
def loadcpu(inset,bdset,netf):    
    netf = netf.to('cpu')
    
    #inset.X.requires_grad = True
    inset.X = inset.X.to('cpu')
    bdset.X = bdset.X.to('cpu')
    inset.right = inset.right.to('cpu')
    bdset.Dright = bdset.Dright.to('cpu')
    bdset.Dlenth = bdset.Dlenth.to('cpu')
    return inset,bdset,netf
    

def Lossf(netf,inset,bdset):
    if inset.X.requires_grad is not True:
        inset.X.requires_grad = True
    
    
    
    insetF = netf.forward(inset.X)
    # print('insetF device: {}'.format(insetF.device))
    
    insetFx, = torch.autograd.grad(insetF, inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=torch.ones(inset.size,1).to(device))
    u_in = insetF#inset.G为netg在inset.X上取值，后面训练时提供，此举为加快迭代速度
    ux = insetFx#复合函数求导，提高迭代效率
    taux, = torch.autograd.grad(ux[:,0:1], inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=torch.ones(inset.size,1).to(device))
    tauy, = torch.autograd.grad(ux[:,1:2], inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=torch.ones(inset.size,1).to(device))
    
    out_in = ((taux[:,0:1] + tauy[:,1:2] + inset.right)**2 + (taux[:,1:2] - tauy[:,0:1])**2).mean()
    ub = netf.forward(bdset.X)  

    out_b = bdset.Dlenth * ((ub - bdset.Dright)**2).mean()
    beta = 5e2
    return out_in + beta*out_b



# Train neural network f
def Trainf(netf, inset, bdset, optimf, epochf):
    print('train neural network f')
    ERROR,BUZHOU = [],[]
    
    lossf = Lossf(netf,inset,bdset)
    lossoptimal = lossf
    trainerror = error(netf.forward(inset.X), inset.u_acc)
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
        trainerror = error(netf.forward(inset.X), inset.u_acc)
        ERROR.append(trainerror.item())
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,lossf:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,lossf.item(),trainerror,ela))
    return ERROR,BUZHOU


prob = 1
bounds = torch.Tensor([0.0,1.0,0.0,1.0]).reshape(2,2)
nx_tr = [41,41]#训练集剖分
nx_te = [101,101]#测试集剖分
    
epochf = 2
lr = 1e-2
tests_num = 1

    # ------------------------------------------------------------------------------------------------
testerror = torch.zeros(tests_num)
for it in range(tests_num):

    inset = INSET(bounds, nx_tr, prob)
    bdset = BDSET(bounds, nx_tr, prob)
    teset = TESET(bounds, nx_te, prob)

        

        
    
    lay = [2,10,10,20,1];netf = Net(lay)
    inset,bdset,netf = loadcuda(inset,bdset,netf)
        #optimg = torch.optim.Adam(netg.parameters(), lr=lr)
        #optimf = torch.optim.Adam(netf.parameters(), lr=lr)
        
    optimf = torch.optim.LBFGS(netf.parameters(), lr=lr,max_iter = 100,history_size=2500,
                                line_search_fn = 'strong_wolfe')


    start_time = time.time()
    ERROR,BUZHOU = Trainf(netf, inset, bdset, optimf, epochf)
    inset,bdset,netf = loadcpu(inset,bdset,netf)
    torch.cuda.empty_cache()
    elapsed = time.time() - start_time
    print('Train time: %.2f' %(elapsed))

    
    netf.load_state_dict(torch.load('best_netf.pkl'))
    te_U = pred(netf,teset.X)
    testerror[it] = error(te_U, teset.u_acc)
    print('testerror = %.3e\n' %(testerror[it].item()))
    
    print(testerror.data)
    testerror_mean = testerror.mean()
    testerror_std = testerror.std()
    print('testerror_mean = %.3e, testerror_std = %.3e'
      %(testerror_mean.item(),testerror_std.item()))


