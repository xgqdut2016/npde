import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
def U(X, order,prob):
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

def A(X, order, ind):
    if order==0:
        if ind==[1,1]: return (X[:,0]+X[:,1])*(X[:,0]+X[:,1]) + 1 # a11
        if ind==[1,2]: return -(X[:,0]+X[:,1])*(X[:,0]-X[:,1])    # a12
        if ind==[2,1]: return -(X[:,0]+X[:,1])*(X[:,0]-X[:,1])    # a21
        if ind==[2,2]: return (X[:,0]-X[:,1])*(X[:,0]-X[:,1]) + 1 # a22
    if order==1:
        if ind==[1,1]: return 2*(X[:,0]+X[:,1])  # a11_x
        if ind==[1,2]: return -2*X[:,0]          # a12_x
        if ind==[2,1]: return 2*X[:,1]           # a21_y
        if ind==[2,2]: return -2*(X[:,0]-X[:,1]) # a22_y
    
def C(X, prob):
    return A(X,1,[1,1])*U(X,[1,0],prob) + A(X,0,[1,1])*U(X,[2,0],prob) + \
            A(X,1,[1,2])*U(X,[0,1],prob) + A(X,0,[1,2])*U(X,[1,1],prob) + \
            A(X,1,[2,1])*U(X,[1,0],prob) + A(X,0,[2,1])*U(X,[1,1],prob) + \
            A(X,1,[2,2])*U(X,[0,1],prob) + A(X,0,[2,2])*U(X,[0,2],prob)

def NEU(X, n, prob):
    return (A(X,0,[1,1])*U(X,[1,0],prob) + A(X,0,[1,2])*U(X,[0,1],prob)) * n[:,0] + \
           (A(X,0,[2,1])*U(X,[1,0],prob) + A(X,0,[2,2])*U(X,[0,1],prob)) * n[:,1]
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
        self.u_acc = U(self.X,[0,0],prob).view(-1,1)#储存内点精确解
        self.right = - C(self.X,prob).view(-1,1)# - nabla A \nabla u  = -c
        self.AM = torch.zeros(self.size,2,2)#储存矩阵A在所有内点的取值，方便损失函数计算 (A \nabla u)* \nabal u
        self.AM[:,0,0] = A(self.X,0,[1,1]);self.AM[:,0,1] = A(self.X,0,[1,2])
        self.AM[:,1,0] = A(self.X,0,[2,1]);self.AM[:,1,1] = A(self.X,0,[2,2])

class BDSET():
    def __init__(self,bound,nx,prob):
        self.dim = 2
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        self.DS = nx[1]#Dirichlet边界采样点数量
        self.NS = 2*nx[0] + nx[1]#Neumann边界采样点数量
        self.Dlenth = bound[1,1] - bound[1,0]
        self.Nlenth = 2*(bound[0,1] - bound[0,0]) + bound[1,1] - bound[1,0]
        self.DX = torch.zeros(self.DS,self.dim)#Dirichlet边界，{-1}*[-1,1]
        self.NX = torch.zeros(self.NS,self.dim)#Neumann边界
        self.Nn = torch.zeros(self.NS,self.dim)#Neumann边界中对应的3个外法向量
        self.Dn = torch.zeros(self.DS,self.dim)
        for i in range(nx[1]):
            self.DX[i,0] = bound[0,0]
            self.DX[i,1] = bound[1,0] + (i + 0.5)*self.hx[1]
            self.Dn[i,0] = - 1;self.Dn[i,1] = 0
        #下面采集Neumann边界点------------------------------------------
        m = 0
        for i in range(nx[0]):
            self.NX[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
            self.NX[m,1] = bound[1,0]
            self.Nn[m,0] = 0
            self.Nn[m,1] = -1
            m = m + 1
        for j in range(nx[1]):
            self.NX[m,0] = bound[0,1]
            self.NX[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
            self.Nn[m,0] = 1
            self.Nn[m,1] = 0
            m = m + 1
        for i in range(nx[0]):
            self.NX[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
            self.NX[m,1] = bound[1,1]
            self.Nn[m,0] = 0
            self.Nn[m,1] = 1
            m = m + 1
        self.Dright = U(self.DX,[0,0],prob).view(-1,1)#储存Dirichlet边界精确解取值
        self.Nright = NEU(self.NX,self.Nn,prob).view(-1,1)#储存Neumann边界上条件
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
        self.u_acc = U(self.X,[0,0],prob).view(-1,1)#储存求解区域网格点对应精确解
np.random.seed(1234)
torch.manual_seed(1234)
class NETG(nn.Module):#u = netf*lenthfactor + netg，此为netg
    def __init__(self):
        super(NETG,self).__init__()
        self.fc1 = torch.nn.Linear(2,10)
        self.fc2 = torch.nn.Linear(10,10)
        self.fc3 = torch.nn.Linear(10,1)
    def forward(self,x):
        out = torch.sin(self.fc1(x))
        out = torch.sin(self.fc2(out)) + x@torch.eye(x.size(1),10)
        return self.fc3(out)
    
class SIN(nn.Module):#u = netg*lenthfactor + netf，此为netg网络所用的激活函数
    def __init__(self,order):
        super(SIN,self).__init__()
        self.e = order
    def forward(self,x):
        return torch.sin(x)**self.e
class Res(nn.Module):
    def __init__(self,input_size,output_size):
        super(Res,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,output_size),
            SIN(1),
            nn.Linear(output_size,output_size),
            SIN(1)
        )
        self.input = input_size
        self.output = output_size
    def forward(self,x):
        x = self.model(x) + x@torch.eye(x.size(1),self.output)#模拟残差网络
        return x
class NETF(nn.Module):#u = netg*lenthfactor + netf，此为netg，此netg逼近内部点取值
    def __init__(self):
        super(NETF,self).__init__()
        self.model = nn.Sequential(
            Res(2,10),
            Res(10,10),
            Res(10,10)
        )
        self.fc = torch.nn.Linear(10,1)
    def forward(self,x):
        out = self.model(x)
        return self.fc(out)

class lenthfactor():
    def __init__(self, bound, mu):
        self.bound = bound; self.dim = 2
        self.hx = self.bound[1]-self.bound[0]
        self.mu = mu
        
    def forward(self,X):
        return (X[:,0:1]-self.bound[0])/self.hx
    
def pred(netg, netf, lenth, X):
    return netg.forward(X) + lenth.forward(X) * netf.forward(X)

def error(u_pred, u_acc):
    return (((u_pred-u_acc)**2).sum() / (u_acc**2).sum()) ** (0.5)

# ----------------------------------------------------------------------------------------------------
def Lossg(netg,bdset):#拟合Dirichlet边界
    ub = netg.forward(bdset.DX)
    return bdset.Dlenth * ((ub - bdset.Dright)**2).sum()

def Lossf(netf,inset,bdset):
    inset.X.requires_grad = True
    insetF = netf.forward(inset.X)
    insetFx, = torch.autograd.grad(insetF, inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=torch.ones(inset.size,1))
    u_in = inset.G + inset.L * insetF#inset.G为netg在inset.X上取值，后面训练时提供，此举为加快迭代速度
    ux = inset.Gx + inset.Lx*insetF + inset.L*insetFx#复合函数求导，提高迭代效率

    temp = (inset.AM@ux.view(-1,inset.dim,1)).view(-1,inset.dim)

    ub = bdset.N_G + bdset.N_L * netf.forward(bdset.NX)

    return 0.5*inset.area * ((temp*ux).sum(1)).mean() \
           - inset.area * (inset.right*u_in).mean() \
           - bdset.Nlenth * (bdset.Nright*ub).mean()
def Traing(netg, bdset, optimg, epochg):
    print('train neural network g')
    lossg = Lossg(netg,bdset)
    lossbest = lossg
    print('epoch:%d,lossf:%.3e'%(0,lossg.item()))
    torch.save(netg.state_dict(),'best_netg.pkl')
    cycle = 100
    for i in range(epochg):
        st = time.time()
        '''
        for j in range(cycle):
            optimg.zero_grad()
            lossg = Lossg(netg,bdset)
            lossg.backward()
            optimg.step()
        '''
        def closure():
            optimg.zero_grad()
            lossg = Lossg(netg,bdset)
            lossg.backward()
            return lossg
        optimg.step(closure)
        
        lossg = Lossg(netg,bdset)
        if lossg < lossbest:
            lossbest = lossg
            torch.save(netg.state_dict(),'best_netg.pkl')
        ela = time.time() - st
        print('epoch:%d,lossg:%.3e,time:%.2f'%((i + 1)*cycle,lossg.item(),ela))

# Train neural network f
def Trainf(netf, inset, bdset, optimf, epochf):
    print('train neural network f')
    ERROR,BUZHOU = [],[]
    lossf = Lossf(netf,inset,bdset)
    lossoptimal = lossf
    trainerror = error(inset.G + inset.L * netf.forward(inset.X), inset.u_acc)
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
        trainerror = error(inset.G + inset.L * netf.forward(inset.X), inset.u_acc)
        ERROR.append(trainerror.item())
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,lossf:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,lossf.item(),trainerror,ela))
    return ERROR,BUZHOU

# Train neural network
def Train(netg, netf, lenth, inset, bdset, optimg, optimf, epochg, epochf):
    
    # Calculate the length factor
    inset.X.requires_grad = True
    inset.L = lenth.forward(inset.X)
    inset.Lx, = torch.autograd.grad(inset.L, inset.X,#计算长度因子关于内部点输入的梯度
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones(inset.size,1))
    bdset.N_L = lenth.forward(bdset.NX)#计算长度因子关于Neumann边界样本点的梯度

    inset.L = inset.L.data; inset.Lx = inset.Lx.data; bdset.N_L = bdset.N_L.data

    # Train neural network g
    Traing(netg, bdset, optimg, epochg)

    netg.load_state_dict(torch.load('best_netg.pkl'))
    inset.X.requires_grad = True
    inset.G = netg.forward(inset.X)
    inset.Gx, = torch.autograd.grad(inset.G, inset.X,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones(inset.size,1))
    bdset.N_G = netg.forward(bdset.NX)

    inset.G = inset.G.data; inset.Gx = inset.Gx.data; bdset.N_G = bdset.N_G.data

    # Train neural network f
    ERROR,BUZHOU = Trainf(netf, inset, bdset, optimf, epochf)
    return ERROR,BUZHOU


def main():
    
    # Configurations
    prob = 4
    bounds = torch.Tensor([-1.0,1.0,-1.0,1.0]).reshape(2,2)
    nx_tr = [60,60]#训练集剖分
    nx_te = [101,101]#测试集剖分
    epochg = 10
    epochf = 10
    lr = 0.01
    tests_num = 1

    # ------------------------------------------------------------------------------------------------
    testerror = torch.zeros(tests_num)
    for it in range(tests_num):

        inset = INSET(bounds, nx_tr, prob)
        bdset = BDSET(bounds, nx_tr, prob)
        teset = TESET(bounds, nx_te, prob)

        lenth = lenthfactor(bounds[0,:], 1)

        netg = NETG()
        netf = NETF()
        #optimg = torch.optim.Adam(netg.parameters(), lr=lr)
        #optimf = torch.optim.Adam(netf.parameters(), lr=lr)
        optimg = torch.optim.LBFGS(netg.parameters(), lr=lr,max_iter = 100,line_search_fn = 'strong_wolfe')
        optimf = torch.optim.LBFGS(netf.parameters(), lr=lr,max_iter = 100,line_search_fn = 'strong_wolfe')

        start_time = time.time()
        ERROR,BUZHOU = Train(netg, netf, lenth, inset, bdset, optimg, optimf, epochg, epochf)
        print(ERROR,BUZHOU)
        elapsed = time.time() - start_time
        print('Train time: %.2f' %(elapsed))

        netg.load_state_dict(torch.load('best_netg.pkl'))
        netf.load_state_dict(torch.load('best_netf.pkl'))
        te_U = pred(netg, netf, lenth, teset.X)
        testerror[it] = error(te_U, teset.u_acc)
        print('testerror = %.3e\n' %(testerror[it].item()))
    
    print(testerror.data)
    testerror_mean = testerror.mean()
    testerror_std = testerror.std()
    print('testerror_mean = %.3e, testerror_std = %.3e'
          %(testerror_mean.item(),testerror_std.item()))
    
if __name__ == '__main__':
    main()

