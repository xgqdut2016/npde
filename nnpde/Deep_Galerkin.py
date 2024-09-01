import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

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
        #self.area = (bound[0,1] - bound[0,0])*(bound[1,1] - bound[1,0])
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        self.size = (nx[0] - 1)*(nx[1] - 1)
        self.nx = nx
        self.bound = bound
        
        self.Node()#生成网格点，这里对应基函数
        self.cai()#采集样本点x，以及给出对于内网格点的索引，方便取出内网格点对应的基函数
        self.test()#取出测试函数在不为0的区域的取值以及对应的偏导数值
        self.u_acc = torch.zeros(self.X.shape[0],self.X.shape[1],1)
        self.right = torch.zeros(self.X.shape[0],self.X.shape[1],1)
        for i in range(self.X.shape[0]):
            self.u_acc[i,:,0] = UU(self.X[i,:,:],[0,0],prob)
            self.right[i,:,0] = FF(self.X[i,:,:],prob)
    def Node(self):#生成网格点(M + 1)*(N + 1)，注意这个和X不是同一个
        self.Nodes_size = (self.nx[0] + 1)*(self.nx[1] + 1)
        self.Nodes = torch.zeros(self.Nodes_size,self.dim)
        m = 0
        for i in range(self.nx[0] + 1):
            for j in range(self.nx[1] + 1):
                self.Nodes[m,0] = self.bound[0,0] + i*self.hx[0]
                self.Nodes[m,1] = self.bound[1,0] + j*self.hx[1]
                m = m + 1
    def cai(self):
        self.index = np.zeros([(self.nx[0] - 1)*(self.nx[1] - 1)],np.int)#这个就是内网格点的索引
        m = 0
        for i in range(1,self.nx[0]):
            for j in range(1,self.nx[1]):#内网格点的数目为(nx[0] - 1)*(nx[1] - 1)
                self.index[m] = i*(self.nx[1] + 1) + j
                m = m + 1
        #ind = self.index
        #plt.scatter(self.Nodes[ind,0],self.Nodes[ind,1])，这行命令可以检查是否取得是内网格点
        self.X = torch.zeros((self.nx[0] - 1)*(self.nx[1] - 1),4,self.dim)
        #以内网格点为中心对应的基函数，周围4个单元取值非0，其中在每个单元采集一个点
        m = 0
        for i in range(1,self.nx[0]):
            for j in range(1,self.nx[1]):
                self.X[m,0,0] = self.bound[0,0] + (i - 0.5)*self.hx[0]
                self.X[m,0,1] = self.bound[1,0] + (j - 0.5)*self.hx[1]
                self.X[m,1,0] = self.bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[m,1,1] = self.bound[1,0] + (j - 0.5)*self.hx[1]
                self.X[m,2,0] = self.bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[m,2,1] = self.bound[1,0] + (j + 0.5)*self.hx[1]
                self.X[m,3,0] = self.bound[0,0] + (i - 0.5)*self.hx[0]
                self.X[m,3,1] = self.bound[1,0] + (j + 0.5)*self.hx[1]
                m = m + 1
    def phi(self,X,order):#[-1,1]*[-1,1]，在原点取值为1，其他网格点取值为0的基函数
        ind00 = (X[:,0] >= -1);ind01 = (X[:,0] >= 0);ind02 = (X[:,0] >= 1)
        ind10 = (X[:,1] >= -1);ind11 = (X[:,1] >= 0);ind12 = (X[:,1] >= 1)
        if order == [0,0]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,0])*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(1 - X[:,0])*(1 + X[:,1]) + \
                    (ind00*~ind01*ind11*~ind12).float()*(1 + X[:,0])*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).float()*(1 - X[:,0])*(1 - X[:,1])
        if order == [1,0]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(-(1 + X[:,1])) + \
                    (ind00*~ind01*ind11*~ind12).float()*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).float()*(-(1 - X[:,1]))
        if order == [0,1]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,0]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(1 - X[:,0]) + \
                    (ind00*~ind01*ind11*~ind12).float()*(-(1 + X[:,0])) + \
                    (ind01*~ind02*ind11*~ind12).float()*(-(1 - X[:,0]))
    def basic(self,X,order,i):#根据网格点的存储顺序，遍历所有网格点，取基函数
        temp = (X - self.Nodes[i,:])/torch.tensor([self.hx[0],self.hx[1]])
        if order == [0,0]:
            return self.phi(temp,order)
        if order == [1,0]:
            return self.phi(temp,order)/self.hx[0]
        if order == [0,1]:
            return self.phi(temp,order)/self.hx[1]
    def test(self):
        self.v = torch.zeros(self.size,4,1)
        #定义测试函数在所有内点的取值，一共有(nx[0] - 1)*(nx[1] - 1)个测试函数，每个测试函数只有4个点上取值非0
        #print(self.v[0,:,0:1].shape,self.basic(self.X[0,:,:],[0,0],self.index[0]).shape)
        for i in range(self.size):
            self.v[i,:,0:1] = self.basic(self.X[i,:,:],[0,0],self.index[i]).view(-1,1)
        self.vx = torch.zeros(self.size,4,self.dim)#定义基函数在对应区域的偏导数
        for i in range(self.size):
            self.vx[i,:,0] = self.basic(self.X[i,:,:],[1,0],self.index[i])
            self.vx[i,:,1] = self.basic(self.X[i,:,:],[0,1],self.index[i])
        

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
class NETG(nn.Module):#u = netf*lenthfactor + netg，此为netg
    def __init__(self):
        super(NETG,self).__init__()
        self.fc1 = torch.nn.Linear(2,10)
        self.fc2 = torch.nn.Linear(10,10)
        self.fc3 = torch.nn.Linear(10,1)
    def forward(self,x):
        out = torch.sin(self.fc1(x))
        out = torch.sin(self.fc2(out)) + x@torch.eye(x.size(-1),10)
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
        x = self.model(x) + x@torch.eye(x.size(-1),self.output)#模拟残差网络
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
    def __init__(self,bound,mu):
        self.mu = mu
        self.bound = bound
        self.dx = bound[:,1] - bound[:,0]
    def forward(self,X):
        L = 1.0
        if X.ndim == 2:#如果是边界点，存储方式为[m,2]
            for i in range(2):
                L = L*(1 - (1 - (X[:,i] - self.bound[i,0])/self.dx[i])**self.mu)
                L = L*(1 - (1 - (self.bound[i,1] - X[:,i])/self.dx[i])**self.mu)
            return L.view(-1,1)
        elif X.ndim == 3:#如果是内点，存储方式为[m,4,2],m表示内网格点数目
            for i in range(2):
                L = L*(1 - (1 - (X[:,:,i] - self.bound[i,0])/self.dx[i])**self.mu)
                L = L*(1 - (1 - (self.bound[i,1] - X[:,:,i])/self.dx[i])**self.mu)
            return L.view(X.shape[0],X.shape[1],1)  
def pred(netg, netf, lenth, X):
    return netg.forward(X) + lenth.forward(X) * netf.forward(X)

def error(u_pred, u_acc):
    return max(abs(u_pred.reshape(-1,1) - u_acc.reshape(-1,1)))
    #return (((u_pred-u_acc)**2).sum() / (u_acc**2).sum()) ** (0.5)

# ----------------------------------------------------------------------------------------------------
def Lossg(netg,bdset):#拟合Dirichlet边界
    ub = netg.forward(bdset.X)
    return bdset.Dlenth * ((ub - bdset.Dright)**2).sum()

def Lossf(netf,inset):
    inset.X.requires_grad = True
    insetF = netf.forward(inset.X)#(81,4,1)
    insetFx, = torch.autograd.grad(insetF, inset.X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(inset.size,4,1))#(81,4,2)
    
   
    ux = inset.Gx + inset.Lx*insetF + inset.L*insetFx#复合函数求导，提高迭代效率
    out = (inset.vx*ux).sum(2) - (inset.v*inset.right).sum(2)#inset.vx=[m,4,2],inset.right = [m,4,1]
    out = out.sum(1)
    #这个就是所有测试函数与MLP形成的损失函数，其中积分就是在对应内网格点得到的测试函数在周围4个单元的的取值
    return (out**2).mean()

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
    lossf = Lossf(netf,inset)
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
            lossf = Lossf(netf,inset)
            lossf.backward()
            return lossf
        optimf.step(closure)
        lossf = Lossf(netf,inset)

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
                                    grad_outputs=torch.ones(inset.size,4,1))
    

    inset.L = inset.L.data; inset.Lx = inset.Lx.data

    # Train neural network g
    Traing(netg, bdset, optimg, epochg)

    netg.load_state_dict(torch.load('best_netg.pkl'))
    inset.X.requires_grad = True
    inset.G = netg.forward(inset.X)
    
    inset.Gx, = torch.autograd.grad(inset.G, inset.X,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones(inset.size,4,1))
    

    inset.G = inset.G.data; inset.Gx = inset.Gx.data

    # Train neural network f
    ERROR,BUZHOU = Trainf(netf, inset, bdset, optimf, epochf)
    return ERROR,BUZHOU


def main():
    
    # Configurations
    prob = 1
    bounds = torch.Tensor([-1.0,1.0,-1.0,1.0]).reshape(2,2)
    nx_tr = [50,50]#训练集剖分
    nx_te = [101,101]#测试集剖分
    epochg = 10
    epochf = 10
    lr = 1e-2
    tests_num = 1

    # ------------------------------------------------------------------------------------------------
    testerror = torch.zeros(tests_num)
    for it in range(tests_num):

        inset = INSET(bounds, nx_tr, prob)
        bdset = BDSET(bounds, nx_tr, prob)
        teset = TESET(bounds, nx_te, prob)

        lenth = lenthfactor(bounds, 1)

        netg = NETG()
        netf = NETF()
        #optimg = torch.optim.Adam(netg.parameters(), lr=lr)
        #optimf = torch.optim.Adam(netf.parameters(), lr=lr)
        optimg = torch.optim.LBFGS(netg.parameters(), lr=lr,max_iter = 100,line_search_fn = 'strong_wolfe')
        optimf = torch.optim.LBFGS(netf.parameters(), lr=lr,max_iter = 100,line_search_fn = 'strong_wolfe')


        start_time = time.time()
        ERROR,BUZHOU = Train(netg, netf, lenth, inset, bdset, optimg, optimf, epochg, epochf)
        #print(ERROR,BUZHOU)
        elapsed = time.time() - start_time
        print('Train time: %.2f' %(elapsed))

        netg.load_state_dict(torch.load('best_netg.pkl'))
        netf.load_state_dict(torch.load('best_netf.pkl'))
        te_U = pred(netg, netf, lenth, teset.X)
        testerror[it] = error(te_U, teset.u_acc)
        print('testerror = %.3e\n' %(testerror[it].item()))
        plt.plot(BUZHOU,ERROR,'r*')
        plt.plot(BUZHOU,ERROR)
        plt.title('the absolute error for example of prob = %d at iteration'%(prob))
        plt.savefig('jvxing.jpg')
    print(testerror.data)
    testerror_mean = testerror.mean()
    testerror_std = testerror.std()
    print('testerror_mean = %.3e, testerror_std = %.3e'
          %(testerror_mean.item(),testerror_std.item()))
    
if __name__ == '__main__':
    main()

