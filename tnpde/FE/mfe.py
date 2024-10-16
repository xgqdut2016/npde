import matplotlib.pyplot as plt
import numpy as np
import time


def UU(X, order,prob):#X表示(x,t)
    if prob==1:
        temp = 10*(X[:,0]+X[:,1])**2 + (X[:,0]-X[:,1])**2 + 0.5
        if order[0]==0 and order[1]==0:
            return np.log(temp)
        if order[0]==1 and order[1]==0:#对x求偏导
            return temp**(-1) * (20*(X[:,0]+X[:,1]) + 2*(X[:,0]-X[:,1]))
        if order[0]==0 and order[1]==1:#对t求偏导
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
                   0.5*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
        if order[0]==1 and order[1]==0:
            return (3*X[:,0]*X[:,0]-1) * \
                   0.5*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
        if order[0]==0 and order[1]==1:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   (np.exp(2*X[:,1])-np.exp(-2*X[:,1]))
        if order[0]==2 and order[1]==0:
            return (6*X[:,0]) * \
                   0.5*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
        if order[0]==1 and order[1]==1:
            return (3*X[:,0]*X[:,0]-1) * \
                   (np.exp(2*X[:,1])-np.exp(-2*X[:,1]))
        if order[0]==0 and order[1]==2:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   2*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
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
def FF(prob,X):
    return -UU(X,[0,2],prob) - UU(X,[2,0],prob)
def NEU(X,prob,n,beta):#定义g(x),beta > 0
    return UU(X,[1,0],prob)*n[:,0] + UU(X,[0,1],prob)*n[:,1] + beta*UU(X,[0,0],prob)
np.random.seed(1234)


class TESET():
    def __init__(self, bounds, nx):
        self.bounds = bounds
        self.nx = nx
        self.hx = [(self.bounds[0,1]-self.bounds[0,0])/self.nx[0],
                   (self.bounds[1,1]-self.bounds[1,0])/self.nx[1]]
        self.DS = nx[0] + 1
        self.DX = np.zeros([self.DS,2])
        for i in range(nx[0] + 1):
            self.DX[i,0] = bounds[0,0] + i*self.hx[0]
            self.DX[i,1] = bounds[1,0]
            
        self.NS = 2*(nx[1] + 1) + nx[0] + 1
        self.NX = np.zeros([self.NS,2])
        m = 0
        for j in range(nx[1] + 1):
            self.NX[m,0] = bounds[0,0]
            self.NX[m,1] = bounds[1,0] + j*self.hx[1]
            m = m + 1
        for i in range(nx[0] + 1):
            self.NX[m,0] = bounds[0,0] + i*self.hx[0]
            self.NX[m,1] = bounds[1,1]
            m = m + 1
        for j in range(nx[1] + 1):
            self.NX[m,0] = bounds[0,1]
            self.NX[m,1] = bounds[1,0] + j*self.hx[1]
            m = m + 1
        self.size = (nx[0] + 1)*(nx[1] + 1)
        self.X = np.zeros([self.size,2])
        for i in range(nx[0] + 1):
            for j in range(nx[1] + 1):
                self.X[i*(nx[1] + 1) + j,0] = bounds[0,0] + i*self.hx[0]
                self.X[i*(nx[1] + 1) + j,1] = bounds[1,0] + j*self.hx[1]


class FENET():
    def __init__(self,bounds,nx):
        self.dim = 2
        self.bounds = bounds
        self.hx = [(bounds[0,1] - bounds[0,0])/nx[0],
                  (bounds[1,1] - bounds[1,0])/nx[1]]
        self.nx = nx
        self.gp_num = 2
        self.gp_pos = [(1 - np.sqrt(3)/3)/2,(1 + np.sqrt(3)/3)/2]
        self.Node()
        self.Unit()
        
    def Node(self):#生成网格点(M + 1)*(N + 1)
        self.Nodes_size = (self.nx[0] + 1)*(self.nx[1] + 1)
        self.Nodes = np.zeros([self.Nodes_size,self.dim])
        m = 0
        for i in range(self.nx[0] + 1):
            for j in range(self.nx[1] + 1):
                self.Nodes[m,0] = self.bounds[0,0] + i*self.hx[0]
                self.Nodes[m,1] = self.bounds[1,0] + j*self.hx[1]
                m = m + 1
    def phi(self,X,order):#[-1,1]*[-1,1]，在原点取值为1，其他网格点取值为0的基函数
        ind00 = (X[:,0] >= -1);ind01 = (X[:,0] >= 0);ind02 = (X[:,0] >= 1)
        ind10 = (X[:,1] >= -1);ind11 = (X[:,1] >= 0);ind12 = (X[:,1] >= 1)
        if order == [0,0]:
            return (ind00*~ind01*ind10*~ind11).astype('float32')*(1 + X[:,0])*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).astype('float32')*(1 - X[:,0])*(1 + X[:,1]) + \
                    (ind00*~ind01*ind11*~ind12).astype('float32')*(1 + X[:,0])*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).astype('float32')*(1 - X[:,0])*(1 - X[:,1])
        if order == [1,0]:
            return (ind00*~ind01*ind10*~ind11).astype('float32')*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).astype('float32')*(-(1 + X[:,1])) + \
                    (ind00*~ind01*ind11*~ind12).astype('float32')*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).astype('float32')*(-(1 - X[:,1]))
        if order == [0,1]:
            return (ind00*~ind01*ind10*~ind11).astype('float32')*(1 + X[:,0]) + \
                    (ind01*~ind02*ind10*~ind11).astype('float32')*(1 - X[:,0]) + \
                    (ind00*~ind01*ind11*~ind12).astype('float32')*(-(1 + X[:,0])) + \
                    (ind01*~ind02*ind11*~ind12).astype('float32')*(-(1 - X[:,0]))
    def basic(self,X,order,i):#根据网格点的存储顺序，遍历所有网格点，取基函数
        temp = (X - self.Nodes[i,:])/np.array([self.hx[0],self.hx[1]])
        if order == [0,0]:
            return self.phi(temp,order)
        if order == [1,0]:
            return self.phi(temp,order)/self.hx[0]
        if order == [0,1]:
            return self.phi(temp,order)/self.hx[1]
    def Unit(self):#生成所有单元，单元数目(M*N)
        self.Units_size = self.nx[0]*self.nx[1]
        self.Units_Nodes = np.zeros([self.Units_size,4],np.int)#每个单元有4个点，记录这4个点的整体编号
        self.Units_Int_Points = np.zeros([self.Units_size,#划分成M*N个小区域，每个区域有4个高斯积分点
                                          self.gp_num*self.gp_num,self.dim])
        
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.Units_Nodes[m,0] = i*(self.nx[1] + 1) + j
                self.Units_Nodes[m,1] = i*(self.nx[1] + 1) + j + 1
                self.Units_Nodes[m,2] = (i + 1)*(self.nx[1] + 1) + j
                self.Units_Nodes[m,3] = (i + 1)*(self.nx[1] + 1) + j + 1
                n = 0
                for k in range(self.gp_num):
                    for l in range(self.gp_num):
                        self.Units_Int_Points[m,n,0] = \
                            self.bounds[0,0] + (i + self.gp_pos[k])*self.hx[0]
                        self.Units_Int_Points[m,n,1] = \
                            self.bounds[1,0] + (j + self.gp_pos[l])*self.hx[1]
                        n = n + 1
                m = m + 1
        #下面开始采集neumann边界上的高斯积分点
        self.Neu_size = 2*self.nx[1] + self.nx[0]#Neumann边界上的线段数目
        self.Neu_Nodes = np.zeros([self.Neu_size,2],np.int)#存储每一个线段的两个端点序号
        self.Units_Bou_Points = np.zeros([self.Neu_size,self.gp_num,self.dim])#存储线段上的两个积分点坐标
        self.dir = np.zeros([self.Neu_size,self.gp_num,self.dim])#存储Neumann边界上的法方向
        self.area = np.zeros([self.Neu_size,1])#存储Neumann边界上单元的长度
        m = 0
        for i in range(self.nx[1]):
            self.Neu_Nodes[m,0] = i
            self.Neu_Nodes[m,1] = i + 1
            self.area[m,0] = self.hx[1]
            for k in range(self.gp_num):
                self.Units_Bou_Points[m,k,0] = self.bounds[0,0]
                self.Units_Bou_Points[m,k,1] = self.bounds[1,0] + (i + self.gp_pos[k])*self.hx[1]
                self.dir[m,k,0] = -1.0;self.dir[m,k,1] = 0.0
                
            m = m + 1
        
        for i in range(self.nx[0]):
            self.Neu_Nodes[m,0] = (i + 1)*(self.nx[1] + 1) - 1
            self.Neu_Nodes[m,1] = (i + 2)*(self.nx[1] + 1) - 1
            self.area[m,0] = self.hx[0]
           
            for k in range(self.gp_num):
                self.Units_Bou_Points[m,k,0] = self.bounds[0,0] + (i + self.gp_pos[k])*self.hx[0]
                self.Units_Bou_Points[m,k,1] = self.bounds[1,1]
                self.dir[m,k,0] = 0.0;self.dir[m,k,1] = 1.0
                
            m = m + 1
        
        for i in range(self.nx[1]):
            self.Neu_Nodes[m,0] = self.nx[0]*(self.nx[1] + 1) - 1 + (self.nx[1] + 1 - i)
            self.Neu_Nodes[m,1] = self.nx[0]*(self.nx[1] + 1) - 1 + (self.nx[1] + 1 - i - 1)
            self.area[m,0] = self.hx[1]
            for k in range(self.gp_num):
                self.Units_Bou_Points[m,k,0] = self.bounds[0,1]
                self.Units_Bou_Points[m,k,1] = self.bounds[1,0] + (self.nx[1] - 1 - i + self.gp_pos[k])*self.hx[1]
                self.dir[m,k,0] = 1.0;self.dir[m,k,1] = 0.0
                
            m = m + 1
    def Int_basic_basic(self,i,j,u_ind):#表示第i,j个基函数的梯度的乘积，以及在第u_ind个区域的积分
        X = self.Units_Int_Points[u_ind,:,:]#[4,2]
        basic0 = np.zeros_like(X)
        basic0[:,0] = self.basic(X,[1,0],i)
        basic0[:,1] = self.basic(X,[0,1],i)
        
        basic1 = np.zeros_like(X)
        basic1[:,0] = self.basic(X,[1,0],j)
        basic1[:,1] = self.basic(X,[0,1],j)
        return ((basic0*basic1).sum(1)).mean()*self.hx[0]*self.hx[1]
    def Int_F_basic(self,i,u_ind,prob):#第i个基函数与右端项乘积，在第u_ind个单元积分
        X = self.Units_Int_Points[u_ind,:,:]
        return (FF(prob,X)*self.basic(X,[0,0],i)).mean()*self.hx[0]*self.hx[1]
    def Bou_basic_basic(self,i,j,u_ind,beta):#第i,j个基函数在第u_ind个Neumann边界上的乘积
        X = self.Units_Bou_Points[u_ind,:,:]#[2,2]
        basic0 = self.basic(X,[0,0],i)
        
        basic1 = self.basic(X,[0,0],j)
        area = self.area[u_ind,0]
        return beta*(basic0*basic1).mean()*area
    def Bou_Neu_basic(self,j,u_ind,prob,beta):
        X = self.Units_Bou_Points[u_ind,:,:]#形状为[2,2],Neumannn边界上的高斯积分点
        area = self.area[u_ind,0]
        n = self.dir[u_ind,:,:]#形状为[2,2],Neumannn边界上的高斯积分点的法方向
        basic = self.basic(X,[0,0],j)
        g = NEU(X,prob,n,beta)
        return (g*basic).mean()*area
    def matrix(self,beta):
        A = np.zeros([self.Nodes_size,self.Nodes_size])#self.Nodes_size = (M+ 1)*(N + 1)
        for m in range(self.Units_size):#self.Units_size = M*N，第m个区域单元的4个积分点
            for k in range(4):
                ind0 = self.Units_Nodes[m,k]#self.Units_Nodes = [M*N,4],第m个区域中第k个网格点，第k个基函数编号
                for l in range(4):
                    ind1 = self.Units_Nodes[m,l]#self.Units_Nodes = [M*N,4],第m个区域中第l网格点，第k个基函数编号
                    #第m个区域上，两个基函数梯度的乘积的积分a(u,v)
                    A[ind0,ind1] += self.Int_basic_basic(ind0,ind1,m)
        for m in range(self.Neu_size):
            for k in range(2):
                ind0 = self.Neu_Nodes[m,k]
                for l in range(2):
                    ind1 = self.Neu_Nodes[m,l]
                    A[ind0,ind1] += self.Bou_basic_basic(ind0,ind1,m,beta)
        for i in range(self.nx[0] + 1):
            ind = i*(self.nx[1] + 1)
            A[ind,:] = np.zeros([1,self.Nodes_size])
            A[ind,ind] = 1.0
        return A
    def right(self,prob,beta):
        b = np.zeros([self.Nodes_size,1])#self.Nodes_size = (M + 1)*(N + 1)
        for m in range(self.Units_size):#self.Units_size = M*N
            for k in range(4):
                ind = self.Units_Nodes[m,k]#第m个单元区域内第k个网格点，第k个基函数的编号
                b[ind] += self.Int_F_basic(ind,m,prob)
        for m in range(self.Neu_size):
            for k in range(2):
                ind = self.Neu_Nodes[m,k]
                b[ind] += self.Bou_Neu_basic(ind,m,prob,beta)
        for i in range(self.nx[0] + 1):
            ind = i*(self.nx[1] + 1)
            b[ind,0] = UU(self.Nodes[ind:ind + 1,:],[0,0],prob)
        return b
    def Uh(self,X,prob,beta):
        uh = np.zeros(X.shape[0])
        A = self.matrix(beta)
        b = self.right(prob,beta)
        Nodes_V = np.linalg.solve(A,b)
        for i in range(self.Nodes_size):#self.Nodes_size = (M + 1)*(N + 1)
            # 计算数据集 关于 基函数中心 的相对位置
            uh += Nodes_V[i]*self.basic(X,[0,0],i)
        return uh.reshape(-1,1)
st = time.time()
bounds = np.array([[0,1.0],[0,1.0]])
nx_tr = [10,10]    # 网格大小
fenet = FENET(bounds,nx_tr)

nx_te = [80,40]
teset = TESET(bounds,nx_te)
X = teset.X
prob = 1
beta = 1
u_acc = UU(X,[0,0],prob).reshape(-1,1)
u_pred = fenet.Uh(X,prob,beta)
error = max(abs(u_acc - u_pred))
x_train = np.linspace(bounds[0,0],bounds[0,1],nx_te[0] + 1)
y_train = np.linspace(bounds[1,0],bounds[1,1],nx_te[1] + 1)
x,y = np.meshgrid(x_train,y_train)
plt.contourf(x,y,u_pred.reshape(nx_te[0] + 1,nx_te[1] + 1).T,40,cmap = 'Blues')
#plt.contourf(x,y,u_acc.reshape(nx_te[0] + 1,nx_te[1] + 1).T,40,cmap = 'Blues')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('prob = %d ,the predict solution at grid = %d * %d'%(prob,nx_te[1] + 1,nx_te[0] + 1))
ela = time.time() - st

print('the error at %d * %d of fenet = %.3e,time:%.3f'%(nx_te[0],nx_te[1],error,ela))
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(fenet.Nodes[:,0],fenet.Nodes[:,1])
ax.scatter(fenet.Units_Int_Points[:,:,0],fenet.Units_Int_Points[:,:,1])

for i in range(fenet.nx[0] + 1):
    plt.plot(fenet.Nodes[i*(fenet.nx[1] + 1):(i + 1)*(fenet.nx[1] + 1),0],fenet.Nodes[i*(fenet.nx[1] + 1):(i + 1)*(fenet.nx[1] + 1),1])
x = np.linspace(bounds[0,0],bounds[0,1],nx_tr[0] + 1)
y = np.linspace(bounds[1,0],bounds[1,1],nx_tr[1] + 1)
for i in range(nx_tr[1] + 1):
    plt.plot(x,y[i]*np.ones_like(x))
plt.savefig('gauss.png')


