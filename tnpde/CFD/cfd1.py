import numpy as np
import time
import matplotlib.pyplot as plt
def UB(x):#函数初值
    n = x.shape[0]
    tmp = np.zeros([n,3])
    leq = (x < 0.3).astype('float32')
    geq = (x > 0.3).astype('float32')
    tmp[:,0] = leq*1 + geq*0.125
    tmp[:,1] = leq*0 + geq*0
    tmp[:,2] = leq*2.5 + geq*0.25
    return tmp

def matrix(U):#计算Roe格式的时候需要求矩阵的分解和绝对值组合
    gama = 1.4
    eps = 1e-7
    n = U.shape[1]
    R = np.ones([n,n]);L = np.zeros([n,n])
    a = np.sqrt(gama*(gama - 1)*(U[0,2]/(U[0,0] + eps) - 0.5*(U[0,1]**2/(U[0,0]**2 + eps))))
    u = U[0,1]/(U[0,0] + eps)
    H = (U[0,2] + (gama - 1)*(U[0,2] - 0.5*U[0,1]**2/(U[0,0] + eps)))/(U[0,0] + eps)
    R[1,0] = u - a;R[1,1] = u;R[1,2] = u + a
    R[2,0] = H - u*a;R[2,1] = 0.5*u**2;R[2,2] = H + u*a
    a31 = u*(u*a + u**2 - 2*H)/(4*a*H - 2*a*u**2)
    a32 = (2*H - u**2 - 2*a*u)/(4*a*H - 2*a*u**2)
    L[0,0] = u/a + a31;L[0,1] = a32 - 1/a;L[0,2] = 1/(2*H - u**2)
    L[1,0] = 1 - u/a - 2*a31;L[1,1] = 1/a - 2*a32;L[1,2] = 2/(u**2 - 2*H)
    L[2,0] = a31;L[2,1] = a32;L[2,2] = 1/(2*H - u**2)
    lam = np.array([u + a,u,u - a])
    return R@np.diag(abs(lam))@L
def FF(U):#计算源函数
    gama = 1.4
    eps = 1e-7
    n = U.shape[0]
    tmp = np.zeros([n,3])
    tmp[:,0] = U[:,1]
    tmp[:,1] = (gama - 1)*U[:,2] + 0.5*(3 - gama)*(U[:,1]**2)/(U[:,0] + eps)
    tmp[:,2] = gama*U[:,2]*U[:,1]/(U[:,0] + eps) + 0.5*(1 - gama)*(U[:,1]**3)/(U[:,0]**2 + eps)
    return tmp
def LAMmax(U):#计算最大特征值
    gama = 1.4
    eps = 1e-7
    a = np.sqrt(gama*(gama - 1)*(U[:,2]/(U[:,0] + eps) - 0.5*(U[:,1]**2/(U[:,0]**2 + eps))))
    tmp = abs(U[:,1]/(U[:,0] + eps)) + a
    
    return max(tmp)

def UU(tmp):#把守恒变量替换成原始变量
    n = tmp.shape[0]
    u = np.zeros([n,3])
    gama = 1.4
    eps = 1e-7
    u[:,0] = tmp[:,0]#rho
    u[:,1] = tmp[:,1]/(tmp[:,0] + eps)
    u[:,2] = (gama - 1)*(tmp[:,2] - 0.5*tmp[:,1]**2/(tmp[:,0] + eps))
    return u
class FD():
    def __init__(self,bound,nx):
        self.bound = bound#定义空间和时间的边界，比计算区域大
        self.hx = (bound[0,1] - bound[0,0])/(nx - 1)#定义步长
        self.X = np.linspace(bound[0,0],bound[0,1],nx)#包含计算区域
        self.nx = nx
        self.gama = 1.4
        self.cpu = [0,1]#空间计算区域
        self.x = np.arange(self.cpu[0],self.cpu[1] + self.hx,self.hx)#离散空间计算区域
        self.xL = np.arange(self.bound[0,0],self.cpu[0] + self.hx,self.hx)#把区域分成3份，这是左
        self.xR = np.arange(self.cpu[1],self.bound[0,1] + self.hx,self.hx)#右
    def solve(self,mode):
        U_old = UB(self.X)
        #print(U_old.shape)
        U_new = np.zeros_like(U_old)
        cfl = 0.9
        #---------------------------------------------------
        if mode == 'LF':
            m = 0
            t = self.bound[1,0]
            while m < 100:
                tau = cfl*self.hx/LAMmax(U_old)
                U_new[0,:] = U_old[1,:]
                for i in range(1,U_old.shape[0] - 1):
                    U_new[i:i + 1,:] = 0.5*(U_old[i - 1:i,:] + U_old[i + 1:i + 2,:]) - \
                    0.5*tau*(FF(U_old[i + 1:i + 2,:]) - FF(U_old[i - 1:i,:]))/self.hx
                U_new[-1,:] = U_old[-2,:]
                if t <= self.bound[1,1] and t + tau >= self.bound[1,1]:
                    tau = self.bound[1,1] - t
                    U_new[0,:] = U_old[1,:]
                    for i in range(1,U_old.shape[0] - 1):
                        U_new[i:i + 1,:] = 0.5*(U_old[i - 1:i,:] + U_old[i + 1:i + 2,:]) - \
                        0.5*tau*(FF(U_old[i + 1:i + 2,:]) - FF(U_old[i - 1:i,:]))/self.hx
                    U_new[-1,:] = U_old[-2,:]
                    t += tau
                    break
                else:
                    t += tau
                    U_old = U_new
                    m = m + 1
            
            print('the mode:%s,the epoch:%d,the problem:%d,the time:%.2f'%('LF',m,2,t))
            return U_new[self.xL.size - 1:self.xL.size + self.x.size - 1,:]
        if mode == 'Mac':
            cfl = 0.9
            m = 0
            t = self.bound[1,0]
            while m < 100:
                eig = LAMmax(U_old)
                
                tau = self.hx*cfl/eig
                W0 = U_old[1:2,:] - tau*(FF(U_old[0:1,:]) - FF(U_old[1:2,:]))/self.hx
                W1 = U_old[0:1,:] - tau*(FF(U_old[1:2,:]) - FF(U_old[0:1,:]))/self.hx
                U_new[0:1] = 0.5*(U_old[0:1] + W1) - 0.5*tau*(FF(W1) - FF(W0))/self.hx
                for i in range(1,U_old.shape[0] - 1):
                    W0 = U_old[i - 1:i,:] - tau*(FF(U_old[i:i + 1,:]) - FF(U_old[i - 1:i,:]))/self.hx
                    W1 = U_old[i:i + 1,:] - tau*(FF(U_old[i + 1:i + 2,:]) - FF(U_old[i:i + 1,:]))/self.hx
                    U_new[i:i + 1,:] = 0.5*(U_old[i:i + 1,:] + W1) - \
                    0.5*tau*(FF(W1) - FF(W0))/self.hx
                W0 = U_old[-2:-1,:] - tau*(FF(U_old[-1:,:]) - FF(U_old[-2:-1,:]))/self.hx
                W1 = U_old[-1:,:] - tau*(FF(U_old[-2:-1,:]) - FF(U_old[-1:,:]))/self.hx
                
                U_new[-1:,:] = 0.5*(U_old[-1:,:] + W1) - 0.5*tau*(FF(W1) - FF(W0))/self.hx
                if t <= self.bound[1,1] and t + tau >= self.bound[1,1]:
                    tau = self.bound[1,1] - t
                    W0 = U_old[1:2,:] - tau*(FF(U_old[0:1,:]) - FF(U_old[1:2,:]))/self.hx
                    W1 = U_old[0:1,:] - tau*(FF(U_old[1:2,:]) - FF(U_old[0:1,:]))/self.hx
                    U_new[0:1] = 0.5*(U_old[0:1] + W1) - 0.5*tau*(FF(W1) - FF(W0))/self.hx
                    for i in range(1,U_old.shape[0] - 1):
                        W0 = U_old[i - 1:i,:] - tau*(FF(U_old[i:i + 1,:]) - FF(U_old[i - 1:i,:]))/self.hx
                        W1 = U_old[i:i + 1,:] - tau*(FF(U_old[i + 1:i + 2,:]) - FF(U_old[i:i + 1,:]))/self.hx
                        U_new[i:i + 1,:] = 0.5*(U_old[i:i + 1,:] + W1) - \
                        0.5*tau*(FF(W1) - FF(W0))/self.hx
                    W0 = U_old[-2:-1,:] - tau*(FF(U_old[-1:,:]) - FF(U_old[-2:-1,:]))/self.hx
                    W1 = U_old[-1:,:] - tau*(FF(U_old[-2:-1,:]) - FF(U_old[-1:,:]))/self.hx
                    U_new[-1:,:] = 0.5*(U_old[-1:,:] + W1) - 0.5*tau*(FF(W1) - FF(W0))/self.hx
                    t += tau
                    break
                else:
                    t += tau
                    U_old = U_new
                    m = m + 1
            print('the mode:%s,the epoch:%d,the problem:%d,the time:%.2f'%('Mac',m,2,t))
            return U_new[self.xL.size - 1:self.xL.size + self.x.size - 1,:]
        if mode == 'Roe':
            cfl = 0.9
            m = 0
            t = self.bound[1,0]
            U_new = U_old.copy()
            while m < 100:
                eig = LAMmax(U_old)
                tau = self.hx*cfl/eig
                mat = matrix(U_old[0:1,:]) + matrix(U_old[1:2,:])
                U_new[0:1,:] = U_old[0:1,:] - 0.5*tau*(U_old[0:1,:] - U_old[1:2,:])@mat.T/self.hx
                for i in range(1,U_old.shape[0] - 1):
                    
                    W0 = 0.5*(FF(U_old[i - 1:i,:]) + FF(U_old[i:i + 1,:])) - \
                    0.5*(U_old[i:i + 1,:] - U_old[i - 1:i,:])@matrix(U_old[i - 1:i,:]).T
                    W1 = 0.5*(FF(U_old[i:i + 1,:]) + FF(U_old[i + 1:i + 2,:])) + \
                    0.5*(U_old[i + 1:i + 2,:] - U_old[i:i + 1,:])@matrix(U_old[i:i + 1,:]).T
                    U_new[i:i + 1,:] = U_old[i:i + 1,:] - tau*(W1 - W0)/self.hx
                
                mat = matrix(U_old[-1:,:]) + matrix(U_old[-2:-1,:])
                U_new[-1:,:] = U_old[-1:,:] - 0.5*tau*(U_old[-1:,:] - U_old[-2:-1,:])@mat.T/self.hx
                if t <= self.bound[1,1] and t + tau >= self.bound[1,1]:
                    tau = self.bound[1,1] - t
                    mat = matrix(U_old[0:1,:]) + matrix(U_old[1:2,:])
                    U_new[0:1,:] = U_old[0:1,:] - 0.5*tau*(U_old[0:1,:] - U_old[1:2,:])@mat.T/self.hx
                    for i in range(1,U_old.shape[0] - 1):
                        W0 = 0.5*(FF(U_old[i - 1:i,:]) + FF(U_old[i:i + 1,:])) - \
                        0.5*(U_old[i:i + 1,:] - U_old[i - 1:i,:])@matrix(U_old[i - 1:i,:]).T
                        W1 = 0.5*(FF(U_old[i:i + 1,:]) + FF(U_old[i + 1:i + 2,:])) + \
                        0.5*(U_old[i + 1:i + 2,:] - U_old[i:i + 1,:])@matrix(U_old[i:i + 1,:]).T
                        U_new[i:i + 1,:] = U_old[i:i + 1,:] - tau*(W1 - W0)/self.hx
                    mat = matrix(U_old[-1:,:]) + matrix(U_old[-2:-1,:])
                    U_new[-1:,:] = U_old[-1:,:] - 0.5*tau*(U_old[-1:,:] - U_old[-2:-1,:])@mat.T/self.hx
                    t += tau
                    break
                else:
                    t += tau
                    m = m + 1
                    U_new = U_old
            print('the mode:%s,the epoch:%d,the problem:%d,the time:%.2f'%('Roe',m,2,t))
            return U_new[self.xL.size - 1:self.xL.size + self.x.size - 1,:]
        
bound = np.array([[-5,5],[0,0.2]])
nx = 501
mode = 'Roe'
fd = FD(bound,nx)

u_pred = UU(fd.solve(mode))
#plt.plot(fd.x,u_pred[:,0],'r*')
plt.plot(fd.x,u_pred[:,0],'r',label = 'rho')
#plt.plot(fd.x,u_pred[:,1],'bo')
plt.plot(fd.x,u_pred[:,1],'b',label = 'u')
#plt.plot(fd.x,u_pred[:,2],'k.')
plt.plot(fd.x,u_pred[:,2],'k',label = 'P')
plt.xlabel('the space:[%.2f,%.2f]'%(fd.cpu[0],fd.cpu[1]))
plt.ylabel('the numerical solutions')
plt.title('the method :%s, at time:%.2f'%(mode,bound[1,1]))
plt.legend(loc = 'upper right')
plt.savefig('%s1.jpg'%(mode))

