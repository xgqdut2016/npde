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
def FF(prob,X):
    return -UU(X,[0,2],prob) - UU(X,[2,0],prob)

class FD():
    def __init__(self,bound,hx,prob):
        self.prob = prob
        self.dim = 2
        self.hx = hx
        self.nx = [int((bound[0,1] - bound[0,0])/self.hx[0]) + 1,int((bound[1,1] - bound[1,0])/self.hx[1]) + 1]
        self.size = self.nx[0]*self.nx[1]
        self.X = np.zeros([self.size,self.dim])
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.X[m,0] = bound[0,0] + i*self.hx[0]
                self.X[m,1] = bound[1,0] + j*self.hx[1]
                m = m + 1
        self.u_acc = UU(self.X,[0,0],self.prob).reshape(-1,1)
    def matrix(self):
        self.A = np.zeros([self.nx[0]*self.nx[1],self.nx[0]*self.nx[1]])
        dx = self.hx[0];dy = self.hx[1]
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                dx = self.hx[0];dy = self.hx[1]
                if i== 0 or i == self.nx[0] - 1 or j == 0 or j == self.nx[1] - 1:
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 1
                
                else:
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j-1] = -dx/dy
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j+1] = -dx/dy
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -dy/dx
                    self.A[i*self.nx[1]+j,(i+1)*self.nx[1]+j] = -dy/dx
        return self.A
    def right(self):
        self.b = np.zeros([self.nx[0]*self.nx[1],1])
        
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                dx = self.hx[0];dy = self.hx[1]
                
                if i== 0 or i == self.nx[0] - 1 or j == 0 or j == self.nx[1] - 1:
                    self.b[i*self.nx[1]+j] = UU(self.X[i*self.nx[1]+j:i*self.nx[1]+j+1,:],[0,0],self.prob)
             
                else:
                    self.b[i*self.nx[1]+j] =  FF(self.prob,self.X[i*self.nx[1]+j:i*self.nx[1]+j+1,:])*dx*dy
        return self.b
    def solve(self):
        A = self.matrix()
        b = self.right()
        u = np.linalg.solve(A,b)
        return u
def error(u_pred,u_acc):
    temp = max(abs(u_pred - u_acc))
    return temp
bound = np.array([[0,1.0],[0,2.0]])
hx = [0.1,0.2]
prob = 1
fd = FD(bound,hx,prob)
u_pred = fd.solve()
u_acc = fd.u_acc
print(error(u_pred,u_acc))

def GS(A,b,x,epoch):
    N = b.shape[0]
    eps = 1e-7
    x_new = x.copy()
    for i in range(epoch):
        x_new[0,0] = (b[0,0] - (A[0,1:N]*x[1:N,0]).sum())/A[0,0]
        for k in range(1,N):
            x_new[k,0] = (b[k,0] - (A[k,0:k - 1]*x_new[0:k - 1,0]).sum() - (A[k,k + 1:N]*x_new[k + 1:N,0]).sum())/A[k,k]
        res = np.linalg.norm(b - A@x_new)
        if res < eps:
            break
        else:
            x = x_new
    print('the end iteration:%d'%(i + 1))
    return x_new
A = fd.matrix()
b = fd.right()
x = b.copy()
epoch = 500
u_p = GS(A,b,x,epoch)
print(error(u_pred,u_acc))

