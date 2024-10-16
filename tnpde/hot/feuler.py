import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
def UU(prob,X,t,ox,ot):#精确解，将空间坐标点排成（M*N，2）的形式计算
    if prob == 1:
        if ox == [0,0] and ot == 0:
            return (X[:,0]**2 + X[:,1]**2)*torch.exp(t*(1 - t))
        if ox == [1,0] and ot == 0:
            return 2*X[:,0]*torch.exp(t*(1 - t))
        if ox == [0,1] and ot == 0:
            return 2*X[:,1]*torch.exp(t*(1 - t))
        if ox == [2,0] and ot == 0:
            return 2*torch.exp(t*(1 - t))*torch.ones(X.shape[0])
        if ox == [0,2] and ot == 0:
            return 2*torch.exp(t*(1 - t))*torch.ones(X.shape[0])
        if ox == [0,0] and ot == 1:
            return (X[:,0]**2 + X[:,1]**2)*(1 - 2*t)*torch.exp(t*(1 - t))
    if prob == 2:
        fenzi = (X[:,0]**2 - X[:,1]**2)*t
        temp = (X[:,0]**2 + X[:,1]**2)*(1 - t) + 0.2
        if ox == [0,0] and ot == 0:
            return fenzi/temp
        if ox == [1,0] and ot == 0:
            return (2*X[:,0]*t*temp - 2*X[:,0]*(1 - t)*fenzi)/(temp**2)
        if ox == [0,1] and ot == 0:
            return (- 2*X[:,1]*t*temp - 2*X[:,1]*(1 - t)*fenzi)/(temp**2)
        if ox == [2,0] and ot == 0:
            return - 2*2*X[:,0]*(1 - t)*(2*X[:,0]*t*temp - 2*X[:,0]*(1 - t)*fenzi)/(temp**3) + \
                    (2*t*temp - 2*(1 - t)*fenzi)/(temp**2)
        if ox == [0,2] and ot == 0:
            return - 2*2*X[:,1]*(1 - t)*(- 2*X[:,1]*t*temp - 2*X[:,1]*(1 - t)*fenzi)/(temp**3) + \
                    (- 2*t*temp - 2*(1 - t)*fenzi)/(temp**2)
        if ox == [0,0] and ot == 1:
            return ((X[:,0]**2 - X[:,1]**2)*temp + (X[:,0]**2 + X[:,1]**2)*fenzi)/(temp**2)
def CC(prob,X,t):
    return UU(prob,X,t,[0,0],1) - UU(prob,X,t,[2,0],0) - UU(prob,X,t,[0,2],0)

X = torch.rand(4,2)
t = torch.tensor([0.1])
prob = 1
print(UU(prob,X,t,[0,0],1),UU(prob,X,t,[2,0],0),UU(prob,X,t,[0,2],0))
#print(CC(prob,X,t))
print(UU(prob,X,t,[0,0],0))
A = torch.zeros(3,4)
print(A[0,:].shape)

class ForEuler():
    def __init__(self,bounds_X,hx,bounds_T,ht,prob):
        self.dim = 2
        self.M = int((bounds_X[0,1] - bounds_X[0,0])/hx[0]) + 1
        self.N = int((bounds_X[1,1] - bounds_X[1,0])/hx[1]) + 1
        self.size = self.M*self.N
        self.hx = hx
        self.prob = prob
        self.X = torch.zeros(self.size,self.dim)
        for i in range(self.M):
            for j in range(self.N):
                self.X[i*self.N + j,0] = bounds_X[0,0] + i*hx[0]
                self.X[i*self.N + j,1] = bounds_X[1,0] + j*hx[1]
        self.nt = int((bounds_T[1] - bounds_T[0])/ht) + 1
        self.ht = ht
        self.T = torch.zeros(self.nt,1)
        for i in range(self.nt):
            self.T[i] = bounds_T[0] + i*ht
        self.u_acc = torch.zeros(self.size,self.nt)#精确解（M*N，nt)
        for i in range(self.nt):
            self.u_acc[:,i] = UU(prob,self.X,self.T[i],[0,0],0)
#采取迭代方式求解        
        
    def matrix(self):
        self.A = torch.zeros(self.size,self.size)
        dx = self.hx[0];dy = self.hx[1]
        dt = self.ht
       
        for i in range(self.M):
            for j in range(self.N):
                if i == 0 or i == self.M - 1 or j == 0 or j == self.N - 1:#当考虑空间边界时，对应一行为0
                    self.A[i*self.N + j,:] = torch.zeros(self.size)
                else:
                    self.A[i*self.N + j,i*self.N + j] = 1 - 2*dt*(1/(dx**2) + 1/(dy**2))
                    self.A[i*self.N + j,i*self.N + j + 1] = dt/(dy**2)
                    self.A[i*self.N + j,i*self.N + j - 1] = dt/(dy**2)
                    self.A[i*self.N + j,(i + 1)*self.N + j] = dt/(dx**2)
                    self.A[i*self.N + j,(i - 1)*self.N + j] = dt/(dx**2)
        return self.A
    def right(self,n):#注意这里迭代过程中要使用n+1
        self.b = torch.zeros(self.size,1)
        dt = self.ht
        for i in range(self.M):
            for j in range(self.N):
                X = self.X[i*self.N + j:i*self.N + j + 1,:]
                if i == 0 or i == self.M - 1 or j == 0 or j == self.N - 1:
                    self.b[i*self.N + j] = UU(self.prob,X,self.T[n],[0,0],0)#当考虑空间边界时，对应的分量就是精确解
                else:
                    self.b[i*self.N + j] = dt*CC(self.prob,X,self.T[n])
        return self.b
   
    def diedai(self):
        mat = self.matrix()
        self.pred = torch.zeros(self.size,self.nt)
        self.pred[:,0] = UU(prob,self.X,self.T[0],[0,0],0)
        for i in range(1,self.nt):
            self.pred[:,i:i + 1] = mat@self.pred[:,i - 1:i] + self.right(i)
        return self.pred
class BackEuler():
    def __init__(self,bounds_X,hx,bounds_T,ht,prob):
        self.dim = 2
        self.M = int((bounds_X[0,1] - bounds_X[0,0])/hx[0]) + 1
        self.N = int((bounds_X[1,1] - bounds_X[1,0])/hx[1]) + 1
        self.size = self.M*self.N
        self.hx = hx
        self.prob = prob
        self.X = torch.zeros(self.size,self.dim)
        for i in range(self.M):
            for j in range(self.N):
                self.X[i*self.N + j,0] = bounds_X[0,0] + i*hx[0]
                self.X[i*self.N + j,1] = bounds_X[1,0] + j*hx[1]
        self.nt = int((bounds_T[1] - bounds_T[0])/ht) + 1
        self.ht = ht
        self.T = torch.zeros(self.nt,1)
        for i in range(self.nt):
            self.T[i] = bounds_T[0] + i*ht
        self.u_acc = torch.zeros(self.size,self.nt)
        for i in range(self.nt):
            self.u_acc[:,i] = UU(prob,self.X,self.T[i],[0,0],0)
    def matrix(self):
        self.A = torch.zeros(self.size,self.size)
        self.B = torch.zeros(self.size,self.size)
        dx = self.hx[0];dy = self.hx[1]
        dt = self.ht
        
        for i in range(self.M):
            for j in range(self.N):
                if i == 0 or i == self.M - 1 or j == 0 or j == self.N - 1:
                    self.A[i*self.N + j,i*self.N + j] = 1
                    self.B[i*self.N + j,:] = torch.zeros(self.size)
                else:
                    self.B[i*self.N + j,i*self.N + j] = 1
                    #------------------
                    self.A[i*self.N + j,i*self.N + j] = 1 + 2*dt*(1/(dx**2) + 1/(dy**2))
                    self.A[i*self.N + j,i*self.N + j + 1] = - dt/(dy**2)
                    self.A[i*self.N + j,i*self.N + j - 1] = - dt/(dy**2)
                    self.A[i*self.N + j,(i + 1)*self.N + j] = - dt/(dx**2)
                    self.A[i*self.N + j,(i - 1)*self.N + j] = - dt/(dx**2)
        return self.A,self.B
    def right(self,n):#注意这里迭代过程中要使用n+1
        self.b = torch.zeros(self.size,1)
        dt = self.ht
        for i in range(self.M):
            for j in range(self.N):
                x = i*self.hx[0];y = j*self.hx[1]
                X = self.X[i*self.N + j:i*self.N + j + 1,:]
                if i == 0 or i == self.M - 1 or j == 0 or j == self.N - 1:
                    self.b[i*self.N + j] = UU(self.prob,X,self.T[n],[0,0],0)
                else:
                    self.b[i*self.N + j] = dt*CC(self.prob,X,self.T[n])
        return self.b
    def diedai(self):
        zuo,you = self.matrix()
        self.pred = torch.zeros(self.size,self.nt)
        self.pred[:,0] = UU(prob,self.X,self.T[0],[0,0],0)
        print(zuo.shape,you.shape,self.pred[:,0].shape)
        for i in range(1,self.nt):
            old = (you@self.pred[:,i - 1:i]).float() + self.right(i)
            self.pred[:,i:i + 1],lu = torch.solve(old,zuo)
        return self.pred


def error(u_pred, u_acc):
    fenzi = ((u_pred - u_acc)**2).sum()
    fenmu = (u_acc**2).sum()
    return (fenzi/fenmu)**(0.5)

bounds_X = torch.tensor([[-1,1],[-1,1]]).float()
bounds_T = torch.tensor([0,1]).float()
hx = [0.1,0.1]
ht = 0.05
prob = 1

fe = ForEuler(bounds_X,hx,bounds_T,ht,prob) 
u_pred = fe.diedai()
u_acc = fe.u_acc

print(error(u_pred,u_acc))

