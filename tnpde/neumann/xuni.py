

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
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
def f(prob,X):
    return -UU(X,[2,0],prob)- UU(X,[0,2],prob)
class FD():
    def __init__(self,bound,hx,prob):
        self.prob = prob
        self.dim = 2
        self.hx = hx
        self.nx = [int((bound[0,1] - bound[0,0])/self.hx[0]) + 1,int((bound[1,1] - bound[1,0])/self.hx[1]) + 1]
        self.size = self.nx[0]*self.nx[1]
        self.X = torch.zeros(self.size,self.dim)
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.X[m,0] = bound[0,0] + i*self.hx[0]
                self.X[m,1] = bound[1,0] + j*self.hx[1]
                m = m + 1
        self.u_acc = UU(self.X,[0,0],prob).view(-1,1)
    def matrix(self):
        self.A = torch.zeros(self.size,self.size)
        dx = self.hx[0];dy = self.hx[1]
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                if i == 0:
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 1
                
                elif (i == self.nx[0] - 1 and j > 0 and j < self.nx[1] - 1):
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j-1] = -dx/dy
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j+1] = -dx/dy
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -2*dy/dx
                    
                elif (j == 0 and i > 0 and i < self.nx[0] - 1):
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j+1] = -2*dx/dy
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -dy/dx
                    self.A[i*self.nx[1]+j,(i+1)*self.nx[1]+j] = -dy/dx
                    
                elif (j == self.nx[1] - 1 and i > 0 and i < self.nx[0] - 1):
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j-1] = -2*dx/dy
                    
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -dy/dx
                    self.A[i*self.nx[1]+j,(i+1)*self.nx[1]+j] = -dy/dx
                
                elif (i == self.nx[0] - 1 and j == 0):
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j+1] = -2*dx/dy
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -2*dy/dx
                elif (i == self.nx[0] - 1 and j == self.nx[1] - 1):
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j-1] = -2*dx/dy
                    
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -2*dy/dx
                elif (i > 0 and i < self.nx[0] - 1 and j > 0 and j < self.nx[1] - 1):
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j] = 2*(dx/dy + dy/dx)
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j-1] = -dx/dy
                    self.A[i*self.nx[1]+j,i*self.nx[1]+j+1] = -dx/dy
                    self.A[i*self.nx[1]+j,(i-1)*self.nx[1]+j] = -dy/dx
                    self.A[i*self.nx[1]+j,(i+1)*self.nx[1]+j] = -dy/dx
                    
                
        return self.A
    def right(self):
        self.b = torch.zeros(self.size,1)
        dx = self.hx[0];dy = self.hx[1]
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                X = self.X[i*self.nx[1]+j:i*self.nx[1]+j+1,:]
                if i == 0:
                    self.b[i*self.nx[1]+j] = UU(X,[0,0],self.prob)
                elif (i == self.nx[0] - 1 and j > 0 and j < self.nx[1] - 1):
                    self.b[i*self.nx[1]+j] = f(self.prob,X)*dx*dy + UU(X,[1,0],self.prob)*2*dy
                elif (j == 0 and i > 0 and i < self.nx[0] - 1):
                    self.b[i*self.nx[1]+j] = f(self.prob,X)*dx*dy - UU(X,[0,1],self.prob)*2*dx
                elif (j == self.nx[1] - 1 and i > 0 and i < self.nx[0] - 1):
                    self.b[i*self.nx[1]+j] = f(self.prob,X)*dx*dy + UU(X,[0,1],self.prob)*2*dx
                
                
                elif (i == self.nx[0] - 1 and j == 0):
                    self.b[i*self.nx[1]+j] = f(self.prob,X)*dx*dy + UU(X,[1,0],self.prob)*2*dy - UU(X,[0,1],self.prob)*2*dx
                elif (i == self.nx[0] - 1 and j == self.nx[1] - 1):
                    self.b[i*self.nx[1]+j] = f(self.prob,X)*dx*dy + UU(X,[1,0],self.prob)*2*dy + UU(X,[0,1],self.prob)*2*dx
                elif (i > 0 and i < self.nx[0] - 1 and j > 0 and j < self.nx[1] - 1):
                    self.b[i*self.nx[1]+j] =  f(self.prob,self.X[i*self.nx[1]+j:i*self.nx[1]+j+1,:])*dx*dy
        return self.b
    def solve(self):
        A = self.matrix()
        b = self.right()
        u,lu = torch.solve(b,A)
        return u
def error(u_pred,u_acc):
    temp = ((u_pred - u_acc)**2).sum()/(u_acc**2).sum()
    return temp**(0.5)
bound = torch.tensor([[0,2],[0,1]]).float()
hx = [0.2,0.1]
prob = 2
fd = FD(bound,hx,prob)
u_pred = fd.solve()
u_acc = fd.u_acc
print(error(u_pred,u_acc))

