import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
def UU(prob,X,t,ox,ot):#重新检查
    a = 1.0;d = 1.0
    if prob == 1:
        if ox == [0,0,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return -a*(fx*torch.sin(a*X[:,1] + d*X[:,2]) + fz*torch.cos(a*X[:,0] + d*X[:,1]))*torch.exp(-t*d**2)
        if ox == [0,0,0] and ot == 1:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return (a*d**2)*(fx*torch.sin(a*X[:,1] + d*X[:,2]) + fz*torch.cos(a*X[:,0] + d*X[:,1]))*torch.exp(-t*d**2)
        if ox == [1,0,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return -a*(s*fx*torch.sin(a*X[:,1] + d*X[:,2]) - a*fz*torch.sin(a*X[:,0] + d*X[:,1]))*torch.exp(-t*d**2)
        if ox == [0,1,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return -a*(a*fx*torch.cos(a*X[:,1] + d*X[:,2]) - d*fz*torch.sin(a*X[:,0] + d*X[:,1]))*torch.exp(-t*d**2)
        if ox == [0,0,1] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return -a*(d*fx*torch.cos(a*X[:,1] + d*X[:,2]) + a*fz*torch.cos(a*X[:,0] + d*X[:,1]))*torch.exp(-t*d**2)
        if ox == [2,0,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return -a*(fx*torch.sin(a*X[:,1] + d*X[:,2])*a**2 - fz*torch.cos(a*X[:,0] + d*X[:,1])*a**2)*torch.exp(-t*d**2)
        if ox == [0,2,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return a*(fx*torch.sin(a*X[:,1] + d*X[:,2])*a**2 + fz*torch.cos(a*X[:,0] + d*X[:,1])*d**2)*torch.exp(-t*d**2)
        if ox == [0,0,2] and ot == 0:
            fx = torch.exp(a*X[:,0]);fz = torch.exp(a*X[:,2])
            return a*(fx*torch.cos(a*X[:,1] + d*X[:,2])*d**2 - fz*torch.cos(a*X[:,0] + d*X[:,1])*a**2)*torch.exp(-t*d**2)
    if prob == 2:
        if ox == [0,0,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return -a*(fy*torch.sin(a*X[:,2] + d*X[:,0]) + fx*torch.cos(a*X[:,1] + d*X[:,2]))*torch.exp(-t*d**2)
        if ox == [0,0,0] and ot == 1:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return (a*d**2)*(fy*torch.sin(a*X[:,2] + d*X[:,0]) + fx*torch.cos(a*X[:,1] + d*X[:,2]))*torch.exp(-t*d**2)
        if ox == [1,0,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return -a*(d*fy*torch.cos(a*X[:,2] + d*X[:,0]) + a*fx*torch.cos(a*X[:,1] + d*X[:,2]))*torch.exp(-t*d**2)
        if ox == [0,1,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return -a*(a*fy*torch.sin(a*X[:,2] + d*X[:,0]) - a*fx*torch.sin(a*X[:,1] + d*X[:,2]))*torch.exp(-t*d**2)
        if ox == [0,0,1] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return -a*(a*fy*torch.cos(a*X[:,2] + d*X[:,0]) - d*fx*torch.sin(a*X[:,1] + d*X[:,2]))*torch.exp(-t*d**2)
        if ox == [2,0,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return a*(fy*torch.sin(a*X[:,2] + d*X[:,0])*d**2 - fx*torch.cos(a*X[:,1] + d*X[:,2])*a**2)*torch.exp(-t*d**2)
        if ox == [0,2,0] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return -a*(fy*torch.sin(a*X[:,2] + d*X[:,0])*a**2 - fx*torch.cos(a*X[:,1] + d*X[:,2])*a**2)*torch.exp(-t*d**2)
        if ox == [0,0,2] and ot == 0:
            fx = torch.exp(a*X[:,0]);fy = torch.exp(a*X[:,1])
            return a*(fy*torch.sin(a*X[:,2] + d*X[:,0])*a**2 + fx*torch.cos(a*X[:,1] + d*X[:,2])*d**2)*torch.exp(-t*d**2)
    if prob == 3:
        if ox == [0,0,0] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return -a*(fz*torch.sin(a*X[:,0] + d*X[:,1]) + fy*torch.cos(a*X[:,2] + d*X[:,0]))*torch.exp(-t*d**2)
        if ox == [0,0,0] and ot == 1:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return (a*d**2)*(fz*torch.sin(a*X[:,0] + d*X[:,1]) + fy*torch.cos(a*X[:,2] + d*X[:,0]))*torch.exp(-t*d**2)
        if ox == [1,0,0] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return -a*(a*fz*torch.cos(a*X[:,0] + d*X[:,1]) - d*fy*torch.sin(a*X[:,2] + d*X[:,0]))*torch.exp(-t*d**2)
        if ox == [0,1,0] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return -a*(d*fz*torch.cos(a*X[:,0] + d*X[:,1]) + a*fy*torch.cos(a*X[:,2] + d*X[:,0]))*torch.exp(-t*d**2)
        if ox == [0,0,1] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return -a*(a*fz*torch.sin(a*X[:,0] + d*X[:,1]) - a*fy*torch.sin(a*X[:,2] + d*X[:,0]))*torch.exp(-t*d**2)
        if ox == [2,0,0] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return a*(fz*torch.sin(a*X[:,0] + d*X[:,1])*a**2 + fy*torch.cos(a*X[:,2] + d*X[:,0])*d**2)*torch.exp(-t*d**2)
        if ox == [0,2,0] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return a*(fz*torch.sin(a*X[:,0] + d*X[:,1])*d**2 - fy*torch.cos(a*X[:,2] + d*X[:,0])*a**2)*torch.exp(-t*d**2)
        if ox == [0,0,2] and ot == 0:
            fy = torch.exp(a*X[:,1]);fz = torch.exp(a*X[:,2])
            return -a*(fz*torch.sin(a*X[:,0] + d*X[:,1])*a**2 - fy*torch.cos(a*X[:,2] + d*X[:,0])*a**2)*torch.exp(-t*d**2)
    if prob == 4:
        if ox == [0,0,0] and ot == 0:
            return (X[:,0]**2 + X[:,1]**2)*X[:,2]*torch.exp(t)
        if ox == [0,0,0] and ot == 1:
            return (X[:,0]**2 + X[:,1]**2)*X[:,2]*torch.exp(t)
        if ox == [1,0,0] and ot == 0:
            return 2*X[:,0]*X[:,2]*torch.exp(t)
        if ox == [0,1,0] and ot == 0:
            return 2*X[:,1]*X[:,2]*torch.exp(t)
        if ox == [0,0,1] and ot == 0:
            return (X[:,0]**2 + X[:,1]**2)*torch.exp(t)
        if ox == [2,0,0] and ot == 0:
            return 2*X[:,2]*torch.exp(t)
        if ox == [0,2,0] and ot == 0:
            return 2*X[:,2]*torch.exp(t)
        if ox == [0,0,2] and ot == 0:
            return 0*X[:,2]
def FF(prob,X,t):
    return UU(prob,X,t,[0,0,0],1) - (UU(prob,X,t,[2,0,0],0) + UU(prob,X,t,[0,2,0],0) + UU(prob,X,t,[0,0,2],0))
class INSET():
    def __init__(self,bound_x,bound_t,nx,nt,prob):
        self.hx = [(bound_x[1] - bound_x[0])/nx,(bound_x[3] - bound_x[2])/nx,(bound_x[5] - bound_x[4])/nx]
        self.ht = (bound_t[1] - bound_t[0])/nt
        self.size = nt*nx
        self.X = torch.zeros(self.size,4)
        for m in range(nt):
            self.X[m*nx:(m + 1)*nx,0] = bound_t[0] + (m + 1)*self.ht
        self.X[:,1:2] = (bound_x[1] - bound_x[0] - 2*self.hx[0])*torch.rand(self.size,1) + bound_x[0] + self.hx[0]
        self.X[:,2:3] = (bound_x[3] - bound_x[2] - 2*self.hx[1])*torch.rand(self.size,1) + bound_x[2] + self.hx[1]
        self.X[:,3:4] = (bound_x[5] - bound_x[4] - 2*self.hx[2])*torch.rand(self.size,1) + bound_x[4] + self.hx[2]
        
        self.u_acc = torch.zeros(self.size,1)
        for m in range(nt):
            self.u_acc[m*nx:(m + 1)*nx,0] = UU(prob,self.X[m*nx:(m + 1)*nx,1:4],bound_t[0] + (m + 1)*self.ht,[0,0,0],0)
        self.right = torch.zeros(self.size,1)
        for m in range(nt):
            self.right[m*nx:(m + 1)*nx,0] = FF(prob,self.X[m*nx:(m + 1)*nx,1:4],bound_t[0] + (m + 1)*self.ht)
            
class BDSET():
    def __init__(self,bound_x,bound_t,nx,nt,prob):
        self.hx = [(bound_x[1] - bound_x[0])/nx,(bound_x[3] - bound_x[2])/nx,(bound_x[5] - bound_x[4])/nx]
        self.ht = (bound_t[1] - bound_t[0])/nt
        self.Dlenth = 2*(bound_x[1] - bound_x[0])*(bound_x[3] - bound_x[2]) +\
        2*(bound_x[5] - bound_x[4])*(bound_x[3] - bound_x[2]) + 2*(bound_x[1] - bound_x[0])*(bound_x[5] - bound_x[4])
        self.Xomega = torch.zeros(nx,4)
        self.Xomega[:,0] = bound_t[0]
        self.Xomega[:,1:2] = (bound_x[1] - bound_x[0] - 2*self.hx[0])*torch.rand(nx,1) + bound_x[0] + self.hx[0]
        self.Xomega[:,2:3] = (bound_x[3] - bound_x[2] - 2*self.hx[1])*torch.rand(nx,1) + bound_x[2] + self.hx[1]
        self.Xomega[:,3:4] = (bound_x[5] - bound_x[4] - 2*self.hx[2])*torch.rand(nx,1) + bound_x[4] + self.hx[2]
        self.rig_omega = UU(prob,self.Xomega[:,1:4],bound_t[0],[0,0,0],0)
        self.Xtime = torch.zeros(6*nx*nt,4)
        for i in range(nt):
            self.Xtime[6*i*nx:6*(i + 1)*nx,0] = bound_t[0] + (i + 1)*self.ht
            self.Xtime[6*i*nx:6*i*nx + nx,1:2] = (bound_x[1] - bound_x[0] - 2*self.hx[0])\
            *torch.rand(nx,1)+ bound_x[0] + self.hx[0]
            self.Xtime[6*i*nx:6*i*nx + nx,2:3] = (bound_x[3] - bound_x[2] - 2*self.hx[1])\
            *torch.rand(nx,1) + bound_x[2] + self.hx[1]
            self.Xtime[6*i*nx:6*i*nx + nx,3] = bound_x[4]
            self.Xtime[6*i*nx + nx:6*i*nx + 2*nx,1:2] = (bound_x[1] - bound_x[0] - 2*self.hx[0])\
            *torch.rand(nx,1) + bound_x[0] + self.hx[0]
            self.Xtime[6*i*nx + nx:6*i*nx + 2*nx,2:3] = (bound_x[3] - bound_x[2] - 2*self.hx[1])\
            *torch.rand(nx,1) + bound_x[2] + self.hx[1]
            self.Xtime[6*i*nx + nx:6*i*nx + 2*nx,3] = bound_x[5]
            #---------------------------------
            self.Xtime[6*i*nx + 2*nx:6*i*nx + 3*nx,1] = bound_x[0]
            self.Xtime[6*i*nx + 2*nx:6*i*nx + 3*nx,2:3] = (bound_x[3] - bound_x[2] - 2*self.hx[1])\
            *torch.rand(nx,1) + bound_x[2] + self.hx[1]
            self.Xtime[6*i*nx + 2*nx:6*i*nx + 3*nx,3:4] = (bound_x[5] - bound_x[4] - 2*self.hx[2])\
            *torch.rand(nx,1) + bound_x[4] + self.hx[2]
            self.Xtime[6*i*nx + 3*nx:6*i*nx + 4*nx,1] = bound_x[1]
            self.Xtime[6*i*nx + 3*nx:6*i*nx + 4*nx,2:3] = (bound_x[3] - bound_x[2] - 2*self.hx[1])\
            *torch.rand(nx,1) + bound_x[2] + self.hx[1]
            self.Xtime[6*i*nx + 3*nx:6*i*nx + 4*nx,3:4] = (bound_x[5] - bound_x[4] - 2*self.hx[2])\
            *torch.rand(nx,1) + bound_x[4] + self.hx[2]
            #-----------------------------------
            self.Xtime[6*i*nx + 4*nx:6*i*nx + 5*nx,1:2] = (bound_x[1] - bound_x[0] - 2*self.hx[0])\
            *torch.rand(nx,1) + bound_x[0] + self.hx[0]
            self.Xtime[6*i*nx + 4*nx:6*i*nx + 5*nx,2] = bound_x[2]
            self.Xtime[6*i*nx + 4*nx:6*i*nx + 5*nx,3:4] = (bound_x[5] - bound_x[4] - 2*self.hx[2])\
            *torch.rand(nx,1) + bound_x[4] + self.hx[2]
            self.Xtime[6*i*nx + 5*nx:6*i*nx + 6*nx,1:2] = (bound_x[1] - bound_x[0] - 2*self.hx[0])\
            *torch.rand(nx,1) + bound_x[0] + self.hx[0]
            self.Xtime[6*i*nx + 5*nx:6*i*nx + 6*nx,2] = bound_x[3]
            self.Xtime[6*i*nx + 5*nx:6*i*nx + 6*nx,3:4] = (bound_x[5] - bound_x[4] - 2*self.hx[2])\
            *torch.rand(nx,1) + bound_x[4] + self.hx[2]
        self.rig_time = torch.zeros(6*nx*nt,1)
        for m in range(nt):
            self.rig_time[m*6*nx:(m + 1)*6*nx,0] = \
            UU(prob,self.Xtime[m*6*nx:(m + 1)*6*nx,1:4],bound_t[0] + (m + 1)*self.ht,[0,0,0],0)
class TESET():
    def __init__(self,bound_x,bound_t,nx,nt,prob):
        self.hx = [(bound_x[1] - bound_x[0])/nx,(bound_x[3] - bound_x[2])/nx,(bound_x[5] - bound_x[4])/nx]
        self.ht = (bound_t[1] - bound_t[0])/nt
        self.size = nt*nx
        self.X = torch.zeros(self.size,4)
        for m in range(nt):
            self.X[m*nx:(m + 1)*nx,0] = bound_t[0] + (m + 1)*self.ht
        self.X[:,1:2] = (bound_x[1] - bound_x[0])*torch.rand(self.size,1) + bound_x[0] 
        self.X[:,2:3] = (bound_x[3] - bound_x[2])*torch.rand(self.size,1) + bound_x[2] 
        self.X[:,3:4] = (bound_x[5] - bound_x[4])*torch.rand(self.size,1) + bound_x[4] 
        
        self.u_acc = torch.zeros(self.size,1)
        for m in range(nt):
            self.u_acc[m*nx:(m + 1)*nx,0] = UU(prob,self.X[m*nx:(m + 1)*nx,1:4],bound_t[0] + (m + 1)*self.ht,[0,0,0],0)
        self.right = torch.zeros(self.size,1)
        for m in range(nt):
            self.right[m*nx:(m + 1)*nx,0] = FF(prob,self.X[m*nx:(m + 1)*nx,1:4],bound_t[0] + (m + 1)*self.ht)      
np.random.seed(1234)
torch.manual_seed(1234)

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
            Res(4,16),
            Res(16,16),
            Res(16,16),
            Res(16,16)
            
        )
        self.fc = torch.nn.Linear(16,1)
    def forward(self,x):
        out = self.model(x)
        return self.fc(out)
def pred(netf,X):
    return netf.forward(X)
    
def error(u_pred,u_acc):
    #return max(abs(u_pred - u_acc))
    return (((u_pred - u_acc)**2).mean())**(0.5)
def Lossf(netf,inset,bdset):#之前方程有一个mu=0.5
    inset.X.requires_grad = True
    insetF = netf.forward(inset.X)
    insetFx, = torch.autograd.grad(insetF, inset.X, create_graph=True, retain_graph=True,
                                   grad_outputs=torch.ones(inset.size,1))
    ux = insetFx
    tauux, = torch.autograd.grad(ux[:,1:2], inset.X, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones(inset.size,1))#u_1的偏导数
    tauuy, = torch.autograd.grad(ux[:,2:3], inset.X, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones(inset.size,1))
    tauuz, = torch.autograd.grad(ux[:,3:4], inset.X, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones(inset.size,1))
    ut = insetFx[:,0:1]
    out_in = ((ut - tauux[:,1:2] - tauuy[:,2:3] - tauuz[:,3:4] - inset.right)**2).mean() 
    
    beta = 1.0
    ub_omega = netf.forward(bdset.Xomega)
    ub_time = netf.forward(bdset.Xtime)
    out_b = bdset.Dlenth*(((ub_omega - bdset.rig_omega)**2).mean() + ((ub_time - bdset.rig_time)**2).mean())
    return torch.sqrt(out_in + beta*out_b)
def Trainf(netf, inset,bdset,optimf, epochf):
    print('train neural network f')
    ERROR,BUZHOU = [],[]
    lossoptimal = Lossf(netf,inset,bdset).data
    trainerror = error(pred(netf,inset.X),inset.u_acc)#---------------------
    print('epoch: %d, loss: %.3e, trainerror: %.3e'
          %(0, lossoptimal.item(), trainerror.item()))
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
        trainerror = error(pred(netf,inset.X),inset.u_acc)
        ERROR.append(trainerror)
        BUZHOU.append((i + 1)*cycle)
        print('epoch:%d,lossf:%.3e,train error:%.3e,time:%.2f'%
             ((i + 1)*cycle,lossf.item(),trainerror,ela))
    return ERROR,BUZHOU
bound_x = torch.tensor([-1.0,1.0,0,1.5,-0.5,1.5])
bound_t = torch.tensor([0,1.0])
nx_tr = 20
nx_te = 40
nt = 10
prob = 2

netf = NETF()
epochf = 10
lr = 1e-2
tests_num = 1
testerror = torch.zeros(tests_num)
for it in range(tests_num):

    inset = INSET(bound_x,bound_t,nx_tr,nt,prob)
    bdset = BDSET(bound_x,bound_t,nx_tr,nt,prob)
    teset = TESET(bound_x,bound_t,nx_te,nt,prob)

    
    netf = NETF()
    optimf = torch.optim.LBFGS(netf.parameters(), lr=lr,max_iter = 100,tolerance_change=1e-14,
                               history_size=3000,line_search_fn = 'strong_wolfe')


    start_time = time.time()
    #ERROR,BUZHOU = Train(netg, netf, lenth, inset, bdset, optimg, optimf, epochg, epochf)
    ERROR,BUZHOU = Trainf(netf, inset, bdset, optimf, epochf)
    elapsed = time.time() - start_time
    print('Train time: %.2f' %(elapsed))

    
    netf.load_state_dict(torch.load('best_netf.pkl'))
    te_U = pred(netf,teset.X)
    testerror[it] = error(te_U, teset.u_acc)
    print('testerror = %.3e\n' %(testerror[it].item()))
    plt.plot(BUZHOU,ERROR,'r*')
    plt.plot(BUZHOU,ERROR)
    plt.title('the absolute error for example of prob = %d at iteration'%(prob))
    #plt.savefig('jvxing.jpg')
print(testerror.data)
testerror_mean = testerror.mean()


