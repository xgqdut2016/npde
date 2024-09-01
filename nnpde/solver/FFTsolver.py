import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import bfgs
from data import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
dtype = torch.float32
'''
fname_y = "data_y.npy"
fname_u = "data_u.npy"
fname_x = "data_x.npy"
'''
fname_y = "C://Users//2001213226//Desktop//graduation//deeponet//data_y.npy"
fname_u = "C://Users//2001213226//Desktop//graduation//deeponet//data_u.npy"
fname_x = "C://Users//2001213226//Desktop//graduation//deeponet//data_x.npy"

data_y = np.load(fname_y)#[fun_size,sample_size]
data_u = np.load(fname_u)#[fun_size,sample_size]
data_x = np.load(fname_x)#[sample_size,dim = 2]
def pre_data(data_u,data_y,data_x,dev):
    data_u = torch.tensor(data_u).type(dtype)
    data_y = torch.tensor(data_y).type(dtype)
    data_x = torch.tensor(data_x).type(dtype)
    fun_num = data_u.shape[0]
    input_x = data_x.repeat(fun_num,1,1)#[fun_size,sample_size,dim = 2]
    input_u = data_u.unsqueeze(2)#[fun_size,sample_size,1]
    input_y = data_y.unsqueeze(2)#[fun_size,sample_size,1]
    return torch.cat([input_u,input_x],dim = -1).to(dev),input_y.to(dev)#[fun_size,sample_size,dim = 3]和#[fun_size,sample_size,1]
class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes,  dtype=torch.cfloat))
        #in_channels，out_channals对应的是空间维度，后面会把维度=3提升到dim = width
        #modes对应的是样本点数目，后面会根据modes把样本点分成两份，分别做矩阵乘积
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        #input = (fun_num,input_size,sample_size),weight = (input_size,out_size,sample_size)->(fun_num,out_size,sample_size)
        #爱因斯坦求和，这里把数组前面两个维度做了一个矩阵乘
        return torch.einsum("bix,iox->box", input, weights)
    def forward(self, x):
        #x = [fun_size,width,sample_size]
        x_ft = torch.fft.rfft2(x)#x_ft = [fun_size,width,sample_size//2 + 1]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros_like(x_ft)
        #特别注意，这里的modes一定要小于sample_size//2 + 1
        out_ft[:, :, :self.modes] = self.compl_mul2d(x_ft[:, :, :self.modes], self.weights1)
        out_ft[:, :, -self.modes:] = self.compl_mul2d(x_ft[:, :, -self.modes:], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        #此时的x形状和一开始一样，也是[fun_size,width,sample_size]
        return x
class Solvernet(torch.nn.Module):
    def __init__(self, width,modes):
        super(Solvernet, self).__init__()
        self.modes = modes
        
        self.width = width
        
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        

        self.fc1 = nn.Linear(self.width, 16)
        self.fc2 = nn.Linear(16, 1)
    def forward(self, x):
        
        x = self.fc0(x)#[fun_num,sample_num,width]
        x = x.permute(0,2,1)
        #x = F.gelu(x)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])
def pred_y(solvernet,x):#x = [fun_size,sample,3]
    return solvernet.forward(x)
def Loss(solvernet,x,data_y):
    y = pred_y(solvernet,x)
    err = ((y - data_y)**2).mean()
    return torch.sqrt(err)
def train_yp(solvernet,x,data_y,optim,scheduler,epoch,optimtype):
    print('Train y&p Neural Network')
    loss = Loss(solvernet,x,data_y)
    print('epoch:%d,loss:%.2e, time: %.2f'
          %(0, loss.item(), 0.00))
    for it in range(epoch):
        st = time.time()
        if optimtype == 'BFGS' or optimtype == 'LBFGS':
            def closure():
                optim.zero_grad()
                loss = Loss(solvernet,x,data_y)
                loss.backward()
                return loss
            optim.step(closure) 
        else:
            for j in range(100):
                optim.zero_grad()
                loss = Loss(solvernet,x,data_y)
                loss.backward()
                optim.step()
        scheduler.step()
        loss = Loss(solvernet,x,data_y)
        ela = time.time() - st
        print('epoch:%d,loss:%.2e, time: %.2f'
              %((it+1), loss.item(), ela))
x,data_y = pre_data(data_u,data_y,data_x,device)
width = 10
modes = data_x.shape[1]//4 + 1
solvernet = Solvernet(width,modes)
fname1 = "fftlay%d-ynet-var.pt"%(width)
solvernet = solvernet.to(device)
lr = 1e0
optimtype = 'Adam'
#optimtype = 'BFGS'
if optimtype == 'BFGS':
    optim = bfgs.BFGS(solvernet.parameters(),
                      lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
else:
    optim = torch.optim.Adam(solvernet.parameters(),
                      lr=lr,weight_decay=1e-4)    
step_size = 100
gamma = 0.5                     
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)      

epoch = 10
#solvernet = torch.load(fname1)
print(x.shape,data_y.shape)
#train_yp(solvernet,x,data_y,optim,scheduler,epoch,optimtype)
torch.save(solvernet,fname1)
solvernet = torch.load(fname1)

#-----------------------
class Testdata():
    def __init__(self,fun_size,sample_size,layers,dtype,seeds):
        self.dim = 2
        tmp = self.quasi_samples(sample_size)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.x = torch.tensor(tmp).type(dtype)
        self.data_y = torch.zeros(fun_size,sample_size)
        self.data_u = torch.zeros(fun_size,sample_size)
        for i in range(fun_size):
            np.random.seed(seeds + i)
            torch.manual_seed(seeds + i)
            net_data = Net(layers,dtype)
            
            self.data_y[i:i + 1,:] = function_y(net_data,len_data,self.x).t()
            self.data_u[i:i + 1,:] = function_u(net_data,len_data,self.x).t()
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
layers = [2,lay_wid,lay_wid,lay_wid,1]

seeds = 10000
fun_size = 3
sample_size = 4000
teset = Testdata(fun_size,sample_size,layers,dtype,seeds)
data_u = teset.data_u.detach().numpy()
data_y = teset.data_y.detach().numpy()
data_x = teset.x.detach().numpy()
print(data_u.shape,data_y.shape,data_x.shape)
x,data_y = pre_data(data_u,data_y,data_x,device)
print(data_y.shape,x.shape)


y = pred_y(solvernet,x)
y_acc = data_y
err = (y - y_acc).reshape(-1,1)
L1 = max(abs(err))
L2 = (err**2).sum()
print('test error:L1:%.2e,L2:%.2e'%(L1,L2))

