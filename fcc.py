import numpy as np
from typing import Union

class FCC:
    def __init__(self, dims: list) -> None:
        self.weights = []
        self.biases = []        
        for dim_in, dim_out in zip(dims,dims[1:]):
            W = np.random.uniform(size=(dim_in,dim_out))/np.sqrt(dim_out)
            b = np.zeros(dim_out)
            self.weights.append(W)
            self.biases.append(b) 
        self.a = []       
        self.z = []
        self.dA = []
        self.db = []
        self.dW = []

    def sigmoid(self, x: Union[np.ndarray,float]) -> Union[np.ndarray,float]:
        return 1/(1+np.exp(-x))            
    
    def forward(self,x: np.ndarray) -> np.ndarray:
        assert len(x.shape) > 1
        assert x.shape[-1] == self.weights[0].shape[0]
        self.a.append(x)
        for W,b in zip(self.weights[:-1],self.biases[:-1]):
            x = x @ W + b
            self.z.append(x)                
            x = np.where(x>=0,x,0)
            self.a.append(x)

        W, b = self.weights[-1],self.biases[-1]
        x = x @ W + b
        self.z.append(x)                
        x = self.sigmoid(x)
        self.a.append(x)
                            
        return x
    
    def reluback(self,x):
        return np.where(x>=0,1,0)

    def backward(self,grad: np.ndarray) -> None:
        self.dW = []
        self.dA = []        
        # assert grad.shape == (b,1)
        # self.da.append(grad)
        # temp_grad = np.sum(grad*self.a[-1]*(1-self.a[-1]) * self.a[-2],axis=0,keepdims=True).T
        dz = grad*self.a[-1]*(1-self.a[-1]) 
        # dw = np.sum(dz*self.a[-2], axis=0)
        
        dw =  self.a[-2].T @ dz
        da = dz @ self.weights[-1].T

        self.dW.append(dw)
        self.dA.append(da)
        
        
        for i in range(2,len(self.weights)+1):
            for el in reversed(self.dA):
                print(el.shape)               
            print(f'a shape: {self.a[-i+1].shape} and i = {i} and {self.dA[-i+1].shape}') 
            dz = self.dA[-i+1]*self.reluback(self.a[-i+1])
            dw = self.a[-i-1].T @ dz            
            da = dz @ self.weights[-i].T
            print(dz.shape, dw.shape,da.shape)
            self.dW = [dw] + self.dW
            self.dA = [da]+ self.dA
            print("Next")
            
        return
    
    def update(self,lr: float = 1e-3 ) -> None:
        for i in range(len(self.weights)):
            self.weights[i] += lr*self.dW[i]
        


X = np.random.normal(0,1,size=(100,20))            
mymodel = FCC([20,15,10,5,1])
coeff = np.random.normal(1,0.01,size=(1,20))
targets = X @ coeff.T + 2
for step in range(100):
    out = mymodel.forward(X)
    loss = (out - targets)**2
    dloss = 2*(out - targets)*out
    mymodel.backward(dloss)
    mymodel.update()
    print(np.sum(loss))





        



            
