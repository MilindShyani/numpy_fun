import numpy as np
from typing import Union, Tuple


class LayerNorm:
    def __init__(self, dim: int) -> None:                
        self.gamma = np.ones(dim).reshape(1,-1)
        self.beta = np.zeros(dim).reshape(1,-1)
        self.eps = 1e-8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self,x: np.ndarray) -> np.ndarray:
        self.mean = np.mean(x,axis=-1,keepdims=True)
        self.std = np.std(x,axis=-1,keepdims = True)
        self.shifted = (x - self.mean)/np.sqrt(self.std**2 + self.eps)
        # shifted has shape (B,D)
        out = self.shifted*self.gamma + self.beta
        return out
    
    def backward(self, grad_in: np.ndarray) -> Tuple[np.ndarray, ...]:        
        self.db = np.sum(grad_in,axis=0,keepdims=True)
        self.dg = np.sum(grad_in*self.shifted,axis=0,keepdims=True )
        self.grad_out = grad_in*self.gamma*1/np.sqrt(self.std**2 + self.eps)
        return self.dg, self.db, self.grad_out
    
    def update(self,lr:float) -> None:
        self.gamma -= lr*self.dg
        self.beta -= lr*self.db


class FCC:
    def __init__(self, dims: list) -> None:
        self.weights = []
        self.biases = []
        self.layernorms = []

        for i, (dim_in, dim_out) in enumerate(zip(dims,dims[1:])):
            W = np.random.uniform(low = -1, high = 1, size=(dim_in,dim_out))/np.sqrt(dim_out)
            b = np.zeros(dim_out)  
            if i < len(dims) - 2:  
                ln = LayerNorm(dim_out)
                self.layernorms.append(ln)
            self.weights.append(W)
            self.biases.append(b) 
                                                 
        self.a = []       
        self.z = []
        self.dA = []
        self.dB = []
        self.dW = []                

    def sigmoid(self, x: Union[np.ndarray,float]) -> Union[np.ndarray,float]:
        return 1/(1+np.exp(-x))                    
    
    def forward(self,x: np.ndarray) -> np.ndarray:
        assert len(x.shape) > 1
        assert x.shape[-1] == self.weights[0].shape[0]
        self.a.append(x)
        for W,b,ln in zip(self.weights[:-1],self.biases[:-1],self.layernorms):
            x = x @ W + b
            x = ln(x)
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
    
    def sigmoidback(self,x):
        return x*(1-x)
        

    def backward(self,grad: np.ndarray) -> None:
        self.dW = []
        self.dA = []
        self.dB = []                                
        self.dA.append(grad)

        dz = self.dA[-1]*self.sigmoidback(self.a[-1])
        
        dw =  self.a[-2].T @ dz
        da = dz @ self.weights[-1].T
        db =  np.sum(dz,0)

        self.dW.append(dw)
        self.dB.append(db)
        self.dA = [da] + self.dA
                
        for i in range(2,len(self.weights)+1):                                    
            dz = self.dA[-i]*self.reluback(self.a[-i])

            _,_, dz = self.layernorms[-i+1].backward(dz)

            dw = self.a[-i-1].T @ dz            
            da = dz @ self.weights[-i].T
            db = np.sum(dz,axis=0) 
            
            self.dW = [dw] + self.dW
            self.dA = [da] + self.dA
            self.dB = [db] + self.dB
                        
        return
    
    def update(self,lr: float = 1e-1 ) -> None:
        for i in range(len(self.weights)):
            assert self.weights[i].shape == self.dW[i].shape
            self.weights[i] -= lr*self.dW[i]
            if i < len(self.weights) - 1 :
                self.layernorms[i].update(lr)
            # print(f'{np.linalg.norm(self.weights[i])} and grad {np.linalg.norm(self.dW[i])}')
        # print("next")
        


X = np.random.normal(0,1,size=(1000,20))            
mymodel = FCC([20,8,5,1])
coeff = np.random.normal(1,0.01,size=(1,20))
# coeff = np.ones((1,20)) + 3
targets = np.tanh(X @ coeff.T + 2)
for step in range(15000):
    out = mymodel.forward(X)
    loss = ((out - targets)**2)
    dloss = 2*(out - targets)/X.shape[0]
    mymodel.backward(dloss)
    mymodel.update(1e-2)
    if not step % 1:
        print(np.mean(loss))





        



            
