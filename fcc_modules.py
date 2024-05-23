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
    
    def backward(self, grad_in: np.ndarray) -> np.ndarray:        
        self.db = np.sum(grad_in,axis=0,keepdims=True)
        self.dg = np.sum(grad_in*self.shifted,axis=0,keepdims=True )
        self.grad_out = grad_in*self.gamma*1/np.sqrt(self.std**2 + self.eps)
        return self.grad_out
    
    def update(self,lr:float) -> None:
        self.gamma -= lr*self.dg
        self.beta -= lr*self.db

class Linear:
    def __init__(self, dim_in: int, dim_out: int) -> None:                
        self.W = np.random.uniform(low = -1, high = 1, size=(dim_in,dim_out))/np.sqrt(dim_out)
        self.b = np.zeros(dim_out)  
            
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self,x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b                                
        self.a = x
        return out
    
    def backward(self, grad_in: np.ndarray) -> np.ndarray:        
        self.dW = self.a.T @ grad_in
        self.db = np.sum(grad_in,0)        
        self.grad_out = grad_in @ self.W.T
        return self.grad_out
    
    def update(self,lr:float) -> None:
        self.W -= lr*self.dW
        self.b -= lr*self.db

class relu:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self,x):
        return np.where(x>=0,x,0)
    
    def backward(self,x):
        return np.where(x>=0,1,0)


class sigmoid:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self,x):
        return 1/(1+np.exp(-x))
    
    def backward(self,x):
        return x*(1-x)

class FCC:
    def __init__(self, dims: list) -> None:
        self.linear_layers = []        
        self.layernorms = []
        self.relu = relu()
        self.sigmoid = sigmoid()
        
        for i, (dim_in, dim_out) in enumerate(zip(dims,dims[1:])):            
            if i < len(dims) - 2:  
                ln = LayerNorm(dim_out)
                self.layernorms.append(ln)                
            self.linear_layers.append(Linear(dim_in,dim_out))            
                                                             
        self.a = []       
        self.dA = []        
    
    def forward(self,x: np.ndarray) -> np.ndarray:
        assert len(x.shape) > 1        
        self.a.append(x)
        for linear,ln in zip(self.linear_layers[:-1],self.layernorms):
            x = linear(x)
            x = ln(x)            
            x = self.relu(x)
            self.a.append(x)       
             
        x = self.linear_layers[-1](x)
           
        x = self.sigmoid(x)
        self.a.append(x)
        return x
              
    def backward(self,grad: np.ndarray) -> None:        
        self.dA = []        
        self.dA.append(grad)
        
        for i in range(1,2): 
            dz = self.dA[-i]*self.sigmoid.backward(self.a[-i])
            da = self.linear_layers[-i].backward(dz)
        
        self.dA = [da] + self.dA
                
        for i in range(2,len(self.linear_layers)+1):                                    
            dz = self.dA[-i]*self.relu.backward(self.a[-i])
            dzp = self.layernorms[-i+1].backward(dz)
            da = self.linear_layers[-i].backward(dzp)                                                
            self.dA = [da] + self.dA                                    
        return
    
    def update(self,lr: float = 1e-1 ) -> None:
        for layer in self.linear_layers:
            layer.update(lr)
            
        for layer in self.layernorms:
            layer.update(lr)
                                
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





        



            
