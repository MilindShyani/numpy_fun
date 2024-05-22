import numpy as np
from typing import Union

class FCC:
    def __init__(self, dims: list) -> None:
        self.weights = []
        self.biases = []
        self.gammas = []        
        self.betas = []
        for dim_in, dim_out in zip(dims,dims[1:]):
            W = np.random.uniform(low = -1, high = 1, size=(dim_in,dim_out))/np.sqrt(dim_out)
            b = np.zeros(dim_out)
            gamma = np.ones(dim_out)
            beta = np.zeros(dim_out)
            self.weights.append(W)
            self.biases.append(b) 
            self.betas.append(beta)
            self.gammas.append(gamma)
            
        self.a = []       
        self.z = []
        self.dA = []
        self.dB = []
        self.dW = []
        self.dG = []
        self.dBT = []

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
        self.dB = []                                
        self.dA.append(grad)

        dz = self.dA[-1]*self.a[-1]*(1-self.a[-1]) 
        
        dw =  self.a[-2].T @ dz
        da = dz @ self.weights[-1].T
        db =  np.sum(dz,0)

        self.dW.append(dw)
        self.dB.append(db)
        self.dA = [da] + self.dA
                
        for i in range(2,len(self.weights)+1):                                    
            dz = self.dA[-i]*self.reluback(self.a[-i])

            dw = self.a[-i-1].T @ dz            
            da = dz @ self.weights[-i].T
            db = np.sum(dz,axis=0) 
            
            self.dW = [dw] + self.dW
            self.dA = [da] + self.dA
            self.dB = [db] + self.dB
                        
        return
    
    def update(self,lr: float = 1e-2 ) -> None:
        for i in range(len(self.weights)):
            assert self.weights[i].shape == self.dW[i].shape
            self.weights[i] -= lr*self.dW[i]
            # print(f'{np.linalg.norm(self.weights[i])} and grad {np.linalg.norm(self.dW[i])}')
        # print("next")
        


X = np.random.normal(0,1,size=(1000,20))            
mymodel = FCC([20,8,5,1])
# coeff = np.random.normal(1,0.01,size=(1,20))
coeff = np.ones((1,20)) + 3
targets = X @ coeff.T + 2
for step in range(150):
    out = mymodel.forward(X)
    loss = ((out - targets)**2)
    dloss = 2*(out - targets)/X.shape[0]
    mymodel.backward(dloss)
    mymodel.update()
    if not step % 1:
        print(np.mean(loss))





        



            
