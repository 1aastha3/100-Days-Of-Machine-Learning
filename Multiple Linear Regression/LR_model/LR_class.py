## importing the libraries
import numpy as np
import matplotlib.pyplot as plt

## creating an object containing the type of gradient descent method(requires type) and regularization (requires regularization type and lamda)

Type_args = {"gradient descent" : "default_batchGD",
             "regularization" : None,
             "lambda" : 0,
             "batch_size" : None}

## create a class

class LR_model:
##creating an initializing attribution function which takes and initializes the values
    def __init__(self, lr, epoch, argument = Type_args):
        self.lr = lr
        self.epoch = epoch
        self.reg = argument["regularization"]
        self.grad_descent = argument["gradient descent"]
        self.lambda = argument["lambda"]
        self.batch_size = argument["batch_size"]
        
## creating a function where we pass our training set, initialize weights, bias and make cost plots

    def __train(self, x, y):
        self.w = np.random.rand(x.shape[-1], 1)
        self.b = np.random.rand()
        
        cost = self.gradient_descent(x, y)
        plt.plot(cost)
        
        return x.dot(self.w) + self.b
    
# creating a function where we iterate through epochs for required type of gradient descent and regulariztaion
  
    def __gradient_descent(self, x, y):
        cost = []  ## we empty cost array after every call
        n = y.shape[0]
        
        for i in range(self.epoch):
            
            if(self.gd == "stochastic_GD"):
                
                index = np.random.randint(0, x.shape[-1])
                x_ = x[index]
                y_ = y[index]
                
            elif(self.gd == "minibatch_GD"):
                
                index = np.random.randint(0, x.shape[-1], self.batch_size)
                x_ = x[[index]]
                y_ = y[[index]]
                
            Y_p = x_.dot(self.w) + self.b
            R = y_ - Y_p    
            Reg = 0
            _Reg = 0
        
## we are adding regularization methods: R1: bias = lamda*sum(weights)**2 and R2: bias = lambda*sum(abs(weights))
        
            if(self.reg == "R1"):
                Reg = self.lambda*(np.sum(np.square(w)))
                _Reg = self.lamda * w
                
            elif(self.reg =="R2"):
                Reg = self.lambda*(np.sum(np.abs(w)))
                _Reg = self.lamda * np.sign(w)
                
## changing weight and bias gradients

            w_g = (-2/n)*x_.T.dot(R)
            b_g = (-2/n)*np.sum(R)
        
            
            self.w = self.w - w_g*self.lr -  self.lr*_Reg
            self.b = self.b - b_g*self.lr
            
            cost.append((1/(2*n))*R.T.dot(R) + Reg)
            
        return cost
    
    def __test(self, x, y):
        return x.dot(self.w) + self.b
    
        
                
            
            
            
        
     
        
        
        