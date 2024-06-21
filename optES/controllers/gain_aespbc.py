import torch
import os 
from optES.controllers.aespbc import AESPBC 
from optES.utils.nn import NeuralNet
from optES.utils.paths import NET_WEIGHTS_PATH  

class GASEPBC(AESPBC):
 
    def __init__(self, model, Kw=1, Kx=1, R=1, dt=0.1):
        ''' 
        Gains-based Algebric Energy Shaping Passivity-Based Controller (GASEPBC)

        @param model: model of the system
        @param Kw: control gain
        @param Kx: control gain
        @param R: control gain 
        @param dt: control sampling time. Default: 0.001
        '''

        # Model  
        self.model = model 
        dim_state = self.model.n
        dim_input = self.model.m

        # AESPBC init  
        super().__init__(
            model = self.model,
            pdiag = self.pdiag,
            R = R,
            Kw = Kw,
            Kx = Kx,
            dt = dt
        )
    
    def freeze_gains(self):
        ''' Freeze the network parameters ''' 
        if self.gains is not None:  
            for param in self.gains: 
                param.requires_grad_(False)

    def unfreeze_gains(self):
        ''' Unfreeze the network parameters '''
        if self.gains is not None:  
            for param in self.gains: 
                param.requires_grad_(True)

    def _dHa(self,x,w):
        ''' [OVERWRITE] gradient of the Residual Hamiltonian'''
        x.requires_grad_(True)
        w.requires_grad_(True)

        # We need to freeze the neural network to AVOID computing and updating 
        # its gradients in the backpropagation step of Ha(x,w)
        self.freeze_gains()

        # Backpropagation step (copute the gradient of Ha(x,w) w.r.t. x and w)
        self.Ha(x,w).backward()

        # We unfreeze the neural network to ALLOW computing and updating 
        # its gradients in the optimization process
        self.unfreeze_gains()
        return torch.cat((x.grad, w.grad))

    def get_action(self,q,p):  
        ''' [OVERWRITE] Get the control action'''

        # Set the model state to the current system state
        self.set_model_state(q,p) 
        
        # We need to set the model in evaluation mode and freeze the parameters to AVOID 
        # computing gradients and updating the weights of the neural network during the
        # Newton-Raphson iterations in the model_prediction() method.
        self.freeze_gains()

        # Compute the next state 
        self.model_prediction()  

        # We restore the model in training mode and unfreeze the parameters to ALLOW
        # computing gradients and updating the weights of the neural network 
        self.unfreeze_gains()

        u = self._beta(self._x, self._next_x) + self._v(self._x, self._next_x, self._w, self._next_w)  
        return u 