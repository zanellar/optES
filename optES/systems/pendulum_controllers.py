import torch
import json
import os 
from optES.controllers.neural_aespbc import NASEPBC 
from optES.controllers.gain_aespbc import GASEPBC  
from optES.controllers.aespbc import AESPBC
from optES.controllers.ebpbc import EBPBC  
from optES.utils.paths import GAINS_PATH 


class PendulumEBPBC(EBPBC): 

    def __init__(self, model, Kp=1, Kd=1, gains_file_name="noname", load_gains=False, perfect_potential_cancellation=False, dt=0.1): 
        ''' 
            The EBPBC controller for the pendulum system.
        '''
        self.gains_file_path = os.path.join(GAINS_PATH, gains_file_name+".json")
        if load_gains: 
            f = open(self.gains_file_path) 
            gains = json.load(f)
            Kp = gains["Kp"]
            Kd = gains["Kd"]

        super().__init__(
            model = model,
            dt = dt,
            Kp = Kp,
            Kd = Kd,
            perfect_potential_cancellation = perfect_potential_cancellation,
        ) 
 

class PendulumNASEPBC(NASEPBC):
 
    def __init__(self, model, Kw=1, Kx=1, R=1, dim_layers=None, activation="relu", weights_file_name="noname", load_net=False, seed=None, dt=0.1):
        '''
            The Neural-ASEPBC controller for the pendulum system.
        '''
        
        # Set seed   
        if seed is not None:
            torch.manual_seed(seed)  
             
        dim_state = model.n
        dim_input = model.m
  
        super().__init__(
            model = model,  
            R = R*torch.eye(2*dim_state),
            Kw = Kw*torch.eye(2*dim_state),
            Kx = Kx*torch.eye(dim_input), 
            weights_file_name = weights_file_name,
            load_net = load_net,
            activation = activation, 
            dim_layers = dim_layers,
            dt = dt
        )
  
 
class PendulumASEPBC(AESPBC):

    def __init__(self, model, Kw=1, Kx=1, R=1, Pgains=dict(k0=1, k1=5), dt=0.1):
        '''
            The ASEPBC controller for the pendulum system.
        '''
        dim_state = model.n
        dim_input = model.m
 
        super().__init__(
            model = model,
            pdiag = self.pdiag,
            R = R*torch.eye(2*dim_state),
            Kw = Kw*torch.eye(2*dim_state),
            Kx = Kx*torch.eye(dim_input), 
            dt = dt
        )
        # gains for feedforward+proportional control in P11 matrix
        self.k0 = Pgains["k0"]
        self.k1 = Pgains["k1"]
        
    def pdiag(self, eq, ep): 
        ''' diagonal values of P11 control matrix'''
        p1 = self.k0 + self.k1 * eq.T @ eq
        return p1

class PendulumGASEPBC(GASEPBC):

    def __init__(self, model, Kw=1, Kx=1, R=1, Pgains=dict(k0=1, k1=5, alpha=0.1), dt=0.1, gains_file_name="noname", load_gains=False):
        '''
            The ASEPBC controller for the pendulum system.
        '''
        dim_state = model.n
        dim_input = model.m
 
        super().__init__(
            model = model, 
            R = R*torch.eye(2*dim_state),
            Kw = Kw*torch.eye(2*dim_state),
            Kx = Kx*torch.eye(dim_input), 
            dt = dt
        )
        
        self.gains = []
        self.gains_file_path = os.path.join(GAINS_PATH, gains_file_name+".json")
        if load_gains: 
            f = open(self.gains_file_path) 
            gains = json.load(f)
            self.gains.append(torch.tensor(gains["k0"], dtype=torch.float))
            self.gains.append(torch.tensor(gains["k1"], dtype=torch.float))
        else:  
            self.gains.append(torch.tensor(Pgains["k0"], dtype=torch.float))
            self.gains.append(torch.tensor(Pgains["k1"], dtype=torch.float))

    def pdiag(self, eq, ep): 
        ''' diagonal values of P11 control matrix'''
        p1 = self.gains[0] + self.gains[1] * eq.T @ eq
        return p1


