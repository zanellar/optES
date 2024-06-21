
import torch
from optES.basemodels.phsys import PHSystemCanonic


class Pendulum(PHSystemCanonic):

    g = 9.81

    def __init__(self, params):
        '''
        Pendulum system in the vertical plane. Implemented using the Hamiltonian formulation. Solving the system using the Newton-Raphson method.
        The system is initialized with the following parameters:
        - m: mass of the pendulum
        - r: length of the pendulum
        - k: spring constant
        - b: damping coefficient
        - dt: sampling time
        '''
        self.params = params
        self.m = torch.tensor(params["model"]["m"]).type(torch.FloatTensor)
        self.r = torch.tensor(params["model"]["r"]).type(torch.FloatTensor)
        self.b = torch.tensor(params["model"]["b"]).type(torch.FloatTensor)
        self.k = torch.tensor(params["model"]["k"]).type(torch.FloatTensor)

        self.M = self.m * self.r**2

        super().__init__(
            D = [self.b],
            B = [1],
            params = params 
        )

    def V(self, q):
        ''' Potential energy of the pendulum system''' 
        V = self.m * self.g * self.r * (1. - torch.cos(q)) + 0.5 * self.k * q**2
        return V

    def K(self,q,p):
        ''' Kinetic energy of the pendulum''' 
        K = 0.5 * p**2 / self.M
        return K
 

    def _dVdq(self, q): # TODO auto-differentiation
        ''' Derivative of potential energy with respect to q '''
        return self.m * self.g * self.r * torch.sin(q) + self.k * q

    # def _dH(self, q, p): # TODO auto-differentiation 
    #     ''' gradient of the Hamiltonian with respect to generalized coordinates "q"'''
    #     dVdq = self._dVdq(q)
    #     dVdp = torch.zeros_like(p)

    #     dKdq = torch.zeros_like(q)
    #     dKdp = p/self.M

    #     dHdq = dVdq + dKdq
    #     dHdp = dVdp + dKdp 

    #     return torch.cat((dHdq, dHdp))