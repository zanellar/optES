import numpy as np
import torch

from optES.utils.ops import root_finder, intregral_approximation


class PHSystemCanonic():

    SAMPLE_VALUE_DISCRETE_GRADIENT = 0
    MEAN_VALUE_DISCRETE_GRADIENT = 1


    def __init__(self, D, B, params, verbose=False):
        '''
        This class implements a generic discrete-time port-Hamiltonian system in canonic form (x=[q,p]) using PyTorch.

        The system is defined by the following equations:
            x(k+1) = x(k) + dt * F * ddH(x(k),x(k+1)) + dt * G * u(k)
            y(k) = G' * ddH(x(k),x(k+1))

        where:
        - x(k) = [q(k),p(k)] is the state vector
        - u(k) is the input vector
        - y(k) is the output vector
        - F = [0, I; -I, -D] is the state transition matrix
        - G = [0; B] is the input matrix
        - ddH(x(k+1),x(k))=\int_{0}^{1}dH(s*x(k+1)+(1-s)*x(k))ds is the mean value dicrete gradient of the Hamiltonian H(x) between x(k) and x(k+1)
            with dH = [dHdq, dHdp]

        The system is implemented using PyTorch and the (implicit) state update equation is solved using the Newton-Raphson method.

        The system is initialized with the following parameters:
        - D: matrix n x n of the state transition matrix F (list)
        - B: matrix n x m of the input matrix G (list)
        - dt: sampling time
        - verbose: if true, prints "q" and "p" every step
        '''
        self.verbose = verbose
        self.params = params

        self.n = np.asarray(D).shape[0] # number of coordinates
        if len(np.asarray(B).shape)>1:
            self.m = np.asarray(B).shape[1] # number of inputs
        else:
            self.m = 1 # number of inputs (assuming there is only one input)

        self.dt = torch.tensor(params["dt"]).type(torch.FloatTensor)

        # auxiliary matrices
        self.On = torch.zeros((self.n, self.n)).type(torch.FloatTensor)
        self.In = torch.eye(self.n).type(torch.FloatTensor)

        # state transition matrix
        self.D = torch.tensor(D).reshape((self.n, self.n)).type(torch.FloatTensor)
        self.F = torch.stack([
            torch.stack([self.On, self.In], dim=1),
            torch.stack([-self.In, -self.D], dim=1)
        ]).reshape((2*self.n, 2*self.n)).type(torch.FloatTensor)

        # input matrix
        self.B = torch.tensor(B).reshape((self.n, self.m)).type(torch.FloatTensor)
        self.G = torch.stack([self.On, self.B]).reshape((2*self.n, self.m)).type(torch.FloatTensor)

        self.q = None # current state
        self.p = None # current state

        self.y = None # output

    def reset(self, q0=None, p0=None):
        '''
        Reset the system to the initial state.
        '''  
        self.q = torch.tensor(np.array(q0)) if not isinstance(q0, torch.Tensor) else q0.clone() 
        self.q = self.q.reshape((self.n,1)).float()

        self.p = torch.tensor(np.array(p0)) if not isinstance(p0, torch.Tensor) else p0.clone()
        self.p = self.p.reshape((self.n,1)).float()
  
        self.x = torch.cat((self.q,self.p)).float()
        self.u = torch.zeros((self.m,1)).float()

        next_x = root_finder(self._eq, self.x, params=self.params)
        ddH = self._disc_dH(self.x, next_x)
        self.y = self.G.T @ ddH

    def H(self,q,p):
        ''' Hamiltonian of the system '''
        return self.K(q,p) + self.V(q)

    def _dH(self,q,p):
        ''' gradient of the Hamiltonian''' 
        # q.requires_grad_(True)
        # p.requires_grad_(True) 
        # H = self.H(q,p)
        # H.backward()
        # q.requires_grad_(False)
        # p.requires_grad_(False)
        # return torch.cat((q.grad, p.grad))
        
        q_ = q.requires_grad_(True)
        p_ = p.requires_grad_(True) 
        dHdq = torch.autograd.grad(self.H(q_,p_), q_)[0]
        dHdp = torch.autograd.grad(self.H(q_,p_), p_)[0]
  
        return torch.cat((dHdq, dHdp))

    def _disc_dH(self, x1, x2):
        '''
        Mean value dicrete gradient of the Hamiltonian H(x) between x(k) and x(k+1)
        @ param x1: torch tensor of shape (n,) representing the starting point, i.e. x(t)
        @ param x2: torch tensor of shape (n,) representing the ending point, i.e. x(t+1)
        @ param samples: number of sample points used to compute the integral with trapezoidal rule.
        '''
        q2, p2 = torch.split(x2,self.n)
        q2 = q2.reshape((self.n,1))
        p2 = p2.reshape((self.n,1))

        q1, p1 = torch.split(x1,self.n)
        q1 = q1.reshape((self.n,1))
        p1 = p1.reshape((self.n,1))

        # integration range [a,b] = [0,1]
        a = torch.tensor(0.)
        b = torch.tensor(1.)

        def f(s):
            iq = s * q1 + (1 - s) * q2
            ip = s * p1 + (1 - s) * p2
            dH = self._dH(iq, ip)
            return dH

        disc_dH = intregral_approximation(f, a, b, params=self.params )
        disc_dH.reshape((2*self.n,1))
        return disc_dH

    def _eq(self,next_x):
        ''' eqution to be solved in in x(t+1) '''

        # compute the discrete gradient of the Hamiltonian
        ddH = self._disc_dH(self.x, next_x)

        # implicit eqaution in x(t+1):
        # x(t+1) - x(t) - dt*F*ddH(x(t),x(t+1)) - dt*G*u = 0
        e = next_x - ( self.x + self.dt *  self.F @ ddH + self.dt * self.G @ self.u)

        return e.reshape((2*self.n,1))
    

    def update_state(self, disc_grad):
        ''' Update the state of the system to the next time instant. 
        @ param disc_grad: if "1" the discrete gradient is computed as the mean value of the continuous gradient between x(t) and x(t+1). 
                            In this case the model is an implicit equation which requires numerical solution.
                           if "0" the discrete gradient is computed as the value of the continuous gradient at x(t).
                            In this case the model is an explicit equation which can be solved analytically (but loses accuracy).
        ''' 
 
        if disc_grad == self.MEAN_VALUE_DISCRETE_GRADIENT:   

            _next_x_sol = root_finder(self._eq, self.x, params=self.params) 

            return _next_x_sol
        
        else:  
            
            # compute the continuous gradient of the Hamiltonian
            q, p = torch.split(self.x, self.n)
            q = q.reshape((self.n,1))
            p = p.reshape((self.n,1))
            dH = self._dH(q, p) 

            # approximated model with continuous gradient 
            next_x = self.x + self.dt *  self.F @ dH + self.dt * self.G @ self.u 

            return next_x

    def step(self, u, disc_grad=1):
        '''
        Simulate the system for one step.
        @ param u: input (numpy.array) shape (m,)
        @ return: state x (numpy.array) shape (n,) and output y (numpy.array) shape (mn,)
        '''
         
        # control input
        if type(u) is not torch.Tensor:
            # print(f"input 'u'={u} IS NOT torch.Tensor")
            u = torch.tensor(u)

        self.u = u.reshape(self.m,1).type(torch.FloatTensor)

        # current state
        self.x = torch.cat((self.q,self.p))

        # next state
        next_x = self.update_state(disc_grad=disc_grad)

        # output
        ddH = self._disc_dH(self.x, next_x)
        self.y = self.G.T @ ddH

        # update current state
        self.x = next_x.clone()
        self.q, self.p = torch.split(self.x,self.n)

        if self.verbose:
            print(f"q={self.q.flatten().detach().numpy()}, p={self.p.flatten().detach().numpy()}")

        x = self.x.clone().flatten().detach().numpy()
        y = self.y.clone().flatten().detach().numpy()
        return x,y
