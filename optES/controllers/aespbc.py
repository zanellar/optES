import numpy as np
import torch
from optES.utils.ops import root_finder, intregral_approximation, transpose


class AESPBC():

    def __init__(self, model, pdiag, R, Kx, Kw, dt, verbose=False):
        '''
        This class implements a generic Passivity-Based Control with Algebric Energy Shaping.
 
        @ param pdiag: function that returns a list of values in the diagonal of the P matrix (torch function) 
        @ param R: control gain (torch matrix)
        @ param Kx: control gain (torch matrix)
        @ param Kw: control gain (torch matrix)
        @ param dt: sampling time
        @ param verbose: if true, prints "q" and "p" every step
        '''
        self.verbose = verbose

        # model 
        self.model = model 
        self.D = self.model.D
        self.F = self.model.F 
        self.B = self.model.B
        self.G = self.model.G

        self.n = np.asarray(self.D).shape[0] # number of coordinates
        if len(np.asarray(self.B).shape)>1:
            self.m = np.asarray(self.B).shape[1] # number of inputs
        else:
            self.m = 1 # number of inputs (assuming there is only one input)

        self.dt = torch.tensor(dt).float()
  
        # control parameters
        self.Kw = Kw
        self.Kx = Kx
        self.R = R  
        self.pdiag = pdiag
   
        # auxiliary matrices
        self.On = torch.zeros((self.n, self.n)).float()
        self.In = torch.eye(self.n).float()

    def set_target(self, qd):
        ''' Set target configuration'''
        self.qd = torch.tensor(qd).float().reshape(self.n,1) 
        self.pd = torch.zeros((self.n,1)).float() 
        self.xd = torch.cat((self.qd,self.pd))

    def set_model_state(self, q, p):
        ''' Set the state '''
 
        q = torch.tensor(np.array(q)) if not isinstance(q, torch.Tensor) else q.clone() 
        q = q.reshape((self.n,1)).float()
        eq = q - self.qd

        p = torch.tensor(np.array(p)) if not isinstance(p, torch.Tensor) else p.clone()
        p = p.reshape((self.n,1)).float()
        ep = p - self.pd
 
        self._x = torch.cat((eq,ep))
        self._w = 0.5*self._x
        
        self._next_x, self._next_w = self._x, self._w

    def reset(self, *args, **kwargs):
        ''' Reset the controller (do nothing) '''
        pass

    def computation_settings(self, max_iter, samples):
        self.max_iter = max_iter
        self.samples = samples

    # def H(self,q,p):
    #     ''' Hamiltonian of the system ''' 
    #     return self.model.K(q,p) + self.model.V(q)

    def _dHdx(self,q,p):
        ''' gradient of the Hamiltonian'''  
        # q.requires_grad_(True)
        # p.requires_grad_(True) 
        # H = self.H(q,p)
        # H.backward()
        # q.requires_grad_(False)
        # p.requires_grad_(False)
        # return torch.cat((q.grad, p.grad))
        return self.model._dH(q,p) 

    def _disc_dHdx(self, x1, x2, samples=10):
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
            dHdx = self._dHdx(iq.clone().detach(), ip.clone().detach())
            return dHdx

        disc_dHdx = intregral_approximation(f, a, b, params=self.model.params)
        disc_dHdx.reshape((2*self.n,1))
        return disc_dHdx

    def Ha(self,x,w, log=False):
        ''' Residual Hamiltonian with potential compensation''' 
        x = x.reshape((2*self.n,1))
        w = w.reshape((2*self.n,1))
        q, _ = torch.split(x, self.n)
        return - self.model.V(q) + 0.5 * transpose(x) @ self._P(w) @ x + 0.5 * transpose(x-w) @ self.R @ (x-w)

    def _dHa(self,x,w):
        ''' gradient of the Residual Hamiltonian'''
        # with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        w.requires_grad_(True)
        Ha = self.Ha(x,w)
        Ha.backward()
        x.requires_grad_(False)
        w.requires_grad_(False)
        return torch.cat((x.grad, w.grad))

    def _disc_dHa(self, x1, x2, w1, w2, samples=10):
        '''
        Mean value dicrete gradient of the residual Hamiltonian Ha(x,w) between [x(k),w(k)] and [x(k+1),w(k+1)]
        @ param x1: torch tensor of shape (n,) representing the starting point, i.e. x(k)
        @ param x2: torch tensor of shape (n,) representing the ending point, i.e. x(k+1)
        @ param w1: torch tensor of shape (n,) representing the starting point, i.e. w(k)
        @ param w2: torch tensor of shape (n,) representing the ending point, i.e. w(k+1)
        @ param samples: number of sample points used to compute the integral with trapezoidal rule.
        '''
        # integration range [a,b] = [0,1]
        a = torch.tensor(0.)
        b = torch.tensor(1.)

        def f(s):
            ix = s * x1 + (1 - s) * x2
            iw = s * w1 + (1 - s) * w2
            dHa = self._dHa(ix.clone().detach(), iw.clone().detach())
            return dHa

        disc_dHa = intregral_approximation(f, a, b, params=self.model.params)

        return disc_dHa

    def _disc_dH_target(self,x1,x2,w1,w2):
        '''
        discrete gradient of the desired Hamiltonian Hd(x,w) = H(x) + Ha(x,w)
        @ param x1: torch tensor of shape (n,) representing the starting point, i.e. x(k)
        @ param x2: torch tensor of shape (n,) representing the ending point, i.e. x(k+1)
        @ param w1: torch tensor of shape (n,) representing the starting point, i.e. w(k)
        @ param w2: torch tensor of shape (n,) representing the ending point, i.e. w(k+1)
        @ return: touple of discrete gradients dH/dx and dH/dw
        '''
        # discrete gradient of the original-model Hamiltonian "H"
        disc_dHdx = self._disc_dHdx(x1 + self.xd, x2 + self.xd)

        # discrete gradient of the extended-model residual Hamiltonian "Ha"
        disc_dHa = self._disc_dHa(x1, x2, w1, w2)
        disc_dHadx, disc_dHadw = torch.split(disc_dHa,2*self.n)

        # discrete gradient of the extended-model desired Hamiltonian "Hd"
        disc_dHdx_target = disc_dHdx + disc_dHadx
        disc_dHdw_target = disc_dHadw

        return disc_dHdx_target, disc_dHdw_target

    def _equation(self, next_z):
        ''' equation to be solved in z(t)=[x(t+1),w(t+1)] '''

        next_x, next_w = torch.split(next_z, 2*self.n)
        next_x = next_x.reshape((2*self.n,1))
        next_w = next_w.reshape((2*self.n,1))

        # discrete gradient of the desired Hamiltonian "Hd" of the extended-model
        disc_dHdx_target, disc_dHdw_target = self._disc_dH_target(self._x, next_x, self._w, next_w)
        disc_dHdx = self._disc_dHdx(self._x + self.xd, next_x + self.xd)

        # control input
        u = self._beta(self._x, next_x) + self._v(self._x, next_x, self._w, next_w)  

        # implicit equation in [x(t+1), w(t+1)]
        ex = next_x - (self._x + self.dt * self.F @ disc_dHdx + self.dt * self.G @ u)
        ew = next_w - (self._w - self.dt * self.Kw @ disc_dHdw_target)

        e = torch.cat((ex,ew))
        return e.reshape((4*self.n,1))

    def model_prediction(self):
        ''' One-step prediction of the extended model dynamics '''
        # solve implicit state eqaution using Newton-Raphson method
        # initialized with the current state [x(t), w(t)]
 
        res = root_finder(
            f = self._equation,
            x0 = torch.cat((self._x,self._w)),
            params=self.model.params 
        )
        
        self._next_x, self._next_w = torch.split(res,2*self.n)  
 

    def _dVdq(self,q):
        ''' Derivative of potential energy with respect to q ''' 
        # q.requires_grad_(True)
        # V = self.model.V(q)
        # V.backward()
        # dVdq = q.grad 
        # q.requires_grad_(False) 
        # return dVdq  
        return self.model._dVdq(q)

    def _disc_dVdq(self, x1, x2, samples=10):
        '''Mean value dicrete gradient'''
        q1, _ = torch.split(x1, self.n)
        q1 = q1.reshape((self.n,1))
        q2, _ = torch.split(x2, self.n)
        q2 = q2.reshape((self.n,1))

        # integration range [a,b] = [0,1]
        a = torch.tensor(0.)
        b = torch.tensor(1.)

        def f(s):
            iq = s * q1 + (1 - s) * q2
            dVdq = self._dVdq(iq.clone().detach())
            return dVdq

        disc_dVdq = intregral_approximation(f, a, b, params=self.model.params)

        return disc_dVdq

    def _P(self, x):
        eq, ep = torch.split(x, self.n)
        eq = eq.reshape((self.n,1))
        ep = ep.reshape((self.n,1))   
        return torch.block_diag(
                    torch.diag(self.pdiag(eq,ep).flatten()),
                    self.On
                    )

    def _beta(self,x,next_x):
        ''' Energy shaping term'''
        Pb = 0.5*(self._P(x) + self._P(next_x))
        disc_dVdq = self._disc_dVdq(x + self.xd, next_x + self.xd)
        Ka = 0.5 * Pb @ (x + next_x) + torch.cat((-disc_dVdq, torch.zeros((self.n,1)).float())) 
        beta = torch.linalg.pinv(self.G) @ self.F @ Ka
        return beta

    def _v(self,x, next_x, w, next_w):
        ''' Damping term '''
        disc_dHdx_target, _ = self._disc_dH_target(x, next_x, w, next_w) 
        v = - self.Kx @ self.G.T @ disc_dHdx_target
        return v
     
    def get_action(self, q, p):  
        ''' Get the control action'''

        # Set the model state to the current system state
        self.set_model_state(q,p) 
         
        # Compute the next state 
        self.model_prediction() 
  
        u = self._beta(self._x, self._next_x) + self._v(self._x, self._next_x, self._w, self._next_w) 
        return u
