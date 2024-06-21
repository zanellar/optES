
import torch
import numpy as np 
from optES.utils.ops import intregral_approximation


class EBPBC():
    def __init__(self, model, Kp, Kd, dt, perfect_potential_cancellation=False):
        '''
        This class implements a generic Passivity-Based Control with Energy Shaping.
        
        @ param model: model of the system
        @ param Kp: proportional control gain
        @ param Kd: derivative control gain.
        @ param dt: controller sampling time
        @ param perfect_potential_cancellation: if true, the potential energy is perfectly cancelled
        ''' 

        self.model = model
        self.dt = dt
        self.Kp = Kp #torch.abs(Kp)
        self.Kd = Kd #torch.abs(Kd)
        self.q = None
        self.perfect_potential_cancellation = perfect_potential_cancellation

    def set_target(self, qd):
        ''' set target position'''
        self.qd = torch.tensor(qd).reshape(self.model.n,1).type(torch.FloatTensor)
        self.pd = torch.tensor(0).reshape(self.model.n,1).type(torch.FloatTensor)
        self.xd = torch.cat((self.qd,self.pd),dim=0)

    def reset(self,q0):
        self.q = q0 

    def _dVdq(self,q):
        ''' Derivative of potential energy with respect to q '''
        # self.model.V(q).backward()
        # return q.grad
        return self.model._dVdq(q)


    def _disc_dVdq(self, q1, q2):
        '''Mean value dicrete gradient of V(q) using trapezoidal rule'''

        # integration range [a,b] = [0,1]
        a = torch.tensor(0.)
        b = torch.tensor(1.)

        def f(s):
            iq = s * q1 + (1 - s) * q2
            dVdq = self._dVdq(iq)
            return dVdq

        disc_dVdq = intregral_approximation(f, a, b, params=self.model.params)

        ''' We need to add an "epsilon" if we use the controller in a system that
            perfectly matches the model, to avoid complete cancellation of the term
            (if fact also disc_dVdq is subject to the optimization, and a complete
            cancellation whould end in a vanishing of the gradient)'''
        if self.perfect_potential_cancellation:
            disc_dVdq = disc_dVdq + 1e3*torch.finfo().eps

        return disc_dVdq

    def energy_shaping(self,q):
        B = self.model.B
        q_old = self.q
        disc_dVdq = self._disc_dVdq(q_old , q )
        beta = - torch.linalg.inv(B) * ( - disc_dVdq + 0.5 * self.Kp * (q_old + q - 2*self.qd))
        return beta

    def dumping_injection(self,y):
        v = self.Kd * y
        return v

    def get_action(self,q,y):
        u = self.energy_shaping(q) + self.dumping_injection(y)
        self.q = q
        return u
