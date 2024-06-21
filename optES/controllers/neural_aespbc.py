import torch
import os 
from optES.controllers.aespbc import AESPBC 
from optES.utils.nn import NeuralNet
from optES.utils.paths import NET_WEIGHTS_PATH  

class NASEPBC(AESPBC):
 
    def __init__(self, model, dim_layers, activation, weights_file_name="noname", load_net=False, Kw=1, Kx=1, R=1, dt=0.1):
        ''' 
        Neural Algebric Energy Shaping Passivity-Based Controller (NASEPBC)

        @param model: model of the system
        @param Kw: control gain
        @param Kx: control gain
        @param R: control gain
        @param dim_layers: list with the number of neurons for each layer of the neural network (e.g. [2,2,2]). Default: None (pdiag k0 + k1 * e.T @ e)
        @param activation: activation function for the neural network. Default: "relu" 
        @param weights_file_name: name of the neural network. Default: "noname"
        @param load_net: load the neural network from the weights file. Default: False
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
  
        # neural network
        self.eval = False
        self.net = None
        if dim_layers is not None and dim_layers != []:
            self.net = NeuralNet(
                # dim_input = dim_state*2,
                dim_input = dim_state,
                dim_output = dim_state,
                dim_layers = dim_layers,
                activation = activation 
            )   
        else: # de-BUG temrporary else branch (replacing the neural network with a quadratic function)

            class ModelFoo():
                def __init__(self,k0,k1):
                    self.k0 = k0
                    self.k1 = k1
                    self.state_dict = lambda : {"k0":self.k0, "k1":self.k1}
                    self.parameters = lambda : [self.k0,self.k1]
                    self.eval = lambda : None
                    self.train = lambda : None

            class QuadFuc():
                def __init__(self):
                    self.k0 = torch.tensor([1.], requires_grad=True)
                    self.k1 = torch.tensor([1.], requires_grad=True)
                    self.model = ModelFoo(self.k0,self.k1)

                def __call__(self, eq):
                    p1 = torch.exp(self.k0) + torch.exp(self.k1) * eq.T @ eq
                    return p1
                
            self.net = QuadFuc()

        self.weights_file_name = weights_file_name
        net_file_path = os.path.join(NET_WEIGHTS_PATH, self.weights_file_name+".pth")

        if load_net: 
            print(f"LOADING {net_file_path}")
            self.net.load_state_dict(torch.load(net_file_path)) 
            self.net.eval()
 
    def freeze_net(self):
        ''' Freeze the network parameters ''' 
        if self.net is not None: 
            self.net.eval()  
            for param in self.net.parameters():
                param.requires_grad_(False)

    def unfreeze_net(self):
        ''' Unfreeze the network parameters '''
        if self.net is not None: 
            self.net.train() 
            for param in self.net.parameters():
                param.requires_grad_(True)

    def set_eval_mode(self):
        ''' Set the network in evaluation mode '''
        self.eval = True
        self.freeze_net()

    def set_train_mode(self):
        ''' Set the network in training mode '''
        self.eval = False
        self.unfreeze_net()

    def _dHa(self,x,w):
        ''' [OVERWRITE] gradient of the Residual Hamiltonian''' 

        # with torch.set_grad_enabled(True):
            
        x.requires_grad_(True)
        w.requires_grad_(True)

        # We need to freeze the neural network to AVOID computing and updating 
        # its gradients in the backpropagation step of Ha(x,w)
        self.freeze_net()

        # Backpropagation step (copute the gradient of Ha(x,w) w.r.t. x and w)
        Ha = self.Ha(x,w)
        Ha.backward()

        # We unfreeze the neural network to ALLOW computing and updating 
        # its gradients in the optimization process
        self.unfreeze_net() if not self.eval else self.freeze_net()

        x.requires_grad_(False)
        w.requires_grad_(False)
        
        return torch.cat((x.grad, w.grad))

    def get_action(self, q, p):  
        ''' [OVERWRITE] Get the control action'''

        # Set the model state to the current system state
        self.set_model_state(q,p) 
        
        # We need to set the model in evaluation mode and freeze the parameters to AVOID 
        # computing gradients and updating the weights of the neural network during the
        # Newton-Raphson iterations in the model_prediction() method.
        self.freeze_net()

        # Compute the next state 
        self.model_prediction()  

        # We restore the model in training mode and unfreeze the parameters to ALLOW
        # computing gradients and updating the weights of the neural network 
        self.unfreeze_net() if not self.eval else self.freeze_net()

        u = self._beta(self._x, self._next_x) + self._v(self._x, self._next_x, self._w, self._next_w) 
        return u

    #####################################################################################################

    def pdiag(self, eq, ep): 
        ''' computes the diagonal matrix P11 using a neural network and returns it as a vector '''
        eq = eq.view(-1,self.model.n) 
        ep = ep.view(-1,self.model.n) 
        ex = eq#torch.cat((eq,ep),1)
        p1 = self.net(ex).view(-1,self.model.n)   
        return p1


