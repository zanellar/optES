import torch  
import wandb
import time
import os
from optES.utils.paths import NET_WEIGHTS_PATH 
from optES.optimization.optimbase import BaseOptimizer

   
############################################################################################################
############################################################################################################
############################################################################################################

class AESPBCNetOptimizer(BaseOptimizer):

    def __init__(self, 
                 model, 
                 controller,  
                 initial_pos, 
                 target_pos,
                 config,
                 debug = False
                 ): 
         
        super().__init__(model = model, 
                         controller = controller,
                         params2optimize = controller.net.parameters, 
                         cost_type = 'l2', 
                         initial_pos = initial_pos,
                         target_pos = target_pos,
                         config = config,
                         debug = debug
                         )
        
        # Wandb
        if self.controller.net is torch.nn.Module:
            wandb.watch(models=self.controller.net, criterion=self.objective, log='all', log_freq=100)
        
    def get_current_result(self):
        ''' Get the current network weights'''
        return self.controller.net.state_dict()
 
    def additional_log(self):
        ''' Log the current network weights'''
        return {} 
    
    def optimize(self):
        super().optimize()
        
        # Save the network weights
        formatted_date = time.strftime("%Y%m%d", time.localtime()) 
        formatted_dim_layers = "_".join([str(dim) for dim in self.config.dim_layers])
        formatted_activation = self.config.activation 
        self.output_file_name = "naespbc_" + formatted_date + "_" + formatted_dim_layers + "_" + formatted_activation + "_" + wandb.run.name  
        self.output_file_path = os.path.join(NET_WEIGHTS_PATH, self.output_file_name + ".pth") 
        torch.save(self.best_res, self.output_file_path)
        print(f"Saved network weights in {self.output_file_path}")

        # artifact = wandb.Artifact(name=save_weights_file_name, type='model')
        # artifact.add_file(save_weights_file_name.split('.')[0]))
        # wandb.log_artifact(artifact)
         
        if self.debug:
            for param_tensor in self.controller.net.state_dict():
                print(param_tensor, "\t", self.controller.net.state_dict()[param_tensor].size())
    

############################################################################################################
############################################################################################################
############################################################################################################

class AESPBCGainOptimizer(BaseOptimizer):

    def __init__(self, 
                 model, 
                 controller,  
                 initial_pos, 
                 target_pos,
                 config,
                 debug = False
                 ): 
        
        # enable gradient in gains
        for param in controller.gains:
            param.requires_grad = True

        def params2optimize():
            return controller.gains

        super().__init__(model = model, 
                         controller = controller,
                         params2optimize = params2optimize, 
                         cost_type = 'l2', 
                         initial_pos = initial_pos,
                         target_pos = target_pos,
                         config = config,
                         debug = False
                         )
         
    def clamp(self):
        ''' Clamp the parameters to positive values''' 
        for param in self.params2optimize():
            param.data.clamp_(min=0.0) 

    def get_current_result(self):
        ''' Get the current gains'''
        res = {}
        for i, param in enumerate(self.params2optimize()):
            res["k"+str(i)] = param.item()
        return res
     
    def additional_log(self):
        ''' Log the current gains'''
        return self.get_current_result()
    
    def optimize(self):
        super().optimize()
         
        formatted_date = time.strftime("%Y%m%d", time.localtime())
        formatted_cost = format(self.cost.item(), ".0e")
        self.output_file_name = "aespbc_" + wandb.run.name + "_" + formatted_date + "_" + formatted_cost 
        self.output_file_path = os.path.join(GAINS_PATH, self.output_file_name + ".json")  
        
        # Save the gains 
        with open(self.output_file_path, 'w') as f:
            json.dump(self.best_res, f)
  

############################################################################################################
############################################################################################################
############################################################################################################

from optES.controllers.ebpbc import EBPBC
from optES.utils.paths import GAINS_PATH 
import json

class EBPBCGainOptimizer(BaseOptimizer):

    def __init__(self, 
                 model,   
                 target_pos, 
                 initial_pos,  
                 config,
                 debug = False
                 ): 
        
        self.target = torch.tensor(target_pos) 
        
        # Gains to optimize
        self.Kp = torch.rand((1,), requires_grad=True)
        self.Kd = torch.rand((1,), requires_grad=True)
        # self.Kp = torch.zeros((1,), requires_grad=True)
        # self.Kd = torch.zeros((1,), requires_grad=True)

        # Controller # TODO: replace with PendulumEBPBC passed from outside as generic controller 
        controller = EBPBC(model, Kp=self.Kp, Kd=self.Kd, perfect_potential_cancellation=True, dt=0.1) # TODO dt hard coded
        controller.set_target(qd=[self.target]) 

        def params2optimize():
            return [self.Kp,self.Kd]
 
        super().__init__(model = model, 
                         controller = controller,
                         params2optimize = params2optimize,
                         cost_type = 'l1', 
                         initial_pos = initial_pos,
                         target_pos = target_pos,
                         config = config,
                         debug = False
                         ) 
        
    def clamp(self):
        ''' Clamp the parameters to positive values'''
        self.Kp.data.clamp_(min=0.0)
        self.Kd.data.clamp_(min=0.0) 

    def get_current_result(self):
        ''' Get the current gains'''
        return dict(kp=self.Kp.item(), kd=self.Kd.item())
    
    def additional_log(self):
        ''' Log the current gains'''
        return self.get_current_result()

    def optimize(self):
        super().optimize()
        
        formatted_date = time.strftime("%Y%m%d", time.localtime())
        formatted_cost = format(self.cost.item(), ".0e")
        self.output_file_name = "ebpbc_" + wandb.run.name + "_" + formatted_date + "_" + formatted_cost 
        self.output_file_path = os.path.join(GAINS_PATH, self.output_file_name + ".json")

        # Save the gains
        with open(self.output_file_path, 'w') as f:
            json.dump(dict(Kp=self.best_res["kp"],Kd=self.best_res["kd"]), f)

