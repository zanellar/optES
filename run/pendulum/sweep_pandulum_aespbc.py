
import os
import json
import yaml
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from optES.systems.pendulum_phmodel import Pendulum 
from optES.optimization.optims import AESPBCGainOptimizer

from optES.systems.pendulum_controllers import PendulumGASEPBC
from optES.utils.paths import PARAMS_PATH, NET_WEIGHTS_PATH
from optES.utils.plot import plot_p11, plot_energy
from optES.utils.eval import Eval
from optES.utils.pos import getpos
  
 
class MainOptimizer():

    def __init__(self, params_file_name) -> None:
 
        # Load parameters
        self.params = json.load(open(os.path.join(PARAMS_PATH, params_file_name+".json")) )

        # Create the system
        self.model = Pendulum(self.params)

        self._do_evaluate_baseline = True

    ##################################################################################
    def run(self): 

        # Initialize wandb
        wandb.init() 

        # Set seed
        if wandb.config.seed>=0:
            torch.manual_seed(wandb.config.seed)  
            np.random.seed(wandb.config.seed)

        # Create the controller
        self.controller = PendulumGASEPBC(
            model = self.model,
            Kw = self.params["controller"]["Kw"],
            Kx = self.params["controller"]["Kx"],
            R = self.params["controller"]["R"],
            dt = self.params["dt"], 
        ) 
        self.controller.set_target(qd=getpos(self.params["target_pos"], n=self.model.n))
        # Visualize the initial controller  
        plot_p11(controller=self.controller, name="initial")
        plot_energy(model=self.model, controller=self.controller, select = ["Vd"], name="initial")

        # Train the controller Network with Gradient Descent
        self.optimizer = AESPBCGainOptimizer(
            model = self.model,  
            controller = self.controller,
            initial_pos = self.params["initial_pos"],
            target_pos = self.params["target_pos"],
            config = wandb.config
        ) 
        self.optimizer.optimize()  

        # Evaluate the best weights of the optimized controller
        self._evaluate_best_model()

        # Evaluate the baseline controllers (only once)
        if self._do_evaluate_baseline:
            self._do_evaluate_baseline = False
            self._evaluate_baseline()
    
    ##################################################################################
    def _evaluate_best_model(self):
        '''
        Evaluate the best weights of the optimized controller
        ''' 
        self.model.reset(
            q0 = getpos(self.params["initial_pos"], n=self.model.n),
            p0 = [0]
        )
        opt_controller = PendulumGASEPBC(
            model = self.model,
            Kw = self.controller.Kw,
            Kx = self.controller.Kx,
            R = self.controller.R, 
            dt = self.controller.dt,
            gains_file_name = self.optimizer.output_file_name,
            load_gains = True,  
        ) 
        opt_controller.set_target(qd=getpos(self.params["target_pos"], n=self.model.n))
            
        eval = Eval(model=self.model, controller=opt_controller, save_best_result=True)
        error,steps = eval.evaluate(horizon=wandb.config.eval_horizon, initial_pos=self.params["initial_pos"], target_pos=self.params["target_pos"])
        print(f"Final Evaluation Error (achieved in {steps} steps)", error)

        # Visualize the optimized controller   
        plot_p11(controller=opt_controller, name="optimized")
        plot_energy(model=self.model, controller=opt_controller, select = ["Vd"])
    

####################################################################################################
# Wandb Sweep
####################################################################################################

# Load wandb sweep config file
config_file_path = os.path.join(PARAMS_PATH,"optimization", "sweep_pendulum_quadratic_aespbc.yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream) 

# Initialize sweep by passing in config
sweep_id = wandb.sweep(
  sweep=config, 
  project='optES' 
) 

# Initialize main optimizer
mainopt = MainOptimizer(params_file_name="pendulum_quadratic_aespbc_uniforminit")

# Start sweep job 
wandb.agent(sweep_id, function=mainopt.run)
  
# Finish sweep job
wandb.finish()
 