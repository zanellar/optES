
import os
import json 
import wandb
import torch
import numpy as np  
from optES.optimization.optims import AESPBCNetOptimizer, AESPBCGainOptimizer
 
from optES.utils.paths import PARAMS_PATH, NET_WEIGHTS_PATH
from optES.utils.plot import plot_p11, plot_energy
from optES.utils.eval import Eval
from optES.utils.pos import getpos
from optES.utils.args import Dict2Args
  
class OptProccess():

    NETWORK = 0
    GAINS = 1

    def __init__(self, 
                 params_file_name, 
                 modelclass, 
                 controllerclass, 
                 baselineclass = None, 
                 params_baseline_file_name = None, 
                 config = None,
                 optim = 0,  
                 debug = False):
        
        self.optim = optim

        if self.optim not in [self.NETWORK, self.GAINS]:
            raise ValueError("Invalid optimization option. Choose between 0 (network) and 1 (gains).")
 
        if config is None:
            self.sweep = True 
        else: 
            self.sweep = False
            self.config = Dict2Args(config["parameters"])

        self.debug = debug
  
        # Load parameters
        self.params = json.load(open(os.path.join(PARAMS_PATH, params_file_name+".json")) ) 
        print(f"Parameters: {self.params}")
 
        # Model
        self.modelclass = modelclass
        self.model = self.modelclass(self.params)

        # Controller
        self.controllerclass = controllerclass

        # Baseline
        self.baselineclass = baselineclass 
        self.params_baseline = json.load(open(os.path.join(PARAMS_PATH, params_baseline_file_name+".json")) ) 
        print(f"Baseline Parameters: {self.params_baseline}")
        self._do_evaluate_baseline = True if self.baselineclass is not None else False

    ##################################################################################
    def run(self): 

        if self.sweep: 
            wandb.init() 
            self.config = wandb.config    
 
        # Set seed
        if self.config.seed>=0:
            torch.manual_seed(self.config.seed)  
            np.random.seed(self.config.seed)

        # Evaluate the baseline controllers (only once)
        if self._do_evaluate_baseline:
            self._do_evaluate_baseline = False
            self._evaluate_baseline()

        # Create the controller
        if self.optim == self.GAINS:  
            self.controller = self.controllerclass(
                model = self.model,
                Kw = self.params["controller"]["Kw"],
                Kx = self.params["controller"]["Kx"],
                R = self.params["controller"]["R"], 
                dt = self.params["dt"], 
                Pgains = self.params["controller"] 
            )  
        if self.optim == self.NETWORK:
            self.controller = self.controllerclass(
                model = self.model,
                Kw = self.params["controller"]["Kw"],
                Kx = self.params["controller"]["Kx"],
                R = self.params["controller"]["R"],
                dim_layers = self.config.dim_layers, 
                activation = self.config.activation, 
                dt = self.params["dt"], 
                seed = self.config.seed
            )  

        self.controller.set_target(qd=getpos(self.params["target_pos"], n=self.model.n))
        
        # Visualize the initial controller  
        plot_p11(controller=self.controller, name="initial", local_plot=self.debug)
        plot_energy(model=self.model, controller=self.controller, select = ["Vd"], name="initial")

        # Train the controller Network with Gradient Descent
        if self.optim == self.NETWORK: 
            self.optimizer = AESPBCNetOptimizer(
                model = self.model,
                controller = self.controller,
                initial_pos = self.params["initial_pos"],
                target_pos = self.params["target_pos"],
                config = self.config,
                debug = self.debug
            ) 
        if self.optim == self.GAINS:
            self.optimizer = AESPBCGainOptimizer(
                model = self.model,
                controller = self.controller,
                initial_pos = self.params["initial_pos"],
                target_pos = self.params["target_pos"],
                config = self.config,
                debug = self.debug
            )
        self.optimizer.optimize()

        # Evaluate the best weights of the optimized controller
        self._evaluate_best_model()

    ##################################################################################
    def _evaluate_best_model(self):
        '''
        Evaluate the best weights of the optimized controller
        '''  
        if self.optim == self.NETWORK:
            opt_controller = self.controllerclass(
                model = self.model,
                Kw = self.controller.Kw,
                Kx = self.controller.Kx,
                R = self.controller.R,
                dim_layers = self.config.dim_layers, 
                activation = self.config.activation,  
                weights_file_name = self.optimizer.output_file_name,
                load_net = True,
                dt = self.controller.dt
            ) 
        if self.optim == self.GAINS:
            opt_controller = self.controllerclass(
                model = self.model,
                Kw = self.controller.Kw,
                Kx = self.controller.Kx,
                R = self.controller.R,  
                load_gains = True,
                dt = self.controller.dt
            )
        opt_controller.set_target(qd=getpos(self.params["target_pos"], n=self.model.n))
            
        eval = Eval(model=self.model, controller=opt_controller, save_best_result=True)
        error,steps = eval.evaluate(title="best", horizon=self.config.eval_horizon, initial_pos=self.params["initial_pos"], target_pos=self.params["target_pos"])
        print(f"Final Evaluation Error (achieved in {steps} steps)", error)

        # Visualize the optimized controller   
        plot_p11(controller=opt_controller, name="optimized", local_plot=self.debug)
        plot_energy(model=self.model, controller=opt_controller, select = ["Vd"])
    
    ##################################################################################
    def _evaluate_baseline(self): 
        '''
        Evaluate the baseline controllers
        '''  
        controller_baseline = self.baselineclass(
            model = self.model,
            Kw = self.params["controller"]["Kw"],
            Kx = self.params["controller"]["Kx"],
            R = self.params["controller"]["R"], 
            Pgains = self.params_baseline["controller"],
            dt = self.params["dt"]
        ) 
        controller_baseline.set_target(qd=getpos(self.params["target_pos"], n=self.model.n))
        # plot_energy(model=self.model, controller=controller_baseline, name="baseline1", select = ["Vd"])
  
        eval = Eval(model=self.model, controller=controller_baseline, save_best_result=False)
        error,steps = eval.evaluate(title="baseline", log=0, horizon=self.config.eval_horizon, initial_pos=self.params["initial_pos"], target_pos=self.params["target_pos"])
 
   