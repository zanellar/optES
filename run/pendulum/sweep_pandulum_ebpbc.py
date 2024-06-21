import json
import os
import wandb
import yaml
import numpy as np
import matplotlib.pyplot as plt
from optES.systems.pendulum_phmodel import Pendulum 
from optES.optimization.optims import EBPBCGainOptimizer
from optES.systems.pendulum_controllers import PendulumEBPBC
from optES.utils.paths import PARAMS_PATH, GAINS_PATH
from optES.utils.eval import Eval
from optES.utils.pos import getpos
   
best_result = {"cost":np.inf, "eval/error":np.inf, "eval/steps":np.inf, "run":""}

def main():

    # Initialize a new wandb run
    wandb.init(group='ebpbc_gains_optim', project='optES') 
 
    # Load parameters 
    params = json.load(open(os.path.join(PARAMS_PATH,"pendulum_ebpbc.json")) )

    # Initialize model
    model = Pendulum(params) 

    # Tune gain with Gradient Descent
    optimizer = EBPBCGainOptimizer(
        model = model,  
        initial_pos = params["initial_pos"],
        target_pos = params["target_pos"],
        config = wandb.config
    ) 
    optimizer.optimize() 

    # Evaluate the optimized controller 
    model.reset(
        q0 = getpos(params["initial_pos"], n=model.n),
        p0 = [0]
    )
    opt_controller = PendulumEBPBC(
        model, 
        gains_file_name = optimizer.output_file_name,
        perfect_potential_cancellation = True,
        load_gains = True,
        dt = params["dt"]
    )
    opt_controller.set_target(qd = getpos(params["target_pos"], n=model.n))
    opt_controller.reset(q0 = model.q)
     
    eval = Eval(model=model, controller=opt_controller, save_best_result=True)
    error,steps = eval.evaluate(horizon=wandb.config.eval_horizon, initial_pos=params["initial_pos"], target_pos=params["target_pos"])
    print(f"\nFinal Evaluation Error (achieved in {steps} steps)\n\n", error)



####################################################################################################
# Wandb Sweep
####################################################################################################

# Load wandb sweep config file
config_file_path = os.path.join(PARAMS_PATH,"optimization", "sweep_pendulum_ebpbc0.yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream) 
 
# Initialize sweep by passing in config
sweep_id = wandb.sweep(
  sweep=config, 
  project='optES' 
) 

# Start sweep job 
wandb.agent(sweep_id, function=main)

# Finish sweep job
wandb.finish() 
