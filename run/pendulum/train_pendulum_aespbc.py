
import os 
import yaml
import wandb 
import argparse

from optES.systems.pendulum_phmodel import Pendulum  
from optES.systems.pendulum_controllers import PendulumGASEPBC, PendulumASEPBC
from optES.utils.paths import PARAMS_PATH   
from optES.optimization.optimproc import OptProccess
  
argparser = argparse.ArgumentParser()
argparser.add_argument('--params_file', type=str, default="pendulum_quadratic_aespbc_uniforminit")
argparser.add_argument('--config_file', type=str, default="train_pendulum_quadratic_aespbc")
argparser.add_argument('--params_baseline_file', type=str, default="pendulum_aespbc")
args = argparser.parse_args()

################################################################################################################

# Load wandb sweep config file
config_file_path = os.path.join(PARAMS_PATH,"optimization", args.config_file+".yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream) 

# Wandb initialization
run = wandb.init( project = "nes_gain", mode="online") 
    
# Initialize main optimizer
mainopt = OptProccess(
    params_file_name = args.params_file,
    modelclass = Pendulum,
    controllerclass = PendulumGASEPBC,
    # baselineclass = PendulumASEPBC, 
    params_baseline_file_name = args.params_baseline_file,
    optim = OptProccess.GAINS,
    config = config)
mainopt.run()
  
# Finish sweep job
wandb.finish()
 