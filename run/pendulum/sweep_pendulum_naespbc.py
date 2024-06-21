
import os 
import yaml
import wandb 

from optES.systems.pendulum_phmodel import Pendulum   
from optES.systems.pendulum_controllers import PendulumNASEPBC, PendulumASEPBC
from optES.utils.paths import PARAMS_PATH   
from optES.optimization.optimproc import OptProccess
   
project = 'nes_pendulum' 
args_config_file = "sweep_pendulum_neural_aespbc"
args_params_file = "pendulum_neural_aespbc_uniforminit"
args_params_baseline_file = "pendulum_aespbc"

####################################################################################################

# Load wandb sweep config file
config_file_path = os.path.join(PARAMS_PATH,"optimization", args_config_file+".yaml")
with open(config_file_path, "r") as stream: 
    config = yaml.safe_load(stream) 

# Initialize sweep by passing in config
sweep_id = wandb.sweep(sweep = config, project = project) 

# Initialize main optimizer
mainopt = OptProccess(
    params_file_name = args_params_file,
    modelclass = Pendulum,
    controllerclass = PendulumNASEPBC,
    baselineclass = PendulumASEPBC,
    params_baseline_file_name = args_params_baseline_file
)

# Start sweep job 
wandb.agent(sweep_id = sweep_id, function = mainopt.run, project = project)
  
# Finish sweep job
wandb.finish()