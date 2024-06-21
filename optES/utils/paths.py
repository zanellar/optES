import os
import optES

_PACKAGE_PATH = optES.__path__[0]  

GAINS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "gains")  
NET_WEIGHTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "weights")  
PARAMS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "params")  
PLOTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "plots")