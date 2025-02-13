# optES # 
## Learning the Optimal Energy-based Control Strategy for Port-Hamiltonian Systems ##
This repository implements a tuning procedure for discrete-time, energy-based regulators for port-Hamiltonian systems. Based on a discrete-time approximation of the plant, the control system is designed within the energy-shaping plus damping injection paradigm. To achieve task performance optimization alongside a guarantee of asymptotic stability, artificial neural networks are integrated as parametric function approximators with passivity-based control. The aim is to employ optimally shaped artificial neural networks to enhance performance during task execution through the solution of an optimization problem.
 
# Notes #

In 'optES/systems' we exted the class 'PHSystemCanonic' for the specific system (e.g. a simple pendulum  'pendulum_phmodel' is already implemeted) 

In 'optES/systems' we exted the classes 'EBPBC', 'GAEBPBC', 'NAEBPBC' for the specific system (e.g. a simple pendulum  'pendulum_controllers' is already implemeted). Note that the class 'GAEBPBC' needs the method 'pdiag', function of the error, to be defined in the specific system class (e.g. for the pendulum some quadratic function is implemented).

The derivative of the Hamiltonian and the potential energy, if not overwritten, are computed using autodifferentiation in the class 'PHSystemCanonic'. This can introduce some error that requires a smaller tollerance in the Newton-Raphson method or a smaller time step in the simulation. When possible, it is better to overwrite the methods '_dH' and '_dV' in the specific system class.

## Installation ##

Setup virtual environment

```
conda env create -f environment.yml
```

Install Package

```
pip install -e .
```

 
## Usage ##

Here is explained how to use the code using the example of the pendulum.

To run the optimization on the neural netowkr weights with a grid search: 
```
python sweep_pendulum_naespbc.py
```
inside the script change the name of the files for: the grid search configuration ('args_config_file'), the system parameters ('args_params_file') and the baseline parameters ('args_params_baseline_file'). See the folliwing files as examples of configurations and parameters: "data/params/optimization/sweep_pendulum_neural_aespbc.yaml", "data/params/pendulum_neural_aespbc_uniforminit.json", "data/params/pendulum_aespbc.json"

To run the test on the optimal solution: 
```
python test_aespbc_pendulum_net.py
```
here change the name of the parameters file ('args_params_file'). And inside the new file specify the path to the optimal weight. See for example "data/params/pendulum_neural_aespbc_fixedinit.json"

# References #
Zanella, R., Macchelli, A. and Stramigioli, S., 2024. Learning the Optimal Energy-based Control Strategy for Port-Hamiltonian Systems. IFAC-papersonline, 58(6), pp.208-213.
## Cite ##
```
@article{zanella2024learning,
  title={Learning the Optimal Energy-based Control Strategy for Port-Hamiltonian Systems},
  author={Zanella, Riccardo and Macchelli, Alessandro and Stramigioli, Stefano},
  journal={IFAC-papersonline},
  volume={58},
  number={6},
  pages={208--213},
  year={2024},
  publisher={Elsevier}
}
```
