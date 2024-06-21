import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from optES.utils.pos import getpos
from optES.systems.pendulum_phmodel import Pendulum 
from optES.systems.pendulum_controllers import PendulumASEPBC, PendulumNASEPBC
from optES.utils.paths import PARAMS_PATH, NET_WEIGHTS_PATH, PLOTS_PATH
from optES.utils.evalfuncs import get_energy, get_p11

plt.rcParams['font.size'] = 16

num_initial_pos = -1
positions = [
    0.40045036478026885,
    0.04273495114418979,
    0.43772634397296073,
    0.14296822275503807,
    0.15268187550060863,
    -0.055841316637243676,
    0.3173167068741257,
    0.3287741880993831,
    0.034871684866407904,
    -0.0927915301866028
]
# positions = np.linspace(-6, 6, 10)

color_naespbc = "tab:orange"
color_baseline = "tab:blue"

# Opening JSON file
f = open(os.path.join(PARAMS_PATH,"pendulum_neural_aespbc_uniforminit.json")) 
# f = open(os.path.join(PARAMS_PATH,"pendulum_neural_aespbc_uniforminit_global.json")) 
params = json.load(f)
  
delta_system = dict(
    dr = 0.0,
    dm = 0.0,
    db = 0.0,
    dk = 0.0
)

params_sys = params
params_sys["model"]["r"] += delta_system["dr"]
params_sys["model"]["m"] += delta_system["dm"]
params_sys["model"]["b"] += delta_system["db"]
params_sys["model"]["k"] += delta_system["dk"]

#### System #####
system = Pendulum(params_sys)

##### Model #####
model = Pendulum(params)

##### Controller #####
weight_files = ["naespbc_20240131_3_3_3_3_relu_serene-flower-2882"]
# weight_files = [f.split(".")[0] for f in os.listdir(NET_WEIGHTS_PATH)]
controllers = {}
 
# Baseline
controllers["baseline"] = PendulumASEPBC(
    model = model,
    Kw = params["controller"]["Kw"],
    Kx = params["controller"]["Kx"],
    R = params["controller"]["R"],
    Pgains = dict(k0=1.36, k1=0.654),
    dt = params["dt"]
)  
controllers["baseline"].set_target(qd=[params["target_pos"]])

# Neural controller (different seeds)
for weights_file_name in weight_files:  

    # check if the weights file is in the folder
    if not os.path.isfile(os.path.join(NET_WEIGHTS_PATH, weights_file_name+".pth")):
        continue

    controller = PendulumNASEPBC(
        model = model,
        Kw = params["controller"]["Kw"],
        R = params["controller"]["R"],
        dim_layers = params["controller"]["dim_layers"],  
        weights_file_name = weights_file_name,
        load_net = True,
        activation = params["controller"]["activation"], 
        dt = params["dt"]
    )
    controller.set_target(qd=[params["target_pos"]])

    controllers[weights_file_name] = controller


################################################################### 
###################################################################
###################################################################
 
if num_initial_pos > 0: 
    positions = [getpos(params["initial_pos"], n = model.n).item()   for i in range(num_initial_pos)] 

data = {} 

# Iterate over different controllers
for controller_id, controller in controllers.items(): 
    
    data[controller_id] = {} 

    print(f"Controller {controller_id}")

    # Iterate over different initial positions
    for initial_pos in positions:
 
        print(f"Initial Position {initial_pos}")

        system.reset(
            q0 = [initial_pos],
            p0 = [0]
        ) 

        model.reset(
            q0 = [initial_pos],
            p0 = [0]
        )
 
        #################################################
        # Evaluate the controller on the system   
        data[controller_id][str(initial_pos)] = {"q":[], "p":[], "Vd":[], "p11":[], "error":[]}

        q,p = system.q, system.p
        data[controller_id][str(initial_pos)]["q"].append(q.flatten().detach().item())
        data[controller_id][str(initial_pos)]["p"].append(p.flatten().detach().item()) 
        data[controller_id][str(initial_pos)]["error"].append(np.linalg.norm(q.detach().numpy()-params["target_pos"], axis=1))

        for t in range(params["model"]["horizon"]):
            u = controller.get_action(system.q, system.p )
            x, y = system.step(u=u)

            q,p = system.q, system.p
            data[controller_id][str(initial_pos)]["q"].append(q.flatten().detach().item())
            data[controller_id][str(initial_pos)]["p"].append(p.flatten().detach().item()) 
            data[controller_id][str(initial_pos)]["error"].append(np.linalg.norm(q.detach().numpy()-params["target_pos"], axis=1))


    # Potential energy of the closed-loop system
    energydata_range, energydata = get_energy(model, controller)
    data[controller_id]["Vd"] = energydata["Vd"] 
    data[controller_id]["Vd_xrange"] = energydata_range.detach().numpy()

    # P11 of the controller
    p11data_range, p11data = get_p11(controller)
    data[controller_id]["p11"] = np.array([p11 for p11 in p11data]) 
    data[controller_id]["p11_xrange"] = p11data_range 
  
################################################################### 
###################################################################
###################################################################
 
# Plotting
fig0, ax0 = plt.subplots(figsize=(5,5))
# ax0.set_title('Error')
fig1, ax1 = plt.subplots(figsize=(5,5))
# ax1.set_title('Position')
fig2, ax2 = plt.subplots(figsize=(5,5))
# ax2.set_title('Momentum')
fig3, ax3 = plt.subplots(figsize=(5,5))
# ax3.set_title('Potential Energy')
fig4, ax4 = plt.subplots(figsize=(5,5))
# ax4.set_title('P11')

def add_plot_convence_margin(ax, data, convergence_margin=0.01, reference=0.0, color="r"):  
    data = np.array(data) 
    
    ax.fill_between(np.linspace(0, data.shape[1], data.shape[1]), reference-convergence_margin, reference+convergence_margin, color='green', alpha=0.2) 
  
    indices = np.where((data <= reference-convergence_margin) | (data >= reference+convergence_margin))   
    max_index = np.max(indices) + 1 
    ax.axvline(x=max_index, color=color, linestyle=':', alpha=0.8) 
    ax.text(max_index, ax.get_ylim()[0], str(max_index), color=color, ha='right', alpha=0.8, fontsize=16)

naespbc_err = []
baseline_err = []
naespbc_pos = []
baseline_pos = []
naespbc_mom = []
baseline_mom = []

for controller_id in controllers.keys():  
    print(f"@@@@ Controller {controller_id}")
    # linestyle = "dashed" if controller_id == "baseline" else "solid"
    linestyle = "solid"
    color = color_baseline if controller_id == "baseline" else color_naespbc

    for initial_pos in positions:

        ### Error
        err = np.array(data[controller_id][str(initial_pos)]["error"])  
        err = np.sum(err, axis=1)
        baseline_err.append(err) if controller_id == "baseline" else naespbc_err.append(err)
        ax0.plot(err, label=controller_id, linestyle=linestyle, color=color)

        ### Position
        pos = data[controller_id][str(initial_pos)]["q"]
        baseline_pos.append(pos) if controller_id == "baseline" else naespbc_pos.append(pos)
        ax1.plot(pos, label=controller_id, linestyle=linestyle, color=color)
        # add_plot_convence_margin(ax1, pos)

        ### Momentum
        mom = data[controller_id][str(initial_pos)]["p"]
        baseline_mom.append(mom) if controller_id == "baseline" else naespbc_mom.append(mom)
        ax2.plot(mom, label=controller_id, linestyle=linestyle, color=color)
        # add_plot_convence_margin(ax2, mom)
        
    ax3.plot(data[controller_id]["Vd_xrange"], data[controller_id]["Vd"].flatten(), label=controller_id, linestyle=linestyle, color=color)
    ax4.plot(data[controller_id]["p11_xrange"], data[controller_id]["p11"].flatten(), label=controller_id, linestyle=linestyle, color=color)

add_plot_convence_margin(ax0, naespbc_err, convergence_margin=0.01, color=color_naespbc)
add_plot_convence_margin(ax0, baseline_err, convergence_margin=0.01, color=color_baseline)
add_plot_convence_margin(ax1, naespbc_pos, convergence_margin=0.01, reference=params["target_pos"], color=color_naespbc)
add_plot_convence_margin(ax1, baseline_pos, convergence_margin=0.01, reference=params["target_pos"], color=color_baseline)
add_plot_convence_margin(ax2, naespbc_mom, convergence_margin=0.01, color=color_naespbc)
add_plot_convence_margin(ax2, baseline_mom, convergence_margin=0.01, color=color_baseline)

ax0.grid()
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax0.margins(x=0)
ax0.margins(y=0)
ax1.margins(x=0)
ax1.margins(y=0)
ax2.margins(x=0)
ax2.margins(y=0)
ax3.margins(x=0)
ax3.margins(y=0)
ax4.margins(x=0)
ax4.margins(y=0)

ax0.set_xlim([0,100]) 
ax1.set_xlim([0,100])
ax2.set_xlim([0,100]) 
ax3.set_ylim([-25,50])
ax3.set_xlim([-6.28,6.28])
ax4.set_ylim([0,4])
ax4.set_xlim([-10,10])

plt.show()

folder_path = os.path.join(PLOTS_PATH,"pendulum")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save plots
fig0.savefig(os.path.join(folder_path, "pendulum_error.pdf"))
fig1.savefig(os.path.join(folder_path, "pendulum_pos.pdf"))
fig2.savefig(os.path.join(folder_path, "pendulum_mom.pdf"))
fig3.savefig(os.path.join(folder_path, "pendulum_pot.pdf"))
fig4.savefig(os.path.join(folder_path, "pendulum_p11.pdf"))

fig0.savefig(os.path.join(folder_path, "pendulum_error.png"))
fig1.savefig(os.path.join(folder_path, "pendulum_pos.png"))
fig2.savefig(os.path.join(folder_path, "pendulum_mom.png"))
fig3.savefig(os.path.join(folder_path, "pendulum_pot.png"))
fig4.savefig(os.path.join(folder_path, "pendulum_p11.png"))

fig0.savefig(os.path.join(folder_path, "pendulum_error.svg"))
fig1.savefig(os.path.join(folder_path, "pendulum_pos.svg"))
fig2.savefig(os.path.join(folder_path, "pendulum_mom.svg"))
fig3.savefig(os.path.join(folder_path, "pendulum_pot.svg"))
fig4.savefig(os.path.join(folder_path, "pendulum_p11.svg"))
