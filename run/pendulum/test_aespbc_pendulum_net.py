import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from optES.systems.pendulum_phmodel import Pendulum 
from optES.systems.pendulum_controllers import PendulumNASEPBC
from optES.utils.paths import PARAMS_PATH

args_params_file = "pendulum_neural_aespbc_fixedinit"
# Opening JSON file
f = open(os.path.join(PARAMS_PATH, args_params_file + ".json")) 
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

#################################################

system = Pendulum(params_sys)
system.reset(
    q0 = [params["initial_pos"]],
    p0 = [0]
)

##### Model #####

model = Pendulum(params)
model.reset(
    q0 = [params["initial_pos"]],
    p0 = [0]
)

#################################################

controller = PendulumNASEPBC(
    model = model,
    Kw = params["controller"]["Kw"],
    R = params["controller"]["R"],
    dim_layers = params["controller"]["dim_layers"],  
    weights_file_name = params["controller"]["weights_file_name"],
    load_net = True,
    activation = params["controller"]["activation"], 
    dt = params["dt"]
)
controller.set_target(qd=[params["target_pos"]])

#################################################

data = dict(q=[],p=[],H=[],V=[],K=[],qw=[],pw=[],u=[],p11=[])

for t in range(params["model"]["horizon"]):
    u = controller.get_action(system.q, system.p )
    x, y = system.step(u=u)

    q,p = system.q, system.p
    data["q"].append(q.flatten().detach().item())
    data["p"].append(p.flatten().detach().item())
    data["H"].append(system.H(q,p).item())
    data["V"].append(system.V(q).flatten().detach().item())
    data["K"].append(system.K(q,p).flatten().detach().item())

    w = controller._next_w
    qw, pw = torch.split(w, system.n)
    data["qw"].append(qw.flatten().detach().item())
    data["pw"].append(pw.flatten().detach().item())
    data["u"].append(u.flatten().detach().item())
    data["p11"].append(controller.pdiag(q,p).flatten().detach().item())
 
## Plot State
plt.figure(figsize=(20,20))
nr, nc = 5,2

plt.subplot(nr,nc,1)
plt.plot(data["q"])
plt.title("Position")
plt.xlabel("k")
plt.ylabel("q")
plt.grid()

plt.subplot(nr,nc,3)
plt.plot(data["p"])
plt.title("Velocity")
plt.xlabel("k")
plt.ylabel("p")
plt.grid()

plt.subplot(nr,nc,5)
error = np.array(data["q"]) - np.ones_like(data["q"])*params["target_pos"]
plt.plot(error)
plt.title("Error")
plt.xlabel("k")
plt.ylabel("e")
plt.grid()

# Plot Energy

plt.subplot(nr,nc,2)
plt.plot(data["K"])
plt.title("Kinetic Energy")
plt.xlabel("k")
plt.ylabel("K")
plt.grid()

plt.subplot(nr,nc,4)
plt.plot(data["V"])
plt.title("Potential Energy")
plt.xlabel("k")
plt.ylabel("V")
plt.grid()

plt.subplot(nr,nc,6)
plt.plot(data["H"])
plt.title("Energy")
plt.xlabel("k")
plt.ylabel("H")
plt.grid()

# Plot Estented state

plt.subplot(nr,nc,7)
plt.plot(data["qw"])
plt.title("Extended Position")
plt.xlabel("k")
plt.ylabel("qw")
plt.grid()

plt.subplot(nr,nc,9)
plt.plot(data["pw"])
plt.title("Extended Velocity")
plt.xlabel("k")
plt.ylabel("pw")
plt.grid()

# Plot control

plt.subplot(nr,nc,8)
plt.plot(data["u"])
plt.title("Control Signal")
plt.xlabel("k")
plt.ylabel("u")
plt.grid()

plt.subplot(nr,nc,10)
plt.plot(data["p11"])
plt.title("p11")
plt.xlabel("k")
plt.ylabel("p11")
plt.grid()

plt.show()

print(f"error : {error[-1]}")