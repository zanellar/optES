import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from optES.systems.pendulum_phmodel import Pendulum 
from optES.systems.pendulum_controllers import PendulumASEPBC
from optES.utils.paths import PARAMS_PATH
from optES.utils.plot import plot_energy

   
# Opening JSON file
f = open(os.path.join(PARAMS_PATH,"pendulum_aespbc.json")) 
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

controller = PendulumASEPBC(
    model = model,
    Kw = params["controller"]["Kw"],
    R = params["controller"]["R"], 
    k0 = params["controller"]["k0"],
    k1 = params["controller"]["k1"],
    dt = params["dt"]
)
controller.set_target(qd=[params["target_pos"]])

#################################################

data = dict(q=[],p=[],H=[],V=[],K=[],qw=[],pw=[],u=[],p11=[],Vd=[],Hcl=[])

for t in range(params["model"]["horizon"]):

    u = controller.get_action(system.q, system.p )
    x, y = system.step(u=u)

    q,p = system.q, system.p
    H = system.H(q, p) 
    x = torch.cat((q,p)).float()
    w = x.clone()
    xd = controller.xd.reshape(x.shape)
    Ha = controller.Ha(x - xd, w - xd) 
    V =  system.V(q)
    K =  system.K(q,p)
    Hcl = H + Ha #- V
    Vd = Hcl - K  
    data["q"].append(q.flatten().detach().item())
    data["p"].append(p.flatten().detach().item())
    data["H"].append(H.item())
    data["V"].append(V.flatten().detach().item())
    data["K"].append(K.flatten().detach().item())
    data["Vd"].append(Vd.flatten().detach().item()) 
    data["Hcl"].append(Hcl.flatten().detach().item())

    w = controller._next_w 
    qw, pw = torch.split(w, system.n)
    data["qw"].append(qw.flatten().detach().item())
    data["pw"].append(pw.flatten().detach().item())
    data["u"].append(u.flatten().detach().item())
    data["p11"].append(controller.pdiag(q,p).flatten().detach().item())
 
## Plot State
fig = plt.figure(figsize=(20,20))
nr, nc = 4,3

plt.subplot(nr,nc,1)
plt.plot(data["q"])
plt.plot(np.ones_like(data["q"])*params["target_pos"], "--")
plt.title("Position")
plt.xlabel("k")
plt.ylabel("q")
plt.grid()

plt.subplot(nr,nc,4)
plt.plot(data["p"])
plt.title("Velocity")
plt.xlabel("k")
plt.ylabel("p")
plt.grid()

plt.subplot(nr,nc,7)
error = np.array(data["q"]) - np.ones_like(data["q"])*params["target_pos"]
plt.plot(error)
plt.title("Error")
plt.xlabel("k")
plt.ylabel("e")
plt.grid()

# Plot Energy

plt.subplot(nr,nc,8)
plt.plot(data["K"])
plt.title("Kinetic Energy")
plt.xlabel("k")
plt.ylabel("K")
plt.grid()

plt.subplot(nr,nc,2)
plt.plot(data["V"])
plt.title("Potential Energy")
plt.xlabel("k")
plt.ylabel("V")
plt.grid()

plt.subplot(nr,nc,5)
plt.plot(data["H"])
plt.title("Hamiltonian")
plt.xlabel("k")
plt.ylabel("H")
plt.grid()

plt.subplot(nr,nc,3)
plt.plot(data["Vd"])
plt.title("Potential Energy Closed-Loop")
plt.xlabel("k")
plt.ylabel("Vd")
plt.grid()

plt.subplot(nr,nc,6)
plt.plot(data["Hcl"])
plt.title("Hamiltonian Closed-Loop")
plt.xlabel("k")
plt.ylabel("Hcl")
plt.grid()

# Plot Estented state

plt.subplot(nr,nc,9)
plt.plot(data["qw"])
plt.title("Extended Position")
plt.xlabel("k")
plt.ylabel("qw")
plt.grid()

plt.subplot(nr,nc,12)
plt.plot(data["pw"])
plt.title("Extended Velocity")
plt.xlabel("k")
plt.ylabel("pw")
plt.grid()

# Plot control

plt.subplot(nr,nc,10)
plt.plot(data["u"])
plt.title("Control Signal")
plt.xlabel("k")
plt.ylabel("u")
plt.grid()

plt.subplot(nr,nc,11)
plt.plot(data["p11"])
plt.title("p11")
plt.xlabel("k")
plt.ylabel("p11")
plt.grid()

fig.tight_layout() 
 
## Plot Energy
plot_energy(model, controller, min=-6.28, max=6.28, step=0.01, name="", select = None, local_plot=True)

print(f"error : {error[-1]}")