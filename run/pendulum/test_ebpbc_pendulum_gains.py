import numpy as np
import os
import json
import matplotlib.pyplot as plt
from optES.systems.pendulum_phmodel import Pendulum 
from optES.systems.pendulum_controllers import PendulumEBPBC
from optES.utils.paths import PARAMS_PATH



# Opening JSON file
f = open(os.path.join(PARAMS_PATH,"pendulum_ebpbc_optim.json")) 
params = json.load(f)
    
delta_system = dict(
    dr=0.0,
    dm=0.0,
    db=0.0,
    dk=0.0
)
params_sys = params
params_sys["model"]["r"] += delta_system["dr"]
params_sys["model"]["m"] += delta_system["dm"]
params_sys["model"]["b"] += delta_system["db"]
params_sys["model"]["k"] += delta_system["dk"]


##### System #####

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

##### Controller #####

controller = PendulumEBPBC(
    model, 
    gains_file_name = params["optimizer"]["gains_file_name"],
    load_gains = True
    )
controller.set_target(qd=[params["target_pos"]])
controller.reset(q0=system.q)

#################################################

data = dict(q=[],p=[],H=[],V=[],K=[],u=[])

for t in range(params["model"]["horizon"]):
    u = controller.get_action(system.q, system.y)
    x, y = system.step(u=u)
    q,p = system.q, system.p
    data["q"].append(q.flatten().detach().item())
    data["p"].append(p.flatten().detach().item())
    data["H"].append(system.H(q,p).item())
    data["V"].append(system.V(q).flatten().detach().item())
    data["K"].append(system.K(q,p).flatten().detach().item())
    data["u"].append(u.flatten().detach().item())

#################################################

## Plot State
plt.figure(figsize=(20,10))

plt.subplot(3,2,1)
plt.plot(data["q"])
plt.title("Position")
plt.xlabel("k")
plt.ylabel("q")
plt.grid()

plt.subplot(3,2,3)
plt.plot(data["p"])
plt.title("Velocity")
plt.xlabel("k")
plt.ylabel("p")
plt.grid()

plt.subplot(3,2,5)
error = np.array(data["q"]) - np.ones_like(data["q"])*params["target_pos"]
plt.plot(error)
plt.title("Error")
plt.xlabel("k")
plt.ylabel("e")
plt.grid()

# Plot Energy

plt.subplot(3,2,2)
plt.plot(data["K"])
plt.title("Kinetic Energy")
plt.xlabel("k")
plt.ylabel("K")
plt.grid()

plt.subplot(3,2,4)
plt.plot(data["V"])
plt.title("Potential Energy")
plt.xlabel("k")
plt.ylabel("V")
plt.grid()

plt.subplot(3,2,6)
plt.plot(data["H"])
plt.title("Energy")
plt.xlabel("k")
plt.ylabel("H")
plt.grid()

plt.show()

##########################################
# plt.figure(figsize=(20,3))

# plt.subplot(1,2,1)
# plt.plot(data["u"])
# plt.title("Control Signal")
# plt.xlabel("k")
# plt.ylabel("u")
# plt.grid()

# plt.show()

print(f"error : {error[-1]}")