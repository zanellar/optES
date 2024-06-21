
import torch
import os
import json
from torchviz import make_dot 
import matplotlib.pyplot as plt
from optES.systems.pendulum_phmodel import Pendulum 
from optES.utils.paths import PARAMS_PATH



# Opening JSON file
f = open(os.path.join(PARAMS_PATH,"pendulum_free.json")) 
params = json.load(f)
 
system = Pendulum(params)
# system = Pendulum(dt=0.1, b=0, k=0)

system.reset(
    q0 = [params["initial_pos"]],
    p0 = [0.]
)
# q = torch.randn(1)
# p = torch.randn(1)
# y = system.H(q,p )
# make_dot(y)

data = dict(q=[],p=[],H=[],V=[],K=[])

for t in range(params["model"]["horizon"]):

    x,y = system.step(u=[0])
    q,p = system.q, system.p
    data["q"].append(q.flatten().detach().item())
    data["p"].append(p.flatten().detach().item())
    data["H"].append(system.H(q,p).item())
    data["V"].append(system.V(q).flatten().detach().item())
    data["K"].append(system.K(q,p).flatten().detach().item())


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