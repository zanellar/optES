import os
import torch
import wandb 
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from optES.utils.evalfuncs import get_energy, get_p11
from optES.utils.paths import PLOTS_PATH

def _plot_wandb_dict(data, xspace, xlabel="x", name="", select = None):
    """
    Plot the dictionary data in wandb 
    """
    
    for key in data.keys():

        if select is not None and key not in select:
            continue
        
        # combine the data into a table  
        data_plot = [[x, y] for (x, y) in zip(xspace, data[key])]

        # Create a wandb table format
        xlabel = xlabel
        ylabel = key
        table = wandb.Table(data=data_plot, columns=[xlabel, ylabel])

        # Use the table to populate various custom charts 
        title = f"{key} {name}" if name != "" else f"{key}"
        print(f"Plotting: {title}") 
        plot = wandb.plot.line(table, x=xlabel, y=ylabel, title=title)
        
        # Log custom tables, which will show up in customizable charts in the UI
        wandb.log({title: plot }) 

########################################################################################################
########################################################################################################
########################################################################################################

def plot_p11(controller, min=-10, max=10, step=0.1, name="", select = None, local_plot=False):
    """
    Plot the p11 value of the controller
    """

    xspace, yspace = get_p11(controller, min, max, step) 

    n = controller.model.n
    data = {f"p11_{i}_eq{j}":yspace[j,i,:].tolist() for j in range(n) for i in range(n)}

    if local_plot:  
        print(f"Plotting: p11") 
        plt.figure(figsize=(20,20))
        for j in range(controller.model.n):
            for i in range(controller.model.n):
                plt.subplot(n,n,i+j*n+1)
                plt.plot(xspace, data[f"p11_{i}_eq{j}"])
                plt.title(f"p11_{i}_eq{j}")
                plt.xlabel("q")
                plt.ylabel("p11")
                plt.grid()
        plt.tight_layout()
        # plt.show()
        filename = name + f"_p11_{i}_eq{j}_" + datetime.now().strftime("%Y%m%d%H%M%S")
        folderpath = os.path.join(PLOTS_PATH, "p11")
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        filepath = os.path.join(folderpath, filename + ".png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else: 
        _plot_wandb_dict(data, xspace=xspace, name=name, xlabel="error", select = select)
 

def plot_energy(model, controller, min=-6.28, max=6.28, step=0.1, name="", select = None, local_plot=False):
    """
    Plot the energy of the system 
    Potential energy and Hamiltonian of the open-loop system: V(q) and H(q)
    Potential energy and Hamiltonian of the closed-loop system: Vd(q) and Hd(q) 
    """  
    
    xspace, yspace = get_energy(model, controller, min, max, step)
    n = controller.model.n
    data = {} 
    data.update({f"Vd_{i}_q{j}":yspace["Vd"][j,i,:].tolist() for j in range(n) for i in range(n)})
    data.update({f"V_{i}_q{j}":yspace["V"][j,i,:].tolist() for j in range(n) for i in range(n)})
     
    if local_plot: 
        # fig = plt.figure(figsize=(20,20))
        # plt.subplot(2,1,1)
        # plt.plot(config_space, data["V"])
        # plt.sca
        # plt.title("Potential Energy of the Open-Loop System")
        # plt.xlabel("q")
        # plt.ylabel("V")
        # plt.grid() 
        # plt.subplot(2,1,2)
        # plt.plot(config_space, data["Vd"])
        # xd = controller.xd[0].flatten().detach().item()
        # Vd = np.array(data["Vd"])[np.where(config_space.numpy() == xd)][0]
        # plt.scatter(xd, Vd, c="r")
        # matplotlib.pyplot.text(xd, Vd, f"({xd:.2f},{Vd:.2f})")
        # plt.title("Potential Energy of the Closed-Loop System")
        # plt.xlabel("q")
        # plt.ylabel("Vd")
        # plt.grid() 
        # fig.tight_layout()
        # plt.show()

        # TODO TEST THE FOLLOWING CODE

        fig, axs = plt.subplots(controller.model.n, figsize=(20, 20))

        for j in range(controller.model.n):
            for i in range(controller.model.n):
                axs[j].plot(xspace, data["V"][j, i, :])
                axs[j].set_title(f"Potential Energy of the Open-Loop System - Dimension {j} - Joint {i}")
                axs[j].set_xlabel("q")
                axs[j].set_ylabel(f"V_{i}_q{j}")
                axs[j].grid()

                axs[j].plot(xspace, data["Vd"][j, i, :])
                xd = controller.xd[0][j].flatten().detach().item()
                Vd = np.array(data["Vd"][j, i, :])[np.where(xspace.numpy() == xd)][0]
                axs[j].scatter(xd, Vd, c="r")
                axs[j].text(xd, Vd, f"({xd:.2f},{Vd:.2f})")
                axs[j].set_title(f"Potential Energy of the Closed-Loop System - Dimension {j} - Joint {i}")
                axs[j].set_xlabel("q")
                axs[j].set_ylabel(f"Vd_{i}_q{j}")
                axs[j].grid()

        fig.tight_layout()
        plt.show()
    else:
        _plot_wandb_dict(data, xspace=xspace, name=name, xlabel="q", select = select)
 