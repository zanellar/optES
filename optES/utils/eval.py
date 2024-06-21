import numpy as np
import wandb 
import torch
import os
import json
from optES.utils.paths import NET_WEIGHTS_PATH
from optES.utils.pos import getpos
from optES.utils.plot import plot_p11

class Eval():
    def __init__(self, model, controller, save_best_result = True, convergence_margin=0.025, debug=False):
        '''
            Evaluate the controller network on the system model.
            @param model: The system model
            @param controller: The controller
            @param save_best_result: Whether to save the best result
            @param convergence_margin: Error threshold for convergence 
            @param log: Level of logging (0: logging "q", "p", "error", 1: logging "eval/steps", "eval/error", 2: logging "q", "p", "error", "eval/steps", "eval/error")
        '''
        self.debug = debug
        self.convergence_margin = convergence_margin
        self.model = model 
        self.controller = controller 
        self.save_best_result = save_best_result
        self.best_result = {"cost":np.inf, "eval/error":np.inf, "eval/steps":np.inf, "run":""}

    def save(self):
        print("Saving...") 
        last_error = np.array(wandb.run.summary["eval/error"]).flatten()[-1]
        last_steps = np.array(wandb.run.summary["eval/steps"]).flatten()[-1]
        if last_error < self.best_result["eval/error"] and last_steps < self.best_result["eval/steps"]:
            self.best_result = dict(
                cost = wandb.run.summary["cost"],
                error = wandb.run.summary["eval/error"],
                steps = wandb.run.summary["eval/steps"], 
                run = wandb.run.name
            )
        print("Best result: ", self.best_result)
        with open(os.path.join(NET_WEIGHTS_PATH, "naespbc_BEST.json"), "w") as f:
            json.dump(self.best_result, f)

    def evaluate(self, horizon, initial_pos, target_pos, controller=None, title="", log=1): 
        '''
            Evaluate the controller network on the system model.
            @param horizon: The prediction horizon
            @param initial_pos: The initial state of the system
            @param target_pos: The target state of the system
            @param title: The title of the evaluation
            @param log: Level of logging (0: logging "q", "p", "error", 1: logging also "eval/steps", "eval/error", 2: logging also "p11")
        '''
        if controller is not None:
            self.controller = controller
            print("Evaluating with a new controller... " + title)

        print("Evaluating... " + title) 
 
        # set the network to evaluation mode if any
        if hasattr(self.controller, "net"): 
            self.controller.set_eval_mode()
                 
        # set the initial position and target position
        initial_pos = getpos(initial_pos, self.model.n)
        target_pos = getpos(target_pos, self.model.n)
        self.model.reset(q0=initial_pos, p0=[0]*self.model.n)
        self.controller.set_target(qd=target_pos)

        # data for logging
        _q = np.zeros((self.model.n,horizon))
        _q.fill(np.nan)
        _p = np.zeros((self.model.n,horizon))
        _p.fill(np.nan)
        _err = np.zeros((1,horizon))
        _err.fill(np.nan)
        data_eval = dict( 
            q = _q,
            p = _p,
            error = _err,
        )
        
        # target state
        xd = self.controller.xd.flatten().detach().numpy()

        convergence_steps = 0
        for t in range(horizon): 
            
            # action
            u = self.controller.get_action(self.model.q, self.model.p) 
            # simulation
            x, y = self.model.step(u=u) 

            # state extract a scalar value if the state is 1D and a vector if the state is 2D
            q = self.model.q.clone().flatten().detach().numpy()
            p = self.model.p.clone().flatten().detach().numpy()
            
            if self.model.n == 1:
                q = q.item()
                p = p.item()

            # state error
            error  = np.linalg.norm(x - xd)
            data_eval["q"][:,t] = q
            data_eval["p"][:,t] = p
            data_eval["error"][:,t] = error 
 
            if error > self.convergence_margin:
                convergence_steps += 1

        # combine the data into a table  
        title = " " + title if title != "" else title
        if log>=0:
            print("Logging...")
            for key in data_eval.keys():  
                value = data_eval[key]
                num_dim = value.shape[0]
                for i in range(num_dim): 
                    label = f"{key}_{i}" if num_dim>1 else key
                    data_plot = [[_x, _y] for (_x, _y) in zip(range(horizon), value[i,:])]
                    table = wandb.Table(data=data_plot, columns=["steps", label])
                    line_plot = wandb.plot.line(table, x="steps", y=label, title=f"{key}_{i}{title}") 
                    wandb.log({f"{key}/{i}{title}": line_plot})
  
            last_error = data_eval["error"][:,-1].item()
            wandb.log({"eval/steps": convergence_steps, "eval/error" : last_error}) 
        
        if log>=2: 
            plot_p11(controller=self.controller, name="tmp", local_plot=True)

        if self.save_best_result:
            self.save()
 
        # set the model back to train mode
        if hasattr(self.controller, "net"): 
            self.controller.set_train_mode()

        return np.mean(data_eval["error"]), convergence_steps
    

    
