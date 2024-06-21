import torch   
import torch.optim.lr_scheduler as lr_scheduler

import os
import numpy as np
import wandb 
from optES.utils.paths import NET_WEIGHTS_PATH 
from optES.utils.eval import Eval 
from optES.utils.pos import getpos
from torchviz import make_dot 
from optES.utils.ops import weighted_norm

class BaseOptimizer():

    def __init__(self, 
                 model, 
                 controller,  
                 params2optimize,
                 initial_pos, 
                 target_pos,
                 cost_type,
                 config, 
                 debug = False
                 ):
        '''
            Train the controller network to minimize the cost function of the closed-loop dynamics.
            The cost function is defined as the sum of the L2 distance between the final and the target state.
            
            @param model: The system model
            @param controller: The controller
            @param params2optimize: The generator of the parameters to optimize (function)
            @param cost_type: The type of cost function ("l2" or "l1")
            @param initial_pos: The initial state of the system
            @param target_pos: The target state of the system
            @param config: The configuration of the optimization which contains:
                    - optimization_steps: The number of optimization steps 
                    - prediction_horizon: The prediction horizon
                    - learning_rate: The learning rate
                    - initial_weight: The weight of the initial state in the cost function
                    - input_weight: The weight of the input in the cost function
                    - state_weight: The weight of the state in the cost function
                    - final_weight: The weight of the final state in the cost function
                    - early_stop_threshold: The threshold for early stopping
                    - early_stop_patience: The patience for early stopping
                    - early_stop_delay: The delay for early stopping
                    - early_stop_epsilon: The epsilon for early stopping
                    - eval_interval: The interval for intermediate evaluation
                    - eval_horizon: The horizon for intermediate evaluation
        '''
        self.debug = debug
        self.config = config
  
        # check if folder exists otherwise create it
        if not os.path.exists(NET_WEIGHTS_PATH):
            os.makedirs(NET_WEIGHTS_PATH)

        # Seed
        if config.seed is not None:
            torch.manual_seed(config.seed)  
            np.random.seed(config.seed)
  
        # Model
        self.horizon = config.prediction_horizon 
        self.model = model
        self.n = self.model.n
        self.initial_pos = getpos(initial_pos, self.n)
        self.model.reset(
            q0 = [self.initial_pos],
            p0 = [0]*self.model.n
        )
        self.q0 = self.model.q

        # Controller
        self.controller = controller
        self.controller.reset(self.initial_pos)
        self.target_pos = getpos(target_pos, self.n)  
        self.controller.set_target(self.target_pos)
        self.dt = self.controller.dt
 
        self.params2optimize = params2optimize 
        

        # Optimization
        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=params2optimize(), 
                lr=config.learning_rate
        )
            
        elif config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=params2optimize(), 
                lr=config.learning_rate, 
                momentum=config.optimizer_momentum, 
                dampening=config.dampening if "dampening" in config else 0.
        )
            
        elif config.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                params=params2optimize(), 
                lr=config.learning_rate, 
                momentum=config.optimizer_momentum
        )
            
        else:
            print(f"WRONG OPTIMIZER: {config.optimizer}. Choose among [andam, sgd, rmsprop]")
        
        self.steps = config.optimization_steps

        # Learning rate scheduler   
        if "learning_rate_decay_factor" in config and "learning_rate_decay_patience" in config:
            self.learning_rate_scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode = 'min', 
                factor = config.learning_rate_decay_factor,
                patience = config.learning_rate_decay_patience,
                cooldown = config.learning_rate_decay_cooldown if "learning_rate_decay_cooldown" in config else 0,
            ) 
        else:
            self.learning_rate_scheduler = None

        # Cost
        if config.num_initial_conditions is not None:
            self.num_x0 = config.num_initial_conditions
        else:
            self.num_x0 = 1
        self.cost = torch.tensor([0.]) 
        self.best_cost = np.inf 

        self.grad_clip_value = config.grad_clip_value  
        
        # Weights of the cost function 
        self.input_weight = torch.tensor(config.input_weight).float().reshape((self.model.m,1))
        self.initial_weight = torch.tensor(config.initial_weight).float().reshape((2*self.model.n,1))
        self.state_weight = torch.tensor(config.state_weight).float().reshape((2*self.model.n,1))
        self.final_weight = torch.tensor(config.final_weight).float().reshape((2*self.model.n,1))

        # Early stopping
        self.early_stop_threshold = config.early_stop_threshold
        self.early_stop_patience = config.early_stop_patience
        self.early_stop_delay = config.early_stop_delay 
        self.early_stop_epsilon = config.early_stop_epsilon

        # Evaluation
        self.eval = Eval(model=self.model, controller=self.controller, save_best_result=False, debug=self.debug )
        self.eval_interval = config.eval_interval
        self.eval_horizon = config.eval_horizon
      
    def objective(self):
        ''' Simulate the closed-loop dynamics and calculate the distance to the target point '''
        res = torch.tensor([0.])  
 
        for i in range(self.num_x0):

            # Reset model
            q0 = getpos(self.initial_pos, self.n)
            qd = getpos(self.target_pos, self.n)

            self.model.reset(q0=q0, p0=torch.zeros((self.n,1)))
            self.controller.reset(q0)
            self.controller.set_target(qd)
 
            # Initial state cost
            x0 = self.model.x
            xd = self.controller.xd    
            res = res + weighted_norm(x0, xd, self.initial_weight)  

            # desired control input at equilibrium
            _qd = torch.tensor(qd).float().reshape(self.n,1) 
            dHqd, dHpd = self.model._dH(_qd, torch.zeros_like(_qd)).split(self.n)

            ud = torch.inverse(self.model.B)@(dHqd + self.model.D @ dHpd)   
 
            for t in range(self.horizon):

                # Get state and output feedbacks
                q = self.model.q
                p = self.model.p

                # Compute the control action from feedback
                u = self.controller.get_action(q,p) 

                # Simulate the close-loop dynamics
                self.model.step(u=u, disc_grad=self.model.SAMPLE_VALUE_DISCRETE_GRADIENT)
                x = self.model.x.clone()
  
                # State and input cost  
                cx = weighted_norm(x, xd, self.state_weight)
                cu = weighted_norm(u, ud, self.input_weight)  
                res = res + (cx + cu)/self.horizon  

            # Final state cost
            res = res + weighted_norm(x, xd, self.final_weight) 

        return res/self.num_x0

    def clamp(self):
        ''' Clamp the parameters (not implemented in the base class) ''' 
        return

    def get_current_result(self):
        ''' Get the current result (not implemented in the base class) ''' 
        raise NotImplementedError 
    
    def additional_log(self):
        ''' Get the log dictionary (not implemented in the base class) ''' 
        return {}

    def optimize(self): 
        
        not_improvement_counter = 0

        # Optimization loop
        for i in range(self.steps): 

            # compute the cost across the prediction horizon
            self.cost = self.objective()
           
            print(f"Step {i} - Cost: {self.cost.item()} - Best Cost: {self.best_cost}")

            # make_dot(self.cost).render("graph", format="png")
            # print("DONE!\n") 
            # input("Press Enter to continue...")
            # # exit()

            # Log the cost to wandb  
            logdict = dict(cost = self.cost.item()) 
            logdict.update(self.additional_log())
            wandb.log(logdict)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # compute the gradients
            self.cost.backward(retain_graph=False)   
            torch.nn.utils.clip_grad_norm_(self.params2optimize(), self.grad_clip_value) # clip gradients

            # print("@@@@@@@@@@@@@@@@@@@", i, "@@@@@@@@@@@@@@@@@@@")  
            # for j,p in enumerate(self.params2optimize()): 
            #     print(j,"\t", "\t", list(p.size()) ) 
            #     print("grad\t", "\t","\t", round(np.linalg.norm(p.grad), 3))
            #     # print("grad\t", "\t","\t", np.round(p.grad.tolist(), 3).flatten()) 
            #     # print("value\t", "\t","\t", np.round(p.data.tolist(), 3).flatten()) 
            #     if not p.requires_grad:
            #         print(p.name, "NOT REQUIRES GRAD") 
            # # input("Press Enter to continue...")
  
            # update the parameters
            self.optimizer.step()

            # update the learning rate
            if self.learning_rate_scheduler is not None:
                self.learning_rate_scheduler.step(self.cost.item())

            # clamp the parameters if needed
            self.clamp()
 
            # early stopping  
            if i > self.early_stop_delay:
                not_improvement_counter += 1 if c-self.best_cost > self.early_stop_epsilon else 0 
                if self.best_cost < self.early_stop_threshold or not_improvement_counter >= self.early_stop_patience:
                    break
   
            # update the best cost
            c = self.cost.item() 
            if c < self.best_cost:
                self.best_cost = c 
                self.best_res = self.get_current_result()
            
            # intermediate evaluation 
            if self.eval_interval is not None and self.eval_horizon is not None:
                if i % self.eval_interval == 0: 
                    error, steps = self.eval.evaluate(
                        controller=self.controller,
                        horizon=self.eval_horizon, 
                        initial_pos=self.initial_pos, 
                        target_pos=self.target_pos, 
                        log=2 if self.debug else 0
                    )
                    print(f"Step {i} - Cost: {c} - Error: {error}/{self.eval.convergence_margin} - Steps: {steps}/{self.eval_horizon}") 