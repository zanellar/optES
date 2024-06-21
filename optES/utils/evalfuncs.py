import torch
import numpy as np

def get_energy(model, controller, min=-6.28, max=6.28, step=0.1):
    
    xspace = torch.arange(min, max, step)

    n = controller.model.n
 
    Vdspace = np.zeros((n,n,len(xspace)))
    Vdspace.fill(np.nan) 
    Vspace = np.zeros((n,n,len(xspace)))
    Vspace.fill(np.nan)

    yspace = dict(
        Vd = Vdspace,
        V = Vspace
    )

    for j in range(n): 
        for k, qj in enumerate(xspace):  
            
            # Set the state of the model with all zeros except for the j-th state  
            q = torch.zeros((n,1)).float()
            q[j] = qj.unsqueeze(0) 

            p = torch.zeros_like(q)
            H = model.H(q, p) 
            x = torch.cat((q,p)).float()
            w = x.clone()
            xd = controller.xd.reshape(x.shape)
            Ha = controller.Ha(x - xd, w - xd) 
            Hd = H + Ha
            V =  model.V(q)
            Vd = Hd - model.K(q,p) - V 

            Vd = Vd.clone().flatten().detach().numpy()
            V = V.clone().flatten().detach().numpy()

            if n == 1:
                Vd = Vd.item()
                V = V.item()
            
            yspace["Vd"][j,:,k] = Vd 
            yspace["V"][j,:,k] = V 
    
    return xspace, yspace 

def get_p11(controller, min=-10, max=10, step=0.1):
     
    xspace = torch.arange(min, max, step)

    n = controller.model.n
 
    yspace = np.zeros((n,n,len(xspace)))
    yspace.fill(np.nan) 

    for j in range(n): 

        for k, xj in enumerate(xspace):  
 
            # Set the state of the model with all zeros except for the j-th state
            x = torch.zeros((n,1)).float()
            x[j] = xj.unsqueeze(0) 

            pdiag = controller.pdiag(x, torch.zeros_like(x)).detach().flatten().numpy()

            if n == 1:
                pdiag = pdiag.item() 
 
            yspace[j,:,k] = pdiag    
        
    return xspace, yspace