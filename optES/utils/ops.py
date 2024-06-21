import numpy as np
import torch
from scipy.optimize import fsolve

def intregral_approximation(f, a, b, params):
    '''
    Approximates the integral of a function f(x) between a and b using the trapezoidal rule.
    @ param f: function to integrate
    @ param a: lower limit of integration
    @ param b: upper limit of integration
    @ param params: dictionary of parameters, including: 
        - integral_samples: number of sample points used to compute the integral with trapezoidal rule.
    @ return: integral of f(x) between a and b
    '''
    samples = params["integral_samples"]

    # step size
    ds = (b - a) / samples

    # evalutation at the extremities
    fa = f(a)
    fb = f(b)

    # initialization
    val = fa + fb

    # compute the sum of the middle terms
    for i in range(1,samples):
        s = a + i*ds
        fs = f(s)
        val = val + 2*fs

    # final result
    integral = val*ds/2

    return integral

def root_finder(f, x0, params):
    if params["root_finder"] == "fsolve":
        return scipy_fsolve(f, x0, params)
    if params["root_finder"]  == "newton":
        return newton_raphson(f, x0, params)
 
def scipy_fsolve(f, x0, params): 
    '''
    Implements the fsolve method to find the root of a function.
    @ param f: function to find the root of f
    @ param x0: initial guess for the root
    @ param params: dictionary of parameters, including:
        - fsolve_tol: tolerance for the norm of the function f(x)
        - fsolve_max_iter: maximum number of iterations (optional)
        
    ValueError: if the maximum number of iterations is exceeded.
    '''

    tol = params.get("root_finder_tol", 1.49012e-8)  # default tolerance in fsolve
    max_iter = params.get("root_finder_max_iter", 1e6)  # default max_iter in fsolve
     
    x0_numpy = x0.clone().float().detach().numpy()
    n = x0_numpy.shape[0] 

    # Convert the PyTorch function and initial guess to a format that fsolve can handle
    # f_numpy = lambda x: f(torch.from_numpy(x)).numpy()
    def f_numpy(x): 
        x = torch.from_numpy(x).reshape((n,1)).float() 
        y = f(x).reshape((n,)).float().detach().numpy()  
        return y
 
    # Call fsolve
    root, info, ier, msg = fsolve(f_numpy, x0_numpy, xtol=tol, maxfev=int(max_iter), full_output=True)
 
    if ier != 1:
        print(f"Warning: fsolve did not converge. Message: {msg}")

    # Convert the root back to a PyTorch tensor
    root = torch.from_numpy(root.reshape((n,1))).float()

    return root


def newton_raphson(f, x0, params): 
    '''
    Implements the Newton-Raphson method to find the root of a PyTorch function.
    @ param f: PyTorch function to find the root of f
    @ param x0: initial guess for the root
    @ param params: dictionary of parameters, including:
        - root_finder_tol: tolerance for the norm of the function f(x)
        - root_finder_tol2: tolerance for the norm of the function f(x) (optional)
        - root_finder_max_iter: maximum number of iterations (optional)
        
    ValueError: if the maximum number of iterations is exceeded.
    '''
    print("Using Newton-Raphson method to find the root of a PyTorch function.")
    tol = params["root_finder_tol"] 
    tol2 = params.get("root_finder_tol2", None) 
    max_iter = params.get("root_finder_max_iter", 1e6) 
    stepsize = params.get("root_finder_stepsize", 1)

    x = x0.clone()
    # print(f"\n\n Newton-Raphson initial guess: x0={x0}")
    n = x.shape[0]
    for i in range(int(max_iter)):
        y = f(x)
        # print(f"Newton-Raphson iteration {i}: y={torch.norm(y)}")
        if torch.norm(y) < tol: 
            # print(f"Newton-Raphson converged after {i} iterations: res={torch.norm(y)}<tol={tol}\n")
            return x 
        if torch.norm(y) == torch.inf:  
            break
        dy = torch.autograd.functional.jacobian(f, x, create_graph=True)  
        dy[dy!=dy] = 0  # replace nan values with 0
        try: 
            x = x.reshape((n,1))
            y = y.reshape((n,1))
            dy = dy.reshape((n,n))
            x = x -  stepsize*torch.matmul(torch.linalg.pinv(dy),y)
            # print(f"x={x}")
            # input("Press Enter to continue...")
        except Exception as e:
            raise ValueError(e)
    
    if tol2 is not None:
        if torch.norm(y) < tol2:
            # print(f"Newton-Raphson converged after {i} iterations: res={torch.norm(y)}<tol={tol2}")
            return x
        else:
            raise ValueError(f"Failed to converge. After {max_iter} iteration: res={torch.norm(y)}>tol2={tol2}")
    else:
        raise ValueError(f"Failed to converge. After {max_iter} iteration: res={torch.norm(y)}>tol={tol}")

def transpose(x):
    '''
    Transpose a torch tensor
    @ param x: torch tensor
    @ return: transpose of x
    '''
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def weighted_norm(data_point1, data_point2, weights):
    """
    Compute the weighted L2 distance between two data points.

    Args:
        data_point1 (torch.Tensor): The first data point (1D tensor).
        data_point2 (torch.Tensor): The second data point (1D tensor) with the same shape as data_point1.
        weights (torch.Tensor): Weights for each element of the data points (1D tensor) with the same shape as data_point1.

    Returns:
        torch.Tensor: The weighted L2 distance as a scalar tensor.
    """
    assert data_point1.shape == data_point2.shape == weights.shape, f"Shapes of data_point1={data_point1.shape}, data_point2={data_point2.shape}, and weights={weights.shape} must be the same."

    diff = data_point1 - data_point2 
    return diff.t() @ torch.diag(weights.flatten()) @ diff