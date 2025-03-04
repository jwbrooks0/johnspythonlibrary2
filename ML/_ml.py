# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:15:23 2025

@author: jbrooks
"""


import numpy as np

# %% grids/domains

def structured_to_flattened(grid):
    """
    Converts a structured grid representation (meshgrid) to a flattened M x N representation.
    
    Parameters:
        grid (tuple of numpy arrays): The structured grid with N arrays of shape (S1, S2, ..., SN).
    
    Returns:
        numpy.ndarray: An (M, N) array where M is the total number of grid points.
        
    Example:
        
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 4)
        X, Y = np.meshgrid(x, y, indexing='ij')  # (5, 4) grids
        
        # Convert to flattened representation
        flattened = structured_to_flattened((X, Y))
        print("Flattened Representation:\n", flattened)
        
        # Convert back to structured representation
        shape = X.shape
        structured = flattened_to_structured(flattened, shape)
        print("\nStructured Representation:")
        for arr in structured:
            print(arr)
    """
    dim = len(grid)  # Number of dimensions
    flattened = np.column_stack([g.ravel() for g in grid])  # Flatten each coordinate array
    return flattened

def flattened_to_structured(flattened, shape):
    """
    Converts a flattened M x N representation back into a structured grid representation.
    
    Parameters:
        flattened (numpy.ndarray): The flattened coordinate array of shape (M, N).
        shape (tuple): The original shape of the structured grid.
    
    Returns:
        tuple of numpy arrays: Structured coordinate arrays of shape `shape`.
        
    Example:
        
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 4)
        X, Y = np.meshgrid(x, y, indexing='ij')  # (5, 4) grids
        
        # Convert to flattened representation
        flattened = structured_to_flattened((X, Y))
        print("Flattened Representation:\n", flattened)
        
        # Convert back to structured representation
        shape = X.shape
        structured = flattened_to_structured(flattened, shape)
        print("\nStructured Representation:")
        for arr in structured:
            print(arr)
    """
    dim = flattened.shape[1]  # Number of dimensions
    structured = tuple(flattened[:, i].reshape(shape) for i in range(dim))
    return structured


def generate_qmc_grid(bounds,
                      num_points,
                      seed=None,
                      ):
    
    """
    Sometimes, I need a random-uniform distribution of points that's a little more evenly distributed.  This is that.  
    
    Examples
    --------
    
    Example 1 ::
        
        ## 1D example
        bounds = [[1, 2]]
        num_points = 35
        seed = 0
        X_qmc = generate_qmc_grid(bounds, num_points, seed=seed)
        print(X_qmc)
        _np.random.seed(seed)
        X_uniform = _np.random.uniform(*bounds[0], size=num_points)
        
        fig, ax = _plt.subplots()
        ax.plot(X_qmc, X_qmc, ls="", marker=".", label="qmc")
        ax.plot(X_uniform, X_uniform, ls="", marker="x", label="random-uniform")
        ax.set_xlim(bounds[0])
        ax.set_title("Note that the random-uniform clumps up more than the qmc.")
        ax.legend()
    
    Example 2 ::
    
        ## 2D example
        bounds = [[1, 2], [5, 8]]
        num_points = 500
        seed = 0
        X_qmc = generate_qmc_grid(bounds, num_points, seed=seed)
        print(X_qmc)
        _np.random.seed(seed)
        X_uniform = _np.zeros(X_qmc.shape)
        X_uniform [:, 0] = _np.random.uniform(*bounds[0], size=num_points)
        X_uniform [:, 1] = _np.random.uniform(*bounds[1], size=num_points)
        
        fig, ax = _plt.subplots()
        ax.plot(X_qmc[:, 0], X_qmc[:, 1], ls="", marker=".", label="qmc")
        ax.plot(X_uniform[:, 0], X_uniform[:, 1], ls="", marker="x", label="random-uniform")
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_title("Note that the random-uniform clumps up more than the qmc.")
        ax.legend()
        
        
    References
    ----------
    
     * https://blog.scientific-python.org/scipy/qmc-basics/
     * https://scipy.github.io/devdocs/reference/stats.qmc.html
        
    """
    assert len(np.shape(bounds)) == 2, "bounds must be a 2D array"
    from scipy.stats import qmc
    
    m, _ = np.shape(bounds)
    
    ## generate quasi-random points
    generator = qmc.Sobol(m, seed=seed)
    grid_normalized = generator.random(num_points)
    
    ## scale values to be within bounds
    grid = grid_normalized * 1.0
    for i, bound in enumerate(bounds):
        diff = np.abs(np.subtract(*bound))
        minn = bound[0]
        print(i, bound, diff, minn)
        grid[:, i] = grid_normalized[:, i] * diff + minn
    
    return grid    


# %% distance metrics
def euclidean_distance(x_1, x_2, w=0):
    """
    Calculates the euclidean distance between two ND arrays.
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html
    """
    
    from scipy.spatial import distance
    return distance(x_1, x_2, w)