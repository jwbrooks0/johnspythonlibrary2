# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:15:23 2025

Misc. Machine learning and data science code.

@author: jbrooks
"""


import numpy as np
import xarray as _xr
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

# %% grids/domains

def generate_nd_grid(bounds, num_points, coord_names=[]):
    """
    Generates a uniform, structured N-dimensional grid within given bounds and returns:
    1. Unstructured grid: shape (M, N)
    2. Structured grid: shape (P, P, ..., P, N)
    3. List of N 1D arrays (coordinate vectors)

    Parameters
    ----------
    bounds : list of array-like
        A list of N elements, each with two values (lower and upper bounds) for one axis.
    num_points : int
        Number of points along each axis (P).

    Returns
    -------
    unstructured : np.ndarray
        Array of shape (M, N), where M = P^N, representing all grid points as rows.

    structured : np.ndarray
        Array of shape (P, ..., P, N), where the last axis stores the N coordinates 
        at each grid point location.

    coord_vectors : list of np.ndarray
        List of N 1D arrays, each of length P, representing the coordinate values along each axis.
        
    coords_xr : list of xr.DataArrays
        List of N 1D xr.DataArrays, each of length P, representing the coordinate values along each axis.

    Raises
    ------
    ValueError
        If inputs are invalid.
        
    Examples
    --------
    Example 1::
        
        bounds = [[0, 1], [0, 2], [0, 3]]  # 3D bounds
        num_points = 4
        unstructured, structured, structured_xr, coord_vectors, coords_xr = generate_nd_grid(bounds, num_points, coord_names=["apples", "bananas", "cherries"])
    """
    if not isinstance(bounds, list) or not all(len(b) == 2 for b in bounds):
        raise ValueError("Bounds must be a list of N elements, each with two values.")
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer.")

    N = len(bounds)
    
    # Generate coordinate vectors
    coord_vectors = [np.linspace(low, high, num_points) for low, high in bounds]

    # Generate meshgrid
    indexing='ij' # use "ij" as my default convention.  Scales well to large dimenions.  For 2D plots, will need to convert (post) to "xy" and matplotlib expects "xy" matrices.
    mesh = np.meshgrid(*coord_vectors, indexing=indexing) # 

    # Unstructured representation: flatten and stack
    unstructured = np.stack([m.flatten() for m in mesh], axis=-1)  # Shape (M, N)

    # Structured representation: stack along new last axis
    structured = np.stack(mesh, axis=-1)  # Shape (P, ..., P, N)
    
    # create an xarray for coords
    coords_xr = []
    for i in range(N):
        if len(coord_names) == N:
            coord_name_i = coord_names[i]
        else:
            coord_name_i = f"x{i+1}"
        x_i = coord_vectors[i]
        x_i_xr = _xr.DataArray(x_i, coords={coord_name_i: x_i})
        coords_xr.append(x_i_xr)
        
    # create an xarray for structured
    structured_xr = []
    for i in range(N):
        temp = _xr.DataArray(structured[..., i], coords=coords_xr)
        structured_xr.append(temp)
        
    return unstructured, structured, structured_xr, coord_vectors, coords_xr




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
        X1, Y1 = flattened_to_structured(flattened, shape)
        print("\nStructured Representation:")
        print(X1, Y1)
    """
    dim = len(grid)  # Number of dimensions
    flattened = np.column_stack([g.ravel() for g in grid])  # Flatten each coordinate array
    return flattened


def unstructured_to_structured(unstructured, structured_shape):
    """
    Convert an unstructured array to a mesh (structured) representation.

    Parameters
    ----------
    unstructured : np.ndarray
        The unstructured array of shape (M, K) or possibly (M,) if you're 
        converting to a single-channel mesh. Typically M = product of all 
        mesh_shape dimensions except perhaps the last one, which can 
        be a 'channel' or 'coordinate' dimension.
        
        For instance, if mesh_shape = (P, P, P, N), then unstructured.shape 
        should be (P^3, N).
        
    structured_shape : tuple of int
        Desired shape of the structured mesh, e.g. (P, P, P, N).

    Returns
    -------
    structured : np.ndarray
        The mesh of shape mesh_shape.

    Raises
    ------
    ValueError
        If the product of the dimensions of `mesh_shape` does not match the 
        total number of elements in `unstructured`.
    
    Examples
    --------
    Example 1::
        
         unstructured, structured, coord_vectors = generate_nd_grid([[0, 1], [0, 2], [0, 3]], 5)
         
         # Convert unstructured -> mesh
         mesh = unstructured_to_structured(unstructured, structured_shape=structured.shape)
         
         if (structured==mesh).sum() == 5 * 5 * 5 * 3:
             print("Success")
    """
    
    # Check total number of elements:
    # The product of mesh_shape must match unstructured.size
    if np.prod(structured_shape) != unstructured.size:
        raise ValueError(
            f"Cannot reshape unstructured array of size {unstructured.size} "
            f"into shape {structured_shape} which has size {np.prod(structured_shape)}."
        )
    
    # Reshape
    structured = unstructured.reshape(structured_shape)
    return structured



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
        X1, Y1 = flattened_to_structured(flattened, shape)
        print("\nStructured Representation:")
        print(X1, Y1)
    """
    dim = flattened.shape[1]  # Number of dimensions
    structured = tuple(flattened[:, i].reshape(shape) for i in range(dim))
    return structured


def generate_qmc_grid(bounds,
                      num_points,
                      seed=None,
                      verbose=False,
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
        if verbose is True:
            print(i, bound, diff, minn)
        grid[:, i] = grid_normalized[:, i] * diff + minn
    
    return grid    


# %% distance metrics

def euclidean_distance_point_to_point(x_1, x_2, w=None):
    """
    Compute the Euclidean distance between two N-dimensional points.

    Parameters
    ----------
    point1 : array_like
        The first point, represented as a list, tuple, or NumPy array.
    point2 : array_like
        The second point, represented as a list, tuple, or NumPy array.

    Returns
    -------
    float
        The Euclidean distance between `point1` and `point2`.

    Examples
    --------
    >>> euclidean_distance_point_to_point([1, 2, 3], [4, 5, 6])
    5.196152422706632
    >>> euclidean_distance_point_to_point([1, 0, 0], [0, 1, 0])
    1.4142135623730951
    """    
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)

    if x_1.shape != x_2.shape:
        raise ValueError("Both points must have the same number of dimensions.")

    return euclidean(x_1, x_2)


def euclidean_distance_pairwise(points):
    """
    Compute the pairwise Euclidean distance matrix for a set of points.

    Parameters
    ----------
    points : array_like, shape (M, N)
        A 2D array of M points in N-dimensional space.

    Returns
    -------
    distances : ndarray, shape (M, M)
        A square matrix where the entry (i, j) represents the Euclidean 
        distance between point i and point j.

    Raises
    ------
    ValueError
        If the input is not a 2D array.

    Examples
    --------
    Example 1::
        
        >>> points = np.array([[1, 2], [3, 4], [5, 6]])
        >>> euclidean_distance_pairwise(points)
        array([[0.        , 2.82842712, 5.65685425],
               [2.82842712, 0.        , 2.82842712],
               [5.65685425, 2.82842712, 0.        ]])
    """
    points = np.asarray(points)

    if points.ndim != 2:
        raise ValueError("Input must be a 2D array of shape (M, N).")

    return cdist(points, points, metric='euclidean')