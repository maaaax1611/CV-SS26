import numpy as np

def computeBilinerWeights(q):
    """Compute bilinear weights for point q
    Args:
        q: 2D point (x, y) 

    Returns:
        weights: 4D vector of bilinear weights for the four surrounding pixels
                 [w(x,y), w(x+1,y), w(x,y+1), w(x+1,y+1)]
    """
    x,y = q
    a = x - np.floor(x)
    b = y - np.floor(y)

    w00 = (1 - a) * (1 - b)   # I(x, y)
    w10 = a * (1 - b)         # I(x+1, y)
    w01 = (1 - a) * b         # I(x, y+1)
    w11 = a * b               # I(x+1, y+1)

    return [w00, w10, w01, w11]


def computeGaussianWeights(winsize, sigma):
    """
    Compute Gaussian weights for a given window size and standard deviation
    Args:
        winsize: tuple (width, height) of the window size
        sigma: standard deviation of the Gaussian

    Returns:
        weights: 2D array of Gaussian weights for the window
    """
    width, height = winsize
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    # transform to centered, normalized coordinates
    x = (np.arange(width) - center_x) / width
    y = (np.arange(height) - center_y) / height
    X, Y = np.meshgrid(x, y)
    weights = np.exp(-(X**2 + Y**2) / (2*sigma**2))
    
    return np.array(weights)


def invertMatrix2x2(A):
    """
    Invert a 2x2 matrix A
    Args:
        A: 2x2 matrix
    
    Returns:
        invA: Inverse of matrix A
    """
    # Compute the determinant
    # det = ad - bc for a 2x2 matrix [[a, b], [c, d]]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    invA = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
    return invA
