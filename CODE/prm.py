"""
Probabilistic Road Map (PRM) Planner

Reference:
Sampling-based algorithms for optimal motion planning
[https://doi.org/10.1177/0278364911406761]
"""


from scipy import interpolate
from math import floor, sqrt
import numpy as np


# Randomly sampling n nodes while avoiding the obstacles.
def SampleFree(n, xL, xH, yL, yH, G, seed=0):
    Dx = xH - xL
    Dy = yH - yL

    sample = np.zeros((0, 2))
    rng = np.random.default_rng(seed)
    while sample.shape[0] < n:
        nNew = int(n) - sample.shape[0]
        raw = rng.random((nNew, 2))
        sample_raw = raw * [Dx, Dy] + [xL, yL]
        # Only store the nodes that lie in free space.
        freeMask = inFreeSpace(sample_raw, G)
        sample = np.append(sample, freeMask, axis=0)

    return sample


# Check if the sampled points lie in free space.
def inFreeSpace(p, G):
    # Find the points that lie within predefined grid.
    inBounds = np.where((p[:,0] >= G.xL) & (p[:,0] <= G.xH) & (p[:,1] >= G.yL) & (p[:,1] <= G.yH))[0].tolist()
    p_bound = p[inBounds]

    # Convert the coordinates into grid indices.
    nX = G.obsMask.shape[1]
    nY = G.obsMask.shape[0]
    obs_width = (G.xH-G.xL)/nX
    obs_height = (G.yH-G.yL)/nY

    p_x = list()
    p_y = list()
    for i, _ in enumerate(p_bound):
        p_x.append(min(nX-1, floor((p_bound[i][0] - G.xL) / obs_width)))
        p_y.append(min(nY-1, floor((G.yH - p_bound[i][1]) / obs_height)))

    # Find the corresponding grid in the obstacles mask.
    obsQuery = list()
    for i in range(len(p_x)):
        obsQuery.append(G.obsMask[p_y[i], p_x[i]]^1)

    # In free space if point is inside the bounds AND
    # if the grid cell it sits in does not have an obstacle
    freeMask = inBounds.copy()
    indices = [i for i, e in enumerate(obsQuery) if e == 0]
    for index in sorted(indices, reverse=True):
        del freeMask[index]

    return p_bound[freeMask]


# Check if the edge between 2 points cross any obstacle.
def CollisionFree(G, v, u):
    # Linearly interpolate points between v and u.
    if abs(u[0] - v[0]) <= 0.001:
        # If the x coordinates are two close just draw a straight line.
        if u[1] > v[1]:
            y = np.arange(v[1], u[1], 0.0005)
        else:
            y = np.arange(u[1], v[1], 0.0005)
        x = np.ones(y.shape[0])*u[0]
    else:
        if u[0] > v[0]:
            x_index = np.array([v[0], u[0]])
            y_index = np.array([v[1], u[1]])
            x = np.arange(v[0], u[0], 0.0005)
        else:
            x_index = np.array([u[0], v[0]])
            y_index = np.array([u[1], v[1]])
            x = np.arange(u[0], v[0], 0.0005)
        f = interpolate.interp1d(x_index, y_index)
        y = f(x)

    linePoints = np.zeros((0, 2))
    for i, _ in enumerate(x):
        linePoints = np.append(linePoints, np.array([[x[i], y[i]]]), axis=0)

    # Check if they are in free space
    freeMask = inFreeSpace(linePoints, G)

    return np.array_equal(linePoints, freeMask)


# Euclindean distance from u to v.
def Euclindean_dist(u, v):
    return np.linalg.norm(u - v)


 # Return nodes within a radius threshold from a referene node.
def Near_radius(G, v, r):
    U = []
    for u in G.nodes.items():
        dist = Euclindean_dist(u[1], v)
        if dist <= r and (u[1][0] != v[0] or u[1][1] != v[1]):
            U.append(u)
    return np.array(U, dtype=object)


# Function to return the minimum distance between a line segment AB and a point E.
def minDistance(A, B, E):
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
    # vector BE
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]
    # vector AE
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]
    # Minimum distance from point E to the line segment
    reqAns = 0

    if (AB_BE > 0):
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = sqrt(x * x + y * y)
    elif (AB_AE < 0):
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = sqrt(x * x + y * y)
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = sqrt(x1 * x1 + y1 * y1)
        reqAns = abs(x1 * y2 - y1 * x2) / mod

    return reqAns
