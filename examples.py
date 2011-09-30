import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def foo(x):
    return x[0]**2 + 10*np.sin(x[1])


def foo_surface_plot(myranges):
    '''Generate 3D line plot showing foo()'''
    global X, Y, Z
    x = np.linspace(*myranges[0], num=100)
    y = np.linspace(*myranges[1], num=100)
    X, Y = np.meshgrid(x, y)
    Z = np.ndarray(X.shape)
    for i, k in zip(range(X.shape[0]), range(X.shape[1])):
        Z[i][k] = foo([X[i][k], Y[i][k]])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.legend()


def foo_plot(myranges):
    '''Generate 3D line plot showing foo()'''
    x = np.linspace(*myranges[0], num=100)
    y = np.linspace(*myranges[1], num=100)
    z = np.array([foo(v) for v in zip(x, y)])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()


if __name__=="__main__":
    # Method 1 (same number of steps for each parameter)
    myranges = ((-50, 50), (-30, -10))
    result = optimize.brute(foo, myranges)

    #Try anneal
    aresult = optimize.anneal(foo, [0, 0])
    X = Y = Z = None
