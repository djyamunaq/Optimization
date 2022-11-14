import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import meshgrid

def f2Plot(x, y):
    J_vals = np.zeros((len(x), len(y)))
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            J_vals[i, j] = f2(np.array([x[i], y[j]]))

    fig = plt.figure()
    X,Y = np.meshgrid(x, y)
    cp = plt.contour(X, Y, J_vals, colors='black', linestyles='dashed', linewidths=1)
    cp = plt.contourf(X, Y, J_vals)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    
    return fig

def f2(x):
    x0 = x[0]
    x1 = x[1]

    return 5*(x0**2) + (x1**2)/2 - 3*(x0 + x1)

def grad2(x):
    x0 = x[0]
    x1 = x[1]

    dx1 = 10*x0 - 3
    dx2 = x1 - 3

    grad = np.array([dx1, dx2])
    
    return grad

def constStepGD(F, x0, gradF, tol=1e-6, Niter=100):
    x_vals = []
    it = Niter
    x = x0
    J = F(x)
    G = gradF(x)
    step = 1e-2

    while it > 0 and np.linalg.norm(G) > tol:
        x = x - step*G
        x_vals.append(x)
        J = F(x)
        G = gradF(x)
        it -= 1

        # print(abs(J))

    return x, x_vals

def optStepGD(F, x0, gradF, A, tol=1e-6, Niter=100):
    x_vals = []
    it = Niter
    x = x0
    J = F(x)
    G = gradF(x)
    d = -G

    while it > 0 and np.linalg.norm(G) > tol:
        # Optimized Step
        step = (np.linalg.norm(G)**2)/(np.dot(d, np.dot(A, d)))   

        # Update values
        x = x + step*d
        x_vals.append(x)
        J = F(x)
        G = gradF(x)
        d = -G

        # print(abs(J))
        # print(step)

        it -= 1

    return x, x_vals

def conjGD(F, x0, gradF, A, tol=1e-6, Niter=100):
    x_vals = []
    it = Niter
    x = x0
    x_vals.append(x0)

    J = F(x)
    G = gradF(x)
    d = -G

    while it > 0 and np.linalg.norm(G) > tol:
        # Optimized step
        step = (np.linalg.norm(G)**2)/(np.dot(d, np.dot(A, d)))   
        x = x + step*d
        # Conjugate direction
        beta = np.dot(d, np.dot(A, gradF(x)))/np.dot(d, np.dot(A, d))
        d = -gradF(x) + beta*d

        # Update values
        x_vals.append(x)
        J = F(x)
        G = gradF(x)

        print('Iteration:', 100 - it)
        print('\tCost: ', abs(J))
        print('\tStep: ', step)
        print('\tbeta: ', beta)

        it -= 1

    return x, x_vals

def main():
    x=np.linspace(-10,10,10)
    x = x.reshape((len(x), 1))
    y=np.linspace(-10,10,10)
    y = y.reshape((len(y), 1))

    f2Plot(x, y)

    tol = 1e-6 
    Niter = 1e2

    # ========================================================================================
    # Constant step gradient descent    
    # res1, x_vals1 =  constStepGD(F=f2, x0=np.array([-6, 6]), gradF=grad2, tol=tol, Niter=Niter)
    # res2, x_vals2 =  constStepGD(F=f2, x0=np.array([0, 0]), gradF=grad2, tol=tol, Niter=Niter)
    # res3, x_vals3 =  constStepGD(F=f2, x0=np.array([-2, -7]), gradF=grad2, tol=tol, Niter=Niter)
    # print(res1)
    # print(res2)
    # print(res3)
    # x_vals1 = np.array(x_vals1)
    # x_vals2 = np.array(x_vals2)
    # x_vals3 = np.array(x_vals3)
    
    # for i in range(len(x_vals1) - 1):
    #     # Plot GD for x0=[-6, 6]
    #     plt.scatter(x_vals1[i, 0], x_vals1[i, 1], color='r', s=1e1)
    #     plt.scatter(x_vals1[i+1, 0], x_vals1[i+1, 1], color='r', s=1e1)
    #     plt.plot((x_vals1[i, 0], x_vals1[i+1, 0]), (x_vals1[i, 1], x_vals1[i+1, 1]), linestyle='solid', color='r', linewidth=1)
    #     # Plot GD for x0=[0, 0]
    #     plt.scatter(x_vals2[i, 0], x_vals2[i, 1], color='g', s=1e1)
    #     plt.scatter(x_vals2[i+1, 0], x_vals2[i+1, 1], color='g', s=1e1)
    #     plt.plot((x_vals2[i, 0], x_vals2[i+1, 0]), (x_vals2[i, 1], x_vals2[i+1, 1]), linestyle='solid', color='g', linewidth=1)
    #     # Plot GD for x0=[-2, -7]
    #     plt.scatter(x_vals3[i, 0], x_vals3[i, 1], color='b', s=1e1)
    #     plt.scatter(x_vals3[i+1, 0], x_vals3[i+1, 1], color='b', s=1e1)
    #     plt.plot((x_vals3[i, 0], x_vals3[i+1, 0]), (x_vals3[i, 1], x_vals3[i+1, 1]), linestyle='solid', color='b', linewidth=1)

    # ========================================================================================
    # Optimized step gradient descent    

    # A = np.array([[10, 0], [0, 1]])
    # res1, x_vals1 =  optStepGD(F=f2, x0=np.array([-6, 6]), A=A, gradF=grad2, tol=tol, Niter=Niter)
    # res2, x_vals2 =  optStepGD(F=f2, x0=np.array([0, 0]), A=A, gradF=grad2, tol=tol, Niter=Niter)
    # res3, x_vals3 =  optStepGD(F=f2, x0=np.array([-2, -7]), A=A, gradF=grad2, tol=tol, Niter=Niter)
    # print(res1)
    # print(res2)
    # print(res3)
    # x_vals1 = np.array(x_vals1)
    # x_vals2 = np.array(x_vals2)
    # x_vals3 = np.array(x_vals3)
    
    # for i in range(len(x_vals1) - 1):
    #     # Plot GD for x0=[-6, 6]
    #     plt.scatter(x_vals1[i, 0], x_vals1[i, 1], color='r', s=1e1)
    #     plt.scatter(x_vals1[i+1, 0], x_vals1[i+1, 1], color='r', s=1e1)
    #     plt.plot((x_vals1[i, 0], x_vals1[i+1, 0]), (x_vals1[i, 1], x_vals1[i+1, 1]), linestyle='solid', color='r', linewidth=1)
    #     # Plot GD for x0=[0, 0]
    #     plt.scatter(x_vals2[i, 0], x_vals2[i, 1], color='g', s=1e1)
    #     plt.scatter(x_vals2[i+1, 0], x_vals2[i+1, 1], color='g', s=1e1)
    #     plt.plot((x_vals2[i, 0], x_vals2[i+1, 0]), (x_vals2[i, 1], x_vals2[i+1, 1]), linestyle='solid', color='g', linewidth=1)
    #     # Plot GD for x0=[-2, -7]
    #     plt.scatter(x_vals3[i, 0], x_vals3[i, 1], color='b', s=1e1)
    #     plt.scatter(x_vals3[i+1, 0], x_vals3[i+1, 1], color='b', s=1e1)
    #     plt.plot((x_vals3[i, 0], x_vals3[i+1, 0]), (x_vals3[i, 1], x_vals3[i+1, 1]), linestyle='solid', color='b', linewidth=1)


    # ========================================================================================
    # Optimized step gradient descent    

    A = np.array([[10, 0], [0, 1]])
    res1, x_vals1 =  conjGD(F=f2, x0=np.array([-6, 6]), A=A, gradF=grad2, tol=tol, Niter=Niter)
    res2, x_vals2 =  conjGD(F=f2, x0=np.array([0, 0]), A=A, gradF=grad2, tol=tol, Niter=Niter)
    res3, x_vals3 =  conjGD(F=f2, x0=np.array([-2, -7]), A=A, gradF=grad2, tol=tol, Niter=Niter)
    print(res1)
    print(res2)
    print(res3)
    x_vals1 = np.array(x_vals1)
    x_vals2 = np.array(x_vals2)
    x_vals3 = np.array(x_vals3)
    
    for i in range(len(x_vals1) - 1):
        # Plot GD for x0=[-6, 6]
        plt.scatter(x_vals1[i, 0], x_vals1[i, 1], color='r', s=1e1)
        plt.scatter(x_vals1[i+1, 0], x_vals1[i+1, 1], color='r', s=1e1)
        plt.plot((x_vals1[i, 0], x_vals1[i+1, 0]), (x_vals1[i, 1], x_vals1[i+1, 1]), linestyle='solid', color='r', linewidth=1)
        # Plot GD for x0=[0, 0]
        plt.scatter(x_vals2[i, 0], x_vals2[i, 1], color='g', s=1e1)
        plt.scatter(x_vals2[i+1, 0], x_vals2[i+1, 1], color='g', s=1e1)
        plt.plot((x_vals2[i, 0], x_vals2[i+1, 0]), (x_vals2[i, 1], x_vals2[i+1, 1]), linestyle='solid', color='g', linewidth=1)
        # Plot GD for x0=[-2, -7]
        plt.scatter(x_vals3[i, 0], x_vals3[i, 1], color='b', s=1e1)
        plt.scatter(x_vals3[i+1, 0], x_vals3[i+1, 1], color='b', s=1e1)
        plt.plot((x_vals3[i, 0], x_vals3[i+1, 0]), (x_vals3[i, 1], x_vals3[i+1, 1]), linestyle='solid', color='b', linewidth=1)

    # ========================================================================================

    plt.show()


if __name__=="__main__":
    main()