import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import meshgrid

# Landmark 1 position
l1cx = -1
l1cy = 0
# Landmark 2 position
l2cx = 1
l2cy = 0
# Landmark 3 position
l3cx = 0
l3cy = 1

# Distance between robot and landmarks
r1 = 1
r2 = 1
r3 = 1
data = np.array([r1, r2, r3])

def plotLandmarks():
    t = np.linspace(0, 7, 100)

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    # Landmark 1
    cx = l1cx
    cy = l1cy
    x = cx + r1*np.cos(t)
    y = cy + r1*np.sin(t) 
    plt.plot(x, y, color='b')
    plt.scatter(cx, cy, color='b')

    # Landmark 2
    cx = l2cx
    cy = l2cy
    x = cx + r2*np.cos(t)
    y = cy + r2*np.sin(t) 
    plt.plot(x, y, color='b')
    plt.scatter(cx, cy, color='b')

    # Landmark 3
    cx = l3cx
    cy = l3cy
    x = cx + r3*np.cos(t)
    y = cy + r3*np.sin(t) 
    plt.plot(x, y, color='b')
    plt.scatter(cx, cy, color='b')

def m(theta):
  x = theta[0]
  y = theta[1]

  # distance to landmark 1
  d1 = ((x - l1cx)**2 + (y - l1cy)**2)**0.5
  # distance to landmark 2
  d2 = ((x - l2cx)**2 + (y - l2cy)**2)**0.5
  # distance to landmark 3
  d3 = ((x - l3cx)**2 + (y - l3cy)**2)**0.5

  return np.array([d1, d2, d3])

def fun(theta):
  return m(theta) - data

def main():
    theta0 = [0, 0]
    res = least_squares(fun, theta0)
    print("Robot pos:", res.x)

    # Print distances calculated from robot position estimated to landmarks    
    robot_pos = np.array(res.x)
    d1 = np.linalg.norm(robot_pos - np.array([l1cx, l1cy]))
    print("d1:", d1) 
    d2 = np.linalg.norm(robot_pos - np.array([l2cx, l2cy]))
    print("d2:", d2)
    d3 = np.linalg.norm(robot_pos - np.array([l3cx, l3cy]))
    print("d3:", d3)

    plotLandmarks()

    # Plot robot position
    plt.scatter(res.x[0], res.x[1], color='r')

    plt.show()

if __name__=="__main__":
    main()