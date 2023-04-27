import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    p0 = 330
    r1= .17
    r2= .12
    r3 = .08

    t = np.linspace(0,5)
    pt1 = p0*np.exp(r1 * t)
    pt2 = p0*np.exp(r2 * t)
    pt3 = p0*np.exp(r3 * t)

    plt.plot(t, pt1, 'g')
    plt.plot(t, pt2, 'y')
    plt.plot(t, pt3, 'r')
    plt.show()
