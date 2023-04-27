import numpy as np
import sys
import matplotlib.pyplot as plt

navcals = []
for fp in sys.argv[1:]:
    navcals.append(np.genfromtxt(fp, delimiter=' '))

o = np.array([0,0,0,1])
u = np.array([150,0,0,1])

ax = plt.figure().add_subplot(projection='3d')
uts = []
i  = 0 
for navcal in navcals:
    i = i+1
    ot = np.matmul(navcal, o)
    ut = np.matmul(navcal, u)
    print(ut)
    uts.append(ut)
    ax.scatter(ut[0], ut[1], ut[2], label=i)
    #ax.quiver(ot[0], ot[1], ot[2], ut[0], ut[1], ut[2], length=1, normalize=True)


ax.legend()
plt.show()
