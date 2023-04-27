import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import misc
import glob
import matplotlib.pyplot as plt

def main():
    
    image = misc.imread("/home/nview/cat.png")
    r = R.from_quat([0,0, np.sin(np.pi/4), np.cos(np.pi/4)])
    #rotated = r.apply(image)

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
