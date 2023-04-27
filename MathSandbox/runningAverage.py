import numpy as np
import matplotlib.pyplot as plt

def main():
    # generate the data
    x = np.random.rand(100)

    # run the running average, printing each time
    runAvg = np.zeros_like(x)
    for i, el in enumerate(x):
        if i==0:
            runAvg[i] = x[i]
        else:
            runAvg[i] = (el + i*runAvg[i-1])/(i+1)

    plt.plot(x)
    plt.plot(runAvg)
    plt.show()
if __name__ == "__main__":
    main()
