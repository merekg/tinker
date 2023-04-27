import numpy as np
import sys
import scipy.io.wavfile as wf
import scipy.signal as sig
import matplotlib.pyplot as plt

wave = np.fromfile(sys.argv[1],sep=' ').reshape((-1,16))
wave_sum = np.sum(wave, axis=1)
for el in wave_sum:
    print(el)

#plt.plot(wave_formatted)
#plt.show()

#wf.write(sys.argv[2], 11000,wave_formatted)
