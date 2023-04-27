#!/usr/bin/env python -B
import matplotlib.pyplot as plt
import numpy as np
__author__ = 'Arvi Cheryauka'

# Handle NRRD / NHDR header i/on maps, in AcqHome + GainMap directory


def image(Img, angle, mapV, minV, maxV):
    nU, nV = Img.shape
    plt.imshow(Img, cmap=mapV, vmin=minV, vmax=maxV)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('X/U, pixels')
    plt.ylabel('Y/V, pixels')
    plt.xlim([1, nU])
    plt.ylim([1, nV])
    plt.text(50, -50, 'Image @ Angle = ' + "{:.3f}".format(angle) + ' deg',
             color='r')
#    plt.colorbar()


def error(erms, emax, angle):
    plt.scatter(angle, np.log10(erms), color='g', marker='.', s=10)
    plt.scatter(angle, np.log10(emax), color='c', marker='.', s=10)
    plt.xlabel('Angle, [degree]')
    plt.ylabel('Log10 distance, [mm]')
    plt.xlim([0, 360])
#    plt.ylim([-2, 1])
    plt.text(300, 0.75, '* RMS', color='g')
    plt.text(300, 0.5,  '* MAX', color='c')
    plt.title('Overall RMS / MAX errors = ' +
              "{:.3f} / {:.3f}".format(erms.mean(), emax.max()) + ' mm',
              color='r')
