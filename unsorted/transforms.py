from scipy.ndimage import rotate, interpolation
import numpy as np
import matplotlib.pyplot as plt

def scaleit(image, factor, isseg=False):
    order = 0 if isseg == True else 3

    height, width,depth = image.shape
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = int(np.round(factor * depth))

    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row     = int((height - zheight) / 2)
        col     = int((width - zwidth) / 2)
        dep     = int((depth - zdepth) / 2)
        newimg[row:row+zheight, col:col+zwidth, dep:dep+zdepth] = interpolation.zoom(image, (float(factor), float(factor), float(factor)), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        dep     = (zdepth - depth) // 2    
        newimg = interpolation.zoom(image[row:row+zheight, col:col+zwidth, dep:dep+zdepth], (float(factor), float(factor), float(factor)), order=order, mode='nearest')  
        
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        return newimg

    else:
        return image


def resampleit(image, dims, isseg=False):
    order = 0 if isseg == True else 5

    image = interpolation.zoom(image, np.array(dims)/np.array(image.shape, dtype=np.float32), order=order, mode='nearest')

    if isseg:
        image[np.where(image==4)]=3
        
    return image if isseg else (image-image.min())/(image.max()-image.min()) 
   

def translateit(image, offset, isseg=False, mode='nearest'):
    order   = 0 if isseg else 5
    mode    ='nearest' if isseg else 'mirror'
    offset  = offset if image.ndim == 2 else (int(offset[0]), int(offset[1]),int(offset[2]))

    return interpolation.shift(image, offset , order=order, mode=mode)


def rotateit(image, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return rotate(image, float(theta), reshape=False, order=order, mode='nearest')


def flipit(image, axes):
    
    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)
    
    return image


def intensifyit(image, factor):

    return image*float(factor)

def noisy(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy


