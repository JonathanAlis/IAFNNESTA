import numpy as np
from scipy import sparse
import sys
import random as rd
import matplotlib.pyplot as plt
from PIL import Image
import waveletDec as wd
import IAFNNESTA
import IAFNNesterov

def help():
    return '''
Here we compare the tv reconstruction with the L1 reconstruction in the wavelet domain
IAFNNesterov also allows to use transformations instead of filters, and we show that 
it present the same results as L1 wavelet reconstruction

'''
print(help())
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def GetNotNansPos(x):
    x=x.reshape((-1,1))
    get_not_nan = lambda xs: [i for (y, i) in zip(xs, range(len(xs))) if y==y]
    return get_not_nan(x)
def AdjointSampling(x,pos,origShape):
    y = np.empty(origShape)
    y[:]=np.nan
    y=y.reshape((-1,1))
    y[pos]=x    
    y=y.reshape(origShape)
    from scipy import interpolate

    array = np.ma.masked_invalid(y)    
    xx, yy = np.meshgrid(np.arange(0, origShape[0]),np.arange(0, origShape[1]))

    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    y = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='linear')

    return y.reshape((-1,1))

# Parameters
# ----------

import k_space
import radial
brain= np.array(Image.open('data/head256.png').convert('L').resize((256,256)))/255
idx=radial.radial2D(40,brain.shape)
#print(type(idx))
A=lambda x: k_space.k_space_sampling(x,brain.shape,idx)
At=lambda x: k_space.adjoint(x,brain.shape,idx)
b=A(brain)
x0=At(b).reshape(brain.shape).real

pattern=np.random.permutation(brain.size)
U=lambda x: wd.WavDec(x,brain.shape,decLevel=3,family='haar',randperm=pattern)
Ut=lambda x: wd.WavRec(x,brain.shape,decLevel=3,family='haar',randperm=pattern)
xw=np.reshape(U(brain),brain.shape)

import time
t=time.time()
xrtv=IAFNNESTA.IAFNNESTA(b,A=A,At=At,H='tv',sig_size=brain.shape,verbose=0,maxit=1000).real
print(time.time()-t)
t=time.time()
xrwav=IAFNNESTA.IAFNNESTA(b,A=A,At=At,U=U,Ut=Ut,sig_size=brain.shape,verbose=0,maxit=1000).real
print(time.time()-t)
t=time.time()
xrwav_as_h=IAFNNesterov.IAFNNesterov(b,A=A,At=At,H=U,Ht=Ut,verbose=0,maxit=1000)[0].real.reshape(brain.shape)
print(time.time()-t)

plt.title('MRI wavelet demo: tv - L1 wavelet - L1 wavelet as H')
plt.imshow(np.hstack((x0,xrtv,xrwav,xrwav_as_h)),cmap='gray')
plt.savefig('demo_wavelet.png')

plt.show()

