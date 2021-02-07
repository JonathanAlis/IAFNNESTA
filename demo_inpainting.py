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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

from IAFNNesterov import IAFNNesterov
import numpy as np
import sys
import random as rd
import matplotlib.pyplot as plt
from PIL import Image
import waveletDec as wd
# Parameters
# ----------
n_angles = 80 
Lambda = 0.64 
mu = 1e-10 
nit = 300  

I=np.zeros((64,64))
for _ in range(int(64*64/100)): #1%
    I[rd.randint(0,63),rd.randint(0,63)]=1.0
plt.imshow(I)
#plt.show()

Is=I
for i in range(64):
    Is[np.random.permutation(64)[:58],i]=np.nan
plt.imshow(Is)
x=Is.reshape(-1,1)
b=x[~np.isnan(x)]
#idxs=np.where(~np.isnan(x))
#b=x[idxs[0]]
print(b)
#plt.show()
#samplingPos=

lena= rgb2gray(np.array(Image.open('data/lena.jpg').resize((128,128)))/255)
#lena=lena+np.random.normal(0,0.1,lena.shape)

U=lambda x: wd.WavDec(x,lena.shape,decLevel=3,family='haar')
Ut=lambda x: wd.WavRec(x,lena.shape,decLevel=3,family='haar')
xw=np.reshape(U(lena),lena.shape)

b=lena.reshape(-1,1)
xr=IAFNNesterov(b,delta=0.1,U=U,Ut=Ut)[0].real
print(xr)
xr=xr.reshape(lena.shape)
plt.imshow(xr)
plt.show()
print(lena.shape)

#plt.imshow(xw)
#plt.show()

lena[40:60,100:120]=np.nan
pos=GetNotNansPos(lena)
#print(GetNotNansPos(lena).shape)
A=lambda x: x[pos]
At=lambda y: AdjointSampling(y,pos,lena.shape)

b=A(lena.reshape((-1,1)))
plt.imshow(At(b).reshape(lena.shape))
print("b")

f, axarr = plt.subplots(1,3)
axarr[0].imshow(lena,cmap='gray')
axarr[1].imshow(xw,cmap='gray')
axarr[2].imshow(At(b).reshape(lena.shape),cmap='gray')
plt.show()

xw
xr=IAFNNesterov(b,delta=0.1,A=A,At=At,U=U,Ut=Ut,verbose=10,maxit=10000)[0].real
plt.imshow(xr.reshape(lena.shape))
plt.show()

#    
