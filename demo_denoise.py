from PIL import Image
import numpy as np
from IAFNNESTA import IAFNNESTA
from IAFNNESTA_UP import IAFNNESTA_UP
import matplotlib.pyplot as plt
import time


def help():
    return '''
Demo denoise
Solves the denoising problem using IAFNNESTA:

argmin_x iaFN(x,h), s.t. ||xn-x||<delta,

and the unconstrained form using IAFNNESTA_UP:

argmin_x lambda iaFN(x,h) + 1/2 ||xn - x||_2^2 

where x is the denoised signal to be obtained, xn is the signal with noise, 
iaFN is the isotropic-anisotropic filtering norm and h are the filters

we tested the isotropic form with TV filter and the isotropic-anisotropic with 2nd order TV filters
    '''

print(help())
lena= np.array(Image.open('data/lena.jpg').convert('L').resize((256,256)))/255
lenan=lena+np.random.normal(0,0.1,lena.shape)

t=time.time()
h=[]
h.append(np.array([[1,-1]]))
h.append(np.array([[1],[-1]]))
xrc1=IAFNNESTA(lenan,lenan.shape,delta=26,L1w=0,L2w=1,verbose=0,maxit=1000,H=h)
print('time constrained problem tv: ',time.time()-t)
t=time.time()
h.append(np.array([[1, -2, 1]]))
h.append(np.array([[1],[-2],[1]]))
xrc2=IAFNNESTA(lenan,lenan.shape,delta=26,L1w=1,L2w=1,verbose=0,maxit=1000,H=h)
print('time iso-aniso constrained problem 2d order: ',time.time()-t)

#unconstrained problem
Lambda=0.1
La=1
t=time.time()
h=[]
h.append(np.array([[1,-1]]))
h.append(np.array([[1],[-1]]))
xru1=IAFNNESTA_UP(lenan,lenan.shape,Lambda=Lambda,La=La,L1w=0,L2w=1,verbose=0,maxit=1000,H=h)
print('time unconstrained problem tv: ',time.time()-t)
t=time.time()
h.append(np.array([[1, -2, 1]]))
h.append(np.array([[1],[-2],[1]]))
xru2=IAFNNESTA_UP(lenan,lenan.shape,Lambda=Lambda,La=La,L1w=1,L2w=1,verbose=0,maxit=1000,H=h)
print('time unconstrained problem 2d order: ',time.time()-t)

fig, axs = plt.subplots(2, 3)
axs[0, 0].imshow(lena,cmap='gray')
axs[0, 0].set_title('Original')
axs[1, 0].imshow(lenan,cmap='gray')
axs[1, 0].set_title('With noise')
axs[0, 1].imshow(xrc1,cmap='gray')
axs[0, 1].set_title('isotropic tv rec')
axs[1, 1].imshow(xrc2,cmap='gray')
axs[1, 1].set_title('iaFN order2 rec')
axs[0, 2].imshow(xru1,cmap='gray')
axs[0, 2].set_title('isotropic UP tv')
axs[1, 2].imshow(xru2,cmap='gray')
axs[1, 2].set_title('iaFN UP order2 tv')
fig.suptitle('Demo denoise', fontsize=16)

plt.savefig('denoise_demo.png')
plt.show()