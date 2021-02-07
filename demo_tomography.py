import matplotlib.pyplot as plt
import numpy as np

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
from IAFNNESTA_UP import IAFNNESTA_UP
from PIL import Image
def help():
    return '''
Demo tomographic reconstruction

Solves the unrestricted reconstruction problem from tomographic radon sampling:

argmin_x iaFN(x,h) + 1/2||Ax-b||,

where x is the reconstructed image, A is the sampling operator(radon), 
b are the measurements. 
iaFN is the isotropic-anisotropic filtering norm and h are the filters

The unrestricted problem is prefered because AAt (undersampled radon and 
its inverse are not projectors).

First we test L1 and TV reconstruction using IAFNNESTA
Next, we test the reconstruction with the 8 filters from the paper 

The measurements are taken 30 angles of the radon transform
    '''
def sinogram_vectorized(x,theta,imshape):       
    image=x.reshape(imshape)    
    sinogram = radon(image, theta=theta, circle=True)
    vec_sino=sinogram.reshape((-1,1))
    return vec_sino

def rec_sinogram(vec_sino,theta):
    sino=vec_sino.reshape((-1,len(theta)))
    rec = iradon(sino, theta=theta, circle=True)
    rec_vec=rec.reshape((-1,1))
    return rec_vec

print(help())
#image = shepp_logan_phantom()
#image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
image=np.array(Image.open('data/tomo.jpg').convert('L').resize((128,128)))/255
num_angles=30
theta = np.linspace(0., 180., num_angles, endpoint=False)

A=lambda x: sinogram_vectorized(x,theta,image.shape)
At=lambda x: rec_sinogram(x,theta)
b=A(image)
print('radon')
tomo_iradon=At(b).reshape(image.shape)
print('l1 unconstrained')
tomol1=IAFNNESTA_UP(b,A=A,At=At,Lambda=0.01,La=1,sig_size=image.shape).real
print('tv unconstrained')
tomotv=IAFNNESTA_UP(b,A=A,At=At,Lambda=0.01,La=1,sig_size=image.shape,H='tv').real
plt.figure(1)
plt.imshow(np.hstack((image,tomo_iradon,tomol1,tomotv)),vmin=0, vmax=1)
plt.title('Tomographic iradon, L1 and TV demo')
plt.gray()
plt.savefig('demo_tomo_l1_tv.png')
plt.show()  


from filters_from_paper import hs

recs=[]   
for i in range(len(hs)):   
    print('filtering norm, with filter',i)     
    rec=IAFNNESTA_UP(b,A=A,At=At,Lambda=0.01,La=1,sig_size=image.shape,H=hs[i]).real
    recs.append(rec)

row1=np.hstack((recs[0],recs[1],recs[2],recs[3]))
row2=np.hstack((recs[4],recs[5],recs[6],recs[7]))
rows=np.vstack((row1,row2))
plt.figure(4)
plt.gray()
plt.imshow(rows,vmin=0, vmax=1)
plt.title('Tomographic unconstrained filtering norms demo')
plt.savefig('demo_tomo_filtering_norms.png')
plt.show()

