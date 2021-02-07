
import k_space
from IAFNNESTA import IAFNNESTA
import radial
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def help():
    return '''
Demo mri reconstruction

Solves the mri reconstruction problem for radial sampling:

argmin_x iaFN(x,h), s.t. ||Ax-b||<delta,

where x is the reconstructed image, A is the sampling operator, b is the 
measurements. 
iaFN is the isotropic-anisotropic filtering norm and h are the filters

First we test L1 and TV reconstruction using IAFNNESTA
Next, we test the reconstruction with the 8 filters from the paper 

The measurements are taken from 40 radial lines in the k-space
    '''

print(help)
brain= np.array(Image.open('data/head256.png').convert('L').resize((128,128)))/255
idx=radial.radial2D(40,brain.shape)
#print(type(idx))
A=lambda x: k_space.k_space_sampling(x,brain.shape,idx)
At=lambda x: k_space.adjoint(x,brain.shape,idx)
b=A(brain)

brainrl1=IAFNNESTA(b,A=A,At=At,sig_size=brain.shape)
brainrl1=brainrl1.real
brainrtv=IAFNNESTA(b,A=A,At=At,sig_size=brain.shape,H='tv')
brainrtv=brainrtv.real
plt.figure(1)
plt.imshow(np.hstack((brain,brainrl1,brainrtv)))
plt.title('MRI L1/TV demo')
plt.gray()
plt.savefig('demo_mri_l1_tv.png')
plt.show(block=False)  

from filters_from_paper import hs

recs=[]   
for i in range(len(hs)):        
    rec=IAFNNESTA(b,A=A,At=At,sig_size=brain.shape,H=hs[i]).real
    recs.append(rec)

row1=np.hstack((recs[0],recs[1],recs[2],recs[3]))
row2=np.hstack((recs[4],recs[5],recs[6],recs[7]))
rows=np.vstack((row1,row2))
plt.figure(4)
plt.gray()
plt.imshow(rows)
plt.title('MRI filtering norms demo')
plt.savefig('demo_mri_filtering_norms.png')
plt.show(block=False)


