import numpy as np 
import math
import matplotlib.pyplot as plt
def radial2D(N,s,golden=True,extend=False, show=False):
    z=np.zeros(s)
    num_points=2*np.max(s)
    golden=math.pi*(3-math.sqrt(5))
    for i in range(N):
        angle=golden*i
        pos=np.linspace(-1,1,num_points)
        for p in pos:
            x=math.sin(angle)*p
            y=math.cos(angle)*p
            midx=math.ceil(s[0]/2)
            midy=math.ceil(s[1]/2)
            xi=math.floor(midx+midx*x)
            yi=math.floor(midy+midy*y)
            if xi<s[0] and yi<s[1]:
                z[xi,yi]=1
    
    if show:
        plt.imshow(z)
        plt.show()

    z=np.fft.ifftshift(z)
    z=z.reshape((-1,1))
    idx= [i for i, x in enumerate(z) if x == 1]
    
    
    return idx

if __name__ == "__main__":
    idx=radial2D(10,(128,128),show=True)
    print(idx)
    y=np.zeros((128,128))
    y=y.reshape((-1,1))
    y[idx]=1
    y=y.reshape((128,128))
    y=np.fft.fftshift(y)
    plt.imshow(y)
    plt.show()
