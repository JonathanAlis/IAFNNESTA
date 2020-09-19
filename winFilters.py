import numpy as np
from scipy import signal

def winFilters(nCoefs,nBands):
    h1D=[]
    #hs.append(np.array([[1,-1]]))
    h1D.append(signal.firwin(nCoefs,1.0/nBands))
    for i in range(1,nBands-1):
        h1D.append(signal.firwin(nCoefs,[(i)/nBands, (i+1)/nBands]))
    h1D.append(signal.firwin(nCoefs,1.0/nBands,pass_zero='highpass'))
    #print(h1D)
    h2D=[]
    for i in range(nBands):
        for j in range(nBands):
            if i!=0 or j!=0:
                h2D.append(np.matrix(h1D[i]).T@np.matrix(h1D[j]))
    return h2D

if __name__ == "__main__":
    #test 2D
    import matplotlib.pyplot as plt
    h=winFilters(11,2)
    sq=np.ceil(np.sqrt(len(h)))
    for i in range(len(h)):
        plt.subplot(sq,sq,i+1)
        spec=np.abs(np.fft.fft2(h[i],(128,128)))
        plt.imshow(spec)
    plt.show()
    from PIL import Image
    lena= np.array(Image.open('data/lena.jpg').convert('L').resize((256,256)))/255
    for i in range(len(h)):
        f=signal.convolve2d(lena, h[i], mode='valid')
        plt.subplot(sq,sq,i+1)
        plt.imshow(f)
    plt.show()