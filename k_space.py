from scipy.fft import fftn, ifftn
import numpy as np
def k_space_sampling(x,ss,idx):
    '''
    x:signal, of total lenght equal prod(ss).
    ss: signal shape
    idx: indexes to take the k-space measurements
    '''

    x=x.reshape(ss)
    X=fftn(x,s=ss)
    X=X.reshape((-1,1))
    return X[idx]

def adjoint(y,ss,idx):
    Y=np.zeros(ss,dtype=complex)
    Y=Y.reshape((-1,1))
    Y[idx]=y
    Y=Y.reshape(ss)
    Y=ifftn(Y,ss)
    return Y.reshape((-1,1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from PIL import Image
    from skimage.transform import resize
    image=np.array(Image.open('data/lena.jpg').convert('L'))
    image=resize(image,(128,128))
    image=image.astype(np.float)/np.max(image[...])
    #random sampling:
    numpoints=int(128*128/2)
    idx=np.random.permutation(np.arange(0, 128*128).tolist())
    idx=idx[0:numpoints-1]
    p=np.zeros((128,128))
    p=p.reshape((-1,1))
    p[idx]=1
    p=p.reshape((128,128))
    y=k_space_sampling(image,(128,128),idx)
    #X_toview=np.log(1+abs(X))
    
    rec=adjoint(y,(128,128),idx)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(p)
    plt.subplot(1,3,3)
    plt.imshow(abs(rec))
    plt.show()