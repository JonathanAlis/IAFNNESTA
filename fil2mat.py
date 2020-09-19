import numpy as np
from scipy import signal, sparse
import ipdb
from itertools import product
import time
import _pickle as cPickle
import gzip
import os
from pathlib import Path
def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
    return object

def fil2mat(hs,ss):
    #hs: list with nd filters
    #ss: tuple containing the signal size
    
    #obtaining the filename for specific filter and shape
    str_list=['f:' +np.array_str(h) for h in hs]
    folder=Path().absolute() / Path('sparseMatrixFilters')
    file='shape:'+str(ss)+'_filters:'+str(str_list)+'.zip'
    filename = folder / file
    #filename=os.path.join('sparseMatrixFilters', 'shape:'+str(ss)+'_filters:'+str(str_list)+'.zip')
    #filename='./sparseMatrixFilters/shape:'+str(ss)+'_filters:'+str(str_list)+'.zip'
    #filename='./sparseMatrixFilters/shape:'+str(ss)+'_filters:'+str(str_list)+'.zip'
    #loading the file if exists
    if os.path.isfile(filename):
        M,Mt,MtM,H=load(filename)
        return M,Mt,MtM,H

    #if not exists, create it
    #flipping the filters:
    hfs=[]
    for i in range(len(hs)):
        try:
            hfs.append(np.asmatrix(np.flip(hs[i])))
        except:
            hfs.append(np.flip(hs[i]))
    N=np.prod(ss)
    H=[]
    
    for i in range(len(hfs)):        
        hf=np.array(hfs[i])
        fs=hfs[i].shape
        coords=[]
        HM=sparse.lil_matrix((N,N))
        row=0
        if len(ss)==1: #1D
            for j in range(ss[0]-fs[0]+1):
                m=np.zeros(ss)
                m[j:j+fs[0]]=hfs[i]
                HM[row,:]=m.reshape((1,N))
                row=row+1
        if len(ss)==2: #2D
            for j in range(ss[0]-fs[0]+1):
                for k in range(ss[1]-fs[1]+1):
                    m=np.zeros(ss)
                    m[j:j+fs[0],k:k+fs[1]]=hfs[i]
                    HM[row,:]=m.reshape((1,N))
                    row=row+1
        if len(ss)==3: #3D
            for j in range(ss[0]-fs[0]+1):
                for k in range(ss[1]-fs[1]+1):
                    for l in range(ss[2]-fs[2]+1):
                        m=np.zeros(ss)
                        m[j:j+fs[0],k:k+fs[1],l:l+fs[2]]=hfs[i]
                        HM[row,:]=m.reshape((1,N))
                        row=row+1
        if len(ss)>3: #nD, slower
            for i in range(len(fs)):
                delta = ss[i]-fs[i]+1
                coords.append(list(range(delta)))
            for indices in product(*coords):
                idx=tuple(slice(indices[i], indices[i]+fs[i]) for i in range(len(fs)))
                m=sparse.lil_matrix(ss)
                m.__setitem__(tuple(idx), hf)  
                HM[row,:]=m.reshape((1,N))#sparse.csr_matrix(m).reshape((1,N))
                row=row+1
        HM=HM.tocsr()
        H.append(HM)


    M=sparse.vstack(H)
    Mt=M.transpose()
    MtM=Mt@M
    #save to file
    try:
        save([M,Mt,MtM,H],filename)
    except:
        print('Could not save the file...')
    #with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([M,Mt,MtM,H], f)
    return M,Mt,MtM,H

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) 

if __name__ == "__main__":
    #test 2D
    x=np.random.random((5,5))
    hs=[]
    hs.append(np.array([[1,-2,1],[1,-2,1],[1,-2,1]]))
    #hs.append(np.array([[1,-4,1],[1,-3,1],[1,-2,1]]))

    print("2D test:")
    print("For random x, of size "+str(x.shape))
    print("And h equal to")
    print(hs[0].shape)
    print("we get:")
    M,_,_,H=fil2mat(hs,x.shape)
    y1=signal.fftconvolve(x,hs[0],'valid')
    y1=y1.reshape((-1,1))
    y2=M@(x.reshape((-1,1)))
    y2=y2[:len(y1)]    
    print(np.c_[y1,y2])
    #3D
    print("3D test:")
    x=np.random.random((5,5,5))
    hs=[]
    hs.append(np.array([2, 3, 5,7,11,13,17,19,23,29,31,37]))
    hs[0]=hs[0].reshape(2,2,3)
    print("For random x, of size "+str(x.shape))
    print("And h equal to")
    print(hs[0].shape)
    print("we get:")
    M,_,_,H=fil2mat(hs,x.shape)
    y1=signal.convolve(x,hs[0],'valid')
    y1=y1.reshape((-1,1))
    y2=M@(x.reshape((-1,1)))
    y2=y2[:len(y1)]
    print(np.c_[y1,y2])
    print("First column: regular convolution; second: convolution with matrix. If both numbers are equal, then it works!!!")