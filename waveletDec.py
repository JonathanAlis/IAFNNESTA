import pywt
from anytree import AnyNode, RenderTree, LevelOrderIter, LevelOrderGroupIter, ZigZagGroupIter
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from PIL import Image
import copy

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def WavDec(x,shape,decLevel=5,family='haar',randperm=None):
    x=x.reshape(shape)
    root=AnyNode(id="root",value=x,shape=x.shape)
    lastParent=root

    for i in range(decLevel):
        coeffs = pywt.dwt2(lastParent.value, family)
        cA, (cH, cV, cD) = coeffs
        childA=AnyNode(id="child"+str(i+1)+"A",value=cA,shape=cA.shape,level=i,parent=lastParent)
        childH=AnyNode(id="child"+str(i+1)+"H",value=cH,shape=cH.shape,level=i,parent=lastParent)
        childV=AnyNode(id="child"+str(i+1)+"V",value=cV,shape=cV.shape,level=i,parent=lastParent)
        childD=AnyNode(id="child"+str(i+1)+"D",value=cD,shape=cD.shape,level=i,parent=lastParent)
        lastParent=childA


    dec=np.zeros(x.shape)
    layers=[children for children in LevelOrderGroupIter(root)]
    for layer in layers[::-1]:
        try:
            layerImg=np.vstack((np.hstack((layer[0].value,layer[1].value)),np.hstack((layer[2].value,layer[3].value))))
            layer[0].parent.value=layerImg
        except:
            break
        
    #plt.imshow(np.log(1+np.abs(root.value)))
    y=root.value.reshape((-1,1))
    if randperm is not None:
        y=y[randperm]
    return y    
    

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def WavRec(y,shape,decLevel=5,family='haar',randperm=None):
    if randperm is not None:
        y=y[inv(randperm)]
    y=y.reshape(shape)
    root=AnyNode(id="root",shape=shape)
    lastParent=root
    lls=shape#lastLevelShape
    for i in range(decLevel):        
        lls=tuple([int(x/2) for x in lls])
        childA=AnyNode(id="child"+str(i+1)+"A",value=np.zeros(lls),shape=lls,level=i,parent=lastParent)
        childH=AnyNode(id="child"+str(i+1)+"H",value=np.zeros(lls),shape=lls,level=i,parent=lastParent)
        childV=AnyNode(id="child"+str(i+1)+"V",value=np.zeros(lls),shape=lls,level=i,parent=lastParent)
        childD=AnyNode(id="child"+str(i+1)+"D",value=np.zeros(lls),shape=lls,level=i,parent=lastParent)
        lastParent=childA
        
    # dec=np.zeros(y.shape)
    layers=[children for children in LevelOrderGroupIter(root)]
    for layer in layers[::-1]:
        try:
            lls=tuple([int(x*2) for x in lls])
            cA=y[0:int(lls[0]/2),0:int(lls[1]/2)]
            cV=y[int(lls[0]/2):lls[0],0:int(lls[1]/2)]
            cH=y[0:int(lls[0]/2),int(lls[1]/2):lls[1]]
            cD=y[int(lls[0]/2):lls[0],int(lls[1]/2):lls[1]]
            y[0:lls[0],0:lls[1]]=pywt.idwt2((cA,(cH,cV,cD)),'haar')
        except:
            break
        
    return y.reshape(-1,1)
        #layer[0] layer
        #for node in layer:
        #    pass
            #print(node.id) 
    #plt.imshow(np.log(1+np.abs(root2.value)))
    #y=root2.value.reshape((-1,1))

if __name__ == "__main__":


    lena= rgb2gray(np.array(Image.open('data/lena.jpg'))/255)
    #lena=lena+np.random.normal(0,0.1,lena.shape)
    print(type(lena))
    w=WavDec(lena.reshape(-1,1),lena.shape)
    print(w.shape)
    plt.plot(w)
    plt.show()
    wt=WavRec(w,lena.shape)
    print(wt.shape)
    plt.plot(wt)
    plt.show()
