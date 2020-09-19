import IAFNNesterov
import numpy as np
from scipy import sparse
import fil2mat
def identity(x):
    return x

def IAFNNESTA(b,sig_size=0,A=identity,At=identity,muf=0.0001,delta=0,L1w=1,L2w=0,verbose=0,MaxIntIter=5,maxit=1000,x0=[],U=identity,Ut=identity,stopTest=1,TolVar = 1e-5,AAtinv=[],normU=1,H=[]):

    if delta<0:
        raise Exception('Delta must not be negative')

    if not callable(A): #If not function
        A=lambda x:np.matmul(A,x)
        At=lambda x:np.matmul(np.transpose(A),x)
    
    b=b.reshape((-1,1))
    Atb=At(b)

    if callable(AAtinv):
        AtAAtb = At( AAtinv(b) )
    else:
        if len(AAtinv)>0:
            AAtinv=lambda x: np.matmul(AAtinv,x)
            AtAAtb = At( AAtinv(b) )
        else: #default
            AtAAtb = Atb
            AAtinv=identity
    
    if len(x0)==0:
        x0 = AtAAtb 

    if len(H)==0:
        Hf=identity
        Hft=identity
    else:            
        if not sparse.issparse(H):
            if isinstance(H, str):
                if H=='tv':
                    hs=[]
                    hs.append(np.array([[1,-1]]))
                    hs.append(np.array([[1],[-1]]))
                    H,_,_,_=fil2mat.fil2mat(hs,sig_size)
                else:
                    print('H not recognized. Must be a sparse matrix, a list of filters or the string tv')            
            else:        
                #list of filters:
                H,_,_,_=fil2mat.fil2mat(H,sig_size) 
        #print(H.shape)
        #print(H)
        #print(type(H))
        Ht=H.transpose()
        Hf=lambda x: H@x
        Hft=lambda x: Ht@x

    HU=lambda x: Hf(U(x))
    UtHt=lambda x: Ut(Hft(x))
    
    if sig_size==0:
        sig_size=Atb.shape

    
        
    typemin=''
    if L1w>0:
        typemin+="iso"
    if L2w>0:
        typemin+="aniso"
    typemin+='tropic '
    if callable(H):
        typemin+='filtering norm '    

    mu0=0
    if L1w>0:
        mu0+=L1w*0.9*np.max(np.linalg.norm(HU(x0),1))
    if L2w>0:
        mu0+=L2w*0.9*np.max(np.linalg.norm(HU(x0),2))

    niter = 0
    Gamma = np.power(muf/mu0,1/MaxIntIter)
    mu = mu0
    Gammat= np.power(TolVar/0.1,1/MaxIntIter)
    TolVar = 0.1
    
    for i in range(MaxIntIter):    
        mu = mu*Gamma
        TolVar=TolVar*Gammat;    
        if verbose>0:
            #if k%verbose==0:
            print("\tBeginning %s Minimization; mu = %g\n" %(typemin,mu))
            

        xk,niter_int,res = IAFNNesterov.IAFNNesterov(b,A=A,At=At,mu=mu,delta=delta,L1w=L1w,L2w=L2w,verbose=verbose,maxit=maxit,x0=x0,U=U,Ut=Ut,stopTest=stopTest,TolVar = TolVar,AAtinv=AAtinv,normU=normU,H=Hf,Ht=Hft)
        
        xplug = xk
        niter = niter_int + niter
        if i==0:
            residuals=res
        else:
            residuals = np.vstack((residuals, res))
    
    return xk.reshape(sig_size)


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    lena= np.array(Image.open('data/lena.jpg').convert('L').resize((256,256)))/255
    lenan=lena+np.random.normal(0,0.1,lena.shape)

    h=[]
    h.append(np.array([[1,-1]]))
    h.append(np.array([[1],[-1]]))
    xr1=IAFNNESTA(lenan,lenan.shape,delta=26,L1w=0,L2w=1,verbose=50,maxit=1000,H=h)
    h.append(np.array([[1, -2, 1]]))
    h.append(np.array([[1],[-2],[1]]))
    xr2=IAFNNESTA(lenan,lenan.shape,delta=26,L1w=1,L2w=1,verbose=50,maxit=1000,H=h)

    plt.figure(1)
    plt.imshow(np.vstack((np.hstack((lena,lenan)),np.hstack((xr1,xr2)))))
    plt.gray()
    plt.title('Denoise demo')
  

    import k_space
    import radial
    brain= np.array(Image.open('data/head256.png').convert('L').resize((128,128)))/255
    idx=radial.radial2D(40,brain.shape)
    #print(type(idx))
    A=lambda x: k_space.k_space_sampling(x,brain.shape,idx)
    At=lambda x: k_space.adjoint(x,brain.shape,idx)
    b=A(brain)
    brainrtv=IAFNNESTA(b,A=A,At=At,sig_size=brain.shape,H='tv')
    brainrtv=brainrtv.real
    plt.figure(2)
    plt.imshow(np.hstack((brain,brainrtv)))
    plt.title('MRI TV demo')

    plt.gray()
    plt.figure(3)
    brainrl1=IAFNNESTA(b,A=A,At=At,sig_size=brain.shape)
    brainrl1=brainrl1.real
    plt.imshow(np.hstack((brain,brainrl1)))
    plt.title('MRI L1 demo')
    #plt.show()  

    import winFilters 
    hs=[]
    h1=[]
    h1.append(np.matrix([1, -1]))  
    h1.append(np.matrix([[1], [-1]]))
    hs.append(h1)
    h2=[]
    h2.append(np.matrix([[1,-1], [1,-1]]))
    h2.append(np.matrix([[1,1], [-1,-1]]))
    h2.append(np.matrix([[1,-1], [-1,1]]))
    hs.append(h2)
    h3=[]
    h3.append(np.matrix([[1,-2,1]]))
    h3.append(np.matrix([[1], [-2],[1]]))
    hs.append(h3)
    h4=[]   
    h4=winFilters.winFilters(3,2) 
    hs.append(h4)
    h5=[]
    h5.append(np.matrix([[1,-1], [1,-1]]))
    h5.append(np.matrix([[1,1], [-1,-1]]))
    h5.append(np.matrix([[1,-1], [-1,1]]))
    h5.append(np.matrix([[1,-2,1]]))
    h5.append(np.matrix([[1], [-2],[1]]))
    hs.append(h5)
    h6=[]
    h6=winFilters.winFilters(3,2) 
    h6.append(np.matrix([[1,-1], [1,-1]]))
    h6.append(np.matrix([[1,1], [-1,-1]]))
    h6.append(np.matrix([[1,-1], [-1,1]]))
    hs.append(h6)
    h7=[]
    h7=winFilters.winFilters(3,2) 
    h7.append(np.matrix([[1,-2,1]]))
    h7.append(np.matrix([[1], [-2],[1]]))
    hs.append(h7)
    h8=[]
    h8=winFilters.winFilters(3,2) 
    h8.append(np.matrix([[1,-1], [1,-1]]))
    h8.append(np.matrix([[1,1], [-1,-1]]))
    h8.append(np.matrix([[1,-1], [-1,1]]))
    h8.append(np.matrix([[1,-2,1]]))
    h8.append(np.matrix([[1], [-2],[1]]))
    hs.append(h8)

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
    plt.show()


