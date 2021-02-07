def help():
    return '''
Isotropic-Anisotropic Filtering Norm Nesterov Algorithm

Solves the filtering norm minimization + quadratic term problem
Nesterov algorithm, with continuation:

    argmin_x lambda iaFN(x,h) + 1/2 ||b - Ax||_2^2 

If no filter is provided, solves the L1.

Continuation is performed by sequentially applying Nesterov's algorithm
with a decreasing sequence of values of  mu0 >= mu >= muf

The observation matrix A must be a projector (non projector not implemented yet)

Inputs:
IAFNNESTA(b,                      #Observed data, a m x 1 array
            A=identity,At=identity,     # measurement matrix and adjoint (either a matrix, function handles)                
            muf=0.0001,                 #final mu value, smaller leads to higher accuracy
            delta,                      #l2 error bound. This enforces how close the variable
                                        #must fit the observations b, i.e. || y - Ax ||_2 <= delta
                                        #If delta = 0, enforces y = Ax
                                        #delta = sqrt(m + 2*sqrt(2*m))*sigma, where sigma=std(noise).
            L1w=1,L2w=0,                #weights of L1 (anisotropic) and L2(isotropic) norms
            verbose=0,                  #whether to print internal steps 
            maxit=1000,                 #maximum iterations at the inner loop
            x0=[],                      #initial solution, if not provided, will be At(b)
            U=identity,Ut=identity,     #Analysis/Synthesis operators
            stopTest=1,                 #stopTest == 1 : stop when the relative change in the objective 
                                        function is less than TolVar
                                        stopTest == 2 : stop with the l_infinity norm of difference in 
                                        the xk variable is less than TolVar 
            TolVar = 1e-5,              #tolerance for the stopping criteria
            AAtinv=[],                  #not implemented
            normU=1,                    #if U is provided, this should be norm(U)
            H=[],Ht=[]):                #filter operations in sparse matrix form
                                        #also accepts the string 'tv' as input, 
                                        #in that case, calculates the tv norm
Outputs:
return  xk,                             #estimated x reconstructed signal
        niter,                          #number of iterations
        residuals                       #first column is the residual at every step,
                                        #second column is the value of f_mu at every step


'''

import numpy as np
from scipy import sparse
import fil2mat
import IAFNNesterov_UP
def identity(x):
    return x

def IAFNNESTA_UP(b,sig_size=0,A=identity,At=identity,muf=0.0001,Lambda=None,La=None,L1w=1,L2w=0,verbose=0,MaxIntIter=5,maxit=1000,x0=[],U=identity,Ut=identity,stopTest=1,TolVar = 1e-5,AAtinv=[],normU=1,H=[]):
    if Lambda is None or La is None:
        print('IAFNNesterov_UP error, must provide Lambda and La')
        exit()


    if not callable(A): #If not function
        A=lambda x:np.matmul(A,x)
        At=lambda x:np.matmul(np.transpose(A),x)
    
    b=b.reshape((-1,1))
    Atb=At(b)
    if sig_size==0:
        sig_size=Atb.shape
        
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
    # Initialization
    N = len(x0)
    wk = np.zeros((N,1)) 
    xk = x0
    
            
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

    muL = Lambda/La
    mu0 = max(mu0,muL)
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
        xk,niter_int,res = IAFNNesterov_UP.IAFNNesterov_UP(b,A=A,At=At,mu=mu,Lambda=Lambda,La=La,L1w=L1w,L2w=L2w,verbose=verbose,maxit=maxit,x0=x0,U=U,Ut=Ut,stopTest=stopTest,TolVar = TolVar,AAtinv=AAtinv,normU=normU,H=Hf,Ht=Hft)
        
        xplug = xk
        niter = niter_int + niter
        if i==0:
            residuals=res
        else:
            residuals = np.vstack((residuals, res))
    
    return xk.reshape(sig_size)


if __name__ == "__main__":
    print(help())
