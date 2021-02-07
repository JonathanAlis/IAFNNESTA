def help():
    return '''
Isotropic-Anisotropic Filtering Norm Nesterov

Solves the filtering norm minimization + quadratic term problem
Nesterov algorithm, without continuation:

     argmin_x ||  iaFN(x) ||_1/2 subjected to ||b - Ax||_2^2 < delta

If no filter is provided, solves the L1.
If continuation is desired, see the function IAFNNESTA.m

The primal prox-function is also adapted by accounting for a first guess
x0 that also tends towards x_muf 

The observation matrix A must be a projector (non projector not implemented yet)

Inputs:
IAFNNesterov(b,                      #Observed data, a m x 1 array
            A=identity,At=identity,     # measurement matrix and adjoint (either a matrix, function handles)                
            mu=0.0001,                  #mu, smaller leads to higher accuracy
            delta,                      #bounds of the solution deviations from measurements
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
Outputs:
return  xk,                             #estimated x reconstructed signal
        niter,                          #number of iterations
        residuals                       #first column is the residual at every step,
                                        #second column is the value of f_mu at every step


'''

import numpy as np
from scipy import sparse
import sys
def identity(x):
    return x


def IAFNNesterov(b,A=identity,At=identity,mu=0.0001,delta=0,L1w=1,L2w=0,verbose=0,maxit=1000,x0=[],U=identity,Ut=identity,stopTest=1,TolVar = 1e-5,AAtinv=[],normU=1,H=identity,Ht=identity):

    if delta<0:
        raise Exception('Delta must not be negative')

    if not callable(A): #If not function
        A=lambda x:np.matmul(A,x)
        At=lambda x:np.matmul(np.transpose(A),x)
    
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
    '''
    if not sparse.issparse(H):
        H=sparse.identity(len(x0))
        Ht=sparse.identity(len(x0))
    else:
        Ht=H.transpose()
    '''
    HU=lambda x: H(U(x))
    UtHt=lambda x: Ut(Ht(x))
    # Initialization
    N = len(x0)
    wk = np.zeros((N,1)) 
    xk = x0
    xold=x0

    #---- Init Variables
    Ak= 0
    Lmu = normU/mu
    yk = xk
    zk = xk
    fmean =sys.float_info.min/10 #smallest positive real number that can be represented
    OK=False
    n = np.floor(np.sqrt(N))

    #%---- Computing Atb
    Atb = At(b)
    Axk = A(xk)#;% only needed if you want to see the residuals
    Ax0 = Axk


#TV, filter, etc.
    
    Lmu1 = 1/Lmu
    SLmu = np.sqrt(Lmu);
    SLmu1 = 1/np.sqrt(Lmu);
    LambdaY = 0
    LambdaZ = 0

    residuals=np.zeros((maxit,2))
    for k in range(maxit):
        if L1w!=0:
            df,fx = Perform_L1_Constraint(xk,mu,HU,UtHt)    
        if L2w!=0:
            df,fx = Perform_L2_filter_constraints(xk,mu,H,Ht,U,Ut)
        
        #ipdb.set_trace()

        cp = xk - 1/Lmu*df  #;  % this is "q" in eq. (3.7) in the paper
        Acp = A( cp )
        AtAcp = At( Acp )
        residuals[k][0] = np.linalg.norm( b-Axk)
        residuals[k][1] = fx             
            
        if delta > 0:
            aux=Lmu*(np.linalg.norm(b-Acp)/delta - 1)
            Lambda = np.max([0,aux])
            gamma = Lambda/(Lambda + Lmu)
            yk = Lambda/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp
            #% for calculating the residual, we'll avoid calling A()
            #% by storing A(yk) here (using A'*A = I):
            Ayk = Lambda/Lmu*(1-gamma)*b + Acp - gamma*Acp       
        else:
            #% if delta is 0, the projection is simplified:
            yk = AtAAtb + cp - AtAcp
            Ayk = b
        

        #stopping criteria            
        qp = np.abs(fx - np.mean(fmean))/np.mean(fmean)

        if stopTest==1:
                #look at the relative change in function value
            if qp <= TolVar and OK:
                break
            if qp <= TolVar and ~OK:
                OK=True
        if stopTest==2:
                #look at the l_inf change from previous iterate
                if k >= 0 and np.linalg.norm( xk - xold, np.inf ) <= TolVar:
                    break

        fmean = np.insert(fmean,0,fx)
        if (len(fmean) > 10): 
            fmean = fmean[0:9]
        #%--- Updating zk
      
        apk =0.5*(k+1)
        Ak = Ak + apk 
        tauk = 2/(k+3) 
        
        wk =  apk*df + wk
        
        #%
        #% zk = Argmin_x Lmu/2 ||b - Ax||_l2^2 + Lmu/2||x - xplug ||_2^2+ <wk,x-xk> 
        #%   s.t. ||b-Ax||_l2 < delta
        #%
        
        cp = x0 - 1/Lmu*wk
        
        Acp = A( cp )
        AtAcp = At( Acp )
        
        if delta > 0:
            Lambda = max(0,Lmu*(np.linalg.norm(b-Acp)/delta - 1))
            gamma = Lambda/(Lambda + Lmu)
            zk = Lambda/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp
            #% for calculating the residual, we'll avoid calling A()
            #% by storing A(zk) here (using A'*A = I):
            Azk = Lambda/Lmu*(1-gamma)*b + Acp - gamma*Acp
        else:
            #% if delta is 0, this is simplified:
            zk = AtAAtb + cp - AtAcp
            Azk = b
        

        #xk
                
        xkp = tauk*zk + (1-tauk)*yk
        xold = xk
        xk=xkp
        Axk = tauk*Azk + (1-tauk)*Ayk
        
        if k%10==0: 
            Axk = A(xk)   #% otherwise slowly lose precision

        if verbose>0:
            if k%verbose==0:
                print("Iter: %3d, fmu: %.3f, Rel. Variation of fmu: %.2e ~ Residual: %.2e" %(k, fx,qp,residuals[k][0]))  
        if np.abs(fx)>1e20 or np.abs(residuals[k][0]) >1e20 or np.isnan(fx):
            print(fx)
            raise Exception('Filtering Norm Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?');
    
    niter = k+1
    #%-- truncate output vectors
    residuals = residuals[1:niter,:]
    return xk,niter,residuals

#%%%%%%%%%%%% PERFORM THE L1 CONSTRAINT %%%%%%%%%%%%%%%%%%

def Perform_L1_Constraint(xk,mu,U,Ut):
#[df,fx,val,uk]
    uk=fx=U(xk)
    uk=np.divide(uk,np.maximum(abs(uk),mu))    
    val = np.real(np.matmul(np.transpose(uk),fx))
    fx = np.real(np.matmul(np.transpose(uk),fx) - mu/2*np.square(uk).sum())
    df=Ut(uk)
    return df,fx

def Perform_L2_filter_constraints(xk,mu,H,Ht,U,Ut):
    x = U(xk)
    Hx = H(x);    
    sq=np.sqrt(np.square(Hx))  
    w = np.maximum(sq,mu)
    u = np.divide(Hx,w)
    fx = np.real(np.matmul(np.transpose(u),H(x)))
    fx-= mu/2 * 1/u.size * np.power(np.linalg.norm(u),2)
    df = Ut(Ht(u))
    return df,fx



#TV
def Perform_TV_Constraint(xk,mu,Dv,Dh,D,U,Ut): #Dv, Dh, sparse matrix
    x = U(xk)
    df = np.zeros(x.shape)

    Dhx = Dh@x
    Dvx = Dv@x
    sq=(np.sqrt(np.square(Dhx)+np.square(Dvx)))  
    tvx = sq.sum()
    w = np.maximum(sq,mu)
    uh = np.divide(Dhx,w)
    uv = np.divide(Dvx,w)
    u = np.concatenate((uh,uv))
    fx = np.real(np.matmul(np.transpose(u),D@x) - mu/2* 1/u.size * (np.transpose(u)*u).sum())


# def randomSampling2D():
# def sample2D(x,):
    #select the smaples of a signal

if __name__ == "__main__":
    print(help())
