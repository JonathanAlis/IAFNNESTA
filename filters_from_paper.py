import winFilters 
import numpy as np
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