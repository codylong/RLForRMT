import numpy as np


def random_wishart(nmod): # pos def metric
    vars = [1./np.float(nmod)**(1./2.) for i in range(nmod + 2)]
    #A = np.array([np.random.normal(loc = 0, scale = vars[pp],size = nmod )for pp in range(nmod)])
    A00 = np.random.normal(loc = 0, scale = vars[nmod + 1])
    A0a = np.array(np.random.normal(loc = 0, scale = vars[nmod],size = nmod))
    Ap0a = np.array(np.random.normal(loc = 0, scale = vars[nmod],size = nmod))
    Aab = np.array([np.random.normal(loc = 0, scale = vars[pp],size = nmod)for pp in range(nmod)])

    P00  = A00**2 + sum([ent**2 for ent in A0a])
    P0a = np.array([A00*ent for ent in Ap0a]) + np.dot(A0a,Aab)
    Pab = np.dot(Aab.transpose(),Aab) + [[Ap0a[i]*Ap0a[j] for i in range(len(Ap0a))] for j in range(len(Ap0a))]
    m = Pab/P00 - [[P0a[i]*P0a[j] for i in range(len(P0a))] for j in range(len(P0a))]/P00**2
    return m

def sample(nmod,numdraws):
    #samplemoments = [[] for kk in range(self.nmoments)] 
    evals = []
    for num in range(numdraws):
        evals  = evals + list(np.linalg.eig(random_wishart(nmod))[0])
    return evals

sample(10,1)