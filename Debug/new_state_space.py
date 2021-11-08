import numpy as np


def random_wishart(nmod): # pos def metric
    #make a list of vars of the appropriate length
    littlevars = [1./np.float(nmod)**(1./2.) for i in range(nmod + 1)]
    vars = []
    for ii in range(nmod + 2):
        vars = vars + littlevars
    A = []
    for rowind in range(nmod + 1):
        A.append([np.random.normal(loc = 0, scale = vars[pp]) for pp in range(rowind*(nmod + 1), (rowind+1)*(nmod + 1))])
    A = np.array(A)
    P = np.dot(A.transpose(),A)
    P00 = P[0][0]
    P0a = [P[0][a] for a in range(1,nmod+1)]
    Pab = [[P[a][b] for a in range(1,nmod+1)] for b in range(1,nmod + 1)]
    m = Pab/P00 - [[P0a[i]*P0a[j] for i in range(len(P0a))] for j in range(len(P0a))]/P00**2
    return m

def sample(nmod,numdraws):
    #samplemoments = [[] for kk in range(self.nmoments)] 
    evals = []
    for num in range(numdraws):
        evals  = evals + list(np.linalg.eig(random_wishart(nmod))[0])
    return evals
allevals = []
for ii in range(10000):
    allevals += sample(10,1)
print min(allevals)