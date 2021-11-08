import cPickle as pickle
import subprocess
import os.path
import time
#from scipy import special
import numpy as np
import numpy as numpy
import math
#from Notebooks.sigthresh import *

codystring  = 'codylong'
jimstring = 'jhhalverson'
#outdir = 'inout2'

def writeScript(nmod,personstring,beta,gam,arch,nd, stepsize, weight, max_steps='default'):
    nmod = np.float(nmod)
    out = "#!/bin/bash"
    out += "\n#SBATCH --job-name=k"+str(nmod)+"beta"+str(beta)+"msteps"+max_steps+"arch"+arch + "numdraws" + str(nd) + "stepsize" + str(stepsize) + "weight" + str(weight)
    out += "\n#SBATCH --output=k"+str(nmod)+"beta"+str(beta)+"msteps"+max_steps+"arch"+arch+ "numdraws" + str(nd) + "stepsize" + str(stepsize) + "weight" + str(weight)+ ".out"
    out += "\n#SBATCH --error=k"+str(nmod)+"beta"+str(beta)+"msteps"+max_steps+"arch"+arch+ "numdraws" + str(nd) + "stepsize" + str(stepsize) + "weight" + str(weight)+  ".err"
    out += "\n#SBATCH --exclusive"
    #old name for discovery node
    #out += "\n#SBATCH --partition=ser-par-10g-5"
    out += "\n#SBATCH --partition=fullnode"
    out += "\n#SBATCH -N 1"
    out += "\n#SBATCH --workdir=/home/" + personstring + "/Bergman/workdir"
    out += "\ncd /home/" + personstring + "/Bergman/"
    out += "\nmpiexec -n 1 python train_a3c_gym.py 32 --steps 20000000000 --env bergmanvmeans-v0 --outdir /scratch/" + personstring + "/BergmanCYvmeans/" + " --gamma " + str(gam)+ ' --mean ' + str(8.63) + ' --moments '+ '\"'+ str([np.float(116.04),np.float(3799.32)]).replace(' ','')  + '\"' + " --nmod " + str(int(nmod))  + " --beta " + str(beta)+ " --eval-interval 2000000 "  + " --arch " + arch + " --numdraws " + str(nd) + " --stepsize " + str(stepsize) + ' --weights '+ '\"'+ str([weight,weight**2]).replace(' ','')  + '\"'
            
    f = open("/home/" + personstring + "/Bergman/scripts/k"+str(nmod)+"beta"+str(beta)+"msteps"+max_steps+"arch"+arch+  "numdraws" + str(nd) + "stepsize" + str(stepsize) + "weight" + str(weight)+ ".job",'w')
    f.write(out)
    f.close()
    output=subprocess.Popen("sbatch /home/" + personstring +  "/Bergman/scripts/k"+str(nmod)+"beta"+str(beta)+"msteps"+max_steps+"arch"+arch+ "numdraws" + str(nd) + "stepsize" + str(stepsize) + "weight" + str(weight)+  ".job",shell=True,stdout=subprocess.PIPE).communicate()[0]
    return output


#November 14 sigma = 1 runs to reproduce wishart:

#for nmod in [3,5,10]:
#    for beta in [.01,.1,1,10]:
#        for gam in [.8,.9,.99,1]:
#            for nd in [100,500,1000]:
#                for stepsize in [1e-1,1e-2,1e-3]:
#                    for arch in ["FFSoftmax"]:
#                        writeScript(nmod,codystring,beta,gam,arch,nd,stepsize)

#November 15 sigma = 1 runs, weighted by moments, to reproduce wishart:

for nmod in [15]:
    for beta in [.1,1]:
        for gam in [.9,.99,1]:
            for nd in [500,1000]:
                for stepsize in [1e-2]:
                    for arch in ["FFSoftmax"]:
                        for weight in [.5,.1]:
                                writeScript(nmod,codystring,beta,gam,arch,nd,stepsize,weight)


