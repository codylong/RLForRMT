import pandas as pd
from lib import *
import os
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import cPickle as pickle
import numpy as np
import scipy
from mpmath import *
import os
import datetime
import itertools


class Wishart_Experiment:

    def __init__(self,dir):
        self.dir = dir
        self.timestamp = self.dir.rsplit('/', 1)[-1]
        if dir[len(dir)-1] != '/': self.dir += '/'
        self.dist = {'worker': [], 'time': [], 'dist':  [], 'state': []}
        self.s = {'worker': [], 'time': [], 'dist':  [], 'state': []}
        self.scores, self.steps, self.gamma, self.beta, self.env, self.nmod, self.stepsize, self.nd, self.weights, self.moments, self.numdraws = {}, None, None, None, None, None, None, None, None, None, None
        self.track_scores = {'worker': [], 'time': [], 'score': []}
        self.lstm, self.mcts, self.outdir, self.discovery_dir = None, None, None, None
        self.has_scores = True
        self.timestamp_datetime = datetime.datetime.strptime(self.timestamp,'%Y%m%dT%H%M%S.%f')

        self.read_command_txt()
        self.read_output_txt()
        print dir
        self.read_scores_txt()

        #if os.path.isfile(os.path.join(self.dir,"metric.pickle")):
        #    self.metric = pickle.load(open(os.path.join(self.dir,"metric.pickle"),'r'))
        #    #self.metric = np.identity(self.nmod)
        #    if len(self.metric) == 1: self.metric = self.metric[0]
#
            #self.check_metric()
 #       else: 
 #           print "Copying metrics over, rerun program."
    

    def read_output_txt(self):
        filename = self.dir + "output.txt"
        try:
            f = open(filename,'r')
        except IOError:
            print self.dir + ": ", filename, 'not found'
            return None
        linelists = [[l.replace(',','').replace('(','').replace(')','').replace(':','').replace('[','').replace(']','') for l in k.split(' ')] for k in f.read().split('\n')]
        f.close()
        df_dict = {}
        for l in linelists:
            #print l
            key, data = l[0], l[1:]
            if key == 'head':
                self.nmod,self.numdraws,self.stepsize = int(data[0]), int(data[1]), float(data[2])
            elif key == "smallestdist": 
                #print data
                if 'j' not in str(data):
                    self.add_dist(data)
            elif key == "rws":
                self.add_score(data)
        
        self.dist_df = pd.DataFrame(self.dist)
        self.s_df = pd.DataFrame(self.s)
        self.trackscore_df = pd.DataFrame(self.track_scores)

    def read_scores_txt(self):
        filename = self.dir + "scores.txt"
        try:
            f = open(filename,'r')
        except IOError:
            print self.dir + ": ", filename, 'not found'
            self.has_scores = False
            return None
        scoresfile = [n.split("\t") for n in f.read().split("\n")]
        f.close()
        scores_head = scoresfile[0]
        scores = [[cast_int_then_float(l) for l in p] for p in scoresfile[1:len(scoresfile)-1]]
        for key in scores_head: 
            self.scores[key] = []
            idx = scores_head.index(key)
            for i in range(len(scores)):
                self.scores[key].append(scores[i][idx])

        if 'mean' not in self.scores or len(self.scores['mean']) == 0: 
            self.has_scores = False
            #print self.command_string(), "has no scores yet"
        self.scores_df = pd.DataFrame(self.scores)

    def read_command_txt(self):
        filename = self.dir + "command.txt"
        try:
            f = open(filename,'r')
        except IOError:
            print self.dir + ": ", filename, 'not found'
            return None
        words = [[l for l in k.split(' ')] for k in f.read().split('\n')][0]
        f.close()
        try:
            f = open(filename,'r')
        except IOError:
            print self.dir + ": ", filename, 'not found'
            return None
        self.c_string_test = f.read()
        f.close()
        if '--steps' in words: self.steps = int(words[words.index('--steps')+1])
        if '--reward-fct' in words: self.reward_fct = words[words.index('--reward-fct')+1]
        if '--env' in words: self.env = words[words.index('--env')+1]
        if '--gamma' in words: self.gamma = float(words[words.index('--gamma')+1])       
        if '--beta' in words: self.beta = float(words[words.index('--beta')+1])  
        if '--moments' in words: self.moments = str(words[words.index('--moments')+1])  
        if '--outdir' in words: self.outdir = words[words.index('--outdir')+1]
        if '--numdraws' in words: self.numdraws = words[words.index('--numdraws')+1]
        if '--stepsize' in words: self.stepsize = words[words.index('--stepsize')+1]
        if '--weights' in words: self.weights = words[words.index('--weights')+1]

        if 'home' in self.outdir: self.outdir = self.outdir.replace("home","scratch") # in case not using scratch

        if '--arch' in words and words[words.index('--arch')+1] == "LSTMFR": self.lstm = "LSTMFR"
        if '--arch' in words and words[words.index('--arch')+1] == "LSTMJH": self.lstm = "LSTMJH"
        if '--arch' in words and words[words.index('--arch')+1] == "LSTMGaussian": self.lstm = "LSTMGaussian"
        train_word = None
        for w in words:
            if 'train' in w:
                train_word = w
                break
        if 'MCTS' in train_word: self.mcts = "MCTS"

    def command_string(self):
        s = self.env
        s += "nmod" + str(self.nmod)
        s += "s" + str(self.steps)
        s += "m" + str(self.moments)
        s += "w" + str(self.weights)
        s += "nd" + str(self.numdraws)
        #s += "sig" + str(self.sigma)
        s += "ss" + str(self.stepsize)
        s += "g" + str(self.gamma)
        s += "b" + str(self.beta)
        if self.mcts != None:
            s+= "_mcts"
        if self.lstm != None:
            s+= "_lstm"
        return s

    def add_dist(self,tup): # add positive
        #print tup
        newt = []
        for t in tup:
            newt.append(t)
        tup = newt
        #print tup
        self.dist['worker'].append(int(tup[0]))
        self.dist['time'].append(int(tup[1]))
        #self.p['cc'].append(float(tup[2].replace('mpf\'','').replace('\'','')))
        self.dist['dist'].append(float(tup[2]))
        self.dist['state'].append([int(float(l)) for l in tup[3:]])


    #def add_s(self,tup): # add solution
    #    newt = []
    #    for t in tup:
    #        if 'mpf' in t: 
    #            newt.append(str_to_mpf(t))
    #        else:
    #            newt.append(t)
    #    tup = newt
    #    self.s['worker'].append(int(tup[0]))
    #    self.s['time'].append(int(tup[1]))
    #    self.s['cc_mpf'].append(tup[2])
    #    self.s['cc'].append(float(tup[2].replace('mpf\'','').replace('\'','')))
    #    self.p['state'].append([int(float(str(l).replace('mpf(\'','').replace('\')',''))) for l in tup[3:]])  

    def add_score(self,tup): # add positive
        newt = []
        for t in tup:
            newt.append(t)
        tup = newt
        if int(tup[1]) != 0:
            self.track_scores['worker'].append(int(tup[0]))
            self.track_scores['time'].append(int(tup[1]))
            self.track_scores['score'].append(float(tup[2]))

    def min_dist(self): 
        dist_min = 1e10
        if len(self.dist_df['dist']) > 1:
            dist_min = min(self.p_df['dist'])
        #s_min = 1e10
        #if len(self.s_df['dist']) > 1:
        #    s_min = min(self.s_df['dist'])
        return float(dist_min)



   # def num_solutions(self): return len(self.s['dist'])

    def exp_desc_dict(self):
        return {'env': self.env, 'gamma': self.gamma, 'beta': self.beta, 'nmod': self.nmod, 'steps': self.steps, 'sigma': self.sigma, 'reward_fct': self.reward_fct, 'moments': self.moments, 'numdraws': self.numdraws, 'stepsize': self.stepsize, 'weights': self.weights}

    def equals(self,wish_exp):

        if self.steps != wish_exp.steps: return False 
        if self.env != wish_exp.env: return False
        #if self.sigma != wish_exp.sigma: return False
        #if self.eps != wish_exp.eps: return False
        if self.nmod != wish_exp.nmod: return False
        if self.gamma != wish_exp.gamma: return False
        if self.beta != wish_exp.beta: return False
        if self.moments != wish_exp.moments: return False
        if self.numdraws != wish_exp.numdraws: return False
        if self.stepsize != wish_exp.stepsize: return False
        if self.weights != wish_exp.weights: return False
        #if self.pow != wish_exp.pow: return False
        #if self.reward_fct != wish_exp.reward_fct: return False
        #if self.metric_index != wish_exp.metric_index: return False
        if self.lstm != wish_exp.lstm: return False
        if self.mcts != wish_exp.mcts: return False

        return True

    def last_nn_folder(self):
        files, dirs = [x for x in os.listdir(self.dir)], []
        self.exps = []
        for d in files:
             if os.path.isdir(self.dir+d):
                if 'finish' in d:
                    return d
                else:
                    dirs.append(int(d))
        if dirs != []:
            return os.path.join(os.path.join(os.path.join(self.discovery_dir,self.dir.split('/')[-3]),self.timestamp),str(max(dirs)))
        if dirs == []:
                return False

    def create_job_file(self, submit_string, home_dir, partition = "fullnode"):
        cmd = self.command_string()
        s = "#!/bin/bash"
        s += "\n#SBATCH --job-name=" + cmd
        s += "\n#SBATCH --output=" + cmd + ".out"
        s += "\n#SBATCH --error=" + cmd + ".err"
        s += "\n#SBATCH --exclusive"
        s += "\n#SBATCH --partition=" + partition
        s += "\n#SBATCH -N 1"
        s += "\n#SBATCH --workdir=" + os.path.join(home_dir,"workdir")
        s += "\ncd " + home_dir
        s += "\nmpiexec -n 1 python " + submit_string
        s += "\n"

        if not os.path.isdir("../scripts/new_jobs_"+self.cur_time):
            os.system("mkdir ../scripts/new_jobs_"+self.cur_time)
        f = open("../scripts/new_jobs_"+self.cur_time+"/"+cmd+".job",'w')
        f.write(s)
        f.close()

    def continue_training(self):
        nndir = self.last_nn_folder()
        if nndir != False:
            nndir = nndir.replace('/home/codylong/Bergman/CYBergmanouts','/scratch/codylong/BergmanCY')
            print 'yo', self.outdir
            if self.mcts != None: 
                python_file = 'train_a3c_gym_MCTS.py'
            else: 
                python_file = 'train_a3c_gym.py'
            substring = python_file + " 32 --steps " + str(self.steps) + " --eval-interval 2000000 --env " + self.env + " --outdir " + self.outdir + " --gamma " + str(self.gamma)+ ' --moments ' + str(self.moments)  + " --nmod " + str(self.nmod)  + " --beta " + str(self.beta)+ " --eval-interval 200000 "  +  " --numdraws " + str(self.numdraws) + " --stepsize " + str(self.stepsize) + ' --weights ' + str(self.weights)  + " --load " + str(nndir)
            if self.lstm != None:
                substring += " --arch " + self.lstm 

            self.create_job_file(substring,self.discovery_dir)

  #  def continue_training_new_origin(self):
   #     nndir = self.last_nn_folder()
   #     if nndir != False:
   #         print 'yo', self.outdir
   #         if self.mcts != None: 
   #             python_file = 'train_a3c_gym_MCTS.py'
   #         else: 
   #             python_file = 'train_a3c_gym.py'
   #         substring = python_file + " 32 --steps " + str(self.steps) + " --eval-interval 2000000 --env " + self.env + " --outdir " + self.outdir + " --gamma " + str(self.gamma) + " --eps " + str(self.eps) + " --nmod " + str(self.nmod) + " --sigma " + str(self.sigma) + " --beta " + str(self.beta)+ " --reward-d-pow " + str(self.pow) + " --origin " + str(self.get_new_origin())
   #         if self.lstm != None:
   #             substring += " --arch " + self.lstm 
#
#            self.create_job_file(substring,self.discovery_dir)

    ###
    # for analyzing the experiments

  #  def get_new_origin(self):
  #      if len(self.p_df['cc']) > 0: 
  #          preorg = self.p_df['state'][list(self.p_df['cc']).index(min(self.p_df['cc']))]
  #          self.compute_origin()
  #          theorg = np.array(self.origin) + np.array(preorg)
  #          org = ''
  #          for i in range(self.nmod-1):
  #              org = org + str(theorg[i]) + ','
  #          org = org + str(theorg[self.nmod - 1])
  #          return org

    def max_score(self): 
        if len(self.scores_df['max']) > 0: return max(self.scores_df['max'])
        return None

    def min_dist(self):
        if len(self.dist_df['dist']) > 0: return min(self.dist_df['dist'])

    #def num_s(self):
    #    return len(self.s_df)

    def dist_plot(self):
        ax = sns.lmplot(x="time",y="dist",data=self.dist_df,fit_reg=False)
        ax.set_xticklabels(rotation=30)
        #ax.set(yscale="log")
        ax2 = plt.gca()
        ax2.set_title(self.command_string())
        plt.ylim(min(self.dist_df['dist'])/3, max(self.dist_df['dist'])*3)
    
    def track_plot(self):
        ax = sns.lmplot(x="time",y="score",data=self.trackscore_df,fit_reg=False)
        ax.set_xticklabels(rotation=30)
        ax.set(yscale="log")
        ax2 = plt.gca()
        ax2.set_title(self.command_string())
        plt.ylim(min(self.trackscore_df['score'])/3, max(self.trackscore_df['score'])*3)

    #def s_plot(self):
     #   ax = sns.lmplot(x="time",y="cc",data=self.s_df,fit_reg=False)
      #  ax.set_xticklabels(rotation=30)
      #  ax.set(yscale="log")
      #  ax2 = plt.gca()
      #  ax2.set_title(self.command_string())
      #  plt.ylim(min(self.s_df['cc'])/3, max(self.s_df['cc'])*3)
        

    def scores_plot(self):
        if self.has_scores == False:
            return False
            #print self.command_string(), "has no scores"
        #print "steps" in self.scores, "mean" in self.scores
        ax = sns.lmplot(x="steps",y="mean",data=self.scores_df,fit_reg=False)
        #ax.set_xticklabels(rotation=30)
        ax.set(yscale="log")
        #ax2 = plt.gca()
        #ax2.set_title(self.command_string())
        plt.ylim(min_pos_mean(self.scores_df)/3, max_pos_mean(self.scores_df)*3)

   # def origin_cc(self): return self.compute_cc([0 for i in range(self.nmod)])

  #  def check_metric(self):
  #      num_checks = 10
  #      for i in range(num_checks):
  #          idx = np.random.randint(0,len(self.p_df))
  #          state, cc = self.p_df['state'][idx], self.p_df['cc'][idx]
  #          cc_new = self.compute_cc(state)
            #if abs(cc_new/cc) > 1.1 or abs(cc_new/cc) < .9:
            #    print "METRIC DOESN'T COMPUTE CC CORRECTLY!",  type(cc), type(cc_new), abs(cc_new/cc), cc, self.compute_cc(state)
            #    break

    # def compute_origin(self):
    #     eig_vals, eig_vecs = np.linalg.eig(np.array([[np.float(ii) for ii in jj] for jj in self.metric]))
    #     eig_vecs = np.transpose(eig_vecs)
    #     new_eig_vecs = []
    #     for i in range(len(eig_vecs)):
    #         new_eig_vecs.append(eig_vecs[i]/eig_vals[i]**.5)
    #     rounded_evecs = [[int(round(entry)) for entry in ev] for ev in new_eig_vecs]
    #     self.origin = [mpf(e) for e in rounded_evecs[np.argmin(eig_vals)]]

    # def compute_cc(self,v):
    #     return mpf(-1) + self.compute_cc_norm(v)

    # def compute_cc_norm(self,v):
    #     if self.origin == None:
    #         self.compute_origin()
    #     s = np.array([mpf(e) for e in v]) + np.array(self.origin)
    #     return np.dot(np.dot(self.metric,s),s)
        
    #def is_interesting(self):
    #     int1, int2, int3, int4 = None, None, None, None
    #     if self.eps < self.sigma * self.sigma * 1e-5 and self.num_s() > 0: int1 = True
    #     if self.min_p_cc() < self.sigma * self.sigma * 1e-5: int2 = True
    #     if self.min_p_cc() < self.compute_cc([0 for i in range(self.nmod)]) * 1e-10: int3 = True
        
    #     #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self.scores_df['steps'], self.scores_df['mean'])
    #     #if slope > 0 and abs(r_value) >= .1: int4 = True
    #     if 'steps' not in self.scores:
    #         print "no scores in this one"
    #         print "command string is ", self.c_string_test
    #     if ('steps' in self.scores and self.scores['steps'] == []):
    #         print "scores printed but no numbers"
    #         print "command string is ", self.c_string_test
    #     if 'steps' not in self.scores or ('steps' in self.scores and self.scores['steps'] == []):
    #         if (datetime.datetime.now()-self.timestamp_datetime).days >= 1:
    #             print "No scores, but over 24 hours! Ignoring."
    #             return False
    #         else:
    #             print "No scores! Not auto-ignoring now, though, since < 24 hours."
    #             return True
    #     else:
    #         smax, smin = max(self.scores['steps']), min(self.scores['steps'])
    #         l = smax - smin
    #         d3 = [self.scores['mean'][i] for i in range(len(self.scores['mean'])) if self.scores['steps'][i] < smin + .01 *l]
    #         d2 = [self.scores['mean'][i] for i in range(len(self.scores['mean'])) if self.scores['steps'][i] > smax - .1 *l]  
    #         s, m = np.std(d3), np.mean(d3)
    #         d1 = [d for d in d3 if d <= m + s]
    #         if np.mean(d2) >= np.mean(d1)*(10**self.pow): int4 = True

    #         if (int1,int2,int3,int4) == (None, None, None, None) and self.nmod != 250: 
    #             return False
    #         else:
    #             return (int1,int2,int3,int4)

class Wishart_Experiment_EQclass:
    def __init__(self,exps):
        self.exps = exps
        #print "num exps seen = ", len(self.exps)
        self.last_exp = self.exps[-1]
        self.exp = self.last_exp # for rep'ing exp-indep things
        self.dirs = [e.dir for e in self.exps]

        for i in range(1,len(self.exps)): # check equivalence
            assert self.exps[i].equals(self.exps[i-1]) 

        ## Compute aggreggate scores
        self.scores = {}
        for key in self.exps[0].scores:
            self.scores[key] = self.exps[0].scores[key]

        if 'steps' in self.scores and len(self.scores['steps']) > 0:
            lastmax = max(self.scores['steps'])
            for i in range(1,len(self.exps)):
                cur_scores = self.exps[i].scores
                if 'steps' in cur_scores:
                    for key in cur_scores:
                        for val in cur_scores[key]:
                            if key == 'steps':
                                self.scores[key].append(val+lastmax)
                            else:
                                self.scores[key].append(val)
                    lastmax += max(cur_scores['steps'])
                
            self.scores_df = pd.DataFrame(self.scores)

        self.has_scores = ('mean' in self.scores and len(self.scores['mean']) != 0)


    def study(self, questions = False):
        print "\n\nStudying " + self.exp.command_string()
        print "Num exps in eq class is:", len(self.exps)
        print self.dirs

        print "min distance: ", self.min_dist()
        print "max score: ", self.max_score()
       # print "num solutions: ", self.num_s()
        #print "origin cc: ", self.exp.compute_cc([0 for i in range(self.exp.nmod)])
        #print "last exp in eq class interesting:", self.last_exp.is_interesting()
       # for e in self.exps:
        #    e.scores_plot()
        #    plt.show()
        self.scores_plot()
        plt.show()
        if questions:
            cont = None
            if not os.path.isfile(self.last_exp.dir+"continue.txt"):
                while cont not in ['y','n']:
                    cont = raw_input("Continue training? (y/n) ")
                if cont == "y":
                    f = open(self.last_exp.dir+"continue.txt",'w')
                    f.write('y')
                    f.close()
                    self.last_exp.continue_training()
            else: 
                print "Already continued training this EQ class."

            # keep = raw_input("Keep? (y/n/i) ")
            # if keep == 'n':
            #     print "Ignoring the experiment in the future."
            #     f = open(self.dir+"ignore.txt",'w')
            #     f.write('y')
            #     f.close()
            # elif keep == 'y':
            #     plt.savefig(os.path.join(self.dir,"scores.png"),dpi=200)
            #     cont = raw_input("Continue training? (y/n) ")
            #     if cont == 'y':
            #         self.continue_training()    

    def scores_plot(self): # aggregate scores plot
        if self.has_scores == False:
            return False
        ax = sns.lmplot(x="steps",y="mean",data=self.scores_df,fit_reg=False)
        ax.set(yscale="log")
        plt.ylim(min_pos_mean(self.scores_df)/3, max_pos_mean(self.scores_df)*3)

    def min_dist(self): return min([e.min_dist() for e in self.exps])
    def max_score(self): return max([e.max_score() for e in self.exps])
    #def num_s(self): return sum([e.num_s() for e in self.exps])


class Wishart_Experiments:

    def __init__(self,parent_dir,discovery_dir="/home/codylong/Bergman/",respect_ignores = True, respect_day = False): 
        cur_t = str(datetime.datetime.now()).replace(" ",'_').replace(":",".")
        cur_t = cur_t[:cur_t.rindex(".")]
        self.parent_dir = parent_dir
        self.discovery_dir = discovery_dir
        if parent_dir[len(parent_dir)-1] != '/': self.parent_dir += '/'
        dirs = [x for x in os.listdir(self.parent_dir)]

        now = datetime.datetime.now()

        self.exps = []
        for d in dirs:
            if os.path.isdir(parent_dir+d) and(respect_ignores == False or not os.path.isfile(parent_dir+d+"/ignore.txt")) and (respect_day == False or (now-datetime.datetime.strptime(d,'%Y%m%dT%H%M%S.%f')).days >= 1):
                print parent_dir + d
                if os.path.isfile(parent_dir+d+"/output.txt") and os.path.isfile(parent_dir+d+"/scores.txt"):
                    self.exps.append(Wishart_Experiment(self.parent_dir + d))
                    self.exps[-1].discovery_dir = self.discovery_dir
                    self.exps[-1].cur_time = cur_t
                    e = self.exps[-1]
                   # if not os.path.isfile(parent_dir+d+"/metric.pickle"):
                    #    os.system("cp ../metrics/metric" + str(e.nmod) + "sig" + str(e.sigma) +"v1.pickle " + os.path.join(e.dir,"metric.pickle"))
        
        print "Computing EQ classes and auto ignores"
        self.eq_classes = self.equiv_classes()
        #self.compute_auto_ignores()
        self.cont_classes = self.get_cont_classes()
    ### hacky stuff to look at continued experiments
    def get_cont_classes(self):
        to_cont = []
        for eq in self.eq_classes:
            if len(eq.exps) > 0:
                to_cont.append(eq)
        return to_cont
    def create_all_scores_plots_cont(self):
        if not os.path.isdir(os.path.join(self.parent_dir,"all_plot_cont")):
            os.system("mkdir " + os.path.join(self.parent_dir,"all_plots_cont"))
        for eq in self.cont_classes:
            if eq.scores_plot() != False: # returns false if no scores yet
                plt.savefig(os.path.join(os.path.join(self.parent_dir,"all_plots_cont"),eq.exps[0].command_string()+"numexps"+str(len(eq.exps)) +".png"),dpi=200)
                plt.clf()



    ############### end hacky stuff
    def reset_ignores(self):
        dirs = [x for x in os.listdir(self.parent_dir)]
        for d in dirs:
            if os.path.isdir(self.parent_dir+d) and os.path.isfile(self.parent_dir+d+"/ignore.txt"):
                os.system("rm " + self.parent_dir+d+"/ignore.txt")

    def min_dist(self):
        cur_min = 1e10
        for exp in self.exps:
            this_min = exp.min_dist()
            if this_min != None:
                if this_min < cur_min: cur_min = this_min
        return cur_min

    def exp_max_score(self):
        cur_max, max_exp = -1e10, None
        for exp in self.exps:
            this_max = exp.max_score()
            if this_max > cur_max: 
                cur_max = this_max
                max_exp = exp
        return max_exp

    #def exp_max_num_solutions(self):
     #   cur_max, max_exp = -1e10, None
     #   for exp in self.exps:
     #       this_max = exp.num_solutions()
     #       if this_max > cur_max: 
     #           cur_max = this_max
     #           max_exp = exp
     #   return max_exp

    def concatenate(self): # returns concatenated DataFrame
        return pd.concat([exp.to_df() for exp in self.exps])

    def equal_exps(self,wish_exp):
        return [e for e in self.exps if e != wish_exp and e.equals(wish_exp)]

    def continue_training(self,wish_exp):
        eqs = self.equal_exps(wish_exp)
        timestamps = [e.timestamp for e in eqs]
        years = [stamp[0:4] for stamp in timestamps]
        months = [stamp[4:6] for stamp in timestamps]
        days = [stamp[6:8] for stamp in timestamps]
        hours = [stamp[8:10] for stamp in timestamps]
        minutes = [stamp[10:12] for stamp in timestamps]
        seconds = [stamp[12:] for stamp in timestamps]
        tosort =  [[timestamps[i],years[i],months[i],days[i],hours[i],minutes[i],seconds[i]] for i in range(len(timestamps))]
        sortedstamps = sorted(tosort, key = operator.itemgetter(1, 2, 3, 4, 5, 6))
        toreverse = [s[0] for s in sortedstamps]
        orderedstamps = toreverse[::-1]
        for e in eqs:
            if e.timestamp == orderedstamps[0]:
                e.continue_training()
                break

    def study_experiments(self, questions = False):
        count = 0
        int_eq_classes = [eq_c for eq_c in self.eq_classes if eq_c.exps[0].is_interesting() != False]
        print "There are", len(int_eq_classes), " interesting equivalence classes"
        for eq_c in int_eq_classes:
            eq_c.study(questions=questions)

    def create_all_scores_plots(self):
        if not os.path.isdir(os.path.join(self.parent_dir,"all_plots")):
            os.system("mkdir " + os.path.join(self.parent_dir,"all_plots"))
        for e in self.exps:
            if e.scores_plot() != False: # returns false if no scores yet
                plt.savefig(os.path.join(os.path.join(self.parent_dir,"all_plots"),e.command_string()+".png"),dpi=200)
                plt.clf()

    def compute_auto_ignores(self):
        for eq_c in self.eq_classes:
            e = eq_c.exps[0]
            if not e.is_interesting():
                print "Ignoring " + e.dir
                f = open(e.dir+"ignore.txt",'w')
                f.write('y')
                f.close()

    def analyze_params(self):
        d = {'gamma': [], 'beta': [], 'nmod': [], 'weights': [], 'stepsize': [], 'steps': [], 'moments': [], 'min_dist': [], 'numdraws': []}
        for e in self.exps:
            d['gamma'].append(e.gamma)
            d['beta'].append(e.beta)
            d['nmod'].append(e.nmod)
            d['weights'].append(e.weights)
            d['stepsize'].append(e.stepsize)
            d['steps'].append(e.steps)
            d['moments'].append(e.moments)
            d['min_dist'].append(e.min_dist())
            d['numdraws'].append(e.numdraws)
        df = pd.DataFrame(d)
        df['log10beta'] = np.log10(df['beta'])
        df['log10min_cc'] = np.log10(df['min_cc'])
        df.hist()
        plt.show()
        pd.plotting.scatter_matrix(df)
        plt.show()

    def equiv_classes(self):
        eq = []
        for e in self.exps:
            notfound = True
            for e2s in eq:
                if e.equals(e2s[0]):
                    e2s.append(e)
                    notfound = False
                    break
            if notfound: eq.append([e])

        new_eq = []
        for eq_c in eq:
            dt = [e.timestamp_datetime for e in eq_c]
            dt.sort()
            #print "len dt = ", len(dt)
            eq_c_new = []
            cur_dt = 0
            #print dt
            #for e in eq_c:
            #    print e.timestamp_datetime
            #    if e.timestamp_datetime == dt[cur_dt]:
            #        eq_c_new.append(e)
            #        cur_dt += 1
            for cur_dt in range(len(dt)):
                for e in eq_c:
                    #print e.timestamp_datetime
                    if e.timestamp_datetime == dt[cur_dt]:
                        eq_c_new.append(e)
            new_eq.append(Wishart_Experiment_EQclass(eq_c_new))
            #print "len exps = ", len(eq_c_new)

        return new_eq

    def eqcs_with_days_diff(self,days_diff): #eq classes with < daysdiff
        now = datetime.datetime.now()
        return [eqc for eqc in self.eq_classes if (now-eqc.last_exp.timestamp_datetime).days <= days_diff]

if __name__=="__main__":
    exps = Wishart_Experiments(sys.argv[1])

