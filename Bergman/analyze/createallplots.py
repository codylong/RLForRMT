from analysis import *

if __name__=="__main__":
    mp.dps = 200
    exps = Wishart_Experiments(sys.argv[1])
    exps.create_all_scores_plots()