import os
import cPickle as pickle
import numpy as np
import sys

resDict = {}
resStates = []

path = str(sys.argv[1])

files = os.listdir(path)

for file in files:
    currentDictArray = []
    # currentStates = []

    # do the dictionary
    if "outputOvercountsRun" in file:
        f = open(path + str(file),'r')

        # read in pickle files
        while True:
            try:
                currentDictArray.append(pickle.load(f))
            except:
                f.close()
                break

        for items in currentDictArray:
            for key, val in items[1].items():
                if key in resDict:
                    resDict[key] = resDict[key] + val
                else:
                    resDict[key] = val

    # # do the states
    # if "outputStatesRun" in file:
    #     f = open(path + str(file),'r')
    #     currentStates = pickle.load(f)
    #     f.close()
    #     for item in currentStates[1]:
    #         if item not in resStates:
    #             resStates.append(item)


# print results 
print "\nNumber of distinct solutions:"
print resDict

# print "\nCurrent States:"
# print resStates

# write results
hnd = open(path + "outputOvercountsRun_results.txt", 'w')
hnd.write(str(resDict))
hnd.close()

# hnd = open(path + "outputOvercountsRun_results", w)
# hnd.write(resStates)
# hnd.close()
