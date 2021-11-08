import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

filename = sys.argv[1]
print "Analyzing", filename

with open(filename) as input:
    #for line in input: data.append(zip(*(line.strip().split('\t'))))
    rawdata = zip(*(line.strip().split('\t') for line in input))

labels = [dat[0] for dat in rawdata]
rawdata = [[float(k) for k in dat[1:]] for dat in rawdata]

rawdata = map(list,zip(*rawdata))
rawdata.sort(key = lambda row: row[0])
rawdata = map(list,zip(*rawdata))

data = pd.DataFrame()
for i in range(len(rawdata)):
    data[labels[i]] = rawdata[i] #labels[k]:rawdata[k] for k in range(len(rawdata))}

###
# Plots

# mean score vs steps
sns.lmplot('steps','mean',data=data,fit_reg=False)
#plt.show()

# avg steps 
data['steps to solve'] = np.array([None]+[(data['steps'][i]-data['steps'][i-1])/(data['episodes'][i]-data['episodes'][i-1])*1.0 for i in range(1,len(data['steps']))])
avgplot = sns.lmplot('steps','steps to solve',data=data,fit_reg=False)
ax = avgplot.axes[0][0]
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
