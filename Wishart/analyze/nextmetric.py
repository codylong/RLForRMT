import os
import sys

# assume running from metrics folder
nmod = sys.argv[1]

existing_metric_files = [f for f in os.listdir(".") if "metric"+str(nmod)+"v" in f]
metric_versions = [int(f[f.index('v')+1:f.index('.')]) for f in existing_metric_files]
metric_versions.sort()

print "Existing metric versions for nmod = ", nmod, ":"
print metric_versions
print "\nNext metric version:", max(metric_versions)+1

