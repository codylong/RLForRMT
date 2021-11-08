import subprocess
import os
import sys

d = sys.argv[1]
files = subprocess.check_output('ls ' + d, shell=True).split('\n')

for my_file in files:
    if my_file[:2] == "cc":
        print d + "/" + my_file
        output=subprocess.Popen("sbatch " +d+"/" + my_file,shell=True,stdout=subprocess.PIPE).communicate()[0]
