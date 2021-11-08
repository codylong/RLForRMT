import subprocess
import sys
import os

file_dir = sys.argv[1]

infile, outfile = 'output.txt', 'output.txt'

dirs = [d for d in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir,d))]

for d in dirs:
	if os.path.isfile(os.path.join(file_dir,os.path.join(d,infile))):
		f = subprocess.check_output("cat "+ os.path.join(file_dir,os.path.join(d,infile)),shell=True)
		lines = f.split("\n")

		pstates = []
		newlines = ""

		for l in lines:
		    if len(l) > 0 and l[0] == 'p':
		        lindex, rindex = l.index('['), l.index(']')
		        s = l[lindex:rindex+1]
		        s = s.replace('mpf(\'','').replace('\')','').replace('.0','')
		        if s not in pstates:
		            pstates.append(s)
		            newlines += l+"\n"
		    elif 'rws' not in l:
		        newlines += l+"\n"

		f = open(os.path.join(file_dir,os.path.join(d,outfile)),"w")
		f.write(newlines)
		f.close()


