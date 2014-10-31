#!/usr/bin/python
'''
replace all " in the R output, so matlab could read it as a normal csv file.
--steven

'''
import os
import sys
for root, dir, files in os.walk("."):
    for file in files:
        if file.lower().endswith('.csv'):
            fdata=open(file).read()
            fdata=fdata.replace('"','')
            fout=open(file,'w')
            fout.write(fdata)
            fout.close()