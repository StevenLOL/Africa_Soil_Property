import os
import sys

tg=sys.argv[1];
ref=sys.argv[2];
ds=open(ref).readlines()
t=ds[0];
ds=[s.strip().split(',')[0] for s in ds]
ds2=open(tg).readlines()
ofile =open(tg,'w')
ofile.write(t);
for i in xrange(1,len(ds)):
	ofile.write(ds[i]+','+ds2[i-1]);
ofile.close();
