import sys
import os
from modelstat import NNStat
import torch as tor

from pydoc import locate


if len(sys.argv) == 1:
    print("Missing argument MODELNAME:")
elif len(sys.argv) > 2:
    print("Too many arguments:")

if len(sys.argv) != 2:
    print("\t> python benchmark.py MODELNAME")
    sys.exit()

mfile = "models/"+sys.argv[1]+".pt"
if not os.path.isfile(mfile):
    print(mfile,"not found")
    sys.exit()
 
Net = locate('models.'+sys.argv[1]+'.Net')

mod = tor.load(mfile)
stat = NNStat()

stat.accuracy(mod)
print()
stat.class_accuracy(mod)