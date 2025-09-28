from __future__ import print_function, division
from TAPS import *
from Confs import Confs
import time
import mdtraj as md
import numpy as np
import shutil

dirPars = '2A_LSD_80/pars'
parFile = 'taps.par'
topFile = 'step7_10.gro'
p0File = 'aligned_first50.xtc'
alignFile = 'align.ndx'
rmsFile = 'rms.ndx'

taps = TAPS(dirPars, parFile, topFile, p0File, alignFile, rmsFile)
p0 = taps.refPath
p1 = p0.rmClose(0.09)
p1.nodes.save('aligned_first50_rc009.xtc')
p1.pcv(dire=dirPars)

