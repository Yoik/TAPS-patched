from __future__ import print_function, division
from TAPS import *
from Confs import Confs
import time
import mdtraj as md
import numpy as np
import shutil
from copy import deepcopy

def digits(s1):
    s2 = "%.3d" % s1
    return s2

# =========================================================================================================
#                                       number of iterations per taps
# =========================================================================================================
n_iter = 14

# ===============================================================================================
#                                                input files
# =========================================================================================================
dirPars = '2A_LSD_80/pars'
parFile = 'taps.par'
topFile = 'step7_10.gro'
p0File = 'raw_first50_rc009.xtc'
alignFile = 'align.ndx'
rmsFile = 'rms.ndx'
# =========================================================================================================
taps = TAPS(dirPars, parFile, topFile, p0File, alignFile, rmsFile)
refPath = taps.refPath
refPath.pcv(dire=dirPars) # for computing lamda

dirEvol = './2A_LSD_80_rc009_raw0/paths'
# ======================================================================================================
#       All iterations finished, deciding convergence by looking av(z) per path again initial path
#           re-grow path with data from that the round with median(av(z)) as final output path
# ======================================================================================================
# compute pcv-z of all paths with respect to initial path, find average value of z
z_av = np.zeros([n_iter+1, 2])
for j in range(1, n_iter + 1):
    iter = 'iter' + digits(j)
    p_nodes = md.load(dirEvol + '/' + iter + '.xtc', top=dirPars + '/' + topFile)
    s,z = refPath.pcv(dirEvol, p_nodes)
    np.savetxt(dirEvol + "/s_" + iter + ".dat", s, fmt='%8.8f')
    np.savetxt(dirEvol + "/z_" + iter + ".dat", z, fmt='%8.8f')                
    z_av[j, 0] = j
    z_av[j, 1] = np.sqrt(np.sum(z) / p_nodes.n_frames) /taps.tolDist
np.savetxt(dirEvol + '/evol_avZ.dat', z_av, fmt='%8.8f')

