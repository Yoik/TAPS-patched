# =======================================================================================================================
#                        global import: future_print, numpy, mdtraj, os, re, argparse
# =======================================================================================================================
from __future__ import print_function, division
import numpy as np
import mdtraj as md
import os
import re
import shutil
import time
import errno
from tqdm import tqdm
from tqdm import tqdm as _tqdm
from contextlib import contextmanager
import sys
# =======================================================================================================================
#                           import python wrappers for MD engines (GROMACS, NAMD, AMBER)
# =======================================================================================================================
import gromacs.setup
import gromacs.run
import gromacs.tools
import gromacs

# For treating gromacs warnings at mdrun as exception in python
#   such that mdrun can be terminated if
import warnings

warnings.simplefilter('error', gromacs.AutoCorrectionWarning)
warnings.simplefilter('error', gromacs.BadParameterWarning)
warnings.simplefilter('error', gromacs.GromacsValueWarning)
warnings.simplefilter('error', gromacs.GromacsFailureWarning)

from Confs import Confs
from PcvInd import PcvInd
from Path import Path


# =======================================================================================================================
#                           import mpi4py for parallel computing
# =======================================================================================================================
import multiprocessing
from mpi4py import MPI
comm = MPI.COMM_WORLD
#print("hello world")
size = comm.Get_size()
rank = comm.Get_rank()
threads = 1
# ======================================================================================================================
#                                       digits formater for iterations: "3 --> 03"
# ======================================================================================================================
def digits(s1):
    s2 = "%.3d" % s1
    return s2
    
# ==================================================================================================================
#                                       wrapper for a single run of umbrella sampling
#                     engine-specific implementation of sampling is realized in this function
# ==================================================================================================================
def runUs(dire, engine, runName, pluName, cvOut, trjName):
    #lock is for parallel process
    if engine == 'GROMACS':
        # redirect gromacs stdout/stderr to per-node log file to avoid noisy shell output
        logfile = os.path.join(dire, 'gromacs.log')
        with _redirect_fds_to_file(logfile):
            meta = gromacs.run.MDrunner(dire, ntomp=threads, deffnm=runName, plumed=pluName)
            meta.run()
        # make sure xtc is complete, under restrains, high energy confs may be generated
        # use trjconv to disregard "unphysical (CV values is nan)" frames
        # meanwhile, there are two other cases to consider
        #  1. when there is no output or only one line in CV file (sampling crashed in the first step)
        #       this can be dealt with by put the endtime at 0
        #  2. when the last line is incomplete
        #       just remove the last line
        keepLineIndex = []
        input = open(dire + '/' + cvOut, 'r+')
        for i, line in enumerate(input):
            if line[0] != "#":
                if not re.match('.*nan.*', line):
                    keepLineIndex.append(i)
        lineCount = len(keepLineIndex)
        output = open(dire + '/colvar_filter', 'w+')
        if lineCount == 1:  # only one line: keep this line
            endTime = 0
            lines = input.readlines()
            line = lines[keepLineIndex[0]]
            output.write(line)
        else:  # many lines in CV file, only remove the last line (incomplete when sampling crashes)
            input = open(dire + '/' + cvOut, 'r+')
            lines = input.readlines()
            output = open(dire + '/colvar_filter', 'w+')
            for k in range(lineCount - 1):  # remove the last line (incomplete when sampling crashes)
                line = lines[keepLineIndex[k]]
                endTime = line.split()[0]
                output.write(line)
        input.close()
        output.close()
        shutil.move(dire + '/' + trjName, dire + '/bak_' + trjName)
        trjconv = gromacs.tools.Trjconv(s=dire + '/' + runName + '.tpr', f=dire + '/bak_' + trjName,
                                        o=dire + '/' + trjName, e=endTime, \
                                        ur='compact', center=True, pbc='mol', input=('Protein', 'System'))
        trjconv.run()
        os.remove(dire + '/bak_' + trjName)
    else:
        raise ValueError("MD engines other than GROMACS are not support yet")

# ======================================================================================================================
#                               class TAPS: encoding methods for each iteration of TAPS
# ======================================================================================================================
class TAPS(object):
    # default structure file, these names are important during sampling and plumed computation
    nodeName = 'node.pdb'
    runName = 'run'
    trjName = 'run.xtc'
    trjFilter = 'filtered.xtc'
    pluName = 'plumed.dat'
    cvOut = 'COLVAR'

    # ==================================================================================================================
    #    constructor: read in taps parameters and relevant files (system topology, initial path, PCV definition)
    # ==================================================================================================================
    def __init__(self, dire='pars', parFile='taps.par', topFile='protein.pdb', p0='path0.xtc', alignFile='align.ndx', \
                 rmsFile='rms.ndx'):

        # check if inputs exists
        if not os.path.isdir(dire):
            raise ValueError("Directory %s for initial path & parameters does not exist" % dire)
        if not os.path.exists(dire + '/' + parFile):
            raise ValueError("Parameters file %s is not found in directory %s" % (parFile, dire))
        if not os.path.exists(dire + '/' + topFile):
            raise ValueError("Structure file %s is not found in directory %s" % (topFile, dire))
        if not os.path.exists(dire + '/' + p0):
            raise ValueError("Trajectory of initial path (%s) is not found in directory '%s'" % (p0, dire))
        if not os.path.exists(dire + '/' + alignFile):
            raise ValueError("Atom index file for alignment (%s) is not found in directory %s" % (alignFile, dire))
        if not os.path.exists(dire + '/' + rmsFile):
            raise ValueError("Atom index file for rms computation (%s) is not found in directory %s" % (rmsFile, dire))

        # record root directory
        self.dirRoot = os.getcwd()

        # record directory for initial path and parameters
        self.dirPar = self.dirRoot + '/' + dire

        # record topology file name and position
        self.topNAME = topFile
        self.topFile = self.dirPar + '/' + topFile

        # record alignment index file position
        self.alignFile = self.dirPar + '/' + alignFile

        # record rms index file position
        self.rmsFile = self.dirPar + '/' + rmsFile

        # load atom indices for PCV definition (alignment & rmsd calculation)
        align = np.fromstring(open(self.dirPar + '/' + alignFile).read(), sep=' ', dtype=np.int32)
        rms = np.fromstring(open(self.dirPar + '/' + rmsFile).read(), sep=' ', dtype=np.int32)
        self.pcvInd = PcvInd(align, rms)

        # load initial refPath (compute initial s, included)
        self.refPath = Path('iter' + digits(0), self.pcvInd)
        self.refPath.loadFromTRJ(self.dirPar + '/' + p0, self.dirPar + '/' + topFile)

        # initialize initial node (extracting from initial path)
        self.initNode = self.refPath.nodes.slice(0)

        # initialize final node (extracting from initial path)
        self.finalNode = self.refPath.nodes.slice(self.refPath.n_nodes - 1)

        # read in parameters for MD and metaD
        fr = open(self.dirPar + '/' + parFile, 'r+')
        pars = fr.read()
        fr.close()

        # MD parameters
        # engine specific input check
        match = re.search("engine=.*\n", pars)
        if match is not None:
            self.engine = re.split('=', match.group(0).rstrip('\n'))[1]
        else:
            raise ValueError("MD engine not given in parameter file %s" % (parFile))
        if self.engine == 'GROMACS':
            match = re.search("groTOP=.*\n", pars)
            if match is not None:
                self.groTOP = re.split('=', match.group(0).rstrip('\n'))[1]
                if not os.path.exists(self.dirPar + '/' + self.groTOP):
                    raise ValueError("GROMACS topology file %s is not found in directory %s" % (self.groTOP, \
                                                                                                self.dirPar))
            else:
                raise ValueError("GROMACS topology file not given in %s" % (parFile))
            match = re.search("groMDP=.*\n", pars)
            if match is not None:
                self.groMDP = re.split('=', match.group(0).rstrip('\n'))[1]
                if not os.path.exists(self.dirPar + '/' + self.groMDP):
                    raise ValueError("GROMACS template mdp file %s is not found in directory %s" % (self.groMDP, \
                                                                                                    self.dirPar))
            else:
                raise ValueError("gromacs mdp file %s not given in %s" % (parFile))
        elif self.engine == 'NAMD':
            raise ValueError('NAMD is not supported yet')
        elif self.engine == 'AMBER':
            raise ValueError('AMBER is not supported yet')
        else:
            raise ValueError("unknown MD engine %s" % self.engine)

        # mode = {serial, parallel, qjob}
        match = re.search("runMode=.*\n", pars)
        if match is not None:
            self.mode = re.split('=', match.group(0).rstrip('\n'))[1]
        else:
            raise ValueError("Mode of running (runMode) not given in parameter file %f" % (parFile))

        # time step
        match = re.search("timeStep=.*\n", pars)
        if match is not None:
            self.timeStep = float(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            raise ValueError("MD timestep (timestep, unit: ps) not given in parameter file %f" % (parFile))

        match = re.search("lenSample=.*\n", pars)
        if match is not None:
            self.lenSample = float(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            raise ValueError("Amount of sampling per taps iteration ('lenSample', unit: ps) not given in \
            parameter file %f" % (parFile))

        # output frequency of trajectories
        match = re.search("freqTRJ=.*\n", pars)
        if match is not None:
            self.freqTRJ = int(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            raise ValueError("Output frequency of sampling trajectories (freqTRJ) not given in parameter file %f" % (parFile))

        fr = open(self.dirPar + '/' + self.groMDP, 'r+')
        linesMDP = fr.readlines()
        fr.close()
        mdpFile = 'md.mdp'
        fw = open(self.dirPar + '/' + mdpFile, 'w+')
        fw.writelines(linesMDP)
        print('nstxout-compressed= %d' % self.freqTRJ, file=fw)
        fw.close()
        self.groMDP = mdpFile

        # output frequency of trajectories
        match = re.search("kappa=.*\n", pars)
        if match is not None:
            self.kappa = int(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            raise ValueError("Wall strength on PCV-s (kappa, 10-50) not given in parameter file %f" % (parFile))

        # tolerable restraining potential to ensure "physically irrelevant" conformations are selected
        # selecting frames with small restrain potential is a more direct approach than ds-s[0]<sTol
        # because it makes the choice independent from the kappa of the restraining potential
        match = re.search("tolRS=.*\n", pars)
        if match is not None:
            self.rsTol = float(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            raise ValueError("Tolerable restraining potential (rsTol) not found in parameter file %s \n  This parameter\
             is crucial for selecting frames from MetaD trajectories" % (parFile))

	    # parameters for path-reparameterization
	    # tolerable distance between neighbor nodes, used for reparameterization
        match = re.search("tolDist=.*\n", pars)
        if match is not None:
            self.tolDist = float(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            raise ValueError("Tolerable maximum distance (tolDist) between neighbor nodes not given in parameter\
             file %s\n  This parameter is crucial for path reparameterzation" % (parFile))

        # tolerable asymmetry factor, determines how much deviation from the used for path reparameterization
        match = re.search("devMID=.*\n", pars)
        if match is not None:
            self.devMID = float(re.split('=', match.group(0).rstrip('\n'))[1])
            if self.devMID > 1 or self.devMID <= 0:
                raise ValueError("Parameter devMID out of range ( 0<devMID<=1 required )")
        else:
            raise ValueError(
                "Tolerable deviation from vertical line between two distant nodes (devMID) is not given in parameter \
                file %s\n  This parameter is crucial for path reparameterzation" % (parFile))

        # tolerable cosTheta, used for reparameterization
        match = re.search("tolCos=.*\n", pars)
        if match is not None:
            self.tolCos = float(re.split('=', match.group(0).rstrip('\n'))[1])
            if self.tolCos > 0.5:
                self.tolCos = 0.5
                print("Tolerable cos(theta) in parameter file %s must be <=0.5\n  setting to 0.5" % (parFile))
        else:
            raise ValueError(
                "Tolerable cos(theta) to select \"middle\" conformations between neighbor nodes is not given in \
                parameter file %s" % (parFile))

        # straightening factor
        sub_i = self.initNode.atom_slice(self.pcvInd.atomSlice)
        sub_f = self.finalNode.atom_slice(self.pcvInd.atomSlice)
        sub_f.superpose(sub_i, 0, self.pcvInd.align)
        dist_term = md.rmsd(sub_f, sub_i, 0, self.pcvInd.rms)
        match = re.search("stf=.*\n", pars)
        if match is not None:
            self.stf = float(re.split('=', match.group(0).rstrip('\n'))[1])
            if ((self.stf < 1) or (self.stf > (dist_term/self.tolDist/2.5))):
                 print("Straightening factor (stf) is out of range (must be 1 <= stf <= d[0,end]/tolDist )")
                 self.stf = dist_term / self.tolDist / 3
                 print("Setting stf as d[0,end]/tolDist/3: stf=", self.stf)
        else:
            print("Straightening Factor for path reparameterization (stf) not given in \
                    parameter file %s" % (parFile))
            self.stf = dist_term / self.tolDist / 3
            print("Setting stf as d[0,end]/tolDist/3: stf=", self.stf)

        # wall position of PCV-Z for MetaD
        match = re.search("zw=.*\n", pars)
        if match is not None:
            self.zw = float(re.split('=', match.group(0).rstrip('\n'))[1])
            self.zw = (self.zw * self.tolDist) ** 2
        else:
            raise ValueError("Wall position of PCV-Z for MetaDynamics (zw, unit: nm^2) not given in parameter file %s" % (parFile))

        # wall strength of PCV-Z for MetaD
        match = re.search("zwK=.*\n", pars)
        if match is not None:
            self.zwK = float(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            self.zwK = self.rsTol / (self.tolDist / 20) ** 2
            # raise ValueError("Kappa for wall on PCV-Z is not given for MetaD in parameter file %s" % (parFile))

        # kappa for targeted MD
        match = re.search("kTMD=.*\n", pars)
        if match is not None:
            self.kTMD = int(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            print("Kappa of targeted MD (kTMD) for path reparameterization is not given in \
                parameter file %s" % (parFile))

        # length of targeted MD 
        # default length of targeted MD
        self.lenTMD = 10
        match = re.search("lenTMD=.*\n", pars)
        if match is not None:
            self.lenTMD = float(re.split('=', match.group(0).rstrip('\n'))[1])
        else:
            print("Length of targeted MD (lenTMD) for path reparameterization is not given in \
                parameter file %s" % (parFile))

        # for storing the directories of meta trajecories of last iteration
        self.lastMeta = None
        self.lastSamples = None


    # ==================================================================================================================
    #                                    Prepare directories & files for MetaD
    #                   1. make directories
    #                   2. store node.pdb for sampling under each directory
    #                   3. specify the MetaD length by self.lenSample / path.n_nodes
    # ==================================================================================================================
    def us_dirs(self, path, dirMeta):
        # input dirMeta is the directory under which, the MetaD sampling and analysis will be performed
        # make sure the path is not empty for MetaD sampling
        if path is None:
            raise ValueError("Path '%s' is empty, can not be sampled" % path.pathName)
        # list to record directories for running
        dirRUNs = []
        for n in tqdm(range(path.n_nodes), desc="meta_dirs", unit="node"):
            dirNode = 'node' + digits(n)
            longDirNode = self.dirRoot + '/' + dirMeta + '/' + dirNode
            if not os.path.exists(longDirNode):
                try:
                    os.makedirs(longDirNode)
                except OSError as error:
                    if error.errno != errno.EEXIST:
                        raise
            nd = path.nodes.slice(n)
            dirRUNs.append(dirNode)
            nodeFile = longDirNode + '/' + self.nodeName
            nd.save(nodeFile)
        # deciding length of each meta using the total amount of sampling
        self.lenMetaD = self.lenSample / path.n_nodes
        return dirRUNs

    # ==================================================================================================================
    #                                   Prepare plumed files for umbrella sampling
    #                       1. plumed input file
    #                       2. path pdb file for PCV definition in plumed2 format
    #                       NOTE: engine-specific running files is implemented in prepSampling()
    # ==================================================================================================================
    def fill_gaps(self, s0_list, delta=0.5):
        # s0_list: [(node_index, s0_value), ...]
        s0_list = sorted(s0_list, key=lambda x: x[1])
        new_points = []
        for (i1, s1), (i2, s2) in zip(s0_list[:-1], s0_list[1:]):
            gap = s2 - s1
            if gap > delta:
                n_insert = int(np.floor(gap / delta))
                for k in range(1, n_insert+1):
                    s_new = s1 + k*delta
                    # 由最近的 node 负责
                    nearest = i1 if abs(s_new - s1) < abs(s2 - s_new) else i2
                    new_points.append((nearest, s_new))
        return new_points

    def umbrella_setup(self, p_bak, dirMeta, dirRUNs):
        if not os.path.exists(dirMeta):
            os.makedirs(dirMeta)

        p = self.refPath
        all_s0 = []

        # === 第一步：收集所有节点的 s0 ===
        for i in tqdm(range(len(dirRUNs)), desc="umbrella_setup (rank {})".format(rank), unit="node"):
            runDir = self.dirRoot + '/' + dirMeta + '/' + dirRUNs[i]
            self.prepSampling(runDir + '/' + self.nodeName, runDir)

            p.exportPCV(dirMeta + '/' + dirRUNs[i])
            p.pcv(dirMeta + '/' + dirRUNs[i])

            node = md.load(runDir + '/' + self.nodeName, top=self.topFile)
            s0, z0 = p.pcv(dirMeta + '/' + dirRUNs[i], node)
            all_s0.append((i, s0))

            # === 写原始节点的 plumed 文件 ===
            pluInput = dirMeta + '/' + dirRUNs[i] + '/' + self.pluName
            with open(pluInput, 'w+') as f:
                atoms = ','.join(str(a+1) for a in self.pcvInd.atomSlice)
                print("WHOLEMOLECULES STRIDE=1 ENTITY0=%s" % atoms, file=f)
                print("p1: PATHMSD REFERENCE=%s LAMBDA=%f NEIGH_STRIDE=4 NEIGH_SIZE=8" %
                      (p.pathName + '_plu.pdb', p.lamda), file=f)
                print("UPPER_WALLS ARG=p1.zzz AT=%f KAPPA=%f EXP=2 EPS=1 OFFSET=0 LABEL=zwall" %
                      (self.zw, self.zwK), file=f)
                print("RESTRAINT ARG=p1.sss KAPPA=%f AT=%f LABEL=res" %
                      (self.kappa, s0), file=f)
                print("PRINT ARG=p1.sss,p1.zzz,res.bias,zwall.bias STRIDE=%d FILE=%s FMT=%%8.16f" %
                      (self.freqTRJ, self.cvOut), file=f)

        # === 第二步：检查间隙并补点 ===
        new_points = self.fill_gaps(all_s0, delta=0.5)

        # === 第三步：为补点生成新目录和 plumed 文件 ===
        import shutil
        for nearest, s_new in new_points:
            # 原始节点目录
            srcDir = self.dirRoot + '/' + dirMeta + '/' + dirRUNs[nearest]
            # 新节点目录名（相对）
            newDirName = dirRUNs[nearest] + "_extra_%d" % int(s_new * 100)
            # 新节点目录（绝对路径）
            newDir = self.dirRoot + '/' + dirMeta + '/' + newDirName

            if not os.path.exists(newDir):
                shutil.copytree(srcDir, newDir)

                # 修改 plumed 文件中的 AT
                pluInput = newDir + '/' + self.pluName
                with open(pluInput, 'r') as f:
                    lines = f.readlines()
                with open(pluInput, 'w') as f:
                    for line in lines:
                        if line.startswith("RESTRAINT ARG=p1.sss"):
                            f.write("RESTRAINT ARG=p1.sss KAPPA=%f AT=%f LABEL=res\n" %
                                    (self.kappa, s_new))
                        else:
                            f.write(line)

                # 把新目录名加入 dirRUNs，方便 umbrella_sample 识别
                dirRUNs.append(newDirName)

        # 返回更新后的 dirRUNs（可选，如果你希望在外部直接用）
        return dirRUNs

    def prepSampling(self, node, dire):
        if self.engine == 'GROMACS':
            logfile = os.path.join(dire, 'gromacs.setup.log')
            with _redirect_fds_to_file(logfile):
                gromacs.setup.MD(dire, mdp=self.dirPar + '/' + self.groMDP, mainselection=None, struct=node, \
                                 top=self.dirPar + '/' + self.groTOP, deffnm=self.runName, runtime=self.lenMetaD, \
                                 dt=self.timeStep, maxwarn=50)
        else:
            raise ValueError("MD engines other than GROMACS are not support yet")

    # ==================================================================================================================
    #                                       perform the actual umbrella sampling
    # ==================================================================================================================
    def umbrella_sample(self, dirMeta, dirRUNs):
        totTRJ = len(dirRUNs)  # the total number of trajectories to run
        # print("%d trajectories to sample in total." % totTRJ)
        if self.mode == 'serial':
            for i in tqdm(range(totTRJ), desc="umbrella_sample", unit="job"):
                runDir = self.dirRoot + '/' + dirMeta + '/' + dirRUNs[i]
                runUs(dire=runDir, engine=self.engine, runName=self.runName, pluName=self.pluName, cvOut=self.cvOut, trjName=self.trjName)
        elif self.mode == 'parallel':
            # record = []
            # lock = multiprocessing.Lock()
            # runDir_list = []
            # cpus = multiprocessing.cpu_count()
            # pool = multiprocessing.Pool(processes=cpus)
            # for i in range(totTRJ):
            #     runDir = self.dirRoot + '/' + dirMeta + '/' + dirRUNs[i]
            #     pool.apply_async(runMeta, args=(runDir, self.engine, self.runName, self.pluName, self.cvOut, self.trjName))
            # pool.close()
            # pool.join()
            N_jobs = totTRJ
            # compute job ids for this rank to avoid overlapping work
            my_tids = list(range(rank, N_jobs, size))
            for tid in tqdm(my_tids, desc="umbrella_sample rank{}".format(rank), unit="job"):
                runDir = self.dirRoot + '/' + dirMeta + '/' + dirRUNs[tid]
                _tqdm.write("+++DEBUG+++ runMeta {} of {} , size {} , rank {}".format(tid, N_jobs, size, rank))
                runUs(dire=runDir, engine=self.engine, runName=self.runName, pluName=self.pluName,
                        cvOut=self.cvOut, trjName=self.trjName)
        elif self.mode == 'qjobs':
            raise ValueError("qjobs version to be implemented")

@contextmanager
def _redirect_fds_to_file(path):
    """Context manager to redirect file descriptors 1 and 2 (stdout/stderr)
    to the given file path. Useful to capture subprocess output from gmx.
    """
    # ensure directory exists
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError:
            pass
    f = open(path, 'a')
    # flush Python-level buffers
    sys.stdout.flush()
    sys.stderr.flush()
    # duplicate fds
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        yield
    finally:
        # flush and restore
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        f.close()

