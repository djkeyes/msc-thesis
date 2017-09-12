#!/bin/bash
#
# job to perform descriptor training
# 
#SBATCH --job-name="greatcourt"
#SBATCH --output=./log/greatcourt.out
#SBATCH --error=./log/greatcourt.err
#
#
#SBATCH --partition=plongx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#
# make the submitter happy and print what is been done

tstamp="`date '+%D %T'`"
hn="`hostname -f`"
jobid=${SLURM_JOB_ID}
jobname=${SLURM_JOB_NAME}
if [ -z "${jobid}" ] ; then
  echo "ERROR: SLURM_JOBID undefined, are you running this script directly ?"
  exit 1
fi

printf "%s: starting %s(%s) on host %s\n" "${tstamp}" "${jobname}" "${jobid}" "${hn}"
echo "**"
echo "** SLURM_CLUSTER_NAME="$SLURM_CLUSTER_NAME
echo "** SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "** SLURM_JOB_ID="$SLURM_JOBID
echo "** SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "** SLURM_NUM_NODES"=$SLURM_NUM_NODES
echo "** SLURMTMPDIR="$SLURMTMPDIR
echo "** working directory = "$SLURM_SUBMIT_DIR
echo
echo "loading modules:"
module load cuda cudnn openmpi tools/libzip gcc
module list
echo "setting paths"
# I think only the runtime paths are necessary. Unless caffe does some dynamic compilation?
export CPATH=/home/dkeyes/local/include:$CPATH
export LIBRARY_PATH=/home/dkeyes/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/dkeyes/local/lib:$LD_LIBRARY_PATH
export PATH=/home/dkeyes/local/bin:$PATH
export PYTHONPATH=$PYTHON_PATH:/home/dkeyes/local/python:/home/dkeyes/git/msc-thesis/python/DescriptorLearning

echo "setup limits"
ulimit -s unlimited
ulimit -a
dt="`date '+%s'`"
srun /home/dkeyes/local/bin/caffe train -gpu all -solver /home/dkeyes/git/msc-thesis/python/DescriptorLearning/greatcourt_solver.prototxt -weights /import/euryale/projects/dkeyes/experiments/denseCorrespondence/greatcourt_iter_4500.caffemodel
stat="$?"
dt=$(( `date '+%s'` - ${dt} ))
echo "job finished, status=$stat, duration=$dt second(s)"
echo

