#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J 3d-siamese-hashes 
#SBATCH -o log/3d-siamese-hashes-%j.out
#SBATCH -e log/3d-siamese-hashes-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
    source /etc/profile.d/modules-basis.sh 
fi

module purge
source ~/activation


TRAINING_FOLDER=training-list/test2_vgg

MODEL_PATH="./$TRAINING_FOLDER/default.model"
HASH_FILE="./$TRAINING_FOLDER/hash_$(date "+%d.%m.%y_%H.%M").npy"

srun python3 compute_hashes.py \
-L $MODEL_PATH \
-H $HASH_FILE \
-F $TRAINING_FOLDER