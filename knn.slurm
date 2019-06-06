#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J 3d-siamese-knn
#SBATCH -o log/3d-siamese-knn-%j.out
#SBATCH -e log/3d-siamese-knn-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
source /etc/profile.d/modules-basis.sh 
fi

module purge
source ~/activation

# specify training folder for containing training and validation links
TRAINING_FOLDER=training-list/11
EPOCHS=0
STEPS=0

KNN_K=1
SIGMA=3.0
THRESHOLD=15.0


# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-e  $EPOCHS     \
-s  $STEPS        \
-aug         \
-knn        \
-k  $KNN_K        \
-si $SIGMA       \
-l $LAMBDA    \
-th $THRESHOLD      \
-V "$TRAINING_FOLDER/visual" \
-L "$TRAINING_FOLDER/default.model"



