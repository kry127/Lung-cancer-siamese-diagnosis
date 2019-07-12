#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J 3d-siamese-training 
#SBATCH -o log/3d-siamese-training-%j.out
#SBATCH -e log/3d-siamese-training-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
source /etc/profile.d/modules-basis.sh 
fi

module purge
source ~/activation

# specify training folder for containing training and validation links
TRAINING_FOLDER=training-list/test2_ord
BATCHSIZE=16
EPOCHS=3
STEPS=500
LEARNING_RATE=0.001
MARGIN=42
LAMBDA1=0.1
LAMBDA2=3
LAMBDA3=1
LAMBDA4=1
LAMBDA5=1
LAMBDA6=1
LAMBDAS="$LAMBDA1;$LAMBDA2;$LAMBDA3;$LAMBDA4;$LAMBDA5;$LAMBDA6"

# create data and algorithm digest
echo "siam resnet bs=$BATCHSIZE e=$EPOCHS s=$STEPS lr=$LEARNING_RATE l=$LAMBDAS m=$MARGIN\
" >> ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-bs $BATCHSIZE      \
-e  $EPOCHS     \
-s  $STEPS        \
-sb           \
-lr $LEARNING_RATE \
-aug         \
-vis        \
-k  1        \
-si 3       \
-th 15.0       \
-m  $MARGIN     \
-l1 $LAMBDA1    \
-l2 $LAMBDA2    \
-l3 $LAMBDA3    \
-l4 $LAMBDA4    \
-l5 $LAMBDA5    \
-l6 $LAMBDA6    \
-V "$TRAINING_FOLDER/visual" \
-L "$TRAINING_FOLDER/default.model" \
-S "$TRAINING_FOLDER/default.model"



# pipeline info:
# https://hpc.nih.gov/docs/job_dependencies.html