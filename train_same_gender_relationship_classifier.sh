#!/bin/bash
#SBATCH --job-name=train_same_gender_relationship_classifier
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=20g
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --account=mihalcea0

OUT_DIR=relationship_type_classifier
MODEL_NAME=facebook/mbart-large-50
LANG='it'

# queue server
# disable internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python $OUT_DIR --model_name $MODEL_NAME --lang $LANG