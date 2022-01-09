#!/bin/bash
#SBATCH --job-name=train_question_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=30g
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --account=mihalcea0
OUT_DIR=data/MT
SOURCE_LANG=es
DATASET='europarl-bilingual'
MODEL_TYPE='mbart'

python train_MT_model.py $OUT_DIR --source_lang $SOURCE_LANG --dataset $DATASET --model_type $MODEL_TYPE