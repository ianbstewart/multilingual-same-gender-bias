#!/bin/bash
#SBATCH --job-name=train_MT_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=30g
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --account=mihalcea0
OUT_DIR=data/MT
SOURCE_LANG=es
DATASET='europarl_bilingual'
MODEL_TYPE='mbart'

# disable internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python train_MT_model.py $OUT_DIR --source_lang $SOURCE_LANG --dataset $DATASET --model_type $MODEL_TYPE
