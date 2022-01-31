#!/bin/bash
#SBATCH --job-name=test_MT_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=30g
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --account=mihalcea0
#LANG=es
#LANG=fr
LANG=it
MODEL_DIR=finetune_translate_mbart_lang="$LANG"/checkpoint-54000/
OUT_DIR=data/MT/data_"$LANG"
TEST_DATA_DIR="$OUT_DIR"/test_data
# offline transformers on server
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python test_MT_model.py $MODEL_DIR $OUT_DIR $TEST_DATA_DIR
