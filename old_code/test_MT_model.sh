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
#SOURCE_LANG=es
#SOURCE_LANG=fr
SOURCE_LANG=it
# default data: test data
#TEST_DATA_DIR=data/MT/data_"$SOURCE_LANG"/test_data
# default data: original model
MODEL_DIR=finetune_translate_mbart_lang="$SOURCE_LANG"/checkpoint-54000/
#OUT_DIR=data/MT/data_"$SOURCE_LANG"
# default data: fine-tuned model
#MODEL_DIR=finetune_translate_mbart_lang="$SOURCE_LANG"/finetune/checkpoint-1500/
#OUT_DIR=data/MT/data_"$SOURCE_LANG"/translation_data_type=relationship_finetune/
# diff/same gender data
TEST_DATA_DIR=data/MT/translation_data_type=relationship_lang="$SOURCE_LANG"/data_"$SOURCE_LANG"/test_data/
#OUT_DIR=data/MT/translation_data_type=relationship_lang="$SOURCE_LANG"/
OUT_DIR=data/MT/translation_data_type=relationship_lang="$SOURCE_LANG"/finetune/
# offline transformers on server
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python test_MT_model.py $MODEL_DIR $OUT_DIR $TEST_DATA_DIR
