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
#SOURCE_LANG=es
#SOURCE_LANG=fr
SOURCE_LANG=it
# "standard" data
#OUT_DIR=data/MT
SAMPLE_SIZE=100000
#DATASET='europarl_bilingual'
# custom data
OUT_DIR=data/MT/translation_data_type=relationship_lang=$SOURCE_LANG/
DATASET=data/MT/translation_data_type=relationship_lang=$SOURCE_LANG/
MODEL_TYPE='mbart'
# pretrained model
MODEL_DIR=finetune_translate_mbart_lang=$SOURCE_LANG/checkpoint-54000/

# disable internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python train_MT_model.py $OUT_DIR --source_lang $SOURCE_LANG --dataset $DATASET --model_type $MODEL_TYPE
#python train_MT_model.py $OUT_DIR --source_lang $SOURCE_LANG --dataset $DATASET --model_type $MODEL_TYPE --sample_size $SAMPLE_SIZE
# pretrained model
python train_MT_model.py $OUT_DIR --source_lang $SOURCE_LANG --dataset $DATASET --model_type $MODEL_TYPE --model_dir $MODEL_DIR