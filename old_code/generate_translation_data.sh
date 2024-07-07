OUT_DIR=data/MT/
DATA_TYPE=relationship
SOURCE_LANGS=("es" "fr" "it")
#SOURCE_LANGS=("it")
MODEL_TYPE='mbart'

python generate_translation_data.py $OUT_DIR --data_type $DATA_TYPE --source_langs "${SOURCE_LANGS[@]}" --model_type $MODEL_TYPE