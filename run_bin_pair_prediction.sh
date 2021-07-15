INPUT_PAIR_FILE=$1
MODEL_CKPT_DIR=$2
DEVICE=$3

OUTPUT_DIR=pred_tmp
mkdir -p $OUTPUT_DIR
rm -rf $OUTPUT_DIR/*

echo "Start preparing input pairs"
python prepare_prediction_file.py \
  --input_pairs $INPUT_PAIR_FILE \
  --entity_dict $MODEL_CKPT_DIR/entities.dict \
  --relation_dict $MODEL_CKPT_DIR/relation.dict \
  --output_dir $OUTPUT_DIR/

TOP_K=`wc -l < $OUTPUT_DIR/head.list`

echo "Start prediction for input pairs"
CUDA_VISIBLE_DEVICES=$DEVICE DGLBACKEND=pytorch dglke_predict \
  --model_path $MODEL_CKPT_DIR \
  --format 'h_r_t' \
  --data_files $OUTPUT_DIR/head.list $OUTPUT_DIR/rel.list $OUTPUT_DIR/tail.list \
  --exec_mode 'triplet_wise' \
  --gpu $DEVICE \
  --topK $TOP_K \
  --output $OUTPUT_DIR/result.tsv

echo "Start aggregating prediction results"
python aggregate_prediction_result.py \
  --result $OUTPUT_DIR/result.tsv \
  --entity_dict $MODEL_CKPT_DIR/entities.dict \
  --relation_dict $MODEL_CKPT_DIR/relation.dict \
  --output agg_result.tsv
