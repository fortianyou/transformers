
# wget -c -e use_proxy=on -e \
#   https_proxy=http://blade_disc:r299eNZXVTUpuyg7@8.217.91.10:12357  \
#   https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin \
#   -O bert-base-cased.bin

# export LTC_IR_DEBUG=1
# export LTC_SAVE_TENSORS_FILE=dump/ltc_ir.txt
# export LTC_SAVE_TENSORS_FMT=text
# # 
# export LTC_METRICS_FILE=dump/ltc_metrics.txt
# rm -rf $LTC_SAVE_TENSORS_FILE

#export APPLY_PENDING_GRAPH_MATERIALIZE=on
export LTC_TS_CUDA=1

#train_script="pytorch/xla_spawn.py --num_cores 1"
#train_script="-m torch.distributed.launch --nproc_per_node 1"

export CUDA_VISIBLE_DEVICES=1
# export DISABLE_TORCH_LTC=on
python3 $train_script \
  pytorch/text-classification/run_glue.py \
  --config_name bert-base-cased \
  --tokenizer_name bert-base-cased \
  --model_name_or_path bert-base-cased.bin \
  --dataset_name imdb  \
  --do_train \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --output_dir /tmp/imdb/ 2>&1 | tee train.ltc.log
