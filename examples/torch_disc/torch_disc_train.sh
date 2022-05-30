
# wget -c -e use_proxy=on -e \
#   https_proxy=http://blade_disc:r299eNZXVTUpuyg7@8.217.91.10:12357  \
#   https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin \
#   -O bert-base-cased.bin

rm -rf dump && mkdir -p dump
#export LTC_IR_DEBUG=1
export LTC_SAVE_TENSORS_FILE=dump/ltc_ir.txt
export LTC_SAVE_TENSORS_FMT=backend

#export APPLY_PENDING_GRAPH_MATERIALIZE=on
#export LTC_EMIT_STEPLOG=1
#train_script="pytorch/xla_spawn.py --num_cores 1"
#train_script="-m torch.distributed.launch --nproc_per_node 1"
#export TORCH_DISC_LTC_DISABLE_DISC=true
export TORCH_DISC_LTC_DUMP_TS_GRAPH_DATA=true
export PYTHONPATH=$(pwd)/../../src
export CUDA_VISIBLE_DEVICES=1
export LTC_DISC_CUDA=1
#export BENCHMARK_ENABLE_NVPROF=1
#export BENCHMARK_DISABLE_TORCH_DISC=1
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
#nvprof --profile-from-start off python $train_script \
python $train_script \
  ../pytorch/text-classification/run_glue.py \
  --config_name bert-base-cased \
  --tokenizer_name bert-base-cased \
  --model_name_or_path bert-base-cased.bin \
  --dataset_name imdb  \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --logging_steps 3000 \
  --save_steps 3000 \
  --max_grad_norm 1.0 \
  --output_dir /tmp/imdb/ 2>&1 | tee train.torch.log
#  --output_dir /tmp/imdb/ 2>&1 | tee train.ltc.log

#  --max_steps 4 \
