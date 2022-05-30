
# wget -c -e use_proxy=on -e \
#   https_proxy=http://blade_disc:r299eNZXVTUpuyg7@8.217.91.10:12357  \
#   https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin \
#   -O bert-base-cased.bin

function set_default_env() {
  export CUDA_VISIBLE_DEVICES=1
  # use the inner transformers Python package to easy to debug
  export PYTHONPATH=$(pwd)/../../src
  # this is used for nvprof
  export LD_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
  # disable ltc and disc accelerator by default
  export BENCHMARK_ENABLE_TORCH_LTC=OFF
  export BENCHMARK_ENABLE_TORCH_DISC=OFF

  # LTC debug envs
  rm -rf dump && mkdir -p dump
  export LTC_SAVE_TENSORS_FILE=dump/ltc_ir.txt
  export LTC_SAVE_TENSORS_FMT=backend
}

source parse_args.sh
set_default_env

if [ "$ENABLE_DISC" == "ON" ]; then
  export LTC_DISC_CUDA=1
  export BENCHMARK_ENABLE_TORCH_LTC=ON
  export BENCHMARK_ENABLE_TORCH_DISC=ON
fi

if [ "$ENABLE_LTC" == "ON" ]; then
  export LTC_TS_CUDA=1
  export BENCHMARK_ENABLE_TORCH_LTC=ON
fi

entry_cmd="python ../pytorch/text-classification/run_glue.py \
  --config_name bert-base-cased \
  --tokenizer_name bert-base-cased \
  --model_name_or_path bert-base-cased.bin \
  --dataset_name imdb  \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --logging_steps 3000 \
  --save_steps 3000 "

if [ "$ENABLE_NVPROF" == "ON" ]; then
  export BENCHMARK_ENABLE_NVPROF=ON
  entry_cmd="nvprof --profile-from-start off $entry_cmd \
  --max_steps=110 "
fi

if [ "$ENABLE_TORCH_PROFILER" == "ON" ]; then
  export BENCHMARK_ENABLE_TORCH_PROFILER=ON
  entry_cmd="$entry_cmd --max_steps 10"
fi

if [ "$DISABLE_GRAD_NORM" == "ON" ]; then
  entry_cmd="$entry_cmd --max_grad_norm 0"
fi

set -ex
$entry_cmd --output_dir /tmp/imdb 2>&1 | tee train.torch.log
