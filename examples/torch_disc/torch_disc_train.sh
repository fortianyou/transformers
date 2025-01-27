
# wget -c -e use_proxy=on -e \
#   https_proxy=http://blade_disc:r299eNZXVTUpuyg7@8.217.91.10:12357  \
#   https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin \
#   -O bert-base-cased.bin

function set_default_env() {
  #export CUDA_VISIBLE_DEVICES=1
  # use the inner transformers Python package to easy to debug
  export PYTHONPATH=$(pwd)/../../src
  # this is used for nvprof
  export LD_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
  # disable ltc and disc accelerator by default
  export BENCHMARK_ENABLE_TORCH_LTC=OFF
  export BENCHMARK_ENABLE_TORCH_DISC=OFF

  # LTC debug envs
  rm -rf dump && mkdir -p dump
  export PYTORCH_NVFUSER_DISABLE_FALLBACK=1
  #export NVIDIA_TF32_OVERRIDE=0
  export TORCH_DISC_USE_TORCH_MLIR=1
  export PYTORCH_JIT_LOG_LEVEL=">>>disc_compiler:>>graph_fuser:>>>register_disc_class"
  export TORCH_MHLO_OP_WHITE_LIST="aten::slice;aten::reshape;aten::permute;aten::embedding;aten::native_layer_norm;aten::native_dropout;aten::addmm;aten::bmm;aten::_softmax;aten::add;aten::sub;aten::mul;aten::div;aten::expand;aten::gelu;aten::_log_softmax;aten::sum;aten::mm;aten::native_dropout_backward;aten::gelu_backward;aten::_softmax_backward_data;aten::native_layer_norm_backward"
  export DROP_LAST_BATCH=ON
  # enable torch-disc replay toolkit or not 
  export TORCH_DISC_ENABLE_REPLAY=false
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

if [ "$ENABLE_XLA" == "ON" ]; then
  export GPU_NUM_DEVICES=1
  launch_script="../pytorch/xla_spawn.py"
fi

TASK_NAME=cola

entry_cmd="python $launch_script ../pytorch/text-classification/run_glue.py \
  --config_name bert-base-cased \
  --tokenizer_name bert-base-cased \
  --model_name_or_path bert-base-cased.bin \
  --task_name $TASK_NAME \
  --dataset_name imdb  \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 48 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --log_level debug \
  --logging_steps 3000 \
  --save_steps 3000 \
  --logging_nan_inf_filter False "



if [ "$ENABLE_NVPROF" == "ON" ]; then
  export BENCHMARK_ENABLE_NVPROF=ON
  entry_cmd="nsys profile -f true -c cudaProfilerApi $entry_cmd --max_steps=120 "

fi

if [ "$ENABLE_TORCH_PROFILER" == "ON" ]; then
  export BENCHMARK_ENABLE_TORCH_PROFILER=ON
  entry_cmd="$entry_cmd --max_steps 4"
fi

if [ "$DISABLE_GRAD_NORM" == "ON" ]; then
  entry_cmd="$entry_cmd --max_grad_norm 0"
fi

entry_cmd="$entry_cmd $@"

set -ex
$entry_cmd --output_dir /tmp/imdb 2>&1 | tee train.torch.log