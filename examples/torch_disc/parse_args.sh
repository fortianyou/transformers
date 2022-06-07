#!/bin/bash
function help() {
  echo "torch_disc_train.sh [--nvprof|--torch-profiler] [--ltc|--disc]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvprof)
      ENABLE_NVPROF="ON"
      shift
      if [ "$ENABLE_PROFILER" == "ON" ];then
        echo "--nvprof is mutually exclusive with --torch-profiler"
        help
        exit
      fi
      ;;
    --torch-profiler)
      ENABLE_TORCH_PROFILER="ON"
      if [ "$ENABLE_NVPROF" == "ON" ];then
        echo "--torch-profiler is mutually exclusive with --nvprof"
        help
        exit
      fi 
      shift
      ;;
    --ltc)
      ENABLE_LTC="ON"
      shift
      ;;
    --xla)
      ENABLE_XLA="ON"
      shift
      ;;
    --disc)
      ENABLE_DISC="ON"
      shift
      ;;
    --no-grad-norm)
      DISABLE_GRAD_NORM="ON"
      shift
      ;;
    -h)
      help
      exit
      ;;
    *)
      return
  esac
done