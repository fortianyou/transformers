# Running GLUE Example with Torch Disc Accelerator

1. build Torch DISC accelerator from source according [this guide](https://github.com/alibaba/BladeDISC/blob/main/docs/build_from_source.md#building-bladedisc-for-pytorch-users).

1. (option)download the pre-trained model from AliCloud OSS if you're in the
backend of WFG.

    ``` bash
    wget http://223502.oss-cn-hangzhou-zmf.aliyuncs.com/temp/bert-base-cased.bin
    ```

1. run the bash script with the options which you needs as the [Command Options](#command-options) section:

    ``` bash
    python -m pip install -r requirements.txt
    bash torch_disc_train.sh
    ```

## Command Options

| option | description|
| -- | -- |
| --nvprof | enable Nvidia Profiler toolkit to profile this benchmark, inject 90~100 iterations only|
| --torch-profiler | enable torch profiler toolkit to profile this benchmark, run 10 iterations only|
| --ltc | enable LTC or not |
| --disc | enable Disc or not |
| --no-grad-norm | disable grad norm or not |
