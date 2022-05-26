# Running GLUE Example with Torch Disc Accelerator

1. build Torch DISC accelerator from source according [this guide](https://github.com/alibaba/BladeDISC/blob/main/docs/build_from_source.md#building-bladedisc-for-pytorch-users).

1. (option)download the pre-trained model from AliCloud OSS if you're in the
backend of WFG.

1. run the following bash command:

    ``` bash
    python -m pip install -r requirements.txt
    bash torch_disc_train.sh
    ```
