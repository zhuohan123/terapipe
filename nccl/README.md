# NCCL configuration

First, download NCCL 2.7.8 for CUDA 11.0, O/S agnostic local installer: https://developer.nvidia.com/nccl/nccl-download.

```bash
# Change CUDA runtime to 11.0
sudo ln -sfn /usr/local/cuda-11.0 /usr/local/cuda
NCCL_VERSION=nccl_2.7.8-1+cuda11.0_x86_64

tar xvf $NCCL_VERSION.txz
# to avoid issues caused by symlinks, we install headers and libs separately.
sudo cp -rf $NCCL_VERSION/include/* /usr/local/cuda/include/
sudo cp -rf $NCCL_VERSION/lib /usr/local/cuda
rm -r $NCCL_VERSION
```

# Build NCCL python wrapper

Run `./build.sh`. Use `round_trip_test.py` for testing.
