## Introduction

This repository provides a Python API for NCCL to send and receive PyTorch tensors.

## Compile
#### compile comm
```bash
cd communication/comm
export LIBRARY_PATH=/path/to/nccl:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/path/to/Python.h:$CPLUS_INCLUDE_PATH
make
```
## End-to-end Workflow

Run the following two commands on the same server with at least 2 GPUs at the same time:
```bash
python test_nccl.py --rank 0
python test_nccl.py --rank 1
```