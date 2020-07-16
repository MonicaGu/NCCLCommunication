## Introduction

This repository provides a Python API for NCCL.

## Compile
#### compile comm
```bash
cd communication/comm
export LIBRARY_PATH=/path/to/nccl:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/path/to/Python.h:$CPLUS_INCLUDE_PATH
make
```
