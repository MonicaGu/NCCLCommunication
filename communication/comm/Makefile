INC	:= $(shell pkg-config python3 --cflags)
LIB	:= -lnccl
all:
	nvcc comm.cpp comm_py_interface.cpp \
	-o nccl.so \
	-shared -Xcompiler -fPIC \
	$(LIB) $(INC) -std=c++11