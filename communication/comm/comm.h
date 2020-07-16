#pragma once
#include "common.h"

class Comm{
  private:
    ncclUniqueId commid;
    ncclComm_t comm;
    cudaStream_t stream;
    int rank_, nrank_, device_;
  public:
    Comm(int nrank, int rank, int device);
    ~Comm();
    const char *getUniqueId();
    void setUniqueId(const char* buffer);
    void commInitRank();
    void syncStream();
    void commDestroy();
    void send(void* data, ssize_t N, ncclDataType_t dtype, int peer);
    void recv(void* data, ssize_t N, ncclDataType_t dtype, int peer);
};