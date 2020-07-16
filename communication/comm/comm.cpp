#include "comm.h"

Comm::Comm(int nrank, int rank, int device) {
  this->rank_ = rank;
  this->nrank_ = nrank;
  this->device_ = device;
  CUDACHECK(cudaSetDevice(device));
  CUDACHECK(cudaStreamCreate(&this->stream));
}

Comm::~Comm() {
  return;
}

const char* Comm::getUniqueId() {
  NCCLCHECK(ncclGetUniqueId(&this->commid));
  return this->commid.internal;
}

void Comm::setUniqueId(const char* buffer) {
  memcpy(this->commid.internal, buffer, 128);
}

void Comm::commInitRank() {
  NCCLCHECK(ncclCommInitRank(&this->comm, this->nrank_, this->commid, this->rank_));
}

void Comm::syncStream() {
  CUDACHECK(cudaStreamSynchronize(this->stream));
}

void Comm::send(void* data, ssize_t N, ncclDataType_t dtype, int peer) {
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(data, N, dtype, peer, this->comm, this->stream));
  NCCLCHECK(ncclGroupEnd());
}

void Comm::recv(void* data, ssize_t N, ncclDataType_t dtype, int peer) {
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclRecv(data, N, dtype, peer, this->comm, this->stream));
  NCCLCHECK(ncclGroupEnd());
}

void Comm::commDestroy() {
  CUDACHECK(cudaStreamDestroy(this->stream));
  NCCLCHECK(ncclCommDestroy(this->comm));
}