# coding=utf-8
import socket
import threading
import torch
from .nccl import PyComm

"""
    Our protocal is like
    |...8 bytes...| .....message..... |
    the first 8 bytes (up to 2^64) indicate the following message's size.
"""

def dtype2nccl(dtype):
    """
    typedef enum { ncclInt8    = 0, ncclChar       = 0,
                ncclUint8      = 1,
                ncclInt32      = 2, ncclInt        = 2,
                ncclUint32     = 3,
                ncclInt64      = 4,
                ncclUint64     = 5,
                ncclFloat16    = 6, ncclHalf       = 6,
                ncclFloat32    = 7, ncclFloat      = 7,
                ncclFloat64    = 8, ncclDouble     = 8,
                ncclNumTypes   = 9 } ncclDataType_t
    """
    mapper = {
        torch.int8 : 0,
        torch.uint8 : 1,
        torch.int32 : 2,
        torch.int : 2,
        torch.int64 : 4,
        torch.float16 : 6,
        torch.half : 6,
        torch.float32 : 7,
        torch.float : 7,
        torch.float64 : 8,
        torch.double : 8
    }
    if dtype not in mapper.keys():
        raise Exception("dtype Not supported :", dtype)
    return mapper[dtype]

class CommunicationHandler:
    def __init__(self, masteraddr, masterport, nrank, rank, device=0):
        """
            @param:
                masteraddr, masterport: (IPv4 addr, port)
                rank: an integer, 0<=rank<nrank
                nrank: total number of process
                device: GPU id
            Step:

        """
        assert(type(device) is int)
        assert(type(rank) is int)
        assert(type(nrank) is int)
        assert(0 <= rank and rank < nrank)
        self.rank=rank
        self.nrank=nrank
        self.sockets = [None for i in range(nrank)]
        self._C = PyComm(
            nrank=nrank,
            rank=rank,
            device=device
        )
        if rank == 0:
            uid = self._C.getUniqueId()
            self.sockets[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sockets[0].bind((masteraddr, masterport))
            self.sockets[0].listen(nrank)
            connection_remain = nrank - 1
            while connection_remain > 0:
                sock, addr = self.sockets[0].accept()
                who = self.__recv_from(sock)
                i = int(who)
                assert(self.sockets[i] is None)
                self.sockets[i] = sock
                connection_remain -= 1
            self.sockets[0].close()
        else:
            self.sockets[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__active_connect(self.sockets[0], (masteraddr, masterport))
        if rank == 0:
            for i in range(1, nrank):
                self.__send_to(self.sockets[i], uid)
        else:
            uid = self.__recv_from(self.sockets[0])
            self._C.setUniqueId(uid)

        self._C.commInitRank()

        for sock in self.sockets:
            if sock is not None:
                sock.close()

        print("Connection Built : rank ", rank)
        return

    def send_to(self, peer, tensor):
        """
            @param:
                peer: an integer, 0<=peer<nrank, the target to send to
                tensor: a torch.Tensor object
        """
        addr = tensor.data_ptr()
        num_elm = tensor.shape.numel()
        dtype = dtype2nccl(tensor.dtype)

        self._C.send(addr, num_elm, dtype, peer)
        return

    def __send_to(self, sock, message):
        size = len(message)
        size_msg = size.to_bytes(length=8, byteorder='big',signed=False)
        sock.send(size_msg)
        length = sock.send(message)

    def recv_from(self, peer, tensor):
        """
            @param:
                peer: an integer, 0<=peer<nrank, the target to receive from
                tensor: a torch.Tensor object
        """
        addr = tensor.data_ptr()
        num_elm = tensor.shape.numel()
        dtype = dtype2nccl(tensor.dtype)

        self._C.recv(addr, num_elm, dtype, peer)
        return

    def sync(self):
        self._C.syncStream()

    def __recv_from(self, sock):
        size_msg = sock.recv(8)
        size = int.from_bytes(size_msg, byteorder='big', signed=False)
        size_received = 0
        size_to_receive = size
        while size_to_receive > 0:
            message_received = sock.recv(size_to_receive)
            if size_received == 0:
                message = message_received
            else:
                message += message_received
            size_received += len(message_received)
            size_to_receive -= len(message_received)
            if size_to_receive == 0:
                break
        return message

    def close(self):
        """
            Close all the communication
        """
        self._C.commDestroy()

    def __active_connect(self, sock, addr):
        while True:
            try:
                sock.connect(addr)
                self.__send_to(sock, str(self.rank).encode())
                break
            except ConnectionError as e:
                continue
        return
