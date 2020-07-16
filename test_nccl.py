import sys
sys.path.append("..")
import torch
import threading
import time
from communication.comm import CommunicationHandler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, help='rank id')
args = parser.parse_args()
addr = "127.0.0.1"
port = 4588

def Worker(i):
    c = CommunicationHandler(addr, port, 2, i, i)
    if i == 1:
        time.sleep(10)
        for j in range(5):
            a = torch.rand(64, 1024).cuda(device=i)
            now = time.time()
            print("sending")
            c.send_to(0, a)
        print("sent", time.time() - now, a)
    else:
        for j in range(5):
            a = torch.rand(64, 1024).cuda(device=i)
            now = time.time()
            print("receiving")
            c.recv_from(1, a)
        print("recvd", time.time() - now, a)
    c.close()


t0 = threading.Thread(target=Worker, args=[args.rank])
t0.start()
