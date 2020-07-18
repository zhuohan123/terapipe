import queue
import torch
import threading

q = queue.Queue()

def _worker1():
  stream = torch.cuda.Stream()
  a = torch.randint(-1000, 1000, (2 ** 30), device='cuda:0')
  s = a.sum().item()
  print(s)
  with torch.cuda.stream(s):
      a = a.to('cuda:1', non_blocking=True)
  q.put((a, stream.record_event()))

def _worker2():
    a, event = q.get()
    torch.cuda.current_stream().wait_event(event)
    print(a.sum().item())

thread_1 = threading.Thread(target=_worker1)
thread_2 = threading.Thread(target=_worker2)

thread_1.start()
thread_2.start()
thread_1.join()
thread_2.join()
