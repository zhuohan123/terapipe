#ifndef __CONSUMERPRODUCERQUEUE_H__
#define __CONSUMERPRODUCERQUEUE_H__

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class ConsumerProducerQueue {
  std::condition_variable cond;
  std::mutex mutex;
  std::queue<T> cpq;
  int maxSize;

public:
  ConsumerProducerQueue(int mxsz=10000000) : maxSize(mxsz) {}

  void add(T request) {
    std::unique_lock<std::mutex> lock(mutex);
    cond.wait(lock, [this]() { return !isFull(); });
    cpq.push(request);
    lock.unlock();
    cond.notify_all();
  }

  void consume(T &request) {
    std::unique_lock<std::mutex> lock(mutex);
    cond.wait(lock, [this]() { return !isEmpty(); });
    request = cpq.front();
    cpq.pop();
    lock.unlock();
    cond.notify_all();
  }

  bool isFull() const { return cpq.size() >= maxSize; }

  bool isEmpty() const { return cpq.size() == 0; }

  int length() const { return cpq.size(); }

  void clear() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!isEmpty()) {
      cpq.pop();
    }
    lock.unlock();
    cond.notify_all();
  }
};

#endif