#pragma once

#include <iostream>
#include <deque>

/**
 * Bounded queue that holds frames from the camera.
 */
template <typename T>
class FrameQueue {
  private:
    std::deque<T> queue;
    size_t max_size;

  public:
    /**
     * Construct a frame queue.
     * @param capacity The size of the queue
     */
    explicit FrameQueue(size_t capacity);

    /**
     * Push an element to the back of the queue.
     * @param element The element to add to the back of the queue
     */
    void enqueue(const T &value);

    /**
     * Remove an element from the front of the queue.
     */
    void dequeue();

    /**
     * Get the front element from the queue.
     */
    T& front();
    const T& front() const;

    /**
     * Get the size of the queue.
     */
    size_t size() const;

    /**
     * Check if the queue is empty.
     */
    bool isEmpty() const;

    /**
     * Check if the queue is full.
     */
    bool isFull() const;

    /**
     * Overloaded stream insertion operator for logging queue items.
     */
    template <typename U>
    friend std::ostream& operator<<(std::ostream &os, const FrameQueue<U> &frame_queue);
};

// ==========================
// 🔹 Function Definitions 🔹
// ==========================

template <typename T>
FrameQueue<T>::FrameQueue(size_t capacity) : max_size(capacity) {};

template <typename T>
void FrameQueue<T>::enqueue(const T &value)
{
  if (queue.size() >= max_size) {
    queue.pop_front();
  }

  queue.push_back(value);
}

template <typename T>
void FrameQueue<T>::dequeue()
{
  if (queue.empty()) {
    return;
  }

  return queue.pop_front();
}

template <typename T>
T& FrameQueue<T>::front()
{
  if (queue.empty()) {
    throw std::runtime_error("FrameQueue is empty");
  }

  return queue.front();
}

template <typename T>
const T& FrameQueue<T>::front() const
{
  if (queue.empty()) {
    throw std::runtime_error("FrameQueue is empty");
  }

  return queue.front();
}

template <typename T>
size_t FrameQueue<T>::size() const
{
  return queue.size();
}

template <typename T>
bool FrameQueue<T>::isEmpty() const
{
  return queue.empty();
}

template <typename T>
bool FrameQueue<T>::isFull() const
{
  return queue.size() == max_size;
}

template <typename T>
std::ostream& operator<<(std::ostream &os, const FrameQueue<T> &frame_queue)
{
  os << "FrameQueue:\n";
  
  for (const T &element : frame_queue.queue) {
    os << element << "\n";
  }

  os << std::endl;

  return os;
}