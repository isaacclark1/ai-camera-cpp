#ifndef THREADPOOL_HPP // Header guard to prevent multiple inclusions
#define THREADPOOL_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/**
 * ThreadPool manages a group of worker threads to execute tasks concurrently.
 * It maintains a thread-safe queue of tasks and uses mutexes and condition variables
 * to coordinate task scheduling and execution. Worker threads continously wait for new tasks
 * and process them until the ThreadPool is instructed to stop (during destruction).
 */
class ThreadPool
{
  private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

  public:
    // Default constructor - creates an empty ThreadPool
    ThreadPool() : stop(false) {}

    // Constructor that creates and starts worker threads
    ThreadPool(size_t numThreads) : stop(false)
    {
      setupThreads(numThreads);
    }

    // Function to create worker threads and store them in the 'workers' vector
    void setupThreads(size_t numThreads)
    {
      for (size_t i = 0; i < numThreads; ++i) {
        // Create a new thread and store it in the vector
        workers.emplace_back([this] { // Lambda function defines worker behaviour
          while (true) { // Infinite loop (until stop flag is set)
            std::function<void()> task; // Task to be executed by thread

            {
              // Lock the task queue to safely access shared resources
              std::unique_lock<std::mutex> lock(this->queueMutex);

              // Wait until there's a task to process or the pool is stopped
              this->condition.wait(lock, [this] {
                return this->stop || !this->tasks.empty();
              });

              // If stop flag is set and no more tasks, exit the loop
              if (this->stop && this->tasks.empty())
                return;

              // Get the next task from the queue
              // This code moves the task into the task variable rather than copying it (which is more expensive than simply moving it)
              // After this tasks.front() is an empty object
              task = std::move(this->tasks.front());
              this->tasks.pop(); // Remove empty task from the front of the queue
            } // Lock is released here

            // Execute the task outside the lock
            task();
          }
        });
      }
    }

    // Function to add a new task to the queue (Thread-safe)
    template <class F>
    // enqueue can accept either an lvalue (locator-value) or an rvalue (right-hand value)
    void enqueue(F &&f)
    {
      {
        // Lock the queue to safely add a task
        std::unique_lock<std::mutex> lock(queueMutex);

        // Store the function as a task in the queue
        // 'forward' ensures perfect forwarding avoiding unnecessary deep copies of f
        tasks.emplace(std::forward<F>(f));
      } // Lock is released here

      // Notify one waiting worker thread that a new task is available
      condition.notify_one();
    }

    // Destructor: Cleans up the ThreadPool and stops all worker threads
    ~ThreadPool()
    {
      {
        // Lock the queue before modifying stop flag
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true; // Set stop flag to true
      }
      
      // Wake up all waiting threads to let them finish
      condition.notify_all();

      // Join all worker threads (wait for them to finish execution)
      for (std::thread &worker : workers) {
        worker.join();
      }
    }
};

#endif // THREADPOOL_HPP