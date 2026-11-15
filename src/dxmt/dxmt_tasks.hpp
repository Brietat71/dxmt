#pragma once

#include "thread.hpp"
#include "util_win32_compat.h"
#include <algorithm>
#include <atomic>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace dxmt {

template <typename Task> struct task_trait {
  Task run_task(Task task);
  bool get_done(Task task);
  void set_done(Task task);
};

/* neutron: two priorities and a pool that grows with the queued work.
   - submit() is for work a draw waits for (pipelines and their shaders), submit_background() for
     speculative work (shader conversion at creation, primed and compute pipelines). Background tasks
     run when no foreground task is queued, on at most half the workers. promote() moves a background
     task a draw now needs to the front (queued), to the foreground once its dependency is done
     (parked), or raises the priority of its worker (running); a foreground task that waits for a
     background one promotes it.
   - The pool grew only when every worker was running at the moment of a submit: a burst of pipelines
     from one draw loop found the workers still waking up and compiled on 2 to 4 threads, while Metal
     compiles about 16 pipelines in parallel on an 18 core M5 Max (2.1 ms wall per pipeline against
     5.4 ms with 4 threads). Now a worker is added while queued tasks outnumber idle workers, up to the
     number of cores (more threads gave no throughput, only longer latency per pipeline). */
template <typename Task> class task_scheduler {
public:
  void submit(Task task);
  void submit_background(Task task);
  void promote(Task task);
  void shutdown();

  task_scheduler();
  ~task_scheduler();

  uint64_t
  get_running_threads() {
    return running.load(std::memory_order_relaxed);
  }

private:
  void worker_func();
  bool should_grow_locked();
  void add_worker();

  dxmt::mutex worker_mutex_;
  dxmt::condition_variable worker_cond_;
  std::deque<Task> task_queue_;
  std::deque<Task> task_background_queue_;
  std::deque<Task> task_continuation_queue_;

  dxmt::mutex deps_mutex_;
  std::unordered_multimap<Task, Task> task_continuation_;
  std::unordered_set<Task> background_parked_; /* background tasks waiting for a dependency */

  std::vector<dxmt::thread> workers_;

  /* what each worker runs, to raise the priority of a background task a draw now waits for */
  struct worker_slot {
    HANDLE thread = nullptr;
    Task task = {};
    bool background = false;
    bool boosted = false;
  };
  std::vector<worker_slot *> slots_; /* under worker_mutex_ */

  std::atomic_bool destroyed = false;
  std::atomic_uint64_t running = 0;
  uint64_t idle = 0;             /* workers waiting for a task, under worker_mutex_ */
  uint64_t running_background = 0; /* under worker_mutex_ */
  uint64_t threads;
  uint64_t max_threads;
  uint64_t max_background;
};

template <typename Task> task_scheduler<Task>::task_scheduler() {
  max_threads = std::max<uint64_t>(dxmt::thread::hardware_concurrency(), 4);
  max_background = std::max<uint64_t>(max_threads / 2, 1);
  workers_.reserve(max_threads);
  threads = 2;

  for (unsigned i = 0; i < threads; i++) {
    workers_.emplace_back([this]() { worker_func(); });
  }
}

template <typename Task> task_scheduler<Task>::~task_scheduler() { shutdown(); }

template <typename Task>
void
task_scheduler<Task>::shutdown() {
  std::vector<dxmt::thread> workers;
  {
    std::unique_lock<dxmt::mutex> lock(worker_mutex_);
    destroyed.store(true);
    workers.swap(workers_);
  }
  worker_cond_.notify_all();

  for (auto &worker : workers) {
    if (worker.joinable())
      worker.join();
  }

  workers_.clear();
}

template <typename Task>
void
task_scheduler<Task>::worker_func() {
  struct task_trait<Task> task_trait;
  std::vector<std::pair<Task, bool>> continuation_buffer;
  /* neutron: Wine on macOS makes a TIME_CRITICAL thread a fixed priority Mach thread of the highest
     importance. Up to half the cores busy with speculative work at that priority took cores from the
     game thread (its draw loop 2x slower in frames that created shaders); background tasks run at
     BELOW_NORMAL (QoS utility), work a draw waits for keeps TIME_CRITICAL. */
  int priority = THREAD_PRIORITY_TIME_CRITICAL;
  SetThreadPriority(GetCurrentThread(), priority);
  worker_slot slot;
  DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &slot.thread, 0, FALSE,
                  DUPLICATE_SAME_ACCESS);
  {
    std::unique_lock<dxmt::mutex> lock(worker_mutex_);
    slots_.push_back(&slot);
  }
  while (!destroyed.load()) {
    Task task;
    bool background = false;
    bool grow = false;
    {
      std::unique_lock<dxmt::mutex> lock(worker_mutex_);

      auto has_work = [this]() {
        return task_queue_.size() || task_continuation_queue_.size() ||
               (task_background_queue_.size() && running_background < max_background);
      };
      if (!has_work()) {
        idle++;
        worker_cond_.wait(lock, [&]() { return has_work() || destroyed.load(); });
        idle--;
      }

      if (!task_continuation_queue_.empty()) {
        task = task_continuation_queue_.front();
        task_continuation_queue_.pop_front();
      } else if (!task_queue_.empty()) {
        task = task_queue_.front();
        task_queue_.pop_front();
      } else if (!task_background_queue_.empty() && running_background < max_background) {
        task = task_background_queue_.front();
        task_background_queue_.pop_front();
        background = true;
        running_background++;
      } else {
        break;
      }
      grow = should_grow_locked();
      slot.task = task;
      slot.background = background;
      if (slot.boosted) {
        slot.boosted = false;
        priority = THREAD_PRIORITY_TIME_CRITICAL;
      }
    }
    if (grow)
      add_worker();
    if (int want = background ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_TIME_CRITICAL; want != priority) {
      priority = want;
      SetThreadPriority(GetCurrentThread(), priority);
    }
    running.fetch_add(1, std::memory_order_relaxed);
    while (true) {
      Task continuation = task_trait.run_task(task);
      if (continuation == task) {
        {
          std::unique_lock<dxmt::mutex> lock(deps_mutex_);
          auto range = task_continuation_.equal_range(continuation);
          for (auto itr = range.first; itr != range.second; ++itr) {
            continuation_buffer.push_back({itr->second, background_parked_.erase(itr->second) != 0});
          }
          task_continuation_.erase(range.first, range.second);
          task_trait.set_done(continuation);
        }
        if (!continuation_buffer.empty()) {
          std::unique_lock<dxmt::mutex> lock(worker_mutex_);
          for (auto &[task, bg] : continuation_buffer) {
            if (bg)
              task_background_queue_.push_back(task);
            else
              task_continuation_queue_.push_back(task);
          }
        }
        worker_cond_.notify_all();
        continuation_buffer.clear();
      } else {
        {
          std::unique_lock<dxmt::mutex> lock(deps_mutex_);
          // spurious dependency
          if (task_trait.get_done(continuation)) {
            continue;
          }
          task_continuation_.insert({continuation, task});
          if (background)
            background_parked_.insert(task);
        }
        /* neutron: the dependency may still wait in the background queue */
        if (!background)
          promote(continuation);
      }
      break;
    }
    running.fetch_sub(1, std::memory_order_relaxed);
    if (background) {
      std::unique_lock<dxmt::mutex> lock(worker_mutex_);
      running_background--;
      slot.background = false;
      if (!task_background_queue_.empty())
        worker_cond_.notify_one();
    }
  }
  {
    std::unique_lock<dxmt::mutex> lock(worker_mutex_);
    slots_.erase(std::find(slots_.begin(), slots_.end(), &slot));
  }
  if (slot.thread)
    CloseHandle(slot.thread);
};

/* adds one worker while queued tasks outnumber the idle workers. Creating a thread takes 1 to 2 ms
   in Wine, so the game thread adds at most one per submit, and only for work a draw waits for; the
   workers add the rest when they take a task. */
template <typename Task>
bool
task_scheduler<Task>::should_grow_locked() {
  if (destroyed.load() || threads >= max_threads)
    return false;
  uint64_t queued = task_queue_.size() + task_continuation_queue_.size() +
                    std::min<uint64_t>(task_background_queue_.size(), max_background - running_background);
  if (queued <= idle)
    return false;
  threads++; /* reserved, add_worker() creates it outside the lock */
  return true;
}

template <typename Task>
void
task_scheduler<Task>::add_worker() {
  dxmt::thread worker([this]() { worker_func(); });
  std::unique_lock<dxmt::mutex> lock(worker_mutex_);
  if (destroyed.load()) {
    /* shutdown() has joined the others already */
    lock.unlock();
    worker_cond_.notify_all();
    worker.join();
    return;
  }
  workers_.push_back(std::move(worker));
}

template <typename Task>
void
task_scheduler<Task>::submit(Task task) {
  bool grow;
  {
    std::unique_lock<dxmt::mutex> lock(worker_mutex_);
    task_queue_.push_back(task);
    grow = idle == 0 && should_grow_locked();
  }
  worker_cond_.notify_one();
  if (grow)
    add_worker();
}

template <typename Task>
void
task_scheduler<Task>::submit_background(Task task) {
  {
    std::unique_lock<dxmt::mutex> lock(worker_mutex_);
    task_background_queue_.push_back(task);
  }
  worker_cond_.notify_one();
}

template <typename Task>
void
task_scheduler<Task>::promote(Task task) {
  {
    /* parked: it goes to the foreground queue once its dependency is done */
    std::unique_lock<dxmt::mutex> lock(deps_mutex_);
    background_parked_.erase(task);
  }
  {
    std::unique_lock<dxmt::mutex> lock(worker_mutex_);
    auto it = std::find(task_background_queue_.begin(), task_background_queue_.end(), task);
    if (it == task_background_queue_.end()) {
      /* already running in the background: raise its worker */
      for (auto slot : slots_) {
        if (slot->task == task && slot->background && !slot->boosted && slot->thread) {
          slot->boosted = true;
          SetThreadPriority(slot->thread, THREAD_PRIORITY_TIME_CRITICAL);
        }
      }
      return;
    }
    task_background_queue_.erase(it);
    task_queue_.push_front(task);
  }
  worker_cond_.notify_one();
}

}; // namespace dxmt
