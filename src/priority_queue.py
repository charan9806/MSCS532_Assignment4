import time
import random
import csv
import os


class Task:
    # simple container for a schedulable unit of work
    _counter = 0  # tiebreaker for stable ordering when priorities match

    def __init__(self, task_id, priority, name="", deadline=None, arrival_time=None):
        self.task_id = task_id
        self.priority = priority
        self.name = name
        self.deadline = deadline
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        Task._counter += 1
        self._seq = Task._counter

    def __repr__(self):
        return f"Task(id={self.task_id}, pri={self.priority}, name='{self.name}')"

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._seq > other._seq  # earlier insertion wins ties


class MaxHeapPriorityQueue:
    # array-backed max-heap; array beats a linked tree because parent/child
    # index math is trivial and memory layout is cache-friendly

    def __init__(self):
        self._heap = []
        # task_id -> heap index, so increase/decrease_key don't need a linear scan
        self._index_map = {}

    def _parent(self, i):
        return (i - 1) // 2

    def _left(self, i):
        return 2 * i + 1

    def _right(self, i):
        return 2 * i + 2

    def _swap(self, i, j):
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index_map[self._heap[i].task_id] = i
        self._index_map[self._heap[j].task_id] = j

    def _sift_up(self, i):
        while i > 0:
            parent = self._parent(i)
            if self._heap[i].priority > self._heap[parent].priority:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._heap)
        while True:
            largest = i
            left = self._left(i)
            right = self._right(i)
            if left < n and self._heap[left].priority > self._heap[largest].priority:
                largest = left
            if right < n and self._heap[right].priority > self._heap[largest].priority:
                largest = right
            if largest == i:
                break
            self._swap(i, largest)
            i = largest

    # --- core operations ---

    def insert(self, task):
        # O(log n) — append and bubble up
        self._heap.append(task)
        idx = len(self._heap) - 1
        self._index_map[task.task_id] = idx
        self._sift_up(idx)

    def extract_max(self):
        # O(log n) — swap root with last, pop, sift down
        if self.is_empty():
            raise IndexError("priority queue is empty")
        self._swap(0, len(self._heap) - 1)
        task = self._heap.pop()
        del self._index_map[task.task_id]
        if not self.is_empty():
            self._sift_down(0)
        return task

    def peek(self):
        if self.is_empty():
            raise IndexError("priority queue is empty")
        return self._heap[0]

    def increase_key(self, task_id, new_priority):
        # O(log n) — update and sift up
        if task_id not in self._index_map:
            raise KeyError(f"task {task_id} not in the queue")
        idx = self._index_map[task_id]
        if new_priority < self._heap[idx].priority:
            raise ValueError(f"new priority {new_priority} is lower than current {self._heap[idx].priority}")
        self._heap[idx].priority = new_priority
        self._sift_up(idx)

    def decrease_key(self, task_id, new_priority):
        # O(log n) — update and sift down
        if task_id not in self._index_map:
            raise KeyError(f"task {task_id} not in the queue")
        idx = self._index_map[task_id]
        if new_priority > self._heap[idx].priority:
            raise ValueError(f"new priority {new_priority} is higher than current {self._heap[idx].priority}")
        self._heap[idx].priority = new_priority
        self._sift_down(idx)

    def is_empty(self):
        return len(self._heap) == 0

    def size(self):
        return len(self._heap)

    def __len__(self):
        return len(self._heap)

    def __contains__(self, task_id):
        return task_id in self._index_map

    def __repr__(self):
        return f"MaxHeapPQ(size={self.size()}, top={self._heap[0] if self._heap else None})"


# --- scheduler demo ---

def run_scheduler_demo():
    pq = MaxHeapPriorityQueue()

    tasks = [
        Task("T1", 3, "Write unit tests", deadline="2025-02-15"),
        Task("T2", 5, "Fix production bug", deadline="2025-02-10"),
        Task("T3", 1, "Update README", deadline="2025-02-20"),
        Task("T4", 4, "Code review", deadline="2025-02-12"),
        Task("T5", 2, "Refactor logging", deadline="2025-02-18"),
        Task("T6", 5, "Security patch", deadline="2025-02-10"),
    ]

    print("=== Task Scheduler Demo ===\n")
    print("Inserting tasks:")
    for t in tasks:
        pq.insert(t)
        print(f"  inserted {t}")

    print(f"\nQueue size: {pq.size()}")
    print(f"Highest priority task: {pq.peek()}")

    print("\nBumping T3 (Update README) priority from 1 to 6...")
    pq.increase_key("T3", 6)
    print(f"New top: {pq.peek()}")

    print("\nProcessing all tasks in priority order:")
    order = 1
    while not pq.is_empty():
        task = pq.extract_max()
        print(f"  {order}. {task}")
        order += 1


# --- benchmarking ---

def run_benchmarks():
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    trials = 3
    results = []

    for n in sizes:
        insert_times, extract_times, inc_times, dec_times = [], [], [], []

        for _ in range(trials):
            pq = MaxHeapPriorityQueue()
            tasks = [Task(f"t{i}", random.randint(1, 1000)) for i in range(n)]

            # time all inserts
            start = time.perf_counter()
            for t in tasks:
                pq.insert(t)
            insert_times.append((time.perf_counter() - start) * 1000)

            # time increase_key on random subset
            sample_size = min(n, 1000)
            inc_ids = [f"t{random.randint(0, n-1)}" for _ in range(sample_size)]
            start = time.perf_counter()
            for tid in inc_ids:
                try:
                    pq.increase_key(tid, random.randint(1001, 5000))
                except (KeyError, ValueError):
                    pass
            inc_times.append((time.perf_counter() - start) * 1000)

            # time decrease_key on random subset
            dec_ids = [f"t{random.randint(0, n-1)}" for _ in range(sample_size)]
            start = time.perf_counter()
            for tid in dec_ids:
                try:
                    pq.decrease_key(tid, random.randint(0, 500))
                except (KeyError, ValueError):
                    pass
            dec_times.append((time.perf_counter() - start) * 1000)

            # time full extraction
            start = time.perf_counter()
            while not pq.is_empty():
                pq.extract_max()
            extract_times.append((time.perf_counter() - start) * 1000)

        row = {
            "n": n,
            "insert_total_ms": round(sum(insert_times) / trials, 4),
            "extract_all_ms": round(sum(extract_times) / trials, 4),
            "increase_key_ms": round(sum(inc_times) / trials, 4),
            "decrease_key_ms": round(sum(dec_times) / trials, 4),
        }
        results.append(row)
        print(f"n={n:>6}: insert={row['insert_total_ms']:>8.2f}ms  extract={row['extract_all_ms']:>8.2f}ms  "
              f"inc_key={row['increase_key_ms']:>6.2f}ms  dec_key={row['decrease_key_ms']:>6.2f}ms")

    return results

def save_benchmark_csv(results, filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "insert_total_ms", "extract_all_ms", "increase_key_ms", "decrease_key_ms"])
        writer.writeheader()
        writer.writerows(results)

def make_benchmark_plots(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    ns = [r["n"] for r in results]

    # total operation time vs n
    plt.figure(figsize=(9, 5))
    plt.plot(ns, [r["insert_total_ms"] for r in results], "b-o", label="Insert All")
    plt.plot(ns, [r["extract_all_ms"] for r in results], "r-s", label="Extract All")
    plt.plot(ns, [r["increase_key_ms"] for r in results], "g-^", label="Increase Key (up to 1k ops)")
    plt.plot(ns, [r["decrease_key_ms"] for r in results], "m-D", label="Decrease Key (up to 1k ops)")
    plt.title("Priority Queue: Total Operation Time vs Input Size")
    plt.xlabel("Number of Elements (n)")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "pq_operations.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # per-element cost for insert and extract
    plt.figure(figsize=(9, 5))
    plt.plot(ns, [r["insert_total_ms"] / r["n"] for r in results], "b-o", label="Insert (per element)")
    plt.plot(ns, [r["extract_all_ms"] / r["n"] for r in results], "r-s", label="Extract (per element)")
    plt.title("Priority Queue: Per-Element Operation Cost")
    plt.xlabel("Number of Elements (n)")
    plt.ylabel("Time per Element (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "pq_per_element.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # insert vs extract scaling (log scale)
    plt.figure(figsize=(9, 5))
    plt.loglog(ns, [r["insert_total_ms"] for r in results], "b-o", label="Insert All")
    plt.loglog(ns, [r["extract_all_ms"] for r in results], "r-s", label="Extract All")
    # reference O(n log n) line
    import math
    ref = [ns[0] * math.log2(ns[0])]
    scale = results[0]["insert_total_ms"] / ref[0]
    ref_line = [n * math.log2(n) * scale for n in ns]
    plt.loglog(ns, ref_line, "k--", alpha=0.5, label="O(n log n) reference")
    plt.title("Priority Queue: Log-Log Scaling")
    plt.xlabel("n")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "pq_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
    os.makedirs(out_dir, exist_ok=True)

    run_scheduler_demo()
    print("\n\n=== Priority Queue Benchmarks ===\n")
    results = run_benchmarks()
    save_benchmark_csv(results, os.path.join(out_dir, "pq_benchmark.csv"))
    make_benchmark_plots(results, out_dir)
    print(f"\nResults saved to {out_dir}/")