import time
import random
import copy
import sys
import csv
import os

sys.setrecursionlimit(50000)


# --- heapsort ---

def max_heapify(arr, heap_size, i):
    # iterative sift-down avoids python's recursion overhead
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < heap_size and arr[left] > arr[largest]:
            largest = left
        if right < heap_size and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

def build_max_heap(arr):
    n = len(arr)
    # start from last non-leaf, work backwards — this is the O(n) build
    for i in range(n // 2 - 1, -1, -1):
        max_heapify(arr, n, i)

def heapsort(arr):
    n = len(arr)
    build_max_heap(arr)
    # pull max to end one at a time, shrink heap
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        max_heapify(arr, i, 0)


# --- quicksort (randomized, lomuto, iterative stack) ---

def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    # explicit stack so we don't blow python's recursion limit on sorted input
    stack = [(low, high)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        pivot_idx = random.randint(lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]
        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        stack.append((lo, i - 1))
        stack.append((i + 1, hi))


# --- merge sort ---

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# --- benchmarking ---

def generate_input(dist, n):
    if dist == "Random":
        return [random.randint(0, n * 10) for _ in range(n)]
    elif dist == "Sorted":
        return list(range(n))
    elif dist == "Reverse-Sorted":
        return list(range(n, 0, -1))
    elif dist == "Repeated":
        return [random.randint(1, 5) for _ in range(n)]
    return []

def time_sort(sort_fn, arr, returns_new=False):
    start = time.perf_counter()
    result = sort_fn(arr)
    end = time.perf_counter()
    return (end - start) * 1000

def run_benchmarks():
    sizes = [100, 500, 1000, 5000, 10000]
    distributions = ["Random", "Sorted", "Reverse-Sorted", "Repeated"]
    trials = 3
    results = []

    for dist in distributions:
        for n in sizes:
            hs_times, qs_times, ms_times = [], [], []
            for _ in range(trials):
                base = generate_input(dist, n)

                arr1 = copy.deepcopy(base)
                hs_times.append(time_sort(heapsort, arr1))

                arr2 = copy.deepcopy(base)
                qs_times.append(time_sort(quicksort, arr2))

                arr3 = copy.deepcopy(base)
                ms_times.append(time_sort(merge_sort, arr3, returns_new=True))

            row = {
                "distribution": dist,
                "size": n,
                "Heapsort": round(sum(hs_times) / trials, 4),
                "Quicksort": round(sum(qs_times) / trials, 4),
                "Merge Sort": round(sum(ms_times) / trials, 4),
            }
            results.append(row)
            print(f"{dist}, n={n}: HS={row['Heapsort']:.4f}ms  QS={row['Quicksort']:.4f}ms  MS={row['Merge Sort']:.4f}ms")

    return results

def save_csv(results, filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["distribution", "size", "Heapsort", "Quicksort", "Merge Sort"])
        writer.writeheader()
        writer.writerows(results)

def make_plots(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    distributions = ["Random", "Sorted", "Reverse-Sorted", "Repeated"]
    colors = {"Heapsort": "b", "Quicksort": "g", "Merge Sort": "r"}
    markers = {"Heapsort": "o", "Quicksort": "s", "Merge Sort": "^"}

    # combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sorting Comparison: Heapsort vs Quicksort vs Merge Sort", fontsize=14, fontweight="bold")
    for idx, dist in enumerate(distributions):
        ax = axes[idx // 2][idx % 2]
        subset = [r for r in results if r["distribution"] == dist]
        sizes = [r["size"] for r in subset]
        for algo in ["Heapsort", "Quicksort", "Merge Sort"]:
            ax.plot(sizes, [r[algo] for r in subset],
                    f"{colors[algo]}-{markers[algo]}", label=algo)
        ax.set_title(dist)
        ax.set_xlabel("n")
        ax.set_ylabel("Time (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sorting_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # individual per-distribution
    for dist in distributions:
        subset = [r for r in results if r["distribution"] == dist]
        sizes = [r["size"] for r in subset]
        plt.figure(figsize=(8, 5))
        for algo in ["Heapsort", "Quicksort", "Merge Sort"]:
            plt.plot(sizes, [r[algo] for r in subset],
                     f"{colors[algo]}-{markers[algo]}", label=algo)
        plt.title(f"Sorting Comparison \u2014 {dist} Input")
        plt.xlabel("Array Size (n)")
        plt.ylabel("Average Time (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = f"sorting_{dist.lower().replace('-', '_')}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
    os.makedirs(out_dir, exist_ok=True)

    print("Running sorting benchmarks...")
    results = run_benchmarks()
    save_csv(results, os.path.join(out_dir, "sorting_results.csv"))
    make_plots(results, out_dir)
    print(f"\nResults saved to {out_dir}/")