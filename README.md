# Assignment 4: Heap Data Structures

This repo has my code and write-up for Assignment 4 on heaps, heapsort, and priority queues.

## Repo Structure

    MSCS532_ASSIGNMENT4/
    ├── Results/                          # auto-generated when you run the scripts
    │   ├── sorting_results.csv
    │   ├── sorting_random.png
    │   ├── sorting_sorted.png
    │   ├── sorting_reverse_sorted.png
    │   ├── sorting_repeated.png
    │   ├── sorting_combined.png
    │   ├── pq_benchmark.csv
    │   ├── pq_operations.png
    │   ├── pq_per_element.png
    │   └── pq_scaling.png
    ├── src/
    │   ├── heapsort_analysis.py          # sorting comparison + timing + plots
    │   ├── priority_queue.py             # priority queue implementation + benchmarks + plots
    │   └── test_algorithms.py            # 61 unit tests across both implementations
    └── README.md

## How to Run

You need Python 3.8+ and matplotlib for the plots. Run everything from the repo root.

Run the sorting benchmarks (generates CSVs and plots in Results/):

    cd src && python heapsort_analysis.py

Run the priority queue demo and benchmarks:

    cd src && python priority_queue.py

Run the tests:

    cd src && python -m unittest test_algorithms -v


## Quick Summary

Heapsort ran consistently at O(n log n) regardless of input pattern. It was about 30-40% slower than quicksort on random data (cache locality differences), but on repeated elements quicksort with Lomuto fell apart at 568 ms while heapsort stayed at 12 ms. Merge sort was competitive everywhere but needs O(n) extra space.

Priority queue operations all confirmed O(log n) bounds. Per-element insert and extract costs stayed flat from n=100 to n=50,000, and the log-log scaling plot sits right on top of the O(n log n) reference line.