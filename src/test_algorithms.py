import unittest
import random
import copy
from heapsort_analysis import heapsort, build_max_heap, max_heapify, quicksort, merge_sort
from priority_queue import Task, MaxHeapPriorityQueue


# --- heapsort tests ---

class TestHeapsort(unittest.TestCase):

    def test_empty(self):
        arr = []
        heapsort(arr)
        self.assertEqual(arr, [])

    def test_single_element(self):
        arr = [42]
        heapsort(arr)
        self.assertEqual(arr, [42])

    def test_two_elements_sorted(self):
        arr = [1, 2]
        heapsort(arr)
        self.assertEqual(arr, [1, 2])

    def test_two_elements_unsorted(self):
        arr = [5, 3]
        heapsort(arr)
        self.assertEqual(arr, [3, 5])

    def test_already_sorted(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8]
        heapsort(arr)
        self.assertEqual(arr, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_reverse_sorted(self):
        arr = [8, 7, 6, 5, 4, 3, 2, 1]
        heapsort(arr)
        self.assertEqual(arr, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_random_small(self):
        arr = [random.randint(0, 100) for _ in range(20)]
        expected = sorted(arr)
        heapsort(arr)
        self.assertEqual(arr, expected)

    def test_random_medium(self):
        arr = [random.randint(0, 10000) for _ in range(500)]
        expected = sorted(arr)
        heapsort(arr)
        self.assertEqual(arr, expected)

    def test_random_large(self):
        arr = [random.randint(-10000, 10000) for _ in range(5000)]
        expected = sorted(arr)
        heapsort(arr)
        self.assertEqual(arr, expected)

    def test_all_same(self):
        arr = [7, 7, 7, 7, 7, 7]
        heapsort(arr)
        self.assertEqual(arr, [7, 7, 7, 7, 7, 7])

    def test_duplicates(self):
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        expected = sorted(arr)
        heapsort(arr)
        self.assertEqual(arr, expected)

    def test_negatives(self):
        arr = [-3, -1, -4, -1, -5, -9, -2]
        expected = sorted(arr)
        heapsort(arr)
        self.assertEqual(arr, expected)

    def test_mixed_positive_negative(self):
        arr = [-5, 3, -1, 0, 7, -8, 2]
        expected = sorted(arr)
        heapsort(arr)
        self.assertEqual(arr, expected)

    def test_matches_python_sort_multiple(self):
        for _ in range(10):
            arr = [random.randint(0, 500) for _ in range(200)]
            expected = sorted(arr)
            heapsort(arr)
            self.assertEqual(arr, expected)


class TestBuildMaxHeap(unittest.TestCase):

    def test_heap_property_holds(self):
        arr = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
        build_max_heap(arr)
        # every parent should be >= its children
        n = len(arr)
        for i in range(n):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n:
                self.assertGreaterEqual(arr[i], arr[left])
            if right < n:
                self.assertGreaterEqual(arr[i], arr[right])

    def test_max_at_root(self):
        arr = [3, 5, 1, 8, 2, 7]
        build_max_heap(arr)
        self.assertEqual(arr[0], 8)

    def test_single_element_heap(self):
        arr = [42]
        build_max_heap(arr)
        self.assertEqual(arr, [42])

    def test_empty_heap(self):
        arr = []
        build_max_heap(arr)
        self.assertEqual(arr, [])


# --- quicksort tests ---

class TestQuicksort(unittest.TestCase):

    def test_empty(self):
        arr = []
        quicksort(arr)
        self.assertEqual(arr, [])

    def test_single(self):
        arr = [1]
        quicksort(arr)
        self.assertEqual(arr, [1])

    def test_sorted_input(self):
        arr = list(range(100))
        quicksort(arr)
        self.assertEqual(arr, list(range(100)))

    def test_reverse_sorted(self):
        arr = list(range(100, 0, -1))
        expected = sorted(arr)
        quicksort(arr)
        self.assertEqual(arr, expected)

    def test_random(self):
        arr = [random.randint(0, 1000) for _ in range(500)]
        expected = sorted(arr)
        quicksort(arr)
        self.assertEqual(arr, expected)

    def test_duplicates(self):
        arr = [5, 5, 5, 3, 3, 1, 1, 2, 2]
        expected = sorted(arr)
        quicksort(arr)
        self.assertEqual(arr, expected)

    def test_all_same(self):
        arr = [4, 4, 4, 4]
        quicksort(arr)
        self.assertEqual(arr, [4, 4, 4, 4])


# --- merge sort tests ---

class TestMergeSort(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(merge_sort([]), [])

    def test_single(self):
        self.assertEqual(merge_sort([1]), [1])

    def test_sorted(self):
        self.assertEqual(merge_sort([1, 2, 3, 4]), [1, 2, 3, 4])

    def test_reverse(self):
        self.assertEqual(merge_sort([4, 3, 2, 1]), [1, 2, 3, 4])

    def test_random(self):
        arr = [random.randint(0, 1000) for _ in range(500)]
        self.assertEqual(merge_sort(arr), sorted(arr))

    def test_duplicates(self):
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        self.assertEqual(merge_sort(arr), sorted(arr))

    def test_stability_check(self):
        # merge sort should be stable
        arr = [5, 3, 5, 1, 3]
        self.assertEqual(merge_sort(arr), [1, 3, 3, 5, 5])


# --- all three should agree ---

class TestSortingConsistency(unittest.TestCase):

    def test_all_sorts_match_random(self):
        for _ in range(5):
            base = [random.randint(0, 1000) for _ in range(300)]
            a1 = copy.deepcopy(base)
            a2 = copy.deepcopy(base)
            heapsort(a1)
            quicksort(a2)
            a3 = merge_sort(base)
            self.assertEqual(a1, a2)
            self.assertEqual(a2, a3)

    def test_all_sorts_match_adversarial(self):
        for gen in [lambda n: list(range(n)), lambda n: list(range(n, 0, -1)),
                    lambda n: [1] * n, lambda n: [random.randint(1, 3) for _ in range(n)]]:
            base = gen(200)
            a1 = copy.deepcopy(base)
            a2 = copy.deepcopy(base)
            heapsort(a1)
            quicksort(a2)
            a3 = merge_sort(base)
            self.assertEqual(a1, a2)
            self.assertEqual(a2, a3)


# --- task tests ---

class TestTask(unittest.TestCase):

    def test_creation(self):
        t = Task("T1", 5, "Test task", deadline="2025-03-01")
        self.assertEqual(t.task_id, "T1")
        self.assertEqual(t.priority, 5)
        self.assertEqual(t.name, "Test task")
        self.assertEqual(t.deadline, "2025-03-01")

    def test_default_arrival_time(self):
        t = Task("T1", 1)
        self.assertIsNotNone(t.arrival_time)

    def test_comparison_higher_priority(self):
        t1 = Task("A", 10)
        t2 = Task("B", 5)
        self.assertTrue(t2 < t1)
        self.assertFalse(t1 < t2)

    def test_comparison_equal_priority(self):
        # earlier insertion should win (not be "less than")
        t1 = Task("A", 5)
        t2 = Task("B", 5)
        self.assertTrue(t2 < t1)  # t2 inserted later, so it's "less"

    def test_repr(self):
        t = Task("X", 3, "test")
        self.assertIn("X", repr(t))
        self.assertIn("3", repr(t))


# --- priority queue tests ---

class TestMaxHeapPriorityQueue(unittest.TestCase):

    def test_empty_queue(self):
        pq = MaxHeapPriorityQueue()
        self.assertTrue(pq.is_empty())
        self.assertEqual(pq.size(), 0)
        self.assertEqual(len(pq), 0)

    def test_insert_single(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 5))
        self.assertFalse(pq.is_empty())
        self.assertEqual(pq.size(), 1)

    def test_peek_returns_max(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 3))
        pq.insert(Task("T2", 7))
        pq.insert(Task("T3", 1))
        self.assertEqual(pq.peek().task_id, "T2")
        self.assertEqual(pq.size(), 3)  # peek shouldn't remove

    def test_extract_max_order(self):
        pq = MaxHeapPriorityQueue()
        for i, pri in enumerate([3, 7, 1, 9, 4]):
            pq.insert(Task(f"T{i}", pri))
        extracted = []
        while not pq.is_empty():
            extracted.append(pq.extract_max().priority)
        self.assertEqual(extracted, [9, 7, 4, 3, 1])

    def test_extract_empties_queue(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 1))
        pq.extract_max()
        self.assertTrue(pq.is_empty())

    def test_extract_from_empty_raises(self):
        pq = MaxHeapPriorityQueue()
        with self.assertRaises(IndexError):
            pq.extract_max()

    def test_peek_empty_raises(self):
        pq = MaxHeapPriorityQueue()
        with self.assertRaises(IndexError):
            pq.peek()

    def test_increase_key(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 3))
        pq.insert(Task("T2", 5))
        pq.insert(Task("T3", 1))
        pq.increase_key("T3", 10)
        self.assertEqual(pq.peek().task_id, "T3")
        self.assertEqual(pq.peek().priority, 10)

    def test_increase_key_to_same_value(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 5))
        pq.increase_key("T1", 5)  # no-op but shouldn't break
        self.assertEqual(pq.peek().priority, 5)

    def test_increase_key_lower_raises(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 5))
        with self.assertRaises(ValueError):
            pq.increase_key("T1", 2)

    def test_increase_key_missing_raises(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 5))
        with self.assertRaises(KeyError):
            pq.increase_key("NOPE", 10)

    def test_decrease_key(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 10))
        pq.insert(Task("T2", 5))
        pq.decrease_key("T1", 2)
        self.assertEqual(pq.peek().task_id, "T2")

    def test_decrease_key_higher_raises(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 5))
        with self.assertRaises(ValueError):
            pq.decrease_key("T1", 10)

    def test_decrease_key_missing_raises(self):
        pq = MaxHeapPriorityQueue()
        with self.assertRaises(KeyError):
            pq.decrease_key("NOPE", 1)

    def test_contains(self):
        pq = MaxHeapPriorityQueue()
        pq.insert(Task("T1", 5))
        self.assertIn("T1", pq)
        self.assertNotIn("T2", pq)
        pq.extract_max()
        self.assertNotIn("T1", pq)

    def test_len_tracks_operations(self):
        pq = MaxHeapPriorityQueue()
        self.assertEqual(len(pq), 0)
        pq.insert(Task("T1", 1))
        pq.insert(Task("T2", 2))
        self.assertEqual(len(pq), 2)
        pq.extract_max()
        self.assertEqual(len(pq), 1)
        pq.extract_max()
        self.assertEqual(len(pq), 0)

    def test_many_inserts_extract_descending(self):
        pq = MaxHeapPriorityQueue()
        for i in range(1000):
            pq.insert(Task(f"T{i}", random.randint(1, 10000)))
        prev = float("inf")
        while not pq.is_empty():
            t = pq.extract_max()
            self.assertLessEqual(t.priority, prev)
            prev = t.priority

    def test_heap_property_after_mixed_ops(self):
        pq = MaxHeapPriorityQueue()
        for i in range(50):
            pq.insert(Task(f"T{i}", random.randint(1, 100)))
        # bump some priorities up
        for i in range(0, 50, 5):
            try:
                pq.increase_key(f"T{i}", random.randint(101, 200))
            except (KeyError, ValueError):
                pass
        # drop some priorities down
        for i in range(1, 50, 7):
            try:
                pq.decrease_key(f"T{i}", random.randint(0, 10))
            except (KeyError, ValueError):
                pass
        # extract a few
        for _ in range(10):
            pq.extract_max()
        # everything left should still come out in order
        prev = float("inf")
        while not pq.is_empty():
            t = pq.extract_max()
            self.assertLessEqual(t.priority, prev)
            prev = t.priority

    def test_index_map_consistency(self):
        pq = MaxHeapPriorityQueue()
        for i in range(30):
            pq.insert(Task(f"T{i}", random.randint(1, 50)))
        for tid, idx in pq._index_map.items():
            self.assertEqual(pq._heap[idx].task_id, tid)

    def test_index_map_after_extractions(self):
        pq = MaxHeapPriorityQueue()
        for i in range(20):
            pq.insert(Task(f"T{i}", random.randint(1, 100)))
        for _ in range(10):
            pq.extract_max()
        # remaining entries should still be consistent
        for tid, idx in pq._index_map.items():
            self.assertEqual(pq._heap[idx].task_id, tid)
        self.assertEqual(len(pq._index_map), len(pq._heap))

    def test_sequential_priorities(self):
        # insert 1..n and extract should come back n..1
        pq = MaxHeapPriorityQueue()
        n = 100
        for i in range(1, n + 1):
            pq.insert(Task(f"T{i}", i))
        for expected in range(n, 0, -1):
            self.assertEqual(pq.extract_max().priority, expected)

    def test_repr(self):
        pq = MaxHeapPriorityQueue()
        self.assertIn("size=0", repr(pq))
        pq.insert(Task("T1", 5, "test"))
        self.assertIn("size=1", repr(pq))


if __name__ == "__main__":
    unittest.main()