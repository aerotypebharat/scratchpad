package com.god.file;

/**
 * AlgorithmPatternsBook.java
 * <p>
 * A single-file, offline-ready collection of 31 algorithmic & system patterns in Java.
 * Each pattern contains:
 * - A short conceptual overview
 * - Intuition / when to use
 * - Key points
 * - Time & space complexity
 * - A concise, runnable Java example (method or small class)
 * <p>
 * Usage:
 * - Open in your IDE and run main() for quick demos.
 * - Each pattern's explanation is in the comment block immediately above the code.
 * <p>
 * Java version: 11+ (works on Java 8 for most code; some Collection helpers require 9+)
 */

import java.util.*;
import java.util.concurrent.*;

public class AlgorithmPatternsBookReverse {


    // ======================================================================================
    // 31) Pattern: Multi-thread (Executor + Callable)
    // ======================================================================================
    /*
     * Concept:
     *   Use ExecutorService for thread pools, Callable/Future for results, CompletableFuture for chaining.
     *
     * Intuition:
     *   Avoid managing Thread objects directly when scaling; use thread pools and high-level constructs.
     *
     * Example:
     *   Submit a Callable, get Future, then get() the result.
     *
     * Complexity:
     *   Threading complexity depends on workload; manage concurrency (locks / atomic) and shutdown pools.
     */
    static void multiThreadExample() throws Exception {
        ExecutorService ex = Executors.newFixedThreadPool(2);
        Callable<Integer> task = () -> {
            Thread.sleep(100); // simulate work
            return 42;
        };
        Future<Integer> f = ex.submit(task);
        System.out.println("Future result: " + f.get()); // blocking
        ex.shutdown();
    }


    // ======================================================================================
    // 30) Pattern: Prefix Sum
    // ======================================================================================
    /*
     * Concept:
     *   Build prefix sums to compute subarray sums quickly: sum(i..j) = prefix[j+1] - prefix[i]
     *
     * Complexity:
     *   Build: O(n)
     *   Query: O(1)
     *   Space: O(n)
     */
    static int[] prefixSum(int[] nums) {
        int[] pref = new int[nums.length + 1];
        for (int i = 0; i < nums.length; i++) pref[i + 1] = pref[i] + nums[i];
        return pref;
    }

    // ======================================================================================
    // 29) Pattern: Ordered Set
    // ======================================================================================
    /*
     * Concept:
     *   Balanced BST-backed set (TreeSet in Java) for ordered operations (min, max, floor, ceiling).
     *
     * Complexity:
     *   Basic ops: O(log n)
     */
    static void orderedSetExample() {
        TreeSet<Integer> ts = new TreeSet<>();
        ts.add(5);
        ts.add(1);
        ts.add(3);
        ts.add(2);
        System.out.println("Min: " + ts.first() + " Max: " + ts.last());
        System.out.println("Ceiling(2) = " + ts.ceiling(2));
    }

    // ======================================================================================
    // 28) Pattern: Union Find (Disjoint Set)
    // ======================================================================================
    /*
     * Concept:
     *   Manage connected components via parent[] and union by rank / path compression.
     *
     * Complexity:
     *   Amortized nearly O(1) per op (inverse Ackermann)
     *   Space: O(n)
     */
    static class UnionFind {
        int[] parent, rank;

        UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
        }

        int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }

        void union(int a, int b) {
            int pa = find(a), pb = find(b);
            if (pa == pb) return;
            if (rank[pa] < rank[pb]) parent[pa] = pb;
            else if (rank[pa] > rank[pb]) parent[pb] = pa;
            else {
                parent[pb] = pa;
                rank[pa]++;
            }
        }
    }

    // ======================================================================================
    // 27) Pattern: Topological Sort (Graph)
    // ======================================================================================
    /*
     * Concept:
     *   Linear ordering of vertices in a DAG. Kahn's algorithm uses in-degrees + queue.
     *
     * Complexity:
     *   Time: O(V + E)
     *   Space: O(V)
     */
    static List<Integer> topologicalSort(int V, int[][] edges) {
        Map<Integer, List<Integer>> g = new HashMap<>();
        int[] indeg = new int[V];
        for (int[] e : edges) {
            g.computeIfAbsent(e[0], k -> new ArrayList<>()).add(e[1]);
            indeg[e[1]]++;
        }
        Queue<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < V; i++) if (indeg[i] == 0) q.offer(i);
        List<Integer> res = new ArrayList<>();
        while (!q.isEmpty()) {
            int u = q.poll();
            res.add(u);
            for (int v : g.getOrDefault(u, Collections.emptyList())) {
                if (--indeg[v] == 0) q.offer(v);
            }
        }
        return res;
    }


    // ======================================================================================
    // 26) Pattern: Trie
    // ======================================================================================
    /*
     * Concept:
     *   A prefix tree for strings — supports insert/search/prefix queries in O(length) time.
     *
     * Complexity:
     *   Insert/Search: O(L) where L is length of word
     *   Space: O(sum of lengths)
     */
    static class TrieNode {
        Map<Character, TrieNode> children = new HashMap<>();
        boolean end;
    }

    static class Trie {
        TrieNode root = new TrieNode();

        void insert(String word) {
            TrieNode cur = root;
            for (char c : word.toCharArray())
                cur = cur.children.computeIfAbsent(c, k -> new TrieNode());
            cur.end = true;
        }

        boolean search(String word) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                cur = cur.children.get(c);
                if (cur == null) return false;
            }
            return cur.end;
        }

        boolean startsWith(String prefix) {
            TrieNode cur = root;
            for (char c : prefix.toCharArray()) {
                cur = cur.children.get(c);
                if (cur == null) return false;
            }
            return true;
        }
    }


    // ======================================================================================
    // 25) Pattern: Backtracking
    // ======================================================================================
    /*
     * Concept:
     *   Exhaustive search that incrementally builds candidates and abandons those that fail constraints.
     *
     * Example:
     *   Generate permutations using recursion + used[] array.
     *
     * Complexity:
     *   Time: O(n! * n) for permutations (heavy), Space: O(n)
     */
    static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrackPermute(nums, new ArrayList<>(), new boolean[nums.length], res);
        return res;
    }

    static void backtrackPermute(int[] nums, List<Integer> temp, boolean[] used, List<List<Integer>> res) {
        if (temp.size() == nums.length) {
            res.add(new ArrayList<>(temp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            temp.add(nums[i]);
            backtrackPermute(nums, temp, used, res);
            used[i] = false;
            temp.remove(temp.size() - 1);
        }
    }


    // ======================================================================================
    // 24) Pattern: Palindromic Subsequence (Dynamic Programming)
    // ======================================================================================
    /*
     * Concept:
     *   Longest Palindromic Subsequence (LPS) via DP on substrings:
     *     dp[i][j] = 2 + dp[i+1][j-1] if s[i]==s[j] else max(dp[i+1][j], dp[i][j-1])
     *
     * Complexity:
     *   Time: O(n^2)
     *   Space: O(n^2) (can be optimized)
     */
    static int longestPalindromicSubseq(String s) {
        int n = s.length();
        if (n == 0) return 0;
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) dp[i][j] = 2 + (i + 1 <= j - 1 ? dp[i + 1][j - 1] : 0);
                else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
        return dp[0][n - 1];
    }


    // ======================================================================================
    // 23) Pattern: Fibonacci (Dynamic Programming)
    // ======================================================================================
    /*
     * Concept:
     *   Use iterative DP (bottom-up) or memoization to compute fib in linear time.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(1) (iterative) or O(n) (memo)
     */
    static int fib(int n) {
        if (n <= 1) return n;
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            int t = a + b;
            a = b;
            b = t;
        }
        return b;
    }


    // ======================================================================================
    // 22) Pattern: 0/1 Knapsack (Dynamic Programming)
    // ======================================================================================
    /*
     * Concept:
     *   Classic DP: dp[i][w] = max value using first i items with capacity w.
     *
     * Complexity:
     *   Time: O(n * W)
     *   Space: O(n * W) (can be optimized to O(W))
     */
    static int knapsack(int[] weights, int[] values, int W) {
        int n = weights.length;
        int[] dp = new int[W + 1];
        for (int i = 0; i < n; i++) {
            for (int w = W; w >= weights[i]; w--) {
                dp[w] = Math.max(dp[w], values[i] + dp[w - weights[i]]);
            }
        }
        return dp[W];
    }


    // ======================================================================================
    // 21) Pattern: Greedy Algorithms
    // ======================================================================================
    /*
     * Concept:
     *   Make the locally optimal choice and rely on greedy-choice property + optimal substructure.
     *
     * Example:
     *   Activity selection (select max number of non-overlapping activities)
     *
     * Complexity:
     *   Time: O(n log n) for sorting by finish time, then O(n)
     *   Space: O(1)
     */
    static int maxActivities(int[] start, int[] end) {
        int n = start.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingInt(i -> end[i]));
        int count = 0, lastEnd = Integer.MIN_VALUE;
        for (int i : idx) {
            if (start[i] >= lastEnd) {
                count++;
                lastEnd = end[i];
            }
        }
        return count;
    }


    // ======================================================================================
    // 20) Pattern: K-way merge
    // ======================================================================================
    /*
     * Concept:
     *   Merge K sorted lists/arrays using a min-heap keyed by current element from each array.
     *
     * Complexity:
     *   Time: O(n log k) where n = total elements across arrays
     *   Space: O(k)
     */
    static List<Integer> mergeKSortedArrays(List<int[]> arrays) {
        List<Integer> res = new ArrayList<>();
        // min-heap of (value, arrayIndex, elementIndex)
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        for (int i = 0; i < arrays.size(); i++) {
            if (arrays.get(i).length > 0) pq.offer(new int[]{arrays.get(i)[0], i, 0});
        }
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            res.add(cur[0]);
            int ai = cur[1], ei = cur[2] + 1;
            if (ei < arrays.get(ai).length) pq.offer(new int[]{arrays.get(ai)[ei], ai, ei});
        }
        return res;
    }


    // ======================================================================================
    // 19) Pattern: Top 'K' Elements
    // ======================================================================================
    /*
     * Concept:
     *   Use a min-heap of size k to keep top k elements in O(n log k) time.
     *
     * Complexity:
     *   Time: O(n log k)
     *   Space: O(k)
     */
    static List<Integer> topK(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int n : nums) {
            pq.offer(n);
            if (pq.size() > k) pq.poll();
        }
        // result is unsorted as a list — to sort, pop into list and reverse
        List<Integer> res = new ArrayList<>(pq);
        Collections.sort(res, Collections.reverseOrder());
        return res;
    }

    // ======================================================================================
    // 18) Pattern: Bitwise XOR
    // ======================================================================================
    /*
     * Concept:
     *   XOR properties: x^x = 0, x^0 = x; useful to find unique elements when others appear twice.
     *
     * Example: Single number where all others appear twice.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(1)
     */
    static int singleNumber(int[] nums) {
        int res = 0;
        for (int n : nums) res ^= n;
        return res;
    }


    // ======================================================================================
    // 17) Pattern: Modified Binary Search
    // ======================================================================================
    /*
     * Concept:
     *   Binary search variants solve nearest/ceiling/floor/first/last occurrences instead
     *   of exact matches by changing condition & boundary updates.
     *
     * Example: Find ceiling (first index >= target)
     *
     * Complexity:
     *   Time: O(log n)
     *   Space: O(1)
     */
    static int searchCeiling(int[] nums, int target) {
        int left = 0, right = nums.length - 1, ans = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) {
                ans = mid;
                right = mid - 1;
            } else left = mid + 1;
        }
        return ans;
    }


    // ======================================================================================
    // 16) Pattern: Subsets
    // ======================================================================================
    /*
     * Concept:
     *   Generate all subsets (power set) iteratively (start from empty set) or backtracking.
     *
     * Complexity:
     *   Time: O(n * 2^n) (number of subsets * cost to copy)
     *   Space: O(2^n * n)
     */
    static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>());
        for (int num : nums) {
            List<List<Integer>> add = new ArrayList<>();
            for (List<Integer> base : res) {
                List<Integer> copy = new ArrayList<>(base);
                copy.add(num);
                add.add(copy);
            }
            res.addAll(add);
        }
        return res;
    }


    // ======================================================================================
    // 15) Pattern: Two Heaps (Median of a Data Stream)
    // ======================================================================================
    /*
     * Concept:
     *   Use two heaps: max-heap for lower half, min-heap for upper half to get median in O(1).
     *
     * Complexity:
     *   addNum: O(log n)
     *   findMedian: O(1)
     *   Space: O(n)
     */
    static class MedianFinder {
        private final PriorityQueue<Integer> lowers = new PriorityQueue<>(Collections.reverseOrder());
        private final PriorityQueue<Integer> highers = new PriorityQueue<>();

        void addNum(int num) {
            lowers.offer(num);
            highers.offer(lowers.poll());
            if (highers.size() > lowers.size()) lowers.offer(highers.poll());
        }

        double findMedian() {
            if (lowers.size() == highers.size()) return (lowers.peek() + highers.peek()) / 2.0;
            return lowers.peek();
        }
    }


    // ======================================================================================
    // 14) Pattern: Island (Matrix traversal)
    // ======================================================================================
    /*
     * Concept:
     *   Grid traversal with DFS/BFS to explore connected components (islands).
     *
     * Typical problem:
     *   Count islands in binary matrix: do DFS from each '1' and mark visited.
     *
     * Complexity:
     *   Time: O(m*n)
     *   Space: O(m*n) recursion stack worst-case (or O(min(m,n)) if smart)
     */
    static void dfsIsland(char[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length) return;
        if (grid[i][j] == '0') return;
        grid[i][j] = '0'; // mark visited (mutate)
        dfsIsland(grid, i + 1, j);
        dfsIsland(grid, i - 1, j);
        dfsIsland(grid, i, j + 1);
        dfsIsland(grid, i, j - 1);
    }


    // ======================================================================================
    // 13) Pattern: Graphs (Adjacency List + BFS)
    // ======================================================================================
    /*
     * Concept:
     *   Represent graph with adjacency lists. BFS/DFS traverse nodes; Dijkstra/Kruskal apply
     *   for weighted/minimum spanning tree etc.
     *
     * Intuition:
     *   Use visited set to avoid cycles. For shortest unweighted paths, BFS works.
     *
     * Complexity (BFS):
     *   Time: O(V + E)
     *   Space: O(V)
     */
    static void bfsGraph(Map<Integer, List<Integer>> graph, int start) {
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> q = new ArrayDeque<>();
        q.offer(start);
        while (!q.isEmpty()) {
            int node = q.poll();
            if (visited.contains(node)) continue;
            visited.add(node);
            System.out.print(node + " ");
            for (int nei : graph.getOrDefault(node, Collections.emptyList())) {
                if (!visited.contains(nei)) q.offer(nei);
            }
        }
        System.out.println();
    }


    // ======================================================================================
    // 12) Pattern: Tree Depth First Search (DFS)
    // ======================================================================================
    /*
     * Concept:
     *   DFS can be recursive (call stack) or iterative using a stack. Preorder/inorder/postorder.
     *
     * Use cases:
     *   Traversal, backtracking on trees, path finding.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(h) recursion stack (h = tree height)
     */
    static void dfsPreorder(TreeNode root) {
        if (root == null) return;
        System.out.print(root.val + " ");
        dfsPreorder(root.left);
        dfsPreorder(root.right);
    }


    // ======================================================================================
    // 11) Pattern: Tree Breadth First Search (BFS)
    // ======================================================================================
    /*
     * Concept:
     *   Generic BFS for graphs/trees using a queue. Level order traversal is a special case.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(n)
     */
    static void bfsTree(TreeNode root) {
        if (root == null) return;
        Queue<TreeNode> q = new ArrayDeque<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode cur = q.poll();
            System.out.print(cur.val + " ");
            if (cur.left != null) q.offer(cur.left);
            if (cur.right != null) q.offer(cur.right);
        }
        System.out.println();
    }

    // ======================================================================================
    // 10) Pattern: Level Order Traversal
    // ======================================================================================
    /*
     * Concept:
     *   BFS applied to trees using a queue to traverse level by level.
     *
     * Intuition:
     *   Use the queue size to delineate levels.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(w) where w is max width of tree (queue)
     */
    static class TreeNode {
        int val;
        TreeNode left, right;

        TreeNode(int v) {
            val = v;
        }
    }

    static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> q = new ArrayDeque<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode n = q.poll();
                level.add(n.val);
                if (n.left != null) q.offer(n.left);
                if (n.right != null) q.offer(n.right);
            }
            res.add(level);
        }
        return res;
    }


    // ======================================================================================
    // 9) Pattern: Hash Maps
    // ======================================================================================
    /*
     * Concept:
     *   Key-value store for O(1) average lookup/insert. Great for frequency counting, index maps.
     *
     * Example: Character frequency.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(k) (k distinct keys)
     */
    static Map<Character, Integer> charFrequency(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) map.put(c, map.getOrDefault(c, 0) + 1);
        return map;
    }


    // ======================================================================================
    // 8) Pattern: Monotonic Stack
    // ======================================================================================
    /*
     * Concept:
     *   Maintain a stack that is either strictly increasing or decreasing to solve "next/prev
     *   greater/smaller" queries in linear time.
     *
     * Intuition:
     *   Each element is pushed/popped at most once, giving O(n).
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(n)
     */
    static int[] nextGreater(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        Deque<Integer> st = new ArrayDeque<>(); // store indices, monotonic decreasing
        for (int i = 0; i < n; i++) {
            while (!st.isEmpty() && nums[i] > nums[st.peek()]) {
                ans[st.pop()] = nums[i];
            }
            st.push(i);
        }
        return ans;
    }


    // ======================================================================================
    // 7) Pattern: Stack
    // ======================================================================================
    /*
     * Concept:
     *   LIFO structure. Useful for parenthesis matching, DFS iterative, expression evaluation.
     *
     * Example: Balanced parentheses.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(n)
     */
    static boolean isValidParentheses(String s) {
        Deque<Character> st = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '(') st.push(')');
            else if (c == '{') st.push('}');
            else if (c == '[') st.push(']');
            else {
                if (st.isEmpty() || st.pop() != c) return false;
            }
        }
        return st.isEmpty();
    }


    // ======================================================================================
    // 6) Pattern: In-place Reversal of a LinkedList
    // ======================================================================================
    /*
     * Concept:
     *   Iteratively rewire `next` pointers to reverse a singly linked list.
     *
     * Intuition:
     *   Keep track of previous node and move through list flipping pointers.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(1)
     */
    static ListNode reverseLinkedList(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }


    // ======================================================================================
    // 5) Pattern: Cyclic Sort
    // ======================================================================================
    /*
     * Concept:
     *   For arrays containing numbers in a limited range (1..n), place each number at its
     *   correct index by swapping until all are in position.
     *
     * Intuition:
     *   Use index as a hash — O(n) time and O(1) space to reorder for detection problems
     *   (missing numbers, duplicates, etc.)
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(1)
     */
    static void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    static void cyclicSort(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            int correct = nums[i] - 1;
            if (correct >= 0 && correct < nums.length && nums[i] != nums[correct])
                swap(nums, i, correct);
            else i++;
        }
    }


    // ======================================================================================
    // 4) Pattern: Merge Intervals
    // ======================================================================================
    /*
     * Concept:
     *   Sort intervals by start time, then merge overlapping ones by comparing current end.
     *
     * Intuition:
     *   Sorting gives an ordering that allows a single pass to combine overlaps.
     *
     * Complexity:
     *   Time: O(n log n) for sorting
     *   Space: O(n) for result (or O(1) additional)
     */
    static List<int[]> mergeIntervals(int[][] intervals) {
        if (intervals == null || intervals.length == 0) return Collections.emptyList();
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        List<int[]> res = new ArrayList<>();
        int[] cur = intervals[0];
        for (int[] next : intervals) {
            if (next[0] <= cur[1]) cur[1] = Math.max(cur[1], next[1]);
            else {
                res.add(cur);
                cur = next;
            }
        }
        res.add(cur);
        return res;
    }


    // ======================================================================================
    // 3) Pattern: Sliding Window
    // ======================================================================================
    /*
     * Concept:
     *   Maintain a window (subarray / substring) using two pointers (left, right).
     *
     * Intuition:
     *   For problems seeking subarray with some property (sum, distincts, etc.), expand right
     *   to include elements and move left to shrink when constraints are violated.
     *
     * Example: Longest substring without repeating characters.
     *
     * Complexity:
     *   Time: O(n) (each element enters/exits window at most once)
     *   Space: O(k) for the window contents (k = charset or distinct count)
     */
    static int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> lastIndex = new HashMap<>();
        int left = 0, maxLen = 0;
        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);
            if (lastIndex.containsKey(c))
                left = Math.max(left, lastIndex.get(c) + 1); // shrink window
            lastIndex.put(c, right);
            maxLen = Math.max(maxLen, right - left + 1);
        }
        return maxLen;
    }


    // ======================================================================================
    // 2) Pattern: Fast & Slow Pointers
    // ======================================================================================
    /*
     * Concept:
     *   Use two pointers that move at different speeds to detect cycles and find meeting points.
     *
     * Intuition:
     *   In a cycle, a faster runner will eventually catch the slower runner.
     *
     * Use cases:
     *   - Cycle detection in linked lists (Floyd's Tortoise and Hare)
     *   - Finding middle node, removing nth-from-end (by shifting)
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(1)
     */
    static class ListNode {
        int val;
        ListNode next;

        ListNode(int v) {
            val = v;
        }
    }

    static boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }


    // ======================================================================================
    // 1) Pattern: Two Pointers
    // ======================================================================================
    /*
     * Concept:
     *   Use two indices that move towards each other (or move in one pass) to find relationships
     *   between elements in a linear structure (array / string).
     *
     * Intuition:
     *   When input is sorted (or can be treated in two directions) and you want to compare pairs,
     *   two pointers avoid O(n^2) brute-force pair comparisons.
     *
     * Key points:
     *   - Works well for sorted arrays, reversed traversal, or fixed-sum pair problems.
     *
     * Complexity:
     *   Time: O(n)
     *   Space: O(1)
     */
    static int[] twoSumSorted(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == target) return new int[]{left, right};
            if (sum < target) left++;
            else right--;
        }
        return new int[]{-1, -1};
    }

    // ======================================================================================
    // Helper & Demo
    // ======================================================================================
    public static void main(String[] args) throws Exception {
        System.out.println("=== Two Pointers (twoSumSorted) ===");
        System.out.println(Arrays.toString(twoSumSorted(new int[]{1, 2, 3, 4, 6}, 6)));

        System.out.println("\n=== Sliding Window (lengthOfLongestSubstring) ===");
        System.out.println(lengthOfLongestSubstring("abcabcbb"));

        System.out.println("\n=== Bitwise XOR (singleNumber) ===");
        System.out.println(singleNumber(new int[]{2, 2, 1}));

        System.out.println("\n=== K-way merge demo ===");
        List<int[]> arrays = new ArrayList<>();
        arrays.add(new int[]{1, 4, 7});
        arrays.add(new int[]{2, 5, 6});
        arrays.add(new int[]{3, 8});
        System.out.println(mergeKSortedArrays(arrays));

        System.out.println("\n=== Trie demo ===");
        Trie trie = new Trie();
        trie.insert("hello");
        trie.insert("help");
        System.out.println("search hello: " + trie.search("hello"));
        System.out.println("startsWith he: " + trie.startsWith("he"));

        System.out.println("\n=== Multi-thread demo ===");
        multiThreadExample();
    }
}
