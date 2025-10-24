package com.god.file;

import java.util.*;
import java.util.concurrent.*;

public class LeetCodePatternsReverse_part1 {

    // PATTERN 31: MULTI-THREAD
    // Use Case: Concurrency, synchronization

    // 31.1: Print in Order

    class Foo {
        private Semaphore firstDone = new Semaphore(0);
        private Semaphore secondDone = new Semaphore(0);

        public Foo() {
        }

        public void first(Runnable printFirst) throws InterruptedException {
            printFirst.run();
            firstDone.release();
        }

        public void second(Runnable printSecond) throws InterruptedException {
            firstDone.acquire();
            printSecond.run();
            secondDone.release();
        }

        public void third(Runnable printThird) throws InterruptedException {
            secondDone.acquire();
            printThird.run();
        }
    }


    // 31.2: Print FooBar Alternately

    class FooBar {
        private int n;
        private Semaphore fooSem = new Semaphore(1);
        private Semaphore barSem = new Semaphore(0);

        public FooBar(int n) {
            this.n = n;
        }

        public void foo(Runnable printFoo) throws InterruptedException {
            for (int i = 0; i < n; i++) {
                fooSem.acquire();
                printFoo.run();
                barSem.release();
            }
        }

        public void bar(Runnable printBar) throws InterruptedException {
            for (int i = 0; i < n; i++) {
                barSem.acquire();
                printBar.run();
                fooSem.release();
            }
        }
    }


    // 31.3: Dining Philosophers

    class DiningPhilosophers {
        private Semaphore[] forks = new Semaphore[5];
        private Semaphore dining = new Semaphore(4); // Allow only 4 philosophers to eat

        public DiningPhilosophers() {
            for (int i = 0; i < 5; i++) {
                forks[i] = new Semaphore(1);
            }
        }

        public void wantsToEat(int philosopher, Runnable pickLeftFork, Runnable pickRightFork, Runnable eat, Runnable putLeftFork, Runnable putRightFork) throws InterruptedException {

            int leftFork = philosopher;
            int rightFork = (philosopher + 1) % 5;

            dining.acquire();

            forks[leftFork].acquire();
            forks[rightFork].acquire();

            pickLeftFork.run();
            pickRightFork.run();

            eat.run();

            putLeftFork.run();
            putRightFork.run();

            forks[leftFork].release();
            forks[rightFork].release();

            dining.release();
        }
    }


    // PATTERN 30: PREFIX SUM
    // Use Case: Range sum queries, subarray problems


    // 30.1: Subarray Sum Equals K
    // Count number of subarrays with sum exactly k
    // Time: O(n), Space: O(n)

    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        prefixSumCount.put(0, 1); // Base case: empty subarray has sum 0

        int count = 0, prefixSum = 0;

        for (int num : nums) {
            prefixSum += num;

            // Check if (prefixSum - k) exists
            if (prefixSumCount.containsKey(prefixSum - k)) {
                count += prefixSumCount.get(prefixSum - k);
            }

            prefixSumCount.put(prefixSum, prefixSumCount.getOrDefault(prefixSum, 0) + 1);
        }
        return count;
    }


    // 30.2: Continuous Subarray Sum
    // Check if there's subarray with sum multiple of k (size >= 2)
    // Time: O(n), Space: O(n)

    public boolean checkSubarraySum(int[] nums, int k) {
        Map<Integer, Integer> remainderIndex = new HashMap<>();
        remainderIndex.put(0, -1); // Handle case where subarray starts from index 0

        int prefixSum = 0;

        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            int remainder = prefixSum % k;

            if (remainderIndex.containsKey(remainder)) {
                // Check if subarray length >= 2
                if (i - remainderIndex.get(remainder) >= 2) {
                    return true;
                }
            } else {
                remainderIndex.put(remainder, i);
            }
        }
        return false;
    }


    // 30.3: Range Sum Query 2D - Immutable
    // Query sum of submatrix in O(1) time after O(mn) preprocessing

    class NumMatrix {
        private int[][] prefixSum;

        public NumMatrix(int[][] matrix) {
            if (matrix.length == 0 || matrix[0].length == 0) return;

            int rows = matrix.length, cols = matrix[0].length;
            prefixSum = new int[rows + 1][cols + 1];

            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= cols; j++) {
                    prefixSum[i][j] = matrix[i - 1][j - 1] + prefixSum[i - 1][j] + prefixSum[i][j - 1] - prefixSum[i - 1][j - 1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return prefixSum[row2 + 1][col2 + 1] - prefixSum[row1][col2 + 1] - prefixSum[row2 + 1][col1] + prefixSum[row1][col1];
        }
    }


    // PATTERN 29: ORDERED SET
    // Use Case: Range queries, sorted data operations


    // 29.1: Contains Duplicate III
    // Check if there are nearby indices with nearby values
    // Time: O(n log k), Space: O(k)

    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        TreeSet<Long> set = new TreeSet<>();

        for (int i = 0; i < nums.length; i++) {
            Long floor = set.floor((long) nums[i] + valueDiff);
            Long ceiling = set.ceiling((long) nums[i] - valueDiff);

            if ((floor != null && floor >= nums[i]) || (ceiling != null && ceiling <= nums[i])) {
                return true;
            }

            set.add((long) nums[i]);

            // Maintain sliding window of size indexDiff
            if (i >= indexDiff) {
                set.remove((long) nums[i - indexDiff]);
            }
        }
        return false;
    }


    // 29.2: Count of Smaller Numbers After Self
    // Count smaller elements to the right using Fenwick Tree
    // Time: O(n log n), Space: O(n)

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        List<Integer> result = new ArrayList<>();

        // Coordinate compression
        int[] sortedNums = nums.clone();
        Arrays.sort(sortedNums);
        Map<Integer, Integer> ranks = new HashMap<>();
        int rank = 0;
        for (int num : sortedNums) {
            if (!ranks.containsKey(num)) {
                ranks.put(num, ++rank);
            }
        }

        // Binary Indexed Tree (Fenwick Tree)
        int[] BIT = new int[rank + 1];

        for (int i = n - 1; i >= 0; i--) {
            int currentRank = ranks.get(nums[i]);
            result.add(queryBIT(BIT, currentRank - 1));
            updateBIT(BIT, currentRank, 1);
        }

        Collections.reverse(result);
        return result;
    }

    private void updateBIT(int[] BIT, int index, int value) {
        while (index < BIT.length) {
            BIT[index] += value;
            index += index & -index;
        }
    }

    private int queryBIT(int[] BIT, int index) {
        int sum = 0;
        while (index > 0) {
            sum += BIT[index];
            index -= index & -index;
        }
        return sum;
    }


    // 29.3: Data Stream as Disjoint Intervals
    // Maintain disjoint intervals from data stream
    // Time: O(log n) per operation, Space: O(n)

    class SummaryRanges {
        private TreeSet<int[]> intervals;
        private Set<Integer> values;

        public SummaryRanges() {
            intervals = new TreeSet<>((a, b) -> a[0] - b[0]);
            values = new HashSet<>();
        }

        public void addNum(int value) {
            if (values.contains(value)) return;
            values.add(value);

            int[] newInterval = {value, value};
            int[] floor = intervals.floor(newInterval);
            int[] ceiling = intervals.ceiling(newInterval);

            // Merge with left interval
            if (floor != null && floor[1] + 1 == value) {
                intervals.remove(floor);
                newInterval[0] = floor[0];
            }

            // Merge with right interval
            if (ceiling != null && value + 1 == ceiling[0]) {
                intervals.remove(ceiling);
                newInterval[1] = ceiling[1];
            }

            intervals.add(newInterval);
        }

        public int[][] getIntervals() {
            return intervals.toArray(new int[intervals.size()][]);
        }
    }


    // PATTERN 28: UNION FIND
    // Use Case: Dynamic connectivity, cycle detection


    // 28.1: Number of Provinces
    // Find connected components in undirected graph
    // Time: O(n^2 α(n)), Space: O(n)

    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        UnionFind uf = new UnionFind(n);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1) {
                    uf.union(i, j);
                }
            }
        }
        return uf.count;
    }


    // 28.2: Redundant Connection
    // Find edge that creates cycle in undirected graph
    // Time: O(n α(n)), Space: O(n)

    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        UnionFind uf = new UnionFind(n + 1);

        for (int[] edge : edges) {
            if (!uf.union(edge[0], edge[1])) {
                return edge;
            }
        }
        return new int[0];
    }


    // 28.3: Number of Islands II
    // Count islands after each addLand operation
    // Time: O(k α(mn)), Space: O(mn)

    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> result = new ArrayList<>();
        UnionFind uf = new UnionFind(m * n);
        boolean[][] grid = new boolean[m][n];

        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        for (int[] pos : positions) {
            int row = pos[0], col = pos[1];

            if (grid[row][col]) {
                result.add(uf.count);
                continue;
            }

            grid[row][col] = true;
            uf.count++;
            int index = row * n + col;

            for (int[] dir : directions) {
                int newRow = row + dir[0];
                int newCol = col + dir[1];

                if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && grid[newRow][newCol]) {
                    int neighborIndex = newRow * n + newCol;
                    uf.union(index, neighborIndex);
                }
            }
            result.add(uf.count);
        }
        return result;
    }

    // Union-Find Data Structure with path compression and union by rank
    class UnionFind {
        int[] parent;
        int[] rank;
        int count;

        public UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            count = n;

            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // Path compression
            }
            return parent[x];
        }

        public boolean union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX == rootY) return false; // Already connected

            // Union by rank
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            count--;
            return true;
        }
    }


    // PATTERN 27: TOPOLOGICAL SORT (GRAPH)
    // Use Case: Dependency resolution, course scheduling


    // 27.1: Course Schedule II
    // Find valid course order using Kahn's algorithm
    // Time: O(V + E), Space: O(V + E)

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        // Build adjacency list
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }

        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            graph.get(pre[1]).add(pre[0]);
            indegree[pre[0]]++;
        }

        // Topological sort using BFS (Kahn's algorithm)
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        int[] result = new int[numCourses];
        int index = 0;

        while (!queue.isEmpty()) {
            int course = queue.poll();
            result[index++] = course;

            for (int neighbor : graph.get(course)) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return index == numCourses ? result : new int[0];
    }


    // 27.2: Alien Dictionary
    // Reconstruct character order from alien dictionary
    // Time: O(C) where C is total characters, Space: O(1)

    public String alienOrder(String[] words) {
        // Build graph
        Map<Character, Set<Character>> graph = new HashMap<>();
        Map<Character, Integer> indegree = new HashMap<>();

        // Initialize
        for (String word : words) {
            for (char c : word.toCharArray()) {
                graph.putIfAbsent(c, new HashSet<>());
                indegree.putIfAbsent(c, 0);
            }
        }

        // Build edges
        for (int i = 0; i < words.length - 1; i++) {
            String word1 = words[i], word2 = words[i + 1];
            int minLen = Math.min(word1.length(), word2.length());

            // Check for invalid order
            if (word1.length() > word2.length() && word1.startsWith(word2)) {
                return "";
            }

            for (int j = 0; j < minLen; j++) {
                char c1 = word1.charAt(j), c2 = word2.charAt(j);
                if (c1 != c2) {
                    if (!graph.get(c1).contains(c2)) {
                        graph.get(c1).add(c2);
                        indegree.put(c2, indegree.get(c2) + 1);
                    }
                    break;
                }
            }
        }

        // Topological sort
        Queue<Character> queue = new LinkedList<>();
        for (char c : indegree.keySet()) {
            if (indegree.get(c) == 0) {
                queue.offer(c);
            }
        }

        StringBuilder result = new StringBuilder();
        while (!queue.isEmpty()) {
            char c = queue.poll();
            result.append(c);

            for (char neighbor : graph.get(c)) {
                indegree.put(neighbor, indegree.get(neighbor) - 1);
                if (indegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return result.length() == indegree.size() ? result.toString() : "";
    }


    // 27.3: Sequence Reconstruction
    // Check if sequence can be uniquely reconstructed
    // Time: O(n + s), Space: O(n)

    public boolean sequenceReconstruction(int[] nums, List<List<Integer>> sequences) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        Map<Integer, Integer> indegree = new HashMap<>();

        // Initialize graph
        for (int num : nums) {
            graph.putIfAbsent(num, new HashSet<>());
            indegree.putIfAbsent(num, 0);
        }

        // Build graph from sequences
        for (List<Integer> seq : sequences) {
            for (int i = 0; i < seq.size() - 1; i++) {
                int from = seq.get(i), to = seq.get(i + 1);
                if (graph.get(from).add(to)) {
                    indegree.put(to, indegree.get(to) + 1);
                }
            }
        }

        // Topological sort
        Queue<Integer> queue = new LinkedList<>();
        for (int num : nums) {
            if (indegree.get(num) == 0) {
                queue.offer(num);
            }
        }

        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            if (queue.size() > 1) return false; // Not unique

            int current = queue.poll();
            result.add(current);

            for (int neighbor : graph.get(current)) {
                indegree.put(neighbor, indegree.get(neighbor) - 1);
                if (indegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // Check if result matches nums
        if (result.size() != nums.length) return false;
        for (int i = 0; i < nums.length; i++) {
            if (result.get(i) != nums[i]) return false;
        }
        return true;
    }


    // PATTERN 26: TRIE
    // Use Case: Prefix searching, dictionary problems


    // 26.1: Implement Trie (Prefix Tree)
    // Time: O(L) for insert/search/startsWith, Space: O(N//L)

    class Trie {
        class TrieNode {
            TrieNode[] children;
            boolean isEnd;

            public TrieNode() {
                children = new TrieNode[26];
                isEnd = false;
            }
        }

        private TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        public void insert(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new TrieNode();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            TrieNode node = searchPrefix(word);
            return node != null && node.isEnd;
        }

        public boolean startsWith(String prefix) {
            return searchPrefix(prefix) != null;
        }

        private TrieNode searchPrefix(String prefix) {
            TrieNode node = root;
            for (char c : prefix.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return null;
                }
                node = node.children[index];
            }
            return node;
        }
    }


    // 26.2: Word Search II
    // Find all words from dictionary in board using Trie
    // Time: O(m//n//4^L), Space: O(k//L) where k is number of words

    public List<String> findWords(char[][] board, String[] words) {
        List<String> result = new ArrayList<>();
        Trie trie = new Trie();

        // Build trie with all words
        for (String word : words) {
            trie.insert(word);
        }

        int rows = board.length, cols = board[0].length;
        boolean[][] visited = new boolean[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dfsWordSearch(board, i, j, trie.root, new StringBuilder(), result, visited);
            }
        }
        return result;
    }

    private void dfsWordSearch(char[][] board, int i, int j, Trie.TrieNode node, StringBuilder current, List<String> result, boolean[][] visited) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || visited[i][j]) {
            return;
        }

        char c = board[i][j];
        int index = c - 'a';
        if (node.children[index] == null) return;

        node = node.children[index];
        current.append(c);
        visited[i][j] = true;

        if (node.isEnd) {
            result.add(current.toString());
            node.isEnd = false; // Avoid duplicates
        }

        // Explore neighbors
        dfsWordSearch(board, i + 1, j, node, current, result, visited);
        dfsWordSearch(board, i - 1, j, node, current, result, visited);
        dfsWordSearch(board, i, j + 1, node, current, result, visited);
        dfsWordSearch(board, i, j - 1, node, current, result, visited);

        // Backtrack
        current.deleteCharAt(current.length() - 1);
        visited[i][j] = false;
    }


    // 26.3: Replace Words
    // Replace words with their shortest root
    // Time: O(n//L), Space: O(n//L)

    public String replaceWords(List<String> dictionary, String sentence) {
        Trie trie = new Trie();

        // Build trie with dictionary
        for (String root : dictionary) {
            trie.insert(root);
        }

        String[] words = sentence.split(" ");
        StringBuilder result = new StringBuilder();

        for (String word : words) {
            if (result.length() > 0) result.append(" ");
            String root = findShortestRoot(trie, word);
            result.append(root != null ? root : word);
        }

        return result.toString();
    }

    private String findShortestRoot(Trie trie, String word) {
        Trie.TrieNode node = trie.root;
        StringBuilder prefix = new StringBuilder();

        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) return null;

            node = node.children[index];
            prefix.append(c);

            if (node.isEnd) return prefix.toString();
        }
        return null;
    }


    // PATTERN 25: BACKTRACKING
    // Use Case: Permutations, combinations, constraint satisfaction


    // 25.1: Permutations
    // Generate all permutations of distinct integers
    // Time: O(n//n!), Space: O(n)

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackPermute(nums, new ArrayList<>(), new boolean[nums.length], result);
        return result;
    }

    private void backtrackPermute(int[] nums, List<Integer> current, boolean[] used, List<List<Integer>> result) {
        if (current.size() == nums.length) {
            result.add(new ArrayList<>(current));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                used[i] = true;
                current.add(nums[i]);
                backtrackPermute(nums, current, used, result);
                current.remove(current.size() - 1);
                used[i] = false;
            }
        }
    }


    // 25.2: Combination Sum
    // Find all combinations that sum to target (reuse allowed)
    // Time: O(n^(t/m)), Space: O(t/m)

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackCombinationSum(candidates, target, 0, new ArrayList<>(), result);
        return result;
    }

    private void backtrackCombinationSum(int[] candidates, int target, int start, List<Integer> current, List<List<Integer>> result) {
        if (target < 0) return;
        if (target == 0) {
            result.add(new ArrayList<>(current));
            return;
        }

        for (int i = start; i < candidates.length; i++) {
            current.add(candidates[i]);
            backtrackCombinationSum(candidates, target - candidates[i], i, current, result);
            current.remove(current.size() - 1);
        }
    }


    // 25.3: N-Queens
    // Place n queens on n×n board so no two attack each other
    // Time: O(n!), Space: O(n^2)

    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        backtrackNQueens(board, 0, result);
        return result;
    }

    private void backtrackNQueens(char[][] board, int row, List<List<String>> result) {
        if (row == board.length) {
            result.add(constructSolution(board));
            return;
        }

        for (int col = 0; col < board.length; col++) {
            if (isValidQueen(board, row, col)) {
                board[row][col] = 'Q';
                backtrackNQueens(board, row + 1, result);
                board[row][col] = '.';
            }
        }
    }

    private boolean isValidQueen(char[][] board, int row, int col) {
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') return false;
        }

        // Check diagonal \
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') return false;
        }

        // Check diagonal /
        for (int i = row - 1, j = col + 1; i >= 0 && j < board.length; i--, j++) {
            if (board[i][j] == 'Q') return false;
        }

        return true;
    }

    private List<String> constructSolution(char[][] board) {
        List<String> solution = new ArrayList<>();
        for (char[] row : board) {
            solution.add(new String(row));
        }
        return solution;
    }


    // PATTERN 24: PALINDROMIC SUBSEQUENCE (DYNAMIC PROGRAMMING)
    // Use Case: Palindrome problems, string manipulation


    // 24.1: Longest Palindromic Subsequence
    // Find length of longest palindromic subsequence
    // Time: O(n^2), Space: O(n^2)

    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];

        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = 2 + dp[i + 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }


    // 24.2: Palindromic Substrings
    // Count all palindromic substrings
    // Time: O(n^2), Space: O(1)

    public int countSubstrings(String s) {
        int count = 0;
        int n = s.length();

        for (int i = 0; i < n; i++) {
            // Odd length palindromes
            count += expandAroundCenter(s, i, i);
            // Even length palindromes
            count += expandAroundCenter(s, i, i + 1);
        }
        return count;
    }

    private int expandAroundCenter(String s, int left, int right) {
        int count = 0;
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            count++;
            left--;
            right++;
        }
        return count;
    }


    // 24.3: Longest Palindromic Substring
    // Find the longest palindromic substring
    // Time: O(n^2), Space: O(1)

    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";

        int start = 0, end = 0;

        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenterLength(s, i, i);
            int len2 = expandAroundCenterLength(s, i, i + 1);
            int len = Math.max(len1, len2);

            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenterLength(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }


    // PATTERN 23: FIBONACCI (DYNAMIC PROGRAMMING)
    // Use Case: Sequence problems, counting ways


    // 23.1: Climbing Stairs
    // Count ways to climb n stairs (1 or 2 steps)
    // Time: O(n), Space: O(1)

    public int climbStairs(int n) {
        if (n <= 2) return n;

        int first = 1, second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }


    // 23.2: House Robber
    // Maximum amount without robbing adjacent houses
    // Time: O(n), Space: O(1)

    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];

        int prev2 = 0, prev1 = 0;
        for (int num : nums) {
            int current = Math.max(prev1, prev2 + num);
            prev2 = prev1;
            prev1 = current;
        }
        return prev1;
    }


    // 23.3: Min Cost Climbing Stairs
    // Find minimum cost to reach top
    // Time: O(n), Space: O(1)

    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int first = 0, second = 0;

        for (int i = 2; i <= n; i++) {
            int current = Math.min(first + cost[i - 2], second + cost[i - 1]);
            first = second;
            second = current;
        }
        return second;
    }


    // PATTERN 22: 0/1 KNAPSACK (DYNAMIC PROGRAMMING)
    // Use Case: Subset sum, partition problems, resource allocation


    // 22.1: Partition Equal Subset Sum
    // Check if array can be partitioned into two equal sum subsets
    // Time: O(n//sum), Space: O(sum)

    public boolean canPartition(int[] nums) {
        int totalSum = 0;
        for (int num : nums) totalSum += num;

        if (totalSum % 2 != 0) return false;

        int target = totalSum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;

        for (int num : nums) {
            for (int j = target; j >= num; j--) {
                dp[j] = dp[j] || dp[j - num];
            }
        }
        return dp[target];
    }


    // 22.2: Target Sum
    // Assign +/- to get target sum
    // Time: O(n//sum), Space: O(sum)

    public int findTargetSumWays(int[] nums, int target) {
        int totalSum = 0;
        for (int num : nums) totalSum += num;

        if (Math.abs(target) > totalSum) return 0;
        if ((totalSum + target) % 2 != 0) return 0;

        int subsetSum = (totalSum + target) / 2;
        int[] dp = new int[subsetSum + 1];
        dp[0] = 1;

        for (int num : nums) {
            for (int j = subsetSum; j >= num; j--) {
                dp[j] += dp[j - num];
            }
        }
        return dp[subsetSum];
    }


    // 22.3: Coin Change II
    // Count number of combinations to make amount
    // Time: O(n//amount), Space: O(amount)

    public int changeCoins(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;

        for (int coin : coins) {
            for (int j = coin; j <= amount; j++) {
                dp[j] += dp[j - coin];
            }
        }
        return dp[amount];
    }


    // PATTERN 21: GREEDY ALGORITHMS
    // Use Case: Optimization problems, local optimal choices


    // 21.1: Jump Game
    // Check if you can reach last index
    // Time: O(n), Space: O(1)

    public boolean canJump(int[] nums) {
        int maxReach = 0;

        for (int i = 0; i < nums.length; i++) {
            if (i > maxReach) return false;
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= nums.length - 1) return true;
        }
        return true;
    }


    // 21.2: Jump Game II
    // Find minimum jumps to reach end
    // Time: O(n), Space: O(1)

    public int jump(int[] nums) {
        int jumps = 0;
        int currentEnd = 0;
        int farthest = 0;

        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);

            if (i == currentEnd) {
                jumps++;
                currentEnd = farthest;

                if (currentEnd >= nums.length - 1) break;
            }
        }
        return jumps;
    }


    // 21.3: Gas Station
    // Find starting gas station to complete circuit
    // Time: O(n), Space: O(1)

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int totalGas = 0, totalCost = 0;
        int currentGas = 0, startIndex = 0;

        for (int i = 0; i < gas.length; i++) {
            totalGas += gas[i];
            totalCost += cost[i];
            currentGas += gas[i] - cost[i];

            if (currentGas < 0) {
                startIndex = i + 1;
                currentGas = 0;
            }
        }

        return totalGas >= totalCost ? startIndex : -1;
    }
}
