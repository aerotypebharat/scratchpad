/**
 * LeetCode Patterns Collection - 31 Patterns, 93 Problems with Java Solutions
 * REVERSE ORDER: Pattern 31 to Pattern 1 for practicing from the end first
 * GitHub: https://github.com/yourusername/leetcode-patterns
 */

import java.util.*;
import java.util.concurrent.*;

public class LeetCodePatternsReverse {
    
    public static void main(String[] args) {
        System.out.println("LeetCode Patterns Collection - REVERSE ORDER (31 to 1)");
        System.out.println("93 Problems - Practice from the end first!");
    }

    // =========================================================================
    // PATTERN 31: MULTI-THREAD
    // Use Case: Concurrency, synchronization
    // =========================================================================
    
    /**
     * 31.1: Print in Order
     */
    class Foo {
        private Semaphore firstDone = new Semaphore(0);
        private Semaphore secondDone = new Semaphore(0);

        public Foo() {}
        
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
    
    /**
     * 31.2: Print FooBar Alternately
     */
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
    
    /**
     * 31.3: Dining Philosophers
     */
    class DiningPhilosophers {
        private Semaphore[] forks = new Semaphore[5];
        private Semaphore dining = new Semaphore(4); // Allow only 4 philosophers to eat

        public DiningPhilosophers() {
            for (int i = 0; i < 5; i++) {
                forks[i] = new Semaphore(1);
            }
        }

        public void wantsToEat(int philosopher,
                               Runnable pickLeftFork,
                               Runnable pickRightFork,
                               Runnable eat,
                               Runnable putLeftFork,
                               Runnable putRightFork) throws InterruptedException {
            
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

    // =========================================================================
    // PATTERN 30: PREFIX SUM
    // Use Case: Range sum queries, subarray problems
    // =========================================================================
    
    /**
     * 30.1: Subarray Sum Equals K
     * Count number of subarrays with sum exactly k
     * Time: O(n), Space: O(n)
     */
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
    
    /**
     * 30.2: Continuous Subarray Sum
     * Check if there's subarray with sum multiple of k (size >= 2)
     * Time: O(n), Space: O(n)
     */
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
    
    /**
     * 30.3: Range Sum Query 2D - Immutable
     * Query sum of submatrix in O(1) time after O(mn) preprocessing
     */
    class NumMatrix {
        private int[][] prefixSum;

        public NumMatrix(int[][] matrix) {
            if (matrix.length == 0 || matrix[0].length == 0) return;
            
            int rows = matrix.length, cols = matrix[0].length;
            prefixSum = new int[rows + 1][cols + 1];
            
            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= cols; j++) {
                    prefixSum[i][j] = matrix[i - 1][j - 1] + 
                                     prefixSum[i - 1][j] + 
                                     prefixSum[i][j - 1] - 
                                     prefixSum[i - 1][j - 1];
                }
            }
        }
        
        public int sumRegion(int row1, int col1, int row2, int col2) {
            return prefixSum[row2 + 1][col2 + 1] - 
                   prefixSum[row1][col2 + 1] - 
                   prefixSum[row2 + 1][col1] + 
                   prefixSum[row1][col1];
        }
    }

    // =========================================================================
    // PATTERN 29: ORDERED SET
    // Use Case: Range queries, sorted data operations
    // =========================================================================
    
    /**
     * 29.1: Contains Duplicate III
     * Check if there are nearby indices with nearby values
     * Time: O(n log k), Space: O(k)
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        TreeSet<Long> set = new TreeSet<>();
        
        for (int i = 0; i < nums.length; i++) {
            Long floor = set.floor((long)nums[i] + valueDiff);
            Long ceiling = set.ceiling((long)nums[i] - valueDiff);
            
            if ((floor != null && floor >= nums[i]) || (ceiling != null && ceiling <= nums[i])) {
                return true;
            }
            
            set.add((long)nums[i]);
            
            // Maintain sliding window of size indexDiff
            if (i >= indexDiff) {
                set.remove((long)nums[i - indexDiff]);
            }
        }
        return false;
    }
    
    /**
     * 29.2: Count of Smaller Numbers After Self
     * Count smaller elements to the right using Fenwick Tree
     * Time: O(n log n), Space: O(n)
     */
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
    
    /**
     * 29.3: Data Stream as Disjoint Intervals
     * Maintain disjoint intervals from data stream
     * Time: O(log n) per operation, Space: O(n)
     */
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

    // =========================================================================
    // PATTERN 28: UNION FIND
    // Use Case: Dynamic connectivity, cycle detection
    // =========================================================================
    
    /**
     * 28.1: Number of Provinces
     * Find connected components in undirected graph
     * Time: O(n^2 α(n)), Space: O(n)
     */
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
    
    /**
     * 28.2: Redundant Connection
     * Find edge that creates cycle in undirected graph
     * Time: O(n α(n)), Space: O(n)
     */
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
    
    /**
     * 28.3: Number of Islands II
     * Count islands after each addLand operation
     * Time: O(k α(mn)), Space: O(mn)
     */
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

    // =========================================================================
    // PATTERN 27: TOPOLOGICAL SORT (GRAPH)
    // Use Case: Dependency resolution, course scheduling
    // =========================================================================
    
    /**
     * 27.1: Course Schedule II
     * Find valid course order using Kahn's algorithm
     * Time: O(V + E), Space: O(V + E)
     */
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
    
    /**
     * 27.2: Alien Dictionary
     * Reconstruct character order from alien dictionary
     * Time: O(C) where C is total characters, Space: O(1)
     */
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
    
    /**
     * 27.3: Sequence Reconstruction
     * Check if sequence can be uniquely reconstructed
     * Time: O(n + s), Space: O(n)
     */
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

    // =========================================================================
    // PATTERN 26: TRIE
    // Use Case: Prefix searching, dictionary problems
    // =========================================================================
    
    /**
     * 26.1: Implement Trie (Prefix Tree)
     * Time: O(L) for insert/search/startsWith, Space: O(N*L)
     */
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
    
    /**
     * 26.2: Word Search II
     * Find all words from dictionary in board using Trie
     * Time: O(m*n*4^L), Space: O(k*L) where k is number of words
     */
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
    
    private void dfsWordSearch(char[][] board, int i, int j, TrieNode node, StringBuilder current, 
                              List<String> result, boolean[][] visited) {
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
    
    /**
     * 26.3: Replace Words
     * Replace words with their shortest root
     * Time: O(n*L), Space: O(n*L)
     */
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
        TrieNode node = trie.root;
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

    // =========================================================================
    // PATTERN 25: BACKTRACKING
    // Use Case: Permutations, combinations, constraint satisfaction
    // =========================================================================
    
    /**
     * 25.1: Permutations
     * Generate all permutations of distinct integers
     * Time: O(n*n!), Space: O(n)
     */
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
    
    /**
     * 25.2: Combination Sum
     * Find all combinations that sum to target (reuse allowed)
     * Time: O(n^(t/m)), Space: O(t/m)
     */
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
    
    /**
     * 25.3: N-Queens
     * Place n queens on n×n board so no two attack each other
     * Time: O(n!), Space: O(n^2)
     */
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

    // =========================================================================
    // PATTERN 24: PALINDROMIC SUBSEQUENCE (DYNAMIC PROGRAMMING)
    // Use Case: Palindrome problems, string manipulation
    // =========================================================================
    
    /**
     * 24.1: Longest Palindromic Subsequence
     * Find length of longest palindromic subsequence
     * Time: O(n^2), Space: O(n^2)
     */
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
    
    /**
     * 24.2: Palindromic Substrings
     * Count all palindromic substrings
     * Time: O(n^2), Space: O(1)
     */
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
    
    /**
     * 24.3: Longest Palindromic Substring
     * Find the longest palindromic substring
     * Time: O(n^2), Space: O(1)
     */
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

    // =========================================================================
    // PATTERN 23: FIBONACCI (DYNAMIC PROGRAMMING)
    // Use Case: Sequence problems, counting ways
    // =========================================================================
    
    /**
     * 23.1: Climbing Stairs
     * Count ways to climb n stairs (1 or 2 steps)
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 23.2: House Robber
     * Maximum amount without robbing adjacent houses
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 23.3: Min Cost Climbing Stairs
     * Find minimum cost to reach top
     * Time: O(n), Space: O(1)
     */
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

    // =========================================================================
    // PATTERN 22: 0/1 KNAPSACK (DYNAMIC PROGRAMMING)
    // Use Case: Subset sum, partition problems, resource allocation
    // =========================================================================
    
    /**
     * 22.1: Partition Equal Subset Sum
     * Check if array can be partitioned into two equal sum subsets
     * Time: O(n*sum), Space: O(sum)
     */
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
    
    /**
     * 22.2: Target Sum
     * Assign +/- to get target sum
     * Time: O(n*sum), Space: O(sum)
     */
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
    
    /**
     * 22.3: Coin Change II
     * Count number of combinations to make amount
     * Time: O(n*amount), Space: O(amount)
     */
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

    // =========================================================================
    // PATTERN 21: GREEDY ALGORITHMS
    // Use Case: Optimization problems, local optimal choices
    // =========================================================================
    
    /**
     * 21.1: Jump Game
     * Check if you can reach last index
     * Time: O(n), Space: O(1)
     */
    public boolean canJump(int[] nums) {
        int maxReach = 0;
        
        for (int i = 0; i < nums.length; i++) {
            if (i > maxReach) return false;
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= nums.length - 1) return true;
        }
        return true;
    }
    
    /**
     * 21.2: Jump Game II
     * Find minimum jumps to reach end
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 21.3: Gas Station
     * Find starting gas station to complete circuit
     * Time: O(n), Space: O(1)
     */
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

    // =========================================================================
    // PATTERN 20: K-WAY MERGE
    // Use Case: Merging multiple sorted arrays/lists
    // =========================================================================
    
    /**
     * 20.1: Merge K Sorted Lists
     * Time: O(N log k), Space: O(k)
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        
        PriorityQueue<ListNode> minHeap = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        // Add head of all lists to heap
        for (ListNode list : lists) {
            if (list != null) {
                minHeap.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!minHeap.isEmpty()) {
            ListNode node = minHeap.poll();
            current.next = node;
            current = current.next;
            
            if (node.next != null) {
                minHeap.offer(node.next);
            }
        }
        return dummy.next;
    }
    
    /**
     * 20.2: Kth Smallest Element in Sorted Matrix
     * Time: O(k log n), Space: O(n)
     */
    public int kthSmallestMatrix(int[][] matrix, int k) {
        int n = matrix.length;
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        
        // Add first element of each row to heap
        for (int i = 0; i < n; i++) {
            minHeap.offer(new int[]{matrix[i][0], i, 0});
        }
        
        int count = 0;
        while (!minHeap.isEmpty()) {
            int[] element = minHeap.poll();
            int value = element[0], row = element[1], col = element[2];
            count++;
            
            if (count == k) return value;
            
            // Add next element from same row
            if (col + 1 < n) {
                minHeap.offer(new int[]{matrix[row][col + 1], row, col + 1});
            }
        }
        return -1;
    }
    
    /**
     * 20.3: Find K Pairs with Smallest Sums
     * Time: O(k log k), Space: O(k)
     */
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0 || k == 0) return result;
        
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> (a[0] + a[1]) - (b[0] + b[1]));
        
        // Add pairs (nums1[i], nums2[0]) for all i
        for (int i = 0; i < Math.min(nums1.length, k); i++) {
            minHeap.offer(new int[]{nums1[i], nums2[0], 0});
        }
        
        while (k-- > 0 && !minHeap.isEmpty()) {
            int[] current = minHeap.poll();
            result.add(Arrays.asList(current[0], current[1]));
            
            int nums2Index = current[2];
            if (nums2Index + 1 < nums2.length) {
                minHeap.offer(new int[]{current[0], nums2[nums2Index + 1], nums2Index + 1});
            }
        }
        return result;
    }

    // =========================================================================
    // PATTERN 19: TOP 'K' ELEMENTS
    // Use Case: K largest/smallest, frequent elements
    // =========================================================================
    
    /**
     * 19.1: Top K Frequent Elements
     * Time: O(n log k), Space: O(n + k)
     */
    public int[] topKFrequent(int[] nums, int k) {
        // Count frequencies
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }
        
        // Min heap to keep top k elements
        PriorityQueue<Map.Entry<Integer, Integer>> minHeap = 
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        
        for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            minHeap.offer(entry);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        
        // Extract results
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = minHeap.poll().getKey();
        }
        return result;
    }
    
    /**
     * 19.2: Kth Largest Element in Array
     * Time: O(n log k), Space: O(k)
     */
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        for (int num : nums) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        return minHeap.peek();
    }
    
    /**
     * 19.3: K Closest Points to Origin
     * Time: O(n log k), Space: O(k)
     */
    public int[][] kClosest(int[][] points, int k) {
        // Max heap to keep k closest points
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(distance(b), distance(a))
        );
        
        for (int[] point : points) {
            maxHeap.offer(point);
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }
        
        int[][] result = new int[k][2];
        for (int i = 0; i < k; i++) {
            result[i] = maxHeap.poll();
        }
        return result;
    }
    
    private int distance(int[] point) {
        return point[0] * point[0] + point[1] * point[1];
    }

    // =========================================================================
    // PATTERN 18: BITWISE XOR
    // Use Case: Finding unique elements, bit manipulation
    // =========================================================================
    
    /**
     * 18.1: Single Number
     * Find the number that appears once (others appear twice)
     * Time: O(n), Space: O(1)
     */
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }
    
    /**
     * 18.2: Missing Number
     * Find missing number in range [0, n]
     * Time: O(n), Space: O(1)
     */
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int result = n; // Initialize with n since it's missing from indices
        
        for (int i = 0; i < n; i++) {
            result ^= i ^ nums[i];
        }
        return result;
    }
    
    /**
     * 18.3: Complement of Base 10 Integer
     * Flip all bits (excluding leading zeros)
     * Time: O(1), Space: O(1)
     */
    public int bitwiseComplement(int n) {
        if (n == 0) return 1;
        
        int bitCount = 0;
        int num = n;
        while (num > 0) {
            bitCount++;
            num >>= 1;
        }
        
        int allOnes = (1 << bitCount) - 1;
        return n ^ allOnes;
    }

    // =========================================================================
    // PATTERN 17: MODIFIED BINARY SEARCH
    // Use Case: Rotated arrays, unknown order, bitonic arrays
    // =========================================================================
    
    /**
     * 17.1: Search in Rotated Sorted Array
     * Time: O(log n), Space: O(1)
     */
    public int searchRotated(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) return mid;
            
            // Check which side is sorted
            if (nums[left] <= nums[mid]) { // Left side is sorted
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else { // Right side is sorted
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
    
    /**
     * 17.2: Find First and Last Position in Sorted Array
     * Time: O(log n), Space: O(1)
     */
    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1, -1};
        if (nums.length == 0) return result;
        
        result[0] = findFirstPosition(nums, target);
        result[1] = findLastPosition(nums, target);
        return result;
    }
    
    private int findFirstPosition(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int index = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
            
            if (nums[mid] == target) index = mid;
        }
        return index;
    }
    
    private int findLastPosition(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int index = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
            
            if (nums[mid] == target) index = mid;
        }
        return index;
    }
    
    /**
     * 17.3: Find Peak Element
     * Time: O(log n), Space: O(1)
     */
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // =========================================================================
    // PATTERN 16: SUBSETS
    // Use Case: Combinations, permutations, power set
    // =========================================================================
    
    /**
     * 16.1: Subsets
     * Generate all possible subsets
     * Time: O(n * 2^n), Space: O(n)
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackSubsets(nums, 0, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrackSubsets(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));
        
        for (int i = start; i < nums.length; i++) {
            current.add(nums[i]);
            backtrackSubsets(nums, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }
    
    /**
     * 16.2: Subsets II (with duplicates)
     * Time: O(n * 2^n), Space: O(n)
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums); // Sort to handle duplicates
        List<List<Integer>> result = new ArrayList<>();
        backtrackSubsetsWithDup(nums, 0, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrackSubsetsWithDup(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));
        
        for (int i = start; i < nums.length; i++) {
            // Skip duplicates
            if (i > start && nums[i] == nums[i - 1]) continue;
            
            current.add(nums[i]);
            backtrackSubsetsWithDup(nums, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }
    
    /**
     * 16.3: Letter Case Permutation
     * Time: O(n * 2^n), Space: O(n)
     */
    public List<String> letterCasePermutation(String s) {
        List<String> result = new ArrayList<>();
        backtrackLetterCase(s.toCharArray(), 0, result);
        return result;
    }
    
    private void backtrackLetterCase(char[] chars, int index, List<String> result) {
        if (index == chars.length) {
            result.add(new String(chars));
            return;
        }
        
        // Skip if digit
        if (Character.isDigit(chars[index])) {
            backtrackLetterCase(chars, index + 1, result);
            return;
        }
        
        // Lower case branch
        chars[index] = Character.toLowerCase(chars[index]);
        backtrackLetterCase(chars, index + 1, result);
        
        // Upper case branch
        chars[index] = Character.toUpperCase(chars[index]);
        backtrackLetterCase(chars, index + 1, result);
    }

    // =========================================================================
    // PATTERN 15: TWO HEAPS
    // Use Case: Finding median, partitioning data streams
    // =========================================================================
    
    /**
     * 15.1: Find Median from Data Stream
     * Time: O(log n) per operation, Space: O(n)
     */
    class MedianFinder {
        private PriorityQueue<Integer> maxHeap; // lower half
        private PriorityQueue<Integer> minHeap; // upper half

        public MedianFinder() {
            maxHeap = new PriorityQueue<>((a, b) -> b - a);
            minHeap = new PriorityQueue<>();
        }
        
        public void addNum(int num) {
            // Add to appropriate heap
            if (maxHeap.isEmpty() || num <= maxHeap.peek()) {
                maxHeap.offer(num);
            } else {
                minHeap.offer(num);
            }
            
            // Balance heaps
            if (maxHeap.size() > minHeap.size() + 1) {
                minHeap.offer(maxHeap.poll());
            } else if (minHeap.size() > maxHeap.size()) {
                maxHeap.offer(minHeap.poll());
            }
        }
        
        public double findMedian() {
            if (maxHeap.size() == minHeap.size()) {
                return (maxHeap.peek() + minHeap.peek()) / 2.0;
            } else {
                return maxHeap.peek();
            }
        }
    }
    
    /**
     * 15.2: Sliding Window Median
     * Time: O(n log k), Space: O(k)
     */
    public double[] medianSlidingWindow(int[] nums, int k) {
        double[] result = new double[nums.length - k + 1];
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> Integer.compare(b, a));
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        for (int i = 0; i < nums.length; i++) {
            // Add new element
            if (maxHeap.isEmpty() || nums[i] <= maxHeap.peek()) {
                maxHeap.offer(nums[i]);
            } else {
                minHeap.offer(nums[i]);
            }
            
            // Remove element leaving window
            if (i >= k) {
                int toRemove = nums[i - k];
                if (toRemove <= maxHeap.peek()) {
                    maxHeap.remove(toRemove);
                } else {
                    minHeap.remove(toRemove);
                }
            }
            
            // Balance heaps
            balanceHeaps(maxHeap, minHeap);
            
            // Calculate median
            if (i >= k - 1) {
                if (maxHeap.size() == minHeap.size()) {
                    result[i - k + 1] = ((double)maxHeap.peek() + (double)minHeap.peek()) / 2.0;
                } else {
                    result[i - k + 1] = maxHeap.peek();
                }
            }
        }
        return result;
    }
    
    private void balanceHeaps(PriorityQueue<Integer> maxHeap, PriorityQueue<Integer> minHeap) {
        while (maxHeap.size() > minHeap.size() + 1) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }
    
    /**
     * 15.3: IPO
     * Maximize capital by selecting projects
     * Time: O(n log n), Space: O(n)
     */
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int n = profits.length;
        int[][] projects = new int[n][2];
        
        for (int i = 0; i < n; i++) {
            projects[i][0] = capital[i];
            projects[i][1] = profits[i];
        }
        
        // Sort by capital required
        Arrays.sort(projects, (a, b) -> a[0] - b[0]);
        
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
        int i = 0;
        
        while (k-- > 0) {
            // Add all affordable projects to max heap
            while (i < n && projects[i][0] <= w) {
                maxHeap.offer(projects[i][1]);
                i++;
            }
            
            if (maxHeap.isEmpty()) break;
            
            // Select project with maximum profit
            w += maxHeap.poll();
        }
        return w;
    }

    // =========================================================================
    // PATTERN 14: ISLAND (MATRIX TRAVERSAL)
    // Use Case: Grid problems, connected components in matrix
    // =========================================================================
    
    /**
     * 14.1: Max Area of Island
     * Time: O(m*n), Space: O(m*n) for recursion stack
     */
    public int maxAreaOfIsland(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int maxArea = 0;
        int rows = grid.length, cols = grid[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    maxArea = Math.max(maxArea, dfsIslandArea(grid, i, j));
                }
            }
        }
        return maxArea;
    }
    
    private int dfsIslandArea(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 1) {
            return 0;
        }
        
        grid[i][j] = 0; // Mark as visited
        
        return 1 + dfsIslandArea(grid, i + 1, j) + dfsIslandArea(grid, i - 1, j) + 
                   dfsIslandArea(grid, i, j + 1) + dfsIslandArea(grid, i, j - 1);
    }
    
    /**
     * 14.2: Number of Closed Islands
     * Time: O(m*n), Space: O(m*n)
     */
    public int closedIsland(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int rows = grid.length, cols = grid[0].length;
        int count = 0;
        
        // Mark boundary islands (they are not closed)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if ((i == 0 || j == 0 || i == rows - 1 || j == cols - 1) && grid[i][j] == 0) {
                    dfsClosedIslands(grid, i, j);
                }
            }
        }
        
        // Count closed islands
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 0) {
                    count++;
                    dfsClosedIslands(grid, i, j);
                }
            }
        }
        return count;
    }
    
    private void dfsClosedIslands(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 0) {
            return;
        }
        
        grid[i][j] = 1; // Mark as visited
        
        dfsClosedIslands(grid, i + 1, j);
        dfsClosedIslands(grid, i - 1, j);
        dfsClosedIslands(grid, i, j + 1);
        dfsClosedIslands(grid, i, j - 1);
    }
    
    /**
     * 14.3: Surrounded Regions
     * Time: O(m*n), Space: O(m*n)
     */
    public void solveSurroundedRegions(char[][] board) {
        if (board == null || board.length == 0) return;
        
        int rows = board.length, cols = board[0].length;
        
        // Mark boundary 'O's and connected regions
        for (int i = 0; i < rows; i++) {
            if (board[i][0] == 'O') dfsSurrounded(board, i, 0);
            if (board[i][cols - 1] == 'O') dfsSurrounded(board, i, cols - 1);
        }
        
        for (int j = 0; j < cols; j++) {
            if (board[0][j] == 'O') dfsSurrounded(board, 0, j);
            if (board[rows - 1][j] == 'O') dfsSurrounded(board, rows - 1, j);
        }
        
        // Flip surrounded 'O's to 'X' and restore marked ones
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == 'M') {
                    board[i][j] = 'O';
                }
            }
        }
    }
    
    private void dfsSurrounded(char[][] board, int i, int j) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != 'O') {
            return;
        }
        
        board[i][j] = 'M'; // Mark as connected to boundary
        
        dfsSurrounded(board, i + 1, j);
        dfsSurrounded(board, i - 1, j);
        dfsSurrounded(board, i, j + 1);
        dfsSurrounded(board, i, j - 1);
    }

    // =========================================================================
    // PATTERN 13: GRAPHS
    // Use Case: Connectivity, traversal, cycle detection
    // =========================================================================
    
    /**
     * 13.1: Number of Islands (DFS)
     * Time: O(m*n), Space: O(m*n)
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int count = 0;
        int rows = grid.length, cols = grid[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfsGrid(grid, i, j);
                }
            }
        }
        return count;
    }
    
    private void dfsGrid(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') {
            return;
        }
        
        grid[i][j] = '0'; // Mark as visited
        
        // Explore all four directions
        dfsGrid(grid, i + 1, j);
        dfsGrid(grid, i - 1, j);
        dfsGrid(grid, i, j + 1);
        dfsGrid(grid, i, j - 1);
    }
    
    /**
     * 13.2: Clone Graph
     * Time: O(V + E), Space: O(V)
     */
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        
        Map<Node, Node> visited = new HashMap<>();
        return dfsCloneGraph(node, visited);
    }
    
    private Node dfsCloneGraph(Node node, Map<Node, Node> visited) {
        if (visited.containsKey(node)) {
            return visited.get(node);
        }
        
        Node clone = new Node(node.val);
        visited.put(node, clone);
        
        for (Node neighbor : node.neighbors) {
            clone.neighbors.add(dfsCloneGraph(neighbor, visited));
        }
        
        return clone;
    }
    
    /**
     * 13.3: Course Schedule (Cycle Detection)
     * Time: O(V + E), Space: O(V + E)
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // Build adjacency list
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        for (int[] pre : prerequisites) {
            graph.get(pre[1]).add(pre[0]);
        }
        
        int[] visited = new int[numCourses]; // 0: unvisited, 1: visiting, 2: visited
        
        for (int i = 0; i < numCourses; i++) {
            if (hasCycleDFS(graph, visited, i)) {
                return false;
            }
        }
        return true;
    }
    
    private boolean hasCycleDFS(List<List<Integer>> graph, int[] visited, int course) {
        if (visited[course] == 1) return true; // Cycle detected
        if (visited[course] == 2) return false; // Already processed
        
        visited[course] = 1; // Mark as visiting
        
        for (int neighbor : graph.get(course)) {
            if (hasCycleDFS(graph, visited, neighbor)) {
                return true;
            }
        }
        
        visited[course] = 2; // Mark as visited
        return false;
    }

    // =========================================================================
    // PATTERN 12: TREE DEPTH FIRST SEARCH (DFS)
    // Use Case: Path sum, tree properties, backtracking in trees
    // =========================================================================
    
    /**
     * 12.1: Path Sum
     * Time: O(n), Space: O(h)
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        
        // Check if it's a leaf node and path sum equals target
        if (root.left == null && root.right == null && root.val == targetSum) {
            return true;
        }
        
        // Recursively check left and right subtrees
        return hasPathSum(root.left, targetSum - root.val) ||
               hasPathSum(root.right, targetSum - root.val);
    }
    
    /**
     * 12.2: Sum Root to Leaf Numbers
     * Time: O(n), Space: O(h)
     */
    public int sumNumbers(TreeNode root) {
        return dfsSumNumbers(root, 0);
    }
    
    private int dfsSumNumbers(TreeNode node, int currentSum) {
        if (node == null) return 0;
        
        currentSum = currentSum * 10 + node.val;
        
        // If leaf node, return the current number
        if (node.left == null && node.right == null) {
            return currentSum;
        }
        
        return dfsSumNumbers(node.left, currentSum) + dfsSumNumbers(node.right, currentSum);
    }
    
    /**
     * 12.3: Binary Tree Maximum Path Sum
     * Time: O(n), Space: O(h)
     */
    private int maxPathSumValue = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxPathSumDFS(root);
        return maxPathSumValue;
    }
    
    private int maxPathSumDFS(TreeNode node) {
        if (node == null) return 0;
        
        // Calculate max path sum from left and right children
        int leftMax = Math.max(maxPathSumDFS(node.left), 0);
        int rightMax = Math.max(maxPathSumDFS(node.right), 0);
        
        // Update global maximum with path through current node
        int pathThroughNode = node.val + leftMax + rightMax;
        maxPathSumValue = Math.max(maxPathSumValue, pathThroughNode);
        
        // Return maximum path sum starting from current node
        return node.val + Math.max(leftMax, rightMax);
    }

    // =========================================================================
    // PATTERN 11: TREE BREADTH FIRST SEARCH (BFS)
    // Use Case: Shortest path, level operations
    // =========================================================================
    
    /**
     * 11.1: Minimum Depth of Binary Tree
     * Time: O(n), Space: O(n)
     */
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 1;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                // Check if it's a leaf node
                if (node.left == null && node.right == null) {
                    return depth;
                }
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            depth++;
        }
        return depth;
    }
    
    /**
     * 11.2: Binary Tree Right Side View
     * Time: O(n), Space: O(n)
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                // Add the last node of each level
                if (i == levelSize - 1) {
                    result.add(node.val);
                }
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
        }
        return result;
    }
    
    /**
     * 11.3: Cousins in Binary Tree
     * Time: O(n), Space: O(n)
     */
    public boolean isCousins(TreeNode root, int x, int y) {
        if (root == null) return false;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            boolean foundX = false, foundY = false;
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                // Check if x and y are children of same parent
                if (node.left != null && node.right != null) {
                    if ((node.left.val == x && node.right.val == y) ||
                        (node.left.val == y && node.right.val == x)) {
                        return false;
                    }
                }
                
                if (node.val == x) foundX = true;
                if (node.val == y) foundY = true;
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            
            if (foundX && foundY) return true;
            if (foundX || foundY) return false;
        }
        return false;
    }

    // =========================================================================
    // PATTERN 10: LEVEL ORDER TRAVERSAL
    // Use Case: Tree traversal level by level
    // =========================================================================
    
    /**
     * 10.1: Binary Tree Level Order Traversal
     * Time: O(n), Space: O(n)
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(currentLevel);
        }
        return result;
    }
    
    /**
     * 10.2: Binary Tree Zigzag Level Order Traversal
     * Time: O(n), Space: O(n)
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean leftToRight = true;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (leftToRight) {
                    currentLevel.add(node.val);
                } else {
                    currentLevel.add(0, node.val);
                }
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(currentLevel);
            leftToRight = !leftToRight;
        }
        return result;
    }
    
    /**
     * 10.3: Average of Levels in Binary Tree
     * Time: O(n), Space: O(n)
     */
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            double levelSum = 0;
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                levelSum += node.val;
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(levelSum / levelSize);
        }
        return result;
    }

    // =========================================================================
    // PATTERN 9: HASH MAPS
    // Use Case: Frequency counting, lookups, caching
    // =========================================================================
    
    /**
     * 9.1: Two Sum
     * Time: O(n), Space: O(n)
     */
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> numMap = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numMap.containsKey(complement)) {
                return new int[]{numMap.get(complement), i};
            }
            numMap.put(nums[i], i);
        }
        return new int[]{-1, -1};
    }
    
    /**
     * 9.2: Group Anagrams
     * Time: O(n*k log k), Space: O(n*k)
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String sorted = new String(chars);
            
            if (!map.containsKey(sorted)) {
                map.put(sorted, new ArrayList<>());
            }
            map.get(sorted).add(str);
        }
        
        return new ArrayList<>(map.values());
    }
    
    /**
     * 9.3: Longest Consecutive Sequence
     * Time: O(n), Space: O(n)
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }
        
        int longest = 0;
        
        for (int num : numSet) {
            // Check if it's the start of a sequence
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                
                while (numSet.contains(currentNum + 1)) {
                    currentNum++;
                    currentStreak++;
                }
                
                longest = Math.max(longest, currentStreak);
            }
        }
        return longest;
    }

    // =========================================================================
    // PATTERN 8: MONOTONIC STACK
    // Use Case: Next greater/smaller element, stock span, histogram problems
    // =========================================================================
    
    /**
     * 8.1: Next Greater Element I
     * Time: O(n), Space: O(n)
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> nextGreater = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        
        for (int num : nums2) {
            while (!stack.isEmpty() && num > stack.peek()) {
                nextGreater.put(stack.pop(), num);
            }
            stack.push(num);
        }
        
        int[] result = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            result[i] = nextGreater.getOrDefault(nums1[i], -1);
        }
        return result;
    }
    
    /**
     * 8.2: Trapping Rain Water
     * Time: O(n), Space: O(n)
     */
    public int trapRainWater(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int totalWater = 0;
        
        for (int i = 0; i < height.length; i++) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int bottom = stack.pop();
                if (stack.isEmpty()) break;
                
                int distance = i - stack.peek() - 1;
                int boundedHeight = Math.min(height[i], height[stack.peek()]) - height[bottom];
                totalWater += distance * boundedHeight;
            }
            stack.push(i);
        }
        return totalWater;
    }
    
    /**
     * 8.3: Largest Rectangle in Histogram
     * Time: O(n), Space: O(n)
     */
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int n = heights.length;
        
        for (int i = 0; i <= n; i++) {
            int currentHeight = (i == n) ? 0 : heights[i];
            
            while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            stack.push(i);
        }
        return maxArea;
    }

    // =========================================================================
    // PATTERN 7: STACK
    // Use Case: LIFO operations, parsing, backtracking
    // =========================================================================
    
    /**
     * 7.1: Valid Parentheses
     * Time: O(n), Space: O(n)
     */
    public boolean isValidParentheses(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) return false;
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == ']' && top != '[') ||
                    (c == '}' && top != '{')) {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }
    
    /**
     * 7.2: Daily Temperatures
     * Time: O(n), Space: O(n)
     */
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int idx = stack.pop();
                result[idx] = i - idx;
            }
            stack.push(i);
        }
        return result;
    }
    
    /**
     * 7.3: Evaluate Reverse Polish Notation
     * Time: O(n), Space: O(n)
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        
        for (String token : tokens) {
            if (token.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (token.equals("-")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a - b);
            } else if (token.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (token.equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a / b);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    // =========================================================================
    // PATTERN 6: IN-PLACE REVERSAL OF LINKEDLIST
    // Use Case: Reverse linked lists or portions of them
    // =========================================================================
    
    /**
     * 6.1: Reverse Linked List
     * Time: O(n), Space: O(1)
     */
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode nextTemp = current.next;
            current.next = prev;
            prev = current;
            current = nextTemp;
        }
        return prev;
    }
    
    /**
     * 6.2: Reverse Linked List II
     * Time: O(n), Space: O(1)
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null || left == right) return head;
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        // Move to the node before reversal start
        for (int i = 1; i < left; i++) {
            prev = prev.next;
        }
        
        ListNode start = prev.next;
        ListNode then = start.next;
        
        // Reverse the sublist
        for (int i = 0; i < right - left; i++) {
            start.next = then.next;
            then.next = prev.next;
            prev.next = then;
            then = start.next;
        }
        
        return dummy.next;
    }
    
    /**
     * 6.3: Reverse Nodes in k-Group
     * Time: O(n), Space: O(1)
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode curr = dummy, prev = dummy, next = dummy;
        
        int count = 0;
        while (curr.next != null) {
            curr = curr.next;
            count++;
        }
        
        while (count >= k) {
            curr = prev.next;
            next = curr.next;
            
            for (int i = 1; i < k; i++) {
                curr.next = next.next;
                next.next = prev.next;
                prev.next = next;
                next = curr.next;
            }
            prev = curr;
            count -= k;
        }
        return dummy.next;
    }

    // =========================================================================
    // PATTERN 5: CYCLIC SORT
    // Use Case: Arrays with numbers in range [1, n], missing/duplicate numbers
    // =========================================================================
    
    /**
     * 5.1: Find All Missing Numbers
     * Time: O(n), Space: O(1)
     */
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            int correctPos = nums[i] - 1;
            if (nums[i] != nums[correctPos]) {
                swap(nums, i, correctPos);
            } else {
                i++;
            }
        }
        
        List<Integer> missing = new ArrayList<>();
        for (i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                missing.add(i + 1);
            }
        }
        return missing;
    }
    
    /**
     * 5.2: Find Duplicate Number using Cyclic Sort
     * Time: O(n), Space: O(1)
     */
    public int findDuplicateCyclic(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            if (nums[i] != i + 1) {
                int correctPos = nums[i] - 1;
                if (nums[i] != nums[correctPos]) {
                    swap(nums, i, correctPos);
                } else {
                    return nums[i];
                }
            } else {
                i++;
            }
        }
        return -1;
    }
    
    /**
     * 5.3: First Missing Positive
     * Time: O(n), Space: O(1)
     */
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        int i = 0;
        
        while (i < n) {
            if (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            } else {
                i++;
            }
        }
        
        for (i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // =========================================================================
    // PATTERN 4: MERGE INTERVALS
    // Use Case: Overlapping intervals, scheduling problems
    // =========================================================================
    
    /**
     * 4.1: Merge Intervals
     * Time: O(n log n), Space: O(n)
     */
    public int[][] mergeIntervals(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        
        // Sort by start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> merged = new ArrayList<>();
        int[] current = intervals[0];
        merged.add(current);
        
        for (int[] interval : intervals) {
            if (interval[0] <= current[1]) {
                // Overlapping intervals, merge
                current[1] = Math.max(current[1], interval[1]);
            } else {
                // Non-overlapping interval, add to list
                current = interval;
                merged.add(current);
            }
        }
        
        return merged.toArray(new int[merged.size()][]);
    }
    
    /**
     * 4.2: Insert Interval
     * Time: O(n), Space: O(n)
     */
    public int[][] insertInterval(int[][] intervals, int[] newInterval) {
        List<int[]> result = new ArrayList<>();
        int i = 0;
        int n = intervals.length;
        
        // Add all intervals ending before newInterval starts
        while (i < n && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i++]);
        }
        
        // Merge overlapping intervals
        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.add(newInterval);
        
        // Add remaining intervals
        while (i < n) {
            result.add(intervals[i++]);
        }
        
        return result.toArray(new int[result.size()][]);
    }
    
    /**
     * 4.3: Meeting Rooms II
     * Time: O(n log n), Space: O(n)
     */
    public int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        int[] starts = new int[intervals.length];
        int[] ends = new int[intervals.length];
        
        for (int i = 0; i < intervals.length; i++) {
            starts[i] = intervals[i][0];
            ends[i] = intervals[i][1];
        }
        
        Arrays.sort(starts);
        Arrays.sort(ends);
        
        int rooms = 0;
        int endPtr = 0;
        
        for (int start : starts) {
            if (start < ends[endPtr]) {
                rooms++;
            } else {
                endPtr++;
            }
        }
        
        return rooms;
    }

    // =========================================================================
    // PATTERN 3: SLIDING WINDOW
    // Use Case: Subarrays/substrings with constraints, fixed/variable window
    // =========================================================================
    
    /**
     * 3.1: Maximum Average Subarray I
     * Time: O(n), Space: O(1)
     */
    public double findMaxAverage(int[] nums, int k) {
        double windowSum = 0;
        
        // Calculate first window
        for (int i = 0; i < k; i++) {
            windowSum += nums[i];
        }
        
        double maxSum = windowSum;
        
        // Slide the window
        for (int i = k; i < nums.length; i++) {
            windowSum += nums[i] - nums[i - k];
            maxSum = Math.max(maxSum, windowSum);
        }
        
        return maxSum / k;
    }
    
    /**
     * 3.2: Longest Substring Without Repeating Characters
     * Time: O(n), Space: O(min(m,n))
     */
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> charIndex = new HashMap<>();
        int left = 0, maxLength = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);
            
            // If character exists and is within current window
            if (charIndex.containsKey(c) && charIndex.get(c) >= left) {
                left = charIndex.get(c) + 1;
            }
            
            charIndex.put(c, right);
            maxLength = Math.max(maxLength, right - left + 1);
        }
        return maxLength;
    }
    
    /**
     * 3.3: Permutation in String
     * Time: O(l1 + l2), Space: O(1)
     */
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) return false;
        
        int[] s1Count = new int[26];
        int[] windowCount = new int[26];
        
        for (char c : s1.toCharArray()) {
            s1Count[c - 'a']++;
        }
        
        // Initial window
        for (int i = 0; i < s1.length(); i++) {
            windowCount[s2.charAt(i) - 'a']++;
        }
        
        if (Arrays.equals(s1Count, windowCount)) return true;
        
        // Slide window
        for (int i = s1.length(); i < s2.length(); i++) {
            windowCount[s2.charAt(i) - 'a']++;
            windowCount[s2.charAt(i - s1.length()) - 'a']--;
            
            if (Arrays.equals(s1Count, windowCount)) return true;
        }
        
        return false;
    }

    // =========================================================================
    // PATTERN 2: FAST & SLOW POINTERS
    // Use Case: Cycle detection, middle element, duplicate finding
    // =========================================================================
    
    /**
     * 2.1: Linked List Cycle Detection
     * Time: O(n), Space: O(1)
     */
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }
    
    /**
     * 2.2: Find Duplicate Number (Floyd's Tortoise and Hare)
     * Time: O(n), Space: O(1)
     */
    public int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];
        
        // Phase 1: Find intersection point
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        
        // Phase 2: Find entrance to cycle
        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
    
    /**
     * 2.3: Middle of Linked List
     * Time: O(n), Space: O(1)
     */
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    // =========================================================================
    // PATTERN 1: TWO POINTERS
    // Use Case: Sorted arrays/lists, pair searching, in-place operations
    // =========================================================================
    
    /**
     * 1.1: Two Sum II - Input Array Is Sorted
     * Time: O(n), Space: O(1)
     */
    public int[] twoSumSorted(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[]{left + 1, right + 1};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        return new int[]{-1, -1};
    }
    
    /**
     * 1.2: Container With Most Water
     * Time: O(n), Space: O(1)
     */
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int maxArea = 0;
        
        while (left < right) {
            int currentArea = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, currentArea);
            
            // Move the pointer with smaller height
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }
    
    /**
     * 1.3: Remove Duplicates from Sorted Array
     * Time: O(n), Space: O(1)
     */
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        
        int uniquePtr = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[uniquePtr]) {
                uniquePtr++;
                nums[uniquePtr] = nums[i];
            }
        }
        return uniquePtr + 1;
    }

    // =========================================================================
    // SUPPORTING DATA STRUCTURES
    // =========================================================================
    
    // ListNode definition for linked list problems
    class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    
    // TreeNode definition for tree problems
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Node definition for graph problems
    class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }
    
    // TrieNode definition for trie problems
    class TrieNode {
        TrieNode[] children;
        boolean isEnd;
        
        public TrieNode() {
            children = new TrieNode[26];
            isEnd = false;
        }
    }
}

/**
 * REVERSE ORDER SUMMARY:
 * Patterns are listed from 31 (Multi-thread) to 1 (Two Pointers)
 * This allows practicing from the end patterns first
 * 
 * Total: 31 Patterns, 93 Problems with complete Java solutions
 * Each pattern includes time/space complexity analysis
 * 
 * Perfect for systematic interview preparation starting from advanced topics!
 */
