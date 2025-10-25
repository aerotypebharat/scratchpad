package com.god.file;

import java.util.*;
import java.util.concurrent.Semaphore;

public class LeetCodePatterns_part2 {


    // PATTERN 11: TREE BREADTH FIRST SEARCH (BFS)
    // Use Case: Shortest path, level operations


    //11.1: Minimum Depth of Binary Tree

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


    //11.2: Binary Tree Right Side View

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


    //11.3: Cousins in Binary Tree

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
                    if ((node.left.val == x && node.right.val == y) || (node.left.val == y && node.right.val == x)) {
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


    // PATTERN 12: TREE DEPTH FIRST SEARCH (DFS)
    // Use Case: Path sum, tree properties, backtracking in trees


    //12.1: Path Sum

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;

        // Check if it's a leaf node and path sum equals target
        if (root.left == null && root.right == null && root.val == targetSum) {
            return true;
        }

        // Recursively check left and right subtrees
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }


    //12.2: Sum Root to Leaf Numbers

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


    //12.3: Binary Tree Maximum Path Sum

    private int maxPathSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxPathSumDFS(root);
        return maxPathSum;
    }

    private int maxPathSumDFS(TreeNode node) {
        if (node == null) return 0;

        // Calculate max path sum from left and right children
        int leftMax = Math.max(maxPathSumDFS(node.left), 0);
        int rightMax = Math.max(maxPathSumDFS(node.right), 0);

        // Update global maximum with path through current node
        int pathThroughNode = node.val + leftMax + rightMax;
        maxPathSum = Math.max(maxPathSum, pathThroughNode);

        // Return maximum path sum starting from current node
        return node.val + Math.max(leftMax, rightMax);
    }


    // PATTERN 13: GRAPHS
    // Use Case: Connectivity, traversal, cycle detection


    //13.1: Number of Islands (DFS)

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


    //13.2: Clone Graph

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


    //13.3: Course Schedule (Cycle Detection)

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


    // PATTERN 14: ISLAND (MATRIX TRAVERSAL)
    // Use Case: Grid problems, connected components in matrix


    //14.1: Max Area of Island

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

        return 1 + dfsIslandArea(grid, i + 1, j) + dfsIslandArea(grid, i - 1, j) + dfsIslandArea(grid, i, j + 1) + dfsIslandArea(grid, i, j - 1);
    }


    //14.2: Number of Closed Islands

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


    //14.3: Surrounded Regions

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


    // PATTERN 15: TWO HEAPS
    // Use Case: Finding median, partitioning data streams


    //15.1: Find Median from Data Stream

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


    //15.2: Sliding Window Median

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
                    result[i - k + 1] = ((double) maxHeap.peek() + (double) minHeap.peek()) / 2.0;
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


    //15.3: IPO

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


    // PATTERN 16: SUBSETS
    // Use Case: Combinations, permutations, power set


    //16.1: Subsets

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


    //16.2: Subsets II (with duplicates)

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


    //16.3: Letter Case Permutation

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


    // PATTERN 17: MODIFIED BINARY SEARCH
    // Use Case: Rotated arrays, unknown order, bitonic arrays


    //17.1: Search in Rotated Sorted Array

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


    //17.2: Find First and Last Position in Sorted Array

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


    //17.3: Find Peak Element

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


    // PATTERN 18: BITWISE XOR
    // Use Case: Finding unique elements, bit manipulation


    //18.1: Single Number

    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    //18.2: Missing Number

    public int missingNumber(int[] nums) {
        int n = nums.length;
        int result = n; // Initialize with n since it's missing from indices

        for (int i = 0; i < n; i++) {
            result ^= i ^ nums[i];
        }
        return result;
    }


    //18.3: Complement of Base 10 Integer

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


    // PATTERN 19: TOP 'K' ELEMENTS
    // Use Case: K largest/smallest, frequent elements


    //19.1: Top K Frequent Elements

    public int[] topKFrequent(int[] nums, int k) {
        // Count frequencies
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        // Min heap to keep top k elements
        PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());

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


    //19.2: Kth Largest Element in Array

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


    //19.3: K Closest Points to Origin

    public int[][] kClosest(int[][] points, int k) {
        // Max heap to keep k closest points
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a, b) -> Integer.compare(distance(b), distance(a)));

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


    // PATTERN 20: K-WAY MERGE
    // Use Case: Merging multiple sorted arrays/lists


    //20.1: Merge K Sorted Lists

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


    //20.2: Kth Smallest Element in Sorted Matrix

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


    //20.3: Find K Pairs with Smallest Sums

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


    // SUPPORTING DATA STRUCTURES


    // ListNode definition for linked list problems
    class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // TreeNode definition for tree problems
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

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
