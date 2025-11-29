import java.util.*;
import java.io.*;

/**
 * Comprehensive Competitive Programming Reference in Java
 * Expanded with advanced topics and algorithms
 */
public class CompetitiveProgrammingReference {
    
    // ========== PART I: BASIC TECHNIQUES ==========
    
    /**
     * 1. Fast I/O and Basic Operations
     */
    static class FastIO {
        BufferedReader br;
        StringTokenizer st;
        PrintWriter pw;
        
        public FastIO() {
            br = new BufferedReader(new InputStreamReader(System.in));
            pw = new PrintWriter(System.out);
        }
        
        String next() {
            while (st == null || !st.hasMoreElements()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }
        
        int nextInt() { return Integer.parseInt(next()); }
        long nextLong() { return Long.parseLong(next()); }
        double nextDouble() { return Double.parseDouble(next()); }
        String nextLine() {
            String str = "";
            try { str = br.readLine(); } 
            catch (IOException e) { e.printStackTrace(); }
            return str;
        }
        
        void print(Object obj) { pw.print(obj); }
        void println(Object obj) { pw.println(obj); }
        void flush() { pw.flush(); }
        void close() { pw.close(); }
    }
    
    /**
     * 2. Advanced Sorting and Searching
     */
    static class AdvancedSorting {
        // Quick Sort implementation
        public static void quickSort(int[] arr, int low, int high) {
            if (low < high) {
                int pi = partition(arr, low, high);
                quickSort(arr, low, pi - 1);
                quickSort(arr, pi + 1, high);
            }
        }
        
        private static int partition(int[] arr, int low, int high) {
            int pivot = arr[high];
            int i = low - 1;
            for (int j = low; j < high; j++) {
                if (arr[j] < pivot) {
                    i++;
                    swap(arr, i, j);
                }
            }
            swap(arr, i + 1, high);
            return i + 1;
        }
        
        private static void swap(int[] arr, int i, int j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
        
        // Merge Sort implementation
        public static void mergeSort(int[] arr, int left, int right) {
            if (left < right) {
                int mid = left + (right - left) / 2;
                mergeSort(arr, left, mid);
                mergeSort(arr, mid + 1, right);
                merge(arr, left, mid, right);
            }
        }
        
        private static void merge(int[] arr, int left, int mid, int right) {
            int n1 = mid - left + 1;
            int n2 = right - mid;
            
            int[] L = new int[n1];
            int[] R = new int[n2];
            
            System.arraycopy(arr, left, L, 0, n1);
            System.arraycopy(arr, mid + 1, R, 0, n2);
            
            int i = 0, j = 0, k = left;
            while (i < n1 && j < n2) {
                if (L[i] <= R[j]) {
                    arr[k++] = L[i++];
                } else {
                    arr[k++] = R[j++];
                }
            }
            
            while (i < n1) arr[k++] = L[i++];
            while (j < n2) arr[k++] = R[j++];
        }
        
        // Lower Bound implementation (first element >= target)
        public static int lowerBound(int[] arr, int target) {
            int left = 0, right = arr.length;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            return left;
        }
        
        // Upper Bound implementation (first element > target)
        public static int upperBound(int[] arr, int target) {
            int left = 0, right = arr.length;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] <= target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            return left;
        }
    }
    
    /**
     * 3. Advanced Data Structures
     */
    static class AdvancedDataStructures {
        // Trie implementation
        static class Trie {
            static class TrieNode {
                TrieNode[] children = new TrieNode[26];
                boolean isEnd;
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
                TrieNode node = root;
                for (char c : word.toCharArray()) {
                    int index = c - 'a';
                    if (node.children[index] == null) return false;
                    node = node.children[index];
                }
                return node.isEnd;
            }
            
            public boolean startsWith(String prefix) {
                TrieNode node = root;
                for (char c : prefix.toCharArray()) {
                    int index = c - 'a';
                    if (node.children[index] == null) return false;
                    node = node.children[index];
                }
                return true;
            }
        }
        
        // Segment Tree implementation
        static class SegmentTree {
            private int[] tree;
            private int n;
            
            public SegmentTree(int[] nums) {
                n = nums.length;
                tree = new int[4 * n];
                buildTree(nums, 0, 0, n - 1);
            }
            
            private void buildTree(int[] nums, int node, int start, int end) {
                if (start == end) {
                    tree[node] = nums[start];
                } else {
                    int mid = start + (end - start) / 2;
                    buildTree(nums, 2 * node + 1, start, mid);
                    buildTree(nums, 2 * node + 2, mid + 1, end);
                    tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
                }
            }
            
            public void update(int index, int value) {
                update(0, 0, n - 1, index, value);
            }
            
            private void update(int node, int start, int end, int index, int value) {
                if (start == end) {
                    tree[node] = value;
                } else {
                    int mid = start + (end - start) / 2;
                    if (index <= mid) {
                        update(2 * node + 1, start, mid, index, value);
                    } else {
                        update(2 * node + 2, mid + 1, end, index, value);
                    }
                    tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
                }
            }
            
            public int query(int left, int right) {
                return query(0, 0, n - 1, left, right);
            }
            
            private int query(int node, int start, int end, int left, int right) {
                if (left > end || right < start) return 0;
                if (left <= start && right >= end) return tree[node];
                
                int mid = start + (end - start) / 2;
                int leftSum = query(2 * node + 1, start, mid, left, right);
                int rightSum = query(2 * node + 2, mid + 1, end, left, right);
                return leftSum + rightSum;
            }
        }
        
        // LRU Cache implementation
        static class LRUCache {
            class DLinkedNode {
                int key;
                int value;
                DLinkedNode prev;
                DLinkedNode next;
            }
            
            private Map<Integer, DLinkedNode> cache = new HashMap<>();
            private int size;
            private int capacity;
            private DLinkedNode head, tail;
            
            public LRUCache(int capacity) {
                this.size = 0;
                this.capacity = capacity;
                head = new DLinkedNode();
                tail = new DLinkedNode();
                head.next = tail;
                tail.prev = head;
            }
            
            public int get(int key) {
                DLinkedNode node = cache.get(key);
                if (node == null) return -1;
                moveToHead(node);
                return node.value;
            }
            
            public void put(int key, int value) {
                DLinkedNode node = cache.get(key);
                if (node == null) {
                    DLinkedNode newNode = new DLinkedNode();
                    newNode.key = key;
                    newNode.value = value;
                    cache.put(key, newNode);
                    addNode(newNode);
                    size++;
                    if (size > capacity) {
                        DLinkedNode tail = popTail();
                        cache.remove(tail.key);
                        size--;
                    }
                } else {
                    node.value = value;
                    moveToHead(node);
                }
            }
            
            private void addNode(DLinkedNode node) {
                node.prev = head;
                node.next = head.next;
                head.next.prev = node;
                head.next = node;
            }
            
            private void removeNode(DLinkedNode node) {
                DLinkedNode prev = node.prev;
                DLinkedNode next = node.next;
                prev.next = next;
                next.prev = prev;
            }
            
            private void moveToHead(DLinkedNode node) {
                removeNode(node);
                addNode(node);
            }
            
            private DLinkedNode popTail() {
                DLinkedNode res = tail.prev;
                removeNode(res);
                return res;
            }
        }
    }
    
    /**
     * 4. Advanced Dynamic Programming
     */
    static class AdvancedDP {
        // Coin Change II (Number of combinations)
        public static int coinChangeII(int[] coins, int amount) {
            int[] dp = new int[amount + 1];
            dp[0] = 1;
            
            for (int coin : coins) {
                for (int i = coin; i <= amount; i++) {
                    dp[i] += dp[i - coin];
                }
            }
            return dp[amount];
        }
        
        // Longest Common Subsequence
        public static int longestCommonSubsequence(String text1, String text2) {
            int m = text1.length(), n = text2.length();
            int[][] dp = new int[m + 1][n + 1];
            
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1] + 1;
                    } else {
                        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                    }
                }
            }
            return dp[m][n];
        }
        
        // Matrix Chain Multiplication
        public static int matrixChainMultiplication(int[] dims) {
            int n = dims.length;
            int[][] dp = new int[n][n];
            
            for (int len = 2; len < n; len++) {
                for (int i = 1; i < n - len + 1; i++) {
                    int j = i + len - 1;
                    dp[i][j] = Integer.MAX_VALUE;
                    for (int k = i; k < j; k++) {
                        int cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j];
                        dp[i][j] = Math.min(dp[i][j], cost);
                    }
                }
            }
            return dp[1][n - 1];
        }
        
        // 0/1 Knapsack with reconstruction
        public static List<Integer> knapsackWithReconstruction(int[] weights, int[] values, int capacity) {
            int n = weights.length;
            int[][] dp = new int[n + 1][capacity + 1];
            
            for (int i = 1; i <= n; i++) {
                for (int w = 0; w <= capacity; w++) {
                    if (weights[i - 1] <= w) {
                        dp[i][w] = Math.max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]]);
                    } else {
                        dp[i][w] = dp[i - 1][w];
                    }
                }
            }
            
            // Reconstruction
            List<Integer> selected = new ArrayList<>();
            int w = capacity;
            for (int i = n; i > 0; i--) {
                if (dp[i][w] != dp[i - 1][w]) {
                    selected.add(i - 1);
                    w -= weights[i - 1];
                }
            }
            Collections.reverse(selected);
            return selected;
        }
    }
    
    // ========== PART II: GRAPH ALGORITHMS ==========
    
    /**
     * 5. Advanced Graph Algorithms
     */
    static class AdvancedGraph {
        // Topological Sort (Kahn's Algorithm)
        public static List<Integer> topologicalSort(int n, int[][] edges) {
            List<List<Integer>> graph = new ArrayList<>();
            int[] indegree = new int[n];
            
            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList<>());
            }
            
            for (int[] edge : edges) {
                graph.get(edge[0]).add(edge[1]);
                indegree[edge[1]]++;
            }
            
            Queue<Integer> queue = new LinkedList<>();
            for (int i = 0; i < n; i++) {
                if (indegree[i] == 0) {
                    queue.offer(i);
                }
            }
            
            List<Integer> result = new ArrayList<>();
            while (!queue.isEmpty()) {
                int node = queue.poll();
                result.add(node);
                
                for (int neighbor : graph.get(node)) {
                    indegree[neighbor]--;
                    if (indegree[neighbor] == 0) {
                        queue.offer(neighbor);
                    }
                }
            }
            
            return result.size() == n ? result : new ArrayList<>(); // if cycle exists
        }
        
        // Floyd-Warshall Algorithm (All Pairs Shortest Path)
        public static int[][] floydWarshall(int n, int[][] edges) {
            int[][] dist = new int[n][n];
            
            // Initialize distances
            for (int i = 0; i < n; i++) {
                Arrays.fill(dist[i], Integer.MAX_VALUE);
                dist[i][i] = 0;
            }
            
            for (int[] edge : edges) {
                dist[edge[0]][edge[1]] = edge[2];
            }
            
            // Floyd-Warshall algorithm
            for (int k = 0; k < n; k++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (dist[i][k] != Integer.MAX_VALUE && dist[k][j] != Integer.MAX_VALUE) {
                            if (dist[i][j] > dist[i][k] + dist[k][j]) {
                                dist[i][j] = dist[i][k] + dist[k][j];
                            }
                        }
                    }
                }
            }
            
            return dist;
        }
        
        // Tarjan's Algorithm for Strongly Connected Components
        public static List<List<Integer>> tarjanSCC(int n, int[][] edges) {
            List<List<Integer>> graph = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList<>());
            }
            for (int[] edge : edges) {
                graph.get(edge[0]).add(edge[1]);
            }
            
            int[] index = new int[n];
            int[] lowlink = new int[n];
            boolean[] onStack = new boolean[n];
            Stack<Integer> stack = new Stack<>();
            List<List<Integer>> scc = new ArrayList<>();
            int[] currentIndex = {0};
            
            for (int i = 0; i < n; i++) {
                if (index[i] == 0) {
                    strongconnect(i, graph, index, lowlink, onStack, stack, scc, currentIndex);
                }
            }
            
            return scc;
        }
        
        private static void strongconnect(int v, List<List<Integer>> graph, int[] index, 
                                        int[] lowlink, boolean[] onStack, Stack<Integer> stack,
                                        List<List<Integer>> scc, int[] currentIndex) {
            index[v] = currentIndex[0];
            lowlink[v] = currentIndex[0];
            currentIndex[0]++;
            stack.push(v);
            onStack[v] = true;
            
            for (int w : graph.get(v)) {
                if (index[w] == 0) {
                    strongconnect(w, graph, index, lowlink, onStack, stack, scc, currentIndex);
                    lowlink[v] = Math.min(lowlink[v], lowlink[w]);
                } else if (onStack[w]) {
                    lowlink[v] = Math.min(lowlink[v], index[w]);
                }
            }
            
            if (lowlink[v] == index[v]) {
                List<Integer> component = new ArrayList<>();
                int w;
                do {
                    w = stack.pop();
                    onStack[w] = false;
                    component.add(w);
                } while (w != v);
                scc.add(component);
            }
        }
        
        // Bellman-Ford Algorithm (Negative weight cycles)
        public static int[] bellmanFord(int n, int[][] edges, int source) {
            int[] dist = new int[n];
            Arrays.fill(dist, Integer.MAX_VALUE);
            dist[source] = 0;
            
            // Relax edges n-1 times
            for (int i = 0; i < n - 1; i++) {
                for (int[] edge : edges) {
                    int u = edge[0], v = edge[1], weight = edge[2];
                    if (dist[u] != Integer.MAX_VALUE && dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                    }
                }
            }
            
            // Check for negative weight cycles
            for (int[] edge : edges) {
                int u = edge[0], v = edge[1], weight = edge[2];
                if (dist[u] != Integer.MAX_VALUE && dist[u] + weight < dist[v]) {
                    System.out.println("Graph contains negative weight cycle");
                    return null;
                }
            }
            
            return dist;
        }
    }
    
    /**
     * 6. Tree Algorithms
     */
    static class TreeAlgorithms {
        // Lowest Common Ancestor (Binary Lifting)
        static class LCA {
            private int n, log;
            private int[] depth;
            private int[][] parent;
            private List<List<Integer>> tree;
            
            public LCA(List<List<Integer>> tree, int root) {
                this.tree = tree;
                n = tree.size();
                log = (int) (Math.log(n) / Math.log(2)) + 1;
                depth = new int[n];
                parent = new int[n][log];
                
                dfs(root, -1, 0);
                precompute();
            }
            
            private void dfs(int node, int par, int d) {
                depth[node] = d;
                parent[node][0] = par;
                for (int child : tree.get(node)) {
                    if (child != par) {
                        dfs(child, node, d + 1);
                    }
                }
            }
            
            private void precompute() {
                for (int j = 1; j < log; j++) {
                    for (int i = 0; i < n; i++) {
                        if (parent[i][j - 1] != -1) {
                            parent[i][j] = parent[parent[i][j - 1]][j - 1];
                        } else {
                            parent[i][j] = -1;
                        }
                    }
                }
            }
            
            public int findLCA(int u, int v) {
                if (depth[u] < depth[v]) {
                    int temp = u;
                    u = v;
                    v = temp;
                }
                
                // Lift u to same depth as v
                for (int i = log - 1; i >= 0; i--) {
                    if (depth[u] - (1 << i) >= depth[v]) {
                        u = parent[u][i];
                    }
                }
                
                if (u == v) return u;
                
                // Lift both until their parents are the same
                for (int i = log - 1; i >= 0; i--) {
                    if (parent[u][i] != parent[v][i]) {
                        u = parent[u][i];
                        v = parent[v][i];
                    }
                }
                
                return parent[u][0];
            }
        }
        
        // Tree Diameter
        public static int[] treeDiameter(List<List<Integer>> tree) {
            int n = tree.size();
            // First BFS from arbitrary node to find farthest node
            int[] bfs1 = bfs(tree, 0);
            // Second BFS from farthest node to find diameter
            int[] bfs2 = bfs(tree, bfs1[0]);
            return new int[]{bfs1[0], bfs2[0], bfs2[1]}; // [end1, end2, diameter]
        }
        
        private static int[] bfs(List<List<Integer>> tree, int start) {
            int n = tree.size();
            int[] dist = new int[n];
            Arrays.fill(dist, -1);
            Queue<Integer> queue = new LinkedList<>();
            
            dist[start] = 0;
            queue.offer(start);
            int farthestNode = start;
            int maxDist = 0;
            
            while (!queue.isEmpty()) {
                int node = queue.poll();
                for (int neighbor : tree.get(node)) {
                    if (dist[neighbor] == -1) {
                        dist[neighbor] = dist[node] + 1;
                        queue.offer(neighbor);
                        if (dist[neighbor] > maxDist) {
                            maxDist = dist[neighbor];
                            farthestNode = neighbor;
                        }
                    }
                }
            }
            
            return new int[]{farthestNode, maxDist};
        }
    }
    
    // ========== PART III: ADVANCED TOPICS ==========
    
    /**
     * 7. String Processing
     */
    static class StringProcessing {
        // Z-Algorithm
        public static int[] zAlgorithm(String s) {
            int n = s.length();
            int[] z = new int[n];
            int l = 0, r = 0;
            
            for (int i = 1; i < n; i++) {
                if (i <= r) {
                    z[i] = Math.min(r - i + 1, z[i - l]);
                }
                while (i + z[i] < n && s.charAt(z[i]) == s.charAt(i + z[i])) {
                    z[i]++;
                }
                if (i + z[i] - 1 > r) {
                    l = i;
                    r = i + z[i] - 1;
                }
            }
            return z;
        }
        
        // Manacher's Algorithm (Longest Palindromic Substring)
        public static String longestPalindrome(String s) {
            if (s == null || s.length() == 0) return "";
            
            String T = preprocess(s);
            int n = T.length();
            int[] P = new int[n];
            int C = 0, R = 0;
            
            for (int i = 1; i < n - 1; i++) {
                int mirror = 2 * C - i;
                
                if (i < R) {
                    P[i] = Math.min(R - i, P[mirror]);
                }
                
                while (T.charAt(i + (1 + P[i])) == T.charAt(i - (1 + P[i]))) {
                    P[i]++;
                }
                
                if (i + P[i] > R) {
                    C = i;
                    R = i + P[i];
                }
            }
            
            int maxLen = 0;
            int centerIndex = 0;
            for (int i = 1; i < n - 1; i++) {
                if (P[i] > maxLen) {
                    maxLen = P[i];
                    centerIndex = i;
                }
            }
            
            int start = (centerIndex - maxLen) / 2;
            return s.substring(start, start + maxLen);
        }
        
        private static String preprocess(String s) {
            StringBuilder sb = new StringBuilder();
            sb.append('^');
            for (int i = 0; i < s.length(); i++) {
                sb.append('#');
                sb.append(s.charAt(i));
            }
            sb.append("#$");
            return sb.toString();
        }
        
        // Rabin-Karp Algorithm (Multiple pattern search)
        public static List<Integer> rabinKarp(String text, String pattern) {
            List<Integer> result = new ArrayList<>();
            if (pattern.length() > text.length()) return result;
            
            long prime = 31;
            long mod = (long) 1e9 + 9;
            int n = text.length(), m = pattern.length();
            
            long[] p_pow = new long[Math.max(n, m)];
            p_pow[0] = 1;
            for (int i = 1; i < p_pow.length; i++) {
                p_pow[i] = (p_pow[i - 1] * prime) % mod;
            }
            
            long[] h_text = new long[n + 1];
            for (int i = 0; i < n; i++) {
                h_text[i + 1] = (h_text[i] + (text.charAt(i) - 'a' + 1) * p_pow[i]) % mod;
            }
            
            long h_pattern = 0;
            for (int i = 0; i < m; i++) {
                h_pattern = (h_pattern + (pattern.charAt(i) - 'a' + 1) * p_pow[i]) % mod;
            }
            
            for (int i = 0; i + m - 1 < n; i++) {
                long curr_hash = (h_text[i + m] - h_text[i] + mod) % mod;
                if (curr_hash == h_pattern * p_pow[i] % mod) {
                    result.add(i);
                }
            }
            
            return result;
        }
    }
    
    /**
     * 8. Advanced Mathematics
     */
    static class AdvancedMathematics {
        // Extended Euclidean Algorithm
        public static int[] extendedGCD(int a, int b) {
            if (b == 0) {
                return new int[]{a, 1, 0};
            }
            int[] result = extendedGCD(b, a % b);
            int gcd = result[0];
            int x1 = result[1];
            int y1 = result[2];
            return new int[]{gcd, y1, x1 - (a / b) * y1};
        }
        
        // Chinese Remainder Theorem
        public static int chineseRemainderTheorem(int[] num, int[] rem) {
            int product = 1;
            for (int n : num) product *= n;
            
            int result = 0;
            for (int i = 0; i < num.length; i++) {
                int pp = product / num[i];
                result += rem[i] * modularInverse(pp, num[i]) * pp;
            }
            return result % product;
        }
        
        private static int modularInverse(int a, int m) {
            int[] gcdResult = extendedGCD(a, m);
            if (gcdResult[0] != 1) return -1; // inverse doesn't exist
            return (gcdResult[1] % m + m) % m;
        }
        
        // Matrix Exponentiation (Fibonacci in O(log n))
        public static long matrixFib(int n) {
            if (n <= 1) return n;
            long[][] base = {{1, 1}, {1, 0}};
            long[][] result = matrixPower(base, n - 1);
            return result[0][0];
        }
        
        private static long[][] matrixMultiply(long[][] A, long[][] B) {
            int n = A.length;
            int m = B[0].length;
            int p = B.length;
            long[][] C = new long[n][m];
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < p; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return C;
        }
        
        private static long[][] matrixPower(long[][] matrix, int power) {
            int n = matrix.length;
            long[][] result = new long[n][n];
            for (int i = 0; i < n; i++) result[i][i] = 1;
            
            long[][] base = matrix;
            while (power > 0) {
                if ((power & 1) == 1) {
                    result = matrixMultiply(result, base);
                }
                base = matrixMultiply(base, base);
                power >>= 1;
            }
            return result;
        }
        
        // Fermat's Little Theorem for modular inverse
        public static long modularInverseFERMAT(long a, long mod) {
            return fastPower(a, mod - 2, mod);
        }
        
        private static long fastPower(long base, long exponent, long mod) {
            long result = 1;
            base %= mod;
            while (exponent > 0) {
                if ((exponent & 1) == 1) {
                    result = (result * base) % mod;
                }
                base = (base * base) % mod;
                exponent >>= 1;
            }
            return result;
        }
    }
    
    /**
     * 9. Computational Geometry
     */
    static class ComputationalGeometry {
        // Point class
        static class Point {
            double x, y;
            Point(double x, double y) { this.x = x; this.y = y; }
        }
        
        // Cross product
        public static double cross(Point a, Point b, Point c) {
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        }
        
        // Convex Hull (Graham Scan)
        public static List<Point> convexHull(Point[] points) {
            if (points.length < 3) return Arrays.asList(points);
            
            // Find the point with the lowest y-coordinate
            int minIndex = 0;
            for (int i = 1; i < points.length; i++) {
                if (points[i].y < points[minIndex].y || 
                    (points[i].y == points[minIndex].y && points[i].x < points[minIndex].x)) {
                    minIndex = i;
                }
            }
            
            // Swap minIndex with first element
            Point temp = points[0];
            points[0] = points[minIndex];
            points[minIndex] = temp;
            
            // Sort by polar angle
            final Point pivot = points[0];
            Arrays.sort(points, 1, points.length, (a, b) -> {
                double angle = cross(pivot, a, b);
                if (angle == 0) {
                    return Double.compare(distance(pivot, a), distance(pivot, b));
                }
                return angle > 0 ? -1 : 1;
            });
            
            // Build hull
            Stack<Point> hull = new Stack<>();
            hull.push(points[0]);
            hull.push(points[1]);
            
            for (int i = 2; i < points.length; i++) {
                Point top = hull.pop();
                while (cross(hull.peek(), top, points[i]) <= 0) {
                    top = hull.pop();
                }
                hull.push(top);
                hull.push(points[i]);
            }
            
            return new ArrayList<>(hull);
        }
        
        private static double distance(Point a, Point b) {
            return Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
        }
        
        // Point in Polygon
        public static boolean pointInPolygon(Point point, List<Point> polygon) {
            int n = polygon.size();
            boolean inside = false;
            
            for (int i = 0, j = n - 1; i < n; j = i++) {
                if ((polygon.get(i).y > point.y) != (polygon.get(j).y > point.y) &&
                    point.x < (polygon.get(j).x - polygon.get(i).x) * (point.y - polygon.get(i).y) / 
                    (polygon.get(j).y - polygon.get(i).y) + polygon.get(i).x) {
                    inside = !inside;
                }
            }
            return inside;
        }
    }
    
    /**
     * 10. Bit Manipulation and DP
     */
    static class BitManipulation {
        // Count set bits
        public static int countSetBits(int n) {
            int count = 0;
            while (n > 0) {
                n &= (n - 1);
                count++;
            }
            return count;
        }
        
        // Check if power of two
        public static boolean isPowerOfTwo(int n) {
            return n > 0 && (n & (n - 1)) == 0;
        }
        
        // Traveling Salesman Problem (TSP) with Bitmask DP
        public static int tsp(int[][] graph) {
            int n = graph.length;
            int[][] dp = new int[1 << n][n];
            
            for (int[] row : dp) Arrays.fill(row, Integer.MAX_VALUE);
            dp[1][0] = 0; // Start at city 0
            
            for (int mask = 1; mask < (1 << n); mask++) {
                for (int i = 0; i < n; i++) {
                    if ((mask & (1 << i)) == 0) continue;
                    for (int j = 0; j < n; j++) {
                        if ((mask & (1 << j)) != 0) continue;
                        if (graph[i][j] != 0 && dp[mask][i] != Integer.MAX_VALUE) {
                            int newMask = mask | (1 << j);
                            dp[newMask][j] = Math.min(dp[newMask][j], dp[mask][i] + graph[i][j]);
                        }
                    }
                }
            }
            
            int result = Integer.MAX_VALUE;
            int fullMask = (1 << n) - 1;
            for (int i = 1; i < n; i++) {
                if (graph[i][0] != 0 && dp[fullMask][i] != Integer.MAX_VALUE) {
                    result = Math.min(result, dp[fullMask][i] + graph[i][0]);
                }
            }
            return result;
        }
    }
    
    // ========== TESTING AND DEMONSTRATION ==========
    
    public static void testAllAlgorithms() {
        System.out.println("=== Testing All Advanced Competitive Programming Algorithms ===\n");
        
        // Test Advanced Sorting
        System.out.println("1. Testing Advanced Sorting:");
        int[] arr = {5, 2, 8, 1, 9, 3};
        AdvancedSorting.quickSort(arr, 0, arr.length - 1);
        System.out.println("Quick Sort Result: " + Arrays.toString(arr));
        
        int lowerBound = AdvancedSorting.lowerBound(arr, 4);
        System.out.println("Lower Bound for 4: " + lowerBound);
        
        // Test Advanced Data Structures
        System.out.println("\n2. Testing Advanced Data Structures:");
        AdvancedDataStructures.Trie trie = new AdvancedDataStructures.Trie();
        trie.insert("apple");
        trie.insert("app");
        System.out.println("Trie Search 'apple': " + trie.search("apple"));
        System.out.println("Trie StartsWith 'app': " + trie.startsWith("app"));
        
        // Test Advanced DP
        System.out.println("\n3. Testing Advanced DP:");
        int lcs = AdvancedDP.longestCommonSubsequence("ABCDGH", "AEDFHR");
        System.out.println("LCS: " + lcs);
        
        int coinChange = AdvancedDP.coinChangeII(new int[]{1, 2, 5}, 5);
        System.out.println("Coin Change II: " + coinChange);
        
        // Test Graph Algorithms
        System.out.println("\n4. Testing Graph Algorithms:");
        int n = 6;
        int[][] edges = {{5, 2}, {5, 0}, {4, 0}, {4, 1}, {2, 3}, {3, 1}};
        List<Integer> topo = AdvancedGraph.topologicalSort(n, edges);
        System.out.println("Topological Sort: " + topo);
        
        // Test String Algorithms
        System.out.println("\n5. Testing String Algorithms:");
        String palindrome = StringProcessing.longestPalindrome("babad");
        System.out.println("Longest Palindrome: " + palindrome);
        
        // Test Mathematics
        System.out.println("\n6. Testing Mathematics:");
        long matrixFib = AdvancedMathematics.matrixFib(10);
        System.out.println("Matrix Fibonacci(10): " + matrixFib);
        
        // Test Bit Manipulation
        System.out.println("\n7. Testing Bit Manipulation:");
        int setBits = BitManipulation.countSetBits(15);
        System.out.println("Set bits in 15: " + setBits);
        
        System.out.println("\n=== All Advanced Tests Completed ===");
    }
    
    // ========== MAIN METHOD ==========
    public static void main(String[] args) {
        // Run all test cases
        testAllAlgorithms();
        
        // Example usage of FastIO
        FastIO io = new FastIO();
        System.out.println("\n=== Fast IO Example ===");
        System.out.println("Enter array size and elements:");
        
        int n = io.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = io.nextInt();
        }
        
        io.println("Array: " + Arrays.toString(arr));
        io.flush();
        io.close();
    }
}