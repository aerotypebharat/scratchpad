import java.util.*;
import java.math.*;

public class CompetitiveProgrammingSnippets {
    
    // 1. Prime Number Algorithms
    public static boolean[] sieveOfEratosthenes(int n) {
        boolean[] isPrime = new boolean[n + 1];
        Arrays.fill(isPrime, true);
        isPrime[0] = isPrime[1] = false;
        for (int i = 2; i * i <= n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        return isPrime;
    }
    
    public static List<Integer> primeFactors(int n) {
        List<Integer> factors = new ArrayList<>();
        for (int i = 2; i * i <= n; i++) {
            while (n % i == 0) {
                factors.add(i);
                n /= i;
            }
        }
        if (n > 1) factors.add(n);
        return factors;
    }
    
    // 2. String Algorithms
    public static int[] calculateZ(String s) {
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
    
    public static int[] computeLPS(String pattern) {
        int[] lps = new int[pattern.length()];
        int len = 0, i = 1;
        while (i < pattern.length()) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        return lps;
    }
    
    // 3. Mathematical Algorithms
    public static int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    
    public static int lcm(int a, int b) {
        return (a * b) / gcd(a, b);
    }
    
    public static long modPow(long base, long exp, long mod) {
        long result = 1;
        base %= mod;
        while (exp > 0) {
            if ((exp & 1) == 1) 
                result = (result * base) % mod;
            base = (base * base) % mod;
            exp >>= 1;
        }
        return result;
    }
    
    // 4. Graph Algorithms - Union Find
    static class UnionFind {
        private int[] parent, rank;
        public UnionFind(int size) {
            parent = new int[size];
            rank = new int[size];
            for (int i = 0; i < size; i++) parent[i] = i;
        }
        public int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }
        public void union(int x, int y) {
            int rootX = find(x), rootY = find(y);
            if (rootX != rootY) {
                if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
                else if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
                else { parent[rootY] = rootX; rank[rootX]++; }
            }
        }
    }
    
    // 5. Binary Search Variations
    public static int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            else if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
    
    public static int lowerBound(int[] arr, int target) {
        int left = 0, right = arr.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] < target) left = mid + 1;
            else right = mid;
        }
        return left;
    }
    
    public static int upperBound(int[] arr, int target) {
        int left = 0, right = arr.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] <= target) left = mid + 1;
            else right = mid;
        }
        return left;
    }
    
    // 6. Combinatorics
    public static int nCr(int n, int r) {
        if (r > n) return 0;
        if (r == 0 || r == n) return 1;
        int[][] C = new int[n + 1][r + 1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= Math.min(i, r); j++) {
                if (j == 0 || j == i) C[i][j] = 1;
                else C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
            }
        }
        return C[n][r];
    }
    
    // 7. Fenwick Tree
    static class FenwickTree {
        private int[] tree;
        public FenwickTree(int size) { tree = new int[size + 1]; }
        public void update(int index, int delta) {
            index++;
            while (index < tree.length) {
                tree[index] += delta;
                index += index & -index;
            }
        }
        public int query(int index) {
            index++;
            int sum = 0;
            while (index > 0) {
                sum += tree[index];
                index -= index & -index;
            }
            return sum;
        }
        public int rangeQuery(int l, int r) {
            return query(r) - query(l - 1);
        }
    }
    
    // 8. Segment Tree
    static class SegmentTree {
        private int[] tree;
        private int n;
        public SegmentTree(int[] arr) {
            n = arr.length;
            tree = new int[4 * n];
            build(arr, 0, n - 1, 0);
        }
        private void build(int[] arr, int start, int end, int node) {
            if (start == end) {
                tree[node] = arr[start];
                return;
            }
            int mid = (start + end) / 2;
            build(arr, start, mid, 2 * node + 1);
            build(arr, mid + 1, end, 2 * node + 2);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
        public void update(int index, int value, int start, int end, int node) {
            if (start == end) {
                tree[node] = value;
                return;
            }
            int mid = (start + end) / 2;
            if (index <= mid) update(index, value, start, mid, 2 * node + 1);
            else update(index, value, mid + 1, end, 2 * node + 2);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
        public int query(int l, int r, int start, int end, int node) {
            if (r < start || l > end) return 0;
            if (l <= start && end <= r) return tree[node];
            int mid = (start + end) / 2;
            return query(l, r, start, mid, 2 * node + 1) + 
                   query(l, r, mid + 1, end, 2 * node + 2);
        }
    }
    
    // 9. Dynamic Programming - Fibonacci
    public static long fibonacci(int n) {
        if (n <= 1) return n;
        long[] dp = new long[n + 1];
        dp[0] = 0; dp[1] = 1;
        for (int i = 2; i <= n; i++) dp[i] = dp[i - 1] + dp[i - 2];
        return dp[n];
    }
    
    // 10. 0/1 Knapsack
    public static int knapsack01(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(dp[i - 1][w], 
                        dp[i - 1][w - weights[i - 1]] + values[i - 1]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][capacity];
    }
    
    // 11. Longest Common Subsequence
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
    
    // 12. BFS Shortest Path
    public static int[] shortestPathBFS(List<List<Integer>> graph, int start) {
        int n = graph.size();
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        dist[start] = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int v : graph.get(u)) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    queue.offer(v);
                }
            }
        }
        return dist;
    }
    
    // 13. Topological Sort
    public static List<Integer> topologicalSort(int n, int[][] edges) {
        List<List<Integer>> graph = new ArrayList<>();
        int[] indegree = new int[n];
        for (int i = 0; i < n; i++) graph.add(new ArrayList<>());
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1];
            graph.get(u).add(v);
            indegree[v]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) if (indegree[i] == 0) queue.offer(i);
        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            result.add(u);
            for (int v : graph.get(u)) {
                indegree[v]--;
                if (indegree[v] == 0) queue.offer(v);
            }
        }
        return result.size() == n ? result : new ArrayList<>();
    }
    
    // 14. Tree Node and Traversals
    static class TreeNode {
        int val;
        TreeNode left, right;
        TreeNode(int x) { val = x; }
    }
    
    public static List<Integer> inorderIterative(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            result.add(curr.val);
            curr = curr.right;
        }
        return result;
    }
    
    // 15. Lowest Common Ancestor (Binary Lifting)
    static class LCA {
        private int n, log;
        private int[][] up;
        private int[] depth;
        public LCA(List<List<Integer>> tree, int root) {
            n = tree.size();
            log = (int) (Math.log(n) / Math.log(2)) + 1;
            up = new int[n][log];
            depth = new int[n];
            dfs(tree, root, root);
        }
        private void dfs(List<List<Integer>> tree, int u, int parent) {
            up[u][0] = parent;
            for (int i = 1; i < log; i++) 
                up[u][i] = up[up[u][i - 1]][i - 1];
            for (int v : tree.get(u)) {
                if (v != parent) {
                    depth[v] = depth[u] + 1;
                    dfs(tree, v, u);
                }
            }
        }
        public int lca(int u, int v) {
            if (depth[u] < depth[v]) return lca(v, u);
            int diff = depth[u] - depth[v];
            for (int i = 0; i < log; i++) 
                if ((diff & (1 << i)) != 0) u = up[u][i];
            if (u == v) return u;
            for (int i = log - 1; i >= 0; i--) {
                if (up[u][i] != up[v][i]) {
                    u = up[u][i];
                    v = up[v][i];
                }
            }
            return up[u][0];
        }
    }
    
    // 16. Matrix Exponentiation
    public static long[][] matrixMultiply(long[][] A, long[][] B, long mod) {
        int n = A.length, m = B[0].length, p = B.length;
        long[][] C = new long[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < p; k++) {
                    C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod;
                }
            }
        }
        return C;
    }
    
    public static long[][] matrixPower(long[][] matrix, long power, long mod) {
        int n = matrix.length;
        long[][] result = new long[n][n];
        for (int i = 0; i < n; i++) result[i][i] = 1;
        long[][] base = matrix;
        while (power > 0) {
            if ((power & 1) == 1) result = matrixMultiply(result, base, mod);
            base = matrixMultiply(base, base, mod);
            power >>= 1;
        }
        return result;
    }
    
    // 17. Geometry - Point and Convex Hull
    static class Point {
        double x, y;
        Point(double x, double y) { this.x = x; this.y = y; }
        Point subtract(Point p) { return new Point(x - p.x, y - p.y); }
        double cross(Point p) { return x * p.y - y * p.x; }
        double distance(Point p) { 
            double dx = x - p.x, dy = y - p.y;
            return Math.sqrt(dx * dx + dy * dy);
        }
    }
    
    public static List<Point> convexHull(Point[] points) {
        if (points.length < 3) return Arrays.asList(points);
        int minY = 0;
        for (int i = 1; i < points.length; i++) {
            if (points[i].y < points[minY].y || 
                (points[i].y == points[minY].y && points[i].x < points[minY].x)) {
                minY = i;
            }
        }
        Point start = points[minY];
        Arrays.sort(points, (p1, p2) -> {
            double o = orientation(start, p1, p2);
            if (o == 0) return Double.compare(start.distance(p1), start.distance(p2));
            return (o == 2) ? -1 : 1;
        });
        Stack<Point> hull = new Stack<>();
        hull.push(points[0]); hull.push(points[1]);
        for (int i = 2; i < points.length; i++) {
            while (hull.size() > 1) {
                Point top = hull.pop();
                Point nextTop = hull.peek();
                if (orientation(nextTop, top, points[i]) == 2) {
                    hull.push(top);
                    break;
                }
            }
            hull.push(points[i]);
        }
        return new ArrayList<>(hull);
    }
    
    private static int orientation(Point a, Point b, Point c) {
        double val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
        if (val == 0) return 0;
        return (val > 0) ? 1 : 2;
    }
    
    // 18. Rabin-Karp String Matching
    private static final int PRIME = 101;
    public static List<Integer> rabinKarpSearch(String text, String pattern) {
        List<Integer> result = new ArrayList<>();
        int n = text.length(), m = pattern.length();
        if (m > n) return result;
        long patternHash = createHash(pattern, m);
        long textHash = createHash(text, m);
        for (int i = 0; i <= n - m; i++) {
            if (patternHash == textHash && checkEqual(text, i, i + m - 1, pattern)) {
                result.add(i);
            }
            if (i < n - m) {
                textHash = recalculateHash(text, i, i + m, textHash, m);
            }
        }
        return result;
    }
    
    private static long createHash(String str, int length) {
        long hash = 0;
        for (int i = 0; i < length; i++) {
            hash += str.charAt(i) * Math.pow(PRIME, i);
        }
        return hash;
    }
    
    private static long recalculateHash(String str, int oldIndex, int newIndex, 
                                      long oldHash, int patternLen) {
        long newHash = oldHash - str.charAt(oldIndex);
        newHash /= PRIME;
        newHash += str.charAt(newIndex) * Math.pow(PRIME, patternLen - 1);
        return newHash;
    }
    
    private static boolean checkEqual(String text, int start, int end, String pattern) {
        for (int i = 0; i < pattern.length(); i++) {
            if (text.charAt(start + i) != pattern.charAt(i)) return false;
        }
        return true;
    }
    
    // 19. Trie Data Structure
    static class TrieNode {
        TrieNode[] children = new TrieNode[26];
        boolean isEnd;
    }
    
    static class Trie {
        private TrieNode root;
        public Trie() { root = new TrieNode(); }
        public void insert(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) node.children[index] = new TrieNode();
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
    
    // 20. Dijkstra's Algorithm
    public static int[] dijkstra(List<List<int[]>> graph, int start) {
        int n = graph.size();
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        pq.offer(new int[]{start, 0});
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int u = current[0], d = current[1];
            if (d > dist[u]) continue;
            for (int[] edge : graph.get(u)) {
                int v = edge[0], weight = edge[1];
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.offer(new int[]{v, dist[v]});
                }
            }
        }
        return dist;
    }
}