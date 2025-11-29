import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class GraphPractice {
    
    // Edge List Representation
    private static final int[][] EDGE_LIST = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 3}, {1, 4}, {1, 5},
        {4, 5}, {4, 6}, {5, 6}
    };
    private static final int NODES = 7;
    
    public static void main(String[] args) {
        System.out.println("=== GRAPH CONVERSIONS PRACTICE ===\n");
        
        // 1. Edge List to Adjacency List
        List<List<Integer>> adjList = edgeListToAdjacencyList(EDGE_LIST, NODES);
        System.out.println("1. ADJACENCY LIST:");
        printAdjacencyList(adjList);
        
        // 2. Edge List to Adjacency Matrix
        int[][] adjMatrix = edgeListToAdjacencyMatrix(EDGE_LIST, NODES);
        System.out.println("\n2. ADJACENCY MATRIX:");
        printAdjacencyMatrix(adjMatrix);
        
        // 3. DSU (Union-Find) from Edge List
        DSU dsu = edgeListToDSU(EDGE_LIST, NODES);
        System.out.println("\n3. DSU (Union-Find):");
        dsu.printDSU();
        
        // 4. BFS Traversal
        System.out.println("\n4. BFS TRAVERSAL:");
        bfs(adjList, 0);
        
        // 5. DFS Traversal
        System.out.println("\n5. DFS TRAVERSAL:");
        dfs(adjList, 0);
        
        // 6. Path Finding
        System.out.println("\n6. PATH FINDING:");
        findPath(adjList, 0, 6);
        
        // 7. Cycle Detection
        System.out.println("\n7. CYCLE DETECTION:");
        detectCycle(adjList);
    }
    
    // ========== CONVERSION METHODS ==========
    
    // 1. Edge List to Adjacency List
    public static List<List<Integer>> edgeListToAdjacencyList(int[][] edges, int n) {
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }
        
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1];
            adjList.get(u).add(v);
            adjList.get(v).add(u); // Undirected graph
        }
        
        // Sort for consistent output (optional)
        for (List<Integer> list : adjList) {
            Collections.sort(list);
        }
        
        return adjList;
    }
    
    // 2. Edge List to Adjacency Matrix
    public static int[][] edgeListToAdjacencyMatrix(int[][] edges, int n) {
        int[][] matrix = new int[n][n];
        
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1];
            matrix[u][v] = 1;
            matrix[v][u] = 1; // Undirected graph
        }
        
        return matrix;
    }
    
    // 3. Edge List to DSU (Union-Find)
    public static DSU edgeListToDSU(int[][] edges, int n) {
        DSU dsu = new DSU(n);
        
        for (int[] edge : edges) {
            dsu.union(edge[0], edge[1]);
        }
        
        return dsu;
    }
    
    // ========== GRAPH ALGORITHMS ==========
    
    // 4. BFS Traversal
    public static void bfs(List<List<Integer>> adjList, int start) {
        boolean[] visited = new boolean[adjList.size()];
        Queue<Integer> queue = new LinkedList<>();
        
        queue.offer(start);
        visited[start] = true;
        
        System.out.print("BFS from node " + start + ": ");
        
        while (!queue.isEmpty()) {
            int current = queue.poll();
            System.out.print(current + " ");
            
            for (int neighbor : adjList.get(current)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
        System.out.println();
    }
    
    // 5. DFS Traversal
    public static void dfs(List<List<Integer>> adjList, int start) {
        boolean[] visited = new boolean[adjList.size()];
        System.out.print("DFS from node " + start + ": ");
        dfsRecursive(adjList, start, visited);
        System.out.println();
    }
    
    private static void dfsRecursive(List<List<Integer>> adjList, int node, boolean[] visited) {
        visited[node] = true;
        System.out.print(node + " ");
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                dfsRecursive(adjList, neighbor, visited);
            }
        }
    }
    
    // 6. Path Finding using BFS
    public static void findPath(List<List<Integer>> adjList, int start, int end) {
        boolean[] visited = new boolean[adjList.size()];
        int[] parent = new int[adjList.size()];
        Arrays.fill(parent, -1);
        Queue<Integer> queue = new LinkedList<>();
        
        queue.offer(start);
        visited[start] = true;
        parent[start] = -1;
        
        while (!queue.isEmpty()) {
            int current = queue.poll();
            
            if (current == end) break;
            
            for (int neighbor : adjList.get(current)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    queue.offer(neighbor);
                }
            }
        }
        
        // Reconstruct path
        if (!visited[end]) {
            System.out.println("No path from " + start + " to " + end);
            return;
        }
        
        List<Integer> path = new ArrayList<>();
        for (int at = end; at != -1; at = parent[at]) {
            path.add(at);
        }
        Collections.reverse(path);
        
        System.out.println("Path from " + start + " to " + end + ": " + path);
    }
    
    // 7. Cycle Detection
    public static void detectCycle(List<List<Integer>> adjList) {
        boolean[] visited = new boolean[adjList.size()];
        
        for (int i = 0; i < adjList.size(); i++) {
            if (!visited[i]) {
                if (hasCycle(adjList, i, -1, visited)) {
                    System.out.println("Cycle detected in the graph!");
                    return;
                }
            }
        }
        System.out.println("No cycle detected in the graph.");
    }
    
    private static boolean hasCycle(List<List<Integer>> adjList, int node, int parent, boolean[] visited) {
        visited[node] = true;
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                if (hasCycle(adjList, neighbor, node, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        return false;
    }
    
    // ========== UTILITY METHODS ==========
    
    public static void printAdjacencyList(List<List<Integer>> adjList) {
        for (int i = 0; i < adjList.size(); i++) {
            System.out.println(i + " -> " + adjList.get(i));
        }
    }
    
    public static void printAdjacencyMatrix(int[][] matrix) {
        System.out.print("   ");
        for (int i = 0; i < matrix.length; i++) {
            System.out.print(i + " ");
        }
        System.out.println();
        
        for (int i = 0; i < matrix.length; i++) {
            System.out.print(i + ": ");
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
    
    // ========== DSU (Union-Find) CLASS ==========
    
    static class DSU {
        private int[] parent;
        private int[] rank;
        
        public DSU(int n) {
            parent = new int[n];
            rank = new int[n];
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
        
        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            
            if (rootX != rootY) {
                // Union by rank
                if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
            }
        }
        
        public void printDSU() {
            Map<Integer, List<Integer>> components = new HashMap<>();
            
            for (int i = 0; i < parent.length; i++) {
                int root = find(i);
                components.computeIfAbsent(root, k -> new ArrayList<>()).add(i);
            }
            
            System.out.println("Connected Components:");
            for (List<Integer> component : components.values()) {
                System.out.println("Component with root " + component.get(0) + ": " + component);
            }
        }
    }
}