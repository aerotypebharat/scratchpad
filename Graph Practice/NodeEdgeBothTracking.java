import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class GraphTrackingPractice {
    
    // Sample graph data
    private static final int[][] EDGES = {
        {0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {5, 3}
    };
    private static final int NODES = 6;
    
    public static void main(String[] args) {
        System.out.println("=== GRAPH TRACKING COMPREHENSIVE PRACTICE ===\n");
        
        List<List<Integer>> adjList = createAdjacencyList(EDGES, NODES);
        System.out.println("Graph: " + adjList + "\n");
        
        // ========== CATEGORY 1: NODE TRACKING ONLY ==========
        System.out.println("CATEGORY 1: NODE TRACKING ONLY");
        System.out.println("=".repeat(50));
        
        // 1.1 Basic DFS
        System.out.println("1.1 Basic DFS:");
        basicDFS(adjList, 0);
        
        // 1.2 Connected Components
        System.out.println("\n1.2 Connected Components:");
        findConnectedComponents(adjList);
        
        // 1.3 BFS Shortest Path
        System.out.println("\n1.3 BFS Shortest Path (0 to 4):");
        bfsShortestPath(adjList, 0, 4);
        
        // 1.4 Cycle Detection (Nodes)
        System.out.println("\n1.4 Cycle Detection:");
        detectCycleNodes(adjList);
        
        // ========== CATEGORY 2: EDGE TRACKING REQUIRED ==========
        System.out.println("\n CATEGORY 2: EDGE TRACKING REQUIRED");
        System.out.println("=".repeat(50));
        
        // 2.1 DFS with Edge Tracking
        System.out.println("2.1 DFS with Edge Tracking:");
        dfsWithEdgeTracking(adjList, 0);
        
        // 2.2 Bridge Detection
        System.out.println("\n2.2 Bridge Detection:");
        findBridges(adjList);
        
        // 2.3 Eulerian Path Check
        System.out.println("\n2.3 Eulerian Path Check:");
        checkEulerian(adjList);
        
        // ========== CATEGORY 3: BOTH NODE & EDGE TRACKING ==========
        System.out.println("\n CATEGORY 3: BOTH NODE & EDGE TRACKING");
        System.out.println("=".repeat(50));
        
        // 3.1 All Paths Between Nodes
        System.out.println("3.1 All Paths from 0 to 4:");
        findAllPaths(adjList, 0, 4);
        
        // 3.2 Hamiltonian Path Check
        System.out.println("\n3.2 Hamiltonian Path Check:");
        checkHamiltonianPath(adjList);
    }
    
    // ==================== UTILITY METHODS ====================
    private static List<List<Integer>> createAdjacencyList(int[][] edges, int n) {
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < n; i++) adjList.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjList.get(edge[0]).add(edge[1]);
            adjList.get(edge[1]).add(edge[0]); // Undirected
        }
        return adjList;
    }
    
    private static void printVisited(boolean[] visited) {
        System.out.print("Visited nodes: ");
        for (int i = 0; i < visited.length; i++) {
            if (visited[i]) System.out.print(i + " ");
        }
        System.out.println();
    }
    
    // ========== CATEGORY 1: NODE TRACKING ONLY ==========
    
    // 1.1 Basic DFS with Node Tracking
    public static void basicDFS(List<List<Integer>> adjList, int start) {
        boolean[] visited = new boolean[adjList.size()];
        System.out.print("DFS traversal: ");
        basicDFSHelper(adjList, start, visited);
        System.out.println();
        printVisited(visited);
    }
    
    private static void basicDFSHelper(List<List<Integer>> adjList, int node, boolean[] visited) {
        visited[node] = true;
        System.out.print(node + " ");
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                basicDFSHelper(adjList, neighbor, visited);
            }
        }
    }
    
    // 1.2 Connected Components
    public static void findConnectedComponents(List<List<Integer>> adjList) {
        boolean[] visited = new boolean[adjList.size()];
        int components = 0;
        
        for (int i = 0; i < adjList.size(); i++) {
            if (!visited[i]) {
                components++;
                System.out.print("Component " + components + ": ");
                dfsComponent(adjList, i, visited);
                System.out.println();
            }
        }
        System.out.println("Total components: " + components);
    }
    
    private static void dfsComponent(List<List<Integer>> adjList, int node, boolean[] visited) {
        visited[node] = true;
        System.out.print(node + " ");
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                dfsComponent(adjList, neighbor, visited);
            }
        }
    }
    
    // 1.3 BFS Shortest Path
    public static void bfsShortestPath(List<List<Integer>> adjList, int start, int end) {
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
            System.out.println("No path exists from " + start + " to " + end);
            return;
        }
        
        List<Integer> path = new ArrayList<>();
        for (int at = end; at != -1; at = parent[at]) {
            path.add(at);
        }
        Collections.reverse(path);
        System.out.println("Shortest path: " + path);
    }
    
    // 1.4 Cycle Detection with Node Tracking
    public static void detectCycleNodes(List<List<Integer>> adjList) {
        boolean[] visited = new boolean[adjList.size()];
        boolean hasCycle = false;
        
        for (int i = 0; i < adjList.size(); i++) {
            if (!visited[i]) {
                if (hasCycleDFS(adjList, i, -1, visited)) {
                    hasCycle = true;
                    break;
                }
            }
        }
        System.out.println("Cycle detected: " + hasCycle);
    }
    
    private static boolean hasCycleDFS(List<List<Integer>> adjList, int node, int parent, boolean[] visited) {
        visited[node] = true;
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                if (hasCycleDFS(adjList, neighbor, node, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        return false;
    }
    
    // ========== CATEGORY 2: EDGE TRACKING REQUIRED ==========
    
    // Edge representation helper class
    static class Edge {
        int from, to;
        Edge(int from, int to) { this.from = from; this.to = to; }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Edge edge = (Edge) o;
            return from == edge.from && to == edge.to;
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(from, to);
        }
        
        @Override
        public String toString() {
            return from + "-" + to;
        }
    }
    
    // 2.1 DFS with Edge Tracking
    public static void dfsWithEdgeTracking(List<List<Integer>> adjList, int start) {
        boolean[] visitedNodes = new boolean[adjList.size()];
        Set<Edge> visitedEdges = new HashSet<>();
        
        System.out.print("Edge-tracked DFS: ");
        dfsEdgeHelper(adjList, start, visitedNodes, visitedEdges);
        System.out.println();
        System.out.println("Visited edges: " + visitedEdges);
    }
    
    private static void dfsEdgeHelper(List<List<Integer>> adjList, int node, 
                                    boolean[] visitedNodes, Set<Edge> visitedEdges) {
        visitedNodes[node] = true;
        System.out.print(node + " ");
        
        for (int neighbor : adjList.get(node)) {
            Edge edge = new Edge(node, neighbor);
            if (!visitedEdges.contains(edge)) {
                visitedEdges.add(edge);
                visitedEdges.add(new Edge(neighbor, node)); // For undirected
                dfsEdgeHelper(adjList, neighbor, visitedNodes, visitedEdges);
            }
        }
    }
    
    // 2.2 Bridge Detection using Tarjan's Algorithm
    public static void findBridges(List<List<Integer>> adjList) {
        int n = adjList.size();
        int[] disc = new int[n]; // Discovery times
        int[] low = new int[n];  // Earliest visited node reachable
        boolean[] visited = new boolean[n];
        List<Edge> bridges = new ArrayList<>();
        
        Arrays.fill(disc, -1);
        AtomicInteger time = new AtomicInteger(0);
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                bridgeDFS(adjList, i, -1, disc, low, visited, bridges, time);
            }
        }
        
        if (bridges.isEmpty()) {
            System.out.println("No bridges found");
        } else {
            System.out.println("Bridges: " + bridges);
        }
    }
    
    private static void bridgeDFS(List<List<Integer>> adjList, int u, int parent,
                                 int[] disc, int[] low, boolean[] visited,
                                 List<Edge> bridges, AtomicInteger time) {
        visited[u] = true;
        disc[u] = low[u] = time.getAndIncrement();
        
        for (int v : adjList.get(u)) {
            if (v == parent) continue;
            
            if (!visited[v]) {
                bridgeDFS(adjList, v, u, disc, low, visited, bridges, time);
                low[u] = Math.min(low[u], low[v]);
                
                // If lowest vertex reachable from subtree under v is below u
                if (low[v] > disc[u]) {
                    bridges.add(new Edge(u, v));
                }
            } else {
                low[u] = Math.min(low[u], disc[v]);
            }
        }
    }
    
    // 2.3 Eulerian Path Check
    public static void checkEulerian(List<List<Integer>> adjList) {
        int oddDegreeCount = 0;
        
        for (int i = 0; i < adjList.size(); i++) {
            if (adjList.get(i).size() % 2 != 0) {
                oddDegreeCount++;
            }
        }
        
        if (oddDegreeCount == 0) {
            System.out.println("Graph has Eulerian Circuit");
        } else if (oddDegreeCount == 2) {
            System.out.println("Graph has Eulerian Path");
        } else {
            System.out.println("Graph has neither Eulerian Circuit nor Path");
        }
    }
    
    // ========== CATEGORY 3: BOTH NODE & EDGE TRACKING ==========
    
    // 3.1 Find All Paths Between Two Nodes
    public static void findAllPaths(List<List<Integer>> adjList, int start, int end) {
        boolean[] visitedNodes = new boolean[adjList.size()];
        List<List<Integer>> allPaths = new ArrayList<>();
        List<Integer> currentPath = new ArrayList<>();
        
        findAllPathsDFS(adjList, start, end, visitedNodes, currentPath, allPaths);
        
        if (allPaths.isEmpty()) {
            System.out.println("No paths found from " + start + " to " + end);
        } else {
            for (int i = 0; i < allPaths.size(); i++) {
                System.out.println("Path " + (i + 1) + ": " + allPaths.get(i));
            }
        }
    }
    
    private static void findAllPathsDFS(List<List<Integer>> adjList, int current, int end,
                                      boolean[] visitedNodes, List<Integer> currentPath,
                                      List<List<Integer>> allPaths) {
        visitedNodes[current] = true;
        currentPath.add(current);
        
        if (current == end) {
            allPaths.add(new ArrayList<>(currentPath));
        } else {
            for (int neighbor : adjList.get(current)) {
                if (!visitedNodes[neighbor]) {
                    findAllPathsDFS(adjList, neighbor, end, visitedNodes, currentPath, allPaths);
                }
            }
        }
        
        // Backtrack
        currentPath.remove(currentPath.size() - 1);
        visitedNodes[current] = false;
    }
    
    // 3.2 Hamiltonian Path Check
    public static void checkHamiltonianPath(List<List<Integer>> adjList) {
        boolean[] visited = new boolean[adjList.size()];
        boolean hasHamiltonian = false;
        
        // Try starting from each node
        for (int i = 0; i < adjList.size(); i++) {
            Arrays.fill(visited, false);
            if (hamiltonianDFS(adjList, i, visited, 1)) {
                hasHamiltonian = true;
                break;
            }
        }
        
        System.out.println("Hamiltonian path exists: " + hasHamiltonian);
    }
    
    private static boolean hamiltonianDFS(List<List<Integer>> adjList, int node,
                                        boolean[] visited, int count) {
        visited[node] = true;
        
        if (count == adjList.size()) {
            return true; // Visited all nodes
        }
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                if (hamiltonianDFS(adjList, neighbor, visited, count + 1)) {
                    return true;
                }
            }
        }
        
        // Backtrack
        visited[node] = false;
        return false;
    }
}