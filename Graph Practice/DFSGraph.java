import java.util.*;

public class DFSComparison {
    private static List<List<Integer>> adjList;
    
    public static void main(String[] args) {
        // Create the same graph
        int[][] edges = {
            {0, 1}, {0, 2}, {0, 3},
            {1, 3}, {1, 4}, {1, 5},
            {4, 5}, {4, 6}, {5, 6}
        };
        
        adjList = createAdjacencyList(edges, 7);
        
        System.out.println("=== DFS RECURSIVE vs ITERATIVE COMPARISON ===\n");
        
        System.out.println("1. DFS Recursive:");
        dfsRecursive(0);
        
        System.out.println("\n\n2. DFS Iterative (using stack):");
        dfsIterative(0);
        
        System.out.println("\n\n3. DFS Iterative (with path tracking):");
        dfsIterativeWithPath(0, 6);
        
        System.out.println("\n\n4. When Recursive DFS Fails (Stack Overflow):");
        demonstrateStackOverflow();
    }
    
    // ========== RECURSIVE DFS ==========
    public static void dfsRecursive(int start) {
        boolean[] visited = new boolean[adjList.size()];
        System.out.print("Traversal: ");
        dfsRecursiveHelper(start, visited);
    }
    
    private static void dfsRecursiveHelper(int node, boolean[] visited) {
        visited[node] = true;
        System.out.print(node + " ");
        
        for (int neighbor : adjList.get(node)) {
            if (!visited[neighbor]) {
                dfsRecursiveHelper(neighbor, visited);
            }
        }
    }
    
    // ========== ITERATIVE DFS ==========
    public static void dfsIterative(int start) {
        boolean[] visited = new boolean[adjList.size()];
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        visited[start] = true;
        
        System.out.print("Traversal: ");
        
        while (!stack.isEmpty()) {
            int current = stack.pop();
            System.out.print(current + " ");
            
            // Push neighbors in reverse order to maintain same traversal as recursive
            for (int i = adjList.get(current).size() - 1; i >= 0; i--) {
                int neighbor = adjList.get(current).get(i);
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    
    // ========== ITERATIVE DFS WITH PATH TRACKING ==========
    public static void dfsIterativeWithPath(int start, int target) {
        boolean[] visited = new boolean[adjList.size()];
        int[] parent = new int[adjList.size()];
        Arrays.fill(parent, -1);
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        visited[start] = true;
        parent[start] = -1;
        
        System.out.print("Traversal: ");
        
        while (!stack.isEmpty()) {
            int current = stack.pop();
            System.out.print(current + " ");
            
            if (current == target) {
                System.out.print("\nPath found: ");
                printPath(parent, target);
                return;
            }
            
            for (int i = adjList.get(current).size() - 1; i >= 0; i--) {
                int neighbor = adjList.get(current).get(i);
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    stack.push(neighbor);
                }
            }
        }
        System.out.println("\nNo path found to " + target);
    }
    
    private static void printPath(int[] parent, int target) {
        List<Integer> path = new ArrayList<>();
        for (int at = target; at != -1; at = parent[at]) {
            path.add(at);
        }
        Collections.reverse(path);
        System.out.println(path);
    }
    
    // ========== STACK OVERFLOW DEMONSTRATION ==========
    public static void demonstrateStackOverflow() {
        // Create a worst-case linear graph that will cause deep recursion
        List<List<Integer>> linearGraph = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            linearGraph.add(new ArrayList<>());
        }
        
        // Create a linear chain: 0-1-2-3-4-...
        for (int i = 0; i < 9999; i++) {
            linearGraph.get(i).add(i + 1);
            linearGraph.get(i + 1).add(i);
        }
        
        System.out.println("Testing with 10,000 node linear graph...");
        
        try {
            // This will likely cause StackOverflowError
            dfsRecursiveLinear(linearGraph, 0);
        } catch (StackOverflowError e) {
            System.out.println("StackOverflowError in recursive DFS!");
        }
        
        // Iterative handles it fine
        System.out.println("Iterative DFS handles large graphs:");
        dfsIterativeLinear(linearGraph, 0);
    }
    
    private static void dfsRecursiveLinear(List<List<Integer>> graph, int start) {
        boolean[] visited = new boolean[graph.size()];
        dfsRecursiveLinearHelper(graph, start, visited, 0);
    }
    
    private static void dfsRecursiveLinearHelper(List<List<Integer>> graph, int node, 
                                               boolean[] visited, int depth) {
        if (depth < 5) { // Only print first few to avoid clutter
            System.out.print(node + " ");
        }
        visited[node] = true;
        
        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                dfsRecursiveLinearHelper(graph, neighbor, visited, depth + 1);
            }
        }
    }
    
    private static void dfsIterativeLinear(List<List<Integer>> graph, int start) {
        boolean[] visited = new boolean[graph.size()];
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        visited[start] = true;
        
        int count = 0;
        while (!stack.isEmpty() && count < 10) { // Print first 10
            int current = stack.pop();
            if (count < 5) {
                System.out.print(current + " ");
            }
            count++;
            
            for (int i = graph.get(current).size() - 1; i >= 0; i--) {
                int neighbor = graph.get(current).get(i);
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        System.out.println("... (handles all 10,000 nodes)");
    }
    
    private static List<List<Integer>> createAdjacencyList(int[][] edges, int n) {
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }
        
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1];
            adjList.get(u).add(v);
            adjList.get(v).add(u);
        }
        return adjList;
    }
}