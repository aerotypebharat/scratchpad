package com.god.file.individual;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Graph {

    // Pattern 13: GRAPHS
    // Use Case: Connectivity, traversal, cycle detection


    // 13.1: Number of Islands (DFS)

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


    // 13.2: Clone Graph

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


    // 13.3: Course Schedule (Cycle Detection)

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


}


// Node definition for graph problems
class Node {
    public int val;
    public List<Node> neighbors;

    public Node(int val) {
        this.val = val;
        neighbors = new ArrayList<>();
    }

}
