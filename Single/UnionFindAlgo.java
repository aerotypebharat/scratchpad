package com.god.file.individual;

import java.util.ArrayList;
import java.util.List;

public class UnionFindAlgo {

    // Pattern 28: UNION FIND
    // Use Case: Dynamic connectivity, cycle detection


    // 28.1: Number of Provinces

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

    // Union-Find Data Structure
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
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        public boolean union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX == rootY) return false;

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


}
