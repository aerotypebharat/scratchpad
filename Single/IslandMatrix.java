package com.god.file.individual;

public class IslandMatrix {


    // Pattern 14: ISLAND (MATRIX TRAVERSAL)
    // Use Case: Grid problems, connected components in matrix


    // 14.1: Max Area of Island

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


    // 14.2: Number of Closed Islands

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


    // 14.3: Surrounded Regions

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


}
