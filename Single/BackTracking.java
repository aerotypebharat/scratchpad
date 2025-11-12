package com.god.file.individual;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BackTracking {

    // Pattern 25: BACKTRACKING
    // Use Case: Permutations, combinations, constraint satisfaction


    // 25.1: Permutations

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


    // 25.2: Combination Sum

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


    // 25.3: N-Queens

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


}
