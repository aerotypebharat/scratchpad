package com.god.file.individual;

public class TreeDFS {


    // Pattern 12: TREE DEPTH FIRST SEARCH (DFS)
    // Use Case: Path sum, tree properties, backtracking in trees


    // 12.1: Path Sum

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;

        // Check if it's a leaf node and path sum equals target
        if (root.left == null && root.right == null && root.val == targetSum) {
            return true;
        }

        // Recursively check left and right subtrees
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }


    // 12.2: Sum Root to Leaf Numbers

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


    // 12.3: Binary Tree Maximum Path Sum

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


}
