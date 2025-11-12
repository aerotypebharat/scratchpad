package com.god.file.individual;

import java.util.HashMap;
import java.util.Map;

public class PrefixSum {

    // Pattern 30: PREFIX SUM
    // Use Case: Range sum queries, subarray problems


    // 30.1: Subarray Sum Equals K

    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        prefixSumCount.put(0, 1);

        int count = 0, prefixSum = 0;

        for (int num : nums) {
            prefixSum += num;

            if (prefixSumCount.containsKey(prefixSum - k)) {
                count += prefixSumCount.get(prefixSum - k);
            }

            prefixSumCount.put(prefixSum, prefixSumCount.getOrDefault(prefixSum, 0) + 1);
        }
        return count;
    }


    // 30.2: Continuous Subarray Sum

    public boolean checkSubarraySum(int[] nums, int k) {
        Map<Integer, Integer> remainderIndex = new HashMap<>();
        remainderIndex.put(0, -1);

        int prefixSum = 0;

        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            int remainder = prefixSum % k;

            if (remainderIndex.containsKey(remainder)) {
                if (i - remainderIndex.get(remainder) >= 2) {
                    return true;
                }
            } else {
                remainderIndex.put(remainder, i);
            }
        }
        return false;
    }


    // 30.3: Range Sum Query 2D - Immutable

    class NumMatrix {
        private int[][] prefixSum;

        public NumMatrix(int[][] matrix) {
            if (matrix.length == 0 || matrix[0].length == 0) return;

            int rows = matrix.length, cols = matrix[0].length;
            prefixSum = new int[rows + 1][cols + 1];

            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= cols; j++) {
                    prefixSum[i][j] = matrix[i - 1][j - 1] + prefixSum[i - 1][j] + prefixSum[i][j - 1] - prefixSum[i - 1][j - 1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return prefixSum[row2 + 1][col2 + 1] - prefixSum[row1][col2 + 1] - prefixSum[row2 + 1][col1] + prefixSum[row1][col1];
        }
    }


}
