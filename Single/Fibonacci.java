package com.god.file.individual;

public class Fibonacci {

    // Pattern 23: FIBONACCI (DYNAMIC PROGRAMMING)
    // Use Case: Sequence problems, counting ways


    // 23.1: Climbing Stairs

    public int climbStairs(int n) {
        if (n <= 2) return n;

        int first = 1, second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }


    // 23.2: House Robber

    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];

        int prev2 = 0, prev1 = 0;
        for (int num : nums) {
            int current = Math.max(prev1, prev2 + num);
            prev2 = prev1;
            prev1 = current;
        }
        return prev1;
    }


    // 23.3: Min Cost Climbing Stairs

    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int first = 0, second = 0;

        for (int i = 2; i <= n; i++) {
            int current = Math.min(first + cost[i - 2], second + cost[i - 1]);
            first = second;
            second = current;
        }
        return second;
    }


}
