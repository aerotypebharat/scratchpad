package com.god.file.individual;

public class TwoPointers {

    // Pattern 1: TWO POINTERS
    // Use Case: Sorted arrays/lists, pair searching, in-place operations



    // 1.1: Two Sum II - Input Array Is Sorted
    // Find two numbers that add up to target in sorted array

    public int[] twoSumSorted(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[]{left + 1, right + 1};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        return new int[]{-1, -1};
    }


    // 1.2: Container With Most Water
    // Find two lines that form container with most water

    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int maxArea = 0;

        while (left < right) {
            int currentArea = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, currentArea);

            // Move the pointer with smaller height
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }


    // 1.3: Remove Duplicates from Sorted Array
    // Remove duplicates in-place, return new length

    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;

        int uniquePtr = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[uniquePtr]) {
                uniquePtr++;
                nums[uniquePtr] = nums[i];
            }
        }
        return uniquePtr + 1;
    }

}
