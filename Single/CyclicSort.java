package com.god.file.individual;

import java.util.ArrayList;
import java.util.List;

public class CyclicSort {

    //
    // Pattern 5: CYCLIC SORT
    // Use Case: Arrays with numbers in range [1, n], missing/duplicate numbers
    //


    // 5.1: Find All Missing Numbers

    public List<Integer> findDisappearedNumbers(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            int correctPos = nums[i] - 1;
            if (nums[i] != nums[correctPos]) {
                swap(nums, i, correctPos);
            } else {
                i++;
            }
        }

        List<Integer> missing = new ArrayList<>();
        for (i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                missing.add(i + 1);
            }
        }
        return missing;
    }


    // 5.2: Find Duplicate Number using Cyclic Sort

    public int findDuplicateCyclic(int[] nums) {
        int i = 0;
        while (i < nums.length) {
            if (nums[i] != i + 1) {
                int correctPos = nums[i] - 1;
                if (nums[i] != nums[correctPos]) {
                    swap(nums, i, correctPos);
                } else {
                    return nums[i];
                }
            } else {
                i++;
            }
        }
        return -1;
    }


    // 5.3: First Missing Positive

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        int i = 0;

        while (i < n) {
            if (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            } else {
                i++;
            }
        }

        for (i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

}
