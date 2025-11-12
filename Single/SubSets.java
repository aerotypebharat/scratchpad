package com.god.file.individual;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SubSets {

    // Pattern 16: SUBSETS
    // Use Case: Combinations, permutations, power set


    // 16.1: Subsets

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackSubsets(nums, 0, new ArrayList<>(), result);
        return result;
    }

    private void backtrackSubsets(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));

        for (int i = start; i < nums.length; i++) {
            current.add(nums[i]);
            backtrackSubsets(nums, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }


    // 16.2: Subsets II (with duplicates)

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums); // Sort to handle duplicates
        List<List<Integer>> result = new ArrayList<>();
        backtrackSubsetsWithDup(nums, 0, new ArrayList<>(), result);
        return result;
    }

    private void backtrackSubsetsWithDup(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));

        for (int i = start; i < nums.length; i++) {
            // Skip duplicates
            if (i > start && nums[i] == nums[i - 1]) continue;

            current.add(nums[i]);
            backtrackSubsetsWithDup(nums, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }


    // 16.3: Letter Case Permutation

    public List<String> letterCasePermutation(String s) {
        List<String> result = new ArrayList<>();
        backtrackLetterCase(s.toCharArray(), 0, result);
        return result;
    }

    private void backtrackLetterCase(char[] chars, int index, List<String> result) {
        if (index == chars.length) {
            result.add(new String(chars));
            return;
        }

        // Skip if digit
        if (Character.isDigit(chars[index])) {
            backtrackLetterCase(chars, index + 1, result);
            return;
        }

        // Lower case branch
        chars[index] = Character.toLowerCase(chars[index]);
        backtrackLetterCase(chars, index + 1, result);

        // Upper case branch
        chars[index] = Character.toUpperCase(chars[index]);
        backtrackLetterCase(chars, index + 1, result);
    }


}
