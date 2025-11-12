package com.god.file.individual;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SlidingWindow {

    //
    // Pattern 3: SLIDING WINDOW
    // Use Case: Subarrays/substrings with constraints, fixed/variable window
    //


    // 3.1: Maximum Average Subarray I

    public double findMaxAverage(int[] nums, int k) {
        double windowSum = 0;

        // Calculate first window
        for (int i = 0; i < k; i++) {
            windowSum += nums[i];
        }

        double maxSum = windowSum;

        // Slide the window
        for (int i = k; i < nums.length; i++) {
            windowSum += nums[i] - nums[i - k];
            maxSum = Math.max(maxSum, windowSum);
        }

        return maxSum / k;
    }


    // 3.2: Longest Substring Without Repeating Characters

    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> charIndex = new HashMap<>();
        int left = 0, maxLength = 0;

        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);

            // If character exists and is within current window
            if (charIndex.containsKey(c) && charIndex.get(c) >= left) {
                left = charIndex.get(c) + 1;
            }

            charIndex.put(c, right);
            maxLength = Math.max(maxLength, right - left + 1);
        }
        return maxLength;
    }


    // 3.3: Permutation in String
    // Check if s2 contains permutation of s1

    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) return false;

        int[] s1Count = new int[26];
        int[] windowCount = new int[26];

        for (char c : s1.toCharArray()) {
            s1Count[c - 'a']++;
        }

        // Initial window
        for (int i = 0; i < s1.length(); i++) {
            windowCount[s2.charAt(i) - 'a']++;
        }

        if (Arrays.equals(s1Count, windowCount)) return true;

        // Slide window
        for (int i = s1.length(); i < s2.length(); i++) {
            windowCount[s2.charAt(i) - 'a']++;
            windowCount[s2.charAt(i - s1.length()) - 'a']--;

            if (Arrays.equals(s1Count, windowCount)) return true;
        }

        return false;
    }

}
