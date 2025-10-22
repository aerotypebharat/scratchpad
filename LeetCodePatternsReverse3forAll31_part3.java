package com.god.file; /**
 * LeetCode Patterns Collection - 31 Patterns, 93 Problems with Java Solutions
 * REVERSE ORDER: Pattern 31 to Pattern 1 for practicing from the end first
 * GitHub: https://github.com/yourusername/leetcode-patterns
 */

import java.util.*;

public class LeetCodePatternsReverse_part3 {
    
    public static void main(String[] args) {
        System.out.println("LeetCode Patterns Collection - REVERSE ORDER (31 to 1)");
        System.out.println("93 Problems - Practice from the end first!");
    }

    // =========================================================================
    // PATTERN 10: LEVEL ORDER TRAVERSAL
    // Use Case: Tree traversal level by level
    // =========================================================================
    
    /**
     * 10.1: Binary Tree Level Order Traversal
     * Time: O(n), Space: O(n)
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(currentLevel);
        }
        return result;
    }
    
    /**
     * 10.2: Binary Tree Zigzag Level Order Traversal
     * Time: O(n), Space: O(n)
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean leftToRight = true;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (leftToRight) {
                    currentLevel.add(node.val);
                } else {
                    currentLevel.add(0, node.val);
                }
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(currentLevel);
            leftToRight = !leftToRight;
        }
        return result;
    }
    
    /**
     * 10.3: Average of Levels in Binary Tree
     * Time: O(n), Space: O(n)
     */
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            double levelSum = 0;
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                levelSum += node.val;
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            result.add(levelSum / levelSize);
        }
        return result;
    }

    // =========================================================================
    // PATTERN 9: HASH MAPS
    // Use Case: Frequency counting, lookups, caching
    // =========================================================================
    
    /**
     * 9.1: Two Sum
     * Time: O(n), Space: O(n)
     */
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> numMap = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numMap.containsKey(complement)) {
                return new int[]{numMap.get(complement), i};
            }
            numMap.put(nums[i], i);
        }
        return new int[]{-1, -1};
    }
    
    /**
     * 9.2: Group Anagrams
     * Time: O(n*k log k), Space: O(n*k)
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String sorted = new String(chars);
            
            if (!map.containsKey(sorted)) {
                map.put(sorted, new ArrayList<>());
            }
            map.get(sorted).add(str);
        }
        
        return new ArrayList<>(map.values());
    }
    
    /**
     * 9.3: Longest Consecutive Sequence
     * Time: O(n), Space: O(n)
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }
        
        int longest = 0;
        
        for (int num : numSet) {
            // Check if it's the start of a sequence
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                
                while (numSet.contains(currentNum + 1)) {
                    currentNum++;
                    currentStreak++;
                }
                
                longest = Math.max(longest, currentStreak);
            }
        }
        return longest;
    }

    // =========================================================================
    // PATTERN 8: MONOTONIC STACK
    // Use Case: Next greater/smaller element, stock span, histogram problems
    // =========================================================================
    
    /**
     * 8.1: Next Greater Element I
     * Time: O(n), Space: O(n)
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> nextGreater = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        
        for (int num : nums2) {
            while (!stack.isEmpty() && num > stack.peek()) {
                nextGreater.put(stack.pop(), num);
            }
            stack.push(num);
        }
        
        int[] result = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            result[i] = nextGreater.getOrDefault(nums1[i], -1);
        }
        return result;
    }
    
    /**
     * 8.2: Trapping Rain Water
     * Time: O(n), Space: O(n)
     */
    public int trapRainWater(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int totalWater = 0;
        
        for (int i = 0; i < height.length; i++) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int bottom = stack.pop();
                if (stack.isEmpty()) break;
                
                int distance = i - stack.peek() - 1;
                int boundedHeight = Math.min(height[i], height[stack.peek()]) - height[bottom];
                totalWater += distance * boundedHeight;
            }
            stack.push(i);
        }
        return totalWater;
    }
    
    /**
     * 8.3: Largest Rectangle in Histogram
     * Time: O(n), Space: O(n)
     */
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int n = heights.length;
        
        for (int i = 0; i <= n; i++) {
            int currentHeight = (i == n) ? 0 : heights[i];
            
            while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            stack.push(i);
        }
        return maxArea;
    }

    // =========================================================================
    // PATTERN 7: STACK
    // Use Case: LIFO operations, parsing, backtracking
    // =========================================================================
    
    /**
     * 7.1: Valid Parentheses
     * Time: O(n), Space: O(n)
     */
    public boolean isValidParentheses(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) return false;
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == ']' && top != '[') ||
                    (c == '}' && top != '{')) {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }
    
    /**
     * 7.2: Daily Temperatures
     * Time: O(n), Space: O(n)
     */
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int idx = stack.pop();
                result[idx] = i - idx;
            }
            stack.push(i);
        }
        return result;
    }
    
    /**
     * 7.3: Evaluate Reverse Polish Notation
     * Time: O(n), Space: O(n)
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        
        for (String token : tokens) {
            if (token.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (token.equals("-")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a - b);
            } else if (token.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (token.equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a / b);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    // =========================================================================
    // PATTERN 6: IN-PLACE REVERSAL OF LINKEDLIST
    // Use Case: Reverse linked lists or portions of them
    // =========================================================================
    
    /**
     * 6.1: Reverse Linked List
     * Time: O(n), Space: O(1)
     */
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode nextTemp = current.next;
            current.next = prev;
            prev = current;
            current = nextTemp;
        }
        return prev;
    }
    
    /**
     * 6.2: Reverse Linked List II
     * Time: O(n), Space: O(1)
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null || left == right) return head;
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        // Move to the node before reversal start
        for (int i = 1; i < left; i++) {
            prev = prev.next;
        }
        
        ListNode start = prev.next;
        ListNode then = start.next;
        
        // Reverse the sublist
        for (int i = 0; i < right - left; i++) {
            start.next = then.next;
            then.next = prev.next;
            prev.next = then;
            then = start.next;
        }
        
        return dummy.next;
    }
    
    /**
     * 6.3: Reverse Nodes in k-Group
     * Time: O(n), Space: O(1)
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode curr = dummy, prev = dummy, next = dummy;
        
        int count = 0;
        while (curr.next != null) {
            curr = curr.next;
            count++;
        }
        
        while (count >= k) {
            curr = prev.next;
            next = curr.next;
            
            for (int i = 1; i < k; i++) {
                curr.next = next.next;
                next.next = prev.next;
                prev.next = next;
                next = curr.next;
            }
            prev = curr;
            count -= k;
        }
        return dummy.next;
    }

    // =========================================================================
    // PATTERN 5: CYCLIC SORT
    // Use Case: Arrays with numbers in range [1, n], missing/duplicate numbers
    // =========================================================================
    
    /**
     * 5.1: Find All Missing Numbers
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 5.2: Find Duplicate Number using Cyclic Sort
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 5.3: First Missing Positive
     * Time: O(n), Space: O(1)
     */
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

    // =========================================================================
    // PATTERN 4: MERGE INTERVALS
    // Use Case: Overlapping intervals, scheduling problems
    // =========================================================================
    
    /**
     * 4.1: Merge Intervals
     * Time: O(n log n), Space: O(n)
     */
    public int[][] mergeIntervals(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        
        // Sort by start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> merged = new ArrayList<>();
        int[] current = intervals[0];
        merged.add(current);
        
        for (int[] interval : intervals) {
            if (interval[0] <= current[1]) {
                // Overlapping intervals, merge
                current[1] = Math.max(current[1], interval[1]);
            } else {
                // Non-overlapping interval, add to list
                current = interval;
                merged.add(current);
            }
        }
        
        return merged.toArray(new int[merged.size()][]);
    }
    
    /**
     * 4.2: Insert Interval
     * Time: O(n), Space: O(n)
     */
    public int[][] insertInterval(int[][] intervals, int[] newInterval) {
        List<int[]> result = new ArrayList<>();
        int i = 0;
        int n = intervals.length;
        
        // Add all intervals ending before newInterval starts
        while (i < n && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i++]);
        }
        
        // Merge overlapping intervals
        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.add(newInterval);
        
        // Add remaining intervals
        while (i < n) {
            result.add(intervals[i++]);
        }
        
        return result.toArray(new int[result.size()][]);
    }
    
    /**
     * 4.3: Meeting Rooms II
     * Time: O(n log n), Space: O(n)
     */
    public int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        int[] starts = new int[intervals.length];
        int[] ends = new int[intervals.length];
        
        for (int i = 0; i < intervals.length; i++) {
            starts[i] = intervals[i][0];
            ends[i] = intervals[i][1];
        }
        
        Arrays.sort(starts);
        Arrays.sort(ends);
        
        int rooms = 0;
        int endPtr = 0;
        
        for (int start : starts) {
            if (start < ends[endPtr]) {
                rooms++;
            } else {
                endPtr++;
            }
        }
        
        return rooms;
    }

    // =========================================================================
    // PATTERN 3: SLIDING WINDOW
    // Use Case: Subarrays/substrings with constraints, fixed/variable window
    // =========================================================================
    
    /**
     * 3.1: Maximum Average Subarray I
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 3.2: Longest Substring Without Repeating Characters
     * Time: O(n), Space: O(min(m,n))
     */
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
    
    /**
     * 3.3: Permutation in String
     * Time: O(l1 + l2), Space: O(1)
     */
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

    // =========================================================================
    // PATTERN 2: FAST & SLOW POINTERS
    // Use Case: Cycle detection, middle element, duplicate finding
    // =========================================================================
    
    /**
     * 2.1: Linked List Cycle Detection
     * Time: O(n), Space: O(1)
     */
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }
    
    /**
     * 2.2: Find Duplicate Number (Floyd's Tortoise and Hare)
     * Time: O(n), Space: O(1)
     */
    public int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];
        
        // Phase 1: Find intersection point
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        
        // Phase 2: Find entrance to cycle
        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
    
    /**
     * 2.3: Middle of Linked List
     * Time: O(n), Space: O(1)
     */
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    // =========================================================================
    // PATTERN 1: TWO POINTERS
    // Use Case: Sorted arrays/lists, pair searching, in-place operations
    // =========================================================================
    
    /**
     * 1.1: Two Sum II - Input Array Is Sorted
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 1.2: Container With Most Water
     * Time: O(n), Space: O(1)
     */
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
    
    /**
     * 1.3: Remove Duplicates from Sorted Array
     * Time: O(n), Space: O(1)
     */
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

    // =========================================================================
    // SUPPORTING DATA STRUCTURES
    // =========================================================================

    // ListNode definition for linked list problems
    class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    // TreeNode definition for tree problems
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

}

/**
 * REVERSE ORDER SUMMARY:
 * Patterns are listed from 31 (Multi-thread) to 1 (Two Pointers)
 * This allows practicing from the end patterns first
 *
 * Total: 31 Patterns, 93 Problems with complete Java solutions
 * Each pattern includes time/space complexity analysis
 *
 * Perfect for systematic interview preparation starting from advanced topics!
 */
