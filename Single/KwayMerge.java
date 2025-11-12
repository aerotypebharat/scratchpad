package com.god.file.individual;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

public class KwayMerge {

    // Pattern 20: K-WAY MERGE
    // Use Case: Merging multiple sorted arrays/lists


    // 20.1: Merge K Sorted Lists

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;

        PriorityQueue<ListNode> minHeap = new PriorityQueue<>((a, b) -> a.val - b.val);

        // Add head of all lists to heap
        for (ListNode list : lists) {
            if (list != null) {
                minHeap.offer(list);
            }
        }

        ListNode dummy = new ListNode(0);
        ListNode current = dummy;

        while (!minHeap.isEmpty()) {
            ListNode node = minHeap.poll();
            current.next = node;
            current = current.next;

            if (node.next != null) {
                minHeap.offer(node.next);
            }
        }
        return dummy.next;
    }


    // 20.2: Kth Smallest Element in Sorted Matrix

    public int kthSmallestMatrix(int[][] matrix, int k) {
        int n = matrix.length;
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);

        // Add first element of each row to heap
        for (int i = 0; i < n; i++) {
            minHeap.offer(new int[]{matrix[i][0], i, 0});
        }

        int count = 0;
        while (!minHeap.isEmpty()) {
            int[] element = minHeap.poll();
            int value = element[0], row = element[1], col = element[2];
            count++;

            if (count == k) return value;

            // Add next element from same row
            if (col + 1 < n) {
                minHeap.offer(new int[]{matrix[row][col + 1], row, col + 1});
            }
        }
        return -1;
    }


    // 20.3: Find K Pairs with Smallest Sums

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0 || k == 0) return result;

        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> (a[0] + a[1]) - (b[0] + b[1]));

        // Add pairs (nums1[i], nums2[0]) for all i
        for (int i = 0; i < Math.min(nums1.length, k); i++) {
            minHeap.offer(new int[]{nums1[i], nums2[0], 0});
        }

        while (k-- > 0 && !minHeap.isEmpty()) {
            int[] current = minHeap.poll();
            result.add(Arrays.asList(current[0], current[1]));

            int nums2Index = current[2];
            if (nums2Index + 1 < nums2.length) {
                minHeap.offer(new int[]{current[0], nums2[nums2Index + 1], nums2Index + 1});
            }
        }
        return result;
    }


}
