package com.god.file.individual;

import java.util.Arrays;
import java.util.PriorityQueue;

public class TwoHeaps {

    // Pattern 15: TWO HEAPS
    // Use Case: Finding median, partitioning data streams


    // 15.1: Find Median from Data Stream

    class MedianFinder {
        private PriorityQueue<Integer> maxHeap; // lower half
        private PriorityQueue<Integer> minHeap; // upper half

        public MedianFinder() {
            maxHeap = new PriorityQueue<>((a, b) -> b - a);
            minHeap = new PriorityQueue<>();
        }

        public void addNum(int num) {
            // Add to appropriate heap
            if (maxHeap.isEmpty() || num <= maxHeap.peek()) {
                maxHeap.offer(num);
            } else {
                minHeap.offer(num);
            }

            // Balance heaps
            if (maxHeap.size() > minHeap.size() + 1) {
                minHeap.offer(maxHeap.poll());
            } else if (minHeap.size() > maxHeap.size()) {
                maxHeap.offer(minHeap.poll());
            }
        }

        public double findMedian() {
            if (maxHeap.size() == minHeap.size()) {
                return (maxHeap.peek() + minHeap.peek()) / 2.0;
            } else {
                return maxHeap.peek();
            }
        }
    }


    // 15.2: Sliding Window Median

    public double[] medianSlidingWindow(int[] nums, int k) {
        double[] result = new double[nums.length - k + 1];
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> Integer.compare(b, a));
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int i = 0; i < nums.length; i++) {
            // Add new element
            if (maxHeap.isEmpty() || nums[i] <= maxHeap.peek()) {
                maxHeap.offer(nums[i]);
            } else {
                minHeap.offer(nums[i]);
            }

            // Remove element leaving window
            if (i >= k) {
                int toRemove = nums[i - k];
                if (toRemove <= maxHeap.peek()) {
                    maxHeap.remove(toRemove);
                } else {
                    minHeap.remove(toRemove);
                }
            }

            // Balance heaps
            balanceHeaps(maxHeap, minHeap);

            // Calculate median
            if (i >= k - 1) {
                if (maxHeap.size() == minHeap.size()) {
                    result[i - k + 1] = ((double) maxHeap.peek() + (double) minHeap.peek()) / 2.0;
                } else {
                    result[i - k + 1] = maxHeap.peek();
                }
            }
        }
        return result;
    }

    private void balanceHeaps(PriorityQueue<Integer> maxHeap, PriorityQueue<Integer> minHeap) {
        while (maxHeap.size() > minHeap.size() + 1) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }


    // 15.3: IPO

    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int n = profits.length;
        int[][] projects = new int[n][2];

        for (int i = 0; i < n; i++) {
            projects[i][0] = capital[i];
            projects[i][1] = profits[i];
        }

        // Sort by capital required
        Arrays.sort(projects, (a, b) -> a[0] - b[0]);

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
        int i = 0;

        while (k-- > 0) {
            // Add all affordable projects to max heap
            while (i < n && projects[i][0] <= w) {
                maxHeap.offer(projects[i][1]);
                i++;
            }

            if (maxHeap.isEmpty()) break;

            // Select project with maximum profit
            w += maxHeap.poll();
        }
        return w;
    }


}
