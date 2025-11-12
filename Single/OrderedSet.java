package com.god.file.individual;

import java.util.*;

public class OrderedSet {

    // Pattern 29: ORDERED SET
    // Use Case: Range queries, sorted data operations


    // 29.1: Contains Duplicate III

    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        TreeSet<Long> set = new TreeSet<>();

        for (int i = 0; i < nums.length; i++) {
            Long floor = set.floor((long) nums[i] + valueDiff);
            Long ceiling = set.ceiling((long) nums[i] - valueDiff);

            if ((floor != null && floor >= nums[i]) || (ceiling != null && ceiling <= nums[i])) {
                return true;
            }

            set.add((long) nums[i]);

            if (i >= indexDiff) {
                set.remove((long) nums[i - indexDiff]);
            }
        }
        return false;
    }


    // 29.2: Count of Smaller Numbers After Self

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        List<Integer> result = new ArrayList<>();

        // Coordinate compression
        int[] sortedNums = nums.clone();
        Arrays.sort(sortedNums);
        Map<Integer, Integer> ranks = new HashMap<>();
        int rank = 0;
        for (int num : sortedNums) {
            if (!ranks.containsKey(num)) {
                ranks.put(num, ++rank);
            }
        }

        // Binary Indexed Tree (Fenwick Tree)
        int[] BIT = new int[rank + 1];

        for (int i = n - 1; i >= 0; i--) {
            int currentRank = ranks.get(nums[i]);
            result.add(queryBIT(BIT, currentRank - 1));
            updateBIT(BIT, currentRank, 1);
        }

        Collections.reverse(result);
        return result;
    }

    private void updateBIT(int[] BIT, int index, int value) {
        while (index < BIT.length) {
            BIT[index] += value;
            index += index & -index;
        }
    }

    private int queryBIT(int[] BIT, int index) {
        int sum = 0;
        while (index > 0) {
            sum += BIT[index];
            index -= index & -index;
        }
        return sum;
    }


    // 29.3: Data Stream as Disjoint Intervals

    class SummaryRanges {
        private TreeSet<int[]> intervals;
        private Set<Integer> values;

        public SummaryRanges() {
            intervals = new TreeSet<>((a, b) -> a[0] - b[0]);
            values = new HashSet<>();
        }

        public void addNum(int value) {
            if (values.contains(value)) return;
            values.add(value);

            int[] newInterval = {value, value};
            int[] floor = intervals.floor(newInterval);
            int[] ceiling = intervals.ceiling(newInterval);

            // Merge with left interval
            if (floor != null && floor[1] + 1 == value) {
                intervals.remove(floor);
                newInterval[0] = floor[0];
            }

            // Merge with right interval
            if (ceiling != null && value + 1 == ceiling[0]) {
                intervals.remove(ceiling);
                newInterval[1] = ceiling[1];
            }

            intervals.add(newInterval);
        }

        public int[][] getIntervals() {
            return intervals.toArray(new int[intervals.size()][]);
        }
    }


}
