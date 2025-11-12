package com.god.file.individual;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MergeInterval {

    //
    // Pattern 4: MERGE INTERVALS
    // Use Case: Overlapping intervals, scheduling problems
    //


    // 4.1: Merge Intervals

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


    // 4.2: Insert Interval

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


    // 4.3: Meeting Rooms II
    // Find minimum conference rooms required

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

}
