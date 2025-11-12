package com.god.file.individual;

import java.util.*;

public class TopoSort {

    // Pattern 27: TOPOLOGICAL SORT (GRAPH)
    // Use Case: Dependency resolution, course scheduling


    // 27.1: Course Schedule II

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        // Build graph
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }

        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            graph.get(pre[1]).add(pre[0]);
            indegree[pre[0]]++;
        }

        // Topological sort using BFS (Kahn's algorithm)
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        int[] result = new int[numCourses];
        int index = 0;

        while (!queue.isEmpty()) {
            int course = queue.poll();
            result[index++] = course;

            for (int neighbor : graph.get(course)) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return index == numCourses ? result : new int[0];
    }


    // 27.2: Alien Dictionary

    public String alienOrder(String[] words) {
        // Build graph
        Map<Character, Set<Character>> graph = new HashMap<>();
        Map<Character, Integer> indegree = new HashMap<>();

        // Initialize
        for (String word : words) {
            for (char c : word.toCharArray()) {
                graph.putIfAbsent(c, new HashSet<>());
                indegree.putIfAbsent(c, 0);
            }
        }

        // Build edges
        for (int i = 0; i < words.length - 1; i++) {
            String word1 = words[i], word2 = words[i + 1];
            int minLen = Math.min(word1.length(), word2.length());

            // Check for invalid order
            if (word1.length() > word2.length() && word1.startsWith(word2)) {
                return "";
            }

            for (int j = 0; j < minLen; j++) {
                char c1 = word1.charAt(j), c2 = word2.charAt(j);
                if (c1 != c2) {
                    if (!graph.get(c1).contains(c2)) {
                        graph.get(c1).add(c2);
                        indegree.put(c2, indegree.get(c2) + 1);
                    }
                    break;
                }
            }
        }

        // Topological sort
        Queue<Character> queue = new LinkedList<>();
        for (char c : indegree.keySet()) {
            if (indegree.get(c) == 0) {
                queue.offer(c);
            }
        }

        StringBuilder result = new StringBuilder();
        while (!queue.isEmpty()) {
            char c = queue.poll();
            result.append(c);

            for (char neighbor : graph.get(c)) {
                indegree.put(neighbor, indegree.get(neighbor) - 1);
                if (indegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return result.length() == indegree.size() ? result.toString() : "";
    }


    // 27.3: Sequence Reconstruction

    public boolean sequenceReconstruction(int[] nums, List<List<Integer>> sequences) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        Map<Integer, Integer> indegree = new HashMap<>();

        // Initialize graph
        for (int num : nums) {
            graph.putIfAbsent(num, new HashSet<>());
            indegree.putIfAbsent(num, 0);
        }

        // Build graph from sequences
        for (List<Integer> seq : sequences) {
            for (int i = 0; i < seq.size() - 1; i++) {
                int from = seq.get(i), to = seq.get(i + 1);
                if (graph.get(from).add(to)) {
                    indegree.put(to, indegree.get(to) + 1);
                }
            }
        }

        // Topological sort
        Queue<Integer> queue = new LinkedList<>();
        for (int num : nums) {
            if (indegree.get(num) == 0) {
                queue.offer(num);
            }
        }

        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            if (queue.size() > 1) return false; // Not unique

            int current = queue.poll();
            result.add(current);

            for (int neighbor : graph.get(current)) {
                indegree.put(neighbor, indegree.get(neighbor) - 1);
                if (indegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // Check if result matches nums
        if (result.size() != nums.length) return false;
        for (int i = 0; i < nums.length; i++) {
            if (result.get(i) != nums[i]) return false;
        }
        return true;
    }


}
