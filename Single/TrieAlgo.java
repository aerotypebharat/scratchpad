package com.god.file.individual;

import java.util.ArrayList;
import java.util.List;

public class TrieAlgo {

    // Pattern 26: TRIE
    // Use Case: Prefix searching, dictionary problems


    // 26.1: Implement Trie (Prefix Tree)
    // Time: O(L) for insert/search/startsWith, Space: O(N*L)

    class Trie {
        class TrieNode {
            Trie.TrieNode[] children;
            boolean isEnd;

            public TrieNode() {
                children = new Trie.TrieNode[26];
                isEnd = false;
            }
        }

        private Trie.TrieNode root;

        public Trie() {
            root = new Trie.TrieNode();
        }

        public void insert(String word) {
            Trie.TrieNode node = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie.TrieNode();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            Trie.TrieNode node = searchPrefix(word);
            return node != null && node.isEnd;
        }

        public boolean startsWith(String prefix) {
            return searchPrefix(prefix) != null;
        }

        private Trie.TrieNode searchPrefix(String prefix) {
            Trie.TrieNode node = root;
            for (char c : prefix.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    return null;
                }
                node = node.children[index];
            }
            return node;
        }
    }


    // 26.2: Word Search II
    // Find all words from dictionary in board using Trie
    // Time: O(m*n*4^L), Space: O(k*L) where k is number of words

    public List<String> findWords(char[][] board, String[] words) {
        List<String> result = new ArrayList<>();
        Trie trie = new Trie();

        // Build trie with all words
        for (String word : words) {
            trie.insert(word);
        }

        int rows = board.length, cols = board[0].length;
        boolean[][] visited = new boolean[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dfsWordSearch(board, i, j, trie.root, new StringBuilder(), result, visited);
            }
        }
        return result;
    }

    private void dfsWordSearch(char[][] board, int i, int j, Trie.TrieNode node, StringBuilder current, List<String> result, boolean[][] visited) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || visited[i][j]) {
            return;
        }

        char c = board[i][j];
        int index = c - 'a';
        if (node.children[index] == null) return;

        node = node.children[index];
        current.append(c);
        visited[i][j] = true;

        if (node.isEnd) {
            result.add(current.toString());
            node.isEnd = false; // Avoid duplicates
        }

        // Explore neighbors
        dfsWordSearch(board, i + 1, j, node, current, result, visited);
        dfsWordSearch(board, i - 1, j, node, current, result, visited);
        dfsWordSearch(board, i, j + 1, node, current, result, visited);
        dfsWordSearch(board, i, j - 1, node, current, result, visited);

        // Backtrack
        current.deleteCharAt(current.length() - 1);
        visited[i][j] = false;
    }


    // 26.3: Replace Words
    // Replace words with their shortest root
    // Time: O(n*L), Space: O(n*L)

    public String replaceWords(List<String> dictionary, String sentence) {
        Trie trie = new Trie();

        // Build trie with dictionary
        for (String root : dictionary) {
            trie.insert(root);
        }

        String[] words = sentence.split(" ");
        StringBuilder result = new StringBuilder();

        for (String word : words) {
            if (result.length() > 0) result.append(" ");
            String root = findShortestRoot(trie, word);
            result.append(root != null ? root : word);
        }

        return result.toString();
    }

    private String findShortestRoot(Trie trie, String word) {
        Trie.TrieNode node = trie.root;
        StringBuilder prefix = new StringBuilder();

        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) return null;

            node = node.children[index];
            prefix.append(c);

            if (node.isEnd) return prefix.toString();
        }
        return null;
    }


}
