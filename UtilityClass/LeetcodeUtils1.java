import java.util.*;
import java.util.stream.*;
import java.util.function.*;
import java.util.Collections;

/**
 * Comprehensive Utility Class for LeetCode Problems
 * Contains helpful methods for Arrays, Strings, Stack, PriorityQueue, Comparator,
 * Queue, Deque, Character, Stream, Set, Map, Collections and common patterns
 */
public class LeetCodeUtils1 {

    // ==================== ARRAYS ====================
    
    /**
     * Array Utility Methods
     */
    public static class ArrayUtils {
        
        // Array initialization
        public static int[] createArray(int... values) {
            return values;
        }
        
        public static Integer[] createBoxedArray(int... values) {
            return Arrays.stream(values).boxed().toArray(Integer[]::new);
        }
        
        // Sorting
        public static void sort(int[] arr) {
            Arrays.sort(arr);
        }
        
        public static void sortDescending(int[] arr) {
            arr = Arrays.stream(arr)
                       .boxed()
                       .sorted(Collections.reverseOrder())
                       .mapToInt(Integer::intValue)
                       .toArray();
        }
        
        public static void sort2D(int[][] arr, int column) {
            Arrays.sort(arr, (a, b) -> a[column] - b[column]);
        }
        
        public static void sort2DDescending(int[][] arr, int column) {
            Arrays.sort(arr, (a, b) -> b[column] - a[column]);
        }
        
        public static void sort2DMultiple(int[][] arr) {
            Arrays.sort(arr, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        }
        
        // Binary Search
        public static int binarySearch(int[] arr, int target) {
            return Arrays.binarySearch(arr, target);
        }
        
        public static int binarySearchLeftMost(int[] arr, int target) {
            int left = 0, right = arr.length - 1;
            int result = -1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] >= target) {
                    if (arr[mid] == target) result = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return result;
        }
        
        public static int binarySearchRightMost(int[] arr, int target) {
            int left = 0, right = arr.length - 1;
            int result = -1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] <= target) {
                    if (arr[mid] == target) result = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return result;
        }
        
        // Array manipulation
        public static void fill(int[] arr, int value) {
            Arrays.fill(arr, value);
        }
        
        public static void fill2D(int[][] arr, int value) {
            for (int[] row : arr) {
                Arrays.fill(row, value);
            }
        }
        
        public static void fillRange(int[] arr, int start, int end, int value) {
            Arrays.fill(arr, start, end, value);
        }
        
        public static int[] copy(int[] arr) {
            return Arrays.copyOf(arr, arr.length);
        }
        
        public static int[] copyRange(int[] arr, int start, int end) {
            return Arrays.copyOfRange(arr, start, end);
        }
        
        public static int[][] deepCopy(int[][] arr) {
            int[][] copy = new int[arr.length][];
            for (int i = 0; i < arr.length; i++) {
                copy[i] = Arrays.copyOf(arr[i], arr[i].length);
            }
            return copy;
        }
        
        // Array conversion
        public static List<Integer> toList(int[] arr) {
            return Arrays.stream(arr).boxed().collect(Collectors.toList());
        }
        
        public static int[] toPrimitiveArray(List<Integer> list) {
            return list.stream().mapToInt(Integer::intValue).toArray();
        }
        
        public static Integer[] toBoxedArray(int[] arr) {
            return Arrays.stream(arr).boxed().toArray(Integer[]::new);
        }
        
        // Array statistics
        public static int sum(int[] arr) {
            return Arrays.stream(arr).sum();
        }
        
        public static int max(int[] arr) {
            return Arrays.stream(arr).max().orElse(Integer.MIN_VALUE);
        }
        
        public static int min(int[] arr) {
            return Arrays.stream(arr).min().orElse(Integer.MAX_VALUE);
        }
        
        public static double average(int[] arr) {
            return Arrays.stream(arr).average().orElse(0.0);
        }
        
        // Array checks
        public static boolean isSorted(int[] arr) {
            for (int i = 1; i < arr.length; i++) {
                if (arr[i] < arr[i-1]) return false;
            }
            return true;
        }
        
        public static boolean contains(int[] arr, int target) {
            return Arrays.stream(arr).anyMatch(x -> x == target);
        }
        
        public static int frequency(int[] arr, int target) {
            return (int) Arrays.stream(arr).filter(x -> x == target).count();
        }
        
        // Array printing
        public static void print(int[] arr) {
            System.out.println(Arrays.toString(arr));
        }
        
        public static void print2D(int[][] arr) {
            for (int[] row : arr) {
                System.out.println(Arrays.toString(row));
            }
        }
        
        // Prefix Sum
        public static int[] prefixSum(int[] arr) {
            int[] prefix = new int[arr.length];
            prefix[0] = arr[0];
            for (int i = 1; i < arr.length; i++) {
                prefix[i] = prefix[i-1] + arr[i];
            }
            return prefix;
        }
        
        public static int[] suffixSum(int[] arr) {
            int[] suffix = new int[arr.length];
            suffix[arr.length-1] = arr[arr.length-1];
            for (int i = arr.length-2; i >= 0; i--) {
                suffix[i] = suffix[i+1] + arr[i];
            }
            return suffix;
        }
    }

    // ==================== STRINGS ====================
    
    public static class StringUtils {
        
        // String creation and conversion
        public static char[] toCharArray(String s) {
            return s.toCharArray();
        }
        
        public static String fromCharArray(char[] chars) {
            return new String(chars);
        }
        
        public static List<Character> toCharList(String s) {
            return s.chars().mapToObj(c -> (char)c).collect(Collectors.toList());
        }
        
        // String reversal
        public static String reverse(String s) {
            return new StringBuilder(s).reverse().toString();
        }
        
        public static String reverseWords(String s) {
            String[] words = s.trim().split("\\s+");
            Collections.reverse(Arrays.asList(words));
            return String.join(" ", words);
        }
        
        // String building
        public static String buildString(String... parts) {
            StringBuilder sb = new StringBuilder();
            for (String part : parts) {
                sb.append(part);
            }
            return sb.toString();
        }
        
        public static String repeat(char c, int count) {
            return String.valueOf(c).repeat(count);
        }
        
        public static String joinWithDelimiter(String delimiter, String... parts) {
            return String.join(delimiter, parts);
        }
        
        // String validation
        public static boolean isPalindrome(String s) {
            int left = 0, right = s.length() - 1;
            while (left < right) {
                if (s.charAt(left++) != s.charAt(right--)) return false;
            }
            return true;
        }
        
        public static boolean isAnagram(String s1, String s2) {
            if (s1.length() != s2.length()) return false;
            char[] chars1 = s1.toCharArray();
            char[] chars2 = s2.toCharArray();
            Arrays.sort(chars1);
            Arrays.sort(chars2);
            return Arrays.equals(chars1, chars2);
        }
        
        public static boolean isSubsequence(String sub, String str) {
            int i = 0, j = 0;
            while (i < sub.length() && j < str.length()) {
                if (sub.charAt(i) == str.charAt(j)) i++;
                j++;
            }
            return i == sub.length();
        }
        
        // String manipulation
        public static String removeChar(String s, char c) {
            return s.replace(String.valueOf(c), "");
        }
        
        public static String removeSpaces(String s) {
            return s.replaceAll("\\s+", "");
        }
        
        public static String toLowerCase(String s) {
            return s.toLowerCase();
        }
        
        public static String toUpperCase(String s) {
            return s.toUpperCase();
        }
        
        // String analysis
        public static Map<Character, Integer> charFrequency(String s) {
            Map<Character, Integer> freq = new HashMap<>();
            for (char c : s.toCharArray()) {
                freq.put(c, freq.getOrDefault(c, 0) + 1);
            }
            return freq;
        }
        
        public static int countOccurrences(String s, char target) {
            return (int) s.chars().filter(c -> c == target).count();
        }
        
        public static String mostFrequentChar(String s) {
            Map<Character, Integer> freq = charFrequency(s);
            return String.valueOf(
                Collections.max(freq.entrySet(), Map.Entry.comparingByValue()).getKey()
            );
        }
        
        // String splitting and joining
        public static String[] splitBySpace(String s) {
            return s.trim().split("\\s+");
        }
        
        public static String[] splitByDelimiter(String s, String delimiter) {
            return s.split(delimiter);
        }
        
        public static String joinList(List<String> list, String delimiter) {
            return String.join(delimiter, list);
        }
    }

    // ==================== STACK ====================
    
    public static class StackUtils {
        
        // Stack creation
        public static <T> Stack<T> createStack() {
            return new Stack<>();
        }
        
        public static <T> Deque<T> createDequeStack() {
            return new ArrayDeque<>();
        }
        
        // Stack operations
        public static <T> void pushAll(Stack<T> stack, Collection<T> items) {
            for (T item : items) {
                stack.push(item);
            }
        }
        
        public static <T> List<T> popMultiple(Stack<T> stack, int count) {
            List<T> result = new ArrayList<>();
            for (int i = 0; i < count && !stack.isEmpty(); i++) {
                result.add(stack.pop());
            }
            Collections.reverse(result); // maintain original order
            return result;
        }
        
        public static <T> T peekOrDefault(Stack<T> stack, T defaultValue) {
            return stack.isEmpty() ? defaultValue : stack.peek();
        }
        
        // Stack conversion
        public static <T> List<T> toList(Stack<T> stack) {
            return new ArrayList<>(stack);
        }
        
        public static <T> Stack<T> fromList(List<T> list) {
            Stack<T> stack = new Stack<>();
            stack.addAll(list);
            return stack;
        }
        
        // Stack algorithms
        public static boolean isValidParentheses(String s) {
            Stack<Character> stack = new Stack<>();
            Map<Character, Character> mapping = Map.of(')', '(', '}', '{', ']', '[');
            
            for (char c : s.toCharArray()) {
                if (mapping.containsValue(c)) {
                    stack.push(c);
                } else if (mapping.containsKey(c)) {
                    if (stack.isEmpty() || stack.pop() != mapping.get(c)) {
                        return false;
                    }
                }
            }
            return stack.isEmpty();
        }
        
        public static int[] nextGreaterElement(int[] arr) {
            int[] result = new int[arr.length];
            Arrays.fill(result, -1);
            Stack<Integer> stack = new Stack<>();
            
            for (int i = 0; i < arr.length; i++) {
                while (!stack.isEmpty() && arr[stack.peek()] < arr[i]) {
                    result[stack.pop()] = arr[i];
                }
                stack.push(i);
            }
            return result;
        }
        
        public static int[] dailyTemperatures(int[] temperatures) {
            int[] result = new int[temperatures.length];
            Stack<Integer> stack = new Stack<>();
            
            for (int i = 0; i < temperatures.length; i++) {
                while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                    int idx = stack.pop();
                    result[idx] = i - idx;
                }
                stack.push(i);
            }
            return result;
        }
    }

    // ==================== PRIORITY QUEUE ====================
    
    public static class PriorityQueueUtils {
        
        // PriorityQueue creation
        public static <T> PriorityQueue<T> createMinHeap() {
            return new PriorityQueue<>();
        }
        
        public static <T> PriorityQueue<T> createMaxHeap() {
            return new PriorityQueue<>(Collections.reverseOrder());
        }
        
        public static PriorityQueue<Integer> createMinHeap(int[] arr) {
            PriorityQueue<Integer> pq = new PriorityQueue<>();
            for (int num : arr) pq.offer(num);
            return pq;
        }
        
        public static PriorityQueue<Integer> createMaxHeap(int[] arr) {
            PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
            for (int num : arr) pq.offer(num);
            return pq;
        }
        
        // Custom comparators
        public static PriorityQueue<int[]> minHeapByFirstElement() {
            return new PriorityQueue<>((a, b) -> a[0] - b[0]);
        }
        
        public static PriorityQueue<int[]> maxHeapByFirstElement() {
            return new PriorityQueue<>((a, b) -> b[0] - a[0]);
        }
        
        public static PriorityQueue<int[]> minHeapBySecondElement() {
            return new PriorityQueue<>((a, b) -> a[1] - b[1]);
        }
        
        public static PriorityQueue<String> maxHeapByLength() {
            return new PriorityQueue<>((a, b) -> b.length() - a.length());
        }
        
        // PriorityQueue operations
        public static <T> List<T> pollAll(PriorityQueue<T> pq) {
            List<T> result = new ArrayList<>();
            while (!pq.isEmpty()) {
                result.add(pq.poll());
            }
            return result;
        }
        
        public static <T> void offerAll(PriorityQueue<T> pq, Collection<T> items) {
            pq.addAll(items);
        }
        
        public static <T> T peekOrDefault(PriorityQueue<T> pq, T defaultValue) {
            return pq.isEmpty() ? defaultValue : pq.peek();
        }
        
        // Common algorithms
        public static int findKthLargest(int[] nums, int k) {
            PriorityQueue<Integer> minHeap = new PriorityQueue<>();
            for (int num : nums) {
                minHeap.offer(num);
                if (minHeap.size() > k) {
                    minHeap.poll();
                }
            }
            return minHeap.peek();
        }
        
        public static int findKthSmallest(int[] nums, int k) {
            PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
            for (int num : nums) {
                maxHeap.offer(num);
                if (maxHeap.size() > k) {
                    maxHeap.poll();
                }
            }
            return maxHeap.peek();
        }
        
        public static List<Integer> topKFrequent(int[] nums, int k) {
            Map<Integer, Integer> freq = new HashMap<>();
            for (int num : nums) {
                freq.put(num, freq.getOrDefault(num, 0) + 1);
            }
            
            PriorityQueue<Map.Entry<Integer, Integer>> pq = 
                new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
            
            for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
                pq.offer(entry);
                if (pq.size() > k) {
                    pq.poll();
                }
            }
            
            List<Integer> result = new ArrayList<>();
            while (!pq.isEmpty()) {
                result.add(pq.poll().getKey());
            }
            Collections.reverse(result);
            return result;
        }
    }

    // ==================== COMPARATOR ====================
    
    public static class ComparatorUtils {
        
        // Basic comparators
        public static Comparator<Integer> ascending() {
            return Integer::compare;
        }
        
        public static Comparator<Integer> descending() {
            return (a, b) -> b - a;
        }
        
        public static Comparator<String> byLength() {
            return Comparator.comparing(String::length);
        }
        
        public static Comparator<String> byLengthDescending() {
            return (a, b) -> b.length() - a.length();
        }
        
        // Array comparators
        public static Comparator<int[]> byFirstElement() {
            return (a, b) -> a[0] - b[0];
        }
        
        public static Comparator<int[]> byFirstElementDescending() {
            return (a, b) -> b[0] - a[0];
        }
        
        public static Comparator<int[]> bySecondElement() {
            return (a, b) -> a[1] - b[1];
        }
        
        public static Comparator<int[]> bySecondElementDescending() {
            return (a, b) -> b[1] - a[1];
        }
        
        public static Comparator<int[]> bySum() {
            return (a, b) -> (a[0] + a[1]) - (b[0] + b[1]);
        }
        
        // String comparators
        public static Comparator<String> lexicographical() {
            return String::compareTo;
        }
        
        public static Comparator<String> lexicographicalDescending() {
            return (a, b) -> b.compareTo(a);
        }
        
        // Custom object comparators
        public static <T> Comparator<T> byField(Function<T, Comparable> fieldExtractor) {
            return (a, b) -> fieldExtractor.apply(a).compareTo(fieldExtractor.apply(b));
        }
        
        public static <T> Comparator<T> byFieldDescending(Function<T, Comparable> fieldExtractor) {
            return (a, b) -> fieldExtractor.apply(b).compareTo(fieldExtractor.apply(a));
        }
        
        // Chained comparators
        public static <T> Comparator<T> chain(Comparator<T> first, Comparator<T> second) {
            return first.thenComparing(second);
        }
        
        // Null-safe comparators
        public static <T extends Comparable<T>> Comparator<T> nullsFirst() {
            return Comparator.nullsFirst(Comparator.naturalOrder());
        }
        
        public static <T extends Comparable<T>> Comparator<T> nullsLast() {
            return Comparator.nullsLast(Comparator.naturalOrder());
        }
    }

    // ==================== QUEUE & DEQUE ====================
    
    public static class QueueUtils {
        
        // Queue creation
        public static <T> Queue<T> createQueue() {
            return new LinkedList<>();
        }
        
        public static <T> Deque<T> createDeque() {
            return new ArrayDeque<>();
        }
        
        public static <T> Queue<T> createQueue(Collection<T> items) {
            return new LinkedList<>(items);
        }
        
        // Queue operations
        public static <T> void enqueueAll(Queue<T> queue, Collection<T> items) {
            queue.addAll(items);
        }
        
        public static <T> List<T> dequeueMultiple(Queue<T> queue, int count) {
            List<T> result = new ArrayList<>();
            for (int i = 0; i < count && !queue.isEmpty(); i++) {
                result.add(queue.poll());
            }
            return result;
        }
        
        public static <T> T peekOrDefault(Queue<T> queue, T defaultValue) {
            return queue.isEmpty() ? defaultValue : queue.peek();
        }
        
        // Deque operations
        public static <T> void addToFront(Deque<T> deque, T item) {
            deque.offerFirst(item);
        }
        
        public static <T> void addToBack(Deque<T> deque, T item) {
            deque.offerLast(item);
        }
        
        public static <T> T removeFromFront(Deque<T> deque) {
            return deque.pollFirst();
        }
        
        public static <T> T removeFromBack(Deque<T> deque) {
            return deque.pollLast();
        }
        
        public static <T> T peekFront(Deque<T> deque) {
            return deque.peekFirst();
        }
        
        public static <T> T peekBack(Deque<T> deque) {
            return deque.peekLast();
        }
        
        // Queue algorithms
        public static List<Integer> slidingWindowMax(int[] nums, int k) {
            List<Integer> result = new ArrayList<>();
            Deque<Integer> deque = new ArrayDeque<>();
            
            for (int i = 0; i < nums.length; i++) {
                // Remove indices that are out of window
                while (!deque.isEmpty() && deque.peekFirst() <= i - k) {
                    deque.pollFirst();
                }
                
                // Remove smaller elements from the back
                while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i]) {
                    deque.pollLast();
                }
                
                deque.offerLast(i);
                
                // Add to result when window is complete
                if (i >= k - 1) {
                    result.add(nums[deque.peekFirst()]);
                }
            }
            return result;
        }
        
        public static List<Integer> bfsLevelOrder(TreeNode root) {
            List<Integer> result = new ArrayList<>();
            if (root == null) return result;
            
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                result.add(node.val);
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            return result;
        }
    }

    // ==================== CHARACTER ====================
    
    public static class CharUtils {
        
        // Character checks
        public static boolean isVowel(char c) {
            c = Character.toLowerCase(c);
            return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
        }
        
        public static boolean isConsonant(char c) {
            return Character.isLetter(c) && !isVowel(c);
        }
        
        public static boolean isDigit(char c) {
            return Character.isDigit(c);
        }
        
        public static boolean isLetter(char c) {
            return Character.isLetter(c);
        }
        
        public static boolean isLetterOrDigit(char c) {
            return Character.isLetterOrDigit(c);
        }
        
        public static boolean isUpperCase(char c) {
            return Character.isUpperCase(c);
        }
        
        public static boolean isLowerCase(char c) {
            return Character.isLowerCase(c);
        }
        
        public static boolean isWhitespace(char c) {
            return Character.isWhitespace(c);
        }
        
        public static boolean isAlphanumeric(char c) {
            return Character.isLetterOrDigit(c);
        }
        
        // Character conversion
        public static char toLowerCase(char c) {
            return Character.toLowerCase(c);
        }
        
        public static char toUpperCase(char c) {
            return Character.toUpperCase(c);
        }
        
        public static int toDigit(char c) {
            return Character.digit(c, 10);
        }
        
        public static char fromDigit(int digit) {
            return Character.forDigit(digit, 10);
        }
        
        // Character operations
        public static char nextChar(char c) {
            return (char) (c + 1);
        }
        
        public static char previousChar(char c) {
            return (char) (c - 1);
        }
        
        public static int distance(char a, char b) {
            return Math.abs(a - b);
        }
        
        // String character operations
        public static String toLowerCase(String s) {
            char[] chars = s.toCharArray();
            for (int i = 0; i < chars.length; i++) {
                chars[i] = Character.toLowerCase(chars[i]);
            }
            return new String(chars);
        }
        
        public static String toUpperCase(String s) {
            char[] chars = s.toCharArray();
            for (int i = 0; i < chars.length; i++) {
                chars[i] = Character.toUpperCase(chars[i]);
            }
            return new String(chars);
        }
        
        public static String reverseString(String s) {
            char[] chars = s.toCharArray();
            int left = 0, right = chars.length - 1;
            while (left < right) {
                char temp = chars[left];
                chars[left] = chars[right];
                chars[right] = temp;
                left++;
                right--;
            }
            return new String(chars);
        }
        
        // Character frequency
        public static Map<Character, Integer> frequency(String s) {
            Map<Character, Integer> freq = new HashMap<>();
            for (char c : s.toCharArray()) {
                freq.put(c, freq.getOrDefault(c, 0) + 1);
            }
            return freq;
        }
        
        public static int countVowels(String s) {
            return (int) s.chars()
                         .mapToObj(c -> (char) c)
                         .filter(CharUtils::isVowel)
                         .count();
        }
        
        public static int countConsonants(String s) {
            return (int) s.chars()
                         .mapToObj(c -> (char) c)
                         .filter(CharUtils::isConsonant)
                         .count();
        }
    }
}