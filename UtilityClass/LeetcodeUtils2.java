class LeetCodeUtils2 {

    // ==================== STREAM API ====================
    
    public static class StreamUtils {
        
        // Array to Stream conversions
        public static IntStream stream(int[] arr) {
            return Arrays.stream(arr);
        }
        
        public static Stream<Integer> boxedStream(int[] arr) {
            return Arrays.stream(arr).boxed();
        }
        
        public static <T> Stream<T> stream(T[] arr) {
            return Arrays.stream(arr);
        }
        
        // Collection to Stream
        public static <T> Stream<T> stream(Collection<T> collection) {
            return collection.stream();
        }
        
        // Filter operations
        public static List<Integer> filterEven(int[] arr) {
            return Arrays.stream(arr)
                        .filter(n -> n % 2 == 0)
                        .boxed()
                        .collect(Collectors.toList());
        }
        
        public static List<Integer> filterOdd(int[] arr) {
            return Arrays.stream(arr)
                        .filter(n -> n % 2 != 0)
                        .boxed()
                        .collect(Collectors.toList());
        }
        
        public static List<Integer> filterPositive(int[] arr) {
            return Arrays.stream(arr)
                        .filter(n -> n > 0)
                        .boxed()
                        .collect(Collectors.toList());
        }
        
        public static List<String> filterByLength(String[] arr, int minLength) {
            return Arrays.stream(arr)
                        .filter(s -> s.length() >= minLength)
                        .collect(Collectors.toList());
        }
        
        // Map operations
        public static List<Integer> squareNumbers(int[] arr) {
            return Arrays.stream(arr)
                        .map(n -> n * n)
                        .boxed()
                        .collect(Collectors.toList());
        }
        
        public static List<Integer> doubleNumbers(int[] arr) {
            return Arrays.stream(arr)
                        .map(n -> n * 2)
                        .boxed()
                        .collect(Collectors.toList());
        }
        
        public static List<String> toUpperCase(String[] arr) {
            return Arrays.stream(arr)
                        .map(String::toUpperCase)
                        .collect(Collectors.toList());
        }
        
        public static List<Integer> stringLengths(String[] arr) {
            return Arrays.stream(arr)
                        .map(String::length)
                        .collect(Collectors.toList());
        }
        
        // Reduce operations
        public static int sum(int[] arr) {
            return Arrays.stream(arr).sum();
        }
        
        public static int product(int[] arr) {
            return Arrays.stream(arr).reduce(1, (a, b) -> a * b);
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
        
        // Statistical operations
        public static IntSummaryStatistics statistics(int[] arr) {
            return Arrays.stream(arr).summaryStatistics();
        }
        
        public static long countDistinct(int[] arr) {
            return Arrays.stream(arr).distinct().count();
        }
        
        public static List<Integer> distinctElements(int[] arr) {
            return Arrays.stream(arr).distinct().boxed().collect(Collectors.toList());
        }
        
        // Grouping and partitioning
        public static Map<Boolean, List<Integer>> partitionByEvenOdd(int[] arr) {
            return Arrays.stream(arr)
                        .boxed()
                        .collect(Collectors.partitioningBy(n -> n % 2 == 0));
        }
        
        public static Map<Integer, List<Integer>> groupByRemainder(int[] arr, int divisor) {
            return Arrays.stream(arr)
                        .boxed()
                        .collect(Collectors.groupingBy(n -> n % divisor));
        }
        
        public static Map<Integer, Long> frequencyMap(int[] arr) {
            return Arrays.stream(arr)
                        .boxed()
                        .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        }
        
        // Sorting with streams
        public static List<Integer> sortAscending(int[] arr) {
            return Arrays.stream(arr)
                        .sorted()
                        .boxed()
                        .collect(Collectors.toList());
        }
        
        public static List<Integer> sortDescending(int[] arr) {
            return Arrays.stream(arr)
                        .boxed()
                        .sorted(Collections.reverseOrder())
                        .collect(Collectors.toList());
        }
        
        public static List<String> sortByLength(String[] arr) {
            return Arrays.stream(arr)
                        .sorted(Comparator.comparing(String::length))
                        .collect(Collectors.toList());
        }
        
        // Finding operations
        public static Optional<Integer> findFirstEven(int[] arr) {
            return Arrays.stream(arr)
                        .filter(n -> n % 2 == 0)
                        .boxed()
                        .findFirst();
        }
        
        public static boolean anyMatchGreaterThan(int[] arr, int threshold) {
            return Arrays.stream(arr).anyMatch(n -> n > threshold);
        }
        
        public static boolean allMatchPositive(int[] arr) {
            return Arrays.stream(arr).allMatch(n -> n > 0);
        }
        
        public static boolean noneMatchNegative(int[] arr) {
            return Arrays.stream(arr).noneMatch(n -> n < 0);
        }
    }

    // ==================== SET ====================
    
    public static class SetUtils {
        
        // Set creation
        public static <T> Set<T> createSet() {
            return new HashSet<>();
        }
        
        public static <T> Set<T> createSet(Collection<T> items) {
            return new HashSet<>(items);
        }
        
        public static <T> LinkedHashSet<T> createLinkedSet() {
            return new LinkedHashSet<>();
        }
        
        public static <T> TreeSet<T> createTreeSet() {
            return new TreeSet<>();
        }
        
        public static Set<Integer> createSet(int[] arr) {
            return Arrays.stream(arr).boxed().collect(Collectors.toSet());
        }
        
        // Set operations
        public static <T> Set<T> union(Set<T> set1, Set<T> set2) {
            Set<T> result = new HashSet<>(set1);
            result.addAll(set2);
            return result;
        }
        
        public static <T> Set<T> intersection(Set<T> set1, Set<T> set2) {
            Set<T> result = new HashSet<>(set1);
            result.retainAll(set2);
            return result;
        }
        
        public static <T> Set<T> difference(Set<T> set1, Set<T> set2) {
            Set<T> result = new HashSet<>(set1);
            result.removeAll(set2);
            return result;
        }
        
        public static <T> Set<T> symmetricDifference(Set<T> set1, Set<T> set2) {
            Set<T> union = union(set1, set2);
            Set<T> intersection = intersection(set1, set2);
            return difference(union, intersection);
        }
        
        public static <T> boolean isSubset(Set<T> subset, Set<T> superset) {
            return superset.containsAll(subset);
        }
        
        public static <T> boolean isSuperset(Set<T> superset, Set<T> subset) {
            return superset.containsAll(subset);
        }
        
        public static <T> boolean areDisjoint(Set<T> set1, Set<T> set2) {
            return intersection(set1, set2).isEmpty();
        }
        
        // Set utilities
        public static <T> void addAll(Set<T> set, T... items) {
            Collections.addAll(set, items);
        }
        
        public static <T> Set<T> copy(Set<T> set) {
            return new HashSet<>(set);
        }
        
        public static <T> List<T> toList(Set<T> set) {
            return new ArrayList<>(set);
        }
        
        public static Set<Integer> toSet(int[] arr) {
            return Arrays.stream(arr).boxed().collect(Collectors.toSet());
        }
        
        public static int[] toArray(Set<Integer> set) {
            return set.stream().mapToInt(Integer::intValue).toArray();
        }
        
        // Set algorithms
        public static int[] findMissingNumbers(int[] arr, int n) {
            Set<Integer> present = createSet(arr);
            List<Integer> missing = new ArrayList<>();
            
            for (int i = 1; i <= n; i++) {
                if (!present.contains(i)) {
                    missing.add(i);
                }
            }
            
            return missing.stream().mapToInt(Integer::intValue).toArray();
        }
        
        public static int[] findDuplicateNumbers(int[] arr) {
            Set<Integer> seen = new HashSet<>();
            Set<Integer> duplicates = new HashSet<>();
            
            for (int num : arr) {
                if (!seen.add(num)) {
                    duplicates.add(num);
                }
            }
            
            return duplicates.stream().mapToInt(Integer::intValue).toArray();
        }
        
        public static boolean containsDuplicate(int[] arr) {
            Set<Integer> set = new HashSet<>();
            for (int num : arr) {
                if (!set.add(num)) return true;
            }
            return false;
        }
        
        public static int countUnique(int[] arr) {
            return createSet(arr).size();
        }
    }

    // ==================== MAP ====================
    
    public static class MapUtils {
        
        // Map creation
        public static <K, V> Map<K, V> createMap() {
            return new HashMap<>();
        }
        
        public static <K, V> LinkedHashMap<K, V> createLinkedMap() {
            return new LinkedHashMap<>();
        }
        
        public static <K, V> TreeMap<K, V> createTreeMap() {
            return new TreeMap<>();
        }
        
        public static Map<Integer, Integer> frequencyMap(int[] arr) {
            Map<Integer, Integer> freq = new HashMap<>();
            for (int num : arr) {
                freq.put(num, freq.getOrDefault(num, 0) + 1);
            }
            return freq;
        }
        
        public static Map<Character, Integer> charFrequency(String s) {
            Map<Character, Integer> freq = new HashMap<>();
            for (char c : s.toCharArray()) {
                freq.put(c, freq.getOrDefault(c, 0) + 1);
            }
            return freq;
        }
        
        // Map operations
        public static <K, V> void increment(Map<K, Integer> map, K key) {
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        
        public static <K, V> void decrement(Map<K, Integer> map, K key) {
            map.put(key, map.getOrDefault(key, 0) - 1);
        }
        
        public static <K, V> V getOrDefault(Map<K, V> map, K key, V defaultValue) {
            return map.getOrDefault(key, defaultValue);
        }
        
        public static <K, V> void putIfAbsent(Map<K, V> map, K key, V value) {
            map.putIfAbsent(key, value);
        }
        
        public static <K, V> Map<K, V> merge(Map<K, V> map1, Map<K, V> map2) {
            Map<K, V> result = new HashMap<>(map1);
            result.putAll(map2);
            return result;
        }
        
        // Map utilities
        public static <K, V> Map<K, V> copy(Map<K, V> map) {
            return new HashMap<>(map);
        }
        
        public static <K, V> List<K> getKeys(Map<K, V> map) {
            return new ArrayList<>(map.keySet());
        }
        
        public static <K, V> List<V> getValues(Map<K, V> map) {
            return new ArrayList<>(map.values());
        }
        
        public static <K, V> List<Map.Entry<K, V>> getEntries(Map<K, V> map) {
            return new ArrayList<>(map.entrySet());
        }
        
        // Map sorting
        public static <K extends Comparable, V> Map<K, V> sortByKey(Map<K, V> map) {
            return new TreeMap<>(map);
        }
        
        public static <K, V extends Comparable> Map<K, V> sortByValue(Map<K, V> map) {
            return map.entrySet()
                     .stream()
                     .sorted(Map.Entry.comparingByValue())
                     .collect(Collectors.toMap(
                         Map.Entry::getKey,
                         Map.Entry::getValue,
                         (e1, e2) -> e1,
                         LinkedHashMap::new
                     ));
        }
        
        public static <K, V extends Comparable> Map<K, V> sortByValueDescending(Map<K, V> map) {
            return map.entrySet()
                     .stream()
                     .sorted(Map.Entry.<K, V>comparingByValue().reversed())
                     .collect(Collectors.toMap(
                         Map.Entry::getKey,
                         Map.Entry::getValue,
                         (e1, e2) -> e1,
                         LinkedHashMap::new
                     ));
        }
        
        // Map algorithms
        public static <K> K mostFrequentKey(Map<K, Integer> frequencyMap) {
            return frequencyMap.entrySet()
                             .stream()
                             .max(Map.Entry.comparingByValue())
                             .map(Map.Entry::getKey)
                             .orElse(null);
        }
        
        public static <K> K leastFrequentKey(Map<K, Integer> frequencyMap) {
            return frequencyMap.entrySet()
                             .stream()
                             .min(Map.Entry.comparingByValue())
                             .map(Map.Entry::getKey)
                             .orElse(null);
        }
        
        public static <K, V> boolean allValuesSatisfy(Map<K, V> map, Predicate<V> predicate) {
            return map.values().stream().allMatch(predicate);
        }
        
        public static <K, V> boolean anyValueSatisfies(Map<K, V> map, Predicate<V> predicate) {
            return map.values().stream().anyMatch(predicate);
        }
        
        public static <K, V> Map<K, V> filterByKey(Map<K, V> map, Predicate<K> predicate) {
            return map.entrySet()
                     .stream()
                     .filter(entry -> predicate.test(entry.getKey()))
                     .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }
        
        public static <K, V> Map<K, V> filterByValue(Map<K, V> map, Predicate<V> predicate) {
            return map.entrySet()
                     .stream()
                     .filter(entry -> predicate.test(entry.getValue()))
                     .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }
        
        // Two-sum utility
        public static int[] twoSum(int[] nums, int target) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < nums.length; i++) {
                int complement = target - nums[i];
                if (map.containsKey(complement)) {
                    return new int[]{map.get(complement), i};
                }
                map.put(nums[i], i);
            }
            return new int[]{-1, -1};
        }
    }

    // ==================== COLLECTIONS ====================
    
    public static class CollectionUtils {
        
        // List operations
        public static <T> List<T> createList(T... items) {
            return new ArrayList<>(Arrays.asList(items));
        }
        
        public static <T> List<T> reverseList(List<T> list) {
            List<T> reversed = new ArrayList<>(list);
            Collections.reverse(reversed);
            return reversed;
        }
        
        public static <T> List<T> sortList(List<T> list) {
            List<T> sorted = new ArrayList<>(list);
            Collections.sort((List) sorted);
            return sorted;
        }
        
        public static <T> List<T> sortList(List<T> list, Comparator<T> comparator) {
            List<T> sorted = new ArrayList<>(list);
            sorted.sort(comparator);
            return sorted;
        }
        
        public static <T> List<T> removeDuplicates(List<T> list) {
            return new ArrayList<>(new LinkedHashSet<>(list));
        }
        
        public static <T> T firstElement(List<T> list) {
            return list.isEmpty() ? null : list.get(0);
        }
        
        public static <T> T lastElement(List<T> list) {
            return list.isEmpty() ? null : list.get(list.size() - 1);
        }
        
        // List algorithms
        public static <T> void swap(List<T> list, int i, int j) {
            Collections.swap(list, i, j);
        }
        
        public static <T> void rotate(List<T> list, int distance) {
            Collections.rotate(list, distance);
        }
        
        public static <T> void shuffle(List<T> list) {
            Collections.shuffle(list);
        }
        
        public static <T> List<T> subList(List<T> list, int from, int to) {
            return new ArrayList<>(list.subList(from, to));
        }
        
        // Collection statistics
        public static <T extends Comparable> T max(Collection<T> collection) {
            return Collections.max(collection);
        }
        
        public static <T extends Comparable> T min(Collection<T> collection) {
            return Collections.min(collection);
        }
        
        public static <T> int frequency(Collection<T> collection, T item) {
            return Collections.frequency(collection, item);
        }
        
        public static <T> boolean areDisjoint(Collection<T> col1, Collection<T> col2) {
            return Collections.disjoint(col1, col2);
        }
        
        // Collection transformation
        public static <T> Set<T> toSet(Collection<T> collection) {
            return new HashSet<>(collection);
        }
        
        public static <T> List<T> toList(Collection<T> collection) {
            return new ArrayList<>(collection);
        }
        
        public static <T> T[] toArray(Collection<T> collection, Class<T> type) {
            return collection.toArray((T[]) java.lang.reflect.Array.newInstance(type, collection.size()));
        }
        
        // Collection checks
        public static <T> boolean isEmpty(Collection<T> collection) {
            return collection == null || collection.isEmpty();
        }
        
        public static <T> boolean isNotEmpty(Collection<T> collection) {
            return !isEmpty(collection);
        }
        
        public static <T> boolean allMatch(Collection<T> collection, Predicate<T> predicate) {
            return collection.stream().allMatch(predicate);
        }
        
        public static <T> boolean anyMatch(Collection<T> collection, Predicate<T> predicate) {
            return collection.stream().anyMatch(predicate);
        }
        
        public static <T> boolean noneMatch(Collection<T> collection, Predicate<T> predicate) {
            return collection.stream().noneMatch(predicate);
        }
    }

    // ==================== COMMON LEETCODE PATTERNS ====================
    
    public static class PatternUtils {
        
        // Two pointers pattern
        public static int twoSumSorted(int[] nums, int target) {
            int left = 0, right = nums.length - 1;
            while (left < right) {
                int sum = nums[left] + nums[right];
                if (sum == target) {
                    return 1; // or return indices
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
            }
            return -1;
        }
        
        // Sliding window pattern
        public static int slidingWindowMaxSum(int[] nums, int k) {
            if (nums.length < k) return -1;
            
            int maxSum = 0;
            for (int i = 0; i < k; i++) {
                maxSum += nums[i];
            }
            
            int windowSum = maxSum;
            for (int i = k; i < nums.length; i++) {
                windowSum += nums[i] - nums[i - k];
                maxSum = Math.max(maxSum, windowSum);
            }
            
            return maxSum;
        }
        
        // Binary search pattern
        public static int binarySearchTemplate(int[] nums, int target) {
            int left = 0, right = nums.length - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (nums[mid] == target) return mid;
                else if (nums[mid] < target) left = mid + 1;
                else right = mid - 1;
            }
            return -1;
        }
        
        // Backtracking template
        public static List<List<Integer>> subsets(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            backtrack(result, new ArrayList<>(), nums, 0);
            return result;
        }
        
        private static void backtrack(List<List<Integer>> result, List<Integer> temp, int[] nums, int start) {
            result.add(new ArrayList<>(temp));
            for (int i = start; i < nums.length; i++) {
                temp.add(nums[i]);
                backtrack(result, temp, nums, i + 1);
                temp.remove(temp.size() - 1);
            }
        }
        
        // DFS template
        public static void dfs(TreeNode node, List<Integer> result) {
            if (node == null) return;
            result.add(node.val);
            dfs(node.left, result);
            dfs(node.right, result);
        }
        
        // BFS template
        public static List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> result = new ArrayList<>();
            if (root == null) return result;
            
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                int levelSize = queue.size();
                List<Integer> level = new ArrayList<>();
                
                for (int i = 0; i < levelSize; i++) {
                    TreeNode node = queue.poll();
                    level.add(node.val);
                    
                    if (node.left != null) queue.offer(node.left);
                    if (node.right != null) queue.offer(node.right);
                }
                
                result.add(level);
            }
            
            return result;
        }
        
        // Dynamic programming - Fibonacci
        public static int fibonacci(int n) {
            if (n <= 1) return n;
            int[] dp = new int[n + 1];
            dp[0] = 0;
            dp[1] = 1;
            for (int i = 2; i <= n; i++) {
                dp[i] = dp[i - 1] + dp[i - 2];
            }
            return dp[n];
        }
        
        // Quick select for Kth element
        public static int quickSelect(int[] nums, int k) {
            return quickSelect(nums, 0, nums.length - 1, k - 1);
        }
        
        private static int quickSelect(int[] nums, int left, int right, int k) {
            if (left == right) return nums[left];
            
            int pivotIndex = partition(nums, left, right);
            
            if (k == pivotIndex) return nums[k];
            else if (k < pivotIndex) return quickSelect(nums, left, pivotIndex - 1, k);
            else return quickSelect(nums, pivotIndex + 1, right, k);
        }
        
        private static int partition(int[] nums, int left, int right) {
            int pivot = nums[right];
            int i = left;
            for (int j = left; j < right; j++) {
                if (nums[j] <= pivot) {
                    swap(nums, i, j);
                    i++;
                }
            }
            swap(nums, i, right);
            return i;
        }
        
        private static void swap(int[] nums, int i, int j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
    }

    // TreeNode class for tree problems
    public static class TreeNode {
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

    // ListNode class for linked list problems
    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    // Test method
    public static void main(String[] args) {
        // Test various utilities
        int[] arr = {1, 2, 3, 4, 5};
        System.out.println("Array sum: " + ArrayUtils.sum(arr));
        System.out.println("Is palindrome: " + StringUtils.isPalindrome("racecar"));
        
        // Test priority queue
        PriorityQueue<Integer> minHeap = PriorityQueueUtils.createMinHeap(arr);
        System.out.println("Min heap peek: " + minHeap.peek());
        
        // Test set operations
        Set<Integer> set1 = SetUtils.createSet(new int[]{1, 2, 3});
        Set<Integer> set2 = SetUtils.createSet(new int[]{2, 3, 4});
        Set<Integer> union = SetUtils.union(set1, set2);
        System.out.println("Union: " + union);
    }
}