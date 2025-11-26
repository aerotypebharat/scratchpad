import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Hash Table Abstract Data Type (Separate Chaining implementation)
 * Core Operations: put, get, remove, containsKey, size
 * Time Complexity: O(1) average case, O(n) worst case
 */
public class HashTable<K, V> {
    
    /**
     * Entry class representing key-value pairs in the hash table
     */
    private static class Entry<K, V> {
        private K key;    // The key for this entry
        private V value;  // The value associated with the key
        
        public Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }
    
    private List<LinkedList<Entry<K, V>>> buckets;  // Array of linked lists (buckets)
    private int capacity;    // Total number of buckets
    private int size;        // Current number of key-value pairs
    
    /**
     * Constructor - Initializes hash table with default capacity
     */
    public HashTable() {
        this(16);  // Default capacity of 16
    }
    
    /**
     * Constructor - Initializes hash table with specified capacity
     * @param capacity The initial number of buckets
     */
    public HashTable(int capacity) {
        this.capacity = capacity;
        this.size = 0;
        this.buckets = new ArrayList<>(capacity);
        
        // Initialize all buckets with empty linked lists
        for (int i = 0; i < capacity; i++) {
            buckets.add(new LinkedList<>());
        }
    }
    
    /**
     * Hash function: Maps a key to a bucket index
     * @param key The key to hash
     * @return The bucket index for this key
     */
    private int hash(K key) {
        return Math.abs(key.hashCode()) % capacity;  // Ensure positive index within bounds
    }
    
    // ABSTRACT BEHAVIOR METHODS
    
    /**
     * PUT Operation: Inserts a key-value pair into the hash table
     * If key already exists, updates the value
     * Time Complexity: O(1) average case
     * @param key The key to insert
     * @param value The value to associate with the key
     */
    public void put(K key, V value) {
        int index = hash(key);                    // Get bucket index
        LinkedList<Entry<K, V>> bucket = buckets.get(index);  // Get the bucket
        
        // Check if key already exists in bucket
        for (Entry<K, V> entry : bucket) {
            if (entry.key.equals(key)) {
                entry.value = value;  // Update existing key
                return;
            }
        }
        
        // Key doesn't exist, add new entry
        bucket.add(new Entry<>(key, value));
        size++;
    }
    
    /**
     * GET Operation: Retrieves the value associated with a key
     * Time Complexity: O(1) average case
     * @param key The key to search for
     * @return The value associated with the key, or null if not found
     */
    public V get(K key) {
        int index = hash(key);
        LinkedList<Entry<K, V>> bucket = buckets.get(index);
        
        // Search for key in the bucket
        for (Entry<K, V> entry : bucket) {
            if (entry.key.equals(key)) {
                return entry.value;  // Key found, return value
            }
        }
        return null;  // Key not found
    }
    
    /**
     * CONTAINS KEY Operation: Checks if a key exists in the hash table
     * Time Complexity: O(1) average case
     * @param key The key to check
     * @return true if key exists, false otherwise
     */
    public boolean containsKey(K key) {
        return get(key) != null;  // Key exists if get returns non-null
    }
    
    /**
     * SIZE Operation: Returns the number of key-value pairs in the hash table
     * Time Complexity: O(1)
     * @return The current size
     */
    public int size() {
        return size;
    }
    
    /**
     * ISEMPTY Operation: Checks if the hash table contains no key-value pairs
     * Time Complexity: O(1)
     * @return true if hash table is empty, false otherwise
     */
    public boolean isEmpty() {
        return size == 0;
    }
    
    /**
     * Utility method to display hash table contents (for debugging)
     */
    public void display() {
        for (int i = 0; i < capacity; i++) {
            System.out.print("Bucket " + i + ": ");
            LinkedList<Entry<K, V>> bucket = buckets.get(i);
            for (Entry<K, V> entry : bucket) {
                System.out.print("[" + entry.key + "=" + entry.value + "] ");
            }
            System.out.println();
        }
    }
}