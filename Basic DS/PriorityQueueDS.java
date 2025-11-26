import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Priority Queue Abstract Data Type (Min-Heap implementation)
 * 
 * A priority queue is an abstract data type where each element has a "priority" associated with it.
 * In a min-priority queue, elements with higher priority (lower value) are served before elements 
 * with lower priority (higher value).
 * 
 * Core Operations: insert, extractMin, peek, isEmpty, size
 * Time Complexity: O(log n) for insert and extractMin, O(1) for peek and size
 * 
 * Implementation Note: Using a binary min-heap where the root is always the smallest element.
 * The heap is stored as a complete binary tree in an array list for efficient operations.
 */
public class PriorityQueue<T extends Comparable<T>> {
    
    /**
     * Internal storage for the heap elements
     * The heap is represented as a complete binary tree in array form:
     * - For node at index i:
     *   - Parent index: (i-1)/2
     *   - Left child index: 2*i + 1
     *   - Right child index: 2*i + 2
     */
    private List<T> heap;
    
    /**
     * Constructor - Initializes an empty priority queue
     * Uses dynamic array list for automatic resizing
     */
    public PriorityQueue() {
        heap = new ArrayList<>();
    }
    
    /**
     * Constructor - Initializes priority queue with specified initial capacity
     * @param capacity The initial capacity of the internal array list
     */
    public PriorityQueue(int capacity) {
        heap = new ArrayList<>(capacity);
    }
    
    // PRIVATE HELPER METHODS FOR HEAP OPERATIONS
    
    /**
     * Calculates the parent index of a given node index
     * @param index The child node index
     * @return The parent node index
     */
    private int parent(int index) {
        return (index - 1) / 2;  // Integer division automatically floors the result
    }
    
    /**
     * Calculates the left child index of a given node index
     * @param index The parent node index
     * @return The left child node index
     */
    private int leftChild(int index) {
        return 2 * index + 1;
    }
    
    /**
     * Calculates the right child index of a given node index
     * @param index The parent node index
     * @return The right child node index
     */
    private int rightChild(int index) {
        return 2 * index + 2;
    }
    
    /**
     * Swaps two elements in the heap
     * @param i Index of first element
     * @param j Index of second element
     */
    private void swap(int i, int j) {
        T temp = heap.get(i);
        heap.set(i, heap.get(j));
        heap.set(j, temp);
    }
    
    /**
     * Restores heap property by moving a node up the tree
     * Used after insertion to maintain min-heap property
     * @param index The index of node to move up
     */
    private void heapifyUp(int index) {
        // Continue until we reach root or heap property is satisfied
        while (index > 0 && heap.get(index).compareTo(heap.get(parent(index))) < 0) {
            // Current node is smaller than parent, so swap them
            swap(index, parent(index));
            // Move up to parent position and continue checking
            index = parent(index);
        }
    }
    
    /**
     * Restores heap property by moving a node down the tree
     * Used after extraction to maintain min-heap property
     * @param index The index of node to move down
     */
    private void heapifyDown(int index) {
        int minIndex = index;      // Track the index of smallest element
        int left = leftChild(index);   // Left child index
        int right = rightChild(index); // Right child index
        
        // Check if left child exists and is smaller than current min
        if (left < heap.size() && heap.get(left).compareTo(heap.get(minIndex)) < 0) {
            minIndex = left;
        }
        
        // Check if right child exists and is smaller than current min
        if (right < heap.size() && heap.get(right).compareTo(heap.get(minIndex)) < 0) {
            minIndex = right;
        }
        
        // If current node is not the smallest, swap and continue heapifying down
        if (index != minIndex) {
            swap(index, minIndex);
            heapifyDown(minIndex);  // Recursively heapify the affected subtree
        }
    }
    
    // ABSTRACT BEHAVIOR METHODS
    
    /**
     * INSERT Operation: Adds an element to the priority queue while maintaining heap property
     * Also known as "offer" or "add" in some implementations
     * Time Complexity: O(log n) - may need to traverse from leaf to root
     * @param value The element to be inserted into the priority queue
     */
    public void insert(T value) {
        // Add new element at the end (maintains complete binary tree property)
        heap.add(value);
        // Restore heap property by moving new element up as needed
        heapifyUp(heap.size() - 1);
    }
    
    /**
     * EXTRACT-MIN Operation: Removes and returns the element with highest priority (minimum value)
     * Also known as "poll" in some implementations
     * Time Complexity: O(log n) - may need to traverse from root to leaf
     * @return The element with the highest priority (minimum value)
     * @throws NoSuchElementException if priority queue is empty
     */
    public T extractMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("Cannot extract from empty priority queue");
        }
        
        // Store the minimum element (always at root in min-heap)
        T min = heap.get(0);
        // Get the last element to move to root
        T last = heap.remove(heap.size() - 1);
        
        if (!isEmpty()) {
            // Move last element to root
            heap.set(0, last);
            // Restore heap property by moving new root down as needed
            heapifyDown(0);
        }
        
        return min;
    }
    
    /**
     * PEEK Operation: Returns the element with highest priority without removing it
     * Also known as "element" or "getMin" in some implementations
     * Time Complexity: O(1) - simply returns root element
     * @return The element with the highest priority (minimum value)
     * @throws NoSuchElementException if priority queue is empty
     */
    public T peek() {
        if (isEmpty()) {
            throw new NoSuchElementException("Cannot peek empty priority queue");
        }
        return heap.get(0);  // Root element is always the minimum in min-heap
    }
    
    /**
     * SIZE Operation: Returns the number of elements in the priority queue
     * Time Complexity: O(1)
     * @return The current number of elements
     */
    public int size() {
        return heap.size();
    }
    
    /**
     * ISEMPTY Operation: Checks if the priority queue contains no elements
     * Time Complexity: O(1)
     * @return true if priority queue is empty, false otherwise
     */
    public boolean isEmpty() {
        return heap.isEmpty();
    }
    
    /**
     * CLEAR Operation: Removes all elements from the priority queue
     * Time Complexity: O(1) - simply clears the internal list
     */
    public void clear() {
        heap.clear();
    }
    
    /**
     * CONTAINS Operation: Checks if the priority queue contains specified element
     * Time Complexity: O(n) - must search through entire heap in worst case
     * @param value The element to search for
     * @return true if element is found, false otherwise
     */
    public boolean contains(T value) {
        return heap.contains(value);  // Linear search through the list
    }
    
    /**
     * Utility method to display priority queue contents (for debugging)
     * Shows the heap array representation
     */
    public void display() {
        System.out.println("Priority Queue (min-heap): " + heap);
    }
    
    /**
     * Utility method to check if heap property is maintained (for testing)
     * @return true if heap property is satisfied, false otherwise
     */
    public boolean isMinHeap() {
        // Check heap property for all non-leaf nodes
        for (int i = 0; i <= parent(heap.size() - 1); i++) {
            int left = leftChild(i);
            int right = rightChild(i);
            
            // Check left child
            if (left < heap.size() && heap.get(i).compareTo(heap.get(left)) > 0) {
                return false;
            }
            
            // Check right child
            if (right < heap.size() && heap.get(i).compareTo(heap.get(right)) > 0) {
                return false;
            }
        }
        return true;
    }
}