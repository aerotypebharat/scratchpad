import java.util.NoSuchElementException;

/**
 * Queue Abstract Data Type (FIFO - First In First Out)
 * Core Operations: enqueue, dequeue, peek, isEmpty, size
 * Time Complexity: O(1) for all operations
 */
public class Queue<T> {
    
    /**
     * Node class representing individual elements in the queue
     */
    private static class Node<T> {
        private T data;         // The actual data stored in this node
        private Node<T> next;   // Reference to the next node in the queue
        
        public Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    private Node<T> front;  // Reference to the front (first) node
    private Node<T> rear;   // Reference to the rear (last) node  
    private int size;       // Current number of elements in the queue
    
    /**
     * Constructor - Initializes an empty queue
     */
    public Queue() {
        front = rear = null;  // Both front and rear are null for empty queue
        size = 0;
    }
    
    // ABSTRACT BEHAVIOR METHODS
    
    /**
     * ENQUEUE Operation: Adds an element to the rear of the queue
     * Time Complexity: O(1)
     * @param item The element to be added to the queue
     */
    public void enqueue(T item) {
        Node<T> newNode = new Node<>(item);  // Create new node
        
        if (isEmpty()) {
            // If queue is empty, new node becomes both front and rear
            front = rear = newNode;
        } else {
            // Add new node after current rear and update rear
            rear.next = newNode;
            rear = newNode;
        }
        size++;
    }
    
    /**
     * DEQUEUE Operation: Removes and returns the front element from the queue
     * Time Complexity: O(1)
     * @return The element that was at the front of the queue
     * @throws NoSuchElementException if queue is empty
     */
    public T dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException();  // Can't dequeue from empty queue
        }
        T item = front.data;  // Store data from front node
        front = front.next;   // Move front to next node
        
        if (front == null) {
            rear = null;  // If queue becomes empty, update rear to null
        }
        size--;
        return item;
    }
    
    /**
     * PEEK Operation: Returns the front element without removing it
     * Time Complexity: O(1)
     * @return The element at the front of the queue
     * @throws NoSuchElementException if queue is empty
     */
    public T peek() {
        if (isEmpty()) {
            throw new NoSuchElementException();  // Can't peek empty queue
        }
        return front.data;
    }
    
    /**
     * ISEMPTY Operation: Checks if the queue contains no elements
     * Time Complexity: O(1)
     * @return true if queue is empty, false otherwise
     */
    public boolean isEmpty() {
        return front == null;  // Queue is empty if front is null
    }
    
    /**
     * SIZE Operation: Returns the number of elements in the queue
     * Time Complexity: O(1)
     * @return The current size of the queue
     */
    public int size() {
        return size;
    }
    
    /**
     * Utility method to display queue contents (for debugging)
     */
    public void display() {
        Node<T> current = front;
        System.out.print("Queue (front to rear): ");
        while (current != null) {
            System.out.print(current.data + " ");
            current = current.next;
        }
        System.out.println();
    }
}