import java.util.EmptyStackException;

/**
 * Stack Abstract Data Type (LIFO - Last In First Out)
 * Core Operations: push, pop, peek, isEmpty, size
 * Time Complexity: O(1) for all operations
 */
public class Stack<T> {
    
    /**
     * Node class representing individual elements in the stack
     * Each node contains data and reference to the next node
     */
    private static class Node<T> {
        private T data;         // The actual data stored in this node
        private Node<T> next;   // Reference to the next node in the stack
        
        public Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    private Node<T> top;  // Reference to the top node of the stack
    private int size;     // Current number of elements in the stack
    
    /**
     * Constructor - Initializes an empty stack
     */
    public Stack() {
        top = null;  // Stack is empty, so top points to null
        size = 0;    // No elements initially
    }
    
    // ABSTRACT BEHAVIOR METHODS
    
    /**
     * PUSH Operation: Adds an element to the top of the stack
     * Time Complexity: O(1)
     * @param item The element to be added to the stack
     */
    public void push(T item) {
        Node<T> newNode = new Node<>(item);  // Create new node with given data
        newNode.next = top;  // New node points to current top
        top = newNode;       // Update top to be the new node
        size++;              // Increment element count
    }
    
    /**
     * POP Operation: Removes and returns the top element from the stack
     * Time Complexity: O(1)
     * @return The element that was at the top of the stack
     * @throws EmptyStackException if stack is empty
     */
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();  // Can't pop from empty stack
        }
        T item = top.data;  // Store data from top node
        top = top.next;     // Move top to next node
        size--;             // Decrement element count
        return item;        // Return the stored data
    }
    
    /**
     * PEEK Operation: Returns the top element without removing it
     * Time Complexity: O(1)
     * @return The element at the top of the stack
     * @throws EmptyStackException if stack is empty
     */
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();  // Can't peek empty stack
        }
        return top.data;  // Return data from top node
    }
    
    /**
     * ISEMPTY Operation: Checks if the stack contains no elements
     * Time Complexity: O(1)
     * @return true if stack is empty, false otherwise
     */
    public boolean isEmpty() {
        return top == null;  // Stack is empty if top is null
    }
    
    /**
     * SIZE Operation: Returns the number of elements in the stack
     * Time Complexity: O(1)
     * @return The current size of the stack
     */
    public int size() {
        return size;
    }
    
    /**
     * Utility method to display stack contents (for debugging)
     */
    public void display() {
        Node<T> current = top;
        System.out.print("Stack (top to bottom): ");
        while (current != null) {
            System.out.print(current.data + " ");
            current = current.next;
        }
        System.out.println();
    }
}