import java.util.NoSuchElementException;

/**
 * Singly Linked List Abstract Data Type
 * Core Operations: addFirst, addLast, removeFirst, removeLast, contains, size
 * Time Complexity: O(1) for head operations, O(n) for tail operations
 */
public class LinkedList<T> {
    
    /**
     * Node class representing individual elements in the linked list
     */
    private static class Node<T> {
        private T data;         // The actual data stored in this node
        private Node<T> next;   // Reference to the next node in the list
        
        public Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    private Node<T> head;  // Reference to the first node in the list
    private int size;      // Current number of elements in the list
    
    /**
     * Constructor - Initializes an empty linked list
     */
    public LinkedList() {
        head = null;
        size = 0;
    }
    
    // ABSTRACT BEHAVIOR METHODS
    
    /**
     * ADD FIRST Operation: Inserts an element at the beginning of the list
     * Time Complexity: O(1)
     * @param data The element to be added
     */
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);  // Create new node
        newNode.next = head;  // New node points to current head
        head = newNode;       // Update head to be the new node
        size++;
    }
    
    /**
     * ADD LAST Operation: Inserts an element at the end of the list
     * Time Complexity: O(n) - must traverse to the end
     * @param data The element to be added
     */
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            // If list is empty, new node becomes head
            head = newNode;
        } else {
            // Traverse to the last node and add new node after it
            Node<T> current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
        size++;
    }
    
    /**
     * REMOVE FIRST Operation: Removes and returns the first element
     * Time Complexity: O(1)
     * @return The element that was at the beginning of the list
     * @throws NoSuchElementException if list is empty
     */
    public T removeFirst() {
        if (head == null) {
            throw new NoSuchElementException();  // Can't remove from empty list
        }
        T data = head.data;  // Store data from head node
        head = head.next;    // Move head to next node
        size--;
        return data;
    }
    
    /**
     * REMOVE LAST Operation: Removes and returns the last element
     * Time Complexity: O(n) - must traverse to the second last node
     * @return The element that was at the end of the list
     * @throws NoSuchElementException if list is empty
     */
    public T removeLast() {
        if (head == null) {
            throw new NoSuchElementException();  // Can't remove from empty list
        }
        
        if (head.next == null) {
            // Only one element in list
            return removeFirst();
        }
        
        // Traverse to the second last node
        Node<T> current = head;
        while (current.next.next != null) {
            current = current.next;
        }
        
        T data = current.next.data;  // Store data from last node
        current.next = null;         // Remove reference to last node
        size--;
        return data;
    }
    
    /**
     * CONTAINS Operation: Checks if the list contains specified element
     * Time Complexity: O(n) - worst case must check all elements
     * @param data The element to search for
     * @return true if element is found, false otherwise
     */
    public boolean contains(T data) {
        Node<T> current = head;
        while (current != null) {
            if (current.data.equals(data)) {
                return true;  // Element found
            }
            current = current.next;
        }
        return false;  // Element not found
    }
    
    /**
     * SIZE Operation: Returns the number of elements in the list
     * Time Complexity: O(1)
     * @return The current size of the list
     */
    public int size() {
        return size;
    }
    
    /**
     * ISEMPTY Operation: Checks if the list contains no elements
     * Time Complexity: O(1)
     * @return true if list is empty, false otherwise
     */
    public boolean isEmpty() {
        return head == null;  // List is empty if head is null
    }
    
    /**
     * Utility method to display list contents (for debugging)
     */
    public void display() {
        Node<T> current = head;
        System.out.print("LinkedList: ");
        while (current != null) {
            System.out.print(current.data + " -> ");
            current = current.next;
        }
        System.out.println("null");
    }
}