/**
 * Binary Search Tree Abstract Data Type
 * Core Operations: insert, search, delete, traversals (in-order, pre-order, post-order)
 * Time Complexity: O(h) where h is height of tree (O(log n) for balanced tree)
 */
public class BinarySearchTree<T extends Comparable<T>> {
    
    /**
     * Node class representing individual elements in the BST
     */
    private static class Node<T> {
        private T data;          // The actual data stored in this node
        private Node<T> left;    // Reference to left child
        private Node<T> right;   // Reference to right child
        
        public Node(T data) {
            this.data = data;
            this.left = this.right = null;
        }
    }
    
    private Node<T> root;  // Reference to the root node of the BST
    
    /**
     * Constructor - Initializes an empty binary search tree
     */
    public BinarySearchTree() {
        root = null;
    }
    
    // ABSTRACT BEHAVIOR METHODS
    
    /**
     * INSERT Operation: Inserts a new element into the BST
     * Time Complexity: O(h) where h is height of tree
     * @param data The element to be inserted
     */
    public void insert(T data) {
        root = insertRec(root, data);  // Call recursive helper method
    }
    
    /**
     * Recursive helper method for insertion
     * @param node Current node in recursion
     * @param data Data to be inserted
     * @return The updated node after insertion
     */
    private Node<T> insertRec(Node<T> node, T data) {
        if (node == null) {
            return new Node<>(data);  // Create new node when reaching null position
        }
        
        // Compare data with current node to decide left or right subtree
        if (data.compareTo(node.data) < 0) {
            node.left = insertRec(node.left, data);   // Insert in left subtree
        } else if (data.compareTo(node.data) > 0) {
            node.right = insertRec(node.right, data); // Insert in right subtree
        }
        // If data equals node.data, do nothing (no duplicates)
        
        return node;
    }
    
    /**
     * SEARCH Operation: Searches for an element in the BST
     * Time Complexity: O(h) where h is height of tree
     * @param data The element to search for
     * @return true if element is found, false otherwise
     */
    public boolean search(T data) {
        return searchRec(root, data);  // Call recursive helper method
    }
    
    /**
     * Recursive helper method for search
     * @param node Current node in recursion
     * @param data Data to search for
     * @return true if data is found, false otherwise
     */
    private boolean searchRec(Node<T> node, T data) {
        if (node == null) {
            return false;  // Reached leaf node, data not found
        }
        
        if (data.compareTo(node.data) == 0) {
            return true;   // Data found at current node
        }
        
        // Recursively search in appropriate subtree
        return data.compareTo(node.data) < 0 ? 
            searchRec(node.left, data) : searchRec(node.right, data);
    }
    
    /**
     * IN-ORDER TRAVERSAL: Visits nodes in sorted order (left, root, right)
     * Time Complexity: O(n) where n is number of nodes
     */
    public void inOrder() {
        System.out.print("In-order (sorted): ");
        inOrderRec(root);
        System.out.println();
    }
    
    private void inOrderRec(Node<T> node) {
        if (node != null) {
            inOrderRec(node.left);      // Visit left subtree
            System.out.print(node.data + " ");  // Visit current node
            inOrderRec(node.right);     // Visit right subtree
        }
    }
    
    /**
     * PRE-ORDER TRAVERSAL: Visits nodes in pre-order (root, left, right)
     * Useful for copying the tree
     */
    public void preOrder() {
        System.out.print("Pre-order: ");
        preOrderRec(root);
        System.out.println();
    }
    
    private void preOrderRec(Node<T> node) {
        if (node != null) {
            System.out.print(node.data + " ");  // Visit current node first
            preOrderRec(node.left);     // Visit left subtree
            preOrderRec(node.right);    // Visit right subtree
        }
    }
    
    /**
     * POST-ORDER TRAVERSAL: Visits nodes in post-order (left, right, root)
     * Useful for deleting the tree
     */
    public void postOrder() {
        System.out.print("Post-order: ");
        postOrderRec(root);
        System.out.println();
    }
    
    private void postOrderRec(Node<T> node) {
        if (node != null) {
            postOrderRec(node.left);    // Visit left subtree first
            postOrderRec(node.right);   // Visit right subtree
            System.out.print(node.data + " ");  // Visit current node last
        }
    }
}