class SegmentTree {
    private int[] tree;
    private int n;
    
    public SegmentTree(int[] nums) {
        n = nums.length;
        tree = new int[4 * n];
        buildTree(nums, 0, 0, n - 1);
    }
    
    private void buildTree(int[] nums, int node, int start, int end) {
        if (start == end) {
            tree[node] = nums[start];
        } else {
            int mid = start + (end - start) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            
            buildTree(nums, leftChild, start, mid);
            buildTree(nums, rightChild, mid + 1, end);
            
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }
    
    public void update(int index, int value, int node, int start, int end) {
        if (start == end) {
            tree[node] = value;
        } else {
            int mid = start + (end - start) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            
            if (index <= mid) {
                update(index, value, leftChild, start, mid);
            } else {
                update(index, value, rightChild, mid + 1, end);
            }
            
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }
    
    public int query(int l, int r, int node, int start, int end) {
        if (l > end || r < start) return 0; // No overlap
        if (l <= start && end <= r) return tree[node]; // Complete overlap
        
        int mid = start + (end - start) / 2;
        int leftChild = 2 * node + 1;
        int rightChild = 2 * node + 2;
        
        int leftSum = query(l, r, leftChild, start, mid);
        int rightSum = query(l, r, rightChild, mid + 1, end);
        
        return leftSum + rightSum;
    }
    
    // Helper methods for easier interface
    public void update(int index, int value) {
        update(index, value, 0, 0, n - 1);
    }
    
    public int query(int l, int r) {
        return query(l, r, 0, 0, n - 1);
    }
}