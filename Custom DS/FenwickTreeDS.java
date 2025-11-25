class FenwickTree {
    private int[] tree;
    
    public FenwickTree(int size) {
        tree = new int[size + 1]; // 1-indexed
    }
    
    // Update index i by adding delta
    public void update(int i, int delta) {
        i++; // Convert to 1-indexed
        while (i < tree.length) {
            tree[i] += delta;
            i += i & -i; // Move to next responsible index
        }
    }
    
    // Query prefix sum [0, i]
    public int query(int i) {
        i++; // Convert to 1-indexed
        int sum = 0;
        while (i > 0) {
            sum += tree[i];
            i -= i & -i; // Move to parent
        }
        return sum;
    }
    
    // Query range sum [l, r]
    public int rangeQuery(int l, int r) {
        return query(r) - (l > 0 ? query(l - 1) : 0);
    }
}