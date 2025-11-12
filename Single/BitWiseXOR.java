package com.god.file.individual;

public class BitWiseXOR {

    // Pattern 18: BITWISE XOR
    // Use Case: Finding unique elements, bit manipulation


    // 18.1: Single Number

    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    // 18.2: Missing Number

    public int missingNumber(int[] nums) {
        int n = nums.length;
        int result = n; // Initialize with n since it's missing from indices

        for (int i = 0; i < n; i++) {
            result ^= i ^ nums[i];
        }
        return result;
    }


    // 18.3: Complement of Base 10 Integer

    public int bitwiseComplement(int n) {
        if (n == 0) return 1;

        int bitCount = 0;
        int num = n;
        while (num > 0) {
            bitCount++;
            num >>= 1;
        }

        int allOnes = (1 << bitCount) - 1;
        return n ^ allOnes;
    }


}
