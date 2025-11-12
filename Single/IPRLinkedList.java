package com.god.file.individual;

public class IPRLinkedList {

    //
    // Pattern 6: IN-PLACE REVERSAL OF LINKEDLIST
    // Use Case: Reverse linked lists or portions of them
    //


    // 6.1: Reverse Linked List

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;

        while (current != null) {
            ListNode nextTemp = current.next;
            current.next = prev;
            prev = current;
            current = nextTemp;
        }
        return prev;
    }


    // 6.2: Reverse Linked List II
    // Reverse from position left to right

    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null || left == right) return head;

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;

        // Move to the node before reversal start
        for (int i = 1; i < left; i++) {
            prev = prev.next;
        }

        ListNode start = prev.next;
        ListNode then = start.next;

        // Reverse the sublist
        for (int i = 0; i < right - left; i++) {
            start.next = then.next;
            then.next = prev.next;
            prev.next = then;
            then = start.next;
        }

        return dummy.next;
    }


    // 6.3: Reverse Nodes in k-Group

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode curr = dummy, prev = dummy, next = dummy;

        int count = 0;
        while (curr.next != null) {
            curr = curr.next;
            count++;
        }

        while (count >= k) {
            curr = prev.next;
            next = curr.next;

            for (int i = 1; i < k; i++) {
                curr.next = next.next;
                next.next = prev.next;
                prev.next = next;
                next = curr.next;
            }
            prev = curr;
            count -= k;
        }
        return dummy.next;
    }

}
