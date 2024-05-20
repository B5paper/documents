#include <linux/init.h>
#include <linux/module.h>
#include <linux/list.h>

struct ListNode {
    struct list_head head;
    int my_data;
};

struct list_head lst_head;

int init_mod(void)
{
    pr_info("in init_mod()...\n");
    INIT_LIST_HEAD(&lst_head);
    struct ListNode node_1 = {
        .my_data = 1
    };
    struct ListNode node_2 = {
        .my_data = 2
    };
    struct ListNode node_3 = {
        .my_data = 3
    };
    struct ListNode node_4 = {
        .my_data = 4
    };
    list_add(&node_1.head, &lst_head);
    list_add(&node_2.head, &lst_head);
    list_add(&node_3.head, &lst_head);
    list_add_tail(&node_4.head, &lst_head);
    struct ListNode *cur;
    int len_count = 0;
    pr_info("traverse list:\n");
    list_for_each_entry(cur, &lst_head, head) {
        pr_info("%d\n", cur->my_data);
        ++len_count;
    }
    pr_info("list len: %d\n", len_count);
    return 0;
}

void exit_mod(void)
{
    pr_info("in exit_mod()...\n");
}

module_init(init_mod);
module_exit(exit_mod);
MODULE_LICENSE("GPL");