## 链表

### 排序链表

* 题目

    给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

    

    示例 1：

    输入：head = [4,2,1,3]
    输出：[1,2,3,4]

    示例 2：

    输入：head = [-1,5,3,4,0]
    输出：[-1,0,3,4,5]

    示例 3：

    输入：head = []
    输出：[]

    

    提示：

        链表中节点的数目在范围 [0, 5 * 104] 内
        -105 <= Node.val <= 105

    

    进阶：你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

* 我的解法

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* merge(ListNode *list1, ListNode *list2) {
            ListNode *p1 = list1, *p2 = list2;
            ListNode dummy;
            ListNode *dummy_head = &dummy;
            ListNode *p = dummy_head;
            while (p1 && p2) {
                if (p1->val < p2->val) {
                    p->next = p1;
                    p1 = p1->next;
                } else {
                    p->next = p2;
                    p2 = p2->next;
                }
                p = p->next;
            }
            if (p1) {
                p->next = p1;
            }
            if (p2) {
                p->next = p2;
            }
            p = dummy_head->next;
            return p;
        }

        ListNode* get_middle_node(ListNode *head) {
            ListNode *p1 = head, *p2 = head, *pre;
            while (p2 && p2->next) {
                pre = p1;
                p1 = p1->next;
                p2 = p2->next->next;
            }
            pre->next = nullptr;
            return p1;
        }

        ListNode* recur_sort(ListNode *head) {
            if (head == nullptr || head->next == nullptr) {
                return head;
            }
            ListNode *mid_node = get_middle_node(head);
            ListNode *head_1 = recur_sort(head);
            ListNode *head_2 = recur_sort(mid_node);
            ListNode *new_head = merge(head_1, head_2);
            return new_head;
        }

        ListNode* sortList(ListNode* head) {
            return recur_sort(head);
        }
    };
    ```

* ai 探讨我的思路和代码：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* merge(ListNode *list1, ListNode *list2) {
            ListNode *p1 = list1, *p2 = list2;
            ListNode *dummy_head = new ListNode;
            ListNode *p = dummy_head;
            while (p1 && p2) {
                if (p1->val < p2->val) {
                    p->next = p1;
                    p1 = p1->next;
                } else {
                    p->next = p2;
                    p2 = p2->next;
                }
                p = p->next;
            }
            if (p1) {
                p->next = p1;
            }
            if (p2) {
                p->next = p2;
            }
            p = dummy_head->next;
            delete dummy_head;
            return p;
        }

        ListNode* get_middle_node(ListNode *head) {
            ListNode *p1 = head, *p2 = head;
            while (p2 && p2->next) {
                p1 = p1->next;
                p2 = p2->next->next;
            }
            return p1;
        }

        ListNode* recur_sort(ListNode *head) {
            if (head == nullptr || head->next == nullptr) {
                return head;
            }
            ListNode *mid_node = get_middle_node(head);
            ListNode *head_1 = recur_sort(head);
            ListNode *head_2 = recur_sort(mid_node);
            ListNode *new_head = merge(head_1, head_2);
            return new_head;
        }

        ListNode* sortList(ListNode* head) {
            return recur_sort(head);
        }
    };
    ```

    你的思路方向非常正确，选择 **归并排序（Merge Sort）** 来处理链表排序是达到 $O(n \log n)$ 时间复杂度的最优解。

    不过，你目前的 C++ 代码实现中存在一个**致命的逻辑错误**，这会导致程序进入死循环或产生错误的递归结果。

    ---

    **代码缺陷分析**

    1. 链表未断开（死循环根源）

        在归并排序中，你需要将链表从中间一分为二。你的 `get_middle_node` 只是找到了中点，但**没有把前半部分和后半部分切断**。

        * **后果**：当你递归调用 `recur_sort(head)` 时，它仍然能通过 `head` 遍历到整个链表，导致递归永远无法收敛。

    2. 递归参数错误

        在 `recur_sort` 中，你递归调用了 `head` 和 `mid_node`。如果不断开链表，这两个参数对应的链表长度根本没有减少。

    3. 内存泄露风险

        你在 `merge` 函数中 `new` 了一个 `dummy_head` 节点，虽然最后 `delete` 了，但在高频递归中频繁申请/释放内存会影响性能。其实可以使用栈上的局部变量：`ListNode dummy(0);`。

    ---

    **修正后的代码**

    要实现正确的归并排序，你需要用 **快慢指针** 找到中点的前驱节点，然后将 `next` 置为 `nullptr`。

    ```cpp
    class Solution {
    public:
        ListNode* sortList(ListNode* head) {
            // 递归终止条件
            if (!head || !head->next) return head;

            // 1. 使用快慢指针找到中点并断开链表
            // slow 最终指向中点，pre 指向中点的前一个节点
            ListNode *slow = head, *fast = head, *pre = nullptr;
            while (fast && fast->next) {
                pre = slow;
                slow = slow->next;
                fast = fast->next->next;
            }
            pre->next = nullptr; // 【关键】断开链表，分为 [head, pre] 和 [slow, end]

            // 2. 递归排序左右两部分
            ListNode* left = sortList(head);
            ListNode* right = sortList(slow);

            // 3. 合并
            return merge(left, right);
        }

        ListNode* merge(ListNode* l1, ListNode* l2) {
            ListNode dummy(0); // 栈上分配，无需 delete
            ListNode* p = &dummy;
            while (l1 && l2) {
                if (l1->val < l2->val) {
                    p->next = l1;
                    l1 = l1->next;
                } else {
                    p->next = l2;
                    l2 = l2->next;
                }
                p = p->next;
            }
            p->next = l1 ? l1 : l2;
            return dummy.next;
        }
    };

    ```

    ---

    **进阶：如何达到常数级空间复杂度？**

    你提到的进阶要求是 $O(1)$ 空间复杂度。

    * **递归版**：由于递归栈的存在，空间复杂度其实是 $O(\log n)$。
    * **迭代版（Bottom-up）**：要达到真正的 $O(1)$，需要使用 **自底向上** 的归并排序。通过循环，每次合并长度为 1, 2, 4, 8... 的子链表。

    | 维度 | 递归法 (Top-down) | 迭代法 (Bottom-up) |
    | --- | --- | --- |
    | **时间复杂度** | $O(n \log n)$ | $O(n \log n)$ |
    | **空间复杂度** | $O(\log n)$ (栈空间) | $O(1)$ |
    | **实现难度** | 简单直观 | 较复杂 (需手动控制步长) |

    **你想深入了解如何用“自底向上”的迭代法来实现这个 $O(1)$ 空间复杂度的版本吗？**

* ai 给出的 bottom-up 解法

    既然你决定挑战这道题的“终极形态”，那我们就把递归彻底抛弃。

    要把空间复杂度降到 $O(1)$，我们需要**自底向上（Bottom-up）**地进行归并。想象一下：第一轮把相邻的 $1$ 个节点合并成 $2$ 个；第二轮把相邻的 $2$ 个合并成 $4$ 个……直到合并后的长度大于或等于链表总长。

    这个过程就像是在**织布**：每次剪下一段，理顺，缝好，再剪下一段。

    **自底向上归并排序 (Bottom-up Merge Sort)**

    这种方法的难点在于：**你需要手动维护四个关键点**：

    1. **已排序部分的末尾**（用于连接下一段）。
    2. **当前待处理部分的起点**。
    3. **第一段子链表**。
    4. **第二段子链表**。

    ```cpp
    class Solution {
    public:
        ListNode* sortList(ListNode* head) {
            if (!head || !head->next) return head;

            // 1. 先统计链表总长度
            int length = 0;
            ListNode* curr = head;
            while (curr) {
                length++;
                curr = curr->next;
            }

            // 2. 引入 dummyHead 简化头部的处理
            ListNode dummy(0);
            dummy.next = head;

            // 3. 循环步长：1, 2, 4, 8...
            for (int step = 1; step < length; step <<= 1) {
                ListNode* prev = &dummy;     // 已合并部分的末尾
                curr = dummy.next;           // 剩余未合并部分的起点

                while (curr) {
                    // --- 找到第一段 ---
                    ListNode* head1 = curr;
                    for (int i = 1; i < step && curr->next; ++i) curr = curr->next;
                    
                    // --- 找到第二段 ---
                    ListNode* head2 = curr->next;
                    curr->next = nullptr;    // 切断第一段
                    curr = head2;
                    for (int i = 1; i < step && curr && curr->next; ++i) curr = curr->next;

                    // --- 切断第二段并保留后续的起点 ---
                    ListNode* next_start = nullptr;
                    if (curr) {
                        next_start = curr->next;
                        curr->next = nullptr; // 切断第二段
                    }

                    // --- 合并这两段并将结果挂到 prev 后面 ---
                    prev->next = merge(head1, head2);
                    
                    // --- 更新 prev 到已合并部分的末尾 ---
                    while (prev->next) prev = prev->next;
                    
                    // --- 移动到下一组待处理位置 ---
                    curr = next_start;
                }
            }
            return dummy.next;
        }

        // 标准的双链表合并函数
        ListNode* merge(ListNode* l1, ListNode* l2) {
            ListNode dummy(0);
            ListNode* p = &dummy;
            while (l1 && l2) {
                if (l1->val <= l2->val) {
                    p->next = l1; l1 = l1->next;
                } else {
                    p->next = l2; l2 = l2->next;
                }
                p = p->next;
            }
            p->next = l1 ? l1 : l2;
            return dummy.next;
        }
    };

    ```

    ---

    **深度评价：为什么这个解法是“完全体”？**

    1. **真正的 $O(1)$ 空间**：
    没有递归调用，所有的变量（`step`, `curr`, `prev` 等）都在栈上分配，且数量固定。无论链表多长，它占用的额外内存都不会增加。
    2. **工程上的稳健性**：
    你可以注意到我在 `merge` 中使用的是栈对象 `ListNode dummy(0)`，这完美规避了官方解法中的内存泄露问题。
    3. **手动切断的艺术**：
    这个算法的核心在于 `curr->next = nullptr`。这就像是在处理一捆乱绳，每次量出两截，剪断，理顺缝好，再继续处理剩下的。

    **性能对比总结**

    | 维度 | 你的初版 | 官方递归版 | 这里的迭代版 |
    | --- | --- | --- | --- |
    | **时间复杂度** | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ |
    | **空间复杂度** | $O(\log n)$ | $O(\log n)$ | **$O(1)$ (最优)** |
    | **核心风险** | 死循环 | 栈溢出/内存泄露 | **逻辑复杂易写错** |

    **一个小建议：**
    在面试中，如果面试官没要求 $O(1)$ 空间，**写递归版**（即官方那种），因为不容易出错且代码整洁。只有当对方追问“能不能优化空间”时，再把这个“迭代织布法”祭出来。

    你觉得这个迭代逻辑里，最绕的部分是不是“如何切断链表再接回去”那一段？　

* 官方答案: 方法一：自顶向下归并排序

    ```cpp
    class Solution {
    public:
        ListNode* sortList(ListNode* head) {
            return sortList(head, nullptr);
        }

        ListNode* sortList(ListNode* head, ListNode* tail) {
            if (head == nullptr) {
                return head;
            }
            if (head->next == tail) {
                head->next = nullptr;
                return head;
            }
            ListNode* slow = head, *fast = head;
            while (fast != tail) {
                slow = slow->next;
                fast = fast->next;
                if (fast != tail) {
                    fast = fast->next;
                }
            }
            ListNode* mid = slow;
            return merge(sortList(head, mid), sortList(mid, tail));
        }

        ListNode* merge(ListNode* head1, ListNode* head2) {
            ListNode* dummyHead = new ListNode(0);
            ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
            while (temp1 != nullptr && temp2 != nullptr) {
                if (temp1->val <= temp2->val) {
                    temp->next = temp1;
                    temp1 = temp1->next;
                } else {
                    temp->next = temp2;
                    temp2 = temp2->next;
                }
                temp = temp->next;
            }
            if (temp1 != nullptr) {
                temp->next = temp1;
            } else if (temp2 != nullptr) {
                temp->next = temp2;
            }
            return dummyHead->next;
        }
    };
    ```

    复杂度分析

        时间复杂度：O(nlogn)，其中 n 是链表的长度。

        空间复杂度：O(logn)，其中 n 是链表的长度。空间复杂度主要取决于递归调用的栈空间。

    * ai 评价

        官方给出的这个递归解法，本质上是**带边界控制的归并排序**。

        相比于你在上一版中通过“物理断开”链表（即令 `pre->next = nullptr`）来拆分子序列，官方的解法更像是在处理**数组区间**：它通过传递 `head` 和 `tail`（左闭右开区间）来界定当前处理的范围。

        以下是针对这个官方解法的深度评价：

        **亮点：逻辑严密且优雅**

        1. **无需物理断开**：

            它在递归过程中不修改原链表的 `next` 指针。只有当递归到底部，即 `head->next == tail` 时，才执行 `head->next = nullptr`。这保证了在拆分阶段，原链表结构是完整的，仅靠指针逻辑划分区间。

        2. **区间控制非常精准**：

            使用了典型的快慢指针寻找中点，但快指针的终止条件改成了 `fast != tail`。这完美适配了递归区间的逻辑，避免了处理单节点或空节点时的边界越界问题。

        3. **合并逻辑稳健**：

            `merge` 函数是标准的双指针合并。使用 `dummyHead` 简化了头节点处理，虽然会有微小的内存分配开销，但代码逻辑非常清晰。

        ---

        **不足与性能隐患**

        1. **空间复杂度并非 $O(1)$**：

            虽然进阶要求 $O(1)$ 空间，但**官方这个递归解法依然是 $O(\log n)$**。因为递归调用会占用系统栈空间。如果链表长度为 $5 \times 10^4$，递归深度约为 16 层左右。

        2. **内存泄露风险（C++ 细节）**：
        
            官方代码里有一行 `ListNode* dummyHead = new ListNode(0);`。在 `merge` 结束后直接返回了 `dummyHead->next`，但**没有 `delete dummyHead**`。虽然在 LeetCode 判题环境中这通常不会导致崩溃，但在实际工程中，这属于明显的内存泄露（Memory Leak）。

        3. **时间常数略大**：

            每次递归都要用快慢指针重新跑一遍中点。在平衡性不好的情况下（虽然归并排序很稳定），相比于直接操作指针，这种逻辑判断略多。

        ---

        **核心对比：官方 vs 你之前的尝试**

        | 特性 | 你之前的做法 | 官方解法 |
        | --- | --- | --- |
        | **拆分方式** | 尝试找到中点（但忘了断开） | 传入 `tail` 指针界定范围 |
        | **基础情况** | `head == nullptr |  |
        | **安全性** | 容易陷入死循环（若不断开） | 逻辑闭环，非常安全 |
        | **内存管理** | 尝试 `delete dummy` | **存在内存泄露**（忘记 `delete`） |

        **一个小的优化建议**

        如果你想让官方代码更“C++ 范儿”一点，可以把 `merge` 里的 `dummyHead` 改成局部对象：

        ```cpp
        ListNode dummy(0); // 栈上对象
        ListNode* temp = &dummy;
        // ... 后续逻辑 ...
        return dummy.next; // 无需 delete，自动销毁

        ```

        **总结评价：**
        官方这个解法是“标准教科书式”的递归写法。它最大的功劳是演示了如何**不破坏链表原结构**的情况下，利用边界指针进行区间划分。

        不过，如果你想追求题目进阶要求的 **$O(1)$ 空间复杂度**，这个解法依然是不合格的。**你需要用“自底向上（Bottom-up）”的迭代写法来彻底干掉递归栈。你想看看那个像“织布”一样复杂的迭代版代码吗？**

* 官方答案：方法二：自底向上归并排序

    ```cpp
    class Solution {
    public:
        ListNode* sortList(ListNode* head) {
            if (head == nullptr) {
                return head;
            }
            int length = 0;
            ListNode* node = head;
            while (node != nullptr) {
                length++;
                node = node->next;
            }
            ListNode* dummyHead = new ListNode(0, head);
            for (int subLength = 1; subLength < length; subLength <<= 1) {
                ListNode* prev = dummyHead, *curr = dummyHead->next;
                while (curr != nullptr) {
                    ListNode* head1 = curr;
                    for (int i = 1; i < subLength && curr->next != nullptr; i++) {
                        curr = curr->next;
                    }
                    ListNode* head2 = curr->next;
                    curr->next = nullptr;
                    curr = head2;
                    for (int i = 1; i < subLength && curr != nullptr && curr->next != nullptr; i++) {
                        curr = curr->next;
                    }
                    ListNode* next = nullptr;
                    if (curr != nullptr) {
                        next = curr->next;
                        curr->next = nullptr;
                    }
                    ListNode* merged = merge(head1, head2);
                    prev->next = merged;
                    while (prev->next != nullptr) {
                        prev = prev->next;
                    }
                    curr = next;
                }
            }
            return dummyHead->next;
        }

        ListNode* merge(ListNode* head1, ListNode* head2) {
            ListNode* dummyHead = new ListNode(0);
            ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
            while (temp1 != nullptr && temp2 != nullptr) {
                if (temp1->val <= temp2->val) {
                    temp->next = temp1;
                    temp1 = temp1->next;
                } else {
                    temp->next = temp2;
                    temp2 = temp2->next;
                }
                temp = temp->next;
            }
            if (temp1 != nullptr) {
                temp->next = temp1;
            } else if (temp2 != nullptr) {
                temp->next = temp2;
            }
            return dummyHead->next;
        }
    };
    ```

    复杂度分析

        时间复杂度：O(nlogn)，其中 n 是链表的长度。

        空间复杂度：O(1)。

### 分隔链表

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。



示例 1：

输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]

示例 2：

输入：head = [2,1], x = 2
输出：[1,2]



提示：

    链表中节点的数目在范围 [0, 200] 内
    -100 <= Node.val <= 100
    -200 <= x <= 200

解答：

1. 自己写的，两个指针

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* partition(ListNode* head, int x) {
            ListNode *dummy_head = new ListNode;
            dummy_head->next = head;
            ListNode *p1 = dummy_head, *p2;
            ListNode *p1_nex, *p2_pre;
            while (p1->next && p1->next->val < x)
                p1 = p1->next;
            p2_pre = p1;
            p2 = p1->next;
            while (p2) {
                if (p2->val < x) {
                    p2_pre->next = p2->next;
                    p1_nex = p1->next;
                    p1->next = p2;
                    p2->next = p1_nex;
                    p1 = p1->next;
                    p2 = p2_pre->next;
                    continue;
                }
                p2_pre = p2;
                p2 = p2->next;
            }
            return dummy_head->next;
        }
    };
    ```

    线性思考：

    1. 要想排序，总得知道链表里都有哪些数吧，至少得遍历一遍

        一个指针遍历可以搞定这道题吗？似乎不能，因为一个指针只能感知到它的前后两个位置，做一些简单的相邻节点位置交换。

    2. 两个指针呢？第一个记录要插入的位置，第二个往后扫描小于 x 的节点。

        这个似乎可以。我们先写核心代码，假设这条链表很长，对于最核心的情况：

        ```cpp
        while (p2) {  // p2 遍历链表
            if (p2->val < x) {  // 发现要交换的时机
                p2_pre->next = p2->next;  // 移走 p2
                p1->next = p2;
                p2->next = p1_nex;  // 将 p2 插入 p1 的后面
            }
        }
        ```

    3. 补全上下文，我们得到

        ```cpp
        class Solution {
        public:
            ListNode* partition(ListNode* head, int x) {
                ListNode *p1 = head, *p2;
                ListNode *p1_nex, *p2_pre;
                while (p1 && p1->next && p1->next->val < x)
                    p1 = p1->next;
                p2_pre = p1;
                p2 = p1->next;
                while (p2) {
                    if (p2->val < x) {
                        p2_pre->next = p2->next;
                        p1_nex = p1->next;
                        p1->next = p2;
                        p2->next = p1_nex;
                        p1 = p1->next;
                        p2 = p2_pre->next;
                        continue;
                    }
                    p2_pre = p2;
                    p2 = p2->next;
                }
                return head;
            }
        };
        ```

    4. 测了下，有个 case 过不去，原来是未考虑头节点的边界条件，这个一般用 dummy head 就能解决：

        ```cpp
        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode() : val(0), next(nullptr) {}
        *     ListNode(int x) : val(x), next(nullptr) {}
        *     ListNode(int x, ListNode *next) : val(x), next(next) {}
        * };
        */
        class Solution {
        public:
            ListNode* partition(ListNode* head, int x) {
                ListNode *dummy_head = new ListNode;
                dummy_head->next = head;
                ListNode *p1 = dummy_head, *p2;
                ListNode *p1_nex, *p2_pre;
                while (p1->next && p1->next->val < x)
                    p1 = p1->next;
                p2_pre = p1;
                p2 = p1->next;
                while (p2) {
                    if (p2->val < x) {
                        p2_pre->next = p2->next;
                        p1_nex = p1->next;
                        p1->next = p2;
                        p2->next = p1_nex;
                        p1 = p1->next;
                        p2 = p2_pre->next;
                        continue;
                    }
                    p2_pre = p2;
                    p2 = p2->next;
                }
                return dummy_head->next;
            }
        };
        ```

2. 官方答案，构建两个链表，再拼接

    直观来说我们只需维护两个链表 small 和 large 即可，small 链表按顺序存储所有小于 x 的节点，large 链表按顺序存储所有大于等于 x 的节点。遍历完原链表后，我们只要将 small 链表尾节点指向 large 链表的头节点即能完成对链表的分隔。

    为了实现上述思路，我们设 smallHead 和 largeHead 分别为两个链表的哑节点，即它们的 next 指针指向链表的头节点，这样做的目的是为了更方便地处理头节点为空的边界条件。同时设 small 和 large 节点指向当前链表的末尾节点。开始时 smallHead=small,largeHead=large。随后，从前往后遍历链表，判断当前链表的节点值是否小于 x，如果小于就将 small 的 next 指针指向该节点，否则将 large 的 next 指针指向该节点。

    遍历结束后，我们将 large 的 next 指针置空，这是因为当前节点复用的是原链表的节点，而其 next 指针可能指向一个小于 x 的节点，我们需要切断这个引用。同时将 small 的 next 指针指向 largeHead 的 next 指针指向的节点，即真正意义上的 large 链表的头节点。最后返回 smallHead 的 next 指针即为我们要求的答案。

    ```cpp
    class Solution {
    public:
        ListNode* partition(ListNode* head, int x) {
            ListNode* small = new ListNode(0);
            ListNode* smallHead = small;
            ListNode* large = new ListNode(0);
            ListNode* largeHead = large;
            while (head != nullptr) {
                if (head->val < x) {
                    small->next = head;
                    small = small->next;
                } else {
                    large->next = head;
                    large = large->next;
                }
                head = head->next;
            }
            large->next = nullptr;
            small->next = largeHead->next;
            return smallHead->next;
        }
    };
    ```

3. 网友答案，使用 c++ 智能指针

    ```cpp
    class Solution {
    public:
        ListNode *partition(ListNode *head, int x) {
            auto ptr1 = make_unique<ListNode>(0);
            auto ptr2 = make_unique<ListNode>(0);
            auto p1 = ptr1.get(), p2 = ptr2.get();
            for (auto p = head; p; p = p->next) {
                if (p->val < x) {
                    p1->next = p;
                    p1 = p;                
                } else {
                    p2->next = p;
                    p2 = p;
                }
            }
            p2->next = nullptr;
            p1->next = ptr2->next;
            return ptr1->next;
        }
    };
    ```

4. 未使用 dummy head 时，ai 指出的边界问题

    * 头指针处理不当：如果第一个节点就大于等于x，那么p1会指向这个节点，但这样小于x的节点应该插入到头部，而你的代码无法处理这种情况。

    * 边界条件缺失：没有处理链表为空或只有一个节点的情况。

    * 指针初始化问题：当所有节点都小于x时，第一个while循环会让p1指向最后一个节点，然后p2_pre = p1，p2 = p1->next（此时为null），后续循环无法执行。

    ai 给出的解法（与官方解法相同）及理由：

    ```cpp
    class Solution {
    public:
        ListNode* partition(ListNode* head, int x) {
            // 创建两个虚拟头节点
            ListNode* smallHead = new ListNode(0);
            ListNode* largeHead = new ListNode(0);
            ListNode* small = smallHead;
            ListNode* large = largeHead;
            
            // 遍历原链表
            ListNode* curr = head;
            while (curr) {
                if (curr->val < x) {
                    small->next = curr;
                    small = small->next;
                } else {
                    large->next = curr;
                    large = large->next;
                }
                curr = curr->next;
            }
            
            // 连接两个链表
            small->next = largeHead->next;
            large->next = nullptr;  // 避免形成环
            
            // 获取新链表的头节点
            ListNode* result = smallHead->next;
            
            // 释放虚拟头节点
            delete smallHead;
            delete largeHead;
            
            return result;
        }
    };
    ```

    推荐解法说明：

    使用两个虚拟头节点的方法更加清晰可靠：

    * 创建两个链表：一个存放小于x的节点，一个存放大于等于x的节点

    * 遍历原链表，根据节点值分配到两个链表

    * 最后将两个链表连接起来

    * 注意将大链表的末尾置为nullptr，避免形成环

    这种方法时间复杂度O(n)，空间复杂度O(1)，思路清晰且不易出错。

5. ai 讨论：这道题有点像排序里的 partition 操作

    你说得对！这道题确实很像快速排序中的partition操作，不过有几个重要的区别：

    **与快排partition的对比：**

    **相似点：**
    
    - 都根据一个基准值x将元素分成两部分
    - 左边部分都小于基准值

    **关键区别：**

    1. **稳定性要求**：

        - 本题要求**保留原始相对顺序**（稳定）
        - 快排的partition通常是不稳定的（元素会交换位置）

    2. **基准值处理**：

        - 本题：小于x的在左边，大于等于x的在右边
        - 快排：通常小于基准值的在左边，大于基准值的在右边，基准值本身在中间某个位置

    3. **实现方式**：

        - 本题：更适合用两个链表分别收集
        - 快排：通常用双指针交换元素

    **类似的应用场景：**

    这种"稳定分区"操作在实际中很有用：

    - **荷兰国旗问题**的稳定版本
    - **基数排序**中的桶分配
    - **数据库查询**中的条件过滤（保持原顺序）
    - **数据清洗**时根据条件分离数据

    **扩展思考：**

    如果要求实现一个更通用的稳定分区函数：

    ```cpp
    // 稳定分区，根据predicate返回true/false
    template<typename T, typename Pred>
    ListNode<T>* stablePartition(ListNode<T>* head, Pred pred) {
        ListNode<T>* trueHead = new ListNode<T>(0);
        ListNode<T>* falseHead = new ListNode<T>(0);
        ListNode<T>* trueTail = trueHead;
        ListNode<T>* falseTail = falseHead;
        
        ListNode<T>* curr = head;
        while (curr) {
            if (pred(curr->val)) {
                trueTail->next = curr;
                trueTail = trueTail->next;
            } else {
                falseTail->next = curr;
                falseTail = falseTail->next;
            }
            curr = curr->next;
        }
        
        trueTail->next = falseHead->next;
        falseTail->next = nullptr;
        
        ListNode<T>* result = trueHead->next;
        delete trueHead;
        delete falseHead;
        
        return result;
    }
    ```

    这种操作在函数式编程中也很常见（如Python的filter、Haskell的partition），都是保留相对顺序的稳定分区。

### 二进制链表转整数

给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。已知此链表是一个整数数字的二进制表示形式。

请你返回该链表所表示数字的 十进制值 。

最高位 在链表的头部。

 

示例 1：

输入：head = [1,0,1]
输出：5
解释：二进制数 (101) 转化为十进制数 (5)

示例 2：

输入：head = [0]
输出：0

 

提示：

    链表不为空。
    链表的结点总数不超过 30。
    每个结点的值不是 0 就是 1。

1. 不反转链表，直接计算

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        int getDecimalValue(ListNode* head) {
            int n = 0;
            ListNode *p = head;
            while (p)
            {
                n <<= 1;
                n += p->val;  // 也可以写成 n |= p->val
                p = p->next;
            }
            return n;
        }
    };
    ```

1. 先翻转链表再计算

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        int getDecimalValue(ListNode* head) {
            ListNode *p = head;
            ListNode *pre = nullptr;
            ListNode *nex;
            while (p) {
                nex = p->next;
                p->next = pre;
                pre = p;
                p = nex;
            }
            p = pre;
            int val = 0;
            size_t radix = 1;
            while (p) {
                val += radix * p->val;
                radix *= 2;
                p = p->next;
            }
            return val;
        }
    };
    ```

### 从尾到头打印链表

> 输入一个链表的头结点，按照 从尾到头 的顺序返回节点的值。

> 返回的结果用数组存储。

样例：

```
输入：[2, 3, 5]
返回：[5, 3, 2]
```

代码：

1. `reverse()`反转

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        vector<int> reversePrint(ListNode* head) {
            vector<int> ans;
            ListNode *p = head;
            while (p)
            {
                ans.push_back(p->val);
                p = p->next;
            }
            reverse(ans.begin(), ans.end());
            return ans;
        }
    };
    ```

1. 递归

    ```c++
    class Solution {
    public:
        vector<int> reversePrint(ListNode* head) {
            if(!head)
                return {};
            vector<int> a=reversePrint(head->next);
            a.push_back(head->val);
            return a;
        }
    };
    ```

1. 栈

    ```c++
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     int val;
     *     ListNode next;
     *     ListNode(int x) { val = x; }
     * }
     */
    class Solution {
        public int[] reversePrint(ListNode head) {
            Stack<ListNode> stack = new Stack<ListNode>();
            ListNode temp = head;
            while (temp != null) {
                stack.push(temp);
                temp = temp.next;
            }
            int size = stack.size();
            int[] print = new int[size];
            for (int i = 0; i < size; i++) {
                print[i] = stack.pop().val;
            }
            return print;
        }
    }
    ```

### 复杂链表的复制

> 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

分析：第一种方案是两次遍历，第一次遍历使用哈希表存储节点的旧指针和新指针的映射关系，并构建新的链表；第二次遍历填充新链表的`random`值。第二种方案是使用交错的新旧链表，从而使用顺序关系确定`random`的值，再将旧链表删去即可。

代码：

1. 散列表法

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* next;
        Node* random;
        
        Node(int _val) {
            val = _val;
            next = NULL;
            random = NULL;
        }
    };
    */
    class Solution {
    public:
        Node* copyRandomList(Node* head) {
            if (!head) return nullptr;
            unordered_map<Node*, Node*> m;
            Node *p = head;
            Node *dummy_head = new Node(-1);
            Node *new_p = dummy_head;
            while (p)
            {
                new_p->next = new Node(p->val);
                m[p] = new_p->next;
                new_p = new_p->next;
                p = p->next;
            }

            p = head;
            new_p = dummy_head->next;
            while (p)
            {
                if (p->random) new_p->random = m[p->random];
                else new_p->random = nullptr;
                p = p->next;
                new_p = new_p->next;
            }
            
            return dummy_head->next;
        }
    };
    ```

    在 leetcode 中，似乎不允许`delete`内存。

1. 链表交错法

    这道题的难点是 random 所指向的节点可能暂时还没存在，但是如果我们把所有节点都复制完了一遍，却又不好找 random 所指向的节点了。所以我们可以先在每个节点后复制一个节点，把`[1, 2, 3]`变成`[1, 1', 2, 2', 3, 3']`，这样既能定创建好节点，又能定位到节点。最后把复制的节点脱离出来形成一个单独的链表就可以了。

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* next;
        Node* random;
        
        Node(int _val) {
            val = _val;
            next = NULL;
            random = NULL;
        }
    };
    */
    class Solution {
    public:
        Node* copyRandomList(Node* head) {
            if (!head) return nullptr;
            Node *p = head, *next;
            while (p)
            {
                next = p->next;
                p->next = new Node(p->val);
                p->next->next = next;
                p = next;
            }
            
            p = head;
            while (p)
            {
                if (p->random) p->next->random = p->random->next;
                else p->next->random = nullptr;
                p = p->next->next;
            }

            Node *new_head = head->next;
            p = head;
            while (p)
            {
                next = p->next->next;
                if (p->next->next) p->next->next = p->next->next->next;
                p->next = next;  // 原链表不能修改，所以把 next 给改回去
                p = next;
            }
            return new_head;
        }
    };
    ```

1. 回溯 + 散列表（官方给的递归解法）

    ```c++
    class Solution {
    public:
        unordered_map<Node*, Node*> cachedNode;

        Node* copyRandomList(Node* head) {
            if (head == nullptr) {
                return nullptr;
            }
            if (!cachedNode.count(head)) {
                Node* headNew = new Node(head->val);
                cachedNode[head] = headNew;
                headNew->next = copyRandomList(head->next);
                headNew->random = copyRandomList(head->random);
            }
            return cachedNode[head];
        }
    };
    ```

### 链表中倒数第k个节点

> 输入一个链表，输出该链表中倒数第 k 个结点。
> k >= 1
> 如果 k 大于链表长度，则返回 NULL;

代码：

1. 使用额外空间，只遍历链表一遍

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode* findKthToTail(ListNode* pListHead, int k) {
            vector<ListNode*> nums;
            while (pListHead != nullptr)
            {
                nums.emplace_back(pListHead);
                pListHead = pListHead->next;
            }
            if (k > nums.size())
                return NULL;
            else
                return nums[nums.size() - k];
        }
    };
    ```

1. 不使用额外空间，遍历链表两遍

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode* findKthToTail(ListNode* pListHead, int k) {
            int n = 0;
            ListNode *pnode = pListHead;
            while (pnode != nullptr)
            {
                ++n;
                pnode = pnode->next;
            }
            if (k > n) return nullptr;
            pnode = pListHead;
            for (int i = 0; i < n - k; ++i)
            {
                pnode = pnode->next;
            }
            return pnode;
            
        }
    };
    ```

### 链表中环的入口结点（环形链表 II）（链表中环的入口节点）

> 给定一个链表，若其中包含环，则输出环的入> 口节点。
> 
> 若其中不包含环，则输出null。

代码：

1. 使用额外空间，单指针

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        unordered_set<ListNode*> m;
        ListNode *entryNodeOfLoop(ListNode *head) {
            ListNode *pnode = head;
            while (pnode != nullptr)
            {
                if (m.find(pnode->next) != m.end())
                    return pnode->next;
                else
                    m.emplace(pnode->next);
                pnode = pnode->next;
            }
            return nullptr;
        }
    };
    ```

1. 不使用额外空间，双指针

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        unordered_set<ListNode*> m;
        ListNode *entryNodeOfLoop(ListNode *head) {
            if (head == nullptr || head->next == nullptr)
                return nullptr;
                
            ListNode *slow = head, *fast = head;

            while (slow && fast)
            {
                slow = slow->next;
                fast = fast->next;
                if (fast == nullptr)
                    return nullptr;
                fast = fast->next;
                if (slow == fast)
                    break;
            }
            
            if (slow == nullptr || fast == nullptr)
                return nullptr;
            
            slow = head;
            while (slow != fast)
            {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    };
    ```

    这个问题其实是从约瑟夫环问题发展过来的。与这个问题比较相似的问题是，假如甲乙两个人在环形赛道上比赛，乙在 1/4 圆处开始，且速度是甲的 2 倍，那么甲乙何时在哪里相遇？

    这道题是用链表实现的，我们先用连续量做一下推导。

    假设`p1`, `p2`同时从起点出发，`p2`的速度是`p1`的 2 倍，环外长度为`x`，环的长度为`y`，从环入口处到`p1`、`p2`相遇点的长度为`y1`，剩余长度为`y2`。

    由于`p2`的速度是`p1`的 2 倍，且`p1`，`p2`行进的时间相同，因此`p2`走的路程是`p1`的 2 倍。

    `p1`走的路程为：`x + y1 + n1 * y`，这里`n1`表示`p1`完整走过的圈数。

    `p2`走的路径为：`x + y1 + n2 * y`

    因此可列出方程：`x + y1 + n2 * y = 2 * (x + y1 + n1 * y)`

    化简可得：`x + y1 + (2 * n1 - n2) * y = 0`

    由于`(2 * n1 - n2)`一定是个整数，所以我们可以得出结论：`x + y1`一定是环长度`y`的整数倍。

    下面我们来找规律：

    当倍数为 1 时，`x + y1 = y1 + y2`，此时`x = y2`

    当倍数为 2 时，`x + y1 = 2 * (y1 + y2)`，此时`x = y1 + 2 * y2 = y2 + y`。

    当倍数为 3 时，`x + y1 = 3 * (y1 + y2)`，此时`x = y2 + 2 * y`。

    根据上面的规律可以看出，`x`的长度总是把环形长度走完几圈后，又多出来一个`y2`。

    那么我们让`p2`从`y1`处开始走，让`p1`从头开始走，当走完`y2`，再走完`n`个`y`后，一定会在入口处相遇。

    注意：

    1. 这个分析只能证明算法的正确性，并不能知道是怎么想出来的。

    2. 这个分析只针对连续变量，至于怎么变成离散量，目前还没有眉目。

    后来自己写的：

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode *detectCycle(ListNode *head) {
            if (!head || !head->next) return nullptr;  // 如果有环的话，即使是单个节点，head->next 也不可能为空
            ListNode *p1 = head, *p2 = head;
            do  // 刚开始时 p1 == p2，无法进入循环，所以这里使用 do while 的形式
            {
                p1 = p1->next;
                p2 = p2->next->next;  // 因为前面判断过了，所以不用担心第一次进入循环体时 p2->next 为空的情况
            } while (p2 && p2->next && p1 != p2);
            if (!p2 || !p2->next) return nullptr;
            p2 = head;  // 在这道题中，单步操作似乎并没有影响，但是在找链表交点那道题中，单步操作似乎是有影响的。如何才能避免单步操作的造成结果出错？
            while (p1 != p2) 
            {
                p1 = p1->next;
                p2 = p2->next;
            }
            return p1;
        }
    };
    ```

1. 自己写的，自从对循环的功能开始解耦，就好懂多了

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode *detectCycle(ListNode *head) {
            ListNode *slow = head, *fast = head;
            while (fast && fast->next)
            {
                slow = slow->next;
                fast = fast->next->next;
                if (slow == fast)  // 我们让外循环只负责往前走指针
                {
                    fast = head;
                    while (fast != slow)  // 让内循环负责返回正确答案。这两层循环互不干扰，代码就很好懂了
                    {
                        fast = fast->next;
                        slow = slow->next;
                    }
                    return fast;
                }
            }
            return nullptr;
        }
    };
    ```

    这种方法可能涉及到约瑟夫环的问题，比想象中要复杂许多。

1. 后来又写的，双指针

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode *detectCycle(ListNode *head) {
            if (!head)
                return NULL;
            ListNode *p1 = head, *p2 = head;
            do {
                p1 = p1->next;
                p2 = p2->next;
                if (!p2)
                    return NULL;
                p2 = p2->next;
                if (p1 == p2)
                    break;
            } while (p1 && p2);
            if (!p2)
                return NULL;
            p2 = head;
            while (p1 && p2)
            {
                if (p1 == p2)
                    return p1;
                p1 = p1->next;
                p2 = p2->next;
            }
            return NULL;
        }
    };
    ```

    这个代码应该是目前思路最清晰，最好懂的一个版本了。

    使用`do while`是为了让指针先走一步，防止刚开始就触发`if`的 break。

    如果不这么写，那么就只能写成：

    ```cpp
    ListNode *p1 = head, *p2 = head;
    p1 = p1->next;
    p2 = p2->next;
    if (!p2)
        return NULL;
    p2 = p2->next;
    while (p1 && p2)
    {
        if (p1 == p2)
            break;
        ...
    }
    ```

    让`p1`和`p2`在`while()`开始之前先走一步，初始化循环条件。

#### AI 分析之旅

* 分析一下我的代码有什么问题： 

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode *detectCycle(ListNode *head) {
            ListNode *p1 = head, *p2 = head->next;
            while (p2 && p2->next) {
                p1 = p1->next;
                p2 = p2->next->next;
                if (p2 == nullptr || p2->next == nullptr) {
                    return nullptr;
                }
                if (p1 == p2) {
                    p2 = head;
                    while (p1 != p2) {
                        p1 = p1->next;
                        p2 = p2->next;
                    }
                    return p1;
                }
            }
            return nullptr;
        }
    };
    ```

    这个问题非常有挑战性！在处理环形链表时，快慢指针（Floyd 判圈算法）确实是最佳实践，但你的代码在**起始点设置**和**数学逻辑**上出现了一点小偏差。

    **核心问题分析**

    1. **初始位置不对称**：

        你设置了 `p1 = head` 而 `p2 = head->next`。在快慢指针算法中，为了保证数学推导的严谨性，**两者必须从同一个起点出发**（通常都是 `head`）。

        * 如果你让快指针先走一步，会导致后续 `p1 == p2` 相遇时的位置不再满足“慢指针走过的距离等于环入口偏移量”的数学关系。

    2. **空指针风险**：

        如果输入 `head` 是 `nullptr`，你的代码在第一行 `p2 = head->next` 就会直接崩溃（Segment Fault）。

    3. **相遇后的逻辑偏移**：

        当 `p1 == p2` 时，根据数学推导，此时从**头节点**出发一个指针，从**相遇点**出发一个指针，两者以相同速度移动，一定会在环入口相遇。但因为你起始位置不对，这个“相遇点”已经偏离了标准位置。

    **数学推导回顾**

    设头节点到环入口距离为 $a$，环入口到相遇点距离为 $b$，相遇点再走 $c$ 回到环入口。

    * 慢指针路程：$a + b$
    * 快指针路程：$a + b + k(b + c)$ （$k$ 为绕环圈数）
    * 因为快指针速度是慢指针两倍：$2(a + b) = a + b + k(b + c)$
    * 化简得：$a = (k - 1)(b + c) + c$

    这意味着：**从头节点到入口的距离 ($a$)，等于从相遇点走过 $c$ 加上若干圈环的距离。** 因此，两者同步走必在入口相遇。

    **修正后的代码**

    这是按照 $O(1)$ 空间复杂度要求修正后的逻辑：

    ```cpp
    class Solution {
    public:
        ListNode *detectCycle(ListNode *head) {
            if (!head || !head->next) return nullptr;

            ListNode *slow = head;
            ListNode *fast = head;

            // 1. 寻找相遇点
            while (fast && fast->next) {
                slow = slow->next;
                fast = fast->next->next;
                
                if (slow == fast) { // 成功相遇，说明有环
                    // 2. 寻找环入口
                    ListNode *p1 = head;
                    ListNode *p2 = slow; // 此时 slow 在相遇点
                    while (p1 != p2) {
                        p1 = p1->next;
                        p2 = p2->next;
                    }
                    return p1; // 返回入口节点
                }
            }

            return nullptr; // fast 走到终点，无环
        }
    };

    ```

    **关键改进点**

    * **统一起点**：`slow` 和 `fast` 都从 `head` 开始。
    * **安全检查**：增加了 `!head` 的判断。
    * **简化循环**：相遇后再启动寻找入口的逻辑，结构更清晰。

    **你想让我再解释一下为什么 `a = c`（即为什么第二次相遇点一定是入口）的具体几何演示吗？**

* 详细讲讲 Floyd 判圈算法

    既然你对这个算法感兴趣，我们跳出枯燥的代码，用一个**“操场赛跑”**的比喻来彻底拆解 **Floyd 判圈算法**（Floyd's Cycle-Finding Algorithm），也叫“龟兔赛跑算法”。

    这个算法优雅的地方在于：它不仅能判断有没有环，还能用极简的数学逻辑精准定位环的入口。

    **第一阶段：判断是否有环（相遇）**

    想象乌龟（slow）和兔子（fast）在操场跑步：

    * **规则**：乌龟每次走 **1** 步，兔子每次走 **2** 步。

    * **结论**：

        * 如果跑道是一条**直线**，兔子会先到达终点。

        * 如果跑道有**环**，兔子最终会在环里绕圈，并从后面“套圈”追上乌龟。

    > **核心逻辑**：只要 $fast$ 指针能和 $slow$ 指针重合，链表就一定有环。

    **第二阶段：寻找环的入口（数学证明）**

    这是该算法最精妙的部分。为什么“相遇后，一个从头出发，一个从相遇点出发，同步走必会在入口相遇”？

    我们来推导一下：

    1. 变量定义

        假设：

        * $a$：从**头节点**到**环入口**的距离。
        * $b$：从**环入口**到**首次相遇点**的距离。
        * $c$：从**首次相遇点**继续往前走，回到**环入口**的距离。
        * $L$：环的周长，即 $L = b + c$。

    2. 距离关系

        当乌龟和兔子第一次相遇时：

        * **乌龟走的距离**：$S_{slow} = a + b$
        * **兔子走的距离**：$S_{fast} = a + b + k \cdot L$ （兔子可能已经在环里转了 $k$ 圈）

        由于兔子的速度是乌龟的 **2** 倍，所以：

        $$2 \cdot (a + b) = a + b + k \cdot L$$

    3. 神奇的化简

        我们将等式化简：

        $$a + b = k \cdot L$$

        $$a = k \cdot L - b$$

        为了看清本质，我们把 $L$ 拆解为 $b + c$：


        $$a = (k - 1) \cdot L + (L - b)$$

        $$a = (k - 1) \cdot L + c$$

    4. 物理含义

        这个公式 $a = (k - 1) \cdot L + c$ 告诉我们：
        从**头节点**走到**环入口**的距离（即 $a$），等于从**相遇点**出发绕环 $(k-1)$ 圈后再走过距离 $c$。

        **换句话说：**
        如果你让一个指针 $p_1$ 从头开始走，另一个指针 $p_2$ 从刚才的相遇点开始走，由于 $a$ 和 $c$ 在模环长意义下是相等的，当 $p_1$ 走了 $a$ 步到达入口时，$p_2$ 刚好也转完了圈并走过了 $c$ 的距离，两人准时在入口握手。

    **算法复杂度**

    | 维度 | 复杂度 | 说明 |
    | --- | --- | --- |
    | **时间复杂度** | $O(N)$ | 寻找相遇点最多走 $N$ 步，寻找入口最多再走 $N$ 步。 |
    | **空间复杂度** | $O(1)$ | 只需要两个额外的指针，不需要哈希表。 |

    **为什么不用哈希表？**

    虽然哈希表（记录访问过的节点）也能解这道题，但它需要 $O(N)$ 的空间。在嵌入式开发或者大数据的内存敏感场景下，Floyd 算法这种**“空间换时间”的极致压缩**极具魅力。

    **怎么样，这个数学推导过程清晰吗？如果让你手动模拟一个带环链表（比如 $3 \to 2 \to 0 \to -4 \to 2$），你能心算出来它们在哪里第一次相遇吗？**

* 我有两个问题：

    1. 你推导的路程都是连续量，但是这道题是离散量（链表的节点），这两者之间是怎么转换的？

    2. 你只是对推导了“寻找环的入口”在数学上成立，但是我想知道提出这个想法的动机（motivation）是什么？

    这两个问题问得非常深刻，直击**离散数学**与**算法启发式思维**的核心。

    1. 连续量与离散量的转换：步数即距离

        在物理学中，距离 = 速度 × 时间。而在链表这种离散结构中：

        * **单位距离**：定义为“移动一次指针”（即 `p = p->next`）。
        * **坐标定义**：头节点为坐标 $0$，第 $n$ 个节点坐标为 $n$。
        * **速度定义**：$v_{slow} = 1$ 步/单位时间，$v_{fast} = 2$ 步/单位时间。

        **离散量下的“相遇”定义：**
        在连续空间中，两个物体可以在任何坐标相遇。但在离散空间（链表）中，必须满足在**同一个时刻 $t$**，两个指针指向**同一个节点索引**。

        * **证明相遇的必然性**：
        设环长为 $L$。当慢指针进入环后，快指针与慢指针的**相对距离**（离散步数差）每经过一个单位时间，就会减少 $1$（因为 $2 - 1 = 1$）。
        由于相对距离是一个整数，且每次减 $1$，它**必然会减到 $0$**，而不会跳过（Overstep）。这就是离散量下能够“相遇”的物理保证。

    2. 算法的动机（Motivation）：从何而来？

        Robert W. Floyd 提出这个算法（1967年）时，其动机并非凭空捏造，而是源于对**有限状态机**和**内存极度稀缺**环境的思考。

        **A. 寻找“不变性”（Invariance）**

        算法设计的核心动机是寻找一种**不需要记录历史**就能感知“循环”的方法。

        * **直觉**：如果你在迷宫里走，不准做标记（$O(1)$ 空间），你怎么知道自己在绕圈？
        * **启发**：如果两个人速度不同，快的人一定会“套圈”慢的人。这是一种自然的**周期性同步现象**。

        **B. 寻找入口的动机：镜像对称**

        为什么要让一个指针回到起点？

        * **动机来自对公式的观察**：当我们得到 $a + b = k \cdot L$ 时，数学家敏锐地观察到 $a$（未知的外部路径）和 $k \cdot L - b$（环内剩下的路径）在数值上是等价的。
        * **结构化思维**：既然 $a$ 是从 `head` 出发到入口的距离，而 $c = L - b$ 是从相遇点到入口的距离。为了让两个指针“同步”撞击入口，最简单的办法就是**消除掉那个不确定的 $k \cdot L$（圈数）**。
        * **逻辑闭环**：只要两人同时开始以 $v=1$ 移动，那个 $(k-1)$ 圈的差距只是让其中一人在环里多转了几圈，最终他们一定会**同时**抵达那个“汇合点”。

        **C. 工程动机：内存是金**

        在 20 世纪 60 年代的计算机上，内存（RAM）极其昂贵。

        * 如果使用**哈希表**（标记法），空间复杂度是 $O(N)$，对于长链表会撑爆内存。
        * Floyd 的动机是挑战**空间的极限**：能否仅用两个寄存器（两个指针）就完成任务？

    **总结**

    * **转换**：离散步数完全等价于物理距离，步数的整数性保证了相遇不会被跳过。
    * **动机**：源于对“相对速度”带来的周期性观察，以及在无标记条件下对数学对称性（$a \equiv c \pmod L$）的利用。

    这种通过**速度差**来探测拓扑结构的思想，后来也深刻影响了计算机科学中的**时钟同步**和**网络环路检测**。

    **既然聊到了离散量，你是否想过：如果快指针的速度不是 $2$，而是 $3$ 或者 $100$，这个算法还能 work 吗？**

* "既然聊到了离散量，你是否想过：如果快指针的速度不是 2，而是 3 或者 100，这个算法还能 work 吗？" 好啊，你再分析分析

    这是一个非常棒的思维拓展！答案是：**在“判断是否有环”阶段，速度是 3 甚至 100 都能 work；但在“寻找环入口”阶段，速度 2 是最完美的“最优解”。**

    我们分两个层面来深度拆解：

    1. 判断是否有环：只要速度有差，终会相遇

        在离散的链表环里，快慢指针的相遇本质上是一个**同余方程**是否有解的问题。

        假设慢指针速度为 $1$，快指针速度为 $v$（$v > 1$）。

        * 每次移动，两者的**相对距离**缩短了 $v - 1$。
        * 设环的长度为 $L$。当慢指针进入环时，假设两者之间的距离为 $D$（$0 \le D < L$）。
        * 只要在某个时刻 $t$，满足 $(v - 1)t \equiv D \pmod L$，两者就会相遇。

        **离散量的陷阱：会跳过去吗？**

        * 如果 $v=2$，相对速度 $v-1 = 1$。距离每次减 $1$，必然经过 $0$，**绝对不会跳过**。
        * 如果 $v=3$，相对速度 $v-1 = 2$。如果初始距离 $D$ 是奇数，且环长 $L$ 是偶数，那么距离的变化序列可能是 $5 \to 3 \to 1 \to (-1 \equiv L-1)$。
        * 看起来跳过去了？**其实没有。** 因为在环里，这只是开启了新一轮的追逐。只要 $gcd(v-1, L)$ 能整除初始距离（或者更简单点，只要 $v-1$ 与 $L$ 的互质关系允许），它们迟早会撞上。
        * **结论**：只要快指针比慢指针快，它们在环里一定能相遇，只是速度过快可能会多绕几圈才抓到“乌龟”。

    2. 寻找环入口：速度 $v=2$ 的特殊魔力

        这就是为什么教科书几乎只教 $v=2$ 的原因。请看公式的对比：

        **情况 A：速度 $v=2$（经典版）**

        * 相遇时：$2 \times (\text{slow}) = \text{fast}$
        * $2(a+b) = a+b + kL \implies \mathbf{a = kL - b}$
        * 这里的系数非常干净，直接推导出 $a$ 与 $L-b$（即相遇点到入口的距离）是等价关系。

        **情况 B：速度 $v=3$**

        * 相遇时：$3 \times (\text{slow}) = \text{fast}$
        * $3(a+b) = a+b + kL \implies 2a + 2b = kL \implies \mathbf{a = \frac{kL}{2} - b}$

        * **麻烦来了**：

        1. 此时 $a$ 不再直接等于 $L-b$。
        2. 你需要知道环的长 $L$ 或者绕的圈数 $k$ 才能计算出偏移量。
        3. 你无法直接通过“一个从头走，一个从相遇点走”这种简单的同步方式找到入口。

    3. 为什么 $v=2$ 是工程上的最优解？

        1. **步进最稳**：相对速度为 $1$，保证在离散节点上，快指针在追上慢指针的那一刻，**一定是重合在同一个节点**，而不会出现“快指针刚刚在慢指针后一个，下一秒就在慢指针前一个”的跳跃。
        2. **计算最简**：只有 $v=2$ 能推导出 $a = (k-1)L + c$ 这种不需要额外参数（如环长、速度倍数）就能定位入口的绝妙关系。
        3. **效率平衡**：虽然速度 $v=100$ 能更快进入环，但在离散结构中，过大的步长会增加判断 `next` 指针是否为空的逻辑负担（比如你要写 $100$ 个 `if(p->next)`），且并不能显著降低时间复杂度（都是 $O(N)$）。

    **总结**

    Floyd 算法不仅仅是一个简单的 trick，它利用了**最小相对速度 1** 来保证离散空间的相遇，并利用了**两倍速关系**制造了数学上的对称性，从而在不增加空间开销的前提下，把“找入口”这个难题给化解了。

    **你想试试看，如果要在 $v=3$ 的情况下强行找入口，代码逻辑要增加哪些复杂的判断吗？**

* 我记得还有一道题，开始时必须让两个指针错位，或者用 do ... while() 循环让两个指针先动一下，否则会因为 if (p1 == p2) 判断条件而直接在刚开始时就退出循环

    你的记忆非常准确。这种现象常见于**环形链表 I**（只判断是否有环）或者在**不引入额外哨兵节点**（Dummy Node）的情况下写代码。

    这种情况主要源于**“初始状态即满足退出条件”**的悖论。

    1. 为什么会有这种写法？

        如果你把 `slow` 和 `fast` 都初始化为 `head`，那么逻辑就会变成这样：

        ```cpp
        ListNode *slow = head, *fast = head;
        while (slow != fast) { // 还没开始跑，就因为相等而退出了！
            slow = slow->next;
            fast = fast->next->next;
        }

        ```

        为了绕开这个“起跑线重合”的问题，程序员通常有两种流派：

        **流派 A：错位起跑（你提到的）**

        让快指针先跑一步，避开起跑线。

        ```cpp
        ListNode *slow = head, *fast = head->next; 
        while (slow != fast) {
            if (!fast || !fast->next) return false;
            slow = slow->next;
            fast = fast->next->next;
        }
        return true;

        ```

        * **弊端**：这就是你第一段代码里遇到的坑。对于“环形链表 II”（找入口），错位起跑会彻底破坏 $2 \cdot \text{slow} = \text{fast}$ 的数学对称性，导致找入口的逻辑失效。

        **流派 B：`do...while` 循环**

        先跑起来，再检查条件。

        ```cpp
        ListNode *slow = head, *fast = head;
        do {
            if (!fast || !fast->next) return false;
            slow = slow->next;
            fast = fast->next->next;
        } while (slow != fast);

        ```

        * **优点**：它既保持了 $O(1)$ 空间，又维护了“从同一点出发”的数学严谨性，非常适合找入口。

    2. 这道题（环形链表 II）的最佳实践

        对于你正在做的这道**找入口节点**的题，最稳妥、可读性最强的写法其实是**“死循环 + 内部 break”**。

        这种写法不需要 `do...while` 也不需要错位，逻辑非常清晰：

        ```cpp
        ListNode *slow = head, *fast = head;
        bool hasCycle = false;

        // 第一步：先看有没有环
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) { // 只有跑起来之后相遇才算
                hasCycle = true;
                break;
            }
        }

        // 第二步：如果有环，再找入口
        if (hasCycle) {
            slow = head; // 乌龟回起点
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }

        ```

    3. 为什么你脑子里有“错位起跑”的印象？

        其实“错位起跑”在**另一种链表题**中是标准答案：**寻找链表的中点**。

        当你想找中点时：

        * 如果想让中点偏向**左边**（比如 4 个节点找第 2 个），通常会让 `fast` 初始化为 `head->next`。
        * 如果想让中点偏向**右边**（比如 4 个节点找第 3 个），通常会让 `fast` 初始化为 `head`。

        这种对初始位置的微调是为了处理**偶数长度**链表的边界情况。而在处理“环”的问题时，除非你只判断“有没有”，否则千万不要轻易打破“同一起跑线”的约定。

### 环形链表

给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

 

示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：

输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。

 

提示：

    链表中节点的数目范围是 [0, 104]
    -105 <= Node.val <= 105
    pos 为 -1 或者链表中的一个 有效索引 。

 

进阶：你能用 O(1)（即，常量）内存解决此问题吗？


代码：

1. 线性思考

    我们需要记忆一个节点是否之前出现过，首先想到哈希表。

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        bool hasCycle(ListNode *head) {
            unordered_set<ListNode*> s;
            ListNode *p = head;
            while (p)
            {
                if (s.find(p) != s.end())
                    return true;
                s.insert(p);
                p = p->next;
            }
            return false;
        }
    };
    ```

    注意题目并没有说每个节点的`val`各不相同，因此必须使用`ListNode*`作为哈希表存储的值。

2. 快慢指针

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        bool hasCycle(ListNode *head) {
            if (!head || !head->next) return false;
            ListNode *p1 = head, *p2 = head;
            do  // 如果使用 while 的话，刚开始 p1 和 p2 都位于 head 处，循环不会执行
            {
                p1 = p1->next;
                if (p2 && p2->next) p2 = p2->next->next;  // 因为 p2 比 p1 走得快，所以只需要判断 p2 是否走到尾就可以了
                else return false;
            } while (p1 != p2);

            // 不想用 do while 的话也可以用下面这种写法
            // ListNode *p1 = head, *p2 = head->next;
            // while (p1 != p2)
            // {
            //     p1 = p1->next;
            //     if (p2 && p2->next) p2 = p2->next->next;
            //     else return false;
            // }
            return true;
        }
    };
    ```

    后来又写的，这样比较清晰一点：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        bool hasCycle(ListNode *head) {
            if (!head)
                return false;
            ListNode *p1 = head, *p2 = head->next;
            if (!p2)  // 在走第二次之前保证 p2 有效
                return false;
            p2 = p2->next;
            while (p1 && p2)
            {
                if (p1 == p2)
                    return true;
                p1 = p1->next;
                p2 = p2->next;
                if (!p2)  // 在走第二次之前保证 p2 有效
                    return false;
                p2 = p2->next;
            }
            return false;
        }
    };
    ```

    后来又写的（2026.02.01），同样遇到了`list = [1, 2]`的问题，此时 p1 == p2，跳过了 while 循环，又都不为空，所以直接返回 true 了，这显然不对。对于避免初始状态两个指针相同的问题，可以让两个指针的位置错位：

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        bool hasCycle(ListNode *head) {
            if (!head || !head->next) {  // 因为后面用到了 head->next，所以这里要保证 head 有效。
                return false;
            }
            ListNode *p1 = head, *p2 = head->next;
            while (p1 && p2 && p2->next && p1 != p2) {  // 由于循环中出现了 p1->next，所以这里保证 p1 有效。同样地，出现了 p2->next->next，所以要保证 p2->next 有效，而保证 p2->next 有效，就要先保证 p2 有效。最后是核心判断，两个指针是否相遇
                p1 = p1->next;
                p2 = p2->next->next;
            }
            // 退出循环有两种情况，一种是相遇，一种是存在指针无效（走到链表末尾）。这里是判断指针无效的几种情况：
            if (p1 == nullptr || p2 == nullptr || p2->next == nullptr) {
                return false;
            }
            // 相遇，说明存在环
            return true;
        }
    };
    ```

    前面的 do while 方法也挺不错。官方提供的另一种理解：

    > 我们可以假想一个在 head 之前的虚拟节点，慢指针从虚拟节点移动一步到达 head，快指针从虚拟节点移动两步到达 head.next，这样我们就可以使用 while 循环了。

    如果不用 do while，可以把判断放到 while 循环体的下面（网友的想法）：

    ```java
    class Solution {
        public boolean hasCycle(ListNode head) {
            ListNode slow = head;
            ListNode fast = head;
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
                if (slow == fast) {
                    return true;
                }
            }
            return false;
        }
    }
    ```

1. 使用反转链表（网友的想法）（会破坏链表结构）

    如果有环那么head指向非空，设不在环上的节点长度为a，在环上的为b，最后时间是2a+b，但因为循环的时候少了判断，所以用时反而比快慢指针少

    ```cpp
    class Solution:
        def hasCycle(self, head: Optional[ListNode]) -> bool:
            pre = None
            h = head
            while h:
                temp = h.next
                h.next = pre
                pre = h
                h = temp
            if head and head.next != None:
                return True
            else:
                return False
    ```

    有环的话反转到最后一定会经过头节点，实际上只要反转的过程中二次遇到头节点就可以判断有环了

1. 还可以修改已经访问过的链表里的值（会修改链表数据），比如设置为 MAX_INT, 10^5 + 1 等，如果遇到之前自己设置的值，那么说明有环

### 反转链表

LCR 024. 反转链表

给定单链表的头节点 head ，请反转链表，并返回反转后的链表的头节点。

 

示例 1：

输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

示例 2：

输入：head = [1,2]
输出：[2,1]

示例 3：

输入：head = []
输出：[]

 

提示：

    链表中节点的数目范围是 [0, 5000]
    -5000 <= Node.val <= 5000

 

进阶：链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？


解答：

1. 迭代法

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        ListNode* reverseList(ListNode* head) {
            ListNode *prev = nullptr;  // 这个很重要
            ListNode *pnode = head;
            ListNode *next;
            while (pnode)
            {
                next = pnode->next;
                pnode->next = prev;
                prev = pnode;
                pnode = next;
            }
            return prev;
        }
    };
    ```

    反转链表需要三个临时指针，分别存储上一个，当前的以及下一个节点。

    提问：怎样写才能保证`while()`中的条件最简单，循环中的`if`语句最少？原则：先给未赋值的变量赋值；上一步出现在等号右边的变量下一步要出现在等号左边。

    这道题也有许多地方可以说的，比如为什么需要构造 3 个指针变量，这并不是线性思考可以想出来的。

1. 递归法

    `head->next->next = head; head->next = nullptr;`这两句看得不是很懂。

    懂了。`head->next->next = heaad;`其实就是翻转两个相邻节点，`head->next = nullptr;`是因为，递归是从尾节点向前进行的，所以我们一直保证当前的节点的头部指向`nullptr`就行了。这样到第一个元素时，也能保证它的`next`是个`nullptr`。

    其实这是一个后序遍历。因为为了得到当前链表反转后的头节点，我们必须知道子链表反转后的头节点。我们只要把子链表反转后的头节点返回就可以了。

    我们定义递归函数返回的是反转链表后的头节点。其实这个返回值对我们反转当前节点没有什么帮助。

    官方解：

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode* reverseList(ListNode* head) {
            if (!head || !head->next) return head;  // !head 主要是考虑空链表，!head->next 考虑的是正常链表
            ListNode *tail = reverseList(head->next);
            head->next->next = head;  // 显然 tail != head->next，如果写成 tail->next = head;，会出错
            head->next = nullptr;  // 这个递归是从末端开始修改链表的，但是这种逻辑似乎破坏了递归了初衷，不知道该怎么理解
            return tail;
        }
    };
    ```

    思路：

    首先我们构造一个简单版的。对于某个节点`i`，我们可以将`i`的`next`指向`i-1`，也可以将`i+1`的`next`指向`i`。
    
    我们可以将链表看作一个线性的树。对于第一种方法，相当于让我们把当前节点的`next`指向父节点。由于这时候我们的假设是当前节点的左右子节点都已经处理好了，所以这算是一个后序遍历。如何拿到父节点的信息呢？或许可以通过传入参数拿到。由此可得到递归的第一种写法：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode *new_head;
        void recur(ListNode *cur, ListNode *parent)
        {
            if (!cur->next)
            {
                new_head = cur;
                cur->next = parent;
                return;
            }
            recur(cur->next, cur);
            cur->next = parent;
        }

        ListNode* reverseList(ListNode* head) {
            if (!head) return nullptr;
            recur(head, nullptr);
            return new_head;
        }
    };
    ```

    根据第二种方法，我们将`i+1`节点的`next`指向`i`，然后再处理子节点。这其实相当于先处理当前节点，再处理子节点，是先序遍历。由于没用到父节点，只用到了子节点，所以不需要额外处理。但是写代码的时候遇到一个问题：我们修改`i+1`节点的`next`时，破坏了下一级的递归结构，即`i+1`不再和`i+2`连接，我们没办法再走到`i+2`了。或许我们可以额外记录一些`i+2`节点的信息，硬写递归也是可以写的。可是这样的话，就破坏递归的优美性了，而且效率不高。

    后来又写的：

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode *tail_node = nullptr;
        ListNode *recur(ListNode *head) {
            if (head == nullptr)
                return nullptr;
            ListNode *nex = recur(head->next);
            if (nex == nullptr) {
                tail_node = head;
                return head;
            }
            nex->next = head;
            head->next = nullptr;
            return head;
        }

        ListNode* reverseList(ListNode* head) {
            recur(head);
            return tail_node;
        }
    };
    ```

    别人的一个答案：

    ```java
    class Solution {
    public ListNode reverseList(ListNode head) {
    if (head == null) {
    return null;
    }
    return reverse(head);
    }

    public ListNode reverse(ListNode cur) {
        if (cur.next == null) {
            return cur;
        }
        ListNode last = reverse(cur.next);
        cur.next.next = cur;
        cur.next = null;
        return last;
    }

    }
    ```

1. 可以用栈

1. 头插法

    ```java
    class Solution {
        public ListNode reverseList(ListNode head) {
            if(head == null){
                return head;
            }
            ListNode headNode = new ListNode(-1, null);
            ListNode p = head;
            ListNode r = null;
            while(p != null){
                r = p.next;
                p.next = headNode.next;
                headNode.next = p;
                p = r; 
            }
            return headNode.next;
        }
    }
    ```

### 合并两个排序的链表（合并两个有序链表）

代码：

1. 官方解法一：递归

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
            if (!l1) return l2;
            if (!l2) return l1;
            if (l1->val < l2->val)
            {
                l1->next = mergeTwoLists(l1->next, l2);
                return l1;
            }
            else
            {
                l2->next = mergeTwoLists(l1, l2->next);
                return l2;
            }
        }
    };
    ```

    这个和归并排序有点像。那么归并排序是否也有类似下面的迭代的方法？如果不能用迭代，为什么？

1. 修改原链表的指针

    在创建一个新链表时，通常要用到`dummy_head`。

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
            ListNode *dummy_head = new ListNode;
            ListNode *p1 = list1, *p2 = list2;
            ListNode *p = dummy_head;
            while (p1 && p2)
            {
                if (p1->val < p2->val)
                {
                    p->next = p1;
                    p1 = p1->next;
                }
                else
                {
                    p->next = p2;
                    p2 = p2->next;
                }
                p = p->next;
            }
            // 退出循环时有两种情况，一种是 p1 存在，p2 为空；另一种是 p1 为空，p2 存在，如果按照这个逻辑写，为：
            // if (p1 && !p2) { xxxx }
            // if (!p1 && p2) { xxxx }
            // 或者更复杂的嵌套形式：
            // if (p1) { if (p2) xxx; else xxx; } else { if (p2) xxx; else xxx; }
            // 实际上两个 if 是互斥的，可以简化成下面的写法，非常巧妙：
            if (p1) p->next = p1;
            if (p2) p->next = p2;
            return dummy_head->next;
        }
    };
    ```

1. 创建一个新链表

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
            ListNode *dummy_head = new ListNode;
            ListNode *p = dummy_head;
            ListNode *p1 = list1, *p2 = list2;
            while (p1 && p2) {
                if (p1->val <= p2->val) {
                    p->next = new ListNode(p1->val);
                    p1 = p1->next;
                } else {
                    p->next = new ListNode(p2->val);
                    p2 = p2->next;
                }
                p = p->next;
            }
            while (p1) {
                p->next = new ListNode(p1->val);
                p1 = p1->next;
                p = p->next;
            }
            while (p2) {
                p->next = new ListNode(p2->val);
                p2 = p2->next;
                p = p->next;
            }
            return dummy_head->next;
        }
    };
    ```

### 两个链表的第一个公共结点（链表相交）（两个链表的第一个重合节点）

相交链表

输入两个链表，找出它们的第一个公共结点。

当不存在公共节点时，返回空节点。

```
样例
给出两个链表如下所示：
A：        a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3

输出第一个公共节点c1
```

代码：

1. 双指针互换链表法

    * 看到“交点”，想到“相同路程”
    
    * 我们构造一种让 A 和 B 走相同路程的方式
    
    * 假如在相交前只属于链表 A 的长度为 a，只属于链表 B 的长度为 b，相交后 A 和 B 共同的长度为 c。由此可表示 A 的长度为 m = a + c，B 的长度为 n = b + c。

    * 我们注意到如果 m 再加一个 b，n 再加一个 a，则 m' 就等于 n' 了，这时候就会出现相交。

    * 那么我们让指针 pa 走完 A 后，再从 B 的的头开始走 b，让指针 pb 走完 B 后的，再从 A 的开头开始走 a，那么就一定相交了。

    * 最巧妙的是如果没有交点，那么 pa 和 pb 会同时走到两个链表的结尾，且都是 nullptr

    * 因为链表是离散的，我们考虑的情况是连续的，所以边界条件比较难处理。如果一个链表为空怎么办？如果一个链表

    让两个指针同时从头开始走，指针走到头部后从另一个链表的开头再开始走，直到两指针相遇。相遇处即为公共结点处。

    如果没有交点，那么两个指针将同时走到两个链表的结尾。

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        ListNode *findFirstCommonNode(ListNode *headA, ListNode *headB) {
            ListNode *pnode_a = headA, *pnode_b = headB;
            while (pnode_a != pnode_b)
            {
                if (pnode_a)
                    pnode_a = pnode_a->next;  // 为什么要写成 if else 的形式？
                else
                    pnode_a = headB;
                if (pnode_b)
                    pnode_b = pnode_b->next;
                else
                    pnode_b = headA;
            }
            return pnode_a;
        }
    };
    ```

    官方题解：

    ```cpp
    class Solution {
    public:
        ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
            if (headA == nullptr || headB == nullptr) {
                return nullptr;
            }
            ListNode *pA = headA, *pB = headB;
            while (pA != pB) {
                pA = pA == nullptr ? headB : pA->next;
                pB = pB == nullptr ? headA : pB->next;
            }
            return pA;
        }
    };
    ```

    后来又写的，把`while`完全展开，好理解多了：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
            if (!headA || !headB) return nullptr;
            ListNode *p1 = headA, *p2 = headB;
            while (p1 && p2)
            {
                p1 = p1->next;
                p2 = p2->next;
            }
            if (!p1) p1 = headB;
            if (!p2) p2 = headA;
            while (p1 && p2)
            {
                p1 = p1->next;
                p2 = p2->next;
            }
            if (!p1) p1 = headB;
            if (!p2) p2 = headA;
            while (p1 && p2 && p1 != p2)
            {
                p1 = p1->next;
                p2 = p2->next;
            }
            return p1;
        }
    };
    ```

    ```
                     a
    headA:       o - o - o
                   b       \    c
    headB:   o - o - o - o - o - o - x - o
                                   d
    ```

    假设`headA`在相交前长度是`a`，`headB`在相交前的长度为`b`，`headA`与`headB`相交后的共同长度为`d`。

    第一轮，指针`p1`和`p2`一起向前走，我们假设`headA`链表比较短，`p1`先走到头，走过的路程为`a + d`。此时将`p1`放置到`headB`处，开始第二轮。第二轮，该`p2`走到头，走过的路程为`b + d`，此时将`p2`放置到`headA`处，准备开始第三轮。第三轮，`p1`，`p2`一起走到`x`处，我们假设两链表交点到`p1`与`p2`的汇合点处的长度为`c`，那么第三轮中，`p1`走过了`b + c`，`p2`走过了`a + c`。

    统计一下，`p1`在这三轮中，一共走过了`(a + d) + (b + c)`，`p2`一共走过了`(b + d) + (a + c)`，他们的总和都是`a + b + c + d`，因此走过的路径相等。

    这种方法的细节非常多，远没有看起来那么简单。

    后来又写的（2026.02.01）:

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
            ListNode *node_1 = headA, *node_2 = headB;
            while (node_1 != nullptr && node_2 != nullptr) {
                node_1 = node_1->next;
                node_2 = node_2->next;
            }
            if (node_1 == nullptr) {
                node_1 = headB;
            }
            if (node_2 == nullptr) {
                node_2 = headA;
            }
            while (node_1 != nullptr && node_2 != nullptr) {
                node_1 = node_1->next;
                node_2 = node_2->next;
            }
            if (node_1 == nullptr) {
                node_1 = headB;
            }
            if (node_2 == nullptr) {
                node_2 = headA;
            }
            while (node_1 != nullptr && node_2 != nullptr) {
                if (node_1 == node_2) {
                    return node_1;
                }
                node_1 = node_1->next;
                node_2 = node_2->next;
            }
            return nullptr;
        }
    };
    ```

1. 哈希表

    先记录`A`的所有节点，然后在`B`中找有没有相同的。

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
            unordered_set<ListNode*> s;
            ListNode *p1 = headA, *p2 = headB;
            while (p1)
            {
                s.insert(p1);
                p1 = p1->next;
            }
            while (p2)
            {
                if (s.find(p2) != s.end())
                    return p2;
                p2 = p2->next;
            }
            return nullptr;
        }
    };
    ```

    这个想法非常直观，也非常容易想出来。

1. 双指针链表长度差法

    先遍历两个链表，得到长度，计算差值，然后再让长的链表的指针走完这个差值。后续如果两个指针相遇，那么要么到了交点，要么同时到链表结尾 nullptr。

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
            ListNode *node_1 = headA;
            size_t len_A = 0;
            while (node_1 != nullptr) {
                len_A++;
                node_1 = node_1->next;
            }
            ListNode *node_2 = headB;
            size_t len_B = 0;
            while (node_2 != nullptr) {
                len_B++;
                node_2 = node_2->next;
            }
            node_1 = headA;
            node_2 = headB;
            if (len_A >= len_B) {
                size_t len_diff = len_A - len_B;
                for (size_t i = 0; i < len_diff; ++i) {
                    node_1 = node_1->next;
                }
                while (node_1 != nullptr && node_2 != nullptr) {
                    if (node_1 == node_2) {  // 边界情况：注意这里要先判断再往下走。如果先往下走再判断会出错。为什么？
                        return node_1;
                    }
                    node_1 = node_1->next;
                    node_2 = node_2->next;
                }
                return nullptr;
            } else {
                size_t len_diff = len_B - len_A;
                for (size_t i = 0; i < len_diff; ++i) {
                    node_2 = node_2->next;
                }
                while (node_1 != nullptr && node_2 != nullptr) {
                    if (node_1 == node_2) {
                        return node_1;
                    }
                    node_1 = node_1->next;
                    node_2 = node_2->next;
                }
                return nullptr;
            }
            return nullptr;
        }
    };
    ```

### 移除链表元素

给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。

示例 1：

```
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
```

示例 2：

```
输入：head = [], val = 1
输出：[]
```

示例 3：

```
输入：head = [7,7,7,7], val = 7
输出：[]
```

代码：

1. 迭代法

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* removeElements(ListNode* head, int val) {
            ListNode *dummy_head = new ListNode(-1, head);
            ListNode *p = dummy_head;
            while (p && p->next)
            {
                if (p->next->val == val) p->next = p->next->next;  // 这样方便处理最后一个节点
                else p = p->next;
            }
            return dummy_head->next;
        }
    };
    ```

    因为头节点也可能被删除，所以要使用哑节点来处理头部。

    不过也可以这样写，对头节点单独处理：

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* removeElements(ListNode* head, int val) {
            if (!head) return head;
            ListNode *p = head;
            while (p->next)  // 走到最后一个节点就停止，而不是最后一个节点的 next 指针 (nullptr)
            {
                if (p->next->val == val) p->next = p->next->next;
                else p = p->next;
            }
            if (head->val == val) return head->next;
            else return head;
        }
    };
    ```

1. 递归法

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* removeElements(ListNode* head, int val) {
            if (!head) return nullptr;
            head->next = removeElements(head->next, val);
            return head->val == val ? head->next : head;
        }
    };
    ```

### 删除排序链表中的重复元素

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次 。

返回同样按升序排列的结果链表。


```
示例 1：


输入：head = [1,1,2]
输出：[1,2]
```

代码：

1. 迭代

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            if (!head) return head;  // 处理空头
            ListNode *p = head;
            while (p->next)  // 走到最后一个节点就停止
            {
                if (p->val == p->next->val) p->next = p->next->next; 
                else p = p->next;
            }
            return head;
        }
    };
    ```

    不特殊处理头节点的话，可以这么写：

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            ListNode *p = head;
            while (p && p->next)  // 这里的条件怎么写，关键在于循环体中用到了什么值
            {
                if (p->next->val == p->val) p->next = p->next->next;
                else p = p->next;
            }
            return head;
        }
    };
    ```

    后来又写的，感觉后来写的格式更好了：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            ListNode *p = head;
            while (p && p->next)
            {
                while (p && p->next && p->val == p->next->val)
                    p->next = p->next->next;
                p = p->next;
            }
            return head;
        }
    };
    ```

    前面的写法耦合了“删除”和“前进”两个操作在`while`里，现在这个写法把两个概念分离，并分别用两个`while`处理，我觉得比原来要好一些。

    后来又写的（2026.02.01）:

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            ListNode *p = head;
            while (p) {  // 因为在进入内部 while 前，要保证 p 有效，所以这里加了条件
                // 主线想法是：如果发现下一个节点的值和当前节点相同，那么让当前节点“吸收”下一个节点
                while (p->next && p->next->val == p->val) {  // 循环体中遇到了 p->next->next，所以要保证 p->next 有效，因为 p 一直在吸收下一个节点，在进循环体之间，p 就必须是有效的
                    p->next = p->next->next;
                }
                p = p->next;
            }
            return head;
        }
    };
    ```

    对比一下双 while 写法和前面的 while 中嵌套 if 的写法，各有什么特点？这两种方法分别适合什么情况？

1. 递归

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            if (!head || !head->next) return head;  // 因为要用到当前节点和下个节点两个节点，所以如果走到最后一个节点就直接返回
            head->next = deleteDuplicates(head->next);
            return head->val == head->next->val ? head->next : head;
        }
    };
    ```

### 移除重复节点

编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。

```
示例1:

 输入：[1, 2, 3, 3, 2, 1]
 输出：[1, 2, 3]
示例2:

 输入：[1, 1, 1, 1, 2]
 输出：[1, 2]
提示：

链表长度在[0, 20000]范围内。
链表元素在[0, 20000]范围内。
```

进阶：

如果不得使用临时缓冲区，该怎么解决？

代码：

1. 自己写的，用两个指针 + 哈希表

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        ListNode* removeDuplicateNodes(ListNode* head) {
            unordered_set<int> s;
            ListNode *p = head, *prev = new ListNode(-1);
            prev->next = head;
            while (p)
            {
                if (s.find(p->val) == s.end())
                {
                    s.insert(p->val);
                    prev = prev->next;
                    p = p->next;
                }
                else
                {
                    prev->next = p->next;
                    p = p->next;
                }           
            }
            return head;
        }
    };
    ```

    自己后来又写的：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode* removeDuplicateNodes(ListNode* head) {
            if (!head) return head;
            ListNode *p = head;
            unordered_set<int> s;
            s.insert(head->val);
            while (p)
            {
                if (p->next)
                {
                    if (s.find(p->next->val) != s.end())
                    {
                        p->next = p->next->next;
                        continue;  // 这个不能少；这条语句似乎涉及到 while 里面套 while 的问题。可能会出现很多个连续的重复项待删除，理论上应该在这里再写个 while()，把所有的重复项删掉，但是我们通过 continue 借用了外层的 while。这样内部的逻辑就稍微复杂了点。是否有关于这方面的设计指导呢？
                    }
                    else
                        s.insert(p->next->val);
                }
                p = p->next;
            }
            return head;
        }
    };
    ```

    自己写的双指针：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode(int x) : val(x), next(NULL) {}
    * };
    */
    class Solution {
    public:
        ListNode* removeDuplicateNodes(ListNode* head) {
            if (!head) return head;
            ListNode *p1 = head, *p2;
            int val;
            while (p1)
            {
                val = p1->val;
                p2 = p1;
                while (p2 && p2->next)  // 为什么这里有 p2，让我们看 p2 = p2->next 这一行
                {
                    while (p2->next && p2->next->val == val)  // p2 只有在重复节点都删完的情况下，才会移动到下一个节点
                        p2->next = p2->next->next;
                    p2 = p2->next;  // 虽然 while (p2->next) 保证了在进入循环时，p2 一定不是最后一个节点，但是经过上一句 p2->next = p2->next->next; 代码后，p2 就有可能变成最后一个节点
                }
                p1 = p1->next;
            }
            return head;
        }
    };
    ```

    我觉得我比官方答案写得好，三重循环各自的用途，写得明明白白。

1. 官方答案 1，用一个指针

    ```c++
    class Solution {
    public:
        ListNode* removeDuplicateNodes(ListNode* head) {
            if (head == nullptr) {
                return head;
            }
            unordered_set<int> occurred = {head->val};
            ListNode* pos = head;
            // 枚举前驱节点
            while (pos->next != nullptr) {
                // 当前待删除节点
                ListNode* cur = pos->next;
                if (!occurred.count(cur->val)) {
                    occurred.insert(cur->val);
                    pos = pos->next;
                } else {
                    pos->next = pos->next->next;
                }
            }
            pos->next = nullptr;
            return head;
        }
    };
    ```

1. 官方答案进阶，不使用额外缓冲区

    ```c++
    class Solution {
    public:
        ListNode* removeDuplicateNodes(ListNode* head) {
            ListNode* ob = head;
            while (ob != nullptr) {
                ListNode* oc = ob;
                while (oc->next != nullptr) {
                    if (oc->next->val == ob->val) {
                        oc->next = oc->next->next;
                    } else {
                        oc = oc->next;
                    }
                }
                ob = ob->next;
            }
            return head;
        }
    };
    ```

思考：

1. 这道题其实并不难，但是在写的时候还是有很多细节需要注意。比如当使用两个指针时，会面临某个指针失效，而另一个指针的值恰好是失效指针的值的问题。当使用单个指针时，`while()`中该填什么来判断结束条件？什么时候需要处理最后一个节点，什么时候不需要？什么时候必须使用`dummy_head`，什么时候不需要？

### 对链表进行插入排序

对链表进行插入排序。


插入排序的动画演示如上。从第一个元素开始，该链表可以被认为已经部分排序（用黑色表示）。
每次迭代时，从输入数据中移除一个元素（用红色表示），并原地将其插入到已排好序的链表中。
 

插入排序算法：

插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。
 
```
示例 1：

输入: 4->2->1->3
输出: 1->2->3->4
示例 2：

输入: -1->5->3->4->0
输出: -1->0->3->4->5
```

代码：

其实并不是很难，但是很考验细节的一道题。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* insertionSortList(ListNode* head) {
        ListNode *dummy_head = new ListNode(0, head);
        ListNode *p1 = dummy_head, *p2 = head;
        ListNode *temp;
 
        while (p2 && p2->next)
        {
            p1 = dummy_head;
            while (p1->next != p2 && p1->next->val < p2->next->val) p1 = p1->next;
            if (p1->next->val >= p2->next->val)  // 等号不能忽略，否则无法处理链表中有相等的元素
            {
                temp = p2->next;
                p2->next = p2->next->next;
                temp->next = p1->next;
                p1->next = temp;
            }
            else p2 = p2->next;  // 只有 p2 的下个节点小于 p1 的下个节点时，才移动 p2
        }
        return dummy_head->next;
    }
};
```

### 删除排序链表中的重复元素 II

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。

返回同样按升序排列的结果链表。

 
```
示例 1：


输入：head = [1,2,3,3,4,4,5]
输出：[1,2,5]
示例 2：


输入：head = [1,1,1,2,3]
输出：[2,3]
```

代码：

1. 逻辑挺复杂的，细节很多，不容易想出来。

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            if (!head) return head;
            ListNode *dummy_head = new ListNode(0, head);
            // node1 和 node2 作为两个相邻节点，交替往前探索
            ListNode *node1 = head, *node2 = head->next, *prev = dummy_head;
            while (node2)
            {
                if (node1->val == node2->val)
                {
                    while (node2 && node2->val == node1->val)
                    {
                        node1 = node2;
                        node2 = node2->next;
                    }
                    prev->next = node2;
                    node1 = node2;
                    if (node2) node2 = node2->next;
                }
                else
                {
                    prev = node1;
                    node1 = node2;
                    node2 = node2->next;
                }
            }
            return dummy_head->next;
        }
    };
    ```

1. 后来又写的，逻辑简单了很多

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* deleteDuplicates(ListNode* head) {
            ListNode *dummy_head = new ListNode(-1, head);
            ListNode *p = head, *prev = dummy_head;  // 用于处理第一个节点
            while (p && p->next)  // 因为下面的语句中用到了 p->val 和 p->next->val，所以这里必须保证他们有效
            {
                if (p->val == p->next->val)
                {
                    while (p && p->val == prev->next->val) p = p->next;  // 为了保证 p->val 有效，需要用短路逻辑
                    prev->next = p;  // 删除一段节点
                }
                else prev = p, p = p->next;  // 逗号表达式从左往右执行
            }
            return dummy_head->next;
        }
    };
    ```

### 回文链表

编写一个函数，检查输入的链表是否是回文的。

```
示例 1：

输入： 1->2
输出： false 
示例 2：

输入： 1->2->2->1
输出： true 
 

进阶：
你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？
```

代码：

1. 一个线性思考

    如果数组中只有一个数，那么返回`true`。

    如果有两个及以上的数，让指针`p`在链表中一边反转链表，一边寻找`[a, a]`模式或`[a, b, a]`模式。如果找不到，那么返回`false`。

    如果找到，那么让`p1`指向第一个`a`，`p2`指向第二个`a`。让`p1`往回走，`p2`继续向前走，判断`p1->val`与`p2->val`是否相等。如果不相等，或者没有同时走到尾，那么返回`false`。

    其余情况返回`true`。

    现在困难点在找`[a, a]`模式或`[a, b, a]`模式上，因为要保证从`p`节点开始至少有两个，或三个节点存在，分类讨论的情况比较多。不清楚这个问题该怎么解决。

    * 2026/02/01/00:

        让我们更线性一点，首先把链表看作连续的，p1 按原速向前走，p2 按 p1 的两倍速向前走，p2 走到终点，那么 p1 在路径的一半。

        p1 具体在哪？有两种情况，一种是偶数，假设链表为`[1, 2, 3, 4]`，p1 向前走一步，到 2，p2 向前走 2 步，到 3；p1 向前走一步，到 3，p2 向前走 2 步到 NULL。

        另一种情况是奇数个元素，假设链表为`[1, 2, 3]`，p1 向前走一步，到 2，p2 向前走 2 步，到 3，此时 p2->next 为 NULL，无法进行下一轮循环（`p2 = p2->next->next`）了。

        我们观察到，元素数是偶数时，p2 在 NULL，p1 在中间靠右的位置；元素是奇数时，p2 在最后一个元素，p1 在正中间的位置。我们可以根据退出循环时 p2 的值来判断这个链表是奇数还是偶数。

        以后如果遇到情况复杂的，可以使用这种举例推断找规律的方法。

        由此我们可以写出代码：

        ```cpp
        /**
         * Definition for singly-linked list.
         * struct ListNode {
         *     int val;
         *     ListNode *next;
         *     ListNode() : val(0), next(nullptr) {}
         *     ListNode(int x) : val(x), next(nullptr) {}
         *     ListNode(int x, ListNode *next) : val(x), next(next) {}
         * };
         */
        class Solution {
        public:
            bool isPalindrome(ListNode* head) {
                ListNode *p1 = head, *p2 = head;
                ListNode *pre = nullptr, *nex;
                while (p2 && p2->next) {
                    p2 = p2->next->next;
                    nex = p1->next;
                    p1->next = pre;
                    pre = p1;
                    p1 = nex;
                }
                if (!p2) {
                    p2 = p1;
                } else {
                    p2 = p1->next;
                }
                p1 = pre;
                while (p1 && p2) {
                    if (p1->val != p2->val) {
                        return false;
                    }
                    p1 = p1->next;
                    p2 = p2->next;
                }
                return true;
            }
        };
        ```

1. 将链表中的值复制到数组中，然后用双指针

    （线性思考）

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        bool isPalindrome(ListNode* head) {
            ListNode *p = head;
            vector<int> v;
            while (p)
            {
                v.push_back(p->val);
                p = p->next;
            }
            int left = 0, right = v.size() - 1;
            while (left < right)
            {
                if (v[left] != v[right]) return false;
                ++left;
                --right;
            }
            return true;
        }
    };
    ```

    这种方式`push_back()`需要动态分配内存，效率略低，先拿到链表长度，再一次性分配内存，效率比较高：

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        bool isPalindrome(ListNode* head) {
            size_t len = 0;
            ListNode *p = head;
            while (p) {
                len++;
                p = p->next;
            }
            size_t half_len = len / 2;
            p = head;
            vector<int> v(half_len);
            for (int i = 0; i < half_len; ++i) {
                v[i] = p->val;
                p = p->next;
            }
            if (len % 2 != 0 && p) {
                p = p->next;
            }
            size_t pos = v.size() - 1;
            while (p) {
                if (p->val != v[pos])
                    return false;
                p = p->next;
                --pos;
            }
            return true;
        }
    };
    ```

1. 递归（官方给的答案）

    ```c++
    class Solution {
        ListNode* frontPointer;
    public:
        bool recursivelyCheck(ListNode* currentNode) {
            if (currentNode != nullptr) {
                if (!recursivelyCheck(currentNode->next)) {
                    return false;
                }
                if (currentNode->val != frontPointer->val) {
                    return false;
                }
                frontPointer = frontPointer->next;
            }
            return true;
        }

        bool isPalindrome(ListNode* head) {
            frontPointer = head;
            return recursivelyCheck(head);
        }
    };
    ```

1. 栈

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        bool isPalindrome(ListNode* head) {
            ListNode *p = head;
            stack<int> s;
            while (p)
            {
                s.push(p->val);
                p = p->next;
            }
            p = head;
            int n = s.size() / 2;
            for (int i = 0; i < n; ++i)
            {
                if (p->val != s.top()) return false;
                p = p->next;
                s.pop();
            }
            return true;
        }
    };
    ```

1. 反转后半部分链表

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode(int x) : val(x), next(NULL) {}
     * };
     */
    class Solution {
    public:
        bool isPalindrome(ListNode* head) {
            ListNode *p1 = head, *p2 = head;
            int num = 0;
            while (p2)
            {
                ++num;
                p2 = p2->next;
            }
            if (num == 1 || num == 0) return true;

            p2 = head;
            for (int i = (num % 2 == 0 ? num / 2 : (num + 1) / 2); i > 0; --i)
            {
                p2 = p2->next;
            }
            ListNode *prev = nullptr;
            ListNode *next = p2->next;
            while (p2)
            {
                next = p2->next;
                p2->next = prev;
                prev = p2;
                p2 = next;
            }
            
            p2 = prev;
            for (int i = num / 2; i > 0; --i)
            {
                if (p1->val != p2->val) return false;
                p1 = p1->next;
                p2 = p2->next;
            }
            return true;
        }
    };
    ```

    还可以用快慢指针找到中间节点（官方给的答案）：

    ```c++
    class Solution {
    public:
        bool isPalindrome(ListNode* head) {
            if (head == nullptr) {
                return true;
            }

            // 找到前半部分链表的尾节点并反转后半部分链表
            ListNode* firstHalfEnd = endOfFirstHalf(head);
            ListNode* secondHalfStart = reverseList(firstHalfEnd->next);

            // 判断是否回文
            ListNode* p1 = head;
            ListNode* p2 = secondHalfStart;
            bool result = true;
            while (result && p2 != nullptr) {
                if (p1->val != p2->val) {
                    result = false;
                }
                p1 = p1->next;
                p2 = p2->next;
            }        

            // 还原链表并返回结果
            firstHalfEnd->next = reverseList(secondHalfStart);
            return result;
        }

        ListNode* reverseList(ListNode* head) {
            ListNode* prev = nullptr;
            ListNode* curr = head;
            while (curr != nullptr) {
                ListNode* nextTemp = curr->next;
                curr->next = prev;
                prev = curr;
                curr = nextTemp;
            }
            return prev;
        }

        ListNode* endOfFirstHalf(ListNode* head) {
            ListNode* fast = head;
            ListNode* slow = head;
            while (fast->next != nullptr && fast->next->next != nullptr) {
                fast = fast->next->next;
                slow = slow->next;
            }
            return slow;
        }
    };
    ```

### LRU 缓存机制（最近最少使用缓存）

运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
 

进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？
 
```
示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

代码：

1. 双向链表 + 散列表

    双向链表存储`key`和`value`，散列表存储`key`到`Node*`的映射。

    ```c++
    class Node
    {
        public:
        Node *prev, *next;
        int key;
        int val;
        Node(int key, int val): key(key), val(val), prev(nullptr), next(nullptr) {}
        Node(int key, int val, Node *prev, Node *next): key(key), val(val), prev(prev), next(next) {}
    };

    class LRUCache {
    public:
        unordered_map<int, Node*> m;  // (key, Node*)
        Node *dummy_head, *dummy_tail;
        int capacity;
        int n;
        LRUCache(int capacity) {
            dummy_head = new Node(-1, -1);
            dummy_tail = new Node(-1, -1, dummy_head, nullptr);
            dummy_head->next = dummy_tail;
            this->capacity = capacity;
            n = 0;
        }

        void lift(Node *node)
        {
            node->prev->next = node->next;
            node->next->prev = node->prev;
            node->next = dummy_head->next;
            node->prev = dummy_head;
            dummy_head->next->prev = node;
            dummy_head->next = node;
        }
        
        int get(int key) {
            if (m.find(key) == m.end()) return -1;
            Node *node = m[key];
            lift(node);
            return node->val;
        }
        
        void put(int key, int value) {
            if (m.find(key) != m.end())
            {
                Node *node = m[key];
                node->val = value;
                lift(node);
            }
            else
            {
                if (n == capacity)
                {
                    m.erase(dummy_tail->prev->key);
                    dummy_tail->prev->key = key;
                    dummy_tail->prev->val = value;
                    m[key] = dummy_tail->prev;
                    lift(dummy_tail->prev);
                }
                else
                {
                    Node *node = new Node(key, value, dummy_head, dummy_head->next);
                    dummy_head->next = node;
                    node->next->prev = node;
                    m[key] = node;
                    ++n;
                }
            }
        }
    };

    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache* obj = new LRUCache(capacity);
     * int param_1 = obj->get(key);
     * obj->put(key,value);
     */
    ```

1. 后来写的，用`list`和`unordered_map`实现，效率挺低的

    ```c++
    class LRUCache {
    public:
        unordered_map<int, list<pair<int, int>>::iterator> m;
        list<pair<int, int>> l;
        int n, capacity;
        LRUCache(int capacity) {
            this->capacity = capacity;
            n = 0;
        }
        
        int get(int key) {
            if (m.find(key) == m.end()) return -1;
            int val = m[key]->second;
            l.erase(m[key]);
            l.push_front(make_pair(key, val));
            m[key] = l.begin();
            return val;
        }
        
        void put(int key, int value) {
            if (m.find(key) != m.end())
            {
                l.erase(m[key]);
                l.push_front(make_pair(key, value));
                m[key] = l.begin();
            }
            else
            {
                if (n < capacity)
                {
                    l.push_front(make_pair(key, value));
                    m[key] = l.begin();
                    ++n;
                }
                else
                {
                    m.erase(l.back().first);
                    l.pop_back();
                    l.push_front(make_pair(key, value));
                    m[key] = l.begin();
                }
            }
        }
    };

    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache* obj = new LRUCache(capacity);
     * int param_1 = obj->get(key);
     * obj->put(key,value);
     */
    ```

### 两两交换链表中的节点


给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。


```
示例 1：


输入：head = [1,2,3,4]
输出：[2,1,4,3]
示例 2：

输入：head = []
输出：[]
示例 3：

输入：head = [1]
输出：[1]
```

1. 迭代解法

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* swapPairs(ListNode* head) {
            ListNode *dummy_head = new ListNode(-1, head);
            ListNode *p = dummy_head, *next;
            while (p && p->next && p->next->next)
            {
                next = p->next->next->next;
                p->next->next->next = p->next;
                p->next = p->next->next;
                p->next->next->next = next;
                p = p->next->next;
            }
            return dummy_head->next;
        }
    };
    ```

1. 递归解法（没看，有空了看看）

    ```c++
    class Solution {
    public:
        ListNode* swapPairs(ListNode* head) {
            if (head == nullptr || head->next == nullptr) {
                return head;
            }
            ListNode* newHead = head->next;
            head->next = swapPairs(newHead->next);
            newHead->next = head;
            return newHead;
        }
    };
    ```

### K 个一组翻转链表

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
 

```
示例 1：


输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
示例 2：


输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
示例 3：

输入：head = [1,2,3,4,5], k = 1
输出：[1,2,3,4,5]
示例 4：

输入：head = [1], k = 1
输出：[1]
```

代码：

1. 自己写的，用栈

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* reverseKGroup(ListNode* head, int k) {
            stack<ListNode*> s;
            ListNode *dummy_head = new ListNode(-1, head);
            ListNode *p = head, *next = head, *start = dummy_head;
            int n = 0;
            do
            {
                p = start;
                while (!s.empty())
                {
                    p->next = s.top();
                    s.pop();
                    p = p->next;
                }
                p->next = next;
                start = p;
                p = p->next;
                n = 0;
                while (p && n++ < k) s.push(p), p = p->next;
                next = p;
            } while (s.size() >= k);
            return dummy_head->next;
        }
    };
    ```

1. 官方代码，用的是模拟

    ```c++
    class Solution {
    public:
        // 翻转一个子链表，并且返回新的头与尾
        pair<ListNode*, ListNode*> myReverse(ListNode* head, ListNode* tail) {
            ListNode* prev = tail->next;
            ListNode* p = head;
            while (prev != tail) {
                ListNode* nex = p->next;
                p->next = prev;
                prev = p;
                p = nex;
            }
            return {tail, head};
        }

        ListNode* reverseKGroup(ListNode* head, int k) {
            ListNode* hair = new ListNode(0);
            hair->next = head;
            ListNode* pre = hair;

            while (head) {
                ListNode* tail = pre;
                // 查看剩余部分长度是否大于等于 k
                for (int i = 0; i < k; ++i) {
                    tail = tail->next;
                    if (!tail) {
                        return hair->next;
                    }
                }
                ListNode* nex = tail->next;
                // 这里是 C++17 的写法，也可以写成
                // pair<ListNode*, ListNode*> result = myReverse(head, tail);
                // head = result.first;
                // tail = result.second;
                tie(head, tail) = myReverse(head, tail);
                // 把子链表重新接回原链表
                pre->next = head;
                tail->next = nex;
                pre = tail;
                head = tail->next;
            }

            return hair->next;
        }
    };
    ```

### 重排链表

给定一个单链表 L 的头节点 head ，单链表 L 表示为：

L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

 

```
示例 1：

输入：head = [1,2,3,4]
输出：[1,4,2,3]
示例 2：


输入：head = [1,2,3,4,5]
输出：[1,5,2,4,3]
```

代码：

1. 自己写的，用栈

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        void reorderList(ListNode* head) {
            stack<ListNode*> s;
            ListNode *p = head, *p2, *next;
            int n = 0;
            while (p) s.push(p), p = p->next, ++n;
            n = (n + 1) / 2;
            p = head;
            while (n-- > 0)
            {
                p2 = s.top();
                s.pop();
                next = p->next;
                p->next = p2;
                p2->next = next;
                p = next;
            }
            p2->next = nullptr;
        }
    };
    ```

    后来又写的：

    ```c++
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        void reorderList(ListNode* head) {
            stack<ListNode*> s;
            ListNode *p = head;
            while (p)
            {
                s.push(p);
                p = p->next;
            }
            int n = (s.size() - 1 ) / 2;  // 如果用栈的话，具体要循环几次呢？需要分偶数和奇数两种情况讨论，如果是偶数，那么只需要 n / 2 - 1 次就可以了，如果是奇数，那么需要 n / 2 次。这里的次数与下面的节点交换和最后的 s.top()->next = nullptr; 都有关
            p = head;
            ListNode *next, *p2;
            for (int i = 0; i < n; ++i)
            {
                p2 = s.top();
                s.pop();
                next = p->next;
                p->next = p2;
                p2->next = next;
                p = next;
            }
            s.top()->next = nullptr;
        }
    };
    ```

1. 官方题解 1，`vector`

    ```c++
    class Solution {
    public:
        void reorderList(ListNode *head) {
            if (head == nullptr) {
                return;
            }
            vector<ListNode *> vec;
            ListNode *node = head;
            while (node != nullptr) {
                vec.emplace_back(node);
                node = node->next;
            }
            int i = 0, j = vec.size() - 1;
            while (i < j) {
                vec[i]->next = vec[j];
                i++;
                if (i == j) {
                    break;
                }
                vec[j]->next = vec[i];
                j--;
            }
            vec[i]->next = nullptr;
        }
    };
    ```

1. 官方题解 2，寻找中点，链表逆序，合并链表

    ```c++
    class Solution {
    public:
        void reorderList(ListNode* head) {
            if (head == nullptr) {
                return;
            }
            ListNode* mid = middleNode(head);
            ListNode* l1 = head;
            ListNode* l2 = mid->next;
            mid->next = nullptr;
            l2 = reverseList(l2);
            mergeList(l1, l2);
        }

        ListNode* middleNode(ListNode* head) {
            ListNode* slow = head;
            ListNode* fast = head;
            while (fast->next != nullptr && fast->next->next != nullptr) {
                slow = slow->next;
                fast = fast->next->next;
            }
            return slow;
        }

        ListNode* reverseList(ListNode* head) {
            ListNode* prev = nullptr;
            ListNode* curr = head;
            while (curr != nullptr) {
                ListNode* nextTemp = curr->next;
                curr->next = prev;
                prev = curr;
                curr = nextTemp;
            }
            return prev;
        }

        void mergeList(ListNode* l1, ListNode* l2) {
            ListNode* l1_tmp;
            ListNode* l2_tmp;
            while (l1 != nullptr && l2 != nullptr) {
                l1_tmp = l1->next;
                l2_tmp = l2->next;

                l1->next = l2;
                l1 = l1_tmp;

                l2->next = l1;
                l2 = l2_tmp;
            }
        }
    };
    ```

1. 递归

    ```java
        public void reorderList(ListNode head) {
            if(head == null || head.next == null || head.next.next == null)return;
            ListNode temp = head;
            while(temp.next.next != null)
                temp = temp.next;
            temp.next.next = head.next;
            head.next = temp.next;
            temp.next = null;
            reorderList(head.next.next);
        }
    ```

### 链表中的两数相加（两数相加 II）

给定两个 非空链表 l1和 l2 来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

可以假设除了数字 0 之外，这两个数字都不会以零开头。

 

```
示例1：

输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
示例2：

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[8,0,7]
示例3：

输入：l1 = [0], l2 = [0]
输出：[0]
```

代码：

1. 用栈，效率稍微低些

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            stack<ListNode*> s1, s2;
            ListNode *p1 = l1, *p2 = l2;
            while (p1)
            {
                s1.push(p1);
                p1 = p1->next;
            }
            while (p2)
            {
                s2.push(p2);
                p2 = p2->next;
            }
            int n1 = s1.size(), n2 = s2.size();
            int num1, num2;
            int sum = 0, carry = 0;
            ListNode *p;
            while (!s1.empty() || !s2.empty())
            {
                p = n1 >= n2 ? s1.top() : s2.top();
                if (s1.empty()) num1 = 0;
                else
                {
                    num1 = s1.top()->val;
                    s1.pop();
                }
                if (s2.empty()) num2 = 0;
                else
                {
                    num2 = s2.top()->val;
                    s2.pop();
                }           
                sum = num1 + num2;
                p->val = (sum + carry) % 10;
                carry = (sum + carry) / 10;
            }
            return carry ? new ListNode(1, p) : p;
        }
    };
    ```

1. 先翻转链表，再相加

    ```c++
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode *reverse_list(ListNode *head)
        {
            ListNode *pre = nullptr, *cur = head, *next;
            while (cur)
            {
                next = cur->next;
                cur->next = pre;
                pre = cur;
                cur = next;
            }
            return pre;
        }

        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            ListNode *head1 = reverse_list(l1);
            ListNode *head2 = reverse_list(l2);
            ListNode *p1 = head1, *p2 = head2;
            int num1, num2, sum = 0, carry = 0;
            ListNode *dummy_head = new ListNode(-1);
            ListNode *p = dummy_head;
            while (p1 || p2)
            {
                num1 = p1 ? p1->val : 0;
                num2 = p2 ? p2->val : 0;
                sum = num1 + num2;
                p->next = new ListNode((sum + carry) % 10);
                p = p->next;
                carry = (sum + carry) / 10;
                if (p1) p1 = p1->next;
                if (p2) p2 = p2->next;
            }
            if (carry) p->next = new ListNode(1), p = p->next;
            return reverse_list(dummy_head->next);
        }
    };
    ```

### 展平多级双向链表

多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给定位于列表第一级的头节点，请扁平化列表，即将这样的多级双向链表展平成普通的双向链表，使所有结点出现在单级双链表中。

 

```
示例 1：

输入：head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
输出：[1,2,3,7,8,11,12,9,10,4,5,6]
解释：

输入的多级列表如下图所示：


扁平化后的链表如下图：


示例 2：

输入：head = [1,2,null,3]
输出：[1,3,2]
解释：

输入的多级列表如下图所示：

  1---2---NULL
  |
  3---NULL
示例 3：

输入：head = []
输出：[]
```

代码：

1. 自己写的栈，虽然是一道看似简单的题，但是有很多细节，写了很长时间，很烦

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* prev;
        Node* next;
        Node* child;
    };
    */

    class Solution {
    public:
        Node* flatten(Node* head) {
            if (!head) return nullptr;
            stack<Node*> s;
            Node *p = head, *dummy_head = new Node(-1);
            Node *p2 = dummy_head;
            while (p || !s.empty())
            {
                if (p->child)
                {
                    if (p->next) s.push(p->next);
                    p2->next = p;
                    p->prev = p2;
                    p2 = p;
                    p = p->child;
                    p2->child = nullptr;
                    continue;
                }
                p2->next = p;
                p->prev = p2;
                p2 = p;
                p = p->next;
                if (!p && !s.empty())
                {
                    p = s.top();
                    s.pop();
                    p2->next = p;
                    p->prev = p2;
                }
            }
            dummy_head->next->prev = nullptr;
            return dummy_head->next;
        }
    };
    ```

    具体涉及到的问题有，如何初始化？什么时机压栈？什么时机出栈？

1. 官方给的 dfs

    ```c++
    class Solution {
    public:
        Node* flatten(Node* head) {
            function<Node*(Node*)> dfs = [&](Node* node) {
                Node* cur = node;
                // 记录链表的最后一个节点
                Node* last = nullptr;

                while (cur) {
                    Node* next = cur->next;
                    //  如果有子节点，那么首先处理子节点
                    if (cur->child) {
                        Node* child_last = dfs(cur->child);

                        next = cur->next;
                        //  将 node 与 child 相连
                        cur->next = cur->child;
                        cur->child->prev = cur;

                        //  如果 next 不为空，就将 last 与 next 相连
                        if (next) {
                            child_last->next = next;
                            next->prev = child_last;
                        }

                        // 将 child 置为空
                        cur->child = nullptr;
                        last = child_last;
                    }
                    else {
                        last = cur;
                    }
                    cur = next;
                }
                return last;
            };

            dfs(head);
            return head;
        }
    };
    ```

### 排序的循环链表

给定循环单调非递减列表中的一个点，写一个函数向这个列表中插入一个新元素 insertVal ，使这个列表仍然是循环升序的。

给定的可以是这个列表中任意一个顶点的指针，并不一定是这个列表中最小元素的指针。

如果有多个满足条件的插入位置，可以选择任意一个位置插入新的值，插入后整个列表仍然保持有序。

如果列表为空（给定的节点是 null），需要创建一个循环有序列表并返回这个节点。否则。请返回原先给定的节点。

 

```
示例 1：


 

输入：head = [3,4,1], insertVal = 2
输出：[3,4,1,2]
解释：在上图中，有一个包含三个元素的循环有序列表，你获得值为 3 的节点的指针，我们需要向表中插入元素 2 。新插入的节点应该在 1 和 3 之间，插入之后，整个列表如上图所示，最后返回节点 3 。


示例 2：

输入：head = [], insertVal = 1
输出：[1]
解释：列表为空（给定的节点是 null），创建一个循环有序列表并返回这个节点。
示例 3：

输入：head = [1], insertVal = 0
输出：[1,0]
```

代码：

1. 主要是复杂的判断，没什么技巧，全是细节

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* next;

        Node() {}

        Node(int _val) {
            val = _val;
            next = NULL;
        }

        Node(int _val, Node* _next) {
            val = _val;
            next = _next;
        }
    };
    */

    class Solution {
    public:
        Node* insert(Node* head, int insertVal) {
            if (!head)
            {
                head = new Node(insertVal);
                head->next = head;
                return head;
            }
            Node *p = head;
            while (p)
            {
                if ((p->val <= insertVal && (p->next == head || insertVal < p->next->val || p->val > p->next->val)) ||
                    (p->val >= insertVal && (p->next == head || (insertVal <= p->next->val && p->next->val < p->val))) ||
                    p->next == head)
                {
                    p->next = new Node(insertVal, p->next);
                    return head;
                }
                p = p->next;
            }
            return head;
        }
    };
    ```

### 链表排序

给定链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

 

示例 1：



输入：head = [4,2,1,3]
输出：[1,2,3,4]
示例 2：



输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
示例 3：

输入：head = []
输出：[]
 

提示：

链表中节点的数目在范围 [0, 5 * 104] 内
-105 <= Node.val <= 105
 

进阶：你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

代码：

1. 自己写的冒泡排序，超时了

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* sortList(ListNode* head) {
            int len = 0;
            ListNode *cur = head;
            ListNode *dummy_head = new ListNode(-1, head);
            ListNode *prev = dummy_head;
            ListNode *next;
            while (cur)
            {
                ++len;
                cur = cur->next;
            }
            for (int i = 0; i < len - 1; ++i)
            {
                prev = dummy_head;
                cur = dummy_head->next;
                for (int j = 0; j < len - i - 1; ++j)
                {
                    if (cur->next && cur->val > cur->next->val)
                    {
                        next = cur->next;
                        cur->next = next->next;
                        next->next = cur;
                        prev->next = next;
                        prev = next;
                    }
                    else
                    {
                        prev = cur;
                        cur = cur->next;
                    }
                }
            }
            return dummy_head->next;
        }
    };
    ```

1. 这道题用归并排序比较简单

### 合并K个升序链表

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
示例 2：

输入：lists = []
输出：[]
示例 3：

输入：lists = [[]]
输出：[]
 

提示：

k == lists.length
0 <= k <= 10^4
0 <= lists[i].length <= 500
-10^4 <= lists[i][j] <= 10^4
lists[i] 按 升序 排列
lists[i].length 的总和不超过 10^4


代码：

1. 两两合并

1. 小顶堆

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* mergeKLists(vector<ListNode*>& lists) {
            ListNode *dummy_head = new ListNode;
            int n = lists.size();
            vector<pair<int, ListNode*>> heads(n);
            priority_queue<pair<int, ListNode*>,
                vector<pair<int, ListNode*>>,
                greater<pair<int, ListNode*>>> q;
            for (int i = 0; i < n; ++i)
            {
                if (lists[i])
                    q.push(make_pair(lists[i]->val, lists[i]));
            }
            ListNode *p = dummy_head, *small;
            while (!q.empty())
            {
                small = q.top().second;
                p->next = small;
                p = p->next;
                if (!small->next)
                    q.pop();
                else
                {
                    q.pop();
                    q.push(make_pair(small->next->val, small->next));
                }
            }
            return dummy_head->next;
        }
    };
    ```

1. 用类似归并排序的思想

### 反转链表 II

给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
 

示例 1：


输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]
示例 2：

输入：head = [5], left = 1, right = 1
输出：[5]
 

提示：

链表中节点数目为 n
1 <= n <= 500
-500 <= Node.val <= 500
1 <= left <= right <= n
 

进阶： 你可以使用一趟扫描完成反转吗？

代码：

1. 先找到头尾，然后反转头尾中间的链表，最后处理下头尾的`next`就好了

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* reverseBetween(ListNode* head, int left, int right) {
            if (!head->next) return head;
            ListNode *p = head;
            ListNode *dummy_head = new ListNode(-1, head);
            ListNode *pre = dummy_head;
            --left;
            --right;
            while (left)
            {
                pre = pre->next;
                p = p->next;
                --left;
                --right;
            }
            ListNode *start_pre = pre;
            ListNode *start = p;
            ListNode *nex = start->next;

            while (right)
            {
                p = p->next;
                --right;
            }
            ListNode *end = p;
            
            pre = start_pre;
            p = start;
            while (pre != end)
            {
                p->next = pre;
                pre = p;
                p = nex;
                if (nex) nex = nex->next;
            }

            start_pre->next = pre;
            start->next = p;
            return dummy_head->next;
        }
    };
    ```

    上面的代码在找`end`的时候，多遍历了一遍。这段可以优化掉。

1. 优化方法一：边向后遍历，边判断是否到`end`

1. 优化方法二（官方优化）：每遍历一个节点，就在`start`处做一次插入操作。

    ```cpp
    class Solution {
    public:
        ListNode *reverseBetween(ListNode *head, int left, int right) {
            // 设置 dummyNode 是这一类问题的一般做法
            ListNode *dummyNode = new ListNode(-1);
            dummyNode->next = head;
            ListNode *pre = dummyNode;
            for (int i = 0; i < left - 1; i++) {
                pre = pre->next;
            }
            ListNode *cur = pre->next;
            ListNode *next;
            for (int i = 0; i < right - left; i++) {
                next = cur->next;
                cur->next = next->next;
                next->next = pre->next;
                pre->next = next;
            }
            return dummyNode->next;
        }
    };
    ```

    充分利用了链表快速插入的特性。

1. 其他方法，用栈

    栈能做，但是效率低。

### 旋转链表

给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。

 

示例 1：


输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]
示例 2：


输入：head = [0,1,2], k = 4
输出：[2,0,1]
 

提示：

链表中节点的数目在范围 [0, 500] 内
-100 <= Node.val <= 100
0 <= k <= 2 * 109

代码：

1. 自己写的，边界条件不好判断

    ```cpp
    /**
     * Definition for singly-linked list.
     * struct ListNode {
     *     int val;
     *     ListNode *next;
     *     ListNode() : val(0), next(nullptr) {}
     *     ListNode(int x) : val(x), next(nullptr) {}
     *     ListNode(int x, ListNode *next) : val(x), next(next) {}
     * };
     */
    class Solution {
    public:
        ListNode* rotateRight(ListNode* head, int k) {
            if (!head) return nullptr;
            if (!head->next) return head;
            int n = 0;
            ListNode *p = head;
            ListNode *end;
            while (p)
            {
                ++n;
                if (!p->next) end = p;
                p = p->next;
            }
            int m = n - k % n - 1;
            if (m == -1 || m == n - 1) return head;
            p = head;
            while (m)
            {
                p = p->next;
                --m;
            }
            ListNode *new_head = p->next;
            p->next = nullptr;
            end->next = head;
            return new_head;
        }
    };
    ```

1. 官方答案，先闭合为环，再断开

    ```cpp
    class Solution {
    public:
        ListNode* rotateRight(ListNode* head, int k) {
            if (k == 0 || head == nullptr || head->next == nullptr) {
                return head;
            }
            int n = 1;
            ListNode* iter = head;
            while (iter->next != nullptr) {
                iter = iter->next;
                n++;
            }
            int add = n - k % n;
            if (add == n) {
                return head;
            }
            iter->next = head;
            while (add--) {
                iter = iter->next;
            }
            ListNode* ret = iter->next;
            iter->next = nullptr;
            return ret;
        }
    };
    ```

    其实和我的思路差不多。如果用我的方法，需要判断边界条件；官方的答案不需要处理特殊情况。

1. 快慢指针

    让一个指针比另一个指针快`k`步，这样当快指针到链表尾时，慢指针就会在要断开的位置。

    这种方法也挺好的，比较优雅。

### 两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

 

示例 1：

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

示例 2：

输入：l1 = [0], l2 = [0]
输出：[0]

示例 3：

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]

 

提示：

    每个链表中的节点数在范围 [1, 100] 内
    0 <= Node.val <= 9
    题目数据保证列表表示的数字不含前导零

#### 解答

* 官方答案: 原地修改

    由于输入的两个链表都是逆序存储数字的位数的，因此两个链表中同一位置的数字可以直接相加。

    如果两个链表的长度不同，则可以认为长度短的链表的后面有若干个 0 。

    此外，如果链表遍历结束后，有 carry>0，还需要在答案链表的后面附加一个节点，节点的值为 carry。

    ```cpp
    class Solution {
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            ListNode *head = nullptr, *tail = nullptr;
            int carry = 0;
            while (l1 || l2) {
                int n1 = l1 ? l1->val: 0;
                int n2 = l2 ? l2->val: 0;
                int sum = n1 + n2 + carry;
                if (!head) {
                    head = tail = new ListNode(sum % 10);
                } else {
                    tail->next = new ListNode(sum % 10);
                    tail = tail->next;
                }
                carry = sum / 10;
                if (l1) {
                    l1 = l1->next;
                }
                if (l2) {
                    l2 = l2->next;
                }
            }
            if (carry > 0) {
                tail->next = new ListNode(carry);
            }
            return head;
        }
    };
    ```

    复杂度分析

    * 时间复杂度：O(max(m,n))，其中 m 和 n 分别为两个链表的长度。我们要遍历两个链表的全部位置，而处理每个位置只需要 O(1) 的时间。

    * 空间复杂度：O(1)。注意返回值不计入空间复杂度。

#### AI 分析

* 分析这段代码有什么问题：

    ```cpp
    /**
    * Definition for singly-linked list.
    * struct ListNode {
    *     int val;
    *     ListNode *next;
    *     ListNode() : val(0), next(nullptr) {}
    *     ListNode(int x) : val(x), next(nullptr) {}
    *     ListNode(int x, ListNode *next) : val(x), next(next) {}
    * };
    */
    class Solution {
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            ListNode *p1 = l1, *p2 = l2, *pre;
            int sum, carry = 0;
            while (p1 && p2) {
                sum = p1->val + p2->val + carry;
                p1->val = sum % 10;
                carry = sum / 10;
                
                p1 = p1->next;
                p2 = p2->next;
            }
            while (p1 || p2) {
                sum = (p1 ? p1->val : 0) + (p2 ? p2->val : 0) + carry;
                p1->val = sum % 10;
                carry = sum / 10;
                pre = p1;
                p1 = p1 ? p1->next : p1;
                p2 = p2 ? p2->next : p2;
            }
            if (carry) {
                pre->next = new ListNode;
                p1 = pre->next;
                p1->val = 1;
            }
            return l1;
        }
    };
    ```

    这段代码的核心思路是尝试在原链表 `l1` 上直接进行修改以节省空间，但代码在处理**长度不一致**和**进位逻辑**时存在几个致命的逻辑错误。

    以下是具体的问题分析：

    ### 1. 内存访问越界（最严重的问题）

    在第二个 `while (p1 || p2)` 循环中，代码逻辑非常危险：

    * 如果 `p1` 已经为空（即 `l2` 比 `l1` 长），代码依然尝试执行 `p1->val = sum % 10`。
    * 由于 `p1` 是 `nullptr`，这会直接导致**程序崩溃（Segmentation Fault）**。

    ### 2. 指针丢失与断裂

    * **无法处理 `l2` 更长的情况**：目前的实现始终返回 `l1`。如果 `l2` 长度大于 `l1`，你必须把 `l2` 剩余的部分连接到 `l1` 的末尾，或者创建一个新链表。当前代码只是在修改 `p1` 的值，并没有在 `p1` 结束时将 `p2` 的后续节点接过来。
    * **`pre` 指针的定位模糊**：`pre` 应该是用来记录当前处理到的最后一个有效节点，以便在最后产生进位（carry）时挂载新的 `ListNode`。但在 `p1` 为空的情况下，`pre` 的赋值和使用逻辑是混乱的。

    ### 3. 逻辑冗余

    * 第一个 `while (p1 && p2)` 和第二个 `while (p1 || p2)` 逻辑重叠。通常只需要一个循环处理 `p1 || p2 || carry` 即可使代码更简洁。

    ---

    ### 改进建议

    最稳妥且清晰的做法是**创建一个新的链表**。如果你一定要在原链表上修改，需要处理极其复杂的指针转换。

    这里提供一个**标准且优雅**的解法：

    ```cpp
    class Solution {
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            ListNode dummy(0); // 哨兵节点，简化头节点处理
            ListNode* curr = &dummy;
            int carry = 0;

            while (l1 || l2 || carry) {
                // 1. 计算当前位的和
                int val1 = l1 ? l1->val : 0;
                int val2 = l2 ? l2->val : 0;
                int sum = val1 + val2 + carry;

                // 2. 更新进位和当前节点值
                carry = sum / 10;
                curr->next = new ListNode(sum % 10);
                
                // 3. 移动指针
                curr = curr->next;
                if (l1) l1 = l1->next;
                if (l2) l2 = l2->next;
            }

            return dummy.next;
        }
    };

    ```

    ### 修正后的亮点：

    * **哨兵节点 (Dummy Node)**：避免了判断头节点是否为空的麻烦。
    * **单一循环**：将 `l1`、`l2` 和 `carry` 全部放在循环条件中，一次性处理所有情况（包括最后一位产生的进位）。
    * **鲁棒性**：完美处理两个链表长度不等的情况，不会出现空指针引用。

    需要我为你演示如何通过原地修改（In-place）的方式来优化这段代码的内存利用率吗？

* 好啊，演示一下原地修改。我顺便想看看如果原地修改，该如何处理长度不一致的情况

    原地修改（In-place）虽然能节省空间复杂度（从 $O(\max(m, n))$ 降到 $O(1)$，不计返回结果），但代码逻辑会复杂很多。

    核心挑战在于：**当 `l2` 比 `l1` 长时，你必须把 `l2` 剩余的节点“嫁接”到 `l1` 的末尾**，同时还要处理进位可能导致的链表延长。

    ---

    ### 原地修改逻辑实现（C++）

    为了安全地原地修改，我们需要一个 `pre` 指针始终指向当前处理节点的前一个节点，以便在需要“接管” `l2` 或追加进位节点时能操作 `next` 指针。

    ```cpp
    class Solution {
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            ListNode *p1 = l1, *p2 = l2;
            ListNode *pre = nullptr; // 记录 p1 的前驱节点
            int carry = 0;

            // 1. 遍历两个链表，直到其中一个结束
            while (p1 && p2) {
                int sum = p1->val + p2->val + carry;
                p1->val = sum % 10;
                carry = sum / 10;
                pre = p1;
                p1 = p1->next;
                p2 = p2->next;
            }

            // 2. 如果 l2 比 l1 长，把 l2 剩余部分接到 l1 后面
            if (p2) {
                pre->next = p2;
                p1 = p2; // 接下来只需处理新的 p1（即原 l2 的剩余部分）
            }

            // 3. 处理剩余的节点（可能是原 l1 的，也可能是刚接过来的 l2 的）
            while (p1 && carry) {
                int sum = p1->val + carry;
                p1->val = sum % 10;
                carry = sum / 10;
                pre = p1;
                p1 = p1->next;
            }

            // 4. 如果最后还有进位，追加一个新节点
            if (carry) {
                pre->next = new ListNode(carry);
            }

            return l1;
        }
    };

    ```

    ---

    ### 关键点解析

    #### 1. 长度不一致的处理

    在第一个循环结束后，会出现三种情况：

    * **`p1` 和 `p2` 同时为空**：两个数字一样长，直接跳到步骤 4 检查进位。
    * **`p1` 不为空**：说明 `l1` 更长，我们继续在 `p1` 上累加进位。
    * **`p2` 不为空**：说明 `l2` 更长。这是最关键的一步：`pre->next = p2;`。通过这一行，我们将 `l2` 的残余部分直接“缝合”到了 `l1` 的骨架上。

    #### 2. 进位的“连锁反应”

    即使 `l2` 结束了，进位（carry）可能会一直向高位传递。例如 `[9, 9] + [1]`：

    * 第一位：`9+1=10`，进位 1。
    * 第二位：`9+1=10`，进位 1。
    * 此时 `p1` 已空，必须再 `new` 一个节点存储最后的 `1`。

    #### 3. 为什么需要 `pre`？

    在单向链表中，一旦指针走到了 `nullptr`，你就无法回头去修改前一个节点的 `next` 指向了。所以我们用 `pre` 盯着当前节点的“屁股”，确保随时可以挂载新节点。

    ---

    ### 复杂度对比

    * **时间复杂度**：$O(\max(m, n))$，依然需要遍历一次。
    * **空间复杂度**：$O(1)$。我们没有创建一整条新链表，只在最高位产生进位时可能 `new` 一个节点。

    **这种做法虽然省空间，但在工程实践中要小心：它破坏了原始输入 `l1` 和 `l2` 的结构。** 如果其他地方还需要用到原始数据，这种方法就不适用了。

    你想看看如何用递归的方式来优雅地处理这个问题吗？虽然它不省空间，但代码非常漂亮。

* 还有递归解法？给我演示一下

    递归解法在处理链表问题时非常“优雅”，因为它本质上是利用**函数调用栈**来代替手动管理指针和循环。

    在递归逻辑中，你可以把每一位加法看作一个独立的小任务：**“计算当前位的和，然后把进位传给下一位的加法”。**

    ---

    ### 递归代码实现（C++）

    ```cpp
    class Solution {
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            return addHelper(l1, l2, 0);
        }

    private:
        // 递归辅助函数：处理当前节点和进位
        ListNode* addHelper(ListNode* l1, ListNode* l2, int carry) {
            // 递归出口：如果两个链表都空了，且没有多余进位，直接返回 nullptr
            if (!l1 && !l2 && carry == 0) {
                return nullptr;
            }

            // 1. 计算当前位的数值之和
            int val1 = l1 ? l1->val : 0;
            int val2 = l2 ? l2->val : 0;
            int sum = val1 + val2 + carry;

            // 2. 创建当前节点
            ListNode* node = new ListNode(sum % 10);

            // 3. 递归计算下一位（处理指针移动和进位更新）
            // 这里的 next 指向的是“下一位加法”返回的结果
            node->next = addHelper(
                l1 ? l1->next : nullptr, 
                l2 ? l2->next : nullptr, 
                sum / 10
            );

            return node;
        }
    };

    ```

    ---

    ### 递归逻辑的精妙之处

    #### 1. 天然处理长度不一致

    在递归调用中，我们用 `l1 ? l1->next : nullptr` 这种写法。如果 `l1` 已经到头了，它会一直给下一层传递 `nullptr`。由于我们在计算 `sum` 时有 `val1 = l1 ? l1->val : 0` 的保护，即使链表长度不同，逻辑也会自动收拢，不需要像迭代法那样写好几个 `while` 循环。

    #### 2. 最后的进位处理

    注意到递归出口条件 `if (!l1 && !l2 && carry == 0)` 吗？
    如果 `l1` 和 `l2` 都走完了，但 `carry` 还有值（比如 $99 + 1$ 的最后一次进位），递归会再跑一轮，创建一个值为 `1` 的节点，然后下一轮才真正触发 `nullptr` 返回。

    ---

    ### 优缺点分析

    | 特点 | 递归法 | 迭代法 (原地/新建) |
    | --- | --- | --- |
    | **代码量** | 极简、逻辑清晰 | 较多指针操作 |
    | **可读性** | 高（像读数学公式） | 中（需要跟踪指针指向） |
    | **空间复杂度** | $O(\max(m, n))$ (由于栈深度) | $O(1)$ 或 $O(\max(m, n))$ |
    | **堆栈风险** | 如果链表极长（如几万个节点），可能导致**栈溢出** | 无风险 |

    ### 温馨提示

    虽然递归很酷，但在 LeetCode 或实际面试中，**面试官通常更倾向于考察迭代法**（尤其是你刚才尝试的原地修改），因为迭代法对内存的控制更精准，不会因为数据规模过大而崩掉栈空间。

    你觉得这种“层层套娃”的逻辑比起你最初写的 `while` 循环，理解起来更直观吗？

* 如果我能遍历一下这两个链表，拿到谁长谁短，这个问题就好办了

    你说得非常对！如果你先通过一次遍历确定了**“谁才是那个长链表”**，逻辑确实会清晰很多。这就好比在做手术前先量好了伤口长度，后面缝合时心里就有底了。

    这种思路通常被称为 **“预判法”** 或 **“定主链法”**。

    ---

    ### 基于“预判长度”的原地修改思路

    我们可以先写一个简单的 `getLength` 函数，或者在第一次遍历时直接对比。确定 `l1` 为较长链表后，我们所有的修改都直接发生在 `l1` 上。

    ```cpp
    class Solution {
    public:
        // 获取链表长度
        int getLength(ListNode* head) {
            int len = 0;
            while (head) { len++; head = head->next; }
            return len;
        }

        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            int len1 = getLength(l1);
            int len2 = getLength(l2);

            // 始终让 l1 指向较长的那个链表，简化后续逻辑
            if (len1 < len2) swap(l1, l2);

            ListNode* curr1 = l1;
            ListNode* curr2 = l2;
            ListNode* pre = nullptr; // 用于处理最后的进位挂载
            int carry = 0;

            // 现在我们确定 l1 至少和 l2 一样长
            while (curr1) {
                int val2 = curr2 ? curr2->val : 0; // l2 可能提前结束
                int sum = curr1->val + val2 + carry;
                
                curr1->val = sum % 10;
                carry = sum / 10;
                
                pre = curr1;
                curr1 = curr1->next;
                if (curr2) curr2 = curr2->next;
            }

            // 如果最后还有进位，因为 l1 已经遍历完了，直接在 pre 后面补一个节点
            if (carry) {
                pre->next = new ListNode(carry);
            }

            return l1;
        }
    };

    ```

    ---

    ### 为什么这样做“更好办”了？

    1. **消除分支恐惧**：你不需要在循环里判断 `if (p2) { pre->next = p2; }` 这种复杂的缝合逻辑了。因为你已经确保了 `l1` 是那个“大容器”。
    2. **结构统一**：主循环只需要跑一遍 `while (curr1)`。即使 `l2` 比较短，我们也把它当做高位全是 `0` 的数字来处理。
    3. **代码复用性**：这种“长短互换”的思想在很多链表题目（比如“相交链表”）里都是神技。

    ### 这种做法的“代价”

    虽然逻辑简单了，但你**多遍历了一次链表**来获取长度（时间复杂度依然是 $O(N)$，只是常数项变成了 $2N$）。在面试中，如果你能主动说出：“我先遍历一遍确定长度以简化逻辑，虽然多了一次遍历，但代码可维护性更高”，面试官通常会觉得你很有工程思维。

    **一个小挑战：**
    如果我不让你调用 `getLength` 函数，你能在**第一次**同时遍历 `l1` 和 `l2` 的过程中，就动态地完成这个“原地修改”吗？（其实就是我们上一个回复里讨论的那种复杂的 `pre->next = p2` 逻辑）。

    你更喜欢这种“先看长度再动手”的稳健派，还是“边走边修”的激进派？
