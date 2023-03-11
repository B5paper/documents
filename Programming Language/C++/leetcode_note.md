# LeetCode Note

Same type of structures and topics may emerge at different diffuculity levels of problems.

## Basic problems

Problems that just have one type of method or solution.

### Double pointers （双指针）

#### 反转字符串

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

 

示例 1：

```
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]
```

代码：

1. 对撞双指针

    ```c++
    class Solution {
    public:
        void reverseString(vector<char>& s) {
            int left = 0, right = s.size() - 1;
            while (left < right)
            {
                swap(s[left], s[right]);
                ++left;
                --right;
            }
        }
    };
    ```

    （相当于标准库里的`reverse()`函数了）

#### 替换空格

请实现一个函数，把字符串中的每个空格替换成"%20"。

你可以假定输入字符串的长度最大是 1000。

注意输出字符串的长度可能大于 1000。

样例：

输入：`"We are happy."`

输出：`"We%20are%20happy."`

**分析**：

简单的方法是创建一个新的字符串做替换，但是这样会消耗额外的空间。为了不消耗额外的空间，可以对原字符串 resize 后，用双指针原地修改。

代码：

1. 用新字符串做替换

    ```c++
    class Solution {
    public:
        string replaceSpaces(string &str) {
            int pos = 0;
            string new_str;
            while (pos != str.size())
            {
                if (str[pos] == ' ')
                    new_str.append("%20");
                else
                    new_str.push_back(str[pos]);
                ++pos;
            }
            return new_str;
        }
    };
    ```

1. 倒序双指针

    ```c++
    class Solution {
    public:
        string replaceSpace(string s) {
            int space_count = 0;
            for (int i = 0; i < s.size(); ++i)
            {
                if (s[i] == ' ') ++space_count;
            }

            int len = s.size();
            s.resize(len - space_count + space_count * 3);
            int left = len - 1, right = s.size() - 1;
            while (left > -1)
            {
                if (s[left] == ' ')
                {
                    s[right--] = '0';
                    s[right--] = '2';
                    s[right--] = '%';
                    --left;
                }
                else
                {
                    s[right--] = s[left--];
                }
            }
            return s;
        }
    };
    ```

    倒序双指针可以做原地修改。

#### 合并两个有序数组

tag: 数组，双指针

给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。

 
```
示例 1：

输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
示例 2：

输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]
```

代码：

逆向双指针。

```c++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int pos1 = m - 1, pos2 = n - 1;
        int pos = nums1.size() - 1;
        while (pos1 > -1 && pos2 > -1)
        {
            nums1[pos] = max(nums1[pos1], nums2[pos2]);
            if (nums1[pos1] > nums2[pos2]) --pos1;
            else if (nums1[pos1] < nums2[pos2]) --pos2;
            else --pos2;
            --pos;
        }

        if (pos1 < 0)
        {
            while (pos2 > -1) nums1[pos--] = nums2[pos2--];
        }
        else if (pos2 < 0)
        {
            return;
        }
    }
};
```

#### 移动零

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例:

```
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
```

说明:

必须在原数组上操作，不能拷贝额外的数组。

尽量减少操作次数。

代码：

1. 快慢指针。很简单。

    ```c++
    class Solution {
    public:
        void moveZeroes(vector<int>& nums) {
            int slow = 0, fast = 0;
            while (fast < nums.size())
            {
                if (nums[fast] != 0)
                    nums[slow++] = nums[fast];
                ++fast;
            }
            while (slow < nums.size())
                nums[slow++] = 0;
        }
    };
    ```

1. 后来自己写的`swap()`版本，效率低了好多：

    ```c++
    class Solution {
    public:
        void moveZeroes(vector<int>& nums) {
            int left = 0, right = 0;
            while (right < nums.size())
            {
                if (nums[right] == 0) ++right;
                else
                {
                    swap(nums[left], nums[right]);
                    ++right;
                    ++left;
                }
            }
        }
    };
    ```

### Prefix sum （前缀和）

（常见的前缀和都是和哈希表配合使用的，因此实际上只使用前缀和的并不多）

### Dynamic programming （动态规划）

#### 斐波那契数列

写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项（即 `F(N)`）。斐波那契数列的定义如下：

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

答案需要取模 `1e9+7`（1000000007），如计算初始结果为：1000000008，请返回 1。

分析：采取朴素的递归法，复杂度呈指数增加，因此可以用自顶向下的动态规划，也可以采用自底向上的动态规划。

* 朴素递归法

  ```c++
  class Solution {
  public:
      int fib(int n) {
          if (n == 0)
              return 0;
          if (n == 1)
              return 1;
          return fib(n-1) + fib(n-2);
      }
  };
  ```

* 自底向上法

  ```c++
  class Solution {
  public:
      int fib(int n) {
          if (n == 0)
              return 0;
          if (n == 1)
              return 1;
          int prevs[2] = {0, 1};
          int rtn;
          for (int i = 2; i <= n; ++i)
          {
              rtn = (prevs[0] % 1000000007 + prevs[1] % 1000000007) % 1000000007;
              prevs[0] = prevs[1];
              prevs[1] = rtn;
          }
          return rtn;
      }
  };
  ```

* 自顶向下法

  * 散列表版

    ```c++
    class Solution {
    public:
        unordered_map<int, int> fibs;
        int recu(int n)
        {
            if (fibs.find(n) != fibs.end())
                return fibs[n];
            else
            {
                int fib_n = (recu(n-1) % 1000000007 + recu(n-2) % 1000000007) % 1000000007;
                fibs[n] = fib_n;
                return fib_n;
            }
        }
    
        int fib(int n) {
            fibs[0] = 0;
            fibs[1] = 1;       
            return recu(n);
        }
    };
    ```

  * 数组版

    ```c++
    class Solution {
    public:
        vector<int> fibs;
        int recu(int n)
        {
            if (fibs[n] != -1)
                return fibs[n];
            else
            {
                int fib_n = (recu(n-1) % 1000000007 + recu(n-2) % 1000000007) % 1000000007;
                fibs[n] = fib_n;
                return fib_n;
            }
        }
    
        int fib(int n) {
            fibs.assign(101, -1);
            fibs[0] = 0;
            fibs[1] = 1;
            return recu(n);
        }
    };
    ```

其中，三种方法用时都是差不多的，自底向上法占用的内存最少，自顶向下法的数组版占用内存第二少，自顶向下法的散列表版占用的内存最多。

#### 爬楼梯

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 n 是一个正整数。

示例 1：

输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
示例 2：

输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶

代码：

1. 动态规划

    ```c++
    class Solution {
    public:
        int climbStairs(int n) {
            vector<int> dp(n+1);
            dp[0] = 1;
            dp[1] = 1;
            for (int i = 2; i <= n; ++i)
                dp[i] = dp[i-1] + dp[i-2];
            return dp[n];
        }
    };
    ```

    这是一道动态规划的经典题目。假设我们要爬第`m`级楼梯，`3 <= m <= n`，有几种方法可以到第`m`级楼梯呢？我们可以从第`m - 1`级楼梯跨一级台阶上去，也可以从第`m - 2`级楼梯跨两级台阶上去。因此我们只需要知道上到第`m - 1`级楼梯有几种方法，到第`m - 2`级楼梯有几种方法，然后将这两种方法数求和就可以了。从第 3 级楼梯开始，每一级楼梯的上法，都可以由前两级楼梯的数据计算出来。我们对前两级楼梯特殊处理，最终可以得到这样的代码：

    ```cpp
    class Solution {
    public:
        int climbStairs(int n) {
            // 前两级楼梯特别处理
            if (n == 1) return 1;
            if (n == 2) return 2;

            // 初始化状态
            int n_1 = 1;
            int n_2 = 2;
            int n_3 = 0;

            // 从第 3 级楼梯开始计算
            for (int i = 3; i <= n; ++i)  // 注意这里的 i 表示的不再是索引，而是楼梯的级数，因此 i 从 3 开始，并且可以取到 n
            {
                n_3 = n_1 + n_2;
                n_1 = n_2;
                n_2 = n_3;
            }
            return n_3;
        }
    };
    ```

#### 使用最小花费爬楼梯

数组的每个下标作为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。

每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。

请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。

 
```
示例 1：

输入：cost = [10, 15, 20]
输出：15
解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费 15 。
 示例 2：

输入：cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出：6
解释：最低花费方式是从 cost[0] 开始，逐个经过那些 1 ，跳过 cost[3] ，一共花费 6 。
```

代码：

1. 自己写的

    ```c++
    class Solution {
    public:
        int minCostClimbingStairs(vector<int>& cost) {
            vector<int> dp(cost.size());
            dp[0] = cost[0];
            dp[1] = cost[1];
            for (int i = 2; i < cost.size(); ++i)
            {
                dp[i] = cost[i] + min(dp[i-1], dp[i-2]);
            }
            return min(dp[cost.size()-1], dp[cost.size()-2]);
        }
    };
    ```

1. 官方给的

    ```c++
    class Solution {
    public:
        int minCostClimbingStairs(vector<int>& cost) {
            int n = cost.size();
            vector<int> dp(n + 1);
            dp[0] = dp[1] = 0;
            for (int i = 2; i <= n; i++) {
                dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
            }
            return dp[n];
        }
    };
    ```

    我觉得官方给的代码并不好，不容易理解和阅读。直接看我写的就行了。

    这道题比较烦人的地方是，最后一个台阶并不在`cost.size()-1`的位置，而是在再上一层的位置。

    现在再来分析最后一个台阶的问题，我们只需要把它作为特殊情况处理就可以了。

#### 打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

示例 1：

> 输入：`[1,2,3,1]`
>
> 输出：`4`
>
> 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
>      偷窃到的最高金额 = 1 + 3 = 4 。

示例 2：

> 输入：`[2,7,9,3,1]`
> 输出：12
> 解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
>      偷窃到的最高金额 = 2 + 9 + 1 = 12 。

1. 自底向上，用数组存储最优结果

    ```c++
    class Solution {
    public:
        int rob(vector<int>& nums) {
            vector<int> dp(nums.size() + 1);
            dp[0] = 0;
            dp[1] = nums[0];
            for (int i = 2; i <= nums.size(); ++i)
                dp[i] = max(dp[i-1], dp[i-2] + nums[i-1]);  // 这里的 nums[i-1] 指的是第 i 家的金额
            return dp[nums.size()];
        }
    };
    ```

2. 自底向上，只保留前两个结果

	```c++
	class Solution {
    public:
        int rob(vector<int>& nums) {
            if (nums.size() == 1)
                return nums[0];
            if (nums.size() == 2)
                return max(nums[0], nums[1]);
	
            int q_im2 = nums[0];
            int q_im1 = max(nums[0], nums[1]);
            int max_val;
            for (int i = 2; i < nums.size(); ++i)
            {
                max_val = max(nums[i] + q_im2, q_im1);
                q_im2 = q_im1;
                q_im1 = max_val;
            }
            return max_val;
        }
    };
	```

### Traversal of a tree （树的遍历）

#### Pre-order traversal （先序遍历）

##### 二叉搜索树中的搜索

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

```
例如，

给定二叉搜索树:

        4
       / \
      2   7
     / \
    1   3

和值: 2
你应该返回如下子树:

      2     
     / \   
    1   3
在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
```

代码：

1. 先序遍历

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        int val;
        TreeNode *ans;

        void dfs(TreeNode *root)
        {
            if (ans) return;  // 剪枝
            if (!root) return;
            if (root->val == val)
            {
                ans = root;
                return;
            }
            if (root->val > val) dfs(root->left);
            else dfs(root->right);
        }

        TreeNode* searchBST(TreeNode* root, int val) {
            this->val = val;
            ans = nullptr;
            dfs(root);
            return ans;
        }
    };
    ```

    二叉搜索树似乎只能使用先序遍历。只有先比较了当前节点的值与目标值的大小后，才能判断是搜索左子树还是右子树。

1. 递归

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        TreeNode* searchBST(TreeNode* root, int val) {
            if (!root) return nullptr;
            if (val == root->val) return root;
            if (val < root->val) return searchBST(root->left, val);
            else return searchBST(root->right, val);
        }
    };
    ```

1. 迭代

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        TreeNode* searchBST(TreeNode* root, int val) {
            while (root)
            {
                if (val == root->val) return root;
                if (val < root->val) root = root->left;
                else root = root->right;
            }
            return nullptr;
        }
    };
    ```

#### Mid-order traversal （中序遍历）

#### Post-order traversal （后序遍历）

#### Level traversal （层序遍历）

#### Backtrack （回溯）

##### 全排列

> 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

代码：

1. 标准回溯法

    ```c++
    class Solution {
    public:
        vector<vector<int>> ans;
        vector<int> temp;
        vector<bool> vis;

        void backtrack(vector<int> &nums)
        {
            if (temp.size() == nums.size())
            {
                ans.push_back(temp);
                return;
            }

            for (int i = 0; i < nums.size(); ++i)
            {
                if (!vis[i])
                {
                    vis[i] = true;
                    temp.push_back(nums[i]);
                    backtrack(nums);
                    temp.pop_back();
                    vis[i] = false;
                }
            }
        }

        vector<vector<int>> permute(vector<int>& nums) {
            vis.assign(nums.size(), false);
            backtrack(nums);
            return ans;
        }
    };
    ```

    在回溯中，我们主要考虑这个问题：假如本层要遍历的是`[a, b, c, d]`，当前节点已经用了`c`，那么当进入下一层时，下一层需要遍历哪些元素？对于这个例子，很明显下一层需要遍历`[a, b, d]`，但是怎么让程序知道这个结果呢？

    答案是用`for` + `vis`。我们只需要记录已经在路径中的元素，然后每次都把所有元素都扫描一遍，然后找到没有用过的元素就可以了。

    显然如果每次都扫描所有元素，效率仍有提升空间。或许可以用一个容器存储剩余的可用元素，每次都从容器中 pop 出来一个元素，然后进入下一层遍历。当子节点遍历结束后，再把这个元素 push 回去。

1. 回潮法，交换位置

    ```c++
    class Solution {
    public:
        vector<vector<int>> permute(vector<int>& nums)
        {
            vector<vector<int>> res;
            dfs(res, nums, 0);
            return res;
        }

        void dfs(vector<vector<int>> &res, vector<int> &nums, int pos)
        {
            if (pos == nums.size())
            {
                res.emplace_back(nums);
                return;
            }

            for (int i = pos; i < nums.size(); ++i)
            {
                swap(nums[i], nums[pos]);  // 我觉得不应该写 swap，这是一种取巧的写法，不能代表回溯法的通用思想
                dfs(res, nums, pos+1);
                swap(nums[i], nums[pos]);
            }
        }
    };
    ```



## Combinational problems

Problems that use two or more simple methods to make a combination, or problems that have several types of solutions to make a trade-off.

## Advanced problems

Problems that use unusual methematics.

## 整数反转

> 给你一个 32 位的有符号整数 `x` ，返回将 `x` 中的数字部分反转后的结果。
>
> 如果反转后整数超过 32 位的有符号整数的范围 `[−2^31, 2^31 − 1]` ，就返回 0。
>
> **假设环境不允许存储 64 位整数（有符号或无符号）。**

分析：

4 字节的 int 类型最大值为`INT32_MAX = 2147483647` ，最小值为`INT32_MIN = -2147483648`。

代码：

    ```c++
    class Solution {
    public:
        int reverse(int x) {
            bool neg = x < 0;
            if (x == INT_MIN) return 0;
            if (neg) x = -x;
            int ans = 0;
            while (x)
            {
                ans += x % 10;
                x /= 10;
                if (neg && ((ans > INT_MAX / 10 && x != 0) || (ans == INT_MAX / 10 && x > 8))) return 0;
                if (!neg && ((ans > INT_MAX / 10 && x != 0) || (ans == INT_MAX / 10 && x > 7))) return 0;
                if (x) ans *= 10;
            }
            return neg ? -ans : ans;
        }
    };
    ```

## 动态规划（dynamic programming）

### 钢链切割问题

> 钢条切割问题：不同长度的钢条有不同的售价，现给定一个长度为`n`的钢条，给定不同长度钢条的售价，求如何切割，使得售价之和最大？
>
> 例如：设钢条长度为`n`，不同长度钢条的售价如下所示：
>
> | 长度$i$ | 1    | 2    | 3    | 4    |  5 | 6 | 7 | 8| 9 | 10 |
> |-|-|-|-|-|-|-|-|-|-|-|
> |价格$p_i$| 1 | 5 | 8  | 9 | 10 | 17 | 17 | 20 | 24 | 30 |

长度为$n$的钢条共有$2^{n-1}$种不同的切割方案，因为在距离钢条左端$i$（$i = 1, 2, \dots, n-1$）英寸处，我们总是可以选择切割或不切割。

设对于某个长为$n$的钢条，最优的切割方案为：$n = i_i + i_2 + \cdots + i_k$（$1 \leq k \leq n$），其中$k$为切割的段数，最大收益为$r_n = p_{i_1} + p_{i_2} + \cdots + p_{i_k}$。

下面是一些测试用例：

| $r_1$ | $r_2$ | $r_3$ | $r_4$ | $r_5$ | $r_6$ | $r_7$ | $r_8$ | $r_9$ | $r_{10}$ |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | -------- |
| 1     | 5     | 8     | 10    | 13    | 17    | 18    | 22    | 25    | 30       |

**分析**：

**思路 1**：

对于切割长度为$n$的钢条的问题，我们总能把它划分成两个子问题：从某处切割，将钢条分成两段，分别计算这两段的最大收益，这两段的最大收益之和就是整个钢条的最大收益：

$$
r_n = \max (p_n, r_1 + r_{n-1}, r_2 + r_{n-2}, \cdots, r_{n-1} + r_1)
$$

其中$p_n$表示不切割钢条。

由这个思路，可以写出代码：

```c++
#include <iostream>
#include <vector>
using namespace std;

int p[] = {1, 5, 8, 9, 10, 17, 17, 20, 24, 30};

int cut_rod_1(int n)
{
    // 返回长度为 n 的钢条的最大利润
    
    if (n == 0)  // 若钢条长度为 0，则利润为 0
        return 0;

    vector<int> temp(n, INT32_MIN);
    temp[0] = p[n-1];  // 不切割时钢条的利润，对应公式中的 p_n
    // 上面一行代码，下标写为 n-1，是因为数组 p 的索引从 0 开始
    for (int i = 1; i < n; ++i)
    {
        temp[i] = cut_rod_1(i) + cut_rod_1(n-i);  // 按照上面分析中的公式编写
    }

    // 找出所有方案的利润最大值，并返回即可
    int max_pro = INT32_MIN;
    for (int i = 0; i < n; ++i)
    {
        if (temp[i] > pro_max)
            max_pro = temp[i];
    }
    return max_pro;
}

int main()
{
    int max_pro = cut_rod_1(9);
    cout << "max pro: " << max_pro << endl;
    return 0;
}
```

其实我们并不需要使用数组`temp`来存储所有的中间数据，因为我们要的只是最大利润值，所以可以对上面代码进行以下优化：

```c++
#include <iostream>
using namespace std;

int p[] = {1, 5, 8, 9, 10, 17, 17, 20, 24, 30};

int cut_rod_1(int n)
{
    if (n == 0)
        return 0;

    int max_pro = p[n-1];
    for (int i = 1; i < n; ++i)
    {
        max_pro = max(max_pro, cut_rod_1(i) + cut_rod_1(n-i));
    }

    return max_pro;
}

int main()
{
    int pro = cut_rod_1(9);
    cout << "max pro: " << pro << endl;
    return 0;
}
```

**最优子结构性质（optimal substructure）**：问题的最优解由相关子问题的最优解组合而成，而这些子问题可以独立求解。或者说，一个问题的最优解包含其子问题的最优解。

**思路 2**：

对于钢条切割问题，除了有上面的划分子问题的思路，还可以按下面的思路划分子问题：将钢条从左边切割下长度为$i$的一段，只对右边剩下的长度为$n-i$的一段继续进行切割。于是我们得到另一个公式：
$$
r_n = \max\limits_{1 \leq i \leq n} (p_i + r_{n-i})
$$
当$i = n$时，表示不对钢条进行切割，收益为$p_n$，而$r_0$ = 0。在这个公式中，原问题的最优解只包含一个相关子问题（右端剩余部分）的解，而不是两个。

按照思路 2，代码实现如下（自顶向下递归实现，朴素递归）：

```c++
#include <iostream>
using namespace std;

int p[] = {1, 5, 8, 9, 10, 17, 17, 20, 24, 30};

int cut_rod_2(int n)
{
    if (n == 0)
        return 0;

    int max_pro = INT32_MIN;
    for (int i = 1; i <= n; ++i)
    {
        // 注意下面这行代码的下标，因为对于数组 p 来説，索引从 0 开始，所以下标需要填 i - 1
        // 而对于 cut_rod_2()，其参数的含义是从从左侧起 n-i 长度处切割，所以不需要减 1
        max_pro = max(max_pro, p[i-1] + cut_rod_2(n-i));
    }
    
    return max_pro;
}

int main()
{
    int pro = cut_rod_2(10);
    cout << "max pro: " << pro << endl;
    return 0;
}
```

这种朴素递归的方法时间复杂度为$O(2^n)$，因为它会反复求解相同的子问题。因此我们可以考虑使用动态规划来进一步优化。

所谓的动态规划，指的是付出额外的内存空间来节省计算时间（将指数时间转换为多项式时间），实现了时空权衡（time-memory trade-off）。

动态规划有两种等价的实现方法：

1. 带备忘的自顶向下法（top-down with memoization）

   仍按自然的递归形式编写过程，但会将每个子问题的解保存在一个数组或散列表中。这个递归过程被称为**带备忘的**（memoized）。

   实现代码如下：

   ```c++
   #include <iostream>
   using namespace std;
   
   int p[] = {1, 5, 8, 9, 10, 17, 17, 20, 24, 30};
   
   int cut_rod_3(int n, int r[])
   {
       // 因为 r 中存储的数据的意义和 cur_rod_3 的返回值相同
       // 所以数组 r 的下标不需要减 1
       if (r[n] >= 0)  // 判断子问题是否被解过
           return r[n];
   
       int max_pro = INT32_MIN;
       for (int i = 1; i <= n; ++i)
       {
           max_pro = max(max_pro, p[i-1] + cut_rod_3(n-i, r));
       }
       
       r[n] = max_pro;
       return max_pro;
   }
   
   int main()
   {
       int n = 10;
       int r[n+1];
       for (int i = 0; i < n+1; ++i)
       {
           r[i] = INT32_MIN;
       }
       r[0] = 0;
       int pro = cut_rod_3(n, r);
       cout << "max pro: " << pro << endl;
       return 0;
   }
   ```

1. 自底向上法（bottom-up method）

   需要定义子问题的规模，当求解某个子问题时，它所依赖的那些更小的子问题都已求解完毕，结果已经保存。这样每个子问题都只需求解一次。

   实现代码如下：

   ```c++
   #include <iostream>
   using namespace std;
   
   int p[] = {1, 5, 8, 9, 10, 17, 17, 20, 24, 30};
   
   int cut_rod_4(int n, int r[])
   {
       for (int i = 1; i <= n; ++i)
       {
           int max_pro = INT32_MIN;
           for (int j = 1; j <= i; ++j)
           {
               max_pro = max(max_pro, p[j-1] + r[i-j]);
           }
           r[i] = max_pro;
       }
       
       return r[n];
   }
   
   int main()
   {
       const int n = 9;
       int r[n+1];
       for (int i = 0; i < n + 1; ++i)
       {
           r[i] = INT32_MIN;
       }
       r[0] = 0;
       int pro = cut_rod_4(n, r);
       cout << "max pro: " << pro << endl;
       return 0;
   }
   ```

   可以看到，对于自底向上法，不需要递归调用函数，因此效率更高一些。

### 矩阵链乘法

> 给定$n$个矩阵的链（$A_1$，$A_2$，$\cdots$，$A_n$），矩阵$A_i$的规模为$p_{i-1} \times p_i (i \leq i \leq n)$，求完全括号化方案，使得计算乘积$A_1 A_2 \cdots A_n$所需标量乘法次数最少。

设$P(n)$表示可供选择的括号化方案的数量。当$n = 1$时，$P(1) = 1$。当$n \geq 2$时，完全括号化的矩阵乘积可描述为两个完全括号化的部分积相乘的形式，而两个部分积的划分点在第$k$个矩阵和第$k+1$个矩阵之间，$k$为$1, 2, \dots, n-1$中的任意一个值。因此，我们可以得到如下递归公式：
$$
P(n) = \begin{cases}
1, &\text{if }n = 1 \\ 
\sum\limits_{k=1}^{n-1}P(k) P(n-k), &\text{if } n \geq 2
\end{cases}
$$
此递归公式产生的序列的增长速度为$\Omega (2^n)$。因此通过暴力搜索穷尽所有可能的括号化方案来寻找最优方案是不可行的。

对于 6 个矩阵相乘，数组`p`存储了每个矩阵的尺寸，比如`p[0] = 30`，`p[1] = 35`表示第一个矩阵的尺寸为$30 \times 35$。

假如我们用数组`int m[n][n]`来保存不同起始位置和结束位置的最小代价，比如`m[0][3]`表示 0 号矩阵到 3 号矩阵相乘的最小代价，用`int s[n][n]`表示不同起始位置和结束位置的最优子问题的分割位置，比如`s[2][5] = 3`表示对于 2 号矩阵到 5 号矩阵，在 3 号矩阵和 4 和矩阵中间分割，那么就可以用动态规划来求解这个问题：

1. 自底向上版本

   ```c++
   #include <iostream>
   #include <vector>
   using namespace std;
   
   int matrix_chain_order(int n, int p[], vector<vector<int>> &m, vector<vector<int>> &s)
   {
       for (int i = 0; i < n; ++i)
       {
           m[i][i] = 0;
       }
   
       int subchain_end;
       for (int l = 2; l <= n; ++l)  // l is the chain length
       {
           subchain_end = n-l+1;
           for (int i = 0; i < subchain_end; ++i)
           {
               int j = i + l - 1;
               m[i][j] = INT32_MAX;
               for (int k = i; k < j; ++k)
               {
                   int q = m[i][k] + m[k+1][j] + p[i] * p[k+1] * p[j+1];
                   if (q < m[i][j])
                   {
                       m[i][j] = q;
                       s[i][j] = k;
                   }
               }
           }
       }
       return m[0][n-1];
   }
   
   void print_optimal_parens(vector<vector<int>> &s, int i, int j)
   {
       if (i == j)
       {
           cout << "A" << i;
       }
       else
       {
           cout << "(";
           print_optimal_parens(s, i, s[i][j]);
           print_optimal_parens(s, s[i][j]+1, j);
           cout << ")";
       }
   }
   
   int main()
   {
       const int n = 6;
       int p[n+1] = {30, 35, 15, 5, 10, 20, 25};
       vector<vector<int>> m(n, vector<int>(n, 0));
       vector<vector<int>> s(n, vector<int>(n, 0));
   
       int res = matrix_chain_order(n, p, m, s);
       cout << res << endl;
       print_optimal_parens(s, 0, 5);
       return 0;
   }
   ```

寻找最优子结构：假如有一种对$A_i A_{i+1} \cdots A_k$括号化的方案不是$A_i A_{i+1} \cdots A_k$的最优解，但能使$A_i A_{i+1} \cdots A_j$的代价更低，那么就会和$A_i A_{i+1} \cdots A_k$的代价 + $A_{k+1} A_{k+2} \cdots A_j$的代价 = $A_i A_{i+1} \cdots A_j$的代价相矛盾。【为什么？因为计数原理】

因此在$A_i A_{i+1} \cdots A_j$的最优括号化方案中，对子链$A_i A_{i+1} \cdots A_k$进行括号化的方法就是它自身的最优括号化方案。

两个子链的最优括号化方案的组合不一定是原链的最优方案，但原链的最优方案一定是两个子链的最优方案的组合。原链的最优方案一定可以在两个子链的所有组合的可能性中找到。

2. 自顶向下版本

### 补充练习

#### 删除并获得点数

> 给你一个整数数组`nums` ，你可以对它进行一些操作。
>
> 每次操作中，选择任意一个`nums[i]`，删除它并获得`nums[i]`的点数。之后，你必须删除每个等于`nums[i] - 1`或`nums[i] + 1`的元素。
>
> 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
>
> 示例 1：
>
> 输入：`nums = [3,4,2]`
> 输出：6
> 解释：
> 删除 4 获得 4 个点数，因此 3 也被删除。
> 之后，删除 2 获得 2 个点数。总共获得 6 个点数。
>
> 示例 2：
>
> 输入：`nums = [2,2,3,3,3,4]`
> 输出：9
> 解释：
> 删除 3 获得 3 个点数，接着要删除两个 2 和 4 。
> 之后，再次删除 3 获得 3 个点数，再次删除 3 获得 3 个点数。
> 总共获得 9 个点数。

* 朴素递归（超出时间限制）

  ```c++
  class Solution {
  public:
      int deleteAndEarn(vector<int>& nums) {
          if (nums.empty())
              return 0;
          if (nums.size() == 1)
              return nums[0];
              
          int q = INT32_MIN;
          
          for (int i = 0; i < nums.size(); ++i)
          {
              vector<int> nums_copy(nums);
              int point = nums_copy[i];
  
              auto iter = nums_copy.begin();
              while (iter != nums_copy.end())
              {
                  
                  if (*iter == point - 1 || *iter == point + 1)
                  {
                      iter = nums_copy.erase(iter);
                  }
                  else
                  {
                      ++iter;
                  }
              }
  
              iter = nums_copy.begin();
              while (iter != nums_copy.end())
              {
                  if (*iter == point)
                  {
                      nums_copy.erase(iter);
                      break;
                  }
                  else
                      ++iter;
              }
  
              q = max(point + deleteAndEarn(nums_copy), q);
          }
  
          return q;
      }
  };
  ```

* 带备忘的自顶向下（超出时间限制）

  ```c++
  class Solution {
  public:
      map<vector<int>, int> m;
      int deleteAndEarn(vector<int>& nums) {
          if (nums.empty())
              return 0;
          if (nums.size() == 1)
              return nums[0];
  
          auto iter = m.find(nums);
          if (iter != m.end())
          {
              return iter->second;
          }
              
          int q = INT32_MIN;
          
          for (int i = 0; i < nums.size(); ++i)
          {
              vector<int> nums_copy(nums);
              int point = nums_copy[i];
  
              auto iter = nums_copy.begin();
              while (iter != nums_copy.end())
              {
                  
                  if (*iter == point - 1 || *iter == point + 1)
                  {
                      iter = nums_copy.erase(iter);
                  }
                  else
                  {
                      ++iter;
                  }
              }
  
              iter = nums_copy.begin();
              while (iter != nums_copy.end())
              {
                  if (*iter == point)
                  {
                      nums_copy.erase(iter);
                      break;
                  }
                  else
                      ++iter;
              }
  
              q = max(point + deleteAndEarn(nums_copy), q);
          }
  
          m[nums] = q;
          return q;
      }
  };
  ```

* 删除一个数字`nums[i]`后，获得的点数是等于`nums[i]`的数字的总和，并且`nums[i]-1`和`nums[i]+1`都不能再选。如果我们对数组中每个数字求一下总和，那么获得的新数组`sums`正好就符合打家劫舍问题。

    ```c++
    class Solution {
    public:
        int deleteAndEarn(vector<int>& nums) {
            // 找最大值
            int max_val = 0;
            for (int i = 0; i < nums.size(); ++i)
                max_val = max(max_val, nums[i]);
            
            // 构造符合打家劫舍的数组
            vector<int> sums(max_val+1);
            for (int i = 0; i < nums.size(); ++i)
                sums[nums[i]] += nums[i];
            
            // 按标准的打家劫舍问题进行处理
            vector<int> dp(sums.size()+1);
            dp[0] = 0;
            dp[1] = sums[0];
            for (int i = 2; i <= sums.size(); ++i)
                dp[i] = max(dp[i-1], dp[i-2]+sums[i-1]);
            return dp[sums.size()];
        }
    };
    ```

    官方的代码：

    ```c++
    class Solution {
    public:
        int rob(vector<int>& nums) {
            if (nums.size() == 1)
                return nums[0];
            if (nums.size() == 2)
                return max(nums[0], nums[1]);

            int q_im2 = nums[0];
            int q_im1 = max(nums[0], nums[1]);
            int max_val;
            for (int i = 2; i < nums.size(); ++i)
            {
                max_val = max(nums[i] + q_im2, q_im1);
                q_im2 = q_im1;
                q_im1 = max_val;
            }
            return max_val;
        }

        int deleteAndEarn(vector<int>& nums) {
            int max_val = -1;
            for (int i = 0; i < nums.size(); ++i)
            {
                max_val = max(max_val, nums[i]);
            }

            vector<int> sums(max_val+1, 0);
            for (int i = 0; i < nums.size(); ++i)
                sums[nums[i]] += nums[i];

            return rob(sums);
        }
    };
    ```
  
* 先排序，再打家劫舍

    因为只有当数字是连续的时候，才符合打家劫舍的情况，所以我们可以先对数组进行排序，把相同的数字合并起来，把数组分成一段一段连续的数组，对每一段数组应用打家劫舍，最后把结果相加起来就好了。

    ```c++
    class Solution {
    private:
        int rob(vector<int> &nums) {
            int size = nums.size();
            if (size == 1) {
                return nums[0];
            }
            int first = nums[0], second = max(nums[0], nums[1]);
            for (int i = 2; i < size; i++) {
                int temp = second;
                second = max(first + nums[i], second);
                first = temp;
            }
            return second;
        }
	
    public:
        int deleteAndEarn(vector<int> &nums) {
            int n = nums.size();
            int ans = 0;
            sort(nums.begin(), nums.end());
            vector<int> sum = {nums[0]};
            for (int i = 1; i < n; ++i) {
                int val = nums[i];
                if (val == nums[i - 1]) {
                    sum.back() += val;
                } else if (val == nums[i - 1] + 1) {
                    sum.push_back(val);
                } else {
                    ans += rob(sum);
                    sum = {val};
                }
            }
            ans += rob(sum);
            return ans;
        }
    };
    ```

	对于给定数据，先对其进行排序，比如某个排序结果为`[2, 2, 3, 3, 5, 6]`，那么从`3`和`5`中间断开，对`[2, 2, 3, 3]`进行打家劫舍，对`[5, 6]`进行另外一次打家劫舍，然后把两次的结果加起来。

#### 打家劫舍 II

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

 
```
示例 1：

输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
示例 2：

输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 3：

输入：nums = [0]
输出：0
```

代码：

1. 两遍动态规划。这道题与原版打家劫舍不同的地方是，住户变成了环形。我们可以将其拆解为两个单列，一个是不偷第一家，一个是不偷最后一家，做两遍动态规划，然后选出最大值即可。

    ```c++
    class Solution {
    public:
        int rob(vector<int>& nums) {
            if (nums.size() == 1) return nums[0];
            
            vector<int> dp(nums.size()+1);
            dp[1] = nums[0];
            for (int i = 2; i <= nums.size() - 1; ++i)
                dp[i] = max(dp[i-1], dp[i-2]+nums[i-1]);
            int ans1 = dp[nums.size()-1];

            dp.assign(dp.size()+1, 0);
            dp[2] = nums[1];
            for (int i = 3; i <= nums.size(); ++i)
                dp[i] = max(dp[i-1], dp[i-2]+nums[i-1]);
            int ans2 = dp[nums.size()];

            return max(ans1, ans2);
        }
    };
    ```

    这道题意在准确控制动态数组的开始、结束、初始值等细节问题。

#### 丑数

> 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
>
> 示例：
>
> 输入: n = 10
>
> 输出: 12
>
> 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
>
> 说明：
>
> 1. `1`是丑数。
> 2. `n`不超过 1690。

代码：

1. 这个应该是最优的解法

    当前要求的数字一定是之前某个数字的 2 倍，3 倍或 5 倍。前面每个数字的 2 倍，3 倍和 5 倍都有可能被取到，所以让三个指针一个一个滑动就可以了。

    ```c++
    class Solution {
    public:
        int nthUglyNumber(int n) {
            if (n == 1)
                return 1;
            int un_idx[3] = {0, 0, 0};
            int un[3];
            int ugly_nums[n];
            ugly_nums[0] = 1;
            for (int i = 1; i < n; ++i)
            {
                un[0] = ugly_nums[un_idx[0]] * 2;
                un[1] = ugly_nums[un_idx[1]] * 3;
                un[2] = ugly_nums[un_idx[2]] * 5; 
                ugly_nums[i] = min(min(un[0], un[1]), un[2]);
                if (ugly_nums[i] == un[0])  // 这三个 if 分开写就很巧妙，因为有可能出现 2 * 3 = 3 * 2 = 6 这样的情况。分开写的 if 可以让三个指针各自判断并递增，避免了重复数字的出现
                    ++un_idx[0];
                if (ugly_nums[i] == un[1])
                    ++un_idx[1];
                if (ugly_nums[i] == un[2])
                    ++un_idx[2];
            }
            return ugly_nums[n-1];
        }
    };
    ```

1. 用循环和库函数的写法

    ```c++
    class Solution {
    public:
        int nthUglyNumber(int n) {
            vector<int> nums(n);
            nums[0] = 1;
            int p[3] = {0}, temp[3];
            int muls[3] = {2, 3, 5};
            for (int i = 1; i < n; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    temp[j] = nums[p[j]] * muls[j];

                nums[i] = *min_element(temp, temp+3);
                
                for (int j = 0; j < 3; ++j)
                    if (nums[i] == temp[j]) ++p[j];
    
            }
            return nums[n-1];
        }
    };
    ```

1. 后来自己写的，明显复杂了一些

    ```c++
    class Solution {
    public:
        int getUglyNumber(int n) {
            if (n == 1)
                return 1;
            
            vector<int> m(n);
            m[0] = 1;
            int prev[3] = {0, 0, 0};
            
            for (int i = 1; i < n; ++i)
            {
                while (m[prev[0]] * 2 <= m[i-1])
                    ++prev[0];
                while (m[prev[1]] * 3 <= m[i-1])
                    ++prev[1];
                while (m[prev[2]] * 5 <= m[i-1])
                    ++prev[2];
                m[i] = min(m[prev[0]] * 2, min(m[prev[1]] * 3, m[prev[2]] * 5));
            }
            return m.back();
        }
    };
    ```

#### 组合总和 IV

> 给你一个由 不同 整数组成的数组`nums` ，和一个目标整数 target 。请你从`nums`中找出并返回总和为 target 的元素组合的个数。
>
> 题目数据保证答案符合 32 位整数范围。
>
> 示例 1：
> ```
> 输入：nums = [1,2,3], target = 4
> 输出：7
> 解释：
> 所有可能的组合为：
> (1, 1, 1, 1)
> (1, 1, 2)
> (1, 2, 1)
> (1, 3)
> (2, 1, 1)
> (2, 2)
> (3, 1)
> 请注意，顺序不同的序列被视作不同的组合。
> ```
> 
> 示例 2：
>
> ```
> 输入：nums = [9], target = 3
> 输出：0
> ```

分析：在形成 target 之前的最后一步，需要从数组中挑一个数。比如数组为`{1, 2, 3}`，target 为 4，那么在形成 4 之前，上一步的 target 可以为 4 - 1 = 3，4 - 2 = 2，或 4 - 3 = 1。如果我们已经计算出 target 为 3，2，1 的个数，那么直接相加就可以了。由此就可以得到递归算法：

$$
n_{target} = \sum\limits_{\mathrm{num} \in \mathrm{nums}}n_{target - \mathrm{num}}
$$

其中，$\mathrm{num}$满足不大于 target。我们直接按自底向上法计算就可以了。

```c++
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<int> num_combs(target+1, 0);
        num_combs[0] = 1;
        for (int i = 1; i <= target; ++i)
        {
            for (int &num: nums)
            {
                if (num <= i && num_combs[i - num] < INT32_MAX - num_combs[i])
                    num_combs[i] += num_combs[i - num];
            }
        }
        return num_combs[target];
    }
};
```

#### 停在原地的方案数

有一个长度为 arrLen 的数组，开始有一个指针在索引 0 处。

每一步操作中，你可以将指针向左或向右移动 1 步，或者停在原地（指针不能被移动到数组范围外）。

给你两个整数 steps 和 arrLen ，请你计算并返回：在恰好执行 steps 次操作以后，指针仍然指向索引 0 处的方案数。

由于答案可能会很大，请返回方案数 模 10^9 + 7 后的结果。

分析：

若在第 n 步停留在 p 位置，那么在第 n - 1 步可能停留在 p - 1, p, p + 1 位置（p - 1 >= 0, p + 1 <= lenArr）。相加即可。

代码：

1. 将结果保存为二维表（超时）

```c++
    class Solution {
    public:
        vector<vector<int>> m;  // (steps, pos)
        
        void disp_m(vector<vector<int>> &m)
        {
            for (int i = 0; i < m.size(); ++i)
            {
                for (int j = 0; j < m[0].size(); ++j)
                {
                    cout << m[i][j] << ", ";
                }
                
                cout << endl;
            }
        }

        int numWays(int steps, int arrLen) {
            m.assign(steps + 1, vector<int>(arrLen, 0));
            m[1][0] = 1;
            int min_len = min(arrLen, steps);
            for (int i = 1; i < min_len; ++i)
            {
                m[i][i] = 1;
            }

            int start, end;
            for (int s = 1; s <= steps; ++s)
            {
                int pos_right = min(s - 1, arrLen - 1);
                for (int pos = 0; pos <= pos_right; ++pos)
                {
                    start = pos == 0 ? 0 : pos - 1;
                    end = pos == arrLen - 1 ? arrLen - 1 : pos + 1;
                    // cout << "s, pos, start, end: " << s << ", " << pos << ", " << start << ", " << end << endl;
                    for (int p = start; p <= end; ++p)
                    {
                        m[s][pos] = ((m[s][pos] % 1000000007) + (m[s-1][p] % 1000000007)) % 1000000007;
                    }
                }
            }
            // disp_m(m);
            return m[steps][0];
        }
    };
    ```

1. 将结果保存为两个数组（只能击败 5%）

    ```c++
    class Solution {
    public:
        int numWays(int steps, int arrLen) {
            vector<int> prev, next(arrLen, 0);
            int min_len = min(arrLen - 1, steps - 1);
            next[0] = 1;
            next[1] = 1;
            
            int start, end;
            for (int s = 2; s <= steps; ++s)
            {
                prev = next;
                
                int pos_right = min(s - 1, arrLen-1);
                next.assign(arrLen, 0);
                if (s < arrLen)
                {
                    next[s] = 1;
                }

                for (int pos = 0; pos <= pos_right; ++pos)
                {
                    start = pos == 0 ? 0 : pos - 1;
                    end = pos == arrLen - 1 ? arrLen - 1 : pos + 1;
                    for (int p = start; p <= end; ++p)
                    {
                        next[pos] = ((next[pos] % 1000000007) + (prev[p] % 1000000007)) % 1000000007;
                    }
                }
            }
            return next[0];
        }
    };
    ```

1. 官方题解 1，将结果保存为二维数组（击败 25%，还没看）

    ```c++
    class Solution {
    public:
        const int MODULO = 1000000007;

        int numWays(int steps, int arrLen) {
            int maxColumn = min(arrLen - 1, steps);
            vector<vector<int>> dp(steps + 1, vector<int>(maxColumn + 1));
            dp[0][0] = 1;
            for (int i = 1; i <= steps; i++) {
                for (int j = 0; j <= maxColumn; j++) {
                    dp[i][j] = dp[i - 1][j];
                    if (j - 1 >= 0) {
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MODULO;
                    }
                    if (j + 1 <= maxColumn) {
                        dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % MODULO;
                    }
                }
            }
            return dp[steps][0];
        }
    };
    ```

1. 官方题解 2，使用两个一维数组保存结果（击败 45%，还没看）

### 一些心得

动态规划的题目可分为下面几类：

1. 一维与位置无关
2. 一维与位置相关
3. 二维与位置相关

## 数组

### 数组中重复的数字

> 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

分析：可以新建一个哈希表（`unordered_set`）来存储数字，若有重复的，直接返回即可。也可以直接在原数组上构建哈希表，若出现冲突，直接返回即可。

代码：

1. 使用新的哈希表

    ```c++
    class Solution {
    public:
        int findRepeatNumber(vector<int>& nums) {
            unordered_set<int> s;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (s.find(nums[i]) == s.end())
                    s.insert(nums[i]);
                else
                    return nums[i];
            }
            return -1;
        }
    };
    ```

1. 通过交换数据，在原数组基础上直接构建哈希表

    ```c++
    class Solution {
    public:
        int findRepeatNumber(vector<int>& nums) {
            int temp;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (nums[i] != i)
                {
                    if (nums[nums[i]] == nums[i])
                        return nums[i];
                    else
                    {
                        temp = nums[nums[i]];
                        nums[nums[i]] = nums[i];
                        nums[i] = temp;
                    }
                }
            }
            return -1;
        }
    };
    ```

1. 排序

    ```c++
    class Solution {
    public:
        int findRepeatNumber(vector<int>& nums) {
            sort(nums.begin(), nums.end());
            int pos = 0;
            while (pos < nums.size() - 1)
            {
                if (nums[pos] == nums[pos+1]) return nums[pos];
                ++pos;
            }
            return -1;
        }
    };
    ```

### 不修改数组找出重复的数字

> 给定一个长度为`n+1`的数组`nums`，数组中所有的数均在`1∼n`的范围内，其中`n≥1`。

> 请找出数组中任意一个重复的数，但不能修改输入的数组。

> 只能使用$O(1)$的额外空间。

**分析**：

使用抽屉原理，首先将`[1, n]`划分成两个子区间：`[1, n/2]`和`[n/2 + 1, n]`，然后统计数组中属于这两个子区间的元素的个数。若区间中元素个数大于区间的长度，那么必然有一个元素是重复的。以此类推，直到最终找到重复的元素。

代码：

```c++
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        int left, right, mid;
        left = 1;
        right = nums.size() - 1;
        mid = (left + right) / 2;
        int left_count = 0, right_count = 0;
        
        while (true)
        {
            left_count = 0;
            right_count = 0;
            
            for (int i = 0; i < nums.size(); ++i)
            {
                if (nums[i] >= left && nums[i] <= mid)
                    ++left_count;
                else if (nums[i] > mid && nums[i] <= right)
                    ++right_count;
            }
            
            if (left == mid && right == mid)
            {
                if (left_count > 1) return left;
                if (right_count > 1) return right;
                break;
            }
            
            if (left_count > mid - left + 1)
                right = mid;
            else
                left = mid + 1;
                
            mid = (left + right) / 2;
        }
        
        return -1;
    }
};
```

#### 正则表达式匹配

```
请实现一个函数用来匹配包括'.'和'*'的正则表达式。

模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。

在本题中，匹配是指字符串的所有字符匹配整个模式。

例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配。

样例
输入：

s="aa"
p="a*"

输出:true
```

代码（还没来得及看）：

```c++
class Solution {
public:
    vector<vector<int>>f;
    int n, m;
    bool isMatch(string s, string p) {
        n = s.size();
        m = p.size();
        f = vector<vector<int>>(n + 1, vector<int>(m + 1, -1));
        return dp(0, 0, s, p);
    }

    bool dp(int x, int y, string &s, string &p)
    {
        if (f[x][y] != -1) return f[x][y];
        if (y == m)
            return f[x][y] = x == n;
        bool first_match = x < n && (s[x] == p[y] || p[y] == '.');
        bool ans;
        if (y + 1 < m && p[y + 1] == '*')
        {
            ans = dp(x, y + 2, s, p) || first_match && dp(x + 1, y, s, p);
        }
        else
            ans = first_match && dp(x + 1, y + 1, s, p);
        return f[x][y] = ans;
    }
};

```

#### 最长公共子序列（LCS）

分析：

设$X_i$表示字符串$X$前$i$个字符，$Y_j$表示字符串$Y$前$j$个字符，$c[i][j]$表示$X_i$和$Y_j$的 LCS 的长度，则可以列出递归式：

$$
c[i][j] = 
\begin{cases}
0, &\text{if } i = 0 \text{ or } j = 0 \\
c[i - 1, j - 1] + 1, &\text{if } i, j \gt 0 \text{ and } x_i = y_j \\
\max (c[i, j - 1], c[i - 1, j]), &\text{if } i, j \gt 0 \text{ and } x_i \neq y_j
\end{cases}
$$

1. 若一个子串长度为零，那么 LCS 的长度也一定为 0
1. 若两个子串长度都不为零，且两个子串末尾字符相同，那么删去末尾字符后，两个子串仍有 LCS，且 LCS 的长度减 1
1. 若两个子串长度都不为零，且末尾字符不同，若 LCS 与$X_i$结尾字符不同，那么 LCS 必定是$X_{i-1}$和$Y_j$的 LCS；若 LCS 与$Y_j$结尾字符不同，那么 LCS 必定是$Y_{j-1}$和$X_i$的 LCS。此时 LCS 的长度为两个子问题的 LCS 长度的较大者。

代码：

1. 二维数组自底向上

    ```c++
    class Solution {
    public:
        int longestCommonSubsequence(string text1, string text2) {
            vector<vector<int>> dp(text1.size()+1, vector<int>(text2.size()+1));
            for (int i = 1; i < text1.size()+1; ++i)
            {
                for (int j = 1; j < text2.size()+1; ++j)
                {
                    if (text1[i-1] == text2[j-1])
                    {
                        dp[i][j] = dp[i-1][j-1] + 1;
                    }
                    else
                    {
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                    }
                }
            }
            return dp[text1.size()][text2.size()];
        }
    };
    ```

    注意，这里`i`和`j`分别表示子串长度，所以在取下标的时候，需要减 1。

1. 一维数组自底向上

    观察二维数组自底向上法，发现只用到了过去一步的状态，因此可以用两个一维数组来代替。

    ```c++
    class Solution {
    public:
        vector<int> c_prev;
        vector<int> c;
        int longestCommonSubsequence(string text1, string text2) {
            c.resize(text2.size() + 1);
            c_prev = c;
            
            for (int i = 0; i <= text1.size(); ++i)
            {
                for (int j = 0; j <= text2.size(); ++j)
                {
                    if (i == 0 || j == 0)
                        c[j] = 0;
                    else if (text1[i-1] == text2[j-1])
                        c[j] = c_prev[j-1] + 1;
                    else
                        c[j] = max(c[j-1], c_prev[j]);
                }
                c_prev = c;
            }
    
            return c[text2.size()];
        }
    };
    ```

#### 两个字符串的删除操作

给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

 
```
示例：

输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
```

代码：

1. 动态规划。先将问题转化为 lcs，然后再计算

    ```c++
    class Solution {
    public:
        int minDistance(string word1, string word2) {
            vector<vector<int>> dp(word1.size()+1, vector<int>(word2.size()+1));
            for (int i = 1; i < word1.size()+1; ++i)
            {
                for (int j = 1; j < word2.size()+1; ++j)
                {
                    if (word1[i-1] == word2[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
                    else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
            return word1.size() - dp[word1.size()][word2.size()] + word2.size() - dp[word1.size()][word2.size()];
        }
    };
    ```

1. 动态规划，不借助 lcs

    ```c++
    class Solution {
    public:
        int minDistance(string word1, string word2) {
            vector<vector<int>> dp(word1.size()+1, vector<int>(word2.size()+1));
            for (int i = 0; i < word1.size()+1; ++i)
            {
                for (int j = 0; j < word2.size()+1; ++j)
                {
                    if (i == 0 || j == 0) dp[i][j] = i + j;  // 注意 i, j 从 0 开始，边界值是动态的
                    else if (word1[i-1] == word2[j-1]) dp[i][j] = dp[i-1][j-1];  // 若相等，不需要删除字符
                    else dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1]);  // 若不等，挑个较小的删除字符数再加 1
                }
            }
            return dp[word1.size()][word2.size()];
        }
    };
    ```

#### 不相交的线

在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：

`nums1[i] == nums2[j]`

且绘制的直线不与任何其他连线（非水平线）相交。

请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。

示例：

```
输入：nums1 = [1,4,2], nums2 = [1,2,4]
输出：2
解释：可以画出两条不交叉的线，如上图所示。 
但无法画出第三条不相交的直线，因为从 nums1[1]=4 到 nums2[2]=4 的直线将与从 nums1[2]=2 到 nums2[1]=2 的直线相交。
```

分析：同最长公共子序列，代码一个字都不用改。我们在分析时，可选某个状态已经为最大连线情况，然后尝试从右端开始逐渐削减连线的数量：若两个末尾数字相同，那么最大连线数量减一；若两个末尾数字不同，那么最大连线数量为减去上侧末尾数字或送去下侧末尾数字得到的最大连线数量的最大者。这样的分析就把我们带回了 LCS。

代码：

```c++
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2)
    {
        vector<int> c_prev(nums2.size()+1);
        vector<int> c = c_prev;
        for (int i = 0; i <= nums1.size(); ++i)
        {
            for (int j = 0; j <= nums2.size(); ++j)
            {
                if (i == 0 || j == 0)
                    c[j] = 0;
                else if (nums1[i-1] == nums2[j-1])
                    c[j] = c_prev[j-1] + 1;
                else
                    c[j] = max(c[j-1], c_prev[j]);
            }
            c_prev = c;
        }
        return c[nums2.size()];
    }
};
```

#### 连续子数组的最大和（最大子数组和）

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。
 
```
示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

分析：

设$m[i]$为前$i$个数字的最大和，要求这前$i$个数字中必须包含$nums[i]$，则有

$$
m[i] = max(m[i-1] + nums[i], nums[i]);
$$

或者

$$
m[i] = 
\begin{cases}
m_{i-1} + n_i, &\text{if } m_{i-1} \gt 0 \\
n_i, &\text{if } m_{i-1} \leq 0
\end{cases}
$$

通过这种方法得到数组$m$后，对$m$进行遍历，找到最大值即可。

（但是感觉这道题不像是动态规划，因为我们定义的数组$m$与原问题相差得比较远）

代码：

1. 使用第一个递归公式

    ```c++
    class Solution {
    public:
        vector<vector<int>> m;
        int maxSubArray(vector<int>& nums) {
            vector<int> m(nums.size());
            m[0] = nums[0];
            for (int i = 1; i < nums.size(); ++i)
            {
                m[i] = max(m[i-1] + nums[i], nums[i]);
            }
            
            int max_val = INT32_MIN;
            for (auto &elm: m)
                if (elm > max_val) max_val = elm;
            return max_val;
        }
    };
    ```

    还是有点想不明白。

1. 使用第二个递归公式

    ```c++
    class Solution {
    public:
        vector<vector<int>> m;
        int maxSubArray(vector<int>& nums) {
            vector<int> m(nums.size());
            m[0] = nums[0];
            for (int i = 1; i < nums.size(); ++i)
            {
                m[i] = m[i-1] > 0 ? m[i-1] + nums[i] : nums[i];
            }
            
            int max_val = INT32_MIN;
            for (auto &elm: m)
                if (elm > max_val) max_val = elm;
            return max_val;
        }
    };
    ```

1. 最简版本

    ```c++
    class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            int s = 0, res = INT32_MIN;

            for (auto &num: nums)
            {
                if (s < 0) s = 0;
                s += num;
                res = max(res, s);
            }

            return res;
        }
    };
    ```

    感觉这个版本才抓到了问题的精髓。如果发现某段数组的和小于零，那么这段数组贡献到结果`res`中去是负收益，我们直接把这段数组抛弃掉就好，然后重新开始。否则我们直接比较结果，取较大的就好了。

    为什么要抛弃掉某段和为负的数组？假如没有这段数组，后面如果涨得话会涨得更猛。

    或者可以这样想：假如我们前面已经准备了一段数组，对于新来的一个数，如果旧数组+新数还没有新数本身大，那我就可以认为旧数组还不如这一个新数，可以把旧数组舍弃掉了。（假设数组是`[..., a, b, c, d, ...]`，现在我们发现`c`是符合要求的新元素，这时候有没有可能从`c`开始向前取元素组成新数组？假如我们取了`[b, c]`，那么`b`不符合我们之前的要求，因此`[..., a]`的和大于等于 0，又因为`c`是符合要求的，所以`[..., a, b]`的和一定是小于 0 的，因此`b`一定小于 0，因此`b`对`[b, c]`的贡献是负值，可以删掉。如果取`[..., b, c]`作为新数组，我们可以把`[..., b]`看作`b'`，一定小于 0。可以验证下这个结论）

1. 前缀和（超时）

    做这道题的时候，我的第一反应是前槡和，可是超时了。

    ```cpp
    class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            vector<int> sum1(nums.size());
            sum1[0] = nums[0];
            for (int i = 1; i < nums.size(); ++i)
            {
                sum1[i] = sum1[i-1] + nums[i];
            }
            int ans = INT32_MIN;
            for (int i = 0; i < nums.size(); ++i)
            {
                for (int j = i; j < nums.size(); ++j)
                {
                    ans = max(ans, sum1[j] - sum1[i] + nums[i]);
                }
            }
            return ans;
        }
    };
    ```

1. 官方答案还给了个线段树，不懂，有空了看看

    ```cpp
    class Solution {
    public:
        struct Status {
            int lSum, rSum, mSum, iSum;
        };

        Status pushUp(Status l, Status r) {
            int iSum = l.iSum + r.iSum;
            int lSum = max(l.lSum, l.iSum + r.lSum);
            int rSum = max(r.rSum, r.iSum + l.rSum);
            int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
            return (Status) {lSum, rSum, mSum, iSum};
        };

        Status get(vector<int> &a, int l, int r) {
            if (l == r) {
                return (Status) {a[l], a[l], a[l], a[l]};
            }
            int m = (l + r) >> 1;
            Status lSub = get(a, l, m);
            Status rSub = get(a, m + 1, r);
            return pushUp(lSub, rSub);
        }

        int maxSubArray(vector<int>& nums) {
            return get(nums, 0, nums.size() - 1).mSum;
        }
    };
    ```

#### 最长递增子序列

> 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
> 
> 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
> 
> ```
> 示例 1：
> 
> 输入：nums = [10,9,2,5,3,7,101,18]
> 输出：4
> 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
> ```

分析：

设数组`m[i]`表示以`nums[i]`结尾的子数列的最长严格递增子列的长度。若`nums[i] > nums[j]`，那么数字`nums[i]`可以放到以`nums[j]`结尾的子列的后面，因此最长连续子列的长度会加一：

$$
m[i] = \max\{ m[i], m[j] + 1 \}, \quad j = 0, 1, \dots, i-1
$$

以`nums[i]`结尾的最长子列的长度，是取以`nums[0]`，`nums[1]`，...，`nums[i-1]`结尾的最长子列中的最大值再加 1，如果`nums[i]`能放到这些子列后面的话。

代码：

1. 动态规划

    ```c++
    class Solution {
    public:
        int lengthOfLIS(vector<int>& nums) {
            vector<int> dp(nums.size(), 1);
            int ans = 1;
            for (int i = 1; i < nums.size(); ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    if (nums[i] > nums[j])
                        dp[i] = max(dp[i], dp[j]+1);
                }
                ans = max(dp[i], ans);
            }
            return ans;
        }
    };
    ```

1. 贪心 + 二分查找

    不考虑使用动态规划的思想，我们维护一个数组`m`，记录实际的最长子列。对原数组进行遍历，若`nums[i]`大于`m`的最大值（在末尾），那么就把元素添加到`m`的末尾；若元素小于`m`的最大值，那么使用二分法找到一个大于等于`nums[i]`的元素`m[l]`，并将`m[l]`替换为`nums[i]`。

    所谓的贪心思想是：子列增长得越慢，那么它就会越长。因此每一步都尽量让子列中的数小一些。

    ```c++
    class Solution {
    public:
        int lengthOfLIS(vector<int>& nums) {
            vector<int> m(nums.size());
            int len = 1;
            m[0] = nums[0];
            
            int l, r, mid;
            for (int i = 1; i < nums.size(); ++i)
            {
                if (nums[i] > m[len-1])
                {
                    m[len++] = nums[i];
                }
                else
                {
                    l = 0;
                    r = len - 1;
                    mid = (l + r) >> 1;
                    while (l < r)
                    {
                        if (m[mid] < nums[i])
                            l = mid + 1;
                        else
                            r = mid;
                        mid = (l + r) >> 1;  // 记得更新 mid
                    }
                    m[l] = min(nums[i], m[l]);
                }
            }

            return len;
        }
    };
    ```

#### 把数字翻译成字符串

给定一个数字，我们按照如下规则把它翻译为字符串：

0 翻译成 a，1 翻译成 b，……，11 翻译成 l，……，25 翻译成 z。

一个数字可能有多个翻译。

例如 12258 有 5 种不同的翻译，它们分别是 bccfi、bwfi、bczi、mcfi 和 mzi。

请编程实现一个函数用来计算一个数字有多少种不同的翻译方法。

```
样例
输入："12258"

输出：5
```

分析：

很简单的动态规划。

代码：

1. 使用`m[l]`表示字符串前`l`个字母有多少种编码方法。

    ```c++
    class Solution {
    public:
        int getTranslationCount(string s) {
            vector<int> m(s.size() + 1, 0);
            m[0] = 1;
            m[1] = 1;
            for (int l = 2; l <= s.size(); ++l)
            {
                if (s[l-2] == '0' || s[l-2] > '2' || (s[l-2] == '2' && s[l-1] > '5'))
                    m[l] = m[l-1];
                else
                    m[l] = m[l-1] + m[l-2];
            }
            
            return m.back();
        }
    };
    ```

1. 由于只用到前两个状态，所以可以对数组`m`做简化。

    ```c++
    class Solution {
    public:
        int getTranslationCount(string s) {
            int m_cur = 1, m_pprev = 1, m_prev = 1;
            for (int l = 2; l <= s.size(); ++l)
            {
                if (s[l-2] == '0' || s[l-2] > '2' || (s[l-2] == '2' && s[l-1] > '5'))
                    m_cur = m_prev;
                else
                    m_cur = m_prev + m_pprev;
                    
                m_pprev = m_prev;
                m_prev = m_cur;
            }
            return m_cur;
        }
    };
    ```

#### 礼物的最大价值

> 在一个 m×n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。
> 
> 你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格直到到达棋盘的右下角。
> 
> 给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？
> 
> 注意：
> 
> m,n>0
> 
> ```
> 样例：
> 
> 输入：
> [
>   [2,3,1],
>   [1,7,1],
>   [4,6,1]
> ]
> 
> 输出：19
> 
> 解释：沿着路径 2→3→7→6→1 可以得到拿到最大价值礼物。
> ```

分析：

很简单的二维动态规划。注意把边界值赋好。

代码：

```c++
class Solution {
public:
    int val;
    int getMaxValue(vector<vector<int>>& grid) {
        if (grid.size() == 1)
            return grid[0][0];
            
        vector<vector<int>> m(grid.size(), vector<int>(grid[0].size()));
        int row = grid.size(), col = grid[0].size();
        m[0][0] = grid[0][0];
        for (int i = 1; i < col; ++i)
            m[0][i] = m[0][i-1] + grid[0][i];
        for (int i = 1; i < row; ++i)
            m[i][0] = m[i-1][0] + grid[i][0];
        
        for (int i = 1; i < row; ++i)
        {
            for (int j = 1; j < col; ++j)
            {
                m[i][j] = max(m[i-1][j], m[i][j-1]) + grid[i][j];
            }
        }
        
        return m[row-1][col-1];
    }
};
```

（如果题目中没说明的话，我们可以直接在原数组`grid`上进行修改，省去了开辟空间）

#### 最长不含重复字符的子字符串（不含重复字符的最长子字符串）

> 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
> 
> 示例 1:
> 
> 输入: "abcabcbb"
> 输出: 3 
> 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

分析：

1. 动态规划 + 散列表

    设`dp[i]`为以`s[i]`结尾的子串的最大不重复子串长度，用`m[c]`记录字符`c`上一次出现的位置。若字符`c`出现在当前区间内（当前区间的长度由`dp[i-1]`确定），则说明遇到重复字符，重新开始计数；若字符`c`未出现在当前区间内，那么可以放心地把当前区间的长度加 1，即`dp[i] = dp[i-1] + 1`。

代码：

1. 动态规划 + 散列表

    其实这个并不能算是动态规划，因为中间有重新开始计数的过程。动态规划应该能把问题分割成子问题才对。这个归类到滑动窗口比较好一点。滑动窗口和散列表的结合。

    ```c++
    class Solution {
    public:
        int lengthOfLongestSubstring(string s) {
            unordered_map<char, int> m;
            int len = 0, ans = 0;
            for (int i = 0; i < s.size(); ++i)
            {
                if (m.find(s[i]) != m.end())
                {
                    if (len >= i - m[s[i]]) len = i - m[s[i]];  // 为什么这里要有等于呢，不写等于好像也通过了
                    else ++len;
                }
                else ++len;
                ans = max(ans, len);
                m[s[i]] = i;
            }
            return ans;
        }
    };
    ```

    注意一个细节，为什么`if (len >= i-m[s[i]]) len = i - m[s[i]];`里有等于呢？本来我们的逻辑应该是当`len`大于`ans`的时候才更新`ans`，所以当`len == i - m[s[i]]`的时候，我们只需要什么都不用做就好了，而不是将长度再加 1。所以实际上应该这样写：

    ```c++
    if (m.find(s[i]) != m.end())
    {
        if (len > i - m[s[i]]) len = i - m[s[i]];
        else if (len == i - m[s[i]]) ;
        else ++len;
    }
    ```

    但是其实更新一下`len`也没有大碍。所以就把第一个条件和第二个条件合并到一起了。

    其实根本不需要哈希表，用数组计数就可以了。更快一点。

1. 直接用哈希集合 + 滑动窗口。逻辑简洁清晰，且根本用不到动态规划

    其实和上面那个是一个道理。只不过上面那种解法是直接定位了长度，这种解法是用滑动窗口一步一步定位长度。

    ```c++
    class Solution {
    public:
        int lengthOfLongestSubstring(string s) {
            unordered_set<char> occ;
            int l = 0, r = 0;
            int len = 0, max_len = 0;
            while (r < s.size())
            {
                if (occ.find(s[r]) == occ.end())  // 若没出现过，则右移滑动窗口右边界
                {
                    occ.insert(s[r]);
                    ++r;
                    ++len;
                    max_len = max(len, max_len);
                }
                else  // 若出现过，则右移滑动窗口左边界，直到上次出现当前字符的位置
                {
                    occ.erase(s[l]);
                    ++l;
                    --len;
                }
            }
            return max_len;
        }
    };
    ```

我觉得这道题根本就不是滑动窗口。滑动窗口的必要条件是，对于给定的滑动窗口，总是能够知道该移动左边界还是移动右边界。但是对于`[a, b, c, a, d, d]`，我们首先可以找到`[a, b, c, a]`这个区间。可是接下来呢？如果将右边界向右移动，那么左边界该怎么办呢？左边界无法知道在哪里停止。

如果`left`向`right`之间没有重复的字符，那么应该`++left`，作为新的起点。如果有和`right`处相同的字符，那么从重复的字符处作为新的起点。这两种情况差异太大，没办法用滑动窗口实现。

这道题只是使用哈希表来记忆一些东西而已。

#### 骰子的点数

将一个骰子投掷 n 次，获得的总点数为 s，s 的可能范围为 n∼6n。

掷出某一点数，可能有多种掷法，例如投掷 2 次，掷出 3 点，共有 [1,2],[2,1] 两种掷法。

请求出投掷 n 次，掷出 n∼6n 点分别有多少种掷法。

```
样例1
输入：n=1

输出：[1, 1, 1, 1, 1, 1]

解释：投掷1次，可能出现的点数为1-6，共计6种。每种点数都只有1种掷法。所以输出[1, 1, 1, 1, 1, 1]。
```

```
样例2
输入：n=2

输出：[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]

解释：投掷2次，可能出现的点数为2-12，共计11种。每种点数可能掷法数目分别为1,2,3,4,5,6,5,4,3,2,1。

      所以输出[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]。
```


代码：

1. 动态规划

    假如要投 3 次，和为 15，那么投第 2 次时的取值范围为`[max(15 - 6, 2), 15 - 1]`，我们取总和就好了。

    ```c++
    class Solution {
    public:
        vector<int> numberOfDice(int n) {
            vector<vector<int>> dp(n, vector<int>(6 * n, 0));
            for (int j = 0; j < 6; ++j)
                dp[0][j] = 1;
            
            for (int i = 1; i < n; ++i)
                for (int j = 0; j < 6 * (i+1); ++j)
                    for (int k = max(j - 6, i - 1); k < j; ++k)
                        dp[i][j] += dp[i-1][k];

            return vector<int>(dp[n-1].begin() + n - 1, dp[n-1].end());
        }
    };
    ```

### n个骰子的点数

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。


你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。


```
示例 1:

输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
示例 2:

输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]
```

1. 动态规划

    `dp[i][j]`表示前`i`个骰子组成数字`j`的方式数量。我们只需要考虑新来一个数字时，它只能掷出 1 ~ 6 这个范围的数字，那么我们只要知道前面的骰子掷出`s - 6`到`s - 1`的数量，然后相加即可。

    ```c++
    class Solution {
    public:
        vector<double> dicesProbability(int n) {
            vector<vector<int>> dp(n + 1, vector<int>(6 * n + 1));
            dp[0][0] = 1;
            for (int i = 1; i <= n; ++i)
            {
                for (int j = i; j <= 6 * i; ++j)
                {
                    for (int k = 1; k <= min(j - i + 1, 6); ++k)
                    {
                        dp[i][j] += dp[i-1][j-k];
                    }
                }
            }
            int sum = accumulate(dp[n].begin()+1, dp[n].end(), 0);
            vector<double> ans(n * 6 - n + 1);
            int p = 0;
            for (int i = n; i <= 6 * n; ++i)
                ans[p++] = (float) dp.back()[i] / sum;
            return ans;
        }
    };
    ```

    注意这道题不能按`dp[i][j]`表示`i`个骰子组成数字`j`的数量理解，因为这样会造成重复计数。比如前 1 个骰子组成数字 1，和前 2 个数字组成数字 2，就会有重复的计数情况。

#### 0-1 背包问题

有 N 件物品和一个容量是 V 的背包。每件物品只能使用一次。

第 i 件物品的体积是 vi，价值是 wi。

求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。

输出最大价值。

输入格式

第一行两个整数，N，V，用空格隔开，分别表示物品数量和背包容积。

接下来有 N 行，每行两个整数 vi,wi，用空格隔开，分别表示第 i 件物品的体积和价值。

输出格式

输出一个整数，表示最大价值。

数据范围

```
0 < N, V ≤ 1000
0 < vi, wi ≤ 1000
```

输入样例

```
4 5
1 2
2 4
3 4
4 5
```

输出样例：

```
8
```

代码：

1. 深度优先搜索（超时）

    ```c++
    #include <iostream>
    #include <vector>

    using namespace std;

    int max_val;
    int num, cap;
    vector<pair<int, int>> v;

    void dfs(int pos, int vol, int val)
    {
        max_val = max(max_val, val);
        
        for (int i = pos; i < num; ++i)
        {
            if (vol + v[i].first <= cap)
            {
                dfs(i+1, vol + v[i].first, val + v[i].second);
            }
        }
    }

    int main()
    {
        cin >> num >> cap;
        v.resize(num);
        for (int i = 0; i < num; ++i)
            cin >> v[i].first >> v[i].second;   
        
        dfs(0, 0, 0);
        cout << max_val << endl;
        return 0;
    }
    ```

1. 动态规划

    `dp[i][j]`表示 i 个物品，承重为 j 的背包的最大价值。

    如果背包承重小于当前物品重量，那么最大值延续上一个最大值。

    如果背包承重大于等于当前物品重量，那么最大值则根据选择或不选择当前物品进行判断。

    ```c++
    #include <iostream>
    #include <vector>

    using namespace std;

    int num, cap;
    vector<int> v;
    vector<int> w;
    vector<vector<int>> dp;

    int get_sol()
    {
        for (int i = 1; i <= num; ++i)
        {
            for (int j = 0; j <= cap; ++j)
            {
                if (j < v[i-1])  // i 表示数量，i - 1 表示索引，很细微的差别
                    dp[i][j] = dp[i-1][j];
                else
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j - v[i-1]] + w[i-1]);  // 这里同理
            }
        }
        return dp[num][cap];
    }

    int main()
    {
        cin >> num >> cap;

        v.resize(num);
        w.resize(num);
        for (int i = 0; i < num; ++i)
            cin >> v[i] >> w[i]; 

        dp.assign(num+1, vector<int>(cap+1));
        dp[0][0] = 0;
        int max_val = get_sol();

        cout << max_val << endl;
        return 0;
    }
    ```

1. 动态规划的优化

    因为在计算`dp[i]`时只用到了`dp[i-1]`所以可以对它进行优化。

    ```c++
    #include <iostream>
    #include <vector>

    using namespace std;

    int num, cap;
    vector<int> v;
    vector<int> w;
    vector<int> dp, dp_prev;

    int get_sol()
    {
        for (int i = 1; i <= num; ++i)
        {
            for (int j = 0; j <= cap; ++j)
            {
                if (j < v[i-1])
                    dp[j] = dp_prev[j];
                else
                    dp[j] = max(dp_prev[j], dp_prev[j - v[i-1]] + w[i-1]);
            }
            dp_prev = dp;
        }
        return dp[cap];
    }

    int main()
    {
        cin >> num >> cap;
        
        v.resize(num);
        w.resize(num);
        for (int i = 0; i < num; ++i)
            cin >> v[i] >> w[i]; 
            
        dp.resize(cap+1);
        dp_prev.resize(cap+1);
        dp_prev[0] = 0;
        dp[0] = 0;
        int max_val = get_sol();
        
        cout << max_val << endl;
        return 0;
    }
    ```

#### 一和零

给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

 

示例 1：

```
输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
输出：4
解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
```

示例 2：

```
输入：strs = ["10", "0", "1"], m = 1, n = 1
输出：2
解释：最大的子集是 {"0", "1"} ，所以答案是 2 。
```

分析：

0 - 1 背包问题的扩展，要满足零和一的数量两个要求，因此是三维动态规划。

代码：

```c++
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<vector<int>>> dp;
        dp.resize(strs.size()+1, vector<vector<int>>(m+1, vector<int>(n+1, 0)));
        
        int zero_count = 0, one_count = 0;
        for (int i = 1; i <= strs.size(); ++i)
        {
            zero_count = 0;
            one_count = 0;
            for (int p = 0; p < strs[i-1].size(); ++p)
            {
                if (strs[i-1][p] == '0') ++zero_count;
                else ++one_count;
            }

            for (int j = 0; j <= m; ++j)
            {
                for (int k = 0; k <= n; ++k)
                {
                    if (j >= zero_count && k >= one_count)
                        dp[i][j][k] = max(dp[i-1][j][k], dp[i-1][j-zero_count][k-one_count]+1);
                    else
                        dp[i][j][k] = dp[i-1][j][k];
                }
            }
        }
        return dp[strs.size()][m][n];
    }
};
```

由于在计算`dp[i]`时只用到了`dp[i-1]`的数值，因此还可以继续优化。

#### 目标和

给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

示例 1：

```
输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

分析：

1. 回溯
1. 动态规划，定义`dp[i][j]`为取前 i 个数字，和为 j 的方案数。假设取负号的数的和为`neg`，那么剩下的数的和就是`sum - neg`，我们要得到的目标和就满足`(sum - neg) - neg = target`。此时`neg = (sum - target) / 2`，且`neg`要满足`neg >= 0`并且`neg`必须为整数。接下来我们只需要在数组里找到和为`neg`的方案数就可以了，这样就转化成了 0 - 1 背包问题。

代码：

1. 回溯

    ```c++
    class Solution {
    public:
        int res;
        int sum;
        void dfs(vector<int> &nums, int target, int pos)
        {
            if (sum == target)
                ++res;

            for (int i = pos; i < nums.size(); ++i)
            {
                sum -= 2 * nums[i];
                dfs(nums, target, i+1);
                sum += 2 * nums[i];
            }
        }

        int findTargetSumWays(vector<int>& nums, int target) {
            res = 0;
            sum = accumulate(nums.begin(), nums.end(), 0);
            dfs(nums, target, 0);
            return res;
        }
    };
    ```

1. 动态规划

    ```c++
    class Solution {
    public:
        int findTargetSumWays(vector<int>& nums, int target) {
            int sum = accumulate(nums.begin(), nums.end(), 0);
            if (sum - target < 0 || (sum - target) % 2 != 0) return 0;

            int neg = (sum - target) >> 1;
            vector<vector<int>> dp(nums.size()+1, vector<int>(neg+1, 0));

            dp[0][0] = 1;
            for (int i = 1; i <= nums.size(); ++i)
            {
                for (int j = 0; j <= neg; ++j)
                {
                    if (j < nums[i-1])
                        dp[i][j] = dp[i-1][j];
                    else
                        dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]];
                }
            }
            return dp[nums.size()][neg];
        }
    };
    ```

    二维的数组还可以优化成一维的。

#### 最后一块石头的重量 II

有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。

每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。

示例 1：

```
输入：stones = [2,7,4,1,8,1]
输出：1
解释：
组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。
```

分析：

相当于在石子重量前加上正负号，选择的负号的石子的重量之和在不超过`sum/2`的情况下，尽量地大。这样就转化成了目标和问题，或 0-1 背包问题。

代码：

```c++
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int sum = accumulate(stones.begin(), stones.end(), 0);
        int neg = sum / 2;
        vector<vector<bool>> dp(stones.size()+1, vector<bool>(neg+1, false));
        dp[0][0] = true;

        for (int i = 1; i <= stones.size(); ++i)
        {
            for (int j = 0; j <= neg; ++j)
            {
                if (j < stones[i-1])
                    dp[i][j] = dp[i-1][j];
                else
                    dp[i][j] = dp[i-1][j] || dp[i-1][j - stones[i-1]];
            }
        }

        for (int j = neg; j > -1; --j)
        {
            if (dp[stones.size()][j]) return sum - 2 * j;
        }
        return 0;
    }
};
```

#### 零钱兑换

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

 
```
示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：

输入：coins = [2], amount = 3
输出：-1
```

代码：

1. 动态规划。设`dp[i]`为凑够`i`元最少需要的硬币数。

    ```c++
    class Solution {
    public:
        int coinChange(vector<int>& coins, int amount) {
            vector<int> dp(amount+1, INT32_MAX - 10);  // 后面有 INT32_MAX + 1，为了防止溢出，这里先减个 10
            dp[0] = 0;  // 初始条件，凑 0 元不需要硬币
            for (int i = 1; i <= amount; ++i)
                for (int j = 0; j < coins.size(); ++j)
                    if (coins[j] <= i)
                        dp[i] = min(dp[i], dp[i-coins[j]]+1);
            return dp[amount] == INT32_MAX-10 ? -1 : dp[amount];
        }
    };
    ```

    以前都是把物件的个数作为第一层循环，为什么这里把`target`作为第一层循环了呢？为什么一定需要两层循环才行？

#### 三角形最小路径和

给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

 
```
示例 1：

输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
示例 2：

输入：triangle = [[-10]]
输出：-10
```

代码：

1. 从上而下的动态规划

    ```c++
    class Solution {
    public:
        int minimumTotal(vector<vector<int>>& triangle) {
            vector<vector<int>> dp(triangle.size(), vector<int>(triangle.size()));  // 每个位置的最小路径和都得算一遍
            dp[0][0] = triangle[0][0];
            for (int i = 1; i < triangle.size(); ++i)
            {
                dp[i][0] = dp[i-1][0] + triangle[i][0];  // 下一行的第一个元素和最后一个元素只有一种选择
                dp[i][i] = dp[i-1][i-1] + triangle[i][i];
                for (int j = 1; j < i; ++j)
                    dp[i][j] = min(dp[i-1][j-1] + triangle[i][j], dp[i-1][j] + triangle[i][j]);
            }

            int res = INT32_MAX;
            for (auto &num: dp.back())
                res = min(res, num);
            return res;
        }
    };
    ```

    这道题还可以从给定的`triangle`上进行状态转移，从而不使用额外的空间。

1. 后来又写的，和以前写的差不多：

    ```c++
    class Solution {
    public:
        int minimumTotal(vector<vector<int>>& triangle) {
            vector<vector<int>> dp;
            dp.push_back({triangle[0][0]});
            for (int i = 1; i < triangle.size(); ++i)
            {
                dp.push_back(vector<int>(triangle[i].size()));
                for (int j = 0; j < triangle[i].size(); ++j)
                {
                    if (j == 0)
                        dp[i][j] = dp[i-1][j] + triangle[i][j];
                    else if (j == triangle[i].size() - 1)
                        dp[i][j] = dp[i-1][j-1] + triangle[i][j];
                    else
                        dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j]; 
                }
            }
            return *min_element(dp.back().begin(), dp.back().end());
        }
    };
    ```

1. 后来又写的，其实可以直接在原数组上修改

    ```cpp
    class Solution {
    public:
        int minimumTotal(vector<vector<int>>& triangle) {
            int m = triangle.size();
            int n;
            for (int i = 1; i < m; ++i)
            {
                n = triangle[i].size();
                triangle[i][0] = triangle[i-1][0] + triangle[i][0];
                triangle[i][n-1] = triangle[i-1][n-2] + triangle[i][n-1];
                for (int j = 1; j < n - 1; ++j)
                    triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j]);
            }
            int ans = INT32_MAX;
            for (int i = 0; i < triangle[m-1].size(); ++i)
                ans = min(ans, triangle[m-1][i]);
            return ans;
        }
    };
    ```

    对于三角形中的每个位置`[i, j]`，到达它只有从上一层的`[i-1, j-1]`和`[i-1, j]`两个节点。所以到达当前节点的最小值，就是从上一层两个位置中选择一个。

    另外我认为头节点和尾节点单独处理比较好，不要放在循环里用`if`处理。

    现在问题来了，假如要输出最小和的 path，该怎么输出呢？

#### 最长回文子串

给你一个字符串 s，找到 s 中最长的回文子串。

示例 1：

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：

输入：s = "cbbd"
输出："bb"

代码：

1. 动态规划。好理解，但耗时长，占空间大，效率低

    `dp[pos][len]`表示从`s[pos]`开始，长度为`len`的一个字符串是否为回文串。

    ```c++
    class Solution {
    public:
        string longestPalindrome(string s) {
            int max_len = 1, p1 = 0;

            vector<vector<bool>> dp(s.size()+1, vector<bool>(s.size()+1));
            for (int i = 0; i < s.size(); ++i)  // 对长度为 1 和 2 的情况单独判断
            {
                dp[i][1] = 1;
                if (i > 0 && s[i-1] == s[i])
                {
                    dp[i-1][2] = true;
                    max_len = 2;
                    p1 = i-1;
                }
            }
    
            for (int len = 3; len <= s.size(); ++len)
            {
                for (int i = 0; i < s.size(); ++i)
                {
                    // 如果右边界没超过边界，且向内缩进的字符串也为回文串，且当前子串的两端字符相等
                    if (i+len-1 < s.size() && dp[i+1][len-2] && s[i] == s[i+len-1])
                    {
                        dp[i][len] = true;
                        if (len > max_len)
                        {
                            max_len = len;
                            p1 = i;
                        }
                    }
                    else dp[i][len] = false;
                }
            }
            return s.substr(p1, max_len);
        }
    };
    ```

    几个月后又写的，一点长进都没有。需要记住很多时间动态规划外层的循环是长度，内层的循环是起始点。

    ```c++
    class Solution {
    public:
        string longestPalindrome(string s) {
            vector<vector<bool>> dp(s.size(), vector<bool>(s.size()+1));
            int max_len = 1, start = 0;

            for (int i = 0; i < s.size(); ++i)
            {
                dp[i][1] = true;
                if (i < s.size() - 1 && s[i] == s[i+1])
                {
                    start = i;
                    max_len = 2;
                    dp[i][2] = true;
                }
            }

            for (int len = 3; len <= s.size(); ++len)
            {
                for (int pos = 0; pos + len <= s.size(); ++pos)
                {
                    if (dp[pos+1][len-2] && s[pos] == s[pos+len-1])
                    {
                        dp[pos][len] = true;
                        if (len > max_len)
                        {
                            max_len = len;
                            start = pos;
                        }
                    }
                }
                
            }
            return s.substr(start, max_len);
        }
    };
    ```

    用`dp[p1][p2]`表示子串`s[p1:p2]`为回文串的另一种写法：

    ```c++
    class Solution {
    public:
        string longestPalindrome(string s) {
            vector<vector<bool>> dp(s.size(), vector<bool>(s.size()));  // dp[i][j] 表示 s[i:j] 为回文串
            int pos = 0, max_len = 1;
            for (int i = 0; i < s.size(); ++i)
            {
                dp[i][i] = true;
                if (i > 0 && s[i-1] == s[i])
                {
                    dp[i-1][i] = true;
                    pos = i - 1;
                    max_len = 2;
                }
            }

            int j;
            for (int len = 3; len <= s.size(); ++len)
            {
                for (int i = 0; i < s.size() - len + 1; ++i)  // 只计算一斜半的矩阵，从而将时间缩短一半
                {
                    j = i + len - 1;
                    if (dp[i+1][j-1] && s[i] == s[j])
                    {
                        dp[i][j] = true;
                        if (len > max_len)
                        {
                            max_len = len;
                            pos = i;
                        } 
                    }
                }
            }

            return s.substr(pos, max_len);
        }
    };
    ```

1. 中心扩散

    官方给出的题解：

    ```c++
    class Solution {
    public:
        pair<int, int> expandAroundCenter(const string& s, int left, int right) {
            while (left >= 0 && right < s.size() && s[left] == s[right]) {
                --left;
                ++right;
            }
            return {left + 1, right - 1};
        }

        string longestPalindrome(string s) {
            int start = 0, end = 0;
            for (int i = 0; i < s.size(); ++i) {
                auto [left1, right1] = expandAroundCenter(s, i, i);
                auto [left2, right2] = expandAroundCenter(s, i, i + 1);
                if (right1 - left1 > end - start) {
                    start = left1;
                    end = right1;
                }
                if (right2 - left2 > end - start) {
                    start = left2;
                    end = right2;
                }
            }
            return s.substr(start, end - start + 1);
        }
    };
    ```

    一份之前自己写的代码，效率很高，但是挺复杂的：

    ```c++
    class Solution {
    public:
        string longestPalindrome(string s)
        {
            if (s.size() < 2)
            {
                return s;
            }

            int pal_begin = 0, pal_len = 0;    

            bool even = false;
            int right;
            for (int mid = 0; mid < s.size(); ++mid)
            {
                even = false;

                // 长度为 1
                if (pal_len < 1)
                {
                    pal_begin = mid;
                    pal_len = 1;
                }
                
                // 长度为 2
                if (s[mid] == s[mid+1])
                {
                    even = true;
                    if (pal_len < 2)
                    {
                        pal_begin = mid;
                        pal_len = 2;
                    }
                }

                // 长度大于等于 3
                if (even)  // 长度为偶数
                {
                    for (int left = mid - 1; left > -1; --left)
                    {
                        right = mid + mid - left + 1;
                        if (right >= s.size())
                            break;

                        if (s[left] == s[right])
                        {
                            if (right - left + 1 > pal_len)
                            {
                                pal_begin = left;
                                pal_len = right - left + 1;
                            }
                        }
                        else
                        {
                            break;
                        }
                        
                    }
                }
                
                // 长度为奇数
                for (int left = mid - 1; left > -1; --left)
                {
                    right = mid + mid - left;
                    if (right >= s.size())
                        break;
                    if (s[left] == s[right])
                    {
                        if (right - left + 1 > pal_len)
                        {
                            pal_begin = left;
                            pal_len = right - left + 1;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                
            }
            return s.substr(pal_begin, pal_len);
        }
    };
    ```

#### 最长回文子序列

给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

```
示例 1：

输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
示例 2：

输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。
```

代码：

1. 动态规划，自己的实现，以长度`l`从短到长遍历。`dp[i][j]`表示从`i`开始到`j`结束的子串的最长回文子序列的长度。

    ```c++
    class Solution {
    public:
        int longestPalindromeSubseq(string s) {
            vector<vector<int>> dp(s.size(), vector<int>(s.size(), 1));
            int j;
            for (int l = 2; l <= s.size(); ++l)
            {
                for (int i = 0; i < s.size() - l + 1; ++i)
                {
                    j = i + l - 1;
                    if (s[i] == s[j])
                    {
                        if (l == 2) dp[i][j] = 2;
                        else dp[i][j] = dp[i+1][j-1] + 2;
                    }
                    else
                    {
                        dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
                    }
                }
            }
            return dp[0][s.size()-1];
        }
    };
    ```

1. 官方给的动态规划的倒序实现。挺有意思的，有时间研究一下

    ```c++
    class Solution {
    public:
        int longestPalindromeSubseq(string s) {
            int n = s.length();
            vector<vector<int>> dp(n, vector<int>(n));
            for (int i = n - 1; i >= 0; i--) {
                dp[i][i] = 1;
                char c1 = s[i];
                for (int j = i + 1; j < n; j++) {
                    char c2 = s[j];
                    if (c1 == c2) {
                        dp[i][j] = dp[i + 1][j - 1] + 2;
                    } else {
                        dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                    }
                }
            }
            return dp[0][n - 1];
        }
    };
    ```

1. 评论里给出的答案，神奇的思路，只需要将字符串翻转，然后求两个字符串的 lcs 就可以了

    ```python
    class Solution(object):
        def longestPalindromeSubseq(self, s):
            """
            :type s: str
            :rtype: int
            """
            n = len(s)
            dp = [[0] * (n + 1) for i in xrange(n + 1)]
            ss = s[::-1]

            for i in xrange(1, n + 1):
                for j in xrange(1, n + 1):
                    if s[i - 1] == ss[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
            
            return dp[n][n]
    ```

#### 不同路径

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

 
```
示例 1：


输入：m = 3, n = 7
输出：28
示例 2：

输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
示例 3：

输入：m = 7, n = 3
输出：28
示例 4：

输入：m = 3, n = 3
输出：6
```

代码：

1. 动态规划

    这道题是“礼物的最大价值”的简单版本。注意边界条件，要设置为 1。

    ```c++
    class Solution {
    public:
        int uniquePaths(int m, int n) {
            vector<vector<int>> dp(m, vector<int>(n));  // dp[i][j] 表示从起点到 (i, j) 的路径数
            int i, j;
            i = 0;
            for (j = 0; j < n; ++j) dp[i][j] = 1;
            j = 0;
            for (i = 0; i < m; ++i) dp[i][j] = 1;

            for (int i = 1; i < m; ++i)
            {
                for (int j = 1; j < n; ++j)
                {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }

            return dp[m-1][n-1];
        }
    };
    ```

#### 等差数列划分

如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。

例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。

子数组 是数组中的一个连续序列。

 
```
示例 1：

输入：nums = [1,2,3,4]
输出：3
解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。
示例 2：

输入：nums = [1]
输出：0
```

代码：

1. 动态规划（能通过，但效率很低）

    ```c++
    class Solution {
    public:
        int numberOfArithmeticSlices(vector<int>& nums) {
            vector<vector<bool>> dp(nums.size(), vector<bool>(nums.size()));  // dp[i][j] 表示 nums[i:j] 为等差数列
            int ans = 0;
            int j;
            int diff;
            for (int len = 1; len <= nums.size(); ++len)
            {
                for (int i = 0; i < nums.size() - len + 1; ++i)
                {
                    j = i + len - 1;
                    if (len < 3)
                    {
                        dp[i][j] = true;
                    }
                    else
                    {
                        diff = nums[i+2] - nums[i+1];
                        if (dp[i+1][j-1] && nums[i+1] - nums[i] == diff && nums[j] - nums[j-1] == diff)
                        {
                            dp[i][j] = true;
                            ++ans;
                        }
                    }
                }
            }
            return ans;
        }
    };
    ```

    后来又写的动态规划：

    ```c++
    class Solution {
    public:
        int numberOfArithmeticSlices(vector<int>& nums) {
            int n = nums.size();
            vector<vector<bool>> dp(n, vector<bool>(n));
            int ans = 0;
            for (int i = 2; i < n; ++i)
            {
                if (nums[i] - nums[i-1] == nums[i-1] - nums[i-2])
                    dp[i-2][i] = true, ++ans;;
            }

            for (int len = 4; len <= n; ++len)
            {
                for (int i = 0; i < n - len + 1; ++i)
                {
                    if (dp[i][i+len-2] && nums[i+len-1] - nums[i+len-2] == nums[i+len-2] - nums[i+len-3])
                        dp[i][i+len-1] = true, ++ans;
                }
            }
            return ans;
        }
    };
    ```

1. 差分计数，找规律（其实类似滑动窗口）

    长度为 3 的差分数组有 1 种结果，长度为 4 的差分数组有 3 种结果，长度为为 5 的差分数组有 6 种结果。可以观察出来结果的差分是 1, 2, 3，以长度为 1 递增。

    ```c++
    class Solution {
    public:
        int numberOfArithmeticSlices(vector<int>& nums) {
            int n = nums.size();
            if (n == 1) {
                return 0;
            }

            int d = nums[0] - nums[1], t = 0;
            int ans = 0;
            // 因为等差数列的长度至少为 3，所以可以从 i=2 开始枚举
            for (int i = 2; i < n; ++i) {
                if (nums[i - 1] - nums[i] == d) {
                    ++t;
                }
                else {
                    d = nums[i - 1] - nums[i];
                    t = 0;
                }
                ans += t;
            }
            return ans;
        }
    };
    ```

#### 解码方法

一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

'A' -> 1
'B' -> 2
...
'Z' -> 26
要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

"AAJF" ，将消息分组为 (1 1 10 6)
"KJF" ，将消息分组为 (11 10 6)
注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

题目数据保证答案肯定是一个 32 位 的整数。

```
示例 1：

输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2：

输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
示例 3：

输入：s = "0"
输出：0
解释：没有字符映射到以 0 开头的数字。
含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
示例 4：

输入：s = "06"
输出：0
解释："06" 不能映射到 "F" ，因为字符串含有前导 0（"6" 和 "06" 在映射中并不等价）。
```

代码：

1. 动态规划

    ```c++
    class Solution {
    public:
        int numDecodings(string s) {
            if (s[0] == '0') return 0;
            vector<int> dp(s.size()+1);
            dp[0] = 1;
            for (int i = 1; i <= s.size(); ++i)
            {
                if (i > 1 && s[i-1] == '0')  // 当前位是 0，无法解码，只能解码两位
                {
                    if (s[i-2] == '1' || s[i-2] == '2') dp[i] = dp[i-2];
                    else return 0;  // 这里要注意，显然 "90", "30" 这些都是无法翻译的
                }
                else if (i > 1 && (s[i-2] == '1' || (s[i-2] == '2' && s[i-1] <= '6')))  // 当前位与前面位的范围在 11 ~ 26 之间，两种解码方式都可以
                    dp[i] = dp[i-1] + dp[i-2];
                else  // 只能解码当前位
                    dp[i] = dp[i-1];
            }
            return dp[s.size()];
        }
    };
    ```

    后来自己又写的版本：

    ```c++
    class Solution {
    public:
        int numDecodings(string s) {
            if (s[0] == '0') return 0;  //  题目中说可能有先导 0，所以直接排除这种情况
            vector<int> dp(s.size()+1);  // 其实没必要用 n + 1，因为我们完全可以认为 dp[i] 代表当字符串以 s[i] 结尾时，可解码的数量。这里用 n + 1，纯粹是为了利用初始值是 1
            dp[0] = 1;  // 设想 "12" 这个字符串，既可以解码成 ['1', '2']，也可以解码成 ["12"]，共两种方法。即 dp[2] = dp[1] + dp[0];，显然这里 dp[0] 应该为 1 才对。所以就手动添加了这个初值。
            dp[1] = 1;  // 直接把第一位也写了，因为后面存在 i - 2 的情况。这样 i 就可以从 2 开始了。
            for (int i = 2; i <= s.size(); ++i)
            {
                if (s[i-1] == '0' && (s[i-2] == '0' || s[i-2] >= '3')) return 0;  // 遇到连续的 0，或 30, 40 这样的数字，说明本字符串无法解码，直接返回 0
                if (s[i-1] == '0' && ((s[i-2] == '1') || (s[i-2] == '2')))  // 若当前字符为 0，那么当上一个字符是 1 或 2 时，只能把上个字符和当前字符作为一个整体解码
                    dp[i] = dp[i-2];
                else if (s[i-2] == '0')  // 如果上个字符为 0，那么当前字符（不可能是 0，前面已经排除过了）无论是几，都只有一种解码方式，那就是和上个字符分开解码
                    dp[i] = dp[i-1];
                else if (s[i-2] == '1' || (s[i-2] == '2' && '1' <= s[i-1] && s[i-1] <= '6'))  // 11 - 19, 21 - 26 这些情况，既可以合起来解码，也可以分开解码
                    dp[i] = dp[i-2] + dp[i-1];
                else  // 剩下的各种情况，比如大于等于 27
                    dp[i] = dp[i-1];
            }
            return dp.back();
        }
    };
    ```

    答案给的版本：

    ```c++
    class Solution {
    public:
        int numDecodings(string s) {
            int n = s.size();
            vector<int> f(n + 1);
            f[0] = 1;
            for (int i = 1; i <= n; ++i) {
                if (s[i - 1] != '0') {  // 这样分两个 if 写很巧妙，有空了学习一下
                    f[i] += f[i - 1];
                }
                if (i > 1 && s[i - 2] != '0' && ((s[i - 2] - '0') * 10 + (s[i - 1] - '0') <= 26)) {
                    f[i] += f[i - 2];
                }
            }
            return f[n];
        }
    };
    ```

    问题：当遇到复杂情况需要分类讨论时，如何才能不重不漏，而又使代码简洁？

1. 优化过的动态规划

    ```c++
    class Solution {
    public:

        int numDecodings(string s) {
            int f_n = 0, f_n_minus_1 = 1, f_n_minus_2;

            for (int i = 1; i <= s.size(); ++i)
            {
                f_n = 0;
                if (s[i - 1] != '0')
                {
                    f_n += f_n_minus_1;
                }

                if (i > 1 && s[i - 2] != '0' && stoi(s.substr(i-2, 2)) <= 26)
                {
                    f_n += f_n_minus_2;
                }

                f_n_minus_2 = f_n_minus_1;
                f_n_minus_1 = f_n;
            }

            return f_n;
        }
    };
    ```

#### 单词拆分

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

```
说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
示例 3：

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

代码：

1. 动态规划

    这道题算比较简单的经典题目了。最优子结构也非常清晰。

    ```c++
    class Solution {
    public:
        bool wordBreak(string s, vector<string>& wordDict) {
            unordered_set<string> m(wordDict.begin(), wordDict.end());
            vector<bool> dp(s.size()+1);
            dp[0] = true;

            for (int i = 1; i <= s.size(); ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    if (dp[j] && m.find(s.substr(j, i-j)) != m.end())  // 将字符串 s[0:i] 切割成两段，查找这两段是否都在 wordDict 中出现
                    {
                        dp[i] = true;
                        break;
                    }
                }
            }

            return dp.back();
        }
    };
    ```

1. 后来又写的

    ```c++
    class Solution {
    public:
        bool wordBreak(string s, vector<string>& wordDict) {
            unordered_set<string> words;
            for (auto &str: wordDict) words.insert(str);
            int n = s.size();
            vector<bool> dp(n);  // dp[i] 表示以 s[i] 结尾的字符串可以被拆分。
            for (int i = 0; i < n; ++i)
            {
                for (int j = -1; j <= i; ++j)
                {
                    if (j == -1)
                    {
                        if (words.find(s.substr(0, i+1)) != words.end())
                        {
                            dp[i] = true;
                            break;
                        } 
                    }
                    else 
                    {
                        if (dp[j] && words.find(s.substr(j+1, i-j)) != words.end())
                        {
                            dp[i] = true;
                            break;
                        }
                    }
                }
            }
            return dp[n-1];
        }
    };
    ```

    显然这次写的麻烦了许多。该怎样才能统一起来呢？

#### 最长递增子序列的个数

给定一个未排序的整数数组，找到最长递增子序列的个数。

```
示例 1:

输入: [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
示例 2:

输入: [2,2,2,2,2]
输出: 5
解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。
```

代码：

动态规划。

```c++
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        vector<int> dp(nums.size(), 1);
        vector<int> count(nums.size(), 1);
        int max_len = 1, ans = 0;
        for (int i = 1; i < nums.size(); ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                if (nums[i] > nums[j])
                {
                    if (dp[j] + 1 > dp[i])  // 若最长长度更新，则更改计数基准
                    {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    }
                    else if (dp[j] + 1 == dp[i])  // 若最长长度没更新，那么继续计数
                    {
                        count[i] += count[j];
                    }
                }
            }
            max_len = max(dp[i], max_len);
        }

        for (int i = 0; i < dp.size(); ++i)
        {
            if (dp[i] == max_len)
            {
                ans += count[i];
            }
        }
        return ans;
    }
};
```

#### 乘积最大子数组

给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

```
示例 1:

输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
示例 2:

输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

代码：

1. 动态规划

    如果前面的乘积是个最小的负数，那么它在当前一步就可能再乘个负数变成最大的正数。

    如题前面的乘积是最大的正数，那么它也有可能变成最小的负数乘积。

    （其实我也不是很懂）

    ```c++
    class Solution {
    public:
        int maxProduct(vector<int>& nums) {
            vector<int> mins(nums.size()), maxs(nums.size());
            mins[0] = nums[0];
            maxs[0] = nums[0];
            for (int i = 1; i < nums.size(); ++i)
            {
                mins[i] = min(mins[i-1] * nums[i], min(nums[i], maxs[i-1] * nums[i]));
                maxs[i] = max(maxs[i-1] * nums[i], max(nums[i], mins[i-1] * nums[i]));
            }
            return *max_element(maxs.begin(), maxs.end());
        }
    };
    ```

#### 乘积为正数的最长子数组长度

给你一个整数数组 nums ，请你求出乘积为正数的最长子数组的长度。

一个数组的子数组是由原数组中零个或者更多个连续数字组成的数组。

请你返回乘积为正数的最长子数组长度。

 
```
示例  1：

输入：nums = [1,-2,-3,4]
输出：4
解释：数组本身乘积就是正数，值为 24 。
示例 2：

输入：nums = [0,1,-2,-3,-4]
输出：3
解释：最长乘积为正数的子数组为 [1,-2,-3] ，乘积为 6 。
注意，我们不能把 0 也包括到子数组中，因为这样乘积为 0 ，不是正数。
示例 3：

输入：nums = [-1,-2,-3,0,1]
输出：2
解释：乘积为正数的最长子数组是 [-1,-2] 或者 [-2,-3] 。
示例 4：

输入：nums = [-1,2]
输出：1
示例 5：

输入：nums = [1,2,3,5,-6,4,0,10]
输出：4
```

代码：

1. 动态规划

    维护两个数组`pos`，`neg`，一个用来记录正数乘积的最大长度，一个用来记录负数乘积的最大长度。

    当一个数为正数时，`pos`肯定会加一，`neg`得看之前的长度是否为 0，如果为 0，那正数没法让`neg`长度加一，如果不为 0，那么正数不会改变之前乘积的正负性，所以`neg`加一。

    当一个数为负数时，之前乘积的正负性改变，具体的分析类似同上。

    当一个数为 0 时，正数乘积长度和负数乘积长度都清零。

    问题：什么样的情况下才需要用到双数组甚至多数组呢？`pos`和`neg`是如何保证结果的正确性的呢？

    ```c++
    class Solution {
    public:
        int getMaxLen(vector<int>& nums) {
            vector<int> pos(nums.size()), neg(nums.size());
            pos[0] = nums[0] > 0 ? 1 : 0;
            neg[0] = nums[0] < 0 ? 1 : 0;
            int ans = pos[0];
            for (int i = 1; i < nums.size(); ++i)
            {
                if (nums[i] > 0)
                {
                    pos[i] = pos[i-1] + 1;
                    neg[i] = neg[i-1] == 0 ? 0 : neg[i-1] + 1;
                }
                else if (nums[i] < 0)
                {
                    pos[i] = neg[i-1] == 0 ? 0 : neg[i-1] + 1;
                    neg[i] = pos[i-1] + 1;
                }
                else
                {
                    pos[i] = 0;
                    neg[i] = 0;
                }
                ans = max(ans, pos[i]);
            }
            return ans;
        }
    };
    ```

#### 分割等和子集

给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```
示例 1：

输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
示例 2：

输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

代码：

1. 动态规划。将问题转化为 0-1 背包问题。

    我们可以取`target`为总和的一半，使用动态规划计算是否能从数组中挑出一些数字组成和为`target`的子集。

    ```c++
    class Solution {
    public:
        bool canPartition(vector<int>& nums) {
            int sum = accumulate(nums.begin(), nums.end(), 0);
            if (sum % 2 != 0) return false;

            int target = sum / 2;
            vector<vector<bool>> dp(nums.size()+1, vector<bool>(target+1));
            for (int i = 0; i <= nums.size(); ++i)
                dp[i][0] = true;
            for (int i = 1; i <= nums.size(); ++i)
            {
                for (int j = 1; j <= target; ++j)
                {
                    if (nums[i-1] <= j)
                        dp[i][j] = dp[i-1][j] | dp[i-1][j-nums[i-1]];  // 这样其实相当于 dp[i][j] = max(dp[i-1][j], dp[i-1][j-nums[i-1]]);
                    else
                        dp[i][j] = dp[i-1][j];
                }
            }
            return dp[nums.size()][target];
        }
    };
    ```

    由于在求`dp[i][j]`时只用到了`dp[i-1][j]`，所以还可以对数组做压缩，以节省空间。

#### 完全平方数

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

 
```
示例 1：

输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
示例 2：

输入：n = 13
输出：2
解释：13 = 4 + 9
```

代码：

1. 动态规划

    ```c++
    class Solution {
    public:
        int numSquares(int n) {
            vector<int> dp(n+1, INT_MAX);
            dp[0] = 0;
            for (int i = 1; i <= n; ++i)
                for (int j = 1; j * j <= i; ++j)
                    dp[i] = min(dp[i], dp[i-j*j]+1);
            return dp[n];
        }
    };
    ```

1. 四平方和定理

    太难了。

或许这道题还可以试试贪心，dfs？

#### 最佳买卖股票时机含冷冻期

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

```
示例:

输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

1. 动态规划

    设状态 0 是持有股票，状态 1 是未持有股票且处于冷冻期，状态 2 是未持有股票且未处于冷冻期。

    `dp[0][i]`表示处于状态 0 时，第 i 天所能得到的最大利润，`dp[1][i]`与`dp[2][i]`类似。

    当第 i 天处于状态 0 时，其上一个状态要么是仍持有股票（状态 0），要么是第 i 天刚刚买入，即之前处于状态 2。

    当第 i 天处于状态 1 时，说明第 i 天卖了股票，那么其上个状态一定是状态 0。

    当第 i 天处于状态 2 时，说明要么之前仍是状态 2，要么之前处于冷冻期（状态 1）。

    依照以上几种状态列出动态规划即可。

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices) {
            vector<int> dp[3];
            for (int i = 0; i < 3; ++i) dp[i].resize(prices.size());
            dp[0][0] = -prices[0];
            dp[1][0] = 0;  // 这个初始值是如何定的？
            dp[2][0] = 0;
            for (int i = 1; i < prices.size(); ++i)
            {
                dp[0][i] = max(dp[0][i-1], dp[2][i-1] - prices[i]);
                dp[1][i] = dp[0][i-1] + prices[i];
                dp[2][i] = max(dp[2][i-1], dp[1][i-1]);
            }
            return max(dp[1].back(), dp[2].back());
        }
    };
    ```

    在定义状态的时候有一个细节：所有的状态定义都是当天过完后处于什么状态，这样状态 2 和状态 3 的后续推理才不矛盾。那么我们是否可以定义当天开始时是什么状态？这样代码又会有什么变动，还能正确吗？

    什么时候才需要定义多个数组及状态来做动态规划？需要定义几个状态？

    之前的股票问题也可以用这样的动态规划做吗？

1. 回溯

    ```java
    public int maxProfit(int[] prices) {
            if (prices == null || prices.length == 0) return 0;
            int res = 0;
            int len = prices.length;
            // dp[i][0] : 持有股票
            // dp[i][1] : 不持有股票，本日卖出，下一天为冷冻期
            // dp[i][2] : 不持有股票，本日无卖出，下一天非冷冻期
            int[][] dp = new int[len][3];

            //初始状态:
            // 1: 第一天持有，只有可能是买入
            dp[0][0] = -prices[0];

            // 其实dp[0][1]、dp[0][2] 都不写，默认为0也对
            // 2. 第0天，相当于当天买入卖出，没有收益，并造成下一天冷冻期，减少选择。综合认为是负收益
            dp[0][1] = Integer.MIN_VALUE;

            // 3. 相当于没买入，收益自然为0
            dp[0][2]=0;

            for (int i = 1; i < len; i++) {
                // 持有股票： 1.昨天持有股票 2.本日买入（条件：昨天不持有，且不是卖出）
                dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][2] - prices[i]);

                //本日卖出 : (条件:昨天持有)
                dp[i][1] = dp[i - 1][0] + prices[i];

                // 不持有，但也不是卖出 : 1.昨天卖出，不持有  2.昨天没卖出，但也不持有
                dp[i][2] = Math.max(dp[i - 1][1], dp[i - 1][2]);
            }

            // 最后一天还持有股票是没有意义的,肯定是不持有的收益高，不用对比 dp[len-1][0]
            return Math.max(dp[len - 1][1], dp[len - 1][2]);
        }
    ```

#### 买卖股票的最佳时机含手续费

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

 
```
示例 1：

输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
输出：8
解释：能够达到的最大利润:  
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8


示例 2：

输入：prices = [1,3,7,5,10,3], fee = 3
输出：6
```

代码：

1. 动态规划

    设状态 0 是第 i 天结束后，手里没有股票；状态 1 是第 i 天结束后，手里有股票。

    `dp[i][0]`和`dp[i][1]`表示不同状态下，第 i 天结束后，所能取得的最大收益。

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices, int fee) {
            vector<vector<int>> dp(prices.size(), vector<int>(2));
            dp[0][0] = 0;
            dp[0][1] = -prices[0];
            for (int i = 1; i < prices.size(); ++i)
            {
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee);
                dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]);
            }
            return max(dp.back()[0], dp.back()[1]);
        }
    };
    ```

1. 贪心

    官方还给出了贪心解法。有时间了再看。

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices, int fee) {
            int n = prices.size();
            int buy = prices[0] + fee;
            int profit = 0;
            for (int i = 1; i < n; ++i) {
                if (prices[i] + fee < buy) {
                    buy = prices[i] + fee;
                }
                else if (prices[i] > buy) {
                    profit += prices[i] - buy;
                    buy = prices[i];
                }
            }
            return profit;
        }
    };
    ```

#### 不同的二叉搜索树

给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

 
```
示例 1：


输入：n = 3
输出：5
示例 2：

输入：n = 1
输出：1
```

代码：

1. 动态规划

    假如现在有`[1, 2, ..., n]` n 个数，我们从中取一个数作为根节点，这个数左边的数组用于构造左子树，右边的数组用于构造右子树。这样按数目划分，可以把左子树和右子树不遗漏地遍历一遍。对于左子树和右子树，则递归地计数。

    设`dp[i]`表示`i`个节点的树的结构，对于当前数组，每个数都有可能被选作根节点，因为需要对当前数组进行遍历。

    ```c++
    class Solution {
    public:
        int numTrees(int n) {
            vector<int> dp(n+1);
            dp[0] = 1;  // 当取数组的第一个数或最后一个数作为根节点时，总有一棵子树的的节点数为 0
            dp[1] = 1;
            for (int i = 2; i <= n; ++i)
            {
                for (int j = 1; j <= i; ++j)
                {
                    dp[i] += dp[j-1] * dp[i-j];
                }
            }
            return dp[n];
        }
    };
    ```

    这道题不能想成每增加一个节点，树上就少一个位置，同时会多出两个位置，每次只需要在空余位置上选一个就可以了。这种想法会造成重复计数，比如`[1,2,3]`和`[1,3,2]`是结构相同的树。

#### 不同路径 II

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

 
```
示例 1：


输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右
示例 2：


输入：obstacleGrid = [[0,1],[0,0]]
输出：1
```

代码：

1. 动态规划，自己写的，很繁琐

    ```c++
    class Solution {
    public:
        int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
            if (obstacleGrid[0][0] == 1) return 0;
            int m = obstacleGrid.size(), n = obstacleGrid[0].size();
            vector<vector<int>> dp(m, vector<int>(n));
            dp[0][0] = 1;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == 0 && j == 0) continue;
                    if (obstacleGrid[i][j] == 1) continue;
                    if (i == 0 && j > 0)
                    {
                        if (obstacleGrid[i][j-1] == 1) dp[i][j] = 0;
                        else dp[i][j] = dp[i][j-1];
                        continue;
                    }
                    if (j == 0 && i > 0)
                    {
                        if (obstacleGrid[i-1][j] == 1) dp[i][j] = 0;
                        else dp[i][j] = dp[i-1][j];
                        continue;
                    }
                    if (obstacleGrid[i-1][j] && !obstacleGrid[i][j-1])
                    {
                        dp[i][j] = dp[i][j-1];
                        continue;
                    }
                    if (!obstacleGrid[i-1][j] && obstacleGrid[i][j-1])
                    {
                        dp[i][j] = dp[i-1][j];
                        continue;
                    }
                    if (i > 0 && j > 0)
                        dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
            return dp[m-1][n-1];
        }
    };
    ```

1. 动态规划 官方给的解答

    ```c++
    class Solution {
    public:
        int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
            int n = obstacleGrid.size(), m = obstacleGrid.at(0).size();
            vector <int> f(m);

            f[0] = (obstacleGrid[0][0] == 0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (obstacleGrid[i][j] == 1) {
                        f[j] = 0;
                        continue;
                    }
                    if (j - 1 >= 0 && obstacleGrid[i][j - 1] == 0) {
                        f[j] += f[j - 1];
                    }
                }
            }

            return f.back();
        }
    };
    ```

## 贪心

### 雪糕的最大数量

夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。

商店中新到 n 支雪糕，用长度为 n 的数组 costs 表示雪糕的定价，其中 costs[i] 表示第 i 支雪糕的现金价格。Tony 一共有 coins 现金可以用于消费，他想要买尽可能多的雪糕。

给你价格数组 costs 和现金量 coins ，请你计算并返回 Tony 用 coins 现金能够买到的雪糕的 最大数量 。

注意：Tony 可以按任意顺序购买雪糕。

示例 1：

```
输入：costs = [1,3,2,4,1], coins = 7
输出：4
解释：Tony 可以买下标为 0、1、2、4 的雪糕，总价为 1 + 3 + 2 + 1 = 7
```

分析：虽然可以把它看作是所有的物品价值为 1 的 01 背包问题，但是这样效率并不高。我们只需要从最便宜的雪糕买起就可以了。


代码：

1. 快速排序

    ```c++
    class Solution {
    public:
        int maxIceCream(vector<int>& costs, int coins) {
            sort(costs.begin(), costs.end());
            int count = 0;
            for (auto &cost: costs)
            {
                if (cost <= coins)
                {
                    ++count;
                    coins -= cost;
                }
                else
                    break;
            }
            return count;
        }
    };
    ```

1. 计数排序

    ```c++
    class Solution {
    public:
        int maxIceCream(vector<int>& costs, int coins) {
            vector<int> v(100001);
            for (auto &cost: costs)
                v[cost]++;
            int c = 0, count = 0;
            for (int i = 1; i < 100001; ++i)
            {
                if (i <= coins)
                {
                    c = min(v[i], coins / i);
                    count += c;
                    coins -= c * i;
                }
                else
                    break;
            }
            return count;
        }
    };
    ```

### 跳跃游戏

给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

```
示例 1：

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
示例 2：

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

代码：

1. 贪心

    维护一个边界值，若当前位置超出边界值，说明不可达。

    注意这道题不需要要跳跃游戏 II 那样既维护一个边界值，又维护一个在到达边界值之前能跳跃的最远距离。因为跳跃游戏 II 需要记录跳跃次数，这道题不需要。

    ```c++
    class Solution {
    public:
        bool canJump(vector<int>& nums) {
            int pos = 0, jump = 0, bound = nums[0];
            while (pos <= bound)
            {
                jump = nums[pos];
                if (pos + jump >= nums.size() - 1) return true;
                bound = max(bound, pos + jump);
                ++pos;
            }
            return false;
        }
    };
    ```

1. 动态规划（效率较低）

    ```c++
    class Solution {
    public:
        bool canJump(vector<int>& nums) {
            vector<int> dp(nums.size()+1);  // dp[i] 表示第 i 个位置（从 1 开始计数）跳到的最远的地方
            dp[0] = 1;
            for (int i = 1; i <= nums.size(); ++i)
            {
                if (i <= dp[i-1])
                    dp[i] = max(dp[i-1], i+nums[i-1]);
            }
            if (dp[nums.size()] >= nums.size()) return true;
            return false;
        }
    };
    ```

### 跳跃游戏 II

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

示例 1:

输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

分析：贪心算法。记录一次跳跃中的边界和最远能达到的距离，每次都跳跃最远的距离即可（不一定是从落点出发）。

代码：

1. 贪心

    ```c++
    class Solution {
    public:
        int jump(vector<int>& nums) {
            int max_pos = 0;
            int max_bound = 0;  // 注意这个不是落点
            int jump_count = 0;
            for (int i = 0; i < nums.size() - 1; ++i)  // 不需要跳到最后一个位置，因为如果 max_bound 正好为 nums.size() - 1，那么会进入 if 语句块，从而 jump_count 多出来一个。
            {
                max_pos = max(max_pos, i + nums[i]);
                if (i == max_bound)
                {
                    max_bound = max_pos;
                    ++jump_count;
                }
            }
            return jump_count;
        }
    };
    ```

    后来又写的：

    ```c++
    class Solution {
    public:
        int jump(vector<int>& nums) {
            int pos = 0, temp_bound = 0, bound = 0;
            int count = 0;
            while (bound < nums.size() - 1)  // 当 bound == nums.size() - 1 时，意味着可以跳到最后一个索引
            {
                temp_bound = max(temp_bound, pos + nums[pos]);
                if (pos == bound)
                {
                    ++count;
                    bound = max(temp_bound, bound);
                }
                ++pos;
            }
            return count;
        }
    };
    ```

1. 效率很低很低的动态规划

    ```c++
    class Solution {
    public:
        int jump(vector<int>& nums) {
            vector<int> dp(nums.size(), 0);  // dp[i] 表示至少跳几次能到 i 这个位置
            for (int i = 1; i < nums.size(); ++i)
            {
                dp[i] = INT32_MAX;
                for (int j = 0; j < i; ++j)
                {
                    if (nums[j] + j >= i)
                        dp[i] = min(dp[i], dp[j] + 1);
                }
            }
            return dp.back();
        }
    };
    ```

### 买卖股票的最佳时机 II

给定一个数组 `prices` ，其中 `prices[i]` 是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 
```
示例 1:

输入: prices = [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
示例 2:

输入: prices = [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
示例 3:

输入: prices = [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

代码：

1. 贪心

    只需要把所有的上坡收集起来就好了。

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices) {
            int ans = 0;
            for (int i = 1; i < prices.size(); ++i)
                ans += max(0, prices[i] - prices[i-1]);
            return ans;
        }
    };
    ```

1. 动态规划

    用`dp[i][0]`表示第`i`天手里没股票时的最大收益，`dp[i][1]`表示第`i`天有股票时的最大收益。

    官方给出的答案，有空了再看看。

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices) {
            int n = prices.size();
            int dp[n][2];
            dp[0][0] = 0, dp[0][1] = -prices[0];
            for (int i = 1; i < n; ++i) {
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
            }
            return dp[n - 1][0];
        }
    };
    ```

### 递增的三元子序列

给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。

如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。

```
示例 1：

输入：nums = [1,2,3,4,5]
输出：true
解释：任何 i < j < k 的三元组都满足题意
示例 2：

输入：nums = [5,4,3,2,1]
输出：false
解释：不存在满足题意的三元组
示例 3：

输入：nums = [2,1,5,0,4,6]
输出：true
解释：三元组 (3, 4, 5) 满足题意，因为 nums[3] == 0 < nums[4] == 4 < nums[5] == 6
```

代码：

1. 贪心。维护一个最大值和一个次大值，如果找到比次大值还小的一个数，那么说明存在递增序列。虽然最大值可能出现在次大值的左边，但是只要次大值存在，那么说明在它的右边曾经存在过一个最大值。我们让最大值和次大值尽量的大，从而为出现递增序列腾出空间。

    ```c++
    class Solution {
    public:
        bool increasingTriplet(vector<int>& nums) {
            int max1 = INT32_MIN, max2 = INT32_MIN;
            for (int i = nums.size() - 1; i > -1; --i)
            {
                if (nums[i] < max2) return true;
                if (nums[i] >= max1) max1 = nums[i];  // 这里的等号必须取到，防止漏掉最大值进入下一个判断语句
                else if (nums[i] > max2) max2 = nums[i];
            }
            return false;
        }
    };
    ```

    展开来讲讲，为什么在 132 模式中，需要用到单调栈，这里就不需要了呢？在 132 模式那道题里，我们其实维护了`max_k`这个值，另外一个值由栈顶元素`s.top()`自然而然地得到。在这道题中，如果真的用单调栈，我们不需要记录出栈的元素，因为出栈的元素一定比当前元素小。我们只需记录所有入栈元素的最大值和最小值即可。那如果这样，还不如不用单调栈。

    这道题还可以维护`min1`和`min2`两个值，原理相同。

1. 动态规划。转换为递增子序列问题。只要找到长度为 3 的递增子序列就可以了。

    ```py
    class Solution(object):
        def increasingTriplet(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            n = len(nums)
            dp = [1] * n
            for i in range(1, n):
                for j in range(0, i):
                    if nums[j] < nums[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
                        if dp[i] == 3:
                            return True
            return False
    ```

### 划分字母区间

字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

 
```
示例：

输入：S = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
```

代码：

1. 自己写的，桶计数

    ```c++
    class Solution {
    public:
        vector<int> partitionLabels(string s) {
            vector<int> ans;
            unordered_set<char> m;
            vector<int> cnt(26);
            for (int i = 0; i < s.size(); ++i)
                ++cnt[s[i]-'a'];
            int len = 0;
            for (int i = 0; i < s.size(); ++i)
            {
                if (m.find(s[i]) == m.end()) m.insert(s[i]);
                --cnt[s[i]-'a'];
                ++len;
                if (cnt[s[i]-'a'] == 0) m.erase(s[i]);
                if (m.empty())
                {
                    ans.push_back(len);
                    len = 0;
                }
            }
            return ans;
        }
    };
    ```

1. 官方答案，记录字符最后一次出现的下标，然后使用当前片段的每个字母最后一次出现的位置更新当前片段的长度

    ```c++
    class Solution {
    public:
        vector<int> partitionLabels(string s) {
            int last[26];
            int length = s.size();
            for (int i = 0; i < length; i++) {
                last[s[i] - 'a'] = i;
            }
            vector<int> partition;
            int start = 0, end = 0;
            for (int i = 0; i < length; i++) {
                end = max(end, last[s[i] - 'a']);
                if (i == end) {
                    partition.push_back(end - start + 1);
                    start = end + 1;
                }
            }
            return partition;
        }
    };
    ```

## 数组

### 两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

```
示例 1：

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
示例 2：

输入：nums = [3,2,4], target = 6
输出：[1,2]
示例 3：

输入：nums = [3,3], target = 6
输出：[0,1]
```

代码：

1. 哈希表

    ```c++
    class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target) {
            unordered_map<int, int> m;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (m.find(target - nums[i]) != m.end())
                    return {i, m[target - nums[i]]};
                else
                    m[nums[i]] = i;
            }
            return {};
        }
    };
    ```

    注意题目中的几个要点：数组无序，有正有负有零；`target`不一定是 0；要求返回索引，而不是返回数组。

    因为要求返回索引，所以我们无法对数组进行排序后用双指针。

### 二维数组中的查找

> 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。

> 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

Example:

输入数组：

```
[
  [1,2,8,9]，
  [2,4,9,12]，
  [4,7,10,13]，
  [6,8,11,15]
]
```

如果输入查找数值为7，则返回`true`，

如果输入查找数值为5，则返回`false`。

**分析**：

从矩阵的右上角开始找，若其比 target 小，那么说明这一行都比 target 小；若其比 target 大，说明 target 就在这一行。

据解析说这种矩阵类似于二叉搜索树，所以从右上角开始搜索就相当于二叉树的搜索效率。但是怎么才能看出来这是一棵二叉搜索树呢？

代码：

```c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.empty()) return false;
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && j > -1)
        {
            if (matrix[i][j] < target) ++i;
            else if (matrix[i][j] > target)  --j;
            else return true;
        }
        return false;
    }
};
```



### 判定字符是否唯一

> 实现一个算法，确定一个字符串 s 的所有字符是否全都不同。
>
> 示例 1：
> 输入: s = "leetcode"
> 输出: false 
> 
> 示例 2：
> 输入: s = "abc"
> 输出: true

代码：

1. 散列表（很平庸的答案）

    ```c++
    class Solution {
    public:
        unordered_set<char> s;
        bool isUnique(string &str) {
            for (int i = 0; i < str.size(); ++i)
            {
                if (s.find(str[i]) != s.end())
                    return false;
                else
                    s.insert(str[i]);
            }
            return true;
        }
    };
    ```

1. 位运算（题目提示不能用额外的数据结构）

    ```c++
    class Solution {
    public:
        bool isUnique(string &str) {
            int bits = 0;
            int char_bit = 0;
            for (int i = 0; i < str.size(); ++i)
            {
                char_bit = 1 << (str[i] - 'a');
                if ((bits & char_bit) != 0)
                    return false;
                else
                    bits |= char_bit;
            }
            return true;
        }
    };
    ```

### 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序。

使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分。

代码：

1. 单指针

    ```c++
    class Solution {
    public:
        void reOrderArray(vector<int> &array) {
            int odd_pos = 0;
            int temp;
            for (int i = 0; i < array.size(); ++i)
            {
                if (array[i] % 2 == 1)
                {
                    temp = array[odd_pos];
                    array[odd_pos] = array[i];
                    array[i] = temp;
                    ++odd_pos;
                }
            }
        }
    };
    ```

    后来又写的单指针：

    ```c++
    class Solution {
    public:
        vector<int> exchange(vector<int>& nums) {
            int left = -1, right = 0;
            int n = nums.size();
            while (right < n)
            {
                if (nums[right] % 2) swap(nums[++left], nums[right]);
                ++right;
            }
            return nums;
        }
    };
    ```

1. 双指针

    ```c++
    class Solution {
    public:
        vector<int> exchange(vector<int>& nums) {
            int n = nums.size();
            int left = 0, right = n - 1;
            while (left < right)
            {
                while (left < right && nums[left] % 2) ++left;
                while (left < right && !nums[right] % 2) --right;
                if (left < right) swap(nums[left], nums[right--]);  // if (left < right) 似乎不需要写
            }
            return nums;
        }
    };
    ```

    小知识：可以用`num & 1`来等价于`num % 2`。

### 顺时针打印矩阵

> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数> 字。
> 
> 样例：
> 
> ```
> 输入：
> [
>   [1, 2, 3, 4],
>   [5, 6, 7, 8],
>   [9,10,11,12]
> ]
> 
> 输出：[1,2,3,4,8,12,11,10,9,5,6,7]
> ```

1. 模拟

    ```c++
    class Solution {
    public:
        vector<int> printMatrix(vector<vector<int> > matrix) {
            vector<int> res;
            if (matrix.empty())
                return res;
            int col_left = 0, col_right = matrix[0].size() - 1;
            int row_top = 0, row_bottom = matrix.size() - 1;
            int direction = 0;
            int pos = 0;
            int count = 0, n = (col_right + 1) * (row_bottom + 1);
            while (count != n)
            {
                if (direction == 0)
                {
                    res.push_back(matrix[row_top][pos]);
                    ++pos;
                    if (pos == col_right + 1)
                    {
                        ++row_top;
                        pos = row_top;
                        direction = 1;
                    }
                }
                
                else if (direction == 1)
                {
                    res.push_back(matrix[pos][col_right]);
                    ++pos;
                    if (pos == row_bottom + 1)
                    {
                        --col_right;
                        pos = col_right;
                        direction = 2;
                    }
                }
                
                else if (direction == 2)
                {
                    res.push_back(matrix[row_bottom][pos]);
                    --pos;
                    if (pos == col_left - 1)
                    {
                        --row_bottom;
                        pos = row_bottom;
                        direction = 3;
                    }
                }
                
                else if (direction == 3)
                {
                    res.push_back(matrix[pos][col_left]);
                    --pos;
                    if (pos == row_top - 1)
                    {
                        ++col_left;
                        pos = col_left;
                        direction = 0;
                    }
                }
                
                ++count;
            }
            return res;
        }
    };
    ```

1. 后来又写的

    ```c++
    class Solution {
    public:
        vector<int> spiralOrder(vector<vector<int>>& matrix) {
            if (matrix.empty()) return {};
            int m = matrix.size(), n = matrix[0].size();
            int num = m * n;
            vector<int> ans(num);
            int left = 0, right = n - 1, top = 0, bottom = m - 1;
            int cnt = 0;
            int pos = 0;
            while (cnt < num)
            {
                while (cnt < num && pos <= right) ans[cnt++] = matrix[top][pos++];  // 每一句 while 都要记得用 cnt < num 做限制，防止越界
                pos = ++top;
                while (cnt < num && pos <= bottom) ans[cnt++] = matrix[pos++][right];
                pos = --right;
                while (cnt < num && pos >= left) ans[cnt++] = matrix[bottom][pos--];
                pos = --bottom;
                while (cnt < num && pos >= top) ans[cnt++] = matrix[pos--][left];
                pos = ++left;
            }
            return ans;
        }
    };
    ```

### 合并排序的数组

给定两个排序后的数组 A 和 B，其中 A 的末端有足够的缓冲空间容纳 B。 编写一个方法，将 B 合并入 A 并排序。

初始化 A 和 B 的元素数量分别为 m 和 n。

示例:

```
输入:
A = [1,2,3,0,0,0], m = 3
B = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```

分析：数组 A 的后面有空余空间，所以可以使用从尾部开始的双指针。

```c++
class Solution {
public:
    void merge(vector<int>& A, int m, vector<int>& B, int n) {
        int i = m - 1, j = n - 1, p = A.size() - 1;
        while (i > -1 && j > -1)
        {
            if (A[i] < B[j])
                A[p] = B[j--];
            else
                A[p] = A[i--];
            --p;
        }
        
        // 如果剩余的是 A 数组中的元素，那么不需要再排了
        // 如果剩余的是 B 数组中的元素，那么需要把剩余的部分复制到 A 数组中
        while (j > -1)
        {
            A[p] = B[j--];
            --p;
        }
    }
};
```

### 罗马数字转整数

1. 全部放到散列表里（只超过 10% 的用户）

    ```c++
    class Solution {
    public:
        int romanToInt(string s) {
            unordered_map<string, int> m({{"IV", 4}, {"IX", 9}, {"XL", 40}, {"XC", 90}, 
            {"CD", 400}, {"CM", 900}, {"I", 1}, {"V", 5}, {"X", 10}, {"L", 50}, {"C", 100},
            {"D", 500}, {"M", 1000}});
            unordered_map<string, int>::iterator iter;
            int num = 0;
            int i = 0;
            while (i < s.size() - 1)
            {
                iter = m.find(s.substr(i, 2));
                if (iter != m.end())
                {
                    num += iter->second;
                    i += 2;
                }
                else
                {
                    iter = m.find(s.substr(i, 1));
                    num += iter->second;
                    ++i;
                }
            }

            if (i < s.size())
            {
                iter = m.find(s.substr(i, 1));
                num += iter->second;
            }
            return num;
        }
    };
    ```

1. 找规律，若小的罗马数字在大的罗马数字之前，那么它被减。（击败了 50%）

    ```c++
    class Solution {
    public:
        int romanToInt(string s) {
            unordered_map<char, int> m({
                {'I', 1},
                {'V', 5},
                {'X', 10},
                {'L', 50},
                {'C', 100},
                {'D', 500},
                {'M', 1000}
            });

            int ans = 0;
            for (int i = 0; i < s.size(); ++i)
            {
                if (i < s.size() - 1 && m[s[i]] < m[s[i+1]]) ans -= m[s[i]];
                else ans += m[s[i]];
            }
            
            return ans;
        }
    };
    ```

### 最小的k个数

输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

示例 1：

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

示例 2：

```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

代码：

1. 建堆（只能击败约 20% 用户）

    ```c++
    class Solution {
    public:
        vector<int> getLeastNumbers(vector<int>& arr, int k) {
            priority_queue<int> q;
            for (int i = 0; i < arr.size(); ++i)
            {
                q.push(arr[i]);
                if (q.size() > k) q.pop();
            }
            vector<int> ans;
            while (!q.empty())
            {
                ans.push_back(q.top());
                q.pop();
            }
            return ans;
        }
    };
    ```

1. 快速选择（acwing 上能 ac，但在 leetcode 上似乎会超时）

    ```c++
    class Solution {
    public:
        vector<int> getLeastNumbers(vector<int>& arr, int k) {
            vector<int> res(k);
            if (k == 0)
                return res;
            for (int i = 0; i < k; ++i)
            {
                res[i] = quick_select(arr, 0, arr.size()-1, i+1);
            }
            return res;
        }

        int quick_select(vector<int> &arr, int l, int r, int k)
        {
            if (l >= r)
                return arr[l];
            int i = l - 1, j = r + 1, x = arr[(l + r) >> 1];
            while (i < j)
            {
                do ++i; while (arr[i] < x);
                do --j; while (arr[j] > x);
                if (i < j)
                    swap(arr[i], arr[j]);
            }
            if (k <= j-l+1)
                return quick_select(arr, l, j, k);
            else
                return quick_select(arr, j+1, r, k-(j-l+1));
        }
    };
    ```

1. 官方题解（没看）

    ```c++
    class Solution {
        int partition(vector<int>& nums, int l, int r) {
            int pivot = nums[r];
            int i = l - 1;
            for (int j = l; j <= r - 1; ++j) {
                if (nums[j] <= pivot) {
                    i = i + 1;
                    swap(nums[i], nums[j]);
                }
            }
            swap(nums[i + 1], nums[r]);
            return i + 1;
        }

        // 基于随机的划分
        int randomized_partition(vector<int>& nums, int l, int r) {
            int i = rand() % (r - l + 1) + l;
            swap(nums[r], nums[i]);
            return partition(nums, l, r);
        }

        void randomized_selected(vector<int>& arr, int l, int r, int k) {
            if (l >= r) {
                return;
            }
            int pos = randomized_partition(arr, l, r);
            int num = pos - l + 1;
            if (k == num) {
                return;
            } else if (k < num) {
                randomized_selected(arr, l, pos - 1, k);
            } else {
                randomized_selected(arr, pos + 1, r, k - num);
            }
        }

    public:
        vector<int> getLeastNumbers(vector<int>& arr, int k) {
            srand((unsigned)time(NULL));
            randomized_selected(arr, 0, (int)arr.size() - 1, k);
            vector<int> vec;
            for (int i = 0; i < k; ++i) {
                vec.push_back(arr[i]);
            }
            return vec;
        }
    };
    ```

1. 另一种快速排序（击败 95% 用户）（没看）

    ```c++
    class Solution {
    public:
        vector<int> getLeastNumbers(vector<int>& arr, int k) {
            if (k >= arr.size()) return arr;
            return quickSort(arr, k, 0, arr.size() - 1);
        }
    private:
        vector<int> quickSort(vector<int>& arr, int k, int l, int r) {
            int i = l, j = r;
            while (i < j) {
                while (i < j && arr[j] >= arr[l]) j--;
                while (i < j && arr[i] <= arr[l]) i++;
                swap(arr[i], arr[j]);
            }
            swap(arr[i], arr[l]);
            if (i > k) return quickSort(arr, k, l, i - 1);
            if (i < k) return quickSort(arr, k, i + 1, r);
            vector<int> res;
            res.assign(arr.begin(), arr.begin() + k);
            return res;
        }
    };
    ```

1. 自己写的快速选择（击败 98% 用户）

    原理是做几次快速排序的`partition()`操作，当`partition`返回的恰好是`k`时，那么必定有`arr[0:k-1] <= arr[k] <= arr[k+1:end]`。

    ```c++
    class Solution {
    public:
        int partition(vector<int> &arr, int left, int right)
        {
            int x = arr[right];
            int i = left - 1, j = left;
            while (j < right)
            {
                if (arr[j] < x) swap(arr[++i], arr[j]);
                ++j;
            }
            swap(arr[i+1], arr[right]);
            return i+1;
        }

        int randomized_partition(vector<int> &arr, int left, int right)
        {
            int idx = rand() % (right - left + 1) + left;
            swap(arr[idx], arr[right]);
            return partition(arr, left, right);
        }

        void quick_select(vector<int> &arr, int left, int right, int k)
        {
            if (left > right) return;
            int idx = randomized_partition(arr, left, right);
            if (idx == k) return;
            else if (idx > k) quick_select(arr, left, idx-1, k);
            else quick_select(arr, idx+1, right, k);
        }

        vector<int> getLeastNumbers(vector<int>& arr, int k) {
            if (k == 0) return {};
            vector<int> ans(k);
            quick_select(arr, 0, arr.size()-1, k);
            int pos = 0;
            do ans[pos] = arr[pos]; while (++pos < k);
            return ans;
        }
    };
    ```

### 回文数

1. 用队列（只能击败 5%）

    ```c++
    class Solution {
    public:
        bool isPalindrome(int x) {
            if (x < 0)
                return false;
            deque<int> q;
            while (x != 0)
            {
                q.push_back(x % 10);
                x = x / 10;
            }

            while (q.size() > 1)
            {
                if (q.front() != q.back())
                    return false;
                else
                {
                    q.pop_front();
                    q.pop_back();
                }
            }

            return true;
        }
    };
    ```

1. 用比较巧妙的方法，只比较一半的数字

    ```c++
    class Solution {
    public:
        bool isPalindrome(int x) {
            if (x < 0 || (x % 10 == 0 && x != 0))
                return false;

            int reverted_num = 0;
            while (x > reverted_num)
            {
                reverted_num = reverted_num * 10 + x % 10;
                x /= 10;
            }

            return x == reverted_num / 10 || x == reverted_num; 
        }
    };
    ```

### 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。
 
```
示例 1:

输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
```

1. 排序

    ```c++
    class Solution {
    public:
        int majorityElement(vector<int>& nums) {
            sort(nums.begin(), nums.end());
            return nums[nums.size() / 2];
        }
    };
    ```

1. 摩尔投票法

    ```c++
    class Solution {
    public:
        int majorityElement(vector<int>& nums) {
            int x = 0, votes = 0;
            for(int num : nums){
                if(votes == 0) x = num;
                votes += num == x ? 1 : -1;
            }
            return x;
        }
    };
    ```

    摩尔投票法找的其实不是众数，而是占一半以上的数。当数组没有超过一半的数，则可能返回非众数，比如`[1, 1, 2, 2, 2, 3, 3]`，最终返回`3`。

    投票法简单来说就是不同则抵消，占半数以上的数字必然留到最后。

    后来又写的：

    ```c++
    class Solution {
    public:
        int majorityElement(vector<int>& nums) {
            int ans = nums[0], count = 1;
            int n = nums.size();
            for (int i = 1; i < n; ++i)
            {
                count += ans == nums[i] ? 1 : -1;
                if (count < 0)
                {
                    ans = nums[i];
                    count = 1;
                }
            }
            return ans;
        }
    };
    ```

    一个条件是`count == 0`，一个条件是`count < 0`，一个不需要重置`count`，一个需要重置`count`，一个不需要处理开头，一个需要处理开头。为什么会这样呢？

### 最长连续序列

> 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
> 
> 进阶：你可以设计并实现时间复杂度为 O(n) 的解决方案吗？
> 
> ```
> 示例 1：
> 
> 输入：nums = [100,4,200,1,3,2]
> 输出：4
> 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
> ```

代码：

1. 使用散列表 + 剪枝

    ```c++
    class Solution {
    public:
        int longestConsecutive(vector<int>& nums) {
            unordered_set<int> m;
            for (auto &num: nums)
                m.insert(num);

            int res = 0;
            int len = 0;
            for (auto num: m)  // 第二次遍历的时候只对散列表遍历就行了
            {
                if (m.find(num - 1) != m.end())
                    continue;
                len = 1;
                while (m.find(++num) != m.end())
                    ++len;
                res = max(res, len);
            }
            return res;
        }
    };
    ```

### 数组中的逆序对

在数组中的两个数字如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。

输入一个数组，求出这个数组中的逆序对的总数。

```
样例
输入：[1,2,3,4,5,6,0]

输出：6
```

分析：

只是归并排序加了一行代码。如果添加到临时数组中的是左子列，说明左子列的当前值比右子列的当前值小，没有逆序对；如果添加到临时数组中的是右子列，说明左子列的当前值比较大，此时产生`mid - i + 1`个逆序对。

代码：

1. 归并排序

    ```c++
    class Solution {
    public:
        vector<int> tmp;
        int count;
        void merge_sort(vector<int> &nums, int l, int r)
        {
            if (l >= r) return;
            int mid = (l + r) >> 1;
            merge_sort(nums, l, mid);
            merge_sort(nums, mid+1, r);
            int i = l, j = mid+1, k = 0;
            while (i <= mid && j <= r)
            {
                if (nums[i] <= nums[j])
                    tmp[k++] = nums[i++];
                else
                {
                    count += mid - i + 1;
                    tmp[k++] = nums[j++];
                }
            }
            while (i <= mid)
                tmp[k++] = nums[i++];
            while (j <= r)
                tmp[k++] = nums[j++];
            for (int i = l, j = 0; i <= r; ++i, ++j)
                nums[i] = tmp[j];
        }
        int inversePairs(vector<int>& nums) {
            tmp.resize(nums.size());
            count = 0;
            merge_sort(nums, 0, nums.size()-1);
            return count;
        }
    };
    ```

1. 树状数组（没看

### 在排序数组中查找数字 I

统计一个数字在排序数组中出现的次数。

```
示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

代码：

1. 二分查找 + 线性查找

    ```c++
    class Solution {
    public:
        int getNumberOfK(vector<int>& nums , int k) {
            if (nums.empty())
                return 0;
                
            int l = 0, r = nums.size() - 1;
            int mid = (l + r) >> 1;
            while (l < r)
            {
                if (nums[mid] < k)
                    l = mid + 1;
                else if (nums[mid] > k)
                    r = mid;
                else
                    break;
                mid = (l + r) >> 1;
            }
            
            int count = 0, tmp = mid;
            while (tmp >= 0 && nums[tmp--] == k) ++count;
            if (mid < r)
            {
                tmp = mid+1;
                while (tmp < nums.size() && nums[tmp++] == k) ++count;
            }
            return count;
        }
    };
    ```

1. 两次二分查找，找到左右边界

    先应用一次左边界二分查找，再应用一次右边界二分查找。

    ```c++
    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int left = 0, right = nums.size() - 1, mid;
            int left_bound, right_bound;
            while (left <= right)
            {
                mid = left + (right - left) / 2;
                if (nums[mid] < target) left = mid + 1;
                else if (nums[mid] > target) right = mid - 1;
                else right = mid - 1;
            }
            if (left >= nums.size() || nums[left] != target) return 0;
            left_bound = left;
            
            right = nums.size() - 1;
            while (left <= right)
            {
                mid = left + (right - left) / 2;
                if (nums[mid] < target) left = mid + 1;
                else if (nums[mid] > target) right = mid - 1;
                else left = mid + 1;
            }
            right_bound = right;

            return right_bound - left_bound + 1;
        }
    };
    ```

### 0 到 n-1 中缺失的数字

一个长度为 n−1 的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围 0 到 n−1 之内。

在范围 0 到 n−1 的 n 个数字中有且只有一个数字不在该数组中，请找出这个数字。

```
样例
输入：[0,1,2,4]

输出：3
```

代码：

1. 二分查找。

    ```c++
    class Solution {
    public:
        int getMissingNumber(vector<int>& nums) {
            if (nums.empty())
                return 0;
            int l = 0, r = nums.size() - 1, m = (l + r) >> 1;
            while (l < r)
            {
                if (nums[m] > m) r = m;
                else l = m+1;
                m = (l + r) >> 1;
            }
            return l == nums[l] ? l + 1 : l;
        }
    };
    ```

1. 更简洁的二分查找

    ```c++
    class Solution {
    public:
        int missingNumber(vector<int>& nums) {
            int left = 0, right = nums.size() - 1;
            int mid;
            while (left <= right)
            {
                mid = left + (right - left) / 2;
                if (nums[mid] > mid) right = mid - 1;
                else if (nums[mid] == mid) left = mid + 1; 
            }
            return left;
        }
    };
    ```

    其实就是根据条件查找一个左边界。左边界又相当于插入位置的索引。条件是索引值是否等于数值。

### 数组中数值和下标相等的元素

假设一个单调递增的数组里的每个元素都是整数并且是唯一的。

请编程实现一个函数找出数组中任意一个数值等于其下标的元素。

例如，在数组 [−3,−1,1,3,5] 中，数字 3 和它的下标相等。

```
样例
输入：[-3, -1, 1, 3, 5]

输出：3
注意:如果不存在，则返回-1。
```

代码：

二分。

```c++
class Solution {
public:
    int getNumberSameAsIndex(vector<int>& nums) {
        int l = 0, r = nums.size() - 1, m = (l + r) >> 1;
        while (l < r)
        {
            if (nums[m] < m) l = m + 1;
            else if (nums[m] > m) r = m;
            else return m;
            m = (l + r) >> 1;
        }
        if (nums[m] == m) return m;
        return -1;
    }
};
```

### 和为S的两个数字

输入一个数组和一个数字 s，在数组中查找两个数，使得它们的和正好是 s。

如果有多对数字的和等于 s，输出任意一对即可。

你可以认为每组输入中都至少含有一组满足条件的输出。

```
样例
输入：[1,2,3,4] , sum=7

输出：[3,4]
```

代码：

```c++
class Solution {
public:
    vector<int> findNumbersWithSum(vector<int>& nums, int target) {
        unordered_set<int> s;
        for (auto &num: nums)
        {
            if (s.find(target - num) != s.end())
                return vector<int>({num, target-num});
            else
                s.insert(num);
        }
    }
};
```

如果数组是递增排序，那么可以用双指针，二分查找：

1. 双指针（这个应该是最快的，比二分查找还快）

    ```c++
    class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target) {
            int n = nums.size();
            int left = 0, right = n - 1;
            while (left < right)
            {
                if (nums[left] + nums[right] < target) ++left;
                else if (nums[left] + nums[right] > target) --right;
                else return {nums[left], nums[right]};
            }
            return {};
        }
    };
    ```

1. 二分查找

    ```c++
    class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target) {
            int start = 0, left, right, mid;
            int n = nums.size();
            int t;
            while (start < n)
            {
                t = target - nums[start];
                left = start + 1;
                right = n - 1;
                if (t <= 0) break;
                if (nums[left] > t || nums[right] < t)
                {
                    ++start;
                    continue;
                }
                while (left <= right)
                {
                    mid = left + (right - left) / 2;
                    if (nums[mid] < t) left = mid + 1;
                    else if (nums[mid] > t) right = mid - 1;
                    else return {nums[start], nums[mid]};
                }
                ++start;
            }
            return {};
        }
    };
    ```

### 和为S的连续正数序列

输入一个正数 S，打印出所有和为 S 的连续正数序列（至少含有两个数）。

例如输入 15，由于 1+2+3+4+5=4+5+6=7+8=15，所以结果打印出 3 个连续序列 1∼5、4∼6 和 7∼8。

```
样例
输入：15

输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

代码：

1. 滑动窗口。

    ```c++
    class Solution {
    public:
        vector<vector<int> > findContinuousSequence(int sum) {
            vector<vector<int>> res;
            int l = 1, r = 2, tmp = 3;
            while (l <= sum / 2)
            {
                if (tmp < sum) tmp += ++r;  // 如果和小了，那么把窗口左端往右移
                else if (tmp > sum) tmp -= l++;  // 如果和大了，那么把窗口右端往左移
                else  // 如果和满足要求，那么添加进结果
                {   
                    vector<int> vec(r - l + 1);  // 如果 r - l + 1 == 0 会发生什么呢？
                    for (int i = 0, j = l; j <= r; ++i, ++j) vec[i] = j;
                    res.push_back(vec);
                    tmp -= l++;  // 注意改变下状态，不然会限入死循环
                }
            }
            return res;
        }
    };
    ```

1. 滑动窗口，后来自己写的

    ```c++
    class Solution {
    public:
        vector<vector<int>> findContinuousSequence(int target) {
            vector<vector<int>> ans;
            int left = 1, right = 2;
            int sum = 3;
            while (left < right)
            {
                if (sum == target)
                {
                    ans.push_back(vector<int>());
                    for (int i = left; i <= right; ++i)
                    {
                        ans.back().push_back(i);
                    }
                    ++right;
                    sum += right;
                }
                else if (sum < target)
                {
                    ++right;
                    sum += right;
                }
                else 
                {
                    sum -= left;
                    ++left;
                }
            }
            return ans;
        }
    };
    ```

    更精简的代码：

    ```c++
    class Solution {
    public:
        vector<vector<int>> findContinuousSequence(int target) {
            vector<vector<int>> ans;
            int left = 1, right = 2;
            int sum = 3;
            int bound = target / 2 + 1;
            while (right <= bound)
            {
                if (sum < target) sum += ++right;
                else if (sum > target) sum -= left++;
                else
                {
                    if (right - left + 1 >= 2)
                    {
                        ans.push_back(vector<int>(right-left+1));
                        iota(ans.back().begin(), ans.back().end(), left);
                    }
                    sum -= left++;
                }
            }
            return ans;
        }
    };
    ```

1. 前缀和（通用复杂版）

    使用哈希表存储以前算过的前缀和，就不用写两层循环了。

    ```c++
    class Solution {
    public:
        vector<vector<int>> findContinuousSequence(int target) {
            unordered_map<int, int> m;
            vector<vector<int>> ans;
            vector<int> left(target / 2 + 2);  // 这里的 +2 是因为，存储 0 值需要多占一个位置，再举个特例，比如当 target 为 5 时，有 2 + 3，此时需要存储 left[0], left[1], left[2], left[3]，而 5 / 2 = 2，因此需要再另外加一个。
            int n = left.size();
            left[0] = 0;  // 边界值，用于处理左侧全部元素都能取到的情况
            m[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                left[i] = left[i-1] + i;  // 这里的 i 既是索引，也是数值
                if (m.find(left[i] - target) != m.end())
                {
                    int j = m[left[i] - target];
                    int len = i - j;  // m[num] 表示前缀和为 num 的索引值，若 left[j] = num，那么其实 j 是不算在区间内的，相当于是上一个区间结束的位置，因此这里不用写成 len = i - j + 1
                    if (len >= 2)
                    {
                        vector<int> temp(len);
                        int pos = 0;
                        j += 1;  // j 是上个区间结束的索引，同时也是数值，这里把 j 递增 1 转换成当前区间开始的数值
                        while (pos < len) temp[pos++] = j++;
                        ans.push_back(temp);
                    }
                }
                m[left[i]] = i;
            }
            return ans;
        }
    };
    ```

    前缀和通常可以用`left[i]`表示`num[0] + num[1] + ... + num[i]`表示，此时`i`是作为索引值，这种形式在做加法的时候常用。还可以用`left[i]`表示`num[0] + num[1] + ... + num[i-1]`，此时`left[0] = 0`，通常用于找连续数组，做减法。

1. 



### 翻转单词顺序

输入一个英文句子，单词之间用一个空格隔开，且句首和句尾没有多余空格。

翻转句子中单词的顺序，但单词内字符的顺序不变。

为简单起见，标点符号和普通字母一样处理。

例如输入字符串"I am a student."，则输出"student. a am I"。

```
样例
输入："I am a student."

输出："student. a am I"
```

代码：

1. 两头和中间没有多余空格的版本

    ```c++
    class Solution {
    public:
        string reverseWords(string s) {
            string res;
            res.resize(s.size());
            int pos = 0;
            int cur, prev = s.size();
            
            // 从后往前找空格，找到一个就把后面的单词复制到答案中
            for (int i = s.size() - 1; i > -1; --i)
            {
                if (s[i] == ' ')
                {
                    cur = i + 1;
                    while (cur < prev) res[pos++] = s[cur++];
                    res[pos++] = ' ';
                    prev = i;
                }
            }

            // 单独考虑第一个单词
            cur = 0;
            while (cur < prev) res[pos++] = s[cur++];
            return res;
        }
    };
    ```

1. 先`reverse()`整个字符串，再`reverse()`每个单词

    这种解法更优雅一点，但是只能处理单词之间的空格为一个时的情况。

### 翻转字符串里的单词

给你一个字符串 s ，逐个翻转字符串中的所有 单词 。

单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。

说明：

输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
翻转后单词间应当仅用一个空格分隔。
翻转后的字符串中不应包含额外的空格。
 
```
示例 1：

输入：s = "the sky is blue"
输出："blue is sky the"
示例 2：

输入：s = "  hello world  "
输出："world hello"
解释：输入字符串可以在前面或者后面包含多余的空格，但是翻转后的字符不能包括。
示例 3：

输入：s = "a good   example"
输出："example good a"
解释：如果两个单词间有多余的空格，将翻转后单词间的空格减少到只含一个。
示例 4：

输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"
示例 5：

输入：s = "Alice does not even like bob"
输出："bob like even not does Alice"
```

1. 模拟逻辑，用指针一个一个找

    ```c++
    class Solution {
    public:
        string reverseWords(string s) {
            if (s.empty())
                return s;

            int start = 0, end = s.size() - 1;
            while (start <= end && s[start] == ' ') ++start;
            while (start <= end && s[end] == ' ') --end;
            
            string res;
            res.resize(end - start + 1);

            int pos = 0, cur, prev = end + 1; 
            for (int i = end; i > start; --i)
            {
                if (s[i] == ' ' && s[i-1] != ' ')
                {
                    cur = i + 1;
                    while (s[cur] == ' ') ++cur;
                    while (cur < prev)  res[pos++] = s[cur++];
                    res[pos++] = ' ';
                    prev = i;
                }
            }
            
            cur = start;
            while (cur < prev)
                res[pos++] = s[cur++];
            return res;
        }
    };
    ```

1. 后来又写的

    ```c++
    class Solution {
    public:
        string reverseWords(string s) {
            string ans;
            int start = 0, end = s.size() - 1;
            while (s[start] == ' ') ++start;
            while (s[end] == ' ') --end;
            ans.resize(end - start + 1);
            int pos = end;
            int pos_new = 0;
            while (pos >= start)
            {
                while (pos >= start && s[pos] != ' ') --pos;
                for (int i = pos + 1; i <= end; ++i)
                    ans[pos_new++] = s[i];
                if (pos+1 != start) ans[pos_new++] = ' ';
                end = pos;
                while (end >= start && s[end] == ' ') --end;
                pos = end;
            }

            for (int i = start; i <= end; ++i)
                ans[pos_new++] = s[i];
            ans = ans.substr(0, pos_new);
            return ans;

        }
    };
    ```

    第二次似乎没第一次写的好。这种题有这样一个问题：尾部或头部的模式与其他部分的模式不一样。比如`I am a student`，从尾往前扫描，每个单词前面都有一个以上的空格，只有第一个单词没有，这时就需要额外的`if`，或者单独写一段代码来处理，很麻烦。

    后来又写的：

    ```c++
    class Solution {
    public:
        string reverseWords(string s) {
            int start = 0, end = s.size() - 1;
            while (start <= end && s[start] == ' ') ++start;  // 注意要加止 start <= end，防止指针越界
            while (start <= end && s[end] == ' ') --end;

            string ans;
            int p1 = end, p2 = end, p;
            while (p1 >= start)
            {
                while (p1 >= start && s[p1] != ' ') --p1;
                p = p1 + 1;
                while (p <= p2) ans.push_back(s[p++]);
                if (p1 >= start) ans.push_back(' ');  // 如果不是第一个单词
                p2 = p1;
                while (p2 >= start && s[p2] == ' ') --p2;
                p1 = p2;
            }
            return ans;
        }
    };
    ```

    这种方式应该是最简了吧，即使对于第一个单词也能正常处理。需要注意的是，在每个`while()`中，如果对指针有改动，那么一定要检测指针的边界值，防止越界。

### 滑动窗口的最大值

给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。

例如，如果输入数组 [2,3,4,2,6,2,5,1] 及滑动窗口的大小 3，那么一共存在 6 个滑动窗口，它们的最大值分别为 [4,4,6,6,6,5]。

注意：

数据保证 k 大于 0，且 k 小于等于数组长度。

```
样例
输入：[2, 3, 4, 2, 6, 2, 5, 1] , k=3

输出: [4, 4, 6, 6, 6, 5]
```

代码：

1. 单调队列

    队列的右端用于模拟单调栈，左端用于模拟窗口的滑动。

    ```c++
    class Solution {
    public:
        vector<int> maxInWindows(vector<int>& nums, int k) {
            deque<int> q;
            vector<int> res;
            for (int i = 0; i < k; ++i)
            {
                while (!q.empty() && q.back() < nums[i]) q.pop_back();  // 这里不能加等号
                q.push_back(nums[i]);
            }
            if (!q.empty()) res.push_back(q.front());
            
            for (int i = k; i < nums.size(); ++i)
            {
                if (nums[i - k] == q.front()) q.pop_front();
                while (!q.empty() && q.back() < nums[i]) q.pop_back();
                q.push_back(nums[i]);
                res.push_back(q.front());
            }
            return res;
        }
    };
    ```

1. 官方题解 1，大顶堆

    ```c++
    class Solution {
    public:
        vector<int> maxSlidingWindow(vector<int>& nums, int k) {
            int n = nums.size();
            priority_queue<pair<int, int>> q;
            for (int i = 0; i < k; ++i) {
                q.emplace(nums[i], i);
            }
            vector<int> ans = {q.top().first};
            for (int i = k; i < n; ++i) {
                q.emplace(nums[i], i);
                while (q.top().second <= i - k) {
                    q.pop();
                }
                ans.push_back(q.top().first);
            }
            return ans;
        }
    };
    ```

1. 官方题解 2，分块 + 预处理。据说和稀疏表非常相似，有空了看看。

    ```c++
    class Solution {
    public:
        vector<int> maxSlidingWindow(vector<int>& nums, int k) {
            int n = nums.size();
            vector<int> prefixMax(n), suffixMax(n);
            for (int i = 0; i < n; ++i) {
                if (i % k == 0) {
                    prefixMax[i] = nums[i];
                }
                else {
                    prefixMax[i] = max(prefixMax[i - 1], nums[i]);
                }
            }
            for (int i = n - 1; i >= 0; --i) {
                if (i == n - 1 || (i + 1) % k == 0) {
                    suffixMax[i] = nums[i];
                }
                else {
                    suffixMax[i] = max(suffixMax[i + 1], nums[i]);
                }
            }

            vector<int> ans;
            for (int i = 0; i <= n - k; ++i) {
                ans.push_back(max(suffixMax[i], prefixMax[i + k - 1]));
            }
            return ans;
        }
    };
    ```

### 盛最多水的容器

给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

分析：双指针法。每次让短板向里前进 1。因为盛水容量由短板决定，短板向里可能会导致短板变长，但长板向里移动一定无法导致短板变长，所以我们每次都选择短板向里前进。

代码：

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int res = 0;
        while (left < right)
        {
            res = max(min(height[left], height[right]) * (right - left), res);
            if (height[left] < height[right]) ++left;
            else --right;
        }
        return res;
    }
};
```

### 接雨水

给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
```

分析：

1. 暴力法。从`i`位置向左向右找最高的，若两侧最高的柱子都比`i`高，说明`i`位置可以接到雨水，最多可接`min(left_max, right_max) - height[i]`这么多雨水；若两侧最高的柱子有一侧比`i`矮，那么`i`就接不到。时间复杂度`O(n^2)`

1. 动态规划。先从左到右遍历一遍数组，找到每个位置的左侧边界，再从右到左遍历一遍数组，找到每个位置的右侧边界，然后再按方法一的暴力法得到结果就好。时间复杂度`O(n)`，空间复杂度`O(n)`。

1. 单调栈。若形成凹槽，则开始收集雨水。时间复杂度`O(n)`，空间复杂度`O(n)`。

1. 双指针。是动态规划的改进。对于某个位置来说，如果它的左侧有比它高的，右侧也有比它高的，那么它一定能接住雨水。具体能接多少雨水，受`min(left_max, right_max) - height[i]`的影响。

代码

1. 暴力法

    ```c++
    class Solution {
    public:
        int trap(vector<int>& height) {
            int ans = 0;
            int left_max = -1, right_max = -1;
            for (int i = 0; i < height.size(); ++i)
            {
                int j = i - 1;
                left_max = -1;
                while (j > -1)
                    left_max = max(left_max, height[j--]);
                j = i + 1;
                right_max = -1;
                while (j < height.size())
                    right_max = max(right_max, height[j++]);
                if (left_max < height[i] || right_max < height[i])
                    continue;
                else
                    ans += min(left_max, right_max) - height[i];   
            }
            return ans;
        }
    };
    ```

    后来又写的：

    ```c++
    class Solution {
    public:
        int trap(vector<int>& height) {
            int n = height.size();
            vector<int> v(n);
            for (int i = 0; i < n; ++i)
            {
                int left = 0, right = 0;
                for (int j = 0; j < i; ++j)
                    left = max(left, height[j]);
                for (int j = n-1; j > i; --j)
                    right = max(right, height[j]);
                if (min(left, right) > height[i])
                    v[i] = min(left, right) - height[i];
            }
            return accumulate(v.begin(), v.end(), 0);
        }
    };
    ```

1. 动态规划

    ```c++
    class Solution {
    public:
        int trap(vector<int>& height) {
            int ans = 0;
            vector<int> left_max(height.size());
            vector<int> right_max(height.size());

            for (int i = 1; i < height.size(); ++i)  // 因为第一介位置和最后一个位置一定接不到雨水，所以不需要考虑
                left_max[i] = max(left_max[i-1], height[i-1]);
            for (int i = height.size() - 2; i > -1; --i)
                right_max[i] = max(right_max[i+1], height[i+1]);
            for (int i = 0; i < height.size(); ++i)
                if (left_max[i] > height[i] && right_max[i] > height[i])
                    ans += min(left_max[i], right_max[i]) - height[i];
            return ans;
        }
    };
    ```

3. 单调栈

    ```c++
    class Solution {
    public:
        int trap(vector<int>& height) {
            stack<int> st;
            int ans = 0;

            st.push(0);
            for (int i = 1; i < height.size(); ++i)
            {
                while (!st.empty() && height[i] > height[st.top()])
                {
                    int mid = st.top();
                    st.pop();
                    if (!st.empty())  // 排除左侧没有边界的情况
                    {
                        int h = min(height[st.top()], height[i]) - height[mid];
                        int w = i - st.top() - 1;
                        ans += h * w;
                    }
                }
                st.push(i);  // 如果 height[i] == height[st.top()]，则相当于是更新右边界
            }

            return ans;
        }
    };
    ```

1. 双指针法

    ```c++
    class Solution {
    public:
        int trap(vector<int>& height) {
            int ans = 0;
            int left = 0, right = height.size() - 1;
            int left_max = 0, right_max = 0;
            while (left < right)
            {
                left_max = max(height[left], left_max);
                right_max = max(height[right], right_max);
                if (height[left] < height[right])
                {
                    ans += left_max - height[left];
                    ++left;
                }
                else
                {
                    ans += right_max - height[right];
                    --right;
                }
            }
            return ans;
        }
    };
    ```

### 判定是否互为字符重排

给定两个字符串 s1 和 s2，请编写一个程序，确定其中一个字符串的字符重新排列后，能否变成另一个字符串。

```
示例 1：

输入: s1 = "abc", s2 = "bca"
输出: true 
示例 2：

输入: s1 = "abc", s2 = "bad"
输出: false
```

代码：

1. 桶计数（也可以用哈希表计数）

    ```c++
    class Solution {
    public:
        bool CheckPermutation(string s1, string s2) {
            vector<int> cnt(26);
            for (auto ch: s1)
                --cnt[ch-'a'];
            for (auto ch: s2)
                ++cnt[ch-'a'];
            for (int i = 0; i < 26; ++i)
            {
                if (cnt[i] != 0) return false;
            }
            return true;
        }
    };
    ```

注意这道题也不能用异或，因为可能出现`s1 = "aa"`，`s2 = "bb"`的情形。

### 字符串的排列（字符串中的变位词）

给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的 子串 。


示例 1：

```
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
```

分析：

1. 滑动窗口。

    ```c++
    class Solution {
    public:
        bool checkInclusion(string s1, string s2) {
            if (s2.size() < s1.size()) return false;
            vector<int> cnt1(26), cnt2(26);
            for (int i = 0; i < s1.size(); ++i)
            {
                ++cnt1[s1[i] - 'a'];
                ++cnt2[s2[i] - 'a'];
            }
            if (cnt1 == cnt2) return true;
            for (int i = s1.size(); i < s2.size(); ++i)
            {
                --cnt2[s2[i - s1.size()] - 'a'];  // 从左侧滑出窗口
                ++cnt2[s2[i] - 'a'];  // 从右侧滑入窗口
                if (cnt1 == cnt2) return true;
            }
            return false;
        }
    };
    ```

1. 双指针

    ```c++
    class Solution {
    public:
        bool checkInclusion(string s1, string s2) {
            if (s2.size() < s1.size()) return false;
            vector<int> cnt(26);
            for (int i = 0; i < s1.size(); ++i)
                --cnt[s1[i] - 'a'];  // 先把 s1 中出现的字符往下踩坑
            
            int l = 0, r = 0, ch;
            while (r < s2.size())
            {
                ch = s2[r] - 'a';
                ++cnt[ch];  // 再用 s2 中出现的字符把坑填平
                while (cnt[ch] > 0)
                {
                    if (l == s.size()) return false;  // 一直增长 l 或许会超过字符串结尾
                    --cnt[s2[l++] - 'a'];  // 若发现坑鼓了起来，说明某个地方出问题了，我们让左指针找到出问题的地方
                }
                
                if (r - l + 1 == s1.size()) return true;  // 如果长度正好相等，那么可以推断出所有的坑都填平了
                ++r;
            }
            return false;
        }
    };
    ```

    这道题我仍然不知道如何精确地判断边界条件。

### 找到字符串中所有字母异位词（字符串中的变位词）

给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指字母相同，但排列不同的字符串。

 
```
示例 1:

输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

示例 2:

输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
```

代码：

1. 滑动窗口

    ```c++
    class Solution {
    public:
        vector<int> findAnagrams(string s, string p) {
            vector<int> ans;
            if (s.size() < p.size()) return ans;
            int left = 0, right = 0;
            vector<int> cnt_s(26, 0), cnt_p(26, 0);
            while (right < p.size())
            {
                ++cnt_s[s[right]-'a'];
                ++cnt_p[p[right]-'a'];
                ++right;
            }
            if (cnt_s == cnt_p) ans.push_back(0);
            while (right < s.size())
            {
                --cnt_s[s[left++]-'a'];
                ++cnt_s[s[right++]-'a'];
                if (cnt_s == cnt_p) ans.push_back(left);
            }
            return ans;
        }
    };
    ```

注：这道题没法用双指针+踩坑法做，因为这种方法没法处理`s = "abab"`，`p = "ab"`的情形。

这道题似乎可以用双指针做，通过双指针 + 长度配合的方式。但我现在还没懂这样做的原理。

### 错误的集合

集合 s 包含从 1 到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个数字复制了成了集合里面的另外一个数字的值，导致集合 丢失了一个数字 并且 有一个数字重复 。

给定一个数组 nums 代表了集合 S 发生错误后的结果。

请你找出重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。

 

示例 1：

```
输入：nums = [1,2,2,4]
输出：[2,3]
```

代码：

1. 先排序，再找规律

    ```c++
    class Solution {
    public:
        vector<int> findErrorNums(vector<int>& nums) {
            sort(nums.begin(), nums.end());
            vector<int> ans(2);
            for (int i = 1; i < nums.size(); ++i)
            {
                if (nums[i] == nums[i-1]) ans[0] = nums[i];
                if (nums[i] - nums[i-1] == 2) ans[1] = nums[i-1] + 1;
            }

            if (nums[0] == 2) ans[1] = 1;  // 丢失头与丢失尾需要单独处理
            if (nums.back() == nums.size() - 1) ans[1] = nums.size();
            return ans;
        }
    };
    ```

1. 哈希表统计（效率最低）

    ```c++
    class Solution {
    public:
        vector<int> findErrorNums(vector<int>& nums) {
            unordered_map<int, int> m;
            for (auto &num: nums)
                m[num]++;
            vector<int> ans(2);
            int c = 0;
            for (int i = 1; i <= nums.size(); ++i)
            {
                c = m[i];
                if (c == 2) ans[0] = i;
                if (c == 0) ans[1] = i;
            }
            return ans;
        }
    };
    ```

1. 异或分组

    只要两个元素出现的次数的奇偶性相同，就可以用异或分组找到它们。

    ```c++
    class Solution {
    public:
        vector<int> findErrorNums(vector<int>& nums) {
            int x = 0;
            for (int &num: nums)
                x ^= num;
            // 将原数组和序号 1 ~ n 经过异或后，出现了两次的数现在出现 3 次，没有出现的数出现了 1 次
            for (int i = 1; i <= nums.size(); ++i)
                x ^= i;

            // 从最低位开始，找一个是 1 的位就好
            // 也可以用与运算直接找到： dig = x & (-x);
            int dig = 1;
            for (int i = 0; i < 32; ++i)
            {
                if (dig & x) break;
                dig <<= 1;
            }
            vector<int> ans(2);
            for (int &num: nums)
            {
                if (dig & num) ans[0] ^= num;
                else ans[1] ^= num;
            }
            for (int i = 1; i <= nums.size(); ++i)
            {
                if (dig & i) ans[0] ^= i;
                else ans[1] ^= i;
            }
            for (int &num: nums)
                if (num == ans[0]) return ans;
            reverse(ans.begin(), ans.end());
            return ans;
        }
    };
    ```



### 两个数组的交集 II

给定两个数组，编写一个函数来计算它们的交集。

 
```
示例 1：

输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
示例 2:

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
```

进阶：

* 如果给定的数组已经排好序呢？你将如何优化你的算法？

    答：见代码 2

* 如果 nums1 的大小比 nums2 小很多，哪种方法更优？

    答：代码 1 的优化

* 如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

    答 1：对应进阶问题三，如果内存十分小，不足以将数组全部载入内存，那么必然也不能使用哈希这类费空间的算法，只能选用空间复杂度最小的算法。归并排序是天然适合外部排序的算法，可以将分割后的子数组写到单个文件中，归并时将小文件合并为更大的文件。当两个数组均排序完成生成两个大文件后，即可使用双指针遍历两个文件，如此可以使空间复杂度最低。

    答 2：采用哈希表法。

代码：

1. 朴素想法，哈希表

    ```c++
    class Solution {
    public:
        vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
            if (nums1.size() > nums2.size())
                return intersect(nums2, nums1);  // 优化，保证哈希表里总是统计较短的集合。这个写法好优雅

            unordered_map<int, int> m;
            for (auto &num: nums1)
                ++m[num];

            vector<int> res;
            for (auto &num: nums2)
            {
                if (m[num])  // 这种应该算利用了语法特性了吧，不喜欢
                {
                    res.emplace_back(num);
                    --m[num];
                }
            }
            return res;
        }
    };
    ```

1. 排序 + 双指针

    ```c++
    class Solution {
    public:
        vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
            sort(nums1.begin(), nums1.end());
            sort(nums2.begin(), nums2.end());
            int p1 = 0, p2 = 0;
            vector<int> res;
            while (p1 != nums1.size() && p2 != nums2.size())
            {
                if (nums1[p1] < nums2[p2]) ++p1;
                else if (nums1[p1] > nums2[p2]) ++p2;
                else
                {
                    res.push_back(nums1[p1]);
                    ++p1;
                    ++p2;
                }
            }
            return res;
        }
    };
    ```

1. 双哈希表（速度好像还行，没有慢到不能接受）

    ```cpp
    class Solution {
    public:
        vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
            vector<int> ans;
            unordered_map<int, int> m1, m2;
            for (int &num: nums1)
                m1[num] += 1;
            for (int &num: nums2)
                m2[num] += 1;
            for (auto &p: m1)
            {
                int key = p.first;
                int val = p.second;
                if (m2.find(key) != m2.end())
                {
                    for (int i = min(val, m2[key]); i > 0; --i)
                    {
                        ans.push_back(key);
                    }
                }
            }
            return ans;
        }
    };
    ```

### 矩阵置零

给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

进阶：

一个直观的解决方案是使用  O(mn) 的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m + n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个仅使用常量空间的解决方案吗？

```
示例 1：

输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]

示例 2：

输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

分析：这道题的难点在，给某一行或某一列赋值 0 后，矩阵到处都是 0，没法判断接下来该给哪行或哪列赋值了。因此需要想办法提前保存好给哪些行或列赋 0。

1. 用队列或额外的矩阵存 0 的位置，然后遍历赋 0。空间复杂度`O(mn)`

1. 用哈希表或两个额外的数组存需要置零的位置。空间复杂度`O(m + n)`。

1. 用矩阵的首行和首列保存需要赋 0 的位置，用额外的两个变量存储首行和首列是否需要赋 0。空间复杂度`O(1)`。

代码：

这道题的难点在于，后来置 0 的地方会干扰原始置 0 的地方。如果不提前保存原始的 0 的位置，那么矩阵经过更改后，我们就无法分辨哪些是之前置 0 的，哪些是之后置 0 的了。

1. 首先想到的是用队列把原始的 0 的索引都保存下来，然后一个一个更改

    ```c++
    class Solution {
    public:
        void setZeroes(vector<vector<int>>& matrix) {
            queue<pair<int, int>> q;
            int m = matrix.size(), n = matrix[0].size();
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (matrix[i][j] == 0)
                        q.push(make_pair(i, j));
                }
            }

            int sx, sy;
            while (!q.empty())
            {
                sx = q.front().first;
                sy = q.front().second;
                q.pop();
                for (int j = 0; j < n; ++j) matrix[sx][j] = 0;
                for (int i = 0; i < m; ++i) matrix[i][sy] = 0;
            }
        }
    };
    ```

    但是这种方法有冗余的操作。

1. 用两个数组，分别标记行和列出现 0 的地方。

    ```c++
    class Solution {
    public:
        void setZeroes(vector<vector<int>>& matrix) {
            int m = matrix.size(), n = matrix[0].size();
            vector<bool> sr(m), sc(n);

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (matrix[i][j] == 0)
                    {
                        sr[i] = true;
                        sc[j] = true;
                    }
                }
            }

            for (int i = 0; i < m; ++i)
            {
                if (sr[i])
                {
                    for (int j = 0; j < n; ++j)
                        matrix[i][j] = 0;
                }
            }

            for (int j = 0; j < n; ++j)
            {
                if (sc[j])
                {
                    for (int i = 0; i < m; ++i)
                        matrix[i][j] = 0;
                }
            }
        }
    };
    ```

1. 用矩阵的首行和首列保存需要赋 0 的位置，用额外的两个变量存储首行和首列是否需要赋 0。

    ```c++
    class Solution {
    public:
        void setZeroes(vector<vector<int>>& matrix) {
            int m = matrix.size(), n = matrix[0].size();
            bool r0 = false, c0 = false;
            for (int j = 0; j < n; ++j)
            {
                if (matrix[0][j] == 0)
                {
                    r0 = true;
                    break;
                }
            }

            for (int i = 0; i < m; ++i)
            {
                if (matrix[i][0] == 0)
                {
                    c0 = true;
                    break;
                }
            }

            for (int i = 1; i < m; ++i)
            {
                for (int j = 1; j < n; ++j)
                {
                    if (matrix[i][j] == 0)
                    {
                        matrix[0][j] = 0;
                        matrix[i][0] = 0;
                    }
                }
            }
    
            for (int j = 1; j < n; ++j)  // 注意这里从 1 开始，避免把首列赋 0
                if (matrix[0][j] == 0)
                    for (int i = 1; i < m; ++i)
                        matrix[i][j] = 0;

            for (int i = 1; i < m; ++i)  // 这里也从 1 开始，避免把首行赋 0
                if (matrix[i][0] == 0)
                    for (int j = 1; j < n; ++j)
                        matrix[i][j] = 0;

            if (r0)
                for (int j = 0; j < n; ++j)
                    matrix[0][j] = 0;
                    
            if (c0)
                for (int i = 0; i < m; ++i)
                    matrix[i][0] = 0;
        }
    };
    ```

### 有效的数独

请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。


> 数字 1-9 在每一行只能出现一次。
> 数字 1-9 在每一列只能出现一次。
> 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
 

注意：

> 一个有效的数独（部分已被填充）不一定是可解的。
> 只需要根据以上规则，验证已经填入的数字是否有效即可。
> 空白格用 '.' 表示。
 
```
示例 1：


输入：board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：true
示例 2：

输入：board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：false
解释：除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。 但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
```

代码：

1. 用一个哈希表遍历 3 遍（自己写的，效率很低）

    ```c++
    class Solution {
    public:
        bool isValidSudoku(vector<vector<char>>& board) {
            unordered_set<char> s;
            int m = board.size(), n = board[0].size();
            for (int i = 0; i < m; ++i)
            {
                s.clear();
                for (int j = 0; j < n; ++j)
                {
                    if (board[i][j] != '.')
                    {
                        if (s.find(board[i][j]) != s.end())
                            return false;
                        else
                            s.insert(board[i][j]);
                    }
                }
            }

            for (int j = 0; j < n; ++j)
            {
                s.clear();
                for (int i = 0; i < m; ++i)
                {
                    if (board[i][j] != '.')
                    {
                        if (s.find(board[i][j]) != s.end())
                            return false;
                        else
                            s.insert(board[i][j]);
                    }
                }
            }

            int x, y;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    s.clear();
                    for (int u = 0; u < 3; ++u)
                    {
                        for (int v = 0; v < 3; ++v)
                        {
                            x = i * 3 + u;
                            y = j * 3 + v;
                            if (board[x][y] != '.')
                            {
                                if (s.find(board[x][y]) != s.end())
                                    return false;
                                else
                                    s.insert(board[x][y]);
                            }
                        }
                    }
                }
            }
            return true;
        }
    };
    ```

1. 用 27 个哈希表，只遍历一遍（效率挺低的）

    ```c++
    class Solution {
    public:
        bool isValidSudoku(vector<vector<char>>& board) {
            vector<unordered_set<char>> sr(9), sc(9), sb(9);  // row, column, block
            char c;
            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    c = board[i][j];
                    if (c != '.')
                    {
                        if (sr[i].find(c) != sr[i].end() ||
                            sc[j].find(c) != sc[j].end() ||
                            sb[i / 3 * 3 + j / 3].find(c) != sc[i / 3 * 3 + j / 3].end())
                            return false;
                        else
                        {
                            sr[i].insert(c);
                            sc[j].insert(c);
                            sb[i / 3 * 3 + j / 3].insert(c);
                        }
                    }
                }
            }
            return true;
        }
    };
    ```

1. 用 27 个数组来替换哈希表（这个效率稍微高一点）

    ```c++
    class Solution {
    public:
        bool isValidSudoku(vector<vector<char>>& board) {
            vector<bool> sr[9], sc[9], sb[9];  // row, column, block
            for (auto &v: sr) v.resize(9);
            for (auto &v: sc) v.resize(9);
            for (auto &v: sb) v.resize(9);
            
            int c;
            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    c = board[i][j];
                    if (c != '.')
                    {
                        c = c - '1';
                        if (sr[i][c] || sc[j][c] || sb[i / 3 * 3 + j / 3][c])
                            return false;
                        else
                        {
                            sr[i][c] = true;
                            sc[j][c] = true;
                            sb[i / 3 * 3 + j / 3][c] = true;
                        }
                    }
                }
            }
            return true;
        }
    };
    ```

    其实我们只需要 27 组有 9 个位的比特来存储信息就够了。一个 32 位整数可以划分 3 组（`3 * 9 = 27 < 32`, `4 * 9 = 36 > 32`），因此只需要`27 / 3 = 9`个整数就可以存储所有信息了。

思考：当题目里出现“是否出现过”，或“出现了几次”时，可以想到哈希表或桶计数。有明确的范围的话用桶计数，没有明确范围用哈希表。

### 寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

 
```
示例 1：

输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
示例 2：

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
示例 3：

输入：nums1 = [0,0], nums2 = [0,0]
输出：0.00000
示例 4：

输入：nums1 = [], nums2 = [1]
输出：1.00000
```

代码：

1. 合并，排序，直接输出中位数。面试估计会被打，但效率还可以。

    ```c++
    class Solution {
    public:
        double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
            vector<int> m(nums1.size() + nums2.size());
            copy(nums1.begin(), nums1.end(), m.begin());
            copy(nums2.begin(), nums2.end(), m.begin()+nums1.size());
            sort(m.begin(), m.end());
            int len = nums1.size() + nums2.size();
            if (len % 2 == 0) return (m[len / 2 - 1] + m[len / 2]) / 2.0;
            return m[len / 2];
        }
    };
    ```

1. 建堆，效率很低很低，只能击败 5%。

    ```c++
    class Solution {
    public:
        priority_queue<int> qmax;
        priority_queue<int, vector<int>, greater<int>> qmin;
        void insert(int num)
        {
            if (qmax.empty()) qmax.push(num);
            else if (qmax.size() == qmin.size())
            {
                qmin.push(num);
                qmax.push(qmin.top());
                qmin.pop();
            }
            else
            {
                qmax.push(num);
                qmin.push(qmax.top());
                qmax.pop();
            }
        }
        
        double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
            for (const int num: nums1) insert(num);
            for (const int num: nums2) insert(num);
            if (qmax.size() == qmin.size()) return (qmax.top() + qmin.top()) / 2.0;
            return qmax.top();
        }
    };
    ```

1. 双指针计数，只要找到中间位置的就可以了

1. 折半查找

    ```c++
    class Solution {
    public:
        int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {
            /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
            * 这里的 "/" 表示整除
            * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
            * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
            * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
            * 这样 pivot 本身最大也只能是第 k-1 小的元素
            * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
            * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
            * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
            */

            int m = nums1.size();
            int n = nums2.size();
            int index1 = 0, index2 = 0;

            while (true) {
                // 边界情况
                if (index1 == m) {
                    return nums2[index2 + k - 1];
                }
                if (index2 == n) {
                    return nums1[index1 + k - 1];
                }
                if (k == 1) {
                    return min(nums1[index1], nums2[index2]);
                }

                // 正常情况
                int newIndex1 = min(index1 + k / 2 - 1, m - 1);
                int newIndex2 = min(index2 + k / 2 - 1, n - 1);
                int pivot1 = nums1[newIndex1];
                int pivot2 = nums2[newIndex2];
                if (pivot1 <= pivot2) {
                    k -= newIndex1 - index1 + 1;
                    index1 = newIndex1 + 1;
                }
                else {
                    k -= newIndex2 - index2 + 1;
                    index2 = newIndex2 + 1;
                }
            }
        }

        double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
            int totalLength = nums1.size() + nums2.size();
            if (totalLength % 2 == 1) {
                return getKthElement(nums1, nums2, (totalLength + 1) / 2);
            }
            else {
                return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
            }
        }
    };
    ```

### 比较含退格的字符串

给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

 

示例 1：

输入：S = "ab#c", T = "ad#c"
输出：true
解释：S 和 T 都会变成 “ac”。
示例 2：

输入：S = "ab##", T = "c#d#"
输出：true
解释：S 和 T 都会变成 “”。
示例 3：

输入：S = "a##c", T = "#a#c"
输出：true
解释：S 和 T 都会变成 “c”。

代码：

1. 使用栈。代码简洁，但效率较低。

```c++

```

1. 使用倒序双指针。代码较复杂，但效率高。

```c++

```

### 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

 
```
示例 1：

输入：strs = ["flower","flow","flight"]
输出："fl"
示例 2：

输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
```

代码：

纵向扫描。（或许可以省略掉找最短字符串的过程，优化一下还能更快）

```c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int idx_min = 0;
        for (int i = 0; i < strs.size(); ++i)
        {
            if (strs[idx_min].size() > strs[i].size()) idx_min = i;
        }

        int pos = 0;
        char ch;
        for (int i = 0; i < strs[idx_min].size(); ++i)
        {
            ch = strs[idx_min][i];
            for (int j = 0; j < strs.size(); ++j)
            {
                if (strs[j][i] != ch) return strs[idx_min].substr(0, i);
            }
        }
        return strs[idx_min];
    }
};
```

### 存在重复元素

给定一个整数数组，判断是否存在重复元素。

如果存在一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。

```
示例 1:

输入: [1,2,3,1]
输出: true
示例 2:

输入: [1,2,3,4]
输出: false
示例 3:

输入: [1,1,1,3,3,4,3,2,4,2]
输出: true
```

代码：

1. 哈希表

    ```c++
    class Solution {
    public:
        bool containsDuplicate(vector<int>& nums) {
            unordered_set<int> s;
            for (auto num: nums)
            {
                if (s.find(num) != s.end()) return true;
                else s.insert(num);
            }
            return false;
        }
    };
    ```

1. 排序

    ```c++
    class Solution {
    public:
        bool containsDuplicate(vector<int>& nums) {
            if (nums.size() == 1) return false;
            sort(nums.begin(), nums.end());
            for (int i = 1; i < nums.size(); ++i)
                if (nums[i] == nums[i-1]) return true;
            return false;
        }
    };
    ```

思考：这道题是否可以用位运算实现呢？不可以，因为我们不知识重复的元素到底重复了几次。不管出现两次还是三次，四次，似乎都没啥好办法。

### 环形子数组的最大和

给定一个由整数数组 A 表示的环形数组 C，求 C 的非空子数组的最大可能和。

在此处，环形数组意味着数组的末端将会与开头相连呈环状。（形式上，当0 <= i < A.length 时 C[i] = A[i]，且当 i >= 0 时 C[i+A.length] = C[i]）

此外，子数组最多只能包含固定缓冲区 A 中的每个元素一次。（形式上，对于子数组 C[i], C[i+1], ..., C[j]，不存在 i <= k1, k2 <= j 其中 k1 % A.length = k2 % A.length）

 
```
示例 1：

输入：[1,-2,3,-2]
输出：3
解释：从子数组 [3] 得到最大和 3
示例 2：

输入：[5,-3,5]
输出：10
解释：从子数组 [5,5] 得到最大和 5 + 5 = 10
示例 3：

输入：[3,-1,2,-1]
输出：4
解释：从子数组 [2,-1,3] 得到最大和 2 + (-1) + 3 = 4
示例 4：

输入：[3,-2,2,-3]
输出：3
解释：从子数组 [3] 和 [3,-2,2] 都可以得到最大和 3
示例 5：

输入：[-2,-3,-1]
输出：-1
解释：从子数组 [-1] 得到最大和 -1
```

代码：

1. 暴力法（超时）

    ```c++
    class Solution {
    public:
        int maxSubarraySumCircular(vector<int>& nums) {
            int ans = INT32_MIN, sum = 0;
            for (int i = 0; i < nums.size(); ++i)
            {
                sum = 0;
                for (int j = i; j < i + nums.size(); ++j)
                {
                    sum += nums[j % nums.size()];
                    ans = max(sum, ans);
                    if (sum < 0) sum = 0;
                }
            }
            return ans;
        }
    };
    ```

1. 贪心 + 后缀和

    ```c++
    class Solution {
    public:
        int maxSubarraySumCircular(vector<int>& nums) {
            int ans = INT_MIN, sum = 0;  // 数组中有负数，所以初始化为负的最小值
            // 第一遍，不考虑环形数组正常求解
            for (int i = 0; i < nums.size(); ++i)
            {
                sum += nums[i];
                ans = max(ans, sum);
                if (sum < 0) sum = 0;
            }

            // 考虑环形数组
            // 从右侧开始求后缀和
            vector<int> rightsum(nums.size());
            rightsum[nums.size() - 1] = nums.back();
            for (int i = nums.size() - 2; i > -1; --i)
            {
                rightsum[i] = rightsum[i+1] + nums[i];
            }

            // 求后缀和的最大值
            // 因为右侧不知道从哪一项开始，所以直接求最大就行了。我们也无需知道从哪一项开始
            vector<int> max_rightsum(nums.size());
            max_rightsum.back() = nums.back();
            for (int i = nums.size() - 2; i > -1; --i)
            {
                max_rightsum[i] = max(rightsum[i], max_rightsum[i+1]);
            }

            // 把数组切成两段，第一段是 [0, i-1]，第二段是 [i+1, end]。
            // 第一段的所有数字是一定要加上的，所以我们直接用 leftsum 来求
            // 第二段不知道真实的开始索引是哪个，但是我们也无需知道，因为有后缀和的最大值 max_rightsum
            int leftsum = 0;
            for (int i = 1; i < nums.size()-1; ++i)
            {
                leftsum += nums[i-1];
                ans = max(ans, leftsum + max_rightsum[i+1]);
            }
            return ans;
        }
    };
    ```

1. 前缀和 + 单调队列

### 数组中的第K个最大元素

给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

 
```
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```

代码：

1. 使用库函数`priority_queue`，速度击败 90%，内存击败 10%

    ```c++
    class Solution {
    public:
        int findKthLargest(vector<int>& nums, int k) {
            priority_queue<int, vector<int>, greater<int>> q;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (q.size() < k) q.push(nums[i]);
                else
                {
                    if (nums[i] > q.top())
                    {
                        q.pop();
                        q.push(nums[i]);
                    }
                }
            }
            return q.top();
        }
    };
    ```

    `priority_queue`默认使用的是`less<int>`作为仿函数，`q.top()`返回的是最大值。这里我们需要将每个数字和优先队列里的最小值作比较，因此使用`greater<int>`作为比较类。

1. 快速选择

    ```c++
    class Solution {
    public:
        int partition(vector<int> &nums, int left, int right)
        {
            int x = nums[right];
            int i = left - 1, j = left;
            while (j < right)
            {
                if (nums[j] < x) swap(nums[++i], nums[j]);
                ++j;  // 既不能写成 if (nums[j++] < x) swap(nums[++i], nums[j]);，也不能写成 if (nums[j] < x) swap(nums[++i], nums[j++]);
            }
            swap(nums[i+1], nums[right]);
            return i+1;
        }

        int randomized_partition(vector<int> &nums, int left, int right)
        {
            int idx = rand() % (right - left + 1) + left;  // 这里要注意，区间的长度并不是 nums.size()
            swap(nums[idx], nums[right]);
            return partition(nums, left, right);
        }

        int quick_select(vector<int> &nums, int k)
        {
            int idx = -1;
            int left = 0, right = nums.size() - 1;
            do
            {
                idx = randomized_partition(nums, left, right);  // 确定第 idx 小的值的索引
                if (idx < k) left = idx + 1;  // 每次缩小一半的搜索范围
                else if (idx > k) right = idx - 1;
            } while (idx != k);
            return nums[k];
        }

        int findKthLargest(vector<int>& nums, int k) {
            srand(time(NULL));
            int num = quick_select(nums, nums.size() - k);  // 最大的第 k 个元素改成最小的 n - k + 1个元素，如果 k > n / 2，则改成搜索最小元素
            return num;
        }
    };
    ```

    后来自己写的：

    ```c++
    class Solution {
    public:
        int partition(vector<int> &nums, int left, int right)
        {
            int x = nums[right];
            int i = left - 1, j = left;
            while (j < right)
            {
                if (nums[j] >= x) swap(nums[++i], nums[j]);
                ++j;
            }
            swap(nums[right], nums[i+1]);
            return i + 1;
        }

        int randomized_partition(vector<int> &nums, int left, int right)
        {
            int idx = rand() % (right - left + 1) + left;
            swap(nums[idx], nums[right]);
            return partition(nums, left, right);
        }

        void quick_select(vector<int> &nums, int left, int right, int k)
        {
            if (left > right) return;
            int idx = randomized_partition(nums, left, right);
            if (idx < k - 1) quick_select(nums, idx + 1, right, k);
            else if (idx > k - 1) quick_select(nums, left, idx - 1, k);
            else return;
        }

        int findKthLargest(vector<int>& nums, int k) {
            quick_select(nums, 0, nums.size() - 1, k);
            return nums[k-1];
        }
    };
    ```

1. 还有些排序的答案

    ```c++
    class Solution {
    public:
        int findKthLargest(vector<int>& nums, int k) {
            return sort(nums.begin(), nums.end(), greater<int>()), nums[k - 1];
        }
    };
    ```

    也可以计数排序。

### 旋转图像

给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

 
```
示例 1：


输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
示例 2：


输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
示例 3：

输入：matrix = [[1]]
输出：[[1]]
示例 4：

输入：matrix = [[1,2],[3,4]]
输出：[[3,1],[4,2]]
```

代码：

1. 辅助矩阵，这样需要额外的存储空间

    ```c++
    class Solution {
    public:
        void rotate(vector<vector<int>>& matrix) {
            int n = matrix.size();
            vector<vector<int>> matrix_copy(n, vector<int>(n));
            matrix_copy = matrix;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    matrix[j][n-i-1] = matrix_copy[i][j];
        }
    };
    ```

1. 找规律法

    第一个元素`matrix[0][0]`被放到了`matrix[0][n-1]`，`matrix[0][n-1]`的元素被放到了`matrix[n-1][n-1]`，这样找规律，最后发现`matrix[i][j]`位置的元素会被放到`matrix[j][n-i-1]`位置上。这样我们只需要临时变量`temp`保存一个位置的数据就可以了。

    虽然我们在推导的时候，是顺时针的顺序想的，但是在赋值的时候，是按逆时针顺序赋值的。这个不重要，只要应用好链式赋值法则就可以了，一般不会出错。

    ```c++
    class Solution {
    public:
        void rotate(vector<vector<int>>& matrix) {
            int n = matrix.size();
            int temp;
            for (int i = 0; i < n / 2; ++i)  // 具体需要对哪些元素做遍历，n / 2 和 (n+1) / 2 这俩条件不好想出来。我也不知道该咋推出来。
            {
                for (int j = 0; j < (n + 1) / 2; ++j)
                {
                    temp = matrix[i][j];
                    matrix[i][j] = matrix[n-j-1][i];
                    matrix[n-j-1][i] = matrix[n-i-1][n-j-1];
                    matrix[n-i-1][n-j-1] = matrix[j][n-i-1];
                    matrix[j][n-i-1] = temp;
                }
            }
        }
    };
    ```

    想到上次旋转数组的循环替换，这道题也是一个道理，不能用`swap()`来代替循环赋值。

1. 翻转法

    首先水平翻转，把上面的元素翻转到下面去，然后主对角线翻转，交换左下和右上两块区域的位置。

    ```c++
    class Solution {
    public:
        void rotate(vector<vector<int>>& matrix) {
            int n = matrix.size();
            for (int i = 0; i < n / 2; ++i)
                for (int j = 0; j < n; ++j)
                    swap(matrix[i][j], matrix[n-i-1][j]);

            for (int i = 0; i < n; ++i)
                for (int j = i+1; j < n; ++j)
                    swap(matrix[i][j], matrix[j][i]);
        }
    };
    ```

    为什么这样做就可以等价于顺时针旋转 90 度？数学上有什么说法没有？假如下次改成逆时钟旋转 90 度，该怎么翻转？

    交换这两者的翻转顺序似乎无法得到正确的答案，为什么翻转不满足交换律？

### 颜色分类

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

 
```
示例 1：

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
示例 2：

输入：nums = [2,0,1]
输出：[0,1,2]
示例 3：

输入：nums = [0]
输出：[0]
示例 4：

输入：nums = [1]
输出：[1]
```

代码：

1. 统计各个元素的数量，然后直接给数组赋值。

1. 单指针两次遍历

    ```c++
    class Solution {
    public:
        void sortColors(vector<int>& nums) {
            int n = nums.size();
            int left = 0, right = n - 1;
            while (left < right)
            {
                while (left < right && nums[left] == 0) ++left;
                while (left < right && nums[right] != 0) --right;
                swap(nums[left], nums[right]);
            }
            right = n - 1;
            while (left < right)
            {
                while (left < right && nums[left] == 1) ++left;
                while (left < right && nums[right] != 1) --right;
                swap(nums[left], nums[right]);
            }
        }
    };
    ```

    第一遍遍历，`left`维护了一个边界，`nums[left]`之左的元素都是 0。第二遍遍历，`left`维护的是 1 的边界。

    这道题和排序的区别是，这道题的元素种类是固定的，并且知道最小最大值，而排序的元素种类是不定的。

1. 双指针一次遍历

    ```c++
    class Solution {
    public:
        void sortColors(vector<int>& nums) {
            int n = nums.size();
            int left = -1, right = n, p = 0;
            while (p < right)
            {
                if (nums[p] == 0) swap(nums[++left], nums[p++]);  // 一开始没有想到这个 p++，以后该怎么避免这种情况呢？
                else if (nums[p] == 2) swap(nums[--right], nums[p]);
                else ++p;
            }
        }
    };
    ```

    `left`维护的是 0 的边界，`right`维护的是 1 的边界。`p`指向当前值，若当前值为 0，则和左侧交换，因为左侧长度增加了，所以`p`指向的当前值的位置也会向右挪动一位，所以需要对`p`进行递增。

### 螺旋矩阵 II

给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

 
```
示例 1：

输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
示例 2：

输入：n = 1
输出：[[1]]
```

代码：

1. 模拟

    ```c++
    class Solution {
    public:
        vector<vector<int>> generateMatrix(int n) {
            vector<vector<int>> ans(n, vector<int>(n));
            int i = 0, j = 0;
            int dir = 0, layer = 0;
            int num = 1;
            while (num <= n * n)
            {
                switch (dir)
                {
                    case 0:
                    i = layer;
                    j = layer;
                    if (num == n * n) ans[i][j] = num++;
                    else while (j < n - layer - 1) ans[i][j++] = num++;
                    dir = 1;
                    break;

                    case 1:
                    while (i < n - layer - 1) ans[i++][j] = num++;
                    dir = 2;
                    break;

                    case 2:
                    while (j > layer) ans[i][j--] = num++;
                    dir = 3;
                    break;

                    case 3:
                    while (i > layer) ans[i--][j] = num++;
                    ++layer;
                    dir = 0;
                    break;
                }
            }
            return ans;
        }
    };
    ```

1. 别人的代码，写得挺好的

    ```java
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int up = 0, down = n - 1, left = 0, right = n - 1, index = 1;
        while(index <= n * n){
            for(int i = left; i <= right; i++){
                res[up][i] = index++;
            }
            up++;
            for(int i = up; i <= down; i++){
                res[i][right] = index++;
            }
            right--;
            for(int i = right; i >= left; i--){
                res[down][i] = index++;
            }
            down--;
            for(int i = down; i >= up; i--){
                res[i][left] = index++;
            }
            left++;
        }
        return res;
    }
    ```

### 字符串相加

给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和并同样以字符串形式返回。

你不能使用任何內建的用于处理大整数的库（比如 BigInteger）， 也不能直接将输入的字符串转换为整数形式。

 
```
示例 1：

输入：num1 = "11", num2 = "123"
输出："134"
示例 2：

输入：num1 = "456", num2 = "77"
输出："533"
示例 3：

输入：num1 = "0", num2 = "0"
输出："0"
```

代码：

1. 模拟

    这道题和两个链表求和挺像的。

    ```c++
    class Solution {
    public:
        string addStrings(string num1, string num2) {
            int x1, x2, carry = 0;
            int pos1 = num1.size() - 1, pos2 = num2.size() - 1;
            string ans(max(num1.size(), num2.size()) + 1, ' ');
            int pos = ans.size() - 1;
            while (pos1 > -1 || pos2 > -1)
            {
                x1 = pos1 > -1 ? num1[pos1--] - '0' : 0;
                x2 = pos2 > -1 ? num2[pos2--] - '0' : 0;
                ans[pos--] = (x1 + x2 + carry) % 10 + '0';
                carry = x1 + x2 + carry >= 10;
            }
            if (carry) ans[0] = 1 + '0';
            return ans[0] == ' ' ? ans.substr(1, ans.npos) : ans;
        }
    };
    ```

### 最长回文串

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

注意:
假设字符串的长度不会超过 1010。

```
示例 1:

输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
```

代码：

1. 桶计数

    ```c++
    class Solution {
    public:
        int longestPalindrome(string s) {
            vector<int> cnt('z' - 'A' + 1);
            for (int i = 0; i < s.size(); ++i)
                ++cnt[s[i] - 'A'];
            int ans = 0;
            bool odd = false;
            for (int i = 0; i < cnt.size(); ++i)
            {
                if (cnt[i] > 1) ans += cnt[i] / 2 * 2;
                if (cnt[i] % 2 == 1) odd = true;
            }
            if (odd) return ans + 1;
            return ans;
        }
    };
    ```

    大写字母在小写字母的前面。另外若`ans`小于`s.size()`，那么说明存在奇数个数的字符，就不需要额外判断了。

### 螺旋矩阵

给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

 
```
示例 1：


输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：


输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

代码：

1. 抄别人写的

    ```c++
    class Solution {
    public:
        vector<int> spiralOrder(vector<vector<int>>& matrix) {
            int top = 0, bottom = matrix.size() - 1, left = 0, right = matrix[0].size() - 1;
            int n = (int) matrix.size() * (int) matrix[0].size();
            vector<int> ans(n);
            int pos = 0, p = 0;
            while (n > 0)
            {
                p = left;
                while (p <= right && n-- > 0) ans[pos++] = matrix[top][p++];  // 这个 n-- 必须有，而且必须写到后面
                ++top;
                p = top;
                while (p <= bottom && n-- > 0) ans[pos++] = matrix[p++][right];
                --right;
                p = right;
                while (p >= left && n-- > 0) ans[pos++] = matrix[bottom][p--];
                --bottom;
                p = bottom;
                while (p >= top && n-- > 0) ans[pos++] = matrix[p--][left];
                ++left;
            }
            return ans;
        }
    };
    ```

### 和大于等于 target 的最短子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

 

```
示例 1：

输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
示例 2：

输入：target = 4, nums = [1,4,4]
输出：1
示例 3：

输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
```

代码：

1. 前缀和 + 二重循环，效率很低

    ```c++
    class Solution {
    public:
        int minSubArrayLen(int target, vector<int>& nums) {
            int n = nums.size();
            vector<int> pre(n+1);
            pre[0] = 0;
            for (int i = 1; i <= n; ++i) pre[i] = pre[i-1] + nums[i-1];
            int ans = INT32_MAX;
            for (int i = 1; i <= n; ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    if (pre[i] - pre[j] >= target)
                    {
                        ans = min(ans, i - j);
                        break;  // 后面的长度一定大于当前的长度，再往后找是没必要的
                        // 加上这样的剪枝，代码可以通过，但是只能击败 5%
                    }
                }
            }
            return ans == INT32_MAX ? 0 : ans;
        }
    };
    ```

1. 前缀和 + 二分查找右边界

    正整数的前缀和一定是递增的，所以可以用二分查找。

    ```c++
    class Solution {
    public:
        int minSubArrayLen(int target, vector<int>& nums) {
            int n = nums.size();
            vector<int> pre(n+1);
            pre[0] = 0;
            for (int i = 1; i <= n; ++i) pre[i] = pre[i-1] + nums[i-1];
            int ans = INT32_MAX;
            int t;
            int left, right, mid;
            for (int i = 1; i <= n; ++i)
            {
                t = pre[i] - target;
                left = 0, right = i - 1;
                while (left <= right)
                {
                    mid = left + (right - left) / 2;
                    if (pre[mid] < t) left = mid + 1;
                    else if (pre[mid] > t) right = mid - 1;
                    else left = mid + 1;
                }
                if (right > -1)
                    ans = min(ans, i - right); 
            }
            return ans == INT32_MAX ? 0 : ans;
        }
    };
    ```

1. 滑动窗口

    ```c++
    class Solution {
    public:
        int minSubArrayLen(int target, vector<int>& nums) {
            int n = nums.size();
            int left = 0, right = -1;
            int sum = 0;  // 题目指定了 nums 中都是正整数，所以初始和为 0 可行
            int ans = INT32_MAX;
            do  // 与 right = -1 配合，初始化滑动窗口，不然 left <= right 无法通过
            {
                if (sum < target && right < n - 1) sum += nums[++right];  // 滑动逻辑和答案判断分离
                // 这里为什么不能写成 right++？这样写应该更好吧？
                else sum -= nums[left++];
                if (sum >= target) ans = min(ans, right - left + 1);  // 这样写似乎不对，有时间了再看看
            } while (left <= right && right < n);
            return ans == INT32_MAX ? 0 : ans;
        }
    };
    ```

    滑动窗口 rethinking：

    ```cpp
    class Solution {
    public:
        int minSubArrayLen(int target, vector<int>& nums) {
            int n = nums.size();
            int left = 0, right = 0;
            int ans = n + 1;
            int sum = nums[0];
            while (left <= right && right < n)
            {
                while (right < n && sum < target)
                {
                    ++right;
                    if (right < n)  // 之所以这里需要判断是因为 right 可能在最后一个元素上停留多次
                        sum += nums[right];
                }

                while (left <= right && sum - nums[left] >= target)
                    sum -= nums[left++];

                if (sum >= target)
                {
                    ans = min(ans, right - left + 1);
                    ++right;
                    if (right < n)
                        sum += nums[right];
                }
            }
            if (ans > n) return 0;
            return ans;
        }
    };
    ```

    为了使边界条件更清楚，我们沿用对循环的分析。在代码 2 中，我们采用先改变状态，再判断状态的方式。

    这道题比较复杂，先主要分析下这个：

    ```cpp
    while (right < n && sum < target)
    {
        ++right;
        if (right < n)
            sum += nums[right];
    }
    ```

    这里之所以用到`if`，是因为先改变索引`right`，然后才判断状态。`for`循环中，索引永远是最后改变的，所以不需要额外判断。在循环中，我们每次改变状态，都要保证改变完后的状态有效。即 改变状态 -> 判断状态有效 -> 使用状态。

思考：如果题目改正数组中的数为整数，该怎么做呢？

### 乘积小于 K 的子数组

给定一个正整数数组 nums和整数 k ，请找出该数组内乘积小于 k 的连续的子数组的个数。

 

```
示例 1:

输入: nums = [10,5,2,6], k = 100
输出: 8
解释: 8 个乘积小于 100 的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
示例 2:

输入: nums = [1,2,3], k = 0
输出: 0
```

代码：

1. 

### 轮转数组

给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

 

示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
示例 2:

输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]
 

提示：

1 <= nums.length <= 105
-231 <= nums[i] <= 231 - 1
0 <= k <= 105
 
代码：

1. 复制一遍数组，然后截取中间一段

    ```cpp
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            k = k % nums.size();
            int N = nums.size();
            nums.resize(nums.size() * 2);
            nums.insert(nums.begin()+N, nums.begin(), nums.begin()+N);
            int i = 0, j = N - k;
            while (i < N)
            {
                nums[i++] = nums[j++];
            }
            nums.resize(N);
        }
    };
    ```

    对比下答案里给的解法：

    ```cpp
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            int n = nums.size();
            vector<int> newArr(n);
            for (int i = 0; i < n; ++i) {
                newArr[(i + k) % n] = nums[i];  // 似乎环状的算法都会用到取模
            }
            nums.assign(newArr.begin(), newArr.end());
        }
    };
    ```

1. 三次翻转数组

    很简洁，不知道怎么想出来的。

    ```cpp
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            k = k % nums.size();
            reverse(nums.begin(), nums.end());
            reverse(nums.begin(), nums.begin() + k);
            reverse(nums.begin() + k, nums.end());
        }
    };
    ``` 

1. 自己写的。。。不知道啥算法

    ```cpp
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            if (k == 0) return;
            int count = 0, pos = 0, start_pos = 0;
            int temp1 = nums[0], temp2;
            while (count < nums.size())
            {
                pos = (pos + k) % nums.size();
                temp2 = nums[pos];
                nums[pos] = temp1;
                temp1 = temp2;
                ++count;

                if (pos == start_pos)
                {
                    start_pos++;
                    if (start_pos >= nums.size()) return;
                    pos = start_pos;
                    temp1 = nums[pos];
                }
            }
        }
    };
    ```

    答案里的环状替换：

    ```cpp
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            int n = nums.size();
            k = k % n;
            int count = gcd(k, n);
            for (int start = 0; start < count; ++start) {
                int current = start;
                int prev = nums[start];
                do {
                    int next = (current + k) % n;
                    swap(nums[next], prev);
                    current = next;
                } while (start != current);
            }
        }
    };
    ```

### 数组中和为 0 的三个数

给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
示例 2：

输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
示例 3：

输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。
```
 

提示：

3 <= nums.length <= 3000
-105 <= nums[i] <= 105

代码：

1. 双指针

    ```cpp
    class Solution {
    public:
        vector<vector<int>> threeSum(vector<int>& nums) {
            vector<vector<int>> ans;
            sort(nums.begin(), nums.end());
            int n = nums.size();
            int start = 0, l = 1, r = n - 1;
            int target;
            int s;
            while (start < n)
            {
                if (nums[start] > 0)  // 剪枝。如果第一个数已经大于 0 了，那么后面再怎么加，也不可能满足要求。
                    return ans;
                target = - nums[start];
                l = start + 1;
                r = n - 1;
                while (l < r)
                {
                    s = nums[l] + nums[r];
                    if (s < target)
                        do ++l; while (l < r && nums[l] == nums[l-1]);
                    else if (s > target)
                        do --r; while (l < r && nums[r] == nums[r+1]);
                    else
                    {
                        ans.push_back(vector<int>({nums[start], nums[l], nums[r]}));
                        do ++l; while (l < r && nums[l] == nums[l-1]);
                        do --r; while (l < r && nums[r] == nums[r+1]);
                    }
                }
                int prev = nums[start];
                do ++start; while (start < n && nums[start] == prev);
            }
            return ans;
        }
    };
    ```

    题目说答案中不可以有重复的结果，那么我们就直接想到排序了。排完序后自然地想到双指针或二分查找。

## 链表

### 从尾到头打印链表

> 输入一个链表的头结点，按照 从尾到头 的顺序返回节点的值。

> 返回的结果用数组存储。

样例：

```
输入：[2, 3, 5]
返回：[5, 3, 2]
```

代码：

这道题倒是不难，不过思维要发散，需要会多种写法。

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

### 环形链表

给定一个链表，判断链表中是否有环。

如果链表中存在环，则返回 true 。 否则，返回 false 。

代码：

可以用哈希表，也可以用快慢指针。

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

### 反转链表

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

1. 递归法

    `head->next->next = head; head->next = nullptr;`这两句看得不是很懂。

    懂了。`head->next->next = heaad;`其实就是翻转两个相邻节点，`head->next = nullptr;`是因为，递归是从尾节点向前进行的，所以我们一直保证当前的节点的头部指向`nullptr`就行了。这样到第一个元素时，也能保证它的`next`是个`nullptr`。

    其实这是一个后序遍历。因为为了得到当前链表反转后的头节点，我们必须知道子链表反转后的头节点。我们只要把子链表反转后的头节点返回就可以了。

    我们定义递归函数返回的是反转链表后的头节点。其实这个返回值对我们反转当前节点没有什么帮助。

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

### 合并两个排序的链表（合并两个有序链表）

代码：

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

1. 直接使用指针

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
            if (p1) p->next = p1;
            if (p2) p->next = p2;
            return dummy_head->next;
        }
    };
    ```

### 两个链表的第一个公共结点（链表相交）（两个链表的第一个重合节点）

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

1. 双指针法。

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
                if (pnode_a) pnode_a = pnode_a->next;  // 为什么要写成 if else 的形式？
                else pnode_a = headB;
                if (pnode_b) pnode_b = pnode_b->next;
                else pnode_b = headA;
            }
            return pnode_a;
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

1. 将链表中的值复制到数组中，然后用双指针

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

## 树

### 遍历

#### 二叉树的前序遍历（先序遍历）

给你二叉树的根节点 root ，返回它节点值的 前序 遍历。

```
示例 1：

输入：root = [1,null,2,3]
输出：[1,2,3]
示例 2：

输入：root = []
输出：[]
```

代码：

1. 递归法

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> res;
        void pre(TreeNode *r)
        {
            if (!r) return;
            res.push_back(r->val);
            if (r->left) pre(r->left);
            if (r->right) pre(r->right);
        }
        vector<int> preorderTraversal(TreeNode* root) {
            pre(root);
            return res;
        }
    };
    ```

1. 栈（广度优先）

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> preorderTraversal(TreeNode* root) {
            vector<int> res;
            if (!root) return res;
            stack<TreeNode*> s;
            s.push(root);
            TreeNode *r;
            while (!s.empty())
            {
                r = s.top();
                s.pop();
                res.push_back(r->val);
                if (r->right) s.push(r->right);
                if (r->left) s.push(r->left);
            }
            return res;
        }
    };
    ```

1. 栈（深度优先）

    我觉得这个不应该叫做深度优先，其实树的先序遍历本身就是深度优先。这种方法只不过是只存储左节点，不存储右节点，从而节约了一半左右的内存。

    下次可以将这个方法改名为“栈（节约内存版）”。

    我觉得这个版本才是真正的栈的做法。因为我们的需求是记录自己在一棵树中的路径，记录一个 path。这个方法刚好满足了这个需求。而上面的“栈（广度优先）”，只是取巧，把右节点也存进去了，从而简化了代码而已。这种方法并不是通用的。

    比如对于中序遍历，如果我们先存右节点，再处理当前节点，再存左节点，会遇到行为与预期不符的情况，因为我们还是先处理当前节点了。我们需要做的是先找到一条 path，然后边退边处理。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> res;
        vector<int> preorderTraversal(TreeNode* root) {
            vector<int> res;
            stack<TreeNode*> s;
            TreeNode *r = root;
            while (r || !s.empty())  // r 存在说明还有左节点没遍历，s 非空表示还有右节点没遍历
            {
                while (r)  // 每遇到一个节点，先向左下一直走到底
                {
                    res.push_back(r->val);
                    s.push(r);
                    r = r->left;
                }

                r = s.top()->right;  // 如果左下方已经搜索完，则再折回来搜索右侧节点
                s.pop();
            }
            return res;
        }
    };
    ```

1. Morris 遍历（看不懂，有时间再看）

    ```c++
    class Solution {
    public:
        vector<int> preorderTraversal(TreeNode *root) {
            vector<int> res;
            if (root == nullptr) {
                return res;
            }

            TreeNode *p1 = root, *p2 = nullptr;

            while (p1 != nullptr) {
                p2 = p1->left;
                if (p2 != nullptr) {
                    while (p2->right != nullptr && p2->right != p1) {
                        p2 = p2->right;
                    }
                    if (p2->right == nullptr) {
                        res.emplace_back(p1->val);
                        p2->right = p1;
                        p1 = p1->left;
                        continue;
                    } else {
                        p2->right = nullptr;
                    }
                } else {
                    res.emplace_back(p1->val);
                }
                p1 = p1->right;
            }
            return res;
        }
    };
    ```

#### 二叉树的中序遍历

给定一个二叉树的根节点 root ，返回它的 中序 遍历。

```
示例 1：

输入：root = [1,null,2,3]
输出：[1,3,2]
示例 2：

输入：root = []
输出：[]
```

代码：

1. 递归

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        vector<int> ans;
        void dfs(TreeNode *root)
        {
            if (!root) return;
            dfs(root->left);
            ans.push_back(root->val);
            dfs(root->right);
        }

        vector<int> inorderTraversal(TreeNode* root) {
            dfs(root);
            return ans;
        }
    };
    ```

1. 迭代，栈

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> inorderTraversal(TreeNode* root) {
            vector<int> res;
            if (!root) return res;
            stack<TreeNode*> s;  // 用来记录左路径
            TreeNode *r = root;  // 注意一开始 s 中不把 root 入栈，反而把 r 赋值为根节点
            while (r || !s.empty())  // r 存在表示开始遍历右子树，s 不为空表示路径还没回退完
            {
                // 如果这里写成下面的形式，那么根节点或右子树的根节点就不会被压栈
                // while (r->left)
                // {
                //    s.push(r->left);
                //    r = r->left;
                // }
                // 如果写成下面这种形式，那么最后一个左叶子节点就不会被压栈
                // while (r->left)
                // {
                //    s.push(r);
                //    r = r->left;
                // }
                // 因此只有下面这种写法，可以记录从根节点到左叶子节点的路径
                while (r)
                {
                    s.push(r);
                    r = r->left;
                }
                r = s.top();
                s.pop();
                res.push_back(r->val);  // 只是把遍历当前节点的顺序改到了这里
                r = r->right;  // 这里很巧妙，因为要防止重复搜索左路径，所以无论 right 是否存在，都把 r 设置为 r->right。若 right 不为 null，那么直接进入右子树进行搜索；如果 right 为 nullptr，那么就不会进入 while (r) 循环中，从而避免了重复搜索左路径。妙啊。
            }
            return res;
        }
    };
    ```

1. morris 遍历（不会）

#### 二叉树的后序遍历

代码：

1. 迭代

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> postorderTraversal(TreeNode* root) {
            vector<int> res;
            TreeNode *r = root, *prev = nullptr;  // prev = nullptr 很巧妙
            stack<TreeNode*> s;
            while (r || !s.empty())
            {
                while (r)
                {
                    s.push(r);
                    r = r->left;
                }

                r = s.top();
                s.pop();

                if (r->right && r->right != prev)  // 如果右下方还有节点，且没访问过
                {
                    s.push(r);  // 将当前节点再压回栈里
                    r = r->right;  // 向右下方探索
                }
                else  // 前面的 while 保证了当前节点没有左节点，这里的 if 保证了没有右节点，或者是正在向上“回退”的过程中
                {
                    res.push_back(r->val);
                    prev = r;  // 更新当前正在回退的节点，防止再搜索右节点
                    r = nullptr;  // 防止再搜索左子树
                }
            }
            return res;
        }
    };
    ```

#### 二叉树的层序遍历

给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。

```
示例：
二叉树：[3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层序遍历结果：

[
  [3],
  [9,20],
  [15,7]
]
```

代码：

1. 使用`nullptr`分层

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<vector<int>> levelOrder(TreeNode* root) {
            vector<vector<int>> res;
            if (!root) return res;
            
            res.push_back(vector<int>());
            queue<TreeNode*> q;
            q.push(root);
            q.push(nullptr);  // 注意这里
            TreeNode *r;
            while (q.size() > 1)  // 注意这里，退出循环的条件
            {
                r = q.front();
                q.pop();
                if (!r)  // 如果遇到 nullptr，那么说明上层结束，此时再入队一个 nullptr，表示下层的开始
                {
                    res.push_back(vector<int>());
                    q.push(nullptr);
                    continue;
                }

                res.back().push_back(r->val);
                if (r->left) q.push(r->left);
                if (r->right) q.push(r->right);
            }
            return res;
        }
    };
    ```

1. 使用`for`分层

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<vector<int>> levelOrder(TreeNode* root) {
            vector<vector<int>> res;
            if (!root) return res;
            queue<TreeNode*> q;
            q.push(root);
            TreeNode *r;
            int num;
            while (!q.empty())
            {
                res.push_back(vector<int>());
                num = q.size();
                for (int i = 0; i < num; ++i)
                {
                    r = q.front();
                    q.pop();
                    res.back().push_back(r->val);
                    if (r->left) q.push(r->left);
                    if (r->right) q.push(r->right);
                }
            }
            return res;
        }
    };
    ```

1. 递归（这个感觉不对，bfs咋可能用递归）

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<vector<int>> res;
        void bfs(TreeNode *r, int level)
        {
            if (res.size() == level) res.push_back(vector<int>());
            res[level].push_back(r->val);
            if (r->left) bfs(r->left, level+1);
            if (r->right) bfs(r->right, level+1);
        }
        vector<vector<int>> levelOrder(TreeNode* root) {
            if (!root) return res;
            bfs(root, 0);
            return res;
        }
    };
    ```

#### 不需要分层的层序遍历（从上到下打印二叉树）

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

例如:

给定二叉树: `[3,9,20,null,null,15,7]`,

```
 3
/ \
9  20
 /  \
15   7
```

返回

```
[3,9,20,15,7]
```
  
代码：

1. 不需要分层的层序遍历

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        vector<int> levelOrder(TreeNode* root) {
            vector<int> ans;
            if (!root) return ans;
            queue<TreeNode*> q;
            q.push(root);
            TreeNode *r;
            while (!q.empty())
            {
                r = q.front();
                q.pop();
                ans.push_back(r->val);
                if (r->left) q.push(r->left);
                if (r->right) q.push(r->right);
            }
            return ans;
        }
    };
    ```

1. 使用 null

#### 二叉树的层序遍历 II

给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

例如：

```
给定二叉树 [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其自底向上的层序遍历为：

[
  [15,7],
  [9,20],
  [3]
]
```

代码：

1. 自己写的，用了 stack

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<vector<int>> levelOrderBottom(TreeNode* root) {
            vector<vector<int>> ans;
            if (!root) return ans;
            stack<TreeNode*> s;
            vector<int> nums;
            queue<TreeNode*> q;
            q.push(root);
            TreeNode *r;
            while (!q.empty())
            {
                int num = q.size();
                nums.push_back(num);
                for (int i = 0; i < num; ++i)
                {
                    r = q.front();
                    q.pop();
                    s.push(r);
                    if (r->right) q.push(r->right);
                    if (r->left) q.push(r->left);
                }
            }
            
            for (int i = nums.size() - 1; i > -1; --i)
            {
                ans.push_back(vector<int>());
                for (int j = 0; j < nums[i]; ++j)
                {
                    ans.back().push_back(s.top()->val);
                    s.pop();
                }
            }

            return ans;
        }
    };
    ```

1. 直接`reverse()`反转层序遍历的答案就可以了

    ```c++
    class Solution {
    public:
        vector<vector<int>> levelOrderBottom(TreeNode* root) {
            auto levelOrder = vector<vector<int>>();
            if (!root) {
                return levelOrder;
            }
            queue<TreeNode*> q;
            q.push(root);
            while (!q.empty()) {
                auto level = vector<int>();
                int size = q.size();
                for (int i = 0; i < size; ++i) {
                    auto node = q.front();
                    q.pop();
                    level.push_back(node->val);
                    if (node->left) {
                        q.push(node->left);
                    }
                    if (node->right) {
                        q.push(node->right);
                    }
                }
                levelOrder.push_back(level);
            }
            reverse(levelOrder.begin(), levelOrder.end());
            return levelOrder;
        }
    };
    ```


### 重建二叉树（从前序与中序遍历序列构造二叉树）

> 输入一棵二叉树前序遍历和中序遍历的结果，请重建该二叉树。
>
> 二叉树中每个节点的值都互不相同；
> 
> 输入的前序遍历和中序遍历一定合法；
>
> 样例：
>
> 给定：
> 前序遍历是：[3, 9, 20, 15, 7]
> 中序遍历是：[9, 3, 15, 20, 7]
> 
> 返回：[3, 9, 20, null, null, 15, 7, null, null, > null, null]
> 返回的二叉树如下所示：
>
> ```
>     3
>    / \
>   9  20
>     /  \
>    15   7
> ```

**分析**：

利用前序遍历的结果找到根节点，再到中序遍历的结果中找到左子树的节点数量，最后递归构建左子树和右子树即可。

代码：

1. 官方答案

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:

        unordered_map<int,int> pos;

        TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
            int n = preorder.size();
            for (int i = 0; i < n; i ++ )
                pos[inorder[i]] = i;
            return dfs(preorder, inorder, 0, n - 1, 0, n - 1);
        }

        TreeNode* dfs(vector<int>&pre, vector<int>&in, int pl, int pr, int il, int ir)
        {
            if (pl > pr) return NULL;
            int k = pos[pre[pl]] - il;
            TreeNode* root = new TreeNode(pre[pl]);
            root->left = dfs(pre, in, pl + 1, pl + k, il, il + k - 1);
            root->right = dfs(pre, in, pl + k + 1, pr, il + k + 1, ir);
            return root;
        }
    };
    ```

1. 自己写的。思路一样，但变量更易懂一些。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        unordered_map<int, int> m;

        TreeNode* dfs(vector<int> &preorder, vector<int> &inorder, int inorder_bound_left, int inorder_bound_right, int preorder_start)
        {
            TreeNode *root = new TreeNode(preorder[preorder_start]);
            if (inorder_bound_left == inorder_bound_right) return root;  // 只剩一个节点

            int pos_root = m[root->val];
            int num_left = pos_root - inorder_bound_left;  // 左子树的节点数目
            int num_right = inorder_bound_right - pos_root;  // 右子树的节点数目
            if (num_left)
                root->left = dfs(preorder, inorder, inorder_bound_left, pos_root - 1, preorder_start + 1);
            if (num_right)
                root->right = dfs(preorder, inorder, pos_root + 1, inorder_bound_right, preorder_start + num_left + 1);
            return root;
        }

        TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
            if (preorder.empty()) return nullptr;
            for (int i = 0; i < inorder.size(); ++i)
                m[inorder[i]] = i;

            TreeNode *root = dfs(preorder, inorder, 0, inorder.size()-1, 0);
            return root;
        }
    };
    ```

1. 自己后来又写的

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        unordered_map<int, int> m;
        TreeNode* dfs(vector<int> &preorder, vector<int> &inorder, int pre_left, int pre_right, int in_left, int in_right)
        {
            if (pre_left > pre_right) return nullptr;
            int idx_root = m[preorder[pre_left]];
            int len_left = idx_root - in_left;
            int len_right = in_right - idx_root;
            TreeNode *r = new TreeNode(preorder[pre_left]);
            r->left = dfs(preorder, inorder, pre_left+1, pre_left+len_left, in_left, idx_root-1);
            r->right = dfs(preorder, inorder, pre_left+len_left+1, pre_right, idx_root+1, in_right);
            return r;
        }

        TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
            for (int i = 0; i < inorder.size(); ++i)
                m[inorder[i]] = i;
            TreeNode *root = dfs(preorder, inorder, 0, preorder.size()-1, 0, inorder.size()-1);
            return root;
        }
    };
    ```

1. 其实`inorder`可以不作为参数传进去，所以后来自己又写了一个版本

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        unordered_map<int, int> m;
        TreeNode* dfs(vector<int> &preorder, int pre_left, int pre_right, int in_left, int in_right)
        {
            if (pre_left > pre_right) return nullptr;
            int root_val = preorder[pre_left];
            int pos = m[root_val];
            int len_left = pos - in_left, len_right = in_right - pos;
            TreeNode *root = new TreeNode(root_val);
            root->left = dfs(preorder, pre_left+1, pre_left+len_left, in_left, pos-1);
            root->right = dfs(preorder, pre_left+len_left+1, pre_right, pos+1, len_right);
            return root;
        }

        TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
            for (int i = 0; i < inorder.size(); ++i)
                m[inorder[i]] = i;
            return dfs(preorder, 0, preorder.size()-1, 0, inorder.size()-1);
        }
    };
    ```

1. 官方给的一种迭代解法（没看

    ```c++
    class Solution {
    public:
        TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
            if (!preorder.size()) {
                return nullptr;
            }
            TreeNode* root = new TreeNode(preorder[0]);
            stack<TreeNode*> stk;
            stk.push(root);
            int inorderIndex = 0;
            for (int i = 1; i < preorder.size(); ++i) {
                int preorderVal = preorder[i];
                TreeNode* node = stk.top();
                if (node->val != inorder[inorderIndex]) {
                    node->left = new TreeNode(preorderVal);
                    stk.push(node->left);
                }
                else {
                    while (!stk.empty() && stk.top()->val == inorder[inorderIndex]) {
                        node = stk.top();
                        stk.pop();
                        ++inorderIndex;
                    }
                    node->right = new TreeNode(preorderVal);
                    stk.push(node->right);
                }
            }
            return root;
        }
    };
    ```

### 二叉树的下一个节点

给定一棵二叉树的其中一个节点，请找出中序遍历序列的下一个节点。

分析：分类讨论，背会就好。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode *father;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL), father(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* inorderSuccessor(TreeNode* p) {
        if (p->right)
        {
            p = p->right;
            while (p->left != nullptr)
                p = p->left;
            return p;
        }
        
        while (p->father && p == p->father->right) p = p->father;
        return p->father;
        
    }
};
```

### 树的子结构

> 输入两棵二叉树 A，B，判断 B 是不是 A 的> 子结构。
> 
> 我们规定空树不是任何树的子结构。
>
> 样例：
> 
> 树 A：
> 
> ```
>      8
>     / \
>    8   7
>   / \
>  9   2
>     / \
>    4   7
> ```
> 
> 树 B：
> 
> ```
>    8
>   / \
>  9   2
> ```
>
> 返回 true

代码：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool hasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (!pRoot1 || !pRoot2)
            return false;
        if (isSame(pRoot1, pRoot2))
            return true;
        return hasSubtree(pRoot1->left, pRoot2) || hasSubtree(pRoot1->right, pRoot2);
    }
    
    bool isSame(TreeNode *pRoot1, TreeNode *pRoot2)
    {
        if (!pRoot2)
            return true;
        if (!pRoot1 || pRoot1->val != pRoot2->val)
            return false;
        return isSame(pRoot1->left, pRoot2->left) && isSame(pRoot1->right, pRoot2->right);
    }
};
```

1. 后来自己写的代码

    按照先序遍历的思路，很好理解：如果当前树`A`正好就是树`B`（其实不是完全相等），那么`B`就是`A`的子结构。否则的话，如果两个子树其中一个和`B`相等，那么也可以认为有子结构。最后根据题意把空树这种特殊情况加上就好了。

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        bool is_same(TreeNode *A, TreeNode *B)  // 这个函数并不是严格意义上的相等
        {
            if (!B) return true;  // 这里不能写成 if (!A && !B)，因为子树 B 可能并不在 A 的底部
            if ((!A && B) || (A && !B)) return false;
            if (A->val == B->val && is_same(A->left, B->left) && is_same(A->right, B->right))
                return true;
            return false;
        }

        bool is_subtree(TreeNode *A, TreeNode *B)
        {
            if (!B || !A) return false;
            if (is_same(A, B)) return true;
            if (is_subtree(A->left, B) || is_subtree(A->right, B)) return true;
            return false;
        }

        bool isSubStructure(TreeNode* A, TreeNode* B) {
            return is_subtree(A, B);
        }
    };
    ```

### 另一棵树的子树

给你两棵二叉树 root 和 subRoot 。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true ；否则，返回 false 。

二叉树 tree 的一棵子树包括 tree 的某个节点和这个节点的所有后代节点。tree 也可以看做它自身的一棵子树。

 
```
示例 1：


输入：root = [3,4,5,1,2], subRoot = [4,1,2]
输出：true
示例 2：


输入：root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
输出：false
```

代码：

1. dfs

    不断移动根节点，找到一个相同的子树

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        bool isSame(TreeNode *root1, TreeNode *root2)
        {
            if (!root1 && !root2) return true;
            if ((root1 && !root2) || (!root1 && root2)) return false;
            if (root1->val != root2->val) return false;
            return isSame(root1->left, root2->left) && isSame(root1->right, root2->right);
        }

        bool isSubtree(TreeNode* root, TreeNode* subRoot) {
            if (!root) return false;  // subRoot 不可能为空，而且也不改变，所以只要检测 root 就可以了
            return isSame(root, subRoot) || isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);  // isSame() 要写到前面，否则可能会漏掉 root = [1, 1], subRoot = [1] 这种情况
        }
    };
    ```

1. 将树序列化成先序遍历序列，同时引入`lNULL`和`rNULL`保证序列的唯一性，接着使用 kmp 来做高效匹配

    （有时间再看）

    ```c++
    class Solution {
    public:
        vector <int> sOrder, tOrder;
        int maxElement, lNull, rNull;

        void getMaxElement(TreeNode *o) {
            if (!o) {
                return;
            }
            maxElement = max(maxElement, o->val);
            getMaxElement(o->left);
            getMaxElement(o->right);
        }

        void getDfsOrder(TreeNode *o, vector <int> &tar) {
            if (!o) {
                return;
            }
            tar.push_back(o->val);
            if (o->left) {
                getDfsOrder(o->left, tar);
            } else {
                tar.push_back(lNull);
            }
            if (o->right) {
                getDfsOrder(o->right, tar);
            } else {
                tar.push_back(rNull);
            }
        }

        bool kmp() {
            int sLen = sOrder.size(), tLen = tOrder.size();
            vector <int> fail(tOrder.size(), -1);
            for (int i = 1, j = -1; i < tLen; ++i) {
                while (j != -1 && tOrder[i] != tOrder[j + 1]) {
                    j = fail[j];
                }
                if (tOrder[i] == tOrder[j + 1]) {
                    ++j;
                }
                fail[i] = j;
            }
            for (int i = 0, j = -1; i < sLen; ++i) {
                while (j != -1 && sOrder[i] != tOrder[j + 1]) {
                    j = fail[j];
                }
                if (sOrder[i] == tOrder[j + 1]) {
                    ++j;
                }
                if (j == tLen - 1) {
                    return true;
                }
            }
            return false;
        }

        bool isSubtree(TreeNode* s, TreeNode* t) {
            maxElement = INT_MIN;
            getMaxElement(s);
            getMaxElement(t);
            lNull = maxElement + 1;
            rNull = maxElement + 2;

            getDfsOrder(s, sOrder);
            getDfsOrder(t, tOrder);

            return kmp();
        }
    };
    ```

### 二叉树的镜像（翻转二叉树）

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1

 

示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]


代码：

1. 先序遍历

    这种思路认为对于每个根节点，只要交换两个子节点就可以了。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        void mirror(TreeNode* root) {
            if (!root) return;
            swap(root->left, root->right);
            mirror(root->left);
            mirror(root->right);
        }
    };
    ```

    由于交换两个子节点不会影响子树，也不会影响父节点，所以其实先序遍历，中序遍历，后序遍历都是可以的。下面这段代码是后序遍历的例子。

    ```cpp
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        void dfs(TreeNode *r)
        {
            if (!r) return;
            dfs(r->left);
            dfs(r->right);
            swap(r->left, r->right);
        }

        TreeNode* invertTree(TreeNode* root) {
            dfs(root);
            return root;
        }
    };
    ```

1. 自己写的：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        void bfs(TreeNode *r)
        {
            swap(r->left, r->right);
            if (r->left) bfs(r->left);
            if (r->right) bfs(r->right);
        }
        TreeNode* invertTree(TreeNode* root) {
            if (!root) return root;
            bfs(root);
            return root;
        }
    };
    ```

1. 自己写的递归。这种方法利用了函数的返回值。

    这种思路是后序遍历，它认为镜像就是交换两个子树。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        TreeNode* mirrorTree(TreeNode* root) {
            if (!root) return nullptr;
            TreeNode *left, *right;
            // 注意这里不能写成
            // root->left = mirrorTree(root->right);
            // root->right = mirrorTree(root->left);
            // 这样会导致循环递归
            left = mirrorTree(root->right);
            right = mirrorTree(root->left);
            root->left = left;
            root->right = right;
            return root;
        }
    };
    ```

### 对称的二叉树（对称二叉树）

分析：

1. 递归

    两棵子树互为镜像，当且仅当：

    1. 两个子树的根节点值相等

    1. 第一棵子树的左子树和第二棵子树的右子树互为镜像，且第一棵子树的右子树和第二棵子树的左子树互为镜像。

代码：

1. 递归

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        bool isSymmetric(TreeNode* root) {
            if (!root)  // 空树是对称的
                return true;
            if (dfs(root->left, root->right))  // 
                return true;
        }
        
        bool dfs(TreeNode *p, TreeNode *q)
        {
            if (!p || !q)  return !p && !q;
            return p->val == q->val && dfs(p->left, q->right) && dfs(p->right, q->left);
        }
    };
    ```

    自己在 leetcode 上写的版本，好理解：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool is_sym(TreeNode *r1, TreeNode *r2)
        {
            if (!r1 && !r2) return true;  // 左右都为空时，是对称的
            if (r1 && r2 && r1->val == r2->val && is_sym(r1->left, r2->right) && is_sym(r1->right, r2->left)) // 左右都不为空，且两个树的顶部节点相等，且树1的左节点和树2的右节点相等，树1的右节点和树2的左节点相等，那么树1和树2是对称的
                return true;
            return false;  // 左右有一个为空，另一个不为空；或左右节点都存在，但不满足对称条件时，都不对称
        }
        bool isSymmetric(TreeNode* root) {
            if (!root) return true;
            return is_sym(root->left, root->right);
        }
    };
    ```

    后来又写的版本：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        bool is_sym(TreeNode *p, TreeNode *q)
        {
            if (!p && !q) return true;
            if (p && !q) return false;
            if (!p && q) return false;
            if (p->val != q->val) return false;
            return is_sym(p->left, q->right) && is_sym(p->right, q->left);
        }

        bool isSymmetric(TreeNode* root) {
            if (!root) return true;
            return is_sym(root->left, root->right);
        }
    };
    ```

1. 迭代（栈）

    对根节点的左子树，使用中序遍历；对根节点的右子树，使用反中序遍历。若两个子树互为镜像，当且仅当同时遍历两棵子树时，对应节点的值相等。（还没看）


    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        bool isSymmetric(TreeNode* root) {
            if (!root) return true;
            stack<TreeNode*> left, right;
            TreeNode *lc = root->left;
            TreeNode *rc = root->right;
            while(lc || rc || left.size())
            {
                while (lc && rc)
                {
                    left.push(lc), right.push(rc);
                    lc = lc->left, rc = rc->right;
                }
                if (lc || rc) return false;
                lc = left.top(), rc = right.top();
                left.pop(), right.pop();
                if (lc->val != rc->val) return false;
                lc = lc->right, rc = rc->left;
            }
            return true;
        }

    };
    ```

1. 迭代（队列）

    队列中相邻的两个元素应该一模一样才对。若出现不一样，说明不对称。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool isSymmetric(TreeNode* root) {
            if (!root) return true;
            queue<TreeNode*> q;
            q.push(root->left);
            q.push(root->right);
            TreeNode *r1, *r2;
            while (!q.empty())
            {
                r1 = q.front(); q.pop();
                r2 = q.front(); q.pop();
                if (!r1 && !r2) continue;  // 俩节点都为空，则对称
                if (!r1 || !r2) return false;  // 前面排除了两节点都为空，这里只能是其中一个为空
                if (r1->val != r2->val) return false;  // 这里只剩下俩节点都不空的情况了
                q.push(r1->left);
                q.push(r2->right);
                q.push(r1->right);
                q.push(r2->left);
            }
            return true;
        }
    };
    ```

### 不分行从上往下打印二叉树

> 从上往下打印出二叉树的每个结点，同一层的结点按照从左> 到右的顺序打印。
> 
> 样例
> 
> ```
> 输入如下图所示二叉树[8, 12, 2, null, null, 6, > null, 4, null, null, null]
>     8
>    / \
>   12  2
>      /
>     6
>    /
>   4
> 
> 输出：[8, 12, 2, 6, 4]
> ```

代码：

层序遍历：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    queue<TreeNode*> q;
    vector<int> v;
    vector<int> printFromTopToBottom(TreeNode* root) {
        if (!root)
            return v;
        q.push(root);
        while (!q.empty())
        {
            v.push_back(q.front()->val);
            if (q.front()->left)
                q.push(q.front()->left);
            if (q.front()->right)
                q.push(q.front()->right);
            q.pop();
        }
        return v;
    }
};
```

### 分行从上往下打印二叉树

> 从上到下按层打印二叉树，同一层的结点按从左到右的顺序> 打印，每一层打印到一行。
> 
> 样例
> 
> ```
> 输入如下图所示二叉树[8, 12, 2, null, null, 6, null, 4, null, null, null]
>     8
>    / \
>   12  2
>      /
>     6
>    /
>   4
> 
> 输出：[[8], [12, 2], [6], [4]]
> ```

分析：

使用`nullptr`作为分行依据。

代码：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    
    vector<vector<int>> printFromTopToBottom(TreeNode* root) {
        vector<vector<int>> v;
        if (!root)
            return v;
        queue<TreeNode*> q;
        q.push(root);
        q.push(nullptr);
        v.push_back(vector<int>());
        while (!q.empty())
        {
            if (q.front() == nullptr)
            {
                if (q.size() == 1)
                    break;
                v.push_back(vector<int>());
                q.push(nullptr);
                q.pop();
            }
            else
            {
                v.back().push_back(q.front()->val);
                if (q.front()->left)
                    q.push(q.front()->left);
                if (q.front()->right)
                    q.push(q.front()->right);
                q.pop();
            }
        }
        return v;
    }
};
```

### 之字形打印二叉树（二叉树的锯齿形层序遍历）

分析：思路有很多。

代码：

1. 使用 deque，来回插入新节点 

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        vector<vector<int>> printFromTopToBottom(TreeNode* root) {
            vector<vector<int>> v;
            if (!root)
                return v;
            deque<TreeNode*> q;
            bool right_dir = false;
            v.push_back(vector<int>());
            q.push_back(root);
            q.push_back(nullptr);

            while (!q.empty())
            {
                if (right_dir)
                {
                    if (q.back() == nullptr)
                    {
                        if (q.size() == 1)
                            break;
                        v.push_back(vector<int>());
                        right_dir = !right_dir;
                        continue;
                    }
                    
                    v.back().push_back(q.back()->val);
                    if (q.back()->right)
                        q.push_front(q.back()->right);
                    if (q.back()->left)
                        q.push_front(q.back()->left);
                    q.pop_back();
                }
                else
                {
                    if (q.front() == nullptr)
                    {
                        if (q.size() == 1)
                            break;
                        v.push_back(vector<int>());
                        right_dir = !right_dir;
                        continue;
                    }
                
                    v.back().push_back(q.front()->val);
                    if (q.front()->left)
                        q.push_back(q.front()->left);
                    if (q.front()->right)
                        q.push_back(q.front()->right);
                    q.pop_front();
                }
            }
            return v;
        }
    };
    ```

1. 使用双栈

    这里想说的是，虽然栈提供了逆序的功能，但是并不能提供临时存储数据的功能。比如我把一个数压栈后，必须把这个数 pop 出来才能继续处理栈中的其它数据。如果既想用栈的逆序，又想存储，那么只能用多个栈了。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        vector<vector<int>> levelOrder(TreeNode* root) {
            vector<vector<int>> ans;
            if (!root) return ans;
            stack<TreeNode*> s1, s2;
            s1.push(root);
            int size;
            bool ltor = true;
            while (!s1.empty() || !s2.empty())
            {
                ans.push_back({});
                if (ltor)
                {
                    size = s1.size();
                    for (int i = 0; i < size; ++i)
                    {
                        root = s1.top();
                        s1.pop();
                        ans.back().push_back(root->val);
                        if (root->left) s2.push(root->left);
                        if (root->right) s2.push(root->right);
                    }
                }
                else
                {
                    size = s2.size();
                    for (int i = 0; i < size; ++i)
                    {
                        root = s2.top();
                        s2.pop();
                        ans.back().push_back(root->val);
                        if (root->right) s1.push(root->right);
                        if (root->left) s1.push(root->left);
                    }
                }
                ltor = !ltor;
            }
            return ans;
        }
    };
    ```

1. 先正常层序遍历，再按奇偶行判断是否使用逆序`reverse()`

### 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 
```
参考以下这颗二叉搜索树：

     5
    / \
   2   6
  / \
 1   3
示例 1：

输入: [1,6,3,2,5]
输出: false
示例 2：

输入: [1,3,2,6,5]
输出: true
```

代码：

1. 递归法

    ```c++
    class Solution {
    public: 
        vector<int> seq;

        bool verifySequenceOfBST(vector<int> sequence) {
            seq = sequence;
            return dfs(0, seq.size() - 1);
        }
        
        bool dfs(int l, int r) {
            if (l >= r)  // 左右子树都为空，返回 true 终止递归
                return true;
            int k = l;
            while (seq[k] < seq[r])  // 找到右子树的起始位置
                ++k;
            int m = k;
            while (seq[k] > seq[r])  // 在找右子树时已经保证了左子树的正确性，所以这里只需检测右子树的正确性就好
                ++k;
            if (k != r)
                return false;
            return dfs(l, m-1) && dfs(k, m-1);
        }
    };
    ```

    如果我们采用这种方式：根节点大于左子树的根节点，但是小于右子树的根节点；每个子树都做这样的递归判断。这样的方法是不行的。因为根节点的信息不能传递到子树的节点。这种现象和“验证二叉搜索树”是一样的。

1. 单调栈法

    ```py
    class Solution:
        def verifyPostorder(self, postorder: [int]) -> bool:
            stack, root = [], float("+inf")
            for i in range(len(postorder) - 1, -1, -1):
                if postorder[i] > root: return False
                while(stack and postorder[i] < stack[-1]):
                    root = stack.pop()
                stack.append(postorder[i])
            return True
    ```

### 二叉树中和为某一值的路径（路径总和 II）

输入一棵二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。

从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

样例

```
给出二叉树如下所示，并给出num=22。
      5
     / \
    4   6
   /   / \
  12  13  6
 /  \    / \
9    1  5   1

输出：[[5,4,12,1],[5,6,6,5]]
```

代码：

1. dfs

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        vector<vector<int>> res;
        vector<int> path;
        vector<vector<int>> findPath(TreeNode* root, int sum) {
            if (!root) return res;
            dfs(root, sum);
            return res;
        }
        
        void dfs(TreeNode *root, int sum)
        {
            if (!root) return;
            sum = sum - root->val;
            path.push_back(root->val);
            if (!root->left && !root->right && !sum)  // 到叶节点，且和满足要求时，才停止
                res.emplace_back(path);
            if (root->left)
                dfs(root->left, sum);
            if (root->right)
                dfs(root->right, sum);
            path.pop_back();
        }
    };
    ```

1. 后来又写的

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        vector<vector<int>> ans;
        vector<int> temp;
        void dfs(TreeNode *root, int target)
        {
            if (!root) return;
            if (!root->left && !root->right && target == root->val)
            {
                temp.push_back(root->val);
                ans.push_back(temp);
                temp.pop_back();
                return;
            }

            temp.push_back(root->val);
            dfs(root->left, target - root->val);
            dfs(root->right, target - root->val);
            temp.pop_back();     
        }

        vector<vector<int>> pathSum(TreeNode* root, int target) {
            dfs(root, target);
            return ans;
        }
    };
    ```

    从这道题可以看出，如果能保证`push_back()`和`pop_back()`是一对，两个都能执行到，那么就能保证回溯是没有问题的。

    但是下面这种写法是有问题的：

    ```c++
    void dfs(TreeNode *root, int target)
    {
        if (!root) return;
        temp.push_back(root->val);
        if (!root->left && !root->right && target == root->val)
        {
            ans.push_back(temp);
            return;
        }

        dfs(root->left, target - root->val);
        dfs(root->right, target - root->val);
        temp.pop_back();     
    }
    ```

    假如执行完`push_back()`后，进入了第二个`if`分支，那么就无法执行到下面的`pop_back()`了，这样就会出错。

1. 层序遍历

    可以用两个`queue`或者`pair`的形式来存储每个节点和它对应的目标和，用`unordered_map`存储每个节点对应的父节点。如果发现某个叶子节点的值正好等于目标和，那么可以一路往上找，直至根节点，找到一条路径。

1. 既然 dfs，bfs 都可以写，那么其实四种遍历的迭代写法 + 递归写法也都能做这道题。路径的记录可以是哈希表，也可以是临时数组。

    其实这道题是一道树的回溯搜索题。

### 路径总和

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。

```
示例 1：

输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
示例 2：


输入：root = [1,2,3], targetSum = 5
输出：false
```

代码：

1. 先序遍历，回溯写法

    ```cpp
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        bool ans;
        int targetSum;
        int path_sum;

        void dfs(TreeNode *root)
        {
            if (ans)
                return;
            path_sum += root->val;
            if (!root->left && !root->right)
            {
                if (path_sum == targetSum)
                    ans = true;
                path_sum -= root->val;  // 这里要回退清理
                return;
            }
            if (root->left) dfs(root->left);
            if (root->right) dfs(root->right);
            path_sum -= root->val;
        }

        bool hasPathSum(TreeNode* root, int targetSum) {
            if (!root) return false;
            ans = false;
            this->targetSum = targetSum;
            path_sum = 0;
            dfs(root);
            return ans;
        }
    };
    ```

    使用当前节点的数据 -> 遍历左节点 -> 遍历右节点，因此这种方法是标准的先序遍历。

    循环的模式遵循：改变状态 -> 判断状态 -> 进入下一层。

    为什么要在那里回退清理呢，因为我们为了判断一个节点是否为叶子节点，先序遍历比较好理解，如果一个节点没有左右节点，那么它就是叶子节点，我们就需要判断路径和是否为目标值。在假设我们用了先序遍历的前提下，是应该先判断状态，再改变状态呢，还是先改变状态，再判断状态？

    假如我们选择先判断状态，再改变状态，那么就要保证在进入当前节点时，当前节点的状态就已经准备就绪。因此代码可能会这样写：

    ```cpp
    if (root->left)
    {
        path_sum += root->left->val;
        dfs(root->left);
        path_sum -= root->left->val;

    }
    if (root->right)
    {
        path_sum += root->right->val;
        dfs(root->right);
        path_sum -= root->right->val;
    }
    ```

    如果我们选择先改变状态，再判断状态，那么在写遍历子节点时，代码就可以少一些，代价是我们需要在当前函数的**所有出口**处做清理工作。

    （其实给递归函数传递参数，就是“先改变状态，再判断状态”的特殊写法）

1. 递归

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool hasPathSum(TreeNode* root, int targetSum) {
            if (!root) return false;
            if (!root->left && !root->right && targetSum == root->val) return true;  // 注意，必须走到底才算一条路径，所以需要加上 !root->left && !root->right 这个条件
            return hasPathSum(root->left, targetSum - root->val) || 
                hasPathSum(root->right, targetSum - root->val);
        }
    };
    ```

    对于先序遍历来讲，先判断当前节点是否符合要求，再判断子树是否符合要求，根据需要来判断是否保存子树的返回值。对于后序遍历来讲，如果需要先判断子树的状态，再结合当前节点的状态来判断，那么就有可能需要保存子树的返回值。

    后来自已又写的：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool dfs(TreeNode *root, int target)
        {
            if (!root->left && !root->right && target == root->val) return true;
            if (root->left && dfs(root->left, target - root->val)) return true;
            if (root->right && dfs(root->right, target - root->val)) return true;
            return false;
        }

        bool hasPathSum(TreeNode* root, int targetSum) {
            if (!root) return false;
            return dfs(root, targetSum);
        }
    };
    ```

    其实我并不是很清楚，为什么上一种写法有`if (!root) return false;`这个语句。下面那种写法更复杂，但是它限制了不可能搜索到空节点，所以条理更清晰一些。（现在明白了，`if (!root) return false;`是因为当当前节点的左节点或右节点为空时，也会进行一次递归调用，此时需要退出递归。总的来说，要么使用`if`来避免进入递归，要么睚递归函数开头的地方写上出口条件）

    但是要注意，这样写是错的：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool ans;
        void dfs(TreeNode *root, int target)
        {
            if (ans) return;
            if (!root && target == 0)
            {
                ans = true;
                return;
            }
            if (!root && target != 0) return;

            dfs(root->left, target - root->val);
            dfs(root->right, target - root->val);
        }

        bool hasPathSum(TreeNode* root, int targetSum) {
            if (!root) return false;
            ans = false;
            dfs(root, targetSum);
            return ans;
        }
    };
    ```

    这种情况只考虑到了，如果到达某个空节点，发现目标和为零，那么就说明成功找到一条路径。但是实际上，空节点也分两种类型，一种的父节点是叶子节点，即父节点的两个子节点都为空；另一种的父节点只有一个子节点。

    这种遍历本质上还是一个先序遍历，只不过加上了对当前根节点类型的判断。

    然后其实这样写也是可以的：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool ans;
        void dfs(TreeNode *root, int target)
        {
            if (ans) return;
            if (!root) return;
            if (!root->left && !root->right && root->val == target) 
            {
                ans = true;
                return;
            }
            dfs(root->left, target - root->val);
            dfs(root->right, target - root->val);
        }

        bool hasPathSum(TreeNode* root, int targetSum) {
            ans = false;
            dfs(root, targetSum);
            return ans;
        }
    };
    ```

1. 广度优先迭代

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool hasPathSum(TreeNode* root, int targetSum) {
            if (!root) return false;
            queue<pair<TreeNode*, int>> q;
            q.push(make_pair(root, 0));
            TreeNode *r;
            int sum;
            while (!q.empty())
            {
                r = q.front().first;
                sum = q.front().second;
                q.pop();
                if (!r->left && !r->right && r->val + sum == targetSum)
                    return true;
                if (r->left) q.push(make_pair(r->left, r->val + sum));
                if (r->right) q.push(make_pair(r->right, r->val + sum));
            } 
            return false;
        }
    };
    ```

### 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。

要求不能创建任何新的结点，只能调整树中结点指针的指向。

注意：

需要返回双向链表最左侧的节点。

分析：中序遍历加一点点的修改。

代码：

1. dfs + 额外变量记录上一个节点

    为什么不能不使用额外的变量？假如让每个函数都有一个返回值，那么可以代替额外的变量吗？

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        TreeNode *pre = NULL;
        void dfs(TreeNode *root)
        {
            if (!root)
                return;
                
            if (root->left)
                dfs(root->left);
                
            root->left = pre;
            if (pre)
                pre->right = root;
            pre = root;
            
            if (root->right)
                dfs(root->right);
        }
        TreeNode* convert(TreeNode* root) {
            if (!root)
                return nullptr;
            dfs(root);
            while (root->left)
                root = root->left;
            return root;
        }
    };
    ```

1. 如果题目需要改成循环链表，那么还需要将首尾连起来：

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* left;
        Node* right;

        Node() {}

        Node(int _val) {
            val = _val;
            left = NULL;
            right = NULL;
        }

        Node(int _val, Node* _left, Node* _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    };
    */
    class Solution {
    public:
        Node *pre;
        void dfs(Node *root)
        {
            if (!root) return;
            if (root->left) dfs(root->left);
            root->left = pre;
            if (pre) pre->right = root;
            pre = root;
            if (root->right) dfs(root->right);
        }

        Node* treeToDoublyList(Node* root) {
            if (!root) return nullptr;
            pre = nullptr;
            dfs(root);
            Node *left = root, *right = root;
            while (left->left) left = left->left;
            left->left = pre;
            pre->right = left;
            return left;
        }
    };
    ```

1. 迭代写法

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* left;
        Node* right;

        Node() {}

        Node(int _val) {
            val = _val;
            left = NULL;
            right = NULL;
        }

        Node(int _val, Node* _left, Node* _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    };
    */
    class Solution {
    public:
        Node* treeToDoublyList(Node* root) {
            if (!root) return nullptr;
            stack<Node*> s;
            Node *r = root, *pre = nullptr;
            while (r || !s.empty())
            {
                while (r)
                {
                    s.push(r);
                    r = r->left;
                }
                r = s.top();
                s.pop();
                r->left = pre;
                if (pre) pre->right = r;
                pre = r;
                r = r->right;
            }

            Node *left = root;
            while (left->left) left = left->left;
            left->left = pre;
            pre->right = left;
            return left;
        }
    };
    ```

### 序列化二叉树

> 请实现两个函数，分别用来序列化和反序列化二叉树。
> 
> 您需要确保二叉树可以序列化为字符串，并且可以将此字符串反序列化为原始树结构。
> 
> 
> ```
> 样例
> 你可以序列化如下的二叉树
>     8
>    / \
>   12  2
>      / \
>     6   4
> 
> 为："[8, 12, 2, null, null, 6, 4, null, null, null, null]"
> ```

分析：中序遍历似乎是不可行的，因为无法定位根节点。其余三种遍历均可。

1. 前序遍历

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:

        string str;
        int l, r;

        // Encodes a tree to a single string.
        string serialize(TreeNode* root) {
            if (!root)
                return string("null");
            dfs_serialize(root);
            return str;
        }
        
        void dfs_serialize(TreeNode *root)
        {
            if (!root)
            {
                str.append("null,");
            }
            else
            {
                str.append(to_string(root->val));
                str.push_back(',');
                dfs_serialize(root->left);
                dfs_serialize(root->right);
            }
        }
        
        void dfs_deserialize(string &data, TreeNode *root)
        {
            root->val = stoi(data.substr(l, r-l+1));
            l = r + 2;
            
            if (l == data.size())
                return;
            
            if (data[l] != 'n')
            {
                int i = l;
                do ++i; while (data[i] != ',');
                r = i - 1;
                root->left = new TreeNode(0);
                dfs_deserialize(data, root->left);
                l = r + 2;
            }
            else
            {
                int i = l;
                do ++i; while (data[i] != ',');
                l = i + 1;
                
            }

            if (data[l] != 'n')
            {
                int i = l;
                do ++i; while (data[i] != ',');
                r = i - 1;
                root->right = new TreeNode(0);
                dfs_deserialize(data, root->right);
            }
            else
            {
                int i = l;
                do ++i; while (data[i] != ',');
                r = i - 1;
            }
            
        }

        // Decodes your encoded data to tree.
        TreeNode* deserialize(string data) {
            if (data[0] == 'n')
                return nullptr;
            
            TreeNode *root = new TreeNode(0);
            int i = 0;
            do ++i; while (data[i] != ',');
            l = 0;
            r = i-1;
            
            dfs_deserialize(data, root);
            return root;
        }
    };
    ```

    想了想，似乎没啥需要注意的地方。写就完事了。

1. 后序遍历

1. 层序遍历

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Codec {
    public:

        // Encodes a tree to a single string.
        string serialize(TreeNode* root) {
            string ans;
            if (!root) return ans;
            queue<TreeNode*> q;
            q.push(root);
            while (!q.empty())
            {
                root = q.front();
                q.pop();
                if (root) ans.append(to_string(root->val));
                else ans.append("null");
                ans.push_back(',');
                if (root)
                {
                    q.push(root->left);
                    q.push(root->right);
                }
            }
            return ans;
        }

        // Decodes your encoded data to tree.
        TreeNode* deserialize(string data) {
            if (data.empty()) return nullptr;
            int left = 0, right = 0;
            do ++right; while (data[right] != ',');
            int num = stoi(data.substr(left, right - left));
            TreeNode *r = new TreeNode(num);
            queue<TreeNode*> q;
            q.push(r);
            TreeNode *root;
            while (!q.empty() && right < data.size())
            {
                root = q.front();
                q.pop();
                
                left = right + 1;
                do ++right; while (data[right] != ',');
                if (data[left] == 'n') root->left = nullptr;
                else 
                {
                    root->left = new TreeNode(stoi(data.substr(left, right - left)));
                    q.push(root->left);
                }

                left = right + 1;
                do ++right; while (data[right] != ',');
                if (data[left] == 'n') root->right = nullptr;
                else 
                {
                    root->right = new TreeNode(stoi(data.substr(left, right - left)));
                    q.push(root->right);
                }
            }
            return r;
        }
    };

    // Your Codec object will be instantiated and called as such:
    // Codec ser, deser;
    // TreeNode* ans = deser.deserialize(ser.serialize(root));
    ```

    后来又写的

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Codec {
    public:

        // Encodes a tree to a single string.
        string serialize(TreeNode* root) {
            string ans;
            if (!root) return ans;
            queue<TreeNode*> q;
            q.push(root);
            int size;
            while (!q.empty())
            {
                size = q.size();
                for (int i = 0; i < size; ++i)
                {
                    root = q.front();
                    q.pop();
                    ans.append(root ? to_string(root->val) : "null");
                    ans.push_back(',');
                    if (root) q.push(root->left), q.push(root->right);
                }    
            }
            return ans;
        }

        // Decodes your encoded data to tree.
        TreeNode* deserialize(string data) {
            if (data.empty()) return nullptr;
            TreeNode *root, *r;
            int start = 0, end, p = 0;
            while (data[p] != ',') ++p;
            root = new TreeNode(stoi(data.substr(start, p-start)));
            queue<TreeNode*> q;
            q.push(root);
            string temp;
            while (!q.empty())
            {
                r = q.front();
                q.pop();

                start = p + 1;
                do ++p; while (data[p] != ',');
                end = p - 1;
                temp = data.substr(start, end-start+1);
                if (temp != "null") r->left = new TreeNode(stoi(temp)), q.push(r->left);
                
                start = p + 1;
                do ++p; while (data[p] != ',');
                end = p - 1;
                temp = data.substr(start, end-start+1);
                if (temp != "null") r->right = new TreeNode(stoi(temp)), q.push(r->right);
            }
            return root;
        }
    };

    // Your Codec object will be instantiated and called as such:
    // Codec codec;
    // codec.deserialize(codec.serialize(root));
    ```

1. 括号表示编码 + 递归下降解码（没看

    ```c++
    class Codec {
    public:
        string serialize(TreeNode* root) {
            if (!root) {
                return "X";
            }
            auto left = "(" + serialize(root->left) + ")";
            auto right = "(" + serialize(root->right) + ")";
            return left + to_string(root->val) + right;
        }

        inline TreeNode* parseSubtree(const string &data, int &ptr) {
            ++ptr; // 跳过左括号
            auto subtree = parse(data, ptr);
            ++ptr; // 跳过右括号
            return subtree;
        }

        inline int parseInt(const string &data, int &ptr) {
            int x = 0, sgn = 1;
            if (!isdigit(data[ptr])) {
                sgn = -1;
                ++ptr;
            }
            while (isdigit(data[ptr])) {
                x = x * 10 + data[ptr++] - '0';
            }
            return x * sgn;
        }

        TreeNode* parse(const string &data, int &ptr) {
            if (data[ptr] == 'X') {
                ++ptr;
                return nullptr;
            }
            auto cur = new TreeNode(0);
            cur->left = parseSubtree(data, ptr);
            cur->val = parseInt(data, ptr);
            cur->right = parseSubtree(data, ptr);
            return cur;
        }

        TreeNode* deserialize(string data) {
            int ptr = 0;
            return parse(data, ptr);
        }
    };
    ```

### 二叉树的堂兄弟节点

在二叉树中，根节点位于深度 0 处，每个深度为 k 的节点的子节点位于深度 k+1 处。

如果二叉树的两个节点深度相同，但 父节点不同 ，则它们是一对堂兄弟节点。

我们给出了具有唯一值的二叉树的根节点 root ，以及树中两个不同节点的值 x 和 y 。

只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true 。否则，返回 false。

代码：

1. 深度优先

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int dx, dy;
        int rx, ry;
        bool isCousins(TreeNode* root, int x, int y) {
            dx = -1;
            dy = -1;
            rx = -1;
            ry = -1;
            dfs(root, root, 0, x, y);
            if (dx == dy && rx != ry)
                return true;
            return false;
        }

        void dfs(TreeNode *root, TreeNode *p, int depth, int x, int y)
        {
            if (dx != -1 && dy != -1 && rx != -1 && ry != -1)
                return;
            if (root->val == x)
            {
                dx = depth;
                rx = p->val;
            }
            else if (root->val == y)
            {
                dy = depth;
                ry = p->val;
            }

            if (root->left)
                dfs(root->left, root, depth+1, x, y);
            if (root->right)
                dfs(root->right, root, depth+1, x, y);
            
        }
    };
    ```

1. 广度优先

### 二叉搜索树的第k个结点（二叉搜索树中第K小的元素）

给定一棵二叉搜索树，请找出其中的第 k 小的结点。

你可以假设树和 k 都存在，并且 1≤k≤ 树的总结点数。

```
样例
输入：root = [2, 1, 3, null, null, null, null] ，k = 3

    2
   / \
  1   3

输出：3
```


代码：

1. 中序遍历

    中序遍历的结果恰好就是二叉搜索树中节点从小到大排列的结果。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int ans;
        int k;
        
        void dfs(TreeNode *root)
        { 
            if (ans != -1) return;
            if (root->left) dfs(root->left);
            if (--k == 0) ans = root->val;
            if (root->right) dfs(root->right);
        }

        int kthSmallest(TreeNode* root, int k) {
            ans = -1;
            this->k = k;
            dfs(root);
            return ans;
        }
    };
    ```

    中序遍历也有迭代和递归两种写法，这里只写了递归，有空再写写迭代。

    后来又写的，差不多其实：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int k, cur;
        bool found;
        int ans;
        void dfs(TreeNode *root)
        {
            if (!root) return;
            if (found) return;
            dfs(root->left);
            if (++cur == k) ans = root->val, found = true;
            dfs(root->right);
        }

        int kthSmallest(TreeNode* root, int k) {
            this->k = k;
            cur = 0;
            found = false;
            dfs(root);
            return ans;
        }
    };
    ```

1. 统计左子树和右子树的节点数，用类似二分搜索的方式查找

    ```c++
    class MyBst {
    public:
        MyBst(TreeNode *root) {
            this->root = root;
            countNodeNum(root);
        }

        // 返回二叉搜索树中第k小的元素
        int kthSmallest(int k) {
            TreeNode *node = root;
            while (node != nullptr) {
                int left = getNodeNum(node->left);
                if (left < k - 1) {
                    node = node->right;
                    k -= left + 1;
                } else if (left == k - 1) {
                    break;
                } else {
                    node = node->left;
                }
            }
            return node->val;
        }

    private:
        TreeNode *root;
        unordered_map<TreeNode *, int> nodeNum;

        // 统计以node为根结点的子树的结点数
        int countNodeNum(TreeNode * node) {
            if (node == nullptr) {
                return 0;
            }
            nodeNum[node] = 1 + countNodeNum(node->left) + countNodeNum(node->right);
            return nodeNum[node];
        }

        // 获取以node为根结点的子树的结点数
        int getNodeNum(TreeNode * node) {
            if (node != nullptr && nodeNum.count(node)) {
                return nodeNum[node];
            }else{
                return 0;
            }
        }
    };

    class Solution {
    public:
        int kthSmallest(TreeNode* root, int k) {
            MyBst bst(root);
            return bst.kthSmallest(k);
        }
    };
    ```

1. avl 树

    可以应对频繁的插入和删除操作。

    ```c++
    // 平衡二叉搜索树结点
    struct Node {
        int val;
        Node * parent;
        Node * left;
        Node * right;
        int size;
        int height;

        Node(int val) {
            this->val = val;
            this->parent = nullptr;
            this->left = nullptr;
            this->right = nullptr;
            this->height = 0; // 结点高度：以node为根节点的子树的高度（高度定义：叶结点的高度是0）
            this->size = 1; // 结点元素数：以node为根节点的子树的节点总数
        }

        Node(int val, Node * parent) {
            this->val = val;
            this->parent = parent;
            this->left = nullptr;
            this->right = nullptr;
            this->height = 0; // 结点高度：以node为根节点的子树的高度（高度定义：叶结点的高度是0）
            this->size = 1; // 结点元素数：以node为根节点的子树的节点总数
        }

        Node(int val, Node * parent, Node * left, Node * right) {
            this->val = val;
            this->parent = parent;
            this->left = left;
            this->right = right;
            this->height = 0; // 结点高度：以node为根节点的子树的高度（高度定义：叶结点的高度是0）
            this->size = 1; // 结点元素数：以node为根节点的子树的节点总数
        }
    };


    // 平衡二叉搜索树（AVL树）：允许重复值
    class AVL {
    public:
        AVL(vector<int> & vals) {
            if (!vals.empty()) {
                root = build(vals, 0, vals.size() - 1, nullptr);
            }
        }

        // 根据vals[l:r]构造平衡二叉搜索树 -> 返回根结点
        Node * build(vector<int> & vals, int l, int r, Node * parent) {
            int m = (l + r) >> 1;
            Node * node = new Node(vals[m], parent);
            if (l <= m - 1) {
                node->left = build(vals, l, m - 1, node);
            }
            if (m + 1 <= r) {
                node->right = build(vals, m + 1, r, node);
            }
            recompute(node);
            return node;
        }

        // 返回二叉搜索树中第k小的元素
        int kthSmallest(int k) {
            Node * node = root;
            while (node != nullptr) {
                int left = getSize(node->left);
                if (left < k - 1) {
                    node = node->right;
                    k -= left + 1;
                } else if (left == k - 1) {
                    break;
                } else {
                    node = node->left;
                }
            }
            return node->val;
        }

        void insert(int v) {
            if (root == nullptr) {
                root = new Node(v);
            } else {
                // 计算新结点的添加位置
                Node * node = subtreeSearch(root, v);
                bool isAddLeft = v <= node->val; // 是否将新结点添加到node的左子结点
                if (node->val == v) { // 如果值为v的结点已存在
                    if (node->left != nullptr) { // 值为v的结点存在左子结点，则添加到其左子树的最右侧
                        node = subtreeLast(node->left);
                        isAddLeft = false;
                    } else { // 值为v的结点不存在左子结点，则添加到其左子结点
                        isAddLeft = true;
                    }
                }

                // 添加新结点
                Node * leaf = new Node(v, node);
                if (isAddLeft) {
                    node->left = leaf;
                } else {
                    node->right = leaf;
                }

                rebalance(leaf);
            }
        }

        // 删除值为v的结点 -> 返回是否成功删除结点
        bool Delete(int v) {
            if (root == nullptr) {
                return false;
            }

            Node * node = subtreeSearch(root, v);
            if (node->val != v) { // 没有找到需要删除的结点
                return false;
            }

            // 处理当前结点既有左子树也有右子树的情况
            // 若左子树比右子树高度低，则将当前结点替换为右子树最左侧的结点，并移除右子树最左侧的结点
            // 若右子树比左子树高度低，则将当前结点替换为左子树最右侧的结点，并移除左子树最右侧的结点
            if (node->left != nullptr && node->right != nullptr) {
                Node * replacement = nullptr;
                if (node->left->height <= node->right->height) {
                    replacement = subtreeFirst(node->right);
                } else {
                    replacement = subtreeLast(node->left);
                }
                node->val = replacement->val;
                node = replacement;
            }

            Node * parent = node->parent;
            Delete(node);
            rebalance(parent);
            return true;
        }

    private:
        Node * root;

        // 删除结点p并用它的子结点代替它，结点p至多只能有1个子结点
        void Delete(Node * node) {
            if (node->left != nullptr && node->right != nullptr) {
                return;
                // throw new Exception("Node has two children");
            }
            Node * child = node->left != nullptr ? node->left : node->right;
            if (child != nullptr) {
                child->parent = node->parent;
            }
            if (node == root) {
                root = child;
            } else {
                Node * parent = node->parent;
                if (node == parent->left) {
                    parent->left = child;
                } else {
                    parent->right = child;
                }
            }
            node->parent = node;
        }

        // 在以node为根结点的子树中搜索值为v的结点，如果没有值为v的结点，则返回值为v的结点应该在的位置的父结点
        Node * subtreeSearch(Node * node, int v) {
            if (node->val < v && node->right != nullptr) {
                return subtreeSearch(node->right, v);
            } else if (node->val > v && node->left != nullptr) {
                return subtreeSearch(node->left, v);
            } else {
                return node;
            }
        }

        // 重新计算node结点的高度和元素数
        void recompute(Node * node) {
            node->height = 1 + max(getHeight(node->left), getHeight(node->right));
            node->size = 1 + getSize(node->left) + getSize(node->right);
        }

        // 从node结点开始（含node结点）逐个向上重新平衡二叉树，并更新结点高度和元素数
        void rebalance(Node * node) {
            while (node != nullptr) {
                int oldHeight = node->height, oldSize = node->size;
                if (!isBalanced(node)) {
                    node = restructure(tallGrandchild(node));
                    recompute(node->left);
                    recompute(node->right);
                }
                recompute(node);
                if (node->height == oldHeight && node->size == oldSize) {
                    node = nullptr; // 如果结点高度和元素数都没有变化则不需要再继续向上调整
                } else {
                    node = node->parent;
                }
            }
        }

        // 判断node结点是否平衡
        bool isBalanced(Node * node) {
            return abs(getHeight(node->left) - getHeight(node->right)) <= 1;
        }

        // 获取node结点更高的子树
        Node * tallChild(Node * node) {
            if (getHeight(node->left) > getHeight(node->right)) {
                return node->left;
            } else {
                return node->right;
            }
        }

        // 获取node结点更高的子树中的更高的子树
        Node * tallGrandchild(Node * node) {
            Node * child = tallChild(node);
            return tallChild(child);
        }

        // 重新连接父结点和子结点（子结点允许为空）
        static void relink(Node * parent, Node * child, bool isLeft) {
            if (isLeft) {
                parent->left = child;
            } else {
                parent->right = child;
            }
            if (child != nullptr) {
                child->parent = parent;
            }
        }

        // 旋转操作
        void rotate(Node * node) {
            Node * parent = node->parent;
            Node * grandparent = parent->parent;
            if (grandparent == nullptr) {
                root = node;
                node->parent = nullptr;
            } else {
                relink(grandparent, node, parent == grandparent->left);
            }

            if (node == parent->left) {
                relink(parent, node->right, true);
                relink(node, parent, false);
            } else {
                relink(parent, node->left, false);
                relink(node, parent, true);
            }
        }

        // trinode操作
        Node * restructure(Node * node) {
            Node * parent = node->parent;
            Node * grandparent = parent->parent;

            if ((node == parent->right) == (parent == grandparent->right)) { // 处理需要一次旋转的情况
                rotate(parent);
                return parent;
            } else { // 处理需要两次旋转的情况：第1次旋转后即成为需要一次旋转的情况
                rotate(node);
                rotate(node);
                return node;
            }
        }

        // 返回以node为根结点的子树的第1个元素
        static Node * subtreeFirst(Node * node) {
            while (node->left != nullptr) {
                node = node->left;
            }
            return node;
        }

        // 返回以node为根结点的子树的最后1个元素
        static Node * subtreeLast(Node * node) {
            while (node->right != nullptr) {
                node = node->right;
            }
            return node;
        }

        // 获取以node为根结点的子树的高度
        static int getHeight(Node * node) {
            return node != nullptr ? node->height : 0;
        }

        // 获取以node为根结点的子树的结点数
        static int getSize(Node * node) {
            return node != nullptr ? node->size : 0;
        }
    };

    class Solution {
    public:
        int kthSmallest(TreeNode * root, int k) {
            // 中序遍历生成数值列表
            vector<int> inorderList;
            inorder(root, inorderList);
            // 构造平衡二叉搜索树
            AVL avl(inorderList);

            // 模拟1000次插入和删除操作
            vector<int> randomNums(1000);
            std::random_device rd;
            for (int i = 0; i < 1000; ++i) {
                randomNums[i] = rd()%(10001);
                avl.insert(randomNums[i]);
            }
            shuffle(randomNums); // 列表乱序
            for (int i = 0; i < 1000; ++i) {
                avl.Delete(randomNums[i]);
            }

            return avl.kthSmallest(k);
        }

    private:
        void inorder(TreeNode * node, vector<int> & inorderList) {
            if (node->left != nullptr) {
                inorder(node->left, inorderList);
            }
            inorderList.push_back(node->val);
            if (node->right != nullptr) {
                inorder(node->right, inorderList);
            }
        }

        void shuffle(vector<int> & arr) {
            std::random_device rd;
            int length = arr.size();
            for (int i = 0; i < length; i++) {
                int randIndex = rd()%length;
                swap(arr[i],arr[randIndex]);
            }
        }
    };
    ```

### 二叉搜索树的第k大节点

给定一棵二叉搜索树，请找出其中第k大的节点。

 
```
示例 1:

输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
示例 2:

输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

1. 中序遍历

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        int ans;
        int k;

        void dfs(TreeNode *root)
        {
            if (k == 0) return;
            if (!root) return;
            dfs(root->right);
            if (--k == 0) ans = root->val;
            dfs(root->left);
        }

        int kthLargest(TreeNode* root, int k) {
            this->k = k;
            dfs(root);
            return ans;
        }
    };
    ```

    迭代写法：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        int kthLargest(TreeNode* root, int k) {
            stack<TreeNode*> s;
            while (root || !s.empty())
            {
                while (root)
                {
                    s.push(root);
                    root = root->right;
                }
                root = s.top();
                s.pop();
                if (--k == 0) return root->val;
                root = root->left;
            }
            return -1;
        }
    };
    ```

### 二叉树的深度（二叉树的最大深度）

输入一棵二叉树的根结点，求该树的深度。

从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```
样例
输入：二叉树[8, 12, 2, null, null, 6, 4, null, null, null, null]如下图所示：
    8
   / \
  12  2
     / \
    6   4

输出：3
```

代码：

1. dfs

    `dfs`函数使用一个参数：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        int res;
        int tmp;
        
        void dfs(TreeNode *root)
        {
            res = max(++tmp, res);
            if (root->left) dfs(root->left);
            if (root->right) dfs(root->right);
            --tmp;
        }
        
        int treeDepth(TreeNode* root) {
            if (!root)
                return 0;
            res = 0;
            tmp = 0;
            dfs(root);
            return res;
        }
    };
    ```

    `dfs`函数使用两个参数：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        int ans;
        void dfs(TreeNode* root, int d) {
            ans = max(ans, d);
            if (root->left) dfs(root->left, d+1);
            if (root->right) dfs(root->right, d+1);
        }

        int treeDepth(TreeNode* root) {
            if (root == NULL) return 0;
            ans = 0;
            dfs(root, 1);
            return ans;
        }
    };
    ```

    后来又写的：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    * };
    */
    class Solution {
    public:
        int dfs(TreeNode *root)
        {
            if (!root) return 0;
            int left = dfs(root->left);
            int right = dfs(root->right);
            return max(left, right) + 1;
        }

        int maxDepth(TreeNode* root) {
            return dfs(root);
        }
    };
    ```

    后来又写的：

    ```cpp
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int ans;
        int depth;

        void dfs(TreeNode *r)
        {
            if (!r->left && !r->right)  // 判断结束状态
            {
                ++depth;  // 因为代码的逻辑是先改变状态，再判断状态，所以当结束的时候，需要额外地改变状态
                ans = max(ans, depth);  
                --depth;  // 用回溯的写法
                return;
            }
            ++depth;  // 先改变状态
            ans = max(ans, depth);  // 再判断状态
            if (r->left) dfs(r->left);
            if (r->right) dfs(r->right);
            --depth;  // 回溯写法
        }

        int maxDepth(TreeNode* root) {
            if (!root) return 0;
            ans = 0;
            depth = 0;
            dfs(root);
            return ans;
        }
    };
    ```

    现在再看以前的写法，

    ```cpp
    if (root->left) dfs(root->left, d+1);
    if (root->right) dfs(root->right, d+1);
    ```

    这两行隐含地包含了结束条件：如果一个节点既没有左节点，又没有右节点，那么它一定是叶子节点，因此不需要额外地判断递归的停止条件了。我觉得这种写法不是很好，就好像 C 语言中的隐式类型转换一样。

1. 后序遍历的迭代写法

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        int maxDepth(TreeNode* root) {
            stack<TreeNode*> s;
            TreeNode *prev = nullptr;
            int ans = 0;
            while (root || !s.empty())
            {
                while (root)
                {
                    s.push(root);
                    root = root->left;
                }
                ans = max(ans, (int)s.size());
                
                root = s.top();
                s.pop();

                if (root->right && root->right != prev)
                {
                    s.push(root);
                    root = root->right;
                }
                else
                {
                    prev = root;
                    root = nullptr;
                }
                ans = max(ans, (int)s.size());
            }
            return ans;
        }
    };
    ```

1. 层序遍历

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int maxDepth(TreeNode* root) {
            if (!root) return 0;
            queue<TreeNode*> q;
            q.push(root);
            int num = 1;
            TreeNode *r;
            int res = 0;
            while (!q.empty())
            {
                ++res;
                num = q.size();
                for (int i = 0; i < num; ++i)
                {
                    r = q.front();
                    q.pop();
                    if (r->left) q.push(r->left);
                    if (r->right) q.push(r->right);
                }
            }
            return res;
        }
    };
    ```

### 平衡二叉树

输入一棵二叉树的根结点，判断该树是不是平衡二叉树。

如果某二叉树中任意结点的左右子树的深度相差不超过 1，那么它就是一棵平衡二叉树。

注意：

规定空树也是一棵平衡二叉树。

```
样例
输入：二叉树[5,7,11,null,null,12,9,null,null,null,null]如下所示，
    5
   / \
  7  11
    /  \
   12   9

输出：true
```

代码：


代码：

1. 深度优先遍历。

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        bool res;
        int dfs(TreeNode *root)
        {
            int dleft = root->left ? 1 + dfs(root->left) : 0;
            int dright = root->right ? 1 + dfs(root->right) : 0;
            
            if (dleft - dright > 1 || dright - dleft > 1)
                res = false;
                
            return max(dleft, dright);
        }
        
        bool isBalanced(TreeNode* root) {
            if (!root)
                return true;
            res = true;
            dfs(root);
            return res;
        }
    };
    ```

1. 后来自己又写的

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        bool ans;

        int dfs(TreeNode *root)
        {
            if (!ans) return 0;  // 随便返回一个什么就行，用不到这个返回值
            if (!root) return 0;
            int depth_left = dfs(root->left);
            int depth_right = dfs(root->right);
            if (abs(depth_left - depth_right) > 1)
                ans = false; 
            return max(depth_left, depth_right) + 1;
        }

        bool isBalanced(TreeNode* root) {
            ans = true;
            dfs(root);
            return ans;
        }
    };
    ```

    自己后来写的加入了空节点的处理，所以不需要判断左节点或右节点是否存在。代码更简洁了。

    另外这道题是很明显的后序遍历，因为需要比较左右两个子树的深度，才能得出当前节点是否平衡。

### 树中两个结点的最低公共祖先（二叉树的最近公共祖先）

给出一个二叉树，输入两个树节点，求它们的最低公共祖先。

一个树节点的祖先节点包括它本身。

注意：

1. 输入的二叉树不为空；
1. 输入的两个节点一定不为空，且是二叉树中的节点；

样例

```
二叉树[8, 12, 2, null, null, 6, 4, null, null, null, null]如下图所示：
    8
   / \
  12  2
     / \
    6   4

1. 如果输入的树节点为2和12，则输出的最低公共祖先为树节点8。

2. 如果输入的树节点为2和6，则输出的最低公共祖先为树节点2。
```

分析：

如果左右子树中同时有`p`或`q`，那么说明`root`就是最低公共祖先。如果只有左子树有，那么说明公共祖先在左子树中；右子树同理。

其实这就是一个后序遍历的模型，子节点把信息上传给父节点。若某个父节点满足条件则直接返回。

代码：

1. 后序遍历

    为什么是后序遍历？因为一个父节点是否为祖先节点，需要收集子节点的信息。如果两个子节点包含有`p`或`q`，那么父节点就是祖先；如果两个子节点中没有`p`和`q`，但是左右两棵子树中含有`p`或`q`，即若两个子节点中含有`p`或`q`的祖先，那么父节点同样也是祖先。

    有关`nullptr`的处理：如果不想判断`root`本身是否为`nullptr`，那么就需要用`if (root->left)`，`if (root->right)`来保证不要进入有`nullptr`的分支。否则的话需要判断`root`自身是否为`nullptr`。其实判断`root`是否为`nullptr`是个比较好的选择，因为有时候题目中会给出空树的情况。

    有关返回值的存储：因为这里是后序遍历，所以处理完了两个子树的情况才能处理本节点。

    为了写递归，我们定义`dfs()`返回的是某个子树有关`p`或`q`的祖先（并不是`p`和`q`的公共祖先）。这个有点奇怪了，按道理外层函数和内层的递归函数的用处应该一样才对，但是外层函数的作用是返回`p`和`q`的**公共祖先**，而内层函数返回的是`p`或`q`的**祖先**。为什么会这样呢？

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        TreeNode* dfs(TreeNode *root, TreeNode *p, TreeNode *q)
        {
            if (!root) return nullptr;  // 处理到最下面的叶子节点的下一层
            if (root == p || root == q) return root;  // 本节点就是 p 或 q，那么本节点就是祖先
            
            TreeNode *left = dfs(root->left, p, q);  // 后序遍历需要先存储两棵子树的结果，再处理本节点
            TreeNode *right = dfs(root->right, p, q);
            
            if (left && right) return root;  // 如果左右子树各含有 p, q 中的一个，那么当前节点就是公共祖先
            if (!left) return right;  // 如果两个节点在右子树中，那么右子树返回的节点就是公共祖先
            else return left;
        }
        
        TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
            return dfs(root, p, q);
        }
    };
    ```

如果把条件从“二叉树”改成“二叉搜索树”，代码应该这样：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return root;
        if (root->val > p->val && root->val > q->val)  // 若 p，q 都小于 root，那么找左边
            return lowestCommonAncestor(root->left, p, q);
        else if (root->val < p->val && root->val < q->val)  // 都大于 root，那么找右边
            return lowestCommonAncestor(root->right, p, q);
        else  // 一个小于，一个大于，说明这就是公共祖先
            return root;
    }
};
```

迭代版本：

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 迭代
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root
        return None
```

如果还需要记录路径呢？有时间看看官方题解。

```c++
class Solution {
public:
    vector<TreeNode*> getPath(TreeNode* root, TreeNode* target) {
        vector<TreeNode*> path;
        TreeNode* node = root;
        while (node != target) {
            path.push_back(node);
            if (target->val < node->val) {
                node = node->left;
            }
            else {
                node = node->right;
            }
        }
        path.push_back(node);
        return path;
    }

    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        vector<TreeNode*> path_p = getPath(root, p);
        vector<TreeNode*> path_q = getPath(root, q);
        TreeNode* ancestor;
        for (int i = 0; i < path_p.size() && i < path_q.size(); ++i) {
            if (path_p[i] == path_q[i]) {
                ancestor = path_p[i];
            }
            else {
                break;
            }
        }
        return ancestor;
    }
};
```

存储父节点的版本（没看）：

```c++
class Solution {
public:
    unordered_map<int, TreeNode*> fa;
    unordered_map<int, bool> vis;
    void dfs(TreeNode* root){
        if (root->left != nullptr) {
            fa[root->left->val] = root;
            dfs(root->left);
        }
        if (root->right != nullptr) {
            fa[root->right->val] = root;
            dfs(root->right);
        }
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        fa[root->val] = nullptr;
        dfs(root);
        while (p != nullptr) {
            vis[p->val] = true;
            p = fa[p->val];
        }
        while (q != nullptr) {
            if (vis[q->val]) return q;
            q = fa[q->val];
        }
        return nullptr;
    }
};
```

### 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

```
示例 1:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
示例 2:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

代码：

1. 递归

    ```cpp
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */

    class Solution {
    public:
        TreeNode *p, *q;
        TreeNode *ans;
        void dfs(TreeNode *r)
        {
            if (ans) return;
            if (p->val <= r->val && q->val >= r->val)  // 只有最近公共祖先唯一满足这个条件
            {
                ans = r;
                return;
            }
            if (p->val < r->val && q->val < r->val)
                dfs(r->left);
            else if (p->val > r->val && q->val > r->val)
                dfs(r->right);
        }

        TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
            if (p->val > q->val) swap(p, q);  // 保证 p 小 q 大
            ans = nullptr;
            this->p = p;
            this->q = q;
            dfs(root);
            return ans;
        }
    };
    ```

1. 记录路径，两次遍历

    （这道题显然是直接抄了答案，有空再看看）

    ```c++
    class Solution {
    public:
        vector<TreeNode*> getPath(TreeNode* root, TreeNode* target) {
            vector<TreeNode*> path;
            TreeNode* node = root;
            while (node != target) {
                path.push_back(node);
                if (target->val < node->val) {
                    node = node->left;
                }
                else {
                    node = node->right;
                }
            }
            path.push_back(node);
            return path;
        }

        TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
            vector<TreeNode*> path_p = getPath(root, p);
            vector<TreeNode*> path_q = getPath(root, q);
            TreeNode* ancestor;
            for (int i = 0; i < path_p.size() && i < path_q.size(); ++i) {
                if (path_p[i] == path_q[i]) {
                    ancestor = path_p[i];
                }
                else {
                    break;
                }
            }
            return ancestor;
        }
    };
    ```

1. 不记录路径，一次迭代遍历

    ```c++
    class Solution {
    public:
        TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
            TreeNode* ancestor = root;
            while (true) {
                if (p->val < ancestor->val && q->val < ancestor->val) {
                    ancestor = ancestor->left;
                }
                else if (p->val > ancestor->val && q->val > ancestor->val) {
                    ancestor = ancestor->right;
                }
                else {
                    break;
                }
            }
            return ancestor;
        }
    };
    ```

1. 递归解法（不过这是二叉树的通用解法，而不是二叉搜索树的解法）

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
    public:
        TreeNode* dfs(TreeNode *root, TreeNode *p, TreeNode *q)
        {
            if (!root) return nullptr;
            if (root == p || root == q) return root;
            TreeNode *left = dfs(root->left, p, q);
            TreeNode *right = dfs(root->right, p, q);
            if (left && right) return root;
            if (left && !right) return left;
            return right;
        }

        TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
            return dfs(root, p, q);
        }
    };
    ```

### 合并二叉树

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

示例 1:

```
输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
```

代码：

1. 先序遍历

    我觉得这道题很经典。可以用先序遍历做，可以用后序遍历做，也可以用中序遍历做。我们还可以思考一下在递归中函数的返回值的使用。

    对于先序遍历，我们先处理当前节点，然后再处理左子树和右子树。正是因为当前节点、左子树、右子树这三者互不干扰，所以我们可以用不同的遍历方式。

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        TreeNode* dfs(TreeNode *r1, TreeNode *r2)
        {
            // 分情况进行讨论，哪个节点存在返回哪个
            if (!r1 && !r2) return nullptr;
            if (r1 && !r2) return r1;
            if (!r1 && r2)  return r2;
            r1->val += r2->val;  // 若两个节点都存在，则相加节点值，依次处理完左右两个节点后，返回左侧节点值
            r1->left = dfs(r1->left, r2->left);
            r1->right = dfs(r1->right, r2->right);
            return r1;
        }

        TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
            return dfs(root1, root2);
        }
    };
    ```

    先序遍历先处理当前节点，然后分别处理左节点和右节点。

    对于当前节点，可能有存在和不存在两种情况。如果树 1 的当前节点和树 2 的当前节点都不存在，那么递归中止。如果树 1 的当前节点存在，树 2 的当前节点不存在，那么什么都不用做。如果树 1 的当前节点不存在，树 2 的当前节点存在，那么需要拿树 2 的当前节点替换树 1 的当前节点的位置。怎么替换呢？我们需要拿父节点的信息，或者把树 2 的当前节点存储起来后面交给父节点使用。

    如果需要拿父节点信息，那么我们可以用后序遍历，即假设两个子树都已经处理好了，然后处理两个子节点就可以了。

    而如果想把当前节点存储起来，就可以利用函数的返回值了。即上面的 c++ 代码的解法。

    其实很多关于树的题都归结于对根节点存在与否的讨论。

1. 自己写的后序遍历

    ```cpp
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        void recur(TreeNode *r1, TreeNode *r2)
        {
            if (!r1 || !r2)
                return;
            recur(r1->left, r2->left);
            recur(r1->right, r2->right);
            if (r1->left && r2->left)
                r1->left->val += r2->left->val;
            else if (!r1->left && r2->left)
                r1->left = r2->left;
            if (r1->right && r2->right)
                r1->right->val += r2->right->val;
            else if (!r1->right && r2->right)
                r1->right = r2->right;
        }
        TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
            if (root1 && root2)
                root1->val += root2->val;
            else if (!root1 && root2)
                root1 = root2;
            recur(root1, root2);
            return root1;
        }
    };
    ```

    由于后序遍历时，我们处理的是父节点，所以需要对各种情况分类讨论。

1. 自己想出来的解法，挺复杂的，仅作纪念

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        void dfs(TreeNode *r1, TreeNode *r2, TreeNode *p1, TreeNode *p2, bool left)
        {
            if (!r1 && !r2) return;
            else if (!r1 && r2)
            {
                if (left)
                {
                    p1->left = new TreeNode(r2->val);
                    r1 = p1->left;
                }
                else
                {
                    p1->right = new TreeNode(r2->val);
                    r1 = p1->right;
                }
            }
            else if (r1 && !r2)
            {
                if (left)
                {
                    p2->left = new TreeNode(r1->val);
                    r2 = p2->left;
                }
                else
                {
                    p2->right = new TreeNode(r1->val);
                    r2 = p2->right;
                }
            }
            else if (r1 && r2) r1->val += r2->val;
            
            if (r1 || r2)
            {
                dfs(r1->left, r2->left, r1, r2, true);
                dfs(r1->right, r2->right, r1, r2, false);
            }
        }

        TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
            if (!root1 && root2) return root2;
            if (root1 && !root2) return root1;
            dfs(root1, root2, root1, root2, true);
            return root1;
        }
    };
    ```

1. 深度优先

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
            if (!root1) return root2;
            if (!root2) return root1;
            TreeNode *merged = new TreeNode(root1->val + root2->val);
            merged->left = mergeTrees(root1->left, root2->left);
            merged->right = mergeTrees(root1->right, root2->right);
            return merged;
        }
    };
    ```

1. 广度优先（太麻烦了，没看）

    ```c++
    class Solution {
    public:
        TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
            if (t1 == nullptr) {
                return t2;
            }
            if (t2 == nullptr) {
                return t1;
            }
            auto merged = new TreeNode(t1->val + t2->val);
            auto q = queue<TreeNode*>();
            auto queue1 = queue<TreeNode*>();
            auto queue2 = queue<TreeNode*>();
            q.push(merged);
            queue1.push(t1);
            queue2.push(t2);
            while (!queue1.empty() && !queue2.empty()) {
                auto node = q.front(), node1 = queue1.front(), node2 = queue2.front();
                q.pop();
                queue1.pop();
                queue2.pop();
                auto left1 = node1->left, left2 = node2->left, right1 = node1->right, right2 = node2->right;
                if (left1 != nullptr || left2 != nullptr) {
                    if (left1 != nullptr && left2 != nullptr) {
                        auto left = new TreeNode(left1->val + left2->val);
                        node->left = left;
                        q.push(left);
                        queue1.push(left1);
                        queue2.push(left2);
                    } else if (left1 != nullptr) {
                        node->left = left1;
                    } else if (left2 != nullptr) {
                        node->left = left2;
                    }
                }
                if (right1 != nullptr || right2 != nullptr) {
                    if (right1 != nullptr && right2 != nullptr) {
                        auto right = new TreeNode(right1->val + right2->val);
                        node->right = right;
                        q.push(right);
                        queue1.push(right1);
                        queue2.push(right2);
                    } else if (right1 != nullptr) {
                        node->right = right1;
                    } else {
                        node->right = right2;
                    }
                }
            }
            return merged;
        }
    };
    ```

### 填充每个节点的下一个右侧指针

给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

代码：

1. 层序遍历，每层按顺序从左指向右就好了

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* left;
        Node* right;
        Node* next;

        Node() : val(0), left(NULL), right(NULL), next(NULL) {}

        Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

        Node(int _val, Node* _left, Node* _right, Node* _next)
            : val(_val), left(_left), right(_right), next(_next) {}
    };
    */

    class Solution {
    public:
        Node* connect(Node* root) {
            if (!root) return root;
            queue<Node*> q;
            q.push(root);
            q.push(nullptr);
            Node *cur;
            while (q.size() > 1)
            {
                cur = q.front();
                q.pop();
                if (!cur)
                {
                    q.push(nullptr);
                    continue;
                }
                cur->next = q.front();
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }

            return root;
        }
    };
    ```

1. 利用上一层已经创建好的`next`指针，可将空间复杂度降到`O(1)`

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* left;
        Node* right;
        Node* next;

        Node() : val(0), left(NULL), right(NULL), next(NULL) {}

        Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

        Node(int _val, Node* _left, Node* _right, Node* _next)
            : val(_val), left(_left), right(_right), next(_next) {}
    };
    */

    class Solution {
    public:
        Node* connect(Node* root) {
            if (!root) return root;
            Node *start = root, *cur;
            while (start->left)
            {
                cur = start;
                while (cur)
                {
                    cur->left->next = cur->right;
                    if (cur->next) cur->right->next = cur->next->left;
                    cur = cur->next;
                }
                start = start->left;
            }

            return root;
        }
    };
    ```



### 二叉搜索树中的插入操作

给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。

 
```
示例 1：

输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]
解释：另一个满足题目要求可以通过的树是：

示例 2：

输入：root = [40,20,60,10,30,50,70], val = 25
输出：[40,20,60,10,30,50,70,null,null,25]
```

代码：

1. 先序遍历

    ```cpp
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        int val;
        bool done;
        void dfs(TreeNode *r)
        {
            if (done) return;  // 剪枝
            if (val < r->val && !r->left)  // 判断状态。为了条理清晰故意写成这样的，没有与下面合并
            {
                r->left = new TreeNode(val);
                done = true;
                return;
            }
            else if (val > r->val && !r->right)
            {
                r->right = new TreeNode(val);
                done = true;
                return;
            }
            
            if (val < r->val && r->left)  // 搜索下一层
                dfs(r->left);
            else if (val > r->val && r->right)
                dfs(r->right);
        }

        TreeNode* insertIntoBST(TreeNode* root, int val) {
            if (!root)
            {
                root = new TreeNode(val);
                return root;
            }
            this->val = val;
            done = false;
            dfs(root);
            return root;
        }
    };
    ```

    基本想法：
    
    1. 状态检查
    
        如果 val 小于当前节点，并且当前节点没有左节点，那么就把 val 插入到左节点。如果 val 大于当前节点，并且当前节点没有右节点，那么就把 val 插入到右节点。
        
    1. 搜索下一层
    
        如果 val 小于当前节点，并且左节点存在，那么我们就去左边继续找。如果 val 大于当前节点，并且右节点存在，那么我们就去右边继续找。

    1. 如果`root`是空节点，那么特殊处理。

1. 迭代

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        TreeNode* insertIntoBST(TreeNode* root, int val) {
            if (!root) return new TreeNode(val);
            TreeNode *head = root;
            while (root->left || root->right)
            {
                if (val < root->val)
                {
                    if (!root->left)
                    {
                        root->left = new TreeNode(val);
                        return head;
                    }
                    root = root->left;
                }
                else
                {
                    if (!root->right)
                    {
                        root->right= new TreeNode(val);
                        return head;
                    }
                    root = root->right;
                }
            }
            if (val < root->val) root->left = new TreeNode(val);
            else root->right = new TreeNode(val);
            return head;
        }
    };
    ```

    其实当时没想明白，觉得只要节点是叶子节点，那么就停止迭代，对这种情况单独判断。事实上，完全可以写成：

    ```c++
    class Solution {
    public:
        TreeNode* insertIntoBST(TreeNode* root, int val) {
            if (!root) return new TreeNode(val);
            TreeNode *head = root;
            while (true)
            {
                if (val < root->val)
                {
                    if (!root->left)
                    {
                        root->left = new TreeNode(val);
                        return head;
                    }
                    root = root->left;
                }
                else
                {
                    if (!root->right)
                    {
                        root->right= new TreeNode(val);
                        return head;
                    }
                    root = root->right;
                }
            }
            return head;
        }
    };
    ```

    这不禁令我们深思，`while()`括号中究竟该填些什么，才能自然而又效率高呢。如果循环体中的任务是遍历，那么`while()`中就应该写上遍历到末尾的条件；如果任务是搜索，那么就应该写上搜索到或者遍历结束的条件。如果实在难以在一行写完，那么就写成`true`，然后尽量在`while()`的开头写出口条件。

1. 后来又写的迭代，把出口位置放到了前面，但是判断的次数更多了

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        TreeNode* insertIntoBST(TreeNode* root, int val) {
            if (!root) return new TreeNode(val);
            TreeNode *r = root;
            while (true)
            {
                if (val < root->val && !root->left)  // 因为 while 中填的是 true，所以这里先写 while 的出口。出口一共有两个，要么是插入值比当前节点小，并且当前节点没有左节点；要么是插入值比当前节点大，并且当前节点没有右节点。
                {
                    root->left = new TreeNode(val);
                    break;
                }
                else if (val > root->val && !root->right)  // 这里并没有写成 if (root && val > root->val && !root->right)
                {
                    root->right = new TreeNode(val);
                    break;
                }

                if (val < root->val) root = root->left;  // 这里也没有写成 if (root && val < root->val && root->left)
                else root = root->right;
            }
            return r;
        }
    };
    ```

1. 递归

    ```c++
    class Solution {
    public:
        TreeNode* insertIntoBST(TreeNode* root, int val) {
            if (!root) return new TreeNode{val};
            auto&& [v, l, r] = *root;  // 事实上，这里可以写成 auto& [v, l, r]，这种语法叫结构化绑定，可以将 *root 对象中的各个字段绑定到 v, l, r 上。
            if (val < v) 
                l = insertIntoBST(l, val);
            else 
                r = insertIntoBST(r, val);
            return root;
        }
    };
    ```

### 验证二叉搜索树

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

* 节点的左子树只包含小于当前节点的数。
* 节点的右子树只包含大于当前节点的数。
* 所有左子树和右子树自身必须也是二叉搜索树。

```
示例 1:

输入:
    2
   / \
  1   3
输出: true
示例 2:

输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。
```

分析：注意题目中说的是节点的左子树中所有的数都比当前节点小，右子树中所有的数都比当前节点大。所以单纯比较左右叶子节点和当前节点的大小是不行的。

1. 递归，中序遍历

    中序遍历能保证若一个树是有效的二叉搜索数，那么它的遍历结果一定是递增的。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        long pre;  // 官方会测试 INT32_MIN，所以这里用 long
        Solution(): pre(INT64_MIN) {}
        bool isValidBST(TreeNode* root) {
            if (!root) return true;
            if (!isValidBST(root->left)) return false;
            if (root->val <= pre) return false;
            pre = root->val;
            return isValidBST(root->right); 
        }
    };
    ```

    无论先序遍历还是后序遍历，在这道题中都需要解决一个问题：根节点的信息必须直通叶子节点。但是这两种遍历无论是哪个都做不到。为什么中序遍历就能做到这一点呢？【似乎信息并不是直通所有的叶子节点，而是部分叶子节点】

    另外，如何才能不用`long`来解决这个问题呢。

1. 先序遍历

    ```c++
    class Solution {
    public:
        bool helper(TreeNode* root, long long lower, long long upper) {
            if (root == nullptr) {
                return true;
            }
            if (root -> val <= lower || root -> val >= upper) {
                return false;
            }
            return helper(root -> left, lower, root -> val) && helper(root -> right, root -> val, upper);
        }
        bool isValidBST(TreeNode* root) {
            return helper(root, LONG_MIN, LONG_MAX);
        }
    };
    ```

1. 中序遍历的迭代

    ```c++
    class Solution {
    public:
        bool isValidBST(TreeNode* root) {
            stack<TreeNode*> stack;
            long long inorder = (long long)INT_MIN - 1;

            while (!stack.empty() || root != nullptr) {
                while (root != nullptr) {
                    stack.push(root);
                    root = root -> left;
                }
                root = stack.top();
                stack.pop();
                // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
                if (root -> val <= inorder) {
                    return false;
                }
                inorder = root -> val;
                root = root -> right;
            }
            return true;
        }
    };
    ```

1. 另外一种中序遍历的递归

    ```c++
    class Solution {
        long pre = Long.MIN_VALUE; // 记录上一个节点的值，初始值为Long的最小值

        public boolean isValidBST(TreeNode root) {
            return inorder(root);
        }

        // 中序遍历
        private boolean inorder(TreeNode node) {
            if(node == null) return true;
            boolean l = inorder(node.left);
            if(node.val <= pre) return false;
            pre = node.val;
            boolean r = inorder(node.right);
            return l && r;
        }
    }
    ```

1. 自己写了一个，有问题

    ```cpp
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool ans;
        int dfs(TreeNode *r, bool left)
        {
            if (!ans) return 0;
            if (!r->left && !r->right)
                return r->val;
            int max_val = INT32_MIN;
            int min_val = INT32_MAX;
            if (r->left)
                max_val = dfs(r->left, true);
            if (r->right)
                min_val = dfs(r->right, false);
            if (max_val >= r->val || min_val <= r->val)
            {
                ans = false;
                return 0;
            }
            if (left)
                return max(max(max_val, r->val), min_val);
            return min(min(min_val, r->val), max_val);
        }

        bool isValidBST(TreeNode* root) {
            ans = true;
            dfs(root, true);
            return ans;
        }
    };
    ```

    思路是用后序遍历，收集左子树的最大值，右子树的最小值，如果左子树的最大值大于当前节点，或右子树的最小值小于当前节点，那么说明不符合二叉搜索树。

    但是这个思路是错的，因为一个节点可能属于某个节点的左子树，也可能属于其他节点的右子树。这样的话在收集信息的过程中信息就会丢失。

    正确的版本应该是这样的：

    ```cpp
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        bool ans;
        int max_val(TreeNode *r)
        {
            if (!r->right)
                return r->val;
            return max_val(r->right);
        }

        int min_val(TreeNode *r)
        {
            if (!r->left)
                return r->val;
            return min_val(r->left);
        }

        void dfs(TreeNode *r)
        {
            if (!ans) return;
            if (r->left && max_val(r->left) >= r->val)
            {
                ans = false;
                return;
            }
            if (r->right && min_val(r->right) <= r->val)
            {
                ans = false;
                return;
            }
            if (r->left) dfs(r->left);
            if (r->right) dfs(r->right);
        }

        bool isValidBST(TreeNode* root) {
            ans = true;
            dfs(root);
            return ans;
        }
    };
    ```

    对于每个节点，都在左侧找到最大节点，在右侧找到最小节点。而且每个节点都是独立寻找，互不影响。

    我觉得代码还有优化空间，可以把`max_val`和`min_val`改成迭代版本，然后不必找到最后一个节点，发现不符合条件的就立即返回。

    （不清楚这个代码是怎么通过样例的，可能有这样的逻辑链：当子树没问题时，我对当前节点的判断方法就没问题。而当子树有问题时，会在子树的位置定位出问题。但是我判断当前节点，用的是先序遍历，只能保证左子树没问题，不能保证右子树没问题。 ）

### 两数之和 IV - 输入 BST （两数之和 IV - 输入二叉搜索树）

给定一个二叉搜索树和一个目标结果，如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。

```
案例 1:

输入: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

输出: True
 

案例 2:

输入: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 28

输出: False
```

代码：

1. 遍历 + 哈希表

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        unordered_set<int> s;
        bool findTarget(TreeNode* root, int k) {
            if (!root) return false;
            if (s.find(k - root->val) != s.end())
                return true;
            else
                s.insert(root->val);
            return findTarget(root->left, k) || findTarget(root->right, k);
        }
    };
    ```

    还可以层序遍历。

1. BST 的中序遍历是递增的，所以还可以先把中序遍历结果保存下来，然后用双指针

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        bool findTarget(TreeNode* root, int k) {
            vector<int> nums;
            TreeNode *r = root, *prev = nullptr;
            stack<TreeNode*> s;
            while (r || !s.empty())
            {
                while (r)
                {
                    s.push(r);
                    r = r->left;
                }
                r = s.top();
                s.pop();
                nums.push_back(r->val);
                r = r->right;
            }
            int left = 0, right = nums.size() - 1;
            while (left < right)
            {
                if (nums[left] + nums[right] < k) ++left;
                else if (nums[left] + nums[right] > k) --right;
                else return true;
            }
            return false;
        }
    };
    ```

1. 当已知一个根节点的时候，可以用 BST 的性质展开搜索，从而不使用哈希表

    ```cpp
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        bool ans;
        int k;
        TreeNode *found;
        TreeNode *root;
        void search_bst(TreeNode *r, int t)
        {
            if (found) return;
            if (!r) return;
            if (t < r->val) search_bst(r->left, t);
            else if (t > r->val) search_bst(r->right, t);
            else
            {
                found = r;
                return;
            }
        }

        void dfs(TreeNode *r)
        {
            if (ans) return;
            search_bst(root, k - r->val);
            if (found && found != r)
            {
                ans = true;
                return;
            }
            found = nullptr;
            search_bst(root, k - r->val);
            if (found && found != r)
            {
                ans = true;
                return;
            }
            found = nullptr;
            if (r->left) dfs(r->left);
            if (r->right) dfs(r->right);
        }

        bool findTarget(TreeNode* root, int k) {
            ans = false;
            this->k = k;
            this->root = root;
            found = nullptr;
            dfs(root);
            return ans;
        }
    };
    ```

    我们首先用先序遍历确定第一个节点，然后用二叉树搜索确定第二个节点。只不过细节有亿点点多。

    我觉得可以把当前节点存储到类成员中，在二叉搜索树搜索的时候，直接跳过当前节点，或许可以简化流程。

1. Morris 中序遍历加双指针，从两端分别开始迭代遍历

### 二叉树中所有距离为 K 的结点

给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。

返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。

示例：

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
输出：[7,4,1]
解释：
所求结点为与目标结点（值为 5）距离为 2 的结点，
值分别为 7，4，以及 1

注意，输入的 "root" 和 "target" 实际上是树上的结点。
上面的输入仅仅是对这些对象进行了序列化描述。
```

代码：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> res;
    unordered_map<int, TreeNode*> m;
    void find_parent(TreeNode *root)
    {
        if (root->left)
        {
            m[root->left->val] = root;
            find_parent(root->left);
        }
        if (root->right)
        {
            m[root->right->val] = root;
            find_parent(root->right);
        }
    }

    void dfs(TreeNode *root, TreeNode *prev, int k, int dis)
    {
        if (dis == k)
        {
            res.push_back(root->val);
            return;
        }

        // 向下搜索左节点
        if (root->left && root->left != prev)
            dfs(root->left, root, k, dis+1);

        // 向下搜索右节点
        if (root->right && root->right != prev)
            dfs(root->right, root, k, dis+1);

        // 向上搜索
        if (m[root->val] && m[root->val] != prev)
            dfs(m[root->val], root, k, dis+1);
    }

    vector<int> distanceK(TreeNode* root, TreeNode* target, int k)
    {
        find_parent(root);
        dfs(target, nullptr, k, 0);
        return res;
    }
};
```

### 填充每个节点的下一个右侧节点指针 II

给定一个二叉树

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

 

进阶：

你只能使用常量级额外空间。
使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
 
```
示例：

输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化输出按层序遍历顺序（由 next 指针连接），'#' 表示每层的末尾。
```

1. 层序遍历

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* left;
        Node* right;
        Node* next;

        Node() : val(0), left(NULL), right(NULL), next(NULL) {}

        Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

        Node(int _val, Node* _left, Node* _right, Node* _next)
            : val(_val), left(_left), right(_right), next(_next) {}
    };
    */

    class Solution {
    public:
        Node* connect(Node* root) {
            if (!root) return nullptr;
            queue<Node*> q;
            q.push(root);
            Node *node;
            int size;
            while (!q.empty())
            {
                size = q.size();
                for (int i = 0; i < size; ++i)
                {
                    node = q.front();
                    q.pop();
                    if (i != size - 1) node->next = q.front();
                    else node->next = nullptr;
                    if (node->left) q.push(node->left);
                    if (node->right) q.push(node->right);
                }
            }
            return root;
        }
    };
    ```

1. 使用已经建立好的`next`指针

    ```c++
    /*
    // Definition for a Node.
    class Node {
    public:
        int val;
        Node* left;
        Node* right;
        Node* next;

        Node() : val(0), left(NULL), right(NULL), next(NULL) {}

        Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

        Node(int _val, Node* _left, Node* _right, Node* _next)
            : val(_val), left(_left), right(_right), next(_next) {}
    };
    */

    class Solution {
    public:
        void handle(Node* &prev, Node* &p, Node* &next_start)
        {
            if (prev) prev->next = p;  // 如果是下一层的第一个节点，那么跳过
            if (!next_start) next_start = p;  // 因为下一行的起始位置不一定在第一个，所以需要一直找下去，直到找到为止
            prev = p;
        }

        Node* connect(Node* root) {
            if (!root) return nullptr;
            Node *start = root, *prev, *next_start;
            while (start)
            {
                prev = nullptr;  // 在每一层新的开始，prev 和 next_start 都要被初始化为空指针
                next_start = nullptr;
                for (Node *p = start; p != nullptr; p = p->next)
                {
                    if (p->left) handle(prev, p->left, next_start);
                    if (p->right) handle(prev, p->right, next_start);
                }
                start = next_start;
            }
            return root;
        }
    };
    ```

### 相同的树

给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

```
示例 1：


输入：p = [1,2,3], q = [1,2,3]
输出：true
示例 2：


输入：p = [1,2], q = [1,null,2]
输出：false
示例 3：


输入：p = [1,2,1], q = [1,1,2]
输出：false
```

代码：

首先考虑节点的有无，再考虑如果两个节点都存在时值是否相等，最后考虑两侧的子树也必须相同。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool dfs(TreeNode *p, TreeNode *q)
    {
        if ((p && !q) || (!p && q)) return false;
        if (!p && !q) return true;
        if (p->val != q->val) return false;
        return dfs(p->left, q->left) && dfs(p->right, q->right);
    }

    bool isSameTree(TreeNode* p, TreeNode* q) {
        return dfs(p, q);
    }
};
```

### 监控二叉树

给定一个二叉树，我们在树的节点上安装摄像头。

节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。

计算监控树的所有节点所需的最小摄像头数量。

```
示例 1：

输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。
示例 2：

输入：[0,0,null,0,null,0,null,null,0]
输出：2
解释：需要至少两个摄像头来监视树的所有节点。 上图显示了摄像头放置的有效位置之一。
```

代码：

1. 动态规划

    太难了，不清楚为啥一定要定义 3 个状态。看不懂。

    ```c++
    struct Status {
        int a, b, c;
    };

    class Solution {
    public:
        Status dfs(TreeNode* root) {
            if (!root) {
                return {INT_MAX / 2, 0, 0};
            }
            auto [la, lb, lc] = dfs(root->left);
            auto [ra, rb, rc] = dfs(root->right);
            int a = lc + rc + 1;
            int b = min(a, min(la + rb, ra + lb));
            int c = min(a, lb + rb);
            return {a, b, c};
        }

        int minCameraCover(TreeNode* root) {
            auto [a, b, c] = dfs(root);
            return b;
        }
    };
    ```

### 词典中最长的单词

给出一个字符串数组words组成的一本英语词典。从中找出最长的一个单词，该单词是由words词典中其他单词逐步添加一个字母组成。若其中有多个可行的答案，则返回答案中字典序最小的单词。

若无答案，则返回空字符串。

```
示例 1：

输入：
words = ["w","wo","wor","worl", "world"]
输出："world"
解释： 
单词"world"可由"w", "wo", "wor", 和 "worl"添加一个字母组成。
示例 2：

输入：
words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
输出："apple"
解释：
"apply"和"apple"都能由词典中的单词组成。但是"apple"的字典序小于"apply"。
```

1. 前缀树

```c++
class Solution {
public:
    // 字典树定义
    class Node{
    public:
        vector<Node *>nexts = vector<Node *> (26, nullptr);
    };
    // 比较器函数在类中一定要加static
    // 长度相等则比较字典序， 长度不等则长度小的在前面
    static bool cmp(const string& s1, const string& s2){
        return s1.size() == s2.size()? s1 < s2 : s1.size() < s2.size();
    }

    string longestWord(vector<string>& words) {
        int n = words.size();
        // 先将单词进行排序， 那么长度短的单词一定会排在前面，题目中说一次只能新增一个字母
        // 所以可以先看新加入的单词的前n - 1个字母组成的前缀是否被加入过，
        // 如果加入过，则这个新加入的单词可以在前面单词的基础上再加入最后一个字母形成
        // 那么这个单词就可以作为备选项
        // 将这个单词的最后一个字母加入到字典树中，在看该单词的长度是否能够更新最大值
        // 如果没有加入过， 则直接跳过这个单词
        sort(words.begin(), words.end(), cmp);
        int maxlen = 0;
        string maxstr = "";
        Node *root = new Node();
        for(int i = 0; i < n; i++){
            int len = words[i].size();
            Node *node = root;
            int path = 0;
            bool flag = false;
            // 检查前n - 1字符是否已经被加入过了
            for(int j = 0; j < len - 1; j++){
                path = words[i][j] - 'a';
                // 如果之前没有加入过， 则直接跳过这个单词
                if(node->nexts[path] == nullptr){
                    flag = true;
                    break;
                }
                node = node->nexts[path];
            }
            // 如果之前加入过前n - 1个字母， 则可以在加入最后一个字母
            if(flag){  // 改成 if(!flag)
                node->nexts[words[i][len - 1]] = new Node();  // 改成 node->nexts[words[i][len - 1]-'a'] = new Node();
            }
            // 因为排序后，字典序小的单词在前面， 所以长度相等的时候不要更新， 大于的时候才更新
            if(len > maxlen){  // 这几秆应该在 if(flag) 模块里
                maxlen = len;
                maxstr = words[i];
            }
        }
        return maxstr;
    }
};
```

```c++
class Solution {
public:
    class TreeNode {
    public:
        bool end;
        string word;
        TreeNode* next[26];
        TreeNode () {
            word = "";
            end = false;
            for(auto& n : next) n = NULL;
        };
    };
    TreeNode* buildTree(vector<string>& words) {
        TreeNode* root = new TreeNode(), *p;
        for (auto& w : words) {
            p = root;
            int len = w.size();
            for (int i = 0; i < len; i++) {
                int index = w[i] - 'a';
                if (p->next[index] ==  NULL)
                    p->next[index] = new TreeNode();
                p = p->next[index];
            }
            p->end = true;
            p->word = w;
        }
        return root;
    }
    string ans = "";
    int maxDepth = 0;
    void dfs(TreeNode* root,int depth){
        if(depth > 0 && !root->end) return;
        if(depth > maxDepth){
            ans = root->word;
            maxDepth = depth;
        }
        for(int i = 0; i < 26; i ++){
            if(root->next[i] != NULL)
                dfs(root->next[i],depth+1);
        }
    }
    string longestWord(vector<string>& words) {
        TreeNode* root = buildTree(words);
        dfs(root,0);
        return ans;
    }
};
```

1. 暴力法

### 二叉树的直径

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

 
```
示例 :
给定二叉树

          1
         / \
        2   3
       / \     
      4   5    
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。
```

注意：两结点之间的路径长度是以它们之间边的数目表示。

代码：

1. 后序遍历。自己写的。

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        int ans;
        int dfs(TreeNode *root)
        {
            if (!root) return 0;
            if (!root->left && !root->right) return 0;
            int left = dfs(root->left);
            int right = dfs(root->right);
            if (root->left) ans = max(ans, 1 + left);
            if (root->right) ans = max(ans, 1 + right);
            if (root->left && root->right) ans = max(ans, 2 + left + right);
            return max(left, right) + 1;
        }
        int diameterOfBinaryTree(TreeNode* root) {
            ans = 0;
            dfs(root);
            return ans;
        }
    };
    ```

    别人写的：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int ans;
        int dfs(TreeNode *root)
        {
            if (!root) return 0;
            int left = dfs(root->left);
            int right = dfs(root->right);
            ans = max(ans, left + right);
            return max(left, right) + 1;
        }
        int diameterOfBinaryTree(TreeNode* root) {
            ans = 0;
            dfs(root);
            return ans;
        }
    };
    ```

    为何别人写的能更简洁一些？

### 二叉树中的最大路径和

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

```
示例 1：


输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
示例 2：


输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

代码：

1. 后序遍历

    这道题很像动态规划。如果我们知道了左子树的最大路径，右子树的的最大路径，那么只要判断是否加上当前节点就可以了。

    这种先求子问题，再解决当前问题的方法，不就是动态规划的自底向上方法吗。可惜树具有独特的递归结构，不知道该怎么用迭代法自底向上地遍历整个树。那么就暂定用后序遍历来做。

    我们先定义`dfs()`返回的是子树带上根节点的最大路径和。但是仔细一分析发现，子树的最大路径是可以`左节点 + 根节点 + 右节点`这种模式的。但是如果我们把子树的路径作为父节点的最大路径的一部分，就只能是`{根节点 + 左节点, 根节点 + 右节点, 根节点}`这三者中选一个，不能是左节点和右节点都有。

    因此`dfs()`返回的，应该是把子树不同路径情况切割开来最大的那一种。这样我们无法直接将`dfs()`的结果作为答案返回，就只能额外维护一个答案变量。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int ans;
        int dfs(TreeNode *root)
        {
            if (!root) return 0;
            int left = dfs(root->left);
            int right = dfs(root->right);
            int temp[4] = {root->val, root->val + left, root->val + right, root->val + left + right};
            ans = max(ans, *max_element(&temp[0], &temp[4]));  // 维护答案时使用的是 4 个元素
            return *max_element(&temp[0], &temp[3]);  // 递归返回时使用的是前 3 个元素
        }

        int maxPathSum(TreeNode* root) {
            ans = INT32_MIN;
            dfs(root);
            return ans;
        }
    };
    ```

    还有一种比较简洁的写法，巧妙地用 0 来表示排除掉这个路径：

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        int ans;
        int dfs(TreeNode *root)
        {
            if (!root) return 0;
            int left = max(dfs(root->left), 0);  // 如果左子树对路径的贡献为负值，就抛弃这条路径
            int right = max(dfs(root->right), 0);
            ans = max(ans, root->val + left + right);
            return root->val + max(left, right);
        }

        int maxPathSum(TreeNode* root) {
            ans = INT32_MIN;
            dfs(root);
            return ans;
        }
    };
    ```

### 删除二叉搜索树中的节点

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。
 
```
示例 1:

输入：root = [5,3,6,2,4,null,7], key = 3
输出：[5,4,6,2,null,null,7]
解释：给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
另一个正确答案是 [5,2,6,null,4,null,7]。


示例 2:

输入: root = [5,3,6,2,4,null,7], key = 0
输出: [5,3,6,2,4,null,7]
解释: 二叉树不包含值为 0 的节点
示例 3:

输入: root = [], key = 0
输出: []
```

代码：

1. 递归删除，挺麻烦的

    ```c++
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (!root) return nullptr;   // key not found in bst

        if (key < root->val) {
            root->left = deleteNode(root->left, key);
        }
        else if (key > root->val) {
            root->right = deleteNode(root->right, key);
        }
        else {
            // case 1:  if the root itself is a leaf node
            if (!root->left && !root->right) {
                return nullptr;
            }

            // case 2:  if the root only has right child
            if (!root->left && root->right) {
                return root->right;
            }

            // case 3:  if the root only has left child
            if (root->left && !root->right) {
                return root->left;
            }

            // case 4:  if the root has both left and right child
            if (root->left && root->right) {
                //  find the successor from right subtree:
                //  1. the successor must be the samllest element in subtree
                //  2. the successor could be either the right or left child of its ancestor
                auto ancestor = root;
                auto successor = root->right;
                while (successor->left) {
                    ancestor = successor;
                    successor = successor->left;
                }
                root->val = successor->val;
                if (successor == ancestor->right) {  // 这几句不知道是啥意思 
                    ancestor->right = deleteNode(successor, successor->val);
                }
                else {
                    ancestor->left = deleteNode(successor, successor->val);
                }
            }
        }
        return root;
    }
    ```

    二叉搜索树的中序遍历的序列是递增排序的序列。

    ```java
    public LinkedList<Integer> inorder(TreeNode root, LinkedList<Integer> arr) {
        if (root == null) return arr;
        inorder(root.left, arr);
        arr.add(root.val);
        inorder(root.right, arr);
        return arr;
    } 
    ```

    successor 代表的是中序遍历序列的下一个节点，即比当前节点大的最小节点，简称后继节点。 先取当前节点的右节点，然后一直取该节点的左节点，直到左节点为空，则最后指向的节点为后继节点。

    ```java
    public int successor(TreeNode root) {
        root = root.right;
        while (root.left != null) root = root.left;
        return root;
    } 
    ```
    
    predecessor 代表的是中序遍历序列的前一个节点。即比当前节点小的最大节点，简称前驱节点。先取当前节点的左节点，然后取该节点的右节点，直到右节点为空，则最后指向的节点为前驱节点。

    ```java
    public int predecessor(TreeNode root) {
        root = root.left;
        while (root.right != null) root = root.right;
        return root;
    } 
    ```

1. 别人写的迭代

    ```c++
    class Solution {
    public:
        TreeNode* deleteNode(TreeNode* root, int key) {
            if (!root)return root;
            TreeNode* temp = root;
            TreeNode* ret = root;
            //因为根节点没有父节点所以要单独处理
            if (root->val == key){
                TreeNode* l = root->left;
                if (root->right != nullptr){
                    ret = root->right;
                    root = root->right;
                    while (root->left != nullptr){
                        root = root->left;
                    }
                    root->left = l;
                }
                else{
                    ret = root->left;
                }
                return ret;
            }
            //进行比较，遇到目标节点时，若其右子树为空则用左子树代替，
            //若不为空，则用右子树代替，并且需要吧原来左子树拼接到后面
            while (temp != nullptr){
                if (temp->val == key){
                    TreeNode* l = temp->left;
                    if (temp->right != nullptr){
                        if (root->val > temp->val){
                            root->left = temp->right;
                        }
                        else{
                            root->right = temp->right;
                        }
                        root = temp->right;
                        temp = temp->right->left;
                        while (temp != nullptr){
                            root = temp;
                            temp = temp->left;
                        }
                        root->left = l;
                    }
                    else{
                        if (root->val > temp->val){
                            root->left = l;
                        }
                        else{
                            root->right = l;
                        }

                    }
                    return ret;

                }
                else if (temp->val < key){
                    root = temp;
                    temp = temp->right;
                }
                else{
                    root = temp;
                    temp = temp->left;

                }
            }

            return ret;

        }
    };
    ```

1. 一个比较简洁的递归，会增加树的高度

    ```c++
    class Solution {
    public:
        TreeNode* deleteNode(TreeNode* root, int key) 
        {
            if (root == nullptr)    return nullptr;
            if (key > root->val)    root->right = deleteNode(root->right, key);     // 去右子树删除
            else if (key < root->val)    root->left = deleteNode(root->left, key);  // 去左子树删除
            else    // 当前节点就是要删除的节点
            {
                if (! root->left)   return root->right; // 情况1，欲删除节点无左子
                if (! root->right)  return root->left;  // 情况2，欲删除节点无右子
                TreeNode* node = root->right;           // 情况3，欲删除节点左右子都有 
                while (node->left)          // 寻找欲删除节点右子树的最左节点
                    node = node->left;
                node->left = root->left;    // 将欲删除节点的左子树成为其右子树的最左节点的左子树
                root = root->right;         // 欲删除节点的右子顶替其位置，节点被删除
            }
            return root;    
        }
    };
    ```

    有人给出的改进版，这样不会增加树的高度：

    ```c++
    class Solution {
    public:
        TreeNode* deleteNode(TreeNode* root, int key) 
        {
            if (root == nullptr)    return nullptr;
            if (key > root->val)    root->right = deleteNode(root->right, key);     // 去右子树删除
            else if (key < root->val)    root->left = deleteNode(root->left, key);  // 去左子树删除
            else    // 当前节点就是要删除的节点
            {
                if (! root->left)   return root->right; // 情况1，欲删除节点无左子
                if (! root->right)  return root->left;  // 情况2，欲删除节点无右子
                TreeNode* node = root->right;           // 情况3，欲删除节点左右子都有 
                TreeNode* pre=root;
                while (node->left)          // 寻找欲删除节点右子树的最左节点
                {
                    pre=node;
                    node = node->left;
                }
                root->val=node->val; // 欲删除节点的下一个值顶替其位置，节点被删除
            if(pre->left->val==node->val) pre->left = node->right;    // 将下一个值的节点删除   
            else pre->right=node->right;    
            }
            return root;    
        }
    };
    ```

1. 其他的一个 c++ 递归版本

    ```c++
    class Solution {
    public:
        TreeNode* deleteNode(TreeNode* root, int key) {
            //第一种情况，整个树没有找到值为key的节点
            if(root == NULL) return root;
            if(root->val == key){
                //第二种情况，左右节点都为空，直接返回NULL
                //第三种情况，左节点为空，右节点不为空,将右节点作为根节点
                if(root->left == NULL) return root->right;
                //第四种情况，右节点为空，左节点不为空，将左节点作为根节点
                else if(root->right == NULL) return root->left;
                //第五种情况，左右节都不为空,要将目标节点的左孩子插入到目标节点右孩子最左下角的节点后，作为它的左孩子,然后将root节点的右孩子作为右节点，删除root。
                else{
                    TreeNode* cur = root->right;
                    while(cur->left != NULL){
                        cur = cur->left;
                    }
                    cur->left = root->left;
                    TreeNode* node = root;
                    root = root->right;
                    delete node;
                    return root;
                }
            }

            if(root->val > key) root->left = deleteNode(root->left, key);
            if(root->val < key) root->right = deleteNode(root->right, key);
            return root;
        }
    };
    ```

1. 自己写的

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        TreeNode* dfs(TreeNode *r, int key)  // 这个写法其实是把搜索和删除写到一起了。如果是分开写的话，该怎么写呢？分开写和合起来写各有什么样的区别呢？
        {
            if (!r) return nullptr;  // 题目的样例可能会让 root 删除自己
            if (r->left) r->left = dfs(r->left, key);  // 因为有可能删除当前节点，所以写成后续遍历的形式
            if (r->right) r->right = dfs(r->right, key);
            if (r->val == key)
            {
                if (!r->left && !r->right) return nullptr;
                if (!r->right) return r->left;  // 如果当前节点只有左子树，那么让左子树接替当前节点的位置就可以了。如果只有右子树，那么也可以让右
                // if (!r->left) return r->right;  // 但是如果多加一行这个，则会出问题，为什么呢？
                TreeNode *cur = r->right, *prev = r;
                while (cur->left) prev = cur, cur = cur->left;
                r->val = cur->val;
                if (prev == r) prev->right = dfs(r->right, r->right->val);  // 如果右节点没有左子树，那么递归删除
                else prev->left = dfs(cur, cur->val);  // 因为右节点的左节点也可能有右子树，所以这里也需要递归删除
            }
            return r;
        }

        TreeNode* deleteNode(TreeNode* root, int key) {
            return dfs(root, key);  // 因为有可能删除 root 节点
        }
    };
    ```

    首先这道题在考中序遍历，因为二叉搜索树一定满足中序遍历递增，如果我们删除某个节点，就一定要找到中序遍历里它的下一个节点或者上一个节点来接替它的位置。或者是把它的下一个节点和上一个节点拼接起来。

    其次，`dfs`函数之所以有返回值，并且写成后序遍历，是因为这样不用再找某个节点的父节点了。当然，也可以给dfs再加一个`parent`参数来解决这个问题，不过`parent`并不知道当前节点是左节点还是右节点，还需要写`if`来判断，不够优雅。

    再次，之所以分那么多情况讨论，是因为理想的情况总是无法满足。对于理想情况，中序遍历里某个节点的下一个节点是它右节点的左下方一直走到头的节点。但是如果一个节点没有右节点呢？那么它的下一个节点就是父节点，我们可以选择让左子树代替当前节点的位置。可是如果当前节点既没有右节点。也没有左子树呢？那么它就只能删除自己。如果一个节点有右节点，但是它的右节点没有左节点呢？那么我们就把这个右节点接替当前节点的位置。我们一步步倒序排查，才能找到所有非理想条件。而并不是这些非理想条件是一开始就想到的。

    最后，我们看到在有些情况中，会删除当前节点本身，这就是必须要用到后序遍历的原因：用后序遍历保留parent的信息。如果我们总是删除子节点，那么后序遍历就不是必用不可。

### 将有序数组转换为二叉搜索树

给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。


```
示例 1：


输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：

示例 2：


输入：nums = [1,3]
输出：[3,1]
解释：[1,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

代码：

1. 将中间节点（左，或右都可以）作为根节点，递归构建

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        TreeNode* dfs(vector<int> &nums, int left, int right)
        {
            if (left > right) return nullptr;
            int mid = left + (right - left) / 2;
            TreeNode *root = new TreeNode(nums[mid]);
            root->left = dfs(nums, left, mid-1);
            root->right = dfs(nums, mid+1,right);
            return root;
        }

        TreeNode* sortedArrayToBST(vector<int>& nums) {
            return dfs(nums, 0, nums.size()-1);
        }
    };
    ```

### 二叉树的右视图

给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

 

```
示例 1:

输入: [1,2,3,null,5,null,4]
输出: [1,3,4]
示例 2:

输入: [1,null,3]
输出: [1,3]
示例 3:

输入: []
输出: []
```

代码：

1. 层序遍历，记录每一层最后一个节点

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> rightSideView(TreeNode* root) {
            vector<int> ans;
            if (!root) return ans;
            queue<TreeNode*> q;
            q.push(root);
            int size;
            while (!q.empty())
            {
                size = q.size();
                for (int i = 0; i < size; ++i)
                {
                    root = q.front();
                    q.pop();
                    if (i == size - 1) ans.push_back(root->val);
                    if (root->left) q.push(root->left);
                    if (root->right) q.push(root->right);
                }
            }
            return ans;
        }
    };
    ```

1. dfs，先序遍历

    先访问右子树，再访问左子树。因为每层只有一个节点加入到答案中，所以用一个额外的量来标记进行到了第几层。

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class Solution {
    public:
        vector<int> ans;
        int cur_depth;
        void dfs(TreeNode *root, int depth)
        {
            if (!root) return;
            if (depth > cur_depth)
            {
                ans.push_back(root->val);
                ++cur_depth;
            }
            dfs(root->right, depth+1);
            dfs(root->left, depth+1);

        }
        vector<int> rightSideView(TreeNode* root) {
            cur_depth = 0;
            dfs(root, 1);
            return ans;
        }
    };
    ```

### 二叉搜索树迭代器

实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
int next()将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。

你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。

 

```
示例：

输入
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
输出
[null, 3, 7, true, 9, true, 15, true, 20, false]

解释
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // 返回 3
bSTIterator.next();    // 返回 7
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 9
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 15
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 20
bSTIterator.hasNext(); // 返回 False
```

代码：

1. 自己写的，其实就是把中序遍历的迭代写法拆开写了

    ```c++
    /**
    * Definition for a binary tree node.
    * struct TreeNode {
    *     int val;
    *     TreeNode *left;
    *     TreeNode *right;
    *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
    *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    * };
    */
    class BSTIterator {
    public:
        stack<TreeNode*> s;
        TreeNode *cur;

        BSTIterator(TreeNode* root) {
            cur = root;
            while (cur)
            {
                s.push(cur);
                cur = cur->left;
            }
        }
        
        int next() {
            cur = s.top();
            s.pop();
            int ans = cur->val;
            cur = cur->right;
            while (cur)
            {
                s.push(cur);
                cur = cur->left;
            }
            return ans;
        }
        
        bool hasNext() {
            return !s.empty() || cur;
        }
    };

    /**
    * Your BSTIterator object will be instantiated and called as such:
    * BSTIterator* obj = new BSTIterator(root);
    * int param_1 = obj->next();
    * bool param_2 = obj->hasNext();
    */
    ```

1. 官方给的答案，更简洁一点，思路一样

    ```c++
    class BSTIterator {
    private:
        TreeNode* cur;
        stack<TreeNode*> stk;
    public:
        BSTIterator(TreeNode* root): cur(root) {}
        
        int next() {
            while (cur != nullptr) {
                stk.push(cur);
                cur = cur->left;
            }
            cur = stk.top();
            stk.pop();
            int ret = cur->val;
            cur = cur->right;
            return ret;
        }
        
        bool hasNext() {
            return cur != nullptr || !stk.empty();
        }
    };
    ```

1. 还可以先用递归把所有的节点值保存在数组里，然后直接返回

## 栈

设计一个支持push，pop，top等操作并且可以在O(1)时间内检索出最小元素的堆栈。

* `push(x)` – 将元素x插入栈中
* `pop()` – 移除栈顶元素
* `top()` – 得到栈顶元素
* `getMin()` – 得到栈中最小元素

样例：

```c++
MinStack minStack = new MinStack();
minStack.push(-1);
minStack.push(3);
minStack.push(-4);
minStack.getMin();   --> Returns -4.
minStack.pop();
minStack.top();      --> Returns 3.
minStack.getMin();   --> Returns -1.
```

代码：

使用非严格单调递减的辅助栈：

```c++
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int> main_stack, aux_stack;
    MinStack() {
        aux_stack.push(INT32_MAX);
    }
    
    void push(int x) {
        if (x <= aux_stack.top())
        {
            main_stack.push(x);
            aux_stack.push(x);
        }
        else
        {
            main_stack.push(x);
        }
    }
    
    void pop() {
        if (main_stack.top() == aux_stack.top())
        {
            main_stack.pop();
            aux_stack.pop();
        }
        else
        {
            main_stack.pop();
        }
        
    }
    
    int top() {
        return main_stack.top();
    }
    
    int getMin() {
        return aux_stack.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

### 栈的压入、弹出序列

> 输入两个整数序列，第一个序列表示栈的压入顺> 序，请判断第二个序列是否可能为该栈的弹出顺> 序。
> 
> 假设压入栈的所有数字均不相等。
> 
> 例如序列 1,2,3,4,5 是某栈的压入顺序，序列 > 4,5,3,2,1 是该压栈序列对应的一个弹出序列，> 但 4,3,5,1,2 就不可能是该压栈序列的弹出序> 列。
> 
> 注意：若两个序列长度不等则视为并不是一个栈> 的压入、弹出序列。若两个序列都为空，则视为> 是一个栈的压入、弹出序列。
> 
> ```
> 样例
> 输入：[1,2,3,4,5]
>       [4,5,3,2,1]
> 
> 输出：true
> ```

代码：

用辅助栈来模拟栈的压入和弹出操作即可。

```c++
class Solution {
public:
    bool isPopOrder(vector<int> pushV, vector<int> popV)
    {
        if (pushV.empty() && popV.empty())
            return true;
            
        if (pushV.size() != popV.size())
            return false;
            
        stack<int> aux;
        int j = 0;
        for (int i = 0; i < pushV.size(); ++i)
        {
            aux.push(pushV[i]);
            while (!aux.empty() && aux.top() == popV[j])
            {
                aux.pop();
                ++j;
            }
        }
        if (!aux.empty())
            return false;
        return true;
    }
};
```

### 用两个栈实现队列

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

 
```
示例 1：

输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
示例 2：

输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```

代码：

1. 两个栈模拟。看了看以前写的代码，繁琐复杂。现在一遍就能写出来精简的代码，感慨良多。

    ```c++
    class CQueue {
    public:
        stack<int> s1;
        stack<int> s2;
        CQueue() {

        }
        
        void appendTail(int value) {
            s1.push(value);
        }
        
        int deleteHead() {
            if (s2.empty() && s1.empty()) return -1;
            if (s2.empty())
            {
                while (!s1.empty())
                {
                    s2.push(s1.top());
                    s1.pop();
                }
            }
            int num = s2.top();
            s2.pop();
            return num;
        }
    };

    /**
     * Your CQueue object will be instantiated and called as such:
     * CQueue* obj = new CQueue();
     * obj->appendTail(value);
     * int param_2 = obj->deleteHead();
     */
    ```

### 包含min函数的栈

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

```
示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

代码：

1. 单调栈

    ```c++
    class MinStack {
    public:
        /** initialize your data structure here. */
        stack<int> s1, s2;
        MinStack() {

        }
        
        void push(int x) {
            s1.push(x);
            if (s2.empty() || x <= s2.top())
                s2.push(x);
        }
        
        void pop() {
            if (s1.top() == s2.top())
            {
                s1.pop();
                s2.pop();
            }
            else
                s1.pop();
        }
        
        int top() {
            return s1.top();
        }
        
        int min() {
            return s2.top();
        }
    };

    /**
     * Your MinStack object will be instantiated and called as such:
     * MinStack* obj = new MinStack();
     * obj->push(x);
     * obj->pop();
     * int param_3 = obj->top();
     * int param_4 = obj->min();
     */
    ```

    提问：什么时候需要用单调栈，什么时候需要用单调队列呢？

### 下一个更大元素 I

给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。

请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。

nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。

 
```
示例 1:

输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:
    对于 num1 中的数字 4 ，你无法在第二个数组中找到下一个更大的数字，因此输出 -1 。
    对于 num1 中的数字 1 ，第二个数组中数字1右边的下一个较大数字是 3 。
    对于 num1 中的数字 2 ，第二个数组中没有下一个更大的数字，因此输出 -1 。
示例 2:

输入: nums1 = [2,4], nums2 = [1,2,3,4].
输出: [3,-1]
解释:
    对于 num1 中的数字 2 ，第二个数组中的下一个较大数字是 3 。
    对于 num1 中的数字 4 ，第二个数组中没有下一个更大的数字，因此输出 -1 。
```

代码：

1. 单调栈 + 散列表

    这道题是想做这么一件事情：我想知道某个数组中任意元素右侧比它大的数。所以维护一个单调栈，记录某元素右侧大于此元素，但尽量小的数。同时因为数组中的数不重复，所以我们可以用哈希表来记录答案。

    ```c++
    class Solution {
    public:
        vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
            unordered_map<int, int> m;
            stack<int> s;
            for (int i = nums2.size() - 1; i > -1; --i)  // 倒序遍历
            {
                // 注意这里几个语句的顺序
                while (!s.empty() && s.top() < nums2[i]) s.pop();  // 先将栈顶元素与当前元素比较，找到第一个大于当前元素的
                m[nums2[i]] = s.empty() ? -1 : s.top();  // 将答案记录在哈希表中
                s.push(nums2[i]);  // 最后才将当前元素压栈
            }

            vector<int> ans(nums1.size());
            for (int i = 0; i < nums1.size(); ++i)
                ans[i] = m[nums1[i]];
            return ans;
        }
    };
    ```

### 132 模式

给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。

如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。

 
```
示例 1：

输入：nums = [1,2,3,4]
输出：false
解释：序列中不存在 132 模式的子序列。
示例 2：

输入：nums = [3,1,4,2]
输出：true
解释：序列中有 1 个 132 模式的子序列： [1, 4, 2] 。
示例 3：

输入：nums = [-1,3,2,0]
输出：true
解释：序列中有 3 个 132 模式的的子序列：[-1, 3, 2]、[-1, 3, 0] 和 [-1, 2, 0] 。
```

代码：

1. 单调栈

    ```c++
    class Solution {
    public:
        bool find132pattern(vector<int>& nums) {
            stack<int> s;
            s.push(nums.back());
            int max_k = INT32_MIN;
            for (int i = nums.size() - 2; i > -1; --i)
            {
                if (nums[i] < max_k) return true;
                else
                {
                    while (!s.empty() && s.top() < nums[i])
                    {
                        max_k = s.top();
                        s.pop();
                    }
                }
                s.push(nums[i]);
            }
            return false;
        }
    };
    ```

    我依然不是很理解啥时候应该用单调栈。但是关于这道题，我似乎快要理解了。关键点在于，`max_k`，`s.top()`以及`nums[i]`并不一定是连续的。尤其是`max_k`和`s.top()`，它们似乎利用单调栈跨越了空间。

    弹出去的元素一定在待入栈元素的右侧，且比待入栈的元素小。这道题主要用到了这个性质。这时如果我们找到一个元素比弹出去的元素还小，那么就可以判断 132 模式了。

    如果不考虑弹出的元素，那么单调栈总是能找到某个元素右侧第一个比这个元素大的元素。这道题显然是把弹出去的元素也用到了，不容易想到。

1. 暴力 + 哈希表

1. 单调栈 + 二分

### 栈的压入、弹出序列

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。


```
示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
示例 2：

输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

代码：

1. 模拟 以前写的

    先压入一个数，然后弹出尽量多的数

    ```c++
    class Solution {
    public:
        bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
            stack<int> s;
            int j = 0;
            for (int i = 0; i < pushed.size(); ++i)
            {
                s.push(pushed[i]);
                while (!s.empty() && s.top() == popped[j])
                {
                    s.pop();
                    ++j;
                }
            }
            
            if (j == popped.size())
                return true;
            else
                return false;
        }
    };
    ```

    后来写的，感觉还没以前写得好：

    ```c++
    class Solution {
    public:
        bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
            stack<int> s;
            int p2 = 0;
            for (int i = 0; i < pushed.size(); ++i)
            {
                s.push(pushed[i]);
                while (p2 < popped.size() && !s.empty() && s.top() == popped[p2]) s.pop(), ++p2;
            }
            return s.empty() && p2 == popped.size();
        }
    };
    ```

### 最小栈

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

push(x) —— 将元素 x 推入栈中。
pop() —— 删除栈顶的元素。
top() —— 获取栈顶元素。
getMin() —— 检索栈中的最小元素。
 

```
示例:

输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

代码：

1. 辅助栈

    ```c++
    class MinStack {
    public:
        stack<int> s1, s2;
        MinStack() {

        }
        
        void push(int val) {
            s1.push(val);
            if (s2.empty()) s2.push(val);
            else if (!s2.empty() && val <= s2.top()) s2.push(val);
        }
        
        void pop() {
            if (s1.top() == s2.top()) s2.pop();
            s1.pop();
        }
        
        int top() {
            return s1.top();
        }
        
        int getMin() {
            return s2.top();
        }
    };

    /**
    * Your MinStack object will be instantiated and called as such:
    * MinStack* obj = new MinStack();
    * obj->push(val);
    * obj->pop();
    * int param_3 = obj->top();
    * int param_4 = obj->getMin();
    */
    ```

1. 不使用额外空间，记录差值

    ```py
    class MinStack:
        def __init__(self):
            """
            initialize your data structure here.
            """
            self.stack = []
            self.min_value = -1

        def push(self, x: int) -> None:
            if not self.stack:
                self.stack.append(0)
                self.min_value = x
            else:
                diff = x-self.min_value
                self.stack.append(diff)
                self.min_value = self.min_value if diff > 0 else x

        def pop(self) -> None:
            if self.stack:
                diff = self.stack.pop()
                if diff < 0:
                    top = self.min_value
                    self.min_value = top - diff
                else:
                    top = self.min_value + diff
                return top

        def top(self) -> int:
            return self.min_value if self.stack[-1] < 0 else self.stack[-1] + self.min_value

        def getMin(self) -> int:
            return self.min_value if self.stack else -1
    ```

### 后缀表达式

根据 逆波兰表示法，求该后缀表达式的计算结果。

有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

 

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
 

```
示例 1：

输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
示例 2：

输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
示例 3：

输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

代码：

1. 栈的基本应用

    ```c++
    class Solution {
    public:
        int evalRPN(vector<string>& tokens) {
            stack<int> s;
            int n1, n2;
            for (int i = 0; i < tokens.size(); ++i)
            {
                if (tokens[i] == "+")
                {
                    n1 = s.top();
                    s.pop();
                    n2 = s.top();
                    s.pop();
                    s.push(n1 + n2);
                    continue;
                }
                if (tokens[i] == "-")  // 这里如果写成 tokens[i][0] == '-'，那么需要区分负数与负号
                {
                    n1 = s.top();
                    s.pop();
                    n2 = s.top();
                    s.pop();
                    s.push(n2 - n1);  // 做减法和除法时注意 n2 在前，n1 在后
                    continue;
                }
                if (tokens[i] == "*")  // 因为每一个 if 语句块里都是 continue，所以不用写成 else if
                {
                    n1 = s.top();
                    s.pop();
                    n2 = s.top();
                    s.pop();
                    s.push(n1 * n2);
                    continue;
                }
                if (tokens[i] == "/")
                {
                    n1 = s.top();
                    s.pop();
                    n2 = s.top();
                    s.pop();
                    s.push(n2 / n1);
                    continue;
                }
                s.push(stoi(tokens[i]));
            }
            return s.top();
        }
    };
    ```

### 小行星碰撞

给定一个整数数组 asteroids，表示在同一行的小行星。

对于数组中的每一个元素，其绝对值表示小行星的大小，正负表示小行星的移动方向（正表示向右移动，负表示向左移动）。每一颗小行星以相同的速度移动。

找出碰撞后剩下的所有小行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

 

```
示例 1：

输入：asteroids = [5,10,-5]
输出：[5,10]
解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
示例 2：

输入：asteroids = [8,-8]
输出：[]
解释：8 和 -8 碰撞后，两者都发生爆炸。
示例 3：

输入：asteroids = [10,2,-5]
输出：[10]
解释：2 和 -5 发生碰撞后剩下 -5 。10 和 -5 发生碰撞后剩下 10 。
示例 4：

输入：asteroids = [-2,-1,1,2]
输出：[-2,-1,1,2]
解释：-2 和 -1 向左移动，而 1 和 2 向右移动。 由于移动方向相同的行星不会发生碰撞，所以最终没有行星发生碰撞。 
```

代码：

1. 栈的基本用法

    这种应用题似乎就是为栈设计的，没有什么模板

    ```c++
    class Solution {
    public:
        vector<int> asteroidCollision(vector<int>& asteroids) {
            stack<int> s;
            int pos = 0;
            while (pos < asteroids.size())
            {
                if (s.empty() ||
                    asteroids[pos] > 0 && s.top() < 0 || 
                    asteroids[pos] < 0 && s.top() < 0 ||
                    asteroids[pos] > 0 && s.top() > 0)
                {
                    s.push(asteroids[pos]);
                    ++pos;
                }
                else
                {
                    if (abs(s.top()) > abs(asteroids[pos]))
                    {
                        ++pos;
                    }
                    else if (abs(s.top()) == abs(asteroids[pos]))
                    {
                        s.pop();
                        ++pos;
                    }
                    else
                    {
                        s.pop();
                    }
                }
            }
            vector<int> ans;  // 这里可以优化，因为已经知道栈的大小，所以可以指定数组的大小，从后往前填充，这样就不用再 reverse() 了。当然也可以直接把 vector 当成 stack 来用，使用 pop_back() 代替 pop() 就行了。
            while (!s.empty())
            {
                ans.push_back(s.top());
                s.pop();
            }
            reverse(ans.begin(), ans.end());
            return ans;
        }
    };
    ```

    看了看别人的解答，这道题似乎和单调栈有一定关系？

### 每日温度

请根据每日 气温 列表 temperatures ，重新生成一个列表，要求其对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

 

```
示例 1:

输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
示例 2:

输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]
示例 3:

输入: temperatures = [30,60,90]
输出: [1,1,0]
```

代码：

1. 单调栈

    ```c++
    class Solution {
    public:
        vector<int> dailyTemperatures(vector<int>& temperatures) {
            int n = temperatures.size();
            vector<int> ans(n);
            stack<pair<int, int>> s;  // (pos, val)，其实只存下标就可以，val 可以通过 vector 访问到
            for (int i = n - 1; i > -1; --i)
            {
                while (!s.empty() && s.top().second <= temperatures[i]) s.pop();  // 这里是 <=，不是 <
                ans[i] = s.empty() ? 0 : s.top().first - i;
                s.push(make_pair(i, temperatures[i]));
            }
            return ans;
        }
    };
    ```

    这道题和下一个更大元素很像。

### 直方图最大矩形面积

给定非负整数数组 heights ，数组中的数字用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 

```
示例 1:

输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
示例 2：



输入： heights = [2,4]
输出： 4
```

代码：

1. 两个单调栈

    一个用来找当前柱子左侧小于当前柱子的下标，一个用来找当前柱子右侧小于当前柱子的下标，然后用数组把两个结果存起来。

    这样就可以得到以当前柱子为高，两侧边界柱子为长度的矩形的面积。

    ```c++
    class Solution {
    public:
        int largestRectangleArea(vector<int>& heights) {
            int n = heights.size();
            stack<int> s1, s2;
            vector<int> left(n), right(n);
            for (int i = 0; i < n; ++i)
            {
                while (!s1.empty() && heights[s1.top()] >= heights[i]) s1.pop();
                left[i] = s1.empty() ? -1 : s1.top();
                s1.push(i);
            }
            for (int i = n - 1; i > -1; --i)
            {
                while (!s2.empty() && heights[s2.top()] >= heights[i]) s2.pop();
                right[i] = s2.empty() ? n : s2.top();
                s2.push(i);
            }
            int ans = INT32_MIN;
            int area;
            for (int i = 0; i < n; ++i)
            { 
                area = heights[i] == 0 ? 0 : heights[i] * (right[i] - left[i] - 1);
                ans = max(area, ans);
            }
            return ans;
        }
    };
    ```

    这个代码效率挺低的，只击败了 26%，说明有优化空间。

## 队列
  
### 用栈实现队列

> 请用栈实现一个队列，支持如下四种操作：

> push(x) – 将元素x插到队尾；
> pop() – 将队首的元素弹出，并返回该元素；
> peek() – 返回队首元素；
> empty() – 返回队列是否为空；

```c++
class MyQueue {
public:
    /** Initialize your data structure here. */
    stack<int> s1, s2;
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        s1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if (s2.empty())
        {
            while (!s1.empty())
            {
                s2.push(s1.top());
                s1.pop();
            }
        }
        int val = s2.top();
        s2.pop();
        return val;
    }
    
    /** Get the front element. */
    int peek() {
        if (s2.empty())
        {
            while (!s1.empty())
            {
                s2.push(s1.top());
                s1.pop();
            }
        }
        return s2.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        if (s1.empty() && s2.empty())  // return (s1.empty() && s2.empty());
            return true;
        else
            return false;
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */
```

### 队列的最大值

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

```
示例 1：

输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
示例 2：

输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
```

1. 用一个辅助的单调队列

    ```c++
    class MaxQueue {
    public:
        queue<int> q1;
        deque<int> q2;
        
        MaxQueue() {

        }
        
        int max_value() {
            return q2.empty() ? -1 : q2.front();
        }
        
        void push_back(int value) {
            while (!q2.empty() && q2.back() < value) q2.pop_back();
            q2.push_back(value);
            q1.push(value);
        }
        
        int pop_front() {
            if (!q2.empty() && q2.front() == q1.front()) q2.pop_front();
            int temp = q1.empty() ? -1 : q1.front();
            if (!q1.empty()) q1.pop();
            return temp;
        }
    };

    /**
    * Your MaxQueue object will be instantiated and called as such:
    * MaxQueue* obj = new MaxQueue();
    * int param_1 = obj->max_value();
    * obj->push_back(value);
    * int param_3 = obj->pop_front();
    */
    ```

## 旋转数组的最小数字

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
>
> 输入一个升序的数组的一个旋转，输出旋转数组的最小元素。
> 
> 例如数组 {3,4,5,1,2} 为 {1,2,3,4,5} 的一个旋转，该数组> 的最小值为 1。
> 
> 数组可能包含重复项。
> 
> 注意：数组内所含元素非负，若数组大小为 0，请返回 −1。
> 
> 样例
>
> 输入：nums = [2, 2, 2, 0, 1]
> 
> 输出：0

代码：

1. 线性查找

    ```c++
    class Solution {
    public:
        int findMin(vector<int>& nums) {
            if (nums.empty())
                return -1;
            for (int i = 0; i < nums.size() - 1; ++i)
            {
                if (nums[i+1] < nums[i])
                    return nums[i+1];
            }
        }
    };
    ```

1. 二分查找

    ```c++
    class Solution {
    public:
        int minArray(vector<int>& numbers) {
            int l = 0, r = numbers.size() - 1, m = l + (r - l) / 2;
            while (l < r)
            {
                m = l + (r - l) / 2;
                if (numbers[m] > numbers[r])
                    l = m+1;  // 注意这里的加 1，不加 1 的话会陷入死循环
                else if (numbers[m] < numbers[r])
                    r = m;
                else
                    --r;
            }
            return numbers[l];
        }
    };
    ```

    后来写的：

    ```c++
    class Solution {
    public:
        int minArray(vector<int>& numbers) {
            int left = 0, right = numbers.size() - 1, mid;
            while (left <= right)
            {
                mid = left + (right - left) / 2;
                if (numbers[mid] > numbers[right]) left = mid + 1;
                else if (numbers[mid] < numbers[right]) right = mid;  // 这里不能写成 right = mid - 1;
                else right = right - 1;  // 这里不能写成 right = mid - 1
            }
            return numbers[left];
        }
    };
    ```

    我又有点迷了，为啥注释的那两句必须要那么写？为什么每次比较的是`numbers[right]`，不能比较`numbers[left]`吗？为什么要按找左边界的写法来写二分查找？

### 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
 
```
示例 1：

输入：s = "()"
输出：true
示例 2：

输入：s = "()[]{}"
输出：true
示例 3：

输入：s = "(]"
输出：false
```

代码：

遇到右括号就看看栈顶是不是匹配的左括号。

```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> sta;
        for (char ch: s)
        {
            if (ch == '(' || ch == '[' || ch == '{') sta.push(ch);
            else if (ch == ')' && !sta.empty() && sta.top() == '(') sta.pop();
            else if (ch == ']' && !sta.empty() && sta.top() == '[') sta.pop();
            else if (ch == '}' && !sta.empty() && sta.top() == '{') sta.pop();
            else return false;
        }
        if (sta.empty()) return true;
        return false;
    }
};
```

后来又写的比较复杂，当初是怎么想出来的呢？

```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> sta;
        for (int i = 0; i < s.size(); ++i)
        {
            if (s[i] == '(' || s[i] == '[' || s[i] == '{')
                sta.push(s[i]);
            else
            {
                if (s[i] == ')')
                {
                    if (!sta.empty() && sta.top() == '(') sta.pop();
                    else return false;
                }
                else if (s[i] == ']')
                {
                    if (!sta.empty() && sta.top() == '[') sta.pop();
                    else return false;
                }
                else if (s[i] == '}')
                {
                    if (!sta.empty() && sta.top() == '{') sta.pop();
                    else return false;
                }
            }
        }
        if (!sta.empty()) return false;
        return true;
    }
};
```

1. 后来又写的，还是挺简单的

    ```cpp
    class Solution {
    public:
        bool isValid(string s) {
            if (s.size() % 2 == 1) return false;  // 字符串的长度如果是奇数必无效
            stack<char> stk;
            for (int i = 0; i < s.size(); ++i)
            {
                if (s[i] == '(' || s[i] == '[' || s[i] == '{')
                    stk.push(s[i]);
                else
                {
                    if (stk.empty()) return false;
                    if (s[i] == ')' && stk.top() != '(') return false;
                    else if (s[i] == ']' && stk.top() != '[') return false;
                    else if (s[i] == '}' && stk.top() != '{') return false;
                    stk.pop();
                }
            }
            if (!stk.empty()) return false;
            return true;
        }
    };
    ```

    感觉这个版本的逻辑是最好的了。

## 堆

### 数据流中的中位数

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

```
[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
示例 1：

输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]
示例 2：

输入：
["MedianFinder","addNum","findMedian","addNum","findMedian"]
[[],[2],[],[3],[]]
输出：[null,null,2.00000,null,2.50000]
```

分析：用大根堆存储较小的几个数字，小根堆存储较大的几个数字，每次只需要取两个堆的 top 做运算即可。

代码：

```c++
class Solution {
public:
    priority_queue<int, vector<int>, greater<int>> qmin;
    priority_queue<int> qmax;
    
    void insert(int num){
        if (qmin.size() == qmax.size())
        {
            qmin.push(num);
            qmax.push(qmin.top());
            qmin.pop();
        }
        else
        {
            qmax.push(num);
            qmin.push(qmax.top());
            qmax.pop();
        }
    }

    double getMedian(){
        if (qmin.size() == qmax.size())
            return (qmin.top() + qmax.top()) / 2.0;
        else
            return qmax.top();
        
    }
};
```

注意，往大根堆（左侧）里加数据时，先把数据放到小根堆（右侧），然后再把小根堆的 top 放到大根堆（左侧）。这样才能保证大根堆（左侧）的 top 值小于等于小根堆（右侧）的 top 值。

## 字符串

### 字符串中的第一个唯一字符

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

 
```
示例：

s = "leetcode"
返回 0

s = "loveleetcode"
返回 2
```

提示：你可以假定该字符串只包含小写字母。


代码：

可以用哈希表做，但是数组计数似乎要快上很多。

```c++
class Solution {
public:
    int firstUniqChar(string s) {
        vector<int> m(26);
        for (auto &c: s)
            ++m[c - 'a'];
        for (int i = 0; i < s.size(); ++i)
            if (m[s[i] - 'a'] == 1) return i; 
        return -1;
    }
};
```

### 赎金信

给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)

```
示例 1：

输入：ransomNote = "a", magazine = "b"
输出：false
示例 2：

输入：ransomNote = "aa", magazine = "ab"
输出：false
示例 3：

输入：ransomNote = "aa", magazine = "aab"
输出：true
```

提示：

你可以假设两个字符串均只含有小写字母。

代码：

直接用数组计数就可以了。没啥意思。

```c++
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        vector<int> cnt(26);
        for (char &c: magazine) ++cnt[c-'a'];
        for (char &c: ransomNote)
            if (cnt[c-'a']-- == 0) return false;
        return true;
    }
};
```

### 有效的字母异位词

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

```
示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false
```

说明:
你可以假设字符串只包含小写字母。

进阶:
如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

代码：

1. 数组计数

    ```c++
    class Solution {
    public:
        bool isAnagram(string s, string t) {
            if (s.size() != t.size()) return false;
            vector<int> m(26);
            for (char &c: s)
                ++m[c - 'a'];
            for (char &c: t)
            {
                --m[c - 'a'];
                if (m[c - 'a'] < 0) return false;
            }
            return true;
        }
    };
    ```

    后来又写的：

    ```c++
    class Solution {
    public:
        bool isAnagram(string s, string t) {
            vector<int> cnt(26);
            for (char &c: s) ++cnt[c-'a'];
            for (char &c: t) if (cnt[c-'a']-- == 0) return false;
            for (int &num: cnt) if (num != 0) return false;
            return true;
        }
    };
    ```

    注意这里的一个细节，如果不使用`s.size() == t.size()`比较长度，那么就必须最后再检查一遍`cnt`中的计数是否都为 0。如果不想检测，就必须要判断两个字符串长度是否相等。

这里用数组计数还是比较快的，但是如果用是 unicode 字符的话，就只能用哈希表了。

### 有效的变位词

给定两个字符串 s 和 t ，编写一个函数来判断它们是不是一组变位词（字母异位词）。

注意：若 s 和 t 中每个字符出现的次数都相同且字符顺序不完全相同，则称 s 和 t 互为变位词（字母异位词）。

 
```
示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false
示例 3:

输入: s = "a", t = "a"
输出: false
```

提示:

1 <= s.length, t.length <= 5 * 104
s and t 仅包含小写字母

进阶: 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

分析：

1. 桶计数

    ```cpp
    class Solution {
    public:
        bool isAnagram(string s, string t) {
            int ns = s.size(), nt = t.size();
            if (ns != nt) return false;
            if (s == t) return false;  // 根据题目要求，相同的字符串不算异位词
            int cnt_s[26] = {0};  // 因为定义的是数组，所以需要把第一个元素初始化为零，让编译器自动初始化其他字节
            int cnt_t[26] = {0};
            int ps = 0, pt = 0;
            while (ps < ns)
            {
                ++cnt_s[s[ps++] - 'a'];
                ++cnt_t[t[pt++] - 'a'];
            }
            ps = 0;
            pt = 0;
            while (ps < 26)
            {
                if (cnt_s[ps++] != cnt_t[pt++])
                    return false;
            }
            return true;
        }
    };
    ```

1. 官方解法 排序

    ```cpp
    class Solution {
    public:
        bool isAnagram(string s, string t) {
            if (s.length() != t.length() || s == t) {
                return false;
            }
            sort(s.begin(), s.end());
            sort(t.begin(), t.end());
            return s == t;
        }
    };
    ```

    时间复杂度为`O(nlog n + 2n)`，为什么空间复杂度是`O(log n)`？

1. 官方解法 桶计数

    ```cpp
    class Solution {
    public:
        bool isAnagram(string s, string t) {
            if (s.length() != t.length() || s == t) {
                return false;
            }
            vector<int> table(26, 0);
            for (auto& ch: s) {
                table[ch - 'a']++;
            }
            for (auto& ch: t) {
                table[ch - 'a']--;
                if (table[ch - 'a'] < 0) {  // 这里其实使用了抽屉原理。如果两个字符串长度相同，并且字符串并不是异位词，那么 s2 一定有至少一个字符比 s1 多
                    return false;
                }
            }
            return true;
        }
    };
    ```

### 字符串轮转

字符串轮转。给定两个字符串s1和s2，请编写代码检查s2是否为s1旋转而成（比如，waterbottle是erbottlewat旋转后的字符串）。

```
示例1:

 输入：s1 = "waterbottle", s2 = "erbottlewat"
 输出：True
示例2:

 输入：s1 = "aa", s2 = "aba"
 输出：False
```

代码：

1. 小技巧，拼接轮转后的字符串一定会出现轮转前的字符串

    ```c++
    class Solution {
    public:
        bool isFlipedString(string s1, string s2) {
            if (s1.size() != s2.size()) return false;
            return (s2 + s2).find(s1) != string::npos;
        }
    };
    ```

### 左旋转字符串

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

 
```
示例 1：

输入: s = "abcdefg", k = 2
输出: "cdefgab"
示例 2：

输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
```

1. 用 c++ 的字符串相关库函数

    ```c++
    class Solution {
    public:
        string reverseLeftWords(string s, int n) {
            return s.substr(n) + s.substr(0, n);
        }
    };
    ```

1. 循环替换法

    ```c++
    class Solution {
    public:
        string reverseLeftWords(string s, int n) {
            int period = s.size() - n;
            int start = 0, pos1, pos2;
            int cnt = 0;
            char ch1, ch2;
            while (cnt < s.size())
            {
                pos1 = start;
                ch1 = s[pos1];
                do
                {
                    pos2 = (pos1 + period) % s.size();
                    ch2 = s[pos2];
                    s[pos2] = ch1;
                    ch1 = ch2;
                    pos1 = pos2;
                    ++cnt;
                } while (pos1 != start);
                ++start;
            }
            return s;
        }
    };
    ```

1. 创建个新字符串，进行线性复制

### 第一个只出现一次的字符

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

```
示例 1:

输入：s = "abaccdeff"
输出：'b'
示例 2:

输入：s = "" 
输出：' '
```

代码：

1. 桶计数

    这种题可以用哈希表计数，但是题目规定了范围是小写字母，肯定是用数组计数效率最高。

    ```c++
    class Solution {
    public:
        char firstUniqChar(string s) {
            vector<int> cnt(26);
            for (int i = 0; i < s.size(); ++i)
                ++cnt[s[i]-'a'];

            for (int i = 0; i < s.size(); ++i)
                if (cnt[s[i]-'a'] == 1) return s[i];
            
            return ' ';
        }
    };
    ```

### 单词规律

给定一种规律 pattern 和一个字符串 str ，判断 str 是否遵循相同的规律。

这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 str 中的每个非空单词之间存在着双向连接的对应规律。

```
示例1:

输入: pattern = "abba", str = "dog cat cat dog"
输出: true
示例 2:

输入:pattern = "abba", str = "dog cat cat fish"
输出: false
示例 3:

输入: pattern = "aaaa", str = "dog cat cat dog"
输出: false
示例 4:

输入: pattern = "abba", str = "dog dog dog dog"
输出: false
```

1. 哈希映射和哈希集合（自己写的）

    ```c++
    class Solution {
    public:
        bool wordPattern(string pattern, string s) {
            unordered_map<char, string> m;
            unordered_set<string> appeared;
            int start = 0, end = 0, pos = -1;  // pos 赋值 -1，配合下面的 do while，很巧妙地处理了第一个单词
            string temp;
            for (int i = 0; i < pattern.size(); ++i)
            {
                if (pos >= (int)s.size()) return false;  // 如果 pattern 的长度大于 s 中单词的数量
                start = pos + 1;
                do ++pos; while (pos < s.size() && s[pos] != ' ');
                end = pos - 1;
                temp = s.substr(start, end-start+1);
                if (m.find(pattern[i]) == m.end())
                {
                    if (appeared.find(temp) != appeared.end()) return false;
                    appeared.insert(temp);
                    m[pattern[i]] = temp;
                }
                else
                {
                    if (s.substr(start, end-start+1) != m[pattern[i]]) return false;
                }
            }
            if (pos != s.size()) return false;  // 如果 pattern 的长度小于 s 中单词的数量
            return true;
        }
    };
    ```

1. 双哈希表互相映射

    ```c++
    class Solution {
    public:
        bool wordPattern(string pattern, string s) {
            unordered_map<char, string> m1;
            unordered_map<string, char> m2;
            int start = 0, end = 0, pos = -1;
            char ch;
            string str;
            for (int i = 0; i < pattern.size(); ++i)
            {
                if (pos >= (int)s.size()) return false;  // 这里的 int 不要忘了，不然会出问题
                start = pos + 1;
                do ++pos; while (pos < s.size() && s[pos] != ' ');
                end = pos - 1;
                ch = pattern[i];
                str = s.substr(start, end-start+1);
                if (m1.find(ch) == m1.end()) m1[ch] = str;
                else if (m1[ch] != str) return false;
                if (m2.find(str) == m2.end()) m2[str] = ch;
                else if (m2[str] != ch) return false;
            }
            if (pos != s.size()) return false;
            return true;
        }
    };
    ```

### 字母异位词分组（变位词组）

给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

 
```
示例 1:

输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
示例 2:

输入: strs = [""]
输出: [[""]]
示例 3:

输入: strs = ["a"]
输出: [["a"]]
```

代码：

1. 自己写的，超时了

    ```c++
    class Solution {
    public:
        vector<vector<int>> cnt;

        bool is_ectopic(vector<int> cnt, string &str)
        {
            for (char &c: str)
            {
                --cnt[c-'a'];
                if (cnt[c-'a'] < 0) return false;
            }
            for (int i = 0; i < 26; ++i)
                if (cnt[i] != 0) return false;
            return true;
        }

        vector<vector<string>> groupAnagrams(vector<string>& strs) {
            vector<vector<string>> ans;
            unordered_map<string, int> m;
            for (int i = 0; i < strs.size(); ++i)
            {
                string key_str = " ";
                for (auto &[k, v]: m)
                    if (is_ectopic(cnt[v], strs[i])) key_str = k;
                if (key_str != " ")
                    ans[m[key_str]].push_back(strs[i]);
                else
                {
                    cnt.push_back(vector<int>(26));
                    for (char &c: strs[i])
                        ++cnt.back()[c-'a'];
                    ans.push_back(vector<string>());
                    ans.back().push_back(strs[i]);
                    m[strs[i]] = ans.size() - 1;
                }
            }
            return ans;
        }
    };
    ```

1. 官方给的方法一，将排序后的字符串作为哈希的键

    ```c++
    class Solution {
    public:
        vector<vector<string>> groupAnagrams(vector<string>& strs) {
            unordered_map<string, vector<string>> mp;
            for (string& str: strs) {
                string key = str;
                sort(key.begin(), key.end());
                mp[key].emplace_back(str);
            }
            vector<vector<string>> ans;
            for (auto it = mp.begin(); it != mp.end(); ++it) {
                ans.emplace_back(it->second);
            }
            return ans;
        }
    };
    ```

1. 官方给的解法二，将字符串的计数作为哈希表的键

    ```c++
    class Solution {
    public:
        vector<vector<string>> groupAnagrams(vector<string>& strs) {
            // 自定义对 array<int, 26> 类型的哈希函数
            auto arrayHash = [fn = hash<int>{}] (const array<int, 26>& arr) -> size_t {
                return accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num) {
                    return (acc << 1) ^ fn(num);
                });
            };

            unordered_map<array<int, 26>, vector<string>, decltype(arrayHash)> mp(0, arrayHash);
            for (string& str: strs) {
                array<int, 26> counts{};
                int length = str.length();
                for (int i = 0; i < length; ++i) {
                    counts[str[i] - 'a'] ++;
                }
                mp[counts].emplace_back(str);
            }
            vector<vector<string>> ans;
            for (auto it = mp.begin(); it != mp.end(); ++it) {
                ans.emplace_back(it->second);
            }
            return ans;
        }
    };
    ```

1. 自己乱写的`hasher`和`comper`，只能击败 5%（好歹通过了）

    ```c++
    class Solution {
    public:
        class hasher
        {
            public:
            int operator()(string const &s) const
            {
                int hash = 0;
                for (char c: s) hash += c - 'a';
                return hash;
            }
        };

        class comp
        {
            public:
            bool operator()(string const &s1, string const &s2) const
            {
                vector<int> cnt(26, 0);
                for (char c: s1) ++cnt[c-'a'];
                for (char c: s2) --cnt[c-'a'];
                for (int val: cnt) if (val != 0) return false;
                return true;
            }
        };

        vector<vector<string>> groupAnagrams(vector<string>& strs) {
            unordered_map<string, int, hasher, comp> m;
            vector<vector<string>> ans;
            for (int i = 0; i < strs.size(); ++i)
            {
                if (m.find(strs[i]) != m.end())
                {
                    ans[m[strs[i]]].push_back(strs[i]);
                }
                else
                {
                    ans.push_back({strs[i]});
                    m[strs[i]] = ans.size() - 1;
                }
            }
            return ans;
        }
    };
    ```

### 字符串相乘

给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

```
示例 1:

输入: num1 = "2", num2 = "3"
输出: "6"
示例 2:

输入: num1 = "123", num2 = "456"
输出: "56088"
```

代码：

1. 自己写的，效率还挺高，能打败 90%

    ```c++
    class Solution {
    public:
        string multiply(string num1, string num2) {
            int n1 = num1.size(), n2 = num2.size();
            string ans(n1 * n2 + 1, '0');
            int p1 = n1 - 1, p2 = n2 - 1, p = ans.size() - 1;
            int mul = 0, carry = 0, x, x1, x2;
            int base = 0;
            while (p1 > -1)
            {
                p2 = n2 - 1;
                x1 = num1[p1--] - '0';
                p = ans.size() - 1 - base++;
                while (p2 > -1)
                {
                    x2 = num2[p2--] - '0';
                    mul = x1 * x2;
                    x = ans[p] - '0';
                    ans[p] = (mul + carry + x) % 10 + '0';
                    carry = (mul + carry + x) / 10;
                    --p;
                }
                while (carry)
                {
                    x = ans[p] - '0';
                    ans[p--] = (x + carry) % 10 + '0';
                    carry = (x + carry) / 10;
                }
            }
            p = 0;
            while (p < ans.size() && ans[p] == '0') ++p;
            return p == ans.size() ? "0" : ans.substr(p, ans.size()-p);
        }
    };
    ```

1. 官方给的答案 1（并不简单）

    ```c++
    class Solution {
    public:
        string multiply(string num1, string num2) {
            if (num1 == "0" || num2 == "0") {
                return "0";
            }
            string ans = "0";
            int m = num1.size(), n = num2.size();
            for (int i = n - 1; i >= 0; i--) {
                string curr;
                int add = 0;
                for (int j = n - 1; j > i; j--) {
                    curr.push_back(0);
                }
                int y = num2.at(i) - '0';
                for (int j = m - 1; j >= 0; j--) {
                    int x = num1.at(j) - '0';
                    int product = x * y + add;
                    curr.push_back(product % 10);
                    add = product / 10;
                }
                while (add != 0) {
                    curr.push_back(add % 10);
                    add /= 10;
                }
                reverse(curr.begin(), curr.end());
                for (auto &c : curr) {
                    c += '0';
                }
                ans = addStrings(ans, curr);
            }
            return ans;
        }

        string addStrings(string &num1, string &num2) {
            int i = num1.size() - 1, j = num2.size() - 1, add = 0;
            string ans;
            while (i >= 0 || j >= 0 || add != 0) {
                int x = i >= 0 ? num1.at(i) - '0' : 0;
                int y = j >= 0 ? num2.at(j) - '0' : 0;
                int result = x + y + add;
                ans.push_back(result % 10);
                add = result / 10;
                i--;
                j--;
            }
            reverse(ans.begin(), ans.end());
            for (auto &c: ans) {
                c += '0';
            }
            return ans;
        }
    };
    ```

1. 官方给的答案 2（也不简单）

    ```c++
    class Solution {
    public:
        string multiply(string num1, string num2) {
            if (num1 == "0" || num2 == "0") {
                return "0";
            }
            int m = num1.size(), n = num2.size();
            auto ansArr = vector<int>(m + n);
            for (int i = m - 1; i >= 0; i--) {
                int x = num1.at(i) - '0';
                for (int j = n - 1; j >= 0; j--) {
                    int y = num2.at(j) - '0';
                    ansArr[i + j + 1] += x * y;
                }
            }
            for (int i = m + n - 1; i > 0; i--) {
                ansArr[i - 1] += ansArr[i] / 10;
                ansArr[i] %= 10;
            }
            int index = ansArr[0] == 0 ? 1 : 0;
            string ans;
            while (index < m + n) {
                ans.push_back(ansArr[index]);
                index++;
            }
            for (auto &c: ans) {
                c += '0';
            }
            return ans;
        }
    };
    ```

1. 官方对这道题做了升华，可以用 fft 来做，有网友给出的解：

    ```c++
    class Solution {
    public:
        using CP = complex <double>;
        
        static constexpr int MAX_N = 256 + 5;

        double PI;
        int n, aSz, bSz;
        CP a[MAX_N], b[MAX_N], omg[MAX_N], inv[MAX_N];

        void init() {
            PI = acos(-1);
            for (int i = 0; i < n; ++i) {
                omg[i] = CP(cos(2 * PI * i / n), sin(2 * PI * i / n));
                inv[i] = conj(omg[i]);
            }
        }

        void fft(CP *a, CP *omg) {
            int lim = 0;
            while ((1 << lim) < n) ++lim;
            for (int i = 0; i < n; ++i) {
                int t = 0;
                for (int j = 0; j < lim; ++j) {
                    if((i >> j) & 1) t |= (1 << (lim - j - 1));
                }
                if (i < t) swap(a[i], a[t]);
            }
            for (int l = 2; l <= n; l <<= 1) {
                int m = l / 2;
                for (CP *p = a; p != a + n; p += l) {
                    for (int i = 0; i < m; ++i) {
                        CP t = omg[n / l * i] * p[i + m];
                        p[i + m] = p[i] - t;
                        p[i] += t;
                    }
                }
            }
        }

        string run() {
            n = 1;
            while (n < aSz + bSz) n <<= 1;
            init();
            fft(a, omg);
            fft(b, omg);
            for (int i = 0; i < n; ++i) a[i] *= b[i];
            fft(a, inv);
            int len = aSz + bSz - 1;
            vector <int> ans;
            for (int i = 0; i < len; ++i) {
                ans.push_back(int(round(a[i].real() / n)));
            }
            // 处理进位
            int carry = 0;
            for (int i = ans.size() - 1; i >= 0; --i) {
                ans[i] += carry;
                carry = ans[i] / 10;
                ans[i] %= 10;
            }
            string ret;
            if (carry) {
                ret += to_string(carry);
            }
            for (int i = 0; i < ans.size(); ++i) {
                ret.push_back(ans[i] + '0');
            }
            // 处理前导零
            int zeroPtr = 0;
            while (zeroPtr < ret.size() - 1 && ret[zeroPtr] == '0') ++zeroPtr;
            return ret.substr(zeroPtr, INT_MAX);
        }

        string multiply(string num1, string num2) {
            aSz = num1.size();
            bSz = num2.size();
            for (int i = 0; i < aSz; ++i) a[i].real(num1[i] - '0');
            for (int i = 0; i < bSz; ++i) b[i].real(num2[i] - '0');
            return run();
        }
    };
    ```

### 重复的DNA序列

所有 DNA 都由一系列缩写为 'A'，'C'，'G' 和 'T' 的核苷酸组成，例如："ACGAATTCCG"。在研究 DNA 时，识别 DNA 中的重复序列有时会对研究非常有帮助。

编写一个函数来找出所有目标子串，目标子串的长度为 10，且在 DNA 字符串 s 中出现次数超过一次。

```
示例 1：

输入：s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
输出：["AAAAACCCCC","CCCCCAAAAA"]
示例 2：

输入：s = "AAAAAAAAAAAAA"
输出：["AAAAAAAAAA"]
```

代码：

1. 哈希表统计

    ```c++
    class Solution {
    public:
        vector<string> findRepeatedDnaSequences(string s) {
            vector<string> ans;
            unordered_map<string, int> m;
            int n = s.size();
            string temp;
            for (int i = 0; i <= n - 10; ++i)  // 这个等号必须取到
            {
                temp = s.substr(i, 10);
                m[temp]++;
                if (m[temp] == 2) ans.push_back(temp);  // 数量为 2 时才加入答案，保证只加入一次
            }
            return ans;
        }
    };
    ```

1. 将字符串压缩成整数，减少字符串作为哈希的开销

    ```c++
    class Solution {
    public:
        vector<string> findRepeatedDnaSequences(string s) {
            vector<string> ans;
            if (s.size() < 10) return ans;
            unordered_map<char, int> m({{'A', 0}, {'G', 1}, {'C', 2}, {'T', 3}});
            unordered_map<int, int> cnt;
            int n = s.size();
            int num = 0;
            for (int i = 0; i < 10; ++i)
            {
                num <<= 2;
                num |= m[s[i]];
            }
            ++cnt[num];
            for (int i = 10; i < n; ++i)  // 这里不能取到 n
            {
                num <<= 2;
                num &= (1 << 20) - 1;
                num |= m[s[i]];
                ++cnt[num];
                if (cnt[num] == 2) ans.push_back(s.substr(i-9, 10));  // 这里必须是减 9，因为 i 表示的是字符串末尾字符的索引
            }
            return ans;
        }
    };
    ```

### 表示数值的字符串

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。

数值（按顺序）可以分成以下几个部分：

若干空格
一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
若干空格
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分数值列举如下：

["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
部分非数值列举如下：

["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]
 

```
示例 1：

输入：s = "0"
输出：true
示例 2：

输入：s = "e"
输出：false
示例 3：

输入：s = "."
输出：false
示例 4：

输入：s = "    .1  "
输出：true
```

代码：

1. 自己写的模拟（很麻烦）

    ```c++
    class Solution {
    public:
        bool is_decimal(string &s, int start, int end)
        {
            if (s[start] == '+' || s[start] == '-') ++start;
            if (isdigit(s[start]))
            {
                int p = start;
                while (p <= end && isdigit(s[p])) ++p;
                if (p > end) return false;
                if (s[p] != '.') return false;
                ++p;
                while (p <= end) if (!isdigit(s[p++])) return false;
            }
            else
            {
                if (s[start] != '.') return false;
                int p = start + 1;
                if (p > end) return false;
                while (p <= end) if (!isdigit(s[p++])) return false;
            }
            return true;
        }

        bool is_integer(string &s, int start, int end)
        {
            if (s[start] == '+' || s[start] == '-') ++start;
            int p = start;
            if (p > end) return false;
            while (p <= end) if (!isdigit(s[p++])) return false;
            return true;
        }

        bool isNumber(string s) {
            int start = 0, end = s.size() - 1;
            while (start <= end && s[start] == ' ') ++start;
            while (start <= end && s[end] == ' ') --end;
            if (start > end) return false;
            int pos = start;
            while (pos <= end && (s[pos] != 'e' && s[pos] != 'E')) ++pos;
            if (pos > end) return is_decimal(s, start, end) || is_integer(s, start, end);
            return (is_integer(s, start, pos-1) || is_decimal(s, start, pos-1)) && is_integer(s, pos+1, end);
        }
    };
    ```

1. 有限状态机

    ```c++
    class Solution {
    public:
        enum State {
            STATE_INITIAL,
            STATE_INT_SIGN,
            STATE_INTEGER,
            STATE_POINT,
            STATE_POINT_WITHOUT_INT,
            STATE_FRACTION,
            STATE_EXP,
            STATE_EXP_SIGN,
            STATE_EXP_NUMBER,
            STATE_END
        };

        enum CharType {
            CHAR_NUMBER,
            CHAR_EXP,
            CHAR_POINT,
            CHAR_SIGN,
            CHAR_SPACE,
            CHAR_ILLEGAL
        };

        CharType toCharType(char ch) {
            if (ch >= '0' && ch <= '9') {
                return CHAR_NUMBER;
            } else if (ch == 'e' || ch == 'E') {
                return CHAR_EXP;
            } else if (ch == '.') {
                return CHAR_POINT;
            } else if (ch == '+' || ch == '-') {
                return CHAR_SIGN;
            } else if (ch == ' ') {
                return CHAR_SPACE;
            } else {
                return CHAR_ILLEGAL;
            }
        }

        bool isNumber(string s) {
            unordered_map<State, unordered_map<CharType, State>> transfer{
                {
                    STATE_INITIAL, {
                        {CHAR_SPACE, STATE_INITIAL},
                        {CHAR_NUMBER, STATE_INTEGER},
                        {CHAR_POINT, STATE_POINT_WITHOUT_INT},
                        {CHAR_SIGN, STATE_INT_SIGN}
                    }
                }, {
                    STATE_INT_SIGN, {
                        {CHAR_NUMBER, STATE_INTEGER},
                        {CHAR_POINT, STATE_POINT_WITHOUT_INT}
                    }
                }, {
                    STATE_INTEGER, {
                        {CHAR_NUMBER, STATE_INTEGER},
                        {CHAR_EXP, STATE_EXP},
                        {CHAR_POINT, STATE_POINT},
                        {CHAR_SPACE, STATE_END}
                    }
                }, {
                    STATE_POINT, {
                        {CHAR_NUMBER, STATE_FRACTION},
                        {CHAR_EXP, STATE_EXP},
                        {CHAR_SPACE, STATE_END}
                    }
                }, {
                    STATE_POINT_WITHOUT_INT, {
                        {CHAR_NUMBER, STATE_FRACTION}
                    }
                }, {
                    STATE_FRACTION,
                    {
                        {CHAR_NUMBER, STATE_FRACTION},
                        {CHAR_EXP, STATE_EXP},
                        {CHAR_SPACE, STATE_END}
                    }
                }, {
                    STATE_EXP,
                    {
                        {CHAR_NUMBER, STATE_EXP_NUMBER},
                        {CHAR_SIGN, STATE_EXP_SIGN}
                    }
                }, {
                    STATE_EXP_SIGN, {
                        {CHAR_NUMBER, STATE_EXP_NUMBER}
                    }
                }, {
                    STATE_EXP_NUMBER, {
                        {CHAR_NUMBER, STATE_EXP_NUMBER},
                        {CHAR_SPACE, STATE_END}
                    }
                }, {
                    STATE_END, {
                        {CHAR_SPACE, STATE_END}
                    }
                }
            };

            int len = s.length();
            State st = STATE_INITIAL;

            for (int i = 0; i < len; i++) {
                CharType typ = toCharType(s[i]);
                if (transfer[st].find(typ) == transfer[st].end()) {
                    return false;
                } else {
                    st = transfer[st][typ];
                }
            }
            return st == STATE_INTEGER || st == STATE_POINT || st == STATE_FRACTION || st == STATE_EXP_NUMBER || st == STATE_END;
        }
    };
    ```

### 移除无效的括号

给你一个由 '('、')' 和小写字母组成的字符串 s。

你需要从字符串中删除最少数目的 '(' 或者 ')' （可以删除任意位置的括号)，使得剩下的「括号字符串」有效。

请返回任意一个合法字符串。

有效「括号字符串」应当符合以下 任意一条 要求：

空字符串或只包含小写字母的字符串
可以被写作 AB（A 连接 B）的字符串，其中 A 和 B 都是有效「括号字符串」
可以被写作 (A) 的字符串，其中 A 是一个有效的「括号字符串」


```
示例 1：

输入：s = "lee(t(c)o)de)"
输出："lee(t(c)o)de"
解释："lee(t(co)de)" , "lee(t(c)ode)" 也是一个可行答案。
示例 2：

输入：s = "a)b(c)d"
输出："ab(c)d"
示例 3：

输入：s = "))(("
输出：""
解释：空字符串也是有效的
示例 4：

输入：s = "(a(b(c)d)"
输出："a(b(c)d)"
```

代码：

1. 栈 + 哈希表

    一个字符串中的括号是平衡的，当且仅当

    1. 在任何位置，都有左括号的数量大于等于右括号的数量
    1. 在字符串的末尾，右括号的数量等于左括号的数量

    首先扫描一遍字符串，如果发现某个位置右括号的数量大于左括号的数量，那么就删除右括号。

    然后在字符串的末尾，如果发现有剩余的左括号，那么这些左括号也全都删除。

    ```c++
    class Solution {
    public:
        string minRemoveToMakeValid(string s) {
            stack<int> sta;
            unordered_set<int> d;
            for (int i = 0; i < s.size(); ++i)
            {
                if (s[i] == '(') sta.push(i);
                else if (s[i] == ')')
                {
                    if (sta.empty()) d.insert(i);
                    else sta.pop();
                }
            }
            while (!sta.empty()) d.insert(sta.top()), sta.pop();

            string ans;
            for (int i = 0; i < s.size(); ++i)
            {
                if (d.find(i) == d.end()) ans.push_back(s[i]);
            }
            return ans;
        }
    };
    ```

1. 栈，两遍扫描

    正着扫描一遍，删除`)`；逆着扫描一遍，删除`(`。

1. 栈，一遍扫描

    正着扫描一遍，删除`)`；然后按着栈的 size 删除最左侧的`(`。

### 单词长度的最大乘积

给定一个字符串数组 words，请计算当两个字符串 words[i] 和 words[j] 不包含相同字符时，它们长度的乘积的最大值。假设字符串中只包含英语的小写字母。如果没有不包含相同字符的一对字符串，返回 0。

 

```
示例 1:

输入: words = ["abcw","baz","foo","bar","fxyz","abcdef"]
输出: 16 
解释: 这两个单词为 "abcw", "fxyz"。它们不包含相同字符，且长度的乘积最大。
示例 2:

输入: words = ["a","ab","abc","d","cd","bcd","abcd"]
输出: 4 
解释: 这两个单词为 "ab", "cd"。
示例 3:

输入: words = ["a","aa","aaa","aaaa"]
输出: 0 
解释: 不存在这样的两个单词。
```

代码：

1. 自己写的，数组计数，暴力法

    ```c++
    class Solution {
    public:
        int maxProduct(vector<string>& words) {
            vector<int> cnt(26);
            int ans = 0;
            for (int i = 0; i < words.size(); ++i)
            {
                cnt.assign(26, 0);
                for (int j = 0; j < words[i].size(); ++j)
                    ++cnt[words[i][j] - 'a'];
                for (int j = i + 1; j < words.size(); ++j)
                {
                    int pos = 0;
                    while (pos < words[j].size() && !cnt[words[j][pos] - 'a']) ++pos;
                    if (pos == words[j].size())
                    {
                        ans = max(ans, (int) words[i].size() * (int) words[j].size());
                    }
                }
            }
            return ans;
        }
    };
    ```

1. 因为只需要记录字符串中某个字符出现过没有，而不用记录出现了几次，所以可以用 32 位整数来代替数组计数

### 有效的回文

给定一个字符串 s ，验证 s 是否是 回文串 ，只考虑字母和数字字符，可以忽略字母的大小写。

本题中，将空字符串定义为有效的 回文串 。

```
示例 1:

输入: s = "A man, a plan, a canal: Panama"
输出: true
解释："amanaplanacanalpanama" 是回文串
示例 2:

输入: s = "race a car"
输出: false
解释："raceacar" 不是回文串
```

代码：

1. 双指针向中间逼近

    ```c++
    class Solution {
    public:
        bool isPalindrome(string s) {
            int left = 0, right = s.size() - 1;
            while (left < right)
            {
                if (!isalnum(s[left]))
                {
                    ++left;
                    continue;
                }
                if (!isalnum(s[right]))
                {
                    --right;
                    continue;
                }
                if (tolower(s[left]) != tolower(s[right])) return false;
                ++left, --right;
            }
            return true;
        }
    };
    ```

1. 正序滤掉`s`中的非字母和数字，得到`s1`；逆序滤掉`s`中的非字母和数字，得到`s2`。然后拼接`s1 + s2`和`s2 + s1`，看得到的两个字符串是否相同。

1. 滤掉`s`中的非字母和数字后，得到`s1`，然后比较`s1.reverse()`和`s1`是否相同。

### 验证回文字符串 Ⅱ（最多删除一个字符得到回文）

给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。

 

```
示例 1:

输入: s = "aba"
输出: true
示例 2:

输入: s = "abca"
输出: true
解释: 你可以删除c字符。
示例 3:

输入: s = "abc"
输出: false
```

代码：

1. dfs

    ```c++
    class Solution {
    public:
        bool dfs(string &s, int n, int left, int right)
        {
            while (left < right)
            {
                if (s[left] != s[right])
                {   
                    if (n == 0) return false;
                    return dfs(s, n - 1, left + 1, right) || dfs(s, n - 1, left, right - 1);
                }
                else
                    ++left, --right;
            }
            return true;
        }

        bool validPalindrome(string s) {
            return dfs(s, 1, 0, s.size() - 1);
        }
    };
    ```

    这种形式的递归没写过，不知道它的执行流程是什么样的。

1. 后来又写的

    ```c++
    class Solution {
    public:
        bool dfs(string &s, int n, int left, int right)
        {
            if (n < 0) return false;
            while (left < right)
            {
                if (s[left] != s[right])
                    return dfs(s, n - 1, left + 1, right) || dfs(s, n - 1, left, right - 1);
                ++left;
                --right;
            }
            return true;
        }

        bool validPalindrome(string s) {
            return dfs(s, 1, 0, s.size() - 1);
        }
    };
    ```

### 回文子字符串的个数

给定一个字符串 s ，请计算这个字符串中有多少个回文子字符串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

 

```
示例 1：

输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
示例 2：

输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

代码：

1. 动态规划（只能击败 30%）

    ```c++
    class Solution {
    public:
        int countSubstrings(string s) {
            int n = s.size();
            int ans = 0;
            vector<vector<bool>> dp(n, vector<bool>(n));
            for (int i = 0; i < n; ++i)
            {
                dp[i][i] = true;
                ++ans;
                if (i > 0 && s[i-1] == s[i])
                {
                    dp[i-1][i] = true;
                    ++ans;
                }
            }
            int j;
            for (int len = 3; len <= n; ++len)
            {
                for (int i = 0; i < n - len + 1; ++i)
                {
                    j = i + len - 1;
                    if (dp[i+1][j-1] && s[i] == s[j])
                    {
                        dp[i][j] = true;
                        ++ans;
                    }
                }
            }
            return ans;
        }
    };
    ```

1. 分奇数和偶数的中心拓展

    长度为`n`的字符串会生成`2n - 1`组回文中心`[l_i, r_i]`，其中`l_i = i / 2`, `r_i = l_i + (i % 2)`。

    ```c++
    class Solution {
    public:
        int countSubstrings(string s) {
            int n = s.size(), ans = 0;
            for (int i = 0; i < 2 * n - 1; ++i) {
                int l = i / 2, r = i / 2 + i % 2;
                while (l >= 0 && r < n && s[l] == s[r]) {
                    --l;
                    ++r;
                    ++ans;
                }
            }
            return ans;
        }
    };
    ```

1. Manacher 算法（没看，有空了再看看）

    ```c++
    class Solution {
    public:
        int countSubstrings(string s) {
            int n = s.size();
            string t = "$#";
            for (const char &c: s) {
                t += c;
                t += '#';
            }
            n = t.size();
            t += '!';

            auto f = vector <int> (n);
            int iMax = 0, rMax = 0, ans = 0;
            for (int i = 1; i < n; ++i) {
                // 初始化 f[i]
                f[i] = (i <= rMax) ? min(rMax - i + 1, f[2 * iMax - i]) : 1;
                // 中心拓展
                while (t[i + f[i]] == t[i - f[i]]) ++f[i];
                // 动态维护 iMax 和 rMax
                if (i + f[i] - 1 > rMax) {
                    iMax = i;
                    rMax = i + f[i] - 1;
                }
                // 统计答案, 当前贡献为 (f[i] - 1) / 2 上取整
                ans += (f[i] / 2);
            }

            return ans;
        }
    };
    ```

### 外星语言是否排序

某种外星语也使用英文小写字母，但可能顺序 order 不同。字母表的顺序（order）是一些小写字母的排列。

给定一组用外星语书写的单词 words，以及其字母表的顺序 order，只有当给定的单词在这种外星语中按字典序排列时，返回 true；否则，返回 false。

 

```
示例 1：

输入：words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
输出：true
解释：在该语言的字母表中，'h' 位于 'l' 之前，所以单词序列是按字典序排列的。
示例 2：

输入：words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
输出：false
解释：在该语言的字母表中，'d' 位于 'l' 之后，那么 words[0] > words[1]，因此单词序列不是按字典序排列的。
示例 3：

输入：words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
输出：false
解释：当前三个字符 "app" 匹配时，第二个字符串相对短一些，然后根据词典编纂规则 "apple" > "app"，因为 'l' > '∅'，其中 '∅' 是空白字符，定义为比任何其他字符都小
```

代码：

1. 哈希表 + 比较第一个不同字符的顺序

    ```c++
    class Solution {
    public:
        bool isAlienSorted(vector<string>& words, string order) {
            vector<int> ord(26);
            for (int i = 0; i < 26; ++i)
                ord[order[i]-'a'] = i;
            int idx1, idx2, n1, n2, n;
            for (int i = 1; i < words.size(); ++i)
            {
                n1 = words[i-1].size();
                n2 = words[i].size();
                n = max(n1, n2);
                for (int j = 0; j < n; ++j)
                {
                    idx1 = j < n1 ? ord[words[i-1][j] - 'a'] : -1;
                    idx2 = j < n2 ? ord[words[i][j] - 'a'] : -1;
                    if (idx1 < idx2) break;
                    else if (idx1 > idx2) return false;
                }
            }
            return true;
        }
    };
    ```

    所谓字典序，比较的既不是首字符，也不是单词长度，而是第一个不相同字符的顺序。

### 最小时间差

给定一个 24 小时制（小时:分钟 "HH:MM"）的时间列表，找出列表中任意两个时间的最小时间差并以分钟数表示。

 

```
示例 1：

输入：timePoints = ["23:59","00:00"]
输出：1
示例 2：

输入：timePoints = ["00:00","23:59","00:00"]
输出：0
```

代码：

1. 按分钟数排序后比较前后两个时间

    第一次写的，只能击败 5%

    ```c++
    class Solution {
    public:
        int findMinDifference(vector<string>& timePoints) {
            sort(timePoints.begin(), timePoints.end(), [](string &s1, string &s2) {
                return stoi(s1.substr(0, 2)) * 60 + stoi(s1.substr(3, 2)) < stoi(s2.substr(0, 2)) * 60 + stoi(s2.substr(3, 2));
            });
            int ans = INT32_MAX;
            string *str;
            int n1, n2;
            for (int i = 0; i < timePoints.size(); ++i)
            {
                str = i == 0 ? &timePoints.back() : &timePoints[i-1];
                n1 = stoi(str->substr(0, 2)) * 60 + stoi(str->substr(3, 2));
                str = &timePoints[i];
                n2 = stoi(str->substr(0, 2)) * 60 + stoi(str->substr(3, 2)); 
                ans = min(abs(n1 - n2), ans);
                ans = min(ans, 1440 - ans);
            }
            return ans;
        }
    };
    ```

    还可以用`vector<pair<int, int>>`存储`(mins, idx)`，然后再排序，这样更快一点。不过我没试。

## 前缀和

### 矩阵区域和

给你一个 m x n 的矩阵 mat 和一个整数 k ，请你返回一个矩阵 answer ，其中每个 answer[i][j] 是所有满足下述条件的元素 mat[r][c] 的和： 

i - k <= r <= i + k,
j - k <= c <= j + k 且
(r, c) 在矩阵内。
 
```
示例 1：

输入：mat = [[1,2,3],[4,5,6],[7,8,9]], k = 1
输出：[[12,21,16],[27,45,33],[24,39,28]]
示例 2：

输入：mat = [[1,2,3],[4,5,6],[7,8,9]], k = 2
输出：[[45,45,45],[45,45,45],[45,45,45]]
```

代码：

1. 二维前缀和

    假设有`m`行`n`列的数组`A`，为了方便计算，我们定义`m+1`行，`n+1`列的前缀和数组`pre`，其中第一行和第一列都为零。

    `pre[i][j]`表示从左上角的`A[0][0]`一直到`A[i-1][j-1]`这块正方形内所有数字的和。根据这个定义，我们不难得出两个关系：

    `A[i][j] = pre[i+1][j+1] - pre[i][j+1] - pre[i+1][j] + pre[i][j]`

    `pre[i][j] = pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1] + A[i-1][j-1]`

    这道题无非是多出了个判定边界的步骤。

    ```c++
    class Solution {
    public:
        vector<vector<int>> matrixBlockSum(vector<vector<int>>& mat, int k) {
            int m = mat.size(), n = mat[0].size();
            vector<vector<int>> pre(m+1, vector<int>(n+1));
            for (int i = 1; i <= m; ++i)
                for (int j = 1; j <= n; ++j)
                    pre[i][j] = pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1] + mat[i-1][j-1]; 

            vector<vector<int>> ans(m, vector<int>(n));
            int left, right, top, bottom;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    left = max(j-k, 0);  // 在这里判定边界
                    right = min(j+k, n-1);
                    top = max(i-k, 0);
                    bottom = min(i+k, m-1);
                    ans[i][j] = pre[bottom+1][right+1] - pre[bottom+1][left] - pre[top][right+1] + pre[top][left];
                }
            }
            return ans;
        }
    };
    ```

### 元素和小于等于阈值的正方形的最大边长

给你一个大小为 m x n 的矩阵 mat 和一个整数阈值 threshold。

请你返回元素总和小于或等于阈值的正方形区域的最大边长；如果没有这样的正方形区域，则返回 0 。
 
```
示例 1：

输入：mat = [[1,1,3,2,4,3,2],[1,1,3,2,4,3,2],[1,1,3,2,4,3,2]], threshold = 4
输出：2
解释：总和小于或等于 4 的正方形的最大边长为 2，如图所示。
示例 2：

输入：mat = [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]], threshold = 1
输出：0
示例 3：

输入：mat = [[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]], threshold = 6
输出：3
示例 4：

输入：mat = [[18,70],[61,1],[25,85],[14,40],[11,96],[97,96],[63,45]], threshold = 40184
输出：2
```

代码：

1. 二维前缀和

### 二维区域和检索 - 矩阵不可变

给定一个二维矩阵 matrix，以下类型的多个请求：

计算其子矩形范围内元素的总和，该子矩阵的 左上角 为 (row1, col1) ，右下角 为 (row2, col2) 。
实现 NumMatrix 类：

NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
int sumRegion(int row1, int col1, int row2, int col2) 返回 左上角 (row1, col1) 、右下角 (row2, col2) 所描述的子矩阵的元素 总和 。
 
```
示例 1：

输入: 
["NumMatrix","sumRegion","sumRegion","sumRegion"]
[[[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]],[2,1,4,3],[1,1,2,2],[1,2,2,4]]
输出: 
[null, 8, 11, 12]

解释:
NumMatrix numMatrix = new NumMatrix([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]]);
numMatrix.sumRegion(2, 1, 4, 3); // return 8 (红色矩形框的元素总和)
numMatrix.sumRegion(1, 1, 2, 2); // return 11 (绿色矩形框的元素总和)
numMatrix.sumRegion(1, 2, 2, 4); // return 12 (蓝色矩形框的元素总和)
```

代码：

1. 二维前缀和

    ```c++
    class NumMatrix {
    public:
        vector<vector<int>> pre;
        int m, n;

        NumMatrix(vector<vector<int>>& matrix) {
            m = matrix.size();
            n = matrix[0].size();
            pre.assign(m+1, vector<int>(n+1));
            for (int i = 1; i <= m; ++i)
                for (int j = 1; j <= n; ++j)
                    pre[i][j] = pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1] + matrix[i-1][j-1];
        }
        
        int sumRegion(int row1, int col1, int row2, int col2) {
            return pre[row2+1][col2+1] - pre[row1][col2+1] - pre[row2+1][col1] + pre[row1][col1];
        }
    };

    /**
     * Your NumMatrix object will be instantiated and called as such:
     * NumMatrix* obj = new NumMatrix(matrix);
     * int param_1 = obj->sumRegion(row1,col1,row2,col2);
     */
    ```

1. 一维前缀和也可以做，不过时间复杂度不是O(1)。

### 除自身以外数组的乘积

给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

 
```
示例:

输入: [1,2,3,4]
输出: [24,12,8,6]
 

提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。

说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。

进阶：
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
```

1. 前缀积和后缀积

    ```c++
    class Solution {
    public:
        vector<int> productExceptSelf(vector<int>& nums) {
            int n = nums.size();
            vector<int> left(n), right(n);
            left[0] = nums[0];
            for (int i = 1; i < n; ++i)
                left[i] = left[i-1] * nums[i];
            right[n-1] = nums[n-1];
            for (int i = n - 2; i > -1; --i)
                right[i] = right[i+1] * nums[i];
            vector<int> ans(n);
            ans[0] = right[1];
            ans[n-1] = left[n-2];
            for (int i = 1; i < n - 1; ++i)
                ans[i] = left[i-1] * right[i+1];
            return ans;
        }
    };
    ```

1. 为了节省空间，可以将`ans`作为`left`，然后用一个变量来跟踪`right`，倒序遍历`nums`即可。

    ```c++
    class Solution {
    public:
        vector<int> productExceptSelf(vector<int>& nums) {
            int length = nums.size();
            vector<int> answer(length);

            // answer[i] 表示索引 i 左侧所有元素的乘积
            // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
            answer[0] = 1;
            for (int i = 1; i < length; i++) {
                answer[i] = nums[i - 1] * answer[i - 1];
            }

            // R 为右侧所有元素的乘积
            // 刚开始右边没有元素，所以 R = 1
            int R = 1;
            for (int i = length - 1; i >= 0; i--) {
                // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
                answer[i] = answer[i] * R;
                // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
                R *= nums[i];
            }
            return answer;
        }
    };
    ```

1. 一个网友的挺妙的解法，通过变换下标，只使用一个循环就完成了任务。

    ```c++
    class Solution {
    public:
        vector<int> productExceptSelf(vector<int>& nums) {
            int n=nums.size();
            vector<int>ans(n,1);
            int prefix=1,suffix=1;
            for(int i=0;i<n;i++){
                ans[i]*=prefix;
                ans[n-i-1]*=suffix;
                prefix*=nums[i];
                suffix*=nums[n-i-1];
            }
            return ans;
        }
    };
    ```

### 和为 K 的子数组

给你一个整数数组 nums 和一个整数 k ，请你统计并返回该数组中和为 k 的连续子数组的个数。

 
```
示例 1：

输入：nums = [1,1,1], k = 2
输出：2
示例 2：

输入：nums = [1,2,3], k = 3
输出：2
```

代码：

1. 前缀和（超时）

    注意这道题中数组中元素可以为负值，因此不能用滑动窗口，又因为要求子数组必须连续，所以不能用动态规划。最后只剩下前缀和可以用了。如果遇到需要反复查询的操作，需要用到哈希表。而且在查询时似乎不能用二分查找，因为二分查找要求数组有序。

    ```c++
    class Solution {
    public:
        int subarraySum(vector<int>& nums, int k) {
            int n = nums.size();
            vector<int> left(n);
            left[0] = nums[0];
            for (int i = 1; i < n; ++i)
                left[i] = left[i-1] + nums[i];
            int ans = 0, sum;
            for (int i = 0; i < n; ++i)
            {
                for (int j = i; j < n; ++j)
                {
                    if (i == 0) sum = left[j];
                    else sum = left[j] - left[i-1]; 
                    ans += sum == k ? 1 : 0;
                }
            }
            return ans;
        }
    };
    ```

    后来又写的前缀和 + 双循环，虽然仍然超时，但思路明显清晰了许多：

    ```cpp
    class Solution {
    public:
        int subarraySum(vector<int>& nums, int k) {
            int ans = 0;
            int n = nums.size();
            int presum[n];
            presum[0] = nums[0];
            for (int i = 1; i < n; ++i)
                presum[i] = presum[i-1] + nums[i];

            int sum;
            for (int i = 0; i < n; ++i)
            {
                for (int j = i; j < n; ++j)
                {
                    sum = presum[j] - presum[i] + nums[i];
                    if (sum == k)
                        ++ans;
                }
            }
            return ans;
        }
    };
    ```

1. 前缀和 + 哈希表

    ```c++
    class Solution {
    public:
        int subarraySum(vector<int>& nums, int k) {
            int n = nums.size();
            int ans = 0, pre = 0;
            unordered_map<int, int> m;
            m[0]++;
            for (int i = 0; i < n; ++i)
            {
                pre += nums[i];
                if (m.find(pre - k) != m.end())
                    ans += m[pre - k];
                m[pre]++;
            }
            return ans;
        }
    };
    ```

如果给定的数组为正整数，或非负整数，下面的代码能不能跑通呢？

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int ans = 0;
        int left = 0, right = 0;
        int sum = nums[0];
        int n = nums.size();
        while (left <= right && right < n)
        {
            while (sum < k && right < n)
            {
                ++right;
                if (right < n)
                    sum += nums[right];
            }
            while (sum > k && left <= right)
            {
                sum -= nums[left];
                ++left;
            }
            if (sum == k && right - left + 1 >= 1)
            {
                ++ans;
                ++right;
                if (right < n)
                    sum += nums[right];
            }
        }
        return ans;
    }
};
```

### 0 和 1 个数相同的子数组

给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。

 

```
示例 1:

输入: nums = [0,1]
输出: 2
说明: [0, 1] 是具有相同数量 0 和 1 的最长连续子数组。
示例 2:

输入: nums = [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量 0 和 1 的最长连续子数组。
```

代码：

1. 前缀和 + 哈希表

    ```c++
    class Solution {
    public:
        int findMaxLength(vector<int>& nums) {
            int n = nums.size();
            int pre = 0;
            unordered_map<int, int> m;
            m[0] = -1;
            int ans = 0;
            for (int i = 0; i < n; ++i)
            {
                pre += nums[i] ? 1 : -1;
                if (m.find(pre) != m.end())
                    ans = max(i - m[pre], ans);
                else
                    m[pre] = i;
            }
            return ans;
        }
    };
    ```

    为了充分利用题目中的 0 和 1，其实可以不用把 0 转换成 -1，可以直接用`target = nums.size() / 2;`得到 target 值。当然这其中可能有个奇数、偶数，边界情况的问题，可以后续详细讨论。

    可以参考题目：和为 k 的子数组
    

### 左右两边子数组的和相等

给你一个整数数组 nums ，请计算数组的 中心下标 。

数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。

如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。

如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。

 

```
示例 1：

输入：nums = [1,7,3,6,5,6]
输出：3
解释：
中心下标是 3 。
左侧数之和 sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11 ，
右侧数之和 sum = nums[4] + nums[5] = 5 + 6 = 11 ，二者相等。
示例 2：

输入：nums = [1, 2, 3]
输出：-1
解释：
数组中不存在满足此条件的中心下标。
示例 3：

输入：nums = [2, 1, -1]
输出：0
解释：
中心下标是 0 。
左侧数之和 sum = 0 ，（下标 0 左侧不存在元素），
右侧数之和 sum = nums[1] + nums[2] = 1 + -1 = 0 。
```

代码：

1. 前缀和 + 后缀和，自己写的

    ```c++
    class Solution {
    public:
        int pivotIndex(vector<int>& nums) {
            int n = nums.size();
            vector<int> left(n+2), right(n+2);  // 因为题目要求前面和后面都是 0，所以这里需要用到 n + 2 来做对齐
            left[0] = 0, right[n+1] = 0;
            for (int i = 1; i <= n; ++i) left[i] = left[i-1] + nums[i-1];
            for (int i = n; i > 0; --i) right[i] = right[i+1] + nums[i-1];
            for (int i = 1; i <= n; ++i)
                if (left[i-1] == right[i+1]) return i-1;
            return -1; 
        }
    };
    ```

    这个写法虽然简单，但是我仍不明白为什么时候才需要左右各加一个做对齐。

1. 自己写的

    ```cpp
    class Solution {
    public:
        int pivotIndex(vector<int>& nums) {
            int n = nums.size();
            vector<int> presum_l(nums.size()), presum_r(nums.size());
            presum_l[0] = nums[0];
            presum_r.back() = nums.back();
            for (int i = 1; i < n; ++i)
            {
                presum_l[i] = presum_l[i-1] + nums[i];
                presum_r[n-1-i] = presum_r[n-i] + nums[n-1-i];
            }
            if (presum_r[1] == 0)  // 这里的 if，下面的 for，以及后面的 if，必须符合这个顺序才行，不然不符合题目要求的从左到右返回坐标
                return 0;
            for (int i = 1; i < n-1; ++i)
            {
                if (presum_l[i-1] == presum_r[i+1])
                    return i;
            }
            if (presum_l[n-2] == 0)
                return n-1;
            return -1;
        }
    };
    ```

1. 官方给的答案，更简单一点

    ```c++
    class Solution {
    public:
        int pivotIndex(vector<int> &nums) {
            int total = accumulate(nums.begin(), nums.end(), 0);
            int sum = 0;
            for (int i = 0; i < nums.size(); ++i) {
                if (2 * sum + nums[i] == total) {
                    return i;
                }
                sum += nums[i];
            }
            return -1;
        }
    };
    ```

### 二维子矩阵的和

给定一个二维矩阵 matrix，以下类型的多个请求：

计算其子矩形范围内元素的总和，该子矩阵的左上角为 (row1, col1) ，右下角为 (row2, col2) 。
实现 NumMatrix 类：

NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
int sumRegion(int row1, int col1, int row2, int col2) 返回左上角 (row1, col1) 、右下角 (row2, col2) 的子矩阵的元素总和。
 

```
示例 1：

输入: 
["NumMatrix","sumRegion","sumRegion","sumRegion"]
[[[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]],[2,1,4,3],[1,1,2,2],[1,2,2,4]]
输出: 
[null, 8, 11, 12]

解释:
NumMatrix numMatrix = new NumMatrix([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]]);
numMatrix.sumRegion(2, 1, 4, 3); // return 8 (红色矩形框的元素总和)
numMatrix.sumRegion(1, 1, 2, 2); // return 11 (绿色矩形框的元素总和)
numMatrix.sumRegion(1, 2, 2, 4); // return 12 (蓝色矩形框的元素总和)
```

代码：

1. 二维前缀和

    ```c++
    class NumMatrix {
    public:
        vector<vector<int>> pre;
        NumMatrix(vector<vector<int>>& matrix) {
            int m = matrix.size(), n = matrix[0].size();
            pre.assign(m+1, vector<int>(n+1));
            for (int i = 1; i <= m; ++i)
                for (int j = 1; j <= n; ++j)
                    pre[i][j] = pre[i][j-1] + pre[i-1][j] - pre[i-1][j-1] + matrix[i-1][j-1];
        }
        
        int sumRegion(int row1, int col1, int row2, int col2) {
            return pre[row2+1][col2+1] - pre[row2+1][col1] - pre[row1][col2+1] + pre[row1][col1];
        }
    };

    /**
    * Your NumMatrix object will be instantiated and called as such:
    * NumMatrix* obj = new NumMatrix(matrix);
    * int param_1 = obj->sumRegion(row1,col1,row2,col2);
    */
    ```

1. 一维前缀和

### 在区间范围内统计奇数数目

给你两个非负整数 low 和 high 。请你返回 low 和 high 之间（包括二者）奇数的数目。

 
```
示例 1：

输入：low = 3, high = 7
输出：3
解释：3 到 7 之间奇数数字为 [3,5,7] 。
示例 2：

输入：low = 8, high = 10
输出：1
解释：8 到 10 之间奇数数字为 [9] 。
 

提示：

0 <= low <= high <= 10^9
```

代码：

1. 一开始想的，答案可能和除以二有关

    ```cpp
    class Solution {
    public:
        int countOdds(int low, int high) {
            bool low_odd = low % 2;
            bool high_odd = high % 2;
            if (!low_odd && !high_odd) return (high - low) / 2;
            if (low_odd && high_odd) return (high - low) / 2 + 1;
            if (low_odd && !high_odd) return (high - low) / 2 + 1;
            if (!low_odd && high_odd) return (high - low) / 2 + 1;
            return 0;
        }
    };
    ```

    在分类讨论里面，可以稍微简化一下：

    ```cpp
    class Solution {
    public:
        int countOdds(int low, int high) {
            if (!(low % 2) && !(high % 2))
                return (high - low) / 2;
            return (high - low) / 2 + 1;
        }
    };
    ```

1. 前缀和，官方答案的思路

    ```cpp
    class Solution {
    public:
        int pre(int x) {
            return (x + 1) >> 1;
        }
        
        int countOdds(int low, int high) {
            return pre(high) - pre(low - 1);
        }
    };
    ```

    定义`pre(x)`为`[0, x]`中奇数的数量，然后计算出答案。

## 其它

### 求1+2+…+n

> 求 1+2+…+n，要求不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句 (A?B:C)。
> 
> ```
> 样例
> 输入：10
> 
> 输出：55
> ```

代码：

1. 神奇的做法

    ```c++
    class Solution {
    public:
        int getSum(int n) {
            char a[n][n+1];
            return sizeof(a)>>1;
        }
    };
    ```

1. 常规做法（在 leetcode 上无法通过）

    ```c++
    class Solution {
    public:
        int getSum(int n) {
            static int sum = 0;
            sum += n;
            n && getSum(n-1);
            return sum;
        }
    };
    ```

1. leetcode 版本

    ```c++
    class Solution {
    public:
        int sumNums(int n) {
            n && (n += sumNums(n-1));
            return n;
        }
    };
    ```

### 把数组排成最小的数

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

```
示例 1:

输入: [10,2]
输出: "102"
示例 2:

输入: [3,30,34,5,9]
输出: "3033459"
```

分析：

自定义排序。有时间了看看详细的证明。

代码：

```c++
class Solution {
public:
    string printMinNumber(vector<int>& nums) {
        vector<string> strs;
        for (auto &num: nums)
            strs.emplace_back(to_string(num));
        sort(strs.begin(), strs.end(), [](string &s1, string &s2) {return s1 + s2 < s2 + s1;});
        string res;
        for (auto &str: strs)
            res.append(str);
        return res;
    }
};
```

### 只出现一次的数字

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

```
示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4
```

代码：

1. 异或位运算

    0 与任意数异或得到的是任意数，奇数次异或得到的是自己，偶数次异或得到 0。

    ```c++
    class Solution {
    public:
        int singleNumber(vector<int>& nums) {
            int res = 0;
            for (auto &num: nums)
                res ^= num;
            return res;
        }
    };
    ```

1. 排序

1. 两重循环

1. 哈希表

### 数组中只出现一次的两个数字（数组中数字出现的次数）

一个整型数组里除了两个数字之外，其他的数字都出现了两次。

请写程序找出这两个只出现一次的数字。

你可以假设这两个数字一定存在。

```
样例
输入：[1,2,3,3,4,4]

输出：[1,2]
```

代码：

1. 哈希表

    ```c++
    class Solution {
    public:
        vector<int> findNumsAppearOnce(vector<int>& nums)
        {
            unordered_set<int> s;
            for (auto &num: nums)
            {
                if (s.find(num) == s.end())
                    s.insert(num);
                else
                    s.erase(num);
            }
            
            vector<int> res;
            for (auto &num: s)
                res.push_back(num);
            
            return res;
        }
    };
    ```

1. 分组异或

    ```c++
    class Solution {
    public:
        vector<int> findNumsAppearOnce(vector<int>& nums)
        {
            // 对所有数字异或，结果是两个只出现一次的数字的异或
            int div = 0;
            for (auto &num: nums)
                div ^= num;
            
            // 随便找到异或结果里 1 所在的位
            int dig = 1;
            for (int i = 0; i < 32; ++i)
            {
                if (dig & div) break;
                dig <<= 1;
            }

            // 根据 1 所在的位进行分组，两个只出现一次的数字绝对不会被分到同一组里
            vector<int> res(2, 0);
            for (auto &num: nums)
            {
                if (num & dig)
                    res[0] ^= num;
                else
                    res[1] ^= num;
            }
            return res;
        }
    };
    ```

    后来又写的，稍微简洁一点：

    ```c++
    class Solution {
    public:
        vector<int> singleNumbers(vector<int>& nums) {
            int temp = 0;
            for (int &num: nums) temp ^= num;
            int dig = temp & (-temp);
            int num1 = 0, num2 = 0;
            for (int &num: nums)
            {
                if (num & dig) num1 ^= num;
                else num2 ^= num;
            }
            return {num1, num2};
        }
    };
    ```

### 数组中唯一只出现一次的数字（只出现一次的数字 ）

在一个数组中除了一个数字只出现一次之外，其他数字都出现了三次。

请找出那个只出现一次的数字。

你可以假设满足条件的数字一定存在。

思考题：

如果要求只使用 O(n) 的时间和额外 O(1) 的空间，该怎么做呢？

```
样例
输入：[1,1,1,2,2,2,3,4,4,4]

输出：3
```

代码：

1. 按位统计 1

    按位统计 1 出现的次数。出现 3 次的数字，有 1 的位置出现的次数必定是 3 的倍数，因此必须能被 3 整除；而只出现一次的数字，有 1 的位置被 3 整除，必定余 1。因此我们可以根据整除的结果，把只出现一次的数字重新构造出来。

    ```c++
    class Solution {
    public:
        int findNumberAppearingOnce(vector<int>& nums) {
            int res = 0;
            for (int i = 0; i < 32; ++i)
            {
                int count = 0;
                for (auto &num: nums)
                    count += ((num >> i) & 1);
                res |= (count % 3) << i;
            }
            return res;
        }
    };
    ```

    假如把题目改成一个数字出现了 m 次，其余数字都出现了 n 次，是否只要 m < n，就一定可以用这种方法做？

1. 有限状态自动机（没看，有空了在 leetcode 上看看）

    ```py
    class Solution:
        def singleNumber(self, nums: List[int]) -> int:
            ones, twos = 0, 0
            for num in nums:
                ones = ones ^ num & ~twos
                twos = twos ^ num & ~ones
            return ones
    ```

### 不用加减乘除做加法

写一个函数，求两个整数之和，要求在函数体内不得使用 ＋、－、×、÷ 四则运算符号。

```
样例
输入：num1 = 1 , num2 = 2

输出：3
```

分析：

二进制的两个数做异或运算，则得到的是两个数不进位的加法；两个数做与运算再左移 1，得到的是进位情况。我们对进位情况进行循环左移，直到其为零，说明进位结束。

代码：

1. acwing 上可以通过的代码

    ```c++
    class Solution {
    public:
        int add(int num1, int num2){
            int sum = num1, carry = num2;
            int prev_sum = sum;
            while (carry)
            {   
                sum = sum ^ carry;
                carry = (prev_sum & carry) << 1;
                prev_sum = sum;
            }
            return sum;
        }
    };
    ```

1. leetcode 上可以通过的代码

    ```c++
    class Solution {
    public:
        int add(int a, int b) {
            int sum = 0, carry = 0;
            do
            {
                sum = a ^ b;
                carry = (unsigned) (a & b) << 1;  // 这里必须要加 unsigned，不然会报错
                a = sum;
                b = carry;
            } while (carry);
            return sum;
        }
    };
    ```

### 快速幂（数值的整数次方）

```
示例 1：

输入：x = 2.00000, n = 10
输出：1024.00000
示例 2：

输入：x = 2.10000, n = 3
输出：9.26100
示例 3：

输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

代码：

1. 以前的代码，需要用到`long`来存储`-INT32_MIN`这个数。

    原理是，比如数`3.1^19`，可以拆分成`3.1 ^ 16 * 3.1 ^ 2 * 3.1 ^ 1`。将`3.1`与自己相乘后可以得到`3.1 ^ 2`，将`3.1 ^ 2`再与自己相乘可以得到`3.1 ^ 4`，再然后是`3.1 ^ 8`，`3.1 ^ 16`。此时我们只需要取`1, 2, 16`三个位置，就可以得到`3.1 ^ 19`了。而`1, 2, 16`这三个位置，恰好对应`n`中二进位位是`1`的三个位置。

    ```c++
    class Solution {
    public:
        double myPow(double x, int n) {
            bool minus = false;
            long n_copy = n;
            if (n < 0)
            {
                minus = true;
                n_copy = n;
                n_copy = - n_copy;  // 这一行要和上面一行分开写，不然 -n 可能会溢出
            }

            double res = 1;
            while (n_copy)
            {
                if (n_copy & 1) res *= x;
                x *= x;
                n_copy >>= 1;
            }

            if (minus) return 1 / res;
            return res;
        }
    };
    ```

1. 后来写的代码，单独判断`INT32_MIN`，不需要用到`long`。

    ```c++
    class Solution {
    public:
        double myPow(double x, int n) {
            bool minus = false;
            bool int_min = false;
            if (n < 0) minus = true;
            if (n == INT32_MIN) int_min = true;
            if (minus && !int_min) n = -n;
            if (int_min) n = -(n+1);
            double ans = int_min ? x : 1;
            while (n)
            {
                if (n & 1) ans *= x;
                x *= x;
                n = n >> 1;
            }
            if (minus) ans = 1 / ans;
            return ans;
        }
    };
    ```

### 连续的子数组和

给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：

子数组大小 至少为 2 ，且
子数组元素总和为 k 的倍数。
如果存在，返回 true ；否则，返回 false 。

如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。

示例 1：

```
输入：nums = [23,2,4,6,7], k = 6
输出：true
解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。
```

代码：

前缀和。对每个前缀和求余数，若存在两个前缀和的余数相等，则可以整除 k。

```c++
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> m;  // 用于记录第一次出现某个余数的位置
        m[0] = -1;  // 不要忘了这个特殊的前缀和
        int r = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            r = (r + nums[i]) % k;
            if (m.find(r) == m.end())
                m[r] = i;
            else
                if (i - m[r] >= 2) return true;
        }
        return false;
    }
};
```

### 连续数组

给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。

示例 1:

```
输入: nums = [0,1]
输出: 2
说明: [0, 1] 是具有相同数量0和1的最长连续子数组。
```

示例 2:

```
输入: nums = [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量0和1的最长连续子数组。
```

代码：

前缀和。

```c++
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        unordered_map<int, int> m;
        int count = 0;
        m[0] = -1;
        int max_len = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            if (nums[i] == 0) --count;
            else ++count;
            if (m.find(count) != m.end()) max_len = max(max_len, i - m[count]);
            else m[count] = i;
        }
        return max_len;
    }
};
```

### 2 的幂

给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；否则，返回 false 。

如果存在一个整数 x 使得 n == 2x ，则认为 n 是 2 的幂次方。

 
```
示例 1：

输入：n = 1
输出：true
解释：20 = 1
示例 2：

输入：n = 16
输出：true
解释：24 = 16
```

代码：

1. 自己写的，效率很低

    ```c++
    class Solution {
    public:
        bool isPowerOfTwo(int n) {
            if (n <= 0) return false;
            int count = 0;
            for (int i = 0; i < 31; ++i)
            {
                if (n & 1) ++count;
                if (count == 2) return false;
                n >>= 1;
            }
            return true;
        }
    };
    ```

1. 后来自己又写的，思路类似整数除法里的二倍试探法

    ```cpp
    class Solution {
    public:
        bool isPowerOfTwo(int n) {
            int m = 1;
            while (m <= n)
            {
                if (m == n)
                    return true;
                if (m > INT32_MAX / 2)
                    return false;
                m += m;
            }
            return false;
        }
    };
    ```

1. 官方给出的技巧 1

    ```c++
    class Solution {
    public:
        bool isPowerOfTwo(int n) {
            return n > 0 && (n & (n - 1)) == 0;  // n & (n - 1) 可以移除二进制中最低位的 1
        }
    };
    ```

    需要注意`==`的优先级高于`&`。

1. 官方给出的技巧 2

    ```c++
    class Solution {
    public:
        bool isPowerOfTwo(int n) {
            return n > 0 && (n & -n) == n;  // n & (-n) 可以获取二进制表示的最低位的 1
        }
    };
    ```

### 位1的个数（二进制中1的个数）

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

提示：

* 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
* 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。
 
```
示例 1：

输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

代码：

1. 消低位 1 法。

    ```c++
    class Solution {
    public:
        int hammingWeight(uint32_t n) {
            int count = 0;
            while (n)
            {
                n &= (n-1);
                ++count;
            }
            return count;
        }
    };
    ```

1. 古朴的循环检查法

    ```c++
    class Solution {
    public:
        int hammingWeight(uint32_t n) {
            int ans = 0;
            while (n)
            {
                ans += n & 1;
                n >>= 1;
            }
            return ans;
        }
    };
    ```

### 颠倒二进制位

颠倒给定的 32 位无符号整数的二进制位。

 
提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 2 中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。
 

进阶:
如果多次调用这个函数，你将如何优化你的算法？

 
```
示例 1：

输入: 00000010100101000001111010011100
输出: 00111001011110000010100101000000
解释: 输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
     因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
```

代码：

1. 自己写的循环，效率很低

    ```c++
    class Solution {
    public:
        uint32_t reverseBits(uint32_t n) {
            unsigned int temp = 1;  // 如果用 int 的话，temp 右移会在左侧补 1
            for (int i = 0; i < 31; ++i)
                temp <<= 1;
            int res = 0;
            while (n)
            {
                if (n & 1) res |= temp;
                temp >>= 1;
                n >>= 1;
            }
            return res;
        }
    };
    ```

1. 官方给的位运算分治。类似归并排序的思想。我觉得如果不追求极致效率的话，没必要掌握。

    ```c++
    class Solution {
    private:
        const uint32_t M1 = 0x55555555; // 01010101010101010101010101010101
        const uint32_t M2 = 0x33333333; // 00110011001100110011001100110011
        const uint32_t M4 = 0x0f0f0f0f; // 00001111000011110000111100001111
        const uint32_t M8 = 0x00ff00ff; // 00000000111111110000000011111111

    public:
        uint32_t reverseBits(uint32_t n) {
            n = n >> 1 & M1 | (n & M1) << 1;
            n = n >> 2 & M2 | (n & M2) << 2;
            n = n >> 4 & M4 | (n & M4) << 4;
            n = n >> 8 & M8 | (n & M8) << 8;
            return n >> 16 | n << 16;
        }
    };
    ```

1. 后来又写的仿照双指针的写法

    ```cpp
    class Solution {
    public:
        uint32_t reverseBits(uint32_t n) {
            uint32_t ans;
            uint32_t low, high;
            int pos = 0;
            while (pos < 16)
            {
                low = (n >> pos) & 1;
                high = (n >> (32 - pos - 1)) & 1;
                if (low)
                    n |= 1 << (32 - pos - 1);
                else
                    n &= ~(1 << 32 - pos - 1);
                if (high)
                    n |= 1 << pos;
                else
                    n &= ~(1 << pos);
                ++pos;
            }
            return n;
        }
    };
    ```

### 最小好进制

对于给定的整数 n, 如果n的k（k>=2）进制数的所有数位全为1，则称 k（k>=2）是 n 的一个好进制。

以字符串的形式给出 n, 以字符串的形式返回 n 的最小好进制。

 
```
示例 1：

输入："13"
输出："3"
解释：13 的 3 进制是 111。
示例 2：

输入："4681"
输出："8"
解释：4681 的 8 进制是 11111。
示例 3：

输入："1000000000000000000"
输出："999999999999999999"
解释：1000000000000000000 的 999999999999999999 进制是 11。
```

代码：

1. 数学

    这个太难了，还要从数学角度推公式。闲了再看吧。

    ```c++
    class Solution {
    public:
        string smallestGoodBase(string n) {
            long num = stol(n);
            int max_m = log(num) / log(2);
            for (int m = max_m; m > 1; --m)
            {
                int k = floor(pow(num, 1.0 / m));
                long mul = 1, sum = 1;
                for (int i = 1; i <= m; ++i)
                {
                    mul *= k;
                    sum += mul;
                }
                if (sum == num) return to_string(k);
            }
            return to_string(num - 1);
        }
    };
    ```

### 剪绳子

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

```
示例 1：

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

1. 数学推导

    结论是当绳子长度都为 3，或者最后一段是 2 或 4 时，乘积最大。

    ```c++
    class Solution {
    public:
        int cuttingRope(int n) {
            if (n == 2) return 1;
            if (n == 3) return 2;
            int cnt = n / 3, rem;
            rem = n % 3;
            if (rem == 1) return (int) pow(3, cnt-1) * 4;
            if (rem == 2) return (int) pow(3, cnt) * 2;
            return pow(3, cnt);
        }
    };
    ```

1. 动态规划

    其实这个问题可以划分成最优子问题。将某段绳子切成两段，分别计算这两段的最大乘积，这两个乘积的乘积就是整段绳子切割后的最大乘积。【如何证明呢？】

1. 贪心

### 插入、删除和随机访问都是 O(1) 的容器

设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构：

* `insert(val)`：当元素 val 不存在时返回 true ，并向集合中插入该项，否则返回 false 。
* `remove(val)`：当元素 val 存在时返回 true ，并从集合中移除该项，否则返回 false 。
* `getRandom`：随机返回现有集合中的一项。每个元素应该有 相同的概率 被返回。
 

```
示例 :

输入: inputs = ["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
输出: [null, true, false, true, 2, true, false, 2]
解释:
RandomizedSet randomSet = new RandomizedSet();  // 初始化一个空的集合
randomSet.insert(1); // 向集合中插入 1 ， 返回 true 表示 1 被成功地插入

randomSet.remove(2); // 返回 false，表示集合中不存在 2 

randomSet.insert(2); // 向集合中插入 2 返回 true ，集合现在包含 [1,2] 

randomSet.getRandom(); // getRandom 应随机返回 1 或 2 
  
randomSet.remove(1); // 从集合中移除 1 返回 true 。集合现在包含 [2] 

randomSet.insert(2); // 2 已在集合中，所以返回 false 

randomSet.getRandom(); // 由于 2 是集合中唯一的数字，getRandom 总是返回 2 
```

代码：

1. 数组 + 哈希表

    题目中要求插入和删除都是 O(1) 的时间复杂度，因此一定需要一个哈希表来实现这个功能，不然的话其它数据结构查询需要很多时间。题目中还要求能随机返回一个值，因此必须有数组的结构，因为要求随机访问元素。

    此时问题的难点在于删除数组中的元素时，怎样才能在 O(1) 时间复杂度下完成。方法是交换数组末尾元素与待删除元素，并删除此时数组中最后一个元素；同时更新哈希表中末尾元素对应的索引值，以及删除哈希表中待删除元素对应的键值对。

    ```c++
    class RandomizedSet {
    public:
        /** Initialize your data structure here. */
        unordered_map<int, int> m;
        vector<int> v;
        RandomizedSet() {

        }
        
        /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
        bool insert(int val) {
            if (m.find(val) != m.end()) return false;
            v.push_back(val);
            m[val] = v.size() - 1;
            return true;
        }
        
        /** Removes a value from the set. Returns true if the set contained the specified element. */
        bool remove(int val) {
            if (m.find(val) == m.end()) return false;
            m[v.back()] = m[val];
            swap(v[m[val]], v.back());
            m.erase(val);
            v.pop_back();
            return true;
        }
        
        /** Get a random element from the set. */
        int getRandom() {
            return v[rand() % v.size()];
        }
    };

    /**
     * Your RandomizedSet object will be instantiated and called as such:
     * RandomizedSet* obj = new RandomizedSet();
     * bool param_1 = obj->insert(val);
     * bool param_2 = obj->remove(val);
     * int param_3 = obj->getRandom();
     */
    ```

## 整数，进制，位运算

### 整数除法

给定两个整数 a 和 b ，求它们的除法的商 a/b ，要求不得使用乘号 '*'、除号 '/' 以及求余符号 '%' 。

 

注意：

整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231, 231−1]。本题中，如果除法结果溢出，则返回 231 − 1
 

```
示例 1：

输入：a = 15, b = 2
输出：7
解释：15/2 = truncate(7.5) = 7
示例 2：

输入：a = 7, b = -3
输出：-2
解释：7/-3 = truncate(-2.33333..) = -2
示例 3：

输入：a = 0, b = 1
输出：0
示例 4：

输入：a = 1, b = 1
输出：1
```

代码：

这道题并没有想象中的那么简单，从背答案来看很好理解，但是从独立分析的角度看，涉及到了离散与连续的转换，边界情况处理，整数溢出的处理，动态循环语句的编写，值得深入研究。

1. 用减法做（超时）

    ```c++
    class Solution {
    public:
        int divide(int a, int b) {
            int ans = 0;
            bool neg = (a > 0) ^ (b > 0);
            if (a > 0) a = -a;  // 因为最小的负数的绝对值比最大的正数还大一，所以把正数都转换成负数进行运算
            if (b > 0) b = -b;
            while (a <= b)
            {
                a -= b;
                if (ans == INT32_MAX)  // 在递增之前判断是否溢出
                {
                    if (neg) return INT32_MIN;
                    else return INT32_MAX;
                }
                ++ans;
            }
            if (neg) ans = -ans;
            return ans;
        }
    };
    ```

1. 用乘二法做

    第 1 种方法的关键操作有两个，一个是递增，另一个是比较。我们知道递增是一个线性操作，消耗的时间也是线性的。这两个操作让我们联想到二分查找，如果每次都能以折半的方式缩小搜索范围，那么速度就会快很多了。

    在这道题中，对于`a / b`，假如这两个数都是正数。如果`a`的一半大于`b`，那么整除的结果一定大于`2`；如果`a`的四分之一大于`b`，那么整除的结果一定大于`4`。以此类推。
    
    现在来看这样一个问题：如果`a`的 1/4 大于`b`，但`a`的 1/8 小于`b`，那么整除的结果一定大于`4`小于`8`。有没有办法获取更具体的值呢？目前我只有两种思路，一种是计算`a - 4b`，用这个结果再去除以`b`。另一种是将整个问题看作一个二分查找问题，通过二分法不断逼近`b`的精确位置。

    尝试写一下第一种方法：

    ```c++
    int divide(int a, int b) {
        int temp = b;  // 以 b 为基准长度，探测长度每次增加一倍，探测是否超过 a
        int round_quotient = 1;  // 当前轮得到的商
        int ans = 0;  // 总的结果
        while (a >= b)  // 如果 a >= b，那么结果起码是 1。如果 a < b，那结果必然是 0，不需要再算了
        {
            if ((a >> 1) > temp)  // 如果探测长度不到 a 的一半，那么继续翻倍
            {
                temp += temp;  // 题目里说不能用乘法，因此使用加法翻倍
                round_quotient += round_quotient;  // temp 翻了一倍，说明 a 至少大于 2b；temp 再翻一倍，说明 a 至少大于 4b。因此本轮的商也跟着翻倍。
            }
            else  // 如果探测长度超过 a 的一半，那么就该把已经探测到的地方减掉
            {
                ans += round_quotient;
                a -= temp;
                temp = b;  // 重新从基准开始度量
                round_quotient = 1;
            }
        }
        return ans;
    }
    ```

    上面的代码对大部分正数相除的计算是没问题的。我不清楚它是怎么正确运行的，因为我完全是按照连续量进行的思考，并没有考虑离散量的边界值，整数截断之类的。

    想了想，似乎不能用二分查找法做这道题。我们被赋予的运算是有限的，也就是说要么对`a`除以 2（`a >> 1`），要么对某个数乘 2（`temp += temp`）。假如选择每次都对`a`除以 2，那么会以`a`为度量单位，得到`a/2, a/4, a/8, ...`。这时候会有一个问题，假如我们发现`b < a/4`且`b > a/8`，对`[a/8, a/4]`进行二分，得到的是`3a/16`，这时候我们会发现，我们找不到`3/16`对应`b`的几倍了、`16/3 = 5.333...`，但是题目不允许我们用除法，我们无法得到`5.333`这个数字。如果我们换种思路，“既然我们知道`a`大于`4b`，我们只需要`a - 4b = c`，然后再计算`c / b`就可以了”，可这正是第一种思路。我们无法直接计算`4 * b`，只能不断累加，或者先算`2b = b + b`，再算`4b = 2b + 2b`。一切的迹象都指向第一种方法。

    说到底，整数除法指的应该是`a`和`b`的几倍的关系，第二种想法却演变成了`a`和`2`的关系。

    ```c++
    class Solution {
    public:
        int divide(int a, int b) {
            // 先判断答案的正负，然后将除数和被除数都转换成负数
            bool neg = (a > 0) ^ (b > 0);
            if (a > 0) a = -a;
            if (b > 0) b = -b;

            int ans = 0;
            int times = -1;  // 因为后面有 ans += times; 如果 times 不是负数，那么也有可能溢出
            int b_copy = b;
            while (a <= b)  // 是否要取到等号？没搞清楚
            {
                if (b <= (a >> 1))  // 这里的等号必须取到，不清楚是为啥
                {
                    ans += times;
                    a -= b;
                    times = -1;
                    b = b_copy;
                    continue;  // 如果不加这个，下面会直接计算 2 倍，这样就无法计算 1 位的结果了
                }
                b += b;  // 不能用乘号，所以这里用加来代替 * 2
                times += times;
            }
            if (!neg && ans == INT32_MIN) return INT32_MAX;  // 最小的值为 INT32_MIN / 1 = INT32_MIN，因此不需要担心负数溢出，只需要担心正数溢出就可以了
            if (!neg) return -ans;
            return ans;
        }
    };
    ```

    新的循环模板写法：

    ```cpp
    class Solution {
    public:
        int divide(int a, int b) {
            bool sign = (a > 0) ^ (b > 0);
            if (a > 0) a = -a;
            if (b > 0) b = -b;
            int ans = 0;
            int temp = b;
            int n = -1;
            int HALF_MIN = INT32_MIN / 2;
            while (a <= b)
            {
                // 不断向左试探，直到找到一个最大值
                while (temp >= HALF_MIN && a <= temp + temp)  // 前面加 temp >= HALF_MIN 防止后面的 temp + temp 溢出
                {
                    temp += temp;
                    n += n;
                }
                ans += n;
                a -= temp;
                temp = b;
                n = -1;
            }
            if (!sign && ans == INT32_MIN)  // 唯一的额外异常情况
                return INT32_MAX;
            return sign ? ans : -ans;
        }
    };
    ```

### 二进制加法

给定两个 01 字符串 a 和 b ，请计算它们的和，并以二进制字符串的形式输出。

输入为 非空 字符串且只包含数字 1 和 0。

 

```
示例 1:

输入: a = "11", b = "10"
输出: "101"
示例 2:

输入: a = "1010", b = "1011"
输出: "10101"
```

代码：

1. 模拟

    ```c++
    class Solution {
    public:
        string addBinary(string a, string b) {
            int n1 = a.size(), n2 = b.size();
            string ans(max(n1, n2) + 1, '0');
            int p1 = n1 - 1, p2 = n2 - 1, p = ans.size() - 1;
            bool sum = false;
            bool carry = false;
            bool d1, d2;
            while (p1 > -1 || p2 > -1)
            {
                d1 = p1 > -1 ? a[p1--] == '1' : false;
                d2 = p2 > -1 ? b[p2--] == '1' : false;
                sum = d1 ^ d2 ^ carry;
                carry = (d1 && d2) || (d1 && carry) || (d2 && carry);
                ans[p--] = sum + '0';
            }
            ans[p] = carry ? '1' : '0';
            p = 0;
            while (ans[p] == '0') ++p;
            if (p == ans.size()) return "0";
            return ans.substr(p, ans.npos);
        }
    };
    ```

1. 后来自己又写的，虽然长了点，但是更容易理解了

    ```cpp
    class Solution {
    public:
        string addBinary(string a, string b) {
            string ans;
            int carry = 0, cur;  // 为了避免歧义，我们把当前位，进位，当前位之和，这三个信息分开记录
            int sum;
            int p1 = a.size() - 1, p2 = b.size() - 1;
            while (p1 > -1 && p2 > -1)  // 双指针倒序相加
            {
                sum = (a[p1--] - '0') + (b[p2--] - '0') + carry;
                if (sum >= 2)
                {
                    cur = sum - 2;
                    carry = 1;
                }
                else
                {
                    cur = sum;
                    carry = 0;
                }
                ans.push_back(cur + '0');
            }
            while (p1 > -1)  // 把 while 解耦，每个 while 只负责一个功能
            {
                sum = a[p1--] - '0' + carry;
                if (sum >= 2)
                {
                    cur = sum - 2;
                    carry = 1;
                }
                else
                {
                    cur = sum;
                    carry = 0;
                }
                ans.push_back(cur + '0');
            }
            while (p2 > -1)
            {
                sum = b[p2--] - '0' + carry;
                if (sum >= 2)
                {
                    cur = sum - 2;;
                    carry = 1;
                }
                else
                {
                    cur = sum;
                    carry = 0;
                }
                ans.push_back(cur + '0');
            }
            if (carry)  // 不要忘记最后是否进位
                ans.push_back('1');
            reverse(ans.begin(), ans.end());
            return ans;
        }
    };
    ```

1. 答案使用的模拟

    ```cpp
    class Solution {
    public:
        string addBinary(string a, string b) {
            string ans;
            reverse(a.begin(), a.end());
            reverse(b.begin(), b.end());

            int n = max(a.size(), b.size()), carry = 0;
            for (size_t i = 0; i < n; ++i) {
                carry += i < a.size() ? (a.at(i) == '1') : 0;
                carry += i < b.size() ? (b.at(i) == '1') : 0;
                ans.push_back((carry % 2) ? '1' : '0');
                carry /= 2;
            }

            if (carry) {
                ans.push_back('1');
            }
            reverse(ans.begin(), ans.end());

            return ans;
        }
    };
    ```

    我们主要学一下`/`可以计算进位，`%`可以计算当前位。

### 前 n 个数字二进制中 1 的个数

给定一个非负整数 n ，请计算 0 到 n 之间的每个数字的二进制表示中 1 的个数，并输出一个数组。

 

```
示例 1:

输入: n = 2
输出: [0,1,1]
解释: 
0 --> 0
1 --> 1
2 --> 10
示例 2:

输入: n = 5
输出: [0,1,1,2,1,2]
解释:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
```

代码：

1. Brian Kernighan 算法

    使用`n & (n-1)`得到将最低位 1 置 0 后的数字。

    ```c++
    class Solution {
    public:
        vector<int> countBits(int n) {
            vector<int> ans(n+1);
            int cnt = 0;
            int num;
            for (int i = 0; i <= n; ++i)
            {
                num = i;
                cnt = 0;
                while (num)
                {
                    ++cnt;
                    num &= (num - 1);
                }
                ans[i] = cnt;
            }
            return ans;
        }
    };
    ```

1. 动态规划，把问题分解成高位的 1 和其余位的 1

    ```c++
    class Solution {
    public:
        vector<int> countBits(int n) {
            vector<int> ans(n+1);
            int high;
            for (int i = 1; i <= n; ++i)
            {
                if ((i & (i - 1)) == 0) high = i;  // 若发现 i 是 2 的幂，那么说明到了设定最高位的时候
                ans[i] = ans[i-high] + 1;
            }
            return ans;
        }
    };
    ```

## 纯逻辑模拟，找规律

### 从1到n整数中1出现的次数

> 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
> 
> 例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
> 
>  
> ```
> 示例 1：
> 
> 输入：n = 12
> 输出：5
> ```

分析：

比如数字`14165`，我们令`high = 1416`，`cur = 5`，`low = 0`，此时`1 < cur = 5`，所以`cur`所在的位一定能取到 1，我们假设`cur`所在的位就是 1，那么其它位一共有多少种变化情况呢？从`00001`一直到`14161`，一共有`1417`种变化。

令`high = 141`，`cur = 6`，`low = 5`，此时`cur > 1`，我们固定`cur`位为 1，其它位的变化情况数为从`00010`一直到`14119`，`high`每变化一个，`low`可以从 0 变到 9，因此共有`142 * 10 = 1420`种变化。

令`high = 14`，`cur = 1`，`low = 65`，此时`cur = 1`，所以`cur`只能取到 1，且`low`所在的位所取数字不能大于`low`的当前值。因此当`high`所在的位小于`high`当前值时，`low`可以取`00`到`99`；而当`high`所在位的值恰好为`high`时，`high`每变化一个，`low`所在的位的数值可以从`00`变到`low`的当前值，即`low + 1`种变化。因此一共有`14 * 100 + (65 + 1)`种变化。

由此可总结出规律：

`cur = 0`时，出现 1 的次数为`high * digit`；`cur = 1`时，出现 1 的次数为`high * digit + (low + 1)`；`cur > 1`时，出现 1 的次数为`(high + 1) * digit`。由此可模拟写出代码。

代码：

1. 思路 1，按指针讨论

    ```c++
    class Solution {
    public:
        int countDigitOne(int n) {
            unsigned long low = 0, high = n / 10, cur = n % 10;
            unsigned long digit = 1;
            unsigned long res = 0;
            while (high != 0 || cur != 0)  // 退出条件为当前位和高位都为 0，说明已经出界
            {
                if (cur == 0)
                    res += high * digit;
                else if (cur == 1)
                    res += high * digit + low + 1;
                else
                    res += (high + 1) * digit;
                digit *= 10;
                low = n % digit;
                high = n / (digit * 10);
                cur = n % (digit * 10) / digit; 
            }
            return res;
        }
    };
    ```

1. 思路 2，从低位到高位讨论（官方答案）

    比如要统计百位上 1 的个数，首先拿`n`整除`1000`，每个`1000`都会有`100 ~ 199`这 100 个数，这样就有`n / 1000 * 199`个 1 了。然后拿`n`取`1000`取模，设取模结果为`mod`，若`mod < 100`，那么百位上不会有 1；若`100 <= mod < 200`，那么百位会出现`mod - 100 + 1`个 1；如果`mod >= 200`，那么百位上会有完整的`100`个 1。此时百位上最少有 0 个 1，最多有 100 个 1，因此可以用`min`和`max`做限制，就不用再分类讨论了。

    然后对个位、十位、百位等分别做这样的统计就好了。

    ```c++
    class Solution {
    public:
        int countDigitOne(int n) {
            // mulk 表示 10^k
            // 在下面的代码中，可以发现 k 并没有被直接使用到（都是使用 10^k）
            // 但为了让代码看起来更加直观，这里保留了 k
            long long mulk = 1;
            int ans = 0;
            for (int k = 0; n >= mulk; ++k) {
                ans += (n / (mulk * 10)) * mulk + min(max(n % (mulk * 10) - mulk + 1, 0LL), mulk);
                mulk *= 10;
            }
            return ans;
        }
    };
    ```



### Z 字形变换

将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

P   A   H   N
A P L S I I G
Y   I   R
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

请你实现这个将字符串进行指定行数变换的函数：

string convert(string s, int numRows);
 
```
示例 1：

输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"
示例 2：
输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
解释：
P     I    N
A   L S  I G
Y A   H R
P     I
示例 3：

输入：s = "A", numRows = 1
输出："A"
```

代码：

1. 自己写的，只能击败 5%：

    ```c++
    class Solution {
    public:
        string convert(string s, int numRows) {
            if (numRows == 1) return s;
            int period;
            if (s.size() == numRows) period = 0;
            else period = s.size() / (2 * numRows - 2);
            int cols = period * (numRows - 1);
            int chars_remain = (int)s.size() - period * (2 * numRows - 2);
            if (chars_remain > numRows)
                cols += chars_remain - numRows + 1;
            else if (chars_remain > 0)
                cols++;
            
            vector<vector<char>> board(numRows, vector<char>(cols, ' '));
            bool down = true;
            int i = -1, j = 0;
            for (int pos = 0; pos < s.size(); ++pos)
            {
                if (down)
                {
                    i += 1;
                    board[i][j] = s[pos];
                    if (i == numRows - 1) down = false;
                }
                else
                {
                    --i;
                    ++j;
                    board[i][j] = s[pos];
                    if (i == 0) down = true;
                }
            }

            string ans;
            for (int i = 0; i < board.size(); ++i)
            {
                for (int j = 0; j < board[0].size(); ++j)
                {
                    if (board[i][j] != ' ')
                        ans.push_back(board[i][j]);
                }
            }
            return ans;
        }
    };
    ```

1. 官方给出的解法一，跟我的解法区别是把内层的`vector<char>`替换成了`string`，节省不少空间。

    ```c++
    class Solution {
    public:
        string convert(string s, int numRows) {

            if (numRows == 1) return s;

            vector<string> rows(min(numRows, int(s.size())));
            int curRow = 0;
            bool goingDown = false;

            for (char c : s) {
                rows[curRow] += c;
                if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
                curRow += goingDown ? 1 : -1;
            }

            string ret;
            for (string row : rows) ret += row;
            return ret;
        }
    };
    ```



### 数字序列中某一位的数字

> 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
> 
> 请写一个函数，求任意第n位对应的数字。
> 
> ```
> 示例 1：
> 
> 输入：n = 3
> 输出：3
> ```

分析：

找规律模拟。如果`n`在 0 ~ 9 之间，那么直接返回即可。如果它在 10 ~ 99 之间，那么`n`的前 10 个数存储的是 0 ~ 9，`n - 10`每两位存储一个数字，我们拿`(n - 10) / 2 + 10`即可得到存储的是哪一个数字，`(n - 10) % 2`即可得到存储的是这个数字的哪一位（从左往右数）。

由此可找出规律：

```
0 ~ 9:  lower_bound = 0, upper_bound = 9, digit = 1, count = 1 * (9 - 0 + 1) = 10
10 ~ 99:  lower_bound = 10, upper_bound = 99, digit = 2, count = 2 * (99 - 10 + 1) = 180
100 ~ 999:  lower_bound = 100, upper_bound = 999, digit = 3, count = 3 * (999 - 100 + 1) = 1800
...

num = lower_bound + (n - count) / digit;
dig = (n - count) % digit;
```

即取`num`的第`dig`位。

代码：

1. 自己写的

    ```c++
    class Solution {
    public:
        int findNthDigit(int n) {
            if (n < 10)
                return n;
            unsigned long lower_bound = 10, upper_bound = 99, digit = 2;
            unsigned long count = 10;
            while (n > count + digit * (upper_bound - lower_bound + 1))
            {
                count += digit * (upper_bound - lower_bound + 1);
                ++digit;
                lower_bound = upper_bound + 1;
                upper_bound = upper_bound * 10 + 9;
            }
            
            unsigned long num = lower_bound + (n - count) / digit;
            unsigned long dig = (n - count) % digit;
            return to_string(num)[dig] - '0';
        }
    };
    ```

1. 后来又写的

    ```c++
    class Solution {
    public:
        int findNthDigit(int n) {
            int digits = 1;
            int num_start = 1;
            long start = 1, end = 9;
            while (n > end)
            {
                num_start *= 10;
                ++digits;
                start = end + 1;
                end += (long) digits * 9 * num_start;  // 这里必须做类型转换，不然会溢出。即使 int 与 int 相乘也会溢出
            }
            int num = (n - start) / digits + num_start;
            int digit = (n - start) % digits;
            return to_string(num)[digit] - '0';
        }
    };
    ```

### 扑克牌的顺子

从扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这 5 张牌是不是连续的。

2∼10 为数字本身，A 为 1，J 为 11，Q 为 12，K 为 13，大小王可以看做任意数字。

为了方便，大小王均以 0 来表示，并且假设这副牌中大小王均有两张。

```
样例1
输入：[8,9,10,11,12]

输出：true
```

```
样例2
输入：[0,8,9,11,12]

输出：true
```

代码：

1. 对乱序的输入先排序

    ```c++
    class Solution {
    public:
        bool isContinuous( vector<int> numbers) {
            if (numbers.empty())
                return false;
                
            sort(numbers.begin(), numbers.end());
            
            int zero_count = 0;
            while (numbers[zero_count] == 0) ++zero_count;
            
            int cur = numbers[zero_count];
            for (int i = zero_count + 1; i < numbers.size(); ++i)
            {
                if (numbers[i] != cur + 1)
                {
                    if (zero_count == 0) return false;
                    else
                    {
                        --zero_count;
                        --i;  // 两个数字间的距离可能大于 2，比如 1, 4，所以需要重新对当前数字评估
                    }
                }
                ++cur;
            }
            return true;
        }
    };
    ```

1. 使用优先队列排序

    ```c++
    class Solution {
    public:
        bool isStraight(vector<int>& nums) {
            if (nums.empty())
                return false;
                
            priority_queue<int> q;
            int zero_count = 0;
            for (auto &num: nums)
            {
                if (num == 0) ++zero_count;
                else q.push(num);
            }
            
            int end = q.top();
            q.pop();
            while (!q.empty())
            {
                if (q.top() == end - 1)
                {
                    q.pop();
                    --end;
                }
                else
                {
                    if (zero_count-- > 0)
                        q.push(end - 1);
                    else
                        return false;
                }
            }
            
            return true;
        }
    };
    ```

1. 哈希集合和最值

    只要满足：

    1. 数组中没有重复数字
    1. 除 0 之外的最大最小值之差小于 5

    就一定可以组成顺子。不知道这两点结论是怎么推出来的，有空了推推看看。

    ```c++
    class Solution {
    public:
        bool isStraight(vector<int>& nums) {
            int vmin = INT32_MAX, vmax = INT32_MIN;
            vector<bool> cnt(14);
            for (int i = 0; i < nums.size(); ++i)
            {
                if (nums[i] == 0) continue;
                if (cnt[nums[i]]) return false;
                else cnt[nums[i]] = true;
                vmin = min(vmin, nums[i]);
                vmax = max(vmax, nums[i]);
            }
            return vmax - vmin < 5;
        }
    };
    ```

1. 排序

    ```c++
    class Solution {
    public:
        bool isStraight(vector<int>& nums) {
            sort(nums.begin(), nums.end());
            vector<bool> cnt(14);
            int zero_cnt = 0;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (nums[i] == 0)
                {
                    ++zero_cnt;
                    continue;
                }
                if (cnt[nums[i]]) return false;
                else cnt[nums[i]] = true;
            }
            return nums[4] - nums[zero_cnt] < 5;
        }
    };
    ```

### 圆圈中最后剩下的数字

0,1,…,n−1 这 n 个数字 (n>0) 排成一个圆圈，从数字 0 开始每次从这个圆圈里删除第 m 个数字。

求出这个圆圈里剩下的最后一个数字。

```
样例
输入：n=5 , m=3

输出：3
```

代码：

1. 递归

    似乎是用两种坐标表示同一个位置，然后做一个下标变换。具体是怎么实现的仍不是很清楚。另外这道题在用递归实现时，并不是简单的直接实现，因为问题发生了改变。这时候该怎么用递归呢？

    ```c++
    class Solution {
    public:
        int lastRemaining(int n, int m){
            if (n == 1) return 0;
            return (lastRemaining(n-1, m) + m) % n;
        }
    };
    ```

1. 动态规划 / 迭代

    ```c++
    class Solution {
    public:
        int lastRemaining(int n, int m) {
            int f = 0;
            for (int i = 2; i != n + 1; ++i) {
                f = (m + f) % i;
            }
            return f;
        }
    };
    ```

### 找出游戏的获胜者

共有 n 名小伙伴一起做游戏。小伙伴们围成一圈，按 顺时针顺序 从 1 到 n 编号。确切地说，从第 i 名小伙伴顺时针移动一位会到达第 (i+1) 名小伙伴的位置，其中 1 <= i < n ，从第 n 名小伙伴顺时针移动一位会回到第 1 名小伙伴的位置。

游戏遵循如下规则：

从第 1 名小伙伴所在位置 开始 。
沿着顺时针方向数 k 名小伙伴，计数时需要 包含 起始时的那位小伙伴。逐个绕圈进行计数，一些小伙伴可能会被数过不止一次。
你数到的最后一名小伙伴需要离开圈子，并视作输掉游戏。
如果圈子中仍然有不止一名小伙伴，从刚刚输掉的小伙伴的 顺时针下一位 小伙伴 开始，回到步骤 2 继续执行。
否则，圈子中最后一名小伙伴赢得游戏。
给你参与游戏的小伙伴总数 n ，和一个整数 k ，返回游戏的获胜者。

```
示例 1：


输入：n = 5, k = 2
输出：3
解释：游戏运行步骤如下：
1) 从小伙伴 1 开始。
2) 顺时针数 2 名小伙伴，也就是小伙伴 1 和 2 。
3) 小伙伴 2 离开圈子。下一次从小伙伴 3 开始。
4) 顺时针数 2 名小伙伴，也就是小伙伴 3 和 4 。
5) 小伙伴 4 离开圈子。下一次从小伙伴 5 开始。
6) 顺时针数 2 名小伙伴，也就是小伙伴 5 和 1 。
7) 小伙伴 1 离开圈子。下一次从小伙伴 3 开始。
8) 顺时针数 2 名小伙伴，也就是小伙伴 3 和 5 。
9) 小伙伴 5 离开圈子。只剩下小伙伴 3 。所以小伙伴 3 是游戏的获胜者。
示例 2：

输入：n = 6, k = 5
输出：1
解释：小伙伴离开圈子的顺序：5、4、6、2、3 。小伙伴 1 是游戏的获胜者。
```

代码：

1. 递归

    ```c++
    class Solution {
    public:
        int findTheWinner(int n, int k) {
            if (n == 1) return 1;
            return (findTheWinner(n-1, k) + k - 1) % n + 1;
        }
    };
    ```

1. 迭代

    ```c++
    class Solution {
        public int findTheWinner(int n, int k) {
            int luckier = 1;
            for(int i = 2; i <= n; i++) {
                luckier = ((luckier + k - 1) % i) + 1;
            }
            return luckier;
        }
    }
    ```

### 股票的最大利润（买卖股票的最佳时机）

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖 一次 该股票可能获得的利润是多少？

例如一只股票在某些时间节点的价格为 [9,11,8,5,7,12,16,14]。

如果我们能在价格为 5 的时候买入并在价格为 16 时卖出，则能收获最大的利润 11。

```
样例
输入：[9, 11, 8, 5, 7, 12, 16, 14]

输出：11
```

分析：找规律。如果没有发现一个比当前最低点更低的点，那么就扩展搜索更大的利润；如果发现一个比当前最低点更低的点，此时才有机会拿到更大的利润，因此更新最低点，继续扩展搜索更大的利润。

另外一种解释，也挺好的：假如计划在第 i 天卖出股票，那么最大利润的差值一定是在[0, i-1] 之间选最低点买入；所以遍历数组，依次求每个卖出时机的的最大差值，再从中取最大值。

还有一种解释：如果我们发现有一天股票亏了，那么即使我们在当天买入，最坏情况也只是 0，不会亏。所以不如从零再来，更何况之前的最大利润都已经记录过了。

代码：

1. 比较简洁的标准答案

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices) {
            int res = 0;
            int low = prices[0];
            for (auto &price: prices)
            {
                res = max(res, price - low);
                if (price < low) low = price;
            }
            return res;
        }
    };
    ```

    自己写出来的：

    ```cpp
    class Solution {
    public:
        int maxProfit(vector<int>& prices) {
            int ans = 0;
            int start = prices[0];
            for (int i = 1; i < prices.size(); ++i)
            {
                if (prices[i] - start < 0)
                    start = prices[i];
                else
                    ans = max(ans, prices[i] - start);
            }
            return ans;
        }
    };
    ```

1. 后来又写的，虽然思路一样，但是长了不少

    ```c++
    class Solution {
    public:
        int maxProfit(vector<int>& prices) {
            int buy_price = prices[0];
            int profit = 0, ans = 0;
            for (int i = 0; i < prices.size(); ++i)
            {
                if (prices[i] <= buy_price)
                {
                    profit = 0;
                    buy_price = prices[i];
                    continue;
                }
                profit = prices[i] - buy_price;
                ans = max(profit, ans);
            }
            return ans;
        }
    };
    ```

    问题：怎么才能确定循环中语句的顺序，使得每次都写出来最简洁的语句呢？

### 最大子序和

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```
示例 1：

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
示例 2：

输入：nums = [1]
输出：1
示例 3：

输入：nums = [0]
输出：0
示例 4：

输入：nums = [-1]
输出：-1
示例 5：

输入：nums = [-100000]
输出：-100000
```

代码：

1. 动态规划

    ```c++
    class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            int pre = 0, maxAns = nums[0];
            for (const auto &x: nums) {
                pre = max(pre + x, x);
                maxAns = max(maxAns, pre);
            }
            return maxAns;
        }
    };
    ```

    我依然没想明白问题是如何转化到这一步的。

1. 分治（这个似乎和线段树的操作类似，官方给出的解挺复杂的）

    ```c++
    class Solution {
    public:
        struct Status {
            int lSum, rSum, mSum, iSum;
        };

        Status pushUp(Status l, Status r) {
            int iSum = l.iSum + r.iSum;
            int lSum = max(l.lSum, l.iSum + r.lSum);
            int rSum = max(r.rSum, r.iSum + l.rSum);
            int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
            return (Status) {lSum, rSum, mSum, iSum};
        };

        Status get(vector<int> &a, int l, int r) {
            if (l == r) {
                return (Status) {a[l], a[l], a[l], a[l]};
            }
            int m = (l + r) >> 1;
            Status lSub = get(a, l, m);
            Status rSub = get(a, m + 1, r);
            return pushUp(lSub, rSub);
        }

        int maxSubArray(vector<int>& nums) {
            return get(nums, 0, nums.size() - 1).mSum;
        }
    };
    ```

1. 贪心

    ```c++
    class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            int max_sum = INT32_MIN;
            int sum = 0;
            for (int &num: nums)
            {
                sum += num;
                max_sum = max(max_sum, sum);
                if (sum < 0)
                {
                    sum = 0;
                }
            }
            return max_sum;
        }
    };
    ```

### 构建乘积数组

给定一个数组A[0, 1, …, n-1]，请构建一个数组B[0, 1, …, n-1]，其中B中的元素B[i]=A[0]×A[1]×… ×A[i-1]×A[i+1]×…×A[n-1]。

不能使用除法。

```
样例
输入：[1, 2, 3, 4, 5]

输出：[120, 60, 40, 30, 24]
```

思考题：

能不能只使用常数空间？（除了输出的数组之外）

代码：

1. 不使用常数空间，使用两个数组维护`i`两侧的乘积

    ```c++
    class Solution {
    public:
        vector<int> multiply(const vector<int>& A) {
            if (A.empty())
                return vector<int> ();
            if (A.size() == 1)
                return vector<int>({1});
                
            vector<int> res(A.size()), left(A.size()), right(A.size());
            
            left[0] = A[0];
            for (int i = 1; i < A.size(); ++i)
                left[i] = left[i-1] * A[i];
            right.back() = A.back();
            for (int i = A.size() - 2; i > -1; --i)
                right[i] = right[i+1] * A[i];
            
            res[0] = right[1];
            res[A.size()-1] = left[A.size()-2];
            for (int i = 1; i < A.size()-1; ++i)
                res[i] = left[i-1] * right[i+1];
            
            return res;
        }
    };
    ```

1. 如果将右侧乘积结果保存在`res`中，左侧乘积结果保存在单个变量中，即可做到使用常数空间。

    ```c++
    class Solution {
    public:
        vector<int> multiply(const vector<int>& A) {
            if (A.empty())
                return vector<int> ();
            if (A.size() == 1)
                return vector<int>({1});
                
            vector<int> res(A.size());
            int left = A[0];
            
            res.back() = A.back();
            for (int i = A.size() - 2; i > -1; --i)
                res[i] = res[i+1] * A[i];
            
            res.front() = res[1];
            for (int i = 1; i < A.size()-1; ++i)
            {
                res[i] = left * res[i+1];
                left *= A[i];
            }
            res.back() = left;
            
            return res;
        }
    };
    ```

    自己写的：

    ```c++
    class Solution {
    public:
        vector<int> constructArr(vector<int>& a) {
            int n = a.size();
            if (n == 0 || n == 1) return a;
            vector<int> ans(n);
            vector<int> right(n);
            int left = 1;  // 这里必须为 1
            right[n-1] = a[n-1];
            for (int i = n-2; i > -1; --i) right[i] = right[i+1] * a[i];
            ans[0] = right[1];
            for (int i = 0; i < n - 1; ++i)
            {
                ans[i] = left * right[i+1];
                left *= a[i];  // left 必须放在后面
            }
            ans[n-1] = left;  // ans[n-1] 的赋值必须放到最后
            return ans;
        }
    };
    ```

### 把字符串转换成整数

请你写一个函数 StrToInt，实现把字符串转换成整数这个功能。

当然，不能使用 atoi 或者其他类似的库函数。

样例

```
输入："123"

输出：123
```

注意:

你的函数应满足下列条件：

1. 忽略所有行首空格，找到第一个非空格字符，可以是 ‘+/−’ 表示是正数或者负数，紧随其后找到最长的一串连续数字，将其解析成一个整数；
1. 整数后可能有任意非数字字符，请将其忽略；
1. 如果整数长度为 0，则返回 0；
1. 如果整数大于 INT_MAX(231−1)，请返回 INT_MAX；如果整数小于INT_MIN(−231) ，请返回 INT_MIN；

分析：

主要难点是判断是否溢出。

代码：

1. acwing 版

    ```c++
    class Solution {
    public:
        int strToInt(string str) {
            // 消除串首的空格
            int pos = 0;
            while (str[pos] == ' ') ++pos;

            // 判断是正还是负
            bool minus = false;
            if (str[pos] == '-')
            {
                minus = true;
                ++pos;
            }
            else if (str[pos] == '+')
                ++pos;
                
            // 判断是否还有杂项字母
            if (str[pos] < '0' || str[pos] > '9')
                return 0;
            
            // 消除字符串末尾的非数字字符
            int end = str.size() - 1;
            while ('0' > str[end] || str[end] > '9') --end;
            
            int res = 0;
            for (int i = pos; i <= end; ++i)
            {
                // 判断整数溢出
                if (!minus && (res > INT_MAX / 10 || (res == INT_MAX / 10 && str[i] > '7')))
                    return INT_MAX;
                if (minus && (res > INT_MAX / 10 || (res == INT_MAX / 10 && str[i] > '8')))
                    return INT_MIN;
                res = res * 10 + (str[i] - '0');  // 这里的括号不能少，否则在 res * 10 + str[i] 的时候就可能溢出
            }
            
            if (minus) return -res;
            return res;
        }
    };
    ```

1. leetcode 版

    ```c++
    class Solution {
    public:
        int myAtoi(string s) {
            int pos = 0;
            while (s[pos] == ' ') ++pos;

            bool minus = false;
            if (s[pos] == '-')
            {
                minus = true;
                ++pos;
            }
            else if (s[pos] == '+')
                ++pos;

            if (s[pos] < '0' || s[pos] > '9') return 0;

            int res = 0;
            for (int i = pos; i < s.size(); ++i)
            {
                if (s[i] < '0' || s[i] > '9') break;
                if (!minus && (res > INT_MAX / 10 || (res == INT_MAX / 10 && s[i] > '7')))
                    return INT_MAX;
                if (minus && (res < INT_MIN / 10 || (res == INT_MIN / 10 && s[i] > '8')))
                    return INT_MIN;
                if (!minus)
                    res = res * 10 + (s[i] - '0');
                if (minus)
                    res = res * 10 - (s[i] - '0');
            }

            return res;
        }
    };
    ```

1 官方题解：自动机（没看）

    ```c++
    class Automaton {
        string state = "start";
        unordered_map<string, vector<string>> table = {
            {"start", {"start", "signed", "in_number", "end"}},
            {"signed", {"end", "end", "in_number", "end"}},
            {"in_number", {"end", "end", "in_number", "end"}},
            {"end", {"end", "end", "end", "end"}}
        };

        int get_col(char c) {
            if (isspace(c)) return 0;
            if (c == '+' or c == '-') return 1;
            if (isdigit(c)) return 2;
            return 3;
        }
    public:
        int sign = 1;
        long long ans = 0;

        void get(char c) {
            state = table[state][get_col(c)];
            if (state == "in_number") {
                ans = ans * 10 + c - '0';
                ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
            }
            else if (state == "signed")
                sign = c == '+' ? 1 : -1;
        }
    };

    class Solution {
    public:
        int myAtoi(string str) {
            Automaton automaton;
            for (char c : str)
                automaton.get(c);
            return automaton.sign * automaton.ans;
        }
    };
    ```

1. 自己写的有限状态机

    ```c++
    class Solution {
    public:
        int strToInt(string str) {
            vector<unordered_map<char, int>> s({
                {{' ', 0}, {'s', 1}, {'d', 2}, {'o', 3}},  // 0: 起始（空格）
                {{' ', 3}, {'s', 3}, {'d', 2}, {'o', 3}},  // 1: 符号
                {{' ', 3}, {'s', 3}, {'d', 2}, {'o', 3}},  // 2: 数字
                {{' ', 3}, {'s', 3}, {'d', 3}, {'o', 3}}   // 3: 结束
            });

            int ans = 0;
            int cur = 0;
            char trans;
            bool minus = false;
            for (int i = 0; i < str.size(); ++i)
            {
                if (str[i] == ' ') trans = ' ';
                else if (str[i] == '+' || str[i] == '-') trans = 's';
                else if (isdigit(str[i])) trans = 'd';
                else trans = 'o';
                cur = s[cur][trans];  // 先转移状态，再在下面根据当前状态做处理
                
                if (cur == 1 && str[i] == '-') minus = true;
                if (cur == 2)
                {
                    if (minus && ((ans == INT32_MAX / 10 && str[i]-'0' >= 8) || (ans > INT32_MAX / 10))) return INT32_MIN;
                    if (!minus && ((ans == INT32_MAX / 10 && str[i]-'0' >= 7) || (ans > INT32_MAX / 10))) return INT32_MAX;
                    ans *= 10;
                    ans += str[i] - '0';
                }
            }
            if (minus) return -ans;
            return ans;
        }
    };
    ```

### 你能在你最喜欢的那天吃到你最喜欢的糖果吗？

给你一个下标从 0 开始的正整数数组 candiesCount ，其中 candiesCount[i] 表示你拥有的第 i 类糖果的数目。同时给你一个二维数组 queries ，其中 queries[i] = [favoriteTypei, favoriteDayi, dailyCapi] 。

你按照如下规则进行一场游戏：

你从第 0 天开始吃糖果。
你在吃完 所有 第 i - 1 类糖果之前，不能 吃任何一颗第 i 类糖果。
在吃完所有糖果之前，你必须每天 至少 吃 一颗 糖果。
请你构建一个布尔型数组 answer ，满足 answer.length == queries.length 。answer[i] 为 true 的条件是：在每天吃 不超过 dailyCapi 颗糖果的前提下，你可以在第 favoriteDayi 天吃到第 favoriteTypei 类糖果；否则 answer[i] 为 false 。注意，只要满足上面 3 条规则中的第二条规则，你就可以在同一天吃不同类型的糖果。

请你返回得到的数组 answer 。

 

示例 1：

```
输入：candiesCount = [7,4,5,3,8], queries = [[0,2,2],[4,2,4],[2,13,1000000000]]
输出：[true,false,true]
提示：
1- 在第 0 天吃 2 颗糖果(类型 0），第 1 天吃 2 颗糖果（类型 0），第 2 天你可以吃到类型 0 的糖果。
2- 每天你最多吃 4 颗糖果。即使第 0 天吃 4 颗糖果（类型 0），第 1 天吃 4 颗糖果（类型 0 和类型 1），你也没办法在第 2 天吃到类型 4 的糖果。换言之，你没法在每天吃 4 颗糖果的限制下在第 2 天吃到第 4 类糖果。
3- 如果你每天吃 1 颗糖果，你可以在第 13 天吃到类型 2 的糖果。
```

分析：

把糖果一维化，只要比较两个区间是否有交集即可。

代码：

```c++
class Solution {
public:
    vector<bool> canEat(vector<int>& candiesCount, vector<vector<int>>& queries) {
        vector<long long> accu(candiesCount.size()+1);
        accu[0] = 0;
        // 计算每种类型的糖果的起始位置，题解中说这叫“前缀和”
        for (int i = 1; i <= candiesCount.size(); ++i)
        {
            accu[i] = accu[i-1] + candiesCount[i-1];
        }

        long long left, right;
        vector<bool> ans(queries.size());
        for (int i = 0; i < queries.size(); ++i)
        {
            left = queries[i][1];
            right = (long long) queries[i][2] * (long long) (queries[i][1] + 1);
            if (min(right, accu[queries[i][0]+1]) - max(left, accu[queries[i][0]]) > 0)
                ans[i] = true;
            else
                ans[i] = false;
        }
        return ans;
    }
};
```

### 大数阶乘

分析：

代码：

```c++
#include<iostream>
using namespace std;
 
int multiply(int x, int a[], int size)
{
	int carry = 0;
    int p;
    
	for(int i = 0; i < size; ++i)
	{
		p = a[i] * x + carry;
		a[i] = p % 10;
		carry = p / 10;
	}
		
	while(carry != 0)
	{
		a[size] = carry % 10;
		carry = carry / 10;
		size++;		
	}

    return size;
}
     
int main()
{  
	int n, a[1000], i, size = 1;
    a[0] = 1;
 
	cout << "Enter any large number:";
    cin >> n;
    	
    for(i=2;i<=n;++i)
    {
		size=multiply(i, a, size);    		
    }
    	
    for(int i = size-1; i > -1; --i)
    {
		cout << a[i];    	
    }
    	
   	return 0;
}

```

### 最大数

给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。

示例 1：

```
输入：nums = [10,2]
输出："210"
```

分析：可以找规律，也可以直接重载字符串比较函数。

代码：

1. 重载字符串比较函数

    ```c++
    class Solution {
    public:
        class string_comp
        {
            public:
            bool operator()(string &str1, string &str2)
            {
                return str1 + str2 > str2 + str1;
            }

        };
        
        string largestNumber(vector<int>& nums) {
            string res;
            vector<string> strs(nums.size());

            for (int i = 0; i < nums.size(); ++i)
                strs[i] = to_string(nums[i]);
            
            sort(strs.begin(), strs.end(), string_comp());

            for (auto &str: strs)
                res.append(str);
            
            while (res.size() > 1 && res[0] == '0') 
                res = res.substr(1);

            return res;
        }
    };
    ```

1. 找规律（挺复杂的，不建议看）

    ```c++
    class Solution {
    public:
        string largestNumber(vector<int> &nums) {
            sort(nums.begin(), nums.end(), [](const int &x, const int &y) {
                long sx = 10, sy = 10;
                while (sx <= x) {
                    sx *= 10;
                }
                while (sy <= y) {
                    sy *= 10;
                }
                return sy * x + y > sx * y + x;
            });
            if (nums[0] == 0) {
                return "0";
            }
            string ret;
            for (int &x : nums) {
                ret += to_string(x);
            }
            return ret;
        }
    };
    ```

### 数组拆分 I

给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大。

返回该 最大总和 。

示例 1：

```
输入：nums = [1,4,3,2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
所以最大总和为 4
```

分析：一个小数和一个大数搭配势必要牺牲一个大数，所以要让牺牲小一点，让两个数足够地近就好了。

代码：

```c++
class Solution {
public:
    int arrayPairSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0; i < nums.size(); i += 2)
            res += nums[i];
        return res;
    }
};
```

### 森林中的兔子

森林中，每个兔子都有颜色。其中一些兔子（可能是全部）告诉你还有多少其他的兔子和自己有相同的颜色。我们将这些回答放在 answers 数组里。

返回森林中兔子的最少数量。

示例:

```
输入: answers = [1, 1, 2]
输出: 5
解释:
两只回答了 "1" 的兔子可能有相同的颜色，设为红色。
之后回答了 "2" 的兔子不会是红色，否则他们的回答会相互矛盾。
设回答了 "2" 的兔子为蓝色。
此外，森林中还应有另外 2 只蓝色兔子的回答没有包含在数组中。
因此森林中兔子的最少数量是 5: 3 只回答的和 2 只没有回答的。

输入: answers = [10, 10, 10]
输出: 11

输入: answers = []
输出: 0
```

分析：假如有 13 个兔子都说了有 5 个与自己同色，那么可以将其分为 13 = 6 + 6 + 1。此时至少有`3 * 6 = 18`只兔子。不同颜色的兔子不可能说相同的数字，因此分组统计就好。

```c++
class Solution {
public:
    int numRabbits(vector<int>& answers) {
        unordered_map<int, int> count;
        for (auto &ans: answers)
            count[ans]++;
        int ans = 0;
        for (auto &[x, y]: count)
        {
            if (x == 0) ans += y;
            else ans += (y / (x + 1) + (y % (x+1) ? 1 : 0)) * (x + 1);
        }
        return ans;
    }
};
```

注意，三目运算符的优先级很低，需要用小括号括起来。

### 计数质数

统计所有小于非负整数 n 的质数的数量。

 
```
示例 1：

输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
示例 2：

输入：n = 0
输出：0
示例 3：

输入：n = 1
输出：0
```

代码：

1. 暴力计算

    ```c++
    class Solution {
    public:
    
        int countPrimes(int n) {
            if (n == 0 || n == 1) return 0;
            int bound;
            int count = 0;
            bool is_prime = false;
            for (int i = 2; i < n; ++i)
            {
                bound = sqrt(i);
                is_prime = true;
                for (int j = 2; j <= bound; ++j)
                {
                    if (i % j == 0)
                    {
                        is_prime =false;
                        break;
                    }
                }
                if (is_prime) ++count;
            }
            return count;
            
        }
    };
    ```

1. 埃氏筛

    若`x`是质数，则从`x*x`开始，将`x*x, x*(x+1), ...`都标记为合数。如果后续我们遇到一个数，它没有被标记过，那么它就是质数。

    ```c++
    class Solution {
    public:
        int countPrimes(int n) {
            if (n == 0 || n == 1) return 0;
            int count = 0;
            vector<bool> is_prime(n, true);
            for (int i = 2; i < n; ++i)
            {
                if (is_prime[i])
                {
                    ++count;
                    if ((long long) i * i < n)  // 防止 i * i 溢出
                    {
                        for (int j = i * i; j < n; j += i)
                            is_prime[j] = false;
                    }
                }
            }
            return count;
        }
    };
    ```

1. 线性筛（太难了没看）

    ```c++
    class Solution {
    public:
        int countPrimes(int n) {
            vector<int> primes;
            vector<int> isPrime(n, 1);
            for (int i = 2; i < n; ++i) {
                if (isPrime[i]) {
                    primes.push_back(i);
                }
                for (int j = 0; j < primes.size() && i * primes[j] < n; ++j) {
                    isPrime[i * primes[j]] = 0;
                    if (i % primes[j] == 0) {
                        break;
                    }
                }
            }
            return primes.size();
        }
    };
    ```

### 重塑矩阵

在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。

给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。

重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。

如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

示例 1:

```
输入: 
nums = 
[[1,2],
 [3,4]]
r = 1, c = 4
输出: 
[[1,2,3,4]]
解释:
行遍历nums的结果是 [1,2,3,4]。新的矩阵是 1 * 4 矩阵, 用之前的元素值一行一行填充新矩阵。
```

代码：

1. 自己写的，模拟下标，效率一般

    ```c++
    class Solution {
    public:
        vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
            if (mat.size() * mat[0].size() != r * c) return mat;
            vector<vector<int>> res(r, vector<int>(c));
            int count = 0, n = r * c;
            int i1 = 0, j1 = 0, i2 = 0, j2 = 0;
            while (count != n)
            {
                res[i2][j2++] = mat[i1][j1++];
                if (j2 == c)
                {
                    ++i2;
                    j2 = 0;
                }
                if (j1 == mat[0].size())
                {
                    ++i1;
                    j1 = 0;
                }
                ++count;
            }
            return res;
        }
    };
    ```

1. 使用`/`和`%`计算索引

    ```c++
    class Solution {
    public:
        vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
            int m = mat.size(), n = mat[0].size();
            if (m * n != r * c) return mat;

            vector<vector<int>> ans(r, vector<int>(c));
            int count = 0, num = m * n;
            while (count < num)
            {
                ans[count / c][count % c] = mat[count / n][count % n];
                ++count;
            }
            return ans;
        }
    };
    ```

### 杨辉三角

给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。

在杨辉三角中，每个数是它左上方和右上方的数的和。

```
示例:

输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```

代码：

1. 很简单，没啥好说的。

    ```c++
    class Solution {
    public:
        vector<vector<int>> generate(int numRows) {
            vector<vector<int>> res(numRows);
            res[0] = vector<int>({1});
            for (int i = 1; i < numRows; ++i)
            {
                res[i].resize(i+1);
                res[i].front() = 1;
                res[i].back() = 1;
                for (int j = 1; j < i; ++j)
                    res[i][j] = res[i-1][j-1] + res[i-1][j];
            }
            return res;
        }
    };
    ```

1. 后来又写的，几乎一样：

    ```c++
    class Solution {
    public:
        vector<vector<int>> generate(int numRows) {
            vector<vector<int>> ans;
            for (int i = 0; i < numRows; ++i)
            {
                ans.push_back(vector<int>(i+1));
                for (int j = 0; j < ans[i].size(); ++j)
                {
                    if (j == 0 || j == ans[i].size() - 1) ans[i][j] = 1;
                    else ans[i][j] = ans[i-1][j-1] + ans[i-1][j];
                }
            }
            return ans;
        }
    };
    ```

    这道题的两种写法展现了两种处理边界值的思路。一种是提前赋值，然后在遍历时候不考虑这些点。另一种是遍历所有点，但是对边界点特殊考虑。

### 交换字符使得字符串相同

有两个长度相同的字符串 s1 和 s2，且它们其中 只含有 字符 "x" 和 "y"，你需要通过「交换字符」的方式使这两个字符串相同。

每次「交换字符」的时候，你都可以在两个字符串中各选一个字符进行交换。

交换只能发生在两个不同的字符串之间，绝对不能发生在同一个字符串内部。也就是说，我们可以交换 s1[i] 和 s2[j]，但不能交换 s1[i] 和 s1[j]。

最后，请你返回使 s1 和 s2 相同的最小交换次数，如果没有方法能够使得这两个字符串相同，则返回 -1 。

 

示例 1：

输入：s1 = "xx", s2 = "yy"
输出：1
解释：
交换 s1[0] 和 s2[1]，得到 s1 = "yx"，s2 = "yx"。
示例 2：

输入：s1 = "xy", s2 = "yx"
输出：2
解释：
交换 s1[0] 和 s2[0]，得到 s1 = "yy"，s2 = "xx" 。
交换 s1[0] 和 s2[1]，得到 s1 = "xy"，s2 = "xy" 。
注意，你不能交换 s1[0] 和 s1[1] 使得 s1 变成 "yx"，因为我们只能交换属于两个不同字符串的字符。
示例 3：

输入：s1 = "xx", s2 = "xy"
输出：-1
示例 4：

输入：s1 = "xxyyxyxyxx", s2 = "xyyxyxxxyx"
输出：4
 

提示：

1 <= s1.length, s2.length <= 1000
s1, s2 只包含 'x' 或 'y'。

代码：

1. 找规律

    对于

    ```
    xx
    yy
    ```

    类型，只需要交换一次；

    对于

    ```
    xy
    yx
    ```

    类型，只需要交换两次；

    对于

    ```
    xxy
    yxx
    ```

    中间有相同的，可以直接跳过。

    我们优先分别消除`xy-xy`型和`yx-yx`型，如果还有剩余，那么必然还剩一对`xy-yx`型，答案再加 2 就可以了。

    ```cpp
    class Solution {
    public:
        int minimumSwap(string s1, string s2) {
            if (s1.size() != s2.size()) return -1;
            int ans = 0;
            int xy = 0, yx = 0;
            int p = 0;
            int n = s1.size();
            while (p < n)
            {
                if (s1[p] == s2[p])
                {
                    ++p;
                    continue;
                }
                if (s1[p] == 'x' && s2[p] == 'y')
                    ++xy;
                else
                    ++yx;
                ++p;
            }
            if ((xy + yx) % 2 != 0) return -1;
            ans += xy / 2 + yx / 2;
            if (xy % 2 == 1 && yx % 2 == 1)
                ans += 2;
            return ans;
        }
    };
    ```

    我们重点分析为什么不能用其他方法。看到题目中让统计数量，我们马上想到动态规划。可是动态规划要求大问题的最优可以分解为小问题的最优。我们假设前`i`个字符的子串的最小交换次数为`f(0, i-1)`，后面子串的最小交换次数为`f(i, n-1)`，那么`f(0, n-1) = f(0, i-1) + f(i, n-1)`成立吗？显然不成立。首先，存在`[0, i-1]`不可交换，`[i, n-1]`不可交换，但`[0, n-1]`可交换的情况。其次，对于`[xy, yx]`, `[xy, yx]`这两个字符串，如果分别计算交换次数的话，是`2 + 2 = 4`；但如果合在一起计算交换次数的话，是`1 + 1 = 2`。因此无论如何都无法使用动态规划。

### 区间相关

#### 会议室

给定一个会议时间安排的数组 intervals ，每个会议时间都会包括开始和结束的时间 intervals[i] = [starti, endi] ，请你判断一个人是否能够参加这里面的全部会议。

```
示例 1:
输入: intervals = [[0,30],[5,10],[15,20]]
输出: false
解释: 存在重叠区间，一个人在同一时刻只能参加一个会议。

示例 2:
输入: intervals = [[7,10],[2,4]]
输出: true
解释: 不存在重叠区间。
```

代码：

按会议开始时间排序，然后比较下一个会议开始时间和上一个会议结束的时间即可。

```c++
bool get_res(vector<vector<int>> &intervals)
{
    sort(intervals.begin(), intervals.end(), [](vector<int> &v1, vector<int> &v2) {
        return v1.front() < v2.front();
    });  // 似乎不用写匿名函数也可以
    for (int i = 1; i < intervals.size(); ++i)
    {
        if (intervals[i][0] < intervals[i-1][1]) return false;
    }
    return true;
}
```

#### 合并区间

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

示例 1：

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

代码：

1. 以前写的

    ```c++
    class Solution {
    public:
        vector<vector<int>> merge(vector<vector<int>>& intervals) {
            sort(intervals.begin(), intervals.end());
            vector<vector<int>> merged;
            merged.push_back(intervals[0]);
            for (int i = 1; i < intervals.size(); ++i)
            {
                if (intervals[i][0] <= merged.back()[1])  // 别忘了这里的等号
                    merged.back() = vector<int>({merged.back()[0], max(intervals[i][1], merged.back()[1])});  // 若有重叠则合并，注意后面的 max() 作比较
                else
                    merged.push_back(intervals[i]);
            }
            return merged;
        }
    };
    ```

1. 后来又写的

    ```c++
    class Solution {
    public:
        vector<vector<int>> merge(vector<vector<int>>& intervals) {
            sort(intervals.begin(), intervals.end());
            vector<vector<int>> ans;
            ans.push_back(intervals[0]);
            for (int i = 1; i < intervals.size(); ++i)
            {
                if (intervals[i][0] <= ans.back()[1])
                    ans.back()[1] = max(ans.back()[1], intervals[i][1]);
                else
                    ans.push_back(intervals[i]);
            }
            return ans;
        }
    };
    ```

#### 插入区间

给你一个 无重叠的 ，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

 

示例 1：

```
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]
```

示例 2：

```
输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出：[[1,2],[3,10],[12,16]]
解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
```

代码：

1. 因为区间已经按顺序排好了，所以先插入前面几个与`newInterval`无关的区间，再合并与`newInterval`有关的区间，最后把剩下的插入就好了。

    ```c++
    class Solution {
    public:
        vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
            vector<vector<int>> res;
            int idx = 0;
            while (idx < intervals.size() && intervals[idx][1] < newInterval[0])
                res.push_back(intervals[idx++]);
                
            while (idx < intervals.size() && newInterval[1] >= intervals[idx][0])  // 别忘了这里的等号
            {
                newInterval[0] = min(newInterval[0], intervals[idx][0]);
                newInterval[1] = max(newInterval[1], intervals[idx][1]);
                ++idx;
            }
            res.push_back(newInterval);

            while (idx < intervals.size())
                res.push_back(intervals[idx++]);
            return res;
        }
    };
    ```

1. 直接将`newInterval`插入到`intervals`中，然后再排序，再套用“合并区间”的代码就可以了。

#### 删除被覆盖区间

给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。

只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。

在完成所有删除操作后，请你返回列表中剩余区间的数目。

示例：

```
输入：intervals = [[1,4],[3,6],[2,8]]
输出：2
解释：区间 [3,6] 被区间 [2,8] 覆盖，所以它被删除了。
```

代码：

先按左端升序排序，再按右端降序排序，最后仿照着“会议室”的写法写一下就可以了。

```c++
class Solution {
public:
    int removeCoveredIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](vector<int> &v1, vector<int> &v2){
            if (v1[0] < v2[0]) return true;
            else if (v1[0] == v2[0]) return v1[1] > v2[1];
            return false;
        });
        int count = intervals.size();
        for (int i = 1; i < intervals.size(); ++i)
        {
            if (intervals[i][0] >= intervals[i-1][0] && intervals[i][1] <= intervals[i-1][1])
            {
                --count;
                intervals[i] = intervals[i-1];  // 如果被覆盖，拿前面的区间替换后面的区间，不能删除，删除会出问题
            }
        }
        return count;
    }
};
```

#### 汇总区间

给定一个无重复元素的有序整数数组 nums 。

返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。

列表中的每个区间范围 [a,b] 应该按如下格式输出：

"a->b" ，如果 a != b
"a" ，如果 a == b
 

示例 1：

```
输入：nums = [0,1,2,4,5,7]
输出：["0->2","4->5","7"]
解释：区间范围是：
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"
```

代码：

滑动窗口。

```c++
class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> res;
        if (nums.empty()) return res;
        int slow = 0, fast = 1;
        while (fast < nums.size())
        {
            if ((long) nums[fast] - nums[fast-1] != 1)  // 测试样例中有极端的数字，所以用 long
            {
                if (slow == fast-1) res.push_back(to_string(nums[slow]));
                else res.push_back(to_string(nums[slow]) + "->" + to_string(nums[fast-1]));
                slow = fast;
            }
            ++fast;
        }
        if (slow == fast-1) res.push_back(to_string(nums[slow]));
        else res.push_back(to_string(nums[slow]) + "->" + to_string(nums[fast-1]));
        return res;
    }
};
```

#### 俄罗斯套娃信封问题

给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。

 
示例 1：

输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
示例 2：

输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1

分析：

1. 动态规划。同最长递增子序列问题。但是这道题中没有规定要按数组顺序，所以要先对信封排序。排序时先按 w 进行排序，为了使相同的 w 只取一个 h，我们还需对 h 进行降序排列。

代码：

1. 动态规划

    ```c++
    class Solution {
    public:
        int maxEnvelopes(vector<vector<int>>& envelopes) {
            sort(envelopes.begin(), envelopes.end(), [](vector<int> &v1, vector<int> &v2){
                if (v1[0] < v2[0]) return true;
                if (v1[0] == v2[0]) return v1[1] > v2[1];
                return false;
            });
            
            int res = 1;
            vector<int> dp(envelopes.size(), 1);
            for (int i = 1; i < envelopes.size(); ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    if (envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1])
                    {
                        dp[i] = max(dp[i], dp[j]+1);
                    }
                }
                res = max(dp[i], res);
            }
            return res;
        }
    };
    ```

1. 贪心 + 二分

    ```c++
    class Solution {
    public:
        int maxEnvelopes(vector<vector<int>>& envelopes) {
            sort(envelopes.begin(), envelopes.end(), [](vector<int> &v1, vector<int> &v2){
                if (v1[0] == v2[0]) return v1[1] > v2[1];
                return v1[0] < v2[0];
            });

            vector<vector<int>> m;
            m.push_back(envelopes[0]);
            int l, r, mid;
            for (int i = 1; i < envelopes.size(); ++i)
            {
                if (envelopes[i][0] > m.back()[0] && envelopes[i][1] > m.back()[1])
                    m.push_back(envelopes[i]);
                else
                {
                    l = 0;
                    r = m.size() - 1;
                    while (l <= r)
                    {
                        mid = l + (r - l) / 2;
                        if (m[mid][0] < envelopes[i][0] && m[mid][1] < envelopes[i][1]) l = mid + 1;
                        else r = mid - 1;
                    }
                    if (m[l][1] >= envelopes[i][1])
                        m[l] = envelopes[i];
                }
            }
        
            return m.size();
        }
    };
    ```

#### 区间列表的交集

给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。

返回这 两个区间列表的交集 。

形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。

两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。

 
```
示例 1：

输入：firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
示例 2：

输入：firstList = [[1,3],[5,9]], secondList = []
输出：[]
示例 3：

输入：firstList = [], secondList = [[4,8],[10,12]]
输出：[]
示例 4：

输入：firstList = [[1,7]], secondList = [[3,10]]
输出：[[3,7]]
```

代码：

1. 自己写的，复杂度O(mn)，只能击败 5%

    ```c++
    class Solution {
    public:
        vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList) {
            vector<vector<int>> ans;
            int start, end;
            for (int i = 0; i < firstList.size(); ++i)
            {
                for (int j = 0; j < secondList.size(); ++j)
                {
                    start = max(firstList[i][0], secondList[j][0]);
                    end = min(firstList[i][1], secondList[j][1]);
                    if (start <= end)
                    {
                        ans.push_back(vector<int>({start, end}));
                    }
                }
            }
            return ans;
        }
    };
    ```

1. 一个评论里的思路，双指针做剪枝，代码清晰易懂

    ```c++
    class Solution {
    public:
        vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList) {
            vector<vector<int>> ans;
            if (firstList.empty() || secondList.empty()) return ans;
            int start1, end1, start2, end2;
            int i = 0, j = 0;
            while (i < firstList.size() && j < secondList.size())
            {
                start1 = firstList[i][0], end1 = firstList[i][1];
                start2 = secondList[j][0], end2 = secondList[j][1];
                if (start2 > end1) ++i;
                else if (start1 > end2) ++j;
                else
                {
                    ans.push_back(vector<int>({max(start1, start2), min(end1, end2)}));
                    if (end2 > end1) ++i;
                    else ++j;
                }
            }
            return ans;
        }
    };
    ```

#### 无重叠区间

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:

* 可以认为区间的终点总是大于它的起点。
* 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

```
示例 1:

输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
示例 2:

输入: [ [1,2], [1,2], [1,2] ]

输出: 2

解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
示例 3:

输入: [ [1,2], [2,3] ]

输出: 0

解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```

代码：

1. 动态规划（超时）

    令`dp[i]`表示以第`i`个区间结尾的不重叠区间的最大数量。

    ```c++
    class Solution {
    public:
        int eraseOverlapIntervals(vector<vector<int>>& intervals) {
            sort(intervals.begin(), intervals.end());
            int n = intervals.size();
            vector<int> dp(n, 1);
            for (int i = 1; i < n; ++i)
            {
                for (int j = 0; j < i; ++j)
                {
                    if (intervals[j][1] <= intervals[i][0])
                        dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            return n - *max_element(dp.begin(), dp.end());
        }
    };
    ```

1. 贪心

    看不懂这个。

    ```c++
    class Solution {
    public:
        int eraseOverlapIntervals(vector<vector<int>>& intervals) {
            if (intervals.empty()) {
                return 0;
            }
            
            sort(intervals.begin(), intervals.end(), [](const auto& u, const auto& v) {
                return u[1] < v[1];  // 为什么要按末尾排序？
            });

            int n = intervals.size();
            int right = intervals[0][1];
            int ans = 1;
            for (int i = 1; i < n; ++i) {
                if (intervals[i][0] >= right) {  // 怎样保证不重不漏？
                    ++ans;
                    right = intervals[i][1];
                }
            }
            return n - ans;
        }
    };
    ```

### 下一个排列

实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间。

```
示例 1：

输入：nums = [1,2,3]
输出：[1,3,2]
示例 2：

输入：nums = [3,2,1]
输出：[1,2,3]
示例 3：

输入：nums = [1,1,5]
输出：[1,5,1]
```

算法的标准过程：

1. 从后向前查找第一个相邻升序的元素对（i, j），满足`A[i] < A[j]`。此时`[j, end)`必然是降序

1. 在`[j, end)`从后向前查找第一个满足`A[i] < A[k]`的`k`。`A[i]`、`A[k]`分别就是上文所说的【小数】、【大数】。

1. 将`A[i]`与`A[k]`交换

1. 可以断定这时`[j, end)`必然是降序，逆置`[j, end)`，使其升序。

1. 如果在步骤 1 找不到符合的相邻元素对，说明当前`[begin, end)`为一个降序顺序，则直接跳到步骤 4。

代码：

```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        if (nums.size() <= 1) return;
        int i = nums.size() - 2, j = nums.size() - 1, k = nums.size() - 1;

        // find A[i] < A[j]
        while (i >= 0 && nums[i] >= nums[j])
        {
            --i;
            --j;
        }

        // 如果不是最后一个排列
        if (i >= 0)
        {
            while (nums[i] >= nums[k]) --k;
            swap(nums[i], nums[k]);
        }

        // reverse the sequence
        for (int i = j, j = nums.size() - 1; i < j; ++i, --j)
        {
            swap(nums[i], nums[j]);
        }
    }
};
```

### 两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
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
```

思路：

1. 可以将短的链表视为高位为 0

代码：

1. 遍历两遍链表

    第一遍相加，第二遍处理进位。

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
            ListNode *p1 = l1, *p2 = l2;
            ListNode *prev = nullptr;
            while (p1 && p2)
            {
                p1->val += p2->val;
                prev = p1;
                p1 = p1->next;
                p2 = p2->next;
            }

            if (!p1 && p2) prev->next = p2; 

            p1 = l1;
            while (p1)
            {
                if (p1->val > 9)
                {
                    if (p1->next) p1->next->val++;
                    else p1->next = new ListNode(1);
                    p1->val -= 10;
                }
                p1 = p1->next;
            }

            return l1;
        }
    };
    ```

1. 将短的链表视为高位为 0

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
            ListNode *p1 = l1, *p2 = l2;
            ListNode *prev = nullptr;
            while (p1 || p2)
            {
                if (p1)
                {
                    p1->val += p2 ? p2->val : 0;
                    if (p1->val > 9)
                    {
                        p1->val -= 10;
                        if (p1->next) p1->next->val += 1;
                        else p1->next = new ListNode(1);
                    }
                }
                else
                {
                    prev->next = p2;
                    break;
                }
                
                prev = p1;
                p1 = p1->next;
                if (p2) p2 = p2->next;
            }

            return l1;
        }
    };
    ```

1. 后来写的，相加时同时处理进位，然后分别处理链表长度不一的两种情况

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
            ListNode *p1 = l1, *p2 = l2, *prev;
            int sum = 0, carry = 0;
            while (p1 && p2)
            {
                sum = p1->val + p2->val;
                p1->val = (sum + carry) % 10;
                carry = (sum + carry) / 10;
                prev = p1;
                p1 = p1->next;
                p2 = p2->next;
            }
            while (p1)
            {
                sum = p1->val;
                p1->val = (sum + carry) % 10;
                carry = (sum + carry) / 10;
                prev = p1;
                p1 = p1->next;
            }
            if (p2)
            {
                prev->next = p2;
                while (p2)
                {
                    sum = p2->val;
                    p2->val = (sum + carry) % 10;
                    carry = (sum + carry) / 10;
                    prev = p2;
                    p2 = p2->next;
                }
            }
            if (carry) prev->next = new ListNode(1); 
            return l1;
        }
    };
    ```

1. 后来又写的，简洁版

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
            ListNode *p1 = l1, *p2 = l2, *prev;
            int sum = 0, carry = 0;
            while (p1 || p2 || carry)  // carry 也加入判断，不用再单独处理了
            {
                sum = (p1 ? p1->val : 0) + (p2 ? p2->val : 0);  // 因为前面的 while 条件写成了或的形式，所以这里需要加上判断，是取本值还是 0
                if (!p1) {prev->next = new ListNode(0); p1 = prev->next;}  // 如果 p1 不够长，把 p1 补长
                p1->val = (sum + carry) % 10;
                carry = (sum + carry) / 10;
                prev = p1;
                p1 = p1->next;
                if (p2) p2 = p2->next;  // 如果 p2 比 p1 短，那么让 p2 停着不动
            }
            return l1;
        }
    };
    ```

### 整数转罗马数字

罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给你一个整数，将其转为罗马数字。

```
示例 1:

输入: num = 3
输出: "III"
示例 2:

输入: num = 4
输出: "IV"
示例 3:

输入: num = 9
输出: "IX"
示例 4:

输入: num = 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
示例 5:

输入: num = 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

代码：

贪心。

```c++
class Solution {
public:
    string intToRoman(int num) {
        vector<pair<string, int>> v({
            {"I", 1},
            {"IV", 4},
            {"V", 5},
            {"IX", 9},
            {"X", 10},
            {"XL", 40},
            {"L", 50},
            {"XC", 90},
            {"C", 100},
            {"CD", 400},
            {"D", 500},
            {"CM", 900},
            {"M", 1000}
        });

        string ans;
        int i = v.size() - 1;
        while (i > -1)
        {
            if (num >= v[i].second)
            {
                ans.append(v[i].first);
                num -= v[i].second;
            }
            else
                --i;
        }
        return ans;
    }
};
```

### 验证IP地址

编写一个函数来验证输入的字符串是否是有效的 IPv4 或 IPv6 地址。

如果是有效的 IPv4 地址，返回 "IPv4" ；
如果是有效的 IPv6 地址，返回 "IPv6" ；
如果不是上述类型的 IP 地址，返回 "Neither" 。
IPv4 地址由十进制数和点来表示，每个地址包含 4 个十进制数，其范围为 0 - 255， 用(".")分割。比如，172.16.254.1；

同时，IPv4 地址内的数不会以 0 开头。比如，地址 172.16.254.01 是不合法的。

IPv6 地址由 8 组 16 进制的数字来表示，每组表示 16 比特。这些组数字通过 (":")分割。比如,  `2001:0db8:85a3:0000:0000:8a2e:0370:7334` 是一个有效的地址。而且，我们可以加入一些以 0 开头的数字，字母可以使用大写，也可以是小写。所以， `2001:db8:85a3:0:0:8A2E:0370:7334` 也是一个有效的 IPv6 address地址 (即，忽略 0 开头，忽略大小写)。

然而，我们不能因为某个组的值为 0，而使用一个空的组，以至于出现 (::) 的情况。 比如，`2001:0db8:85a3::8A2E:0370:7334` 是无效的 IPv6 地址。

同时，在 IPv6 地址中，多余的 0 也是不被允许的。比如， `02001:0db8:85a3:0000:0000:8a2e:0370:7334` 是无效的。

 
```
示例 1：

输入：IP = "172.16.254.1"
输出："IPv4"
解释：有效的 IPv4 地址，返回 "IPv4"
示例 2：

输入：IP = "2001:0db8:85a3:0:0:8A2E:0370:7334"
输出："IPv6"
解释：有效的 IPv6 地址，返回 "IPv6"
示例 3：

输入：IP = "256.256.256.256"
输出："Neither"
解释：既不是 IPv4 地址，又不是 IPv6 地址
示例 4：

输入：IP = "2001:0db8:85a3:0:0:8A2E:0370:7334:"
输出："Neither"
示例 5：

输入：IP = "1e1.4.5.6"
输出："Neither"
```

代码：

1. 先分割字符串，再分别验证。挺麻烦的。有时间了看看。

    ```c++
    class Solution {
    public:
        string validIPAddress(string IP) {
            if(is4(IP))return "IPv4";
            else if(is6(IP))return "IPv6";
            return "Neither";
        }
        bool is4(string IP){
            vector<string>ip;
            split(IP,ip,'.');
            if(ip.size()!=4)return false;
            for(string s:ip){
                if(s.size()==0||(s.size()>1&&s[0]=='0')||s.size()>3)return false;
                for(char c:s){
                    if(!isdigit(c))return false;
                }
                int digit=stoi(s);
                if(digit<0||digit>255)return false;
            }
            return true;
        }

        bool is6(string IP){
            vector<string>ip;
            split(IP,ip,':');
            if(ip.size()!=8)return false;
            for(string s:ip){
                if(s.size()==0||s.size()>4)return false;
                for(char c:s){
                    if(c<'0'||c>'9'&&c<'A'||c>'F'&&c<'a'||c>'f')return false;
                }
            }
            return true;
        }
        void split(string s,vector<string>&ip,char c){
            stringstream ss(s);
            string tmp;
            while(getline(ss,tmp,c))ip.push_back(tmp);
            if(s.size()>0&&s.back()==c)ip.push_back({});
        }
    };
    ```

### 消除游戏

列表 arr 由在范围 [1, n] 中的所有整数组成，并按严格递增排序。请你对 arr 应用下述算法：

从左到右，删除第一个数字，然后每隔一个数字删除一个，直到到达列表末尾。
重复上面的步骤，但这次是从右到左。也就是，删除最右侧的数字，然后剩下的数字每隔一个删除一个。
不断重复这两步，从左到右和从右到左交替进行，直到只剩下一个数字。
给你整数 n ，返回 arr 最后剩下的数字。

 

```
示例 1：

输入：n = 9
输出：6
解释：
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
arr = [2, 4, 6, 8]
arr = [2, 6]
arr = [6]
示例 2：

输入：n = 1
输出：1
```

代码：

1. 官方，模拟

    ```c++
    class Solution {
    public:
        int lastRemaining(int n) {
            int a1 = 1, an = n;
            int k = 0, cnt = n, step = 1;
            while (cnt > 1) {
                if (k % 2 == 0) { // 正向
                    a1 = a1 + step;
                    an = (cnt % 2 == 0) ? an : an - step;
                } else { // 反向
                    a1 = (cnt % 2 == 0) ? a1 : a1 + step;
                    an = an - step;
                }
                k++;
                cnt = cnt >> 1;
                step = step << 1;
            }
            return a1;
        }
    };
    ```

1. 网友答案，约瑟夫环

    ```java
    class Solution {
        public int lastRemaining(int n) {
            return n == 1 ? 1 : 2 * (n / 2 + 1 - lastRemaining(n / 2));
        }
    }
    ```

## 基本算法

### 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。


示例 1:

```
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4
```

代码：

二分查找的思想很简单，即每次取区间一半的值与目标值比较，看看目标值落在哪个区间内。但是这样的思考仅仅是连续量的思维。我们实际要处理的问题是离散量，因此要处理整除问题，边界值，基数（区间长度）与序数（索引）的区别，在比较时是否加等号。除此之外，我们应该把问题写成递归的形式还是循环的形式？（这两种似乎都可以）如果写成循环的形式，由于我们事先不知道要循环多少次，所以需要写成动态循环。如果写动态循环，那么应该在哪里退出循环？在`while()`中，还是在内部的`if()`中？我们需要在`while()`内部使函数`return`吗？初始化状态是放在循环外，在是在循环内，合并到更新状态中？

这些问题都需要得到解答，因此事实上要完全理解二分搜索，并不是一件容易的事。（这些问题可以看作是“为什么不能那样？”的展开）

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1, m = (l + r) >> 1;  // 似乎应该是 r = nums.size(); ?
        while (l < r)
        {
            if (nums[m] < target) l = m + 1;
            else r = m;
            m = (l + r) >> 1;
        }
        if (nums[l] == target) return l;
        else return -1;
    }
};
```

模板：

1. 正常二分查找 

    ```c++
    int binary_search(int[] nums, int target) {
        int left = 0, right = nums.length - 1; 
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1; 
            } else if(nums[mid] == target) {
                // 直接返回
                return mid;
            }
        }
        // 直接返回
        return -1;
    }
    ```

    这个边界情况我仍然不知道怎么推导出来。似乎有这样的原则在里面：

    1. 每一次缩小搜索范围，都要保证目标值在新的搜索范围内
    1. 要保证可以跳出循环（结束算法）

    假如没有找到 target，那么在返回 -1 时，`left`和`right`代表的意义是什么？

    1. 假设无论之前搜索区间有多长，最终只会缩减到两种情况，一种是`[a, b]`，另一种是`[a, b, c]`。我们要找的数字为`t`。

    2. 对于`[a, b]`，根据我们的原则，`t`一定在搜索范围内，因此`a <= t <= b`；而又因为`t`不等于`a`和`b`，所以`a < t < b`。又因为此时`l = 0`，`r = 1`，我们可以计算出`m = l + (r - l) / 2 = 0`，所以`nums[m] = a < t`。根据代码，`l = m + 1 = 1`，此时计算`m = l + (r - 1) / 2 = 1`，`nums[m] = b > t`。根据代码有`r = m - 1 = 0`。此时`l = 1`，`nums[l] = b`，是第一个比`t`大的元素；`r = 0`，`nums[r] = a`，是第一个比`t`小的元素。

    3. 对于`[a, b, c]`，`t`有两种情况：`a < t < b`，`b < t < c`。

        当`a < t < b`时，`m`首先被计算出来等于`1`，然后`r`会被赋值`0`，此时`l = r = 0`，`nums[l] = nums[r] = a`。接着计算`m = 0`，`l = m + 1 = 1`，退出循环。此时`l`指向第一个比`t`大的元素，`m`指向第一个比`t`小的元素。

        当`b < t < c`时，同理得到`l = 2`，`r = 1`。此时`l`指向第一个比`t`大的元素，`m`指向第一个比`t`小的元素。

        因此对于`[a, b, c]`这种情况，仍然有`l`指向第一个比`t`大的元素，`m`指向第一个比`t`小的元素成立。

    这时我们只要证明“无论之前搜索区间有多长，最终只会缩减到两种情况”就可以了。

    还需要讨论数组两端的边界情况。有空了讨论一下。

1. 搜索左边界

    ```c++
    int left_bound(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] == target) {
                // 别返回，锁定左侧边界（虽然我已经找到目标值了，但是我要找的东西还在左边）
                right = mid - 1;
            }
        }
        // 最后要检查 left 越界的情况
        if (left >= nums.length || nums[left] != target)
            return -1;
        return left;  // 为什么返回的是 left 而不是 right ？
    }
    ```

    如果不加`nums[left] != target`的限制，那么如果目标值不在数组中，则会返回第一个大于目标值的下标。比如`[2,3,5]`里找`4`，则会返回`2`。

    如果不加`left >= nums.length`的限制，那么如果目标值比数组中任何一个数都大，则返回数组的长度。

    如果这两个条件都不写，其实就是“搜索插入位置”这道题了。

    问题：什么是左边界？当我们在找左边界时我们在找什么？

    另一种形式：

    ```c++
    int left_bound(int[] nums, int target) {
        if (nums.length == 0) return -1;
        int left = 0;
        int right = nums.length; // 注意
        
        while (left < right) { // 注意
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                right = mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid; // 注意
            }
        }

        // 最后要检查 left 越界的情况
        if (left >= nums.length || nums[left] != target)
            return -1;
        return left;
    }
    ```

1. 搜索右边界

    ```c++
    int right_bound(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] == target) {
                // 别返回，锁定右侧边界（虽然我已经找到目标值了，但是我要找的东西还在右边）
                left = mid + 1;
            }
        }
        // 最后要检查 right 越界的情况
        if (right < 0 || nums[right] != target)
            return -1;
        return right;
    }
    ```

    如果数字不存在于数组中，且没有`nums[right] != target`的限制，那么返回的是第一个小于`traget`的下标。

    另一种形式：

    ```c++
    int right_bound(vector<int> &nums, int target) {
        if (nums.size() == 0) return -1;
        int left = 0, right = nums.size();
        
        while (left < right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                left = mid + 1; // 注意
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid;
            }
        }
        // 这里改为检查 right 越界的情况，见下图
        if (left == 0 || nums[left-1] != target) return -1;
        return left-1;
    }
    ```

### 第一个错误的版本

你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。

假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。

你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

 
示例 1：

```
输入：n = 5, bad = 4
输出：4
解释：
调用 isBadVersion(3) -> false 
调用 isBadVersion(5) -> true 
调用 isBadVersion(4) -> true
所以，4 是第一个错误的版本。
```

分析：相当于二分法求满足条件的左边界。

代码：

```c++
// The API isBadVersion is defined for you.
// bool isBadVersion(int version);

class Solution {
public:
    int firstBadVersion(int n) {
        int l = 1, r = n, m;
        while (l < r)
        {
            m = l + ((r - l) >> 1);
            if (isBadVersion(m)) r = m;
            else l = m + 1;
        }
        return l;
    }
};
```

后来写的：

```c++
// The API isBadVersion is defined for you.
// bool isBadVersion(int version);

class Solution {
public:
    int firstBadVersion(int n) {
        int left = 1, right = n, mid;
        while (left <= right)
        {
            mid = left + (right - left) / 2;
            if (isBadVersion(mid)) right = mid - 1;
            else left = mid + 1;
        }
        return left;
    }
};
```

### 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

```
示例 1:

输入: [1,3,5,6], 5
输出: 2
示例 2:

输入: [1,3,5,6], 2
输出: 1
示例 3:

输入: [1,3,5,6], 7
输出: 4
```

代码：

1. 带等号版

    ```c++
    class Solution {
    public:
        int searchInsert(vector<int>& nums, int target) {
            int l = 0, r = nums.size() - 1;
            int m;
            while (l <= r)
            {
                m = l + (r - l) / 2;
                if (nums[m] == target) return m;
                else if (nums[m] < target) l = m + 1;
                else r = m - 1;
            }
            return l;
        }
    };
    ```

1. 不带等号版

    ```c++
    class Solution {
    public:
        int searchInsert(vector<int>& nums, int target) {
            int l = 0, r = nums.size();
            int m;
            while (l < r)
            {
                m = l + (r - l) / 2;
                if (nums[m] < target) l = m + 1;
                else r = m;
            }
            return l;
        }
    };
    ```

这道题等价于两个子题目的组合：

1. 给定有序数组和指定元素，如果元素存在，那么返回它所对应的索引
2. 给定有序数组和指定元素，如果元素不存在，那么返回第一个比这个元素大的元素的索引

恰好“二分查找”的正常写法可以同时满足这两个子题目。

### 在排序数组中查找元素的第一个和最后一个位置

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：

你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
 

示例 1：

输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
示例 2：

输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
示例 3：

输入：nums = [], target = 0
输出：[-1,-1]

分析：

其实就是用二分法找到左端点和右端点。

代码：

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int l = 0, r = nums.size(), m;
        int first, last;
        while (l < r)  // 因为 [l, r) 是左闭右开，所以这里不带等号
        {
            m = l + ((r - l) >> 1);
            if (nums[m] < target) l = m + 1;
            else r = m;  // 相等或大于时，收缩右端点
        }
        if (l >= nums.size() || nums[l] != target) return vector<int>({-1, -1});  // 检查是否找到
        first = l;

        l = 0, r = nums.size();
        while (l < r)
        {
            m = l + ((r - l) >> 1);
            if (nums[m] <= target) l = m + 1;  // 相等或小于时，收缩左端点
            else r = m;
        }
        last = l - 1;  // 前面检查过了，这里就不用检查了
        return vector<int>({first, last});
    }
};
```

### 搜索旋转排序数组

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

 

示例 1：

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
示例 2：

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
示例 3：

输入：nums = [1], target = 0
输出：-1

代码：

1. 三次二分查找，先确定旋转点，再在两个有序区间分别查找。（自己写的，效率很低）

    ```c++
    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int k = 0, l = 0, r = nums.size(), m;
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums[l] < nums[m]) l = m;  // 这里不能写成 m+1，否则会漏
                else r = m; 
            }
            k = l;  // 如果 l >= nums.size()，那么相当于直接跳过了下面的第二个循环，因此这里不用判断 l 的范围

            l = 0, r = k+1;  // r 必须写成 k + 1，因为是左闭左开区间 [l, r)
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums[m] < target) l = m + 1;
                else if (nums[m] > target) r = m;
                else return m;
            }

            l = k, r = nums.size();
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums[m] < target) l = m + 1;
                else if (nums[m] > target) r = m;
                else return m;
            }

            return -1;
        }
    };
    ```

1. 只判断 target 是否落在有序序列中

    ```c++
    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int k = 0, l = 0, r = nums.size(), m;
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums[m] == target) return m;
                if (nums[0] <= nums[m])  // m falls into left sequence
                {
                    if (nums[l] <= target && target < nums[m]) r = m;  // 若 target 在左有序序列中，则缩小右端点
                    else l = m + 1;  // 否则缩小左端点
                }
                else  // m falls into right sequence
                {
                    if (nums[m] <= target && target <= nums.back()) l = m + 1;  // 同理，注意第二个小于等于，等号是可以取到的
                    else r = m; 
                }
            }

            return -1;
        }
    };
    ```

### 搜索二维矩阵

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

```
示例：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true

输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false
```

代码：

两次二分查找。

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int l = 0, r = matrix.size(), m;
        while (l < r)
        {
            m = l + ((r - l) >> 1);
            if (matrix[m][0] < target) l = m + 1;
            else r = m;
        }
        if (l < matrix.size() && matrix[l][0] == target) return true;  // l 实际上是插入的索引（参考“搜索插入位置”），因此若能搜索到，直接返回就可以了；否则需要对 l 进行减 1。
        if (l <= 0) return false;  // target 数比第一个数字还小
        int row = l - 1;

        l = 0, r = matrix[row].size();
        while (l < r)
        {
            m = l + ((r - l) >> 1);
            if (matrix[row][m] == target) return true;
            if (matrix[row][m] < target) l = m + 1;
            else r = m;
        }
        return false;
    }
};
```

### 搜索二维矩阵 II

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
 
```
示例 1：

输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
示例 2：


输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

代码：

1. 从右上开始搜索

    ```c++
    class Solution {
    public:
        bool searchMatrix(vector<vector<int>>& matrix, int target) {
            int i = 0, j = matrix[0].size() - 1;
            while (i < matrix.size() && j > -1)
            {
                if (matrix[i][j] < target) ++i;
                else if (matrix[i][j] > target) --j;
                else return true;
            }
            return false;
        }
    };
    ```

### 寻找旋转排序数组中的最小值

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

 
```
示例 1：

输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
示例 2：

输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。
示例 3：

输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
```

代码：

1. 二分查找，根据`nums[0]`判断`m`的位置

    ```c++
    class Solution {
    public:
        int findMin(vector<int>& nums) {
            int l = 0, r = nums.size(), m;
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums[0] <= nums[m]) l = m + 1;  // 等号必须得有，这样当遇到 [4,5,2,3] 这样的时，l 能走到 2 处
                else r = m;
            }
            if (l >= nums.size()) return nums[0];  // l 若超出范围，那么序列是顺序的
            return nums[l];
        }
    };
    ```

1. 二分查找，根据`nums.back()`判断`m`位置

    ```c++
    class Solution {
    public:
        int findMin(vector<int>& nums) {
            int l = 0, r = nums.size(), m;
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums.back() < nums[m]) l = m + 1;  // 和右端点比较时，这里也不用写等号
                else r = m;
            }
            return nums[l];  // 如果和右端点比较，那么 l 一定不会超出范围
        }
    };
    ```

### 寻找峰值

峰值元素是指其值大于左右相邻值的元素。

给你一个输入数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

 
```
示例 1：

输入：nums = [1,2,3,1]
输出：2
解释：3 是峰值元素，你的函数应该返回其索引 2。
示例 2：

输入：nums = [1,2,1,3,5,6,4]
输出：1 或 5 
解释：你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。
```

代码：

1. 二分法，自己写的，有点复杂

    ```c++
    class Solution {
    public:
        int findPeakElement(vector<int>& nums) {
            if (nums.size() == 1) return 0;
            if (nums.size() == 2) return nums[0] > nums[1] ? 0 : 1;
            int l = 0, r = nums.size(), m;
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (m > 0 && m < nums.size()-1 && nums[m-1] < nums[m] && nums[m] > nums[m+1]) return m;
                if (m < nums.size()-1 && nums[m] < nums[m+1]) l = m + 1;
                else r = m;
            }
            return l;
        }
    };
    ```

1. 二分法改进版

    ```c++
    class Solution {
    public:
        int findPeakElement(vector<int>& nums) {
            int l = 0, r = nums.size() - 1, m;  // 这里 r 要减 1，因为下面有 nums[m+1]，所以这样要保证右边界取不到
            while (l < r)
            {
                m = l + ((r - l) >> 1);
                if (nums[m] < nums[m+1]) l = m + 1;
                else r = m;
            }
            return l;
        }
    };
    ```

### 冒泡排序

```c++
void bubble_sort(vector<int> &nums)
{
    for (int i = 0; i < nums.size() - 1; ++i)  // 一共需要执行 n - 1 轮，因为每次执行完一轮，都能保证数组末尾的 i+1 个元素是递增的顺序。我们只需要保证数组末尾 n - 1 个元素是递增的就可以了，这时第一个元素一定是最小的。所以一共需要 n - 1 轮
    {
        for (int j = 0; j < nums.size() - 1 - i; ++j)  // 因为需要比较当前元素和下一个元素，所以 j 不能取到当前区间末尾的元素，对于第一轮，需要比较 n - 1 次，j 的范围为 [0, n-2]；第二轮，最后一个元素已经确定了，此时需要比较 n - 2 次，j 的范围为 [0, n - 3]。以此类推，得到 j < nums.size() - 1 - i。
        {
            if (nums[j] > nums[j+1]) swap(nums[j], nums[j+1]);  // i 只是确定轮数，这里只用到了 j
        }
    }
}
```

### 选择排序

每次遍历数组，选到最小的或最大的元素，放到数组的前面或后面。

```c++
void select_sort(vector<int> &nums)
{
    for (int i = 0; i < nums.size() - 1; ++i)  // 每轮排序完后，前 i + 1 个数的位置已经确定，最后一个数就不需要排了，因此一共需要 n - 1 轮
    {
        int min_idx = i;
        for (int j = i + 1; j < nums.size(); ++j)  // 因为前面已经令 min_idx = i 了，所以 j 不必从 i 开始，直接从 i + 1 开始就可以了
        {
            if (nums[j] < nums[min_idx])
                min_idx = j;
        }
        swap(nums[i], nums[min_idx]);
    }
}
```

### 插入排序

```c++
void insert_sort(vector<int> &nums)
{
    for (int i = 1; i < nums.size(); ++i)  // 一共需要 n - 1 轮，因为第 k 轮能确定前 k 个数的顺序，而最后一个数不需要排序
    {
        int pre_idx = i - 1;
        int cur_num = nums[i];
        while (pre_idx >= 0 && nums[pre_idx] > cur_num)  // 我们需要找到第一个小于等于 cur_num 的数的索引
        {
            nums[pre_idx+1] = nums[pre_idx];  // 把数字往后移一个位置
            --pre_idx;
        }
        nums[pre_idx+1] = cur_num;  // 把当前数字放到小于等于它的数字的后面
    }
}
```

### 希尔排序

```c++
void shell_sort(vector<int> &nums)
{
    int temp, gap = 1;
    while (gap < nums.size() / 3)
        gap = gap * 3 + 1;
    for (; gap > 0; gap = floor(gap / 3))
    {
        for (int i = gap, i < nums.size(); ++i)
        {
            temp = nums[i];
            for (int j = i - gap; j >= 0 && nums[j] > temp; j -= gap)
            {
                nums[j + gap] = nums[j];
            }
            nums[j + gap] = temp;
        }
    }
}
```

有时间了再看吧，一时半会细节消化不完。

### 归并排序

1. 算法导论版本

    ```c++
    void merge(vector<int> &nums, int left, int mid, int right)
    {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        vector<int> L(n1+1), R(n2+1);
        for (int i = 0; i < n1; ++i)
            L[i] = nums[left + i];
        for (int j = 0; j < n2; ++j)
            R[j] = nums[mid + j + 1];
        L[n1] = INT32_MAX;
        R[n2] = INT32_MAX;
        int pos1 = 0, pos2 = 0, pos = 0;
        while (pos <= right)
        {
            if (L[pos1] <= R[pos2]) nums[pos++] = L[pos1++];
            else nums[pos++] = R[pos2++];
        }
    }

    void merge_sort(vector<int> &nums, int left, int right)
    {
        if (left < right)
        {
            int mid = (left + right) >> 1;
            merge_sort(nums, left, mid);
            merge_sort(nums, mid+1, right);
            merge(nums, left, mid, right);
        }
    }
    ```

    另一种写法;

    ```c++
    void merge(vector<int> &nums, int left, int mid, int right)
    {
        int n1 = mid - left + 1;  // 设想 nums = [1, 2, 3], left = 0, mid = 1, right = 2，如果 n1 = 1 - 0 = 1, n2 = 2 - 1 = 1，那么 n1 + n2 就总共有 2 个数而不是 3 个了。所以对于奇数个的数组，我们往左数组多分一个数。
        int n2 = right - mid;
        vector<int> v1(nums.begin()+left, nums.begin()+left+n1), v2(nums.begin()+left+n1, nums.begin()+left+n1+n2);  // 注意赋值时要加上 left
        int pos = left, pos1 = 0, pos2 = 0;  // 注意 pos 是从 left 开始的
        while (pos1 < n1 && pos2 < n2) nums[pos++] = v1[pos1] <= v2[pos2] ? v1[pos1++] : v2[pos2++];  // 按照算法导论上的版本，v1 和 v2 都有一个哨兵值 INT32_MAX，这样我们便可以这样写了：
        // while (pos <= right) nums[pos++] = v1[pos1] <= v2[pos2] ? v1[pos1++] : v2[pos2++];
        // 因为如果 v1 和 v2 中的某个指针到达末尾，会停在哨兵值那里不动，等待着另一个数组把值赋完
        // 但是这个版本里没有用到哨兵值，所以需要另外写两个 while，把剩余的值赋完
        while (pos1 < n1) nums[pos++] = v1[pos1++];
        while (pos2 < n2) nums[pos++] = v2[pos2++];
    }

    void merge_sort(vector<int> &nums, int left, int right)
    {
        if (left >= right) return;
        int mid = (left + right) >> 1;
        merge_sort(nums, left, mid); // 这里并不像快速排序那样每轮都能确定一个中间值，而是把整个数组完整地分成两份，所以这里需要写 mid 而不是 mid - 1
        merge_sort(nums, mid+1, right);
        merge(nums, left, mid, right);  // 深度优先搜索，直到找到左数组或右数组是两个数的时候开始合并
    }
    ```

1. acwing 版本

    ```c++
    vector<int> tmp;
    void merge_sort(vector<int> &q, int l, int r)
    {
        if (l >= r) return;
        int mid = l + r >> 1;
        merge_sort(q, l, mid);
        merge_sort(q, mid+1, r);
        int k = 0, i = l, j = mid + 1;
        while (i <= mid && j <= r)
        {
            if (q[i] <= q[j])
                tmp[k++] = q[i++];
            else
                tmp[k++] = q[j++];
        }

        while (i <= mid)
            tmp[k++] = q[i++];

        while (j <= r)
            tmp[k++] = q[j++];

        for (i = l, j = 0; i <= r; ++i, ++j)
            q[i] = tmp[j];
    }
    ```

1. leetcode 题解版本

    ```c++
    class Solution {
        vector<int> tmp;
        void mergeSort(vector<int>& nums, int l, int r) {
            if (l >= r) return;
            int mid = (l + r) >> 1;
            mergeSort(nums, l, mid);
            mergeSort(nums, mid + 1, r);
            int i = l, j = mid + 1;
            int cnt = 0;
            while (i <= mid && j <= r) {
                if (nums[i] <= nums[j]) {
                    tmp[cnt++] = nums[i++];
                }
                else {
                    tmp[cnt++] = nums[j++];
                }
            }
            while (i <= mid) {
                tmp[cnt++] = nums[i++];
            }
            while (j <= r) {
                tmp[cnt++] = nums[j++];
            }
            for (int i = 0; i < r - l + 1; ++i) {
                nums[i + l] = tmp[i];
            }
        }
    public:
        vector<int> sortArray(vector<int>& nums) {
            tmp.resize((int)nums.size(), 0);
            mergeSort(nums, 0, (int)nums.size() - 1);
            return nums;
        }
    };
    ```

### Quicksort（快速排序）

快速排序的思想是分治，找到一个数`x`，通过一些操作，使得`x`左边的数都小于等于`x`，`x`右边的数都大于等于`x`。然后再对`x`左边的数执行同样的操作，对`x`右边的数也执行同样的操作。直到需要操作的数只剩一个，就可以结束递归了。

1. 算法导论上的实现方式

    ```c++
    class Solution {
    public:
        int partition(vector<int> &nums, int left, int right)
        {
            int x = nums[right];
            int i = left - 1, j = left;
            while (j < right)  // right 是哨兵节点，所以不能取到
            {
                if (nums[j] < x) swap(nums[++i], nums[j]);  // 这里会做很多的原地交换，不如令 i = left，若检测到 i == j，则跳过
                ++j;
            }
            swap(nums[i+1], nums[right]);
            return i+1;
        }

        void quick_sort(vector<int> &nums, int left, int right)
        {
            if (left >= right) return;  // 这里的等号可以取到吗？
            int i = partition(nums, left, right);
            quick_sort(nums, left, i-1);
            quick_sort(nums, i+1, right);
        }

        vector<int> sortArray(vector<int>& nums) {
            quick_sort(nums, 0, nums.size()-1);
            return nums;
        }
    };
    ```

    当`nums`已经排好序时，`quick_sort(nums, left, i-1)`会被反复调用，以至于超时。

    后来又统计了下`while`循环的执行次数，和 acwing 上的调用次数差不多，所以超时好像并不是做了冗余的操作。我又懵了。

1. leetcode 上的实现方式

    为了避免特殊情况下的超时问题，leetcode 将`nums[right]`修改为了随机取值。

    ```c++
    class Solution {
    public:
        int partition(vector<int> &nums, int left, int right)
        {
            int x = nums[right];
            int i = left - 1, j = left;
            while (j < right)
            {
                if (nums[j] < x)
                {
                    ++i;
                    swap(nums[i], nums[j]);
                }
                ++j;
            }
            swap(nums[++i], nums[right]);
            return i;
        }

        int randomize_partition(vector<int> &nums, int left, int right)
        {
            int idx = rand() % (right - left + 1) + left;
            swap(nums[idx], nums[right]);
            return partition(nums, left, right);
        }

        void quick_sort(vector<int> &nums, int left, int right)
        {
            if (left >= right) return;
            int i = randomize_partition(nums, left, right);
            quick_sort(nums, left, i-1);
            quick_sort(nums, i+1, right);
        }

        vector<int> sortArray(vector<int>& nums) {
            quick_sort(nums, 0, nums.size()-1);
            return nums;
        }
    };
    ```

1. acwing 上的实现方式

    ```c++
    class Solution {
    public:
        void quick_sort(vector<int> &nums, int left, int right)
        {
            if (left < right)
            {
                int x = nums[(left + right) >> 1];
                int i = left - 1, j = right + 1;
                while (i < j)
                {
                    do ++i; while (nums[i] < x);
                    do --j; while (nums[j] > x);
                    if (i < j) swap(nums[i], nums[j]);
                }
                quick_sort(nums, left, j);  // 这里必须写成 j，不能是 j - 1，为什么？
                quick_sort(nums, j+1, right);
            }
        }
        vector<int> sortArray(vector<int>& nums) {
            quick_sort(nums, 0, nums.size()-1);
            return nums;
        }
    };
    ```

1. 取 left 作为 pivot 的实现方式

    ```c++
    //严蔚敏《数据结构》标准分割函数
    Paritition1(int A[], int low, int high) {
    int pivot = A[low];
    while (low < high) {
        while (low < high && A[high] >= pivot) {
        --high;
        }
        A[low] = A[high];
        while (low < high && A[low] <= pivot) {
        ++low;
        }
        A[high] = A[low];
    }
    A[low] = pivot;
    return low;
    }

    void QuickSort(int A[], int low, int high) //快排母函数
    {
    if (low < high) {
        int pivot = Paritition1(A, low, high);
        QuickSort(A, low, pivot - 1);
        QuickSort(A, pivot + 1, high);
    }
    }
    ```

1. 某个博客上的实现

    ```c++
    /**
     * 快速排序：C++
     *
     * @author skywang
     * @date 2014/03/11
     */

    #include <iostream>
    using namespace std;

    /*
    * 快速排序
    *
    * 参数说明：
    *     a -- 待排序的数组
    *     l -- 数组的左边界(例如，从起始位置开始排序，则l=0)
    *     r -- 数组的右边界(例如，排序截至到数组末尾，则r=a.length-1)
    */
    void quickSort(int* a, int l, int r)
    {
        if (l < r)
        {
            int i,j,x;

            i = l;
            j = r;
            x = a[i];
            while (i < j)
            {
                while(i < j && a[j] > x)
                    j--; // 从右向左找第一个小于x的数
                if(i < j)
                    a[i++] = a[j];
                while(i < j && a[i] < x)
                    i++; // 从左向右找第一个大于x的数
                if(i < j)
                    a[j--] = a[i];
            }
            a[i] = x;
            quickSort(a, l, i-1); /* 递归调用 */
            quickSort(a, i+1, r); /* 递归调用 */
        }
    }

    int main()
    {
        int i;
        int a[] = {30,40,60,10,20,50};
        int ilen = (sizeof(a)) / (sizeof(a[0]));

        cout << "before sort:";
        for (i=0; i<ilen; i++)
            cout << a[i] << " ";
        cout << endl;

        quickSort(a, 0, ilen-1);

        cout << "after  sort:";
        for (i=0; i<ilen; i++)
            cout << a[i] << " ";
        cout << endl;

        return 0;
    }
    ```

### 堆排序

1. leetcode `写代码的火车`版本

    这里维护的是小顶堆。这个版本挺不错的，清晰易懂。

    ```c++
    class Solution {
    public:
        vector<int> h;
        int s;  // 统计堆中元素的数量，亦是最后一个节点的下标

        void down(int x)
        {
            int t = x;  // 这个地方很巧妙，主要有两个作用，一是两次判断父节点该与哪个子节点交换，二是判断是否需要交换。
            if (2 * x <= s && h[t] > h[2 * x]) t = 2 * x;  // 如果左子节点存在，那么判断是否需要和左节点交换位置
            if (2 * x + 1 <= s && h[t] > h[2 * x + 1]) t = 2 * x + 1;  // 如果右子节点存在，那么判断是否需要和右节点交换位置
            if (t != x) swap(h[x], h[t]), down(t);  // 递归调用，直到该元素沉到它该去的地方（为何不能循环调用呢）
        }

        void up(int x)
        {
            while (x / 2 >= 1 && h[x] < h[x / 2])  // 如果父节点存在，那么判断是否应该交换位置
            {
                swap(h[x], h[x / 2]);
                x /= 2;  // 迭代上升
            }
        }

        void push(int num)  // 插入节点其实就是往数组末尾加入一个节点，然后让它上升就好了
        {
            h[++s] = num;
            up(s);
        }

        void pop()  // 删除一个节点就是把头尾节点交换位置，然后减少数组长度，再让头节点下降
        {
            swap(h[1], h[s--]);  // 为了保证 左节点的索引是 2 * x，右节点的索引是 2 * x + 1，起始索引从 1 开始
            down(1);
        }

        int top()
        {
            return h[1];
        }

        vector<int> sortArray(vector<int>& nums) {
            h.resize(50001);  // 比题目要求的最长数组长度再多一个数，因为索引从 1 开始
            s = 0;  // 起始时节点的数量为 0

            for (int i = 0; i < nums.size(); ++i)
                push(nums[i]);

            for (int i = 0; i < nums.size(); ++i)
            {
                nums[i] = top();
                pop();
            }
            return nums;
        }
    };
    ```

1. github js 版本

    ```c++
    var len;    // 因为声明的多个函数都需要数据长度，所以把len设置成为全局变量

    function buildMaxHeap(arr) {   // 建立大顶堆
        len = arr.length;
        for (var i = Math.floor(len/2); i >= 0; i--) {
            heapify(arr, i);
        }
    }

    function heapify(arr, i) {     // 堆调整
        var left = 2 * i + 1,
            right = 2 * i + 2,
            largest = i;

        if (left < len && arr[left] > arr[largest]) {
            largest = left;
        }

        if (right < len && arr[right] > arr[largest]) {
            largest = right;
        }

        if (largest != i) {
            swap(arr, i, largest);
            heapify(arr, largest);
        }
    }

    function swap(arr, i, j) {
        var temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    function heapSort(arr) {
        buildMaxHeap(arr);

        for (var i = arr.length-1; i > 0; i--) {
            swap(arr, 0, i);
            len--;
            heapify(arr, 0);
        }
        return arr;
    }
    ```

1. leetcode 题解版本

    ```c++
    class Solution {
        void maxHeapify(vector<int>& nums, int i, int len) {
            for (; (i << 1) + 1 <= len;) {
                int lson = (i << 1) + 1;
                int rson = (i << 1) + 2;
                int large;
                if (lson <= len && nums[lson] > nums[i]) {
                    large = lson;
                } else {
                    large = i;
                }
                if (rson <= len && nums[rson] > nums[large]) {
                    large = rson;
                }
                if (large != i) {
                    swap(nums[i], nums[large]);
                    i = large;
                } else {
                    break;
                }
            }
        }
        void buildMaxHeap(vector<int>& nums, int len) {
            for (int i = len / 2; i >= 0; --i) {
                maxHeapify(nums, i, len);
            }
        }
        void heapSort(vector<int>& nums) {
            int len = (int)nums.size() - 1;
            buildMaxHeap(nums, len);
            for (int i = len; i >= 1; --i) {
                swap(nums[i], nums[0]);
                len -= 1;
                maxHeapify(nums, 0, len);
            }
        }
    public:
        vector<int> sortArray(vector<int>& nums) {
            heapSort(nums);
            return nums;
        }
    };
    ```

### 计数排序

1. github js 版

    这种方式似乎没有统计最大值和最小值，所以只能用于数字都是非负整数的情况

    ```c++
    function countingSort(arr, maxValue) {
        var bucket = new Array(maxValue+1),
            sortedIndex = 0;
            arrLen = arr.length,
            bucketLen = maxValue + 1;

        for (var i = 0; i < arrLen; i++) {
            if (!bucket[arr[i]]) {  // 这个 if 语句有啥用？答：bucket[arr[i]]可能为 empty。JS 数组类似于 python，可以放入不同类型的元素
                bucket[arr[i]] = 0;
            }
            bucket[arr[i]]++;
        }

        for (var j = 0; j < bucketLen; j++) {
            while(bucket[j] > 0) {
                arr[sortedIndex++] = j;
                bucket[j]--;
            }
        }

        return arr;
    }
    ```

1. 统计最值版

    这个速度还是挺快的。

    ```c++
    class Solution {
    public:
        vector<int> sortArray(vector<int>& nums) {
            int min_val = INT32_MAX, max_val = INT32_MIN;
            for (int i = 0; i < nums.size(); ++i)
            {
                min_val = min(nums[i], min_val);
                max_val = max(nums[i], max_val);
            }

            vector<int> count(max_val - min_val + 1);
            for (int i = 0; i < nums.size(); ++i)
                ++count[nums[i] - min_val];
                
            int pos = 0, pos_cnt = 0;
            while (pos_cnt < count.size())
            {
                while (count[pos_cnt]--) nums[pos++] = min_val + pos_cnt;
                ++pos_cnt;
            }
            return nums;
        }
    };
    ```

### 桶排序

github js 代码：

```js
function bucketSort(arr, bucketSize) {
    if (arr.length === 0) {
      return arr;
    }

    var i;
    var minValue = arr[0];
    var maxValue = arr[0];
    for (i = 1; i < arr.length; i++) {
      if (arr[i] < minValue) {
          minValue = arr[i];                // 输入数据的最小值
      } else if (arr[i] > maxValue) {
          maxValue = arr[i];                // 输入数据的最大值
      }
    }

    //桶的初始化
    var DEFAULT_BUCKET_SIZE = 5;            // 设置桶的默认数量为5
    bucketSize = bucketSize || DEFAULT_BUCKET_SIZE;
    var bucketCount = Math.floor((maxValue - minValue) / bucketSize) + 1;   
    var buckets = new Array(bucketCount);
    for (i = 0; i < buckets.length; i++) {
        buckets[i] = [];
    }

    //利用映射函数将数据分配到各个桶中
    for (i = 0; i < arr.length; i++) {
        buckets[Math.floor((arr[i] - minValue) / bucketSize)].push(arr[i]);
    }

    arr.length = 0;
    for (i = 0; i < buckets.length; i++) {
        insertionSort(buckets[i]);                      // 对每个桶进行排序，这里使用了插入排序
        for (var j = 0; j < buckets[i].length; j++) {
            arr.push(buckets[i][j]);                      
        }
    }

    return arr;
}
```

### 基数排序

github python 版本：

```py
def radix(arr):
    
    digit = 0
    max_digit = 1
    max_value = max(arr)
    #找出列表中最大的位数
    while 10**max_digit < max_value:
        max_digit = max_digit + 1
    
    while digit < max_digit:
        temp = [[] for i in range(10)]
        for i in arr:
            #求出每一个元素的个、十、百位的值
            t = int((i/10**digit)%10)
            temp[t].append(i)
        
        coll = []
        for bucket in temp:
            for i in bucket:
                coll.append(i)
                
        arr = coll
        digit = digit + 1

    return arr
```

### 前 K 个高频元素

给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

 
```
示例 1:

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]
```

1. 堆

1. 快速选择

    先用哈希表统计各个元素出现的数量，再将它们以`pair`的形式放到`vector`里，然后再对`vector`中元素的`second`进行快速选择。

    下面这个解法是我自己写的，没有用哈希表做统计，而是做了位置映射。。。差不多吧，效果。

    ```c++
    class Solution {
    public:
        int partition(vector<pair<int, int>> &nums, int left, int right)
        {
            int x = nums[right].second;
            int i = left - 1, j = left;
            while (j < right)
            {
                if (nums[j].second > x) swap(nums[++i], nums[j]);
                ++j;
            }
            swap(nums[i+1], nums[right]);
            return i+1;
        }

        int randomized_partition(vector<pair<int, int>> &nums, int left, int right)
        {
            int idx = rand() % (right - left + 1) + left;
            swap(nums[idx], nums[right]);
            return partition(nums, left, right);
        }

        void quick_select(vector<pair<int, int>> &nums, int left, int right, int k)
        {
            if (left >= right) return;
            int idx = randomized_partition(nums, left, right);
            if (idx < k) quick_select(nums, idx+1, right, k);
            else if (idx > k) quick_select(nums, left, idx-1, k);
            else return;
        }

        vector<int> topKFrequent(vector<int>& nums, int k) {
            srand(time(NULL));
            unordered_map<int, int> m;
            vector<pair<int, int>> v;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (m.find(nums[i]) == m.end())
                {
                    m[nums[i]] = v.size();
                    v.push_back(make_pair(nums[i], 1));
                }
                else
                    v[m[nums[i]]].second++;
            }
            quick_select(v, 0, v.size()-1, k-1);
            vector<int> ans;
            for (int i = 0; i < k; ++i)
            {
                ans.push_back(v[i].first);
            }
            return ans;
        }
    };
    ```

### 根据字符出现频率排序

给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

```
示例 1:

输入:
"tree"

输出:
"eert"

解释:
'e'出现两次，'r'和't'都只出现一次。
因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
示例 2:

输入:
"cccaaa"

输出:
"cccaaa"

解释:
'c'和'a'都出现三次。此外，"aaaccc"也是有效的答案。
注意"cacaca"是不正确的，因为相同的字母必须放在一起。
示例 3:

输入:
"Aabb"

输出:
"bbAa"

解释:
此外，"bbaA"也是一个有效的答案，但"Aabb"是不正确的。
注意'A'和'a'被认为是两种不同的字符。
```

代码：

1. 自己写的快速排序（因为不是快速选择，所以还不如用库函数`sort()`）

    ```c++
    class Solution {
    public:
        int partition(vector<pair<int, int>> &nums, int left, int right)
        {
            int x = nums[right].second;
            int i = left - 1, j = left;
            while (j < right)
            {
                if (nums[j].second > x) swap(nums[++i], nums[j]);
                ++j;
            }
            swap(nums[i+1], nums[right]);
            return i+1;
        }

        int randomized_partition(vector<pair<int, int>> &nums, int left, int right)
        {
            int idx = rand() % (right - left + 1) + left;
            swap(nums[idx], nums[right]);
            return partition(nums, left, right);
        }

        void quick_sort(vector<pair<int, int>> &nums, int left, int right)
        {
            if (left >= right) return;
            int idx = randomized_partition(nums, left, right);
            quick_sort(nums, left, idx - 1);
            quick_sort(nums, idx + 1, right);
        }

        string frequencySort(string s) {
            vector<pair<int, int>> cnt(128);
            for (int i = 0; i < 128; ++i)
                cnt[i].first = i;

            for (int i = 0; i < s.size(); ++i)
                ++cnt[s[i]].second;
            
            quick_sort(cnt, 0, cnt.size() - 1);
            int p = 0;
            int n;
            for (int i = 0; i < 128; ++i)
            {
                n = cnt[i].second;
                while (n--) s[p++] = cnt[i].first;
            }
            return s;
        }
    };
    ```

1. 官方答案，桶排序

    因为字符最多也就 128 个，所以用桶排序的效率更高。

    ```c++
    class Solution {
    public:
        string frequencySort(string s) {
            unordered_map<char, int> mp;
            int maxFreq = 0;
            int length = s.size();
            for (auto &ch : s) {
                maxFreq = max(maxFreq, ++mp[ch]);
            }
            vector<string> buckets(maxFreq + 1);
            for (auto &[ch, num] : mp) {
                buckets[num].push_back(ch);
            }
            string ret;
            for (int i = maxFreq; i > 0; i--) {
                string &bucket = buckets[i];
                for (auto &ch : bucket) {
                    for (int k = 0; k < i; k++) {
                        ret.push_back(ch);
                    }
                }
            }
            return ret;
        }
    };
    ```

1. 网友给出的一种方法，但是所说效率很低，按道理不应该，有时间调查一下为啥

    ```c++
    class Solution {
    public:
        string frequencySort(string s) {
            unordered_map<char, int> mp;
            for(auto ch:s) mp[ch]++;
            sort(s.begin(),s.end(),[&](const char &a, const char &b){
                return mp[a]==mp[b] ? a>b : mp[a]>mp[b];
            });
            return s;
        }
    };
    ```

### 最接近原点的 K 个点

我们有一个由平面上的点组成的列表 points。需要从中找出 K 个距离原点 (0, 0) 最近的点。

（这里，平面上两点之间的距离是欧几里德距离。）

你可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。

 

```
示例 1：

输入：points = [[1,3],[-2,2]], K = 1
输出：[[-2,2]]
解释： 
(1, 3) 和原点之间的距离为 sqrt(10)，
(-2, 2) 和原点之间的距离为 sqrt(8)，
由于 sqrt(8) < sqrt(10)，(-2, 2) 离原点更近。
我们只需要距离原点最近的 K = 1 个点，所以答案就是 [[-2,2]]。
示例 2：

输入：points = [[3,3],[5,-1],[-2,4]], K = 2
输出：[[3,3],[-2,4]]
（答案 [[-2,4],[3,3]] 也会被接受。）
```

代码：

普普通通的一道题，可以用排序，堆，快速选择

1. 排序

    ```c++
    class Solution {
    public:
        vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
            int n = points.size();
            vector<pair<int, int>> dist(n);
            for (int i = 0; i < n; ++i)
            {
                dist[i].first = i;
                dist[i].second = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            }
            sort(dist.begin(), dist.end(), [](pair<int, int> &a, pair<int, int> &b){
                return a.second < b.second;
            });

            vector<vector<int>> ans(k, vector<int>(2));
            for (int i = 0; i < k; ++i)
                ans[i] = points[dist[i].first];
            return ans;
        }
    };
    ```

1. 堆

1. 快速选择

### 双指针

#### 有序数组的平方

给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。


示例 1：

```
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]
```

代码：

双指针逆序。

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        vector<int> res(nums.size());
        int pos = nums.size() - 1;
        while (l <= r)
        {
            if (abs(nums[l]) < abs(nums[r]))
            {
                res[pos] = nums[r] * nums[r];
                --r;
            }
            else
            {
                res[pos] = nums[l] * nums[l];
                ++l;
            }
            --pos;
        }
        return res;
    }
};
```

#### 旋转数组

给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

 

进阶：

尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？
 

示例 1:

```
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]
```

代码：

1. 朴素法

    ```c++
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            k = k % nums.size();  // k 有可能比 nums.size() 大
            vector<int> res(nums.size());
            int rev_pos = nums.size() - k;
            int pos = 0;
            while (rev_pos < nums.size())
                res[pos++] = nums[rev_pos++];
            int for_pos = 0;
            while (pos < nums.size())
                res[pos++] = nums[for_pos++];
            nums = res;
        }
    };
    ```

    或者

    ```c++
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            int n = nums.size();
            vector<int> newArr(n);
            for (int i = 0; i < n; ++i) {
                newArr[(i + k) % n] = nums[i];
            }
            nums.assign(newArr.begin(), newArr.end());
        }
    };
    ```

1. 调库法（这个应该是中等难度的解法）

    ```c++
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            k = k % nums.size();
            reverse(nums.begin(), nums.end());
            reverse(nums.begin(), nums.begin() + k);
            reverse(nums.begin() + k, nums.end());
        }
    };
    ```

    这道题之所以被归入双指针，是因为如果我们手写`reverse()`函数的话，需要这样写：

    ```c++
    void reverse(vector<int> &nums, int i, int j)
    {
        while (i < j)
        {
            swap(nums[i], nums[j]);
            ++i;
            --j;
        }
    }
    ```

1. 循环替换法

    设想把这个数组看成个周期函数，每个数字的位置都会变到`(pos + k) % nums.size()`的新位置。但是仅仅这样会导致循环，有些数字遍历不到，所以我们设置起始位置，如果每个数字都已经遍历一遍了，那么就退出循环。

    ```c++
    class Solution {
    public:
        void rotate(vector<int>& nums, int k) {
            if (k == 0) return;
            int count = 0, pos = 0, start_pos = 0;
            int temp1 = nums[0], temp2;
            while (count < nums.size())
            {
                pos = (pos + k) % nums.size();
                temp2 = nums[pos];
                nums[pos] = temp1;
                temp1 = temp2;
                ++count;

                if (pos == start_pos)
                {
                    start_pos++;
                    if (start_pos >= nums.size()) return;  // 防止起始位置超出数组长度
                    pos = start_pos;
                    temp1 = nums[pos];
                }
            }
        }
    };
    ```



#### 两数之和 II - 输入有序数组（排序数组中两个数字之和）

给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。

你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

 
示例 1：

```
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

代码：

可以用双指针，也可以用二分查找。不过推荐双指针，因为实现简单。

如果要在有序数组中找到和为定值的两个数，那么可以用双指针。为什么双指针一定能找到解？我们可以做这样的证明：

假设有这样的数组：`[a ... b ... c, d ... e ... f]`，我们要找的解为`[b, e]`。双指针分别从`a`，`f`开始向内遍历，总会有一侧指针先到达`b`或者先到达`e`。我们假设右侧指针先到达`e`，此时左侧指针一定在`b`的左边（假如左侧指针在`b`上，或到了`b`的右边，那么就和我们的假设相违背了。），此时左侧指针指向的元素与右侧指针指向的元素之和是小于 target 的。根据算法，此时只能继续移动左侧指针，直到和等于 target。因此双指针法一定会找到正确的两个数。

（我只能证明双指针是正确的，但没办法知道它是怎么通过线性和非线性思维被想出来的）

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int l = 0, r = numbers.size() - 1;
        int sum;
        while (l < r)
        {
            sum = numbers[l] + numbers[r];
            if (sum == target) return vector<int>({l+1, r+1});
            else if (sum > target) --r;
            else ++l;
        }
        return vector<int>();
    }
};
```

#### 反转字符串中的单词 III

给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。


示例：

```
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
```

代码：

1. 对每个单词进行双指针倒序就可以了。。

    ```c++
    class Solution {
    public:
        string reverseWords(string s) {
            int l = 0, r = 0, pos = 0;
            do
            {
                l = pos;
                r = pos;
                while (r < s.size() && s[r] != ' ') ++r;
                pos = r + 1;
                --r;
                while (l < r)
                    swap(s[l++], s[r--]);
            } while (pos < s.size());
            return s;
        }
    };
    ```

1. 后来写的，虽然不简洁，但是更好懂了

    ```c++
    class Solution {
    public:
        void reverse(string &s, int start, int end)
        {
            while (start < end)
            {
                swap(s[start], s[end]);
                ++start;
                --end;
            }
        }

        string reverseWords(string s) {
            int start = 0, end = 0;
            while (end < s.size())
            {
                while (end < s.size() && s[end] != ' ') ++end;
                reverse(s, start, end-1);
                start = end + 1;
                ++end;
            }
            reverse(s, start, end-1);
            return s;
        }
    };
    ```

    这道题虽然简单，但是其中的双重循环，已经有复杂循环的雏形了。如果把内层的`while()`改成`if`，然后借用外层`while`进行循环，又该怎么写呢？

#### 链表的中间节点

给定一个头结点为 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

 

示例 1：

```
输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
```

代码：

1. 快慢指针

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
        ListNode* middleNode(ListNode* head) {
            ListNode *slow = head, *fast = head;
            while (fast && fast->next)  // 无论两步走到尾，还是一步走到尾，都直接停止
            {
                slow = slow->next;
                fast = fast->next->next;
            }
            return slow;
        }
    };
    ```

    有几个问题：

    1. 为什么不从`dummy_head`开始？如果从`dummy_head`开始，会出现什么问题？

    1. 如何保证`fast`停下的时候，`slow`会停在中点，或中点的下一个节点，而不是中点的上一个节点？

#### 删除链表的倒数第 N 个结点

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

代码：

1. vector 版

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
        ListNode* removeNthFromEnd(ListNode* head, int n) {
            ListNode *dummy_head = new ListNode(0, head);
            vector<ListNode*> v;
            ListNode *p = dummy_head;
            while (p)
            {
                v.push_back(p);
                p = p->next;
            }
            p = v[v.size() - n - 1];
            p->next = p->next->next;
            return dummy_head->next;
        }
    };
    ```

1. 双指针版

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
        ListNode* removeNthFromEnd(ListNode* head, int n) {
            ListNode *dummy_head = new ListNode(0, head);
            ListNode *slow = dummy_head, *fast = dummy_head;
            while (n--) fast = fast->next;
            while (fast->next)  // 这里很有意思，如果写成 while (fast)，那么循环结束时，slow 正好在要删除的节点上。但是我们需要的是它的上一个节点，所以这里 fast 并没有走到空值，而是提前停了下来
            {
                slow = slow->next;
                fast = fast->next;
            }
            slow->next = slow->next->next;
            return dummy_head->next;
        }
    };
    ```

    （可能会处理到头节点时，通常使用`dummy_head`使代码逻辑更简洁一些）

    除了使用`while (fast->next)`保证`slow`正好指在待删除节点的上一个节点，还可以使用`ListNode *fast = head;`或者先`++n;`，再`while (n--)`，这两种方式也能实现同样的效果。

    那么问题来了：能否找到一种统一的模型，来表示某个指针走了多少步后，处于哪个位置，和`slow`之间的关系？

1. 扫描两遍链表

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
        ListNode* removeNthFromEnd(ListNode* head, int n) {
            int num = 0;
            ListNode *p = head;
            while (p)
            {
                ++num;
                p = p->next;
            }

            ListNode *dummy_head = new ListNode(0, head);
            p = dummy_head;
            while (num - n)
            {
                p = p->next;
                --num;
            }

            p->next = p->next->next;
            return dummy_head->next;
        }
    };
    ```

#### 三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
示例 2：

输入：nums = []
输出：[]
示例 3：

输入：nums = [0]
输出：[]
```

代码：

对数组排序，然后将某个数`nums[i]`的相反数作为`target`，只需要在剩下的数组中找到两个数和为`target`就可以了。问题就转化成`两数之和 II - 输入有序数组`。此时可以用双指针搜索。

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        int left, right, target;
        int n = nums.size();  // 注意这里啊，假如 nums.size() == 0，到了下面 nums.size() - 1 会变成一个很大的数，这里做了 unsigned 到 signed 的转换，才能保证不错！所以这里很重要。
        int start = 0;
        while (start < n - 2)  // 因为后面要用到 nums[left]，nums[left+1] 以及 nums[right]，nums[right-1]，所以至少需要 3 个数。其实也可以写成 start < n，因为下面的 while (left < right) 保证了不会出问题。只不过在剪枝时，要写成
        // if (left < n - 1 && nums[left] + nums[left+1] > target ||
        //     right > start + 1 && nums[right] + nums[right-1] < target)
        // {
        //     do ++start; 
        //     while (start < n && nums[start] == nums[start-1]);
        //     continue;
        // }
        {
            left = start + 1;
            right = nums.size() - 1;
            target = -nums[start];
            if (num[start] > 0) return ans; // 剪枝
            if (nums[start]+nums[left]+nums[left+1] > 0 ||
                nums[start]+nums[right]+nums[right-1] < 0)   // 剪枝
            {
                do ++start; while (start < n && nums[start] == nums[start-1]);
                continue;
            }
            
            while (left < right)
            {
                if (nums[left] + nums[right] == target)
                {
                    ans.push_back(vector<int>({nums[start], nums[left], nums[right]}));
                    do ++left; while (left < right && nums[left] == nums[left-1]);
                }
                else if (nums[left] + nums[right] < target)  // 若相加偏小，则向右移动左指针
                    do ++left; while (left < right && nums[left] == nums[left-1]);  // 略过重复项，注意 left < right 这个细节，不然左指针会跑到数组末尾
                else  // 若相加偏大，则向左移动右指针
                    do --right; while (left < right && nums[right] == nums[right+1]);
            }
            do ++start; while (start < n && nums[start] == nums[start-1]);
        }
        return ans;
    }
};
```

问题：这道题如果不排序，可以使用回溯算法做出来吗？

#### 乘积小于K的子数组

给定一个正整数数组 nums和整数 k 。

请找出该数组内乘积小于 k 的连续的子数组的个数。

 
```
示例 1:

输入: nums = [10,5,2,6], k = 100
输出: 8
解释: 8个乘积小于100的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
示例 2:

输入: nums = [1,2,3], k = 0
输出: 0
```

代码：

滑动窗口。先扩展右端点，当不满足要求时，收缩左端点。

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        if (k <= 1) return 0;  // 当 k == 1 或 0 时，left 会不断向右走，导致 left > right，ans 变成负数
        int ans = 0;
        int left = 0, right = 0, prod = 1;
        while (right < nums.size())
        {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            ans += right - left + 1;
            ++right;
        }
        return ans;
    }
};
```

#### 长度最小的子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

 
```
示例 1：

输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
示例 2：

输入：target = 4, nums = [1,4,4]
输出：1
示例 3：

输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
```

代码：

滑动窗口。代码同“乘积小于K的子数组”几乎一模一样。

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0, right = 0, sum = 0;
        int ans = INT32_MAX;
        while (right < nums.size())
        {
            sum += nums[right];
            while (sum >= target)
            {
                ans = min(ans, right - left + 1);
                sum -= nums[left++];
            }
            ++right;
        }
        return ans == INT32_MAX ? 0 : ans;
    }
};
```

#### 四数之和

给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按 任意顺序 返回答案 。

 
```
示例 1：

输入：nums = [1,0,-1,0,-2,2], target = 0
输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
示例 2：

输入：nums = [2,2,2,2,2], target = 8
输出：[[2,2,2,2]]
```

代码：

1. 模仿三数之和，自己写的

    ```c++
    class Solution {
    public:
        vector<vector<int>> fourSum(vector<int>& nums, int target) {
            sort(nums.begin(), nums.end());
            vector<vector<int>> ans;
            for (int i = 0; i < nums.size(); ++i)
            {
                if (i > 0 && nums[i] == nums[i-1]) continue;  // 跳过重复数字
                int t1 = nums[i];
                for (int j = i + 1; j < nums.size(); ++j)
                {
                    if (j > i + 1 && nums[j] == nums[j-1]) continue;  // 跳过重复数字
                    int t2 = nums[j];
                    int t = target - t1 - t2;
                    int left = j + 1, right = nums.size() - 1;
                    while (left < right)
                    {
                        int sum = nums[left] + nums[right];
                        if (sum < t)
                            do ++left; while (left < right && nums[left] == nums[left - 1]);
                        else if (sum > t)
                            do --right; while (left < right && nums[right] == nums[right + 1]);
                        else
                        {
                            ans.push_back(vector<int>({t1, t2, nums[left], nums[right]}));
                            do ++left; while (left < right && nums[left] == nums[left - 1]);
                        }
                    }
                }
            }
            return ans;
        }
    };
    ```

1. 带上剪枝的官方答案

    ```c++
    class Solution {
    public:
        vector<vector<int>> fourSum(vector<int>& nums, int target) {
            vector<vector<int>> quadruplets;
            if (nums.size() < 4) {
                return quadruplets;
            }
            sort(nums.begin(), nums.end());
            int length = nums.size();
            for (int i = 0; i < length - 3; i++) {
                if (i > 0 && nums[i] == nums[i - 1]) {
                    continue;
                }
                if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                    break;
                }
                if ((long) nums[i] + nums[length - 3] + nums[length - 2] + nums[length - 1] < target) {
                    continue;
                }
                for (int j = i + 1; j < length - 2; j++) {
                    if (j > i + 1 && nums[j] == nums[j - 1]) {
                        continue;
                    }
                    if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) {
                        break;
                    }
                    if ((long) nums[i] + nums[j] + nums[length - 2] + nums[length - 1] < target) {
                        continue;
                    }
                    int left = j + 1, right = length - 1;
                    while (left < right) {
                        int sum = nums[i] + nums[j] + nums[left] + nums[right];
                        if (sum == target) {
                            quadruplets.push_back({nums[i], nums[j], nums[left], nums[right]});
                            while (left < right && nums[left] == nums[left + 1]) {
                                left++;
                            }
                            left++;
                            while (left < right && nums[right] == nums[right - 1]) {
                                right--;
                            }
                            right--;
                        } else if (sum < target) {
                            left++;
                        } else {
                            right--;
                        }
                    }
                }
            }
            return quadruplets;
        }
    };
    ```

### 深度优先搜索 / 广度优先搜索

#### 图像渲染

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。

给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。

为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。

最后返回经过上色渲染后的图像。

示例 1:

输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。

代码：

1. 广度优先搜索

    ```c++
    class Solution {
    public:
        vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
            int old_color = image[sr][sc];
            if (old_color == newColor) return image;  // 这行不要忘，否则会死循环，因为无法标记哪些处理过了，哪些未处理
            const int dx[4] = {1, 0, 0, -1};
            const int dy[4] = {0, -1, 1, 0};
            queue<pair<int, int>> q;
            q.push(make_pair(sr, sc));
            image[sr][sc] = newColor;  // 如果选择了在发现新位置时就染色，那么需要额外处理第一个位置
            int x, y;
            while (!q.empty())
            {
                sr = q.front().first;
                sc = q.front().second;
                q.pop();
                for (int i = 0; i < 4; ++i)
                {
                    x = sr + dx[i];
                    y = sc + dy[i];
                    if (x >= 0 && x < image.size() && y >= 0 && y < image[0].size() && image[x][y] == old_color)
                    {
                        image[x][y] = newColor;  // 我觉得在这里染色是最科学的，如果使用 sr, sc 染色，那么还需要额外考虑重复搜索的问题
                        q.push(make_pair(x, y));
                    }
                }
            }
            return image;
        }
    };
    ```

1. 深度优先搜索

    ```c++
    class Solution {
    public:
        const int dx[4] = {1, 0, 0, -1};
        const int dy[4] = {0, -1, 1, 0};
        int old_color;
        int new_color;

        void dfs(vector<vector<int>> &image, int sr, int sc)
        {
            image[sr][sc] = new_color;
            int x, y;
            for (int i = 0; i < 4; ++i)
            {
                x = sr + dx[i];
                y = sc + dy[i];
                if (x >= 0 && x < image.size() && y >= 0 && y < image[0].size() && image[x][y] == old_color)
                {
                    dfs(image, x, y);
                }
            }
        }

        vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
            old_color = image[sr][sc];
            new_color = newColor;
            if (old_color == new_color) return image;
            dfs(image, sr, sc);
            return image;
        }
    };
    ```

    这样写`dfs()`也挺不错的：

    ```c++
    void dfs(vector<vector<int>> &image, int sr, int sc)
    {
        if (sr < 0 || sr >= image.size() || sc < 0 || sc >= image[0].size() || image[sr][sc] != old_color)
            return;

        image[sr][sc] = new_color;

        dfs(image, x+1, y);
        dfs(image, x-1, y);
        dfs(image, x, y+1);
        dfs(image, x, y-1);
    }
    ```

#### 岛屿的最大面积

给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

 

示例 1:

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1 。
```

1. 广度优先

    ```c++
    class Solution {
    public:
        int maxAreaOfIsland(vector<vector<int>>& grid) {
            const int dx[4] = {0, 0, 1, -1};
            const int dy[4] = {-1, 1, 0, 0};
            queue<pair<int, int>> q;
            int count = 0, max_count = 0;
            for (int i = 0; i < grid.size(); ++i)
            {
                for (int j = 0; j < grid[0].size(); ++j)
                {
                    if (grid[i][j])  // 找到岛屿就开始搜索
                    {
                        q.push(make_pair(i, j));
                        count = 0;
                        while (!q.empty())
                        {
                            int sr = q.front().first, sc = q.front().second;
                            q.pop();
                            if (grid[sr][sc] == 0) continue;

                            ++count;
                            grid[sr][sc] = 0;  // 访问过的地方置 0，防止再次访问

                            int x, y;
                            for (int i = 0; i < 4; ++i)
                            {
                                x = sr + dx[i];
                                y = sc + dy[i];
                                if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y])
                                {
                                    q.push(make_pair(x, y));
                                }
                            }
                        }
                        max_count = max(max_count, count);
                    }
                }
            }
            return max_count;
        }
    };
    ```

    ```c++
    class Solution {
    public:
        const int dx[4] = {-1, 1, 0, 0};
        const int dy[4] = {0, 0, -1, 1};
        int maxAreaOfIsland(vector<vector<int>>& grid) {
            int ans = 0, area = 0;
            for (int i = 0; i < grid.size(); ++i)
            {
                for (int j = 0; j < grid[0].size(); ++j)
                {
                    if (grid[i][j] == 1)
                    {
                        area = 1;
                        queue<pair<int, int>> q;
                        q.push(make_pair(i, j));
                        grid[i][j] = 0;  // 这里需要提前置 0
                        int sx, sy, x, y;
                        while (!q.empty())
                        {
                            // 不能写成 sx, sy = q.front().first, q.front().second; 会出错
                            sx = q.front().first;
                            sy = q.front().second;
                            q.pop();
                            for (int k = 0; k < 4; ++k)
                            {
                                x = sx + dx[k];
                                y = sy + dy[k];
                                if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == 1)
                                {
                                    ++area;
                                    q.push(make_pair(x, y));
                                    grid[x][y] = 0;  // 直接在这里中止掉下次的搜索，就不用再在前面写 continue 了，队列里也没有重复的点了
                                }
                            }
                        }
                        ans = max(ans, area);
                    }
                }
            }
            return ans;
        }
    };
    ```

1. 深度优先（递归实现）

    ```c++
    class Solution {
    public:
        const int dx[4] = {0, 0, 1, -1};
        const int dy[4] = {-1, 1, 0, 0};
        int dfs(vector<vector<int>> &grid, int sx, int sy)
        {
            if (sx < 0 || sx >= grid.size() || sy < 0 || sy >= grid[0].size() || grid[sx][sy] == 0)
                return 0;
            
            grid[sx][sy] = 0;

            int area = 1;
            for (int i = 0; i < 4; ++i)
                area += dfs(grid, sx+dx[i], sy+dy[i]);
                
            return area;
        }

        int maxAreaOfIsland(vector<vector<int>>& grid) {
            int max_area = 0;
            for (int i = 0; i < grid.size(); ++i)
                for (int j = 0; j < grid[0].size(); ++j)
                    max_area = max(max_area, dfs(grid, i, j));
            return max_area;
        }
    };
    ```

1. 深度优先（栈实现）（没看，但应该不难）

    ```c++
    class Solution {
    public:
        int maxAreaOfIsland(vector<vector<int>>& grid) {
            int ans = 0;
            for (int i = 0; i != grid.size(); ++i) {
                for (int j = 0; j != grid[0].size(); ++j) {
                    int cur = 0;
                    stack<int> stacki;
                    stack<int> stackj;
                    stacki.push(i);
                    stackj.push(j);
                    while (!stacki.empty()) {
                        int cur_i = stacki.top(), cur_j = stackj.top();
                        stacki.pop();
                        stackj.pop();
                        if (cur_i < 0 || cur_j < 0 || cur_i == grid.size() || cur_j == grid[0].size() || grid[cur_i][cur_j] != 1) {
                            continue;
                        }
                        ++cur;
                        grid[cur_i][cur_j] = 0;
                        int di[4] = {0, 0, 1, -1};
                        int dj[4] = {1, -1, 0, 0};
                        for (int index = 0; index != 4; ++index) {
                            int next_i = cur_i + di[index], next_j = cur_j + dj[index];
                            stacki.push(next_i);
                            stackj.push(next_j);
                        }
                    }
                    ans = max(ans, cur);
                }
            }
            return ans;
        }
    };
    ```

#### 01矩阵

给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

 
```
示例 1：

输入：
[[0,0,0],
 [0,1,0],
 [0,0,0]]

输出：
[[0,0,0],
 [0,1,0],
 [0,0,0]]
示例 2：

输入：
[[0,0,0],
 [0,1,0],
 [1,1,1]]

输出：
[[0,0,0],
 [0,1,0],
 [1,2,1]]
```

代码：

1. 还有一种把 1 压入队列的广度优先搜索，有机会了试试

1. 多源广度优先搜索

    这种方法很巧妙。首先利用了 0 到 0 的距离是 0，正好是 0 本身。其次从 0 往周围一层一层渲染，第一层渲染的一定是距离 0 最近的一圈，而渲染过的都不再渲染，距离也就已经确定下来了。接下来把这些确定过距离的点再次当作源点渲染下一层，又能确定下一层的距离。

    或许可以看看图的多源最短路径搜索。

    ```c++
    class Solution {
    public:
        vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
            queue<pair<int, int>> q;
            for (int i = 0; i < mat.size(); ++i)
            {
                for (int j = 0; j < mat[0].size(); ++j)
                {
                    if (mat[i][j] == 0) q.push(make_pair(i, j));  // 将所有为 0 的位置入队
                    else mat[i][j] = -1;  // 后续用于标记是否搜索过
                }
            }

            const int dx[4] = {0, 0, -1, 1};
            const int dy[4] = {-1, 1, 0, 0};
            int sx, sy, x, y;
            while (!q.empty())
            {
                sx = q.front().first;
                sy = q.front().second;
                q.pop();
                for (int i = 0; i < 4; ++i)
                {
                    x = sx + dx[i];
                    y = sy + dy[i];
                    if (x >= 0 && x < mat.size() && y >= 0 && y < mat[0].size() && mat[x][y] == -1)
                    {
                        mat[x][y] = mat[sx][sy] + 1;  // 在这里计算距离
                        q.push(make_pair(x, y));
                    }
                }
            }
            return mat;
        }
    };
    ```

1. 动态规划

    设$f(i, j)$表示点$(i, j)$到 0 的最短距离，则有：

    $$f(i, j) = \begin{cases} 1 + \min(f(i-1, j), f(i, j-1), f(i+1, j), f(i, j+1)) &\text{if }mat[i][j] == 1 \\
    0 &\text{if }mat[i][j] == 0
    \end{cases}$$

    对于某一个 1，从左、上、左上计算一次最短距离，再从右、下、右下计算一次最短距离，取两者的最短值即可。

    ```c++
    class Solution {
    public:
        vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
            vector<vector<int>> res(mat.size(), vector<int>(mat[0].size()));
            for (int i = 0; i < mat.size(); ++i)
            {
                for (int j = 0; j < mat[0].size(); ++j)
                {
                    if (mat[i][j] == 0) res[i][j] = 0;
                    else res[i][j] = INT32_MAX  / 2;  // 因为后面要 +1，为了防止溢出，这里就除以 2 了
                }
            }

            for (int i = 0; i < mat.size(); ++i)
            {
                for (int j = 0; j < mat[0].size(); ++j)
                {
                    if (i - 1 >= 0) res[i][j] = min(res[i-1][j]+1, res[i][j]);  // 这种写法很巧妙，有机会了研究研究
                    if (j - 1 >= 0) res[i][j] = min(res[i][j-1]+1, res[i][j]);
                }
            }

            for (int i = mat.size() - 1; i > -1; --i)
            {
                for (int j = mat[0].size() - 1; j > -1; --j)
                {
                    if (i + 1 < mat.size()) res[i][j] = min(res[i+1][j]+1, res[i][j]);
                    if (j + 1 < mat[0].size()) res[i][j] = min(res[i][j+1]+1, res[i][j]);
                }
            }

            return res;
        }
    };
    ```

    下面是自己写的一个版本，更好理解一些：

    ```c++
    class Solution {
    public:
        vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
            vector<vector<int>> dp(mat.size(), vector<int>(mat[0].size()));
            for (int i = 0; i < mat.size(); ++i)
            {
                for (int j = 0; j < mat[0].size(); ++j)
                {
                    if (mat[i][j] == 0) dp[i][j] = 0;
                    else dp[i][j] = INT_MAX / 2;
                }
            }

            for (int i = 0; i < mat.size(); ++i)
            {
                for (int j = 0; j < mat[0].size(); ++j)
                {
                    if (i == 0 && j == 0) continue;  // 左上角那个点，不用动
                    else if (i == 0 && j > 0)  // 第一行，只需要顾及左边的就可以了
                    {
                        dp[i][j] = min(dp[i][j-1] + 1, dp[i][j]);
                        continue;
                    }
                    else if (i > 0 && j == 0)  // 第一列，只需顾及上边的就可以了
                    {
                        dp[i][j] = min(dp[i-1][j] + 1, dp[i][j]);
                        continue;
                    }
                    else  // 第二行开始到最后一行，第二列开始到最后一列，每次都取左和上两个方向的最小值
                    {
                        dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]) + 1, dp[i][j]);
                    }
                }
            }

            // 从右下角往左上角再来做一次动态规划
            // 右上角的最短路径已经被归并到右方向上了，左下角的最短路径也被归并到左方向上了
            // 所以右上和左下就不必再处理了
            for (int i = mat.size() - 1; i > -1; --i)
            {
                for (int j = mat[0].size() - 1; j > - 1; --j)
                {
                    if (i == mat.size() - 1 && j == mat[0].size() - 1) continue;
                    else if (i == mat.size() - 1 && j < mat[0].size() - 1)
                    {
                        dp[i][j] = min(dp[i][j+1] + 1, dp[i][j]);
                        continue;
                    }
                    else if (i < mat.size() - 1 && j == mat[0].size() - 1)
                    {
                        dp[i][j] = min(dp[i+1][j] + 1, dp[i][j]);
                        continue;
                    }
                    else
                    {
                        dp[i][j] = min(min(dp[i+1][j], dp[i][j+1]) + 1, dp[i][j]);
                    }
                }
            }
            return dp;
        }
    };
    ```

#### 地图分析

你现在手里有一份大小为 N x N 的 网格 grid，上面的每个 单元格 都用 0 和 1 标记好了。其中 0 代表海洋，1 代表陆地，请你找出一个海洋单元格，这个海洋单元格到离它最近的陆地单元格的距离是最大的。

我们这里说的距离是「曼哈顿距离」（ Manhattan Distance）：(x0, y0) 和 (x1, y1) 这两个单元格之间的距离是 |x0 - x1| + |y0 - y1| 。

如果网格上只有陆地或者海洋，请返回 -1。


示例 1：

```
输入：[[1,0,1],[0,0,0],[1,0,1]]
输出：2
解释： 
海洋单元格 (1, 1) 和所有陆地单元格之间的距离都达到最大，最大距离为 2。
```

代码：

1. bfs

    ```c++
    class Solution {
    public:
        int maxDistance(vector<vector<int>>& grid) {
            queue<pair<int, int>> q;
            for (int i = 0; i < grid.size(); ++i)
            {
                for (int j = 0; j < grid[0].size(); ++j)
                {
                    if (grid[i][j] == 1)
                    {
                        q.push(make_pair(i, j));
                        grid[i][j] = 0;  // 标记为 0，用于计数
                    }
                    else
                        grid[i][j] = -1;  // 标记为 -1，用于标记是否访问过
                }
            }
            if (q.empty()) return -1;
            if (q.size() == grid.size() * grid[0].size()) return -1;
            
            int sx, sy, x, y;
            int max_dis = -1;
            const int dx[4] = {0, 0, -1, 1};
            const int dy[4] = {-1, 1, 0, 0};
            while (!q.empty())
            {
                sx = q.front().first;
                sy = q.front().second;
                q.pop();
                for (int i = 0; i < 4; ++i)
                {
                    x = sx + dx[i];
                    y = sy + dy[i];
                    if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == -1)
                    {
                        grid[x][y] = grid[sx][sy] + 1;
                        max_dis = max(grid[x][y], max_dis);
                        q.push(make_pair(x, y));
                    }
                }
            }
            return max_dis;
        }
    };
    ```

官方还给出了 dijkstra 版，多源 bfs 版，spfa 版 以及动态规划版，有时间再看看题解：<https://leetcode-cn.com/problems/as-far-from-land-as-possible/solution/di-tu-fen-xi-by-leetcode-solution/>

#### 腐烂的橘子

在给定的网格中，每个单元格可以有以下三个值之一：

值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。

 
```
示例 1：

输入：[[2,1,1],[1,1,0],[0,1,1]]
输出：4
示例 2：

输入：[[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个正向上。
示例 3：

输入：[[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
```

代码：

1. bfs。将题目看作多源腐烂桔子到最远新鲜桔子的距离。

    ```c++
    class Solution {
    public:
        int orangesRotting(vector<vector<int>>& grid) {
            queue<pair<int, int>> q;
            int fresh_count = 0;  // 统计新鲜橘子是否还有剩余
            int m = grid.size(), n = grid[0].size();
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (grid[i][j] == 2)
                    {
                        q.push(make_pair(i, j));
                        grid[i][j] = 0;  // 用于计数时间
                    }
                    if (grid[i][j] == 1)
                    {
                        ++fresh_count;
                        grid[i][j] = -1;  // 用于标记要搜索的地方
                    }
                }
            }

            int dx[4] = {0, 0, -1, 1};
            int dy[4] = {-1, 1, 0, 0};
            int sx, sy, x, y;
            int res = 0;
            while (!q.empty())
            {
                sx = q.front().first;
                sy = q.front().second;
                q.pop();
                for (int i = 0; i < 4; ++i)
                {
                    x = sx + dx[i];
                    y = sy + dy[i];
                    if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == -1)
                    {
                        q.push(make_pair(x, y));
                        grid[x][y] = grid[sx][sy] + 1;
                        res = max(res, grid[x][y]);
                        --fresh_count;
                    }
                }
            }
            return fresh_count > 0 ? -1 : res;  // 若还剩余有新鲜橘子，则返回 -1
        }
    };
    ```

1. 同样是多源 bsf。不过这次看作 bfs 向外渲染的次数。

    ```c++
    class Solution {
    public:
        int orangesRotting(vector<vector<int>>& grid) {
            queue<pair<int, int>> q;
            int num = 0;
            for (int i = 0; i < grid.size(); ++i)
            {
                for (int j = 0; j < grid[0].size(); ++j)
                {
                    if (grid[i][j] == 2) q.push(make_pair(i, j));
                    if (grid[i][j] == 1) ++num;
                }
            }
            if (num == 0) return 0;

            const int dx[4] = {-1, 1, 0, 0};
            const int dy[4] = {0, 0, -1, 1};
            int sx, sy, x, y;
            int cnt = -1, size;
            while (!q.empty())
            {
                size = q.size();
                for (int i = 0; i < size; ++i)
                {
                    sx = q.front().first;
                    sy = q.front().second;
                    q.pop();
                    for (int j = 0; j < 4; ++j)
                    {
                        x = sx + dx[j];
                        y = sy + dy[j];
                        if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == 1)
                        {
                            grid[x][y] = 2;
                            q.push(make_pair(x, y));
                            --num;
                        }
                    }
                }
                ++cnt;
            }

            if (num == 0) return cnt;
            return -1;
        }
    };
    ```

#### 岛屿数量

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```
示例 1：

输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
示例 2：

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

代码：

1. 广度优先搜索

    ```c++
    class Solution {
    public:
        int numIslands(vector<vector<char>>& grid) {
            const int dx[] = {-1, 0, 0, 1};
            const int dy[] = {0, -1, 1, 0};
            queue<pair<int, int>> q;
            int count = 0;
            int sx, sy, x, y;
            for (int i = 0; i < grid.size(); ++i)
            {
                for (int j = 0; j < grid[0].size(); ++j)
                {
                    if (grid[i][j] == '1')
                    {
                        ++count;
                        q.push(make_pair(i, j));
                        while (!q.empty())
                        {
                            sx = q.front().first, sy = q.front().second;
                            q.pop();
                            if (grid[sx][sy] == '0') continue;  // 对于已经搜索过的地方直接跳过，不加这一行会重复搜索很多
                            grid[sx][sy] = '0';
                            for (int i = 0; i < 4; ++i)
                            {
                                x = sx + dx[i];
                                y = sy + dy[i];
                                if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == '1')
                                {
                                    q.push(make_pair(x, y));
                                }
                                    
                            }
                        }
                    }
                }
            }
            return count;
        }
    };
    ```

1. 深度优先搜索

    ```c++
    class Solution {
    public:
        const int dx[4] = {0, 0, -1, 1};
        const int dy[4] = {-1, 1, 0, 0};
        int ans;

        void dfs(vector<vector<char>> &grid, int sx, int sy)
        {
            grid[sx][sy] = '0';  // 这里不用写 if (grid[sx][sy] == '1') continue;，不存在这种情况
            int x, y;
            for (int i = 0; i < 4; ++i)
            {
                x = sx + dx[i];
                y = sy + dy[i];
                if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == '1')
                {
                    dfs(grid, x, y);
                }
            }
        }

        int numIslands(vector<vector<char>>& grid) {
            ans = 0;
            for (int i = 0; i < grid.size(); ++i)
            {
                for (int j = 0; j < grid[0].size(); ++j)
                {
                    if (grid[i][j] == '1')
                    {
                        ++ans;
                        dfs(grid, i, j);
                    }
                }
            }
            return ans;
        }
    };
    ```

#### 省份数量

有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。

 
```
示例 1：


输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
示例 2：


输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
```

代码：

1. 深度优先

    ```c++
    class Solution {
    public:
        vector<bool> visited;
        void dfs(vector<vector<int>> &isConnected, int i)
        {
            visited[i] = true;
            for (int j = 0; j < isConnected.size(); ++j)
            {
                if (isConnected[i][j] && !visited[j])
                    dfs(isConnected, j);
            }
        }

        int findCircleNum(vector<vector<int>>& isConnected) {
            int ans = 0;
            visited.assign(isConnected.size(), false);
            for (int i = 0; i < isConnected.size(); ++i)
            {
                if (!visited[i])
                {
                    ++ans;
                    dfs(isConnected, i);
                }
            }
            return ans;
        }
    };
    ```

1. 广度优先

    ```c++
    class Solution {
    public:
        int findCircleNum(vector<vector<int>>& isConnected) {
            int ans = 0;
            vector<bool> visited(isConnected.size(), false);
            queue<int> q;
            int j;
            for (int i = 0; i < isConnected.size(); ++i)
            {
                if (!visited[i])
                {
                    ++ans;
                    q.push(i);
                    while (!q.empty())
                    {
                        j = q.front();
                        q.pop();
                        if (visited[j]) continue;  // 不加这一行时间会多出很多
                        visited[j] = true;
                        for (int k = 0; k < isConnected.size(); ++k)
                        {
                            if (isConnected[j][k] && !visited[k])
                                q.push(k);
                        }
                    }
                }
            }
            return ans;
        }
    };
    ```

#### 二进制矩阵中的最短路径

给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。

二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：

路径途经的所有单元格都的值都是 0 。
路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
畅通路径的长度 是该路径途经的单元格总数。

 
```
示例 1：


输入：grid = [[0,1],[1,0]]
输出：2
示例 2：


输入：grid = [[0,0,0],[1,1,0],[1,1,0]]
输出：4
示例 3：

输入：grid = [[1,0,0],[1,1,0],[1,1,0]]
输出：-1
```

代码：

1. 很简单的 bfs。只不过把之前的 4 个搜索区域现在改成了 8 个。

    ```c++
    class Solution {
    public:
        int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
            queue<pair<int, int>> q;
            const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
            const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
            q.push(make_pair(0, 0));
            int size;
            int ans = 0;
            int sx, sy, x, y;
            while (!q.empty())
            {
                ++ans;
                size = q.size();
                for (int i = 0; i < size; ++i)
                {
                    sx = q.front().first, sy = q.front().second;
                    q.pop();
                    if (grid[sx][sy]) continue;
                    if (sx == grid.size() - 1 && sy == grid[0].size() - 1) return ans;
                    grid[sx][sy] = 1;

                    for (int j = 0; j < 8; ++j)
                    {
                        x = sx + dx[j];
                        y = sy + dy[j];
                        if (x >= 0 && x < grid.size() && y >= 0 && y < grid[0].size() && grid[x][y] == 0)
                        {
                            q.push(make_pair(x, y));
                        }
                    }
                }
            }
            return -1;
        }
    };
    ```

#### 被围绕的区域

给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
 
```
示例 1：


输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
示例 2：

输入：board = [["X"]]
输出：[["X"]]
```

代码：

1. 自己写的 bfs，效率挺低的。思路是用另外一个队列来存已经更改过的位置，如果找到某个边界上有`O`，那么就把之前改过的全改过来。

    这样会重复搜索，比如已经找过某个位置的`'O'`，它旁边的`'O'`会再被搜索一遍。

    ```c++
    class Solution {
    public:
        void solve(vector<vector<char>>& board) {
            queue<pair<int, int>> q, backup;
            const int dx[4] = {0, 0, -1, 1};
            const int dy[4] = {-1, 1, 0, 0};
            int sx, sy, x, y;
            int size;
            for (int i = 0; i < board.size(); ++i)
            {
                for (int j = 0; j < board[0].size(); ++j)
                {
                    if (board[i][j] == 'O')
                    {
                        q.push(make_pair(i, j));
                        backup.push(make_pair(i, j));
                        while (!q.empty())
                        {
                            sx = q.front().first, sy = q.front().second;
                            q.pop();
                            if (board[sx][sy] == 'X') continue;

                            if (sx == 0 || sx == board.size() - 1 || sy == 0 || sy == board[0].size() - 1)
                            {
                                while (!backup.empty())
                                {
                                    x = backup.front().first, y = backup.front().second;
                                    backup.pop();
                                    board[x][y] = 'O';
                                }
                                while (!q.empty()) q.pop();
                                break;
                            }

                            board[sx][sy] = 'X';

                            for (int k = 0; k < 4; ++k)
                            {
                                x = sx + dx[k];
                                y = sy + dy[k];
                                if (x >= 0 && y >= 0 && x < board.size() && y < board[0].size() && board[x][y] == 'O')
                                {
                                    q.push(make_pair(x, y));
                                    backup.push(make_pair(x, y));
                                }
                            }
                        }
                        while (!backup.empty()) backup.pop();
                    }
                }
            }
        }
    };
    ```

1. 官方的 bfs。先从四周搜索起，遇到`'O'`就 bfs，将其全部转换成`'A'`。然后再遍历一遍，遇到`'A'`就转换回`'O'`，遇到`'O'`就转换成`'X'`（这些`'O'`必然是被`'X'`包围着的）

    ```c++
    class Solution {
    public:
        const int dx[4] = {0, 0, -1, 1};
        const int dy[4] = {-1, 1, 0, 0};

        void solve(vector<vector<char>>& board) {
            queue<pair<int, int>> q;
            int i, j;
            for (j = 0; j < board[0].size(); ++j)
            {
                if (board[0][j] == 'O')
                    q.push(make_pair(0, j));
                if (board[board.size()-1][j] == 'O')
                    q.push(make_pair(board.size()-1, j));
            }
    
            for (i = 0; i < board.size(); ++i)
            {
                if (board[i][0] == 'O')
                    q.push(make_pair(i, 0));
                if (board[i][board[0].size()-1] == 'O')
                    q.push(make_pair(i, board[0].size() - 1));
            }

            int sx, sy, x, y;
            while (!q.empty())
            {
                sx = q.front().first, sy = q.front().second;
                q.pop();
                if (board[sx][sy] == 'A') continue;
                board[sx][sy] = 'A';
                for (int i = 0; i < 4; ++i)
                {
                    x = sx + dx[i];
                    y = sy + dy[i];
                    if (x >= 0 && x < board.size() && y >= 0 && y < board[0].size() && board[x][y] == 'O')
                    {
                        q.push(make_pair(x, y));
                    }
                }
            }

            for (i = 0; i < board.size(); ++i)
            {
                for (j = 0; j < board[0].size(); ++j)
                {
                    if (board[i][j] == 'A')
                        board[i][j] = 'O';
                    else if (board[i][j] == 'O')
                        board[i][j] = 'X';
                }
            }
        }
    };
    ```

1. 深度优先的一个版本

    ```c++
    class Solution {
    public:
        void dfs(vector<vector<char>> &board, int sx, int sy)
        {
            if (sx < 0 || sx >= board.size() || sy < 0 || sy >= board[0].size() || board[sx][sy] != 'O') return;

            board[sx][sy] = 'A';

            dfs(board, sx-1, sy);
            dfs(board, sx+1, sy);
            dfs(board, sx, sy-1);
            dfs(board, sx, sy+1);
        }

        void solve(vector<vector<char>>& board) {
            int i, j;
            for (j = 0; j < board[0].size(); ++j)
            {
                if (board[0][j] == 'O')
                    dfs(board, 0, j);
                if (board[board.size()-1][j] == 'O')
                    dfs(board, board.size()-1, j);
            }
    
            for (i = 0; i < board.size(); ++i)
            {
                if (board[i][0] == 'O')
                    dfs(board, i, 0);
                if (board[i][board[0].size()-1] == 'O')
                    dfs(board, i, board[0].size()-1);
            }

            for (i = 0; i < board.size(); ++i)
            {
                for (j = 0; j < board[0].size(); ++j)
                {
                    if (board[i][j] == 'A')
                        board[i][j] = 'O';
                    else if (board[i][j] == 'O')
                        board[i][j] = 'X';
                }
            }
        }
    };
    ```

#### 机器人的运动范围

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 
```
示例 1：

输入：m = 2, n = 3, k = 1
输出：3
示例 2：

输入：m = 3, n = 1, k = 0
输出：1
```

代码：

1. 因为不涉及到回退回去再找，所以直接用 bfs

    ```c++
    class Solution {
    public:
        int ans;
        vector<vector<bool>> vis;
        const int dx[4] = {-1, 1, 0, 0};
        const int dy[4] = {0, 0, -1, 1};

        void bfs(int m, int n, int k)
        {
            queue<pair<int, int>> q;
            q.push(make_pair(0, 0));
            int sx, sy, x, y;
            while (!q.empty())
            {
                auto [sx, sy] = q.front();
                q.pop();
                if (vis[sx][sy]) continue;
                vis[sx][sy] = true;
                ++ans;
                for (int i = 0; i < 4; ++i)
                {
                    x = sx + dx[i];
                    y = sy + dy[i];
                    if (x >= 0 && x < m && y >= 0 && y < n && !vis[x][y])
                    {
                        if (x % 10 + x / 10 + y % 10 + y / 10 <= k)
                            q.push(make_pair(x, y));
                    }
                }
            }
        }

        int movingCount(int m, int n, int k) {
            ans = 0;
            vis.assign(m, vector<bool>(n));
            bfs(m, n, k);
            return ans;
        }
    };
    ```

1. bfs，不使用`continue`的写法

    ```c++
    class Solution {
    public:
        int ans;
        vector<vector<bool>> vis;
        const int dx[4] = {-1, 1, 0, 0};
        const int dy[4] = {0, 0, -1, 1};

        void bfs(int m, int n, int k)
        {
            queue<pair<int, int>> q;
            q.push(make_pair(0, 0));
            ans = 1;
            vis[0][0] = true;
            int sx, sy, x, y;
            while (!q.empty())
            {
                auto [sx, sy] = q.front();
                q.pop();
                
                for (int i = 0; i < 4; ++i)
                {
                    x = sx + dx[i];
                    y = sy + dy[i];
                    if (x >= 0 && x < m && y >= 0 && y < n && !vis[x][y])
                    {
                        if (x % 10 + x / 10 + y % 10 + y / 10 <= k)
                        {
                            ++ans;
                            q.push(make_pair(x, y));
                            vis[x][y] = true;
                        }
                    }
                }
            }
        }

        int movingCount(int m, int n, int k) {
            ans = 0;
            vis.assign(m, vector<bool>(n));
            bfs(m, n, k);
            return ans;
        }
    };
    ```

1. 动态规划

    通过找规律可以发现，某个位置若能走到，那么它的左侧或上侧至少有一个能走到。

    ```c++
    class Solution {
    public:
        int movingCount(int m, int n, int k) {
            int ans = 1;
            vector<vector<bool>> dp(m, vector<bool>(n));
            dp[0][0] = true;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == 0 && j == 0) continue;
                    else if (i == 0 && j > 0)
                    {
                        if (dp[i][j-1] && i % 10 + i / 10 + j % 10 + j / 10 <= k)
                        {
                            dp[i][j] = true;
                            ++ans;
                        }
                            
                    }
                    else if (j == 0 && i > 0)
                    {
                        if (dp[i-1][j] && i % 10 + i / 10 + j % 10 + j / 10 <= k)
                        {
                            ++ans;
                            dp[i][j] = true;
                        }
                    }
                    else
                    {
                        if ((dp[i-1][j] || dp[i][j-1]) && i % 10 + i / 10 + j % 10 + j / 10 <= k)
                        {
                            ++ans;
                            dp[i][j] = true;
                        }
                    }
                }
            }
            return ans;
        }
    };
    ```

    答案的写法，更简洁了一些：

    ```c++
    class Solution {
        int get(int x) {
            int res=0;
            for (; x; x /= 10){
                res += x % 10;
            }
            return res;
        }
    public:
        int movingCount(int m, int n, int k) {
            if (!k) return 1;
            vector<vector<int> > vis(m, vector<int>(n, 0));
            int ans = 1;
            vis[0][0] = 1;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if ((i == 0 && j == 0) || get(i) + get(j) > k) continue;
                    // 边界判断
                    if (i - 1 >= 0) vis[i][j] |= vis[i - 1][j];
                    if (j - 1 >= 0) vis[i][j] |= vis[i][j - 1];
                    ans += vis[i][j];
                }
            }
            return ans;
        }
    };
    ```

### 递归/回溯

问题与解答：

1. 何时用 for？何时不用？（括号生成不用 for，电话号码的组合、全排列、组合，这些需要用）



#### 含有重复数字的全排列（全排列 II）

> 输入一组数字（可能包含重复数字），输出其所有的排列方式。
> 
> ```
> 样例
> 输入：[1,2,3]
> 
> 输出：
>       [
>         [1,2,3],
>         [1,3,2],
>         [2,1,3],
>         [2,3,1],
>         [3,1,2],
>         [3,2,1]
>       ]
> ```

代码：

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> perm;
    vector<bool> vis;
    
    void dfs(vector<int> &nums)
    {
        if (perm.size() == nums.size())
        {
            res.push_back(perm);
            return;  // 因为下面 i 是从 0 开始的，所以这里我们提前中止了。否则下面的代码会发现 vis 全为 true，从而自然中止
        }
        
        for (int i = 0; i < nums.size(); ++i)  // 注意，这里是从 0 开始的，而不是从 pos 开始
        {
            if (vis[i] || (i > 0 && nums[i] == nums[i-1] && !vis[i-1]))
                continue;
            perm.emplace_back(nums[i]);
            vis[i] = true;
            dfs(nums);
            vis[i] = false;
            perm.pop_back();
        }
    }
    
    vector<vector<int>> permutation(vector<int>& nums) {
        vis.assign(nums.size(), false);
        sort(nums.begin(), nums.end());
        dfs(nums);
        return res;
    }
};
```

看得不是很懂。

一些笔记：

1. 目标：如果发现数字重复了，那么当前情况就不考虑了（剪枝）。比如`[1, 2, 2']`存在后，当`[1]`遇到第二个`2'`，发现与第一个`2`重复，就直接剪枝。

1. 本数字与上个数字相同且上个数字未访问过时 -> 跳过。目的：保证每次都是拿从左往右第一个未被填过的数字。

    其实保证每次拿的是下一段的新数字。不想解释了，有空了做个动画演示下更形象。

#### 字符串的排列

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

```
示例:

输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

代码：

1. 回溯法，同“含有重复数字的全排列”。

    ```c++
    class Solution {
    public:
        vector<string> ans;
        string temp;
        vector<bool> vis;

        void backtrack(string &s)
        {
            if (temp.size() == s.size())
            {
                ans.push_back(temp);
                return;
            }

            for (int i = 0; i < s.size(); ++i)
            {
                if (vis[i]) continue;
                if (i > 0 && !vis[i-1] && s[i] == s[i-1]) continue;
                vis[i] = true;
                temp.push_back(s[i]);
                backtrack(s);
                temp.pop_back();
                vis[i] = false;
            }
        }

        vector<string> permutation(string s) {
            sort(s.begin(), s.end());
            vis.resize(s.size());
            backtrack(s);
            return ans;
        }
    };
    ```

1. 下一个排列，这样就不用去重了（没看）

    ```c++
    class Solution {
    public:
        bool nextPermutation(string& s) {
            int i = s.size() - 2;
            while (i >= 0 && s[i] >= s[i + 1]) {
                i--;
            }
            if (i < 0) {
                return false;
            }
            int j = s.size() - 1;
            while (j >= 0 && s[i] >= s[j]) {
                j--;
            }
            swap(s[i], s[j]);
            reverse(s.begin() + i + 1, s.end());
            return true;
        }

        vector<string> permutation(string s) {
            vector<string> ret;
            sort(s.begin(), s.end());
            do {
                ret.push_back(s);
            } while (nextPermutation(s));
            return ret;
        }
    };
    ```


#### 字母大小写全排列

给定一个字符串S，通过将字符串S中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。

 
```
示例：
输入：S = "a1b2"
输出：["a1b2", "a1B2", "A1b2", "A1B2"]

输入：S = "3z4"
输出：["3z4", "3Z4"]

输入：S = "12345"
输出：["12345"]
```

代码：

1. 树的先序遍历

    ```cpp
    class Solution {
    public:
        vector<string> ans;
        string temp;
        void backtrack(string &s, int pos)
        {
            if (temp.size() >= s.size())
            {
                ans.push_back(temp);
                return;
            }
            if (s[pos] >= '0' && s[pos] <= '9')
            {
                temp.push_back(s[pos]);
                backtrack(s, pos+1);
                temp.pop_back();
            }
            else
            {
                temp.push_back(s[pos]);
                backtrack(s, pos+1);
                temp.pop_back();
                char ch;
                if ('a' <= s[pos] && s[pos] <= 'z')
                    ch = s[pos] - 'a' + 'A';
                else
                    ch = s[pos] - 'A' + 'a';
                temp.push_back(ch);
                backtrack(s, pos+1);
                temp.pop_back();
            }  
        }

        vector<string> letterCasePermutation(string s) {
            backtrack(s, 0);
            return ans;
        }
    };
    ```

    对于`a1b2`这个字符串，其实我们要做的是遍历这样一棵树：

    ```
            o
           / \
          a   A
          |   |
          1   1
         / \ / \
        b  B b  B
        |  | |  |
        2  2 2  2
    ```

    事实上，我们需要做的是一个先序遍历，并记录路径（即代码中的`temp`变量），如果达到叶子节点，那么把路径 append 到答案中。

    我们并没有类似`TreeNode`这样的数据结构，因此在递归遍历下一个节点时，其实是与当前遍历的深度（字符串的索引 pos）配合，凭空产生的节点。本题中只有两个节点，比较简单，所以直接手写了。如果遇到很多的节点，可以用`for`进行遍历。

    在函数的局部变量和代码流程中，记录了一部分的路径信息，因此可以不需要`temp`，我们可以直接对`s`进行 in-place 的修改，然后将`s`添加到`ans`中即可。这样可以再省点内存。但是我觉得用`temp`的解法更通用。

    下面是个 in-place 形式的例子：

    ```c++
    class Solution {
    public:
        vector<string> ans;

        void backtrack(string &s, int pos)
        {
            if (pos == s.size())
            {
                ans.push_back(s);
                return;
            }

            if (isdigit(s[pos]))
                backtrack(s, pos + 1);
            else
            {
                s[pos] = tolower(s[pos]);
                backtrack(s, pos + 1);

                s[pos] = toupper(s[pos]);
                backtrack(s, pos + 1);
                s[pos] = tolower(s[pos]);
            }
        }

        vector<string> letterCasePermutation(string s) {
            backtrack(s, 0);
            return ans;
        }
    };
    ```

1. 自上而下递归（类似二叉树的前序遍历）

    ```c++
    class Solution {
    public:
        vector<string> letterCasePermutation(string S) {
            vector<string> res;
            dfs(S, 0, res);
            return res;
        }

    private:
        void dfs(string s, int n, vector<string> &res) {
            if (s.size() == n) return res.push_back(s);
            if (isdigit(s[n])) return dfs(s, n + 1, res);
            dfs(s, n + 1, res);
            s[n] ^= (1 << 5);
            dfs(s, n + 1, res);
        }
    };
    ```

1. 自下而上递归（类似二叉树的中序遍历）

    ```c++
    class Solution {
    public:
        vector<string> letterCasePermutation(string S) {
            vector<string> res;
            dfs(S, 0, res);
            return res;
        }

    private:
        void dfs(string s, int n, vector<string> &res) {
            if (s.size() == n) return res.push_back(s);
            dfs(s, n + 1, res); //① 注意这里的 dfs 函数有 2层意思
            if (isdigit(s[n])) return; // 忽略数字，直接返回
            s[n] ^= (1 << 5); // 切换大小写
            dfs(s, n + 1, res); // 回溯第二分支
        }
    };
    ```

1. 迭代回溯（类似二叉树的层序遍历，BFS）

    ```c++
    class Solution {
    public:
        vector<string> letterCasePermutation(string S) {
            deque<string> worker;
            worker.push_back(S);
            for (int i = 0; i < S.size(); ++i) {
                if (isdigit(S[i])) continue;
                for (int j = worker.size(); j > 0; --j) {
                    auto sub = worker.front();
                    worker.pop_front();
                    worker.push_back(sub);
                    sub[i] ^= (1 << 5);
                    worker.push_back(sub);
                }
            }
            return vector<string>(worker.begin(), worker.end());
        }
    };
    ```

1. bfs，每次遇到字母就分成两份，分别添加字母的小写和大写

    ```java
    class Solution {
        public List<String> letterCasePermutation(String S) {
            List<StringBuilder> ans = new ArrayList();
            ans.add(new StringBuilder());

            for (char c: S.toCharArray()) {
                int n = ans.size();
                if (Character.isLetter(c)) {
                    for (int i = 0; i < n; ++i) {
                        ans.add(new StringBuilder(ans.get(i)));
                        ans.get(i).append(Character.toLowerCase(c));
                        ans.get(n+i).append(Character.toUpperCase(c));
                    }
                } else {
                    for (int i = 0; i < n; ++i)
                        ans.get(i).append(c);
                }
            }

            List<String> finalans = new ArrayList();
            for (StringBuilder sb: ans)
                finalans.add(sb.toString());
            return finalans;
        }
    }
    ```

#### 子集

给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

 
```
示例 1：

输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：

输入：nums = [0]
输出：[[],[0]]
```

代码：

1. 普通回溯

    ```c++
    class Solution {
    public:
        vector<vector<int>> res;
        vector<int> temp;
        
        void trace_back(vector<int> &nums, int pos)
        {
            res.push_back(temp);  // 每一种情况都添入到结果中。其实这个问题可以看作是基本版本，组合 I 和组合 II，组合总和 I，组合总和 II 等问题，都是在这个基本版本上做修改
            for (int i = pos; i < nums.size(); ++i)
            {
                temp.push_back(nums[i]);
                trace_back(nums, i+1);  // 每次只需要从当前元素的右侧去选就可以了
                temp.pop_back();
            }
        }
        vector<vector<int>> subsets(vector<int>& nums) {
            trace_back(nums, 0);
            return res;
        }
    };
    ```

1. 利用二进制表示某个元素是否被选入集合

1. 用纯递归实现

    每个元素都有放/不放两种选择，用`cur`记录遍历到了哪个元素，如果遍历到了末尾，那么把一个子集加入到结果中。

    ```c++
    class Solution {
    public:
        vector<int> t;
        vector<vector<int>> ans;

        void dfs(int cur, vector<int>& nums) {
            if (cur == nums.size()) {
                ans.push_back(t);
                return;
            }
            t.push_back(nums[cur]);
            dfs(cur + 1, nums);
            t.pop_back();
            dfs(cur + 1, nums);
        }

        vector<vector<int>> subsets(vector<int>& nums) {
            dfs(0, nums);
            return ans;
        }
    };
    ```

#### 子集 II

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

```
示例 1：

输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]

示例 2：

输入：nums = [0]
输出：[[],[0]]
```

代码：

回溯。相当于含有重复元素的“子集”。

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> temp;
    vector<bool> vis;
    void backtrack(vector<int> &nums, int pos)
    {
        res.push_back(temp);
        for (int i = pos; i < nums.size(); ++i)
        {
            // if (vis[i]) continue;  // 这行代码没必要，因为 i 不是从 0 开始，而是从 pos 开始，所以总是在前进
            if (i > 0 && !vis[i-1] && nums[i-1] == nums[i]) continue;
            vis[i] = true;
            temp.push_back(nums[i]);
            backtrack(nums, i+1);
            temp.pop_back();
            vis[i] = false;
        }
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vis.assign(nums.size(), false);
        sort(nums.begin(), nums.end());
        backtrack(nums, 0);
        return res;
    }
};
```

#### 组合

给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。

示例:

输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

代码：

1. 回溯

    ```c++
    class Solution {
    public:
        vector<vector<int>> res;
        vector<int> temp;
        
        void dfs(int n, int k, int pos)
        {
            if (temp.size() + (n - pos + 1) < k) return;  // 剪枝，如果即使把剩下的所有数字都加上去也凑不够结果，那么就直接放弃
            if (temp.size() == k)
            {
                res.push_back(temp);
                return;
            }

            for (int i = pos; i < n; ++i)  // 注意这里 i 是从 pos 开始的，表示之前考虑过的都不用再考虑了
            {
                temp.push_back(i+1);
                dfs(n, k, i+1);
                temp.pop_back();
            }
        }

        vector<vector<int>> combine(int n, int k) {
            dfs(n, k, 0);
            return res;
        }
    };
    ```

1. 递归

    纯递归版的答案。没怎么看。

    ```c++
    class Solution {
    public:
        vector<int> temp;
        vector<vector<int>> ans;

        void dfs(int cur, int n, int k) {
            // 剪枝：temp 长度加上区间 [cur, n] 的长度小于 k，不可能构造出长度为 k 的 temp
            if (temp.size() + (n - cur + 1) < k) {
                return;
            }
            // 记录合法的答案
            if (temp.size() == k) {
                ans.push_back(temp);
                return;
            }
            // 考虑选择当前位置
            temp.push_back(cur);
            dfs(cur + 1, n, k);
            temp.pop_back();
            // 考虑不选择当前位置
            dfs(cur + 1, n, k);
        }

        vector<vector<int>> combine(int n, int k) {
            dfs(1, n, k);
            return ans;
        }
    };
    ```

1. 非递归（字典序法）实现组合型枚举（官方给出的第二种答案，有点难，没看）

    ```c++
    class Solution {
    public:
        vector<int> temp;
        vector<vector<int>> ans;

        vector<vector<int>> combine(int n, int k) {
            // 初始化
            // 将 temp 中 [0, k - 1] 每个位置 i 设置为 i + 1，即 [0, k - 1] 存 [1, k]
            // 末尾加一位 n + 1 作为哨兵
            for (int i = 1; i <= k; ++i) {
                temp.push_back(i);
            }
            temp.push_back(n + 1);
            
            int j = 0;
            while (j < k) {
                ans.emplace_back(temp.begin(), temp.begin() + k);
                j = 0;
                // 寻找第一个 temp[j] + 1 != temp[j + 1] 的位置 t
                // 我们需要把 [0, t - 1] 区间内的每个位置重置成 [1, t]
                while (j < k && temp[j] + 1 == temp[j + 1]) {
                    temp[j] = j + 1;
                    ++j;
                }
                // j 是第一个 temp[j] + 1 != temp[j + 1] 的位置
                ++temp[j];
            }
            return ans;
        }
    };
    ```

#### 组合总合 I（每个数字可以用多次）

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。

``` 
示例 1：

输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]
```

代码：

1. 回溯算法。

    ```c++
    class Solution {
    public:
        vector<vector<int>> ans;
        vector<int> temp;

        void backtrack(vector<int> &candidates, int target, int pos)
        {
            if (target == 0)
            {
                ans.push_back(temp);
                return;
            }

            for (int i = pos; i < candidates.size(); ++i)
            {
                if (candidates[i] <= target)
                {
                    temp.push_back(candidates[i]);
                    backtrack(candidates, target-candidates[i], i);
                    temp.pop_back();
                }
            }
        }
        
        vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
            backtrack(candidates, target, 0);
            return ans;
        }
    };
    ```

    为了防止`[2,3,2], [2,2,3]`这样的重复的答案出现，这里使用了`pos`。并且不需要排序。为什么呢？

#### 组合总和 II（每个数字只能用一次）

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

说明：

所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。

```
示例 1:

输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

代码：

1. 回溯。和含有重复数字的全排列很像。

    ```c++
    class Solution {
    public:
        vector<vector<int>> ans;
        vector<int> temp;
        vector<bool> vis;
        
        void backtrack(vector<int> &candidates, int target, int pos)
        {
            if (target == 0)
            {
                ans.push_back(temp);
                return;
            }

            for (int i = pos; i < candidates.size(); ++i)
            {
                if (i > 0 && candidates[i-1] == candidates[i] && !vis[i-1])
                    continue;

                if (candidates[i] <= target)
                {
                    vis[i] = true;
                    temp.push_back(candidates[i]);
                    backtrack(candidates, target - candidates[i], i+1);
                    temp.pop_back();
                    vis[i] = false;
                }
            }
        }

        vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
            sort(candidates.begin(), candidates.end());
            vis.assign(candidates.size(), false);
            backtrack(candidates, target, 0);
            return ans;
        }
    };
    ```

1. 第二个版本

    ```c++
    class Solution {
    public:
        vector<vector<int>> ans;
        vector<int> temp;
        vector<bool> vis;

        void backtrack(vector<int> &candidates, int target, int pos)
        {
            if (target < 0) return;

            if (target == 0)
            {
                ans.push_back(temp);
                return;
            }

            for (int i = pos; i < candidates.size(); ++i)
            {
                if (i > 0 && !vis[i-1] && candidates[i-1] == candidates[i]) continue;

                temp.push_back(candidates[i]);
                vis[i] = true;
                target -= candidates[i];
                backtrack(candidates, target, i+1);
                target += candidates[i];
                vis[i] = false;
                temp.pop_back();
            }
        }

        vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
            sort(candidates.begin(), candidates.end());
            vis.assign(candidates.size(), false);
            backtrack(candidates, target, 0);
            return ans;
        }
    };
    ```

#### 组合总和 III

找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

说明：

1. 所有数字都是正整数。
1. 解集不能包含重复的组合。 

```
示例 1:

输入: k = 3, n = 7
输出: [[1,2,4]]
示例 2:

输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
```

代码：

回溯。

```c++
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> temp;

    void backtrack(int k, int n, int pos)
    {
        if (temp.size() == k && n == 0)
        {
            ans.push_back(temp);
            return;
        }

        for (int i = pos; i < 9; ++i)
        {
            if (i+1 <= n)
            {
                temp.push_back(i+1);
                backtrack(k, n-(i+1), i+1);
                temp.pop_back();
            }
        }
    }

    vector<vector<int>> combinationSum3(int k, int n) {
        backtrack(k, n, 0);
        return ans;
    }
};
```

#### 二进制手表

二进制手表顶部有 4 个 LED 代表 小时（0-11），底部的 6 个 LED 代表 分钟（0-59）。每个 LED 代表一个 0 或 1，最低位在右侧。

给你一个整数 turnedOn ，表示当前亮着的 LED 的数量，返回二进制手表可以表示的所有可能时间。你可以 按任意顺序 返回答案。

小时不会以零开头：

例如，"01:00" 是无效的时间，正确的写法应该是 "1:00" 。
分钟必须由两位数组成，可能会以零开头：

例如，"10:2" 是无效的时间，正确的写法应该是 "10:02" 。
 
```
示例 1：

输入：turnedOn = 1
输出：["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]
示例 2：

输入：turnedOn = 9
输出：[]
```

代码：

1. 枚举合法的时间，判断二进制的 1 的个数是否满足要求

    ```c++
    class Solution {
    public:
        vector<string> readBinaryWatch(int turnedOn) {
            vector<string> res;
            int count, x;
            for (int i = 0; i < 12; ++i)
            {
                for (int j = 0; j < 60; ++j)
                {
                    count = 0;
                    x = i;
                    while (x) { x &= (x-1); ++count;}
                    x = j;
                    while (x) { x &= (x-1); ++count;}
                    if (count == turnedOn)
                        res.push_back(to_string(i) + (j < 10 ? ":0" : ":") + to_string(j));
                }
            }
            return res;
        }
    };
    ```

1. 枚举合法的 10 个二进制位，判断时分是否满足要求

    有空查查资料，了解下怎么枚举满足恰好有`turnedOn`个 1 的二进制数。

1. 回溯

    ```c++
    class Solution {
    public:
        vector<string> res;
        vector<int> bits;
        int turnedOn;
        void backtrack(int pos, int count)
        {
            if (count == turnedOn)
            {
                int minute = 0, hour = 0;
                int p = 9;
                for (int i = 0; i < 6; ++i)
                {
                    minute += bits[p] * pow(2, i);
                    --p;
                }
                for (int i = 0; i < 4; ++i)
                {
                    hour += bits[p] * pow(2, i);
                    --p;
                }
                if (minute < 60 && hour < 12)
                    res.push_back(to_string(hour) + (minute < 10 ? ":0" : ":") + to_string(minute));
                return;
            }

            for (int i = pos; i < 10; ++i)
            {
                bits[i] = 1;
                backtrack(i+1, count+1);  // 注意这里只递归调用一次，而不是调用两次
                bits[i] = 0;
            }
        }
        vector<string> readBinaryWatch(int turnedOn) {
            bits.assign(10, 0);
            this->turnedOn = turnedOn;
            backtrack(0, 0);
            return res;
        }
    };
    ```

#### 分割回文串

给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

回文串 是正着读和反着读都一样的字符串。

```
示例 1：

输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
示例 2：

输入：s = "a"
输出：[["a"]]
```

代码：

1. 动态规划 + 回溯

    自己写的一份，效率很低，只击败了 5%：

    ```c++
    class Solution {
    public:
        vector<vector<bool>> dp;
        vector<vector<string>> res;
        vector<string> temp;
        int count;
        void backtrack(string &s, int pos)
        {
            if (count == s.size())
            {
                res.push_back(temp);
            }
            for (int i = pos; i < s.size(); ++i)
            {
                for (int len = 1; len <= s.size(); ++len)
                {
                    if (i + len - 1 < s.size() && dp[i][len])
                    {
                        temp.push_back(s.substr(i, len));
                        count += len;
                        backtrack(s, i+len);
                        temp.pop_back();
                        count -= len;
                    }
                }
            }
        }
        vector<vector<string>> partition(string s) {
            count = 0;
            dp.resize(s.size(), vector<bool>(s.size()+1));
            for (int i = 0; i < s.size(); ++i)
            {
                dp[i][1] = true;
                if (i > 0 && s[i-1] == s[i])
                    dp[i-1][2] = true;
            }
            for (int len = 3; len <= s.size(); ++len)
            {
                for (int i = 0; i < s.size(); ++i)
                {
                    if (i + len - 1 < s.size() && dp[i+1][len-2] && s[i] == s[i+len-1])
                        dp[i][len] = true;
                }
            }
            backtrack(s, 0);
            return res;
        }
    };
    ```

    官方给出的两份代码，思路差不多，但写法上有优化，有时间了理解理解：

    ```c++
    class Solution {
    private:
        vector<vector<int>> f;
        vector<vector<string>> ret;
        vector<string> ans;
        int n;

    public:
        void dfs(const string& s, int i) {
            if (i == n) {
                ret.push_back(ans);
                return;
            }
            for (int j = i; j < n; ++j) {
                if (f[i][j]) {
                    ans.push_back(s.substr(i, j - i + 1));
                    dfs(s, j + 1);
                    ans.pop_back();
                }
            }
        }

        vector<vector<string>> partition(string s) {
            n = s.size();
            f.assign(n, vector<int>(n, true));

            for (int i = n - 1; i >= 0; --i) {
                for (int j = i + 1; j < n; ++j) {
                    f[i][j] = (s[i] == s[j]) && f[i + 1][j - 1];
                }
            }

            dfs(s, 0);
            return ret;
        }
    };
    ```

    ```c++
    class Solution {
    private:
        vector<vector<int>> f;
        vector<vector<string>> ret;
        vector<string> ans;
        int n;

    public:
        void dfs(const string& s, int i) {
            if (i == n) {
                ret.push_back(ans);
                return;
            }
            for (int j = i; j < n; ++j) {
                if (isPalindrome(s, i, j) == 1) {
                    ans.push_back(s.substr(i, j - i + 1));
                    dfs(s, j + 1);
                    ans.pop_back();
                }
            }
        }

        // 记忆化搜索中，f[i][j] = 0 表示未搜索，1 表示是回文串，-1 表示不是回文串
        int isPalindrome(const string& s, int i, int j) {
            if (f[i][j]) {
                return f[i][j];
            }
            if (i >= j) {
                return f[i][j] = 1;
            }
            return f[i][j] = (s[i] == s[j] ? isPalindrome(s, i + 1, j - 1) : -1);
        }

        vector<vector<string>> partition(string s) {
            n = s.size();
            f.assign(n, vector<int>(n));

            dfs(s, 0);
            return ret;
        }
    };
    ```

#### 单词搜索

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

 
```
示例 1：


输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
示例 2：


输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
示例 3：


输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

代码：

一个标准的回溯。这种问题看起来和岛屿问题很像，但是不能用 bfs。当我们实际用 bfs 做的时候，会发现无法处理“当某个方向错误时，倒退到某个分岔节点搜索其他路径”这种情况。如果发现整条线路都是错的，或许还可以用栈来保存并恢复搜索过的节点，但是如果是半中间遇到分岔的情况，就很难用栈处理了，因为栈不知道在哪分岔。所以这道题必须用回溯法来做。但是反过来想，回溯法是一种特殊的 dfs，而 dfs 都能改写成栈，那么是否可以用栈来记录路径写回溯呢？

还可以将访问过的地方改为`#`，回溯结束后再改回来，这样就不用开辟一个新的`vis`了。

1. 回溯

    ```c++
    class Solution {
    public:
        vector<vector<bool>> vis;
        int m, n;
        const int dx[4] = {0, 0, -1, 1};
        const int dy[4] = {-1, 1, 0, 0};
        bool found;

        void backtrack(vector<vector<char>> &board, int sx, int sy, string &word, int pos)
        {
            if (found) return;  // 剪枝

            if (board[sx][sy] == word[pos])  // 剪枝
            {
                if (pos == word.size() - 1)
                {
                    found = true;
                    return;
                }
                else
                {
                    vis[sx][sy] = true;
                    for (int i = 0; i < 4; ++i)
                    {
                        int x = sx + dx[i];
                        int y = sy + dy[i];
                        if (x >= 0 && x < m && y >= 0 && y < n && !vis[x][y])
                        {
                            backtrack(board, x, y, word, pos+1);
                        }
                    }
                    vis[sx][sy] = false;
                } 
            }
        }

        bool exist(vector<vector<char>>& board, string word) {
            found = false;
            m = board.size();
            n = board[0].size();
            vis.assign(m, vector<bool>(n, false));
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    backtrack(board, i, j, word, 0);
                    if (found) return true;  // 剪枝
                }
            }
            return false;
        }
    };
    ```

1. 回溯  后来又写的

    ```c++
    class Solution {
    public:
        const int dx[4] = {-1, 1, 0, 0};
        const int dy[4] = {0, 0, -1, 1};
        bool ans;
        void backtrack(vector<vector<char>> &board, string &word, int sx, int sy, int pos)
        {
            if (ans) return;
            if (pos >= word.size()) return;
            if (pos == word.size() - 1 && board[sx][sy] == word[pos])
            {
                ans = true;
                return;
            }
            if (board[sx][sy] != word[pos]) return;

            char ch = board[sx][sy];
            board[sx][sy] = '#';
            for (int i = 0; i < 4; ++i)
            {
                int x = sx + dx[i];
                int y = sy + dy[i];
                if (x >= 0 && x < board.size() && y >= 0 && y < board[0].size() && board[x][y] != '#')
                {
                    backtrack(board, word, x, y, pos+1);
                }
            }
            board[sx][sy] = ch;
        }

        bool exist(vector<vector<char>>& board, string word) {
            ans = false;
            for (int i = 0; i < board.size(); ++i)
            {
                for (int j = 0; j < board[0].size(); ++j)
                {
                    backtrack(board, word, i, j, 0);
                    if (ans) return true;
                }
            }
            return ans;
        }
    };
    ```

#### 八皇后

设计一种算法，打印 N 皇后在 N × N 棋盘上的各种摆法，其中每个皇后都不同行、不同列，也不在对角线上。这里的“对角线”指的是所有的对角线，不只是平分整个棋盘的那两条对角线。

注意：本题相对原题做了扩展

示例:

```
 输入：4
 输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
 解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
```

代码：

经典的回溯。

```c++
class Solution {
public:
    vector<vector<string>> res;
    vector<pair<int, int>> pos;
    vector<string> temp;
    int n;
    void backtrack(int sx)
    {
        if (pos.size() != sx) return;  // 剪枝。如果有一行找不到合适的位置，那么这个分支就不要了

        if (pos.size() == n)
        {
            temp.assign(n, string(n, '.'));
            for (auto &[x, y]: pos)
                temp[x][y] = 'Q';
            res.push_back(temp);
        }

        for (int i = sx; i < n; ++i)  // 每行找一个
        {
            bool placed = false;  // 剪枝
            for (int j = 0; j < n; ++j)
            {
                bool valid = true;
                for (auto &[x, y]: pos)
                {
                    if (j == y || x + y == i + j || x - y == i - j)  // 如果同列有其它皇后，或 左下-右上斜线上有其他皇后，或左上-右下斜线上有其他皇后
                    {
                        valid = false;
                        break;
                    }
                }
                if (valid)
                {
                    pos.push_back(make_pair(i, j));
                    backtrack(i + 1);
                    pos.pop_back();
                    placed = true;
                }
            }
            if (!placed) return;
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;
        backtrack(0);
        return res;
    }
};
```

#### 括号生成

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

 
```
示例 1：

输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
示例 2：

输入：n = 1
输出：["()"]
```

代码：

1. 回溯

    每个位置要么添一个左括号，要么添一个右括号，每次前进一步。

    回溯算法在做更改时，是从后往前更改的。比如`((()))`，先更改的可以是倒数的两个，改成`(()())`

    ```c++
    class Solution {
    public:
        vector<string> res;
        string temp;
        void backtrack(int n, int nleft, int nright)
        {
            if (nleft == n && nright == n)
            {
                res.push_back(temp);
                return;
            }

            if (nleft < n)
            {
                temp.push_back('(');
                backtrack(n, nleft+1, nright);
                temp.pop_back();
            }

            if (nright < nleft)
            {
                temp.push_back(')');
                backtrack(n, nleft, nright+1);
                temp.pop_back();
            }
            
        }
        vector<string> generateParenthesis(int n) {
            backtrack(n, 0, 0);
            return res;
        }
    };
    ```

#### 解数独


#### 所有可能的路径

给一个有 n 个结点的有向无环图，找到所有从 0 到 n-1 的路径并输出（不要求按顺序）

二维数组的第 i 个数组中的单元都表示有向图中 i 号结点所能到达的下一些结点（译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a ）空就是没有下一个结点了。

 
```
示例 1：

输入：graph = [[1,2],[3],[3],[]]
输出：[[0,1,3],[0,2,3]]
解释：有两条路径 0 -> 1 -> 3 和 0 -> 2 -> 3
示例 2：

输入：graph = [[4,3,1],[3,2,4],[3],[4],[]]
输出：[[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]
示例 3：

输入：graph = [[1],[]]
输出：[[0,1]]
示例 4：

输入：graph = [[1,2,3],[2],[3],[]]
输出：[[0,1,2,3],[0,2,3],[0,3]]
示例 5：

输入：graph = [[1,3],[2],[3],[]]
输出：[[0,1,2,3],[0,3]]
```

代码：

```c++
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;
    void dfs(vector<vector<int>> &graph, int node)
    {
        if (node == graph.size() - 1)
        {
            ans.push_back(path);
            return;
        }

        for (int i = 0; i < graph[node].size(); ++i)
        {
            path.push_back(graph[node][i]);
            dfs(graph, graph[node][i]);
            path.pop_back();
        }
    }

    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        path.push_back(0);
        dfs(graph, 0);
        return ans;
    }
};
```

#### 电话号码的字母组合

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

```
示例 1：

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
示例 2：

输入：digits = ""
输出：[]
示例 3：

输入：digits = "2"
输出：["a","b","c"]
```

代码：

1. 回溯。与之前回溯不同的地方是，每次循环的次数都不一样。

    ```c++
    class Solution {
    public:
        vector<string> ans;
        string temp;
        unordered_map<char, string> board;

        void backtrack(string &digits, int pos)
        {
            if (temp.size() == digits.size())
            {
                ans.push_back(temp);
                return;
            }

            string str = board[digits[pos]];
            for (int i = 0; i < str.size(); ++i)
            {
                temp.push_back(str[i]);
                backtrack(digits, pos+1);  // 如果是外层循环，那么只与 pos 有关，与 i 无关（大部分回溯时用的还是 i+1，暂时没发现什么规律）
                temp.pop_back();
            }
        }

        vector<string> letterCombinations(string digits) {
            if (digits.empty()) return ans;
            board = {{'2', "abc"}, {'3', "def"}, {'4', "ghi"}, {'5', "jkl"}, {'6', "mno"},
                {'7', "pqrs"}, {'8', "tuv"}, {'9', "wxyz"}};
            backtrack(digits, 0);
            return ans;
        }
    };
    ```

1. 队列

    每次拼接一个数字键上的字母，直到把指定的数字键都拼接完。

    ```c++
    class Solution {
    public:
        vector<string> ans;
        string temp;
        unordered_map<char, string> board;

        vector<string> letterCombinations(string digits) {
            vector<string> ans;
            if (digits.empty()) return ans;

            board = {{'2', "abc"}, {'3', "def"}, {'4', "ghi"}, {'5', "jkl"}, {'6', "mno"},
                {'7', "pqrs"}, {'8', "tuv"}, {'9', "wxyz"}};
            queue<string> q({""});
            char ch;
            string str;
            int size;
            for (int i = 0; i < digits.size(); ++i)
            {
                ch = digits[i];
                str = board[ch];
                size = q.size();
                for (int j = 0; j < size; ++j)
                {
                    for (int k = 0; k < str.size(); ++k)
                    {
                        q.push(q.front() + str[k]);
                    }
                    q.pop();
                }
            }

            while (!q.empty())
            {
                ans.push_back(q.front());
                q.pop();
            } 
            return ans;
        }
    };
    ```

#### 最佳观光组合

给你一个正整数数组 values，其中 values[i] 表示第 i 个观光景点的评分，并且两个景点 i 和 j 之间的 距离 为 j - i。

一对景点（i < j）组成的观光组合的得分为 values[i] + values[j] + i - j ，也就是景点的评分之和 减去 它们两者之间的距离。

返回一对观光景点能取得的最高分。

 
```
示例 1：

输入：values = [8,1,5,2,6]
输出：11
解释：i = 0, j = 2, values[i] + values[j] + i - j = 8 + 5 + 0 - 2 = 11
示例 2：

输入：values = [1,2]
输出：2
```

代码：

1. 找规律

    最大化的式子可以拆分成`values[i] + i + values[j] - j`，其中`values[j] - j`是固定的，我们只需要维护`values[i] + i`的最大值就可以了。这样只需要遍历一遍数组。

    ```c++
    class Solution {
    public:
        int maxScoreSightseeingPair(vector<int>& values) {
            int max_left = values[0];
            int ans = 0;
            for (int i = 1; i < values.size(); ++i)
            {
                ans = max(ans, max_left + values[i] - i);
                max_left = max(max_left, values[i] + i);
            }
            return ans;
        }
    };
    ```

    这首题似乎和动态规划有一定关联。但是关联又不那么强。不知道该怎么分解最优子问题。

### 图

#### 二分图

存在一个 无向图 ，图中有 n 个节点。其中每个节点都有一个介于 0 到 n - 1 之间的唯一编号。

给定一个二维数组 graph ，表示图，其中 graph[u] 是一个节点数组，由节点 u 的邻接节点组成。形式上，对于 graph[u] 中的每个 v ，都存在一条位于节点 u 和节点 v 之间的无向边。该无向图同时具有以下属性：

不存在自环（graph[u] 不包含 u）。
不存在平行边（graph[u] 不包含重复值）。
如果 v 在 graph[u] 内，那么 u 也应该在 graph[v] 内（该图是无向图）
这个图可能不是连通图，也就是说两个节点 u 和 v 之间可能不存在一条连通彼此的路径。
二分图 定义：如果能将一个图的节点集合分割成两个独立的子集 A 和 B ，并使图中的每一条边的两个节点一个来自 A 集合，一个来自 B 集合，就将这个图称为 二分图 。

如果图是二分图，返回 true ；否则，返回 false 。

 
```
示例 1：

输入：graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
输出：false
解释：不能将节点分割成两个独立的子集，以使每条边都连通一个子集中的一个节点与另一个子集中的一个节点。
示例 2：



输入：graph = [[1,3],[0,2],[1,3],[0,2]]
输出：true
解释：可以将节点分成两组: {0, 2} 和 {1, 3} 。
```

代码：

1. bfs 染色。如果从某个节点出发，碰到相邻节点和自己同色，那么必定无法二分。

    ```c++
    class Solution {
    public:
        vector<int> colors;

        bool bfs(vector<vector<int>> &graph, int s)
        {
            colors[s] = 0;
            queue<int> q;
            q.push(s);
            int rnode;
            while (!q.empty())
            {
                rnode = q.front();
                q.pop();
                for (int i = 0; i < graph[rnode].size(); ++i)
                {
                    if (colors[graph[rnode][i]] == -1)  // 未染色的相邻节点染上和自己反色
                    {
                        colors[graph[rnode][i]] = 1 - colors[rnode];
                        q.push(graph[rnode][i]);
                    }
                    else if (colors[graph[rnode][i]] == colors[rnode]) return false;  // 若相邻节点和自己同色，那么一定无法二分
                }
            }
            return true;
        }

        bool isBipartite(vector<vector<int>>& graph) {
            colors.assign(graph.size(), -1);
            for (int i = 0; i < graph.size(); ++i)
            {
                if (colors[i] == -1)  // 可能有多个子图，所以一个子图一个子图判断
                {
                    if (!bfs(graph, i)) return false;
                }
            }
            return true;
        }
    };
    ```

1. dfs

#### 找到小镇的法官

小镇里有 n 个人，按从 1 到 n 的顺序编号。传言称，这些人中有一个暗地里是小镇法官。

如果小镇法官真的存在，那么：

小镇法官不会信任任何人。
每个人（除了小镇法官）都信任这位小镇法官。
只有一个人同时满足属性 1 和属性 2 。
给你一个数组 trust ，其中 trust[i] = [ai, bi] 表示编号为 ai 的人信任编号为 bi 的人。

如果小镇法官存在并且可以确定他的身份，请返回该法官的编号；否则，返回 -1 。

 

```
示例 1：

输入：n = 2, trust = [[1,2]]
输出：2
示例 2：

输入：n = 3, trust = [[1,3],[2,3]]
输出：3
示例 3：

输入：n = 3, trust = [[1,3],[2,3],[3,1]]
输出：-1
```

代码：

1. 统计入度和出度

    ```c++
    class Solution {
    public:
        int findJudge(int n, vector<vector<int>>& trust) {
            vector<int> in_degree(n), out_degree(n);
            int m = trust.size();
            for (int i = 0; i < m; ++i)
            {
                ++out_degree[trust[i][0]-1];
                ++in_degree[trust[i][1]-1];
            }
            for (int i = 0; i < n; ++i)
            {
                if (in_degree[i] == n - 1 && out_degree[i] == 0)
                    return i+1;
            }
            return -1;
        }
    };
    ```

#### 可以到达所有点的最少点数目

给你一个 有向无环图 ， n 个节点编号为 0 到 n-1 ，以及一个边数组 edges ，其中 edges[i] = [fromi, toi] 表示一条从点  fromi 到点 toi 的有向边。

找到最小的点集使得从这些点出发能到达图中所有点。题目保证解存在且唯一。

你可以以任意顺序返回这些节点编号。

 

```
示例 1：

输入：n = 6, edges = [[0,1],[0,2],[2,5],[3,4],[4,2]]
输出：[0,3]
解释：从单个节点出发无法到达所有节点。从 0 出发我们可以到达 [0,1,2,5] 。从 3 出发我们可以到达 [3,4,2,5] 。所以我们输出 [0,3] 。
示例 2：


输入：n = 5, edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]
输出：[0,2,3]
解释：注意到节点 0，3 和 2 无法从其他节点到达，所以我们必须将它们包含在结果点集中，这些点都能到达节点 1 和 4 。
```

代码：

1. 找入度为 0 的节点

    ```c++
    class Solution {
    public:
        vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
            vector<int> in_degree(n);
            for (int i = 0; i < edges.size(); ++i)
                ++in_degree[edges[i][1]];

            vector<int> ans;
            for (int i = 0; i < n; ++i)
                if (in_degree[i] == 0) ans.push_back(i);
            return ans;
        }
    };
    ```

#### 钥匙和房间

有 n 个房间，房间按从 0 到 n - 1 编号。最初，除 0 号房间外的其余所有房间都被锁住。你的目标是进入所有的房间。然而，你不能在没有获得钥匙的时候进入锁住的房间。

当你进入一个房间，你可能会在里面找到一套不同的钥匙，每把钥匙上都有对应的房间号，即表示钥匙可以打开的房间。你可以拿上所有钥匙去解锁其他房间。

给你一个数组 rooms 其中 rooms[i] 是你进入 i 号房间可以获得的钥匙集合。如果能进入 所有 房间返回 true，否则返回 false。

 

```
示例 1：

输入：rooms = [[1],[2],[3],[]]
输出：true
解释：
我们从 0 号房间开始，拿到钥匙 1。
之后我们去 1 号房间，拿到钥匙 2。
然后我们去 2 号房间，拿到钥匙 3。
最后我们去了 3 号房间。
由于我们能够进入每个房间，我们返回 true。
示例 2：

输入：rooms = [[1,3],[3,0,1],[2],[0]]
输出：false
解释：我们不能进入 2 号房间。
```

代码：

1. 自己写的，bfs，看看是否所有节点都能到达

    ```c++
    class Solution {
    public:
        bool canVisitAllRooms(vector<vector<int>>& rooms) {
            vector<bool> cnt(rooms.size());
            queue<int> q;
            q.push(0);
            int cur, next;
            while (!q.empty())
            {
                cur = q.front();
                q.pop();
                for (int i = 0; i < rooms[cur].size(); ++i)
                {
                    next = rooms[cur][i];
                    if (!cnt[next])
                    {
                        q.push(next);
                        cnt[next] = true;
                    }
                }
            }

            for (int i = 1; i < rooms.size(); ++i)
                if (!cnt[i]) return false;
            return  true;
        }
    };
    ```

1. 答案给的 dfs

    ```c++
    class Solution {
    public:
        vector<int> vis;
        int num;

        void dfs(vector<vector<int>>& rooms, int x) {
            vis[x] = true;
            num++;
            for (auto& it : rooms[x]) {
                if (!vis[it]) {
                    dfs(rooms, it);
                }
            }
        }

        bool canVisitAllRooms(vector<vector<int>>& rooms) {
            int n = rooms.size();
            num = 0;
            vis.resize(n);
            dfs(rooms, 0);
            return num == n;
        }
    };
    ```

1. 答案给的 bfs，直接统计能到达的房间的总数就可以了

    ```c++
    class Solution {
    public:
        bool canVisitAllRooms(vector<vector<int>>& rooms) {
            int n = rooms.size(), num = 0;
            vector<int> vis(n);
            queue<int> que;
            vis[0] = true;
            que.emplace(0);
            while (!que.empty()) {
                int x = que.front();
                que.pop();
                num++;
                for (auto& it : rooms[x]) {
                    if (!vis[it]) {
                        vis[it] = true;
                        que.emplace(it);
                    }
                }
            }
            return num == n;
        }
    };
    ```

#### 字符串中第二大的数字

给你一个混合字符串 s ，请你返回 s 中 第二大 的数字，如果不存在第二大的数字，请你返回 -1 。

混合字符串 由小写英文字母和数字组成。

 
```
示例 1：

输入：s = "dfa12321afd"
输出：2
解释：出现在 s 中的数字包括 [1, 2, 3] 。第二大的数字是 2 。
```

```
示例 2：

输入：s = "abc1111"
输出：-1
解释：出现在 s 中的数字只包含 [1] 。没有第二大的数字。
```

提示：

* 1 <= s.length <= 500
* s 只包含小写英文字母和（或）数字。

代码：

1. c++

    可以用`set`自动排序的特性，可以使用`priority_queue`大顶堆。不过这道题因为只需要找到第 2 大的数字，所以可以直接模拟：

    ```c++
    class Solution {
    public:
        int secondHighest(string s) {
            int n1 = -1, n2 = -1;
            int n;
            for (char &c: s)
            {
                if (c >= '0' && c <= '9')
                {
                    n = c - '0';
                    if (n > n1)
                    {
                        n2 = n1;
                        n1 = n;
                    }
                    else if (n < n1)
                    {
                        if (n > n2) n2 = n;
                    }
                }
            }
            return n2;
        }
    };
    ```

    假如要求使用`secondHighest(string s, int k)`返回第`k`大的数字，那么就没办法使用模拟了，这时候该怎么办？

    由于这道题中数字只有 0 ~ 9，或许可以用这个特性做些优化。

1. rust

    仿照 c++ 的写法。占的内存比别人的多，不知道该怎么缩减内存了。

    ```rust
    impl Solution {
        pub fn second_highest(s: String) -> i32 {
            let mut n1 = -1;
            let mut n2 = -1;
            let mut n;
            for c in s.bytes() {
                if c >= b'0' && c <= b'9' {
                    n = (c - b'0') as i32;
                    if n > n1 {
                        n2 = n1;
                        n1 = n;
                    }
                    else if n < n1 && n > n2 {
                        n2 = n;
                    }
                }
            }
            return n2;
        }
    }
    ```

## 各种算法中需要注意的细节

### bfs

1. 我们需要区别哪些位置处理过了，哪些还没有。如果不加区分，轻则会导致重复处理一些点，重则会导致死循环。

    通常情况下，处理过的地方用别的数字和未处理过的地方区别开。如果无法区分，则会导致死循环。

    我们在赋新值时，最好是在`for()`里赋值，这样可以保证处理过的点不会被添加到队列中。同时应注意第一个位置需要在`where`循环外处理。
    
    如果我们在`q.pop()`后赋新值，那么就必须要加上`if (grid[sx][sy] == old_value) continue;`避免重复处理。

### 动态规划

1. 开辟数组的大小究竟是`v.size()`还是`v.size()+1`，取决于`dp[i]`代表的意义。如果这个意义是从`0`开始索引，那么直接用`v.size()`大小的数组 就可以了；如果`i`代表的意义是个数，那么就需要`v.size()+1`大小的数组。

1. 在`for()`循环中，`i`是从 0 开始还是从 1 开始，或者是从 2 开始，取决于`dp[i] = xxx`的表达式。如果式中出现了`dp[i-1]`，那么要么需要加上条件`if (i > 0)`，要么`i`就需要从 1 开始。`i`是否从 2 开始也同理。

    如果`i`代表的意义是计数，那么也会存在从 1 开始的情况。

1. 常见的问题模板：从一些数字中找多少个数，使得和为定值，问有几种找法，或找法数最小是多少。

### 滑动窗口

1. 常见的要求是连续子列

### 回溯

有关回溯的思考：回溯的整体思想是每个位置把所有可能的情况都试一遍，其特征代码是先选择再撤销。

1. 为何要用 for ？什么时候需要用 for，什么时候不需要用？

### 位运算

1. `n & (n-1)`返回的是将`n`中二进制最低位的`1`置`0`后的数字。

1. `n & (-n)`可以获取二进制表示的最低位的 1

1. `a ^ b`得到的是`a`和`b`丢失进位情况的相加结果，`(a & b) << 1`得到的是进位情况。

1. 将从低向高数的第`n`位置 1。置 1 使用`|=`。

    ```cpp
    int n = 3;
    int bit = 1;
    int num = 0;
    num |= bit << n-1;  // 00000000000000000000000000000100
    ```

1. 将从低向高数的第`n`位置 0。置 0 使用`&=`。

    ```cpp
    int n = 3;
    int bit = 1;
    int num = 0xffffffff;
    num &= ~(bit << n-1);  // 11111111111111111111111111111011
    ```

1. 判断从低往高第`n`位是否为 1

    ```cpp
    int n = 3;
    int num;
    bool is_bit_1 = (num >> n) & 1;
    ```

1. 判断从高往低第`n`位是否为 1

    ```cpp
    bool is_bit_1 = (num >> (31 - n)) & 1;
    ```