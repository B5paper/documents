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

### 和相等的子数组

给你一个下标从 0 开始的整数数组 nums ，判断是否存在 两个 长度为 2 的子数组且它们的 和 相等。注意，这两个子数组起始位置的下标必须 不相同 。

如果这样的子数组存在，请返回 true，否则返回 false 。

子数组 是一个数组中一段连续非空的元素组成的序列。

 

示例 1：

输入：nums = [4,2,4]
输出：true
解释：元素为 [4,2] 和 [2,4] 的子数组有相同的和 6 。
示例 2：

输入：nums = [1,2,3,4,5]
输出：false
解释：没有长度为 2 的两个子数组和相等。
示例 3：

输入：nums = [0,0,0]
输出：true
解释：子数组 [nums[0],nums[1]] 和 [nums[1],nums[2]] 的和相等，都为 0 。
注意即使子数组的元素相同，这两个子数组也视为不相同的子数组，因为它们在原数组中的起始位置不同。
 

提示：

2 <= nums.length <= 1000
-109 <= nums[i] <= 109

代码：

1. 一开始自己写的，没有看到长度固定为 2，所以超时了

    ```cpp
    class Solution {
    public:
        bool findSubarrays(vector<int>& nums) {
            int n = nums.size();
            long long sum_1 = 0, sum_2 = 0;
            vector<long long> presum(n, 0);
            presum[0] = nums[0];
            for (int i = 1; i < n; ++i)
                presum[i] = presum[i-1] + nums[i];
            for (int i1 = 0; i1 < n - 1; ++i1)
            {
                for (int i2 = i1; i2 < n; ++i2)
                {
                    sum_1 = presum[i2] - presum[i1] + nums[i1];
                    for (int j1 = i1+1; j1 < n; ++j1)
                    {
                        for (int j2 = j1+1; j2 < n; ++j2)
                        {
                            sum_2 = presum[j2] - presum[j1] + nums[j1];
                            if (sum_1 == sum_2 && j2 - j1 + 1 == 2 && i2 - i1 + 1 == 2)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }
    };
    ```

    这样写的其实是按长度不固定的情况来写的。

1. 二重循环

    ```cpp
    class Solution {
    public:
        bool findSubarrays(vector<int>& nums) {
            int n = nums.size();
            int sum_1 = 0, sum_2 = 0;
            for (int i = 0; i < n - 1; ++i)
            {
                sum_1 = nums[i] + nums[i+1];
                for (int j = i + 1; j < n - 1; ++j)
                {
                    sum_2 = nums[j] + nums[j+1];
                    if (sum_1 == sum_2)
                        return true;
                }
            }
            return false;
        }
    };
    ```

    为什么这道题可以使用二重循环？因为题目中说数组中的元素是整数，所以没法用双指针构造滑动窗口。

1. 哈希表

    ```cpp
    class Solution {
    public:
        bool findSubarrays(vector<int>& nums) {
            int n = nums.size();
            unordered_set<int> s;
            int sum;
            for (int i = 0; i < n - 1; ++i)
            {
                sum = nums[i] + nums[i+1];
                if (s.find(sum) == s.end())
                    s.insert(sum);
                else
                    return true;
            }
            return false;
        }
    };
    ```

    完全类似两数之和的思路。

1. 先求出所有的2元素数组和，再判断是否有重复

    判断重复的方法有两种，一种是排序，看是否存在相邻的重复元素；另一种是使用哈希表，看哈希表的大小是否等于数组的长度。

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

1. 题目的意思显然是不让用排序，那首先我们写一个暴力的思路，即三重循环搜索

    对于数组中的每一个数，我们都假设它是一个连续子序列的起点，然后开始尝试向后搜索，直到找不到为止。

    ```cpp
    int get_ans(vector<int> &nums)
    {
        int max_len = 1;
        for (int i = 0; i < nums.size(); ++i)
        {
            int start = nums[i];
            int cur = start + 1;
            bool found = true;
            int len = 1;
            while (found)
            {
                found = false;
                for (int j = 0; j < nums.size(); ++j)
                {
                    if (nums[j] == cur)
                    {
                        found = true;
                        break;
                    }
                }
                if (found)
                {
                    ++len;
                    ++cur;
                }
            }
            max_len = max(len, max_len);
        }
        return max_len;
    }
    ```

    对于每个起点，向后搜索每一个数时，都需要重新遍历数组。假设一共有 m 个连续序列的区间，序列平均长度为 v，数组的总长度为 n，那么时间复杂度就是`O(mvn)`。

    假如所有数字都不连续，即`v = 1`，那么`m = n`，时间复杂度为`O(n^2)`。

    假如整个区间只有一个连续序列，即`m = 1`，那么`v = n`，时间复杂度为`O(n^2)`。

2. 为了提高搜索效率，可以将数据都存入哈希表中

    ```cpp
    int get_ans(vector<int> &nums)
    {
        unordered_set<int> s;
        for (int num: nums)
            s.insert(num);
        int max_len = 1;
        for (int i = 0; i < nums.size(); ++i)
        {
            int start = nums[i];
            int cur = start + 1;
            int len = 1;
            while (s.find(cur) != s.end())
            {
                ++len;
                ++cur;
            }
            max_len = max(len, max_len);
        }
        return max_len;
    }
    ```

    这样就省去了最内层的遍历，我们只需要对于每个起始位置的数字，在向后搜索时，都从哈希表里找就可以了。

    此时的时间复杂度为`O(n + mv)`。

3. 由于我们每次都从一个假定的起始位置向后搜索，所以实际上是做了许多重复的搜索的

    比如数组`[5, 3, 1, 2, 4]`，从`1`开始搜索了`[2, 3, 4, 5]`；从 2 开始又搜索了`[3, 4, 5]`，从 3 开始搜索了`[4, 5]`，从 4 开始搜索了`[5]`。

    为了避免这些重复搜索，对于序列`[a, ..., b, ..., c]`，假设我们刚开始遇到的是`b`，那么先向前搜索到`a`，再向后搜索到`c`。这样只需要搜索一遍就可以找到整个序列。对于已经搜索过的数字，我们也将它们存在哈希表里，防止被二次搜索。

    ```cpp
    int get_ans(vector<int> &nums)
    {
        unordered_set<int> s, vis;
        for (int num: nums)
            s.insert(num);
        int max_len = 1;
        for (int i = 0; i < nums.size(); ++i)
        {
            if (vis.find(nums[i]) != vis.end())
                continue;
            int start = nums[i];
            int pre = start - 1;
            int nex = start + 1;
            while (s.find(pre) != s.end())
            {
                vis.insert(pre);
                --pre;
            }
            while (s.find(nex) != s.end())
            {
                vis.insert(nex);
                ++nex;
            }
            int len = (nex - 1) - (pre + 1) + 1;
            max_len = max(len, max_len);
        }
        return max_len;
    }
    ```

    这份代码可以通过，但只能击败 5%.

    目前还不清楚官方答案是怎么想出来的，功力没有达到。

4. 官方答案，使用散列表 + 剪枝

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

1. 首先想到的是暴力解法双循环

    将任意一个位置`i`作为起点，然后遍历`i+1`一直到数组末尾，找一个终点，不断计算面积，然后比较其和最大面积谁大。

    ```cpp
    class Solution {
    public:
        int maxArea(vector<int>& height) {
            int max_area = 0;
            for (int i = 0; i < height.size(); ++i)
            {
                int h1 = height[i];
                for (int j = i+1; j < height.size(); ++j)
                {
                    int h2 = height[j];
                    int area = (j - i) * min(h1, h2);
                    max_area = max(area, max_area);
                }
            }
            return max_area;
        }
    };
    ```

    这个方法当然会超时。

2. 我们研究一下有没有可以剪枝的地方

    对于已经找到的一组`(i, j, h1, h2, area)`，`i`不变时，`j`一定是最优解。

    这时再增加`i`，如果`h1`变得越来越低，那么就没必要再遍历`j`了。

    假如根据上次的`j`再向右找解，由于我们知道上次的`j`已经是最优解，所以右侧的`j`与当前的`i`肯定比最优解差一点。而随着`i`的增加，`h1`的降低，面积只会越来越小。因此`j`右侧的就不用再找了。

    对于`j`左侧，同理，`j - 1`会减小，再加上`h1`减小，面积也只会越来越小。

    综上，如果`h1`变小，那么没必要再遍历`j`了。由此写出代码：

    ```cpp
    class Solution {
    public:
        int maxArea(vector<int>& height) {
            int max_area = 0;
            int hl = -1, hr = -1;
            for (int i = 0; i < height.size(); ++i)
            {
                int h1 = height[i];
                if (h1 < hl)
                    continue;
                for (int j = i+1; j < height.size(); ++j)
                {
                    int h2 = height[j];
                    int area = (j - i) * min(h1, h2);
                    if (area > max_area)
                    {
                        max_area = area;
                        hl = h1;
                        hr = h2;
                    }
                }
            }
            return max_area;
        }
    };
    ```

    可以感受到，这其中似乎有一种不等式传递的感觉。有空了深入探索一下。

    这个方法只能击败 5%。

    我们可以对称地想，对于最优的`i`，将`j`从右往左搜索，如果`j`对应的`h2`越来越小，那么也可以跳过这些`j`：

    ```cpp
    class Solution {
    public:
        int maxArea(vector<int>& height) {
            int max_area = 0;
            int hl = -1, hr = -1;
            for (int i = 0; i < height.size(); ++i)
            {
                int h1 = height[i];
                if (h1 < hl)
                    continue;
                for (int j = height.size() - 1; j > i; --j)
                {
                    int h2 = height[j];
                    if (h2 < hr)
                        continue;
                    int area = (j - i) * min(h1, h2);
                    if (area > max_area)
                    {
                        max_area = area;
                        hl = h1;
                        hr = h2;
                    }
                }
            }
            return max_area;
        }
    };
    ```

    由于跳过的是内循环，所以其实对效率影响不大。

3. 官方答案：双指针

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

    核心思路是对撞双指针，然后每次都变动较短的一边。

    目前并不清楚是怎么想到双指针的。官方给出的解也是突然跳出来的，直接证明双指针的正确性，并没有解析是怎么优化得来的。

    （或许可以像方法二中那样思考，不断地变换固定左边界和右边界，自然地推导出双指针？）

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

5. 一个朴素但行不通的方法的尝试：找凹槽

    我们在遍历数组的时候，可以想象在下山和上山。每经历过一个下坡和上坡，都会形成一个凹槽，这个凹槽里一定可以接到雨水。那我们只需要找到这些凹槽，再叠加每个凹槽接到的雨水的面积就可以了。

    为了记录初始化、下坡和上坡的状态，我们可以使用一个变量来模拟状态机：

    ```cpp
    int get_ans(vector<int> &nums)
    {
        int ans = 0;
        int l = 0, r = l + 1;
        int state = 0;  // 1 for down, 2 for up
        bool downed = false;
        int concave_count = 0;
        while (r < nums.size())
        {
            if (state == 0)
            {
                if (nums[r] < nums[r-1])
                {
                    state = 1;
                    downed = true;
                    l = r - 1;
                    ++r;
                    continue;
                }
                ++r;
                continue;
            }

            if (state == 1)
            {
                if (nums[r] <= nums[r-1])
                    state = 1;
                else
                    state = 2;
                ++r;
                continue;
            }

            if (state == 2)
            {
                if (nums[r] >= nums[r-1])
                    state = 2;
                else
                {
                    state = 1;
                    concave_count++;
                    l = r - 1;
                }
                ++r;
                continue;
            }
        }

        if (downed && state == 2)
        {
            concave_count++;
        }
        ans = concave_count;
        return ans;
    }
    ```

    但是这样的理解其实是不准确的，如果遇到`w`形的凹槽（中间两个小的，包含在一个大的里），是算一个凹槽（一个大的），还是两个凹槽（两个小的），还是三个凹槽（一大两小）？

    这道题其实有点像括号的匹配，在下降沿时做记录，在上升沿时计算横向长度，抵消下降沿的记录。

    之所以想成这样，根本原因可能是没有想到各种凹槽的情况。

    联想节点：凹槽 - 单调栈

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

    后来又写的：

    ```cpp
    class Solution {
    public:
        vector<int> findAnagrams(string s, string p) {
            vector<int> ans;
            if (s.size() < p.size()) return ans;
            int cnt[26] = {0};
            for (int i = 0; i < p.size(); ++i)
            {
                --cnt[p[i] - 'a'];
                ++cnt[s[i] - 'a'];
            }

            bool all_zero = true;
            for (int i = 0; i < 26; ++i)
            {
                if (cnt[i] != 0)
                {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero)
                ans.push_back(0);
            for (int i = p.size(); i < s.size(); ++i)
            {
                ++cnt[s[i] - 'a'];
                --cnt[s[i-p.size()] - 'a'];
                all_zero = true;
                for (int i = 0; i < 26; ++i)
                {
                    if (cnt[i] != 0)
                    {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero)
                    ans.push_back(i - p.size() + 1);
            }
            return ans;
        }
    };
    ```

    后来写的改成了一个数组，不过总体原理和原来差不多。

    这道题应该还可以用 KMP 算法来做。回头试一下。

    我们还可以维护一个当前窗口`cnt`中 0 的数量，这样每次只比较这个数量是否等于 26 就可以了，不需要比较 26 次。

1. 先使得`p`中的词都在`s`中出现过，再判断长度是否正好是`p`的长度

    数组`cnt[26]`可以记录`int`值，正数表示`s`中出现过，`p`中没出现过；负数表示`p`中出现过，`s`中没出现过。

    我们在最外层循环里对`cnt`数组遍历，在内层循环中不断移动右指针，使得`cnt[i]`等 0，接下来检查下一个`cnt`中的字符。等`cnt`中的所有负数都大于等于 0 后，我们再移动左指针，使得`cnt`中所有的正数都等于 0。如果此时又有负数，那就继续移动右指针。直到`cnt`中所有数都等于 0，那么就可以记录左指针的位置到答案中了。

注：这道题没法用双指针+踩坑法做，因为这种方法没法处理`s = "abab"`，`p = "ab"`的情形。

1. 每次右指针只往右走一步，如果右指针出现了一个`p`中没有的字符，那么就收缩左指针，直到删去这个字符为止。此时比较滑动窗口的长度，若长度和`p`相同，那么就一定和`p`是异位词。

    我觉得这个方法利用了抽屉原理。如果现在有 5 个数字，取值`[1, 10]`，现在已知这 5 个数字的和是 5，请求这几个数字分别是多少？只有一种情况，那就是全是 1。我不知道这个原理叫什么名字。


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

    后面又写的，是完全按照这个思路来的：

    ```cpp
    class Solution {
    public:
        void setZeroes(vector<vector<int>>& matrix) {
            int n_row = matrix.size();
            int n_col = matrix[0].size();
            bool zero_first_row = false, zero_first_col = false;
            for (int i = 0; i < n_row; ++i)
            {
                if (matrix[i][0] == 0)
                    zero_first_col = true;
            }
            for (int j = 0; j < n_col; ++j)
            {
                if (matrix[0][j] == 0)
                    zero_first_row = true;
            }

            for (int i = 0; i < n_row; ++i)
            {
                for (int j = 0; j < n_col; ++j)
                {
                    if (matrix[i][j] == 0)
                    {
                        matrix[0][j] = 0;
                        matrix[i][0] = 0;
                    }
                }
            }

            for (int i = 1; i < n_row; ++i)
            {
                if (matrix[i][0] == 0)
                {
                    for (int j = 1; j < n_col; ++j)
                    {
                        matrix[i][j] = 0;
                    }
                }
            }

            for (int j = 1; j < n_col; ++j)
            {
                if (matrix[0][j] == 0)
                {
                    for (int i = 1; i < n_row; ++i)
                    {
                        matrix[i][j] = 0;
                    }
                }
            }

            if (zero_first_col)
            {
                for (int i = 0; i < n_row; ++i)
                {
                    matrix[i][0] = 0;
                }
            }
            if (zero_first_row)
            {
                for (int j = 0; j < n_col; ++j)
                {
                    matrix[0][j] = 0;
                }
            }
        }
    };
    ```

    感觉这道题考察的是一个数据依赖的问题，我们可以将清零的坐标缓存到第一行，第一列，但是是否清零第一行，第一列，又取决于第一行/第一列原有的数据，这个不能被覆盖，必须提前存起来。

    注意到第一行有 0 被清零只影响到第 1 行，第 1 列有 0 被清零只影响第 1 列。因此还需要额外两个变量来存储这个状态。

    这个数据依赖的问题，或许可以扩展到环形缓冲区，环形数组，反转链表？

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

* ai 讨论

    * 分析一下我的代码有什么问题： 

        ```cpp
        class Solution {
        public:
            double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
                int n1 = nums1.size();
                int n2 = nums2.size();
                int n = n1 + n2;
                int half_n = n / 2;
                int p1 = 0, p2 = 0;
                int cnt = 0;
                int cur;
                while (p1 < n1 && p2 < n2) {
                    if (cnt == half_n) {
                        break;
                    }
                    if (nums1[p1] < nums2[p2]) {
                        cur = nums1[p1];
                        ++p1;
                    } else {
                        cur = nums2[p2];
                        ++p2;
                    }
                    ++cnt;
                }
                return cur;
            }
        };
        ```

        这是一个经典的算法问题。你的代码直觉上是想通过**双指针遍历**来到达中位数的位置，但目前存在几个核心逻辑问题。

        1. 时间复杂度不符合要求

            题目明确要求时间复杂度为 $O(\log(m+n))$。
            你的代码使用的是双指针线性扫描，时间复杂度是 $O(m+n)$。虽然在 $m+n \le 2000$ 的数据规模下能过，但在面试或严格的算法评价中，这不符合预期。

        2. 代码逻辑的具体漏洞
            
            即便暂不考虑复杂度，代码在逻辑上也有以下几个问题：

            * **没有处理偶数情况：** 如果总长度 $n$ 是偶数，中位数应该是中间两个数的平均值值值，而你的代码只返回了一个 `cur`。
            * **提前跳出循环的逻辑错误：** 当 `cnt == half_n` 时你直接 `break` 了，但这会导致 `cur` 还是上一次迭代的值。且如果其中一个数组已经遍历完了，`while` 循环会直接终止，此时 `cnt` 可能还没达到 `half_n`。
            * **边界问题：** 如果其中一个数组为空，或者 `half_n` 很大，你的代码可能会跳过逻辑或返回未初始化的 `cur`。

        ---

        3. 正确的思路：二分查找 $O(\log(m+n))$

            要达到 $\log$ 级别，我们需要在两个数组上同时进行“折半查找”。我们可以通过寻找**第 $k$ 小的数**来解决这个问题。

            * 若 $m+n$ 是奇数，找第 $(m+n)/2 + 1$ 小的数。
            * 若 $m+n$ 是偶数，找第 $(m+n)/2$ 和第 $(m+n)/2 + 1$ 小的平均值。

        优化后的 C++ 代码示例：

        ```cpp
        class Solution {
        public:
            double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
                int totalLength = nums1.size() + nums2.size();
                if (totalLength % 2 == 1) {
                    return getKthElement(nums1, nums2, totalLength / 2 + 1);
                } else {
                    return (getKthElement(nums1, nums2, totalLength / 2) + 
                            getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
                }
            }

            int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {
                int m = nums1.size(), n = nums2.size();
                int index1 = 0, index2 = 0;

                while (true) {
                    // 边界情况
                    if (index1 == m) return nums2[index2 + k - 1];
                    if (index2 == n) return nums1[index1 + k - 1];
                    if (k == 1) return min(nums1[index1], nums2[index2]);

                    // 核心逻辑：每次排除 k/2 个元素
                    int newIndex1 = min(index1 + k / 2 - 1, m - 1);
                    int newIndex2 = min(index2 + k / 2 - 1, n - 1);
                    
                    if (nums1[newIndex1] <= nums2[newIndex2]) {
                        k -= (newIndex1 - index1 + 1);
                        index1 = newIndex1 + 1;
                    } else {
                        k -= (newIndex2 - index2 + 1);
                        index2 = newIndex2 + 1;
                    }
                }
            }
        };
        ```

        **为什么这个更好？**

        1.  **真正的 $O(\log(m+n))$**：每次比较后，我们都能排除掉其中一个数组中前 $k/2$ 个元素。
        2.  **严谨性**：处理了奇偶数情况，并且通过辅助函数 `getKthElement` 让逻辑非常清晰。
        3.  **空间优化**：不需要合并数组，空间复杂度为 $O(1)$。

        ---

        **你想让我详细解释一下这个“第 k 小元素”排除法的数学原理吗？**

    * 好啊，给我详细讲讲

        理解这个算法的关键在于：**我们不需要合并数组，只需要不断地“排除”掉不可能成为中位数的区间。**

        这个方法的核心逻辑是：如果我们想找第 $k$ 小的元素，就比较两个数组中第 $k/2$ 个元素的大小。

        ---

        1. 核心逻辑：二分排除法

            假设我们要找第 $k$ 小的数（对于中位数， $k \approx \frac{m+n}{2}$）。

            我们在 `nums1` 中取前 $k/2$ 个元素，在 `nums2` 中也取前 $k/2$ 个元素。
            设 $A = \text{nums1}[k/2 - 1]$，$B = \text{nums2}[k/2 - 1]$。

            * **如果 $A \le B$：**
                这意味着 `nums1` 中的前 $k/2$ 个元素，在合并后的有序数组中，**最多**也只能排在第 $k$ 位（即当 `nums2` 的前 $k/2-1$ 个元素全都比它们小时）。
                因此，`nums1` 的这前 $k/2$ 个元素**绝对不可能是第 $k$ 小的数**，可以全部剔除。
            * **如果 $A > B$：**
                同理，`nums2` 中的前 $k/2$ 个元素都不可能是第 $k$ 小的数，可以全部剔除。

        2. 图解过程

            假设 `nums1 = [1, 3, 4, 9]`, `nums2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
            总长度 14，找中位数即找第 7 小和第 8 小的平均值。我们以找 **第 7 小 ($k=7$)** 为例：

            1.  **第一轮 ($k=7, k/2=3$)：**
                * 比较 `nums1` 的第 3 个数 (**4**) 和 `nums2` 的第 3 个数 (**3**)。
                * 因为 $4 > 3$，所以排除 `nums2` 的前 3 个数 `[1, 2, 3]`。
                * **剩余 $k = 7 - 3 = 4$。**

            2.  **第二轮 ($k=4, k/2=2$)：**
                * 比较 `nums1` 的前 2 个数中的第 2 个 (**3**) 和 `nums2` (剩余部分) 的第 2 个 (**5**)。
                * 因为 $3 < 5$，所以排除 `nums1` 的前 2 个数 `[1, 3]`。
                * **剩余 $k = 4 - 2 = 2$。**

            3.  **第三轮 ($k=2, k/2=1$)：**
                * 比较 `nums1` (剩余 `[4, 9]`) 的第 1 个 (**4**) 和 `nums2` (剩余 `[4, 5...]`) 的第 1 个 (**4**)。
                * 相等时随便删一边，假设删 `nums1` 的第一个数 `[4]`。
                * **剩余 $k = 2 - 1 = 1$。**

            4.  **第四轮 ($k=1$)：**
                * 当 $k=1$ 时，只需比较两个数组当前的最小值。
                * `min(nums1[0], nums2[0])`，此时 `nums1` 剩 `[9]`，`nums2` 剩 `[4, 5...]`。
                * 结果为 **4**。

        ---

        3. 三个边界条件（必须处理）

            在代码实现中，有三种情况会打破上面的递归/循环：

            1.  **数组越界**：如果一个数组的长度小于 $k/2$，我们直接取该数组的**最后一个**元素进行比较。剔除时也相应地剔除实际比较的个数。
            2.  **数组为空**：如果一个数组被剔除光了，剩下的第 $k$ 小元素直接从另一个数组里数出来即可。
            3.  **$k=1$**：当我们要找第 1 小的元素时，直接比较两个数组头部的最小值。

        4. 为什么复杂度是 $O(\log(m+n))$？

            因为每次比较我们都会剔除掉 $\approx k/2$ 个元素。
            而 $k$ 的初始值约为 $\frac{m+n}{2}$。
            这种**每次规模减半**的操作，正是对数级复杂度的典型特征。

        ---

        **总结**

        你的原始代码是“一步一个脚印”去数，而这个算法是“成块成块地扔掉”。

        **你觉得这个“排除法”在处理偶数情况（需要调用两次函数）时，会有性能浪费吗？还是你想看看更极致的“一次二分”解法？**

    * 好啊，都讲讲

        这是一个非常棒的进阶思考。我们先聊聊你刚才担心的“性能浪费”，再带你看那个被公认为中位数问题“终极方案”的**割点二分法**。

        ---

        1. 两次调用 `getKthElement` 是浪费吗？

            在 $m+n$ 为偶数时，我们需要找第 $k$ 小和第 $k+1$ 小的数。

            * **直觉上：** 感觉找了两次，时间翻倍了。
            * **实际上：** 在大 $O$ 复杂度分析中，$2 \cdot \log(m+n)$ 依然等于 $O(\log(m+n))$。
            * **微观性能：** 确实存在重复计算。因为找第 $k$ 小的过程和找第 $k+1$ 小的过程路径高度重合。但在 LeetCode 或实际面试中，这种写法因为**逻辑极其清晰、不容易写错**，通常是首选。

            如果你追求极致的“一次搞定”，那就必须祭出下面这个基于**划分数组（Partition）**的方法。

        ---

        2. 极致解法：割点二分法 (Partition Algorithm)

            这个方法的逻辑不是“找第 $k$ 个数”，而是“把两个数组各切一刀”。

            **核心思想**

            我们要把 `nums1` 切成两半（左 $A_1$, 右 $A_2$），把 `nums2` 也切成两半（左 $B_1$, 右 $B_2$）。
            使得：
            1.  **数量相等：** 左半部分的总数 $= \text{右半部分的总数}$（或者多一个）。
            2.  **左小右大：** 左边所有数 $\le$ 右边所有数。

            此时，中位数就在这四个分界点数值之间！

            **如何寻找切割点？**

            我们只需要在**较短的数组**上进行二分查找。

            假设 `nums1` 长度为 $m$，切割位置为 $i$（左边有 $i$ 个元素）。
            为了满足数量相等，`nums2` 的切割位置 $j$ 必须是 $\frac{m+n+1}{2} - i$。

            **我们要寻找一个 $i$，使得：**
            * $nums1[i-1] \le nums2[j]$ （$A$ 的左边最大 $\le B$ 的右边最小）
            * $nums2[j-1] \le nums1[i]$ （$B$ 的左边最大 $\le A$ 的右边最小）

            **割点二分法的 C++ 实现**

            ```cpp
            class Solution {
            public:
                double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
                    // 确保 nums1 是短的那个，这样二分范围更小，且 j 不会是负数
                    if (nums1.size() > nums2.size()) return findMedianSortedArrays(nums2, nums1);

                    int m = nums1.size();
                    int n = nums2.size();
                    int left = 0, right = m;
                    
                    // 我们要在 [0, m] 之间寻找切割位置 i
                    while (left <= right) {
                        int i = (left + right) / 2;
                        int j = (m + n + 1) / 2 - i;

                        // 边界处理：如果切割点在最左或最右，用无穷大/小代替
                        int nums1LeftMax = (i == 0) ? INT_MIN : nums1[i - 1];
                        int nums1RightMin = (i == m) ? INT_MAX : nums1[i];
                        int nums2LeftMax = (j == 0) ? INT_MIN : nums2[j - 1];
                        int nums2RightMin = (j == n) ? INT_MAX : nums2[j];

                        if (nums1LeftMax <= nums2RightMin && nums2LeftMax <= nums1RightMin) {
                            // 找到了完美的切割点！
                            if ((m + n) % 2 == 1) {
                                return max(nums1LeftMax, nums2LeftMax);
                            } else {
                                return (max(nums1LeftMax, nums2LeftMax) + min(nums1RightMin, nums2RightMin)) / 2.0;
                            }
                        } else if (nums1LeftMax > nums2RightMin) {
                            // nums1 左边太大了，切割点 i 往左挪
                            right = i - 1;
                        } else {
                            // nums1 左边太小了，切割点 i 往右挪
                            left = i + 1;
                        }
                    }
                    return 0.0;
                }
            };
            ```

        ---

        3. 两种 $O(\log(m+n))$ 方法的对比

            | 特性 | **方法一：找第 K 小 (Kth Element)** | **方法二：割点二分 (Partition)** |
            | :--- | :--- | :--- |
            | **思维难度** | 较直观（类似二分查找） | 较抽象（需要考虑四个边界值） |
            | **代码量** | 略多（通常需要辅助函数） | 较少（一个循环搞定） |
            | **边界处理** | 主要是 $k=1$ 和数组空 | 需要处理 `INT_MIN/MAX` |
            | **适用场景** | **推荐：** 逻辑稳健，不容易写错 | **进阶：** 追求单次二分和极致性能 |

        **学习建议：**

        1.  **面试时**：先写出“找第 $k$ 小”的解法，因为它出错概率低，且能向面试官展示你对递归/迭代二分的深刻理解。
        2.  **刷题时**：尝试理解“割点二分”，这能极大提升你对“有序序列性质”的认知。

        **你现在的代码逻辑里，哪个部分让你觉得最难转弯？是处理奇偶数，还是二分时的索引变换？**

    * 先帮我把最开始那个双指针的代码修改好，我想先把最笨的办法跑通

        明白，咱们先把这个“数数”的笨办法打磨得圆圆满满。

        要让双指针法跑通，核心需要解决三个问题：
        1.  **记录两个位置**：因为偶数长度需要中间两个数的平均值，所以我们需要两个变量 `left`（上一个数）和 `right`（当前数）。
        2.  **准确的停止计数**：遍历到第 `(m+n)/2` 个元素就够了。
        3.  **越界处理**：当一个数组走完时，要能自动去另一个数组里继续数。

        **修改后的双指针代码 ($O(m+n)$)**

        ```cpp
        class Solution {
        public:
            double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
                int m = nums1.size();
                int n = nums2.size();
                int len = m + n;
                
                // left 用来存上一次遍历到的值，right 存当前遍历到的值
                // 这样无论是奇数还是偶数，最后都能拿到需要的中位数
                int left = -1, right = -1;
                int p1 = 0, p2 = 0;
                
                // 只需要遍历到总长度的一半位置即可
                for (int i = 0; i <= len / 2; ++i) {
                    left = right; // 迭代前，先把当前的给“上一个”
                    
                    // 判断该走哪一个指针：
                    // 1. p1 没走完
                    // 2. 且 (p2 走完了 或者 nums1[p1] 更小)
                    if (p1 < m && (p2 >= n || nums1[p1] < nums2[p2])) {
                        right = nums1[p1++];
                    } else {
                        right = nums2[p2++];
                    }
                }
                
                // 判断奇偶性
                if (len % 2 == 0) {
                    return (left + right) / 2.0;
                } else {
                    return right;
                }
            }
        };
        ```

        ---

        **你的原代码错在哪了？**

        对比一下，你的原稿主要有这三个“坑”：

        1.  **`cur` 的更新时机**：你的 `cur` 在 `break` 之后没有更新到最终目标位，且循环条件 `while (p1 < n1 && p2 < n2)` 导致如果一个数组为空，循环根本不会进去。
        2.  **单一变量的局限**：你只用了一个 `cur`。如果是偶数长度（比如 4），中位数是第 2 个和第 3 个的平均值，只存一个变量是拿不到“前一个”数的。
        3.  **停止条件 `cnt == half_n`**：这个判断在循环体开头，会导致你少走一步。

        **为什么这个“笨办法”也有价值？**

        虽然它的时间复杂度是 $O(m+n)$，不符合题目 $O(\log(m+n))$ 的进阶要求，但它的**空间复杂度是 $O(1)$**。相比于直接把两个数组 `merge` 之后再找中位数（空间 $O(m+n)$），这种双指针法在内存消耗上已经是非常优秀的了。

        **这段代码在 LeetCode 上是可以直接通过的。你想现在运行试试，还是我再带你看看如何把这段逻辑封装成之前说的二分法？**

    * 为什么 i <= len / 2 时，left 和 right 正好处于中间？ 

        这是一个非常经典的问题。理解这个逻辑的关键在于：**中位数的位置是由总长度决定的，而循环的步数正好对应了我们在合并序列中“走”过的路程。**

        1. 直观理解：路程与终点
            
            想象你正要把两个排好序的数组合并成一个大的有序数组。
            * 如果总长度 `len` 是 **5**（奇数），中位数在索引 **2**（第3个数）。
            * 如果总长度 `len` 是 **4**（偶数），中位数是索引 **1** 和 **2**（第2、3个数）的平均值。

            当你循环执行 `i = 0` 到 `i <= len / 2` 时：

            | 总长度 `len` | `len / 2` 的值 | 循环次数 | `right` 最终停留的索引 | `left` 最终停留的索引 |
            | :--- | :--- | :--- | :--- | :--- |
            | **5** (奇数) | 2 | 3次 (0, 1, 2) | **索引 2** (中位数) | 索引 1 |
            | **4** (偶数) | 2 | 3次 (0, 1, 2) | **索引 2** (右中位数) | **索引 1** (左中位数) |

        ---

        2. 为什么 `left` 和 `right` 配合得刚刚好？

            代码中有一行非常关键：`left = right;`。这保证了 **`left` 永远比 `right` 慢一步**。

            * **最后一次迭代前 ($i = len/2$)：**
                此时 `right` 存储的是索引为 $\frac{len}{2} - 1$ 的元素。
            * **进入最后一次迭代：**
                1.  `left = right;` 执行。此时 `left` 接管了索引 $\frac{len}{2} - 1$ 的值。
                2.  根据条件判断，指针移动，`right` 更新为索引 $\frac{len}{2}$ 的值。

        ---

        3. 结果拆解

            **情况 A：总长度为奇数（例如 $len = 5$）**

            * 循环结束时，`right` 到达了索引 $2$。
            * 根据定义，奇数长度的中位数就是中间那个数。
            * **代码返回：** `right`。完全正确。

            **情况 B：总长度为偶数（例如 $len = 4$）**

            * 循环结束时，`right` 到达了索引 $2$（右半部分的第一个数）。
            * 由于 `left` 慢一步，它停留在索引 $1$（左半部分的最后一个数）。
            * **代码返回：** $(left + right) / 2.0$。正好是两个中间数的平均值。

        ---

        **深度思考：复杂度小贴士**

        虽然这段代码逻辑非常清晰且节省了空间（空间复杂度 $O(1)$），但它的时间复杂度是 $O(m+n)$。

        在 LeetCode 的题目要求中，通常希望达到 **$O(\log(m+n))$**。那是通过**二分查找**（每次排除掉一半的元素）来实现的，逻辑会比这种双指针遍历复杂得多。

        **需要我为你演示一下如何用二分查找法将这个 $O(m+n)$ 优化到 $O(\log(m+n))$ 吗？**

    * 所以没有通用的定理来研究是否“到中间”这件事，更多的是基于观察法和归纳法？

        其实，这件事背后是有**严谨的数学逻辑**支持的，并不完全靠“观察和归纳”。

        在算法领域，这属于**线性扫描（Linear Scan）**和**双指针（Two Pointers）**的范畴。我们可以从以下三个维度来拆解这个“到中间”的定理：

        1. 索引偏移量定理 (Index Offset)
            
            对于任何长度为 $L$ 的有序序列（假设索引从 $0$ 开始），其中位数的逻辑位置是固定的：
            * **第 $k$ 小的数**，其索引是 $k-1$。
            * 中位数对应的“目标索引”位置是 $\lfloor \frac{L-1}{2} \rfloor$ 和 $\lceil \frac{L-1}{2} \rceil$。

            在你的代码中，循环条件是 `i <= len / 2`。
            * 当循环结束时，`i` 实际上运行了 $\lfloor \frac{len}{2} \rfloor + 1$ 次。
            * **结论：** 计数器 `i` 实际上是在模拟一个合并后的虚拟数组的索引。当你“走”到第 $\frac{len}{2}$ 步时，你自然就踩在了物理中点上。

        ---

        2. 状态转移方程（逻辑滞后）
            
            你代码中的 `left = right` 是一种经典的**状态平移（State Shifting）**。这在数字信号处理或动态规划中非常常见。

            如果我们定义 $S_i$ 为合并序列中第 $i$ 个元素，那么：
            * 在第 $i$ 次迭代开始时：`right` $= S_{i-1}$
            * 执行 `left = right` 后：`left` $= S_{i-1}$
            * 执行更新后：`right` $= S_i$

            这保证了在任何时刻，这两个变量构成了一个**滑动窗口**：
            $$[left, right] = [S_{i-1}, S_i]$$
            当循环停止在 $i = \lfloor \frac{len}{2} \rfloor$ 时，这个窗口正好卡在了中轴线上。

        ---

        3. “不变性”证明 (Invariant Proof)
            
            在计算机科学中，我们用**循环不变式 (Loop Invariant)** 来证明算法的正确性。

            对于这段代码，循环不变式可以描述为：
            > “在第 $i$ 次迭代结束时，`right` 总是等于两个有序数组合并后第 $i+1$ 小的元素。”

            * **初始化：** $i=0$ 时，`right` 拿到两个数组中最小的（第1小），成立。
            * **保持：** 如果第 $i$ 次成立，第 $i+1$ 次通过比较 `p1` 和 `p2` 指向的值，必然选出全局第 $i+2$ 小的数。
            * **终止：** 当 $i$ 达到 $\frac{len}{2}$ 时，`right` 必然是第 $\frac{len}{2} + 1$ 小的数。

        ---

        **总结：是定理还是直觉？**

        虽然看起来像“试出来的”，但它本质上是 **“归并排序（Merge Sort）的局部应用”** 结合 **“中位数定义”**。

        * **通用规律：** 如果你需要处理有序序列的“中间”或“相邻”关系，使用 `prev` 和 `curr` 两个变量随迭代平移，是一种处理**偶数/奇数对称性**的标准工程范式。

        这种方法虽然好理解，但在大数据量下，它本质上还是**线性查找**。

        **如果你感兴趣，我们可以聊聊那个更像“定理”的 $O(\log(n))$ 解法：它是基于“二分切分”和“左右部分数量相等”这个数学约束来暴力推导出的，那才是纯粹的数学美感。要看看那个推导过程吗？**

    * 为什么采用 for (int i = 0; i <= len / 2; ++i)，而不是 while (p1 < m && p2 < n)

        这是一个非常敏锐的观察！简单来说，`while (p1 < m && p2 < n)` 只能处理**两个数组都有剩余元素**的情况，而 `for` 循环（配合内部的 `if` 逻辑）处理的是**合并序列的全局进度**。

        以下是采用 `for` 循环的三大核心原因：

        1. 目标导向 vs. 条件导向
            
            * **`for (i = 0; i <= len / 2; ...)` 是“目标导向”：**
                我们的目标是找到合并后的第 $k$ 个元素（即中位数）。不管元素来自 `nums1` 还是 `nums2`，也不管其中一个数组是否已经取完，我们**必须走够这么多步**才能到达中点。
            * **`while (p1 < m && p2 < n)` 是“条件导向”：**
                这个循环在任何一个数组耗尽时就会**提前终止**。如果中位数恰好在那个较长数组的剩余部分里，你就永远走不到中点。

        ---

        2. 边界处理的简洁性
            
            如果使用 `while (p1 < m && p2 < n)`，你必须在循环结束后追加逻辑来处理未走完的数组。代码会变成这样：

            ```cpp
            // 这种写法非常冗长
            while (p1 < m && p2 < n && i <= len / 2) {
                left = right;
                right = (nums1[p1] < nums2[p2]) ? nums1[p1++] : nums2[p2++];
                i++;
            }
            // 还要处理 nums1 没走完的情况
            while (p1 < m && i <= len / 2) {
                left = right;
                right = nums1[p1++];
                i++;
            }
            // 还要处理 nums2 没走完的情况
            while (p2 < n && i <= len / 2) {
                left = right;
                right = nums2[p2++];
                i++;
            }
            ```

            而你给出的 `for` 循环通过内部的一个**复合判断条件**巧妙地融合了这三种情况：
            `if (p1 < m && (p2 >= n || nums1[p1] < nums2[p2]))`
            * `p1 < m`：确保 `nums1` 还有得选。
            * `p2 >= n`：**关键点！** 如果 `nums2` 已经走完了，即便 `nums1` 的值很大，也只能选 `nums1`。
            * `nums1[p1] < nums2[p2]`：常规的大小比较。

        ---

        3. 计数器的“虚拟合并”意义
            
            正如前面提到的，中位数是基于**总长度 `len`** 的物理位置。
            使用 `for (i = 0; i <= len / 2; ++i)` 实际上是在模拟：
            > “假设我已经把两个数组合并成了 `combined` 数组，我现在直接访问 `combined[0...len/2]`。”

            这种写法将**“谁小取谁”**的逻辑和**“取到第几个了”**的计数逻辑解耦了。计数器只管走步数，`if` 只管挑数字。

        **总结**

        * `while` 循环适合处理**“只要还有数据就继续”**的场景。
        * `for` 循环在这个题里代表**“必须精准移动到第 $k$ 步”**。

        这种写法在面试中非常受欢迎，因为它展示了你对**短路逻辑（Short-circuit logic）**的运用——即用一个 `if` 优雅地处理了数组越界的边界情况。

        **你对 `p2 >= n || nums1[p1] < nums2[p2]` 这一行中 `||` 的短路触发顺序（先判空再访问值）是否完全理解了呢？这可是防止程序崩溃的小细节。**

    * 再讲讲优先队列的解法吧

        虽然对于这道题来说，**优先队列（Heap/Priority Queue）**的效率不如双指针（$O(m+n)$）和二分法（$O(\log(m+n))$），但它是一种极其通用的**“求第 K 大/小”**的思维模型。

        在处理“从多个有序/无序序列中寻找中位数”时，优先队列是非常强大的工具。

        ---

        1. 思路一：合并 K 个有序链表的思维 ($O((m+n) \log 2)$)
            
            这种方法本质上是用一个小顶堆来模拟双指针。

            * **操作：** 把两个数组的第一个元素及其来源放入小顶堆。
            * **循环：** 每次弹出堆顶（最小的数），然后把这个数所在数组的下一个元素推入堆中。
            * **计数：** 弹出的第 $(m+n)/2$ 个和第 $(m+n)/2 + 1$ 个数就是我们要的。

            **评价：** 在只有两个数组时，这比双指针慢（因为维护堆需要 $\log$ 开销），但当数组变成 **K 个**时，这是最优解。

        ---

        2. 思路二：对顶堆（大顶堆 + 小顶堆）

            这是流数据（Dynamic Stream）求中位数的经典解法。它的核心是**维护两个堆**，将所有数字分为“较小的一半”和“较大的一半”。

            * **大顶堆 (Max-heap)：** 存储较小的那一半元素。堆顶是这一半里的**最大值**。
            * **小顶堆 (Min-heap)：** 存储较大的那一半元素。堆顶是这一半里的**最小值**。

            **核心性质：**

            1.  **数量平衡：** 大顶堆的大小等于小顶堆，或者比小顶堆多 1。
            2.  **有序：** 大顶堆的堆顶 $\le$ 小顶堆的堆顶。

            **此时中位数：**
            * 如果总数是奇数：中位数就是**大顶堆的堆顶**。
            * 如果总数是偶数：中位数就是**(大顶堆堆顶 + 小顶堆堆顶) / 2**。

        ---

        3. 代码实现（对顶堆版）

            虽然本题数组是静态的，直接用堆有点“杀鸡用牛刀”，但代码逻辑非常有启发性：

            ```cpp
            class Solution {
            public:
                double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
                    priority_queue<int> maxHeap; // 大顶堆（存左半边）
                    priority_queue<int, vector<int>, greater<int>> minHeap; // 小顶堆（存右半边）

                    // 辅助函数：向对顶堆添加元素并保持平衡
                    auto addNum = [&](int num) {
                        maxHeap.push(num);
                        minHeap.push(maxHeap.top());
                        maxHeap.pop();

                        if (maxHeap.size() < minHeap.size()) {
                            maxHeap.push(minHeap.top());
                            minHeap.pop();
                        }
                    };

                    // 把两个数组所有元素丢进去（这一步其实是 O((m+n)log(m+n)))
                    for (int x : nums1) addNum(x);
                    for (int x : nums2) addNum(x);

                    if ((nums1.size() + nums2.size()) % 2 != 0) {
                        return maxHeap.top();
                    } else {
                        return (maxHeap.top() + minHeap.top()) / 2.0;
                    }
                }
            };
            ```

        ---

        4. 方案对比：为什么这题不首选堆？

            | 方法 | 时间复杂度 | 空间复杂度 | 特点 |
            | :--- | :--- | :--- | :--- |
            | **二分法** | $O(\log(m+n))$ | $O(1)$ | **最优解**，符合题目限制。 |
            | **双指针** | $O(m+n)$ | $O(1)$ | **最稳**，面试保底方案。 |
            | **对顶堆** | $O((m+n) \log(m+n))$ | $O(m+n)$ | **最通用**，适合处理**动态数据流**。 |

            ### 什么时候该用堆？
            如果题目变成：“**数据是源源不断传过来的（Data Stream），请随时返回当前已接收数据的中位数**”，那么二分法和双指针就失效了（因为数组一直在变），这时候**对顶堆**就是唯一的王者。

        ---

        **总结一下：**
        你现在的武器库里已经有“小米加步枪”（双指针）、“狙击枪”（二分法）和“机关枪”（对顶堆）了。

        **对于这个特定的 LeetCode 第一难题，你觉得哪种思维方式最让你感到“惊艳”？**


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
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        stack<char> stk1, stk2;
        for (int i = 0; i < s.size(); ++i)
        {
            if (s[i] == '#')
            {
                if (!stk1.empty())
                    stk1.pop();
            }
            else
                stk1.push(s[i]);
        }
        for (int i = 0; i < t.size(); ++i)
        {
            if (t[i] == '#')
            {
                if (!stk2.empty())
                    stk2.pop();
            }
                
            else
                stk2.push(t[i]);
        }
        return stk1 == stk2;
    }
};
```

1. 使用倒序双指针。代码较复杂，但效率高。

```c++

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

思考：这道题是否可以用位运算实现呢？不可以，因为我们不知道重复的元素到底重复了几次。不管出现两次还是三次，四次，似乎都没啥好办法。

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

### 和大于等于 target 的最短子数组（长度最小的子数组）

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

    后来又写的（也会超时）：

    ```cpp
    class Solution {
    public:
        int minSubArrayLen(int target, vector<int>& nums) {
            int n = nums.size();
            vector<int> presum(n, 0);  // 不可以用 int presum[n] = {0}; 因为无法申请动态的数组。但是在本地的 c++ 编译器上是可以的。奇怪。
            presum[0] = nums[0];  // 申请数量为 n 的数组，不申请 n+1 个，这样更直观
            for (int i = 1; i < n; ++i)
            {
                presum[i] = presum[i-1] + nums[i];
            }

            int s = 0;
            int ans = INT32_MAX;
            for (int i = 0; i < n; ++i)
            {
                for (int j = i; j < n; ++j)
                {
                    s = presum[j] - presum[i] + nums[i];
                    if (s >= target)
                    {
                        ans = min(ans, j - i + 1);
                        break;
                    }
                }
            }
            if (ans == INT32_MAX) return 0;
            return ans;
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


### 矩阵对角线元素的和

给你一个正方形矩阵 mat，请你返回矩阵对角线元素的和。

请你返回在矩阵主对角线上的元素和副对角线上且不在主对角线上元素的和。

 

示例  1：



输入：mat = [[1,2,3],
            [4,5,6],
            [7,8,9]]
输出：25
解释：对角线的和为：1 + 5 + 9 + 3 + 7 = 25
请注意，元素 mat[1][1] = 5 只会被计算一次。
示例  2：

输入：mat = [[1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1]]
输出：8
示例 3：

输入：mat = [[5]]
输出：5
 

提示：

n == mat.length == mat[i].length
1 <= n <= 100
1 <= mat[i][j] <= 100

代码：

1. 自己写的

    ```cpp
    class Solution {
    public:
        int diagonalSum(vector<vector<int>>& mat) {
            int sum = 0;
            int n = mat.size();
            for (int i = 0; i < n; ++i)
            {
                sum += mat[i][i];
                if (n % 2 && i == n / 2) continue;  // 这里的判断太多了，只需要像答案一样在最后返回之前减去就好了
                sum += mat[i][n-i-1];
            }
            return sum;
        }
    };
    ```

1. 答案

    ```cpp
    class Solution {
    public:
        int diagonalSum(vector<vector<int>>& mat) {
            int n = mat.size(), sum = 0, mid = n / 2;
            for (int i = 0; i < n; ++i) {
                sum += mat[i][i] + mat[i][n - 1 - i];
            }
            return sum - mat[mid][mid] * (n & 1);  // 可以用 n % 2 判断奇偶，也可以用 n & 1 判断奇偶
        }
    };
    ```

### 找不同

给定两个字符串 s 和 t ，它们只包含小写字母。

字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。

请找出在 t 中被添加的字母。

 

示例 1：

输入：s = "abcd", t = "abcde"
输出："e"
解释：'e' 是那个被添加的字母。
示例 2：

输入：s = "", t = "y"
输出："y"
 

提示：

0 <= s.length <= 1000
t.length == s.length + 1
s 和 t 只包含小写字母

代码：

1. 数组统计

    ```cpp
    class Solution {
    public:
        char findTheDifference(string s, string t) {
            int cnt[26] = {0};
            for (int i = 0; i < s.size(); ++i)
                --cnt[s[i] - 'a'];
            for (int i = 0; i < t.size(); ++i)
            {
                ++cnt[t[i] - 'a'];
                if (cnt[t[i] - 'a'] == 1)
                    return t[i];
            }
            return 0;
        }
    };
    ```

1. 答案解法一，将字符串看作是 ascii 码，先求和，再相减

    ```cpp
    class Solution {
    public:
        char findTheDifference(string s, string t) {
            int as = 0, at = 0;
            for (char ch: s) {
                as += ch;
            }
            for (char ch: t) {
                at += ch;
            }
            return at - as;
        }
    };
    ```

1. 答案二，异或位运算求只出现一次的字符

    ```cpp
    class Solution {
    public:
        char findTheDifference(string s, string t) {
            int ret = 0;
            for (char ch: s) {
                ret ^= ch;
            }
            for (char ch: t) {
                ret ^= ch;
            }
            return ret;
        }
    };
    ```

### 区域和检索 - 数组可修改

给你一个数组 nums ，请你完成两类查询。

其中一类查询要求 更新 数组 nums 下标对应的值
另一类查询要求返回数组 nums 中索引 left 和索引 right 之间（ 包含 ）的nums元素的 和 ，其中 left <= right
实现 NumArray 类：

NumArray(int[] nums) 用整数数组 nums 初始化对象
void update(int index, int val) 将 nums[index] 的值 更新 为 val
int sumRange(int left, int right) 返回数组 nums 中索引 left 和索引 right 之间（ 包含 ）的nums元素的 和 （即，nums[left] + nums[left + 1], ..., nums[right]）
 
```
示例 1：

输入：
["NumArray", "sumRange", "update", "sumRange"]
[[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
输出：
[null, 9, null, 8]

解释：
NumArray numArray = new NumArray([1, 3, 5]);
numArray.sumRange(0, 2); // 返回 1 + 3 + 5 = 9
numArray.update(1, 2);   // nums = [1,2,5]
numArray.sumRange(0, 2); // 返回 1 + 2 + 5 = 8
 

提示：

1 <= nums.length <= 3 * 104
-100 <= nums[i] <= 100
0 <= index < nums.length
-100 <= val <= 100
0 <= left <= right < nums.length
调用 update 和 sumRange 方法次数不大于 3 * 104 
```

代码：

1. 直接求解，会超时

    ```cpp
    class NumArray {
    public:
        NumArray(vector<int>& nums) {
            this->nums = nums;
        }
        
        void update(int index, int val) {
            nums[index] = val;
        }
        
        int sumRange(int left, int right) {
            return accumulate(nums.begin() + left, nums.begin() + right + 1, 0);
        }

        private:
        vector<int> nums;
    };

    /**
     * Your NumArray object will be instantiated and called as such:
     * NumArray* obj = new NumArray(nums);
     * obj->update(index,val);
     * int param_2 = obj->sumRange(left,right);
     */
    ```

1. 分治法

    为了尽量利用已经计算过的值，我们可以对原数组分块，每次只更新有改动的块就可以了。

    ```cpp
    class NumArray {
    public:
        NumArray(vector<int>& nums) {
            this->nums = nums;
            block_size = sqrt(nums.size());
            int n = nums.size() / block_size;
            if (n * block_size == nums.size())
                block_sums.assign(n, 0);
            else
                block_sums.assign(n+1, 0);
            for (int i = 0; i < block_sums.size(); ++i)
            {
                for (int j = i * block_size; j < (i+1) * block_size && j < nums.size(); ++j)
                {
                    block_sums[i] += nums[j];
                }
            }
        }
        
        void update(int index, int val) {
            int idx = index / block_size;
            block_sums[idx] -= nums[index];
            block_sums[idx] += val;
            nums[index] = val;
        }
        
        int sumRange(int left, int right) {
            int ans = 0;
            int i_block = left / block_size;
            int j_block = right / block_size;
            if (i_block == j_block)
            {
                for (int i = left; i <= right; ++i)
                    ans += nums[i];
                return ans;
            }
            int left_end = (i_block + 1) * block_size - 1;
            int right_start = j_block * block_size;
            ans += accumulate(nums.begin()+left, nums.begin()+left_end+1, 0);
            ans += accumulate(nums.begin()+right_start, nums.begin()+right+1, 0);
            for (int i = i_block + 1; i < j_block; ++i)
                ans += block_sums[i];
            return ans;
        }

        private:
        vector<int> nums;
        vector<int> block_sums;
        int block_size;
    };

    /**
    * Your NumArray object will be instantiated and called as such:
    * NumArray* obj = new NumArray(nums);
    * obj->update(index,val);
    * int param_2 = obj->sumRange(left,right);
    */
    ```

1. 线段树

    将一个区间先分成两段，左叶子节点存储左半段的总和，右叶子节点存储右半段的总和，本节点存储当前区间的总和。

    每次更新节点，先找到叶子节点，然后自底向上更新路径。

    如果需要查询求和，只需要找到对应的节点相加就可以了。

    ```cpp
    class NumArray {
    private:
        vector<int> segmentTree;
        int n;

        void build(int node, int s, int e, vector<int> &nums) {
            if (s == e) {
                segmentTree[node] = nums[s];
                return;
            }
            int m = s + (e - s) / 2;
            build(node * 2 + 1, s, m, nums);
            build(node * 2 + 2, m + 1, e, nums);
            segmentTree[node] = segmentTree[node * 2 + 1] + segmentTree[node * 2 + 2];
        }

        void change(int index, int val, int node, int s, int e) {
            if (s == e) {
                segmentTree[node] = val;
                return;
            }
            int m = s + (e - s) / 2;
            if (index <= m) {
                change(index, val, node * 2 + 1, s, m);
            } else {
                change(index, val, node * 2 + 2, m + 1, e);
            }
            segmentTree[node] = segmentTree[node * 2 + 1] + segmentTree[node * 2 + 2];
        }

        int range(int left, int right, int node, int s, int e) {
            if (left == s && right == e) {
                return segmentTree[node];
            }
            int m = s + (e - s) / 2;
            if (right <= m) {
                return range(left, right, node * 2 + 1, s, m);
            } else if (left > m) {
                return range(left, right, node * 2 + 2, m + 1, e);
            } else {
                return range(left, m, node * 2 + 1, s, m) + range(m + 1, right, node * 2 + 2, m + 1, e);
            }
        }

    public:
        NumArray(vector<int>& nums) : n(nums.size()), segmentTree(nums.size() * 4) {
            build(0, 0, n - 1, nums);
        }

        void update(int index, int val) {
            change(index, val, 0, 0, n - 1);
        }

        int sumRange(int left, int right) {
            return range(left, right, 0, 0, n - 1);
        }
    };
    ```

    官方解，有时间了看看。

1. 树状数组 官方答案，有时间了看看

    ```cpp
    class NumArray {
    private:
        vector<int> tree;
        vector<int> &nums;

        int lowBit(int x) {
            return x & -x;
        }

        void add(int index, int val) {
            while (index < tree.size()) {
                tree[index] += val;
                index += lowBit(index);
            }
        }

        int prefixSum(int index) {
            int sum = 0;
            while (index > 0) {
                sum += tree[index];
                index -= lowBit(index);
            }
            return sum;
        }

    public:
        NumArray(vector<int>& nums) : tree(nums.size() + 1), nums(nums) {
            for (int i = 0; i < nums.size(); i++) {
                add(i + 1, nums[i]);
            }
        }

        void update(int index, int val) {
            add(index + 1, val - nums[index]);
            nums[index] = val;
        }

        int sumRange(int left, int right) {
            return prefixSum(right + 1) - prefixSum(left);
        }
    };
    ```

### 删除有序数组中的重复项 II

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

 

说明：

为什么返回数值是整数，但输出的答案是数组呢？

请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
 

示例 1：

输入：nums = [1,1,1,2,2,3]
输出：5, nums = [1,1,2,2,3]
解释：函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。 不需要考虑数组中超出新长度后面的元素。
示例 2：

输入：nums = [0,0,1,1,1,1,2,3,3]
输出：7, nums = [0,0,1,1,2,3,3]
解释：函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。 不需要考虑数组中超出新长度后面的元素。
 

提示：

1 <= nums.length <= 3 * 104
-104 <= nums[i] <= 104
nums 已按升序排列

代码：

1. 双指针，自己写的，还行

    ```cpp
    class Solution {
    public:
        int removeDuplicates(vector<int>& nums) {
            int left = 0, right = 0;
            int n = nums.size();
            if (n < 3) return n;
            while (right < n - 2)
            {
                if (nums[right] == nums[right+1] && nums[right+2] == nums[right])
                {
                    ++right;
                    continue;
                }
                nums[left] = nums[right];
                ++left;
                ++right;
            }
            nums[left++] = nums[right++];
            nums[left++] = nums[right++];
            return left;
        }
    };
    ```

1. 双指针，官方解

    ```cpp
    class Solution {
    public:
        int removeDuplicates(vector<int>& nums) {
            int n = nums.size();
            if (n <= 2) {
                return n;
            }
            int slow = 2, fast = 2;
            while (fast < n) {
                if (nums[slow - 2] != nums[fast]) {
                    nums[slow] = nums[fast];
                    ++slow;
                }
                ++fast;
            }
            return slow;
        }
    };
    ```

### 最接近的三数之和

给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。

返回这三个数的和。

假定每组输入只存在恰好一个解。

 

示例 1：

输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
示例 2：

输入：nums = [0,0,0], target = 1
输出：0
 

提示：

3 <= nums.length <= 1000
-1000 <= nums[i] <= 1000
-104 <= target <= 104

代码：

1. 排序加双指针

    ```cpp
    class Solution {
    public:
        int threeSumClosest(vector<int>& nums, int target) {
            sort(nums.begin(), nums.end());
            int n = nums.size();
            int left, right;
            int sum;
            int diff = INT32_MAX;
            int ans;
            for (int i = 0; i < n - 2; ++i)
            {
                left = i + 1;
                right = n - 1;;
                sum = nums[left] + nums[right] + nums[i];
                if (abs(target - sum) < diff)
                {
                    ans = sum;
                    diff = abs(target - sum);
                }
                while (left < right)
                {
                    sum = nums[left] + nums[right] + nums[i];
                    if (sum < target)
                        ++left;
                    else if (sum > target)
                        --right;
                    else
                        return sum;
                }
                if (abs(target - sum) < diff)
                {
                    ans = sum;
                    diff = abs(target - sum);
                }
            }
            return ans;
        }
    };
    ```

    双指针其实是遍历了所有情况，而不是到了某个位置就停止。

    另外，别人的解法都是在`while`中，每移动一步指针就尝试更新一下答案。我的这个解法只在双指针停止后才更新答案，却仍然通过了所有的测试用例，为什么？

1. 这道题用回溯法该怎么做呢？

### 算术三元组的数目

给你一个下标从 0 开始、严格递增 的整数数组 nums 和一个正整数 diff 。如果满足下述全部条件，则三元组 (i, j, k) 就是一个 算术三元组 ：

i < j < k ，
nums[j] - nums[i] == diff 且
nums[k] - nums[j] == diff
返回不同 算术三元组 的数目。

 
```
示例 1：

输入：nums = [0,1,4,6,7,10], diff = 3
输出：2
解释：
(1, 2, 4) 是算术三元组：7 - 4 == 3 且 4 - 1 == 3 。
(2, 4, 5) 是算术三元组：10 - 7 == 3 且 7 - 4 == 3 。
示例 2：

输入：nums = [4,5,6,7,8,9], diff = 2
输出：2
解释：
(0, 2, 4) 是算术三元组：8 - 6 == 2 且 6 - 4 == 2 。
(1, 3, 5) 是算术三元组：9 - 7 == 2 且 7 - 5 == 2 。
 

提示：

3 <= nums.length <= 200
0 <= nums[i] <= 200
1 <= diff <= 50
nums 严格 递增
```

代码：

1. 三重循环（竟然没超时，而且击败了 78% 的用户，很奇怪）

    ```cpp
    class Solution {
    public:
        int arithmeticTriplets(vector<int>& nums, int diff) {
            int n = nums.size();
            int ans = 0;
            for (int i = 0; i < n; ++i)
            {
                for (int j = i + 1; j < n; ++j)
                {
                    if (nums[j] - nums[i] != diff) continue;
                    for (int k = j + 1; k < n; ++k)
                    {
                        if (nums[k] - nums[j] == diff)
                            ++ans;
                    }
                }
            }
            return ans;
        }
    };
    ```

1. 哈希表

    类似两数之和的思想。

    ```cpp
    class Solution {
    public:
        int arithmeticTriplets(vector<int>& nums, int diff) {
            unordered_set<int> s;
            int n = nums.size();
            int ans = 0;
            for (int i = 0; i < n; ++i)
            {
                if (s.find(nums[i] - diff) != s.end() &&
                    s.find(nums[i] - diff - diff) != s.end())
                    ++ans;
                s.insert(nums[i]);
            }
            return ans;
        }
    };
    ```

1. 三指针

    由于数组单调递增，所以在方法 1 的三重循环里面，对于每个`i`，解都是唯一的。当`i`增加的时候，`j`和`k`也一定都是增加的。也就是说`j`不需要再搜索`[i+1,...,j]`这段区间的内容了，`k`也不需要再搜索`[j+1,...,k]`这段区间的内容了。

    ```cpp
    class Solution {
    public:
        int arithmeticTriplets(vector<int>& nums, int diff) {
            int n = nums.size();
            int ans = 0;
            int i = 0, j = 1, k = 2;
            while (i < n - 2 && j < n - 1 && k < n)
            {
                if (i >= j) j = i + 1;
                while (j < n - 1 && nums[j] - nums[i] < diff) ++j;  // 注意，这里是 < diff，而不是 != diff
                if (j >= k) k = j + 1;
                while (k < n && nums[k] - nums[j] < diff) ++k;
                if (j < n - 1 && k < n &&  // 由于前面的 while 退出条件是多个，所以这里需要再验证一遍
                    nums[j] - nums[i] == diff &&
                    nums[k] - nums[j] == diff) ++ans;
                ++i;
            }
            return ans;
        }
    };
    ```

1. 如果我们搜索到了一组`(i, j, k)`，显然`(j, k)`已经满足`diff`的条件了，我们能否把这个 pair 存起来，当`i`遍历到`j`时，直接把这个 pair 取出来继续向后搜索？（有点像 KMP 算法）
