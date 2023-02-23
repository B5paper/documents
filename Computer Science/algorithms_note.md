# Algorithms

致力于线性直觉思考 + 非线性联想的分析方式，拒绝记忆模板化的答案。

分析代码中的任何细节，对常见的写法进行不常见的提问：为什么这样是对的？为什么不可以那样？对的只有这一种写法吗？



## Basic

### 循环的执行次数

1. `for`循环的次数

    对数组的索引进行遍历：

    如果索引从`0`开始，条件是`< n`，比如`for (int i = 0; i < n; ++i)`，那么一共需要执行`n`次。

    如果索引从`n-1`开始，条件是大于`> -1`，比如`for (int i = n-1; i > -1; --i)`，那么也是需要执行`n`次。

    索引模型（一定要两端都能取到）：

    ```
    [0, ..., n-1]

    [i, ..., j]  长度：j - i + 1
    ```

## Sort

### Bubble sort

```c++
void bubble_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - 1 - i; ++j)
        {
            if (nums[j] < nums[j+1])
                swap(nums[j], nums[j+1]);
        }
    }
}
```

1. 外层循环确定了执行的轮数，一共需要执行`n-1`轮。每执行一轮，都能保证当前索引区间`[0, n-1-i]`中最后一个元素是最大的，也就是说整个数组最后的`i+1`个元素是递增顺序。第一个元素就不需要再排了，所以一共需要`n-1`轮。

1. 内层循环的变量`j`确定了比较两个数字时，前一个数字的索引的位置，位置的范围为`[0, n-2-i]`，即每轮需要比较`n-1-i`次。因为需要比较当前元素和下一个元素，所以`j`不能取到当前区间末尾的元素。

### Select sort

```c++
void select_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 0; i < n - 1; ++i)
    {
        int min_idx = i;
        for (int j = i + 1; j < n; ++j)
        {
            if (nums[j] < nums[min_idx])
                min_idx = j;
        }
        swap[nums[i], nums[min_idx]];
    }
}
```

1. 外层循环确定了执行的轮数，一共需要`n-1`轮。每轮结束后，都能保证当前索引区间`[i, n-1]`中第一个元素是最小的，即整个数组的前`i+1`个元素是递增顺序。最后一个元素不需要再排了，所以一共需要`n-1`轮。

1. 内层循环确定了比较的次数，一共需要比较`n-1-i`次。

### Insertion sort

插入排序，灵感来源于我们对扑克牌进行排序。假设现在手里有了一些已经排好序的牌，现在来了一张新牌，我们直接的想法是把比这张牌大的牌都往后移动一个位置，然后把这张牌插入进去。

对于算法，假设目前区间是这样的：`[a ... b, c, d ... e]`，其中`[a ... b]`已经排好序，`c`是要处理的下一个元素，`[d ... e]`是无序的。一个简单的想法是让`b`和`c`进行比较，如果`b`大于`c`，那么交换这两个数的位置。然后再让`c`和前一个元素比较，如果`c`小于前一个元素，那么就交换位置。直到最后`c`和`a`进行比较，并判断是否交换位置。

代码实现如下：

```c++
void insertion_sort(vector<int> &nums)
{
    for (int i = 1; i < nums.size(); ++i)  // 第一个元素组成的区间不需要维护，因此 i 从 1 开始
    {
        for (int j = i; j > 0; --j)  // 从后往前处理，因为后面要用到 j-1，所以 j 最小取到 1
        {
            if (nums[j] < nums[j-1])  // 比较并交换 j 和 j-1 对应的两个元素
                swap(nums[j], nums[j-1]);
            else
                break;  // 维护有序区间结束，退出循环
        }
    }
}
```

上面的代码中，`i`表示的不是处理轮数，而是要 insert 的 number 的索引。`j`表示的是两个相邻 number 的索引。

事实上，`swap()`表示的是三个操作：一次存储临时值，两次赋值。如果我们事先把新来的数字存储起来，然后只是向右移动序列，等合适的位置空出来后，再把新来的数字放进去，就可以避免一直`swap()`了。代码如下：

```c++
void insertion_sort(vector<int> &nums)
{
    for (int i = 1; i < nums.size(); ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1 && nums[j] > num)
        {
            nums[j+1] = nums[j];
            --j;
        }
        nums[j+1] = num;
    }
}
```

### Merge sort

merge sort 的灵感灵源是拿两副已经排好序的扑克牌，每次只比较最上面的一张的大小，每次都只把较小的一张放下来。这样当手里的两副牌都比较完后，我们得到的第三副牌就是已经排好序的。

我们首先要把牌分成两摞，依次分下去，直到每摞都只有 0 张或 1 张。事实上，这个过程是展开了一棵二叉树。

然后对于合并的过程，我们只需要创建一个临时数组，把两摞按大小合并进去就可以了，最后再把这个临时数组复制到原数组上。

```c++
void merge(vector<int> &nums, int l, int r)
{
    vector<int> temp(r - l + 1);
    int m = l + (r - l) / 2;
    int i = l, j = m + 1;
    int k = 0;
    while (i <= m && j <= r)
    {
        if (nums[i] < nums[j])
            temp[k++] = nums[i++];
        else
            temp[k++] = nums[j++];
    }
    while (i <= m)
        temp[k++] = nums[i++];
    while (j <= r)
        temp[k++] = nums[j++];
    k = 0;
    i = l;
    while (i <= r)
        nums[i++] = temp[k++];
}

void merge_sort_helper(vector<int> &nums, int l, int r)
{
    if (l >= r)
        return;
    int m = l + (r - l) / 2;
    merge_sort_helper(nums, l, m);  // 这个看起来像是一棵二叉树的先序遍历。既然先序遍历可以写，那么其他类型的遍历可以写吗？
    merge_sort_helper(nums, m+1, r);
    merge(nums, l, r);  // merge 的过程，可以把两个子节点看作两副牌，然后父节点看作临时数组
}

void merge_sort(vector<int> &nums)
{
    merge_sort_helper(nums, 0, nums.size() - 1);
}
```

问题：

`merge_sort_helper()`函数的写法是如何推导出来的？这个函数有没有其他的写法？

### Quick sort

快速排序有一点点像归并排序的前半段，有点像冒泡排序。灵感可能来自于分治。

基本的想法是先设置一个分界元素，再对两侧进行排序。然后再设置下一个分界元素，再对两侧进行排序：

```cpp

```

#### 有关边界情况的问题

如何解决这些代码里的边界问题呢？

```cpp
int quicksort_helper(vector<int> &nums, int l, int r, int idx_pivot)
{
    int val = nums[idx_pivot];
    swap(nums[idx_pivot], nums[r]);
    int i = l, j = r-1;
    while (i < j)  // 假如对于数组 [3, 4]，i = 0, j = 0，此时不会进入循环
    {
        while (i < j && nums[i] < val) ++i;
        while (i < j && nums[j] >= val) --j;
        swap(nums[i], nums[j]);
    }
    swap(nums[i], nums[r]);  // 这里 r = 1, i = 0，会把 [3, 4] 交换成 [4, 3]
    return i;
}

void partition(vector<int> &nums, int l, int r)  // 先处理当前节点，再处理左节点，再处理右节点
{
    if (r - l + 1 <= 1)
    {
        cout << "partition: [" << l << ", " << r << "], not enough length" << endl;
        return;
    }
    int idx_pivot = l + (r - l) / 2;
    cout << "partition: [" << l << ", " << r <<
     "], idx: " << idx_pivot << ", x: " << nums[idx_pivot] << ", ";
    int delim = quicksort_helper(nums, l, r, idx_pivot);
    
    cout << "delim: " << delim << ", " << "nums: ";
    for (auto &num: nums)
    {
        cout << num << ", ";
    }
    cout << endl;
    partition(nums, l, delim);
    partition(nums, delim + 1, r);
}

void quicksort(vector<int> &nums)
{
    partition(nums, 0, nums.size() - 1);
}

int main()
{
    vector<int> nums({5, 3, 1, 2, 4});
    quicksort(nums);
    for (auto &num: nums)
        cout << num << ", ";
    cout << endl;
    return 0;
}
```

## Binary search

给定一个`n`个元素有序的（升序）整型数组`nums`和一个目标值`target`，写一个函数搜索`nums`中的`target`，如果目标值存在返回下标，否则返回`-1`。

分析：无论数组中的元素是正数，负数，还是零，只要数组**有序**，所以可以用二分查找。二分查找可以判断某个值是否在数组中，找到这个值的某个索引，或者找到这个值的左边界或右边界。

### exact element

代码：

```c++
int binary_search(vector<int> &nums, int target) {
    int left = 0, right = nums.size() - 1, mid;  // 这里 right 取到最后一个元素，而不是 nums.size()，因为下面的 left <= right 也取到等号了。假如 right 在这里取到 nums.size()，那么假如在 [2, 3, 4] 中找 5，left 就会取到 3，mid 也会取到 3，从而数组越界（即，如果搜索一个大于最大值的数，会导致 left 和 mid 越界）
    while (left <= right) {  // 因为下面在更新 right 的时候用的是 mid - 1，所以这里的等号必须带上。否则在找 [2, 3, 4, 5] 中的 5 时，最后两掓是 left = 2, right = 3，mid = 2，然后 left = mid + 1 = 3，因为 left < right，就会跳出循环，从而 mid 无法取到 3，最终无法找到 5
        mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;  // 为什么这里是 left = mid + 1，而不是 left = mid
        else if (nums[mid] > target) right = mid - 1;  // 为什么这里是 right = mid - 1，而不是 right = mid
        else return mid;
    }
    return -1;
}
```

二分查找的要求：

1. `mid`能够取到所有元素的下标，尤其是边界处的。
1. 尽量不要找重复的元素（比如某个区间边界处）


### left bound

### right bound

## Math

### 最大公约数

```java
static int gcd(int p, int q) {
        if (q == 0) return p;
        int r = p % q;
        return gcd(q, r);
    }
```

## Array-processing

* 找最大值

    find the maximum of the array values

    ```java
    static int findMax(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length - 1; ++i)
            if (arr[i] > max) max = arr[i];
        return max;
    }
    ```

* 计算均值

    compute the average of the array values

    ```java
    static double calcMean(double[] arr) {
        double n = arr.length;
        double sum = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            sum += arr[i];
        }
        return sum / n;
    }
    ```

* 复制另一个数组

    copy to another array

    ```java
    int N = a.length;
    double[] b = new double[N];
    for (int i = 0; i < N; i++)
        b[i] = a[i]; 
    ```

* 反转数组

    reverse the elements within an array

    ```java
    static void reverseArray(int[] arr) {
        int left = 0, right = arr.length - 1;
        int n = arr.length;
        n /= 2;
        int temp;
        for (int i = 0; i < n; ++i) {
            temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            ++left;
            --right;
        }
    }
    ```

    ```java
    int N = a.length;
    for (int i = 0; i < N/2; i++)
    {
        double temp = a[i];
        a[i] = a[N-1-i];
        a[N-i-1] = temp;
    }
    ```

* 矩阵乘法

    ```java
    static double[][] matMultiply(double[][] A, double[][] B) {
        int m = A.length, l = A[0].length, n = B[0].length;
        double[][] C = new double[m][n];
        for (int k = 0; k < l; ++k) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
    ```

## 树，递归，与数据保持

计算机的很多算法都是树的变形。树与递归也经常有联系。这两者中间有个重要的问题，即我们如何记录数据？

常见的数据有 4 种：函数的参数，函数的返回值，函数中的变量，独立在函数之外的数据。

## 树的节点与递归

判断一棵树是否为空，可以在`if()`中处理，也可以交给递归处理。这两种有什么区别呢？

看一个例子：假如我们想对一棵树进行先序遍历，第一种方法为

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
    vector<int> ans;
    vector<int> preorderTraversal(TreeNode* root) {
        if (!root)
            return ans;
        ans.push_back(root->val);
        if (root->left)
            preorderTraversal(root->left);
        if (root->right)
            preorderTraversal(root->right);
        return ans;
    }
};
```

这种方式完全以`if`的形式判断当前节点有没有左右子节点。前面的`if (!root)`实际上只是判断外部的输入是否为空树，并不能起到判断空节点的作用。

我们也可以把它写成递归处理的方式：

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
    vector<int> ans;
    void pre(TreeNode *r)
    {
        if (!r)
            return;
        ans.push_back(r->val);
        pre(r->left);
        pre(r->right);
    }

    vector<int> preorderTraversal(TreeNode* root) {
        pre(root);
        return ans;
    }
};
```

这种情况把子节点是否存在作为递归的终止条件来判断了。

问题：什么情况下必须用第一种方式？什么情况下可以用第二种方式？

## 离散与连续

很多问题如果不考虑边界情况，把所有量作为连续量来处理，那么问题会简单很多。如果考虑边界值，情况则会变得复杂。如何处理离散值的边界问题？如何才能保证使用连续量进行思考，使用离散值来完善算法细节？

## 循环与控制

我们在编写一个循环时，是直接使用`for`，`while`，`if...else...`，`break/continue`编写，还是先写一些特例，找到共同规律后再简写成一个循环？

### insertion sort 中的循环

插入排序虽然说起来原理很简单，但是有一些细节没想清楚：

我们首先要把大于 key 值的牌往后挪，空出来一个位置，等位置空出来后，再把 key 插进去。

假设第 i 张牌是 key 牌，我们选择 key 牌前面的一张牌开始比较，即`j = i - 1`，最后一张要比较的牌为`j = 0`：

```cpp
for (int j = i - 1; j > -1; --j)
```

对于每一张牌，它如果比 key 牌大，那么需要后移：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j]  // 这句执行结束后，nums[j] 这个位置就空出来了
}
```

如果发现某张牌比 key 小，或等于 key，那么就可以结束循环了：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j]
    else
        break;
}
```

此时我们需要把空出来的位置插入 key 值，可以选择在循环内插入，也可以选择在循环外。假如在循环内：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j]
    else
    {
        nums[j+1] = num;  // 如果发现某张牌比 key 小，那么下一张牌一定是属于 key 的位置
        break;
    }
}
```

但是结果并不是正确的，因为当`j = 0`进入循环后，假如发现`j = 1`处的值比`num`大，会执行`nums[1] = nums[0]`。接着`j--`，变成`-1`，不再进入循环。可是这时`nums[0]`处的值已经空出来了，`elxe`语句也并未得到执行。由于我们选择在循环内部处理 key 值，所以需要再一次进入循环。那么就必须把`j > -1`改成`j > -2`。但是这样的话，`nums[j]`就会越界，所以我们需要对这种情况进行单独处理：

```cpp
for (int j = i - 1; j > -2; --j)
{
    if (j == -1)
    {
        nums[0] = num;
        break;
    }

    if (nums[j] > num)
        nums[j+1] = nums[j];
    else
    {
        nums[j+1] = num;
        break;
    }
}
```

这样就没问题了。

如果我们把处理写到外面：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j];
    else
        break;
}

nums[j+1] = num;
```

当`j = 0`时，发现`nums[0] > num`，也会移动，并使`j`递减得到`-1`。这样我们在外面使用`nums[j+1]`仍是有效的。可是问题在于，`j`是在`for`里面定义的，不是在`for`外面定义的。我们把`j`的定义也移到外面：

```cpp
for (int i = 1; i < n; ++i)
{
    int num = nums[i];
    int j = i - 1;
    for (; j > -1; --j)
    {
        if (nums[j] > num)
            nums[j+1] = nums[j];
        else
            break;
    }
    nums[j+1] = num;
}
```

这样是没问题的。可是这样的话，`for`就失去了它的功能，不如把改成`while`：

```cpp
void insertion_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 1; i < n; ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1)
        {
            if (nums[j] > num)
            {
                nums[j+1] = nums[j];
                --j;
            }
            else
                break;
        }
        nums[j+1] = num;
    }
}
```

`else`单独一行，有点丑，可以和上面的`if`换下位置：

```cpp
void insertion_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 1; i < n; ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1)
        {
            if (nums[j] <= num)
                break;
            nums[j+1] = nums[j];
            --j;

        }
        nums[j+1] = num;
    }
}
```

`if`单独一行，也有点繁琐，可以和`while`合一下：

```cpp
void insertion_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 1; i < n; ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1 && nums[j] > num)
        {
            nums[j+1] = nums[j];
            --j;
        }
        nums[j+1] = num;
    }
}
```

现在我们的问题来了：为什么在循环内部进行插入的代码需要单独处理边界情况，而在循环外部进行插入的代码不需要处理边界？为什么第二种情况的循环变量一定要写在循环外面？对于`for`和`while`，我们该如何才能快速选择正确的那个？有没有可能先`--j`，再执行其他语句？

## 尝试与状态改变

有时候我们会遇到这样的情况：先尝试一些运算看看行不行，如果不行的话就不改变状态。如果没问题，那么实际执行运算。

这种问题有两种处理思路，一种是对逻辑进行模拟，然后计算，如果不满足条件就不进入下一个循环或下一层递归；一种是使用递归，直接实际改变状态，然后在下一个函数的入口处检测是否满足递归的退出条件。

这两种方法哪种更好一点呢？

