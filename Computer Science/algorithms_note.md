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
        for (int j = 0; j < n - 1 - i; ++i)
        {
            if (nums[j] < nums[j+1])
                swap(nums[j], nums[j+1]);
        }
    }
}
```

1. 外层循环确定了执行的轮数，一共需要执行`n-1`轮。每执行一轮，都能保证当前索引区间`[0, n-1-i]`中最后一个元素是最大的，也就是说整个数组最后的`i+1`个元素是递增顺序。第一个元素就不需要再排了，所以一共需要`n-1`轮。

1. 内层循环确定了比较两个数字时，前一个数字的索引的位置，位置的范围为`[0, n-2-i]`，即每轮需要比较`n-1-i`次。因为需要比较当前元素和下一个元素，所以`j`不能取到当前区间末尾的元素。

### select sort

```c++
void select_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 0; i < n - i; ++i)
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

## 离散与连续

很多问题如果不考虑边界情况，把所有量作为连续量来处理，那么问题会简单很多。如果考虑边界值，情况则会变得复杂。如何处理离散值的边界问题？如何才能保证使用连续量进行思考，使用离散值来完善算法细节？

## 循环与控制

我们在编写一个循环时，是直接使用`for`，`while`，`if...else...`，`break/continue`编写，还是先写一些特例，找到共同规律后再简写成一个循环？

## 尝试与状态改变

有时候我们会遇到这样的情况：先尝试一些运算看看行不行，如果不行的话就不改变状态。如果没问题，那么实际执行运算。

这种问题有两种处理思路，一种是对逻辑进行模拟，然后计算，如果不满足条件就不进入下一个循环或下一层递归；一种是使用递归，直接实际改变状态，然后在下一个函数的入口处检测是否满足递归的退出条件。

这两种方法哪种更好一点呢？

