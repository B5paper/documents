# Algorithms

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
