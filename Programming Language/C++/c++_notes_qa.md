# C++ Notes QA

[unit]
[u_0]
使用 STL 对一个`vector<int> vec`数组进行乱序排列。数据为：`[1, 2, 3, 4, 5]`
[u_1]
```cpp
#include <algorithm>
#include <random>
using namespace std;

int main()
{
    vector<int> vec{1, 2, 3, 4, 5};
    shuffle(vec.begin(), vec.end(), mt19937{random_device{}()});
    return 0;
}
```
