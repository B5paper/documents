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

[unit]
[u_0]
在函数的形参中，是否有`const`修饰可以看作不同的类型吗？
[u_1]
函数的重载要求形参是不同的类型，所以可以使用函数是否可以重载判断有无`const`修饰的参数是否是不同的类型。

```cpp
void test(int a)
{
    cout << "without const" << endl;
}

void test(const int a)
{
    cout << "with const" << endl;
}

int main()
{
    int a = 3;
    test(a);
    return 0;
}
```

上面的代码会报错。但是换成引用的话就正常了：

```c++
void test(int &a)
{
    cout << "without const" << endl;
}

void test(const int &a)
{
    cout << "with const" << endl;
}

int main()
{
    int a = 3;
    test(a);
    return 0;
}
```

输出：

```
without const
```

因为函数的形参总是一个副本，无论有没有`const`都无所谓。而引用是关系到变量实际上是否被修改的，所以需要对`const`加以区分。
