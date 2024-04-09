# C++ Notes QA

[unit]
[idx]
0
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
[idx]
1
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

[unit]
[idx]
2
[u_0]
使用 fstream 按行读取文件，将所有内容保存到 string lines 中
[u_1]
```cpp
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

string read_file(string file_path)
{
    string file_content;
    ifstream ifs(file_path);  // ifstream，默认打开文件的模式是 ios::read，因此不需要再指定
    if (!ifs.is_open())
    {
        cout << "fail to open file: " << file_path << endl;
    }
    string line;
    while (ifs.good())  // 如果遇到文件末尾，ifs.good() 会返回 false
    {
        getline(ifs, line);  // 每次读取一行，line 的末尾不包含 \n
        file_content.append(line);
        file_content.push_back('\n');  // 手动添加 \n
    }
    return file_content;
}

int main()
{
    string content = read_file("./hello.txt");
    cout << content << endl;
    return 0;
}
```

[unit]
[u_0]
解释 move 语义。
[u_1]
`std::move()`可以将一个左值或左值引用 remove reference 后，转换成右值引用。
右值引用表示一个新的类型，被以右值引用为形参的函数处理，比如 move 构造函数，move 赋值函数。
通常这些函数会改变对象的 allocator，并使原对象失效。

[unit]
[u_0]
`nth_element()`的作用是什么，对于数组`[3, 4, 5, 2, 1]`和`nth = 3`，给出 example code。
[u_1]
`nth_element()`可以保证第`nth`的数据一定在按序排好的正确的位置上，并且保证`nth`之前的数据一定小于等于`nth`之后的数据。

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;

int main()
{
    vector<int> arr{3, 4, 5, 2, 1};
    nth_element(arr.begin(), arr.begin() + 2, arr.end());
    for (int num: arr)
        printf("%d, ", num);
    putchar('\n');
    return 0;
}
```

输出：

```
2, 1, 3, 4, 5,
```

