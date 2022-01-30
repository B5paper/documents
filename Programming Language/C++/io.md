# 常见编程题目的输入处理

## 已知每行的输入个数，空格分隔，不需要判断输入是否终止

计算 a + b

输入描述：

> 输入包括两个正整数a,b(1 <= a, b <= 10^9),输入数据包括多组。

输出描述:

> 输出 a+b 的结果

示例：

输入：

```
1 5
10 20
```

输出：

```
6
30
```

代码：

```c++
#include<iostream>
using namespace std;
 
int main()
{
    int a, b;
    while (cin >> a >> b)
    {
        cout << a+b << endl;
    }
    return 0;
}
```

## 已知每行的输入个数，空格分隔，且已知有多少行

计算 a+b

输入描述:

> 输入第一行包括一个数据组数t(1 <= t <= 100)
> 接下来每行包括两个正整数a,b(1 <= a, b <= 10^9)

输出描述：

> 输出a+b的结果

Example:

输入：

```
2
1 5
10 20
```

输出：

```
6
30
```

代码：

```c++
#include <iostream>
using namespace std;

int main()
{
    int t;
    cin >> t;
    int a, b;
    for (int i = 0; i < t; ++i)
    {
        cin >> a >> b;
        cout << a + b << endl;
    }
    return 0;
}
```

## 每行有不定的输入

```
1 2 3
4 5
0 0 0 0 0
```

代码：

```c++
#include <iostream>
using namespace std;

int main()
{
    int temp, sum = 0;
    while (cin >> temp)
    {
        sum += temp;
        if (cin.peek() == '\n')
        {
            cout << sum << endl;
            sum = 0;
        } 
    }
    return 0;
}
```

## 每行用逗号分隔，不知有多少行

主要思路是先用`cin`读一行，把结果存到`stringstream`对象`sstr`里，然后再用`sstr`对存储的内容进行再次解析，如果想把解析的结果存到`string`对象里，那么唯一的方法就是调用`std::getline(sstr, str, ',')`。使用`sstr.getline()`之类的函数，只能把结果存到`char*`缓冲区里。

```c++
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
using namespace std;

int main()
{
    string str;
    vector<string> strs;
    stringstream sstr;
    while (cin >> str)  // 因为是从标准输入读入，所以永远不会结束，cin 永远有效
    {
        sstr.str(str);  // 也可以写成 sstr << str;
        while (getline(sstr, str, ','))  // 如果想把结果存到 string 对象里，那么只能用 std::getline()
        {
            strs.push_back(str);
        }

        // 也可以这样写：
        // while (sstr >> str)
        // {
        //     strs.push_back(str);
        //     if (sstr.peek() == ',')
        //         sstr.ignore();
        // }
        sort(strs.begin(), strs.end());
        for (int i = 0; i < strs.size()-1; ++i)
            cout << strs[i] << ",";
        cout << strs.back();
        cout << endl;
        strs.clear();
        sstr.clear();  // sstr 在解析完一行后，是无效状态，需要把它的状态重置，以解析下一行
    }

    return 0;
}
```