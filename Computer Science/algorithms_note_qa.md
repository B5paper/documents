# Algorithms Note QA

[unit]
[u_0]
已经图有`n`个节点，其邻接矩阵如下，请使用 dfs 对图进行遍历，排序后输出每个节点的值。
```
0, 1, 0, 0, 0
0, 0, 1, 0, 0
1, 0, 0, 0, 0
0, 0, 0, 1, 0
0, 0, 0, 0, 1
```
[u_1]
(empty)

[unit]
[u_0]
请找出图中的所有“团”。
[u_1]
(empty)

[unit]
[u_0]
写一个函数`revstr()`，反转字符串`string str = "hello, world"`。
[u_1]
```cpp
void revstr(string &str)
{
    int i = 0, j = str.size() - 1;
    while (i < j)
    {
        swap(str[i], str[j]);
        ++i;
        --j;
    }
}

int main()
{
    string str = "hello, world";
    cout << str << endl;
    revstr(str);
    cout << str << endl;
    return 0;
}
```

[unit]
[u_0]
123
[u_1]
456
