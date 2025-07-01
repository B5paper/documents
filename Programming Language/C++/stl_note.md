# STL Note

## cached

* `string_view`本质存储的是 begin pointer + length，并不是一个 null-terminated 的字符串，所以无法提供`.c_str()`。

    example:

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <iostream>
    using namespace std;

    int main()
    {
        string str {"hello, world"};
        string_view strv(str.c_str(), 5);
        cout << strv << endl;
        printf("strv: %s\n", strv.data());

        return 0;
    }
    ```

    output:

    ```
    hello
    strv: hello, world
    ```

    可以看到，`cout`能正常输出 string view 的内容，而`printf()`会寻找`\0`标记的字符串。

* `string_view`的`.compare()`方法与`strcmp()`用法一致；`.substr()`返回的仍是`string_view`。

    example:

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <iostream>
    using namespace std;

    int main()
    {
        string_view strv_1 {"hello"};
        string_view strv_2 {"hello"};
        int ret = strv_1.compare(strv_2);
        printf("ret: %d\n", ret);

        string_view strv {"hello, world"};
        string_view sub_strv = strv.substr(1, 5);
        cout << "strv: " << strv << endl;
        cout << "sub_strv: " << sub_strv << endl;

        return 0;
    }
    ```

    output:

    ```
    ret: 0
    strv: hello, world
    sub_strv: ello,
    ```

* c++20 中的`string_view`不再是 read only，具体增加了什么功能不太清楚。

* 如果`string_view`指向的对象被销毁，那么`string_view`的内容是未定义的：

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <iostream>
    using namespace std;

    string_view get_string_view()
    {
        string msg = "hello, world";
        return string_view(msg);
    }

    int main()
    {
        string_view strv = get_string_view();
        cout << strv << endl;
        return 0;
    }
    ```

    output:

    ```
    @��;�vH&�
    ```

* c++ stl 容器中的 erase

    如果是顺序容器，比如 vector，那么`erase()`只支持传入 iterator 进行删除。比如要删第 1 个元素，那么就调用`vec.erase(vec.begin());`，如果要删除第 3 个元素，那么就调用`vec.erase(vec.begin() + 2)`。
    
    vector 的`erase()`无法通过传入元素引用的方式进行删除，比如`vec.erase(vec.back());`会编译报错。

    如果是关联容器，比如`unordered_map`, `set`等，那么`erase()`既支持传入 iterator，也支持传入 key 的 const 左值引用：

    ```cpp
    #include <stdio.h>
    #include <vector>
    #include <string>
    #include <iostream>
    #include <unordered_map>
    #include <map>
    using namespace std;

    template<typename T>
    void print(T &container) {
        for (auto iter = container.begin(); iter != container.end(); ++iter) {
            cout << *iter << ", ";
        }
        cout << endl;
    }

    ostream& operator<<(ostream &out, const pair<string, int> &p) {
        printf("(%s, %d)", p.first.c_str(), p.second);
        return out;
    }

    int main(int argc, const char **argv) {
        unordered_map<string, int> vec {
            {"hello", 1},
            {"world", 2},
            {"nihao", 3},
            {"zaijian", 4}
        };
        vec.erase("hello");
        print(vec);

        multimap<string, int> m {
            {"hello", 1},
            {"world", 2},
            {"hello", 3},
        };
        m.erase("hello");
        print(m);
        
        return 0;
    }
    ```

    output:

    ```
    (zaijian, 4), (nihao, 3), (world, 2), 
    (world, 2), 
    ```

    但是关联容器没有`.back()`方法，因此调用`m.erase(m.back())`无从谈起。

    如果是多 key 值容器，那么所有与传入参数相等的 key 都会被删除。

* vector 中的元素删除

    如果使用索引的话，可以在删除当前元素后，手动将索引减一。

    example:

    ```cpp
    #include <stdio.h>
    #include <vector>
    using namespace std;

    int main()
    {
        vector<int> vec{1, 2, 3, 4, 4, 4, 3};

        for (int i = 0; i < vec.size(); ++i)
        {
            if (vec[i] == 4) {
                vec.erase(vec.begin() + i);
                --i;
            }
        }

        for (int i = 0; i < vec.size(); ++i)
        {
            printf("%d, ", vec[i]);
        }
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 3,
    ```

    如果使用迭代器，那么必须使用 while 循环：

    ```cpp
    #include <stdio.h>
    #include <vector>
    using namespace std;

    int main()
    {
        vector<int> vec{1, 2, 3, 4, 4, 4, 3};

        auto it = vec.begin();
        while (it != vec.end())
        {
            if (*it == 4)
            {
                it = vec.erase(it);
                continue;
            }
            ++it;
        }

        for (int i = 0; i < vec.size(); ++i)
        {
            printf("%d, ", vec[i]);
        }
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 3,
    ```

    如果使用了 erase，那么就不需要再`++it`，其他情况都需要`++it`。这种处理方式或许也适合其他的容器，比如`map`。（如果 for 里使用`--it`，可能也可以支持 map？）

    如果使用 for 循环，因为并不总是需要`++it`，所以会出错：

    ```cpp
    #include <stdio.h>
    #include <vector>
    using namespace std;

    int main()
    {
        vector<int> vec{1, 2, 3, 4, 4, 4, 3};

        for (auto it = vec.begin(); it != vec.end(); ++it)
        {
            if (*it == 4) {
                it = vec.erase(it);
            }
        }

        for (int i = 0; i < vec.size(); ++i)
        {
            printf("%d, ", vec[i]);
        }
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 4, 3,
    ```

* c++ accumulate usage

    `std::accumulate()`在头文件`<numeric>`中，不在`<algorithm>`，默认实现累加功能，用户也可以利用其实现累乘等自定义功能。

    example:

    ```cpp
    #include <numeric>
    #include <vector>
    #include <stdio.h>
    #include <functional>
    using namespace std;

    int prod_func(int a, int b)
    {
        return a * b;
    }

    int main()
    {
        int arr[5] = {1, 2, 3, 4, 5};
        int s = accumulate(arr, arr + 5, 0);
        printf("s = %d\n", s);

        vector<int> vec{1, 2, 3, 4, 5};
        s = accumulate(vec.begin(), vec.end(), 2);
        printf("s = %d\n", s);

        int p = accumulate(vec.begin(), vec.end(), 1, prod_func);
        printf("p = %d\n", p);

        auto custom_func_1 = [](int a, int b) {
            return a * b;
        };
        int res = accumulate(vec.begin(), vec.end(), 1, custom_func_1);
        printf("custom_func_1 res = %d\n", res);

        function<int(int, int)> custom_func_2 = [](int a, int b) {
            return a + b;
        };
        res = accumulate(vec.begin(), vec.end(), 1, custom_func_2);
        printf("custom_func_2 res = %d\n", res);

        return 0;
    }
    ```

    output:

    ```
    s = 15
    s = 17
    p = 120
    custom_func_1 res = 120
    custom_func_2 res = 16
    ```

    `accumulate()`第三个参数是个初始值，如果没有特别需求，填 0 就可以。

    `prod_func`使用普通函数作为自定义函数，`custom_func_1`使用 lambda 表达式，`custom_func_2`使用 function 对象作为自定义函数。

* string view

    ```cpp
    #include <stdio.h>
    #include <string>
    // #include <string_view>
    #include <iostream>
    using namespace std;

    int main()
    {
        const char *c_str = "c hello, world";
        string_view strv(c_str);
        printf("%s\n", strv.begin());
        cout << strv << endl;

        string cpp_str{"cpp hello, world"};
        string_view strv_2(cpp_str);
        printf("%s\n", strv_2.begin());
        printf("%s\n", strv_2.data());
        cout << strv_2 << endl;

        return 0;
    }
    ```

    output:

    ```
    c hello, world
    c hello, world
    cpp hello, world
    cpp hello, world
    cpp hello, world
    ```

    说明：

    1. `string_view()`没有接收`string`对象为参数的构造函数，不清楚`string_view strv_2(cpp_str);`是怎么通过编译并运行的。

        2025/06/28/00: `string_view strv_2(cpp_str);`这个函数，vscode 在括号时使用`ctrl + shift + space`无法调出参数提示，而在上面的`string_view strv(c_str);`可以调出，不清楚为什么。vscode 的 c++ 扩展中，c++ 标准使用的是 c++ 17。

    2. `string_view`没有`.c_str()`方法，只有`.data()`和`.begin()`迭代器。比较奇怪的是迭代器是个 iterator 对象，但是仍能使用`%s` print 出字符串内容。

* vector 扩容会导致引用失效

    ```cpp
    #include <stdio.h>
    #include <vector>
    using namespace std;

    struct Node
    {
        int val;
    };

    vector<Node> nodes;
    int invoke_cnt = 0;
    void recur_construct_chain(Node &parent)
    {
        invoke_cnt++;
        if (invoke_cnt > 5)
            return;
        nodes.emplace_back();
        Node &new_node = nodes.back();
        new_node.val = nodes.size() - 1;
        printf("new node %d, parent: %d\n", new_node.val, parent.val);
        recur_construct_chain(new_node);
    }

    int main()
    {
        printf("test 1:\n");
        nodes.reserve(3);
        nodes.push_back({0});
        recur_construct_chain(nodes[0]);
        putchar('\n');

        printf("test 2:\n");
        invoke_cnt = 0;
        nodes.clear();
        nodes.shrink_to_fit();
        nodes.reserve(10);
        nodes.push_back({0});
        recur_construct_chain(nodes[0]);

        return 0;
    }
    ```

    output:

    ```
    test 1:
    new node 1, parent: 0
    new node 2, parent: 1
    new node 3, parent: 277658299
    new node 4, parent: 3
    new node 5, parent: 4

    test 2:
    new node 1, parent: 0
    new node 2, parent: 1
    new node 3, parent: 2
    new node 4, parent: 3
    new node 5, parent: 4
    ```

    可以看到，在 test 1 中，我们 reseve 了 3 个元素，当新添加第 4 个元素（node 3）时，vector 发生了扩容，导致 parent 引用失效，输出了非法内存中的数值。

    在 test 2 中，我们 reseve 了 10 个元素，大于需要的 6 个元素，vector 不发生扩容，因此 parent 都是正常的，没有失效。

    在这两个例子中我们可以看出，即使使用引用拿到 vector 中的元素，当 vector 发生扩容或缩容时，也会发生引用失效的情况。

    说明：

    1. 调用完`nodes.clear()`后，必须调用`nodes.shrink_to_fit();`才能让 vector 的实际内存占用缩小到 0。

        调用`nodes.reserve(0);`无法实现这个效果。

    如果我们在 vector 中存储指针，便可以解决这个问题：

    ```cpp
    #include <stdio.h>
    #include <vector>
    using namespace std;

    struct Node
    {
        int val;
    };

    vector<Node*> nodes;
    int invoke_cnt = 0;
    void recur_construct_chain(Node &parent)
    {
        invoke_cnt++;
        if (invoke_cnt > 5)
            return;
        nodes.push_back(new Node);
        Node &new_node = *nodes.back();
        new_node.val = nodes.size() - 1;
        printf("new node %d, parent: %d\n", new_node.val, parent.val);
        recur_construct_chain(new_node);
    }

    int main()
    {
        nodes.reserve(3);
        nodes.push_back(new Node{0});
        recur_construct_chain(*nodes[0]);
        for (Node *node_ptr : nodes)
            delete node_ptr;
        putchar('\n');
        return 0;
    }
    ```

    output:

    ```
    new node 1, parent: 0
    new node 2, parent: 1
    new node 3, parent: 2
    new node 4, parent: 3
    new node 5, parent: 4
    ```

* 使用`partial_sum()`计算前缀和

    example:

    ```cpp
    #include <numeric>
    #include <vector>
    #include <stdio.h>
    using namespace std;

    int main()
    {
        int arr[] = {1, 2, 3, 4, 5};
        int res[5] = {0};
        partial_sum(arr, arr+5, res);
        for (int i = 0; i < 5; ++i)
            printf("%d, ", res[i]);
        putchar('\n');

        vector<int> vec{5, 4, 3, 2, 1};
        vector<int> presum(5, 0);
        partial_sum(vec.begin(), vec.end(), presum.begin());
        for (int i = 0; i < 5; ++i)
            printf("%d, ", presum[i]);
        putchar('\n');
        
        return 0;
    }
    ```

    output:

    ```
    1, 3, 6, 10, 15,
    5, 9, 12, 14, 15,
    ```

    `partial_sum()`在`<numeric>`头文件中。如果将迭代器的`begin()`和`end()`作为参数填进去，那么得到的就是标准的前缀和。

* 使用自定义处理函数的`partial_sum()`

    example:

    ```cpp
    #include <numeric>
    #include <functional>
    #include <vector>
    #include <stdio.h>
    using namespace std;

    int custom_sum(int a, int b)
    {
        return a + b + 1;
    }

    int main()
    {
        int arr[] = {1, 2, 3, 4, 5};
        int res[5] = {0};
        partial_sum(arr, arr+5, res, custom_sum);
        for (int i = 0; i < 5; ++i)
            printf("%d, ", res[i]);
        putchar('\n');

        partial_sum(arr, arr+5, res, multiplies<int>());
        for (int i = 0; i < 5; ++i)
            printf("%d, ", res[i]);
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    1, 4, 8, 13, 19,
    1, 2, 6, 24, 120,
    ```

    其中，`multiplies`来自头文件`<functional>`。

* c++ 中 stringstream 处理多次转换

    example:

    ```cpp
    #include <sstream>
    #include <string>
    #include <stdio.h>
    #include <string.h>
    using namespace std;

    int main()
    {
        const char *int_str = "42";
        const char *flt_str = "3.1415926535";
        int int_val;
        float flt_val;

        // test 1
        stringstream ss;
        ss << int_str;
        ss >> int_val;
        printf("int val: %d\n", int_val);

        ss.clear();
        ss << flt_str;
        ss >> flt_val;
        printf("flt val: %f\n", flt_val);

        ss.clear();
        ss << 123;
        printf("str: %s\n", ss.str().c_str());

        // test 2
        printf("ss: %s\n", ss.str().c_str());
        ss.str("empty");
        printf("ss: %s\n", ss.str().c_str());

        // test 3
        stringstream ss_2;
        size_t cur_size = 0;
        char *buf_1m = (char*) malloc(1024 * 1024);  // 1 MB
        memset(buf_1m, ' ', 1024 * 1024);
        buf_1m[1024 * 1024 - 1] = '\0';
        while (cur_size < 512 * 1024 * 1024) {  // 512 MB
            ss_2 << buf_1m;
            if (ss_2.str().size() > cur_size) {
                cur_size = ss_2.str().size();
            } else {
                break;
            }
        }
        printf("cur size: %llu MB\n", cur_size / 1024 / 1024);

        return 0;
    }
    ```

    output:

    ```
    int val: 42
    flt val: 3.141593
    str: 423.1415926535123
    ss: 423.1415926535123
    ss: empty
    cur size: 512 MB
    ```

    其中 test 1 展示了先处理 int 型的字符串，再处理 float 型的字符串，在转换完 int 后，需要调用`ss.clear()`方法重置状态。

    test 2 的输出为`423.1415926535123`，说明调用`.clear()`后 stringstream 内部缓存的字符串并没有被真正清空，必须使用`.str("")`显式赋值，才能清空 stringstream 缓冲区的内容。

    test 3 探索的 stringstream 的缓冲区最大能有多大，是否有到了最大限制后自动清空缓冲的操作。目前测试下来，到了 512 MB 仍然能正常运行，猜测 buffer 大小可能是无限大的。这警示我们在使用同一个 stringstream 对象多次转换时，不仅要调用`c.lear()`，还需要关注 buffer 的大小，不然就可能无限制地增大，占满内存。

* stringstream 可以解析以空格`' '`, tab `'\t'`, 换行`'\n'`作为分隔符的字符串

    example:

    ```cpp
    #include <sstream>
    #include <string>
    #include <stdio.h>
    #include <string.h>
    using namespace std;

    int main()
    {
        // test 1
        int int_val;
        stringstream ss;
        ss << "3.14";
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        putchar('\n');

        // test 2
        ss.clear();
        ss.str("3,14");
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        putchar('\n');

        // test 3
        ss.clear();
        ss.str("3 14");
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        putchar('\n');

        // test 4
        ss.clear();
        ss.str("3\t14\t15");
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        putchar('\n');

        // test 5
        ss.clear();
        ss.str("3\n14");
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        putchar('\n');

        // test 6
        ss.clear();
        ss.str("3  14\t\t15\n\n\n926");
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        ss >> int_val;
        printf("int val: %d\n", int_val);
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    int val: 3
    int val: 0
    int val: 0

    int val: 3
    int val: 0

    int val: 3
    int val: 14
    int val: 14

    int val: 3
    int val: 14
    int val: 15

    int val: 3
    int val: 14

    int val: 3
    int val: 14
    int val: 15
    int val: 926

    ```

    test 1 尝试将一个小数解析为整数，stringstream 只解析了整数部分，后续的解析都直接输出 0，看来是有解析错误。

    test 2 尝试使用逗号作分隔符，解析整数，可以看到逗号是无法识别的。

    test 3 使用空格作为分隔符，可以正常连续解析，当整数数量不够解析时，stringstream 总是输出最后一个正确解析的整数。

    test 4 用于测试`'\t'`，可以正常解析。

    test 5 用于测试`'\n'`，可以正常解析。

    test 6 用于测试多个空格，多个制表符，多个换行符分隔的整数，可以正常解析。

* `stringstream`同时具有`ostringstream`和`istringstream`的作用

    example:

    ```cpp
    #include <sstream>
    #include <string>
    #include "stdio.h"
    using namespace std;

    int main()
    {
        int int_val = 42;
        float flt_val = 3.1415926;
        const char *msg = "hello, world";

        // test 1
        stringstream ss;
        ss << "int val: " << int_val << ", float val: " << flt_val << ", msg: " << msg << endl;
        string str = ss.str();
        printf("%s\n", str.c_str());

        // test 2
        stringstream ss_2;
        ss_2 << "1234";
        int val_2;
        ss_2 >> val_2;
        printf("%d\n", val_2);

        // test 3
        istringstream iss("3.14");
        float flt_val_3;
        iss >> flt_val_3;
        printf("%f\n", flt_val_3);

        return 0;
    }
    ```

    output:

    ```
    int val: 42, float val: 3.14159, msg: hello, world

    1234
    3.140000
    ```

    test 1 展示了`stringstream`作为`ostringstream`使用时的功能。test 2 进行了一个字符串到数值的转换，展示了`stringstream`既作为`ostringstream`，又作为`istringstream`的功能。
    
    test 3 展示了`istringstream`的用法，它可以把字符串转换成数值类型（其它类型也可以），可以看到`iss`只能在初始化时被赋值，初始化后不能再有类似`iss << "123";`这样的语句。

* c++ 中`sstream`的使用

    如果想把`cout`输出到一个字符串里而不是控制台，那么需要使用`sstream`。

    example:

    ```c++
    #include <sstream>
    #include <string>
    // #include <iostream>
    #include "stdio.h"
    using namespace std;

    int main()
    {
        int int_val = 42;
        float flt_val = 3.1415926;
        const char *msg = "hello, world";

        // test 1
        ostringstream oss;
        oss << "int val: " << int_val << ", float val: " << flt_val << ", msg: " << msg << endl;
        string str = oss.str();
        printf("%s\n", str.c_str());

        // test 2
        oss << "this is a new value of oss" << endl;
        str = oss.str();
        printf("%s\n", str.c_str());

        // test 3
        str = move(oss.str());
        oss << "new oss val 2" << endl;
        str = oss.str();
        printf("%s\n", str.c_str());

        return 0;
    }
    ```

    output:

    ```
    int val: 42, float val: 3.14159, msg: hello, world

    int val: 42, float val: 3.14159, msg: hello, world
    this is a new value of oss

    int val: 42, float val: 3.14159, msg: hello, world
    this is a new value of oss
    new oss val 2

    ```

    其中，test 1 展示了`ostringstream`的常见用法。`endl`之类的控制字符也能被正常识别。

    test 2 说明`oss`的 buffer 似乎是可以动态扩容的，不会因为赋值给了 string 就清空。

    test 3 说明`oss`的 buffer string 不会因为右值引用而被轻易销毁，那么`oss.str()`概率是每次都新建的匿名对象。

* c++ 如果 string 中`\0`后仍有字符，那么使用`str_2 = str_1`仍会保留末尾的无效字符，而`str_2 = str_1.c_str()`会清除无效字符。

    example:

    ```cpp
    #include <cstring>
    #include <string>
    #include <iostream>
    using namespace std;

    int main() {
        string str_1;
        str_1.resize(100);
        strcpy(str_1.data(), "hello, world");

        string str_2 = str_1;
        cout << str_2.size() << endl;

        str_2 = str_1.c_str();
        cout << str_2.size() << endl;

        return 0;
    }
    ```

    output:

    ```
    100
    12
    ```

* `string`会比较 string 中的全部字符，不止是有效字符

    ```cpp
    #include <string>
    #include <stdio.h>
    using namespace std;

    int main()
    {
        string str_1;
        str_1.resize(24, 0);
        sprintf(str_1.data(), "hello");

        string str_2;
        str_2.resize(16, 0);
        sprintf(str_2.data(), "hello");

        if (str_1 == str_2) {
            printf("str_1 == str_2\n");
        } else {
            printf("str_1 != str_2\n");
        }

        string str_3;
        str_3.resize(16, 0);
        sprintf(str_3.data(), "hello");
        str_3[str_3.size() - 1] = '1';

        if (str_2 == str_3) {
            printf("str_2 == str_3\n");
        } else {
            printf("str_2 != str_3\n");
        }

        return 0;
    }
    ```

    output:

    ```
    str_1 != str_2
    str_2 != str_3
    ```

    上面代码中，`str_1`和`str_2`的长度不同因此不相等，`str_2`与`str_3`虽然长度相同，并且有效字符也相同，但是存在非有效字符不同的情况，因此也不相等。

    使用`cout`输出三个字符串：

    ```cpp
    cout << str_1 << endl;
    cout << str_2 << endl;
    cout << str_3 << endl;
    ```

    output:

    ```
    hello
    hello
    hello1
    ```

* cpp 中，对 string 进行`clear()`，会使其`size()`变为 0，但是`capacity()`不会改变。

    ```cpp
    #include <string>
    #include <iostream>
    using namespace std;

    int main()
    {
        string str{"hello, world"};
        cout << str.size() << endl;
        str.clear();
        cout << str << endl;
        cout << str.size() << endl;
        cout << str.capacity() << endl;
        return 0;
    }
    ```

    output:

    ```
    12

    0
    15
    ```

* cpp 中，使用自定义的`operator[]`可以实现使用`[]`取 const 元素的功能

    ```cpp
    #include <unordered_map>
    #include <string>
    #include <iostream>
    #include <initializer_list>
    using namespace std;

    struct MyStruct
    {
        MyStruct(const initializer_list<pair<const char*, const char*>> &init_list)
        {
            for (auto &entry : init_list)
            {
                m.emplace(entry.first, entry.second);
            }
        }

        const string& operator[](const string &key) const
        {
            return m.at(key);
        }

        unordered_map<string, string> m;
    };


    int main()
    {
        const MyStruct obj
        {
            {"hello", "val_1"},
            {"world", "val_2"}
        };

        string key = "hello";
        // const string &val = obj.m[key];  // compiling error
        const string &val = obj[key];
        cout << "key: " << key << ", val: " << val << endl;
        return 0;
    }
    ```

    output:

    ```
    key: hello, val: val_1
    ```

    需要在`const string& operator[](const string &key)`后面加上`const`，才能在`main()`中使用`obj[key]`的方式返回成员的 const 引用。

    说明：

    * 即使在`operator[]()`后加了`const`，标记这是一个 const 函数，依然无法使用`return m[key];`返回`const string&`。

        翻看`unordered_map`源码可发现，`operator[]`的实现方式如下：

        ```cpp
        mapped_type&
        operator[](const key_type& __k)
        { return _M_h[__k]; }

        mapped_type&
        operator[](key_type&& __k)
        { return _M_h[std::move(__k)]; }
        ```

        可以看到并不是一个`const`函数，因此虽然函数返回的是`const Type&`，保证不会在函数外修改，但是有可能在`operator[]()`中对 const 成员进行修改，这是编译器不允许的。

        而源码中`at()`函数的实现如下：

        ```cpp
        mapped_type&
        at(const key_type& __k)
        { return _M_h.at(__k); }

        const mapped_type&
        at(const key_type& __k) const
        { return _M_h.at(__k); }
        ```

        当返回非 const 引用时，`at()`函数不是 const 的；当返回 const 引用时，`at()`函数是 const 的。这样就保证了无论外面是需要 const 引用，还是非 const 引用，`at()`都可以正常工作。

* 在`vector`存指针，并通过指针建立复杂的拓扑关系，并不是一个明智的选择

    假如现在有两个 vector:

    ```
    0, 1, 2, 3

    4, 5, 6, 7
    ```

    其中每个元素通过指针的方式建立拓扑，连接关系如下：

    ```
    0 -> 5 -> 2 -> 7 -> 3 -> 6 -> 1 -> 4
    ```

    假如现在在 vector 1 中添加新的元素`8, 9, 10, 11, 12`，此时肯定会触发 vector 1 的扩容，那么 vector 2 中元素所指向的 vector 1 中的元素的指针全都会失效。

    vector 1 扩容时，其中的每个元素都会被调用拷贝构造函数，copy constructor。这是元素唯一知道扩容的消息，按道理此时应该通知 vector 2 中的元素，重新构建新的 next 指针。但是由于链表是单向的，vector 1 中的元素无法快速定位到其在 vector 1 中的 prev 元素，比如元素 2，我们可以快速找到 2 的 next 元素为 7，但是无法直接找到 2 的 prev 元素为 5。此时我们只能遍历 vector 2，找到所有指向 2 的元素。这样会导致效率低下。

    一种更好的解决方案是在建立拓扑关系时，使用索引，而不是指针。只有当我们保证没有 realloc 时，才使用指针建立拓扑。

* c++ 中，如果`stol()`无法解析，那么会直接抛出异常

    ```cpp
    int main()
    {
        string str = "h123";
        long val = stol(str, NULL, 10);
        cout << val << endl;

        return 0;
    }
    ```

    output:

    ```
    terminate called after throwing an instance of 'std::invalid_argument'
      what():  stol
    Aborted (core dumped)
    ```

* c++ 的 unordered map 的 operator[] 似乎不支持返回 const 引用

    ```cpp
    struct MyStruc
    {
        unordered_map<string, string> attrs;
    };

    void test(const MyStruc &struc)
    {
        const string &val_1 = struc.attrs["key_1"];  // compile error
        const string &val_2 = struc.attrs.at("key_2");  // OK
    }

    int main()
    {
        MyStruc struc {
            {
                {"key_1", "hello"},
                {"key_2", "world"}
            }
        };

        for (auto iter = struc.attrs.begin(); iter != struc.attrs.end(); ++iter)
        {
            cout << iter->first << ": " << iter->second << endl;
        }
        putchar('\n');

        test(struc);

        return 0;
    }
    ```

    可能是考虑到`[]`不处理异常，而确实有可能 unordered map 中找不到 key 值，从而不知道返回什么。而`at()`是处理异常的，因此如果找不到 key，直接抛出异常就可以了。

* `sort()`对数组排序时，末尾迭代器需要填数组最后一个元素的后一位的指针

    example:

    ```cpp
    #include <algorithm>
    #include <vector>
    #include <iostream>
    using namespace std;

    void print_arr(int arr[], int len)
    {
        for (int i = 0; i < len; ++i)
        {
            printf("%d, ", arr[i]);
        }
        putchar('\n');
    }

    int main()
    {
        int arr[] = {4, 5, 2, 1, 3};
        int arr_len = 5;
        print_arr(arr, arr_len);

        sort(&arr[0], &arr[arr_len - 1], [](int &obj_1, int &obj_2){
            if (obj_1 > obj_2)
                return true;
            return false;
        });
        print_arr(arr, arr_len);

        sort(&arr[0], &arr[arr_len], [](int &obj_1, int &obj_2){
            if (obj_1 > obj_2)
                return true;
            return false;
        });
        print_arr(arr, arr_len);
        
        return 0;
    }
    ```

    output:

    ```
    4, 5, 2, 1, 3, 
    5, 4, 2, 1, 3, 
    5, 4, 3, 2, 1,
    ```

* partial sort

    `partial_sort()`可以找到数组中前`n`个最大/最小的值，并对其进行排序。

    > Rearranges elements such that the range `[first, middle)` contains the sorted `middle − first` smallest elements in the range `[first, last)`.

    syntax:

    ```cpp
    template< class RandomIt >
    void partial_sort( RandomIt first, RandomIt middle, RandomIt last );

    template< class RandomIt, class Compare >
    void partial_sort( RandomIt first, RandomIt middle, RandomIt last, Compare comp );
    ```

    example:

    ```cpp
    #include <vector>
    #include <iostream>
    #include <algorithm>
    using namespace std;

    void print_vec(vector<int> &vec)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            cout << vec[i] << ", ";
        }
        cout << endl;
    }

    int main()
    {
        vector<int> vec{5, 4, 2, 1, 3};
        print_vec(vec);
        int n = 3;
        cout << "n = " << n << endl;
        partial_sort(vec.begin(), vec.begin() + n, vec.end());
        print_vec(vec);
        return 0;
    }
    ```

    output:

    ```
    5, 4, 2, 1, 3,
    n = 3
    1, 2, 3, 5, 4,
    ```

    可以看到 middle 的迭代器是不被包含的。当`n = 3`时，只对最小的 3 个数进行排序。

    `partial_sort()`也支持自定义 compare 函数：

    ```cpp
    #include <vector>
    #include <iostream>
    #include <algorithm>
    using namespace std;

    void print_vec(vector<int> &vec)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            cout << vec[i] << ", ";
        }
        cout << endl;
    }

    int main()
    {
        vector<int> vec{3, 1, 2, 5, 4};
        print_vec(vec);
        int n = 3;
        cout << "n = " << n << endl;
        partial_sort(vec.begin(), vec.begin() + n, vec.end(),
            [](int &a, int &b) {
                if (a > b)
                    return true;
                return false;
            }
        );
        print_vec(vec);
        return 0;
    }
    ```

    output:

    ```
    3, 1, 2, 5, 4,
    n = 3
    5, 4, 3, 1, 2,
    ```

    注：

    * 如果是比较简单的小于/大于功能，compare 使用的 lambda 表达式可以替换为`less()`或`greater()`。

        或者`less{}`，`greater{}`，用初始化列表创建匿名对象。

    问题：
    
    1. 如果想找到第`k`小/大的数，是否有算法比 partial sort 更快？

    1. 除了前面的两个 syntax，还有加入 policy 模板参数的 syntax，不知道干嘛用的

        ```cpp
        template< class ExecutionPolicy, class RandomIt >
        void partial_sort( ExecutionPolicy&& policy, RandomIt first, RandomIt middle, RandomIt last );

        template< class ExecutionPolicy, class RandomIt, class Compare >
        void partial_sort( ExecutionPolicy&& policy, RandomIt first, RandomIt middle, RandomIt last, Compare comp );
        ```

    1. 调研`nth_element()`, `partial_sort_copy()`, `stable_sort()`, `ranges::partial_sort()`

* lower bound, upper bound

    `lower_bound()`是找到**大于等于**指定元素的第一个元素，`upper_bound()`找到**大于**指定元素的第一个元素。

    ```cpp
    #include <vector>
    #include <iostream>
    #include <algorithm>
    using namespace std;

    int find_lower_upper_bound(vector<int> &vec, int val)
    {
        auto iter = lower_bound(vec.begin(), vec.end(), val);
        if (iter == vec.end())
        {
            cout << "fail to find lower bound" << endl;
            return -1;
        }
        int idx = distance(vec.begin(), iter);
        cout << "lower bound idx: " << idx << ", val: " << *iter << endl;

        iter = upper_bound(vec.begin(), vec.end(), val);
        if (iter == vec.end())
        {
            cout << "fail to find upper bound" << endl;
            return -1;
        }
        idx = distance(vec.begin(), iter);
        cout << "upper bound idx: " << idx << ", val: " << *iter << endl;

        return 0;
    }

    int main()
    {
        vector<int> vec{1, 2, 3, 4, 4, 6, 7};
        for (int elm: vec)
            cout << elm << ", ";
        cout << endl << endl;

        int val = 4;
        cout << "val: " << val << endl;
        find_lower_upper_bound(vec, val);
        cout << endl;

        val = 5;
        cout << "val: " << val << endl;
        find_lower_upper_bound(vec, val);
        cout << endl;

        val = 7;
        cout << "val: " << val << endl;
        find_lower_upper_bound(vec, val);

        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 4, 4, 6, 7,

    val: 4
    lower bound idx: 3, val: 4
    upper bound idx: 5, val: 6

    val: 5
    lower bound idx: 5, val: 6
    upper bound idx: 5, val: 6

    val: 7
    lower bound idx: 6, val: 7
    fail to find upper bound
    ```

    注意，`lower_bound()`和`upper_bound()`都是使用二分查找法，要求原数组有序。

    问题：如果`vec`是从小到大排列，我想找到小于等于`val`的第一个数，以及小于`val`的第一个数，该怎么办？如果`vec`已经是从大到小排列，又该如何找大于等于`val`的第一个作业君，大于`val`的第一个数；以及小于等于`val`的第一个数，小于`val`的第一个数。

* `find_if_not()`的使用逻辑与`find_if()`相似，只不过变成寻找第一个不满足条件的元素

    ```cpp
    #include <vector>
    #include <iostream>
    #include <algorithm>
    using namespace std;

    int main()
    {
        vector<int> vec{2, 5, 3, 4, 5, 1};
        int val = 2;
        vector<int>::iterator iter = find_if_not(vec.begin(), vec.end(), [&](int obj){
            if (obj == val)
                return true;
            return false;
        });
        if (iter == vec.end())
        {
            cout << "fail to find the element not equal to " << val << endl;
            return 0;
        }
        int idx = distance(vec.begin(), iter);
        cout << "the elm " << *iter << " is not equal to " << val << ", idx: " << idx << endl;
        return 0;
    }
    ```

    output:

    ```
    the elm 5 is not equal to 2, idx: 1
    ```

* `find_if()`

    syntax:

    ```cpp
    template<class InputIt, class UnaryPred>
    InputIt find_if(InputIt first, InputIt last, UnaryPred p);

    template<class ExecutionPolicy, class ForwardIt, class UnaryPred>
    ForwardIt find_if(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, UnaryPred p);
    ```

    可以看到，第 2 个 syntax 多了一个`policy`，目前不知道这个参数是干嘛用的。

    example:

    ```cpp
    #include <vector>
    #include <iostream>
    #include <algorithm>
    using namespace std;

    int main()
    {
        vector<int> vec{2, 3, 1, 5, 4};
        int val = 5;
        auto iter = find_if(vec.begin(), vec.end(), [val](int &obj){
            if (obj == val)
                return true;
            return false;
        });
        if (iter == vec.end())
        {
            cout << "fail to find " << val << endl;
            return 0;
        }
        int pos = distance(vec.begin(), iter);
        cout << val << " is at index " << pos << endl;
        return 0;
    }
    ```

    output:

    ```
    5 is at index 3
    ```

    说明：

    * lambda 表达式中的`int &obj`可以写成`int obj`, `const int &obj`, `const int obj`，效果都一样的。

    * lambda 表达式的捕捉列表`[val]`可以写成`[&val]`, `[&]`, `[=]`

    * 如果`vec`中有多个符合要求元素，那么会返回找到的第 1 个元素的迭代器。

        如果需要找到所有符合要求的元素，可以这样写：

        ```cpp
        #include <vector>
        #include <iostream>
        #include <algorithm>
        using namespace std;

        int main()
        {
            vector<int> vec{2, 3, 5, 5, 4, 5};
            int val = 5;
            auto iter = find_if(vec.begin(), vec.end(), [&](int &obj) {
                if (obj == val)
                    return true;
                return false;
            });
            if (iter == vec.end())
            {
                cout << "fail to find " << val << endl;
                return 0;
            }
            int pos = distance(vec.begin(), iter);
            cout << val << " is at index " << pos << endl;
            while (true)
            {
                iter = find_if(iter + 1, vec.end(), [&](int &obj) {
                    if (obj == val)
                        return true;
                    return false;
                });
                if (iter == vec.end())
                    break;
                pos = distance(vec.begin(), iter);
                cout << val << " is at index " << pos << endl;
            }
            return 0;
        }
        ```

        output:

        ```
        5 is at index 2
        5 is at index 3
        5 is at index 5
        ```

        比较核心的一句是循环执行`iter = find_if(iter + 1, ...`，直到结尾。

    * `find_if()`可以处理任意具有迭代器的容器

        ```cpp
        #include <vector>
        #include <iostream>
        #include <algorithm>
        using namespace std;

        int main()
        {
            string str{"hello, world"};
            char ch = 'l';
            auto iter = find_if(str.begin(), str.end(), [=](char obj) {
                if (obj == ch)
                    return true;
                return false;
            });
            if (iter == str.end())
            {
                cout << "fail to find " << ch << endl;
                return 0;
            }
            int pos = distance(str.begin(), iter);
            cout << ch << " is at index " << pos << endl;
            while (true)
            {
                iter = find_if(iter + 1, str.end(), [=](char obj) {
                    if (obj == ch)
                        return true;
                    return false;
                });
                if (iter == str.end())
                    break;
                pos = distance(str.begin(), iter);
                cout << ch << " is at index " << pos << endl;
            }
            return 0;
        }
        ```

        output:

        ```
        l is at index 2
        l is at index 3
        l is at index 10
        ```

    * 问题：对于自定义的容器，该如何为其实现迭代器，并对接到`find()`等算法上？

* c++ 中`string`插入一个字符时，不支持`str.insert(int pos, char c)`的形式，只支持`str.insert(iterator iter, char c)`，因此需要用这种方式插入字符：

    ```cpp
    #include <string>
    #include <iostream>
    using namespace std;

    int main()
    {
        string str{"helloworld"};
        int pos = 5;
        str.insert(str.begin() + pos, ' ');
        cout << str << endl;
        return 0;
    }
    ```

    output:

    ```
    hello world
    ```

    其原型为

    ```cpp
    iterator insert (const_iterator p, char c);
    ```

    `insert()`可以解释为，在迭代器`p`所在的元素之前插入`c`；也可以解释为，使得迭代器`p`位置处的字符为`c`，同时不破坏原字符串前后的有序关系。

    除了这种方式，还可以使用 fill 的方式：

    ```cpp
    string& insert (size_t pos,   size_t n, char c);
    iterator insert (const_iterator p, size_t n, char c);
    ```

    example:

    ```cpp
    #include <string>
    #include <iostream>
    using namespace std;

    int main()
    {
        string str{"helloworld"};
        int pos = 5;
        str.insert(pos, 1, ' ');
        cout << str << endl;
        pos = 11;
        str.insert(str.begin() + pos, 1, '!');
        cout << str << endl;
        return 0;
    }
    ```

    output:

    ```
    hello world
    hello world!
    ```

* 有关指向`vector<>`的指针的变动的临时解决方案

    ```cpp
    #include <iostream>
    #include <vector>
    using namespace std;

    struct MyStruc
    {
        int val;
        vector<MyStruc> vec;
        MyStruc *parent;

        MyStruc(const MyStruc &obj_src)
        {
            val = obj_src.val;
            vec = obj_src.vec;
            parent = obj_src.parent;
            for (auto &struc: vec)
            {
                struc.parent = this;
            }
        }

        MyStruc(int &val, vector<MyStruc> &&vec, MyStruc * &&parent)
            : val(val), vec(vec), parent(parent) { }
    };

    int main()
    {
        int num_outer = 10, num_inner = 5;
        printf("num_outer: %d, num_innter: %d\n", num_outer, num_inner);

        vector<MyStruc> strucs;

        for (int i = 0; i < num_outer; ++i)
        {
            strucs.push_back({i, {}, NULL});
            for (int j = 0; j < num_inner; ++j)
            {
                strucs[i].vec.push_back({j, {}, &strucs[i]});
                printf("strucs[i].vec[j].parent: %p, &strucs[i]: %p\n",
                    strucs[i].vec[j].parent, &strucs[i]);
            }
        }

        for (int i = 0; i < num_outer; ++i)
        {
            for (int j = 0; j < num_inner; ++j)
            {
                if (strucs[i].vec[j].parent != &strucs[i])
                {
                    printf("i = %d, j = %d, parent is not corrent\n", i, j);
                    printf("strucs[i].vec[j].parent: %p, &strucs[i]: %p\n",
                        strucs[i].vec[j].parent, &strucs[i]);
                    return 0;
                }
            }
        }
        printf("all parents are correct.\n");

        return 0;
    }
    ```

    output:

    ```
    num_outer: 10, num_innter: 5
    strucs[i].vec[j].parent: 0x6529294682c0, &strucs[i]: 0x6529294682c0
    strucs[i].vec[j].parent: 0x6529294682c0, &strucs[i]: 0x6529294682c0
    strucs[i].vec[j].parent: 0x6529294682c0, &strucs[i]: 0x6529294682c0
    strucs[i].vec[j].parent: 0x6529294682c0, &strucs[i]: 0x6529294682c0
    strucs[i].vec[j].parent: 0x6529294682c0, &strucs[i]: 0x6529294682c0
    strucs[i].vec[j].parent: 0x652929468348, &strucs[i]: 0x652929468348
    strucs[i].vec[j].parent: 0x652929468348, &strucs[i]: 0x652929468348
    strucs[i].vec[j].parent: 0x652929468348, &strucs[i]: 0x652929468348
    strucs[i].vec[j].parent: 0x652929468348, &strucs[i]: 0x652929468348
    strucs[i].vec[j].parent: 0x652929468348, &strucs[i]: 0x652929468348
    strucs[i].vec[j].parent: 0x6529294683d0, &strucs[i]: 0x6529294683d0
    strucs[i].vec[j].parent: 0x6529294683d0, &strucs[i]: 0x6529294683d0
    strucs[i].vec[j].parent: 0x6529294683d0, &strucs[i]: 0x6529294683d0
    strucs[i].vec[j].parent: 0x6529294683d0, &strucs[i]: 0x6529294683d0
    strucs[i].vec[j].parent: 0x6529294683d0, &strucs[i]: 0x6529294683d0
    strucs[i].vec[j].parent: 0x6529294683f8, &strucs[i]: 0x6529294683f8
    strucs[i].vec[j].parent: 0x6529294683f8, &strucs[i]: 0x6529294683f8
    strucs[i].vec[j].parent: 0x6529294683f8, &strucs[i]: 0x6529294683f8
    strucs[i].vec[j].parent: 0x6529294683f8, &strucs[i]: 0x6529294683f8
    strucs[i].vec[j].parent: 0x6529294683f8, &strucs[i]: 0x6529294683f8
    strucs[i].vec[j].parent: 0x652929468af0, &strucs[i]: 0x652929468af0
    strucs[i].vec[j].parent: 0x652929468af0, &strucs[i]: 0x652929468af0
    strucs[i].vec[j].parent: 0x652929468af0, &strucs[i]: 0x652929468af0
    strucs[i].vec[j].parent: 0x652929468af0, &strucs[i]: 0x652929468af0
    strucs[i].vec[j].parent: 0x652929468af0, &strucs[i]: 0x652929468af0
    strucs[i].vec[j].parent: 0x652929468b18, &strucs[i]: 0x652929468b18
    strucs[i].vec[j].parent: 0x652929468b18, &strucs[i]: 0x652929468b18
    strucs[i].vec[j].parent: 0x652929468b18, &strucs[i]: 0x652929468b18
    strucs[i].vec[j].parent: 0x652929468b18, &strucs[i]: 0x652929468b18
    strucs[i].vec[j].parent: 0x652929468b18, &strucs[i]: 0x652929468b18
    strucs[i].vec[j].parent: 0x652929468b40, &strucs[i]: 0x652929468b40
    strucs[i].vec[j].parent: 0x652929468b40, &strucs[i]: 0x652929468b40
    strucs[i].vec[j].parent: 0x652929468b40, &strucs[i]: 0x652929468b40
    strucs[i].vec[j].parent: 0x652929468b40, &strucs[i]: 0x652929468b40
    strucs[i].vec[j].parent: 0x652929468b40, &strucs[i]: 0x652929468b40
    strucs[i].vec[j].parent: 0x652929468b68, &strucs[i]: 0x652929468b68
    strucs[i].vec[j].parent: 0x652929468b68, &strucs[i]: 0x652929468b68
    strucs[i].vec[j].parent: 0x652929468b68, &strucs[i]: 0x652929468b68
    strucs[i].vec[j].parent: 0x652929468b68, &strucs[i]: 0x652929468b68
    strucs[i].vec[j].parent: 0x652929468b68, &strucs[i]: 0x652929468b68
    strucs[i].vec[j].parent: 0x6529294691f0, &strucs[i]: 0x6529294691f0
    strucs[i].vec[j].parent: 0x6529294691f0, &strucs[i]: 0x6529294691f0
    strucs[i].vec[j].parent: 0x6529294691f0, &strucs[i]: 0x6529294691f0
    strucs[i].vec[j].parent: 0x6529294691f0, &strucs[i]: 0x6529294691f0
    strucs[i].vec[j].parent: 0x6529294691f0, &strucs[i]: 0x6529294691f0
    strucs[i].vec[j].parent: 0x652929469218, &strucs[i]: 0x652929469218
    strucs[i].vec[j].parent: 0x652929469218, &strucs[i]: 0x652929469218
    strucs[i].vec[j].parent: 0x652929469218, &strucs[i]: 0x652929469218
    strucs[i].vec[j].parent: 0x652929469218, &strucs[i]: 0x652929469218
    strucs[i].vec[j].parent: 0x652929469218, &strucs[i]: 0x652929469218
    all parents are correct.
    ```

    `vector<>`在自动重新分配内存时，会调用元素的拷贝构造函数，我们为了填充正确的`parent`，需要自定义拷贝构造函数。思路是遍历成员`vector<>`中的所有元素，使之指向当前`struct`。

    由于自定义了拷贝构造函数，所以默认的构造函数失效，而`initializer_list`又会用到默认构造函数（为什么？），所以我们还需要再简单实现一下默认构造函数。

    ```cpp
    MyStruc(int &val, vector<MyStruc> &&vec, MyStruc *&&parent)
        : val(val), vec(vec), parent(parent) { }
    ```

    由于我们在下面代码的初始化列表中使用的是`{i, {}, NULL}`，即左值，右值，右值，因此默认构造函数的参数列表也对应为 左值引用，右值引用，右值引用。

* `vector`与指针混用时，指针可能失效

    ```cpp
    #include <iostream>
    #include <vector>
    using namespace std;

    struct MyStruc
    {
        int val;
        vector<MyStruc> vec;
        MyStruc *parent;
    };

    int main()
    {
        int num_outer = 5, num_inner = 1;
        printf("num_outer: %d, num_innter: %d\n", num_outer, num_inner);

        vector<MyStruc> strucs;
        // strucs.reserve(10);

        for (int i = 0; i < num_outer; ++i)
        {
            strucs.push_back({i, {}, NULL});
            for (int j = 0; j < num_inner; ++j)
            {
                strucs[i].vec.push_back({j, {}, &strucs[i]});
                printf("strucs[i].vec[j].parent: %p, &strucs[i]: %p\n",
                    strucs[i].vec[j].parent, &strucs[i]);
            }
        }

        for (int i = 0; i < num_outer; ++i)
        {
            for (int j = 0; j < num_inner; ++j)
            {
                if (strucs[i].vec[j].parent != &strucs[i])
                {
                    printf("i = %d, j = %d, parent is not corrent\n", i, j);
                    printf("strucs[i].vec[j].parent: %p, &strucs[i]: %p\n",
                        strucs[i].vec[j].parent, &strucs[i]);
                    return 0;
                }
            }
        }
        printf("all parents are correct.\n");
        return 0;
    }
    ```

    output:

    ```
    num_outer: 5, num_innter: 1
    strucs[i].vec[j].parent: 0x58b9e1ef72c0, &strucs[i]: 0x58b9e1ef72c0
    strucs[i].vec[j].parent: 0x58b9e1ef7348, &strucs[i]: 0x58b9e1ef7348
    strucs[i].vec[j].parent: 0x58b9e1ef73d0, &strucs[i]: 0x58b9e1ef73d0
    strucs[i].vec[j].parent: 0x58b9e1ef73f8, &strucs[i]: 0x58b9e1ef73f8
    strucs[i].vec[j].parent: 0x58b9e1ef7530, &strucs[i]: 0x58b9e1ef7530
    i = 0, j = 0, parent is not corrent
    strucs[i].vec[j].parent: 0x58b9e1ef72c0, &strucs[i]: 0x58b9e1ef7490
    ```

    将`// strucs.reserve(10);`这行注释取消掉后，输出为：

    ```
    num_outer: 5, num_innter: 1
    strucs[i].vec[j].parent: 0x59a3456902c0, &strucs[i]: 0x59a3456902c0
    strucs[i].vec[j].parent: 0x59a3456902e8, &strucs[i]: 0x59a3456902e8
    strucs[i].vec[j].parent: 0x59a345690310, &strucs[i]: 0x59a345690310
    strucs[i].vec[j].parent: 0x59a345690338, &strucs[i]: 0x59a345690338
    strucs[i].vec[j].parent: 0x59a345690360, &strucs[i]: 0x59a345690360
    all parents are correct.
    ```

    由此可见，当`parent`指针指向`vector<>`中的元素时，由于`vector`会动态释放申请内存，所以`parent`有可能变成野指针。

    智能指针是否可以解决这个问题？如果智能指针也不行，那么如何实现`parent`这个功能？

* `find()`在`<algorithm>`库中，主要功能是线性搜索

    如果传给`find()`的是自定义的 struct，那么需要重载`operator==`，此时`operator==()`的参数必须有`const`修饰。如果只是用`==`比较两个 struct 对象是否相等的话就不需要`const`。目前不清楚为什么。

    example:

    ```cpp
    #include <iostream>
    #include <string>
    #include <vector>
    #include <algorithm>
    using namespace std;

    struct Substruc
    {
        string val_1;
        int val_2;
    };

    struct MyStruc
    {
        int val_1;
        string val_2;
        Substruc sub_struc;

        bool operator==(const MyStruc &obj_2)  // 这里必须有 const
        {
            if (this->val_1 == obj_2.val_1
                && this->val_2 == obj_2.val_2
                && this->sub_struc.val_1 == obj_2.sub_struc.val_1
                && this->sub_struc.val_2 == obj_2.sub_struc.val_2
            )
                return true;
            return false;
        }
    };

    int main()
    {
        vector<MyStruc> vec;

        vec.push_back({3, "3", {"3", 3}});
        vec.push_back({1, "1", {"1", 1}});
        vec.push_back({2, "2", {"2", 2}});

        MyStruc val{1, "1", {"1", 1}};
        auto iter = find(vec.begin(), vec.end(), val);
        cout << "idx: " << distance(vec.begin(), iter) << ", " << iter->val_1 << endl;
        
        return 0;
    }
    ```

    output:

    ```
    idx: 1, 1
    ```

    `find()`还支持传入匿名对象：

    `auto iter = find(vec.begin(), vec.end(), MyStruc{1, "1", {"1", 1}});`

    但是`find()`不支持直接传入初始化列表：

    `auto iter = find(vec.begin(), vec.end(), {1, "1", {"1", 1}});` -> error，目前不清楚为什么。

    如果重载全局`operator==`，那么第一个参数可以不加 const，但是第二个参数必须加 const：

    ```cpp
    #include <iostream>
    #include <string>
    #include <vector>
    #include <algorithm>
    using namespace std;

    struct Substruc
    {
        string val_1;
        int val_2;
    };

    struct MyStruc
    {
        int val_1;
        string val_2;
        Substruc sub_struc;
    };

    // 通过 find() 间接调用 == 时，operator==() 的第二个参数必须加 const,
    // 第一个参数可以不加
    bool operator==(MyStruc &obj_1, const MyStruc &obj_2)
    {
        if (obj_1.val_1 == obj_2.val_1
            && obj_1.val_2 == obj_2.val_2
            && obj_1.sub_struc.val_1 == obj_2.sub_struc.val_1
            && obj_1.sub_struc.val_2 == obj_2.sub_struc.val_2
        )
            return true;
        return false;
    }

    int main()
    {
        vector<MyStruc> vec;

        vec.push_back({3, "3", {"3", 3}});
        vec.push_back({1, "1", {"1", 1}});
        vec.push_back({2, "2", {"2", 2}});

        MyStruc val{1, "1", {"1", 1}};
        auto iter = find(vec.begin(), vec.end(), val);
        cout << "idx: " << distance(vec.begin(), iter) << ", " << iter->val_1 << endl;
        
        return 0;
    }
    ```

* `lower_bound()`与各种类型

    ```cpp
    #include <algorithm>
    #include <iostream>
    #include <vector>
    #include <unordered_set>
    #include <map>
    using namespace std;

    int main()
    {
        int value = 4;

        // test 1, vector<int>
        vector<int> vec{1, 2, 3, 4, 5, 4, 3, 2};
        auto iter = lower_bound(vec.begin(), vec.end(), value);
        int idx = distance(vec.begin(), iter);
        cout << "idx: " << idx << ", " << "val: " << *iter << endl;

        // test 2, unordered_set<int>
        unordered_set<int> m_set{1, 2, 3, 4, 5,};
        auto iter_2 = lower_bound(m_set.begin(), m_set.end(), value);
        idx = distance(m_set.begin(), iter_2);
        cout << "idx: " << idx << endl;
        // cout << "idx: " << idx << ", " << "val: " << *iter_2  << endl;

        // test 3, vector<pait<int, string>>
        vector<pair<int, string>> str_pairs {
            {4, "this is 4"},
            {1, "this is 1"},
            {2, "this is 2"},
            {3, "this is 3"}
        };
        pair<int, string> value_2 = {4, "this is 4321"};
        auto iter_3 = lower_bound(str_pairs.begin(), str_pairs.end(), value_2,
            [](pair<int, string> &p_1, const pair<int, string> &p_2) {
                if (p_1.first < p_2.first)
                    return true;
                return false;
            }
        );
        idx = distance(str_pairs.begin(), iter_3);
        cout << "idx: " << idx << ", " << "val: " << iter_3->first << ", " << iter_3->second << endl;

        // test 4, map<int, string>
        map<int, string> m_map {
            {1, "this is 1"},
            {2, "this is 2"},
            {3, "this is 3"},
            {4, "this is 4"}
        };
        pair<int, string> value_3 = {4, "aaa"};
        auto iter_4 = lower_bound(m_map.begin(), m_map.end(), value_3,
            [](const pair<int, string> &p_1, const pair<int, string> &p_2){
                if (p_1.first < p_2.first)
                    return true;
                return false;
            }
        );
        idx = distance(m_map.begin(), iter_4);
        cout << "idx: " << idx << ", " << "val: " << iter_4->first << ", " << iter_4->second << endl;

        return 0;
    }
    ```

    output:

    ```
    idx: 3, val: 4
    idx: 5
    idx: 4, val: -1414812757,
    idx: 3, val: 4, this is 4
    ```

    `lower_bound()`的作用有两种描述：

    1. 找到大于等于`value`的第一个元素的迭代器。

    2. 找到一个迭代器，在此 insert `value`后，使得原数组仍然保持有序。

    `lower_bound()`使用二分查找法查找元素，因此要求原数组中的元素有序。

    说明：

    * `lower_bound()`返回的并不是下标，而是迭代器，可以使用`distance()`找到迭代器对应的下标

    * `unordered_set<>`哈希表无法使用`lower_bound()`，强行`*iter`会报错 segment fault

    * 如果容器中存储的是复合结构，比如`pair<>`，那么需要自己实现 comp 函数。test 3 中，lambda 表达式的第一个参数可以不是 const 引用，但是第二个参数必须是 const 引用，否则无法通过编译。目前不清楚原因。

    * test 4 中，lambda 表达式的两个参数都要求为 const 引用。目前不清楚原因。

    * 可以看到，由于`map`在存数据时，总是有序的，所以 test 3 找到的值是错的，而 test 4 找到的值是对的

    * 除了使用 lambda 表达式外，还可以使用 struct 函数，或`std::function`，或者为自定义对象实现`operator<`，目前还不清楚怎么用。

    * `value`除了可以是值，左值对象，还可以是右值对象：

        ```cpp
        auto iter_3 = lower_bound(str_pairs.begin(), str_pairs.end(), pair<int, string>{4, "this is 4321"},
            [](pair<int, string> &p_1, const pair<int, string> &p_2) {
                if (p_1.first < p_2.first)
                    return true;
                return false;
            }
        );
        ```

* c++ 中`string`的几种构造方式

    除了移动构造 move 没有给出例子，其他的都给出了例子。

    ```cpp
    #include <iostream>
    #include <vector>
    #include <string>
    using namespace std;

    int main()
    {
        string str("hello, world");  // from c-string (4)  string (const char* s);
        cout << string(str) << endl;  // copy (2)  string (const string& str);
        cout << string(str, 0, 5) << endl;  // substring (3)  string (const string& str, size_t pos, size_t len = npos);
        cout << string(str, 7) << endl;    
        cout << string("hello, world", 5) << endl;  // from buffer (5)  string (const char* s, size_t n);
        cout << string(5, 'a') << endl;  // fill (6)  string (size_t n, char c);
        vector<char> vec{'a', 'b', 'c', 'd', 'e'};
        cout << string(vec.begin(), vec.end()) << endl;  // range (7)	template <class InputIterator>  string  (InputIterator first, InputIterator last);
        char buf[64] = {'e', 'd', 'c', 'b', 'a'};
        cout << string(&buf[0], &buf[5]) << endl;  // range (7)	template <class InputIterator>  string  (InputIterator first, InputIterator last);
        cout << string{'h', 'e', 'l', 'l', 'o'} << endl;  // initializer list (8)  string (initializer_list<char> il);
        
        // 另外有一个移动构造函数 move (9)	string (string&& str) noexcept; 未在这里写出
        return 0;
    }
    ```

    output:

    ```
    hello, world
    hello
    world
    hello
    aaaaa
    abcde
    edcba
    hello
    ```

    使用`string()`构造出的其实是一个匿名对象，交由`<<`运算符处理时，相当于处理了右值引用。

    可以重点记下 substring 和 range 的用法。

* c++ `find()`

    c++ 中的`find()`是线性搜索，在使用时需要加上`#include <algorithm>`头文件。

    example:

    `main.cpp`:

    ```cpp
    #include <algorithm>
    #include <iostream>
    #include <vector>
    #include <string>
    using namespace std;

    struct MyClass
    {
        int val;
        string name;

        MyClass(int val, string name):
            val(val), name(name) {}

        bool operator==(const MyClass &obj_2)
        {
            if (val == obj_2.val && name == obj_2.name)
                return true;
            return false;
        }
    };

    bool operator==(const MyClass &obj_1, const MyClass &obj_2)
    {
        if (obj_1.val == obj_2.val && obj_1.name == obj_2.name)
            return true;
        return false;
    }

    int main()
    {
        vector<int> arr{1, 2, 3, 4, 5, 6, 7, 8};
        vector<int>::iterator iter = find(arr.begin(), arr.end(), 3);
        if (iter != arr.end())
        {
            cout << "found object: " <<  *iter << endl;
        }

        vector<MyClass> vec{
            {5, "hehe"},
            {7, "haha"},
            {12, "byebye"}
        };
        MyClass obj_to_find{7, "haha"};
        vector<MyClass>::iterator my_iter = find(
            vec.begin(), vec.end(), obj_to_find
        );
        if (my_iter != vec.end())
        {
            cout << "found obj, val: " << my_iter->val << ", name: " << my_iter->name << endl;
        }
        return 0;
    }
    ```

    compie: `g++ -g main.cpp -o main`

    run: `./main`

    output:

    ```
    found object: 3
    found obj, val: 7, name: haha
    ```

    通常`find()`需要传入起始和结束的迭代器，第三个参数传入一个`val`，`find()`会线性搜索容器，若找到元素，则返回元素的迭代器；若未找到，则返回结束位置的迭代器。

    迭代器可以看作是指针，直接对迭代器解引用就可以得到值。

    如果是自定义的类型，`find()`的第三个参数可以传入左值对象（比如上面例子中的`obj_to_find`），也可以传入右值对象，比如`MyClass{7, "haha"}`或`MyClass(7, "haha")`。但是看`find()`的函数原型，并没有显示专门对右值的处理，目前不清楚他是怎么能接收右值参数的。

    对于自定义的类型，还需要实现`operator==()`。这个函数可以在类里实现，也可以作为一个全局函数，两者只要实现一个就可以。上面的 example 示范了这两种写法。如果两种实现都写了，那么会优先调用成员函数。

    在写`operator==()`时，其参数必须加`const`，不加会报错。

    实现了`operator==()`，不代表可以判断`obj_1 != obj_2`。

* `random_device{}`只是创建一个`random_device`类型的匿名对象，而它本身又是个仿函数，所以还需要再加一个`()`来返回一个随机值。

    注：有空了把 stl random 相关的东西都整理一下，把这条放到 random 条目下

## vector

使用`swap`收缩内存空间：

```c++
#include <vector>

int main()
{
    vector<int> a(100);
    a.clear();
    a.resize(3);
    vector<int>(a).swap(a);
    return 0;
}
```

## string

可以通过指定数据起始地址和字节数的方式填充数据：

```cpp
string buf;
char *c = "abcdefagasdfw";
buf.append(c, 8);  // 给定数据的起始地址和个数
buf.append(c, c + 8);  // 也可以给定两个指针（或两个迭代器）
```

使用`str.data()`和`str.c_str()`获得的内存地址是一样的。`string`会在`str.size()`个字节后添加一个额外的`\0`，所以如果我们使用`memcpy()`往`str.data()`中写入字符串时，不需要担心程序会找不到字符串的结尾。

Example:

```cpp
#include <string>
#include <string.h>
using namespace std;

int main()
{
    char str_1[8];
    string str_2;
    str_2.resize(8);
    char src[9] = "abcdefgh";
    memcpy(str_1, src, 8);
    memcpy(str_2.data(), src, 8);
    printf("str_1: %s\n", str_1);
    printf("str_2: %s\n", str_2.c_str());
    return 0;
}
```

输出：

```
str_1: abcdefghabcdefgh
str_2: abcdefgh
```

可以看到`string`类型`resize()`为 8 个字节后，仍能正常显示字符串。但是`char str_1[8]`显示异常。

`assign()`会同时改变 string 的内容和 size。

## deque

deque 也支持随机访问。其原理是用一个中控器记录各小段数据的地址。

## unordered_map

自定义键值类型

1. 使用`std::function`

    ```c++
    #include <unordered_map>
    #include <functional>
    #include <string>
    using namespace std;

    class Person
    {
        public:
        string name;
        int age;

        Person(string n, int a)
        {
            name = n;
            age = a;
        }

        bool operator==(const Person &p) const
        {
            return name == p.name && age == p.age;
        }
    }

    size_t person_hash(const Person &p)
    {
        return hash<string>()(p.name) ^ hash<int>()(p.age);
    }

    int main()
    {
        unordered_map<Person, int, function<size_t(const Person &p)>> ids(100, person_hash);
        // 还可以写成：
        // unordered_map<Person, int, decltype(&person_hash)> ids(100, person_hash);
        // 还可以使用 lambda 函数
        //unordered_map<Person, int, function<size_t(const Person &p)>> ids(100, [](const Person &p){
        //    return hash<string>()(p.name) ^ hash<int>()(p.age);
        //});
        ids[Person("Mark", 17)] = 40561;
        ids[Person("Andrew", 16)] = 40562;
        return 0;
    }
    ```

1. 使用伪函数

    ```c++
    #include <string>
    #include <unordered_map>
    #include <functional>
    using namespace std;

    class Person
    {
        public:
        string name;
        int age;

        Person (string n, int a)
        {
            name = n;
            age = a;
        }

        bool operator==(const Person &p) const
        {
            return name == p.name && age == p.age;
        }
    };

    struct hash_name
    {
        size_t operator()(const Person &p) const
        {
            return hash<string>()(p.name) ^ hash<int>()(p.age);
        }
    };

    int main()
    {
        unordered_map<Person, int, hash_name> ids;
        ids[Person("Mark", 17)] = 40561;
        ids[Person("Andrew", 16)] = 40562;
        return 0;
    }
    ```

1. 模板定制

    ```c++
    #include <unordered_map>
    #include <string>
    #include <functional>
    using namespace std;

    typedef pair<string,string> Name;

    namespace std
    {
        template <>
        class hash<Name>
        {
            public:
            size_t operator()(const Name &name) const
            {
                return hash<string>()(name.first) ^ hash<string>()(name.second);
            }
        };
    };

    int main()
    {
        unordered_map<Name, int> ids;
        ids[Name("Mark", "Nelson")] = 40561;
        ids[Name("Andrew", "Binstock")] = 40562;
        return 0;
    }
    ```

    同理，`equal_to`函数也可以这样写。

1. `insert()`需要这样写：`insert({key_1, val_1})`。有时间了看下为什么。

### 插入元素时的 copy 机制

考虑这样一个问题：

我们实现了一个 class，在构造函数申请内存，在析构函数中释放内存。现在我们想要把这个对象放到`unordered_map`中，使用`string`作为键。可能会写出这样的代码：

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

class B
{
    public:
    B(const char *src) {
        cout << "constructor" << endl;
        buf = (char*) malloc(16);
        memset(buf, 0, 16);
        memcpy(buf, src, strlen(src));
        cout << "buf: " << buf << endl;
    }

    ~B() {
        cout << "destructor" << endl;
        free(buf);
    }

    char *buf;
};

int main()
{
    unordered_map<string, B> m;
    m.insert(make_pair("b", B("hello")));
    return 0;
}
```

直接运行会报错：

```
constructor
buf: hello
destructor
destructor
free(): double free detected in tcache 2
Aborted (core dumped)
```

为什么？我们试试只`make_pair()`:

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

class B
{
    public:
    B(const char *src) {
        cout << "constructor" << endl;
        buf = (char*) malloc(16);
        memset(buf, 0, 16);
        memcpy(buf, src, strlen(src));
        cout << "buf: " << buf << endl;
    }

    ~B() {
        cout << "destructor" << endl;
        free(buf);
    }

    char *buf;
};

int main()
{
    pair<string, B> p("b", B("hello"));
    return 0;
}
```

输出：

```
constructor
buf: hello
destructor
destructor
free(): double free detected in tcache 2
Aborted (core dumped)
```

和前面的输出相同，看来问题就在这个`make_pair()`上。我们看`make_pair()`的原型：

```cpp
template <class T1, class T2>  pair<V1,V2> make_pair (T1&& x, T2&& y);  // see below for definition of V1 and V2
```

可以看到传递进去的参数是引用，并没有调用`B`的构造函数。但是当 pair 对象被析构时，会调用`B`的析构函数吗？我们做个实验试试：

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

class B
{
    public:
    B(const char *src) {
        cout << "constructor" << endl;
        buf = (char*) malloc(16);
        memset(buf, 0, 16);
        memcpy(buf, src, strlen(src));
        cout << "buf: " << buf << endl;
    }

    ~B() {
        cout << "destructor" << endl;
        free(buf);
    }

    char *buf;
};

int main()
{
    unordered_map<string, B> m;
    B b("hello");
    cout << "break 1" << endl;
    {
        pair<string, B> p("b", b);
        cout << "break 2" << endl;
    }
    cout << "break 3" << endl;
    return 0;
}
```

输出：

```
constructor
buf: hello
break 1
break 2
destructor
break 3
destructor
free(): double free detected in tcache 2
Aborted (core dumped)
```

`break 2`和`break 3`之间的`destructor`证明了，在离开作用域销毁`p`时，调用了`b`的析构函数。然而在整个`main()`函数结束时，`b`也要被销毁，因此`b`的析构函数被再次调用。这样就导致了内存错误发生。

对于使用匿名对象的写法也是同理，

```cpp
auto p = make_pair("b", B("hello"));
```

`B("hello")`构造了一个匿名对象，它被按引用传入`make_pair()`后，马上被析构。当`p`离开它的作用域时，匿名对象的析构函数再次被调用，这样也导致了内存错误。

`unordered_map`也是同理，无论是使用`insert()`还是`emplace()`插入元素，都无法避免析构函数被调用两次的问题。为了解决这个问题，我们需要将对象的作用域控制在`unordered_map`内，由`unordered_map`负责构造和析构，而对外是不可见的。这时我们就需要用到`piecewise_construct`了：

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

class B
{
    public:
    B(const char *src) {
        cout << "constructor" << endl;
        buf = (char*) malloc(16);
        memset(buf, 0, 16);
        memcpy(buf, src, strlen(src));
        cout << "buf: " << buf << endl;
    }

    ~B() {
        cout << "destructor" << endl;
        free(buf);
    }

    char *buf;
};

int main()
{
    unordered_map<string, B> m;
    m.emplace(
        piecewise_construct, 
        forward_as_tuple("b"),
        forward_as_tuple("hello")
    );
    return 0;
}
```

输出：

```
constructor
buf: hello
destructor
```

这样就解决了报错的问题。

## list

Ref: <https://www.geeksforgeeks.org/list-cpp-stl/>

`list`是一个双向链表。

Examples:

```cpp
#include <list>
using namespace std;

void show_list(list<int> &lst)
{
    for (list<int>::iterator it = lst.begin(); it != lst.end(); ++it)  // 也可以使用 auto &num: lst
    {
        cout << *it << ", ";
    }
    cout << endl;
}

int main()
{
    list<int> lst({1, 3, 2});  // 1, 3, 2
    lst.push_front(4);  // 4, 1, 3, 2
    lst.push_back(5);  // 4, 1, 3, 2, 5
    lst.front();  // 4
    lst.back();  // 5
    lst.pop_front();  // 1, 3, 2, 5
    lst.pop_back();  // 1, 3, 2
    lst.reverse();  // 2, 3, 1
    lst.insert(++lst.begin(), 4);  // 2, 4, 3, 1
    lst.remove(2);  // 4, 3, 1
    lst.sort();  // 1, 3, 4
    show_list(lst);
    return 0;
}
```

## algorithm

* `for_each(iterator beg, iterator end, _func)`

    `_func`可以是仿函数对象，也可以是函数名。

* `transform(beg1, end1, beg2, _func)`

    目标容器需要提前开辟空间。

* `find(beg, end, value)`

    返回迭代器。如果是自定义类型，需要重载`==`。

* `find_if(beg, end, func_obj)`

* `adjacent_find(beg, end)`

    查找相邻重复元素，返回第一个元素的迭代器。

* `bool binary_search(beg, end, value)`

    二分查找，要求序列是有序的。

* `count(beg, end, val)`

    如果容器中有自定义类型对象，那么需要重载`==`。

* `count_if(beg, end, func_obj)`

* `sort(beg, end, func_obj)`

* `random_shuffle(beg, end)`

    随机打乱容器中元素顺序。

    可以用`srand((unsigned int)time(NULL));`设置种子。

* `merge(beg1, end1, beg2, end2, dest);`

    将两个容器中元素合并，放到第三个容器中。

* `reverse(beg, end);`

* `copy(beg, end, dest)`

* `replace(beg, end, old_val, new_val)`

* `replace_if(beg, end, func_obj, new_val)`

* `swap(container c1, container c2);`

* `accumulate(beg, end, init_val)`

    计算区间总和。

* `fill(beg, end, val)`

* `set_intersection(beg1, end1, beg2, end2, beg3)`

* `set_union(beg1, end1, beg2, end2, beg3)`

* `set_difference(beg1, end1, beg2, end2, beg3)`

    最坏情况是取两容器中较大的`size`。

* `nth_element()`

    最常见的用法：

    ```cpp
    template< class RandomIt >
    void nth_element( RandomIt first, RandomIt nth, RandomIt last );
    ```

    `nth_element()`可以保证第`nth`的数据一定在按序排好的正确的位置上，并且保证`nth`之前的数据一定小于等于`nth`之后的数据。

    example:

    ```cpp
    #include <algorithm>
    #include <iostream>
    using namespace std;

    template<typename T>
    ostream& operator<<(ostream &cout, vector<T> &arr)
    {
        cout << "[";
        for (int i = 0; i < arr.size() - 1; ++i)
        {
            cout << arr[i] << ", ";
        }
        cout << arr.back() << "]";
        return cout;
    }

    int main()
    {
        vector<int> arr(10);
        for (int i = 0; i < 10; ++i)
            arr[i] = rand() % 20;
        cout << arr << endl;
        nth_element(arr.begin(), arr.begin() + 4, arr.end());
        cout << arr << endl;
        return 0;
    }
    ```

    输出：

    ```
    [3, 6, 17, 15, 13, 15, 6, 12, 9, 1]
    [6, 3, 1, 6, 9, 12, 13, 15, 15, 17]
    ```

    可以看到，`arr[4]`的元素的位置是对的，即`9`。在`9`之前的数字都小于 9，在`9`之后的数字都大于 9.

    这个函数通常被用来找中位数。

    如果需要自定义比较函数，可以参考文档：

    Ref: <https://en.cppreference.com/w/cpp/algorithm/nth_element>

**内建函数对象**

`#include <functional>`

```c++
template<class T> T plus<T>
template<class T> T minus<T>
template<class T> T multiplies<T>
template<class T> T divides<T>
template<class T> T modulus<T>
template<class T> T negate<T>
template<class T> bool equal_to<T>
template<class T> bool not_equal_to<T>
template<class T> bool greater<T>
template<class T> bool greater_equal<T>
template<class T> bool less<T>
template<class T> bool less_equal<T>
template<class T> bool logical_and<T>
template<class T> bool logical_not<T>
template<class T> bool logical_or<T>

int main()
{
    negate<int> n;
    cout << n(50) << endl;  // -50

    plus<int> p;
    cout << p(3, 5) << endl;  // 8

    return 0;
}
```

## 字符串相关

* `isalnum(c)`：当`c`是字母或数字时为真

* `isalpha(c)`：当`c`是字母时为真

* `iscntrl(c)`：当`c`是控制字符时为真

* `isdigit(c)`：当`c`是数字时为真

* `isgraph(c)`：当`c`不是空格但可以打印时为真

* `islower(c)`：当`c`是小写字母时为真

* `isprint(c)`：当`c`是可打印字符时为真

* `ispunct(c)`：当`c`是标点符号时为真

* `isspace(c)`：当`c`是空白时为真（空格，横向制表符，纵向制表符，回车符，换行符，进纸符）

* `isupper(c)`：当`c`是大写字母时为真

* `isxdigit(c)`：当`c`是十六进制数字时为真

* `tolower(c)`：当`c`是大写字母，输出对应的小写字母；否则原样输出`c`

* `toupper(c)`：当`c`是小写字母，输出对应折大写字母；否则原样输出`c`

## numeric

* `accumulate`

* 

## algorithm

`#include <algorithm>`

* Reduction

    * `std::all_of`

        Returns true if pred returns true for all the elements in the range `[first,last)` or if the range is empty, and false otherwise.

        ```cpp
        template <class InputIterator, class UnaryPredicate> 
        bool all_of (InputIterator first, InputIterator last, UnaryPredicate pred);
        ```

        Example:

        ```cpp
        // all_of example
        #include <iostream>     // std::cout
        #include <algorithm>    // std::all_of
        #include <array>        // std::array

        int main () {
            std::array<int,8> foo = {3,5,7,11,13,17,19,23};

            if ( std::all_of(foo.begin(), foo.end(), [](int i){return i%2;}) )
                std::cout << "All the elements are odd numbers.\n";

            return 0;
        }
        ```

        注意，这几个函数中，空集都是返回 true 的。

        问题：当传入匿名函数后，`all_of`是如何判断匿名函数的返回值类型是不是`bool`的？

    * `std::any_of`

        ```cpp
        template <class InputIterator, class UnaryPredicate>
        bool any_of (InputIterator first, InputIterator last, UnaryPredicate pred);
        ```

    * `std::none_of`

        Returns true if pred returns false for all the elements in the range [first,last) or if the range is empty, and false otherwise.

        ```cpp
        template <class InputIterator, class UnaryPredicate>
        bool none_of (InputIterator first, InputIterator last, UnaryPredicate pred);
        ```

    * `std::for_each`

        ```cpp
        template <class InputIterator, class Function>
        Function for_each (InputIterator first, InputIterator last, Function fn);
        ```

        Example

        ```cpp
        // for_each example
        #include <iostream>     // std::cout
        #include <algorithm>    // std::for_each
        #include <vector>       // std::vector

        void myfunction (int i) {  // function:
            std::cout << ' ' << i;
        }

        struct myclass {           // function object type:
            void operator() (int i) {std::cout << ' ' << i;}
        } myobject;

        int main () {
            std::vector<int> myvector;
            myvector.push_back(10);
            myvector.push_back(20);
            myvector.push_back(30);

            std::cout << "myvector contains:";
            for_each (myvector.begin(), myvector.end(), myfunction);
            std::cout << '\n';

            // or:
            std::cout << "myvector contains:";
            for_each (myvector.begin(), myvector.end(), myobject);
            std::cout << '\n';

            return 0;
        }
        ```

        `for_each`的用法，第三个参数`Function`可以是普通函数，伪函数，匿名函数，也可以是`function`函数。

* Conditionally find an element

    * `std::find`

        ```cpp
        template <class InputIterator, class T>
        InputIterator find (InputIterator first, InputIterator last, const T& val);
        ```
        
        Example:

        ```cpp
        // find example
        #include <iostream>     // std::cout
        #include <algorithm>    // std::find
        #include <vector>       // std::vector

        int main () {
            // using std::find with array and pointer:
            int myints[] = { 10, 20, 30, 40 };
            int * p;

            p = std::find (myints, myints+4, 30);
            if (p != myints+4)
                std::cout << "Element found in myints: " << *p << '\n';
            else
                std::cout << "Element not found in myints\n";

            // using std::find with vector and iterator:
            std::vector<int> myvector (myints,myints+4);
            std::vector<int>::iterator it;

            it = find (myvector.begin(), myvector.end(), 30);
            if (it != myvector.end())
                std::cout << "Element found in myvector: " << *it << '\n';
            else
                std::cout << "Element not found in myvector\n";

            return 0;
        }
        ```

        从 example 中可以看出，如果给`find`传递的是数组地址，那么`find`会返回 pointer；如果给`find`传递的是 iterator，那么`find`会返回 iterator。这个是怎么实现的呢？

    * `std::find_if`

        ```cpp
        template <class InputIterator, class UnaryPredicate>
        InputIterator find_if (InputIterator first, InputIterator last, UnaryPredicate pred);
        ```

        Example:

        ```cpp
        #include <iostream
        #include <algorithm>    // std::find_if
        #include <vector>

        bool IsOdd (int i) {
            return ((i%2)==1);
        }

        int main () {
            std::vector<int> myvector;

            myvector.push_back(10);
            myvector.push_back(25);
            myvector.push_back(40);
            myvector.push_back(55);

            std::vector<int>::iterator it = std::find_if (myvector.begin(), myvector.end(), IsOdd);
            std::cout << "The first odd value is " << *it << '\n';

            return 0;
        }
        ```

    * `find_if_not`

* Find subsequences

    * `search`

    * `find_end`

    * `search_n`

* Counting

    * `std::count`

    * `std::count_if`

* Other

    * `std::adjacent_find`

    * `std::is_permutation`

    * `std::equal`

## initializer list

如果一个函数的参数形式是`std::initializer_list`类型的对象，那么我们就可以用`{}`的形式将参数传递进去：

```cpp
#include <iostream>
using namespace std;

void func(initializer_list<float> &&il)
{
    cout << "len of inizializer list: " << il.size() << endl;
    auto iter = il.begin();

    for (auto iter = il.begin(); iter != il.end(); ++iter)
    {
        cout << *iter << endl;
    }
}

int main()
{
    func({1, 2, 3});
    return 0;
}
```

输出：

```
len of inizializer list: 3
1
2
3
```

可以看到`initializer_list`是一个模板类，它要求大括号中的元素都是相同的类型。

`initializer_list`无法被继承：

```cpp
#include <iostream>
using namespace std;

template<typename T>
class MyIL: public initializer_list<T>
{

};

void func(MyIL<float> &&il)
{
    cout << "len of inizializer list: " << il.size() << endl;
    auto iter = il.begin();

    for (auto iter = il.begin(); iter != il.end(); ++iter)
    {
        cout << *iter << endl;
    }
}

int main()
{
    func({1, 2, 3});
    return 0;
}
```

上面的代码会出现编译错误：

```
Starting build...
/usr/bin/g++-11 -fdiagnostics-color=always -g *.cpp -o /home/hlc/Documents/Projects/cpp_test/main
main.cpp: In function ‘int main()’:
main.cpp:53:9: error: invalid initialization of reference of type ‘MyIL<float>&&’ from expression of type ‘<brace-enclosed initializer list>’
   53 |     func({1, 2, 3});
      |     ~~~~^~~~~~~~~~~
main.cpp:40:25: note: in passing argument 1 of ‘void func(MyIL<float>&&)’
   40 | void func(MyIL<float> &&il)
      |           ~~~~~~~~~~~~~~^~

Build finished with error(s).
```

可以看到经过继承后，使用初始化列表进行初始化的特性就消失了。说明在编译器的实现中，大括号`{}`和`initializer_list`是强绑定在一起的。

其他有关初始化列表的一些资料：

1. <https://medium.com/@its.me.siddh/modern-c-series-std-initializer-list-why-what-how-184899326a49>

## limits

这个标准库里存着一些整数、浮点数编码相关的信息，比如最大值，最小值，最小间隔，数字位数之类的。

使用时需加上`<limits>`头文件。

```cpp
#include <iostream>
#include <limits>
using namespace std;

int main() {
    cout
    << numeric_limits<int>::min() << ", "
    << numeric_limits<int>::max() << ", "
    << numeric_limits<int>::digits << endl;

    cout
    << numeric_limits<float>::min() << ", "
    << numeric_limits<float>::max() << ", "
    << numeric_limits<float>::infinity() << endl;

    return 0;
}
```

输出：

```
-2147483648, 2147483647, 31
1.17549e-38, 3.40282e+38, inf
```

## miscellaneous

1. `function`

    `#include <functional>`

    Examples:

    ```c++
    #include <functional>
    using namespace std;

    // a function
    int half(int x) {return x / 2;}

    // a function object class
    struct third_t {
        int operator()(int x) {return x/3;}
    };

    // a class with data members
    struct MyValue {
        int value;
        int fifth() {return value/5;}
    };

    int main()
    {
        function<int(int)> fn1 = half;  // function
        function<int(int)> fn2 = &half;  // function pointer
        function<int(int)> fn3 = third_t();  // function object
        function<int(int)> fn4 = [](int x){return x / 4};  // lambda expression
        function<int(int)> fn5 = negate<int>();  // standard function object

        fn1(60);  // 30
        fn2(60);  // 30
        fn3(60);  // 20
        fn4(60);  // 15
        fn5(60);  // -60

        // stuff with members
        function<int(MyValue&)> value = &MyValue::value;  // pointer to data member
        function<int(MyValue&)> fifth = &MyValue::fifth;  // pointer to member function

        MyValue sixty {60};
        value(sixty);  // 60
        fifth(sixty);  // 12

        return 0;
    }
    ```

1. `iota`生成 range

    ```c++
    vector<int> vec(5);
    iota(vec.begin(), vec.end(), 0);  // vec: [0, 1, 2, 3, 4]
    ```

## tuple

`tuple`用于存储不同类型的几个变量。

简单的用法：

```cpp
#include <tuple>  // for std::tuple, std::make_tuple, std::get
#include <iostream>

int main()
{
    std::tuple<int, const char*> t;
    t = std::make_tuple(1, "hello");
    std::cout << std::get<0>(t) << std::endl;
    std::cout << std::get<1>(t) << std::endl;
    return 0;
}
```

output:

```
1
hello
```

使用`tuple`时，需要先`#include <tuple>`头文件。

可以看到，在`t`变量中，先存了一个`int`类型数据，再存了一个指针类型的数据。

我们使用`make_tuple()`来创建 tuple 对象，使用`get()`取得不同索引处的数据。

`get()`其实返回的是一个引用，因此也能拿来赋值：

```cpp
#include <tuple>  // for std::tuple, std::make_tuple, std::get
#include <iostream>
using namespace std;

int main()
{
    tuple<int, const char*> t(2, "hello");  // 使用构造函数初始化对象
    get<1>(t) = "world";  // get() 返回的是一个引用，因此可以对一个函数赋值
    cout << get<1>(t) << endl;
    return 0;
}
```

output:

```
world
```

在上面的代码中，可以看出`get()`返回的是一个引用。

可以使用`tuple_size_v<T>`模板对象的值拿到 tuple 的元素个数：

```cpp
#include <tuple>
#include <iostream>
using namespace std;

int main()
{
    tuple<int, const char*> t(2, "hello");
    cout << tuple_size<tuple<int, const char*>>::value << endl;
    cout << tuple_size<decltype(t)>::value << endl;
    cout << tuple_size_v<decltype(t)> << endl;
    return 0;
}
```

output:

```
2
2
2
```

可以看到，这些实现都是在编译期实现的，都是模板元编程的应用。

`swap()`可以交换两个 tuple 对象的内容。`swap(0`既是 method，又是 function。

可以使用`tie()`和`ignore`将 tuple 中的数据取出来：

```cpp
#include <tuple>
#include <iostream>
using namespace std;

int main()
{
    tuple<int, const char*, float> t(2, "hello", 1.1);
    int ival;
    float fval;
    const char *msg;
    tie(ival, msg, fval) = t;
    cout << ival << endl;
    cout << msg << endl;
    cout << fval << endl;

    tie(ival, ignore, fval) = t;
    tie(ignore, ignore, fval) = t;
    return 0;
}
```

output:

```
2
hello
1.1
```

可以使用`tuple_cat()`将两个 tuple 拼接到一起。目前没怎么用过，所以就不详细写了。

tuple 可以存不同类型的对象，array 只能存同一类型的对象。`pair`只能存两个对象，`tuple`可以存 0 个或多个对象。拿`tuple`存多个对象，由于只能使用索引来获得引用，对象失去了名字，所以过段时间很容易忘记这个`tuple`存的是什么东西。综合看来，`tuple`比较适合临时存一些数据，比如函数的多个返回值。也适合存一些意义非常明确，或者与变量名关系不大的数据，比如三维空间的 xyz 坐标，person 的 id 和姓名等等。

## extent

`std::extent`是一个模板类，使用 traits 技术获得数组在某个维度的长度。

```cpp
#include <iostream>

int main()
{
    std::cout << std::extent_v<int[3][4]> << std::endl;  // 3
    std::cout << std::extent_v<int[3][4], 0> << std::endl;  // 3
    std::cout << std::extent_v<int[3][4], 1> << std::endl;  // 4
    std::cout << std::extent<int[3][4], 1>::value << std::endl;  // 5
    return 0;
}
```

输出：

```
3
3
4
4
```

`std::extent`可以和`decltype`结合起来使用，推测一个数组实例的维度长度：

```cpp
#include <iostream>

int main()
{
    int arr[3][4];
    std::cout << std::extent_v<decltype(arr)> << std::endl; // 3
    using arr_type = decltype(arr);
    std::cout << std::extent_v<arr_type> << std::endl;  // 3
    return 0;
}
```

如果先取了索引，再拿 extent，那么会得到 0，因为索引只是一个 reference。比如`arr[0]`的类型，实际上是`int (&)[3]`。我们可以 remove 掉 reference，从而得到正确的 extent：

```cpp
#include <iostream>

int main()
{
    int arr[3][4];
    std::cout << std::extent_v<decltype(arr[0])> << std::endl;  // 0
    std::cout << std::extent_v<std::remove_reference_t<decltype(arr[0])>> << std::endl;  // 4
    return 0;
}
```

如果将数组传递给函数参数，那么会丢失第一个维度的 extent：

```cpp
#include <iostream>

void print_arr_extent(int arr[3][4])
{
    using t_0= std::remove_reference_t<decltype(arr)>;
    std::cout << std::extent_v<t_0> << std::endl;  // 0
    using t_1 = std::remove_reference_t<decltype(arr[0])>;
    std::cout << std::extent_v<t_1> << std::endl;  // 4
    using t_2 = decltype(arr[0]);
    std::cout << std::extent_v<t_2> << std::endl;  // 0
}

int main()
{
    int arr[3][4];
    print_arr_extent(arr);
    return 0;
}
```

不清楚该怎么解决这个问题。