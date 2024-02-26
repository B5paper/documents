# STL Note

## cached

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
        // find_if example
        #include <iostream>     // std::cout
        #include <algorithm>    // std::find_if
        #include <vector>       // std::vector

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