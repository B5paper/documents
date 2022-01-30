# STL Note

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