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

1. `insert()`需要这样写：`insert({key_1, val_1})`。有时间了看下为什么。

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