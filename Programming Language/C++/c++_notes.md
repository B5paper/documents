# C++ Notes

微软出的 c/c++ tutorial，挺好的，有时间了看看：<https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/fopen-wfopen?view=msvc-170>

## cached

* 在`struct`中，`add_elm()`成员函数应该返回引用还是返回指针？

    ```cpp
    #include <vector>
    using namespace std;

    struct MyElm {
        int val;
    };

    struct MyStruc {
        vector<MyElm*> elms;

        MyElm& add_elm_1() {
            elms.push_back(new MyElm);
            return *elms.back();
        }

        MyElm* add_elm_2() {
            elms.push_back(new MyElm);
            return elms.back();
        }

        ~MyStruc() {
            for (MyElm *elm : elms) {
                delete elm;
            }
        }
    };

    int main() {
        MyStruc struc;

        for (int i = 0; i < 3; ++i) {
            MyElm &elm = struc.add_elm_1();  // ok
            elm.val = i;
        }

        MyElm &elm_1 = struc.add_elm_1();
        elm_1.val = 3;
        // MyElm &elm_1 = struc.add_elm_1();  // error

        MyElm *elm_ptr = nullptr;
        elm_ptr = struc.add_elm_2();
        elm_ptr->val = 4;
        elm_ptr = struc.add_elm_2();  // ok
        elm_ptr->val = 5;

        return 0;
    }
    ```

    如果在循环中，那么反复对引用赋值是可以的。如果在循环外，无法用新引用的定义覆盖旧引用。但是指针没这个问题。整体看来，指针更灵活一些。

* const 全局变量不会有冲突

    `src_1.cpp`:

    ```cpp
    int global_val = 123;
    ```

    `src_2.cpp`:

    ```cpp
    #include <stdio.h>

    int global_val = 456;

    int main() {
        printf("global val: %d\n", global_val);
        return 0;
    }
    ```

    compile:

    `g++ src_1.cpp src_2.cpp -o main`

    compiling output:

    ```
    /usr/bin/ld: /tmp/ccAXjHwL.o:(.data+0x0): multiple definition of `global_val'; /tmp/ccozWBXZ.o:(.data+0x0): first defined here
    collect2: error: ld returned 1 exit status
    ```

    可以看到，如果不同的实现文件里有相同名称的全局变量，那么会编译时报错。

    如果全局变量是 const 的，那么不会有这个问题。

    `src_1.cpp`:

    ```cpp
    const int global_val = 123;
    ```

    `src_2.cpp`:

    ```cpp
    #include <stdio.h>

    int global_val = 456;

    int main() {
        printf("global val: %d\n", global_val);
        return 0;
    }
    ```

    compile:

    `g++ src_1.cpp src_2.cpp -o main`

    run:

    `./main`

    output:

    ```
    global val: 456
    ```

    这是因为 const 全局变量都会被自动添加`private`属性。

* 如果前面定义了`int gpu`，后面不可以使用`TopoNode* gpu`重新定义，编译器会报错。`int gpu`定义在函数参数里也不行。

    ```
    main_5.cpp: In function ‘void func(int)’:
    main_5.cpp:8:12: error: declaration of ‘MyCls* aaa’ shadows a parameter
        8 |     MyCls *aaa = (MyCls*) 0x01;
          |            ^~~
    main_5.cpp:7:15: note: ‘int aaa’ previously declared here
        7 | void func(int aaa) {
          |           ~~~~^~~
    ```

* c++ 中模板类和基类为模板类的模板类，都可以作为聚合类

    ```cpp
    #include <vector>
    #include <stdio.h>
    #include <unordered_map>
    using namespace std;

    template<typename T>
    struct VertexBase {
        int id;
        T type;
    };

    template<typename T, typename P>
    struct Vertex: public VertexBase<T> {
        P msg;
    };

    int main() {
        VertexBase<int> vert_base{1, 2};
        Vertex<int, const char*> vert{1, 2, "haha"};
        printf("id: %d, type: %d, msg: %s\n", vert.id, vert.type, vert.msg);
        
        vector<VertexBase<int>> base_verts;
        // base_verts.emplace_back(3, 4);  // error
        vector<Vertex<int, const char*>> verts;
        // verts.emplace_back(1, 2, "hello");  // error
        return 0;
    }
    ```

    但是`emplace()`和`emplace_back()`，需要类必须自定义构造函数来支持。

    如果调用`base_verts.push_back({3, 4});`，那么本质是先用`{3, 4}`聚合构造（Aggregate Initialization）了一个对象，然后调用`push_back()`的重载版本，此时如果`VertexBase`有移动构造函数，那么调用`push_back(VertexBase&&)`，并调用`VertexBase`的移动构造函数，如果没有，那么调用`push_back(const VertexBase&)`，并调用`VertexBase`的拷贝构造函数。

* `sizeof(void)`

    `sizeof(void)`本身无意义，gcc/g++ 可以通过编译，输出为`1`，但是会报 warning。

    example:

    ```cpp
    #include <stdio.h>

    int main() {
        size_t len_void = sizeof(void);
        printf("len_void: %lu\n", len_void);
        return 0;
    }
    ```

    compile output:

    ```
    main_5.cpp: In function ‘int main()’:
    main_5.cpp:4:23: warning: invalid application of ‘sizeof’ to a void type [-Wpointer-arith]
        4 |     size_t len_void = sizeof(void);
          |                       ^~~~~~~~~~~~
    ```

    output:

    ```
    len_void: 1
    ```

* c++ `vector`调用`resize()`时，会保留尽可能多的已有元素，仅增加/删除需要改动的部分。

    example:

    ```cpp
    #include <vector>
    #include <stdio.h>
    using namespace std;

    void print_vec(vector<int> &vec) {
        for (int i = 0; i < vec.size(); ++i) {
            printf("%d, ", vec[i]);
        }
        putchar('\n');
    }

    int main() {
        vector<int> vec{1, 2, 3};
        vec.resize(10);
        print_vec(vec);
        vec[3] = 4;
        vec[4] = 5;
        vec.resize(2);
        print_vec(vec);
        return 0;
    }
    ```

    可以看到，扩容时，仅在`vec`末尾补充 0，前面的`1, 2, 3`并未被修改。而在缩容时，`1, 2`也得到了保留。

    如果在调用`resize()`时提供了`val`，那么使用`val`对新增元素进行初始化。

    调用`resize()`时，如果申请了新的内存，将旧的元素的内容移动到新的内存上去，那么会调用移动构造函数或拷贝构造函数，旧的元素会调用析构函数进行销毁。具体的调用规则如下：

    * 如果元素类型具有 noexcept 的移动构造函数，vector 会优先调用 移动构造函数（高效，避免不必要的拷贝）。

    * 如果移动构造函数可能抛出异常（非 noexcept），vector 会保守地调用 拷贝构造函数（保证强异常安全性）。

    * 如果元素类型不可移动（仅支持拷贝），强制使用 拷贝构造函数。

* 关于 c++ 中数组名的赋值问题

    如果数组名是函数参数，那么在函数内部，数组名可以被赋值为新的指针，如果数组名在函数外部，那么在函数内部无法修改数组名所指的内容：

    ```cpp
    #include <stdio.h>
    #include <stdlib.h>

    void func(int arr[]) {
        int *new_arr = (int*) malloc(sizeof(int) * 3);
        new_arr[0] = 3;
        new_arr[1] = 2;
        new_arr[2] = 1;
        arr = new_arr;
        printf("%d, %d, %d\n", arr[0], arr[1], arr[2]);
        free(new_arr);
    }

    void func_2(int *arr[]) {
        int *new_arr = (int*) malloc(sizeof(int) * 3);
        new_arr[0] = 3;
        new_arr[1] = 2;
        new_arr[2] = 1;
        *arr = new_arr;
        printf("%d, %d, %d\n", (*arr)[0], (*arr)[1], (*arr)[2]);
        // free(new_arr);
    }

    int main() {
        int arr[3] = {1, 2, 3};
        func(arr);
        printf("arr: %p, %d, %d, %d\n", arr, arr[0], arr[1], arr[2]);
        func_2((int**) &arr);
        printf("arr: %p, %d, %d, %d\n", arr, arr[0], arr[1], arr[2]);
        return 0;
    }
    ```

    output:

    ```
    3, 2, 1
    arr: 0x7fffc728817c, 1, 2, 3
    3, 2, 1
    arr: 0x7fffc728817c, -1680833888, 22371, 3
    ```

    在`func()`内部，`arr`的含义被成功改变，而在`func_2()`调用过后，`arr`的值仍保持和原来一样。

* c++ 中聚合类与初始化

    如果一个类定义了构造函数，那么就无法自动初始化。

    example:

    ```cpp
    #include <string>
    using namespace std;

    struct A {
        string a;
        int b;
    };

    int main() {
        A obj_1({"hello", 123});
        A obj_2{"hello", 123};
        return 0;
    }
    ```

    上面是个聚合类，两种方式都可以正常初始化。

    ```cpp
    #include <string>
    using namespace std;

    struct A {
        string a;
        int b;

        A() {}
    };

    int main() {
        A obj_1({"hello", 123});  // error
        A obj_2{"hello", 123};  // error
        return 0;
    }
    ```

    由于没有对应的构造函数，上面的两种方式都会编译失败。

    ```cpp
    #include <string>
    using namespace std;

    struct A {
        string a;
        int b;

        A() {}
        A(string aa, int bb): a(aa), b(bb) {}
    };

    int main() {
        A obj_1({"hello", 123});
        A obj_2{"hello", 123};
        return 0;
    }
    ```

    上面补充了对应的构造函数，所以可以通过编译。

    `A obj_1({"hello", 123});`的含义应该是先用`{}`去生成一个匿名对象，再把这个匿名对象通过移动构造函数传给`obj_1`。所以本质还是聚合类的初始化，并不是`{}`作为 initializer list 可以存放不同类型的元素。

* c++ 中，如果一个构造函数被声明为`explicit`，那么就无法使用等号进行初始化，只能使用括号来初始化。

    ```cpp
    struct B;

    struct A {
        int val;
        A() {}
        explicit A(const B &b);
    };

    struct B {
        int val;
    };

    A::A(const B &b) {
        val = b.val;
    }

    int main() {
        A a;
        a.val = 123;
        B b;
        b.val = 456;

        A c = b;  // Error
        A d(b);  // OK

        return 0;
    }
    ```

    compiling output:

    ```
    main_5.cpp: In function ‘int main()’:
    main_5.cpp:23:11: error: conversion from ‘B’ to non-scalar type ‘A’ requested
       23 |     A c = b;  // Error
          |           ^
    make: *** [Makefile:4: main] Error 1
    ```

    如果不写`explicit`，则可正常通过编译：

    ```cpp
    struct B;

    struct A {
        int val;
        A() {}
        A(const B &b);
    };

    struct B {
        int val;
    };

    A::A(const B &b) {
        val = b.val;
    }

    int main() {
        A a;
        a.val = 123;
        B b;
        b.val = 456;

        A c = b;  // OK
        A d(b);  // OK

        return 0;
    }
    ```

* `const char*` / `char*`不能转换为`string&`，只能转换为`const string&`或`string&&`。

* 使用`typename`告诉编译器当前的名字是个类型名

    example:

    如果直接编译下面的代码，是编译不通的：

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <unordered_map>
    using namespace std;

    struct A {
        typedef unordered_map<string, string> Dict;
        Dict dict;
    };

    template<typename T>
    struct B {
        T::Dict dict_b;
    };

    int main() {
        B<A> b;
        return 0;
    }
    ```

    编译输出：

    ```
    main_5.cpp:14:5: error: need ‘typename’ before ‘T::Dict’ because ‘T’ is a dependent scope
       14 |     T::Dict dict_b;
          |     ^
          |     typename 
    make: *** [Makefile:2: main] Error 1
    ```

    此时我们必须在`T::Dict dict_b;`前加一个`typename`，才能编译通过：

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <unordered_map>
    using namespace std;

    struct A {
        typedef unordered_map<string, string> Dict;
        Dict dict;
    };

    template<typename T>
    struct B {
        typename T::Dict dict_b;
    };

    int main() {
        B<A> b;
        return 0;
    }
    ```

    默认情况下，编译器认为依赖未知类型`T`的符号都是普通成员，而不是类型名。如果想告诉编译器这个符号是个类型，而不是成员变量或函数，那么必须使用`typename`。

    下面这篇文章讲得很清楚：

    <https://www.ibm.com/docs/en/zos/2.4.0?topic=only-typename-keyword-c>

* 必须使用 unique ptr 的场景

    按传统的方式需要调用两次函数，第一次得到 num，然后用户 malloc 内存拿到 buffer，第二次调用函数往 buffer 里填数据。

    如果在这里用 unique ptr，那么只需要调用一次函数就可以了，申请的内存也会自动释放。

* c++ 不允许对指针重载`==`，否则会造成指针的比较定义混淆

    ```cpp
    // OK
    bool operator==(XmlTag &tag_1, const XmlTag &tag_2) {
        //　...
    }

    // Error
    bool operator==(const XmlTag *tag_1, const XmlTag *tag_2) {
        //　...
    }
    ```

    那么如果在`vector`里存指针，比如`vector<XmlTag*>`，该如何使用`std::find()`呢？

    2025/07/04/00: 或许可以使用`std::find_if()`，`find_if()`接收一个 lambda 表达式，这个匿名函数的参数可以为指针。

* 由于`initializer_list`必须要指定模板类型，所以多个`initializer_list`是非常有必须的。

    ```cpp
    #include <stdio.h>
    #include <vector>
    #include <string>
    #include <iostream>
    using namespace std;


    struct MyObj {
        vector<int> ints;
        vector<string> strs;
        MyObj(initializer_list<int> &&init_list_1,
            initializer_list<string> &&init_list_2) {
            for (int val : init_list_1) {
                ints.push_back(val);
            }
            for (const string &str : init_list_2) {
                strs.push_back(str);
            }
        }
    };

    int main()
    {
        MyObj obj {
            {1, 2, 3},
            {"hello", "world"},
        };
        for (int val : obj.ints) {
            printf("%d, ", val);
        }
        putchar('\n');
        for (string &str : obj.strs) {
            printf("%s, ", str.c_str());
        }
        putchar('\n');
        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 
    hello, world, 
    a: nihao
    val: 456
    ```

* `emplace()`和`emplace_back()`

    `emplace()`有点像`insert()`，需要指定要插入元素的位置的迭代器。`emplace_back()`则是在容器的末尾添加。

    ```cpp
    #include <stdio.h>
    #include <vector>
    #include <string>
    using namespace std;

    struct MyObj {
        string msg;
        int val;
        MyObj(const string& m, int v) : msg(m), val(v) {}
    };

    int main() {
        vector<MyObj> objs;
        objs.emplace_back("hello", 123);
        objs.emplace_back("world", 456);
        objs.emplace(objs.begin() + 1, "nihao", 789);
        for (int i = 0; i < objs.size(); ++i) {
            printf("msg: %s, val: %d\n", objs[i].msg.c_str(), objs[i].val);
        }
        return 0;
    }
    ```

    output:

    ```
    msg: hello, val: 123
    msg: nihao, val: 789
    msg: world, val: 456
    ```

    `emplace()`在构造元素时，需要调用构造函数，无法使用 c++ 的聚合构造功能。

* c++ 中的结构体初始化

    ```cpp
    #include <stdio.h>
    #include <vector>
    #include <string>
    #include <iostream>
    using namespace std;

    struct AAA {
        string msg;
        int val;
    };

    struct MyObj {
        vector<int> ints;
        vector<string> strs;
        AAA a;
    };

    int main()
    {
        MyObj obj {
            {1, 2, 3},
            {"hello", "world"},
            "nihao", 456  // {"nihao", 456}
        };
        for (int val : obj.ints) {
            printf("%d, ", val);
        }
        putchar('\n');
        for (string &str : obj.strs) {
            printf("%s, ", str.c_str());
        }
        putchar('\n');
        printf("a: %s\n", obj.a.msg.c_str());
        printf("val: %d\n", obj.a.val);
        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 
    hello, world, 
    a: nihao
    val: 456
    ```

    在没有自定义构造函数的情况下，仅需使用内嵌的大括号消除歧义，使可直接用大括号来初始化一个对象。内部的大括号也并不是必须的，`"nihao", 456`的写法和`{"nihao", 456}`的写法都是正确的。

    目前并不清楚其原理。

* c++ 中指向数组的引用

    指向数组的引用可以保留数组的长度信息。

    example:

    ```cpp
    #include <stdio.h>
    using namespace std;

    int main() {
        int arr[] = {1, 2, 3, 4, 5};

        int (&arr_r)[] = arr;
        printf("arr_r[2]: %d\n", arr_r[2]);

        // printf("sizeof(arr_r): %lu\n", sizeof(arr_r));  // error

        int (&arr_r_2)[5] = arr;
        printf("sizeof(arr_r_2): %lu\n", sizeof(arr_r_2));

        int (&arr_r_3)[sizeof(arr) / sizeof(int)] = arr;
        printf("sizeof(arr_r_3): %lu\n", sizeof(arr_r_3));

        return 0;
    }
    ```

    output:

    ```
    arr_r[2]: 3
    sizeof(arr_r_2): 20
    sizeof(arr_r_3): 20
    ```

    如果在初始化数组引用时，没有指定元素个数，那么在调用`sizeof(arr_r)`时会编译报错。

    元素个数既可以手动指定，也可以使用`sizeof()`计算出来。如果指定的元素个数与原数组不相同，也会编译时报错。

* 在函数参数中处理指向数组的引用

    需要用类似`int (&arr_r)[5]`的方式传递参数，元素个数必须要填，否则会报错。

    ```cpp
    #include <stdio.h>
    using namespace std;

    void print_arr(int (&arr_r)[5]) {
        int N = sizeof(arr_r) / sizeof(int);
        for (int i = 0; i < N; ++i) {
            printf("%d, ", arr_r[i]);
        }
        putchar('\n');

        for (int val : arr_r) {
            printf("%d, ", val);
        }
        putchar('\n');
    }

    int main() {
        int arr[] = {1, 2, 3, 4, 5};
        print_arr(arr);
        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5, 
    ```

    如果在模板中使用则更方便，不需要手动指定元素个数，元素个数可以自动推导出来：

    ```cpp
    #include <stdio.h>
    using namespace std;

    template<typename T, size_t N>
    void print_arr(T (&arr_r)[N]) {
        for (int i = 0; i < N; ++i) {
            printf("%d, ", arr_r[i]);
        }
        putchar('\n');

        for (int val : arr_r) {
            printf("%d, ", val);
        }
        putchar('\n');
    }

    int main() {
        int arr[] = {1, 2, 3, 4, 5};
        print_arr(arr);
        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5,
    ```

* 16 进制字符串解析

    ```cpp
    #include <stdio.h>
    #include <iostream>
    #include <stdlib.h>  // strtol()
    #include <string>  // std::stoi()
    using namespace std;

    int main()
    {
        const char *str = "0x1234";

        size_t idx;
        int val_1 = std::stoi(str, &idx, 16);
        printf("val_1: %d, idx: %lu\n", val_1, idx);

        char *end_ptr;
        int val_2 = strtol(str, &end_ptr, 16);
        printf("val: %d, end ch: %c\n", val_2, *end_ptr);

        const char *str_2 = "1234";

        val_1 = std::stoi(str_2, &idx, 16);
        printf("val_1: %d, idx: %lu\n", val_1, idx);

        val_2 = strtol(str_2, &end_ptr, 16);
        printf("val: %d, end ch: %c\n", val_2, *end_ptr);

        return 0;
    }
    ```

    output:

    ```
    val_1: 4660, idx: 6
    val: 4660, end ch: 
    val_1: 4660, idx: 4
    val: 4660, end ch:
    ```

    `std::stoi()`和`strtol()`都可以正常解析带`0x`或不带`0x`的十六进制字符串。`x`对大小写不敏感，也可以写成`0X`。

    `stoi()`第 2 个参数返回解析结束时的索引，相当于`end_ptr`的作用。

* `stoi()`与`strtol()`的区别

    `stoi()`是一个作用于`const string&`的 c++ 函数，在`<string>`头文件中。
    
    `strol()`是一个作用于 c style string 的 C 函数，在`<stdlib.h>`头文件中。

    example:

    ```cpp
    #include <stdio.h>
    #include <iostream>
    #include <stdlib.h>  // strtol()
    #include <string>  // std::stoi()
    using namespace std;

    int main()
    {
        const char *str = "123, 456";

        size_t idx;
        int val_1 = std::stoi(str, &idx, 10);
        printf("val_1: %d, idx: %lu\n", val_1, idx);

        char *end_ptr;
        int val_2 = strtol(str, &end_ptr, 10);
        printf("val_2: %d, end ch: %c\n", val_2, *end_ptr);

        return 0;
    }
    ```

    output:

    ```
    val_1: 123, idx: 3
    val_2: 123, end ch: ,
    ```

    两个函数都会从字符串起始位置开始尝试解析，直到遇到无效字符为止，返回遇到的第一个无效字符的位置。

    如果一个有效的数字都解析不出来，那么`std::stoi()`会报 exception 退出程序。`strtol()`会返回 0，但并不会报错，用户只能通过判断`end_ptr`与 start ptr 是否相等来判断是否正常解析。

* 在 struct 内初始化的变量，既可以是 const 的，也可以是非 const 的，但是不能是 static 的。

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <iostream>
    using namespace std;

    struct MyType {
        string msg = "hello";
        const int val = 456;
    };

    int main()
    {
        MyType a;
        cout << a.msg << endl;
        cout << a.val << endl;
        return 0;
    }
    ```

    output:

    ```
    hello
    456
    ```

* c++ 中`override`表示此函数为覆盖基类的虚函数，编译器会检查此函数是否确实覆盖了基类虚函数，如果没有，则报错。`final`则表示此函数不应再被派生类的虚函数覆盖。

* copy constructor 起作用的时机

    如果我们希望把 src_obj 的数据交给 dst_obj 管理，并且释放 src_obj，比如`dst_obj.add_child_obj(src_obj);`，那么有两种方案，一种是

    ```cpp
    dst_obj.add_child_obj(new Obj(src_obj));
    // 此时可以放心 delete src_obj;
    ```

    另一种是

    ```cpp
    src_obj->parent = dst_obj;
    dst_obj.child_objs.push_back(src_obj);
    parent_of_src_obj.child_objs.erase(parent_of_src_obj.child_objs.begin() + src_obj_idx);
    ```

    上面代码中，`Obj(src_obj)`实际上是调用了 copy constructor，创建了一个新的实体，并释放了旧实体。

    第二种方案是只移动了指针，并没有创建一个新的实体。

* 树形结构的递归释放

    如果一个 obj 里面管理有 child objs，那么 obj 在析构时不需要递归释放，只需要释放自己管理的 child objs 就可以了，程序会自动递归析构，完成递归释放的功能。

    ```cpp
    #include <iostream>
    #include <vector>
    using namespace std;

    struct Obj {
        int id;
        vector<Obj*> child_objs;

        Obj* add_child_obj(int id) {
            Obj *new_obj = new Obj;
            new_obj->id = id;
            child_objs.push_back(new_obj);
            return new_obj;
        }

        ~Obj() {
            for (Obj *child_obj : child_objs) {
                delete child_obj;
            }
        }
    };

    int main() {
        Obj root_obj;
        root_obj.add_child_obj(0);
        Obj *child_obj_1 = root_obj.add_child_obj(1);
        child_obj_1->add_child_obj(0);
        child_obj_1->add_child_obj(1);
        Obj *child_obj_2 = child_obj_1->add_child_obj(2);
        child_obj_2->add_child_obj(0);
        child_obj_2->add_child_obj(1);
        child_obj_1->add_child_obj(2);
        return 0;
    }
    ```

    可以看到，`～Obj()`并不是递归的。我们不需要考虑那么多，只负责自己这一部分就可以了。

* c++ 中的 raw 字符串

    c++ 可以使用`R"()"`设置 raw 字符串，字符串中的所有内容都不做解析，空白、换行和制表都不会被忽略。

    ```cpp
    #include <iostream>
    #include <string>
    using namespace std;

    int main() {
        string str = R"(aaaa
        hello,
        world
        \t
        )";
        cout << str << endl;
        return 0;
    }
    ```

    output:

    ```
    aaaa
        hello,
        world
        \t
    
    ```

* 如果设计了两个独立的 class，又需要两个 class 合作实现某个功能，那么目前先额外写一个全局函数来解决

    ```cpp
    struct A {

    };

    struct B {

    };

    int my_func(A &a, B &b) {
        // ...
    }
    ```

    每个 class 尽量只查询、修改自己维护的数据，不要访问其他 class 的数据。比如尽量不法要这样写：

    ```cpp
    struct B {
        void mem_func(A &a) {
            // ....
        }
    };
    ```

* 必须使用 malloc / new 的场景：vert and edge，如果删除 vert，idx 发生改变，那么所有的 edge 都需要重新处理。只有 malloc / new 能保证地址不变。

    其实本质相当于一次解耦，如果使用 idx 1 -> idx 2 的方式，idx 1 是连续的，对外暴露，idx 2 是离散的（或者总是按顺序增大），作为一个 id 总是不变，那么在删除时，就可以保证删除一个 idx 2，其他的 idx 2 不变。

* 大端与小端

    大端：数据的高位，保存在内存的低地址中。

    小端：数据的低位，保存在内存的低地址中。

    大端与小端的判断：通常有两种方式，一种是将 int 或 short 数据的指针强制转换成 char 指针，再判断 char 指针的数据对应 int/short 的高位数据还是低位。另一种方式是使用 union 判断，原理是一样的。

    通常网络使用的数据使用大端存储，内存中的数据使用小端存储。
    
    内存中使用小端存储有个很好的优势：

    ```cpp
    int a = 1;  // 4 bytes
    short *pb = &a;  // 2 bytes, 1
    short b = *pb;  // 1
    ```

    通过截断做类型转换非常方便，用不到高位时，高位会被自动舍弃，数值保持不变。使用大端就做不到这一点。

* c++ and move

    `move()`的作用是把左值引用转换成右值引用，包含在头文件`<utility>`中，其实现如下：

    ```cpp
    // (since C++11) (until C++14)
    template< class T >
    typename std::remove_reference<T>::type&& move(T&& t) noexcept;

    // (since C++14)
    template< class T >
    constexpr std::remove_reference_t<T>&& move(T&& t) noexcept;
    ```

    通常情况下，如果我们临时构造出来一个对象，并将这个对象用于初始化容器中的一个元素时，会产生不必要的复制，比如`vec.push_back("hello, world");`, `vec.push_back({123, "hello"})`，另一个常见的例子是函数的返回值`vector<string> ret = get_strs();`。由于这些匿名对象在创建后很快会被释放，编译器会帮忙优化这个操作，在匿名对象创建完成后，直接将匿名对象的内存（指针）赋值给容器中的元素，这样就避免了容器中的元素又申请赋值一遍内存。

    这些优化都是安全的，用户也很少能感知到。但是如果用户希望很明确地告诉编译器，自己创建的一个变量/对象后面用不到了，希望把这个对象的值快速赋给另一个对象，就需要用到`std::move()`函数了。由于用户创建的对象是一个左值，所以 c++ 库函数感知不到用户的意思，此时用户使用`move()`会将其类型转换为一个右值引用，c++ 库函数就知道用户不想保留这个值了。

    example:

    ```cpp
    #include <vector>
    #include <utility>
    #include <string>
    #include <stdio.h>
    using namespace std;

    void my_move_str_func(string &&str)
    {
        printf("in my_move_str_func, str: %s\n", str.c_str());
    }

    int main(void)
    {
        vector<string> strs;
        string str = "hello, world";
        // my_move_str_func(str);  // compile error
        my_move_str_func(move(str));
        printf("after my_move_str_func, str: %s\n", str.c_str());
        strs.push_back(move(str));
        printf("after vector push back right value ref, str: %s\n", str.c_str());
        return 0;
    }
    ```

    output:

    ```
    in my_move_str_func, str: hello, world
    after my_move_str_func, str: hello, world
    after vector push back right value ref, str:
    ```

    可以看到，用户的函数`my_move_str_func()`接收一个右值引用，但是函数内并没有操作`str`的内存，因此`main()`函数中，`str`正常输出。而`strs.push_back()`函数接收右值引用后，内部操作了`str`的内存，再回到`main()`函数，`str`已经无法正常显示了。此时`str`变成一个无效变量。

    说明：

    * 函数参数为右值引用时，实参只接收右值引用，不接收 const 左值引用。函数的参数为`const`左值引用时，既接收左值，又接收右值引用。

    * 如果同时有 const 左值引用和右值引用作为参数的函数，那么优先调用右值引用版本。

    * 左值和右值可以互相转换

        一个右值作为参数传给函数后，它在函数内部就变成了左值，因为函数参数为它赋予了名字。

        example:

        ```cpp
        #include <utility>
        #include <string>
        #include <stdio.h>
        using namespace std;

        void left_or_right_ref_test(string &str)
        {
            printf("str is left reference\n");
        }

        void left_or_right_ref_test(string &&str)
        {
            printf("str is right reference\n");
        }

        void my_move_str_func(string &&str)
        {
            printf("in my_move_str_func, str: %s\n", str.c_str());
            left_or_right_ref_test(str);
        }

        int main(void)
        {
            string str = "hello, world";
            my_move_str_func(move(str));
            return 0;
        }
        ```

        output:

        ```
        in my_move_str_func, str: hello, world
        str is left reference
        ```

* 如果一个全局变量是 const 变量，那么它默认是 static 的。此时如果直接在其他文件里`exteran const GlobalVarType global_val;`引用到这个全局变量，编译器会报错变量未定义。

    解决方案是在给全局变量定义时，加上`extern`关键字。

    example:

    ```cpp
    extern const GlobalVar<string> global_var {
        "hello, world"
    };
    ```

    这样就可以在别的文件里，使用`extern const GlobalVar<string> global_var;`引用到这个变量了。

* c++ 使用别的文件中的全局变量

    比如在`global_var.h`中声明了个结构体`struct GlobalVar {};`，在`global_var.cpp`中想定义个全局变量，让`user_1.cpp`，`user_2.cpp`都用到这个全局变量，那么可以这样写：

    `global_var.h`:

    ```cpp
    #ifndef GLOBAL_VAR_H
    #define GLOBAL_VAR_H

    struct GlobalVar {
        int val_int;
        const char *val_str;
    };

    #endif
    ```

    `global_var.cpp`:

    ```cpp
    #include "global_var.h"

    GlobalVar global_var {
        123,
        "hello, world"
    };
    ```

    `user_1.h`:

    ```cpp
    #ifndef USER_1_H
    #define USER_1_H

    int user_1_print_global_var_int();

    #endif
    ```

    `user_1.cpp`:

    ```cpp
    #include <stdio.h>
    #include "user_1.h"
    #include "global_var.h"

    extern GlobalVar global_var;

    int user_1_print_global_var_int() {
        printf("in user_1, global var int: %d\n", global_var.val_int);
        return 0;
    }
    ```

    `user_2.h`:

    ```cpp
    #ifndef USER_2_H
    #define USER_2_H

    int user_2_print_global_var_str();

    #endif
    ```

    `user_2.cpp`:

    ```cpp
    #include <stdio.h>
    #include "user_2.h"
    #include "global_var.h"

    extern GlobalVar global_var;

    int user_2_print_global_var_str() {
        printf("in user_2, global val str: %s\n", global_var.val_str);
        return 0;
    }
    ```

    `main.cpp`:

    ```cpp
    #include <stdio.h>
    #include "user_1.h"
    #include "user_2.h"
    #include "global_var.h"

    extern GlobalVar global_var;

    int main() {
        user_1_print_global_var_int();
        user_2_print_global_var_str();

        printf("in main(), global var:\n");
        printf("    val int: %d\n", global_var.val_int);
        printf("    val str: %s\n", global_var.val_str);

        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.cpp user_1.o user_2.o global_var.o
    	g++ -g main.cpp user_1.o user_2.o global_var.o -o main

    global_var.o: global_var.h global_var.cpp
    	g++ -g -c global_var.cpp -o global_var.o

    user_1.o: user_1.h user_1.cpp
    	g++ -g -c user_1.cpp -o user_1.o

    user_2.o: user_2.h user_2.cpp
    	g++ -g -c user_2.cpp -o user_2.o

    clean:
    	rm -f main *.o
    ```

    output:

    ```
    in user_1, global var int: 123
    in user_2, global val str: hello, world
    in main(), global var:
        val int: 123
        val str: hello, world
    ```

    其中，`extern`表示这个变量的定义不在当前`.cpp`文件中，而在其他`.cpp`或`.o`文件里。
    
    注意在 makefile 中，编译`user_1.o`和`user_2.o`时并没有用到`global_var.o`，在最后生成`main`，也就是 link 环节才会用到`global_var.o`。

* 如果两个 struct 定义在不同的文件里，那么不可能在两个 struct 中互相包含对方的成员实体

    example:

    `header_1.h`:

    ```cpp
    #ifndef HEADER_1_H
    #define HEADER_1_H

    #include "header_2.h"

    struct A {
        int val_1;
        float val_2;
    };

    #endif
    ```

    `header_2.h`:

    ```cpp
    #ifndef HEADER_2_H
    #define HEADER_2_H

    #include "header_1.h"

    // struct A;

    struct B {
        int val_1;
        double val_2;
        A obj_a;
    };

    #endif
    ```

    `impl_1.cpp`:

    ```cpp
    #include "header_1.h"
    ```

    `impl_2.cpp`:

    ```cpp
    #include "header_2.h"
    ```

    `Makefile`:

    ```makefile
    all: impl_1.o impl_2.o

    impl_1.o: header_1.h impl_1.cpp
    	g++ -g -c impl_1.cpp -o impl_1.o

    impl_2.o: header_2.h impl_2.cpp
    	g++ -g -c impl_2.cpp -o impl_2.o
    ```

    run:

    `make`

    output:

    ```
    g++ -g -c impl_1.cpp -o impl_1.o
    In file included from header_1.h:4,
                     from impl_1.cpp:1:
    header_2.h:11:5: error: ‘A’ does not name a type
       11 |     A obj_a;
          |     ^
    make: *** [Makefile:4: impl_1.o] Error 1
    ```

    报错的过程如下：

    1. 根据 makefile 中的内容，先编译`impl_1.o`，此时会打开`impl_1.cpp`，读取`header_1.h`的内容

    2. 在`header_1.h`中，当执行到`#include "header_2.h"`时，跳转去读`header_2.h`的内容

    3. 在`header_2.h`中，又遇到了读取`header_1.h`的内容，但是由于有多重包含防范，所以不再读`header_1.h`的内容。继续往后走，`struct B`中定义了`A obj_a;`，编译器不知道这个`A`从哪来的，所以直接报错了。

    这种问题被称为结构体相互依赖（Mutual Dependency），通常解决方案是使用指针代替实体：

    `header_2.h`:

    ```cpp
    #ifndef HEADER_2_H
    #define HEADER_2_H

    #include "header_1.h"

    struct A;

    struct B {
        int val_1;
        double val_2;
        A *obj_a;
    };

    #endif
    ```

    此时即可通过编译。

    我们复盘下编译器的解析流程：

    1. 根据 makefile 中的内容，先编译`impl_1.o`，此时会打开`impl_1.cpp`，读取`header_1.h`的内容

    2. 在`header_1.h`中，当执行到`#include "header_2.h"`时，跳转去读`header_2.h`的内容

    3. `header_2.h`又包含了`header_1.h`，此时不再读取`header_1.h`的内容，继续往下走，`struct A;`告诉编译器`A`是一个`struct`，继续往下走，`A *obj_a;`表示`B`中有一个`A`的指针，`B`不需要知道`sizeof(A)`，只需要知道`A*`占 8 个字节就可以了。对`B`的解析至此结束，通过编译。

    `header_2.h`中声明`A`的行为`struct A;`叫前向声明（Forward Declaration）。

* 如果两个 struct 互相依赖，那么即使把它们放到同一个文件里，也无法通过编译

    `header_1.h`:

    ```cpp
    #ifndef HEADER_1_H
    #define HEADER_1_H

    struct B;

    struct A {
        int val_1;
        float val_2;
        B obj_b;
    };

    struct B {
        int val_1;
        double val_2;
        A obj_a;
    };

    #endif
    ```

    `impl_1.cpp`:

    ```cpp
    #include "header_1.h"
    ```

    compile:

    ```bash
    g++ -g -c impl_1.cpp -o impl_1.o
    ```

    compile output:

    ```
    In file included from impl_1.cpp:1:
    header_1.h:11:7: error: field ‘obj_b’ has incomplete type ‘B’
       11 |     B obj_b;
          |       ^~~~~
    header_1.h:4:8: note: forward declaration of ‘struct B’
        4 | struct B;
          |        ^
    make: *** [Makefile:4: impl_1.o] Error 1
    ```

* `decltype` and function

    ```cpp
    using FuncType = int(int&, int);
    int add_to(int &des, int ori);
    FuncType *pf = add_to;
    int a = 4;
    pf(a, 2);

    decltype(add_to) *pf = add_to;
    ```

* `decltype`不会实际计算表达式的值，编译器分析表达式并得到它的类型

    函数调用也算一种表达式，因此不必担心在使用`decltype`时真正执行了函数。

    `decltype`加数组，不负责把数组转换成指针，所以其结果仍是数组。

    ```cpp
    int i = 42, *p = &i, &r = i;
    decltype(*p) c = i;  // *p 是左值，c is a int&
    decltype(r + 0) b;  // r + 0 是右值，b is a int
    ```

    `decltype(expr)`, 如果`expr`返回左值，那么`decltype`返回该类型的左值引用；如果`expr`返回右值，那么`decltype`返回表达式结果本来的类型。

    ```cpp
    int i = 42;
    decltype((i)) ri = i;  // ri is a int&
    int *p = &i;
    decltype((p)) temp = p;  // temp is a int* &
    ```

* c++ 左值与右值

    左值是指那些在表达式执行结束后依然存在的数据，也就是持久性的数据；右值是指那些在表达式执行结束后不再存在的数据，也就是临时性的数据。有一种很简单的方法来区分左值和右值，对表达式取地址，如果编译器不报错就为左值，否则为右值。

* initializer list 本质是右值

    ```cpp
    #include <unordered_map>
    #include <string>
    #include <utility>
    #include <cstdio>
    using namespace std;

    enum Color {
        RED,
        BLUE,
        GREEN,
        YELLOW
    };

    struct LookupTable {
        unordered_map<Color, string> lut;
        // 这里必须加 const，因为 main() 中给出的列表是右值，
        // 我们必须使用 const 左值，或者直接使用右值引用来接收它
        explicit LookupTable(const initializer_list<
            pair<Color, string>> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                lut.emplace(iter->first, iter->second);
            }
        }
        const string& operator[](const Color &color) const {
            return lut.at(color);
        }
    };

    LookupTable Color_Id_To_String {
        {RED, "red"},
        {BLUE, "blue"},
        {GREEN, "green"},
        {YELLOW, "yellow"}
    };

    int main() {
        Color color = GREEN;
        printf("color is %s\n", Color_Id_To_String[color].c_str());
        return 0;
    }

    ```

    output:

    ```
    color is green
    ```

* 如果一个模板类的基类仍是模板类，那么必须使用`this`指针才能访问到基类中的成员

    ```cpp
    #include <cstdio>
    using namespace std;

    template<typename T>
    struct BaseClass {
        T val;
        void print_msg() {
            printf("hello from base class, val: %d\n", val);
        }
    };

    template<typename T>
    struct MyClass: public BaseClass<T> {
        void invoke_func_from_base_class(T input_val) {
            // val = input_val;  // error
            // print_msg();  // error
            this->val = input_val;
            this->print_msg();
        }
    };

    int main() {
        MyClass<int> obj;
        obj.invoke_func_from_base_class(123);
        return 0;
    }
    ```

    如果基类的类型在编译时期就已经确定，那么可以不使用 this 指针：

    ```cpp
    #include <cstdio>
    using namespace std;

    template<typename T>
    struct BaseClass {
        T val;
        void print_msg() {
            printf("hello from base class, val: %d\n", val);
        }
    };

    template<typename T>
    struct MyClass: public BaseClass<int> {
        void invoke_func_from_base_class(T input_val) {
            val = input_val;  // ok
            print_msg();  // ok
        }
    };

    int main() {
        MyClass<int> obj;
        obj.invoke_func_from_base_class(123);
        return 0;
    }
    ```

* cpp 初始化 struct / class 中的 static 变量时，必须这样写：

    ```cpp
    #include <iostream>
    using namespace std;

    struct MyStruc
    {
        static int aaa;
    };

    int MyStruc::aaa = 123;

    int main()
    {
        printf("aaa: %d\n", MyStruc::aaa);
        return 0;
    }
    ```

    output:

    ```
    aaa: 123
    ```

    不能写成：

    ```cpp
    struct MyStruc
    {
        static int aaa = 1;
    };
    ```

* cpp decltype

    `auto`要求变量必须初始化，而`decltype`不要求。因为`auto`是根据变量的初始值来推导出变量类型的，而`decltype`可以写成下面的形式：

    ```cpp
    decltype(exp) varname;
    ```

    当`decltype`作用于函数时，函数并没有被调用：

    ```cpp
    #include <iostream>
    using namespace std;

    int my_func(int a, int b)
    {
        printf("in my_func()...\n");
        return a + b;
    }

    int main()
    {
        decltype(my_func(1, 2)) c = 123;
        printf("c = %d\n", c);
        return 0;
    }
    ```

    output:

    ```
    c = 123
    ```

    如果使用`()`包裹一个变量，或者在`decltype()`中填返回左值的表达式，就会被推导成引用：

    ```cpp
    #include <iostream>
    using namespace std;

    int main()
    {
        int a = 1, b = 2;
        // decltype((a)) c = 3;  // error, c is a int&
        // decltype(b = a + b) c = 3;  // error, c is a int&
        decltype(3) c = 3;  // ok, c is a int
        decltype(a + b) d = 3;  // ok, d is a int
        decltype((a + b)) e = 3;  // ok, e is a int
        return 0;
    }
    ```

* define 是`#define new old`, typedef 是`typedef old new;`，与 define 正好相反

* cpp 中，引用的地址就是原对象的地址

    ```cpp
    #include <string>
    #include <stdio.h>
    using namespace std;

    void print_pointer(string &str)
    {
        printf("in print_pointer(), %p\n", &str);
    }

    int main()
    {
        string str = "hello, world";
        printf("in main(), %p\n", &str);
        print_pointer(str);
        return 0;
    }
    ```

    output:

    ```
    in main(), 0x7ffd8a8531d0
    in print_pointer(), 0x7ffd8a8531d0
    ```

* `uint64_t`在 C++ 的`<cstdint>`头文件中

* 有关移位 + 类型转换

    ```cpp
    #include <stdio.h>
    #include <stdint.h>

    int main()
    {
        uint16_t a = 1;
        uint32_t b = a << 17;
        printf("b: %u\n", b);
        return 0;
    }
    ```

    查看反汇编，当`b`的类型为`uint32_t`时，会使用`movzx`先把`a`移动到 32 位寄存器`eax`中，然后再对寄存器`eax`移位。`movzx`指令的功能是用 0 填充高位，并送入寄存器，因此只适用于无符号整数。当`b`的类型为`uint16_t`时，则会使用普通`mov`指令将`a`送入寄存器。

    是否写成`uint32_t b = (uint32_t) a << 17;`，对汇编指令无影响。

    将 32 位数据转换成 16 数据同理：

    ```cpp
    #include <stdio.h>
    #include <stdint.h>

    int main()
    {
        uint32_t a = 1;
        uint16_t b = a << 14;
        printf("b: %u\n", b);
        return 0;
    }
    ```

    无论是否写成`uint16_t b = (uint32_t) a << 14;`，`a`的数据都会被直接按 16 字节加载到寄存器`eax`中，然后再进行移位。

    综上，我们移位时，最好不要发生类型转换，如果一定要转换，那么移位的位数不要超过 dst 变量的位宽。

    注：

    * 一个细节：当左移位的位数大于等于`dst`变量的位宽时，编译器生成的汇编会直接把数据`0`写入到`dst`中。

* cpp 中当 struct 自定义了构造函数后，就无法再使用`.xxx = yyy;`初始化了。

* 右值引用可以转换成 const 左值引用，说明右值引用有与左值引用相似的特性，那么为什么右值引用要求不能修改被引用对象的值呢？

* 关于指针，左值，右值

    ```cpp
    #include <iostream>
    using namespace std;

    void print_val_1(int *&p)
    {
        cout << *p << endl;
    }

    void print_val_2(int *&&p)
    {
        cout << *p << endl;
    }

    void print_val_3(int *const &p)
    {
        cout << *p << endl;
    }

    int main()
    {
        int val = 123;
        int *pval = &val;

        // pval 是个左值，可以直接走左值引用
        print_val_1(pval);
        // &val 是个匿名对象，是个右值，可以正常走右值引用
        print_val_2(&val);
        // &val 是右值，可以自动转换成 const 左值引用
        print_val_3(&val);
        
        // move 可以把左值转换成右值引用，这样就可以走右值引用的通道了
        print_val_2(move(pval));
        // pval 本身就是左值，当然可以走到左值引用的通道
        print_val_3(pval);

        // move 将左值 pval 转换成右值引用，但是右值引用又可以自动转换成左值引用，
        // 因此走 print_val_3() 这条通道也是没问题的
        print_val_3(move(pval));

        return 0;
    }
    ```

* 有关指针，引用，与`const`

    ```cpp
    #include <iostream>
    using namespace std;

    void test_1()
    {
        int a = 123;

        const int *pa = &a;
        cout << a << endl;
        // *pa = 456;  // error
        // pa 指向 const int 类型，因此不能修改值

        int *const pa_2 = &a;
        *pa_2 = 456;
        cout << a << endl;
        // pa_2 指向 int 类型，因此可以使用 *pa_2 修改值
        // 但是 pa_2 不能再指向其他对象，不允许使用 pa_2 = xxx; 修改 pa_2

        // int *const pa_3;  // error
        // const 指针必须在初始化时赋值，否则会报错

        int &ra = a;
        ra = 789;
        cout << a << endl;

        const int &ra_2 = a;
        // ra_2 = 123;  // error
        // ra_2 指向 const int 类型，因此无法修改值

        // int &const ra_3 = a;  // error
        // 不允许创建 const 引用
    }

    void test_2()
    {
        int a = 123;
        int *pa = &a;

        int *&rpa = pa;
        *rpa = 456;
        cout << a << endl;

        const int *cpa = &a;
        // const int *&rpa_2 = pa;  // error
        // pa 是指向 int 的指针，const int *& 要求指针指向 const int 类型，
        // 因此 error
        const int *&rpa_2 = cpa;
        // *cpa = 789;  // error
        // 显然 const int * 无法修改原对象的值

        int *const &rpa_3 = pa;
        *rpa_3 = 789;
        cout << a << endl;
        // 由于引用的特殊性，rpa_3 本身就无法再被赋值，因此这里的 const 没有用处

        // int **ppa_4 = &rpa_3;  // error
        // 如果没有前面的 int *const 保证，我们这里就可以拿到 pa 的地址，然后使用
        // *ppa_4 = &a; 修改 pa 所指向的对象
        // 由此可见，T *const & 的主要作用是防止指针指向其他对象
    }

    int main()
    {
        cout << "test 1:" << endl;
        test_1();

        cout << endl;

        cout << "test 2:" << endl;
        test_2();

        return 0;
    }
    ```

    output:

    ```
    test 1:
    123
    456
    789

    test 2:
    456
    789
    ```

    使用`T *const p = xxx;`定义的指针，一方面无法改变其指向，比如`p = yyy;`；另一方面也无法对其取址后赋值给更高权限的二级指针，比如：`T **pp = &p`，但是如果二级指针保证不改变一级指针的内容，那么是允许的：`T *const *pp = &p;`。

* 无论 c 还是 c++，`struct`对象都不允许直接比较相等，即使 struct 内都是内置类型（比如`float`, `int`）也不行。

    如果需要比较相等，C 中需要自己实现全局函数，C++ 中可以重载`operator==`。

    c++ example:

    ```cpp
    #include <iostream>
    #include <string>
    using namespace std;

    struct TestStruc
    {
        int val_1;
        string val_2;

        bool operator==(TestStruc &obj_2)  // const TestStruc & is not necessray
        {
            if (val_1 == obj_2.val_1 &&
                val_2 == obj_2.val_2
            )
                return true;
            return false;
        }
    };

    int main()
    {
        TestStruc obj_1{1, "hello"};
        TestStruc obj_2{1, "hello"};
        bool result = obj_1 == obj_2;
        cout << result << endl;
        return 0;
    }
    ```

    output:

    ```
    1
    ```

    c++ example 2:

    ```cpp
    #include <iostream>
    using namespace std;

    struct TestStruc
    {
        int val_1;
        string val_2;
    };

    bool operator==(TestStruc &obj_1, TestStruc &obj_2)
    {
        if (obj_1.val_1 == obj_2.val_1 &&
            obj_2.val_2 == obj_2.val_2
        )
            return true;
        return false;
    }

    int main()
    {
        TestStruc obj_1{1, "hello"};
        TestStruc obj_2{1, "hello"};
        bool result = obj_1 == obj_2;
        cout << result << endl;
        return 0;
    }
    ```

    output:

    ```
    1
    ```

* c++ 对新 struct 进行初始化时，允许使用`.xxx = vvv`的方式，但是必须按照成员的顺序，不能乱序

    examples:

    * 不写`.xxx`进行初始化

        ```cpp
        #include <iostream>
        #include <string>
        #include <vector>
        #include <unordered_map>
        using namespace std;

        ostream& operator<<(ostream &ost, vector<int> &vec)
        {
            for (int i = 0; i < vec.size(); ++i)
            {
                if (i < vec.size() - 1)
                {
                    cout << vec[i] << ", ";
                }
                else
                {
                    cout << vec[i];
                }
            }
            return ost;
        }

        ostream& operator<<(ostream &ost, unordered_map<string, int> &m)
        {
            cout << "{";
            int cnt = 0;
            for (auto iter = m.begin(); iter != m.end(); ++iter)
            {
                cout << "\"" << iter->first << "\"" << ": " << iter->second;
                cnt++;
                if (cnt < m.size())
                {
                    cout << ", ";
                }
            }
            cout << "}";
            return ost;
        }

        struct MyClass
        {
            int val_1;
            string val_2;
            vector<int> val_3;
            unordered_map<string, int> val_4;
        };

        int main()
        {
            MyClass my_obj {
                123,
                "hello",
                {1, 2, 3, 4},
                {
                    {"hello", 1},
                    {"world", 2}
                }
            };

            cout << "val_1: " << my_obj.val_1 << endl;
            cout << "val_2: " << my_obj.val_2 << endl;
            cout << "val_3: " << my_obj.val_3 << endl;
            cout << "val_4: " << my_obj.val_4 << endl;

            return 0;
        }
        ```

        output:

        ```
        val_1: 123
        val_2: hello
        val_3: 1, 2, 3, 4
        val_4: {"world": 2, "hello": 1}
        ```

    * 使用`.xxx`进行初始化

        ```cpp
        #include <iostream>
        #include <string>
        #include <vector>
        #include <unordered_map>
        using namespace std;

        ostream& operator<<(ostream &ost, vector<int> &vec)
        {
            for (int i = 0; i < vec.size(); ++i)
            {
                if (i < vec.size() - 1)
                {
                    cout << vec[i] << ", ";
                }
                else
                {
                    cout << vec[i];
                }
            }
            return ost;
        }

        ostream& operator<<(ostream &ost, unordered_map<string, int> &m)
        {
            cout << "{";
            int cnt = 0;
            for (auto iter = m.begin(); iter != m.end(); ++iter)
            {
                cout << "\"" << iter->first << "\"" << ": " << iter->second;
                cnt++;
                if (cnt < m.size())
                {
                    cout << ", ";
                }
            }
            cout << "}";
            return ost;
        }

        struct MyClass
        {
            int val_1;
            string val_2;
            vector<int> val_3;
            unordered_map<string, int> val_4;
        };

        int main()
        {
            MyClass my_obj {
                .val_1 = 123,
                .val_2 = "hello",
                .val_3 = {1, 2, 3, 4},
                .val_4 = {
                    {"hello", 1},
                    {"world", 2}
                }
            };

            cout << "val_1: " << my_obj.val_1 << endl;
            cout << "val_2: " << my_obj.val_2 << endl;
            cout << "val_3: " << my_obj.val_3 << endl;
            cout << "val_4: " << my_obj.val_4 << endl;

            return 0;
        }
        ```

        output:

        ```
        val_1: 123
        val_2: hello
        val_3: 1, 2, 3, 4
        val_4: {"world": 2, "hello": 1}
        ```

    * 使用`.xxx`选择性地初始化

        ```cpp
        #include <iostream>
        #include <string>
        #include <vector>
        #include <unordered_map>
        using namespace std;

        ostream& operator<<(ostream &ost, vector<int> &vec)
        {
            for (int i = 0; i < vec.size(); ++i)
            {
                if (i < vec.size() - 1)
                {
                    cout << vec[i] << ", ";
                }
                else
                {
                    cout << vec[i];
                }
            }
            return ost;
        }

        ostream& operator<<(ostream &ost, unordered_map<string, int> &m)
        {
            cout << "{";
            int cnt = 0;
            for (auto iter = m.begin(); iter != m.end(); ++iter)
            {
                cout << "\"" << iter->first << "\"" << ": " << iter->second;
                cnt++;
                if (cnt < m.size())
                {
                    cout << ", ";
                }
            }
            cout << "}";
            return ost;
        }

        struct MyClass
        {
            int val_1;
            string val_2;
            vector<int> val_3;
            unordered_map<string, int> val_4;
        };

        int main()
        {
            MyClass my_obj {
                .val_2 = "hello",
                .val_4 = {
                    {"hello", 1},
                    {"world", 2}
                }
            };

            cout << "val_1: " << my_obj.val_1 << endl;
            cout << "val_2: " << my_obj.val_2 << endl;
            cout << "val_3: " << my_obj.val_3 << endl;
            cout << "val_4: " << my_obj.val_4 << endl;

            return 0;
        }
        ```

        output:

        ```
        val_1: 0
        val_2: hello
        val_3: 
        val_4: {"world": 2, "hello": 1}
        ```

    * 乱序（out of order）初始化，报错

        ```cpp
        #include <iostream>
        #include <string>
        #include <vector>
        #include <unordered_map>
        using namespace std;

        ostream& operator<<(ostream &ost, vector<int> &vec)
        {
            for (int i = 0; i < vec.size(); ++i)
            {
                if (i < vec.size() - 1)
                {
                    cout << vec[i] << ", ";
                }
                else
                {
                    cout << vec[i];
                }
            }
            return ost;
        }

        ostream& operator<<(ostream &ost, unordered_map<string, int> &m)
        {
            cout << "{";
            int cnt = 0;
            for (auto iter = m.begin(); iter != m.end(); ++iter)
            {
                cout << "\"" << iter->first << "\"" << ": " << iter->second;
                cnt++;
                if (cnt < m.size())
                {
                    cout << ", ";
                }
            }
            cout << "}";
            return ost;
        }

        struct MyClass
        {
            int val_1;
            string val_2;
            vector<int> val_3;
            unordered_map<string, int> val_4;
        };

        int main()
        {
            MyClass my_obj {
                .val_4 = {
                    {"hello", 1},
                    {"world", 2}
                },
                .val_2 = "hello"
            };

            cout << "val_1: " << my_obj.val_1 << endl;
            cout << "val_2: " << my_obj.val_2 << endl;
            cout << "val_3: " << my_obj.val_3 << endl;
            cout << "val_4: " << my_obj.val_4 << endl;

            return 0;
        }
        ```

        compile output:

        ```
        g++ -g main.cpp -I/home/hlc/Documents/Projects/boost_1_87_0 -o main
        main.cpp: In function ‘int main()’:
        main.cpp:56:5: error: designator order for field ‘MyClass::val_2’ does not match declaration order in ‘MyClass’
           56 |     };
              |     ^
        make: *** [Makefile:2: main] Error 1
        ```

* c++ 中`decltype`的用法

    ```cpp
    #include <iostream>
    using namespace std;

    int main()
    {
        int a = 3;
        decltype(a) b = 5;
        cout << "b is " << b << endl;
        return 0;
    }
    ```

    output:

    ```
    b is 5
    ```

    在`decltype(expression)`中，当`expression`是一个变量名时，`decltype`会推导出该变量的类型，当`expression`是一个函数调用时，`decltype`会推导出该函数的返回值类型。当`expression`是一个表达式时，`decltype`会推导出该表达式的类型。

    example:

    ```
    int x = 5;
    int& y = x;
    decltype(x) z1 = x;  // z1 is a int
    decltype(y) z2 = y;  // z2 is a int&
    decltype(x + y) z3 = x + y;  // z3 is a int
    decltype(std::cout << ｘ) z4 = std::cout << x;  // z4 is a std::ostream&
    ```

* c++ 中，如果没有自定义的构造函数，那么只能使用`MyStruct{xxx, yyy}`来初始化对象，不能使用`MyStruct(xxx, yyy)`初始化对象。

* c++ 代码中的`obj_1 == obj_2`会调用到`operator==()`，两个`struct`对象不能比大小，也不能默认按值判断相等。

* cuda 中，使用 struct 辅助实现偏特化

    因为 c++ 不允许模板函数的偏特化，所以我们使用 struct 辅助一下。

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include "../utils/cumem_hlc.h"
    #include "../utils/timeit.h"

    enum Op
    {
        op_sum,
        op_minus
    };

    template<typename T, Op op>
    struct Calc;

    template<typename T>
    struct Calc<T, op_sum>
    {
        __device__ static void do_calc(T *a, T *b, T *out)
        {
            *out = *a + *b; 
        }
    };

    template<typename T>
    struct Calc<T, op_minus>
    {
        __device__ static void do_calc(T *a, T *b, T *out)
        {
            *out = *a - *b;
        }
    };

    template<typename T, Op op>
    __global__ void do_calc(T *a, T *b, T *out)
    {
        Calc<T, op>::do_calc(a, b, out);
    }

    int main()
    {
        float *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, sizeof(float));
        cudaMalloc(&cubuf_2, sizeof(float));
        assign_cubuf_rand_int(cubuf_1, 1);
        assign_cubuf_rand_int(cubuf_2, 1);
        print_cubuf(cubuf_1, 1);
        print_cubuf(cubuf_2, 1);
        do_calc<float, op_sum><<<1, 1>>>(cubuf_1, cubuf_2, cubuf_1);
        cudaDeviceSynchronize();
        printf("after:\n");
        print_cubuf(cubuf_1, 1);
        print_cubuf(cubuf_2, 1);
        return 0;
    }
    ```

    compile: `vcc -g -G main.cu -o main`

    run: `./main`

    output:

    ```
    3.0, specialized as float
    1.0, specialized as float
    after:
    4.0, specialized as float
    1.0, specialized as float
    ```

* c/cpp 似乎没法直接把一个 val 转换成一个 class/union 的对象，但是可以通过指针转换 + 解引用来完成

    ```cpp
    #include <stdio.h>

    union MyUnion
    {
        float val;
        char spices[4];
    };

    int main()
    {
        float val = 123;
        // MyUnion val_union = (MyUnion) val;  // Error
        MyUnion val_union = *(MyUnion*) &val;
        printf("val_union.val: %f\n", val_union.val);
        return 0;
    }
    ```

    output:

    ```
    val_union.val: 123.000000
    ```

* c++ 模板无法通过隐式推断根据返回值类型推断模板参数

    ```cpp
    #include <stdio.h>

    template <typename T>
    T add(int a, int b)
    {
        return a + b;
    }

    template<>
    float add(int a, int b)
    {
        printf("specialized as float\n");
        return a + b;
    }

    template<>
    int add(int a, int b)
    {
        printf("specialized as int\n");
        return a + b;
    }

    int main()
    {
        int a, b;
        a = 1;
        b = 2;
        float c = add<float>(a, b);
        int d = add<int>(a, b);
        // int d = add(a, b);  // error
        //  float c = add<int>(a, b);  // ok
        printf("c: %.1f, d: %d\n", c, d);
        return 0;
    }
    ```

    output:

    ```
    specialized as float
    specialized as int
    c: 3.0, d: 3
    ```

* C++ 中，当一个指针由`float*`转换为`void*`时，必须加上强制类型转换`(void*)`。使用隐式类型转换编译器会报错。

* 类是模板，成员函数也是模板的类内实现和类外实现

    ```cpp
    #include <stdio.h>

    template<typename T>
    struct Printer
    {
        T val;

        template<typename P>
        void print_type_size(P func_val)
        {
            int size_val = sizeof(T);
            int size_func_val = sizeof(P);
            printf("val size: %d, func val size: %d\n",
                size_val, size_func_val);
        }

        template<typename P>
        void print_type_size_2(P func_val);
    };

    int main()
    {
        Printer<float> printer;
        printer.print_type_size(123);
        printer.print_type_size<double>(123);
        printer.print_type_size_2<long>(456);
        return 0;
    }

    template<typename T>
    template<typename P>
    void Printer<T>::print_type_size_2(P func_val)
    {
        int size_val = sizeof(T);
        int size_func_val = sizeof(P);
        printf("in print 2, val size: %d, func val size: %d\n",
            size_val, size_func_val);
    }
    ```

    output:

    ```
    val size: 4, func val size: 4
    val size: 4, func val size: 8
    in print 2, val size: 4, func val size: 8
    ```

    在外部实现函数时，必须写两个`template`:

    ```cpp
    template<typename T>
    template<typename P>
    ```

    不能这样写：

    `template<typename T, typename P>`

    也不能把 T 和 P 的顺序搞反。

* 特化的模板类里的模板函数，使用特化模板类参数，不能使用通用模板类的模板参数

    ```cpp
    #include <stdio.h>

    template<typename T>
    struct Printer
    {
        T val;

        template<typename P>
        void print_type_size(P func_val)
        {
            int size_val = sizeof(T);
            int size_func_val = sizeof(P);
            printf("val size: %d, func val size: %d\n",
                size_val, size_func_val);
        }
    };

    template<typename T>
    struct BytePack
    {
        static const int size = sizeof(T);
        T val;
    };

    template<typename T>
    struct Printer<BytePack<T>>
    {
        // template<typename P>
        // void print_val(BytePack<T> bytes, P val)
        // {
        //     printf("in specialized Printer, bytes val: %d, bytes size: %d, val: %d\n",
        //         BytePack<T>::size, bytes.val, val);
        // }

        template<typename P>
        void print_val(BytePack<T> bytes, P val);
    };

    int main()
    {
        Printer<float> printer;
        printer.print_type_size<double>(123);
        Printer<BytePack<long>> printer_2;
        BytePack<long> bytes;
        bytes.val = 456;
        printer_2.print_val(bytes, 789);
        return 0;
    }

    template<typename T>
    template<typename P>
    void Printer<BytePack<T>>::print_val(BytePack<T> bytes, P val)
    {
        printf("in specialized Printer, bytes val: %ld, bytes size: %d, val: %d\n",
            bytes.val, BytePack<T>::size, val);
    }
    ```

    output:

    ```
    val size: 4, func val size: 8
    in specialized Printer, bytes val: 456, bytes size: 8, val: 789
    ```

* 函数的默认参数

    ```cpp
    #include <stdio.h>

    void print_aligned(int width = 10);
    void print_aligned_2(int width);

    void print_aligned_2(int width = 10)
    {
        printf("%*s\n", width, "world");
    }

    // void print_aligned_3(int width = 15);

    // Error !
    // void print_aligned_3(int width = 10)
    // {
    //     printf("%*s\n", width, "world");
    // }

    int main()
    {
        printf("0123456789\n");
        print_aligned();
        print_aligned_2();
        // print_aligned_3();
        return 0;
    }

    void print_aligned(int width)
    {
        printf("%*s\n", width, "hello");
    }

    // Error !
    // void print_aligned_2(int width = 10)
    // {
    //     printf("%*s\n", width, "world");
    // }
    ```

    output:

    ```
    0123456789
         hello
         world
    ```

    函数的默认参数可以只出现在声明里，也可以只出现在定义里，但是出现在定义里时，必须比函数被调用的时机要早。

    默认参数不可以同时出现在声明和定义里，否则会编译时报错。

    类成员函数同理。

* 类外定义的函数进行类内成员初始化，写法与类内没什么不同

    ```cpp
    #include <stdio.h>

    template<typename T>
    struct MyType
    {
        T val;

        MyType(T &input_val);
    };

    template<typename T>
    MyType<T>::MyType(T &input_val): val(input_val)
    {
        printf("val is %d\n", val);
    }

    int main()
    {
        int val = 123;
        MyType<int> my_type(val);

        return 0;
    }
    ```

    output:

    ```
    val is 123
    ```

* 模板参数分别是默认，特化为内置类型，特化为模板类的类模板声明与函数的实现方式

    ```cpp
    #include <stdio.h>
    #include <stdlib.h>

    template<typename T>
    struct BytePack
    {
        static const int size = sizeof(T);
    };

    template<typename ValType>
    struct Printer
    {
        // void print_bytes()
        // {
        //     printf("in default Printer, type size: %lu bytes\n", sizeof(ValType));
        // }

        void print_bytes();
    };

    template<>
    struct Printer<float>
    {
        // void print_bytes()
        // {
        //     printf("in specialized float class, size: %lu bytes\n", sizeof(float));
        // }

        void print_bytes();
    };

    template<typename T>
    struct Printer<BytePack<T>>
    {
        // void print_bytes()
        // {
        //     printf("type T size: %d bytes\n", BytePack<T>::size);
        // }

        void print_bytes();
    };

    int main()
    {
        Printer<BytePack<float>> printer;
        printer.print_bytes();

        Printer<float> printer_2;
        printer_2.print_bytes();

        Printer<long long> printer_3;
        printer_3.print_bytes();

        return 0;
    }

    template<typename ValType>
    void Printer<ValType>::print_bytes()
    {
        printf("in default Printer, type size: %lu bytes\n", sizeof(ValType));
    }

    void Printer<float>::print_bytes()
    {
        printf("in specialized float class, size: %lu bytes\n", sizeof(float));
    }

    template<typename T>
    void Printer<BytePack<T>>::print_bytes()
    {
        printf("in specialized BytePack<T> class, type T size: %d bytes\n", BytePack<T>::size);
    }
    ```

    output:

    ```
    in specialized BytePack<T> class, type T size: 4 bytes
    in specialized float class, size: 4 bytes
    in default Printer, type size: 8 bytes
    ```

    可以看到，根据 main 函数中传入的不同类型，分别走不同的模板类调用成员函数。

    当特化类型为内置类型`float`时，类声明前使用了`template<>`，而函数的实现前不可以加`template<>`。

* 模板函数

    允许模板函数重载（函数名相同，但是被认为是不同的函数），无论模板参数不同还是函数参数不同都可以，

    唯一要求是可以推导出来最合适的匹配

    如果无法推导必须使用 <> 指定类型

    `template func_name<type>()`是模板的实例化

    `template<> func_name<type>()`是模板的特化

* 使用`typeid()`判断模板参数`T`的类型会编译时报 warning

* 关于数值类型模板参数与 if 展开

    有时我们会写这样的函数：

    ```cpp
    #include <stdio.h>

    enum Operation
    {
        ADD,
        MINUS,
        MULTIPLY,
        DIVIDE
    };

    void do_calc(int a, int b, int *out, Operation op)
    {
        if (op == ADD)
        {
            *out = a + b;
        }
        else if (op == MINUS)
        {
            *out = a - b;
        }
        else if (op == MULTIPLY)
        {
            *out = a * b;
        }
        else if (op == DIVIDE)
        {
            *out = a / b;
        }
        else
        {
            printf("unknown operation: %d\n", op);
        }
    }

    int main()
    {
        int a = 2, b = 3;
        int c;
        do_calc(a, b, &c, MULTIPLY);
        printf("%d * %d = %d\n", a, b, c);
        return 0;
    }
    ```

    output:

    ```
    2 * 3 = 6
    ```

    但是这样有一个问题，如果我们传入的`op`是`DIVIDE`，那至少要经过 if 语句判断 4 次才行。如果我们希望尽量让对应类型的操作走入到对应的分支，尽量减少 if 的判断，该怎么办？

    我们尝试使用模板：

    `main.cpp`:

    ```cpp
    #include <stdio.h>

    enum Operation
    {
        ADD,
        MINUS,
        MULTIPLY,
        DIVIDE
    };

    template<Operation op>
    void do_calc(int a, int b, int *out)
    {
        if (op == ADD)
        {
            *out = a + b;
        }
        else if (op == MINUS)
        {
            *out = a - b;
        }
        else if (op == MULTIPLY)
        {
            *out = a * b;
        }
        else if (op == DIVIDE)
        {
            *out = a / b;
        }
        else
        {
            printf("unknown operation: %d\n", op);
        }
    }

    int main()
    {
        int a = 2, b = 3;
        int c;
        do_calc<MULTIPLY>(a, b, &c);
        printf("%d * %d = %d\n", a, b, c);
        return 0;
    }
    ```

    此时模板会在编译时展开，不同的 op，函数展开的内容也不一样。在调用时，会直接进入对应函数的入口，

    可以尝试在`if (op == ADD)`处与`else if (op == MINUS)`处加上断点，这两个断点是走不到的。

* 一些面向对象设计的问题

    * 若类`B`依赖`A`，但`B`并不拥有`A`，那么如何保证在`B`有效时`A`一定有效，且`A`发生改变后`B`能使用`A`改变后的值？

    * 如果`A`，`B`结合起来才能实现某些功能，功能的实现放到`C`中，那么`C`该如何设计才能满足这个要求？

        方案 1：`A`，`B`作为`C`的成员。不可以，因为`A`，`B`也可以独立完成一些功能。

        方案 2：`C`保存`A`，`B`的指针，但是会有失效的问题。

        方案 3：`C`保存`A`，`B`的智能指针。

        方案 4：创建一个类`D`，`D`负责维护`A`，`B`，`C`的依赖关系。

* c++ class member as a reference

    ```cpp
    #include <stdio.h>

    class A
    {
        public:
        A(int &val_1, int val_2)
        :ref_1(val_1), ref_2(val_2) {
            
        }
        int &ref_1;
        int &ref_2;
    };

    int main()
    {
        int val_1 = 3;
        int val_2 = 3;
        A a(val_1, val_2);
        printf("a.ref_1: %d\n", a.ref_1);
        a.ref_1 = 2;
        printf("after changing a.ref_1, val_1: %d\n", val_1);
        printf("a.ref_2: %d\n", a.ref_2);
        a.ref_2 = 2;
        printf("after changing a.ref_2, val_2: %d\n", val_2);
        return 0;
    }
    ```

    output:

    ```
    a.ref_1: 3
    after changing a.ref_1, val_1: 2
    a.ref_2: 3
    after changing a.ref_2, val_2: 3
    ```

    Reference must be assigned a value when the object created if it serves as a class member.

    The reference member can accept a rvalue: `A a(val_1, 3);`

* c++ `isdigit()`判断一个`char`字符是否为数字

    目前只要`#include <iostream>`，就可以使用`std::isdigit()`。

    不清楚初始的时候出自哪里。

* c++ piecewise construct 中添加多个类型时，不同构造函数参数数量的处理：

    ```cpp
    void add_local_buf(string buf_name, int elm_size, int elm_num)
    {
        int buf_size = elm_size * elm_num;
        local_bufs.emplace(
            piecewise_construct,
            forward_as_tuple(buf_name),
            forward_as_tuple(buf_name, elm_size, elm_num, ctx)
        );
    }
    ```

* `remove_const_t<>`作用于指针时，移除的是指针的`const`

    example:

    `remove_const_t<const char *const>`，移除的其实是第二个`const`。

* c++ 中 template 无法和`typeid()`合用

    ```cpp
    template<typename T, typename...Args>
    void _set_args(OclKern &kern, T arg, Args...args)
    {
        if (typeid(arg) == typeid(const char*) ||
            typeid(arg) == typeid(char*) ||
            typeid(arg) == typeid(string))
        {
            OclBuf &buf = global_ocl_env->bufs.at(arg);
            kern.sa(buf);
        }
        else
        {
            kern.sa(arg);
        }
        _set_args(kern, args...);
    }
    ```

    比如这段代码，在编译时期就会出错。编译器不走 if, else 分支，只看类型是否匹配。

    应该改成这样的：

    ```cpp
    template<typename T, typename...Args>
    enable_if_t<is_same_v<T, const char*>, void>
    _set_args(OclKern &kern, T arg, Args...args)
    {
        OclBuf &buf = global_ocl_env->bufs.at(arg);
        kern.sa(buf);
        _set_args(kern, args...);
    }

    template<typename T, typename...Args>
    enable_if_t<!is_same_v<T, const char*>, void>
    _set_args(OclKern &kern, T arg, Args...args)
    {
        kern.sa(arg);
        _set_args(kern, args...);
    }
    ```

    这样在编译时期就能走对应的通路。

    (这个功能叫什么？它和模板函数的重载，特化，有什么不同？可以不使用模板实现吗？)

    注意这个模板函数只匹配了`const char*`这一种类型，如果未来有`string`，`char*`等类型，还需要用`conjunction_v<>`等命令去匹配。

* `unique_ptr`释放其所掌握的指针：

    ```cpp
    OclEnv *p_ocl_env = global_ocl_env.release();
    delete p_ocl_env;
    ```

    注意只有`unique_ptr`有这个功能，`shared_ptr`没有这个功能。

* `array<int, 26> arr;`没有 clear 的功能，有两种方式可以实现赋值

    1. 使用初始化赋值

        ```cpp
        array<int, 26> arr = {1};  // 将 26 个数据都赋值为 1
        ```

        这个初始化方法在 for 中仍然适用。

    2. 使用`fill()`赋值

        ```cpp
        array<int, 26> arr;
        arr.fill(3);  // 将 26 个数据赋值为 3
        ```

    如果在 for 中声明变量，但是不初始化，那么并不会每次都赋初始值：

    ```cpp
    for (int i = 0; i < 5; ++i)
    {
        array<int, 10> arr;
        for (int num: arr)
        {
            printf("%d, ", num);
        }
        arr[i] = i;
        putchar('\n');
    }
    ```

    output:

    ```
    1, 2, 3, 32512, -394097016, 32512, -394983164, 32512, -394096808, 32512, 
    0, 2, 3, 32512, -394097016, 32512, -394983164, 32512, -394096808, 32512, 
    0, 1, 3, 32512, -394097016, 32512, -394983164, 32512, -394096808, 32512, 
    0, 1, 2, 32512, -394097016, 32512, -394983164, 32512, -394096808, 32512, 
    0, 1, 2, 3, -394097016, 32512, -394983164, 32512, -394096808, 32512,
    ```

    `array<>`没有`fill()`和`assign()`方法。

* `vector<int[26]> v;`实际存放的是`int*`指针，具体的 26 个 int 的存储空间需要自己去申请

    因此没有办法做到`v.push_back()`这种事，同样地，`v.push_back({})`，`v.push_back({1, 2, 3})`，`v.push_back(int[26])`这些也都无法实现。

    但是可以使用`vector<array<int, 26>> v;`实现存放数组的功能。

* c++ 中 unordered_map 无法使用`vector<int>`之类的数据作为 key，但是可以作为 value

    也无法使用`array<>`作为 key。

    如果需要这些数据类型作为 key，需要自己写哈希函数和比较函数。

* c/c++ 函数的声明与定义不在同一个文件的编译方法

    `aaa.h`:

    ```cpp
    extern int add(int a, int b);
    ```

    `bbb.cpp`:

    ```cpp
    int add(int a, int b)
    {
        return a + b;
    }
    ```
    
    `main.cpp`:

    ```cpp
    #include "aaa.h"
    #include <stdio.h>

    int main()
    {
        int a = 1, b = 2;
        int c = add(a, b);
        printf("%d + %d = %d\n", a, b, c);
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.cpp bbb.o
        g++ -g main.cpp bbb.o -o main
    ```

    测试了下，对于非内核态的普通 c++ 程序，可以把`aaa.h`里的`extern`去掉。

    看来只要提供有函数定义的的`.o`文件，就可以通过编译。

    内核态的程序不清楚。

* c++ 中函数指针支持`vector`等类型

    example:

    ```cpp
    int (*get_ans)(vector<int> &&) = get_ans_1;
    ```

* c++ 中不可以使用一个 vector 去初始化一个 queue

    也不可以使用 vector 的迭代器 begin(), end() 去初始化一个 queue

    queue 没有 assign() 功能，也没有 resize() 功能。

    queue 不支持初始化列表。

* 创建一个`vector`数组

    ```cpp
    #include <vector>
    #include <iostream>
    using namespace std;

    int main()
    {
        vector<int> vecs[3];
        vecs[0].resize(5);
        vecs[1].resize(5, 0);
        vecs[2].assign(5, 0);
        for (int i = 0; i < 3; ++i)
        {
            for (int elm: vecs[i])
            {
                cout << elm << ", ";
            }
            cout << endl;
        }
        return 0;
    }
    ```

* c++ 中初始化 struct 的方法

    ```cpp
    #include <iostream>
    using namespace std;

    struct AAA {
        int val_a;
        long val_b;
    };

    struct AAA a {

    };

    struct AAA b = {

    };

    struct AAA c {
        .val_a = 1
    };

    struct AAA d = {
        .val_a = 2
    };

    int main()
    {
        cout << a.val_a << endl;
        cout << b.val_a << endl;
        cout << c.val_a << endl;
        cout << d.val_a << endl;
        return 0;
    }
    ```

    output:

    ```
    0
    0
    1
    2
    ```

    在 c 与 gnuc 中，由于不支持初始化列表（initialization list），所以不支持`struct AAA a { };`这样的初始化方法，但是支持加`=`的初始化方法。

* 有关 mem 对象的生存周期分配

    如果两个 mem 对象互相依赖，那么它们应该被一个更大的对象管理，由这个更大的对象控制它们的生命周期。

* c/c++ 中，变量名/对象实体其实代表的是内存地址，对内存地址操作肯定是最快，效率最高的

    把各种对象映射成使用字符串索引，虽然降低了效率，但是提高了便利性。

* c++ 中，对`operator[]`加 template 并不能像想象中一样可以这么写：

    ```cpp
    my_vec<float>[3] = 4.0f;
    ```

    反而会非常麻烦：

    ```cpp
    // emplementation
    template<typename T>
    T& operator[](size_t idx) {
        return *((T*)mem+idx);
    }
    
    // usage
    buf_mem.operator[]<float>(0) = 1.0f;
    ```

    此时还不如定义一个普通函数。

* c++ 中，如果 piecewise construct 一个`vector<string, pair<string, MyClass>>`，该怎么传递参数？

* c++ lambda and function pointer

    An example of lambda expression:

    ```cpp
    #include <stdio.h>

    int main()
    {
        auto print_val = [](int val) -> int {
            printf("hello, %d\n", val);
            return 1;
        };
        int rtv = print_val(3);
        printf("return value: %d\n", rtv);
        return 0;
    }
    ```

    output:

    ```
    hello, 3
    return value: 1
    ```

    the return value type of `-> int` is optional. You can miss it and the compiler will help you complete it.

    But you can't give a wrong type for return value. The compiler g++ will throw error if your return type hint is not identical to the real return value.

    The syntax of capture list:

    * `[ ]`: Capture nothing.

    * `[ = ]`: Capture everything by value. It gives only read access.

    * `[ & ]`: Capture everything by reference. It gives both read and write access.

    * `[ =, & x]`: Capture everything by value, and x variable by reference.

    * `[ =x, & ]`: Capture everything by reference, and x by value.

    function pointer:

    ```cpp
    #include <stdio.h>

    void print_ab(int a, int b) {
        printf("a = %d, b = %d\n", a, b);
    }

    int main()
    {
        void (*func)(int, int) = print_ab;
        func(1, 2);
        return 0;
    }
    ```

    output:

    ```
    a = 1, b = 2
    ```

    capture list example:

    ```cpp
    #include <stdio.h>

    int main()
    {
        int x = 0, a = 1, b = 2;
        auto test_fun = [=, &x]() {
            printf("%d + %d = %d\n", a, b, a + b);
            x++;
        };
        test_fun();
        printf("x is %d\n", x);
        return 0;
    }
    ```

    output:

    ```
    1 + 2 = 3
    x is 1
    ```

    A lambda function with a non-empty capture list can't be converted to a C-style function pointer.

* c++ 模板函数多参数

	对于`Args...args`，是可以无参数传递的。但如果是`(T arg, Args...args)`，这样就不行了，要求至少有一个参数。

	```cpp
	#include <iostream>
	using namespace std;

	template<typename T, typename...Args>
	void test_1(T param_1, Args...args)
	{
		cout << "in test_1" << endl;
	}

	template<typename...Args>
	void test_2(Args...args)
	{
		cout << "in test_2" << endl;
	}

	int main()
	{
		// test_1();  // error
		test_1("hello");
		test_1("hello", "world");

		test_2();
		test_2("hello");
		test_2("hello", "world");

		return 0;
	}
	```

	output:

	```
	in test_1
	in test_1
	in test_2
	in test_2
	in test_2
	```

* c++ 中`unorderd_map`使用`[]`获取元素时，要求元素的类型必须有不带参数的构造函数。

	但是使用`at()`就可以避开这个。

* c/c++ 中，不可以将`char**`隐式转换为`const char**`

    ```cpp
    void test(const char **strs)
    {

    }

    int main()
    {
        char **strs = nullptr;
        test(strs);
        return 0;
    }
    ```

    编译：

    ```bash
    g++ main.cpp
    ```

    输出：

    ```
    main.cpp: In function ‘int main()’:
    main.cpp:9:10: error: invalid conversion from ‘char**’ to ‘const char**’ [-fpermissive]
        9 |     test(strs);
        |          ^~~~
        |          |
        |          char**
    main.cpp:1:24: note:   initializing argument 1 of ‘void test(const char**)’
        1 | void test(const char **strs)
        |           ~~~~~~~~~~~~~^~~~
    ```

    不清楚为什么。

* c++ 中，对于模板函数，即使通用类型在参数列表中消失，也算函数的重载

    但是实际无法正常调用，因为无法推导出`T`的类型：

    ```cpp
    #include <iostream>
    using namespace std;

    template<typename T>
    void print(int a, int b)
    {
        cout << a << ", " << b << endl;
    }

    template<typename T>
    void print(int a, int b, T c)
    {
        cout << a << ", " << b << ", " << c << endl;
    }

    int main()
    {
        print(1, 2, 3);  // OK
        print(4, 5);  // Error
        print<int>(4, 5);  // OK, T = int
        return 0;
    }
    ```

    比如上面的代码，`void print(int a, int b)`的参数列表并没有给出`T`，所以尽管它算作`print()`的一个重载，但是后面的代码`print(4, 5);`并不能通过编译，因为无法推导出`T`的类型。

    如果我们像这样`print<int>(4, 5);`指定了`T`的类型，那么是可以正常运行的。

* 使用`fopen("xxx", r+")`并不能使`ftell(f)`返回文件的长度。

    C 语言中可以这样读文件：

    ```cpp
    FILE *f = fopen("kernels.cl", "r");
    fseek(f, 0, SEEK_END);
    size_t p_program_length = ftell(f);
    char *program_content = (char*) malloc(p_program_length);
    fseek(f, 0, SEEK_SET);
    fread(program_content, p_program_length, 1, f);
    ```

* 有关树的返回

    对于树的数据结构，函数从哪里进入就得从哪里返回，因此对树进行展开其实是一个搜集当前节点以及子树信息的过程。

    因此一个二叉树要返回的信息，其实是对三个信息的综合：当前节点，左子树，右子树。

    因此广义上看，似乎无论什么遍历，到最后都是后序遍历？

* c++ 的 move 其实是移动了 memory allocator 的证据

    ```cpp
    #include <vector>
    #include <stdio.h>
    using namespace std;

    int main()
    {
        vector<int> arr_1{1, 2, 3, 4, 5};
        for (auto &num: arr_1)
            printf("%d, ", num);
        putchar('\n');

        vector<int> arr_2 = move(arr_1);
        for (int &num: arr_1)
            printf("%d, ", num);
        putchar('\n');
        return 0;
    }
    ```

    output:

    ```
    1, 2, 3, 4, 5, 

    ```

    上面这段代码只有第一次 printf 的时候有输出，第二次 printf 的时候没有输出，说明`arr_1`已经失效了。

    如果 print `arr_1.size()`，也会输出 0.

* c++ 中，形参使用 const 和非 const 是相同类型是因为形参按值传递似乎站不住脚，因为做了个实验，const 的对象无法修改成员数据，const 的内置类型也无法修改自身，这样的代码无法通过编译。

    因此 const 实际限制修改的是形参，而不管实参是按值传递还是按引用传递。

    这样一来，const type 和 type 为什么会是相同的类型，就又不得而知了。

## Variable

**Basic types**

| Type | Space | Range |
| - | - | - |
| short | 2 bytes | [-2^15, 2^15-1], [-32768, 32767] |
| int | 4 bytes | [-2^31, 2^31-1] |
| long | 4 bytes or 8 bytes | [-2^31, 2^31-1] |
| long long | 8 bytes | [-2^63, 2^63-1] |
| float | 4 bytes | 7 位有效数字 |
| double | 8 bytes | 15 ~ 16 位有效数字 |
| bool | 1 bytes | true, false |

Example:

```cpp
#include <iostream>
#include <cstdlib>
using namespace std;

int main()
{
    char *types[] = {"short", "int", "long", "long long", 
        "bool", "char", "char*", "int*"};
    int length[] = {sizeof(short), sizeof(int),
        sizeof(long), sizeof(long long), 
        sizeof(bool), sizeof(char), sizeof(char*), sizeof(int*)};
    int len = sizeof(types) / sizeof(char*);
    for (int i = 0; i < len; ++i)
    {
        printf("%s: %d, ", types[i], length[i]); 
    }
    cout << endl;
    return 0; 
}
```

Output:

```
short: 2, int: 4, long: 4, long long: 8, bool: 1, char: 1, char*: 8, int*: 8,
```

**Integer type**

Interger numbers are stored in the memory by the radix complement.

Examples:

```
1: 00000000000000000000000000000001
-1: 11111111111111111111111111111111
```

**Literal variables**

字面常量小数默认为`double`类型。

默认情况下，使用`cout`输出一个小数，会显示出 6 位有效数字。使用`printf("%f")`输出一个小数，会显示到小数点后 6 位。

这两种方式都会做四舍五入。但是四舍五入的方式有些奇怪，并不是严格按照`5`加一，`4.xx`舍弃的方法来的。具体情况似乎和底层的二进制存储方式有关。有时间了再看看。

**char and ASCII**

对于`char`类型变量，`0 ~ 31`存储的是非可打印字符，`32 ~ 127`存储的是可打印字符。

Note: `32`是空格` `，`33`是叹号`!`，`34`是双引号`"`，`126`是波浪线`~`，`127`是 DEL。

ASCII table: <https://www.cs.cmu.edu/~pattis/15-1XX/common/handouts/ascii.html>

## 数组与字符串

数组与指针：

```cpp
int arr[3] = {1, 2, 3};
cout << (int)arr << endl;  // 数组名即数组首地址
cout << (int)&arr[0] << endl;  // 第一个元素的地址
cout << (int)&arr[1] << endl;  // 第二个元素的地址
```

二维数组的定义：

```cpp
int arr[row][col];
int arr[row][col] = {{data_1, data_2}, {data_3, data_4}, ...};
int arr[row][col] = {data_1, data_2, data_3, data_4, ...};
int arr[][col] = {data_1, data_2, data_3, data_4, ...};
```

### 有关数组的初始化

1. 全局数组中的每个元素会被编译器初始化为 0。

1. 局部数组编译器不会自动初始化。通常可以用`memset`将其初始化为 0：`memset(arr, 0, sizeof(arr));`。`memset`是以字节为单位对内存数据进行赋值的，因此如果赋的值非零，就不能用`memset`了。

    Example:

    ```cpp
    #include <iostream>
    using namespace std;

    int main()
    {
        int arr[30];
        for (int i = 0; i < 30; ++i)
            cout << arr[i] << ", ";
        cout << endl;
        return 0;
    }
    ```

    Output:

    ```
    0, 0, 0, 0, -1696722216, 529, 1150278918, 32766, 0, 0, 0, 0, 1973229152, 32758, 24, 0, 0, 0, 0, 0, 16, 0, 1973228793, 32758, 0, 0, 52, 0, -1696708304, 529,
    ```

1. 如果只初始化了几个值，那么剩下的数据似乎会被自动初始化为 0：

    ```cpp
    int main()
    {
        int arr[5] = {1, 2};  // {1, 2, 0, 0, 0}
        return 0;
    }
    ```

    （不清楚这一点是否被编译器保证，因此写程序时不应该依赖这个特性）

### 字符串

C 风格的字符串是一种特殊的数组，即数组中存储的是`char`类型的数据。

字符串定义：

```c++
char str[] = "hello, world!";
const char *str = "nihao, zaijian";
```

## 运算符

有符号整数的`>>`，左侧会位移`1`。

前置递增：先递增变量，再计算表达式值。

后置递增：计算表达式值/传递函数参数，再递增变量。

三目运算符：`condition ? statement_1 : statement_2`。

## 函数

**有关引用**

1. 不要返回局部变量的引用

    ```cpp
    int& test()
    {
        int a = 3;
        return a;
    }

    int main()
    {
        int &ra = test();
        ra = 1;  // run time error
        return 0;
    }
    ```

    局部变量`a`在函数返回时已经被释放，此时对`ra`操作会内存错误。

1. 如果局部变量是静态的，那么就可以返回引用，甚至可以修改静态变量

    ```cpp
    int& test()
    {
        static int a = 3;
        return a;
    }

    int main()
    {
        int &ra = test();
        cout << ra << endl;
        ra = 4;
        cout << test() << endl;
        test() = 5;
        cout << test() << endl;
        return 0;
    }
    ```

**函数的默认参数**

函数的声明和实现只能有一个有默认参数。

占位参数：

```cpp
int test(int a, int)
{
    return 0;
}

int test2(int a, int = 3)  // 占位参数也可以有默认参数
{
    return 0;
}

int main()
{
    test(3);
    test(3, 3);
}
```

### 函数重载

1. 只修改函数的返回值类型不能视为重载

1. 函数重载必须在同一作用域下

1. `const`可以作为不同的类型吗？

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

    想了想，这种设计是正确的。因为函数的形参总是一个副本，无论有没有`const`都无所谓。而引用是关系到变量实际上是否被修改的，所以需要对`const`加以区分。

    `int *a`和`const int *a`也被认为是不同的参数，可以在函数重载时通过编译。并且`a`可以在函数中被赋值。

    ```cpp
    void test(int *a)
    {
        a = nullptr;
    }

    void test(const int *a)
    {
        a = nullptr;
    }

    int main()
    {
        return 0;
    }
    ```

    指针其实也是传递了一个副本，但是确能通过编译，但是指针的类型其实依赖于变量的类型，这里变量的类型并不是一个副本变量的类型，而是原变量的类型，因此指针被认为是不同的类型。

1. 默认参数

    ```c++
    void test(int a, int b = 3) {}
    void test(int a) {}
    int main()
    {
        test(3);  // error
        test(3, 2);  // OK
        return 0;
    }
    ```

### 函数调用协议

影响函数参数的入栈方式，栈内数据的清除方式，编译器函数名的修饰规则等。

* `__stdcall`：windows api 默认的函数调用协议

    函数参数由右向左入栈。函数调用结束后由被调用函数清除栈内数据。

    C 语言编译器函数名修饰：`_functionname@number`。其中`number`为参数字节数。

    C++ 编译器的函数名修饰：`?functionname@@YG******@Z`

    其中`******`为函数返回值类型和参数类型表。

* `__cdecl`：c/c++ 默认的函数调用协议

    函数参数由右向左入栈。函数调用结束后由函数调用者清除栈内数据。

    C 编译器函数名修饰：`_functionname`

    C++ 编译器函数名修饰：`?functionname@@YA******@Z`

* `__fastcall`：适用于性能要求较高的场合

    从左开始不大于 4 字节的参数放入 cpu 的 ecx 和 edx 寄存器，其余参数从右向左入栈。

    函数调用结束后由被调用函数清除栈内数据。

    C 编译器函数名修饰：`@functionname@number`

    C++ 编译器函数名修饰：`?functionname@@YI******@Z`

有关栈内数据清除方式的细节：

1. 不同编译器设定的栈结构不尽相同，跨开发平台时由函数调用者清除栈内数据不可行。

1. 某些函数的参数是可变的，如printf函数，这样的函数只能由函数调用者清除栈内数据。

1. 由调用者清除栈内数据时，每次调用都包含清除栈内数据的代码，故可执行文件较大。

显然，如果不进行特殊处理，C 编译器和 C++ 编译器编译出来的函数无法互相调用。

一个例子：

```c++
// 一个 __stdcall 函数的声明
int __stdcall add(int x, int y);

// 函数指针
int (__stdcall *lpAddFun)(int, int);
```

ref: <https://stackoverflow.com/questions/1789807/function-pointer-as-an-argument>

**注：在 windows 下，使用 mingw 的 g++/gcc 进行编译，好像没用，根本没有`@`。有机会再研究。**

### 函数的变长参数（variable arguments）

Variable length argument is a feature that allows a function to receive any number of arguments.

C 语言版本：

```cpp
#include <iostream>
#include <stdarg.h>  // 必须包含这个头文件才能使用变长参数
using namespace std;

float fmin(int elm_num, ...)  // 也可以写成 float fmin(int elm_num...)
{
    float min_val, val;
    va_list args;
    va_start(args, elm_num);
    min_val = va_arg(args, double);  // 即使字面量是 float 类型，这里也强制使用 double
    for (int i = 1; i < elm_num; ++i)
    {
        val = va_arg(args, double);
        if (val < min_val)
            min_val = val;
    }
    va_end(args);
    return min_val;
}

int main()
{
    float min_val = fmin(2.1, 0.5, 1.2);
    cout << "min val is: " << min_val << endl;
    return 0;
}
```

说明：

1. C 语言通过`va_list`，`va_start`，`va_copy`，`va_arg`，`va_end`这 5 个关键字实现变长参数。

    `va_list`用于声明一个特殊变量，操作参数列表。

    `va_start(args, elm_num)`表示从`elm_num`后面的那个参数开始遍历。

    `va_arg(args, type)`根据指定的类型获得参数。注意，传入参数的字面量首先会被强制转化成至少`double`，`int`精度。因此这里的`type`不能填`float`，`short`，`bool`等。

    `va_end()`：释放空间，结束参数的遍历。

    `va_copy()`目前暂时用不上，不知道有什么用。

2. 如果我们执行的是

    ```cpp
    fmin(1, 2, 3);
    ```

    那么函数的输出会是 0。因为字面量`int`类型到`double`类型的转换会出错。

3. 只有完全清楚字面量类型，promotion 类型，`va_arg()`类型之间的转换规则，才有可能写出一个正确的变长参数函数。任何一个环节出错都有可能导致类型转换错误。

    这个挺麻烦的，我的建议是尽量不要使用变长参数。

Ref: 

1. <https://www.scaler.com/topics/cpp/functions-with-variable-number-of-arguments-in-cpp/>

2. <https://www.geeksforgeeks.org/variable-length-argument-c/>

3. <https://www.tutorialspoint.com/cprogramming/c_variable_arguments.htm>

C++ 版本：

基本使用方法：

```cpp
#include <iostream>
using namespace std;

template<typename T>
void print(T param)  // 模板函数的重载，如果只剩一个参数了，那么走这个分支
{
    cout << param << endl;
}

template<typename T, typename ...Ts>  // 使用 typename... 类型名 定义变长参数列表
void print(T param, Ts... params)  // 使用 类型名...形参名 定义函数参数
{
    cout << param << ", ";
    print(params...);  // 使用 形参名... 传递变长参数给下一个函数
}

int main()
{
    print("hello", 123, true);
    return 0;
}
```

输出：

```
hello, 123, 1
```

可以看到，c++ 通过模板 + 递归处理参数的方式实现变长参数，在编译期写好不同参数类型，不同参数长度的函数。

`print(T param, Ts... params)`表示每次处理只处理一个参数，即`param`，剩下的参数`params`递归交给下一个函数处理。如果我们想一次处理两人个参数，当然可以写成`print(T param_1, P param_2)`。

当只剩一个参数时，通过调用重载的模板函数`print(T param)`进行处理，在这里终止递归调用。至此，处理完所有的参数。

这里是每次处理两个参数的例子：

```cpp
#include <iostream>
using namespace std;

template<typename T, typename P>
void print(T param_1, P param_2)
{
    cout << "[" << param_1 << ", " << param_2 << "]" << endl;
}

template<typename T, typename P, typename ...Ts>
void print(T param_1, P param_2, Ts... params)
{
    cout << "[" << param_1 << ", " << param_2 << "]" << ", ";
    print(params...);
}

int main()
{
    print("hello", 123, "world", 456);
    return 0;
}
```

输出：

```
[hello, 123], [world, 456]
```

上面的例子，在传入参数时，都使用按值传递。如果是不涉及内存分配问题，或者体量比较小的参数还好。如果涉及到内存的问题，或者想按引用传递，那么可以使用下面的方法实现完美转发：

```cpp
#include <iostream>
using namespace std;

void aaa(float &val)
{
    cout << "left val: " << val << endl;
}

void aaa(float &&val)
{
    cout << "right val: " << val << endl;
}

template<typename T>
void test(T &&param)
{
    cout << typeid(param).name() << endl;
    aaa(forward<T>(param));
}

template<typename T, typename ...Ts>
void test(T &&param, Ts&&... params)
{
    cout << typeid(param).name() << endl;
    aaa(forward<T>(param));
    test(forward<Ts>(params)...);  // 学习一下变长参数完美转发的方法
}

int main()
{
    float val = 1;
    int val2 = 1;
    test(val, 1.0f, val2, move(val));
    return 0;
}
```

输出：

```
f
left val: 1
f
right val: 1
i
right val: 1
f
right val: 1
```

注意其中的`val2`，虽然是左值，但是`aaa()`接收的是`float`，因此编译器会将`val2`在传入`aaa()`时，进行一次隐式类型转换。类型转换后的值是个右值，所以`aaa()`看到的就是一个右值参数。

### 给函数传递数组

一些 example:

```cpp
#include <iostream>
using namespace std;

void func_1(int arr[], int arr_len) {
    cout << arr[0] << endl;
    arr[0] = 0;
}

void func_2(int arr[][3], int dim0, int dim1) {
    cout << arr[0][0] << endl;
    arr[0][0] = 0;
}

void func_3(int *arr, int dim0, int dim1) {
    cout << *(arr + 0 * dim1 + 0) << endl;
    *(arr + 0 * dim1 + 0)  = 0;
}

void func_4(int (*arr)[3], int dim0, int dim1) {
    cout << arr[0][0] << endl;
    arr[0][0] = 0;
}

void func_5(int (*arr)[3][4], int dim0, int dim1, int dim2) {
    cout << arr[0][0][0] << endl;
    arr[0][0][0] = 0;
}

int main() {
    int arr[3] = {1, 2, 3};
    func_1(arr, 3);
    cout << arr[0] << endl;

    int arr_2[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };
    func_2(arr_2, 2, 3);
    cout << arr_2[0][0] << endl;

    arr_2[0][0] = 1;
    func_3((int*)arr_2, 2, 3);
    cout << arr_2[0][0] << endl;

    arr_2[0][0] = 1;
    func_4(arr_2, 2, 3);
    cout << arr_2[0][0] << endl;

    int arr_3[2][3][4] = {1};
    func_5(arr_3, 2, 3, 4);
    cout << arr_3[0][0][0] << endl;
    return 0;
}
```

输出：

```
1
0
1
0
1
0
1
0
1
0
```

在函数内部无论如何写都无法用`sizeof(arr)`拿到数组的实际长度。

数组作为参数传递时，一直都传递的是指针。

## Pointer and reference 指针与引用

### Basic usage of a pointer

指针常见的用法是指向一个`new`创建的对象：

```cpp
struct MyClass
{
    int m_a;
    int m_b;
};

int main()
{
    int *p = new int(3);
    cout << *p << endl;
    MyClass *pobj = new MyClass({10, 11});
    cout << pobj->m_a << ", " << pobj->m_b << endl;
    delete p;
    delete pobj;
    return 0;
}
```

`new`会返回新创建对象的地址，我们将它赋给指针即可。在不需要使用对象的时候，我们需要手动调用`delete`释放内存。

1. 如果用`new`创建一个对象，但是不使用`delete`，而是用`free`释放掉内存，会发生什么？

    试了下，内存会被正确释放掉，但是不会调用析构函数。如果不递归调用析构函数的话，感觉基类的成员的内存也不会被释放，这样就内存泄露了。

1. 使用`free`释放完内存后，传递给`free()`的指针的值不会改变。

    （想了想确实是这样，传递给`free`的是指针的副本，又不是指针本身，当然不可能被改变）

1. 使用`delete`或`free`释放一个局部变量的内存，会发生什么？

    会在运行时报错。准确地说，在 debug 模式下会报错，在 release 模式下会卡住。

### 指针的运算

对某个变量或对象取完地址后，进行`+`，`-`运算时，并不是按照一个字节一个字节来的，而是按照变量/对象的类型实际的 size 来的。

```cpp
int a = 3;
int *pa = &a;
cout << sizeof(a) << ", " 
    << (long long)pa << ", " 
    << (long long)(pa+1) << endl;
```

Output:

```
4, 612068490332, 612068490336
```

可以看到变量`a`占了 4 个字节，对`pa`进行加 1 时，内存地址的偏移也是 4 个字节。

如果我们对`pa`进行重新解释，那么加减运算的步长也会随之改变：

```cpp
int a = 3;
int *pa = &a;
cout << sizeof(short) << ", " 
    << (long long)pa << ", " 
    << (long long)((short*)pa + 1) << endl;
```

Output:

```
2, 4297063068, 4297063070
```

在上面代码中，我们把`pa`重新解释成`short*`，因为 short 只占用 2 个字节，所以`pa + 1`在内存中也是移动两个字节。

类型转换的优先级要高于`+/-`运算的优先级，所以不需要再加一层括号`((short*)pa)+1`。

### 指针与数组

数组指针：

```cpp
int main() {
    int arr_1[3] = {1, 2, 3};
    int arr_2[3] = {2, 3, 4};

    int *parr_1 = arr_1;
    cout << arr_1[0] << endl;
    parr_1[0] = 0;
    cout << arr_1[0] << endl;

    arr_1[0] = 1;
    int *parr_2[] = {arr_1, arr_2};
    cout << arr_1[0] << endl;
    parr_2[0][0] = 0;
    cout << arr_1[0] << endl;
    
    return 0;
}
```

输出：

```
1
0
1
0
```

### Reference

c++ 规定不允许有元素类型为引用的数组。

Ref: <https://stackoverflow.com/questions/1164266/why-are-arrays-of-references-illegal>

### 函数的返回值为对象

```cpp
#include <iostream>
#include <vector>
using namespace std;

class A
{
    public:
    A(int val) {
        cout << "cons: " << val << endl;
        this->val = val;
        arr.resize(3);
    }

    ~A() {
        cout << "des: " << val << ", " << arr.size() << endl;
    }

    int val;
    vector<int> arr;
};

A func()
{
    return A(1);
}

int main()
{
    A a(2);
    a.arr.resize(4);
    a = func();
    cout << a.arr.size() << endl;
    cout << "here" << endl;
}
```

输出：

```
cons: 2
cons: 1
des: 1, 3
3
here
des: 1, 3
```

可以看到，用函数的返回值去赋值一个对象时，对象会先在函数内部构造，在函数返回时调用析构函数销毁对象。在赋值时，会发生 shadow copy，如果 struct 中原先有 vector，那么函数外部的对象的 vector 会被销毁，然后 shadow copy 成函数内部生成的对象的 vector。

最终退出程序时，再销毁外部的 struct 对象。

也就是说，当我们使用函数的返回值来赋值一个外部 struct 对象时，外部已经存在的对象会被销毁，然后 shadow copy 成函数内部创建的对象。

如果外部对象的构造函数和析构函数有申请/释放内存的操作，那么就要小心了，可能会把已经申请好的内存释放掉。

## 函数，指针与引用

函数指针原始的用法：

```cpp
#include <iostream>
using namespace std;

void print_msg(const char *msg)
{
    cout << msg << endl;
}

int main()
{
    void (*print)(const char *) = print_msg;
    print_msg("hello");
    print("world");
    return 0;
}
```

输出：

```
hello
world
```

使用`typedef`定义函数指针类型的别名：

```cpp
#include <iostream>
using namespace std;

typedef void (*Pfunc)(const char*msg);

void print_msg(const char *msg)
{
    cout << msg << endl;
}

int main()
{
    Pfunc print = print_msg;
    print_msg("hello");
    print("world");
    return 0;
}
```

强制类型转换：

```c
int test(char *msg, int val)
{
    return 0;
}

void *test_2(int *val, float bbb)
{
    return 0;
}

int main()
{
    int (*pfn)(char*, int) = test;
    pfn = (int(*)(char *, int)) test_2;
    return 0;
}
```

让函数返回一个函数指针：

```cpp
#include <iostream>
using namespace std;

typedef void (*Pfunc)(int, int);

int add(int a, int b)
{
    return a + b;
}

int substract(int a, int b)
{
    return a - b;
}

int (*get_calc_op_func(char operation_type))(int, int)
{
    if (operation_type == '+') {
        return add;
    } else if (operation_type == '-') {
        return substract;
    }
    else {
        cout << "unknown operation" << endl;
        return nullptr;
    }
}

int main()
{
    int a = 10, b = 3;
    int (*calc_operation)(int, int) = get_calc_op_func('+');
    cout << calc_operation(a, b) << endl;
    calc_operation = get_calc_op_func('-');
    cout << calc_operation(a, b) << endl;
    return 0;
}
```

输出：

```
13
7
```

使用`typedef`可以让代码更简洁一些：

```cpp
#include <iostream>
using namespace std;

typedef int (*Pfunc)(int, int);

int add(int a, int b)
{
    return a + b;
}

int substract(int a, int b)
{
    return a - b;
}

Pfunc get_calc_op_func(char operation_type)
{
    if (operation_type == '+') {
        return add;
    } else if (operation_type == '-') {
        return substract;
    }
    else {
        cout << "unknown operation" << endl;
        return nullptr;
    }
}

int main()
{
    int a = 10, b = 3;
    Pfunc calc_operation = get_calc_op_func('+');
    cout << calc_operation(a, b) << endl;
    calc_operation = get_calc_op_func('-');
    cout << calc_operation(a, b) << endl;
    return 0;
}
```

如果要返回一个 lambda 表达式，那么没有办法 capture:

```cpp
#include <iostream>
using namespace std;

typedef void (*Pfunc)(const char *msg);

Pfunc get_print_func()
{
    return [](const char *msg) {  // OK
        cout << msg << endl;
    };
}

int main()
{
    Pfunc print = get_print_func();
    print("hello");
    return 0;
}
```

下面这个例子会发生编译错误：

```cpp
#include <iostream>
using namespace std;

typedef void (*Pfunc)();

Pfunc get_print_func(const char *msg)
{
    return [msg]() {  // error, can't capture variables
        cout << msg << endl;
    };
}

int main()
{
    Pfunc print = get_print_func("hello");
    print();
    return 0;
}
```

编译输出：

```
Starting build...
/usr/bin/g++-11 -fdiagnostics-color=always -g *.cpp -o /home/hlc/Documents/Projects/cpp_test/main
main.cpp: In function ‘void (* get_print_func(const char*))()’:
main.cpp:41:5: error: cannot convert ‘get_print_func(const char*)::<lambda()>’ to ‘Pfunc’ {aka ‘void (*)()’} in return
   41 |     };
      |     ^

Build finished with error(s).
```

编译器说没办法将 lambda 表达式转换成函数指针。

这时候就必须要用到 c++ 提供的`std::function`了：

```cpp
#include <iostream>
#include <functional>  // 使用 funtion 必须要加上头文件 functional
using namespace std;

function<void()> get_print_func(const char *msg)
{
    return [msg]() {
        cout << msg << endl;
    };
}

int main()
{
    function print = get_print_func("hello");
    print();
    return 0;
}
```

输出：

```
hello
```

如果不希望填模板参数，还可以直接使用`auto`：

```cpp
#include <iostream>
using namespace std;

auto get_print_func(const char *msg)
{
    return [msg]() {
        cout << msg << endl;
    };
}

int main()
{
    auto print = get_print_func("hello");
    print();
    return 0;
}
```

ref:

1. <https://www.scaler.com/topics/cpp/function-pointer-cpp/>

1. <https://www.learncpp.com/cpp-tutorial/function-pointers/>

1. <https://www.geeksforgeeks.org/function-pointer-in-cpp/>

1. <https://stackoverflow.com/questions/4295432/typedef-function-pointer>

1. <https://www.gamedev.net/forums/topic/687109-function-that-returns-a-function-pointer/>

1. <https://www.geeksforgeeks.org/returning-a-function-pointer-from-a-function-in-c-cpp/>

1. <https://stackoverflow.com/questions/28746744/passing-capturing-lambda-as-function-pointer>

1. <https://www.nextptr.com/question/qa1224899171/converting-captureless-generic-lambda-to-function-pointers>

1. <https://zhuanlan.zhihu.com/p/390883475>

1. <https://www.geeksforgeeks.org/working-and-examples-of-bind-in-cpp-stl/>

### const 左值作为参数参数

如果想让一个函数既接收左值参数，又接收右值参数，还不想重载，也不想写模板实现完美转发，那么可以试一试`const`左值引用。

```cpp
#include <iostream>
#include <string>
using namespace std;

void print(const string &msg)
{
    cout << msg << endl;
}

int main()
{
    print("hello");  // OK, "hello" 会先被转换成匿名对象 string("hello")，然后再转换成 const 左值引用

    string hello_world = "hello, world";
    print(hello_world);  // OK，本身就是左值引用
    return 0;
}
```

想一想，通常右值引用都是匿名对象，我们硬用右值引用改匿名对象也没什么意义。如果懒得定模板，`const`左值引用确实是个比较好的选择。

## Struct

### 在初始化时将 struct 所有字段都置 0

```cpp
#include <iostream>
using namespace std;

struct Struc
{
    int val;
    char *pchar;
};

int main()
{
    Struc struc_1;
    Struc struc_2{};  // 后面加括号，或写成 Struc struc_2 = {}; 的形式，可以让所有字段都赋 0
    cout << struc_1.val << ", " << (int*)struc_1.pchar << endl;
    cout << struc_2.val << ", " << (int*)struc_2.pchar << endl;
    return 0;
}
```

输出：

```
-136146744, 0x7ffff7e28e88
0, 0
```

## Class

### Access

* Public field:

    All the class members declared under public will be available to everyone. The data members and member functions declared public can be accessed by other classes too. The public members of a class can be accessed from anywhere in the program using the direct member access operator (.) with the object of that class.

    外部可以访问到，谁都可以访问。

* Protected field:

    Protected access modifier is similar to that of private access modifiers, the difference is that the class member declared as Protected are inaccessible outside the class but they can be accessed by any subclass(derived class) of that class.

    外部访问不到，只有本类和继承类可以访问。

    Non-public constructors are useful when there are construction requirements that cannot be guaranteed solely by the constructor. For instance, if an initialization method needs to be called right after the constructor, or if the object needs to register itself with some container/manager object, this must be done outside the constructor. By limiting access to the constructor and providing only a factory method, you can ensure that any instance a user receives will fulfill all of its guarantees. This is also commonly used to implement a Singleton, which is really just another guarantee the class makes (that there will only be a single instance).

`struct`:

```cpp
struct STRU
{
    int val;
    double val2;
} s;  // 顺便创建个 STRU 类型的对象叫 s

int main()
{
    // 两种默认初始化的方法
    STRU s2 = {1, 2};
    STRU s3({1, 2});

    STRU sarr[3];  // 结构体数组
    return 0;
}
```

### 构造与析构

#### 构造函数

Ref: <https://www.geeksforgeeks.org/constructors-c/>

有空了看看，查漏补缺。

```cpp
class A
{
    public:
    A() {}
    A(int 3) {}
};

int main()
{
    A a();  // error
    A a;  // OK
    A a = A();  // OK
    A a = 3;  // OK, A a = A(3);
}
```

不能用`A a();`来调用默认构造函数，编译器会认为它是一个函数的声明。因此只能用`A a;`来调用默认构造函数。

注意`A a = A(xxx);`只会调用一次构造函数，而不是先调用构造函数，再调用赋值运算符。（但是别人的代码似乎是先调用了构造函数，然后调用了移动构造函数。奇怪，不清楚哪个版本是对的。）

```cpp
A test_2() {
    A a;
    a.val = 3;
    return a;
}

int main(int argc, char* argv[])
{
    A obj = test_2();
    return 0;
}
```

上面的代码本来需要先调用一次构造函数，再调用一次复制构造函数（如果实现了移动构造函数，会调用移动构造函数）。但是编译器有返回值优化（RVO，Retuan value optimization），所以只会调用一次构造函数。如果想关掉这个优化功能，可以加上`-fno-elide-constructors`参数。

提问：是在`return a;`的时候调用复制/移动构造函数，还是在`A obj = test_2();`时候调用复制/移动构造函数？

#### 析构函数（destructor）

有关容器、构造函数与析构函数的问题：

```cpp
#include <vector>
#include <iostream>
using namespace std;

class A
{
    public:
    A() {
        cout << "in cons" << endl;
    }

    ~A() {
        cout << "in des" << endl;
    }
};

int main()
{
    A a;
    vector<A> vec;
    vec.push_back(a);
    cout << "here" << endl;
    return 0;
}
```

这段代码的输出为：

```
in cons
here
in des
in des
```

可以看到，构造函数调用了一次，析构函数调用了两次。如果某段代码里，构造函数里有申请内存的操作，析构函数里有释放内存的操作，那岂不是会释放两次内存，造成程序崩溃？

如果在`vector`里直接构造对象，就没有问题了：

```cpp
#include <vector>
#include <iostream>
using namespace std;

class A
{
    public:
    A() {
        cout << "in cons" << endl;
    }

    ~A() {
        cout << "in des" << endl;
    }
};

int main()
{
    // A a;
    vector<A> vec;
    vec.emplace_back();
    cout << "here" << endl;
    return 0;
}
```

输出：

```
in cons
here
in des
```

因为`push_back()`至少需要提供一个已经存在的对象，无论是匿名的还是非匿名的，而`emplace_back()`不需要已经存在的对象，所以`push_back()`无法实现和`emplace_back()`相同的功能。

如果容器是`unordered_map`之类的，那么在构造对象时，可以使用`emplace()`加`piecewise_construct`实现原地构造对象。

我们还可以得到结论，对于已经存在的对象，`emplace_back()`和`push_back()`作用相同，调用一次构造函数，两次析构函数。对于不存在的对象，`emplace_back()`只会调用一次析构函数。

#### 拷贝构造函数，复制构造函数（copy constructor）

当出现以下情况时，会调用复制构造函数：

1. 用已存在的对象初始化另一个对象

    ```cpp
    A a;
    A b(a);
    A c = a;
    ```

1. 函数按值传递参数

    ```cpp
    int func(A a) {}
    int main()
    {
        A a;
        func(a);
        return 0;
    }
    ```

1. 函数按值返回对象

编译器提供了默认的复制构造函数，这个构造函数执行的是“浅复制（shadow copy）”：如果成员中有`vector<T>`之类的对象，复制构造函数会递归地调用成员对象的复制构造函数。如果成员中有`int`，`int*`之类的基本类型，那么会直接按值进行复制。

对于`vector`，底层是调用了

```cpp
constexpr vector(const vector& __x)
: _Base(__x.size(),
_Alloc_traits::_S_select_on_copy(__x._M_get_Tp_allocator()))
{
    this->_M_impl._M_finish =
        std::__uninitialized_copy_a(
            __x.begin(), __x.end(),
            this->_M_impl._M_start,
            _M_get_Tp_allocator());
}
```

可以看到，这个复制构造函数是把底层的数据又复制了一遍。

如果我们的类中有指针，那么就需要深复制（deep copy），防止底层数据被多个对象共享，或者同一块内存被多次释放。

#### 赋值运算符（operator=）

**拷贝构造函数**

```cpp
class A
{
    A(const A &obj)
    {
        // do something
    }
};

int main()
{
    A a;
    A b(a);
    A c = A(a);
    A d = c;
}
```

不要用拷贝构造函数初始化一个匿名对象，编译器会认为它是一个对象的声明：

```cpp
A a;
A(a);  // equals A a;
```

调用拷贝构造函数的时机：

当其它类对象作为本类成员，构造时先构造类对象，再构造自身，析构的顺序相反。

有关拷贝构造函数一些需要注意的地方：

1. 传入的参数可以是`const`对象的引用，也可以是非`const`对象的引用。但是通常将形参设置为`const`对象的引用。因为这样既可以传入`const`对象，也可以传入非`const`对象。

C++ 为一个类默认提供 4 个函数：

1. 构造函数

1. 析构函数

1. 拷贝构造函数（按值浅拷贝）

1. 赋值运算符`operator=`（按值浅拷贝）

**析构函数**

先调用当前类的析构函数，再调用成员变量的类的析构函数。

**静态成员**

1. 静态方法只能访问静态成员变量。

1. 静态成员变量在类内声明，在类外初始化。

    ```cpp
    class MyClass
    {
        public:
        static int a;
    }

    int MyClass::a = 1;

    int main()
    {
        cout << MyClass::a << endl;
        return 0;
    }
    ```

1. 静态成员函数可以通过对象访问，也可以通过类名访问。

1. 只有非静态成员变量才属于对象。

1. 空对象占用的内存空间为 1 字节，只有一个成员变量`int a`的对象占 4 个字节。成员函数不占用对象的空间。

1. 如果某个成员方法不涉及到成员属性，可以用空指针访问成员方法（并没有什么用）。

**this**

this 指针的本质是指针常量（`MyClass * const this;`）。

this 指针的用处：

1. 当形参和成员变量同名时，可以用 this 指针来区分
1. 在类的非静态成员函数中返回对象本身，可使用`return *this;`

**常函数与常对象**

```c++
class MyClass
{
    public:
    void show_val() const  // 修饰 this: const MyClass * const this;
    {
        cout << ca << endl;
    }

    void change_val() const
    {
        ca = 1;  // error
        cb = 2;  // OK
    }

    private:
    int ca;
    mutable int cb;
};
```

常对象只能调用常函数：

```c++
class MyClass
{
    public:
    void show_val() const {}
    void change_val() {}
}

int main()
{
    const MyClass obj;
    obj.show_val();  // OK
    obj.change_val();  // error
    return 0;
}
```

**友元**

友元允许类外的一些东西访问类的私有成员。

```c++
class A;
class B;
class C;

class C
{
    public:
    void change_val(A &a);
};

class A
{
    public:
    friend void change_val(A &a);
    friend B;
    friend void C::change_val(A &a);

    public:
    void show_val() {cout << val << endl;}

    private:
    int val;
};

class B
{
    public:
    void change_val(A &a);
};

void B::change_val(A &a)
{
    a.val = 3;
}

void change_val(A &a)
{
    a.val = 2;
}

void C::change_val(A &a)
{
    a.val = 4;
}


int main()
{
    A a;
    a.show_val();
    change_val(a);
    a.show_val();
    B b;
    b.change_val(a);
    a.show_val();
    C c;
    c.change_val(a);
    a.show_val();
    return 0;
}
```

输出：

```
0
2
3
4
```

注意各个类声明的顺序。有时候报错说类名找不到，有时候报错说函数名找不到。

#### 编译器提供的函数

构造函数的创建规则：

1. 若用户定义了构造函数，则不提供默认构造函数，但仍提供默认拷贝构造函数

1. 若用户定义了拷贝构造函数，则不提供其他的构造函数

编译器给每个`class`默认提供 4 种函数：

1. 默认构造函数

1. 默认析构函数

1. 默认拷贝构造函数，对属性进行值拷贝

1. 赋值运算符`operator=`，对属性进行值拷贝

### Memory model of a simple object

```cpp
class A
{
    public:
    A(char a, int b, char c): m_a(a), m_b(b), m_c(c) {}
    void func_1() {}
    void func_2() {}

    char m_a;  // 1 byte, but aligned to 4 bytes
    int m_b;  // 4 bytes
    char m_c;  // 1 byte, but aligned to 4 bytes
};

int main()
{
    A obj_a('h', 456, 'h');
    cout << sizeof(obj_a) << endl;
    printf("%d, %d\n", *(char *)&obj_a, *(int*)((char*)&obj_a+4));
    for (int i = 0; i < 12; ++i)
        printf("%d, ", *((char*)&obj_a + i));
    cout << endl;
    int num = 456;
    for (int i = 0; i < 4; ++i)
        printf("%d, ", *((char*)&num + i));
    printf("\n");
    return 0;
}
```

Output:

```
12
104, 456
104, 0, 0, 0, -56, 1, 0, 0, 104, 1, 0, 0,
-56, 1, 0, 0,
```

我们可以看到，对象`obj_a`占用 12 个字节，在类`A`内部有 3 个成员变量，其中因为`int m_b`占了 4 个节点，所以剩下的两个`char`类型也都按最长的 4 字节对齐。

首先我们取对象地址`&obj_a`，然后将其内存解释为`char`类型：`(char*)&obj_a`，最后对其解引用，即可得到值`'h'`对应的整数 104：`*(char*)&obj_a`。

然后我们尝试把存进去的 456 取出来。首先取对象地址：`&obj_a`，然后将内存解释为`char`类型：`(char*)&obj_a`，这一步是为了在对其进行`+`操作时，步长为 1 个字节。接下来对其加 4，即可跳到第二个成员变量处：`(char*)&obj_a + 4`，最后我们将这个内存解释为`int`类型，并对其解引用，即可得到结果 456：`*(int*)((char*)&obj_a + 4)`。

后面的几行代码，一个字节一个字节地打印出 12 个字节对应的内容，可以验证存储的确实是`'h', 456, 'h'`。

### Memory model of a object whos class inherits from another class

### 杂项 Miscellaneous

* 如果 class 中一个 member function 中有 static 变量，那么这个变量会被所有实例的修改影响

    ```cpp
    #include <iostream>
    using namespace std;

    class A
    {
        public:
        void count() {
            static int val = 0;
            ++val;
            cout << "val is " << val << endl;
        }
    };

    int main()
    {
        A a, b;
        a.count();
        b.count();
        return 0;
    }
    ```

    输出：

    ```
    val is 1
    val is 2
    ```

### 继承

`public`继承无法访问基类中`private`字段的内容，其余字段中的内容不变；`protected`继承无法访问基类中`private`的内容，但会把`public`内容中的属性修改成`protected`；`private`继承无法访问基类中`private`的内容，但会把`public`和`protected`的内容改成`private`属性。

在继承时，父类中所有非静态成员属性都会被子类继承下去，但是被编译器隐藏了，访问不到。

继承中构造和析构的顺序：先构造父类，再构造子类；析构的顺序相反。

在继承时，如果基类没有默认构造函数，那么需要子类在初始化列表里显式调用基类的构造函数：

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    A(int a, int b): a(a), b(b) {}

    int a;
    int b;
};

class B: public A
{
    public:
    B(): A(1, 2) {}  // 在初始化列表中显式调用 A 的构造函数
};

int main()
{
    B obj_b;
    cout << obj_b.a << endl;
}
```

Output:

```
1
2
```

**访问父类中隐藏的变量**

可以加上父类的作用域来访问：`s.Base::m_A`，`s.Base::func()`。

如果子类有同名的成员函数，那么父类中就算有重载的函数，也会被隐藏掉，除非使用基类的作用域来访问。

静态成员的处理方式和非静态成员相同。

**多继承**

`class ChildClass: public Base1, protected Base2;`

多继承中的同名成员也需要用作用域来区分。

**菱形继承问题**

基类`A`，有`B`和`C`两个派生类，`D`又继承于`B`和`C`，此时`D`有两份`A`的数据。

为了让`D`只有一份数据，可以使用虛继承：

```c++
class B: virtual public A;
class C: virtual public A;
```

此时可以不使用作用域运算符来指定`A`的数据，可以直接用`对象.var`来访问`A`中的成员。

此时`B`和`C`会继承一个`vbptr`，指向`vbtable`，其中记录了`A`中`var`的偏移量，从而可以唯一地指定一个偏移量。

**多态**

静态多态：函数重载，运算符重载

动态多态：派生类和虚函数实现运行时多态

静态多态在编译阶段确定函数地址，动态多态在运行阶段确定函数地址。

对于虚函数，可以使用基类的指针/引用调用子类的方法。

子类方法的`virtual`可写可不写。

多态的原理：

基类会有一个`vfptr`指针，指向`vftable`，表内有虚函数的地址。子类函数重写父类的虚函数时，子类中虚函数表内部会替换成子类的虚函数地址。

纯虚函数：

`virtual void func_name() = 0;`

定义了纯虚函数的类称为抽象类。抽象类无法实例化对象。抽象类的子类必须重写抽象类中的纯虚函数，否则也会为抽象类。

虚析构与纯虚析构：

`delete`父类指针的时候，并不调用子类的析构函数，因此会造成子类堆区的内存泄漏。解决方法是把父类的析构函数改成虚函数。

纯虚析构：`virtual ~Animal() = 0;`，但是它还需要一个实现，才能通过编译：

```c++
Animal::~Animal() {}
```

### 虚函数

#### The usage of virtual functions

如果我们定义了虚函数，再配合指向派生类对象的基类指针，那么就可以用基类指针调用派生类的函数。

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    virtual void print()
    {
        cout << "in class A" << endl;
    }
};

class B: public A
{
    public:
    virtual void print()
    {
        cout << "in class B" << endl;
    }
};

class C: public A
{
    public:
    virtual void print()
    {
        cout << "in class C" << endl;
    }
};

int main()
{
    A *ptr = nullptr;
    B b;
    C c;
    ptr = &b;
    ptr->print();
    ptr = &c;
    ptr->print();
    return 0;
}
```

Output:

```
in class B
in class C
```

如果我们写好了框架，那么只需要用基类指针调用派生类中的方法就可以了：

```cpp
#include <iostream>
using namespace std;

class Animal
{
    public:
    virtual void sing() {}
};

class Bird: public Animal
{
    public:
    virtual void sing()
    {
        cout << "bird sing ..." << endl;
    }
};

class Cat: public Animal
{
    public:
    virtual void sing()
    {
        cout << "cat sing ..." << endl;
    }
};

void give_a_performance(Animal* animals[], int n)
{
    for (int i = 0; i < n; ++i)
    {
        animals[i]->sing();
    }
}

int main()
{
    Bird bird;
    Cat cat;
    Animal *animals[] = {&bird, &cat};
    give_a_performance(animals, 2);
    return 0;
}
```

作为一个框架开发者，我们只需要把`class Animal`这个类写出来，然后把`give_a_performance`这个函数使用`Animal`实现，剩下的事我们就不需要管了。用户在使用我们的框架时，只需要写一个类继承`Animal`，并实现`sing()`虚函数，就可以直接调用我们框架中的`give_a_performance`函数了。这种效果非常像回调函数，思想很像控制反转。（是不是可以推导出，凡是要开发一个框架，都需要有一点控制反转的思想在里面？）

#### Mechanism of virtual functions

Virtual functions use virtual function table to find the correct entry address of correspoding function.

```cpp
class A
{
    public:
    virtual void func_a() {}  // 类 B 没有重写这个虚函数，可以预测到类 A 和类 B 的虚函数表中，这个虚函数的地址相同
    virtual void func_b() {}
};

class B: public A
{
    public:
    virtual void func_b() {};  // 类 B 重写了这个虚函数，可以预测到类 A 和类 B 的虚函数表中，这个虚函数的地址不同
    void func_c() {}  // 普通函数将不会出现在虚函数表中
};

int main()
{
    A a;
    B b;

    cout << "Address of objects:" << endl;
    cout << "a: " << &a << endl;  // 不可以用 printf("%x", &a);，printf 好像会截断内存地址的高几位。不清楚为什么。cout 打印出的内存地址是可以在 gdb 里调试的，但是 printf 不能。
    cout << "b: " << &b << endl;
    cout << endl;

    cout << "Class A virtual function table:" << endl;
    void *base = (void *)*(long long*)&a;  // (long long*) 的作用为指示起始地址为 &a 的内存中存储的数据占用 8 个字节，这个解释是为了 *(long long*) 前面那个 * 解引用时候用。不写 (long long*) 的话，编译器不知道 &a 对应的内存中存储着多长字节，什么格式的数据。(void*) 则纯粹是为了说明解引用得到的数据是一个指针。因为所有的指针都占用 8 个字节，所以我们可以把 (void*) 换成 (int*)，或 (char*) 都可以，只不过 base 的类型也要跟着换。
    for (int i = -2; i < 2; ++i)
        cout << "base[" << i << "]: " << (void*)*((long long*)base + i) << endl;  // 这里的 (long long*) 同理，表示 base 指向的对象占 8 个字节（即一个指针）。而 (void*) 是为了告诉 << 运算符，后面跟的是一个指针，一个内存地址。不然 << 会以十进制（long long）的形式打印出内存地址
    cout << endl;

    cout << "Class B virtual function table:" << endl;
    base = (void *)*(long long*)&b;
    for (int i = -2; i < 2; ++i)
        cout << "base[" << i << "]: " << (void*)*((long long*)base + i) << endl;

    return 0;
}
```

Output:

```
Address of objects:
a: 0xa07e1ff9c8
b: 0xa07e1ff9b0

Class A virtual function table:
base[-2]: 0
base[-1]: 0x7ff6a5ec4590
base[0]: 0x7ff6a5ec2b60
base[1]: 0x7ff6a5ec2ba0

Class B virtual function table:
base[-2]: 0
base[-1]: 0x7ff6a5ec45a0
base[0]: 0x7ff6a5ec2b60
base[1]: 0x7ff6a5ec2bd0
```

可以看到，`base[0]: 0x7ff6a5ec2b60`其实就是`func_a()`的地址，类`A`和类`B`的虚函数表中，这个地址是一样的。

而`base[1]`是两个类中`func_b()`的地址，由于`B`对`A`中的`func_a()`进行了重写，因此这两个地址不一样。

大概过程是这样的：先把基类所有虚函数放到当前类的虚函数表里，然后再把当前类里实现的虚函数在表里做替换。注意，基类的普通成员函数和派生类的普通成员函数不会出现在虚函数表中。

另外，`base[-1]`是一个`type_info`类型的对象的地址，用于`typeid()`运算符返回对应的类型。`base[-2]`是一个偏移，目前不清楚是什么意思。有时间了看看。

使用 GDB 查看虚函数表：<https://stackoverflow.com/questions/54079937/how-to-print-virtual-function-of-the-vtable-at-a-specific-address-when-debugging>

说明：

1. 可以用`cout << (void*)&A::func_a << endl;`直接打印出成员函数`func_a()`的地址。这个地址和虚函数表中的地址是一致的。

    但是问题是，为什么无法使用下面的代码输出地址：

    ```cpp
    void (A::*fptr_a)() = &A::func_a;
    cout << (void*)fptr_a << endl;
    ```

    这样只会输出`0x1`。为什么？

### RTTI

RTTI 指的是 Runtime Type Identification，即运行时类型识别。

类型的本质是对内存的解释。C++ 中为了支持面向对象和多态，在代码运行时，某个类型的指针可能并不会指向本类型，这就需要我们动态识别指针所指的类型。在 C++ 中，这一机制被称为 RTTI。

RTTI 的具体实现有两种方式，

1. `typeid()`：返回其表达式或类型名的实际类型

1. `dynamic_cast()`：将基类的指针或引用安全地转换为派生类类型的指针或引用

#### typeid 简介

`typeid()`是一个操作符，用于拿到类型或对象的类型信息：

```cpp
#include <iostream>
using namespace std;

int main()
{
    int a = 3;
    if (typeid(a) == typeid(int))
    {
        cout << "the type of the obj is int" << endl;
    }
    return 0;
}
```

输出：

```
the type of the obj is int
```

`typeid()`返回的实际是一个`std::type_info`类型的对象的引用，

如果表达式的类型是类类型且至少包含一个虚函数，则`typeid`操作符会在运行时动态确定表达式的类型；否则，`typeid`返回表达式的静态类型，在编译期就可以计算。

c++ 标准规定了`type_info`类型必须实现下面四种运算：

```cpp
t1 == t2  // 如果两个对象类型相同，则返回 true，否则返回 false
t1 != t2  // 如果两个对象类型不同，则返回 false，否则返回 true
t.name()  // 返回一个 c-style 字符串，通过一定的规则对类型命名
t1.before(t2)  // t1 是否出现在 t2 之前
```

如果一个基类指针`p`指向一个派生类对象，且基类中有虚函数，那么`typeid(*p)`的类型为派生类，这个过程在运行时确定：

```cpp
#include <iostream>
using namespace std;

class A {};
class B: public A {};

class C {
    virtual void func() {}
};
class D: public C {};

int main()
{
    A *pa = new B;
    if (typeid(*pa) == typeid(A)) {
        cout << "the type of *pa is A" << endl;
    } else if (typeid(*pa) == typeid(B)) {
        cout << "the type of *pa is B" << endl;
    }

    C *pc = new D;
    if (typeid(*pc) == typeid(C)) {
        cout << "the type of *pc is C" << endl;
    } else if (typeid(*pc) == typeid(D)) {
        cout << "the type of *pc is D" << endl;
    }
    return 0;
}
```

输出：

```
the type of *pa is A
the type of *pc is D
```

（这个例子可以看出来，没有虚函数的派生类毫无意义）

`typeid`相关的异常使用`bad_typeid`类型处理。

### 类型转换

c++ 中有关类型转换的 tutorial: <https://cplusplus.com/doc/oldtutorial/typecasting/>

常用的显式类型转换：

`(type)expression`

我们可以在表达式前面加上小括号括起来的类型，进行强制类型转换。

Example:

```cpp

```

四种类型转换运算符：

* `const_cast<type> (expr)`

    The `const_cast` operator is used to explicitly override const and/or volatile in a cast. The target type must be the same as the source type except for the alteration of its const or volatile attributes. This type of casting manipulates the const attribute of the passed object, either to be set or removed.

    只试了试去除掉`const`修饰，好像确实有用。但是结果很奇怪：

    ```cpp
    #include <iostream>
    using namespace std;

    int main()
    {
        const int a = 3;
        int *pa = const_cast<int*>(&a);
        *pa = 4;
        cout << a << endl;
        cout << *pa << endl;
        cout << endl;
        
        *(int*)&a = 5;
        cout << a << endl;
        cout << *(int*)&a << endl;
        cout << *pa << endl;
        return 0;
    }
    ```

    Output:

    ```
    3
    4

    3
    5
    5
    ```

    编译器似乎提前计算好了一些常量表达式中的内容。但是`a`内存里的值确实被修改了。

* `dynamic_cast<type> (expr)`

    The `dynamic_cast` performs a runtime cast that verifies the validity of the cast. If the cast cannot be made, the cast fails and the expression evaluates to null. A `dynamic_cast` performs casts on polymorphic types and can cast a `A*` pointer into a `B*` pointer only if the object being pointed to actually is a B object.

* `reinterpret_cast<type> (expr)`

    The `reinterpret_cast` operator changes a pointer to any other type of pointer. It also allows casting from pointer to an integer type and vice versa.

    C 语言风格的强制类型转换。突出一个乱转换。我愿称之为最强转换。

* `static_cast<type> (expr)`

    The `static_cast` operator performs a nonpolymorphic cast. For example, it can be used to cast a base class pointer into a derived class pointer.

## 运算符重载

**成员函数运算符重载**

* `+`，`-`，`*`，`/`

    ```c++
    class A
    {
        public:
        int operator+(A &a)
        {
            return val + a.val;
        }

        private:
        int val;
    };



    int main()
    {
        A a, b;
        cout << a + b << endl;  // a.operator+(b)
        return 0;
    }
    ```

* `<<`, `>>`

    不要使用成员函数重载`<<`运算符，因为无法实现`cout`在左侧（`obj.operator<<(cout)`相当于`obj << cout`）。

* `++`, `--`

    前置`++`：

    ```c++
    class A
    {
        public:
        A& operator++()  // 前置
        {
            ++val;
            return *this;
        }

        A operator++(int)  // 后置，使用 int 占位参数区分前置，返回值而不是引用
        {
            A temp = *this;
            ++val;
            return temp;
        }

        private:
        int val;
    };

    int main()
    {

    }
    ```

* `=`

    `void operator=(A &obj);`，或者`A& operator=(A &obj);`，可以实现链式赋值。

* `>`，`<`，`==`

    ```c++
    bool opeartor== (Person &p);
    bool opeartor!= (Person &p);
    ```

* `()`函数调用运算符

    ```c++
    class MyPrint
    {
        public:
        void opeartor() (string text)
        {
            cout << text << endl;
        }
    };
    
    int main()
    {
        MyPrint mprint;
        mprint("hello, world");

        // 或者使用匿名函数对象
        MyPrint()("hello, world");
    }
    ```
    
**全局运算符重载**

* `+`, `-`, `*`, `/`

    ```c++
    int operator+(A &a1, A &a2)
    {
        return a1.val + a2.val;
    }

    int main()
    {
        A a, b;
        cout << a + b << endl;  // operator+(a, b)
    }
    ```

* `<<`, `>>`

    ```c++
    ostream& operator<< (ostream& cout, A &a)
    {
        cout << a.val << endl;  // 可以将这个函数作为类 A 的友元，从而可以访问私有变量
        return cout;
    }

    int main()
    {
        A a;
        cout << a << endl;
        return 0;
    }
    ```

* `++`, `--`

    前置递增（返回引用，可以实现链式`++`，如果不想实现链式`++`，返回`void`也是可以的）：

    ```c++
    MyInteger& operator++()
    {
        m_num++;
        return *this;
    }
    ```

    后置递增（使用`int`表示占位参数，返回的时候要返回值，而不是引用，因为`temp`是局部对象）：

    ```c++
    MyInteger operator++(int)
    {
        MyInteger temp = *this;
        m_num++;
        return temp;
    }
    ```



## I/O

**streambuf**

每个`istream`或`ostream`对象都维护着一个`streambuf`对象作为为 stream 的缓冲区。即缓冲区的内容填满后，再进行一次 I/O 操作，并刷新缓冲区，这样可以提高效率。

`<streambuf>`库里定义的 streambuf 有两种，一种是用于处理窄字节的`streambuf`，一种是用于处理宽字节的`wstreambuf`，它们都是模板类`basic_streambuf`的实例化类。

这篇笔记主要关注窄字节的缓冲区。

基类`streambuf`似乎是个抽象类。我们通常使用它的派生类：`filebuf`或`stringbuf`。在创建`stream`对象的时候，会自动内置一个`streambuf`缓冲区对象，当然我们也可以将`stream`对象与我们指定的`streambuf`对象绑定：

Example 1:

对 buffer 和 stream 对象混合输入数据

```c++
#include <iostream>
#include <stringbuf>
#include <sstream>
using namespace std;

int main()
{
    stringbuf buffer;
    ostream os(&buffer);

    buffer.sputn("hello, ", 7);
    os << "world!\n";

    cout << buffer.str();
    return 0;
}
```

注意：`stringbuf`定义在`sstream`中，而不是`streambuf`中。

Example 2:

获取和修改`cout`的缓冲区。

```c++
#include <iostream>
#include <sstream>   // stringbuf 定义在 sstream 中   
using namespace std;

int main ()
{
    stringbuf buffer;
    streambuf *cout_buf = cout.rdbuf();  // 获取 cout 的 streambuf
    cout.rdbuf(&buffer);  // 修改 cout 的 streambuf

    cout << "hello, world!" << endl;  // 此时的数据实际是写入了 buffer 中，不会在屏幕上显示
    cout.rdbuf(cout_buf);  // 将 cout 的缓冲区再改回来，此时会输出到 console
    cout << buffer.str();

    return 0;
}
```

**文件读写相关**：

```c++
// 文件存在，并且没有记录
char ch;
ifs >> ch;
if (ifs.eof())
{
    cout << "文件为空！" << endl;
    this->m_EmpNum = 0;
    this->m_FileIsEmpty = true;
    this->m_EmpArray = NULL;
    ifs.close();
    return;
}
```

### Read formated strings from stdin

* Read strings seperated by `,`, `\n`, and `\t`

    ```cpp
    #include <iostream>
    #include <string>
    using namespace std;

    int main()
    {
        string strs[3];
        for (int i = 0; i < 3; ++i)
            cin >> strs[i];
        for (auto &str: strs)
            cout << str << endl;
        return 0;
    }
    ```

    `cin` will read the character from stream until it encounters ` `, `'\n` or `\t`. `cin` will drop these delimiters even if they are repeated or combined. Then `cin` will parse the string and convert it to the corresponding type.

* Read the whole line

    We can use `std::getline()` to get the whole line.

    Example:

    ```cpp
    #include <iostream>
    #include <string>
    using namespace std;

    int main()
    {
        string str;
        getline(cin, str);
        cout << str << endl;
        return 0;
    }
    ```

* Read strings seperated by specific delimiter

    * Single delimiter

        ```cpp
        #include <iostream>
        #include <string>
        #include <cstdio>
        using namespace std;

        /*
        1,2,3
        4,5,6
        */

        int main()
        {
            int arr[2][3];
            string str;
            int m = 2, n = 3;
            char ch;
            int i = 0, j = 0;
            while (m--)
            {
                str.clear();
                j = 0;
                while (ch = getchar())
                {
                    if (ch == '\n')
                    {
                        arr[i][j++] = stoi(str);
                        break;
                    }
                    if (ch == ',')
                    {
                        arr[i][j++] = stoi(str);
                        str.clear();
                        continue;
                    }
                    str.push_back(ch);
                }
                ++i;
            }

            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    cout << arr[i][j] << ", ";
                }
                cout << endl;
            }
            return 0;
        }`
        ```

        这段代码并不容易写出来。因为我们假设不知道列数，每列都使用`'\n`来判断一行的结束。

    * Multiple delimiter

### Get a char from input stream

All of these functions read a character from input and return an integer value. The integer is returned to accommodate a special value used to indicate failure. The value EOF is generally used for this purpose.

#### `getc()`

It reads a single character from a given input stream and returns the corresponding integer value (typically ASCII value of read character) on success. It returns EOF on failure.

Syntax:

```cpp
int getc(FILE *stream); 
```

Example:

```cpp
#include <stdio.h>
int main()
{
   printf("%c", getc(stdin));
   return(0);
}
```

这个函数需要用户输入字符，并且按下回车后，才会读取第一个字符。

问题：

1. `getc()`可以读取任意的 input stream。那么除了`stdin`之外的 input stream 都有什么呢？有什么实际读取的例子？

#### `getchar()`

The difference between `getc()` and `getchar()` is `getc()` can read from any input stream, but `getchar()` reads from standard input. So `getchar()` is equivalent to `getc(stdin)`.

Syntax:

```cpp
int getchar(void); 
```

Example:

```cpp
int main()
{
   printf("%c", getchar());
   return 0;
}
```

#### `getch()`

getch() is a nonstandard function and is present in conio.h header file which is mostly used by MS-DOS compilers like Turbo C. It is not part of the C standard library or ISO C, nor is it defined by POSIX (Source: http://en.wikipedia.org/wiki/Conio.h)
Like above functions, it reads also a single character from keyboard. But it does not use any buffer, so the entered character is immediately returned without waiting for the enter key.

这个函数不读取缓冲区，会当按键按下的时候立即返回。如果输入中文，程序会直接崩溃。

Syntax:

```cpp
int getch();
```

Example:

```cpp
#include <stdio.h>
#include <conio.h>
int main()
{
   printf("%c", getch());   
   return 0;
}
```

#### `getche()`

Like getch(), this is also a non-standard function present in conio.h. It reads a single character from the keyboard and displays immediately on output screen without waiting for enter key.

`getch()`只读取不回显，`getche()`会回显。这两个函数都会返回 ascii 的 int 值。

Syntax:

```cpp
int getche(void); 
```

Example:

```cpp
#include <stdio.h>
#include <conio.h>
// Example for getche() in C
int main()
{
  printf("%c", getche());
  return 0;
}
```

**文件读写**

常用的 flags:

* `ios::in`：为读文件而打开文件
* `ios::out`：为写文件而打开文件
* `ios::ate`：初始位置：文件尾
* `ios::app`：追加方式写文件
* `ios::trunc`：如果文件存在，先删除，再创建
* `ios::binary`：二进制方式

可以用`|`组合多个 flags。

### file IO

#### C 语言方法

* `fgets`

    ```cpp
    char *fgets(char *str, int n, FILE *stream)
    ```

    头文件：`#include <stdio.h>`

    从`stream`中读取`n - 1`个字节到缓冲区`buf`中，最后一个字节会被填充为`\0`。如果读取了`n - 1`个字节，或者遇到`'\n'`，则返回`str`。如果遇到 EOF，或出错，则返回`NULL`。

    `fgets()`读取的字符串，会包含`\n`。

    宽字符版：`fgetws()`。注意`fgetws()`并不能把文件中的 UTF-8 编码的字符自动转换成 UTF16-LE (Unicode)。它只会占用`wchar_t`的高位字节，并将低位字节置 0。

    `fgets`返回有效指针有两个条件，一个是遇到`\n`，另一个是`buf`读满。

    如果我们希望`fgets()`读完一行就停止，可以使用`fgets()`往缓冲区中循环读入数，然后直到遇到一行结尾就结束。如果`fgets()`没有读到换行，那么会把 buffer 填满，最后一个字节填`'\0'`，如果读到了换行，那么会在 buffer 中间的某个位置填`\n\0`。怎么检测呢？
    
    1. 我们可以使用`strlen()`或`wcslen()`检测 buffer 中字符串的长度，如果长度为`n - 1`，那么说明 buffer 被写满了。此时我们检测倒数第 2 个字符是否为`\n`，如果是`\n`，说明读到了行尾，如果不是，说明没到行尾。如果长度小于`n-1`，那么说明 buffer 没有被写满，一定是到行尾了。

        这种方法每次都至少需要遍历一遍 buffer，效率比较低。

    1. 每次都把 buffer 置零，然后检测倒数第二个字符就可以了。如果字符为`'\0'`或`'\n'`，那么说明读到了结尾。如果不是这两个，那么说明还没到结尾。

        其实不需要把整个 buffer 置零，只需要把倒数第二个字符置 0 就可以了。

        这种解决方法的代码：

        ```cpp
        vector<wstring> read_file_to_lines(string file_path)
        {
            FILE *file = fopen(file_path.c_str(), "r");
            const int BUFSIZE = 128;
            wchar_t buf[BUFSIZE];
            wstring line;
            vector<wstring> lines;
            int i = 0;
            while (!feof(file))
            {
                line.clear();
                while (true)
                {
                    buf[BUFSIZE - 2] = L'\0';
                    fgetws(buf, BUFSIZE, file);
                    line.append(buf);
                    if (buf[BUFSIZE - 2] == L'\0' || buf[BUFSIZE - 2] == L'\n')
                        break;
                }
                lines.push_back(line);
            }
            return lines;
        }
        ```

        稍微改进一下：

        ```cpp
        vector<string> read_file_to_lines(const string &file_path)
        {
            vector<string> lines;
            string line;
            const int BUFSIZE = 128;
            char buf[BUFSIZE];
            FILE *f = fopen(file_path.c_str(), "r");
            while (!feof(f))
            {
                line.clear();
                do
                {
                    buf[BUFSIZE - 2] = 0;
                    fgets(buf, BUFSIZE, f);
                    line.append(buf);
                } while (buf[BUFSIZE - 2] != '\0' && buf[BUFSIZE - 2] != '\n');
                lines.push_back(line);
            }
            return lines;
        }
        ```

* `fread`

    Syntax:

    ```cpp
    size_t fread(void *__restrict__ _DstBuf, size_t _ElementSize, size_t _Count, FILE *__restrict__ _File)
    ```

* 可以使用`ate`的方式打开文件，这样可以用`tellg()`得到文件的大小，从而创建合适的缓冲区

    ```cpp
    string read_file(const char *file_path)
    {
        ifstream ifs(file_path, ios::ate | ios::binary);
        if (!ifs.is_open())
        {
            cout << "fail to open the file " << file_path << endl;
            exit(-1);
        }
        size_t file_size = (size_t) ifs.tellg();
        string buffer;
        buffer.resize(file_size);
        ifs.seekg(0);
        ifs.read(buffer.data(), file_size);
        ifs.close();
        return buffer;
    }
    ```

#### C++ flavor

Ref: 

1. <https://cplusplus.com/doc/tutorial/files/>

2. <https://www.udacity.com/blog/2021/05/how-to-read-from-a-file-in-cpp.html>

    展示了用流式处理，按行处理，单字符处理等多种读取文件的方法。

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

说明：

1. 在手动添加`\n`的时候，如果文件的最后一行的末尾没有`\n`，那么我们读取的`content`会多出一个`\n`，这样会导致读取的内容和原文件不符。

    但是这个问题也不算是什么大问题，目前也没有找到什么比较好的解决办法，所以这段代码还是比较实用的。

#### filesystem (since C++ 17)

没错，c++ 直到 c++17 才支持文件系统。

```cpp
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;

int main()
{
    fs::path p("./main");  // 使用字符串创建 path
    string relative_path = p.c_str();  // path 转换成 c-style string
    string absolute_path = fs::absolute(p).c_str();  // 使用 path 构造绝对路径 absolute
    cout << "relative: " << relative_path << endl;  // 不清楚为啥会多个点，不过路径确实没错
    cout << "absolute: " << absolute_path << endl;

    absolute_path = fs::absolute("./main").c_str();  // 也可以直接使用 C 字符串构造 absolute
    cout << "absolute: " << absolute_path << endl;

    string cur_path = fs::current_path();  // 当前目录，当前目录也可以直接用赋值转换成 string
    cout << fs::current_path() << endl;  // path 也可以直接用 cout 输出
    cout << cur_path << endl;
    return 0;
}
```

输出：

```
relative: ./main
absolute: /home/hlc/Documents/Projects/cpp_test/./main
absolute: /home/hlc/Documents/Projects/cpp_test/./main
"/home/hlc/Documents/Projects/cpp_test"
/home/hlc/Documents/Projects/cpp_test
```

其余的一些功能，判断是文件还是目录，判断是否存在，遍历目录，拿到文件/目录的元信息（权限，inode之类的），新建，删除，读写追加文件，指定工作目录，以及 socket 之类的系统虚拟文件的读写，等有空了再看吧。

还有一些常用的功能，比如拼接路径，修改文件名，路径截取，正斜杠和反斜杠的处理，上一级目录，有空了也看看。

记得编译器要打开 c++ 17 的支持，不然会报错的。 visual studio 默认还在用 c++ 14 的标准。

```cpp
#include <filesystem>
namespace fs = std::filesystem;
```

常用的几个类：

* `fs::path`

    `fs::path`几乎只负责处理字符串，不与文件系统产生交互。

    `fs::path`在`string`的基础上，增加了一些处理文件扩展名，绝对路径与相对路径转换，路径拼接之类的与路径相关的字符串处理功能。

    `fs::path`的路径拼接可以使用`/`完成，也可以使用`append()`方法。
    
    注：
    
    1. cpp reference 中对`append()`和`concat()`的解释：

        `append`, `operator/=`: appends elements to the path with a directory separator
    
        `concat`, `operator+=`: concatenates two paths without introducing a directory separator

        可以看到，`append()`与`/`是自动添加分隔符，而`concat()`与`+`只做字符串拷贝。

        example:

        ```cpp
        #include <filesystem>
        #include <string>
        #include <iostream>
        using namespace std;
        namespace fs = std::filesystem;

        int main()
        {
            fs::path base_dir = "./my_dir";
            string file_name = "hello_world.txt";  // 注意，file_name 不能是 fs::path 类型，不然 append() 会报错
            base_dir.append(file_name);  // append() 只接收各种字符串作为参数，不接收 fs::path 类型
            cout << base_dir << endl;
            return 0;
        }
        ```

        输出：

        ```
        "./my_dir/hello_world.txt"
        ```

        可以看到`file_name`本身没有`/`，但是输出中增加了一个分隔符`/`。

        Ref: <https://en.cppreference.com/w/cpp/filesystem/path/append>

    2. 感觉`path`的设计不是很好，既然底层是`string`，没有什么特殊的数据结构，又必须和`string`打交道，还不能保证`path`对象总是有效，那为什么还要专门设计这个 class？直接用`string` + 路径相关的处理函数不是更好？

* `fs::directory_entry`

    从这个类开始，才开始和系统的文件系统交互。
    
    这个类很奇怪，从名字上看，它是个目录对象，但实际上初始化这个类的，是一个`path`，或者说是一个`string`，因此`directory_entry`可能是一个目录，也可能是一个文件。

    example：判断一个`path`是一个目录还是一个文件

    ```cpp
    #include <filesystem>
    #include <string>
    #include <iostream>
    using namespace std;
    namespace fs = std::filesystem;

    int main()
    {
        fs::path file_path("./hello.txt");
        fs::path dir_path("./world");
        fs::directory_entry file_entry(file_path);
        fs::directory_entry dir_entry(dir_path);
        cout << file_entry << (file_entry.is_regular_file() ? " is " : " isn't ") << "a file." << endl;
        if (dir_entry.is_directory())
        {
            cout << dir_entry << " is a directory." << endl;
        }
        else
        {
            cout << dir_entry << " isn't a directory." << endl;
        }
        return 0;
    }
    ```

    输出：

    ```
    "./hello.txt" is a file.
    "./world" is a directory.
    ```

* `fs::directory_iterator`

    如果我们需要对一个目录进行遍历，那么就需要用到`directory_iterator`对象。

    example:

    ```
    #include <filesystem>
    #include <string>
    #include <iostream>
    using namespace std;
    namespace fs = std::filesystem;

    int main()
    {
        fs::path cur_dir = ".";
        fs::directory_iterator dir_iter(cur_dir);
        for (const fs::directory_entry &dir_entry: dir_iter)  // 只能写成 const 形式
        {
            cout << dir_entry << endl;
        }
        return 9;
    }
    ```

    输出：

    ```
    "./main"
    "./main.cpp"
    "./hello.txt"
    "./world"
    "./.vscode"
    ```

    注：

    1. 官方的 example 也全都写成这种 range for 的形式，虽然`directory_iterator`也可以直接用`->`或`*`获得`directory_entry`对象，也可以配合`std::begin()`和`std::end()`拿到头尾，但是似乎并没有人这样用。

        我们只需要按照官方的 example 使用它就可以了。

    2. `directory_iterator`的行为有点像迭代器（iterator），有点像容器（container），又有点像生成器（generator），其实它就是四不像，难用。

    3. 构建`directory_iterator`时，并不会检查 path 是否有效。如果 path 不存在，那么会直接报 runtime error。

        可以使用`fs::exists()`对 path 的有效性进行检查。

    一个比较系统的教程：<https://www.studyplan.dev/pro-cpp/file-system>

### cout

* cout 指定小数位数

    方案 1：

    ```cpp
    #include <iostream>
    #include <iomanip>
    using namespace std;

    int main()
    {
        float pi = 3.1415926;
        cout << fixed << setprecision(2) << pi << endl;
        return 0;
    }
    ```

    方案 2：

    ```cpp
    #include <iostream>
    using namespace std;

    int main()
    {
        float pi = 3.1415926;
        cout.setf(ios::showpoint);  // optional
        cout.setf(ios::fixed);
        cout.precision(2);  // 3.14
        cout << pi << endl;
        return 0;
    }
    ```

    Ref: <https://stackoverflow.com/questions/5907031/printing-the-correct-number-of-decimal-points-with-cout>

## 模板

**函数模板**

```c++
template<typename T>  // 或 template<class T>
void func(T a) {cout << a << endl;}

int main()
{
    func("hello, world");  // 自动类型推导
    func<int>(1);  // 显式指定类型
}
```

如果模板中只指定了类型，但是并没有使用，那么只能显式调用：

```c++
template<typename T>
void func() {cout << "hello, world!" << endl;}

int main()
{
    func<int>();
    return 0;
}
```

普通函数调用时可以发生隐式类型转换，模板函数在自动类型推导时不会发生隐式类型转换，在显式调用时会发生隐式类型转换。

普通函数与函数横板的调用规则：

1. 如果函数模板和普通函数都可以实现，优先调用普通函数

    ```c++
    void print(int a, int b)
    {
        cout << "normal function" << endl;
    }

    template<typename T>
    void print(T a, T b)
    {
        cout << "template function" << endl;
    }
    ```

1. 可以通过空模板参数列表来强制调用函数模板

    `print<>(a, b);`

1. 函数模板也可以发生重载

    ```c++
    template<typename T>
    void print(T a, T b, T c)
    {
        cout << "template function" << endl;
    }
    ```

1. 如果函数模板可以产生更好的匹配，优先调用函数模板。（能不发生隐式类型转换尽量不发生转换）

如果处理自定义的类型，可以这样定义模板函数，走专用通道：

```c++
template<> bool myCompare(Person &p1, Person &p2)
{
    // ...
}
```

**类模板**

```c++
template<typename NameType, typename AgeType>
class Person
{
    public:
    Person(NameType name, AgeType age)
    {
        this->m_name = name;
        this->m_age = age;
    }

    NameType m_name;
    AgeType m_age;
}
```

类模板只能显式指定类型，不能自动类型推导。类模板可以在类型参数列表中有默认参数。

类模板中的成员函数并不是在一开始就创建的，而是在调用时再创建。

**类模板对象做函数参数**

```c++
// 1. 指定传入类型（最常用）
void printPerson1(Person<string, int> &p)
{
    p.showPershow();
}

// 2. 参数模板化
template<class T1, class T2>
void printPerson2(Person<T1, T2> &p)
{
    p.showPerson();
    cout << "T1: " << typeid(T1).name() << endl;
    cout << "T2: " << typeid(T2).name() << endl;
}

// 3. 整个类模板化
template<class T>
void printPerson3(T &p)
{
    p.showPerson();
}
```

**类模板与继承**

当子类继承的父类是一个类模板时，子类在声明的时候，要指定出父类中`T`的类型。若不指定，编译器无法给子类分配内存。如果想灵活指定出父类中`T`的类型，子类也需变为类模板。

```c++
template<class T>
class Base
{
    T m;
};

class Son: public Base<int>
{

};

template<typename T, typename T2>
class Son2: public Base<T>
{
    T2 obj;
};
```

**类模板成员函数的类外实现**

```c++
template<class T1, class T2>
Person<T1, T2>::Person(T1 name, T2 age)
{
    // ...
}
```

若分文件编写时，要么把类模板中的函数实现都放到头文件中，然后写成`.hpp`文件，要么在第三个文件中`#include "MyTemplateClass.cpp"`。这是因为类模板中成员函数在调用阶段才创建。

**类模板与友元**

```c++
// 全局函数类内实现（建议使用这个）
template<class T1, class T2>
class Person
{
    friend void printPerson(Person<T1, T2> &p)
    {
        cout << p.m_name << endl;
    }
}

// 全局函数类外实现（注意声明的顺序）
template<class T1, class T2> class Person;

template<class T1, class T2>
void printPerson(Person<T1, T2> &p)
{
    cout << p.m_name << endl;
}

template<class T1, class T2>
class Person
{
    // 用 <> 来区分普通函数
    friend void printPerson<>(Person<T1, T2> &p);
};
```

**模板与实例化**

如果我们实现了一个函数模板：

`my_header.h`

```cpp
template<typename T>
void set_ocl_kernel_args(cl_kernel k, T &arg);
```

`my_source.cpp`:

```cpp
#include "my_header.h"

template <typename T>
void set_ocl_kernel_args(cl_kernel k, T &arg)
{
    int size = sizeof(arg);
    T *ptr = &arg;
    printf("size: %d, addr: %p\n", size, ptr);
}
```

如果直接对这两个文件进行编译，由于编译器不知道`T`的具体类型，所以无法确定参数实际占了多少内存，最终无法定位内存地址，从而无法通过编译。

一种解决办法是在头文件里写模板类和函数的实现，另一种办法是在编译时就指定好可能会用到的类型：

`my_source.cpp`:

```cpp
#include "my_header.h"

template <typename T>
void set_ocl_kernel_args(cl_kernel k, T &arg)
{
    int size = sizeof(arg);
    T *ptr = &arg;
    printf("size: %d, addr: %p\n", size, ptr);
}

template void set_ocl_kernel_args<cl_mem>(cl_kernel k, cl_mem &param);
```

最后的这一行叫模板的实例化（instantiation）。

另一个模板实例化的 example：

`my_header.h`:

```cpp
struct OclKernelArg
{
    void *ptr;
    size_t size;
};

struct OclKernelArgCollector
{
    template<typename T> OclKernelArgCollector& sa(T &arg);  // set argument

    vector<OclKernelArg> kernel_args;
};
```

`my_source.cpp`:

```cpp
#include "my_header.h"

template<typename T>
OclKernelArgCollector& OclKernelArgCollector::sa(T &arg)
{
    kernel_args.push_back({&arg, sizeof(arg)});
    return *this;
}

template OclKernelArgCollector& OclKernelArgCollector::sa<cl_mem>(cl_mem &arg);
template OclKernelArgCollector& OclKernelArgCollector::sa<float>(float &arg);
template OclKernelArgCollector& OclKernelArgCollector::sa<char*>(char* &arg);
```

这段代码是对类中的模板成员函数进行实例化。虽然需要对每一个类型都实例化一次，但是这段代码对我来说是可以接受的。如果能有一些字符串处理工具自动填实例类型就更好了。

Ref:

1. <https://stackoverflow.com/questions/4933056/how-do-i-explicitly-instantiate-a-template-function>

1. <https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file>

### c++ 不允许对函数模板偏特化（Partial specialization of function templates is not allowed）

如果我们实现了一个函数模板，但是对于某些指定的类型，我们希望有指定的处理方法，那么就可以用**特化**（注意，这里是特化，不是偏特化）来实现：

```cpp
#include <iostream>
using namespace std;

template<typename T>
T my_add(T a, T b)
{
    cout << "in function template" << endl;
    return a + b;
}

template<>  // 由于不需要用到新的通用类型，所以这里的类型参数列表是空的
float my_add<float>(float a, float b)  // 如果这里写成 float my_add(float a, float b)，那么就变成了函数重载
{
    cout << "in specialization of function template" << endl;
    return a + b;
}

int main()
{
    float a = 1, b = 2;
    float c = my_add(a, b);
    cout << a << " + " << b << " = " << c << endl;

    int v_1 = 1, v_2 = 2;
    int v_3 = my_add(v_1, v_2);
    cout << v_1 << " + " << v_2 << " = " << v_3 << endl;

    return 0;
}
```

输出：

```
in specialization of function template
1 + 2 = 3
in function template
1 + 2 = 3
```

特化要求模板参数列表为空，并且在函数名称后用尖括号指定要特化的参数，并在形参处用特化的类型替换通用类型。

如果模板参数列表里有多个参数，我们可以使用**全特化**来指定要特殊处理的类型：

```cpp
#include <iostream>
using namespace std;

template<typename T, typename P>
T my_round_add(T a, P b)
{
    cout << "in function template" << endl;
    return a + b;
}

template<>
float my_round_add<float, int>(float a, int b)
{
    cout << "in specialization of function template" << endl;
    return (int)(a + 0.5) + b;
}

int main()
{
    float a = 1.5;
    int b = 2;
    float c = my_round_add(a, b);
    cout << a << " + " << b << " = " << c << endl;

    int v_1 = 1, v_2 = 2;
    int v_3 = my_round_add(v_1, v_2);
    cout << v_1 << " + " << v_2 << " = " << v_3 << endl;

    return 0;
}
```

输出：

```cpp
in specialization of function template
1.5 + 2 = 4
in function template
1 + 2 = 3
```

但是如果只想特殊处理部分的模板类型呢，可以吗？

```cpp
#include <iostream>
using namespace std;

template<typename T, typename P>
T my_round_add(T a, P b)
{
    cout << "in function template" << endl;
    return a + b;
}

template<typename P>
float my_round_add<float, P>(float a, int b)
{
    cout << "in specialization of function template" << endl;
    return (int)(a + 0.5) + b;
}

int main()
{
    float a = 1.5;
    int b = 2;
    float c = my_round_add(a, b);
    cout << a << " + " << b << " = " << c << endl;

    int v_1 = 1, v_2 = 2;
    int v_3 = my_round_add(v_1, v_2);
    cout << v_1 << " + " << v_2 << " = " << v_3 << endl;

    return 0;
}
```

上面的代码会在编译时报错：

```
Starting build...
/usr/bin/g++-11 -fdiagnostics-color=always -g *.cpp -o /home/hlc/Documents/Projects/cpp_test/main
main.cpp:39:7: error: non-class, non-variable partial specialization ‘my_round_add<float, P>’ is not allowed
   39 | float my_round_add<float, P>(float a, int b)
      |       ^~~~~~~~~~~~~~~~~~~~~~

Build finished with error(s).
```

只指定部分模板参数的行为叫偏特化。显然函数模板不允许偏特化。

虽然函数不允许偏特化，但是`class`是允许偏特化的：

```cpp
#include <iostream>
using namespace std;

template<typename T, typename P>
struct my_round_add
{
    public:
    T operator()(T a, P b) {
        cout << "in class template" << endl;
        return a + b;
    }
};

template<typename P>
struct my_round_add<float, P>
{
    public:
    float operator()(float a, P b) {
        cout << "in specialization of class template" << endl;
        return (int)(a + 0.5) + b;
    }
};

int main()
{
    float a = 1.5;
    int b = 2;
    float c = my_round_add<float, int>()(a, b);  // 在调用时需指定模板参数
    cout << a << " + " << b << " = " << c << endl;

    int v_1 = 1, v_2 = 2;
    int v_3 = my_round_add<int, int>()(v_1, v_2);
    cout << v_1 << " + " << v_2 << " = " << v_3 << endl;

    return 0;
}
```

输出：

```
in specialization of class template
1.5 + 2 = 4
in class template
1 + 2 = 3
```

可以看到，`class`和`struct`支持模板偏特化，但是需要在调用时指定类型列表。

那么如果我们有一些函数偏特化的需求，该怎么处理呢？可以放弃使用模板，使用函数重载直接实现功能：

```cpp
#include <iostream>
using namespace std;

template<typename T, typename P>
T my_round_add(T a, P b)
{
    cout << "in function template" << endl;
    return a + b;
}

template<typename P>
float my_round_add(float a, P b)  // 使用函数重载，而不是模板的特化
{
    cout << "in overload function" << endl;
    return (int)(a + 0.5) + b;
}

int main()
{
    float a = 1.5;
    int b = 2;
    float c = my_round_add(a, b);
    cout << a << " + " << b << " = " << c << endl;

    int v_1 = 1, v_2 = 2;
    int v_3 = my_round_add(v_1, v_2);
    cout << v_1 << " + " << v_2 << " = " << v_3 << endl;

    return 0;
}
```

输出：

```
in overload function
1.5 + 2 = 4
in function template
1 + 2 = 3
```

编译器会根据函数重载的调用规则，调用最匹配的模板函数实现，这样就解决了偏特化的需求的问题。

### 类模板偏特化的一个例子

假如我想实现一个模板类`class vec3`，当我不指定模板参数时，它默认是`float`类型，当指定模板参数时，使用指定的类型。为了使用`std::array`的一些特性，我希望`vec3`继承`std::array<T, 3>`，并且可以使用列表初始化。想要实现的效果如下：

```cpp
int main()
{
    vec3 v1 = {1, 2, 3};  // float
    vec3 v2 = {1, 2.2, 3};  // float
    vec3 v3<int> = {2, 3, 4};  // int
    vec3 v4<double> = {2, 3, 4};  // double
}
```

实现的方法如下：

```cpp
#include <array>
#include <iostream>
using namespace std;

struct PlaceHolder;

template<typename T = PlaceHolder, typename P = float>  // T 用来占位，判断是否人为指定类型，P 用来绕过 initializer_list
class vec3: public array<T, 3>
{
    public:
    vec3(P args...) = delete;  // 下面的 initializer_list 走不通的话，会走这一行对模板参数进行推导（deduction）
    vec3(initializer_list<P> &&il) {  // 这里的 P 默认就是 float
        cout << typeid(T).name() << ", " << typeid(P).name() << endl;
        for (int i = 0; i < 3; ++i)
            (*this)[i] = *(il.begin() + i);
    }
};

template<typename P>
class vec3<PlaceHolder, P>: public array<float, 3>  // partial specialization，偏特化来实现默认参数类型 float
{
    public:
    vec3(P args...) = delete;
    vec3(initializer_list<float> &&il) {
        cout << "type: float" << endl;
        for (int i = 0; i < 3; ++i)
            (*this)[i] = *(il.begin() + i);
    }
};

template<typename T, typename P>
ostream& operator<<(ostream &cout, vec3<T, P> &vec)
{
    cout << "[" << vec[0] << ", ";
    cout << vec[1] << ", ";
    cout << vec[2] << "]";
    return cout;
}

int main() {
    vec3 v_1{0, 1, 2};  // float
    vec3 v_2{0, 1.2, 3};  // float
    vec3 v_3{1.3, 2, 3};  // float
    vec3<int> v_4 = {3, 4, 5};  // int
    vec3<double> v_5 = {2, 3, 4};  // double

    cout << v_1 << endl;
    cout << v_2 << endl;
    cout << v_3 << endl;
    cout << v_4 << endl;
    cout << v_5 << endl;
    return 0;
}
```

输出：

```
type: float
type: float
type: float
i, f
d, f
[0, 1, 2]
[0, 1.2, 3]
[1.3, 2, 3]
[3, 4, 5]
[2, 3, 4]
```

可以看到虽然看起来实现了，但是会偷偷将 double 精度的类型转换成 float，再转回 double。这样会损失精度。这个方案并不完美。

### traits 技术

在写模板类的时候，如果我们想实现类似

```cpp
if (T.type == float) {
    // do something
} else if (T.type == int) {
    // do something
} else {
    // do another thing
}
```

这样的功能，那么可以使用 traits 技巧把一个类型映射到一个`enum`值上。

```cpp
#include <iostream>
#include <vector>
using namespace std;

class MyType {};

enum Types
{
    tp_float,
    tp_int,
    tp_my_type,
    tp_unkonwn_type
};

template<typename T>
struct type_traits
{
    Types type = tp_unkonwn_type;
};

template<>
struct type_traits<float>
{
    Types type = Types::tp_float;
};

template<>
struct type_traits<int>
{
    Types type = Types::tp_int;
};

template<>
struct type_traits<MyType>
{
    Types type = Types::tp_my_type;
};

template<typename T>
class Container
{
    public:
    Container(T obj) {
        if (type_traits<T>().type == Types::tp_float) {
            cout << "this is a float type" << endl;
        } else if (type_traits<T>().type == Types::tp_int) {
            cout << "this is a int type" << endl;
        } else if (type_traits<T>().type == Types::tp_my_type) {
            cout << "this is a MyType type" << endl;
        } else {
            cout << "unknown type" << endl;
        }
        vec.push_back(obj);
    }

    vector<T> vec;
};

int main()
{
    float val_1 = 1;
    int val_2 = 2;
    MyType val_3;
    double val_4 = 1;
    Container c_1(val_1);
    Container c_2(val_2);
    Container c_3(val_3);
    Container c_4(val_4);
    return 0;
}
```

输出：

```cpp
this is a float type
this is a int type
this is a MyType type
unknown type
```

可以看到，我们使用模板的特化，将指定类型映射到一个整数上，从而对不同的类型选择不同的分支。这个映射的过程发生在编译期，并且不依赖编译器对 rtti 的实现（比如`typeid()`的实现），因此方便又可靠。

Ref: <https://www.zhihu.com/tardis/zm/art/413864991?source_id=1003>

Other resources:

1. <https://leimao.github.io/blog/CPP-Traits/>

### 在使用 trait 时，可以使用 static const 成员避免创建对象

```cpp
#include <stdio.h>
#include <vector>
#include <iostream>
using namespace std;

class MyType {};

enum Types
{
    tp_float,
    tp_int,
    tp_my_type,
    tp_unkonwn_type
};

template<typename T>
struct type_traits
{
    static const Types type = tp_unkonwn_type;
};

template<>
struct type_traits<float>
{
    static const Types type = Types::tp_float;
};

template<>
struct type_traits<int>
{
    static const Types type = Types::tp_int;
};

template<>
struct type_traits<MyType>
{
    static const Types type = Types::tp_my_type;
};

template<typename T>
class Container
{
    public:
    Container(T obj) {
        if (type_traits<T>::type == Types::tp_float) {
            cout << "this is a float type" << endl;
        } else if (type_traits<T>::type == Types::tp_int) {
            cout << "this is a int type" << endl;
        } else if (type_traits<T>::type == Types::tp_my_type) {
            cout << "this is a MyType type" << endl;
        } else {
            cout << "unknown type" << endl;
        }
        vec.push_back(obj);
    }

    vector<T> vec;
};

int main()
{
    float val_1 = 1;
    int val_2 = 2;
    MyType val_3;
    double val_4 = 1;
    Container c_1(val_1);
    Container c_2(val_2);
    Container c_3(val_3);
    Container c_4(val_4);
    return 0;
}
```

output:

```
this is a float type
this is a int type
this is a MyType type
unknown type
```

注：

1. trait 本质还是使用模板的特化路径选择，只不过把这个过程提前了，从而可以在正常的函数里直接使用 enum 来选择 if 分支

2. 如果不使用 trait，或许只能使用`typeid()`了。trait 是在编译时做选择，`typeid`是在运行时做选择，还不太一样。

### 使用 traits 技术使模板对指定的类型生效

```cpp
#include <array>
#include <vector>
#include <iostream>
using namespace std;
typedef array<float, 3> vec3;

template<typename T1, typename T2>
enable_if_t<conjunction_v<
        is_same<remove_reference_t<T1>, vec3>,
        is_same<remove_reference_t<T2>, vec3>>, vec3>
operator+(T1 &&vec_1, T2 &&vec_2) {
    return {vec_1[0] + vec_2[0], vec_1[1] + vec_2[1], vec_1[2] + vec_2[2]};
}

int main() {
    vec3 vec_1, vec_2;
    vec3 vec = vec_1 + vec3{1, 2, 3};
    string hello = "hello";
    const char *str = (hello + ", world").c_str();
    return 0;
}
```

## 谓词

返回`bool`的仿函数称为谓词。接收一个参数的谓词称为一元谓词，接收两个参数的谓词称为二元谓词。

## lvalue and rvalue

简单地说，有变量名的是左值，没有变量名的是右值。

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    int val;
};

void func(A &a)  // 接收一个左值引用
{
    a.val = 2;
}

int main()
{
    A a;  // a 是一个左值
    a.val = 1;
    func(a);
    cout << a.val << endl;
    return 0;
}
```

如果我们想给`func()`传递一个匿名对象（右值），那么就会报错：

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    int val;
};

void func(A &a)
{
    a.val = 2;
}

int main()
{
    func(A());
    return 0;
}
```

报错内容：

```
/usr/bin/g++-11 -fdiagnostics-color=always -g *.cpp -o /home/hlc/Documents/Projects/cpp_test/main
main.cpp: In function ‘int main()’:
main.cpp:17:10: error: cannot bind non-const lvalue reference of type ‘A&’ to an rvalue of type ‘A’
   17 |     func(A());
      |          ^~~
main.cpp:10:14: note:   initializing argument 1 of ‘void func(A&)’
   10 | void func(A &a)
      |           ~~~^
```

此时如果将函数声明`void func(A &a)`改成`void func(A &&a)`，也是可以通过编译并正常运行的：

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    int val;
};

void func(A &&a)
{
    a.val = 2;
}

int main()
{
    func(A());
    return 0;
}
```

目前找到的一个解决办法是：

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    int val;
};

void func(A &a)
{
    a.val = 2;
}

int main()
{
    func(const_cast<A&>(static_cast<const A&>(A())));
    return 0;
}
```

目前并不清楚`static_cast<>`为什么不能直接转换成`A&`，也不是很明白类型转换的原理。

另外一种比较常见的解决方法是完美转发，但是完美转发需要用到模板，这一点不太好。

**Value Categories**

Ref: <https://poby.medium.com/c-lvalue-and-rvalue-aa0c07a095fa>

Value categories properties of expressions

* lvalue

* rvalue

Identity:

* does the expression have a name?

* does the expression have a memory location?

* can you take the address of the expression?

Definition

lvalue

* has a name

* must be able to take its address of memory location via &

```cpp
Foo *pfoo = new Foo;
```

`pfoo` is a lvalue

* the data type is “pointer to Foo”
* It has a name
* It can take the address of the pfoo

`pfoo` is a lvalue

even if it doesn't have a name, we can still take the address of `*pfoo`.

```cpp
int i = 1; // i is lvalue
i = 0; // lvalue can be assigned with a value
int* p = &i; // lvalue can take its address of memory location via &
 
// function returning a reference is lvalue
int& foo();
foo() = 9; // foo() is an lvalue hence it can get assigned
int* p = &foo(); // foo() is lvalue hence it can take it's address
```

rvalue

* does NOT have a name
* can’t take its address
* literals (such as 2, “hello world”, true, or nullptr)
* can’t be assigned with a value
* temporary objects returned from functions.
* lifetime usually ends with the current statement

```cpp
int j = 0;
j = 4; // j is lvalue// a function returning a value is rvalue
int boo();
j = boo(); // np, j is lvalue, boo() is an rvalue
int* p = &boo(); // error, cannot take the address of an rvalue
```

获得右值的地址的一个方法：

```cpp
string content;
const char ** addr = (const char**)&static_cast<const char* const &>(content.c_str());
```

看起来是将右值转换成了一个 const 左值引用，再对 const 左值引用取 const 地址。

**Pass by value vs Pass by reference**

Pass by Value

```cpp
class Foo{};
void func(Foo f);    // f is passed by value
Foo foo;             // foo is lvalue
func(foo);           // call with an lvalue - valid
func(Foo{});         // call with an rvalue - valid
```

Pass by Reference

```cpp
class Foo{};
void func(Foo& f);   // f is passed by lvalue reference
Foo foo;             // foo is lvalue
func(foo);           // call with an lvalue - valid
func(Foo{});         // call with an rvalue - error
```

to fix the above,

void func(const Foo& f); // f is passed by const reference

or

```cpp
class Foo{};
void func(Foo&& f);  // f is passed by rvalue reference
Foo foo;             // foo is lvalue
// func(foo);        // call with an lvalue - error
func(std::move(foo)) // need to cast lvalue foo to rvalue using move
func(Foo{});         // call with an rvalue - valid
```

References

lvalue reference

* called method or function can/will modify the data
* the caller will observe any modifications made

const reference

* called method or function can NOT modify the data

rvalue reference

* called method or function can/will modify the data
* the caller will not and should not observe any modification made
* declared using &&

C++ 11 extended the notion of rvalues by letting you bind an rvalue (category) to an rvalue reference.

`std::move()`可以将一个左值变成一个右值。如果我们的函数接收的参数为右值引用，但是要传进去的是一个左值，那么就可以用`move`将它变成一个右值：

```cpp
#include <iostream>
using namespace std;

void add(int &&a, int &&b, int &&c)
{
    c = a + b;
}

int main()
{
    add(1, 2, 3);  // OK
    int a = 1, b = 2, c;
    add(move(a), move(b), move(c));  // OK
    cout << "c is " << c << endl;  // c is 3
    return 0;
}
```

下面这个 example 会无法通过编译：

```cpp
#include <iostream>
using namespace std;

void add(int &a, int &b, int &c)
{
    c = a + b;
}

int main()
{
    add(1, 2, 3);  // error
    cout << "c is " << c << endl;
    return 0;
}
```

This is invalid. Since the callee can change but the caller can’t see the change made by the callee. So, this is wrong.

注：我觉得这句话并不是关键，关键在于，右值本来就是匿名对象，而匿名对象其实是一个“中间量”，它会被自动地创建，又被自动地销毁，我们不能知道，也没必要知道它的内存占用。右值引用则让我们多了一份选择，匿名对象可以自动被创建，但是当你被销毁时，请留下你已经创建好的内存。右值引用只是多了一个处理引用的入口。

左值引用说明对象是在函数外部申请的内存，由函数外部处理。右值引用说明对象是编译器自动创建的内存，至于这个内存要不要销毁，要看编程者的选择。

显然，如果我们`move`一个左值，就是在告诉函数，这个对象在函数外面本来是手动申请内存的，但是现在把这个内存控制权交给函数内部了，函数内部可以拿走它的内存，也可以不拿走。c++ 并不保持`move`过后的对象仍有效，这需要程序员自己保证。

```cpp
#include <iostream>
using namespace std;

class A
{
    public:
    int val;
};

void modify_rvalue(A &&obj)
{
    delete &obj;
}

int main()
{
    A a;
    a.val = 3;
    cout << a.val << endl;
    modify_rvalue(move(a));
    cout << a.val << endl;
    return 0;
}
```

比如上面这段代码，运行时输出为：

```
3
free(): invalid pointer
Aborted (core dumped)
```

其实我们将对象`move`到函数内部后，这个对象已经被销毁了，但是函数外部的我们是不知道的。c++ 并不能保证 move 完后的对象仍是有效的。

一种解决办法是在参数声明处加上`const`修饰：

```cpp
#include <iostream>
#include <string>
using namespace std;

void print(const string &str)
{
    cout << str << endl;
}

int main()
{
    print("hello");
    return 0;
}
```

另一种方法是同时提供 lvalue 和 rvalue 作为函数参数：

```cpp
#include <iostream>
#include <string>
using namespace std;

void print(string &str)
{
    cout << str << endl;
}

void print(string &&str)
{
    cout << str << endl;
}

int main()
{
    print("hello");
    return 0;
}
```

这样看来，只需要把所有的函数都声明成右值参数，如果遇到左值实参，只需要`move()`将其变成右值就可以了。

## key words explanation

### `using`

c++ 中 using 的三种用法:

1. 导入命名空间

    ```cpp
    using namespace std;
    using std::cout;
    ```

2. 指定类型别名

    相当于`typedef`

    ```cpp
    typedef int T;
    using T = int;
    ```

3. 在派生类中引用基类成员

    ```cpp
    #include <iostream>
    using namespace std;

    class A
    {
        public:
        void print_val() { cout << val << endl; }

        protected:
        int val;
    };

    class B: private A
    {
        public:
        using A::val;
        using A::print_val;
    };

    int main()
    {
        B b;
        b.val = 3;
        b.print_val();
        return 0;
    }
    ```

    输出：

    ```
    3
    ```

    上面的代码中，`A`中的`print_val()`和`val`都对`B`可见，但是`B`使用 private 继承，会使得`A`中的`public`字段和`protected`字段在`B`中都变成`private`字段。

    但是我们在`class B`中，在`public`字段中使用`using`重新引用`val`和`print_val`这两个名字，可以将这两个成员变成`public`字段，从而对外可见。
    
### `namespace`

`namespace`是为了同名变量互不干扰：

```cpp
#include <iostream>
using namespace std;

namespace haha
{
    int val;
    void print_val() {
        cout << val << endl;
    }
}

namespace hehe
{
    int val;
    void print_val() {
        cout << val << endl;
    }
}

int main()
{
    haha::val = 1;
    haha::print_val();

    hehe::val = 2;
    hehe::print_val();
    return 0;
}
```

输出：

```
1
2
```

`namespace`的大括号后面没有分号`;`。

可以在`namespace`中声明变量，函数，类，在`namespace`外面进行定义或初始化。

```cpp
#include <iostream>
using namespace std;

namespace haha
{
    void print_hello();
    class A;
}

void haha::print_hello() {
    cout << "hello" << endl;
}

class haha::A {
    public:
    void print_val() {
        cout << val << endl;
    }
    int val;
};

int main()
{
    haha::print_hello();

    haha::A a;
    a.val = 123;
    a.print_val();
    return 0;
}
```

output:

```
hello
123
```

类成员可以在`namespace`中声明，也可以在外部声明：

```cpp
#include <iostream>
using namespace std;

namespace haha
{
    class A;
    class B {  // 在内部声明类成员
        public:
        int val;
        void print_val();
    };
}

class haha::A {  // 在外部声明类成员
    public:
    void print_val() {
        cout << val << endl;
    }
    int val;
};

void haha::B::print_val() {
    cout << val << endl;
}

int main()
{
    haha::A a;
    a.val = 123;
    a.print_val();
    haha::B b;
    b.val = 456;
    b.print_val();
    return 0;
}
```

output:

```
123
456
```

namespace 可以嵌套，还可以使用`using`来指定使用哪个名字：

```cpp
#include <iostream>
using namespace std;

namespace haha
{
    void print_hello() {
        cout << "hello" << endl;
    }

    namespace hehe
    {
        void print_world() {
            cout << "world" << endl;
        }
    }
}

using haha::print_hello;
using namespace haha::hehe;

int main()
{
    print_hello();
    print_world();
    return 0;
}
```

output:

```
hello
world
```

`using`相当于预处理命令，并不是真正的程序命令，因此想要在运行时更换 namespace 是做不到的：

```cpp
#include <iostream>
using namespace std;

namespace haha
{
    void print_msg() {
        cout << "hello" << endl;
    }
}

namespace hehe
{
    void print_msg() {
        cout << "world" << endl;
    }
}

int main()
{
    using haha::print_msg;
    print_msg();
    using hehe::print_msg;
    print_msg();
    return 0;
}
```

这段代码会报编译错误，第二个`print_msg()`有歧义。因为在`using hehe::print_msg;`后，`print_msg`有了两处来源，因此会报`ambiguous`错误。但是第一个`print_msg();`是没问题的。因此只删去第二处`print_msg();`，程序也可以正常运行。

## C++11/17/20 new features

### move

Suppose we want to traverse a set of vectors, a simple thought is to use reference binding:

```cpp
#include <vector>
#include <iostream>
using namespace std;

int main()
{
    vector<int> vecs[3] = {
        vector<int>({1, 2, 3}), 
        vector<int>({4, 5, 6}),
        vector<int>({7, 8, 9})};
    for (auto &vec: vecs)
    {
        for (auto &item: vec)
        {
            cout << item << ", ";
        }
        cout << endl;
    }
    return 0;
}
```

Output:

```
4, 5, 6,
4, 5, 6,
7, 8, 9,
```

This method does make sense. But for a reference `&`, it is actually a constant pointer binding with each element in `vecs`. However, every pointer occupies 8 bytes in a 64-bit system. For the vector array with 3 elements, it needs create an 8-byte pointer and destroy that in every round of `for` loop. This is low effeciency.

How about moving a new object to a bound reference?

```cpp
vector<int> &rvec = vecs[0];
rvec = vecs[1];
```

Unfortunately, this method doesn't make sense. Because `rves` is equal to `vecs[0]`, the second line just means `vecs[0] = vecs[1];`, and the data of `vecs[0]` will be replaced with the data of `vecs[1]`.

We can also use pointer, but pointer is not safe.

How to only move the ownership of 'allocated memory' to a new variable?

In C++, we can ragard the 'allocated memory' as a right value, and a named variable with allocated memory as a left value.

C++11 provided a function `move` which can convert a left value to a right value, so that we can reassign the allocated memory to another variable.

`move`的源代码：

```cpp
template<typename _Tp>
_GLIBCXX_NODISCARD
constexpr typename std::remove_reference<_Tp>::type&&
move(_Tp&& __t) noexcept
{ return static_cast<typename std::remove_reference<_Tp>::type&&>(__t); }
```

我们可以看到，`move`仅仅是做了一个强制类型转换，把左值转换成右值引用并返回。

紧接着，如果某个类型实现了移动构造函数，会直接调用移动构造函数，而不是拷贝构造函数：

```cpp
A& operator=(const A &&obj) {}
```

在移动构造函数中，我们需要实现的是，把`obj`的 allocator 的指针赋给当前对象`*this`。如果某个对象的 allocator 不是我们写的，我们还需要在移动构造函数内继续调用`move`：`this -> xxx = move(obj.xxx);`，直到找到别人的移动构造函数为止。

注：短字符串并不一定会一直调用移动构造函数，这个叫做 SSO 优化：<https://stackoverflow.com/questions/21694302/what-are-the-mechanics-of-short-string-optimization-in-libc>。

实验：

```cpp
void move_test(std::string&& s) {
    std::string s2 = std::move(s);
    std::cout << "; After move: " << std::hex << reinterpret_cast<uintptr_t>(s2.data()) << std::endl;
}

int main()
{
    std::string sbase;

    for (size_t len=0; len < 32; ++len) {
        std::string s1 = sbase;
        std::cout << "Length " << len << " - Before move: " << std::hex << reinterpret_cast<uintptr_t>(s1.data());
        move_test(std::move(s1));
        sbase += 'a';
    }
}
```

有时间了可以看看。

不可以返回函数内部值的右值，因为对象会在函数结束时被销毁：

```cpp
vector<int>&& test()
{
    vector<int> aaa;
    return move(aaa);
}

int main(int argc, char* argv[])
{
    vector<int> aaa = test();
    return 0;
}
```

### 完美转发

一个右值被作为参数传入函数中后，由于它有了名字，所以它会变成一个左值。

```cpp
#include <iostream>
#include <utility>
using namespace std;

void test_move_2(string &&str)
{
    cout << "in test move 2, right value" << endl;
}

void test_move_2(string &str)
{
    cout << "in test move 2, left value" << endl;
}

void test_move(string &&str)
{
    cout << "in test move, right value" << endl;
    test_move_2(str);
}

void test_move(string &str)
{
    cout << "in test move, left value" << endl;
    test_move_2(str);
}

int main()
{
    string str(500, 'a');
    test_move(move(str));
    return 0;
}
```

Output:

```
in test move, right value
in test move 2, left value
```

这显然与我们的想法不符。我们希望一个右值引用被传入其他函数后，仍然是右值引用；一个左值引用传入其他函数后，仍然是左值引用。有一个简单的方法，我们可以在处理右值引用的函数里，再次调用`move`：

```cpp
#include <iostream>
#include <utility>
using namespace std;

void test_move_2(string &&str)
{
    cout << "in test move 2, right value" << endl;
}

void test_move_2(string &str)
{
    cout << "in test move 2, left value" << endl;
}

void test_move(string &&str)
{
    cout << "in test move, right value" << endl;
    test_move_2(move(str));  // 在这里加上 move，使得左值再次变成右值
}

void test_move(string &str)
{
    cout << "in test move, left value" << endl;
    test_move_2(str);
}

int main()
{
    string str(500, 'a');
    test_move(move(str));
    return 0;
}
```

Output:

```
in test move, right value
in test move 2, right value
```

假如我们的`test_move`函数并没有实际处理内存，而只是希望下一级函数可以处理内存。那么当前函数就不需要关心传入的参数是左值还是右值了，只要按原样转发给下一级函数就可以了。这种转发称为完美转发。通常由`forward()`函数配合模板实现：

```cpp
#include <iostream>
#include <utility>
using namespace std;

void test_move_2(string &&str)
{
    cout << "in test move 2, right value" << endl;
}

void test_move_2(string &str)
{
    cout << "in test move 2, left value" << endl;
}

template<class T>
void test_move(T &&str)
{
    cout << "in test move, no matter what type of value" << endl;
    test_move_2(forward<T>(str));
}

int main()
{
    string str(500, 'a');
    test_move(str);
    cout << "------" << endl;
    test_move(move(str));
    return 0;
}
```

输出：

```
in test move, no matter what type of value
in test move 2, left value
------
in test move, no matter what type of value
in test move 2, right value
```

我们可以看到，左值被转发成了左值，右值被转发成了右值，而`test_move()`本身不需要关心传入的参数是什么值。这样我们就实现了完美转发。

其中，

1. `forward()`是 STL 库`<utility>`中的一个函数，可以按原类型进行转发引用

1. `T&&`指的是通用引用，既可以是左值引用，也可以是右值引用。而`T&`只代表左值引用。为了使用这个特性，使得`test_move`既接受左值引用，也接受右值引用，我们需要引入模板语法。

### c++ 17 初始化一个 struct

    指定字段进行初始化：

    ```cpp
    #include <iostream>
    using namespace std;

    struct A
    {
        int val;
        int val_2;
    };

    int main()
    {
        A a = {
            .val = 1,
            .val_2 = 2
        };

        cout << a.val << endl;
        return 0;
    }
    ```

## Multithreading programming 多线程

### thread

```cpp
#include <iostream>
#include <thread>
using namespace std;

void print_1(int a)  // use a normal function
{
    cout << "method 1: " << a << endl;
}

struct MyFunc  // use a pseudo-funciton
{
    void operator()(int a) {
        cout << "method 2: " << a << endl;
    }
} print_2;

int main()
{
    auto print_3 = [](int a){  // use a function object
        cout << "method 3: " << a << endl;
    };

    thread thd_1(print_1, 111);
    thread thd_2(print_2, 222);
    thread thd_3(print_3, 333);
    thd_1.join();
    thd_2.join();
    thd_3.join();
    return 0;
}
```

Output：

```
method 1: method 2: 222111

method 3: 333
```

Notice that because `iostream` is not thread-safe, the output is disordered.

Notes:

1. 如果不调用`thread_obj.join()`，主线程不会等子线程结束。在整个进程退出时，子线程会被强制结束。

Determain whether the current thread is the main thread:

```cpp
#include <iostream>
#include <thread>
using namespace std;

thread::id main_id = this_thread::get_id();

void print_1(int a)
{
    thread::id id = this_thread::get_id();
    if (id == main_id)
    {
        cout << "this is main thraed" << endl;
    }
    else
    {
        cout << "this is child thread with incoming parameter: " << a << endl;
    }
}

int main()
{
    thread thd(print_1, 123);
    print_1(321);
    thd.join();
    return 0;
}
```

**按引用传递参数**

```cpp
#include <iostream>
#include <thread>
using namespace std;

struct A
{
    int val;
};

void thread_func(A &obj)
{
    obj.val = 2;
}

int main()
{
    A obj;
    obj.val = 1;

    thread thd(thread_func, ref(obj));  // 如果直接传递 obj，会编译时报错
    thd.join();
    cout << obj.val << endl;

    return 0;
}
```

我们可以让线程函数接受引用类型的参数，但是在传递参数的时候需要加上`ref()`。原因我不是很懂。

具体的解释：<https://stackoverflow.com/questions/61985888/why-the-compiler-complains-that-stdthread-arguments-must-be-invocable-after-co>

### Mutex and semephore

如果我们想使用三个线程把一个数字从 0 加到 15，那么可以这样写代码：

```cpp
#include <thread>
#include <iostream>
using namespace std;

int idx = 0;

void print_idx()
{
    for (int i = 0; i < 5; ++i)
    {
        ++idx;
        cout << "idx is: " << idx << endl;
    }
}

int main()
{
    thread thds[3];
    for (int i = 0; i < 3; ++i)
    {
        thds[i] = thread(print_idx);
    }
    for (int i = 0; i < 3; ++i)
    {
        thds[i].join();
    }
    return 0;
}
```

可是这样的话程序的输出总是很不稳定：

```
idx is: idx is: idx is: 23
idx is: 
idx is: 5
2idx is: 6
idx is: 7
idx is: 8
4

idx is: 10
idx is: idx is: 11
idx is: 12
10idx is: 13

idx is: 14
idx is: 15
```

如果我们想要稳定的输出，可以使用`mutex`：

```cpp
#include <mutex>
#include <thread>
#include <iostream>
using namespace std;

int idx = 0;
mutex mtx;

void print_idx()
{
    for (int i = 0; i < 5; ++i)
    {
        mtx.lock();
        ++idx;
        cout << "idx is: " << idx << endl;
        mtx.unlock();
    }
}

int main()
{
    thread thds[3];
    for (int i = 0; i < 3; ++i)
    {
        thds[i] = thread(print_idx);
    }
    for (int i = 0; i < 3; ++i)
    {
        thds[i].join();
    }
    return 0;
}
```

这样就能得到稳定的输出：

```
idx is: 1
idx is: 2
idx is: 3
idx is: 4
idx is: 5
idx is: 6
idx is: 7
idx is: 8
idx is: 9
idx is: 10
idx is: 11
idx is: 12
idx is: 13
idx is: 14
idx is: 15
```

在这个例子中，`mutex`的作用其实是开启了一个临界区，使用`mtx.lock()`和`mtx.unlock()`配对保护临界区中的操作。

对于这种配对使用的场景，还可以使用`locl_guard`更便捷地实现：

```cpp
#include <mutex>
#include <thread>
#include <iostream>
using namespace std;

int idx = 0;
mutex mtx;

void print_idx()
{
    for (int i = 0; i < 5; ++i)
    {
        lock_guard<mutex> mtx_guard(mtx);
        ++idx;
        cout << "idx is: " << idx << endl;
    }
}

int main()
{
    thread thds[3];
    for (int i = 0; i < 3; ++i)
    {
        thds[i] = thread(print_idx);
    }
    for (int i = 0; i < 3; ++i)
    {
        thds[i].join();
    }
    return 0;
}
```

输出：

```
idx is: 1
idx is: 2
idx is: 3
idx is: 4
idx is: 5
idx is: 6
idx is: 7
idx is: 8
idx is: 9
idx is: 10
idx is: 11
idx is: 12
idx is: 13
idx is: 14
idx is: 15
```

`mtx_guard`对象的存在就保证了这个作用域是个临界区，当`mtx_guard`被销毁时，`mtx`会被自动释放。

Ref: <https://www.modernescpp.com/index.php/prefer-locks-to-mutexes/>

mutex 中的`try_lock()`可以无阻塞地尝试 lock，相反，`lock()`则是阻塞式地尝试 lock。

考虑这样一个场景：如果我们希望线程 1 完成填充数组中的数据后，线程 2 对数组中数据进行计算。这时候我们发现 mutex 就不太适用了。使用 mutex 可能会这样写：

```cpp
#include <mutex>
#include <thread>
#include <iostream>
using namespace std;

mutex mtx;
float arr[4];

void fill_array()
{
    mtx.lock();
    for (int i = 0; i < 4; ++i)
    {
        arr[i] = i;
    }
    mtx.unlock();
}

void get_array_sum()
{
    mtx.lock();
    int s = 0;
    for (int i = 0; i < 4; ++i)
    {
        s += arr[i];
    }
    mtx.unlock();
    cout << "sum is " << s << endl;
}

int main()
{
    thread thd_1(fill_array);
    thread thd_2(get_array_sum);
    
    thd_2.join();
    thd_1.join();
    return 0;
}
```

其实这和临界区也没有什么区别了。或许条件变量（conditional variable）可以解决这个问题，有时间了看看。

### 线程函数按引用传递参数相关的内存问题

如果传递的参数是局部变量，那么局部变量会在函数执行前就申请好，并不会因为离开了一个函数中的大括号括起来的作用域就被释放。此时在线程子函数中，参数的引用并不会失效。

如果传递的参数是使用`new`，`malloc()`等动态申请的内存，那么变量的内存在子线程外部被释放掉后，参数的引用会失效，之后的行为都是未定义的。

### ref 与 cref

如果需要按引用向线程函数传递参数，一个朴素的想法可能是这样的：

```cpp
#include <thread>
#include <iostream>
using namespace std;

struct A
{
    int val;
};

void thd_func(A &a)
{
    a.val = 2;
}

int main()
{
    A a;
    a.val = 1;
    thread thd = thread(thd_func, a);
    thd.join();
    cout << a.val << endl;
    return 0;
}
```

但是这样会编译出错：

```
Starting build...
/usr/bin/g++-11 -fdiagnostics-color=always -g *.cpp -o /home/hlc/Documents/Projects/cpp_test/main
In file included from /usr/include/c++/11/thread:43,
                 from main.cpp:32:
/usr/include/c++/11/bits/std_thread.h: In instantiation of ‘std::thread::thread(_Callable&&, _Args&& ...) [with _Callable = void (&)(A&); _Args = {A&}; <template-parameter-1-3> = void]’:
main.cpp:50:36:   required from here
/usr/include/c++/11/bits/std_thread.h:130:72: error: static assertion failed: std::thread arguments must be invocable after conversion to rvalues
  130 |                                       typename decay<_Args>::type...>::value,
      |                                                                        ^~~~~
/usr/include/c++/11/bits/std_thread.h:130:72: note: ‘std::integral_constant<bool, false>::value’ evaluates to false
/usr/include/c++/11/bits/std_thread.h: In instantiation of ‘struct std::thread::_Invoker<std::tuple<void (*)(A&), A> >’:
/usr/include/c++/11/bits/std_thread.h:203:13:   required from ‘struct std::thread::_State_impl<std::thread::_Invoker<std::tuple<void (*)(A&), A> > >’
/usr/include/c++/11/bits/std_thread.h:143:29:   required from ‘std::thread::thread(_Callable&&, _Args&& ...) [with _Callable = void (&)(A&); _Args = {A&}; <template-parameter-1-3> = void]’
main.cpp:50:36:   required from here
/usr/include/c++/11/bits/std_thread.h:258:11: error: no type named ‘type’ in ‘struct std::thread::_Invoker<std::tuple<void (*)(A&), A> >::__result<std::tuple<void (*)(A&), A> >’
  258 |           _M_invoke(_Index_tuple<_Ind...>)
      |           ^~~~~~~~~
/usr/include/c++/11/bits/std_thread.h:262:9: error: no type named ‘type’ in ‘struct std::thread::_Invoker<std::tuple<void (*)(A&), A> >::__result<std::tuple<void (*)(A&), A> >’
  262 |         operator()()
      |         ^~~~~~~~

Build finished with error(s).
```

此时可以使用`ref`来完成这个功能：

```cpp
#include <thread>
#include <iostream>
using namespace std;

struct A
{
    int val;
};

void thd_func(A &a)
{
    a.val = 2;
}

int main()
{
    A a;
    a.val = 1;
    thread thd = thread(thd_func, ref(a));
    thd.join();
    cout << a.val << endl;
    return 0;
}
```

编译并运行，输出为：

```
2
```

不清楚`ref`详细的原理，只知道这么使用就行了。

如果不想传`ref()`，又想按引用传递对象，那么可以传指针，比如`void thd_func(A *a)`，这样也是可以的。

（理论上，c++ 的引用也是一种指针，不清楚为啥不能实现）

更多资料：

<https://en.cppreference.com/w/cpp/utility/functional/ref>

## Smart pointers

`#include <memory>`

### auto_ptr

`auto_ptr`是最早被引入 c++ 的智能指针，从 c++11 开始被弃用。

This class template is deprecated as of C++11. unique_ptr is a new facility with similar functionality, but with improved security.

auto_ptr is a smart pointer that manages an object obtained via a new expression and deletes that object when auto_ptr itself is destroyed.

Example:

```cpp
int main()
{
    auto_ptr<int> p(new int(3));
    cout << *p << endl;
    cout << p.get() << endl;
    cout << &*p << endl;
    cout << *p.get() << endl;
    return 0;
}
```

Output:

```
3
0x22d0a3549e0
0x22d0a3549e0
3
```

我们可以直接对智能指针`p`进行解引用，得到实际的对象。也可以用`get()`方法得到指针存储的对象的地址。

`auto_ptr` is based on an exclusive ownership model i.e. two pointers of the same type can’t point to the same resource at the same time.

Example:

```cpp
int main()
{
    auto_ptr<int> p(new int(3));
    cout << "object memory address: " << p.get() << endl;
    cout << "object content: " << *p << endl;
    cout << "--------" << endl;
    auto_ptr<int> p2 = p;
    cout << "p content: " << p.get() << endl;
    cout << "p2 content: " << p2.get() << endl;
    cout << "p2 object content: " << *p2 << endl;
    // cout << "p object content: " << *p << endl;  // Error
    return 0;
}
```

Output:

```
object memory address: 0x181483649e0
object content: 3
--------
p content: 0
p2 content: 0x181483649e0
p2 object content: 3
```

可以看到，当执行完赋值操作后，`p`被赋为空指针，`p2`指向`p`的内容。这其实是一个 transfer 的过程，而不是一个 copy 的过程。如果我们再对`p`解引用，那么会运行时报错。

为什么`auto_ptr`被抛弃？因为它无法使得两个指针指向同一个对象，所以不支持 stl container （为什么不支持？）。

### unique_ptr

`unique_ptr`和`auto_ptr`功能差不多，只不过当它对 move 语义和 copy 语义区分得更清。

`unique_ptr`禁止了 copy from lvalue 操作：

```cpp
unique_ptr(const unique_ptr&) = delete;
unique_ptr& operator=(const unique_ptr&) = delete;
```

因此我们无法使用一个`unique_ptr`对另一个`unique_ptr`赋值：

```cpp
int main()
{
    unique_ptr<int> p1(new int(3));
    unique_ptr<int> p2 = p1;  // Error, copy constructor is forbidden
    p2 = p1;  // Error, operator with lvalue is forbidden
    unique_ptr<int> p3 = move(p1);  // Ok, move constructor is allowed
    p3 = move(p1);  // Ok, operator with rvalue is allowed
    return 0;
}
```

我们只能使用 rvalue 对`unique_ptr`赋值。

如果需要延迟给`unique_ptr`赋值，那么可以用`make_unique()`:

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main()
{
    unique_ptr<int> pval;
    pval = make_unique<int>(3);
    cout << *pval << endl;
    return 0;
}
```

Ref: <https://www.learncpp.com/cpp-tutorial/stdunique_ptr/>

### shared_ptr

A shared_ptr is a container for raw pointers. It is a reference counting ownership model i.e. it maintains the reference count of its contained pointer in cooperation with all copies of the shared_ptr.

Example:

```cpp
int main()
{
    shared_ptr<int> sp_1(new int(3));
    cout << *sp_1 << endl;
    cout << sp_1.get() << endl;
    cout << sp_1.use_count() << endl;

    cout << "------" << endl;
    shared_ptr<int> sp_2(sp_1);
    cout << *sp_2 << endl;
    cout << sp_2.get() << endl;
    cout << sp_1.use_count() << endl;
    cout << sp_2.use_count() << endl;

    cout << "------" << endl;
    sp_1.reset();
    cout << sp_1.get() << endl;
    cout << sp_1.use_count() << endl;
    cout << sp_2.use_count() << endl;
    return 0;
}
```

Output:

```
3
0x1665db249e0
1
------
3
0x1665db249e0
2
2
------
0
0
1
```

可以看到，`sp_1`和`sp_2`都指向同一个对象。他们的引用计数都变成了 2。当`sp_1`重置时，`sp_1`的引用计数变成 0，`sp_2`的引用计数变成 1。

如果一开始没有对`shared_ptr`初始化，可以使用`make_shared()`函数，或者`reset()`方法进行赋值。

* `make_shared`

    Header file:

    `#include <memory>`

    Syntax:

    ```cpp
    template <class T, class... Args>
    shared_ptr<T> make_shared (Args&&... args);
    ```

    `make_shared`可以使用`new`创建一个`T`类型的对象，并返回一个指向该对象的`shared_ptr`指针。`make_shared`中的参数会传递给`new`调用的构造函数。

    Example:

    ```cpp

    ```


**循环引用问题（Cyclic Dependency）**

假如现在有一个类 A，一个类 B，每个类中都有一个对方类的智能指针，那么就会造成循环引用问题：

```cpp
class A;
class B;

class A
{
    public:
    shared_ptr<B> sp;
    string name;

    ~A() {
        cout << name << " in A destructor" << endl;
    }
};

class B
{
    public:
    shared_ptr<A> sp;
    string name;

    ~B() {
        cout << name << " in B destructor" << endl;
    }
};

void test_cyclic_ref()
{
    shared_ptr<A> pa(new A);
    shared_ptr<B> pb(new B);
    pa->name = "pa";
    pb->name = "pb";
    pa->sp = pb;
    pb->sp = pa;
    
    shared_ptr<A> pc(new A);
    pc->name = "pc";
}

int main()
{
    test_cyclic_ref();
    return 0;
}
```

Output:

```
pc in A deconstructor
```

可以看到，`pa`和`pb`指向的对象并非调用析构函数，只有`pc`指向的对象调用了析构函数。因此在离开函数`test_cyclic_ref`时，`pa`和`pb`指向的对象的内存并未被释放。

当程序想释放对象`a`的内存时，发现`a.sp`所指向的内存（即对象`b`）并未被销毁，因此需要先销毁对象`b`。可是当程序准备释放对象`b`的内存时，发现`b.sp`所指向的内存（即对象`a`）并未被销毁，于是又会去先销毁对象`a`。这样就造成了循环。

### weak_ptr

为了解决这个问题，我们引入`weak_ptr`。

A weak_ptr is created as a copy of shared_ptr. It provides access to an object that is owned by one or more shared_ptr instances but does not participate in reference counting.

The existence or destruction of weak_ptr has no effect on the shared_ptr or its other copies. It is required in some cases to break circular references between shared_ptr instances. 

由于`weak_ptr`只是`shared_ptr`的复制，而不参与引用计数，所以可以打破这引用循环。只需要把类 A 中的`shared_ptr`改成`weak_ptr`即可：

```cpp
class A;
class B;

class A
{
    public:
    weak_ptr<B> wp;  // 将某个类中的 shared_ptr 改成 weak_ptr
    string name;

    ~A() {
        cout << name << " in A destructor" << endl;
    }
};

class B
{
    public:
    shared_ptr<A> sp;
    string name;

    ~B() {
        cout << name << " in B destructor" << endl;
    }
};

void test_cyclic_ref()
{
    shared_ptr<A> pa(new A);
    shared_ptr<B> pb(new B);
    pa->name = "pa";
    pb->name = "pb";
    pa->wp = pb;
    pb->sp = pa;
    
    shared_ptr<A> pc(new A);
    pc->name = "pc";
}

int main()
{
    test_cyclic_ref();
    return 0;
}
```

Output:

```
pc in A destructor
pb in B destructor
pa in A destructor
```

可以看到，每个对象的内存，都被正确释放了。

问题：

1. 析构顺序

    如果交换

    ```cpp
    shared_ptr<A> pa(new A);
    shared_ptr<B> pb(new B);
    ```

    这两行，即变成
    
    ```cpp
    shared_ptr<B> pb(new B);
    shared_ptr<A> pa(new A);
    ```

    上面的代码仍然会输出：

    ```
    pc in A destructor
    pb in B destructor
    pa in A destructor
    ```

    按道理，按照清栈顺序，应该先处理`pa`，再处理`pb`，为什么输出结果仍不变？

1. 这段代码其实是在外部也申请了两个`shared_ptr`指针。如果仅有`A`，`B`两个对象，内部的指针互相指，那么也会造成循环引用吗？

    会。代码如下：

    ```cpp
    class A;
    class B;

    class A
    {
        public:
        shared_ptr<B> sp;
        string name;

        ~A() {
            cout << name << " in A destructor" << endl;
        }
    };

    class B
    {
        public:
        shared_ptr<A> sp;
        string name;

        ~B() {
            cout << name << " in B destructor" << endl;
        }
    };

    void test_cyclic_ref()
    {
        A *pa = new A;
        B *pb = new B;
        pa->name = "pa";
        pb->name = "pb";
        pa->sp.reset(pb);
        pb->sp.reset(pa);
        
        shared_ptr<A> pc(new A);
        pc->name = "pc";

        cout << pa->sp.use_count() << endl;
        cout << pb->sp.use_count() << endl;
    }

    int main()
    {
        test_cyclic_ref();
        return 0;
    }
    ```

    Output:

    ```
    1
    1
    pc in A destructor
    ```

    唯一的区别是，使用`shared_ptr`的代码，其指针的引用数为 2。而这段代码的引用数为 1。

## Coroutine 协程

### cache

* 一个 c++ coroutine 的例子

    ```cpp
    struct MyCoroutine {
        struct promise_type {
            char ch;
            MyCoroutine get_return_object() {
                return MyCoroutine(coroutine_handle<promise_type>::from_promise(*this));
            }
            suspend_always initial_suspend() {return {}; }
            suspend_never final_suspend() noexcept {return {};}
            void return_void() {}
            void unhandled_exception() {}
            suspend_always yield_value(char &ch) {
                this->ch = ch;
                return {};
            }
        };

        coroutine_handle<promise_type> h;
        MyCoroutine(coroutine_handle<promise_type> h): h(h) {}
        ~MyCoroutine() {
            h.destroy();
        }

        char operator()() {
            h();
            return h.promise().ch;
        }
    };

    MyCoroutine myCoroutine()
    {
        static string msg = "hello";
        for (int i = 0; i < 5; ++i)
        {
            co_yield msg[i];
        }
        co_return;
    }

    int main()
    {
        MyCoroutine next_ch = myCoroutine();
        for (int i = 0; i < 5; ++i)
            cout << next_ch() << endl;
        return 0;
    }
    ```

    编译：

    ```bash
    g++ -std=c++20 -g main.cpp -o main
    ```

    运行：

    ```
    ./main
    ```

    输出：

    ```
    h
    e
    l
    l
    o
    ```

    说明：

    * 一个普通`struct`中需要包含一个叫`promise_type`的 struct，他们互相协同工作，才能完成协程的功能。

    * `MyCoroutine myCoroutine()`表示这个函数返回（创建）一个`MyCoroutine`对象。
    
        这个对象在刚被创建的时候，就调用了`initial_suspend()`函数。由于`initial_suspend()`被`suspend_always`修饰，所以执行完这个函数后，就停了下来。

        我们在调用`myCoroutine()`时，并没有去执行函数中的内容，而是去创建了一个`MyCoroutine`对象，这点与普通的函数调用不同，其实挺别扭的。但是由于函数的返回值是`MyCoroutine`，又刚好对接到传统 c++ 的类型语法检查。

    * 在调用`next_ch()`后，`operator()`开始被执行，`h();`表示继续执行`myCoroutine()`

        函数执行到`co_yield`语句后，进入`yield_value()`。这个执行完后，则会停下来，继续执行`return h.promise().ch;`，此时会返回一个值。

    * `co_yield msg[i];`等价于调用`yield_value()`

        这个函数通常算一些数据，然后给存起来，存起来的结果后面由`operator()`返回。

        这个过程可以使用 move 语义加速。

    * `myCoroutine()`函数其实相当于托管给了`coroutine_handle`，由这个 handle 来控制函数的执行流程。

        这个 handle 同时保存了 promise 的信息和协程函数的信息。因此它扮演了很重要的角色。

    * c++ 23 有`generator`模板类，非常好用，今明两天目前的 gcc-12 还没实现。

    ref: <https://en.cppreference.com/w/cpp/language/coroutines>

* c++ coroutine 在 c 中是否有更简单的方式手动实现？

    c++ 的协程，c 的手动实现，切换线程实现协程，哪个的效率更高？

### note

A coroutine is any function that contains a `co_return`, `co_yield` or `co_await`.

协程主要用来处理单线程异步操作。和 JavaScript 挺像的。

一个简单的 example:

```cpp
#include <coroutine>
using namespace std;

struct Task {
    struct promise_type {
        Task get_return_object() { return {}; }
        suspend_never initial_suspend() {return {}; }
        suspend_never final_suspend() noexcept {return {};}
        void return_void() {}
        void unhandled_exception() {}
    };
};

Task myCoroutine() {
    co_return;
}

int main() {
    Task x = myCoroutine();
    return 0;
}
```

`Task`是我们随便定义的一个结构体，里面必须要有一个`promise_type`子结构体才行。

`co_return`关键字会先调用`get_return_object()`生成一个 promise 对象，然后 promise 对象会调用`initial_suspend()`函数，然后调用`return_void()`函数，最后调用`final_suspend()`函数，退出整个程序。

`promise_type`是一个标记型的类型，表示了外部类有 coroutine 的功能，除此之外`promise_type`什么用都没有。我们需要手动实现`promise_type`中的一些方法，以便当外部类被当作 yield 型调用时，可以实现异步 IO 的功能。

* `co_await`

    一个一元操作符，等待一个 promise 对象返回。

    Example:

    ```cpp
    co_await expr;
    ```

    其中`expr`必须是`awaitable`类型。

* `awaiter`

    定义了`await_ready`、`await_suspend`和`await_resume`方法的类型。

* `co_yield expr;`等价于`co_await promise.yield_value(expr);`

* `suspend_always`

    函数返回一个值后，promise 对象总是挂起。

* `co_yield`或`co_return`会先调用`get_return_object()`返回一个 promise 的外类对象。

* `promise_type`的`resume()`函数可以让 promise 对象继续调用`yield_value()`方法。

* 如果拿到了一个 promise 对象的 handle，那么可以用`handle.promise()`返回 promise 对象

* 可以用`coroutine_handle<promise_type>::from_promise()`创建一个指定 promise 对象的 handle

* awaitable 代表一系列类型，不是一个具体的类型

    常见的 awaitable 系列的类型：

    * `suspend_always`
    * `suspend_never `

* 如果一个 promise type 提供了`await_transform(expr)`方法，那么`co_await expr;`应会变成`co_await promise.await_transform(expr);`

* awaitable 系列类型对象

    定义了几个方法的`struct`对象即为 awaitable 系列类型的对象

    ```cpp
    struct Sleeper {
        constexpr bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h) const {
            auto t = std::jthread([h,l = length] {
                std::this_thread::sleep_for(l);
                h.resume();
            });
        }
        constexpr void await_resume() const noexcept {}
        const std::chrono::duration<int, std::milli> length;
    };

    Task myCoroutine() {
        using namespace std::chrono_literals;
        auto before = std::chrono::steady_clock::now();
        co_await Sleeper{200ms};
        auto after = std::chrono::steady_clock::now();
        std::cout << "Slept for " << (after-before) / 1ms << " ms\n";
    }
    ```

    当我们调用`co_await abaitable;`时，会先调用`await_ready()`方法，如果这个方法返回`true`，会继续执行主程序。如果返回`false`，那么会继续调用`await_suspend()`方法。在`await_suspend()`方法的最后，我们需要调用`resume()`方法，此时会接着执行`await_resume()`方法。执行完`await_resume()`后，会继续执行主程序。


## Lambda Expression （匿名函数）

```cpp
int main()
{
    auto f = [](int a)
    {
        cout << a << endl;
    };
    f(123);
    return 0;
}
```

谨慎使用`&`捕捉外界的变量：

```cpp
#include <iostream>
using namespace std;

auto get_print_func(const char *msg)  // 如果在这里把改成 const char *&msg，那么将会获得正确输出
{
    return [&msg]() {
        cout << msg << endl;
    };
}

int main()
{   
    const char *msg = "hello";
    cout << msg << endl;
    auto print = get_print_func(msg);
    print();
    return 0;
}
```

输出：

```
hello
����
```

上面的程序中，参数的函数`msg`在进入函数时被生成，在函数返回时被销毁，而`[&msg]`是捕捉`msg`的引用，当`get_print_func()`返回时，这个引用就已经无效了。后面我们再调用`print()`就会发生错误输出。

因此`[&]`捕获的变量必须要在外界调用时依然有效才可以。

## Exception 异常处理

## 工程化

1. 头文件`xxx.h`中，只写对应`xxx.cpp`文件中出现的函数，尽量不写其他用到的头文件，仅保证头文件中没有语法错误即可。`xxx.cpp`中需要包含实现函数时用到的头文件。这样比较清晰，不容易乱。

1. 跨文件的全局变量

    在头文件`a.h`中写声明：`extern int aaa;`，在其他某个实现文件`xxx.cpp`中写定义：`int aaa;`。在其余地方，比如`main.cpp`，只要 include 了`a.h`，就可以使用这个变量。

1. 动态链接与静态链接

    * 动态链接

        `mylib.h`:

        ```cpp
        int add(int a, int b);
        ```

        `mylib.cpp`:

        ```cpp
        int add(int a, int b)
        {
            return a + b;
        }
        ```

        编译 lib：

        ```bash
        g++ -shared mylib.cpp -o libmylib.so
        ```

        如果是 windows 平台，那么使用

        ```bash
        g++ -shared mylib.cpp -o libmylib.dll
        ```

        测试程序：

        `main.cpp`:

        ```cpp
        #include <iostream>
        #include "mylib.h"
        using namespace std;

        int main()
        {
            int a = 1, b = 2;
            int c = add(a, b);
            cout << c << endl;
            return 0;
        }
        ```

        编译：

        ```bash
        g++ main.cpp -L. -lmylib -o main
        ```

        运行：

        ```
        ./main.exe
        ```

        只有当`libmylib.dll`和`main.exe`在同一个文件夹下时，`main.exe`才能成功运行。如果`libmylib.dll`不在当前目录下，那么需要将它放到搜索目录里。

    * 静态链接

        编译 lib:

        ```bash
        g++ -c mylib.cpp -o mylib.obj
        ar rcs mylib.lib mylib.obj
        ```

        编译测试程序：

        ```bash
        g++ main.cpp mylib.lib -o main.exe
        ```

        此时可以把`main.exe`放到电脑的任何文件夹里运行。

        不清楚此时 glibc 是否是静态链接的。

    Ref:

    1. <https://stackoverflow.com/questions/64455245/how-to-write-a-static-c-library-and-link-it-to-an-executable-using-g-on-wind>

    1. <https://www.systutorials.com/how-to-statically-link-c-and-c-programs-on-linux-with-gcc/>

    1. <https://www.herongyang.com/Linux-Apps/GCC-c-to-Build-Static-Library-Files.html>

    1. <https://medium.com/@neunhoef/static-binaries-for-a-c-application-f7c76f8041cf>

    1. <https://blog.habets.se/2013/01/Compiling-C++-statically.html>

    我们无法将很多的`.so`文件合并成一个，或者将很多的`.so`文件静态链接到一个程序上。

    Ref: <https://stackoverflow.com/questions/915128/merge-multiple-so-shared-libraries>

## Topics

### character encoding

`codecvt.h`头文件专门用于处理字符编码。

`locale.h`有些函数对`codecvt.h`中的函数进行了封装，可以更方便地处理`wchar_t`之类的字符。

Examples:

utf-8 与 utf-16 字符串之间的互相转换：

```cpp
#include <codecvt>
#include <locale>
#include <string>
using namespace std;

int main()
{
    wstring_convert<codecvt_utf8_utf16<wchar_t>, wchar_t> convertor;

    // 将 utf-8 编码的“你好”转换成 utf-16 编码的“你好”
    wstring wstr = convertor.from_bytes("\xE4\xBD\xA0\xE5\xA5\xBD");

    // 将 utf-16 le 编码的“你好”转换成 utf-8 编码的"你好"
    string str = convertor.to_bytes((wchar_t*)"\x60\x4f\x7d\x59\0");  // 末尾的 \0 是必须的，我猜可能是因为 wchar_t 必须有两个字节都为 \0，才能作为字符串的结尾

    return 0;
}
```

我们可以在编码查询网站<https://www.qqxiuzi.cn/bianma/zifuji.php>上查到，`你好`的 utf-8 编码为`E4BDA0E5A5BD`，utf-16le 编码为`604F7D59`。前面这两串编码都是从低字节向高字节来写的。由于程序中实际使用的是小端存储模式，所以如果两个字节合起来读，实际上“你”的编码为`4f60`，“好”的编码为`7d59`。

`<codecvt>`头文件中有三个 class:

* `std::codecvt_utf8`：用于在 utf-8 和 UCS-2 or UCS-4 之间相互转换

* `std::codecvt_utf16`：用于在 utf-8 和 Elem (either UCS-2 or UCS-4) 之间互相转换

* `std::codecvt_utf8_utf16`：用于在 utf-8 和 utf-16 之间互相转换

我们主要看`codecvt_utf8_utf16`怎么用。

```cpp
template < class Elem, unsigned long MaxCode = 0x10ffffUL, codecvt_mode Mode = (codecvt_mode)0 >
class codecvt_utf8_utf16
: public codecvt <Elem, char, mbstate_t>
```

这是一个模板类，其中`codecvt_utf8_utf16`的模板参数`Elem`表示“内部”数据使用的编码方式，`codecvt`的第二个模板参数`char`表示“外部”数据使用的编码方式。

由于`char`表示以字节进行编码，所以指的是 utf-8。而`Elem`可以取值`wchar_t`，`char16_t `，`char32_t`。

常用函数：

* `std::codecvt::in()`

    Syntax:

    ```cpp
    result in (
        state_type& state,   
        const extern_type* from, 
        const extern_type* from_end, 
        const extern_type*& from_next,        
        intern_type* to, 
        intern_type* to_limit, 
        intern_type*& to_next
    ) const;
    ```

    由于对`codecvt_utf8_utf16`类型为说，“内部”数据指的是`wchar_t`之类的固定长度编码字符串，“外部”数据指的是 utf-8 编码的字符串。所以`in()`函数的作用就是把 utf-8 转换成`wchar_t`之类。

    Parameters:

    * `state`: 盲猜存储 shift 的作用。实际上它是一个`int`类型的值，直接填 0 就行。（可能需要做个类型转换）

    * `from`：utf-8 字符串的起始地址，`from_end`指的是字符串的结束地址。它们组成了一个左闭右开的区间`[from, from_end)`。

    * `from_next`：无论什么原因导致函数返回（比如成功解析完字符串，或者转换失败），`from_next`都指向待转换的字符串中，下一个待翻译的字节。

    * `to`：指定缓冲区的地址。同理，`to`和`to_limit`构成了左闭右开的区间`[to, to_limit)`。

    * `to_next`：无论什么原因导致函数返回，`to_next`都指向下一个待填充的缓冲区。

    Example:

    ```cpp
    #include <codecvt>
    #include <iostream>
    using namespace std;

    int main()
    {
        codecvt_utf8_utf16<wchar_t> cvt;
        char *utf8_nihao = "\xE4\xBD\xA0\xE5\xA5\xBD";
        const char *from_next;
        wchar_t utf16_buf[3];
        wchar_t *dst_next;
        mbstate_t mbs = mbstate_t();
        cvt.in(mbs, utf8_nihao, utf8_nihao + 6, from_next,
            utf16_buf, utf16_buf + 2, dst_next);
        for (int i = 0; i < 4; ++i)
        {
            cout << hex << (int)*((char*)utf16_buf + i) << ", ";
        }
        cout << endl;
        return 0;
    }
    ```

    输出：

    ```
    60, 4f, 7d, 59,
    ```

* `std::codecvt::out`

    Syntax:

    ```cpp
    result out (
        state_type& state,   
        const intern_type* from, 
        const intern_type* from_end, 
        const intern_type*& from_next,        
        extern_type* to, 
        extern_type* to_limit, 
        extern_type*& to_next
    ) const;
    ```

    各个参数的作用可以参考`in()`，这里就不再详细写了。

### String processing

* 去除字符串左侧的指定字符（一个或多个）

    * 一个

    * 多个

* 去除字符串左侧的指定子串

* 去除字符串右侧的指定字符（一个或多个）

    * 一个

    * 多个

* 去除字符串右侧的指定子串

* 分隔以`,`以及空白字符隔开的字符串

    ```cpp

    ```

* 模式匹配

### 内存与对象

本话题主要讨论对象在内存中存储时遇到的各种问题。

#### 有关 vector 和 array 的一个猜想

如果是`vector<vector<int>> arr`，那么`arr`不会占用一大块连续的内存；如果是`vector<array<int, 3>> arr`，那么`arr`会占用一大块连续的内存。

#### 有关指针与 struct

如果不使用指针，那么只能把包含这个对象的 struct 的指针传来传去，然后配合 id 或哈希值拿到具体的资源。

* 假如一个场景中包含了很多物体，我想根据物体对象，拿到它所在的场景，该怎么写 c++ 代码？

    显然要用到指针。但是需不需要用到智能指针？

    一个比较简单的情况：

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>
    using namespace std;

    struct Scene;

    struct Object
    {
        string name;
        Scene *pscene;

        string get_scene_name();
    };

    struct Scene
    {
        string name;
        vector<Object> objs;
    };

    string Object::get_scene_name() {
        return pscene->name;
    }

    int main()
    {
        Scene scene;
        scene.name = "m_scene";
        Object obj;
        obj.name = "hello";
        obj.pscene = &scene;
        scene.objs.push_back(obj);

        Object &inner_obj = scene.objs[0];
        cout << "scene name: " << inner_obj.get_scene_name() << endl;
        return09;
    }
    ```

    输出：

    ```
    scene name: m_scene
    ```

    但是如果 scene 在 obj 不知道的时候，偷偷失效了，会发生什么？

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>
    using namespace std;

    struct Scene;

    struct Object
    {
        string name;
        Scene *pscene;

        string get_scene_name();
    };

    struct Scene
    {
        string name;
        vector<Object> objs;
    };

    string Object::get_scene_name() {
        return pscene->name;
    }

    int main()
    {
        Scene *scene = new Scene;
        scene->name = "m_scene";
        Object obj;
        obj.name = "hello";
        obj.pscene = scene;
        scene->objs.push_back(obj);

        Object &inner_obj = scene->objs[0];
        delete scene;  // 在 inner_obj 不知道的时候，scene 已经失效了
        cout << "scene name: " << inner_obj.get_scene_name() << endl;
        return 0;
    }
    ```

    这段代码会报运行时错误：

    ```
    terminate called after throwing an instance of 'std::length_error'
        what():  basic_string::_M_create
    ```

    可以看到`scene`会失效，`inner_obj`本身也会失效。

    假如两个对象没有包含的关系，即一个对象中，不使用 stl 容器包含另一个对象，会发生什么？

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>
    using namespace std;

    struct B;

    struct A
    {
        string name;
        B *pb;
    };

    struct B
    {
        string name;
        A *pa;
    };

    int main()
    {
        A *a = new A;
        a->name = "a";
        B *b = new B;
        b->name = "b";
        b->pa = a;
        a->pb = b;

        delete a;
        cout << b->name << endl;
        cout << b->pa->name << endl;  // 在这一行会出现运行时错误
        return 0;
    }
    ```

    可见，一个对象的消失，并不会通知另外一个对象。

    如果換成智能指针呢？会发生什么？

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>
    using namespace std;

    struct B;

    struct A
    {
        string name;
        shared_ptr<B> pb;
        A() {
            cout << "in A constructor" << endl;
        }
        ~A() {
            cout << "in A destructor" << endl;
        }
    };

    struct B
    {
        string name;
        shared_ptr<A> pa;
        B() {
            cout << "in B constructor" << endl;
        }
        ~B() {
            cout << "in B destructor" << endl;
        }
    };

    int main()
    {
        A *a = new A;
        a->name = "a";
        B *b = new B;
        b->name = "b";
        b->pa = make_shared<A>(*a);
        a->pb = make_shared<B>(*b);

        delete a;
        cout << b->name << endl;
        cout << b->pa->name << endl;
        return 0;
    }
    ```

    输出：

    ```
    in A constructor
    in B constructor
    in A destructor
    in B destructor
    b
    a
    ```

    可以看到，在`delete a;`时，`A`和`B`两个对象就已经被销毁了，后面的输出其实是不可靠的。

    有一个想法是给`A`和`B`互设友元，这样当某个对象失效的时候，可以在析构函数里告诉互相链接的对象。但是这样要求你提前知道哪些类是需要互相通信的，而且当类的数量变多的时候，通信连接数会指数增长。

    其实智能指针和 move 语义也都只能保证有关联的两个指针，如果一个有效，那么另一个也一定有效。无法保证一个失效时，对另外一个指针进行通知；或者让一个指针总是能查询与之相关联的另一个指针是否有效。

    或许可以增加一个全局对象，或者用一个额外的类，来对资源进行管理，提供对象是否失效的查询接口。这样要求每次有新的对象创建，必须在这个资源管理对象中进行注册，当对象被销毁时，在资源管理对象中反注册。这样就又涉及到整个程序的设计思想了，是否真的需要这样做，还得权衡代码复杂度和性能。

    如果让`B`对象包含`A`的指针，如果仅仅是为了传参方便，我觉得没有必要这样做。如果他们确实存在包含关系，那么只需要用到 c++ 的局部变量内存管理就好了。如果两个对象存在依赖关系，那么可以考虑用智能指针。

### 构造函数与析构函数调用次数不一致

c++ 中将一个自动管理的对象（比如局部变量，全局变量等）交给另外一个自动管理的系统（比如 shared_ptr，stl 容器）时，由于会调用移动构造函数，所以有可能不调用构造函数，造成构造函数和析构函数调用的次数不成对，最终导致内存错误。

一个经典的错误是，在程序结束时，构造函数调用一次，析构函数调用两次。

example:

```cpp
#include <iostream>
#include <memory>
using namespace std;

struct B;

struct A
{
    int val;
    shared_ptr<B> to_b;
    A() {
        cout << "in A constructor" << endl;
    }
    ~A() {
        cout << "in A destructor, val: " << val << endl;
    }
};

struct B
{
    int val;
    weak_ptr<A> to_a;
    B() {
        cout << "in B constructor" << endl;
    }
    ~B() {
        cout << "in B destructor, val: " << val << endl;
    }
};

int main()
{
    A a;
    a.val = 1;
    B b;
    b.val = 2;
    a.to_b = make_shared<B>(b);
    b.to_a = make_shared<A>(a);
    return 0;
}
```

输出：

```
in A constructor
in B constructor
in A destructor, val: 1
in B destructor, val: 2
in A destructor, val: 1
in B destructor, val: 2
```

### extern C

### cache

* 如果一个函数同时在`extern "C"`和`extern "C"`以外的区域声明或定义，那么它会被编译器视为定义了两次

* gcc 编译器编译 g++ 编译器编译出来的`.o`文件，会报错没有 c++ 标准库。如果只链接 g++ 编译出来的`.so`文件，则没有问题。

* gcc 编译器不识别`extern "C"`, g++ 才识别

    所以如果一个头文件被 c 和 c++ 共用，那么可以这样写：

    ```cpp
    #ifdef __cplusplus
    extern "C" {
    #endif

    void func_1(int aaa);
    int func_2(char *bbb);
    // ....

    #ifdef __cplusplus
    }
    #endif
    ```

    ref: <https://stackoverflow.com/questions/43602910/extern-c-causing-an-error-expected-before-string-constant>

### note

## gcc attribute

```cpp
#include <iostream>
using namespace std;

template <typename T>
inline T __attribute__((const)) my_max(const T x, const T y)
{
    if (x > y)
        return x;
    else
        return y;
}

int main()
{
    const int a = 3, b = 4;
    int c = my_max(a, b);
    cout << c << endl;
}
```

似乎会用于 gcc 的优化，有空了看看。

Ref: <https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html>

## Miscellaneous

1. 宏

    ```c++
    #define CONVAR 15
    ```

1. 标识符只能由字母、数字、下划线组成。标识符第一个字符只能是字母或下划线。

1. `sizeof`

    ```c++
    sizeof(int)
    sizeof(long long)

    int a = 3;
    sizeof(a);

    int arr[] = {1, 2, 3};
    sizeof(arr)  // 3 * 4
    sizeof(arr) / sizeof(arr[0])  // 3

    int arr2D[3][2];
    sizeof(arr2D)  // 6 * 4
    sizeof(arr2D[0])  // 2 * 4
    sizeof(arr2D[0][0])  // 4

    ```

1. 转义字符与 ASCII

1. 对于 32 位操作系统，指针占用 4 个字节；对于 64 位操作系统，指针占用 8 个字节。

1. 指针

    常量指针：`const int *p = &a;`，一个指向常量的指针，指针本身可以被修改。

    指针常量：`int * const p = &a;`，一个指向 int 类型的 const 指针，指针本身不可被修改。

    `const`既修饰指针，又修饰常量：`const int * const p = &a;`，指针和指针所指向的值都不可修改。

    指针可以接收数组名，使用下标访问：

    ```c++
    int arr[3] = {1, 2, 3};
    int *parr = arr;
    parr[0];
    ```

    但是这样就没法用`sizeof`获取数组的字节数了：

    ```c++
    sizeof(parr)  // 8
    sizeof(arr)  // 3 * 4 = 12
    ```

1. 内存分区模型

    1. 代码区：用于存储二进制代码，由操作系统管理

        代码区共享且只读。

    1. 全局区：用于存储全局变量，静态变量及常量（字符串常量和全局 const 常量，但不包含局部 const 常量）

        由操作系统负责分配内存和释放。

    1. 栈区：由编译器自动分配释放，存放函数的参数值，局部变量等

    1. 堆区：由程序员分配和释放，若程序员不释放，程序结束时由操作系统回收

    程序编译生成后，只有代码区和全局区。

1. `new`

    ```c++
    int *p = new int(3);
    delete p;

    int *p = new int[10];
    delete []p;

    ```

1. 引用

    引用的本质是指针常量

    常量引用：

    ```c++
    int a = 3;
    const int &ra = a;
    const int &ref = 10;  // int temp = 10;  const int &ref = temp;
    ```

1. `size_t`（`unsigned long long`）与`int`类型作大小比较时，会出错

    ```c++
    size_t a = 1;
    int b = -1;
    a > b  // false
    (int) a > b  // true
    ```

1. 匿名函数

    ```cpp
    int main()
    {
        int a = 30;
        all_of(v.begin(), v.end(), [](int &a){
            // xxx
            a = 40;
            return true;
        });
        return 0;
    }
    ```

    在匿名函数的参数中，使用引用`int &a`作为参数时，外部的变量会被修改。使用`int a`作为参数时，外部的变量会传递进来一个副本。

1. 如果使用``malloc`申请的内存不够，会发生什么？

    试了下，在不合法的地方写入数据是没事的，还能正常使用，可能是没修改到重要的内存，比如别的函数的入口地址之类的。但是在`free()`的时候会报错。如果不`free`直接结束程序，也不会报错。

1. `xxx.size()`是一个无符号整数，如果`xxx`为空，那么`xxx.size() - 1`就会回绕成最大无符号整数。

    代码里如果有：

    ```cpp
    for (int i = 0; i < xxx.size() - 1; ++i)
    {
        // do something...
    }
    ```

    就需要特别注意。

1. 为什么要把基类析构函数声明为虚函数？

    假设现在有`class B: public A`，`A *obj = new B;`，且`A`的虚构函数没有声明为虚函数，那么当我们`delete obj;`时，只会调用`A`中的析构函数，不会调用`B`中的析构函数。

1. 处理程序的输入

    ```cpp
    // 可以使用这种
    int main(int argc, char** argv)

    // 也可以使用这种
    int main(int argc, char* argv[])
    ```

    如果没有输入，`argc`为 1，`argv[0]`为在 terminal 中运行这个程序输入的 path。注意这个 path 可以是相对路径，也可以是绝对路径。它总是与 terminal 中输入的命令相同。

1. choose without replacement

    ```cpp
    #include <iostream>
    #include <unordered_map>
    #include <algorithm>
    #include <vector>
    #include <string>
    #include <random>
    using namespace std;

    int main(int argc, char* argv[])
    {
        vector<int> v{3, 2, 5, 4, 1};
        int buf[3];
        sample(v.begin(), v.end(), buf, 3, mt19937{random_device{}()});
        for (auto item: buf)
            cout << item << ",";
        cout << endl;
        return 0;
    }
    ```

    注意`sample()`不会改变元素在 container 中的顺序。如果需要随机顺序，还得再`shuffle()`一下。

1. shuffle

    ```cpp
    vector<int> vv{3,4,5,6,7};
    shuffle(vv.begin(), vv.end(), mt19937{random_device{}()});
    for (auto item: vv)
        cout << item << ",";
    cout << endl;
    ```

1. unicode 字符处理

    使用`wchar_t`和`wstring`。

    ```cpp
    wchar_t wc[] = L"你";
    cout << sizeof(wc) << ", " << sizeof(wc[0]) << endl;
    wstring ws{L"你好"};
    ```

    `wchar_t`占两个字节。

    `wcout`无法输出的问题：

    ```cpp
    #include <iostream>
    #include <locale>
    #include <string>
    #include <codecvt>

    int main()
    {
        std::ios_base::sync_with_stdio(false);

        std::locale utf8( std::locale(), new std::codecvt_utf8_utf16<wchar_t> );
        std::wcout.imbue(utf8);

        std::wstring w(L"Bilişim Sistemleri Mühendisliğine Giriş");
        std::wcout << w << '\n';
    }
    ```

    `wstring`和`string`之间的转换，好像还挺麻烦的：

    ```cpp
    #include <codecvt>
    #include <string>

    std::wstring utf8ToUtf16(const std::string& utf8Str)
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
        return conv.from_bytes(utf8Str);
    }

    std::string utf16ToUtf8(const std::wstring& utf16Str)
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
        return conv.to_bytes(utf16Str);
    }
    ```

1. `memset()`是按字节填充数据的。如果我们要填充的数据大于一个字节，就没法用这个函数。另外，如果缓冲区是`wchar_t`类型，那么需要用`sizeof(wchar_t) * BUFFSIZE`计算填充长度。

1. `-fpermissive`可以返回右值的地址。

1. 有关程序的编码问题，乱码问题，深度解析：<https://blog.csdn.net/qq_50052051/article/details/123690739>

1. filter containers in c++

    <https://www.cppstories.com/2021/filter-cpp-containers/>

1. 读 utf-8 文件：<https://stackoverflow.com/questions/4775437/read-unicode-utf-8-file-into-wstring>

1. 将 char string 转换成 wchar_t string: <https://learncplusplus.org/convert-a-char-array-to-a-wide-string-in-modern-c-software/>

    <https://codereview.stackexchange.com/questions/419/converting-between-stdwstring-and-stdstring>

1. wcout 不输出

    <https://stackoverflow.com/questions/50053386/wcout-does-not-output-as-desired>

1. c++ 中中文字符的处理：<https://blog.csdn.net/orz_3399/article/details/53415987>

1. 大端模式与小端模式

    cpu 在读内存的时候，是一个字节一个字节地读的，但是有些数据，需要把连续几个字节合起来用才能表示一个有效值。我们该怎么把这几个字节连起来呢？比如，要把两个 8 位的数据类型，组成一个 16 位的数据类型，我们实际上拥有的是这样的：

    `[6, 5, 4, 3, 2, 1, 0], [6, 5, 4, 3, 2, 1, 0]`

    我们可以把这两段内存直接合并成一个：

    `[13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]`

    上面这种模式就是**大端模式**，即低内存地址存储高位字节的数据。

    我们也可以把它分成两份：

    `[6, 5, 4, 3, 2, 1, 0], [13, 12, 11, 10, 9, 8, 7]`

    上面这种存储模式称为**小端模式**，即低内存地址存储低位字节的数据。

    又例如，对于两个字节的数据（比如`short`）`0x1234`，它的大端模式为`[0x12, 0x34]`，小端模式为`[0x34, 0x12]`。

    windows 上的 mingw64 g++ 使用的是小端模式。我们可以写程序验证：

    ```cpp
    #include <iostream>
    using namespace std;

    int main()
    {
        short val = 0x1234;
        cout << hex << val << endl;
        cout << hex << (int)*(char*)&val << ", " 
            << (int)*((char*)&val + 1) << endl;
            
        return 0;
    }
    ```

    output:

    ```
    1234
    34, 12
    ```

    可以看到，低位内存的数据为`0x34`，对应`0x1234`的低位；高位内存的数据为`0x12`，对应`0x1234`的高位。因此为小端模式。

1. 将一个十六进制字符串转换成整数

    ```cpp
    std::string s = "5f0066";
    int num = std::stoi(s, 0, 16);
    ```

    或者这样：

    ```cpp
    std::stringstream str;
    std::string s1 = "5f0066";
    str << s1;
    int value;
    str >> std::hex >> value;
    ```

1. union

    用 union 定义的变量，同一时间只有一个被激活，但他们共用同一块内存。

    ```cpp
    struct Test
    {
        void *ptr;
        union {
            int64_t size_i64;
            int32_t size_i32;
        };
    };

    int main()
    {
        Test t;
        t.size_i64 = 3;
        printf("%d", t.size_i32);  // 3
    } 
    ```

1. 全局变量的使用

    如果我们对全局变量的 class 实现了自定义的构造函数，那么我们就无法无参数地初始化全局变量了。

    对应的解决办法是使用指针/智能指针，并在`test.cpp`中维护相关的内存：

    `main.cpp`:

    ```cpp
    #include <iostream>
    #include "test.h"
    using namespace std;

    int main()
    {
        init_global_env(3, 4);
        cout << "val: " << pa->m_val << endl;
        cout << "val 2: " << pa_2.get()->m_val << endl;
        exit_global_env();
        return 0;
    }
    ```

    `test.h`:

    ```cpp
    #include <memory>
    using namespace std;

    class A
    {
        public:
        A(int val): m_val(val) {}
        int m_val;
    };

    void init_global_env(int param_1, int param_2);
    void exit_global_env();

    extern A *pa;
    extern shared_ptr<A> pa_2;
    ```

    `test.cpp`:

    ```
    #include "test.h"
    #include <memory>
    using namespace std;

    A *pa;
    shared_ptr<A> pa_2;

    void init_global_env(int param_1, int param_2)
    {
        pa = new A(param_1);
        pa_2 = make_shared<A>(param_2);
    }

    void exit_global_env()
    {
        delete pa;
    }
    ```

    编译：

    ```bash
    g++ test.h test.cpp main.cpp -o main
    ```

    输出：

    ```
    val: 3
    val 2: 4
    ```

    可以看到，其实没有什么比较好的方法。还需要额外借助几个函数来完成初始化和销毁的工作。

    * 2025052900: 如果对 class 使用了自定义构造函数导致无法默认构造，那么使用new 或者智能指针就可以默认构造了吗？

1. `getchar()`在 windows 下和 linux 下的表现

    在 windows 下，输入`a`回车，会得到两个字符，对应的 int 值分别是`97`和`10`：

    ```cpp
    int main()
    {

        int ch;
        ch = getchar();
        cout << ch << endl;
        ch = getchar();
        cout << ch << endl;
        return 0;
    }
    ```

    输出：

    ```
    97
    10
    ```

    linux 下的输出与此完全相同。

1. 用 g++ 编译时报错：`error adding symbols: DSO missing from command line`

    解决方案：
    
    加一个编译 flag: `-Wl,--copy-dt-needed-entries`

    Ref: <https://stackoverflow.com/questions/19901934/libpthread-so-0-error-adding-symbols-dso-missing-from-command-line>


1. 有关程序中大量使用随机数的问题

    `rand()`的速度很慢，`mt19937`也只快了 3 倍。如果代码中大量用到`rand()`，那么 cpu 会分配很多时间（90% 甚至更多）在系统调用上。

    如果是对随机性不敏感的场合（比如图片渲染），可以使用打表的方法，提前生成一些随机数，在用的时候直接读取数组就可以了。

    ```cpp
    vector<float> rand_nums;
    const total_rand_nums_count = 1024;
    int rand_num_idx = 0;

    float get_rand_num()
    {
        if (rand_num_idx >= total_rand_nums_count)
            rand_num_idx = 0;
        return rand_nums[rand_num_idx++];
    }

    int main()
    {
        // initialize random numbers table
        rand_nums.resize(total_rand_nums_count);
        for (int i = 0; i < total_rand_nums_count; ++i)
        {
            rand_nums[i] = (float)rand() / RAND_MAX;
        }

        // usage
        float rand_num = get_rand_num();
        return 0;
    }
    ```

    如果现在是多线程，又该怎么办？如果还使用上面的代码，由于`rand_num_idx`并不是线程安全的，所以`rand_nums[rand_num_idx++]`可能会数组越界。

    一个比较自然的想法是加锁：

    ```cpp
    mutex mtx;

    int main()
    {
        // usage
        {
            lock_gurad g(mtx)
            float rand_num = get_rand_num();
        }
    }
    ```

    每个线程在获取随机数时，一个一个访问，保证`rand_num_idx`每次都只递增 1，这样就不会数组越界了。但是加锁同`rand()`一样，会造成性能严重下降，导致 cpu 跑不满，或者 cpu 被浪费在系统调用上。

    如果不加锁，又不想数组越界，还有一种方法是使用`rand_num_idx++ % total_rand_nums_count`作为下标进行访问，然后为了防止`rand_num_idx`递增溢出，在每次访问前或后尝试清零：

    ```cpp
    float get_rand_num()
    {
        if (rand_num_idx >= total_rand_nums_count)
            rand_num_idx = 0;
        return rand_nums[rand_num_idx % total_rand_nums_count++];
    }

    int main()
    {
        // usage
        float rand_num = get_rand_num();
    }
    ```

    这样可能会造成有的随机数没有被访问到，有的随机数被访问了多次。但是由于我们是随机性不敏感的场合，所以这样的代价是可以承受的。使用这样的方法可以把 cpu 跑满。

    如果想让每个随机数都访问到，而且不加锁，不越界，可以让每个线程维护一个自己的`rand_num_idx`，或者干脆每个线程维护一个自己的随机数表。

1. `array<int, 3> vec;`和`int vec[3];`该如何选择？

    `array<int, 3>`比`int vec[3]`多了一个`=`运算符，比较方便三个数一起赋值。

    `vector`里可以存`array`，但是不能直接存`int []`数组，如果一定要存数组，可以使用一个`struct`将数组包裹起来，再将 struct 对象存到`vector`中。

    因此如果是存一个向量，那么可以选择`array<float, 3>`，这样可以直接使用`pos_1 = pos_2;`这样的功能。

    如果是在`struct`中定义一组数据，那么可以使用`type []`数组。

* 如果一个甚类`A`有两个派生类`B`，`C`，其中`AA`，`B`有方法`func()`，`C`没有。同时`B`和`C`实现了`A`所有的虚函数。

    现在假设有一个对象`obj`，可能是`B`类，也可能是`C`类型，这个对象被强制转换成基类`A`的指针`pobj`。那么此时使用`pobj`调用`func()`方法，如果`obj`是类型`C`，程序会不会出错？这样的代码又为什么能通过编译？

* 关于固定长度的数组，还可以这样写：

    ```cpp
    typedef std::array<int, 10> MyIntArray;
    using MyIntArray = std::array<int, 10>;.
    ```

    别人写的一些模板类也可以做到：

    ```cpp
    inline void nodeleter(void*) {}

    /// Array of T with ownership. Like \see std::unique_ptr<T[]> but with size tracking.
    /// @tparam T Element type.
    template <typename T>
    class unique_array : public std::unique_ptr<T[],void (*)(void*)>
    {   size_t Size;
    private:
        typedef std::unique_ptr<T[],void (*)(void*)> base;
    protected:
        unique_array(T* ptr, size_t size, void (*deleter)(void*)) noexcept : base(ptr, deleter), Size(size) {}
        void reset(T* ptr, size_t size) noexcept { base::reset(ptr); Size = size; }
    public:
        constexpr unique_array() noexcept : base(nullptr, operator delete[]), Size(0) {}
        explicit unique_array(size_t size) : base(new T[size], operator delete[]), Size(size) {}
        template <size_t N> unique_array(T(&arr)[N]) : base(arr, &nodeleter), Size(N) {}
        unique_array(unique_array<T>&& r) : base(move(r)), Size(r.Size) { r.Size = 0; }
        void reset(size_t size = 0) { base::reset(size ? new T[size] : nullptr); Size = size; }
        void swap(unique_array<T>&& other) noexcept { base::swap(other); std::swap(Size, other.Size); }
        void assign(const unique_array<T>& r) const { assert(Size == r.Size); std::copy(r.begin(), r.end(), begin()); }
        const unique_array<T>& operator =(const unique_array<T>& r) const { assign(r); return *this; }
        size_t size() const noexcept { return Size; }
        T* begin() const noexcept { return base::get(); }
        T* end() const noexcept { return begin() + Size; }
        T& operator[](size_t i) const { assert(i < Size); return base::operator[](i); }
        unique_array<T> slice(size_t start, size_t count) const noexcept
        {   assert(start + count <= Size); return unique_array<T>(begin() + start, count, &nodeleter); }
    };
    ```

    模板类：

    ```cpp
    #ifndef LEFTICUS_TOOLS_SIMPLE_STACK_VECTOR_HPP
    #define LEFTICUS_TOOLS_SIMPLE_STACK_VECTOR_HPP

    #include <array>
    #include <cstdint>
    #include <stdexcept>
    #include <vector>

    namespace lefticus::tools {


    // changes from std::vector
    //  * capacity if fixed at compile-time
    //  * it never allocates
    //  * items must be default constructible
    //  * items are never destroyed until the entire stack_vector
    //    is destroyed.
    //  * iterators are never invalidated
    //  * capacity() and max_size() are now static functions
    //  * should be fully C++17 usable within constexpr
    template<typename Contained, std::size_t Capacity> struct simple_stack_vector
    {
    using value_type = Contained;
    using data_type = std::array<value_type, Capacity>;
    using size_type = typename data_type::size_type;
    using difference_type = typename data_type::difference_type;
    using reference = value_type &;
    using const_reference = const value_type &;

    static_assert(std::is_default_constructible_v<Contained>);

    using iterator = typename data_type::iterator;
    using const_iterator = typename data_type::const_iterator;
    using reverse_iterator = typename data_type::reverse_iterator;
    using const_reverse_iterator = typename data_type::const_reverse_iterator;

    constexpr simple_stack_vector() = default;
    constexpr explicit simple_stack_vector(std::initializer_list<value_type> values)
    {
        for (const auto &value : values) { push_back(value); }
    }

    template<typename OtherContained, std::size_t OtherSize>
    constexpr explicit simple_stack_vector(const simple_stack_vector<OtherContained, OtherSize> &other)
    {
        for (const auto &value : other) { push_back(Contained{ value }); }
    }

    template<typename Type> constexpr explicit simple_stack_vector(const std::vector<Type> &values)
    {
        for (const auto &value : values) { push_back(Contained{ value }); }
    }

    template<typename Itr> constexpr simple_stack_vector(Itr begin, Itr end)
    {
        while (begin != end) {
        push_back(*begin);
        ++begin;
        }
    }

    [[nodiscard]] constexpr iterator begin() noexcept { return data_.begin(); }

    [[nodiscard]] constexpr const_iterator begin() const noexcept { return data_.cbegin(); }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return data_.cbegin(); }

    [[nodiscard]] constexpr iterator end() noexcept
    {
        return std::next(data_.begin(), static_cast<difference_type>(size_));
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept
    {
        return std::next(data_.cbegin(), static_cast<difference_type>(size_));
    }

    [[nodiscard]] constexpr value_type &front() noexcept { return data_.front(); }
    [[nodiscard]] constexpr const value_type &front() const noexcept { return data_.front(); }
    [[nodiscard]] constexpr value_type &back() noexcept { return data_.back(); }
    [[nodiscard]] constexpr const value_type &back() const noexcept { return data_.back(); }

    [[nodiscard]] constexpr const_iterator cend() const noexcept { return end(); }

    [[nodiscard]] constexpr reverse_iterator rbegin() noexcept
    {
        return std::next(data_.rbegin(), static_cast<difference_type>(Capacity - size_));
    }

    [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept
    {
        return std::next(data_.crbegin(), static_cast<difference_type>(Capacity - size_));
    }
    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }

    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] constexpr reverse_iterator rend() noexcept { return data_.rend(); }

    [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept { return data_.crend(); }

    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return data_.crend(); }

    template<typename Value> constexpr value_type &push_back(Value &&value)
    {
        if (size_ == Capacity) { throw std::length_error("push_back would exceed static capacity"); }
        data_[size_] = std::forward<Value>(value);
        return data_[size_++];
    }

    template<typename... Param> constexpr value_type &emplace_back(Param &&...param)
    {
        if (size_ == Capacity) { throw std::length_error("emplace_back would exceed static capacity"); }
        data_[size_] = value_type{ std::forward<Param>(param)... };
        return data_[size_++];
    }

    [[nodiscard]] constexpr value_type &operator[](const std::size_t idx) noexcept { return data_[idx]; }

    [[nodiscard]] constexpr const value_type &operator[](const std::size_t idx) const noexcept { return data_[idx]; }

    [[nodiscard]] constexpr value_type &at(const std::size_t idx)
    {
        if (idx > size_) { throw std::out_of_range("index past end of stack_vector"); }
        return data_[idx];
    }

    [[nodiscard]] constexpr const value_type &at(const std::size_t idx) const
    {
        if (idx > size_) { throw std::out_of_range("index past end of stack_vector"); }
        return data_[idx];
    }

    // resets the size to 0, but does not destroy any existing objects
    constexpr void clear() { size_ = 0; }


    // cppcheck-suppress functionStatic
    constexpr void reserve(size_type new_capacity)
    {
        if (new_capacity > Capacity) { throw std::length_error("new capacity would exceed max_size for stack_vector"); }
    }

    // cppcheck-suppress functionStatic
    [[nodiscard]] constexpr static size_type capacity() noexcept { return Capacity; }

    // cppcheck-suppress functionStatic
    [[nodiscard]] constexpr static size_type max_size() noexcept { return Capacity; }

    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }


    constexpr void resize(const size_type new_size)
    {
        if (new_size <= size_) {
        size_ = new_size;
        } else {
        if (new_size > Capacity) {
            throw std::length_error("resize would exceed static capacity");
        } else {
            auto old_end = end();
            size_ = new_size;
            auto new_end = end();
            while (old_end != new_end) {
            *old_end = data_type{};
            ++old_end;
            }
        }
        }
    }

    constexpr void pop_back() noexcept { --size_; }

    // cppcheck-suppress functionStatic
    constexpr void shrink_to_fit() noexcept
    {
        // nothing to do here
    }


    private:
    // default initializing to make it more C++17 friendly
    data_type data_{};
    size_type size_{};
    };


    template<typename Contained, std::size_t LHSSize, std::size_t RHSSize>
    [[nodiscard]] constexpr bool operator==(const simple_stack_vector<Contained, LHSSize> &lhs,
    const simple_stack_vector<Contained, RHSSize> &rhs)
    {
    if (lhs.size() == rhs.size()) {
        for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
        if (lhs[idx] != rhs[idx]) { return false; }
        }
        return true;
    }

    return false;
    }

    }// namespace lefticus::tools


    #endif
    ```

    学习一下`std::span`.