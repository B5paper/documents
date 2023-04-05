# C++ Notes

微软出的 c/c++ tutorial，挺好的，有时间了看看：<https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/fopen-wfopen?view=msvc-170>

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

**string and `char*`**

字符串定义：

```c++
char str[] = "hello, world!";
```

## 数组

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
        int &ra = test();  // error
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

### 函数的默认参数

函数的声明和实现只能有一个有默认参数。

占位参数：

```cpp
int test(int a, int)
{
    return 0;
}

int test2(int a, int = 3)  // 点位参数也可以有默认参数
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

**注：在 windows 下，使用 mingw 的 g++/gcc 进行编译，好像没用，根本没有`@`。有机会再研究。**

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

### Reference

## 函数，指针与引用

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

The difference between getc() and getchar() is getc() can read from any input stream, but getchar() reads from standard input. So getchar() is equivalent to getc(stdin).

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

    宽字符版：`fgetws()`

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

## 谓词

返回`bool`的仿函数称为谓词。接收一个参数的谓词称为一元谓词，接收两个参数的谓词称为二元谓词。

## lvalue and rvalue

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

Determian whether the current thread is the main thread:

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

### Mutex and semephore

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

## Exception 异常处理

## 工程化

1. 头文件`xxx.h`中，只写对应`xxx.cpp`文件中出现的函数，尽量不写其他用到的头文件，仅保证头文件中没有语法错误即可。`xxx.cpp`中需要包含实现函数时用到的头文件。这样比较清晰，不容易乱。

## Topics

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

    函数指针：``

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