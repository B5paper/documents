# C++ Notes

## Variable

| Type | Space | Range |
| - | - | - |
| short | 2 bytes | [-2^15, 2^15-1], [-32768, 32767] |
| int | 4 bytes | [-2^31, 2^31-1] |
| long | 4 bytes or 8 bytes | [-2^31, 2^31-1] |
| long long | 8 bytes | [-2^63, 2^63-1] |
| float | 4 bytes | 7 位有效数字 |
| double | 8 bytes | 15 ~ 16 位有效数字 |
| bool | 1 bytes | true, false |

字面常量小数默认为`double`类型。

默认情况下，输出一个小数，会显示出 6 位有效数字。

对于`char`类型变量，`0 ~ 31`存储的是非可打印字符，`32 ~ 127`存储的是可打印字符。

字符串定义：

```c++
char str[] = "hello, world!";
```

## 数组

数组与指针：

```c++
int arr[3] = {1, 2, 3};
cout << (int)arr << endl;  // 数组名即数组首地址
cout << (int)&arr[0] << endl;  // 第一个元素的地址
cout << (int)&arr[1] << endl;  // 第二个元素的地址
```

二维数组的定义：

```c++
int arr[row][col];
int arr[row][col] = {{data_1, data_2}, {data_3, data_4}, ...};
int arr[row][col] = {data_1, data_2, data_3, data_4, ...};
int arr[][col] = {data_1, data_2, data_3, data_4, ...};
```


## 运算符

有符号整数的`>>`，左侧会位移`1`。

前置递增：先递增变量，再计算表达式值。

后置递增：计算表达式值/传递函数参数，再递增变量。

三目运算符：`condition ? statement_1 : statement_2`。

## 有关数组的初始化

1. 全局数组中的每个元素会被编译器初始化为 0。

1. 局部数组编译器不会自动初始化。通常可以用`memset`将其初始化为 0：`memset(arr, 0, sizeof(arr));`。`memset`是以字节为单位对内存数据进行赋值的，因此如果赋的值非零，就不能用`memset`了。

1. 如果只初始化了几个值，那么剩下的数据似乎会被自动初始化为 0：

    ```c++
    int main()
    {
        int arr[5] = {1, 2};  // {1, 2, 0, 0, 0}
        return 0;
    }
    ```

## 函数

**有关引用**

1. 不要返回局部变量的引用

    ```c++
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

    ```c++
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

```c++
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

**函数重载**

1. 只修改函数的返回值类型不能视为重载

1. 函数重载必须在同一作用域下

1. `const`可以作为不同的类型吗？

    ```c++
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

## Class

`struct`:

```c++
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

**构造函数**

```c++
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

构造函数的创建规则：

1. 若用户定义了构造函数，则不提供默认构造函数，但仍提供默认拷贝构造函数

1. 若用户定义了拷贝构造函数，则不提供其他的构造函数

编译器给每个`class`默认提供 4 种函数：

1. 默认构造函数

1. 默认析构函数

1. 默认拷贝构造函数，对属性进行值拷贝

1. 赋值运算符`operator=`，对属性进行值拷贝

**拷贝构造函数**

```c++
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

```c++
A a;
A(a);  // equals A a;
```

调用拷贝构造函数的时机：

1. 用已存在的对象初始化另一个对象

    ```c++
    A a;
    A b(a);
    A b = a;
    ```

1. 函数按值传递参数

    ```c++
    int func(A a) {}
    int main()
    {
        A a;
        func(a);
        return 0;
    }
    ```

1. 函数按值返回对象

当其它类对象作为本类成员，构造时先构造类对象，再构造自身，析构的顺序相反。

有关拷贝构造函数一些需要注意的地方：

1. 传入的参数可以是`const`对象的引用，也可以是非`const`对象的引用。但是通常将形参设置为`const`对象的引用。因为这样既可以传入`const`对象，也可以传入非`const`对象。

C++ 为一个类默认提供 4 个函数：

1. 构造函数
1. 析构函数
1. 拷贝构造函数（按值浅拷贝）
1. 赋值运算符`operator=`（按值浅拷贝）

**静态成员**

1. 静态方法只能访问静态成员变量。

1. 静态成员变量在类内声明，在类外初始化。

    ```c++
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

## 继承

`public`继承无法访问基类中`private`字段的内容，其余字段中的内容不变；`protected`继承无法访问基类中`private`的内容，但会把`public`内容中的属性修改成`protected`；`private`继承无法访问基类中`private`的内容，但会把`public`和`protected`的内容改成`private`属性。

在继承时，父类中所有非静态成员属性都会被子类继承下去，但是被编译器隐藏了，访问不到。

继承中构造和析构的顺序：先构造父类，再构造子类；析构的顺序相反。

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

## 函数调用协议

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

**文件读写**

常用的 flags:

* `ios::in`：为读文件而打开文件
* `ios::out`：为写文件而打开文件
* `ios::ate`：初始位置：文件尾
* `ios::app`：追加方式写文件
* `ios::trunc`：如果文件存在，先删除，再创建
* `ios::binary`：二进制方式

可以用`|`组合多个 flags。

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

