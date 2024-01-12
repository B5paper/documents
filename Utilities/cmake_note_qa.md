# cmake note qa

[unit]
[u_0]
请给下面一个 hello world 程序写一个`CMakeLists.txt`文件。

`main.cpp`

```cpp
#include <iostream>
using namespace std;

int main()
{
    cout << "hello" << endl;
    return 0;
}
```
[u_1]
`CMakeLists.txt`:

```cmake
project(hello)
add_executable(main main.cpp)
```

编译：

```bash
mkdir build && cd build
cmake ..
make
```

运行：

```
./main
```

[unit]
[u_0]
使用 debug 模式编译项目。
[u_1]
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

[uint]
[u_0]
编译一个自定义的库，要求输出`libmymath.so`

`my_math.h`:

```cpp
int times(int a, int b);
```

`my_math.cpp`:

```cpp
#include "my_math.h"

int times(int a, int b)
{
    return a * b;
}
```
[u_1]

`CMakeLists.txt`:

```cmake
project(aaa)
add_library(bbb SHARED my_math.cpp)
```

编译：

```bash
mkdir build && cd build
cmake ..
make
```

此时会在`build`目录下生成`libbbb.so`。

[unit]
[u_0]
根据以下代码，编译一个库`libhmath.so`。

`hmath.h`:

```cpp

```

`hmath.cpp`:

```cpp

```

[u_1]
(empty)

[unit]
[u_0]
使用 cmake 根据下面的代码编译出`libhmath.so`库，然后编译出可执行文件。

`main.cpp`:

```cpp
#include "mymath.h"
#include <iostream>
using namespace std;

int main()
{
    int a = 3, b = 5;
    int c = times(a, b);
    cout << a << " * " << b << " = " << c << endl;
    return 0;
}
```
[u_1]
`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.17)
project(MyProject)
add_executable(main main.cpp)
target_include_directories(main PUBLIC .)
target_link_directories(main PUBLIC .)
target_link_libraries(main mymath)
```

[unit]
[u_0]
创建一个名为`hello_msg`的变量，其值为`"hello world"`，并将其打印出来。
[u_1]
```cmake
set(hello_msg "hello world")
message(${hello_msg})
```

[unit]
[u_0]
使用`greeting_msg="hello world" cmake ..` 执行 cmake，打印出环境变量`greeting_msg`，并且设置一个新的环境变量`aaa=bbb`
[u_1]
(empty)

[unit]
[u_0]
使用 if 语句，当变量`a`的值为`haha`时输出`hello`，当变量`a`的值为`hehe`时输出`world`。
先让`a`的值为`haha`，再让`a`的值为`hehe`，分别执行 if 语句。
[u_1]
(empty)
