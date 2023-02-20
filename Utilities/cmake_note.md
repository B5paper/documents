# CMake Note

官方的 tutorial 写得并不是很好。可以参考这篇 tutorial。

Tutorials:

* <<https://medium.com/@onur.dundar1/cmake-tutorial-585dd180109b>>

* <https://cliutils.gitlab.io/modern-cmake/>

CMake 用于跨平台，跨编程语言，跨编译器地构建编译系统，这个编译系统通常是 Makefile 文件。

我们通过编写`CMakeLists.txt`文件和`.cmake`文件，来控制编译流程。这两种文件的说明如下：

* `CMakeLists.txt` file is placed at the source of the project you want to build.

* `CMakeLists.txt` is placed at the root of the source tree of any application, library it will work for.

* If there are multiple modules, and each module can be compiled and built separately, `CMakeLists.txt` can be inserted into the sub folder.

* `.cmake` files can be used as scripts, which runs `cmake` command to prepare environment pre-processing or split tasks which can be written outside of `CMakeLists.txt`.

* `.cmake` files can also define modules for projects. These projects can be separated build processes for libraries or extra methods for complex, multi-module projects.

一个 example:

```cmake
cmake_minimum_required(VERSION 3.20)

# set the project name
project(HelloWorld)

# add the executable
add_executable(HelloWorld main.cpp)
```

可以使用`cmake CMakeLists.txt`打印出详细的配置信息。

其中`cmake_minimum_required()`，`project()`这些类似于函数的代码，叫做 CMake commands。

一些常用的 built-in commands:

* `message`: prints given message

* `cmake_minimum_required`: sets minimum version of cmake to be uses

* `add_executable`: adds executable target with given name.

* `add_library`: adds a library target to be build from listed source files

* `add_subdirectory`: adds a subdirectory to build

还有一些类似编程语言的语句：

* `if`, `elif`, `endif`

* `while`, `endwhile`

* `foreach`, `endforeach`

* `list`

* `return`

* `set_property(assign value to variable)`

更多的 cmake commands 可以在文档中查到：<https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html>

CMake 中建议使用缩进，但并不是强制的。CMake 不使用`;`来标记每行的结尾。

`cmake --build .`: build the target directly. For alternative, we can just run `make` instead.

`project(myproj VERSION 1.0)`: set the project name and version.

`configure_file(TutorialConfig.h.in TutorialConfig.h)`: 不明白这个干什么用的.

```cmake
target_include_directories(Tutorial PUBLIC "${PROJECT_BINARY_DIR}")
```

add the search path of include files.

```cmake
# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

如果调用其他库，假设现在有一个`MathFunctions`文件夹，里面有`MathFunctions.h`和`mysqrt.cxx`文件，其中有函数：`mysqrt()`。我们需要在这个文件夹下创建一个`CMakeLists.txt`，文件中包含以下内容：

```cmake
add_library(MathFunctions mysqrt.cxx)
```

在主程序中，需要在`CMakeLists.txt`中加入:

`add_subdirectory(MathFunctions)`

添加一个选项：

`option(USE_MYMATH "Use tutorial provided math implementation" ON)`

在 cmake 中可以使用`if`语句来利用选项：

```cmake
if(USE_MATH)
    add_subdirectory(MathFunctions)
    list(APPEND EXTRA_LIBS MathFunctions)
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# add the executable
add_executable(Tutorial totorial.cxx)
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC 
"${PROJECT_BINARY_DIR}"
${EXTRA_INCLUDES}
)
```

我们首先需要在 config 文件中添加宏：

`TutorialConfig.h`:

`#cmakedefine USE_MYMATH`

此时在我们的 cpp 文件中，就可以使用`USE_MYMATH`宏了：

```
#ifdef USE_MYMATH
# include "MathFunctions.h"
#endif

#ifdef USE_MYMATH
    const double outputValue = mysqrt(inputValue);
#else
    const double outputValue = sqrt(inputValue);
#endif
```

生成一个 static library:

`add_library(test_library STATIC calc.cpp)`

添加头文件：

`include_directories(includes/math)`

`include_directories(includes/general)`

These two commands make the headers located in `general` and `math` available for including from the sources of all targets.

cmake 有一些预定义变量，用于设置编译器，测试环境之类的。常见的预定义变量如下：

* `CMAKE_CXX_FLAGS`：指定编译器的参数。

    比如`set(CMAKE_CXX_FLAGS "-Wall")`

    还可以只添加，不覆盖：`set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")`

* `CMAKE_BINARY_DIR`: Full path to top level of build tree and binary output folder, by default it is defined as top level of build tree.

* `CMAKE_HOME_DIRECTORY`: Path to top of source tree.

* `CMAKE_SOURCE_DIR`: Full path to top level of source tree

* `CMAKE_INCLUDE_PATH`: Path used to find file, path

cmake 中，commands 是大小写不敏感的，变量是大小写敏感的。对变量进行命名时，可以使用短连接号`-`。

有关 variables 的参考：<https://cmake.org/cmake/help/v3.0/manual/cmake-language.7.html#variables>

全部 predefined variables 参考：<https://cmake.org/cmake/help/v3.0/manual/cmake-variables.7.html#manual:cmake-variables(7)>

可以使用`${<variable_name>}`得到一个变量的值：

```cmake
message("CXX Standard: ${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD 14)
```

我们也可以定义自己的变量：

```cmake
set(TRIAL_VARIABLE "VALUE")
message("${TRIAL_VARIABLE}")
```

CMake 中的所有变量都是字符串类型。list 是被分号`;`分隔的字符串：

```cmake
set(files a.txt b.txt c.txt)
# sets files to "a.txt;b.txt;c.txt"
```

可以用`foreach` command 将 list 中的字符串提取出来：

```cmake
foreach(file ${files})
    message("Filename: ${file}")
endforeach()
```

创建 debug 版本的 build system: `cmake -DCMAKE_BUILD_TYPE=Debug -H.  -Bbuild/Debug`

创建 release 版本的 build system: `cmake -DCMAKE_BUILD_TYPE=Release -H. -Bbuild/Release`

platform checks: <https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Write-Platform-Checks>