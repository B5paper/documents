# CMake Note

Materials:

1. quick cmake tutorial: <https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html>

官方的 tutorial 写得并不是很好。可以参考这篇 tutorial。

Tutorials:

* <https://medium.com/@onur.dundar1/cmake-tutorial-585dd180109b>

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

这些 commands 有时候先后顺序不敏感，有时候又会敏感。很奇怪。不清楚是声明式的还是命令式的。

还有一些类似编程语言的语句：

* `if`, `elif`, `endif`

* `while`, `endwhile`

* `foreach`, `endforeach`

* `list`

* `return`

* `set_property(assign value to variable)`

All conditional statements should be ended with its corresponding end command (endif, endwhile, endforeachetc)

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

CMake can generates a cache file that is designed to be used with a graphical editor. 

Build logic and definitions with CMake language is written either in CMakeLists.txt or a file ends with `<project_name>.cmake`. But as a best practice, main script is named as CMakeLists.txt instead of cmake.

## API Ref

* `add_executable`

    Add an executable to the project using the specified source files.

    Syntax (Normal Executables):

    ```s
    add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
    ```

    Adds an executable target called `<name>` to be built from the source files listed in the command invocation. The `<name>` corresponds to the logical target name and must be globally unique within a project. The actual file name of the executable built is constructed based on conventions of the native platform (such as `<name>.exe` or just `<name>`).

    设置生成可执行文件的名称以及其源文件。这里的源文件可以不写后缀。

    Syntax (Imported Executables):

    ```s
    add_executable(<name> IMPORTED [GLOBAL])
    ```

    Syntax (Imported Executables):

    ```s
    add_executable(<name> IMPORTED [GLOBAL])
    ```

* `add_library`

    * Normal Libraries

        Syntax:

        ```s
        add_library(<name> [STATIC | SHARED | MODULE]
                    [EXCLUDE_FROM_ALL]
                    [<source>...])
        ```

        Adds a library target called `<name>` to be built from the source files listed in the command invocation. The `<name> `corresponds to the logical target name and must be globally unique within a project. The actual file name of the library built is constructed based on conventions of the native platform (such as `lib<name>.a` or `<name>.lib`).

        `STATIC`, `SHARED`, or `MODULE` may be given to specify the type of library to be created. STATIC libraries are archives of object files for use when linking other targets. SHARED libraries are linked dynamically and loaded at runtime. MODULE libraries are plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality. If no type is given explicitly the type is STATIC or SHARED based on whether the current value of the variable `BUILD_SHARED_LIBS` is ON. For SHARED and MODULE libraries the POSITION_INDEPENDENT_CODE target property is set to ON automatically. A SHARED library may be marked with the FRAMEWORK target property to create an macOS Framework.

        Example:

        ```cmake
        project(HelloWorld)
        add_library(my_lib SHARED my_lib.cpp)
        add_executable(main my_lib main.cpp)
        ```

        在`add_executable()`的时候，只需要把自己的库加上去就可以了。

    * Object Libraries

        Syntax:

        ```s
        add_library(<name> OBJECT [<source>...])
        ```

    * Interface Libraries

        Syntax:

        ```s
        add_library(<name> INTERFACE)
        ```

    * Imported Libraries

        Syntax:

        ```s
        add_library(<name> <type> IMPORTED [GLOBAL])
        ```

    * Alias Libraries

        Syntax:

        ```s
        add_library(<name> ALIAS <target>)
        ```

* `add_subdirectory`

    Add a subdirectory to the build.

    Syntax:

    ```s
    add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL] [SYSTEM])
    ```

    Adds a subdirectory to the build. The `source_dir` specifies the directory in which the source `CMakeLists.txt` and code files are located. If it is a relative path, it will be evaluated with respect to the current directory (the typical usage), but it may also be an absolute path. The `binary_dir` specifies the directory in which to place the output files. If it is a relative path, it will be evaluated with respect to the current output directory, but it may also be an absolute path. If `binary_dir` is not specified, the value of `source_dir`, before expanding any relative path, will be used (the typical usage). The `CMakeLists.txt` file in the specified source directory will be processed immediately by CMake before processing in the current input file continues beyond this command.

* `include`

    Load and run CMake code from a file or module.

    Syntax:

    ```s
    include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>]
                      [NO_POLICY_SCOPE])
    ```

    Loads and runs CMake code from the file given. Variable reads and writes access the scope of the caller (dynamic scoping). If OPTIONAL is present, then no error is raised if the file does not exist. If RESULT_VARIABLE is given the variable <var> will be set to the full filename which has been included or NOTFOUND if it failed.

    If a module is specified instead of a file, the file with name <modulename>.cmake is searched first in CMAKE_MODULE_PATH, then in the CMake module directory. There is one exception to this: if the file which calls include() is located itself in the CMake builtin module directory, then first the CMake builtin module directory is searched and CMAKE_MODULE_PATH afterwards. See also policy CMP0017.

* `target_link_libraries`

    Specify libraries or flags to use when linking a given target and/or its dependents. Usage requirements from linked library targets will be propagated. Usage requirements of a target's dependencies affect compilation of its own sources.

    * Overview

        Syntax:

        ```s
        target_link_libraries(<target> ... <item>... ...)
        ```

        The named `<target>` must have been created by a command such as `add_executable()` or `add_library()` and must not be an ALIAS target. If policy `CMP0079` is not set to `NEW` then the target must have been created in the current directory. Repeated calls for the same <target> append items in the order called.

        Each `<item>` may be:

        * A library target name: The generated link line will have the full path to the linkable library file associated with the target. The buildsystem will have a dependency to re-link <target> if the library file changes.

            The named target must be created by add_library() within the project or as an IMPORTED library. If it is created within the project an ordering dependency will automatically be added in the build system to make sure the named library target is up-to-date before the <target> links.

            If an imported library has the IMPORTED_NO_SONAME target property set, CMake may ask the linker to search for the library instead of using the full path (e.g. /usr/lib/libfoo.so becomes -lfoo).

            The full path to the target's artifact will be quoted/escaped for the shell automatically.

        * A full path to a library file: The generated link line will normally preserve the full path to the file. The buildsystem will have a dependency to re-link <target> if the library file changes.

            There are some cases where CMake may ask the linker to search for the library (e.g. /usr/lib/libfoo.so becomes -lfoo), such as when a shared library is detected to have no SONAME field. See policy CMP0060 for discussion of another case.

            If the library file is in a macOS framework, the Headers directory of the framework will also be processed as a usage requirement. This has the same effect as passing the framework directory as an include directory.

            New in version 3.8: On Visual Studio Generators for VS 2010 and above, library files ending in .targets will be treated as MSBuild targets files and imported into generated project files. This is not supported by other generators.

            The full path to the library file will be quoted/escaped for the shell automatically.

        * A plain library name: The generated link line will ask the linker to search for the library (e.g. foo becomes -lfoo or foo.lib).

            The library name/flag is treated as a command-line string fragment and will be used with no extra quoting or escaping.

        * A link flag: Item names starting with -, but not -l or -framework, are treated as linker flags. Note that such flags will be treated like any other library link item for purposes of transitive dependencies, so they are generally safe to specify only as private link items that will not propagate to dependents.

            Link flags specified here are inserted into the link command in the same place as the link libraries. This might not be correct, depending on the linker. Use the LINK_OPTIONS target property or target_link_options() command to add link flags explicitly. The flags will then be placed at the toolchain-defined flag position in the link command.

            New in version 3.13: LINK_OPTIONS target property and target_link_options() command. For earlier versions of CMake, use LINK_FLAGS property instead.

            The link flag is treated as a command-line string fragment and will be used with no extra quoting or escaping.

        * A generator expression: A $<...> generator expression may evaluate to any of the above items or to a semicolon-separated list of them. If the ... contains any ; characters, e.g. after evaluation of a ${list} variable, be sure to use an explicitly quoted argument "$<...>" so that this command receives it as a single <item>.

            Additionally, a generator expression may be used as a fragment of any of the above items, e.g. foo$<1:_d>.

            Note that generator expressions will not be used in OLD handling of policy CMP0003 or policy CMP0004.

        * A debug, optimized, or general keyword immediately followed by another <item>. The item following such a keyword will be used only for the corresponding build configuration. The debug keyword corresponds to the Debug configuration (or to configurations named in the DEBUG_CONFIGURATIONS global property if it is set). The optimized keyword corresponds to all other configurations. The general keyword corresponds to all configurations, and is purely optional. Higher granularity may be achieved for per-configuration rules by creating and linking to IMPORTED library targets. These keywords are interpreted immediately by this command and therefore have no special meaning when produced by a generator expression.

        Items containing ::, such as Foo::Bar, are assumed to be IMPORTED or ALIAS library target names and will cause an error if no such target exists. See policy CMP0028.

    * Libraries for a Target and/or its Dependents

        Syntax:

        ```s
        target_link_libraries(<target>
                      <PRIVATE|PUBLIC|INTERFACE> <item>...
                     [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
        ```

        The `PUBLIC`, `PRIVATE` and `INTERFACE` scope keywords can be used to specify both the link dependencies and the link interface in one command.

        Libraries and targets following PUBLIC are linked to, and are made part of the link interface. Libraries and targets following PRIVATE are linked to, but are not made part of the link interface. Libraries following INTERFACE are appended to the link interface and are not used for linking <target>.

    * Libraries for both a Target and its Dependents

        Syntax:

        ```s
        target_link_libraries(<target> <item>...)
        ```

        Library dependencies are transitive by default with this signature. When this target is linked into another target then the libraries linked to this target will appear on the link line for the other target too. This transitive "link interface" is stored in the INTERFACE_LINK_LIBRARIES target property and may be overridden by setting the property directly. When CMP0022 is not set to NEW, transitive linking is built in but may be overridden by the LINK_INTERFACE_LIBRARIES property. Calls to other signatures of this command may set the property making any libraries linked exclusively by this signature private.

    * Linking Object Libraries

        Object Libraries may be used as the <target> (first) argument of target_link_libraries to specify dependencies of their sources on other libraries. For example, the code

        ```cmake
        add_library(A SHARED a.c)
        target_compile_definitions(A PUBLIC A)

        add_library(obj OBJECT obj.c)
        target_compile_definitions(obj PUBLIC OBJ)
        target_link_libraries(obj PUBLIC A)
        ```

        compiles obj.c with -DA -DOBJ and establishes usage requirements for obj that propagate to its dependents.

        Normal libraries and executables may link to Object Libraries to get their objects and usage requirements. Continuing the above example, the code

        ```cmake
        add_library(B SHARED b.c)
        target_link_libraries(B PUBLIC obj)
        ```

        compiles b.c with -DA -DOBJ, creates shared library B with object files from b.c and obj.c, and links B to A. Furthermore, the code

        ```cmake
        add_executable(main main.c)
        target_link_libraries(main B)
        ```

        compiles main.c with -DA -DOBJ and links executable main to B and A. The object library's usage requirements are propagated transitively through B, but its object files are not.

        Object Libraries may "link" to other object libraries to get usage requirements, but since they do not have a link step nothing is done with their object files. Continuing from the above example, the code:

        ```cmake
        add_library(obj2 OBJECT obj2.c)
        target_link_libraries(obj2 PUBLIC obj)

        add_executable(main2 main2.c)
        target_link_libraries(main2 obj2)
        ```

        compiles `obj2.c` with -DA -DOBJ, creates executable main2 with object files from main2.c and obj2.c, and links main2 to A.

        In other words, when Object Libraries appear in a target's INTERFACE_LINK_LIBRARIES property they will be treated as Interface Libraries, but when they appear in a target's LINK_LIBRARIES property their object files will be included in the link too.

* `set`

    Set a normal, cache, or environment variable to a given value.

    * Set Normal Variable

        Syntax:

        ```s
        set(<variable> <value>... [PARENT_SCOPE])
        ```

        Set or unset `<variable>` in the current function or directory scope:

        * If at least one `<value>...` is given, set the variable to that value.

        * If no value is given, unset the variable. This is equivalent to `unset(<variable>)`.

        If the `PARENT_SCOPE` option is given the variable will be set in the scope above the current scope. Each new directory or function() command creates a new scope. A scope can also be created with the block() command. This command will set the value of a variable into the parent directory, calling function or encompassing scope (whichever is applicable to the case at hand). The previous state of the variable's value stays the same in the current scope (e.g., if it was undefined before, it is still undefined and if it had a value, it is still that value).

        The block(PROPAGATE) and return(PROPAGATE) commands can be used as an alternate method to the set(PARENT_SCOPE) and unset(PARENT_SCOPE) commands to update the parent scope.

    
    * Set Cache Entry

        Syntax:

        ```s
        set(<variable> <value>... CACHE <type> <docstring> [FORCE])
        ```

    * Set Environment Variable

        Syntax:

        ```s
        set(ENV{<variable>} [<value>])
        ```

* `message`

    Log a message.

    * General messages

        Syntax:

        ```s
        message([<mode>] "message text" ...)
        ```

        Record the specified message text in the log. If more than one message string is given, they are concatenated into a single message with no separator between the strings.

        The optional <mode> keyword determines the type of message, which influences the way the message is handled:

        * `FATAL_ERROR`

            CMake Error, stop processing and generation.

            The cmake(1) executable will return a non-zero exit code.

        * `SEND_ERROR`

            CMake Error, continue processing, but skip generation.

        * `WARNING`

            CMake Warning, continue processing.

        * `AUTHOR_WARNING`

            CMake Warning (dev), continue processing.

        * `DEPRECATION`

            CMake Deprecation Error or Warning if variable CMAKE_ERROR_DEPRECATED or CMAKE_WARN_DEPRECATED is enabled, respectively, else no message.

        * `(none)` or `NOTICE`

            Important message printed to stderr to attract user's attention.

        * `STATUS`

            The main interesting messages that project users might be interested in. Ideally these should be concise, no more than a single line, but still informative.

        * `VERBOSE`

            Detailed informational messages intended for project users. These messages should provide additional details that won't be of interest in most cases, but which may be useful to those building the project when they want deeper insight into what's happening.

        * `DEBUG`

            Detailed informational messages intended for developers working on the project itself as opposed to users who just want to build it. These messages will not typically be of interest to other users building the project and will often be closely related to internal implementation details.

        * `TRACE`

            Fine-grained messages with very low-level implementation details. Messages using this log level would normally only be temporary and would expect to be removed before releasing the project, packaging up the files, etc.


    * Configure Log

        Syntax:

        ```s
        message(CONFIGURE_LOG <text>...)
        ```

* `find_package`

    Find a package (usually provided by something external to the project), and load its package-specific details. Calls to this command can also be intercepted by dependency providers.

    * Search Modes

        

    * Basic Signature

        Syntax:

        ```s
        find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [REGISTRY_VIEW  (64|32|64_32|32_64|HOST|TARGET|BOTH)]
             [GLOBAL]
             [NO_POLICY_SCOPE]
             [BYPASS_PROVIDER])
        ```

## Problems shooting

1. `file RPATH_CHANGE could not write new RPATH:`

    后面会跟一个`xxx.so`文件的路径。

    这个错误原因是 llvm 在编译的时候，主机上内存不够导致编译中途崩溃，但是仍然创建了一些大小为 0 的`xxx.so`文件。第二次编译时，编译器发现了这些已经存在的文件，无法重新写入。

    解决办法：

    使用`ls -lh`查看编译目录下所有大小为 0 的`xxx.so`空文件，把它们都删掉就好了。

    注：

    llvm 在编译时会占用大量内存，有些文件在 link 时候占用内存达到 12G 左右，所以编译线程数不要设置太大。