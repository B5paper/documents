# Boost Note

boost 是一个 c++ template library，里面以头文件为主，只有少量库需要编译，大部分的库都不需要编译。

## cache

* `libboost-all-dev`

    `apt install libboost-all-dev`

    boost 的头文件会被安装到`/usr/include/boost`，预编译的 binary 会被安装到`/usr/lib/x86_64-linux-gnu/`。

    在使用 boost 库并编译自己的代码时，可能需要`g++ main.cpp -lboost_thread -lboost_system`这样编译。

    （仍然不太清楚，为什么 boost 主打 header only，但是仍有预编译的库？是否 header only 仅适用于模板库，预编译的库则是非模板库？）

## semaphore

一个经典的例子，两个线程按顺序增加一个变量。这里用 semaphore 来模拟一个 mutex。

`main.cpp`:

```cpp
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
using namespace boost::interprocess;
using ip_sem = interprocess_semaphore;

#include <thread>
using std::thread;
#include <iostream>
using std::cout, std::endl;

int count = 0;
ip_sem data_available(1);

void thd_func()
{
    for (int i = 0; i < 10; ++i)
    {
        data_available.wait();
        ++count;
        cout << count << endl;
        data_available.post();
    }
}

int main()
{
    thread thds[2];
    thds[0] = thread(thd_func);
    thds[1] = thread(thd_func);
    for (int i = 0; i < 2; ++i)
        thds[i].join();
    return 0;
}
```

`Makefile`:

```makefile
main: main.cpp
	g++ main.cpp -I/home/hlc/Softwares/boost/include -o main

clean:
	rm -rf main
```

编译：

```bash
make
```

运行：

```bash
./main
```

输出：

```
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
```

如果不使用 semaphore，可能得到这样的结果：

`main.cpp`:

```cpp
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
using namespace boost::interprocess;
using ip_sem = interprocess_semaphore;

#include <thread>
using std::thread;
#include <iostream>
using std::cout, std::endl;

int count = 0;
ip_sem data_available(1);

void thd_func()
{
    for (int i = 0; i < 10; ++i)
    {
        // data_available.wait();
        ++count;
        cout << count << endl;
        // data_available.post();
    }
}

int main()
{
    thread thds[2];
    thds[0] = thread(thd_func);
    thds[1] = thread(thd_func);
    for (int i = 0; i < 2; ++i)
        thds[i].join();
    return 0;
}
```

输出：

```
1
3
4
5
6
7
8
9
10
11
2
12
13
14
15
16
17
18
19
20
```

boost 中有关进程线程同步的问题，更详细的资料可以参考这里：

<https://www.boost.org/doc/libs/1_35_0/doc/html/interprocess/synchronization_mechanisms.html#interprocess.synchronization_mechanisms.semaphores>