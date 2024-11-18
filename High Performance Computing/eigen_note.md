# Eigen Note

eigen 是一个线性代数库，可以做很多的矩阵运算。

official site: <https://eigen.tuxfamily.org/index.php?title=Main_Page>

tutorial: <https://eigen.tuxfamily.org/dox/GettingStarted.html>

## cache

* 可以直接使用 apt 安装这个库

    `apt install libeigen3-dev`

    在包含头文件时，则需要使用`#include <eigen3/Eigen/Dense>`

## note

eigen 是一个头文件库，不需要编译，只要设置好 include 目录就可以了。

Example:

`main.cpp`:

```cpp
#include <iostream>
#include <Eigen/Dense>
using std::cout, std::endl;
using Eigen::MatrixXd;
 
int main()
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    cout << m << endl;
    return 0;
}
```

`Makefile`:

```makefile
main: main.cpp
	g++ -g main.cpp -I/home/hlc/Softwares/eigen/eigen-3.4.0/ -o main
```

compile command:

```bash
make
```

run:

```bash
./main
```

output:

```
  3  -1
2.5 1.5
```