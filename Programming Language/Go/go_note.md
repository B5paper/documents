# Go Note

下载和安装：

在官网上下载需要的版本：<https://go.dev/doc/install>

解压后，把`bin`文件夹添加到`PATH`里就可以了。

使用`go version`检查是否安装成功。

## cache

* 既然 Go 这么强，为什么没有人用 Go 去写算子？ GPU 算子。

    这问题问到**高性能计算 / AI 框架**的根上了。
    一句话先给结论：
    **不是 Go 不强，而是 GPU 算子根本不吃“语言性能”这一套，Go 的优势完全用不上，短板却全踩中了。**

    下面用最直白、工程化的逻辑讲清楚。

    ---

    1. 写 GPU 算子，比拼的根本不是语言本身速度

        算子（kernel）真正的性能瓶颈只有三个：
        1. **GPU 指令发射效率**
        2. **显存读写、cache 复用**
        3. **向量化、张量排布、访存模式**

        这些**完全由编译器（NVCC/MLIR/LLVM）和硬件决定**，跟你用 C++、Go、Java 没关系。

        - C++ 写算子 → 编译成 PTX → GPU 执行
        - Go 写算子 → 也得编译成 PTX → 一样执行

        **语言本身的执行速度 ≠ 算子在 GPU 上的速度**
        Go 再快，也快不到 GPU 里去。

    ---

    2. 算子生态被两条路锁死了，Go 根本挤不进去

        （1）CUDA / C++ / Triton 垄断生态
        所有成熟框架：
        - PyTorch
        - TensorFlow
        - TVM / XLA
        - OneDNN / MKL / CUTLASS / CUB

        **底层全是 C++ + CUDA C / PTX / Triton**

        框架和算子之间是强绑定的：
        - 内存布局
        - 类型系统
        - 设备管理
        - 流/事件同步
        - 分布式 NCCL

        Go 想插一脚，等于**重写整个深度学习栈**。

        （2）Python 负责上层，C++ 负责底层，是行业最优解
        - Python：灵活、快写、生态爆炸
        - C++/CUDA：极致性能、零开销

        **Go 夹在中间，两头不讨好**
        - 比灵活不如 Python
        - 比底层控制、零开销、编译器生态不如 C++

    ---

    3. Go 天生不适合写 GPU 算子的几个致命问题

        （1）Go 没有精细内存控制
        算子需要：
        - 手动内存对齐
        - 裸指针操作
        - 非安全内存拷贝
        - 精细化显存池管理
        - 与 CUDA Runtime 零成本交互

        Go 的设计理念就是**杜绝 unsafe、杜绝裸指针、杜绝手动管理**。
        你要用 Go 写算子 = 全程 `unsafe`，等于废掉 Go 最核心优势。

        （2）Go 的 runtime 会干扰 GPU 执行
        Go 自带：
        - GMP 调度
        - GC
        - 协程抢占

        这些对**高性能 CUDA 流、同步、核函数启动**都是干扰。
        算子要求**精确控制 CPU 线程**，Go 做不到。

        （3）没有编译器支持
        CUDA 编译器只认 C++ / C。
        Go 想要编译 GPU 代码，必须：
        - 调用外部 NVCC
        - 用 CGO 桥接
        - 本质还是 C++ 写算子

        **CGO 开销巨大，完全抵消 Go 优势**

        （4）算子需要极致的 AOT 编译与优化
        - 模板元编程
        - 常量折叠
        - 向量化展开
        - 针对不同 SM 架构特化

        这些 C++ 天生拉满，Go 完全不擅长。

    ---

    4. 那 Go 适合干什么？

        Go 的优势是：
        - 网络服务
        - 高并发后端
        - 云原生
        - 调度、网关、控制面
        - 低延迟服务端

        **它是写“调度器”的语言，不是写“计算核”的语言。**

        类比：
        - Go = 调度大卡车的**交通指挥中心**
        - C++/CUDA = 卡车**发动机**

        你不会用指挥中心去造发动机。

    ---

    5. 真正用来写 GPU 算子的语言是什么？

        - **高性能算子：C++ + CUDA C**
        - **自动生成算子：Triton（类 Python）**
        - **未来方向：MLIR、方言 IR、专用编译器**
        - **绝对不会是：Go、Java、Python（运行时层不算）**

    ---

    **终极总结**

    1. GPU 算子性能不靠语言，靠**编译器和硬件**
    2. Go 的优势（并发、简单、安全）在算子领域完全没用
    3. Go 的劣势（无裸指针、无精细内存控制、runtime干扰、CGO开销）全踩中
    4. 生态被 C++/CUDA 锁死，没有替换动力
    5. Go 适合写**框架宿主、调度、服务端**，不适合写**计算核心**

    一句话：
    **Go 很强，但它强的地方，跟 GPU 算子没关系。**

    如果你愿意，我可以给你画一张
    **AI框架分层：哪一层用 Go、哪一层用 C++、哪一层用 CUDA、哪一层用 Python**
    的清晰结构图。

* Go 编译出来不是 binary 文件吗？为什么还有 runtime？

    这是一个**特别关键、特别容易被误解**的点。
    我用最直白、最底层的逻辑给你讲清楚：

    **一句话先破误区**
    **Go 编译出来确实是单个二进制文件（binary），但它里面自带了一整个 runtime 虚拟机。**
    它不是像 C 那样“光秃秃的机器码”，而是**自带运行环境的可执行文件**。

    ---

    1. 什么是 Go runtime？

        Go 编译后的二进制里，**静态链接了一整套核心组件**：
        - **GMP 调度器**（管理 goroutine）
        - **GC 垃圾回收器**
        - **内存分配器**（mspan、mcache）
        - **信号处理**
        - **栈管理、栈扩容**
        - **channel 实现**
        - **系统调用封装**

        这些代码**被直接编译进你的可执行文件**，所以：
        - 不需要外部依赖
        - 是单个 binary
        - **但运行时必须跑这套 runtime**

        这就像：
        - C 程序 = 一个人赤手空拳跑
        - Go 程序 = 一个人**开着一辆车**跑
        车就是 runtime。

    ---

    2. 为什么 C++ 没有 runtime？

        C++ 编译后是**纯机器码**，直接跑在 CPU 上：
        - 自己管理栈、堆
        - 自己调用系统调用
        - 没有 GC
        - 没有调度器
        - 没有自动协程

        C++ 只有标准库（libc/libstdc++），**没有语言 runtime**。

    ---

    3. 为什么 Go 一定要带 runtime？

        因为 Go 的核心特性**全部依赖 runtime 实现**：
        1. **goroutine** —— 不是 OS 线程，是 runtime 模拟的轻量级线程
        2. **channel** —— runtime 内部实现的同步队列
        3. **GC** ——  runtime 定时扫描回收
        4. **defer** —— runtime 管理链表
        5. **panic/recover** —— runtime 处理异常
        6. **网络轮续（netpoller）** —— runtime 接管网络 IO

        **没有 runtime，Go 语言本身就不存在。**

    ---

    4. 那为什么它还是 binary？

        因为 Go 采用**静态链接**：
        - 把 runtime
        - 把标准库
        - 把你的代码

        全部**编译、链接成一个独立文件**。

        它是 binary，但**不是精简的机器码 binary**，
        而是**自带虚拟机的 binary**。

        类比更清楚：
        - Java：`.class` + 外部 JVM
        - Go：`可执行文件` = **代码 + 内置迷你JVM**
        - C++：`可执行文件` = 纯代码

    ---

    5. 这对写 GPU 算子意味着什么？

        回到你最开始关心的问题：

        Go 不适合写 GPU 算子，原因就在这里：
        1. **runtime 会抢占线程**
           GPU 启动 kernel 需要精确、低延迟、不被打断的 CPU 线程。
           Go runtime 会随时调度、抢占、切换 goroutine，**会造成启动抖动**。

        2. **GC 会随时暂停**
           高并发场景下 GC 停顿虽然小，但**算子启动路径不允许任何意外暂停**。

        3. **无法精确控制线程绑定**
           CUDA 喜欢一个 CPU 线程绑一个 GPU 流，Go 不允许你控制线程。

        4. **无法裸指针零开销操作**
           Go 的 unsafe 只是勉强能用，不是设计用来做极致底层的。

    ---

    **终极总结（最精炼版）**

    - Go 是 binary ✅
    - 但它是 **自带 runtime 的 binary** ✅
    - runtime = 内置调度器 + GC + 内存管理 ✅
    - 这让 Go 不适合做**极致底层、极致低延迟、精确控制线程**的事情
    - 所以**不适合写 GPU 算子**

    一句话：
    **Go 是自带引擎的汽车，C++ 是直接踩油门的赛车。**

    需要我给你对比一下 **C++、Go、Java 的 runtime 差异** 吗？一眼就能看懂三者本质区别。


## Hello, world

`main.go`：

```go
package main
import "fmt"  // 必须要用

func main() {  // 大括号必须在这里
    fmt.Println("hello world")  // 语句后没有分号
}
```

编译：

`go build main.go`

编译完后会生成二进制文件。

运行：

`./main`

编译和运行：

`go run xxx.go`

## 项目管理

项目管理：

go 会从环境变量`GOPATH`中找包目录。每个项目都要有一个`main`包，并且有且只有一个`main`函数。

```go
package main
import . "fmt"  // 可以直接调用 fmt 里的函数

func main() {
    Println("this is a test")
}
```

给包起别名：

```go
import io "fmt"

func main() {
    io.Println("this is a test")
}
```

忽略某个包：

```go
import _ "fmt"
// 或者
import {
    _ "fmt"
}
```

这样用于调用包中的`init`函数，而不使用这个包。 

可以用`go env`显示环境变量。

同一个目录，包名必须一样。同一个目录，调用别的文件里的函数，直接调用即可，无须包名。

不同目录，包名不一样。子包里可以写成

```go
package calc
```

包中函数名称如果小写，那么为包的私有函数。首字母如果大写，那么可以被外部调用。

导入包时，会先执行`init`函数。

`GOBIN`目录用于存储

自动生成`bin`或`pkg`目录，需要使用`go install`命令。 

## 变量

变量名：字母，下划线，数字

变量声明：

```go
var a int
var b, c int
a = 10
var d int = 10
c := 30  // 自动推导类型
a, b := 10, 20

var a float64
var (
    a int
    b float64 = 2.0
    c = 3
)
```

1. 声明的变量必须使用

    如果存在未被使用的变量，那么会无法通过编译。

    这里的使用，目前指的是打印出来，有 IO。

1. 没有初始化的变量默认值为 0

swap:

```go
i, j = j, i
```

匿名变量：

```go
tmp, _ = i, j
```

常量：

```go
const a int = 10
const b = 10  // 不能使用 const b := 10

const (
    i int = 10
    j float64 = 3.14
    c = 5
)
```

`iota`是常量自动生成器，每隔一行，自动累加 1，遇到`const`重置为 0。

```go
const (
    a = iota  // 0
    b = iota  // 1
    c = iota  // 2
    d, e, f = iota, iota, iota  // 3
)
```

常见类型：

* `bool`

    长度为 1 字节，取值：`true`，`false`

* `byte`

    长度为 1 字节，`uint8`别名

    ```go
    var ch type
    ch = 'a'
    ```

* `rune`

    长度为 4 字节，专用于存储 unicode 编码，等价于 unit32。

* `int`，`uint`

    长度为 4 或 8 字节

* `int8`，`uint8`，`int16`，`uint16`，`int32`，`uint32`，`int64`，`uint64`

* `float32`，`float64`

    浮点数的字面量默认是 float64 类型。

* `complex64`，`complex128`

    ```go
    var t complex128
    t = 2.1 + 3.14i
    t2 := 3.3 + 4.4i
    real(t)
    imag(t)
    ```

* `uintptr`

    长度为 4 个字节或 8 个字节，主要用于存储指针

* `string`

    `utf-8`字符串

    ```go
    var str1 string
    str1 = "abc"
    str2 := "cba"
    len(str2)  // 字符串长度
    ```

整型与布尔类型不能互相转换

```go
var flag bool
var ch byte
var num int

int(flag)  // error
bool(num)  // error

num = ch  // error
num = int(ch)  // OK
```

类型别名：

```go
type bigint int64
type {
    long int64
    char byte
}

var a bigint
```

## 运算符

`+`, `-`, `*`, `/`, `%`

`++`, `--`：这两个只有后置自增（自减），没有前置

`==`, `!=`, `<`, `>`, `<=`, `>=`

`!`, `&&`, `||`：这里的与和或为短路运算吗？

`=`, `+=`, `-=`, `*=`, `/=`, `%=`

位运算符：

`<<=`, `>>=`, `&=`, `^=`, `|=`

`&`：取地址运算符

`*`：取值运算符

运算符优先级：

7. `^`, `!`
6. `*`, `/`, `%`, `<<`, `>>`, `&`, `&^`
5. `+`, `-`, `|`, `^`
4. `==`, `!=`, `<`, `<=`, `>=`, `>`
3. `<-`
2. `&&`
1. `||`

## 控制语句

```go
s := "abc"
if s == "abc" {
    fmt.Println("hello")   
}

// if 支持一个初始化语句，初始化语句和判断条件以分号分隔
if a: = 10; a == 10 {
    fmt.Println("hello")
}

if xxx {
    xxxx
} else {
    xxxx
}

if xxx {
    ...
} else if {
    ...
} else {
    ...
}

switch score := 1; score {  // 可以初始化条件
    case 1:
        fmt.Printf("hello")
        break // 不需要写，默认有 break
    case 2:
        xxx
        break
    case 3:
        xxx
        fallthrough  // 强制执行后面的 case 代码
    case 4, 5, 6:  // 可以匹配多个值
        xxx
    case score > 9:  // 可以放置条件
        xxx
    default:
        xxx
}

sum := 0
for i := 1; i <= 10; i++ {
    sum += i
}

a := "abc"
for i, ch := range a {
    xxx
}

for i := range a {  // 第二个返回值被丢弃

}

for i, _ := range a {  // 同上

}

for {  // 死循环

}

goto End  // 跳转
End:
    fmt.Println("here")
```

## 函数

```go
func test() (a, b, c int, int) {
    return 1, 2, 3, 4
}

// 也可以
func test() (a, b, c int) {
    a := 1
    b := 2
    c := 3
    return  // 不能省略
}

var c, d, e int
c, d, e = test()

func myfunc(a, b int, c string, d) {
    xxx
}

// 不定参数
func myfunc2(args ...int) {
    len(args)  // 获取参数数量
    args[0]  // 用索引访问元素
}
```

函数定义在调用的前面还是后面无所谓。

函数参数按值传递。数组也是按值传递。

不定参数必须放在形参的最后一个。

```go
func myfunc(temp ...int) {

}

func test(args ...int) {
    myfunc(args...)  // 传递所有参数
    myfunc(args[:2]...)  // 从 args[2] 开始传递所有元素
    myfunc(args[:2]...)  // 0 ~ 2，不包括 2
}
```

函数名首字母小写为`private`，大写即为`public`。

函数也是一种数据类型：

```go
func minus(a, b int) int {
    return a - b
}

type FuncType func(int, int) int

func main() {
    var fTest FuncType
    fTest = minus
    fTest(2, 1)
}


// 回调函数实现多态
type FuncType func(int, int) int
func Calc(a, b int, fTest FuncType) (result int) {
    fmt.Println("Calc")
    result = fTest(a, b)
    return
}
```

闭包：

```go
func main() {
    a := 10
    str := "mike"

    // 匿名函数
    f1 := func() {
        fmt.Println("a = ", a)
        fmt.Println("str = ", str)
    }

    f1()  // 调用匿名函数


    type FuncType func()
    var f2 FuncType
    f2 = f1
    f2()

    // 定义匿名函数，同时调用
    func() {
        fmt.Printf("a = %d, str = %s\n", a, str)
    } ()

    // 带参数的匿名函数
    f3 := func(i, j int) {
        fmt.Println(i, j)
    }
    f3(10, 20)

    // 定义匿名函数，同时调用
    func(i, j int) {
        fmt.Println(i, j)
    } (10, 20)
    
    // 匿名函数，有参有返回值
    x, y := func(i, j int) (max, min int) {
        if i > j {
            max = i
            min = j
        } else {
            max = j
            min = i
        }
        return
    } (10, 20)
}
```

闭包以引用方式捕获外部变量：

```go
func main() {
    a := 10
    str := "mike"

    func() {
        a = 666
        str = "go"  // 此时 main 函数中的变量会被修改
    } ()
}
```

闭包形成了一个独立的空间：

```go
func test() func() int {
    var x int
    return func() int {
        x++
        return x * x
    }
}

func main() {
    f := test()
    f()  // 1
    f()  // 4
}
```

变量的生命周期不由它的作用域决定。

`defer`延迟调用：

```go
func main() {
    defer fmt.Println("bbb")  // 在 main 函数结束前调用
    fmt.Println("aaa")
}
```

多个`defer`函数的执行顺序：先写的后调用。即使某次调用发生异常，其余的也会被依次调用。

```go
func main() {
    a := 10
    b := 20

    defer func() {
        fmt.Printf("a = %d, b = %d\n", a, b)
    } ()

    a = 111
    b = 222
    fmt.Printf("in main function: a = %d, b = %d\n", a, b)
}


func main() {
    a := 10
    a := 20

    defer func(a, b) {
        fmt.Printf("a = %d, b = %d\n", a, b)
    } (a, b)  // 传参会先执行，调用会后执行

    a = 111
    b = 222
    fmt.Printf("in main function: a = %d, b = %d\n", a, b)
}
```

## 作用域

```go
{  // 可以用大括号定义一个新作用域
    i := 10
    fmt.Println("i = ", i)
}

if a := 3; a > 3 {  // 这个 a 也是局部变量

}
```

全局变量：

```go
package main
import "fmt"

var a type  // 全局变量

func main() {
    var a int  // 局部变量覆盖全局变量
}
```

## IO

```go
fmt.Println("a = ", a)  // 这个不能格式化，但会自动换行
fmt.Printf("a = %d\n", a)  // 这个可以格式化
```

输入：

```go
package main
import "fmt"

func main() {
    var a int
    fmt.Printf("输入变量 a: ")
    fmt.Scanf("%d", &a)
    fmt.Scan(&a)
}
```

`%T`表示类型

用户传递的参数：

```go
package main
import "fmt"
import "os"

func main() {
    list := os.Args

    n := len(list)
    fmt.Println("n = ", n)

    for i := 0; i < n; i++ {
        fmt.Printf("list[%d] = %s\n", i, list[i])
    }
}
```

## 指针

```go
var p *int
p = &a

*p  // 解引用
*p = 2
p = nil  // 空指针

p = new(int)  // 不需要手动释放内存

func swap(p1, p2 *int) {
    *p1, *p2 = *p2, *p1
}

func modify(p *[5]int) {  // 数组指针
    (*p)[0] = 123
}

// 切片
a := []int {1, 2, 3}
slice := a[0:3:5]  // [low:high:max]，不包括 high，max - low 为容量
cap(slice)
len(slice)
slice = append(slice, 5)

// make(slice, len, cap) 显式创建切片
s2 := make([]int, 5, 10)

s3 := make([]int, 5)  // 长度和容量相等

s[n]
s[:]
s[low:]
s[:high]
s[low:high]
s[low:high:max]  // 比如 s[:6:8]
len(s)
cap(s)
```

数组与切片区别：

1. 数组`[]`里的长度为固定常量，不能修改长度，len 和 cap 都是固定的
1. 切片`[]`里为空，或者为`...`，切片的长度可以不固定
1. 切片的底层数据是数组

常用方法：

```go
s1 := []int {}
s1 = append(s1, 1)  // 在原切片末尾添加元素，如果超过原来容量，通常以两倍容量扩容
copy(dstSlice, srcSlice)
```

切片作为函数参数时是按引用传递。

```go
func InitData(s []int) {

}
```

## 数组

```go
var id [50]int
id[0]
len(a)

var a [5]int = [5]int{1, 2, 3, 4, 5}
b := [5]int{1, 2, 3, 4, 5}
c := [5]int{1, 2, 3}  // 剩下的初始化为 0
d := [5]int{2: 10, 4: 20}  // 选择性赋值

var a [3][4]int
b := [3][4]int {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}  // 必须在同一行
c := [3][4]int {{1, 2, 3}, {5, 6, 7, 8}, {9, 10, 11, 12}}
d := [3][4]int {1:{1, 2, 3, 4}}
fmt.Println(d)  // 数组可以直接打印

a == b  // 数组可以直接比较
var d [5]int
d = a  // 同类型的数组可以赋值
```

给函数传递参数时，数组是按值传递，而 slice 是按引用传递。

## 随机数

```go
import "math/rand"
import "time"
func main() {
    rand.Seed(123)
    rand.Int()
    rand.Seed(time.Now().UnixNano())
    rand.Intn(100)  // 限制在 100 内
}
```

## struct

需要`type`和`struct`关键字配合使用来实现一个 struct.

```go
type struct_name struct {
  member1 datatype
  member2 datatype
  member3 datatype
  ...
} 
```

example:

```go
package main
import ("fmt")

type Person struct {
  name string
  age int
  job string
  salary int
}

func main() {
  var pers1 Person
  var pers2 Person

  // Pers1 specification
  pers1.name = "Hege"
  pers1.age = 45
  pers1.job = "Teacher"
  pers1.salary = 6000

  // Access and print Pers1 info
  fmt.Println("Name: ", pers1.name)
  fmt.Println("Age: ", pers1.age)
  fmt.Println("Job: ", pers1.job)
  fmt.Println("Salary: ", pers1.salary)
}
```

另外一种初始化 struct 的方式：

```go
res := author{
    name:      "Sona",
    branch:    "CSE",
    particles: 203,
    salary:    34000,
}
```

## map

创建：

```go
package main
import ("fmt")

func main() {
  var a = map[string]string{"brand": "Ford", "model": "Mustang", "year": "1964"}
  b := map[string]int{"Oslo": 1, "Bergen": 2, "Trondheim": 3, "Stavanger": 4}

  fmt.Printf("a\t%v\n", a)
  fmt.Printf("b\t%v\n", b)
}
```

Allowed Key Types

The map key can be of any data type for which the equality operator (==) is defined. These include:

    Booleans
    Numbers
    Strings
    Arrays
    Pointers
    Structs
    Interfaces (as long as the dynamic type supports equality)

Invalid key types are:

    Slices
    Maps
    Functions

These types are invalid because the equality operator (==) is not defined for them.

Allowed Value Types

The map values can be any type.

```go
info := map[int]string {
    110: "mike",
    111: "yoyo"
}

var m1 map[int]string  // map 只有 len，没有 cap
m2 := make(map[int]string)
m3 := make(map[int]string, 10)  // 指定长度，会自动扩容
m3[1] = "mike"
m4 := map[int]string {1: "mike", 2: "go"}

// 遍历
for key, val := range m {
    fmt.Printf("%d, %s\n", key, value)
}

// 判断是否存在
val, ok := m[1]  // val 为 key 对应的值，ok 为 key 值是否存在

// 删除一个值
delete(m, 1)  // 删除 key 为 1 的内容

// map 作为函数参数时按引用传递
func test(m map[int]string) {
    // ...
}
```

## method

如果一个 type 或者一个 struct 定义在了**当前**`package`内，那么就可以给他添加一些 method：

```go
func(reciver_name Type) method_name(parameter_list)(return_type){
    // Code
}
```

这个 method 的调用类似于 c++/java 里的成员函数。

example:

```go
package main
 
import "fmt"
 
// Author structure
type author struct {
    name      string
    branch    string
    particles int
    salary    int
}
 
// Method with a receiver
// of author type
func (a author) show() {
 
    fmt.Println("Author's Name: ", a.name)
    fmt.Println("Branch Name: ", a.branch)
    fmt.Println("Published articles: ", a.particles)
    fmt.Println("Salary: ", a.salary)
}
 
// Main function
func main() {
 
    // Initializing the values
    // of the author structure
    res := author{
        name:      "Sona",
        branch:    "CSE",
        particles: 203,
        salary:    34000,
    }
 
    // Calling the method
    res.show()
}
```

内置类型如果通过`type`定义，那么也可以为其定义 method:

```go
package main
 
import "fmt"
 
// Type definition
type data int
 
// Defining a method with
// non-struct type receiver
func (d1 data) multiply(d2 data) data {
    return d1 * d2
}
 
/*
// if you try to run this code,
// then compiler will throw an error
func(d1 int)multiply(d2 int)int{
return d1 * d2
}
*/
 
// Main function
func main() {
    value1 := data(23)
    value2 := data(20)
    res := value1.multiply(value2)
    fmt.Println("Final result: ", res)
}
```

注意，上面的 method 都是按值传递对象。如果需要按引用传递对象，那么需要传递指针：

```go
func (p *Type) method_name(...Type) Type {
    // Code
}
```

example:

```go

// Go program to illustrate pointer receiver
package main
 
import "fmt"
 
// Author structure
type author struct {
    name      string
    branch    string
    particles int
}
 
// Method with a receiver of author type
func (a *author) show(abranch string) {
    (*a).branch = abranch
}
 
// Main function
func main() {
 
    // Initializing the values
    // of the author structure
    res := author{
        name:   "Sona",
        branch: "CSE",
    }
 
    fmt.Println("Author's name: ", res.name)
    fmt.Println("Branch Name(Before): ", res.branch)
 
    // Creating a pointer
    p := &res
 
    // Calling the show method
    p.show("ECE")
    fmt.Println("Author's name: ", res.name)
    fmt.Println("Branch Name(After): ", res.branch)
}
```

输出：

```
Author's name:  Sona
Branch Name(Before):  CSE
Author's name:  Sona
Branch Name(After):  ECE
```

method 可以在传实参时自动加引用或者解引用，最终的效果以 method 参数列表里为准：

```go

// Go program to illustrate how the
// method can accept pointer and value
 
package main
 
import "fmt"
 
// Author structure
type author struct {
    name   string
    branch string
}
 
// Method with a pointer
// receiver of author type
func (a *author) show_1(abranch string) {
    (*a).branch = abranch
}
 
// Method with a value
// receiver of author type
func (a author) show_2() {
 
    a.name = "Gourav"
    fmt.Println("Author's name(Before) : ", a.name)
}
 
// Main function
func main() {
 
    // Initializing the values
    // of the author structure
    res := author{
        name:   "Sona",
        branch: "CSE",
    }
 
    fmt.Println("Branch Name(Before): ", res.branch)
 
    // Calling the show_1 method
    // (pointer method) with value
    res.show_1("ECE")
    fmt.Println("Branch Name(After): ", res.branch)
 
    // Calling the show_2 method
    // (value method) with a pointer
    (&res).show_2()
    fmt.Println("Author's name(After): ", res.name)
}
```

## interface

可以使用`interface`指定一些方法，类似 c++ 的纯虚函数：

```go
package main

import "fmt"

type Print interface {
	print()
}

type MyStr struct {
	str string
}

func (my_str *MyStr) print() { // 对 interface 的实现
	fmt.Println(my_str.str)
}

func show_msg(msg Print) { // 只有实现了 Print 接口的对象，才能调用该函数
	msg.print()
}

func main() {
	msg := MyStr{
		str: "hello",
	}
	show_msg(&msg)
}
```

## Miscellaneous

1. 一个 bug 解决方案：<https://blog.csdn.net/a1056139525/article/details/122270531>
