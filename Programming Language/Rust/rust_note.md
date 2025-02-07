# Rust Note

## cache

* rust 中启动一个子进程：

    ```rust
    std::process::Command::new("cat")
        .arg("test.txt")
        .spawn().expect("fail to start the process");
    ```

* rust 中的`drop`

    假如现在有 3 个变量`a`, `b`, `c`，其中`b`使用了`a`的引用，但是后面`c`也想用`a`，那么可以使用`drop()`把`b` drop 掉，返回`a`的所有权：

    ```rs
    fn test() {
        let a = A::new();
        let b = B::new();
        let c = C::new();
        b.use(&mut a);
        drop(b);
        c.get_ownership(a);
    }
    ```

* rust err 中最好返回`String`

* rust 中`&mut <type>`，需要使用`*`解引用后才能得到值

    这一点和 c++ 不太一样，c++ 中的引用只是一个别名，不需要`*`解引用。

* rust: can't borrow from a `Weak`

    ref: <https://stackoverflow.com/questions/34821389/borrowing-from-weakt>

* 无论是 c++ 还是 rust，在标准库中都不能直接 terminate 一个线程。

    要么从让线程自己退出，要么让主进程退出。

    在 c++ 中强制`delete`一个`thread`对象，会导致主程序立即崩溃。

* `let line = line.strip_suffix('\n').unwrap_or(line);`

    rust 中可以这么用`unwrap_or()`。

    `unwrap_of_default()`返回的是这个类型的默认值，比如 0 或空字符串等。

* 可以用`cargo add <package_name>`在`Cargo.toml`里添加一个新的 dependency

* rust 的`std::time`基本只能拿来计时用，不能用于处理日期方面的东西

    如果要处理日期相关，必须用`chrono`库。

* rust 中`into_xxx`通常是将自身所有权转换成另一个对象的类型，`xxx()`则通常只是 borrow。

* cargo config aliyun mirror

    <https://developer.aliyun.com/mirror/crates.io-index/>

## note

系统编程指的是编写：

* 操作系统
* 各种设备驱动
* 文件系统
* 数据库
* 运行在廉价设备或必须极端可靠设备上的代码
* 加解密程序
* 媒体编解码器（读写音频、视频和图片文件的软件）
* 媒体处理器（如语音识别或图片编辑软件）
* 内存管理程序（如实现垃圾收集器）
* 文本渲染程序（将文本和字体转换为像素）
* 高级编程语言（如 JavaScript 或 Python）
* 网络程序
* 虚拟化及软件容器
* 科学模拟程序
* 游戏

简言之，系统编程是一种资源受限的编程。这种编程需要对每个字节和每个 CPU 时钟周期精打细算。

使用 rust 的项目：Servo

C99 中对*未定义行为*的定义：

> **未定义行为**
>
> 由于使用不可移植或错误的程序构造，或者使用错误的数据导致的行为，本国际标准对此不作要求。

数组下标越界就是一个未定义行为。未定义操作并非只产生意想不到的结果，事实上这种情况下程序无论做任何事情都是被允许的。

为了生成更快的代码，C99 授予编译器全权。这个标准没有让编译器负责检测和处理可疑的行为（比如数组越界），而是让程序员负责保证这种情况永远不会发生。

如果将一个程序写得不可能在执行时导致未定义行为，那么就称这个程序为**定义良好的**（well defined）。如果一种语言的安全检查可以保证所有程序都定义良好，那么就称这种语言是**类型安全的**。C 和 C++ 不是类型安全的，Python 是类型安全的。

## Fast startup

```rust
println!("hello, world!");

let a: i32 = 3;
let b: &str = "hello";
let c: char = 'c';
let d: bool = true;
let mut e: [i32; 3] = [3, 2, 1];
for elm in &e {
    print!("{}, ", elm);
}
```

## Installation

rust 的一些库依赖于 C 编译器。

网站：<https://rustup.rs>

* linux

    `curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh`

    除此之外，还需要安装 C 语言支持：`apt install build-essential`

* windows

    这个网站的安装包似乎会把 rust 直接装到 c 盘。

    除了安装 rust 相关的包，还需要安装 visual studio。（或许 msvc build tools 也可以用，但是我没试过）

有关 rustup：

* `rustup update`：更新 rust 到最新版本
* `rustup self uninstall`：卸载 rust 和 rustup
* `rustup doc`：在浏览器中查看本地文档

工具：

* `cargo`：编译管理器，包管理器，通用工具。
* `rustc`：rust 的编译器。
* `rustdoc`：rust 文档工具。
* `rustfmt`：自动格式化工具

## hello, world 程序

创建一个文件夹`hello`，进入文件夹，创建一个新文件`main.rs`，写入

```rust
fn main() {
    println!("hello, world!");
}
```

编译：`rustc main.rs`，生成二进制文件

运行：`./main`

## 项目管理

创建一个项目：`cargo new <project_name>`

此时会自动初始化一个`git`仓库，即在`<project_name>`目录下创建`.git`和`.gitignore`。

Example:

`cargo new hello`

（也可以在一个已经有 git 的目录下使用`cargo new`对这个目录进行 cargo 的初始化。可以使用`cargo new --vcs=<vcs_name>`使用其他版本的 vcs。)

（`cargo new --bin hello`。`--bin`表示这个项目是一个可执行文件，而不是一个库。）

`Cargo.toml`保存这个项目的元数据，里面的内容差不多是这样：

```toml
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2021"

[dependencies]
```

其中每个方括号`[]`表示一个 section 的标题。

在 rust 中，代码包被称为`crate`。

cargo 还创建了个`.git`和`.gitignore`，如果不需要这么做，可以加上`--cvs none`。

常用的 cargo 命令：

* `cargo clean`

    清理生成的文件。

* `cargo build`

    构建项目，可执行文件被放在`target/debug`目录下。首次运行`cargo build`时，会在项目目录下生成一个`Cargo.lock`文件，它用于记录项目依赖的实际版本。这个文件由 cargo 负责管理，我们不需要动这个文件。

    Parameters:

    * `--release`
    
        `cargo build --release`可以生成 release 版本的程序。

* `cargo check`可以检查项目是否可以通过编译。

* `cargo run`

    构建并运行整个项目。在这个包的任何目录都可以调用`cargo run`。

* `rustup doc --std`

    在本地浏览器中查看标准库文档。

## 变量与常量

**变量**

cargo 中变量默认都是不可变（immutable）的，运行下面的代码会报错``cannot assign twice to immutable variable `x` ``。

```rust
fn main() {
    let x = 5;
    println!("The value of x is: {x}");
    x = 6;
    println!("The value of x is: {x}");
}
```

可以在变量名称前加`mut`将这个变量作为可变的：`let mut x = 5;`

**shadow**

rust 允许用一个新值来隐藏（shadow）之前的值，这个功能常用在需要转换值类型的场景。

```rust
fn main() {
    let x = 5;
    let x = x + 1;  // 创建了一个新变量，并隐藏了前面的 x
    {
        let x = x + 2;
        println!("The value of x in the inner scope is: {x}");
        println!("The value of x is: {x}");
    }
}
```

输出：

```
The value of x in the inner scope is: 12
The value of x is: 6
```

隐藏与赋值的对比：

```rust
let spaces = "    ";
let spaces = spaces.len();  // OK

let mut spaces = "    ";
spaces = spaces.len();  // error，改变类型是不允许的
```

**常量**

常量需要使用`const`声明：

```rust
const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
```

定义常量时，必须指定类型。（为什么？）

## 常用基本类型

rust 是静态类型的语言，即无须实际运行程序，编译器就可以检查所有可能的执行路径，确保程序以与类型一致的方式使用每一个值。

**整数类型**

| 类型 | 范围 |
| - | - |
| `u8` | $0$ ~ $2^8 - 1$ |
| `u16` | $0$ ~ $2^{16} - 1$ |
| `u32` | $0$ ~ $2^{32} - 1$ |
| `u64` | $0$ ~ $2^{64} - 1$ |
| `usize` | $0$ ~ $2^{32} - 1$ 或 $2^{64} - 1$
| `i8` | $-2^7$ ~ $2^7 - 1$ |
| `i16` | $-2^{15}$ ~ $2^{15} - 1$ |
| `i32` | $-2^{31}$ ~ $2^{31} - 1$ |
| `i64` | $-2^{63}$ ~ $2^{63} - 1$ |
| `isize` | $-2^{31}$ ~ $2^{31} - 1$ 或 $-2^{63}$ ~ $2^{63} - 1$ |

其中有符号整数使用补码表示。用`u8`表示字节值。`usize`和`isize`可看作是 C++ 中的`size_t`和`ptrdiff_t`。在 32 位机器上是 32 位长，在 64 位机器上是 64 位长。

rust 要求数组索引必须是`usize`值。

rust 会检查算术操作中是否有整数溢出：

```rust
let big_val = std::i32::MAX;
let x = big_val + 1;  // panic: 算法操作溢出
```

可以使用特定的方法来指定结果翻转为负值：

```rust
let x = big_val.wrapping_add(1);  // OK
```

在 debug 模式下，溢出会引发 panic，而在 release 模式下，会执行一种叫做二进制补码包装（two's complement wrapping）的操作，即回绕。

整数字面量可以通过一个后缀表示类型，比如`42u8`，`1729isize`。如果没有后缀，那么 rust 会根据上下文推断，如果有多种可能性，会优先选择`i32`。

整数字面量可以使用前缀`0x`，`0o`和`0b`分别表示十六进制、八进制和二进制。数字里还可以插入下划线，方便阅读。

| 字面量 | 类型 | 十进制值 |
| - | - | - |
| `116i8` | `i8` | `116` |
| `0xcafeu32` | `u32` | `51966` |
| `0b0010_1010` | 推断 | `42` |
| `0o106` | 推断 | `70` |

rust 还提供了字节字面量（byte literal），它只是`u8`类型的另一种写法：`b'A'`等于`65u8`。有一些转义字符需要特殊处理：

| 字符 | 字符字面量 | 对等的数值 |
| - | - | - |
| 单引号（`'`） | `b'\''` | `39u8` |
| 反斜杠（`\`） | `b'\\'` | `92u8` |
| 换行 | `b'\n'` | `10u8` |
| 回车 | `b'\r'` | `13u8` |
| 制表符 | `b'\t'` | `9u8` |

对于无法打印的字符，可以用十六进制写出它们的编码，比如`b'\x1b'`。

可以使用`as`实现整数之间的转换：

```rust
assert_eq!(10_i8 as u16, 10_u16);  // ok
assert_eq!(2525_u16 as i16, 2525_i16);  // ok
assert_eq!(-1_i16 as i32, -1_i32);  // 以符号填充（没看懂）
assert_eq!(65535_u16 as i32, 65535_i32);  // 以零填充（没看懂）

assert_eq!(1000_i16 as u8, 232_u8);  // 1000 % (2^8) = 232
assert_eq!(65535_u32 as i16, -1_i16);  // 不懂，猜一下，可能是 65535 转换成无符号二进制后为 1111 1111 1111 1111，将它作为补码再转换成十进制，得到 -1

assert_eq!(-1_i8 as u8, 255_u8);
assert_eq!(255_u8 as i8, -1_i8);
```

整数的方法：

* `pow()`

    ```rust
    assert_eq!(2u16.pow(4), 16);
    ```

* `abs()`

    ```rust
    assert_eq!((-4i32).abs(), 4);
    ```

* `count_ones()`

    ```rust
    assert_eq!(0b101101u8.count_ones(), 4);
    ```

* `clone()`

    把自己复制一份，然后返回。

* `clone_from()`

    ```rust
    core::clone::Clone
    pub fn clone_from(&mut self, source: &Self)
    where
        Self: Destruct,
    ```

    `a.clone_from(&b)`等价于`a = b.clone()`。不过第一种方法可以复用`a`已经申请的内存空间。

* `clone_into()`

    ```rust
    alloc::borrow
    fn clone_into(&self, target: &mut T)
    ```

    把当前值（通常是个临时变量） clone 一份，赋值给`target`。

    Examples:

    ```rust
    let mut s: String = String::new();
    "hello".clone_into(&mut s);

    let mut v: Vec<i32> = Vec::new();
    [1, 2][..].clone_into(&mut v);
    ```

* `from()`

    ```rust
    core::convert::From
    pub fn from(value: T) -> Self
    ```

    Converts to this type from the input type.

    其实就是对有 ownership 的值做了个类型转换，转换成当前类型的值。

* `into()`

    ```rust
    core::convert::Into
    pub fn into(self) -> T
    ```

    Converts this type into the (usually inferred) input type.

    我觉得这个相当于 c 语言的隐式类型转换了。

* `cmp()`

    ```rust
    core::cmp::impls
    fn cmp(&self, other: &i32) -> Ordering
    ```

    Example

    ```rust
    use std::cmp::Ordering;

    assert_eq!(5.cmp(&10), Ordering::Less);
    assert_eq!(10.cmp(&5), Ordering::Greater);
    assert_eq!(5.cmp(&5), Ordering::Equal);
    ```

* `partial_cmp()`

    ```rust
    core::cmp::impls
    fn partial_cmp(&self, other: &i32) -> Option<Ordering>
    ```

    即使有一个值不存在也能比较。

    Examples:

    ```rust
    use std::cmp::Ordering;

    let result = 1.0.partial_cmp(&2.0);
    assert_eq!(result, Some(Ordering::Less));

    let result = 1.0.partial_cmp(&1.0);
    assert_eq!(result, Some(Ordering::Equal));

    let result = 2.0.partial_cmp(&1.0);
    assert_eq!(result, Some(Ordering::Greater));
    When comparison is impossible:

    let result = f64::NAN.partial_cmp(&1.0);
    assert_eq!(result, None);
    ```

如果根据上下文能够推断出类型，那么就不需要指定类型。但是上面几个例子无法确定出类型，所以需要后缀来确定类型。

**浮点类型**

| 类型 | 精度 | 范围 |
| - | - | - |
| `f32` | IEEE 单精度（至少 6 位小数） | 约 $-3.4 \times 10^{38}$ 到 $3.4 \times 16^{38}$ |
| `f64` | IEEE 双精度（至少 15 位小数） | 约 $-1.8 \times 10^{308}$ 到 $1.8 \times 10^{308}$ |

对于浮点字面量的通用形式：`31415.926e-4f64`，其中`31415`是整数部分，`.926`是小数部分，`e-4`是指数部分，`f64`是类型后缀。浮点数值中除了整数部分，其他部分都是可选的，但小数部分、指数和类型后缀这三者中至少要有一个存在，这样才能将它跟整数字面量区分开。小数部分也可以只有一个小数点。

如果浮点字面量中没有类型后缀，那么 rust 会根据上下文推断它是`f32`还是`f64`，如果两种都有可能，则默认为`f64`。rust 不会将浮点类型推断为整数类型，反之亦然。

| 字面量 | 类型 | 数学值 |
| - | - | - |
| `-1.5625` | 推断 | -1.5625 |
| `2.` | 推断 | 2 |
| `0.25` | 推断 | 0.25 |
| `1e4` | 推断 | 10000 |
| `40f32` | `f32` | 40 |
| `9.109_383_56e-31f64` | `f64` | 约 $9.10938356 \times 10^{-31}$ |

`std::f32`和`std::f64`中定义有特殊常量：`INFINITY`, `NEG_INFINITY`, `NAN`, `MIN`, `MAX`。`std::f32:consts`和`std::f64::consts`提供了各种常用的数学常量，比如`E`, `PI`。

`f32`和`f64`也提供完整的数学计算方法，比如`2f64.sqrt()`。

```rust
assert_eq!(5f32.sqrt() * 5f32.sqrt(), 5.);
assert_eq!(-1.01f64.floor(), -1.0);
assert!((-1. / std::f32::INFINITY).is_sign_negative());

println!("{}", (2.0).sqrt());  // error
println!("{}", (2.0_f64).sqrt());  // ok
println!("{}", f64::sqrt(2.0));  // ok
```

rust 几乎不做隐式类型转换。如果它推断不出来到底是`f32`还是`f64`，就直接放弃了。

**布尔类型**

`bool`类型有两个值：`true`和`false`。rust 中不允许除了`bool`类型以外的其它类型值作为`if`，`while`等语句的条件。

可以使用`as`操作符把`bool`值转换为整数类型：

```rust
assert_eq!(false as i32, 0);
assert_eq!(true as i32, 1);
```

但是`as`不能将整数转换成布尔值。

**字符类型**

`char`以 32 位值的形式表示单个 Unicode 字符。但是对字符串或文本流使用 utf-8 编码，所以`String`类型中的数据是 utf-8 字节的序列，而不是字符的数组。

字符字面量是以单引号括起来的 Unicode 字符，其中一些特殊字符也需要用反斜杠转义：`\'`, `\\`, `\n`, `\r`, `\t`。

如果愿意，可以在`char`类型里写出字符的十六进制 Unicode 码点：

* 如果字符码点范围在`U+0000`到`U+007F`之间（即 ASCII 字符集），可以将该字符写成`'\xHH'`形式，其中`HH`是 2 位十六进制数字。比如`*`和`\x2A`是相等的。

* 任何 Unicode 字符都可以写作`\u{HHHHHH}`，其中`HHHHHH`是 1 到 6 位十六进制数字。比如`'\u{CA0}'`表示坎纳达语中的某个字符。

`char`类型保存的 Unicode 码点范围只能在`0x0000`到`0xD7FF`之间或`0xE000`到`0x10FFFF`之间。

如果需要，可以使用`as`把`char`转换为整数类型，但如果目的类型小于 32 位，字符值的高位会被截断：

```rust
assert_eq!('*' as i32, 42);
assert_eq!('ಠ' as u16, 0xca0);
assert_eq!('ಠ' as i8, -0x60);
```

`u8`是唯一可以转换为`char`的整数类型。另外`std::char::from_u32`可以将`u32`值转换为`Option<char>`值：如果该`u32`是不被许可的 Unicode 码点，就返回`None`；否则返回`Some(c)`，其中`c`是转换后的`char`值。

标准库中还提供了一些与`char`类型相关的方法：

```rust
assert_eq!('*'.is_alphabetic(), false);
assert_eq!('β'.is_alphabetic(), true);
assert_eq!('8'.to_digit(10), Some(8));
assert_eq!('ಠ'.len_utf8(), 3);
assert_eq!(std::char::from_digit(2, 10), Some('2'));
```

**类型推断**

rust 支持类型推断，因此下面两种写法是等价的：

```rust
fn build_vector() -> Vec<i16> {
    let mut v: Vec<i16> = Vec::<i16>::new();
    v.push(10i16);
    v.push(20i16);
    v
}
```

```rust
fn build_vector() -> Vec<i16> {
    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    v
}
```

rust 不会把`bool`类型与其他数值类型之间自动相互转换。这点与 C 不同。因此数值不能作为`if`等语句的条件表达式。

## 元组（tuple），数组（array）

* tuple

    元组是用圆括号括起来的几个值，比如`("Brazil", 1985)`，它的类型是`(&str, i32)`。可以用`t.0`，`t.1`访问元组`t`的元素。

    也可以显式地指定元组中元素的类型：

    ```rust
    fn main() {
        let tup: (i32, f64, u8) = (500, 6.4, 1);

        let (x, y, z) = tup;  // 解构 destructure
    }
    ```

    元组中元素的类型可以不同，而数组则要求相同类型。元组只允许用常量作为索引，比如`t.4`，而不能是变量，比如`t.i`或`t[i]`。

    ```rust
    fn main() {
        let x: (i32, f64, u8) = (500, 6.4, 1);
        let elm_1 = x.0;
        let elm_2 = x.1;
        let elm_3 = x.2;
    }
    ```

    函数如果想返回多个值时可以返回一个元组：

    ```rust
    fn split_at(&self, mid: usize) -> (&str, &str);
    ```

    利用函数的返回值：

    ```rust
    let text = "I see the eigenvalue in thine eye";
    let (head, tail) = text.split_at(21);
    assert_eq!(head, "I see the eigenvalue ");
    assert_eq!(tail, "in thine eye");
    ```

    将元组作为函数参数：

    ```rust
    fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<(), std::io::Error>
    {
        ...
    }
    ```

    零元组`()`又称为基元类型（unit type），又翻译为单元元组。没有返回值的函数的返回类型就是`()`。

* array

    数组要求其中的元素的类型必须相同。

    ```rust
    fn main() {
        let a = [1, 2, 3, 4, 5];
        let a: [i32; 5] = [1, 2, 3, 4, 5];  // 指定类型和数量
        let a = [3; 5];  // 数组初始化为相同值

        // 用下标访问数组元素
        let first = a[0];
        let second = a[1];
    }
    ```

    rust 会检测数组中的下标是否越界，如果越界，则会 panic。

    * 遍历 traverse

        可以用`for`对数组中的元素进行遍历：

        ```rust
        fn main() {
            let arr = [5, 4, 3, 2, 1];
            for x in arr {
                print!("{}, ", arr);
            }
        }
        ```

        如果想得到元素的引用，可以这样写：

        ```rust
        fn main() {
            let arr = [5, 4, 3, 2, 1];
            for x in &arr {
                print!("{}, ", x);
            }
        }

        在前面的例子中，`x`的类型是`i32`，而这个例子中`x`的类型是`&i32`。


## 引用（reference）

```rust
let a = &1;
```

注：

1. 在这个例子中，`1`是在栈上分配内存，还是在堆上分配内存，还是在静态存储区分配内存？`a`是一个引用，还是一个指针？

    我觉得应该是在栈上分配内存。当引用的生命周期结束后，`1`所占的内存自然被释放。

如果不想在函数返回时释放参数的内存，那么就必须将其作为返回值返回：

```rust
fn main() {
    let s1 = String::from("hello");
    let (s2, len) = calculate_length(s1);
    println!("The length of '{}' is {}.", s2, len);
}

fn calcualte_length(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)
}
```

如果不想这么麻烦，可以使用引用：

```rust
fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);
    println!("The length of '{}' is {}.", s1, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

引用允许你使用值但不获取其所有权。引用实际上是指向变量的指针。我们并不能拥有引用，所以当引用停止使用时，它所指向的值也不会被丢弃。

创建一个引用的行为称为借用（borrowing）。

我们无法通过引用改变堆中内容的值：

```rust
fn main() {
    let s = String::from("hello");
    change(&s);
}

fn change(some_string: &String) {
    some_string.push_str(", world!");
}
```

如果想要修改，必须使用可变引用：

```rust
fn main() {
    let mut s = String::from("hello");
    change(&mut s);
}

fn change(some_thing: &mut String) {
    some_string.push_str(", world!");
}
```

在同一时间只能有一个对某一特定数据的可变引用，对同一个变量创建两个可变引用会报错：

```rust
let mut s = String::from("hello");
let r1 = &mut s;
let r2 = &mut s;
println!("{}, {}", r1, r2);
```

这个设计有助于防止数据竞争（data race）：

1. 两个或更多指针同时访问同一数据
1. 至少有一个指针被用来写入数据
1. 没有同步数据访问的机制

可以在不同作用域中拥有多个可变引用：

```rust
let mut s = String::from("hello");

{
    let r1 = &mut s;
}

let r2 = &mut s;
```

一个变量的可变引用与不可变引用不可以同时出现在同一个作用域中：

```rust
let mut s = String::from("hello");

let r1 = &s;  // OK
let r2 = &s;  // OK
let r3 = &mut s;  // Error

println!("{}, {}, and {}", r1, r2, r3);
```

`r2`的作用域一直到`println!`这一行，所以`r3`的作用域在`r2`的作用域范围内，所以会报错。

**引用的作用域**

一个引用的作用域为从声明的地方开始一直持续到最后一次使用为止。引用的作用域结束后，可以创建新引用：

```rust
let mut s = String::from("hello");

let r1 = &s;  // OK
let r2 = &s;  // OK
println!("{} and {}", r1, r2);  // r1 和 r2 的使用域结束

let r3 = &mut s;  // OK
println!("{}", r3);
```

但是这样写的话就会报错：

```rust
fn main() {
    let mut s = String::from("hello");
    let r1 = &s;
    let r2 = &mut s;
    println!("{}", r1);  // 这里引用 r1 的作用域大于 r2 的作用域，而又因为在 r1 作用域内，只能创建不可变引用，所以这里会报错。
}
```

编译器在作用域结束之前判断不再使用的引用的能力被称为非词法使用域生命周期（Non-Lexical Lifttimes，NLL）。

**引用不允许作为函数的返回值**

引用不允许被函数返回，因为它可能指向一个被释放的变量：

```rust
fn main() {
    let reference_to_nothing = dangle();
}

fn dangle() -> &String {
    let s = String::from("hello");
    &s
}
```

通常我们会返回的变量的值本身，而不是引用。

一些推论：

1. 在任意给定时间，要么只能有一个可变引用，要么只能有多个不可变引用
1. 引用必须总是有效的

**引用与 mut**

```rust
let mut r = &list_node;  // 这个 mut 指的是 r 可以被重新赋值，指向其他的引用
r = &list_node_2;  // OK

let r = &list_node;
r = &list_node_2;  // Error
```

```rust
let r = &mut list_node;  // 这个 mut 指的是 r 可以修改 list_node 中的内容
r.val = 3;  // OK

let r = &list_node;
r.val = 3;  // Error
```



## 字符串

### &str

rust 中常用的字符串有两种，一种为`str`，一种为`String`。

* `str`

    rust 中字面量的字符串为`str`类型，由于它被写进程序的静态存储区，所以我们拿不到它的所有权，只能拿到它的引用：

    ```rust
    let mystr: &str = "hello, world";
    ```

    Indexing：

    因为`str`字符串和`String`字符串存储的是 UTF-8 编码，而 UTF-8 编码是变长编码，无法达到常数时间复杂度的索引，因此它们都不能直接索引。在 UTF-8 编码中，ASCII 字符占用 1 个字节，汉字占用 3 个字节，有一些表情符号占用为 4 个字节。因此处理起来很麻烦。

    ```rust
    let s = "hello, world";
    s.as_bytes().nth(1)  // Some('e'). as_bytes() 会返回 &[u8]，将 utf-8 字符串拆解成单个字节
    s.chars().nth(3)  // Some('l').  chars() 会根据 s 生成一个 Chars 类型的新对象，因此 s 仍有所有权。
    ```

    Slicing:

    ```rust
    let s = "hello, 你好";
    &s[0..2]  // &str, "he"
    &s[7..10]  // &str, "你", 注意一个汉字占 3 个字节
    &s[8..10]  // error，因为 utf-8 解析错误，报错会说 byte index 8 is not a char boundary
    ```

    Properties:

    ```rust
    let s = "hello, 你好";
    s.len()  // 13,  返回字符串的字节数
    ```

    Methods:

    ```rust
    is_char_boundary()  // 检查一个整数是否为 char 的 boundary
    as_ptr()
    as_mut_ptr()
    len()
    capacity()
    ```

### String

如果想对字符串进行修改，我们需要使用标准库中提供的`String`类型，它会在堆上申请内存，存储字符串内容。

由于标准库中`String`会自动被导入，所以在程序中可以直接拿来使用。

创建一个字符串：

```rust
// create a string from &str
let hello = String::from("hello, world!");  // 不可变
let mut hello = String::from("Hello, ");  // 可变

// craete a string from utf8 vec
let sparkle_heart = vec![240, 159, 146, 150];
let sparkle_heart = String::from_utf8(sparkle_heart).unwrap();
assert_eq!("💖", sparkle_heart);

let a = vec![65, 66, 97, 98];
let mystr = String::from_utf8(a).unwrap();  // ABab
```

修改字符串：

```rust
let mut s = String::from("hello, ");
s.push('w');
s.push_str("orld!");
```

`str`中的 indexing 和 slicing 规则同样适用于`String`。

修改字符串中的数据：

```rust
let mut mystr = String::from("你好\nhello\r\nworld");
unsafe {
    let mut v = mystr.as_mut_vec();  // 返回一个 &mut Vec<u8> 引用
    v[7] = b'w';
}
println!("{}", mystr);  // 你好\nwello\r\nworld
```

常用的方法：

* `chars()`

    Syntax:

    ```rust
    pub fn chars(&self) -> Chars<'_>
    ```

    Returns an iterator over the chars of a string slice.

    Example:

    ```rust
    let mut mystr = String::from("你好hello");
    let mut chars = mystr.chars();  // 必须写成 mut，才能调用 next()
    println!("{}", chars.nth(1).unwrap());  // 好
    println!("{}", chars.next().unwrap());  // h
    println!("{}", chars.count());  // 4, count() will consumes the iterator
    println!("{}", chars.next().unwrap());  // Error
    ```

    如果`chars.next()`读到字符串的末尾，那么会返回`None`。

* `char_indices()`

    Syntax:

    ```rust
    pub fn char_indices(&self) -> CharIndices<'_>
    ```

    Returns an iterator over the chars of a string slice, and their positions.

    Examples:

    ```rust
    let mut mystr = String::from("你好hello");
    let mut ch_idx = mystr.char_indices();
    println!("{:?}", ch_idx.nth(1).unwrap());  // (3, '好')
    println!("{:?}", ch_idx.next().unwrap());  // (6, 'h')
    println!("{:?}", ch_idx.count());  // 4
    println!("{}", ch_idx.next().unwrap());  // Error
    ```

    这个方法的功能和`chars()`差不多，只不过`next()`返回的是一个`Some((idx, ch))`。（`Some`内部是一个 tuple）

* `bytes()`

    Syntax:

    ```rust
    pub fn bytes(&self) -> Bytes<'_>
    ```

    An iterator over the bytes of a string slice.

    Examples:

    ```rust
    let mut mystr = String::from("你好hello");
    let mut bytes = mystr.bytes();
    println!("{:?}", bytes.nth(3).unwrap());  // 229
    println!("{:?}", bytes.next().unwrap());  // 165
    println!("{:?}", bytes.next().unwrap());  // 189
    println!("{:?}", bytes.next().unwrap());  // 104, h
    println!("{:?}", bytes.next().unwrap());  // 101, e
    println!("{:?}", bytes.count());  // 3
    println!("{:?}", String::from_utf8(vec![229, 165, 189]));  // Ok("好")
    ```

* `split_whitespace()`

    Syntax:

    ```rust
    pub fn split_whitespace(&self) -> SplitWhitespace<'_>
    ```

    Splits a string slice by whitespace.

    The iterator returned will return string slices that are sub-slices of the original string slice, separated by any amount of whitespace.

    Examples:

    ```rust

    ```

* `split_ascii_whitespace()`

    Syntax:

    ```rust
    pub fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace<'_>
    ```

    Splits a string slice by ASCII whitespace.

    The iterator returned will return string slices that are sub-slices of the original string slice, separated by any amount of ASCII whitespace.

    以空格` `，制表符`\t`，换行符`\n`等分隔一个字符串。返回一个迭代器。

* `lines()`

    Syntax:

    ```rust
    pub fn lines(&self) -> Lines<'_>
    ```

    An iterator over the lines of a string, as string slices.

    以`\n`或`\r\n`分隔行，每次返回一行内容，不包含末尾的`'\n`，`\r\n`。

    Examples:

    ```rust
    let mut mystr = String::from("你好\nhello\r\nworld");
    let mut lines = mystr.lines();
    println!("{:?}", lines.next());
    println!("{:?}", lines.next());
    println!("{:?}", lines.next());
    println!("{:?}", lines.next());
    ```

    输出：

    ```
    Some("你好")
    Some("hello")
    Some("world")
    None
    ```

* `encode_utf16()`

    Syntax:

    ```rust
    pub fn encode_utf16(&self) -> EncodeUtf16<'_>
    ```

    Returns an iterator of u16 over the string encoded as UTF-16.

    将 utf-8 编码的字符串一个一个地转换成 utf-16 编码的字符。

    Example:

    ```rust
    let mut mystr = String::from("你好\nhello\r\nworld");
    let mut utf16 = mystr.encode_utf16();
    println!("{:?}", utf16.next());
    println!("{:?}", utf16.next());
    println!("{:?}", utf16.next());
    println!("{:?}", utf16.next());
    ```

    输出：

    ```
    Some(20320)
    Some(22909)
    Some(10)
    Some(104)
    ```

    感觉这个函数可以用于把 utf-8 编码的内容转换成 utf-16 编码，比如文本文件的编码格式转换。

* `contains()`

    Syntax:

    ```rust
    pub fn contains<'a, P>(&'a self, pat: P) -> bool
    where
        P: Pattern<'a>,
    ```

    Returns true if the given pattern matches a sub-slice of this string slice.

    Returns false if it does not.

    The pattern can be a `&str`, char, a slice of chars, or a function or closure that determines if a character matches.

    Example:

    ```rust
    let mut mystr = String::from("你好\nhello\r\nworld");
    println!("{:?}", mystr.contains("hello"));
    println!("{:?}", mystr.contains('好'));
    println!("{:?}", mystr.contains(|c| c == 'w'));
    ```

    输出：

    ```
    true
    true
    true
    ```

* `starts_with()`

    Syntax:

    ```rust
    pub fn starts_with<'a, P>(&'a self, pat: P) -> bool
    where
        P: Pattern<'a>,
    ```

    Returns true if the given pattern matches a prefix of this string slice.

* `ends_with()`

    Syntax:

    ```rust
    pub fn ends_with<'a, P>(&'a self, pat: P) -> bool
    where
        P: Pattern<'a>,
        <P as Pattern<'a>>::Searcher: ReverseSearcher<'a>,
    ```

    Returns true if the given pattern matches a suffix of this string slice.

* `find()`

    Syntax:

    ```rust
    pub fn find<'a, P>(&'a self, pat: P) -> Option<usize>
    where
        P: Pattern<'a>,
    ```

    Returns the byte index of the first character of this string slice that matches the pattern.

    Returns None if the pattern doesn’t match.

    这个函数好像只能找到`pat`第一次出现的地方。如果要找到字符串中所有的`pat`，可以考虑`match()`相关的函数。

* `replace()`

    替换字符串，返回一个新的字符串。

rust 核心语言中只有`str`和`slice`，前者通常以`&str`的形式出现，而后者为一个引用。

我们比较常用的`String`定义在标准库中：

```rust
let mut s = String::new();

let data = "initial contents";
let s = data.to_string();
let s = "initial contents".to_string();
let s = String::from("initial contents");
```

修改字符串：

```rust
let mut s = String::from("foo");
s.push_str("bar");
```

`push_str()`接受的参数为`slice`，因此并不会获得字符串的所有权：

```rust
let mut s1 = String::from("foo");
let s2 = "bar";
s1.push_str(s2);
println!("s2 is {}", s2);
```

添加一个字符：

```rust
let mut s = String::from("lo");
s.push('l');
```

使用`+`拼接两个字符串会使第一个字符串失效：

```rust
let s1 = String::from("Hello, ");
let s2 = String::from("world!");
let s3 = s1 + &s2;  // s1 can't be used again
```

`+`运算符调用的是

```rust
fn add(self, s: &str) -> String {
```

`self`没有使用`&self`，因此会发生 move。

`+`可以拼接多个字符串：

```rust
let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = String::from("toe");

let s = s1 + "-" + &s2 + "-" + &s3;
// 等价于
let s = format!("{}-{}-{}", s1, s2, s3);
```

rust 中字符串`String`不支持索引：

```rust
let s1 = String::from("hello");
let h = s1[0];  // error
```

但是可以支持 slice:

```rust
let hello = "你好世界";
let s = &hello[0..4];  // 前 4 个字节
let s = &hello[0..1];  // error
```

通常访问字符的方式是将字符串拆分成多个字符（char）：

```rust
for c in "你好".chars() {
    println!("{}", c);
}
```

也可以使用`bytes()`方法返回字节：

```rust
for b in "你好".bytes() {
    println!("{}", b);
}
```



实际上，用花括号括起来的任何代码块都可以看作一个表达式：

```rust
{
    println!("evaluating cos x");
    x.cos()
}
```

测试：

```rust
#[test]
fn test_gcd() {
    assert_eq!(gcd(14, 15), 1);
    assert_eq!(gcd(2 * 3 * 5 * 11 * 17, 
                3 * 7 * 11 * 13 * 19),
                3 * 11);
}
```

`test_gcd`在常规编译时会被跳过，但在使用`cargo test`命令运行程序时会包含并自动调用。

## slice

slice 是数组或字符串中一部分值的引用。

```rust
let s = String::from("hello world");
let hello = &s[0..5];
let world = &s[6..11];
```

一个字符串的 slice，没见过，学习一下:

```rust
let x: &[_] = &['1', '2'];
```

`Range`语法：

```rust
let s = String::from("hello");
let len = s.len();
let slice = &s[3..len];
let slice = &s[..2];
let slice = &s[3..];
let slice = &s[..];
```

函数可以返回一个 slice：

```rust
fn main() {
    let s = String::from("hello, world");
    let slice = get_hello();
}

fn get_hello(&s: String) -> &Str {
    &s[..5]
}
```

字符串字面值也是一个 slice，即其类型为`&str`。

也可以向函数直接传递 slice：

```rust
fn first_word(s: &str) -> &str {
    s[..5]
}

fn main() {
    let my_string = String::from("hello, world!");

    let word = first_word(&my_string[0..6]);
    let word = first_word(&my_string[..]);
    let word = first_word(&my_string);

    let my_string_literal = "hello world";

    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);

    let word = first_word(my_string_literal);
}
```

数组也可以有 slice：

```rust
let a = [1, 2, 3, 4, 5];
let slice = &a[1..3];  // slice 的类型为 &[i32]
assert_eq!(slice, &[2, 3]);  // true
```

## 枚举（enum）

枚举类型英文为 enum，每种可能的取值被称为枚举成员（variant）。

```rust
enum IpAddrKind {
    V4,
    V6
}

fn main() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;
}
```

此时`IpAddrKind`就成了一种类型，`V4`和`V6`都成了这个类型的一个取值。

还可以将枚举值和其它类型关联起来：

```rust
enum IpAddr {
    V4(String),
    V6(String)
}

fn main() {
    let home = IpAddr::V4(String::from("127.0.0.1"));
    let loopback = IpAddr::V6(String::from("::1"));
}

enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String)
}

fn main() {
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));
}
```

此时可以把枚举成员的名字看作是一个构建枚举实例的函数。

可以将任意类型的数据作为枚举成员构造函数的参数：基本类型，结构体，另一个枚举。

枚举的定义还可以更加复杂：

```rust
enum Message {
    Quit,
    Move {x: i32, y: i32},
    Write(String),
    ChangeColor(i32, i32, i32),
}
```

我们使用用不同的结构体实现和上面类似的效果：

```rust
struct QuitMessage;  // 类单元结构体
struct MoveMessage {
    x: i32,
    y: i32,
}
struct WriteMessage(String);  // 元组结构体
struct ChangeColorMessage(i32, i32, i32);  // 元组结构体
```

可以在枚举上定义方法：

```rust
impl Message {
    fn call(&self) {
        // ...
    }
}

let m = Message::Write(String::from("hello"));
m.call();
```

两个特殊的枚举：

* `Option`

    `Option`是一个标准库中定义的枚举，它要解决的问题是如何表示一个变量要么有值，要么没值。

    标准库中`Option`的定义如下：

    ```rust
    enum Option<T> {
        None,
        Some(T),
    }
    ```

    其中有两个成员，一个是`Some`，一个是`None`。

    ```rust
    let some_number: Option<i32> = Some(5);
    let some_string: Option<&str> = Some("a string");

    let absent_number: Option<i32> = None;
    ```

    下面这段代码会报错，因为`Option<i8>`和`i8`两种类型之间无法相加：

    ```rust
    let x: i8 = 5;
    let y: Option<i8> = Some(5);

    let sum = x + y;  // error
    ```

    这种方式可以限制空值的泛滥以增加代码安全性。

    `unwrap_or_else()`接收一个闭包作为参数，然后将绑定的值传递到闭包参数里，这样就可以处理一些额外的逻辑了：

    ```rust
    use std::process;

    fn test(val: i32) -> Result<bool, &str> {
        if val == 3 {
            Ok(true)
        } else {
            Err("val is not 3")
        }
    }

    fn main() {
        let result = test(4).unwrap_or_else(|err| {
            println!("the error message is {}", err);
            process::exit(1);
        })
    }
    ```

    Option 常用的方法：

    * `take()`：返回`Option`中的元素，并将原`Option`置为`None`。

        Example:

        ```rust
        let mut p1: Option<Box<String>> = Some(Box::new(String::from("hello")));
        println!("{}", p1 == None);
        let p2 = p1.take();
        println!("{}", p1 == None);
        println!("{}", p2 == Some(Box::new(String::from("hello"))));
        ```

        输出：

        ```
        false
        true
        true
        ```

        因为`take()`会对`p1`进行更改，所以要求`p1`必须是`mut`的。

    * `insert`

        Syntax:

        ```rust
        pub const fn insert(&mut self, value: T) -> &mut T
        where
            T: ~const Destruct,
        ```

        `insert()`会用一个新值替换掉`Option<T>`类型中原来的值，然后将原来的值 drop 掉。

        `insert()`会返回一个`&mut T`类型的值，因此要求`Option<T>`对象本身也必须是`mut`的。

    * 比较

        如果 struct `T`实现了`PartialEq` trait，那么两个`Option<T>`就可以比较是否相等。

    * `unwrap()`

        提取`Option<T>`类型里面的值，并返回。`unwrap()`会发生 move 语义。

    * `replace(val)`

        使用`val`替换`Some()`中的值，并返回`Option<T>`类型的旧值。

    * `as_ref()`

        Converts from &Option<T> to Option<&T>.

    * `as_mut()`

        Converts from `&mut Option<T>` to `Option<&mut T>`.

        如果我们不想动`Option<T>`中的对象（不想发生 move 操作），那么直接用`as_ref()`或`as_mut()`，然后`unwrap()`就可以了。不`as_ref()`或`as_mut()`，直接`unwrap()`会发生 move。

        如果对象`Option<T>`不是 mut 的，那么无法`as_mut()`

    * `inspect()`

        对里面的元素作用一个匿名函数。

    * `is_some()`

    * `is_none()`

    * `is_some_and()`


* `Result`

    `Result`是一种枚举类型，其成员为`Ok`和`Err`。如果`io::Result`实例的值为`Err`，那么`expect()`会中止程序，并返回传入的字符串参数内容；如果实例的值为`Ok`，那么`expcet`会获取`Ok`中的值并原样返回。

    如果不处理`Result`实例，那么编译器会给出警告。

    ```rust
    enum Result<T, E> {
        Ok(T),
        Err(E),
    }
    ```

    ```rust
    use std::fs::File;

    fn main() {
        let f = File::open("hello.txt");
        let f = match f {
            Ok(file) => file,  // file is a std::fs::File 类型
            Err(error) => panic!("Problem opening the file {:?}", error),  // error 是一个 std::io::Error 类型
        };
    }
    ```

    还可以写一个更完善的处理方式：

    ```rust
    use std::fs::File;
    use std::io::ErrorKind;

    fn main() {
        let f = File::open("hello.txt");
        let f = match f {
            Ok(file) => file,
            Err(error) => match error.kind() {
                ErrorKind::NotFound => match File::create("hello.txt") {
                    Ok(fc) => fc,
                    Err(e) => panic!("Problem creating the file: {:?}", e),
                },
            }
        }
    }
    ```

    使用闭包达到同样的效果：

    ```rust
    use std::fs::File;
    use std::io::ErrorKind;

    fn main() {
        let f = File::open("hello.txt").unwrap_or_else(|Error| {
            if error.kind() == ErrorKind::NotFound {
                File::create("hello.txt").unwrap_or_else(|error| {
                    panic!("Problem creating the file: {:?}", error);
                }) 
            } else {
                panic!("Problem opening the file: {:?}", error);
            }
        });
    }
    ```

**`Result`与`Option`上常用的方法**

* `unwarp`:

    如果`Result`值是成员`Ok`，那么`unwrap`会返回`Ok`中的值。如果`Result`是成员`Err`，`unwrap`会调用`panic!`。

    ```rust
    use std::fs::File;

    fn main() {
        let f = File::open("hello.txt").unwrap();
    }
    ```

* `expect`

    和`unwrap`相似，但它允许接受一个参数：

    ```rust
    use std::fs::File;

    fn main() {
        let f = File::open("hello.txt").expect("Failed to open hello.txt");
    }
    ```

    可以返回 Error，这个过程叫做 传播（propagating）错误：

    ```rust
    use std::fs::File;
    use std::io::{self, Read};

    fn read_username_from_file() -> Result<String, io::Error> {
        let f = File::open("hello.txt");

        let mut f = match f {
            Ok(file) => file,
            Err(e) => return Err(e),
        };

        let mut s = String::new();

        match f.read_to_string(&mut s) {
            Ok(_) => Ok(s),
            Err(e) => Err(e),
        }
    }
    ```

    可以使用`?`运算符进一步简化：

    ```rust
    use std::fs::File;
    use std::io;
    use std::io::Read;

    fn read_username_from_file() -> Result<String, io::Error> {
        let mut f = File::open("hello.txt")?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        Ok(s)
    }
    ```

    如果`Result`的值是`Ok`，那么程序继续执行；否则将`Err`值作为函数的返回值返回。`?`会调用`From` trait，将指定的`Err`值的类型转换成函数的返回值类型。我们需要在 trait 中实现所有的转换方式。

    还可以进一步缩减：

    ```rust
    use std::fs::File;
    use std::io;
    use std::io::Read;

    fn read_username_from_file() -> Result<String, io::Error> {
        let mut s = String::new();
        File::open("hello.txt")?.read_to_string(&mut s)?;
        Ok(s)
    }

    这个函数过于常见，以至于标准库提供了这个函数：

    ```rust
    use std::fs;
    use std::io;

    fn read_username_from_file() -> Result<String, io::Error> {
        fs::read_to_string("hello.txt")
    }
    ```

* `?`

    `?`同样可以用于处理``Option`：

    ```rust
    fn last_char_of_first_line(text: &str) -> Option<char> {
        text.lines().next()?.chars().last()
    }
    ```

    `?`不会在`Result`和`Option`间自动转换。


    `main`函数也可以返回`Result`:

    ```rust
    use std::error::Error;
    use std::fs::File;

    fn main() -> Result<(), Box<dyn Error>> {
        let f = File::open("hello.txt")?;
        Ok(())
    }
    ```

## struct

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn main() {
    let mut user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    }

    user1.email = String::from("anotheremail@ecample.com");
}
```

rust 不允许只将某个字段标记为可变。

我们可以在函数中返回这个实例：

```rust
fn build_user(email: String, username: String) -> User {
    User {
        email: email,
        username: username,
        active: true,
        sign_in_count: 1,
    }
}
```

字段初始化简写语法（field init shorhand），这个似乎允许你使用已有的变量给字段赋值：

```rust
fn build_user(email: String, username: String) -> User {
    User {
        email,
        username,
        active: true,
        sign_in_count: 1,
    }
}
```

结构体更新语法：

```rust
fn main() {
    let user2 = User {
        active: user1.active,
        username: user1.username,
        email: String::from("another@example.com"),
        sign_in_count: user1.sign_in_count,
    };
}

// 等同于
fn main() {
    let user2 = User {
        email: String::from("another@example.com"),
        ..user1
    };
}
```

这种方式相当于等号`=`，即按字段移动所有权。此时`user1.username`会失效，如果再在后面使用`user1.username`，那么编译器会报错。

元组结构体：可以不用给结构体中的字段起名字，在访问时可以使用下标来访问：

```rust
struct Color(i32, i32, i32);
struct Color(i32, i32, i32);
fn main() {
    let black = Color(1, 2, 3);
    let oringin = Point(4, 5, 6);

    println!("{}", oringin.0);
}
```

类单元结构体（unit-like structs）：没有字段的结构体。

```rust
struct AlowaysEqual;

fn main() {
    let subject = AlwaysEqual;
}
```

结构体中可以定义方法：

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32
}

impl Rectangle {
    fn area(self: &Self) -> u32 {
        self.width * self.height
    }
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50
    };

    println!("area is {}", rect1.area());
}
```

结构体作为函数的参数：

```rust
impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.wdith && self.height > other.height
    }
}
```

不把`&self`作为第一个参数时，则函数定义成类型上的函数：
 
```rust
impl Rectangle {
    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

fn main() {
    let sq = Rectangle::square(3);  // 使用 :: 调用
}
```

可以将方法分散在多个`impl`模块中，这样等同于在一个`impl`模块中：

```rust
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}
```

## panic

rust 中`assert!`宏检查失败时的突然终止叫做**诧异**（panic）。除了`debug_assert!`宏可以在编译时被跳过，剩下的宏都不会跳过。

```rust
fn gcd(mut n: u64, mut m: u64) -> u64 {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            let t = m;
            m = n;
            n = t;
        }
        m = m % n;
    }
    n
}
```

## 所有权（ownership）

rust 使用两个承诺保证这门语言是安全的：

1. 由程序员决定每个值的生命周期。rust 会在程序员的控制下迅速释放与某个值关联的内容和其他资源。

1. 程序永远不会在一个对象被释放后还使用指向它的指针。

c 和 c++ 只遵循第一个承诺。

rust 中大多数类型的赋值是把值从源变量转移（move）到目标变量，然后源变量变成未初始化状态。

这样带来的结果是赋值的代价很小，并且变量值的所有者清晰，代价是如果需要这些值的副本，必须显式调用：

```rust
let s = vec!["udon".to_string(), "ramen".to_string(), "soba".to_string()];
let t = s.clone();
let u = s.clone();
```

所有权规则：

1. Rust 中的每一个值都有一个被称为其所有者（owner）的变量。
1. 值在任一时刻有且只有一个所有者。
1. 当所有者（变量）离开作用域，这个值将被丢弃。

* 移动（move）

    ```rust
    let s1 = String::from("hello");
    let s2 = s1;
    println!("{}, world!", s1);  // error
    ```

* 克隆（clone）

    ```rust
    let s1 = String::from("hello");
    let s2 = s1.clone();

    println!("s1 = {}, s2 = {}", s1, s2);
    ```

* 只在栈上的数据：拷贝

    ```rust
    let x = 5;
    let y = x;

    println!("x = {}, y = {}", x, y);
    ```

    实现了 copy trait 的类型都遵循直接复制，不存在浅拷贝和深拷贝。下面是一些例子：

    * 整数，浮点数，布尔类型，字符类型
    * 元组，当且仅当其包含的类型也都实现`Copy`的时候。

将值传递给函数可能会发生移动，也可能发生复制：

```rust
fn main() {
    let s = String::from("hello");
    my_func_1(s);  // s 发生移动，这行代码结束后 s 不再有效
    let x = 5;
    my_func_2(x);  // x 发生复制，这行代码结束后 x 依然有效
}

fn my_func_1(some_string: String) {  // some_string 进入作用域
    println!("{}", some_string);
}  // some_string 移出作用域，并调用 drop 方法释放内存

fn makes_copy(some_integer: i32) {  // some_integer 进入作用域
    println!("{}", some_integer);
}  // 这里 some_integer 被移出作用域
```

函数的返回值也是移交所有权：

```rust
fn main() {
    let s1 = my_func_1();  // 函数返回值转移给 s1

    let s2 = String::from("hello");  // s2 进入作用域

    let s3 = String::my_func_2(s2);  // s2 被移动到函数内，函数返回值移动给 s3
}  // s3 被移出作用域并释放内存，s2 被移出作用域，s1 被移出作用域并释放内存

fn my_func_1() -> String {
    let some_string = String::from("yours");
    some_string
}

fn my_func_2(a_string: String) -> String {
    a_string
}
```

partially move:

struct 对象中只有部分字段发生了 move，那么这个行为称为 partially move。partially moved 的对象不能再调用方法。但是可以继续 move 其他字段。

## 表达式

语句（statements）是执行一些操作但不返回值的指令，表达式（expressions）计算并产生一个值。

常见的语句：

```rust
// 给变量绑定值
let y = 6;

// 定义函数
fn main() {
    let y = 6;
}
```

因为赋值语句不是表达式，所以 rust 中不能这样写：

```rust
let x = y = 6;
```

常见的表达式：

```rust
// 代码块
{
    let x = 3;
    x + 1
}

// if 语句
if n < 5 {
    println!("condition was true");
} else {
    println!("condition was false");
}
```

表达式的结尾没有分号，如果在表达式的结尾加上分号，那么它就变成了语句。实际上，语句的返回值是单位类型`()`，表示不返回值。

如果函数体中最后一行代码是一个表达式，且表达式末尾没有分号，那么这个表达式的值就是函数的返回值。`return`语句一般只用于在函数的中间提前返回。


因为`if`语句是表达式，所以 rust 代码可以这样写：

```rust
fn main() {
    let condition = true;
    let number = if condition { 5 } else { 6 };
    println!("The value of number is: {number}");
}
```

如果`if`语句中两个分支的值不是同一个类型，编译器会报错：

```rust
fn main() {
    let condition = true;
    let number = if condition {5} else {"six"};
    println!("The value of number is: {number}");
}
```

* `loop`

    ```rust
    fn main() {
        loop {
            println!("again!");
        }
    }
    ```

    ```rust
    fn main() {
        let mut counter = 0;
        let result = loop {
            counter += 1;
            if counter == 10 {
                break counter * 2;  // break 用于返回值
            }
        };  // 注意，这里的分号指的是 let result = xxx; 语句的分号
        // 而不是 loop 语句的分号

        println!("The result is {result}");
    }
    ```

    使用循环标签跳出嵌套循环：

    ```rust
    fn main() {
        let mut count = 0;
        'counting_up: loop {
            println!("count = {count}");
            let mut remaining = 10;

            loop {
                println!("remaining = {remaining}");
                if remaining == 9 {
                    break;
                }
                if count == 2 {
                    break 'counting_up;
                }
                remaining -= 1;
            }
            count += 1;
        }
        println!("End count = {count}");
    }
    ```

* `while`

    ```rust
    fn main() {
        let mut number = 3;
        while number != 0 {
            println!("{number}!");
            number -= 1;
        }
        println!("LIFTOFF!!!");
    }
    ```

* `for`

    ```rust
    fn main() {
        let a = [10, 20, 30, 40, 50];

        for element in a {
            println!("the value is: {element}");
        }
    }
    ```

    在`for`中使用`Range`类型：

    ```rust
    fn main() {
        for number in (1..4).rev() {
            println!("{number}!");
        }
        println!("LIFTOFF!!!");
    }
    ```

rust 中的`if`和`match`可以产生值：

```rust
pixels[r * bounds.0 + c] = 
    match escapes(Complex { re: point.0, im: point.1 }, 255) {
        None => 0,
        Some(count) => 255 - count as u8
    };
```

```rust
let status = 
    if cpu.templerature <= MAX_TEMP {
        HttpStatus::Ok
    } else {
        HttpStatus::ServerError
    };
```

```rust
println!("Inside the vat, you see {}.",
    match vat.contents {
        Some(brain) => brain.desc(),
        None => "nothing of interest"
    });
```

* `match`

    一个`match`表达式由分支（arms）构成。一个分支包含一个模式（pattern）和表达式开头的值与分支模式相匹配时应该执行的代码。

    ```rust
    match guess.cmp(&secret_number) {
        Ordering::Less => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal => println!("You win!"),
    }
    ```

    ```rust
    let guess: u32 = match guess.trim().parse() {
        Ok(num) => num,
        Err(_) => continue,
    };
    ```

    `match`表达式可以处理枚举变量：

    ```rust
    enum Coin {
        Penny,
        Nickel,
        Dime,
        Quarter
    }

    fn value_in_cents(coin: Coin) -> u8 {
        match coin {
            Coin::Penny => {
                println!("Lucky penny!");
                1
            },
            Coin::Nickel => 5,
            Coin::Dime => 10,
            Coin::Quarter => 25,
        }
    }
    ```

    `match`本身也是个表达式，它的返回值的类型是通过外部的上下文推导得出的，而不是通过分支得出的。

    `match`表达式可以自动从`enum`类型的变量中提取值：

    ```rust
    #[derive(Debug)]
    enum UsState {
        Alabama,
        Alaska,
        // --snip--
    }

    enum Coin {
        Penny,
        Nickel,
        Dime,
        Quarter(UsState),
    }

    fn value_in_cents(coin: Coin) -> u8 {
        match coin {
            Coin::Penny => 1,
            Coin::Nickel => 5,
            Coin::Dime => 10,
            Coin::Quarter(state) => {
                println!("State quarter from {:?}!", state);
                25
            }  // 注意看这个 branch，它提取出 coins 实例中的 UsState 的实际值
        }
    }

    fn main() {
        value_in_cents(Coin::Quarter(UsState::Alaska));
    }
    ```

    使用`match`处理`Option`类型：

    ```rust
    fn plus_one(x: Option<i32>) -> Option<i32> {
        match x {
            None => None,
            Some(i) => Some(i + 1),
        }
    }

    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);
    ```

    Rust 要求`match`必须考虑到所有情况，否则编译器会报错。这种特性被称为 exhaustive。

    可以用`other`处理其他所有情况：

    ```rust
    let dice_roll = 9;
    match dice_roll {
        3 => add_fancy_hat(),
        7 => remove_fancy_hat(),
        other => move_player(other),  // 此时 dice_roll 的值将被绑定给 other，other 这个变量也是我们自定义的
    }

    fn add_fancy_hat() {}
    fn remove_fancy_hat() {}
    fn move_player(num_spaces: u8) {}
    ```

    通配分支被要求放到最后，因为模式是按顺序匹配的。也可以使用`_`告诉编译器这个值不想被使用：

    ```rust
    let dice_roll = 9;
    match dice_rool {
        3 => add_fancy_hat(),
        7 => remove_fancy_hat(),
        _ => reroll(),
    }

    fn add_fancy_hat() {}
    fn remove_fancy_hat() {}
    fn reroll() {}
    ```

    可以使用`()`表示不执行任何代码：

    ```rust
    match dice_roll {
        3 => add_fancy_hat(),
        7 => remove_fancy_hat(),
        _ => (),
    }
    ```

* `if`

    ```rust
    fn main() {
        let number = 6;
        if number % 4 == 0 {
            println!("number is divisible by 4");
        } else if number % 3 == 0 {
            println!("number is divisible by 3");
        } else if number % 2 == 0 {
            println!("number is divisible by 2");
        } else {
            println!("number is not divisible by 4, 3, or 2");
        }
    }
    ```

在 rust 中，块也是表达式，因此可以产生值。

```rust
let display_name = match post.author() {
    Some(author) => author.name(),
    None => {
        let network_info = post.get_network_metadata()?;
        let ip = network_info.client_address();
        ip.to_string()
    }
};
```

分号`;`结尾的表达式其值为`()`。

对于`let`声明，分号是必需的：

```rust
let dandelion_control = puffball.open();
```

空语句可以出现在块中：

```rust
loop {
    work();
    play();
    ;
}
```

rust 遵循 c 的传统，允许出现这种情况。空语句除了传达一丝淡淡的惆怅外什么也不干。这里提到它也只是出于圆满的考虑。

* `if let`

    下面两段代码是等价的：

    ```rust
    let config_max = Some(3u8);
    match config_max {
        Some(max) => println!("The maximum is configured to be {}", max),
        _ => (),
    }
    ```

    ```rust
    let config_max = Some(3u8);
    if let Some(max) = config_max {
        println!("The maximum is configured to be {}", max);
    }
    ```

    可以在`if let`后增加一个`else`分支，表示`_ => do_something,`。下面两个代码片段等价：

    ```rust
    let mut count = 0;
    match coin {
        Coin::Quarter(state) => println!("State quarter from {:?}!", state),
        _ => count += 1,
    }
    ```

    ```rust
    let mut count = 0;
    if let Coin::Quarter(state) = coin {
        println!("State quarter from {:?}!", state);
    } else {
        count += 1;
    }
    ```

    也可以使用`while let Some(val) = xxx {}`

## 函数

rust 中函数的定义出现在调用之前还是之后都无所谓，只要在与调用处同一作用域就行。

函数的参数：

```rust
fn main() {
    another_function(5);
}

fn another_function(x: i32) {  // 类型注解是必须的
    println!("The value of x is: {x}");
}
```

* 数组作为函数参数

    如果参数是可变参数，需要加上`mut`：

    ```rust
    fn myfunc(mut arr: [i32; 5]) {
        // ...
    }
    ```

* 引用作为函数参数

    如果想做 in-place 的修改，可以使用引用：

    ```rust
    fn main() {
        let mut arr = [1, 2, 3, 4, 5];
        reverse(&mut arr);
        println!("{:?}", arr);
        
    }

    fn reverse(arr: &mut [i32; 5]) {
        let mut i = 0;
        let mut j = arr.len() - 1;
        let mut temp;
        while i < j {
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i += 1;
            j -= 1;
        }
    }
    ```

    输出：

    `[5, 4, 3, 2, 1]`

    注：

    1. 非`mut`的变量可以拿到`&mut`引用吗？

函数可以指定返回值的类型：

```rust
fn five() -> i32 {
    5
}

fn main() {
    let x = five();
    println!("The value of x is: {x}");
}
```

* 数组作为函数返回值

    ```rust
    fn main() {
        let mut arr = [1, 2, 3, 4, 5];
        arr = reverse(arr);
        println!("{:?}", arr);
        
    }

    fn reverse(mut arr: [i32; 5]) -> [i32; 5] {
        let mut i = 0;
        let mut j = arr.len() - 1;
        let mut temp;
        while i < j {
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i += 1;
            j -= 1;
        }
        arr
    }
    ```

    注：

    1. 不明白这样是把数组又复制了一份，还是只传递了数组的地址

## 泛型（generics）

```rust
fn largest<T>(list: &[T]) -> T {

}

struct Point<T> {
    x: T,
    y: T,
}

fn main() {
    let integer = Point {x: 5, y: 10};
    let float = Point {x: 1.0, y: 4.0};
}

struct Point<T, U> {
    x: T,
    y: U,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// 单独为某一种类型实现泛型
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}
```

函数和`impl`使用不一样的类型：

```rust
struct Point<X1, Y1> {
    x: X1,
    y: Y1,
}

impl<X1, Y1> Point<X1, Y1> {
    fn mixup<X2, Y2>(self, other: Point<X2, Y2>) -> Point<X1, Y2> {
        Point {
            x: self.x,
            y: other.y
        }
    }
}

fn main() {
    let p1 = Point {x: 5, y: 10.4};
    let p2 = Point {x: "Hello", y: 'c'};

    let p3 = p1.mixup(p2);
    println!("p3.x = {}, p3.y = {}", p3.x, p3.y);
}
```

`Point`里是`X1`，`Y1`，`mixup()`返回的是`X1`，`Y2`，为什么编译器可以把`Y2`转换成`Y1`？

泛型中`impl`后面跟的可以和`Type`后面跟的不一样：

```rust
use std::fmt::Display;

struct Pair<T> {
    x: T,
    y: T
}

impl<T> Pair<T> {  // 为什么这里要写两个 T？这两个 T 有什么不同的含义
    fn new(x: T, y: T) -> Self {
        Self {x, y}
    }
}

impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("The largest member is x = {}", self.x);
        } else {
            println!("The largest member is y = {}", self.y);
        }
    }
}
```

还可以有条件地实现`impl`：

```rust
impl<T: Display> ToString for T {

}
```

上面的例子中，凡是实现了`Display` trait 的类型，都可以实现当前`ToString`这个泛型，从而转换成字符串。

## 集合（collections）

### vector

* 创建，初始化：

    ```rust
    let v: Vec<i32> = Vec::new();
    let vv = vec![1, 2, 3];
    let mut v = Vec::from([0; 5]);
    let mut v: Vec<Vec<i32>> = vec![vec![0; 4]; 3];  // 3 行 4 列的二维数组
    let mut v = Vec::from_iter((0..5).into_iter());  // 使用 range 创建
    ```

添加元素：

```rust
let mut v = Vec::new();
v.push(5);
v.push(6);
```

访问：

```rust
let third: &i32 = &v[2];  // 若越界，会 panic

match v.get(2) {
    Some(third) => println!({}, third),
    None => println!("there is nothing"),
}  // get 返回的是一个 Option<&T>
```

vector 的引用会考虑到释放内存和指向新内存，因此元素的引用与 vector 的行为要适应：

```rust
let mut v = vec![1, 2, 3];
let first = &v[0];
v.push(6);
println!("the first elm is: {}", first);  // 会报错，first 必须是 mut 才行，因为 vector 有可能释放旧内存，申请新内存。
```

遍历：

```rust
let v = vec![100, 32, 57];
for i in &v {
    println!("{}", i);
}

let mut v = vec![1, 2, 3];
for i in &mut v {
    *i += 50;
}
```

vector 可以通过 enum 存储不同的类型：

```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String)
}

let row = vec![
    SpreadsheetCell::Int(3),
    SpreadsheetCell::Text(String::from("blue")),
    SpreadsheetCell::Float(10.12),
];
```

### hash map

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);
```

也可以使用 iter 结合 vector 来创建 HashMap:

```rust
use std::collections::HashMap;

let teams = vec![String::from("Blue"), String::from("Yellow")];
let initial_scores = vec![10, 50];

let mut scores: HashMap<_, _> = teams.info_iter().zip(initial_scores.into_iter()).collect();
```

使用`HashMap<_, _>`下划线占位，rust 可以推断出`HashMap`所包含的类型。

HashMap 同样涉及所有权的问题：

```rust
use std::collections.HashMap;
let filed_name = String::from("Favorite color");
let field_value = String::from("Blue");

let mut map = HashMap::new();
map.insert(field_name, field_value);  // ownership has beed moved
```

访问：

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

let team_name = String::from("Blue");
let score = scores.get(&team_name);  // Some(10)

for (key, value) in &scores {
    println!("{}: {}", key, value);
}
```

更新时直接使用`insert()`就可以了。

可以使用`entry()`检查一个键是否存在：

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);

scores.entry(String::from("Yellow")).or_insert(50);
scores.entry(String::from("Blue")).or_insert(50);

println!("{:?}", scores);
```

`entry()`返回一个枚举`Entry`。`or_insert`表示键对应的值存在时就返回这个值的可变引用，若不存在，则插入新值并返回新值的可变引用。

若不存在则插入，若存在则增加计数：

```rust
use std::collections::HashMap;

let text = "hello world wonderful world";

let mut map = HashMap::new();

for word in text.split_whitespace() {
    let count = map.entry(word).or_insert(0);
    *count += 1;
}

println!("{:?}", map)
```

### BinaryHeap

默认是大顶堆。

```rust
use std::collections::BinaryHeap;
fn main() {
    let mut q = BinaryHeap::new();
    q.push(3);
    q.push(5);
    q.push(2);
    println!("{}", q.peek().unwrap());
    q.pop();
    println!("{}", q.peek().unwrap());
}
```

输出：

```
5
3
```

如果想使用小顶堆，可以使用`std::cmp::Reverse;`改变整数的`Ord` trait：

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn main() {
    let mut q = BinaryHeap::new();
    q.push(Reverse(3));
    q.push(Reverse(5));
    q.push(Reverse(2));
    println!("{}", q.peek().unwrap().0);
    q.pop();
    println!("{}", q.peek().unwrap().0);
}
```

输出：

```
2
3
```

常用的一些 methods:

* `pub fn into_sorted_vec(self) -> Vec<T, Global>`

    Consumes the BinaryHeap and returns a vector in sorted (ascending) order.

    不知道这里写的 Global 是什么意思。在 example 里面看，好像 Global 没什么用。

    ```rust
    use std::collections::BinaryHeap;

    let mut heap = BinaryHeap::from([1, 2, 4, 5, 7]);
    heap.push(6);
    heap.push(3);

    let vec = heap.into_sorted_vec();
    assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7]);
    ```

* `pub fn append(&mut self, other: &mut BinaryHeap<T>)`

    Moves all the elements of other into self, leaving other empty.

    Example:

    ```rust
    use std::collections::BinaryHeap;

    let mut a = BinaryHeap::from([-10, 1, 2, 3, 3]);
    let mut b = BinaryHeap::from([-20, 5, 43]);

    a.append(&mut b);

    assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
    assert!(b.is_empty());
    ```

* `pub fn iter(&self) -> Iter<'_, T>`

    Returns an iterator visiting all values in the underlying vector, in arbitrary order.

    ```rust
    use std::collections::BinaryHeap;
    let heap = BinaryHeap::from([1, 2, 3, 4]);

    // Print 1, 2, 3, 4 in arbitrary order
    for x in heap.iter() {
        println!("{x}");
    }
    ```

* `pub fn capacity(&self) -> usize`

    Returns the number of elements the binary heap can hold without reallocating.

* `pub fn reserve(&mut self, additional: usize)`

    Reserves capacity for at least additional elements more than the current length. The allocator may reserve more space to speculatively avoid frequent allocations. After calling reserve, capacity will be greater than or equal to self.len() + additional. Does nothing if capacity is already sufficient.

* `pub fn into_vec(self) -> Vec<T, Global>`

    Consumes the BinaryHeap and returns the underlying vector in arbitrary order.

* `pub fn len(&self) -> usize`

    Returns the length of the binary heap.

* `pub fn is_empty(&self) -> bool`

    Checks if the binary heap is empty.

* `pub fn drain(&mut self) -> Drain<'_, T>`

    Clears the binary heap, returning an iterator over the removed elements in arbitrary order. If the iterator is dropped before being fully consumed, it drops the remaining elements in arbitrary order.

    The returned iterator keeps a mutable borrow on the heap to optimize its implementation.

    Example:

    ```rust
    use std::collections::BinaryHeap;
    let mut heap = BinaryHeap::from([1, 3]);

    assert!(!heap.is_empty());

    for x in heap.drain() {
        println!("{x}");
    }

    assert!(heap.is_empty());
    ```

* `pub fn clear(&mut self)`

    Drops all items from the binary heap.

### VecDeque

增加了`push_back()`，`push_front()`，`pop_back()`，`pop_front()`这些功能。剩下的功能和`Vec`差不多。

### LinkedList

双向链表。

### BTreeMap, BTreeSet

二叉搜索树实现的类似 c++ 中`map`和`set`的数据结构。不清楚 rust 中实现的是不是红黑树。有时间了再研究吧，顺便把 b-tree，二叉平衡搜索树和红黑树全部研究一遍。

```rust
let mut t = BTreeSet::new();
t.insert(5);
t.insert(4);
t.insert(3);
println!("{}", t.first().unwrap());
let m: Option<&i32> = t.get(&8);
```


## trait

trait 不可以有自己的成员变量。

### Basic usage

* 实现一个简单的 trait

    ```rust
    pub trait Summary {
        fn summarize(&self) -> String;
    }

    pub struct NewsArticle {
        pub headline: String,
        pub location: String,
        pub author: String,
        pub content: String,
    }

    impl Summary for NewsArticle {
        fn summarize(&self) -> String {
            format!("{}, by {} ({})", self.headline, self.author, self.location)
        }
    }

    pub struct Tweet {
        pub username: String,
        pub content: String,
        pub reply: bool,
        pub retweet: bool,
    }

    impl Summary for Tweet {
        fn summarize(&self) -> String {
            format!("{}: {}", self.username, self.content)
        }
    }
    ```

* trait 与泛型

    ```rust
    use std::fmt::Display;

    struct Pair<T> {
        x: T,
        y: T,
    }

    impl<T> Pair<T> {
        fn new(x: T, y: T) -> Self {
            Self {
                x,
                y,
            }
        }
    }

    impl<T: Display + PartialOrd> Pair<T> {  // 对不同泛型实现不同的 struct 方法
        fn cmp_display(&self) {
            if self.x >= self.y {
                println!("The largest member is x = {}", self.x);
            } else {
                println!("The largest member is y = {}", self.y);
            }
        }
    }
    ```

    还可以为一个有条件约束的泛型实现一个 trait：

    ```rust
    impl<T: Display> ToString for T {
        // --snip--
    }
    ```

    这样的好处是不需要写一个具体的 struct，可以匹配到很多的 struct：

    ```rust
    trait Print {
        fn print(&self);
    }

    trait PrintExtraMsg {
        fn print_ext(&self);
    }

    struct MessageType_1 {
        val: i32
    }

    struct MessageType_2 {
        msg: String
    }

    impl Print for MessageType_1 {
        fn print(&self) {
            println!("Message i32: {}", self.val);
        }
    }

    impl Print for MessageType_2 {
        fn print(&self) {
            println!("Message str: {}", self.msg);
        }
    }

    impl<T: Print> PrintExtraMsg for T {  // 本来需要为 MessageType_1 和 MessageType_2 分别实现 PrintExtraMsg trait，但是我们在这里用泛型代替两个 struct，只需要写一个实现就可以了
        fn print_ext(&self) {
            self.print();  // 这种实现方法的缺点是不能拿到 struct 的成员变量，只能调用上级 trait 的函数
            println!("extra greetings");
        }
    }

    fn main() {
        let mut obj = MessageType_1 {
            val: 42
        };
        obj.print_ext();
    }
    ```

可以为自己的类型实现外部 trait，但是不能为外部类型实现外部 trait。这点与 c++ 有非常大不同。

trait 的默认实现：

```rust
pub trait Summary {
    fn summarize(&self) -> String {
        String::from("(Read more...)");
    }
}

impl Summary for NewsArticle {}
```

在 trait 中，默认实现允许调用相同 trait 中的其他方法：

```rust
pub trait Summary {
    fn summarize_author(&self) -> String;

    fn summarize(&self) -> String {
        format!("(Read more from {}...)", self.summarize_author())
    }
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
}
```

可以将`trait`作为参数：

```rust
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}
```

`impl Summary`表示某个实现了`Summary` trait 的类型。

这种写法等价于：

```rust
pub fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}
```

同时实现多个 trait 的情形：

```rust
pub fn notify(item: &(impl Summary + Display)) {

}

// 或者这样写
pub fn notify<T: Summary + Display>(item: &T) {

}
```

可以使用`where`简化泛型的写法：

```rust
fn some_function<T: Display + Clone, U: Clone + Debug>(t: &T, u: &U) -> i32 {

}

// 可以使用 where 写成
fn some_function<T, U>(t: &T, u: &U) -> i32
    where T: Display + Clone,
        U: Clone + Debug
{

}   
```

返回值也可以使用`impl` trait 语法：

```rust
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("of course, as you probably already know, people"),
        reply: false,
        retweet: false,
    }
}
```

但是有时候返回值的类型的可能性不止一种，就无法用这样的方法：

```rust
fn returns_summarizable(switch: bool) -> impl Summary {
    if switch {
        NewsArticle {
            headline: String::from(
                "Penguins win the Stanley Cup Championship!",
            ),
            location: String::from("Pittsburgh, PA, USA"),
            author: String::from("Iceburgh"),
            content: String::from(
                "The Pittsburgh Penguins once again are the best \ hockey team in the NHL.",
            ),
        }
    } else {
        Tweet {
            username: String::from("horse_ebooks"),
            content: String::from(
                "of course, as you probably already know, people",
            ),
            reply: false,
            retweet: false
        }
    }
}
```

一个实现了寻找最大值的泛型函数（这节应该放到 generic 主题里去）：

```rust
fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
    let mut largest = list[0];
    for &item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];

    let result = largest(&number_list);
    println!("The largest number is {}", result);

    let char_list = vec!['y', 'm', 'a', 'q'];

    let result = largest(&char_list);
    println!("The largest char is {}", result);
}
```

### Copy trait

rust 中的很多类型都没实现`Copy` trait，只有少数的几个内置基本类型实现了`Copy` trait。

如果一个类型实现了`Copy` trait，那么在`=`时就会优先 copy，否则会选择 move。

实现一个类型的 copy trait，可以选择直接使用`derive`：

```rust
#[derive(Copy, Clone)]
struct MyStruc {
    x: i32,
    y: f64
}
```

也可以自己实现：

```rust
struct MyStruc {
    x: i32
}

impl Copy for MyStruc {

}

impl Clone for MyStruc {
    fn clone(&self) -> Self {
        MyStruc {
            x: self.x
        }
    }
}
```

`Copy` trait 的实现依赖`Clone`，所以需要先实现`Clone`。

## 生命周期

生命周期注解：

```rust
&i32
&'a i32
&'a mut i32
```

生命周期注解不影响生命周期。

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

生命周期注解向函数保证，函数的返回引用的生命周期等于`x`和`y`中较短的那个。同时也告诉编译器，在函数外部，函数返回值的生命周期与传入参数中生命周期较短的那个一致。

不一定两个参数都要有生命周期注解：

```rust
fn longest<'a>(x: &'a str, y: &str) -> &'a str {
    x
}
```

如果函数返回值的生命周期与参数完全没有关系，那么也会出错：

```rust
fn longest<'a>(x: &str, y: &str) -> &'a str {
    let result = String::from("really long string");
    result.as_str()
}
```

结构体中的生命周期注解：

```rust
struct ImportantExcerpt<'a'> {
    part: &'a str,  // ImportantExcerpt 的实例不能比其 part 字段中的引用存在的更久
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a 'a'");
    let i = ImportantExcerpt {
        part: first_sentence,
    }
}
```

这个注解意味着`ImportantExcerpt`的实例不能比其`part`字段中的引用存在得更久。

生命周期的省略规则（lifetime elision rules）：

函数或方法的参数的生命周期被称为输入生命周期（input lifetimes），返回值的生命周期被称为输出生命周期（output lifetimes）。

1. 每一个是引用的参数都有它自己的生命周期参数。

    即有一个引用参数的函数有一个生命周期参数：`fn foo<'a>(x: &'a i32)`，有两个引用参数的函数有两个不同的生命周期函数，`fn foo<'a, 'b>(x: &'a i32, y: &'b i32)`

1. 如果只有一个输入生命周期参数，那么它被赋予所有输出生命周期参数

    `fn foo<'a>(x: &'a i32) -> &'a i32`

1. 如果方法有多个输入生命周期参数并且其中一个参数是`&self`或`&mut self`，那么所有输出生命周期参数被赋予`self`的生命周期。

examples:

```rust
fn my_func_1(s: &str) -> &str {}

fn my_func_1<'a>(s: &'a str) -> &str {}  // 使用第一条规则，为每个输入参数赋予一个生命周期注解

fn my_func_2<'a>(s: &'a str) -> &'a str {}  // 应用第二条规则，将输入参数的生命周期注解赋给输出参数，这样就得到了输出参数的生命周期


// 第二个例子
fn my_func_2(x: &str, y: &str) -> &str {}

fn my_func_2<'a, 'b>(x: &'a str, y: &'b str) -> &str {}  // 应用第一条规则，为每个输入参数赋予一个生命周期

// 此时应用第二条规则，我们发现对于输出参数的生命周期，选择 'a 和 'b 存在歧义，因此无法推断出输出参数的生命周期。此时编译品就会报错
```

如果在应用了三个规则后，编译器仍然没有计算出返回参数的生命周期，那么就会报错。

rust 要求被引用对象的生命周期大于等于引用的生命周期。否则编译器会报错。

有时函数无法返回引用，是因为它不知道返回的引用在外部的生命周期与其所指向对象的生命周期的关系。

`fn func<'a>(x: &'a i32)`表示函数``func`的生命周期不能超过引用`x`的生命周期。这样编译器就知道了返回值和参数之间的关系，编译器只需要检查生命周期是否满足要求就可以了，如果满足，那么一定不会发生悬垂引用（即引用指向一块已经被释放的内存）。

函数中的生命周期语法用于将函数的多个参数与其返回值的生命周期进行关联。

结构体中字段的生命周期：

1. 结构体中的字段如果是引用，那么必须要加生命周期注解。

    ```rust
    struct MyStruc<'a> {
        my_int: &'a i32,
        my_str: &'a str,
        my_bool: &bool,  // error
    }
    ```

    结构体中的生命周期注解表示字段的生命周期大于等于结构体实例对象的生命周期。

    ```rust
    struct MyStruc<'a, 'b> {
        my_str: &'a str,
        my_int: &'b i32,
    }
    ```

1. 结构体中方法的生命周期既可以使用结构体的生命周期注解，也可以使用自己的生命周期注解

    ```rust
    struct MyStruc<'a> {
        m_val: &'a i32,
    }

    impl<'a> MyStruc<'a> {
        fn plus_val<'b>(&'a self, &'b val) -> i32 {
            self.m_val
        }
    }
    ```

## 错误处理

## 迭代器 Iterators

### CharIndices

`String`对象有个`char_indices()`方法，可以返回一个`CharIndices`类型的对象。

## 包和模块

### 包

重新编译一个项目，查看它用了哪些包：

```bash
$ cargo clean
$ cargo build --verbose
```

在`main.rs`中，

```rust
extern crate package_name;
```

表示`package_name`是外部库，并不是此项目本身的代码。

我们在`Cargo.toml`文件里可以指定包对应的版本：

```toml
[dependencies]
num = "0.1.27"
image = "0.6.1"
crossbeam = "0.2.8"
```

Cargo 会从 github 上（也可能是<https://crates.io>）下载这些包的源代码，然后对每个包运行一次`rustc`，并加上`--crate-type lib`参数（或`cargo new --lib xxx_proj`），生成一个`.rlib`文件。在编译主程序时，`rustc`会加上`--crate-type bin`参数，生成一个可执行的二进制文件。

cargo 会自动处理依赖，因此我们不必把每个 dependency 的依赖都列出来。

`cargo build --release`会编译优化的代码。此时不会检查整数溢出，还会跳过`debug_assert!()`断言，另外它们针对诧异生成的栈追踪信息一般不太可靠。

可以放到`Cargo.toml`文件中的几种配置：

* `[profile.dev]`：`cargo build`
* `[profile.release]`：`cargo build --release`
* `[profile.test]`：`cargo test`

如果想分析程序占用 CPU 的时间，那么需要既开启优化，又添加调试符号（symbol），此时需要在`Cargo.toml`中添加如下代码：

```toml
[profile.release]
debug = true
```

rust 默认导入一些库，称为预导入(prelude)。

一些常见的预导入：

`String`, `Vec`

构建文档并在浏览器中打开：

`cargo doc --open`

package 中会包含多个 crate。package 最多包含一个 library crate，可以包含多个 binary crate，这两者至少要有一个。

### 模块

* 先看一种最简单的情况，在`main.rs`中定义一个`mod`

    `src/main.rs`:

    ```rust
    mod MyMod {  // 如果 mod 出现在当前文件中，那么就不需要用 mod/use 之类的引入
        pub struct MyStru {  // 不使用 pub 的话，即使是对当前文件，也是不可见的
            pub val: i32,  // struct 是否 pub 与 val 是否 pub，是独立的
        }

        impl MyStru {
            pub fn print(&self) {  // 不使用 pub 的话，下面 m_obj.print() 无法被调用
                println!("{}", self.val);
            }
        }
    }

    fn main() {
        let m_obj = MyMod::MyStru {  // 即使是当前文件，也需要用 mod 名称索引
            val: 42
        };
        m_obj.print();
    }
    ```

* 比如说想在`main.rs`里调用其他文件里的 struct，trait，函数等，可以这样：

    `src/main.rs`:

    ```rust
    mod greeting;  // 引入 mod。只有先引入，才能用 use。同目录下，文件名就是 mod 名。
    use greeting::*;  // use 的作用仅仅是简化路径

    fn main() {
        let m_grt = MyGreeting {
            greeting: String::from("hello"),
        };
        m_grt.print();
        print_hello_world();
    }
    ```

    `src/greeting.rs`:

    ```rust
    pub trait SayGreeting {  // 默认情况下，mod 中的东西，以及 struct 中的成员，都只对当前 mod 和下级 mod 可见，对上级 mod 不可见。
        fn print(&self);
    }

    pub struct MyGreeting {
        pub greeting: String,  // 如果这个 pub 不写，那么 main.rs 中无法直接在 MyGreeting 中填写字段
    }

    impl SayGreeting for MyGreeting {
        fn print(&self) {
            println!("{}", self.greeting);
        }
    }

    pub fn print_hello_world() {
        println!("hello, world from greeting.rs");
    }
    ```

    输出：

    ```
    hello
    hello, world from greeting.rs
    ```

* 文件夹算作一个 mod，下面的文件也算作一个 mod

* 

`cargo create --lib restaurant`

```rust
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}
        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}
```

模块结构：

``` 
crate
 |_____front_of_house
       |____ hosting
       |     |___ add_to_waitlist
       |     |___ seat_at_table
       |
       |____ serving
             |___ take_order
             |___ serve_order
             |___ take_payment
       
```

其中`crate`模块即为根模块，其对应的文件为`src/main.rs`或`src/lib.rs`

* 绝对路径（absolute path）

    若被调用函数和调用函数在同一个`crate`中，那么可以使用以`crate`为根的绝对路径。

    `crate::front_of_house::hosting::add_to_waitlist();`

* 相对路径（relative path）

    `front_of_house::hosting::add_to_waitlist();`

父模块中定义的东西可以被子模块看见，但是子模块的东西无法被父模块看到。siblings 之间的关系是可以互相看到的。若想把暴露出来 ，需要加上`pub`关键字：

`src/lib.rs`

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // absoluate path
    crate::front_of_house::hosting::add_to_waitlist();

    // relative path
    front_of_house::hosting::add_to_waitlist();
}
```

可以使用`super`访问上一层 mod：

```rust
fn server_order() {}

mod back_of_house {
    fn fix_incorrect_order() {
        cook_order();
        super::serve_order();
    }

    fn cook_order() {}
}
```

同样的访问关系也适用于`struct`和`enum`：

```rust
mod back_of_house {
    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String,  // seasonal_fruit can't be changed out of back_of_house mod
    }

    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {  // because the presence of private member, this constructor is necessary.
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
}
```

如果将枚举类型设置为`pub`，那么它的所有成员都变成公有：

```rust
mod back_of_house {
    pub enum Appetizer {
        Soup,
        Salad,
    }
}

pub fn eat_at_restaurant() {
    let order1 = back_of_house::Appetizer::Soup;
    let order2 = back_of_house::Appetizer::Salad;
}
```

可以使用`use`来简化路径：

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use create::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
}
```

`use`也可以使用相对路径：

```rust
use self::front_of_house::hosting;
```

也可以直接使用`use`指定到函数级别，但是一般不这么用。这是为了区分本地函数和其他`mod`中的函数。

使用`as`指定别名：

```rust
use std::fmt::Result;
use std::io::Result as IoResult;
```

重导出（re-exporting）：

```rust
pub use crate::front_of_house::hosting;
```

这样别人就可以从这个 mod 中再次将`hosting`导入到他们自己的 mod 中了。

一些简便写法：

```rust
use std::cmp::Ordering;
use std::io;

// 等价于
use std::{cmp::Ordering, io};


use std::io;
use std::io::Write;
// 等价于
use std::io::{self, Write};
```

引入一个路径下的所有公有项：

```rust
use std::collections::*;
```

多个文件：

`src/lib.rs`

```rust
mod front_of_house;  // 在这里声明使用的模块（文件名）

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
}
```

`src/front_of_house.rs`

```rust
pub mod hosting {
    pub fn add_to_waitlist() {}
}
```

在上面的文件中，`hosting`模块有两层，我们可以继续解构。在`src/front_of_house`下创建文件：`hosting.rs`。然后修改两个文件内容如下：

`src/front_of_house.rs`：

```rust
pub mod hosting;
```

`src/front_of_house/hosting.rs`:

```rust
pub fn add_to_waitlist() {}
```

其它的无需改变，也可以编译运行。

`lib.rs`已经算是子模块了，在`main.rs`需要使用`use <proj_name>::xxx`引入，或者直接使用`<proj_name>::xxx`调用函数。

## 类

`SomeClass::func`表示`func`是`SomeClass`类型的一个关联函数（associated function）。在其他语言中它被称为静态方法（static method）。

## IO

* `println!`宏

    字符串的格式化输出：

    ```rust
    let x = 5;
    let y = 10;
    println!("x = {} and y = {}", x, y);
    println!("x = {x} and y = {y}");
    ```

### read from stdin

```rust
use std::io;

fn get_line() -> String {
    let mut line = String::new();
    let result = io::stdin().read_line(&mut line);
    match result {
        Err(err) => println!("error meg: {err}"),
        Ok(ok) => println!("read bytes: {ok}")
    }
    line
}

fn main() {
    let line = get_line();
    println!("get line: {line}");
}
```

输入：

```
hello, world
```

输出：

```
hello, world
read bytes: 14
get line: hello, world

```

可以看到，`read_line()`不会删除换行符，在 windows 上多占了`\r\n`两个字符，长度为 14。

按引用传递参数：

```rust
use std::io;

fn get_line(line: &mut String) {
    let result = io::stdin().read_line(line);
    match result {
        Err(err) => println!("error meg: {err}"),
        Ok(ok) => println!("read bytes: {ok}")
    }
}

fn main() {
    let mut line = String::new();
    get_line(&mut line);
    println!("get line: {line}");
}
```

先传递进去，再返回回来：

```rust
use std::io;

fn get_line(mut line: String) -> String {
    let result = io::stdin().read_line(&mut line);
    match result {
        Err(err) => println!("error meg: {err}"),
        Ok(ok) => println!("read bytes: {ok}")
    }
    line
}

fn main() {
    let mut line = String::new();
    line = get_line(line);
    println!("get line: {line}");
}
```

## 智能指针

### Box

`Box`是智能指针，用于在堆上分配内存。智能指针不同于结构体的地方在于其实现了`Deref`和`Drop` trait。

`String`和`Vec<T>`都是智能指针。

常用的智能指针：

* `Box<T>`：用于在堆上分配值
* `Rc<T>`：一个引用计数类型，其数据可以有多个所有者
* `Ref<T>`和`RefMut<T>`，通过`RefCell<T>`访问。（`RefCell<T>`是一个在运行时而不是在编译时执行借用规则的类型）。

在堆上分配内存：

```rust
let p = Box::new(5);
println!("{}", p);
```

有些类型里需要指向自己的指针：

```rust
use crate::List::*;
enum List {
    Con(i32, Box<List>),
    Nil,
}

fn main() {
    let l = Con(3, Box::new(Con(5, Box::new(Nil))));
}
```

`Deref` trait 可以将智能指针当作常规引用处理。

```cpp
use std::ops::Deref;
struct MyBox<T>(T);
impl<T> MyBox<T> {
    fn new(t: T) -> MyBox<T> {
        MyBox(t)
    }
}

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn main() {
    let a = MyBox::new(3);
    println!("{}", *a);  // *a 实际调用了 *(a.deref())，不过这个知不知道都无所谓
}
```

Rust 在发现类型和 trait 实现满足三种情况时会进行 Deref 强制转换：

* 当 `T: Deref<Target=U>` 时从 `&T` 到 `&U`。
* 当 `T: DerefMut<Target=U>` 时从 `&mut T` 到 `&mut U`。
* 当 `T: Deref<Target=U>` 时从 `&mut T` 到 `&U`。

Rust 也会将可变引用强转为不可变引用。但是反之是 不可能 的：不可变引用永远也不能强转为可变引用。

### `Rc<T>`

```rust
use std::rc::Rc;

let five = Rc::new(5);

let _ = Rc::clone(&five);
```

常用方法：

* `pub fn new(value: T) -> Rc<T>`

    创建一个新 Rc。

* `pub fn pin(value: T) -> Pin<Rc<T>>`

    Constructs a new Pin<Rc<T>>. If T does not implement Unpin, then value will be pinned in memory and unable to be moved.

* `pub fn into_raw(this: Rc<T>) -> *const T`

    Consumes the Rc, returning the wrapped pointer.

    To avoid a memory leak the pointer must be converted back to an Rc using Rc::from_raw.

* `pub fn as_ptr(this: &Rc<T>) -> *const T`

    Provides a raw pointer to the data.

    The counts are not affected in any way and the Rc is not consumed. The pointer is valid for as long there are strong counts in the Rc.

* `pub fn downgrade(this: &Rc<T>) -> Weak<T>`

    Creates a new Weak pointer to this allocation.

    Example:

    ```rust
    use std::rc::Rc;

    let five = Rc::new(5);

    let weak_five = Rc::downgrade(&five);
    ```

* `pub fn weak_count(this: &Rc<T>) -> usize`

    Gets the number of Weak pointers to this allocation.

* `pub fn strong_count(this: &Rc<T>) -> usize`

    Gets the number of strong (Rc) pointers to this allocation.

* `pub fn get_mut(this: &mut Rc<T>) -> Option<&mut T>`

    Returns a mutable reference into the given Rc, if there are no other Rc or Weak pointers to the same allocation.

    Returns None otherwise, because it is not safe to mutate a shared value.

    Example:

    ```rust
    use std::rc::Rc;

    let mut x = Rc::new(3);
    *Rc::get_mut(&mut x).unwrap() = 4;
    assert_eq!(*x, 4);

    let _y = Rc::clone(&x);
    assert!(Rc::get_mut(&mut x).is_none());
    ```

* `fn as_ref(&self) -> &T`

    Converts this type into a shared reference of the (usually inferred) input type.

* `fn borrow(&self) -> &T`

    Immutably borrows from an owned value.

* `fn clone(&self) -> Rc<T>`

    Makes a clone of the Rc pointer.

    This creates another pointer to the same allocation, increasing the strong reference count.

    Example:

    ```rust
    use std::rc::Rc;

    let five = Rc::new(5);

    let _ = Rc::clone(&five);
    ```

* `fn borrow_mut(&mut self) -> &mut T`

    Mutably borrows from an owned value.

### `RefCell<T>`

不同于 Rc<T>，RefCell<T> 代表其数据的唯一的所有权。

对于引用和 Box<T>，借用规则的不可变性作用于编译时。对于 RefCell<T>，这些不可变性作用于 运行时。对于引用，如果违反这些规则，会得到一个编译错误。而对于 RefCell<T>，如果违反这些规则程序会 panic 并退出。

如下为选择 Box<T>，Rc<T> 或 RefCell<T> 的理由：

`Rc<T>`允许相同数据有多个所有者；`Box<T>`和`RefCell<T>`有单一所有者。
`Box<T>`允许在编译时执行不可变或可变借用检查；`Rc<T>`仅允许在编译时执行不可变借用检查；`RefCell<T>`允许在运行时执行不可变或可变借用检查。
因为`RefCell<T>`允许在运行时执行可变借用检查，所以我们可以在即便`RefCell<T>`自身是不可变的情况下修改其内部的值。

`RefCell<T>`可以通过调用`borrow_mut()`，把不可变引用变成可变引用。

当创建不可变和可变引用时，我们分别使用`&`和`&mut`语法。对于 `RefCell<T>` 来说，则是`borrow`和`borrow_mut`方法，这属于`RefCell<T>`安全 API 的一部分。`borrow`方法返回`Ref<T>`类型的智能指针，`borrow_mut`方法返回 `RefMut<T>`类型的智能指针。这两个类型都实现了`Deref`，所以可以当作常规引用对待。

`RefCell<T>`记录当前有多少个活动的`Ref<T>`和`RefMut<T>`智能指针。每次调用`borrow`，`RefCell<T>`将活动的不可变借用计数加一。当`Ref<T>`值离开作用域时，不可变借用计数减一。就像编译时借用规则一样，`RefCell<T>`在任何时候只允许有多个不可变借用或一个可变借用。

如果我们尝试违反这些规则，相比引用时的编译时错误，`RefCell<T>`的实现会在运行时出现 panic：

```rust
let mystr = String::from("hello");
let rc = RefCell::new(mystr);
let b = rc.borrow_mut();
let c = rc.borrow_mut();
```

上面的几行代码可以通过编译，但是一旦运行会直接 panic。

但是这样写就没问题：

```rust
let mystr = String::from("hello");
let rc = RefCell::new(mystr);
rc.borrow_mut();
rc.borrow_mut();
println!("{}", rc.borrow_mut());
println!("{}", rc.borrow_mut());
```

### Cell

`Cell`比较像一个只包含了一个元素的容器。在写`struct`时，可以把成员设置成`Cell`类型的变量，可以通过对`Cell`进行插入，删除，替换等操作，间接地对里面的元素进行修改。

```rust
use std::cell::*;

struct Msg {
    greeting: String,
    person: Cell<String>
}

impl Msg {
    fn print(&self) {
        println!("{}, {}", self.greeting, self.person.take());  // 这里的 take() 会把 Cell 里面的东西拿出来，然后在原地放一个 None
    }
}

fn main() {
    let obj = Msg {  // 注意 obj 不是 mut 对象，但是后面我们仍可以对其中的 Cell 成员进行修改
        greeting: String::from("hello"),
        person: Cell::from(String::from("abc")),
    };
    obj.print();
    obj.person.set(String::from("xyz"));  // set() 可以往 Cell 里放置一个变量
    obj.print();
}
```

输出：

```
hello, abc
hello, xyz
```

## iterator

有关 for 内赋值的问题：

```rust
fn main() {
    let array: [i32; 3] = [3, 2, 1];
    let mut iter = array.iter();
    let mut next = iter.next();
    while next != None {
        print!("{} ", next.unwrap());
        next = iter.next();
    }
}
```

上面这段代码是没问题的。但假如现在只有一个 unmutable 的`iter`，该如何在 for 内对`next`赋值呢？因为需要对`next`进行修改，所以`next`必须被设置成`mut`，但是我们只有不可修改的`iter`，所以`iter.next()`也是 unmutable 的，无法赋值给`next`。这样就有了矛盾，不知道该怎么解决。

## tests

可以给一个函数添加`test`属性将这个函数标记为测试函数：

```rust
#[test]
fn my_test() {
    assert_eq!(3, 3)
}

fn main() {

}
```

然后使用`cargo test`调用程序中的所有测试函数。每一个测试函数都在一个单独的子线程中执行。

在测试函数中通常使用这几个宏：

```rust
assert!()
assert_eq!()
assert_ne!()

assert!(&bool, &str, params)  // 带有错误提示信息的 assert!
panic!(&str, params)  // 打印错误信息的 panic!
```

`assert_eq!`和`assert_ne!`宏在底层分别使用了`==`和`!=`。当断言失败时，这些宏会使用调试格式打印出其参数，这意味着被比较的值必需实现了`PartialEq`和`Debug trait`。通常可以直接在自定义的枚举或结构体上加上`#[derive(PartialEq, Debug)]`注解。

如果已经能确定某个测试一定会出现`panic!`，可以使用`#[should_panic]`标签：

```rust
#[test]
#[should_panic]
fn my_test() {
    assert_eq!(2 + 3, 4)
}
```

此时，如果函数运行没有错误，那么反而会`panic!`。

还可以让一个测试函数返回`Result<T, E>`类型：

```rust
#[test]
fn my_test() -> Result<(), String> {
    if 2 + 2 == 4 {
        Ok(())
    } else {
        Err(String::from("Something is wrong in my_test()."))
    }
}
```

然后使用`assert!(value.is_err())`判断这个函数的返回值。当函数返回`Result<T, E>`类型时，不能对函数使用`#[should_panic]`注解。

`cargo test`生成的二进制文件的默认行为是并行的运行所有测试，并截获测试运行过程中产生的输出，阻止他们被显示出来，使得阅读测试结果相关的内容变得更容易。

可以将一部分命令行参数传递给`cargo test`，而将另外一部分传递给生成的测试二进制文件。为了分隔这两种参数，需要先列出传递给`cargo test`的参数，接着是分隔符`--`，再之后是传递给测试二进制文件的参数。运行`cargo test --help`会提示`cargo test`的有关参数，而运行`cargo test -- --help`可以提示在分隔符`--`之后使用的有关参数。

可以使用`cargo test -- --test-threads=1`指定并行测试的线程数量。

默认情况下，当测试通过时，Rust 的测试库会截获打印到标准输出的所有内容。比如在测试中调用了`println!`而测试通过了，我们将不会在终端看到`println!`的输出：只会看到说明测试通过的提示行。如果测试失败了，则会看到所有标准输出和其他错误信息。

可以使用`cargo test -- --show-output`显示所有输出。

可以使用`cargo test <pattern>`指定要测试的函数，所有函数名包含了`<pattern>`函数都会被测试。注意，`mod`的名称也是函数名的一部分，因此可以把某个 mod 名指定为`<pattern>`来测试整个 mod 中的函数。

如果不想执行某些测试，只能在代码为函数加上`#[ignore]`。如果只希望运行被忽略的测试，可以使用`cargo test -- --ignored`。如果希望不管是否忽略都要运行全部测试，可以运行`cargo test -- --include-ignored`。

**单元测试（unit tests）**

单元测试主要测试私有接口。即自己测试自己写的代码。

单元测试与他们要测试的代码共同存放在位于`src`目录下相同的文件中。规范是在每个文件中创建包含测试函数的`tests`模块，并使用`cfg(test)`标注模块。`#[cfg(test)]`注解告诉 Rust 只在执行`cargo test`时才编译和运行测试代码，而在运行`cargo build`时不这么做。

file: `src/lib.rs`

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
```

**集成测试（integration tests）**

集成测试主要是测试别人的库。

首先需要在项目根目录下创建一个`tests`文件夹，然后再在这个文件夹中创建一些文件进行测试，每个文件都会被编译成一个 crate。

file: `tests/integration_test.rs`

```rust
use adder;

#[test]
fn it_adds_two() {
    assert_eq!(4, adder::add_two(2));
}
```

Cargo 只会在运行`cargo test`时编译这个目录中的文件。

在集成测试中，仍然可以使用`cargo test <pattern>`执行特定的测试，也可以使用`cargo test --test <file_name>`执行某个指定文件中的测试。

`tests`目录中的子目录不会被作为单独的`crate`编译或作为一个测试结果部分出现在测试输出中。因此可以创建一些文件夹，在文件夹中创建一些 module，作为其他测试函数的公共调用部分。

`String`实现了`Deref<Target = str>`，并且继承了`str`的所有方法，因此当函数将`&str`作为参数时，`String`可以自动转换为`&str`。 

## 多线程

```rust
use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from the spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    for i in 1..5 {
        println!("hi number {} from the main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap();
}
```

自动判断所有权：

```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];

    let handle = thread::spawn(|| {
        println!("Here's a vector: {:?}", v);  // 自动推断出需要 borrow v，但是并不知道 v 在外面存活多长时间，所以无法通过编译
    });

    handle.join().unwrap();
}
```

指定 move 所有权：

```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];

    let handle = thread::spawn(move || {
        println!("Here's a vector: {:?}", v);
    });

    handle.join().unwrap();
}
```

线程间的消息传递：

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
    });

    let received = rx.recv().unwrap();
    println!("Got: {}", received);
}
```

如果需要非阻塞地接收消息，可以使用`try_recv`。

经过信道发送的值会发生move操作：

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
        println!("val is {}", val);  // Error，val 已经被 move，不能再使用
    });

    let received = rx.recv().unwrap();
    println!("Got: {}", received);
}
```

将`rx`作为迭代器接收消息：

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }
}
```

通过`clone`使得多个线程同时发送消息：

```rust
    // --snip--

    let (tx, rx) = mpsc::channel();

    let tx1 = tx.clone();
    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx1.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    thread::spawn(move || {
        let vals = vec![
            String::from("more"),
            String::from("messages"),
            String::from("for"),
            String::from("you"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }

    // --snip--
```

使用共享状态在线程间通信：

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));  // Arc 支持原子操作，防止在修改引用计数时发生数据竞争。使用 Mutex 保证在线程中不发生竞争。如果不使用 Arc，那么 Mutex 会发生 move。
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();

            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

## 其他

1. 注释

    rust 的注释使用`//`

1. `rustup`的代理设置

    参考这个网址：<https://rust-lang.github.io/rustup/network-proxies.html>

    windows 中的系统代理和手动代理对`rustup`都没有作用。

1. vscode 中`rust-analyzer`的正常运行需要安装`rust-src`

    安装方法：

    `rustup update`

    `rustup component add rust-src`

1. 使用 deubg 模式打印结构体信息

    ```rust
    #[derive(Debug)]
    struct Rectangle {
        width: u32,
        height: u32
    }

    fn main() {
        let rect1 = Rectangle {
            width: 30,
            height: 50
        };

        println!("rect1 is {:?}", rect1);
        // 也可以把 {:?} 换成 {:#?}，这样会自动换行
    }
    ```

1. `dbg!()`

    `dbg!`宏可以把信息输出到`stderr`。它接受一个表达式的所有权，并返回该值的所有权。

    ```rust
    #[derive(Debug)]
    struct Rectangle {
        width: u32,
        height: u32
    }

    fn main() {
        let scale = 2;
        let rect1 = Rectangle {
            width: dbg!(30 * scale),
            height: 50
        };

        dbg!(&rect1);
    }
    ```

1. 自动引用和解引用

    ```rust
    p1.distance(&p2);
    (&p1).distance(&p2);
    ```

    上面这两行代码是等价的。

1. 设置在 panic 时终止

    ```toml
    [profile.release]
    panic = 'abort'
    ```

    调用 panic:

    ```rust
    fn main() {
        panic!("crash and burn");
    }
    ```

1. 使用 rust 刷题时，记得整数变量声明成类型`i32`，然后在索引时变成`usize`：`a as usize`。

    如果直接声明成`usize`，那么整数的运算对减法不封闭，可能会出现`2 - 3`这样的溢出 panic，或者直接得到回绕的值。

    如果都声明成`i32`，那么没有办法直接索引，因为索引只能使用`usize`。

1. 一行定义多个变量：

    ```rust
    let (mut a, mut b): (i32, i32) = (1, 2);
    ```

    `mut`每次只能修饰一个变量，不能修饰多个。

1. How to pretty-print a Rust HashMap in GDB?

    <https://stackoverflow.com/questions/50179667/how-do-i-pretty-print-a-rust-hashmap-in-gdb>

1. 无论运算符两侧是否是引用，运算完的结果都是一个值

1. 容器 insert 或 push 进去的，都必须是值，不能是引用

1. `type`

    Define an alias for an existing type.

    The syntax is `type Name = ExistingType;`.

1. 将一个`Range`转换成一个`Vec`

    ```rust
    let b = (0..5).collect::<Vec<i32>>();
    ```

    注意`collect`后面写的类型是容器，而不是`i32`。

## Appended

1. 闭包（closure）

    ```rust

    fn get_increased_val(init_val: u32) -> impl FnMut() -> u32 {
        let mut count = init_val;
        move || -> u32 {
            count += 1;
            count
        }
    }

    fn main() {
        let mut get_val = get_increased_val(0);
        println!("{}", get_val());
        println!("{}", get_val());
    }
    ```

1. `String`

1. cargo 换源

1. 为什么`&"hello, world"`和`"hello, world"`以及`&&"hello, world"`指的都是`&str`？

1. 猜数字中的一些小问题

    ```rust
    use std::io::{self, Write};
    fn main() {
        let target = 35;
        let mut guess: i32 = 0;
        let mut buf = String::new();
        while guess != target {
            print!("input the guess number: ");  //　没有 \n　时，会先缓存不刷新，因此下面需要强制刷新
            io::stdout().flush().unwrap();
            buf.clear();  //　不清空时，字符串找不到 \0，后面的解析会出问题
            io::stdin().read_line(&mut buf).unwrap();
            let line_str = buf.trim();
            guess = i32::from_str_radix(line_str, 10).unwrap();  // 好像也可以使用 line_str.parse()
            if guess > target {
                println!("what you guessed is bigger than the secert number.");
            } else if guess < target {
                println!("what you guessed is smaller than the secret number.");
            }
        }
        println!("correct number. good job.");
    }
    ```

1. 整数运算

    不允许`i32`和`i64`之间互相隐式转换。也就是说，对于`i32`来说，算术运算只能发生在两个`i32`之间。如果是一个`i32`加一个`i64`，那么会报错。

    通过溢出报错，加上非相同类型不能运算，基本就避免了各种溢出的问题。但是程序员会累一些，需要处理溢出异常和类型转换。

1. 配置代理 proxy

1. 有关 rustup 的教程

    Ref: <https://rust-lang.github.io/rustup/index.html>

1. cargo 只下载，不编译

    `cargo fetch`

1. `std::fs`模块

    与文件操作相关的库在`std::fs`模块中。这个模块需要和`std::io`中的 trait 配合使用。

    引入：

    ```rust
    use std::fs::File;
    ```

    打开一个文件：

    ```rust
    fn main() -> std::io::Result<()> {
        // open a file with write mode. if the file existed, the existed file will be covered.
        let mut file = File::create("aaa.txt")?;  

        // open a file with read-only mode
        let mut file2 = File::open("bbb.txt")?;  

        // open a file with append mode
        let mut file3 = File::options().append(true).open("ccc.txt")?;  

        // open a file with binary mode
    }
    ```

    读文本文件：

    * 一次性把所有内容读完

        ```rust
        use std::{fs::File, io::Read};  // io::Read 是必须的，read_to_string() 函数要用到这个

        fn main() -> std::io::Result<()> {
            let file_path = "/home/hlc/Documents/Projects/rust_test/hello.txt";
            let mut file = File::open(file_path)?;
            let mut lines = String::new();
            file.read_to_string(&mut lines)?;
            println!("{}", lines);
            Ok(())
        }
        ```

        由于`read_to_string`接受的参数是`&mut String`，所以`file`必须被指定为`mut`。但是如果我们从别的函数中只拿到了非`mut`的`file`，那么似乎就没办法调用`read_to_string()`了，即使只想看看文件内容，不想做任何更改也不行。这种情况该怎么办呢？

    * 一行一行地读

        <https://stackoverflow.com/questions/45882329/read-large-files-line-by-line-in-rust>

        ```rust
        use std::fs::File;
        use std::io::{self, prelude::*, BufReader};

        fn main() -> io::Result<()> {
            let file = File::open("foo.txt")?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                println!("{}", line?);
            }

            Ok(())
        }
        ```

        不清楚`BufReader()`会申请多大的 buffer，是固定的还是动态分配内存的？ 

    * 一次读指定字节数

        ```rust

        ```

    常用 methods:

1. `std::io`

    `std::io`模块为标准输入输出，文件读写提供了统一的接口。

    常用的 trait：

    * `impl Read for File`

        * `fn read(&mut self, buf: &mut [u8]) -> Result<usize>`

            把 buffer 填满，或者读到文件/输入的结尾

        * `fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize>`

            由`Vec<u8>`负责管理内存，`read_to_end()`把所有内容读到`buf`里。

            ```rust
            let mut buf = Vec::new();
            let mut n = file.read_to_end(&mut buf).unwrap();
            ```

        * `fn read_to_string(&mut self, buf: &mut String) -> Result<usize>`

            由`String`负责管理内存，`read_to_string()`把所有内容读到`buf`里。

        * `fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>`

            不清楚这个和`read()`有啥区别

        * `fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> Result<usize>`

            依次往不同 buffers 中写数据

            Example:

            ```rust
            use std::{fs::File, io::Read};
            use std::io::IoSliceMut;

            fn main() -> std::io::Result<()> {
                let file_path = "example.txt";
                let mut file = File::open(file_path)?;
                let mut lines = String::new();
                let mut buf1: [u8; 15] = [0; 15];
                let mut buf2: [u8; 10] = [0; 10];
                let mut bufs = &mut [
                    IoSliceMut::new(&mut buf1),
                    IoSliceMut::new(&mut buf2),
                ][..];

                let mut n = file.read_vectored(&mut bufs).unwrap();

                println!("Succeed to read {} bytes.", n);
                Ok(())
            }
            ```

1. convert a string to a vector of chars

    ```rust
    fn main() {
        let s = "Hello world!";
        let char_vec: Vec<char> = s.chars().collect();  // or collect::<Vec<char>>()
        for c in char_vec {
            println!("{}", c);
        }
    }
    ```

1. convert a vector to an array

    这个其实不好实现，因为 array 必须要初始化，这样的话我们就需要知道 array 有几个元素。但是 vector 是动态分配内存的，我们不知道有几个元素。因此将`Vec`转换成`[u8]`比较困难。

    目前只简单找了一种方法。有时间的话，再查查其他方法吧。

    需要复制的话，可以使用这个方法：

    ```rust
    fn demo<T>(v: Vec<T>) -> [T; 32] where T: Copy {
        let slice = v.as_slice();
        let array: [T; 32] = match slice.try_into() {
            Ok(ba) => ba,
            Err(_) => panic!("Expected a Vec of length {} but it was {}", 32, v.len()),
        };
        array
    }
    ```

1. convert `&[u8]` to `String`

    ```rust
    std::str::from_utf8(byte_array).unwrap().to_string();
    ```

    也可以直接调用`String`的`from_utf8()`：

    ```rust
    let str = String::from_utf8([b'a', b'b', b'c'].to_vec()).unwrap();
    ```

    字符串转换大全：<https://gist.github.com/jimmychu0807/9a89355e642afad0d2aeda52e6ad2424>

1. convert `[char; N]` to `String`

    ```rust
    let arr = ['h', 'e', 'l', 'l', 'o'];
    let mut str: String = arr.iter().collect();
    ```
    
    或者可以`String::from_iter()`

1. substring

    有一个`substring` crate。也可以使用`.get()`方法。

1. 按行读取文件

    <https://doc.rust-lang.org/rust-by-example/std_misc/file/read_lines.html>

1. reverse a string

    ```rust
    fn main() {
        let foo = "palimpsest";
        println!("{}", foo.chars().rev().collect::<String>());
    }
    ```

1. 有关`rand` crate

    正常情况下，需要

    ```rust
    use rand::{self, Rng};
    ```

    才能调用`rand`中的函数：

    ```rust
    fn main()　{
        let mut rng = rand::thread_rng();
    }
    ```

    但是事实是只需要

    ```rust
    use rand::Rng;
    ```

    就可以调用`rand`中的函数了。

    为什么？

1. 给程序传递参数

    ```rust
    use std::env;

    fn main() {
        let args: Vec<String> = env::args().collect();
        let program_path: &String = &args[0];
        let first_arg: &String = &args[1];
    }
    ```

    可以用`env::args().len()`判断参数的数量。

    这里得到的`program_path`指的是 bash 是以什么样的路径运行程序的。因此这个变量既有可能是绝对路径，也有可能是相对路径。

    如果我们使用`cargo run`运行程序，那么得到的是相对路径，通常为`target/debug/<program-name>`

1. 字符串拼接

    ```rust
    let mut s1 = "Hello,".to_string();
    let s2 = "world".to_string();
    s1 += &s2;  // 运算符似乎也会对所有权有影响，所以 s2 需要写引用

    let s = "Hello," + "world"; // Can't use + with two &str
    let s = s1 + &s2; // Move s1 to s, and concats s2 to s
    ```

    ```rust
    let s1 = String::from("Hello"); let s2 = String::from("world."); let s = format!("{}, {}", s1, s2);
    ```

    Ref: <http://www.codebaoku.com/it-rust/it-rust-string-concat.html>

    有时间了看看这个网站，还有挺多的技巧。

1. 在给`pringln!`传递参数时，会自动把参数变成引用，因此不会发生 move 操作。

1. `Vec<String>`和`Vec<&String>`以及`Vec<&str>`有什么异同？

1. `unwrap()`也会发生 move 操作

1. `println!()`不会发生 move 操作。

    ```rust
    let mut mystr = String::from("hello");
    println!("{}", mystr);
    println!("{}", mystr);
    ```

    输出：

    ```
    hello
    hello
    ```

1. 使用`Box<>`可以拿到对象的成员，此时成员如果没有实现`Copy` trait，那么会发生 move。使用 ref （比如`&`和`&mut`）只能拿到对象成员的 ref，如果强行拿，会先看成员有没有`Copy` trait，如果没有的话，会发生 move。然而 ref 不允许产生 move 语义，因此会无法通过编译。

    如果是`& Box<>`或`&mut Box<>`，那么它其实只代表了`Box<>`，因为编译器会自动解引用。（即使自动解引用，我们也只能拿到 Box 所指对象或对象的成员的引用）

    我们也无法通过解引用`*`拿到一个 ref 对应的对象。

1. `Box`的`as_mut()`和`as_ref()`也都是返回对象的 ref。

1. 可以从 struct 中 move 出一个成员，但是很难从其他容器中 move 出一个成员，包括 Option 和 Vec 等。

    最常见的做法是拿出他们的引用。如果一定要拿出来，可以使用`take()`，`replace()`，`insert()`，`swap()`等相关的内存操作。

    即使是`i32`，`char`这些对象，我们也可以把它看作是一个 coontainer。

1. `Box`被解引用或者被赋值给其他变量后，会发生 move 操作。

    如果我们通过`Box`拿到内容的引用，那么可以替换`Box`所指向的内容：
    
    ```rust
    let mut s1 = String::from("hello");
    let mut s2 = String::from("world");
    let mut r = Box::new(s1);
    let a = &mut *r;  // 这样不会发生 move。但是 let a = *r; 会发生 move。
    replace(a, s2);
    println!("{}", r);
    ```

    输出：

    ```
    world
    ```

1. 假如往智能指针里放一个变量，那么变量会发生 move。假如往智能指针里放一个引用，那么智能指针其实和引用也就没什么区别了。

1. 那么无法通过解引用`*`从`Rc`中拿到内容的 ownership，也无法拿到内容的`&mut`。对于其他情况，`Rc`几乎和内容对象完全相同。

    可以使用`borrow()`和`borrow_mut()`拿到内容的引用。

    如果内容是`RefCell`，那么拿到的引用是`Ref`或`RefMut`

1. 可以使用`clone()`将`Rc`指向另外一个对象

1. `RefMut`有个`replace()`方法，可以替换掉里面的内容

    如果我们有一个`Rc<RefMut<String>>`的对象，想把里面的`String`替换掉，那么可以使用`r.replace()`（假设`r`就是那个`Rc`对象）。`r`会自动解引用，得到`RefMut`，然后我们调用`RefMut`的`replace()`替换里面的值。

    `r.get_mut()`可以返回`&mut String`，也可以用这个加上`std::mem::replace()`替换里面的值。

    Example:

    ```rust
    fn my_test() {
        let mut str1 = String::from("hello");
        let mut str2 = String::from("world");
        let mut r = Rc::new(RefCell::new(str1));
        let a: Ref<String> = r.borrow();
        let a: RefMut<String> = r.borrow_mut();
        let a: &mut String = r.get_mut();
        let a: &RefCell<String> = r.as_ref();
    }
    ```

1. 当把具体的引用传给 longest 时，那生命周期 'a 的大小就是 x 和 y 的作用域的重合部分，换句话说，'a 的大小将等于 x 和 y 中较小的那个。由于返回值的生命周期也被标记为 'a，因此返回值的生命周期也是 x 和 y 中作用域较小的那个。

    生命周期语法用来将函数的多个引用参数和返回值的作用域关联到一起，一旦关联到一起后，Rust 就拥有充分的信息来确保我们的操作是内存安全的

    构体引用的字符串活得比结构体久

1. 将`&str`转换成`String`

    ```rust
    fn alphabet_position(text: &str) -> String {
        let s = text
            .chars()
            .into_iter()
            .filter(|&c| c.is_alphabetic())
            .map(|c| c.to_ascii_uppercase())
            .map(|c| c as u8)
            .map(|c| (c - 64u8).to_string())
            .collect();
        s
    }
    ```

1. 将`char`转换为`u8`

1. How to clear the Cargo cache?

    <https://stackoverflow.com/questions/25072930/how-to-clear-the-cargo-cache>

1. cargo 清除 cache

    <https://stackoverflow.com/questions/25072930/how-to-clear-the-cargo-cache>
