# Rust Note

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

## Installation

网站：<https://rustup.rs>，这个网站的安装包似乎会把 rust 直接装到 c 盘。

工具：

* `cargo`：编译管理器，包管理器，通用工具。
* `rustc`：rust 的编译器。
* `rustdoc`：rust 文档工具。

## 项目管理

创建一个 project：`cargo new --bin hello`。`--bin`表示这个项目是一个可执行文件，而不是一个库。

`Cargo.toml`保存这个项目的元数据。

cargo 还创建了个`.git`和`.gitignore`，如果不需要这么做，可以加上`--cvs none`。

在这个包的任何目录都可以调用`cargo run`，构建并运行整个项目。

`cargo clean`可以清理生成的文件。

在本地浏览器中查看标准库文档：`rustup doc --std`

## 基本语法

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

如果函数体中最后一行代码是一个表达式，且表达式末尾没有分号，那么这个表达式的值就是函数的返回值。`return`语句一般只用于在函数的中间提前返回。

实际上，用花括号括起来的任何代码块者可以看作一个表达式：

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

## 基本类型

rust 是静态类型的语言，即无须实际运行程序，编译器就可以检查所有可能的执行路径，确保程序以与类型一致的方式使用每一个值。

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

* 整数类型

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

    整数也可以有方法：

    ```rust
    assert_eq!(2u16.pow(4), 16);
    assert_eq!((-4i32).abs(), 4);
    assert_eq!(0b101101u8.count_ones(), 4);
    ```

    如果根据上下文能够推断出类型，那么就不需要指定类型。但是上面几个例子无法确定出类型，所以需要后缀来确定类型。

* 浮点类型

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

    rust 几乎不做隐式类型转换。如果它抢断不出来到底是`f32`还是`f64`，就直接放弃了。

* 布尔类型

    `bool`类型有两个值：`true`和`false`。rust 中不允许除了`bool`类型以外的其它类型值作为`if`，`while`等语句的条件。

    可以使用`as`操作符把`bool`值转换为整数类型：

    ```rust
    assert_eq!(false as i32, 0);
    assert_eq!(true as i32, 1);
    ```

    但是`as`不能将整数转换成布尔值。

* 字符类型

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

### 元组（tuple）

元组是用圆括号括起来的几个值，比如`("Brazil", 1985)`，它的类型是`(&str, i32)`。可以用`t.0`，`t.1`访问元组`t`的元素。

元组中元素的类型可以不同，而数组则要求相同类型。元组只允许用常量作为索引，比如`t.4`，而不能是变量，比如`t.i`或`t[i]`。

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

零元组`()`又称为基元类型（unit type）。没有返回值的函数的返回类型就是`()`。

### 指针类型

* 引用

* Box

* 原始指针

* 数组、向量和切片

* 数组

* 向量

* 逐个元素地构建向量

* 切片

### 字符串类型

## 所有权

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

## 引用

## 表达式

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

## 错误处理

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

Cargo 会从 github 上下载这些包的源代码，然后对每个包运行一次`rustc`，并加上`--crate-type lib`参数，生成一个`.rlib`文件。在编译主程序时，`rustc`会加上`--crate-type bin`参数，生成一个可执行的二进制文件。

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

### 模块