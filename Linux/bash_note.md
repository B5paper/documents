# Bash Note

Reference: <https://www.computerhope.com/unix.htm>

## cache

* bash 中常见信号与触发条件

    | 信号名称 | 编号 |	触发条件 |
    | - | - | - |
    | `INT` | 2 | Ctrl + C 中断 |
    | `TERM` | 15 | 默认的 kill 命令 |
    | `EXIT` | 0 | 脚本退出时（非真实信号） |
    | `ERR` | - | 命令执行失败时（非真实信号） |
    | `DEBUG` | - | 每条命令执行后（非真实信号） |

    example:

    `main.sh`：

    ```bash
    trap "echo INT triggered!; exit 1" INT
    trap "echo TERM triggered; exit 1" TERM
    trap "echo EXIT triggered; exit 1" EXIT
    trap "echo ERR triggered; exit 1" ERR
    trap "echo DEBUG triggered" DEBUG

    while true; do
        echo "current time: $(date)"
        sleep 1
    done
    ```

    run: `bash main.sh`

    使用不同的方式触发 signal 后，output 如下：

    * `Ctrl` + `C`

        ```
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:41:38 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:41:39 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:41:40 CST
        DEBUG triggered
        ^CDEBUG triggered
        INT triggered!
        DEBUG triggered
        DEBUG triggered
        EXIT triggered
        DEBUG triggered
        ```

    * kill

        ```
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:44:15 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:44:16 CST
        DEBUG triggered
        DEBUG triggered
        DEBUG triggered
        current time: 2025年 07月 01日 星期二 15:44:17 CST
        DEBUG triggered
        DEBUG triggered
        TERM triggered
        DEBUG triggered
        DEBUG triggered
        EXIT triggered
        DEBUG triggered
        ```

    所有的信号对大小写不敏感，即`INT`和`int`是等价的，其他的同理。

* bash 中的 trap 可以让用户指定函数或命令去处理 signal 信号

    example:

    `main.sh`:

    ```bash
    trap "echo manully terminated!; exit 1" INT

    while true; do
        echo "current time: $(date)"
        sleep 1
    done
    ```

    执行：`bash ./main.sh`，等待几秒后，按`Ctrl` + `C`。

    output:

    ```
    current time: 2025年 06月 30日 星期一 13:03:09 CST
    current time: 2025年 06月 30日 星期一 13:03:10 CST
    current time: 2025年 06月 30日 星期一 13:03:11 CST
    current time: 2025年 06月 30日 星期一 13:03:12 CST
    ^Cmanully terminated!
    ```

* `set -e`: 任何命令返回非零（失败）状态时，立即退出脚本

    可以通过`set +e`关闭这一行为。

    example:

    ```bash
    set -e
    cd haha
    echo hello
    ```

    (`haha`文件夹不存在)

    output:

    ```
    main.sh: line 2: cd: haha: No such file or directory
    ```

    退出 bash 脚本后，`echo $?`的值为`1`。

    ```bash
    # set -e
    cd haha
    echo hello
    ```

    output:

    ```
    main.sh: line 2: cd: haha: No such file or directory
    hello
    ```

    `set -e`等价于`set -o errexit`。

* `set -u`: 遇到未定义的变量时，报错并退出（防止误用空变量）

    example:

    ```bash
    set -u
    echo "$my_var"
    echo "hello"
    ```

    output:

    ```
    main.sh: line 2: my_var: unbound variable
    ```

    ```bash
    # set -u
    echo "$my_var"
    echo "hello"
    ```

    output:

    ```

    hello
    ```

* `set -o pipefail`

    管道命令`|`中任意一个子命令失败时，整个管道返回非零状态。
    
    example:

    `main.sh`:

    ```bash
    set -o pipefail
    cd haha | echo "hello"
    echo $?
    ```

    run: `bash main.sh`

    output:

    ```
    main.sh: line 2: cd: haha: No such file or directory
    hello
    1
    ```

    如果不设置`pipefail`，则只返回最后一个命令的状态：

    ```bash
    # set -o pipefail
    cd haha | echo "hello"
    echo $?
    ```

    run: `bash main.sh`

    output:

    ```
    hello
    main.sh: line 2: cd: haha: No such file or directory
    0
    ```

    注意，`set -o pipefail`只改变了管道命令的返回值，并不会使 bash 脚本退出。

    `set +o pipefail`可以关闭这个参数。

* bash array 使用小括号来定义：`arr=(elm_1 elm_2 elm_3)`

* bash array 使用`[]`作为下标，并从 0 开始索引，`${arr[0]}`, `${arr[1]}`

* 当使用`@`或`*`作为索引时，会索引数组中的所有元素：`${arr[@]}`, `${arr[*]}`

* 使用`${#arr[@]}`或`${#arr[*]}`可以获得数组的长度

* bash 中打印数组中的字符串，每个一行

    ```bash
    arr=(hello world nihao zaijian)
    arr_len=${#arr[@]}
    i=0
    while [ $i -lt $arr_len ] ; do
        echo ${arr[i]}
        i=$((i+1))
    done
    ```

* bash escape single quote

    下面是 bash 中使用单引号组成的字符串的 example 和解释：

    ```bash
    echo 'hello there'
    echo 'hello
       there'

    echo 'hello 'there'
      aaa'
    echo 'hello \'there'
      bbb'

    echo $'hello 'there'
       ccc'
    echo $'hello \'there'
    ```

    output:

    ```
    hello there
    hello
      there
    hello there
      aaa
    hello \there
      bbb
    hello there
      ccc
    hello 'there
    ```

    说明：

    1. 使用单引号`'`括起的 bash 字符串，里面的大部分都按照原始字符解释

        比如前两个 echo，

        ```bash
        echo 'hello there'
        echo 'hello
           there'
        ```

        第二个 echo 有一个换行，echo 输出的字符串也照原样换行了。

    2. 如果想在单引号创建的字符串里加入额外的单引号，这样写是不行的：

        `echo 'hello 'there'`

        这样会被 bash 识别为三个部分：

        1. 第一个字符串`'hello '`

        2. 第二个字符串`there`

        3. 第三个不完整字符串`'`

            这个字符串只写了左单引号，没有写字符串内容和右单引号，因此 bash 会继续往后找另外一个单引号，作为字符串的结尾。

        为了验证这个猜想，上面第三个 echo 换行后把右单引号补全：

        ```bash
        echo 'hello 'there'
          aaa'
        ```

        而对应的输出为：

        ```
        hello there
          aaa
        ```

        与我们的预期相符。

    3. 第四个 echo 的字符串，想使用`\'`对单引号进行转义，但是却输出了`\`，该如何解释？

        ```bash
        echo 'hello \'there'
          bbb'
        ```

        由于单引号将字符串看作 raw string，所以 bash 将其解释为 3 个字符串：

        1. 第一个字符串`'hello \'`

        2. 第二个字符串`there`

        3. 第三个字符串，换行 + 两个空格 + bbb：

            ```
            '
              bbb'
            ```

        程序的输出也符合预期。

    4. 如果想在单引号括起的字符串中加入单引号，可以在字符串前加一个美元符号`$`，再在字符串中对单引号进行转义

        第 5 个 echo:

        ```bash
        echo $'hello 'there'
           ccc'
        ```

        虽然使用了`$`，但是并未对字符串中的单引号进行转义，因此 bash 仍认为它是三个字符串：

        1. `'hello '`

        2. `there`

        3. 换行 + 3 空格 + `ccc`

        第 6 个 echo:

        ```bash
        echo $'hello \'there'
        ```

        满足了在字符串前加`$`，并且在字符串中对单引号进行了`\'`转义，因此输出与预期一致。

## Variables

Example:

```bash
#!/usr/bin/bash
my_var=value
echo $my_var
```

* 在定义变量时，等号前后不能有空格

* bash 对大小写敏感

如果`value`中有空格，可以使用单引号`'`或双引号`"`将它们括起来。

单引号不对字符串转义修改，双引号允许你对字符串中的内容替换。

我们可以用`$()`执行命令并将结果返回到字符串中：

`myvar=$( ls /etc | wc -l )`

或者`myvar=$(ls -lh)`（小括号两边的空格不是必须的）

如果结果是多行输出，那么换行符都会被删除，从而并成单行的结果。

* Check if a variable is set in Bash

    Ref: <https://stackoverflow.com/questions/3601515/how-to-check-if-a-variable-is-set-in-bash>

* 可以用`export`导出一个变量，使得一个新的脚本在执行时，以值传递到新脚本中：

    `export var1`

    因为是按值传递，所以原脚本中的`var1`与新脚本中的`var1`修改一个并不会影响另外一个。

## User input

```bash
read var1
```

`read`命令可以读取输入并将其存到变量`var1`中。输入字符串中的空格，单引号，双引号，不会被特殊处理。但是如果输入左方向键，右方向键等，则会有回显乱码。退格键可以正常使用。反斜杠`\` + 换行会被特殊处理。反斜杠`\`加`n`，`t`等字符，不会按转义字符处理。

`-p`参数可以给出输入提示，`-s`则是 silent 模式，输入不回显。

```bash
#!/bin/bash
read -p 'Username: ' uservar
read -sp 'Password: ' passvar
```

读取多个变量：`read var1 var2 var3`。在输入变量时，变量之间使用空格分隔。如果输入的变量数大于指定的变量数，那么会把多余的输入都存储到最后一个输入中。如果输入的变量少于指定的变量，那么多余的变量会保持空白。

假如一个文件有多行内容，使用`cat text.txt | ./test.sh`的方式给脚本`read`，`read`会只处理第一行。换句话说，`read`只按空格对字符串分隔，而不按换行符、制表符等分隔。

在 bash 中，`STDIN`, `STDOUT`, `STDERR`分别对应 3 个 linux 文件：

* `STDIN`: `/proc/<processID>/fd/0`

* `STDOUT`: `/proc/<processID>/fd/1`

* `STDERR`: `/proc/<processID>/fd/2`

同时 linux 还给出了这些文件的快捷方式：

* `STDIN`: `/dev/stdin` or `/proc/self/fd/0`

* `STDOUT`: `/dev/stdout` or `/proc/self/fd/1`

* `STDERR`: `/dev/stderr` or `/proc/self/fd/2`

其中`fd`指的是 file descriptor。

Example:

```bash
#!/bin/bash
# A basic summary of my sales report
echo Here is a summary of the sales data:
echo ====================================
echo
cat /dev/stdin | cut -d' ' -f 2,3 | sort
```

输入与输出：

```
user@bash: cat salesdata.txt
Fred apples 20 August 4
Susy oranges 5 August 7
Mark watermelons 12 August 10
Terry peaches 7 August 15
user@bash:
user@bash: cat salesdata.txt | ./summary
Here is a summary of the sales data:
====================================
apples 20
oranges 5
peaches 7
watermelons 12
user@bash:
```

使用 flag 的例子：

```bash
while getopts u:a:f: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        a) arg=${OPTARG};;
        f) fullname=${OPTARG};;
    esac
done
echo "Username: $username";
echo "Age: $age";
echo "Full Name: $fullname";
```

我们还可以使用`$@`拿到所有的参数：

```bash
i=1
for user in "$@"
do
    echo "Username - $i: $user"
    i=$((i + 1))
done
```

还可以用`shift`改变`$1`这些参数的指向：

```bash
i=1;
j=$#;
while [ $i -le $j ] 
do
    echo "Username - $i: $1";
    i=$((i + 1));
    shift 1;
done
```

**Special variables**:

1. `$0`: The name of the Bash script.

    实际上，这个变量存储的是脚本被调用时的路径名。

    如果用`./my_script.sh`调用，那么`$0`就等于`./my_script.sh`；如果用`/path/to/my_script.sh`绝对路径调用，那么`$0`就等于`/path/to/my_script.sh`；如果使用`bash my_script.sh`调用，那么`$0`就等于`my_script.sh`。

1. `$1` - `$9`: The first 9 arguments to the Bash script.

1. `$#`: How many arguments supplied to the Bash script.

1. `$@`: All the arguments supplied to the Bash script.

1. `$?`: The exit status of the most recently run process.

1. `$$`: The process ID of the current script.

1. `$USER`: The username of the user running the script.

1. `$HOSTNAME`: The hostname of the machine the script is running on.

1. `$SECONDS`: The number of seconds since the script was started.

1. `$RANDOM`: Returns a different random number each time is it referred to.

1. `$LINENO`: Returns the current line number in the Bash script.

1. `env`命令可列出其它可用的环境变量。

## Arithmetic

`let`可以让我们做算术运算。

```bash
#!/bin/bash
# Basic arithmetic using let

let a=5+4
echo $a # 9

let "a = 5 + 4"
echo $a # 9

let a++
echo $a # 10

let "a = 4 * 5"
echo $a # 20

let "a = $1 + 30"
echo $a # 30 + first command line argument
```

可以使用的运算符：

* `+`, `-`, `\*`, `/`
* `var++`, `var--`
* `%`

`expr`的用法和`let`相似，只不过它直接输出表达式的内容，而且不需要将表达式使用双引号括起来。

`expr item1 operator item2`

Examples:

```bash
#!/bin/bash
# Basic arithmetic using expr
expr 5 + 4
expr "5 + 4"
expr 5+4
expr 5 \* $1
expr 11 % 2
a=$( expr 10 - 3 )
echo $a # 7
```

输出

```
user@bash: ./expr_example.sh 12
9
5 + 4
5+4
60
1
7
user@bash:
```

我们也可以用双括号做运算：

`$(( expression ))`

Examples:

```bash
#!/bin/bash
# Basic arithmetic using double parentheses

a=$(( 4 + 5 ))
echo $a # 9

a=$((3+5))
echo $a # 8

b=$(( a + 3 ))
echo $b # 11

b=$(( $a + 4 ))
echo $b # 12

(( b++ ))
echo $b # 13

(( b += 3 ))
echo $b # 16

a=$(( 4 * 5 ))
echo $a # 20
```

可以使用`${#variable}`得到一个变量的长度：

```bash
#!/bin/bash
# Show the length of a variable.

a='Hello World'
echo ${#a} # 11

b=4953
echo ${#b} # 4
```

## If statements

```bash
if [ <some test> ]
then
    <commands>
fi
```

Example:

```bash
#!/bin/bash
# Basic if statement

if [ $1 -gt 100 ]
then
    echo Hey that\'s a large number.
    pwd
fi

date
```

输出：

```
user@bash: ./if_example.sh 15
Sat 21 Aug 3:01:25 2021
user@bash: ./if_example.sh 150
Hey that's a large number.
/home/ryan/bin
Sat 21 Aug 3:01:25 2021
user@bash:
```

`[]`相当于命令`test`，常见的测试命令如下：

|Operator|Description|
|-|-|
|`! EXPRESSION`|The `EXPRESSION` is false.|
|`-n STRING`|The length of `STRING` is greater than zero.|
|`-z STRING`|The length of `STRING` is zero (ie is empty).|
|`STRING1 = STRING2`|`STRING1` is equal to `STRING2`|
|`STRING1 != STRING2`|`STRING1` is not equal to `STRING2`|
|`INTEGER1 -eq INTEGER2`|`INTEGER1` is numerically equal to `INTEGER2`|
|`INTEGER1 -gt INTEGER2`|`INTEGER1` is numerically greater than `INTEGER2`|
|`INTEGER1 -lt INTEGER2`|`INTEGER1` is numerically less than `INTEGER2`|
|`-d FILE`|`FILE` exists and is a directory.|
|`-e FILE`|`FILE` exists.|
|`-r FILE`|`FILE` exists and the read permisson is granted.|
|`-s FILE`|`FILE` exists and it's size is greater than zero (ie. it is not empty).|
|`-w FILE`|`FILE` exists and the write permission is granted.|
|`-x FILE`|`FILE` exists and the execute permission is granted.|

Examples:

```
user@bash: test 001 = 1
user@bash: echo $?
1
user@bash: test 001 -eq 1
user@bash: echo $?
0
user@bash: touch myfile
user@bash: test -s myfile
user@bash: echo $?
1
user@bash: ls /etc > myfile
user@bash: test -s myfile
user@bash: echo $?
0
user@bash: 
```

bash 对缩进没有要求，但是最好还是缩进。

嵌套的 if 语句：

```bash
#!/bin/bash
# Nested if statements
if [ $1 -gt 100 ]
then
    echo Hey that\'s a large number.

    if (( $1 % 2 == 0 ))
    then
        echo And is also an even number.
    fi
fi
```

if else 语句：

```bash
if [ <some test> ]
then
    <commands>
else
    <other commands>
fi
```

Example:

```bash
#!/bin/bash
# else example

if [ $# -eq 1 ]
then
    nl $1
else
    nl /dev/stdin
fi
```

if elif else:

```bash
if [ <some test> ]
then
    <commands>
elif [ <some test> ]
then
    <different commands>
else
    <other commands>
fi
```

Example:

```bash
#!/bin/bash
# elif statements

if [ $1 -ge 18 ]
then
    echo You may go to the party.
elif [ $2 == 'yes' ]
then
    echo You may go to the party but be back before midnight.
else
    echo You may not go to the party.
fi
```

You can have as many `elif` branches as you like. The final `else` is also optional.

Boolean Operations:

and: `&&`, or: `||`

Example:

```bash
#!/bin/bash
# and example

if [ -r $1 ] && [ -s $1 ]
then
    echo This file is useful.
fi
```

```bash
#!/bin/bash
# or example
if [ $USER == 'bob' ] || [ $USER == 'andy' ]
then
    ls -alh
else
    ls
fi
```

Case Statements:

```bash
case <variable> in
<pattern 1>)
    <commands>
    ;;
<pattern 2>)
    <other commands>
    ;;
esac
```

Example:

```bash
#!/bin/bash
# case example
case $1 in
    start)
        echo starting
        ;;
    stop)
        echo stoping
        ;;
    restart)
        echo restarting
        ;;
    *)
        echo don\'t know
        ;;
esac
```

Another example:

```bash
#!/bin/bash
# Print a message about disk useage.
space_free=$( df -h | awk '{ print $5 }' | sort -n | tail -n 1 | sed 's/%//' )
case $space_free in
    [1-5]*)
        echo Plenty of disk space available
        ;;
    [6-7]*)
        echo There could be a problem in the near future
        ;;
    8*)
        echo Maybe we should look at clearing out old files
        ;;
    9*)
        echo We could have a serious problem on our hands soon
        ;;
    *)
        echo Something is not quite right here
        ;;
esac
```

## Loops

while:

```bash
while [ <some test> ]
do
    <commands>
done
```

Example:

```bash
#!/bin/bash
# Basic while loop

counter=1
while [ $counter -le 10 ]
do
    echo $counter
    ((counter++))
done

echo All done
```

until:

```bash
until [ <some test> ]
do
    <commands>
done
```

Example:

```bash
#!/bin/bash
# Basic until loop

counter=1
until [ $counter -gt 10 ]
do
    echo $counter
    ((counter++))
done

echo All done
```

for:

```bash
for var in <list>
do
    <commands>
done
```

The list is defined as a series of strings, seperated by spaces.

Example:

```bash
#!/bin/bash
# Basic for loop

names='Stan Kyle Cartman'
for name in $names
do
    echo $name
done

echo All done
```

```bash
#!/bin/bash
# Basic range in for loop
for value in {1..5}
do
    echo $value
done
echo All done
```

```bash
#!/bin/bash
# Basic range with steps for loop

for value in {10..0..2}
do
    echo $value
done
echo All done
```

```bash
#!/bin/bash
# Make a php copy of any html files

for value in $1/*.html
do
    cp $value $1/$( basename -s .html $value ).php
done
```

```bash
x=5
for ((i=1; i<=x; i++))
do
    echo $i
done
```

break 与 continue:

```bash
#!/bin/bash
# Make a backup set of files

for value in $1/*
do
    used=$( df $1 | tail -1 | awk '{ print $5 }' | sed 's/%//' )
    if [ $used -gt 90 ]
    then
        echo Low disk space 1>&2
        break
    fi
    cp $value $1/backup/
done
```

```bash
#!/bin/bash
# Make a backup set of files

for value in $1/*
do
    if [ ! -r $value ]
    then
        echo $value not readable 1>&2
        continue
    fi
    cp $value $1/backup/
done
```

select:

```bash
select var in <list>
do
    <commands>
done
```

`select`可以创建一个菜单。当接收到`EOF` signal，或者执行`break`语句时退出菜单循环。

Example:

```bash
#!/bin/bash
# A simple menu system

names='Kyle Cartman Stan Quit'

PS3='Select character: '

select name in $names
do
    if [ $name == 'Quit' ]
    then
        break
    fi
    echo Hello $name
done
echo Bye
```

修改变量`PS3`可以改变提示语。

## Functions

```bash
# form 1
function_name () {
    <commands>
}

# form 2
function function_name {
    <commands>
}
```

bash 中的函数不能有形参，但给函数传递参数时，可以用`$1`，`$2`等：

```bash
#!/bin/bash
# Passing arguments to a function

print_something () {
    echo Hello $1
}

print_something Mars
print_something Jupiter
```

函数不能有返回值，但是可以返回一个状态（status）。

```bash
#!/bin/bash
# Setting a return status for a function

print_something () {
    echo Hello $1
    return 5
}

print_something Mars
print_something Jupiter
echo The previous function has a return value of $?
```

如果想返回字符串，我们可以把函数当作一个命令来执行：

```bash
#!/bin/bash
# Setting a return value to a function

lines_in_file () {
    cat $1 | wc -l
}

num_lines=$( lines_in_file $1 )
echo The file $1 has $num_lines lines in it.
```

如果我们创建一个变量，那么它默认是`global`属性的。如果我们在函数中使用`local`关键字创建一个变量，那么这个变量就只能在函数中可见：

`local var_name=<var_value>`

Example:

```bash
#!/bin/bash
# Experimenting with variable scope

var_change () {
    local var='local 1'
    echo Inside function: var1 is $var1 : var2 is $var2
    var1='changed again'
    var2='2 changed again'
}

var1='global 1'
var2='global 2'

echo Before function call: var1 is $var1 : var2 is $var2

var_change

echo After function call: var1 is $var1 : var2 is $var2
```

我们可以使用重名函数来覆盖 linux 命令：

```bash
#!/bin/bash
# Create a wrapper around the command ls

ls () {
    command ls -lh
}

ls
```

因为函数的优先级较高，所以想区别命令和函数时，需要在 linux 前加`command`。如果不加`command`，就会形成递归调用。

## User interface

## Common used command

### test

`test`命令没有输出，但是有一个 exit status，`0`代表`true`，`1`代表`false`。

Example:

```bash
num=4
if (test $num -gt 5)
then
    echo "yes"
else
    echo "no"
fi 
```

`test`命令也可被方括号`[]`替代：

```bash
file="/etc/passwd"
if [ -e $file ]
then
    echo "whew"
else
    echo "uh-oh"
fi
```

Syntax:

1. File tests:

    ```bash
    test [-a] [-b] [-c] [-d] [-e] [-f] [-g] [-h] [-L] [-k] [-p] [-r] [-s] [-S] [-u] [-w] [-x] [-O] [-G] [-N] [file]

    test -t fd

    test file1 {-nt | -ot | -ef} file2
    ```

1. String tests:

    ```bash
    test [-n | -z] string

    test string1 {= | != | < | >} string2
    ```

1. Shell options and variables:

    ```bash
    test -o option

    test {-v | -R} var
    ```

1. Simple logic (test if values are `null`):

    ```bash
    test [!] expr

    test expr1 {-a | -o} expr2
    ```

1. Numerical comparison (for integer values only; bash doesn't do floating point math):

    ```bash
    test arg1 {-eq | -ne | -lt | -le | -gt | -ge} arg2
    ```

Options:

1. `-a file`

    Returns true if *file* exists. Does the same thing as `-e`. Both are included for compatibility reasons with legacy versions of Unix.

1. `-b file`

    Returns true if *file* is "block-special." Block-special files are similar to regular files, but are stored on block devices — special areas on the storage device that are written or read one block (sector) at a time.

1. `-c file`

    Returns true if *file* is "character-special." Character-special files are written or read byte-by-byte (one character at a time), immediately, to a special device. For example, `/dev/urandom` is a character-special file.

1. `-d file`

    Returns true if *file* is a directory.

1. `-e file`

    Returns true if *file* exists. Does the same thing as `-a`. Both are included for compatibility reasons with legacy versions of Unix.

1. `-f file`

    Returns true if *file* exists, and is a regular file.

1. `-g file`

    Returns true if *file* has the setgid bit set.

1. `-h file`

	Returns true if *file* is a symbolic link. Does the same thing as `-L`. Both are included for compatibility reasons with legacy versions of Unix.

1. `-L file`

    Returns true if *file* is a symbolic link. Does the same thing as `-h`. Both are included for compatibility reasons with legacy versions of Unix.

1. `-k file`

    Returns true if *file* has its sticky bit set.

1. `-p file`

    Returns true if the file is a named pipe, e.g., as created with the command `mkfifo`.

1. `-r file`

    Returns true if *file* is readable by the user running `test`.

1. `-s file`

    Returns true if *file* exists, and is not empty.

1. `-S file`

    Returns true if *file* is a socket.

1. `-t fd`

    Returns true if file descriptor *fd* is opened on a terminal.

1. `-u file`

    Returns true if *file* has the setuid bit set.

1. `-w file`

    Returns true if the user running `test` has write permission to *file*, i.e., make changes to it.

1. `-x file`

    Returns true if *file* is executable by the user running `test`.

1. `-O file`

	Returns true if *file* is owned by the user running `test`.

1. `-G file`

    Returns true if *file* is owned by the group of the user running `test`.

1. `-N file`

    Returns true if *file* was modified since the last time it was read.

1. `file1 -nt file2`

    Returns true if *file1* is newer (has a newer modification date/time) than *file2*.

1. `file1 -ot file2`

    Returns true if *file1* is older (has an older modification date/time) than *file2*.

1. `file1 -ef file2`

    Returns true if *file1* is a hard link to *file2*.

1. `test [-n] string`

    Returns true if *string* is not empty. Operates the same with or without `-n`.

    For example, if `mystr=""`, then `test "$mystr"` and `test -n "$mystr"` would both be false. If `mystr="Not empty"`, then `test "$mystr"` and `test -n "$mystr"` would both be true.

1. `-z string`

    Returns true if string *string* is empty, i.e., `""`.

1. `string1 = string2`

    Returns true if *string1* and *string2* are equal, i.e., contain the same characters.

1. `string1 != string2`

    Returns true if *string1* and *string2* are not equal.

1. `string1 < string2`

    Returns true if *string1* sorts before *string2* lexicographically, according to ASCII numbering, based on the first character of the string. For instance, `test "Apple" < "Banana"` is true, but `test "Apple" < "banana"` is false, because all lowercase letters have a lower ASCII number than their uppercase counterparts.

    **Tip**: Enclose any variable names in double quotes to protect whitespace. Also, escape the less than symbol with a backslash to prevent bash from interpreting as a redirection operator. For instance, use t`est "$str1" \< "$str2"` instead of `test $str1 < $str2`. The latter command will try to read from a file whose name is the value of variable *str2*. For more information, see redirection in bash.

1. `string1 > string2`

    Returns true if *string1* sorts after *string2* lexicographically, according to the ASCII numbering. As noted above, use `test "$str1" \> "$str2"` instead of `test $str1 > $str2`. The latter command creates or overwrites a file whose name is the value of variable *str2*.

1. `-o option`

    Returns true if the shell option *opt* is enabled.

1. `-v var`

    Returns true if the shell variable *var* is set.

1. `-R var`

    Returns true if the shell variable *var* is set, and is a name reference. (It's possible this refers to an *indirect reference*, as described in Parameter expansion in bash.)

1. `! expr`

    Returns true if and only if the expression *expr* is null.

1. `expr1 -a expr2`

    Returns true if expressions *expr1* and *expr2* are both not null.

1. `expr1 -o expr2`

    Returns true if either of the expressions *expr1* or *expr2* are not null.

1. `arg1 -eq arg2`

    True if argument *arg1* equals *arg2*.

1. `arg1 -ne arg2`

    True if argument *arg1* is not equal to *arg2*.

1. `arg1 -lt arg2`

    True if numeric value *arg1* is less than *arg2*.

1. `arg1 -le arg2`

    True if numeric value *arg1* is less than or equal to *arg2*.

1. `arg1 -gt arg2`

    True if numeric value *arg1* is greater than *arg2*.

1. `arg1 -ge arg2`

    True if numeric value *arg1* is greater than or equal to *arg2*.

Notes:

1. All arguments to test must be separated by a space, including all operators.

1. The `<` and `>` operators are lexicographical comparisons, based on ASCII numbering. They are not numerical operators (instead, use `-lt`, `-gt`, etc. for comparing numbers).

1. The precise behavior of `test`, depending on the number of arguments provided, is as follows:

    | #<br>args | test behavior |
    | - | - |
    | 0 | Always return false. |
    | 1 | Return true, if and only if the expression is not null. |
    | 2 | If the first argument is `!`, return true if and only if the expression is null. <br> If the first argument if one of the other unary operators (`-a`, `-b`, etc.), return true if and only if the unary test of the second argument is true. <br> If the first argument is not an unary operator, return false. |
    | 3 | The following conditions are applied in the order listed. <br> If the second argument is one of the binary conditional operators listed above, the result is the binary test using the first and third arguments as operands. Binary conditional operators are those which take two operands, e.g., `-nt`, `-eq`, `<`, etc. <br> The `-a` and `-o` operators are considered binary operators when there are three arguments. <br> If the first argument is `!`, the value is the negation of the two-argument test using the second and third arguments. <br> If the first argument is exactly `(` and the third argument is exactly `)`, the result is the one-argument test of the second argument. In other words, `( expr )` returns the value of `expr`. This special case exists as a way to override the normal precedence of operations. <br> Otherwise, the expression is false. |
    | 4 | If the first argument is `!`, the result is the negation of the three-argument expression composed of the remaining arguments. <br> Otherwise, the expression is parsed and evaluated according to precedence using the rules listed above. |
    | 5+ | The expression is parsed and evaluated according to precedence using the rules listed above. |

Exit status

`0` for true, `1` for false. Anything greater than 1 indicates an error or malformed command.

`$?` can be used to get the exit status of `test`.

### trap

Syntax:

`trap COMMAND SIGNALS...`

Examples:

1. `EXIT`

    ```bash
    tempfile=/tmp/tmpdata
    trap "rm -f $tempfile" EXIT
    ```

    ```bash
    function cleanup()
    {
        # ...
    }
    
    trap cleanup EXIT
    ```

    Note that if you send a `kill -9` to your script, it will not execute the `EXIT` trap before exiting.

1. `SIGINT`

    `SIGINT` can catch Ctrl-C.

    ```bash
    ctrlc_count=0

    function no_ctrlc()
    {
        let ctrlc_count++
        echo
        if [[ $ctrlc_count == 1 ]]; then
            echo "Stop that."
        elif [[ $ctrlc_count == 2 ]]; then
            echo "Once more and I quit."
        else
            echo "That's it. I quit."
            exit
        fi
    }

    trap no_ctrlc SIGINT

    while true
    do
        echo Sleeping
        sleep 10
    done
    ```

1. `-`

    ```bash
    # Run something important, no Ctrl-C allowed.
    trap "" SIGINT
    important_command

    # Less important stuff from here on out, Ctrl-C allowed.
    trap - SIGINT
    not_so_important_command
    ```

1. `USR1`

    ```bash
    nopens=0
    function show_opens()
    {
        echo "Seen $nopens sudo session opens"
    }

    sudo journalctl -f | while read line
    do
        if [[ -z "$trap_set" ]]; then
            trap_set=1
            echo "Trap set in $BASHPID"
            trap show_opens USR1
        fi
        if [[ $line =~ sudo.*session.*opened ]]; then
            let nopens++
        fi
    done
    ```

    使用：

    ```
    $ sudo -k  # reset the sudo timestamp
    $ bash bkgnd.sh &
    [1] 1000
    Trap set in 1002
    $ kill -SIGUSR1 1002
    ```

## String manipulation

1. string length

    `${#string}`

    `expr length $string`

    `expr "$string" : '.*'`

1. extract substring

    `${string:position}`

    `${string:position:length}`

    (0-based indexing)

    If the $string parameter is `*` or `@`, then this extracts the positional parameters, [1] starting at `$position`.

    `expr substr $string $position $length`
    
    `expr match "$string" '\($substring\)'`, `expr "$string" : '\($substring\)'`: Extracts `$substring` at beginning of `$string`, where `$substring` is a regular expression.

    `expr match "$string" '.*\($substring\)'`, `expr "$string" : '.*\($substring\)'`: Extracts `$substring` at end of `$string`, where `$substring` is a regular expression.

    Examples:

    ```bash
    #!/bin/bash

    var="Welcom to the geekstuff"
    echo ${var:15}  # geekstuff
    echo ${var:15:4}  # geek

    stringZ=abcABC123ABCabc
    echo ${stringZ:-4}  # abcABC123ABCabc
    echo ${stringZ:(-4)}  # Cabc
    echo ${stringZ: -4}  # Cabc

    echo ${*:2}  # the second and following positional parameters
    echo ${@:2}  # same as above
    echo ${*:2:3}  # three positional parameters, starting at the second

    echo `expr substr $stringZ 1 2`  # ab
    echo `expr substr $stringZ 4 3`  # ABC
    ```

1. Shortest substring match

    delete the shortest match of `$substring` from front of `$string`: `${string#substring}`

    delete the shortest match of `$substring` from back of `$string`: `${string%substring}`

    Examples:

    ```bash
    #!/bin/bash

    filename="bash.string.txt"
    echo ${filename#*.}  # string.txt
    echo ${filename%.*}  # bash.string
    ```

1. Longest substring match

    `${string##substring}`

    `${string%%substring}`

1. Find and replace string values

    Replace only first match: `${string/pattern/replacement}`

    Replace all the matches: `${string//pattern/replacement}`

    Replace beginning and end: `${string/#pattern/replacement}`, `${string/%pattern/replacement}`

    Examples:

    ```bash
    #!/bin/bash

    filename="bash.string.txt"
    echo ${filename/str*./operations.}  # bash.operations.txt

    filename="Path of the bash is /bin/bash"
    echo ${filename//bash/sh}  # Path of the sh is /bin/sh

    filename="/root/admin/monitoring/process.sh"

    echo ${filename/#\/root/\/tmp}  # /tmp/admin/monitoring/process.sh
    echo ${filename/%.*/.ksh}  # /root/admin/monitoring/process.ksh
    ```

1. Length of matching substring at beginning of string

    `expr match "$string" '$substring'`

    `expr "$string" : '$substring'`

    Note: `$substring` is a regular expression.

    Example:

    ```bash
    stringZ=abcABC123ABCabc
    #       |------|
    #       12345678

    echo `expr match "$stringZ" 'abc[A-Z]*.2'`   # 8
    echo `expr "$stringZ" : 'abc[A-Z]*.2'`       # 8
    ```

1. index

    `expr index $string $substring`

    Example:

    ```bash
    stringZ=abcABC123ABCabc
    #       123456 ...
    echo `expr index "$stringZ" C12`            # 6
                                                # C position.

    echo `expr index "$stringZ" 1c`             # 3
    # 'c' (in #3 position) matches before '1'.
    ```

1. convert a string to lower case in Bash

    posix:

    1. tr
    
        `echo "$a" | tr '[:upper:]' '[:lower:]'`

    1. awk
    
        `echo "$a" | awk '{print tolower($0)}'`

    Ref: <https://stackoverflow.com/questions/2264428/how-to-convert-a-string-to-lower-case-in-bash>

1. Converting a Bash array into a delimited string

    ```bash
    ids="1 2 3 4";echo ${ids// /|}
    ```

    output:

    ```
    1|2|3|4
    ```

    Ref: <https://stackoverflow.com/questions/13470413/converting-a-bash-array-into-a-delimited-string>

## Subshell

Materials:

1. <https://tldp.org/LDP/abs/html/subshells.html>

* parent shell create a new subshell and get the pid of the subshell

    Ref: <https://stackoverflow.com/questions/20573621/bash-get-process-id-of-a-process-started-in-subshell>


## File

bash 通常通过标准输入输出和文件进行交互。

按行读取文件并回显：

`content.txt`:

```
hello      world
nihao
zaijian
```

`test.sh`

```bash
#!/bin/bash

file_path=./content.txt
while read -r line
do
    echo "$line"
done < $file_path
```

输出：

```
hello      world
nihao
zaijian
```

可以看到，文件通过重定向的方式，被`read`函数捕获到，每次处理一行。

说明：

1. 在`echo "$line"`时，必须加上双引号，如果不加，bash 会首先把`$line`展开为带空格的字符串列表，然后按多个参数给`echo`输出。此时第一行就会变成`hello world`。

1. 如果`content.txt`的最后一行的末尾没有`\n`，那么`read`在读取完最后一行后，会返回 false，导致 while 循环退出，从而不会打印最后一行。但是此时最后一行的值已经存在了`$line`变量中，我们还可以在 while 外部将其打印出来。

    这种情况最好的解决办法就是让每旧文件的末尾都最好带有`\n`，使得 bash 能正常处理。

    如果遇到别人写的文本文档没有`\n`，自己写脚本处理时，可以检测跳出循环后的`$line`值是否为 empty。如果非空，说明最后一行仍有内容。

1. 如果`content.txt`文件里有反斜杠`\`，那么`read`会先将反斜杠后面的字符处理为转义字符，然后再将值存入到`$line`变量中。

    这显然不是我们所希望的，所以需要给`read`加上`-r`参数，保证所有字符都不会被转义。

1. 参考资料上的`read`写法是`IFS= read -r line`，不清楚`IFS`有什么用处。

    Ref: <https://www.cyberciti.biz/faq/unix-howto-read-line-by-line-from-file/>

## Miscellaneous

1. 有关重定向符的用法

    * `command > filename`

        将输出重定向到文件，如果文件存在，则会被覆盖。

        其实这行命令应该这么理解：`[command] [> filename]`，把它分成两个部分。第二部分`> filename`的完整版应该是`1> filename`，`1`表示 stdout。

        例子：

        `echo "hello, world" > hello.txt`
    
    * `[descriptor]> filename`

        把文件描述符重定向到某个文件。`descriptor`可以是`0`，`1`，`2`，其中`0`表示 stdin，`1`表示 stdout，`2`表示 stderr。

        注意，文件描述符和`>`中间不能有空格，而后面的 filename 与`>`之间可以有空格，也可以没有。

        Example:

        file name: `echo_stderr.sh`

        ```bash
        #!/bin/bash

        echo "hello, world!" >& 2
        ```

        `./echo_stderr.sh 2> hello.txt`

        在这个例子中，`echo_stderr.sh`在 stderr 进行输出。而下面的命令`./echo_stderr.sh 2> hello.txt`将 stderr 重定向到`hello.txt`文件。因此可以在文件中得到输出的内容。

        注意，如果将上述命令替换为`./echo_stderr.sh 1> hello.txt`或`./echo_stderr.sh > hello.txt`，则仍会在屏幕中进行输出，`hello.txt`文件中不会有任何内容。

    * `[descriptor_1]>& <descriptor_2>`

        将一个文件描述符重定向到另一个文件描述符。

        例子：

        `echo "helo, world" >& 2`

        将 stdout 重定向到 stderr。

        注：

        * `[descriptor_1]>&`这 3 个符号之间不能有空格，而`<descriptor_2>`之前可以有空格，也可以没有。

        * 如果省略不写`[descriptor_1]`，那么`[descriptor_1]`默认为`1`。

    * `command &> filename`

        将 stdout 和 stderr 都重定向到指定文件。注意`&>`前没有其他的参数。

        这个命令等价于`1>filename 2>&1`

        一些参考资料：<https://stackoverflow.com/questions/24793069/what-does-do-in-bash>

1. 有关标准输入输出，标准错误

    * stdin: `/proc/<PID>/fd/0`，`/proc/self/fd/0`，`/dev/stdin`，`0`，这几种都是等价的，下面的同理。

    * stdout: `/proc/<PID>/fd/1`，`/proc/self/fd/1`，`/dev/stdout`，`1`

    * stderr: `/proc/<PID>/fd/2`，`/proc/self/fd/2`，`/dev/stderr`，`2`

    程序可以通过 stdin 接收管道（pipe）传递的值：

    `recv.sh`:

    ```bash
    #!/bin/bash
    $input$(cat /dev/stdin)
    echo "recv: $input"
    ```

    `echo "hello" | ./recv.sh`

1. 有关进程和子进程

    当前进程的 PID：`echo "in current process, PID: $$"`

    可以用小括号打开一个子进程：`(echo "in child process, PID: $BASHPID")`（好像也可以用`$PPID`获得子进程（subshell）的 pid，但我没试过）

    注意子进程的 PID 不可以用`$$`获得。因为子进程会从父进程中继承一些环境变量。

    除了使用小括号可以打开一个子进程外，执行一个别的脚本`./xxx.sh`或`bash xxx.sh`也会打开一个子进程。

    将子进程放到后台执行：`(echo "hello") &`

    （似乎所有的后台命令都是开了个新进程，比如`echo hello &`）

    获得子进程的返回值（exit code）：

    1. 如果是前台的子进程，可以使用`$?`获得

        ```bash
        (echo "in child process"; exit 1)
        echo "the exit code of the child process: $?"
        ```

    1. 如果是后台的子进程，必须使用`wait`获得

        ```bash
        exit 1 &
        wait $!  # $! is the PID of the latest background process
        echo "the exit code of the background child process is $?"
        ```

        `wait`命令会等待子进程执行完成，并将其返回值传递到`$?`变量中。

    `$VAR=$(command)$`也会开启一个子进程：
    
    ```bash
    #!/bin/bash
    child_pid=$(echo $BASHPID)
    echo "child pid: $child_pid"
    echo "current pid: $$"
    ```

1. 有关字符串处理

* if statement

    <https://acloudguru.com/blog/engineering/conditions-in-bash-scripting-if-statements>

    <https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php>

* bash 中默认函数内外的变量都是全局变量。可以用`local xxxx`或`local xxx=xxxx`定义局部变量。

1. debug a bash script

    vscode 里有个插件叫 bash debug 可以对 bash 设置断点。但是如果是手动调试的话，可以看看下面这些链接：

    * <https://unix.stackexchange.com/questions/521775/how-to-debug-trace-bash-function>

    * <https://www.shell-tips.com/bash/debug-script/#gsc.tab=0>

* 有关返回值

    前台进程的返回值可以直接由`$?`获得：

    ```bash
    echo "hello" | grep hello
    echo $?  # 显示 0
    echo "hello" | grep world
    echo $?  # 字符串中没有 world，返回 1
    ```

    后台进程的返回值：

    后台进程的返回值不能直接获得，必须由`wait <PID>`才能获得。

    我们可以首先由`$!`得到最近后台进程的 pid，然后通过`wait $!`得到后台进程的返回值。

    Ref:

    1. <https://stackoverflow.com/questions/1570262/get-exit-code-of-a-background-process>

    1. <https://www.baeldung.com/linux/background-process-get-exit-code>

* `if`中的字符串在使用`-n`比较时，要加`""`

    Supposing `b` is an undefined variable，

    ```bash
    if [ -n "$b" ]; then echo "hello"; fi
    ```

    output： nothing

    ```bash
    if [ -n $b ]; then echo "hello"; fi
    ```

    output:

    ```
    hello
    ```

* EOF

    There is no method to echo an `EOF` directly. But there are some ways to trigger an `EOF`:

    1. reaching the end of a file

    1. pressing key bindings to `EOF` (`Ctrl + D` by default)

    1. `cat <<EOF`

        Ref: <https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash>

* `bash -c`

    <https://unix.stackexchange.com/questions/144514/add-arguments-to-bash-c>

* learning materials

    1. <https://askubuntu.com/questions/121866/why-does-bash-remove-n-in-cat-file>

    1. <https://stackoverflow.com/questions/10028820/bash-wait-with-timeout>

    1. <https://stackoverflow.com/questions/42615374/the-linux-timeout-command-and-exit-codes>

    1. <https://stackoverflow.com/questions/13296863/difference-between-wait-and-sleep>

    1. <https://superuser.com/questions/1699317/permission-denied-on-dev-stderr-after-sudo>

    1. <https://unix.stackexchange.com/questions/38538/bash-dev-stderr-permission-denied>

    1. <https://stackoverflow.com/questions/37981492/how-to-timeout-a-group-of-commands-in-bash>

    1. <https://linuxconfig.org/bash-scripting-tutorial>

    1. <https://www.javatpoint.com/bash-split-string>

    1. <https://www.tutorialkart.com/bash-shell-scripting/bash-split-string/>

    1. <https://www.freecodecamp.org/news/bash-array-how-to-declare-an-array-of-strings-in-a-bash-script/>

    1. <https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays>

    1. <https://linuxhint.com/bash_split_examples/>

    1. <https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash>

    1. <https://stackoverflow.com/questions/3601515/how-to-check-if-a-variable-is-set-in-bash>

    1. <https://stackoverflow.com/questions/71768999/how-to-merge-when-you-get-error-hint-you-have-divergent-branches-and-need-to-s>

    1. <https://stackoverflow.com/questions/9954794/execute-a-shell-function-with-timeout>

1. How to Create Bash Aliases

    <https://linuxize.com/post/how-to-create-bash-aliases/>


