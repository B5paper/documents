# Bash Note

Reference: <https://www.computerhope.com/unix.htm>

## Variables

Special variables:

1. `$0`: The name of the Bash script.

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

定义，使用变量：

```bash
#!/usr/bin/bash
variable=value
echo $variable
```

等号前后不能有空格。bash 对大小写敏感。如果`value`中有空格，可以使用单引号`'`或双引号`"`将它们括起来。

单引号不对字符串转义修改，双引号允许你对字符串中的内容替换。

我们可以用`$()`执行命令并将结果返回到字符串中：

`myvar=$( ls /etc | wc -l )`

如果结果是多行输出，那么换行符都会被删除，从而并成单行的结果。

可以用`export`导出一个变量，使得一个新的脚本在执行时，以值传递到新脚本中：

`export var1`

因为是按值传递，所以原脚本中的`var1`与新脚本中的`var1`修改一个并不会影响另外一个。

## User input

```bash
read var1
```

`read`命令可以读取输入并将其存到变量`var1`中。

`-p`参数可以给出输入提示，`-s`则是 silent 模式，输入不回显。

```bash
#!/bin/bash
read -p 'Username: ' uservar
read -sp 'Password: ' passvar
```

读取多个变量：`read var1 var2 var3`。在输入变量时，变量之间使用空格分隔。如果输入的变量数大于指定的变量数，那么会把多余的输入都存储到最后一个输入中。如果输入的变量少于指定的变量，那么多余的变量会保持空白。

在 bash 中，`STDIN`, `STDOUT`, `STDERR`分别对应 3 个 linux 文件：

* `STDIN`: `/proc/<processID>/fd/0`

* `STDOUT`: `/proc/<processID>/fd/1`

* `STDERR`: `/proc/<processID>/fd/2`

同时 linux 还给出了这些文件的快捷方式：

* `STDIN`: `/dev/sdin` or `/proc/self/fd/0`

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
