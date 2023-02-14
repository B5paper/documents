# Makefile Note

hello world:

```makefile
hello:
	echo "hello world"
```

Makefiles must be indented using TABs and not spaces or `make` will fail.

rules:

```makefile
targets: prerequisites
    command
    command
    command
```

* The `targets` are file names, separated by spaces. Typically, there is only one per rule.

* The `commands` are a series of steps typically used to make the target(s). These need to start with a tab character, not spaces.

* The `prerequisites` are also file names, separated by spaces. These files need to exist before the commands for the target are run. These are also called *dependencies*.

Example:

```makefile
blah: blah.o
	cc blah.o -o blah  # Runs third

blah.o: blah.c
	cc -c blah.c -o blah.o  # Runs second

blah.c:
	echo "int main() { return 0; }" > blah.c  # Runs first
```

依赖链分析：`blah`的生成需要`blah.o`，`make`程序往下找到`blah.o`目标，发现它依赖于`blah.c`；然后`make`程序发现`blah.c`什么都不信赖，所以直接就执行对应的命令了。

`make`程序会将`Makefile`中的第一个 target 作为 default target。

如果某个 target 已经被创建，那么`make`就会跳过这个 target。

如果某个 target 不创建任何文件，那么每次`make`，它被依赖时，总是执行。常见的`clean`选项就利用了这个特点：

```makefile
some_file:
    touch some_file

clean:
    rm -rf some_file
```

此时如果执行`make`，那么会执行`some_file` target。如果执行`make clean`，那么会执行`clean` target。

使用变量：

```makefile
files := file1 file2
some_file: $(files)
	echo "Look at this variable: " $(files)
	touch some_file

file1:
	touch file1
file2:
	touch file2

clean:
	rm -f file1 file2 some_file
```

取变量值可以使用`$()`，也可以使用`${}`。

说明：

1. 这个例子里变量用`:=`赋值了，也可以使用`=`。

另外一个例子：

```makefile
x := dude

all:
	echo $(x)
	echo ${x}

	# Bad practice, but works
	echo $x 
```

自己写的一个例子：

```makefile
lib_src = hello.c

main: main.c hello.o
	cc main.c hello.o -o main

hello.o: hello.c
	cc -c ${lib_src} -o hello.o

clean:
	rm -f main

echo_lib_src:
	echo ${lib_src}
```

执行多个 targets:

```makefile
all: one two three

one:
	touch one
two:
	touch two
three:
	touch three

clean:
	rm -f one two three
```

multiple targets:

```makefile
all: f1.o f2.o

f1.o f2.o:
	echo $@
# Equivalent to:
# f1.o:
#	 echo f1.o
# f2.o:
#	 echo f2.o
```

其中`$@`表示 target name。

`make`的执行结果：

```
echo f1.o
f1.o
echo f2.o
f2.o
```

说明：

1. 如果执行`make`或`make all`，那么会分别执行`make f1.o`和`make f2.o`。如果执行`make f1.o`，那么只会执行`make f1.o`；`f2.o`同理。如果执行`make f1.o f2.o`，那么效果同`make`和`make all`。

**Wildcard**

常用的 wildcard 有`*`和`%`。

* `*` wildcard