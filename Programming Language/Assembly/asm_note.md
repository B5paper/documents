# Assembly X86 Win32 Note

tutorial: <https://www.cs.virginia.edu/~evans/cs216/guides/x86.html>

nasm tutorial: <https://www.tutorialspoint.com/assembly_programming/assembly_basic_syntax.htm>

resources: <https://visualmasm.com/resources/>

tutorial: <http://www.interq.or.jp/chubu/r6/masm32/tute/tute001.html>

masm tutorial: <https://www.jj2007.eu/Masm32_Tips_Tricks_and_Traps.htm>

books:

1. Assembly Language Step-By-Step, P273

1. Assembly Language for x86 Processors, P88

1. x86汇编语言 从实模式到保护模式，有空了看下。<https://book.douban.com/subject/20492528/>

Reference:

1. 常见的各种命令，都可以在这里查：<https://www.felixcloutier.com/x86/>

汇编编译器：

masm32: <https://www.masm32.com/index.htm>

如果只是想学习 8086 汇编，可以使用 emu8086。

visual studio 提供了 masm，但是被集成在 visual studio 中，无法单独下载。有时间了学习下：<https://learn.microsoft.com/en-us/cpp/assembler/masm/masm-for-x64-ml64-exe?view=msvc-170>。

有时间可以了解下 bochs: <https://www.cnblogs.com/diyuanxu/articles/16367513.html>

其他的可能会用到的 debugger: <https://softwarerecs.stackexchange.com/questions/31746/assembly-debugger>

使用 gdb 对汇编进行调试：<http://dbp-consulting.com/tutorials/debugging/basicAsmDebuggingGDB.html>

还有一些没探索过的资料：

1. <https://thestarman.pcministry.com/asm/win32/index.html>

## cache

* linux 下如何编译 x86 asm？

	在 Linux 下编译 x86 汇编程序有多种方式，主要取决于你使用的汇编器。以下是几种常见的方法：

	## 1. 使用 GNU 汇编器 (GAS)

	### 编译 Intel 语法汇编
	```bash
	# 编译为目标文件
	as --32 -o program.o program.s

	# 链接为可执行文件
	ld -m elf_i386 -o program program.o
	```

	### 编译 AT&T 语法汇编（GAS 默认语法）
	```bash
	# 编译和链接一步完成
	gcc -m32 -o program program.s
	```

	## 2. 使用 NASM 汇编器

	### 安装 NASM
	```bash
	# Ubuntu/Debian
	sudo apt-get install nasm

	# CentOS/RHEL
	sudo yum install nasm
	```

	### 编译 Intel 语法汇编
	```bash
	# 编译为目标文件（32位）
	nasm -f elf32 -o program.o program.asm

	# 链接（32位）
	ld -m elf_i386 -o program program.o

	# 或者使用 gcc 链接（会包含标准库）
	gcc -m32 -o program program.o
	```

	## 3. 完整的编译示例

	### 示例汇编代码（hello.asm）
	```nasm
	section .data
		hello db 'Hello, World!', 0xa
		hello_len equ $ - hello

	section .text
		global _start

	_start:
		; 写入系统调用
		mov eax, 4          ; sys_write
		mov ebx, 1          ; stdout
		mov ecx, hello      ; 字符串地址
		mov edx, hello_len  ; 字符串长度
		int 0x80

		; 退出系统调用
		mov eax, 1          ; sys_exit
		mov ebx, 0          ; 退出状态
		int 0x80
	```

	### 编译步骤
	```bash
	# 使用 NASM
	nasm -f elf32 hello.asm -o hello.o
	ld -m elf_i386 hello.o -o hello

	# 或者一步完成
	nasm -f elf32 hello.asm && ld -m elf_i386 hello.o -o hello
	```

	## 4. 使用 GCC 内联汇编

	对于简单的汇编代码，也可以使用 GCC 的内联汇编功能：

	```c
	// inline_asm.c
	#include <stdio.h>

	int main() {
		int a = 5, b = 10, result;
		
		__asm__ volatile (
			"addl %%ebx, %%eax;"
			: "=a" (result)
			: "a" (a), "b" (b)
		);
		
		printf("Result: %d\n", result);
		return 0;
	}
	```

	编译：
	```bash
	gcc -m32 -o inline_asm inline_asm.c
	```

	## 5. 64位汇编编译

	对于 x86_64 汇编，编译命令稍有不同：

	```bash
	# NASM 64位
	nasm -f elf64 program.asm -o program.o
	ld -m elf_x86_64 program.o -o program

	# 或者使用 gcc
	gcc -no-pie -o program program.o
	```

	## 常用工具

	### 反汇编查看代码
	```bash
	objdump -d program
	```

	### 调试汇编程序
	```bash
	gdb program
	```

	## 注意事项

	1. **安装必要的库**：对于 32 位程序，可能需要安装 32 位支持库
	   ```bash
	   sudo apt-get install gcc-multilib
	   ```

	2. **语法差异**：GAS 使用 AT&T 语法，NASM 使用 Intel 语法

	3. **系统调用**：不同架构的系统调用号可能不同

	选择哪种方法主要取决于你的偏好和项目需求。NASM 通常更易读，而 GAS 与 GCC 工具链集成更好。

    * 可以使用 masm 吗

        **不推荐在 Linux 下使用 MASM**，因为 MASM 是 Microsoft 的汇编器，主要设计用于 Windows 环境。不过，有几种替代方案：

        ## 1. 为什么 MASM 不适合 Linux

        - **平台限制**: MASM 是 Windows 原生工具
        - **目标文件格式**: MASM 生成 COFF/PE 格式，而 Linux 使用 ELF 格式
        - **系统调用**: MASM 使用 Windows API，Linux 使用不同的系统调用接口

        ## 2. 推荐的替代方案

        ### NASM (最接近 MASM 语法)
        ```bash
        # 安装
        sudo apt-get install nasm

        # 编译示例（语法与 MASM 相似）
        nasm -f elf32 program.asm -o program.o
        ld -m elf_i386 program.o -o program
        ```

        ### YASM (NASM 的改进版)
        ```bash
        # 安装
        sudo apt-get install yasm

        # 使用方式与 NASM 类似
        yasm -f elf32 program.asm -o program.o
        ld -m elf_i386 program.o -o program
        ```

        ## 3. MASM 到 NASM 的语法转换示例

        ### MASM 语法
        ```masm
        .model small
        .stack 100h
        .data
            msg db 'Hello World!$'
        .code
        start:
            mov ax, @data
            mov ds, ax
            mov dx, offset msg
            mov ah, 9
            int 21h
            mov ah, 4ch
            int 21h
        end start
        ```

        ### 对应的 NASM/Linux 语法
        ```nasm
        section .data
            msg db 'Hello World!', 0xa
            msg_len equ $ - msg

        section .text
            global _start

        _start:
            ; write system call
            mov eax, 4          ; sys_write
            mov ebx, 1          ; stdout
            mov ecx, msg        ; message
            mov edx, msg_len    ; message length
            int 0x80

            ; exit system call
            mov eax, 1          ; sys_exit
            mov ebx, 0          ; exit status
            int 0x80
        ```

        ## 4. 如果必须使用 MASM

        ### 通过 Wine 运行（不推荐）
        ```bash
        # 安装 Wine
        sudo apt-get install wine

        # 运行 MASM（需要 Windows 版本的 MASM）
        wine masm.exe program.asm
        ```

        ### 在 Windows 虚拟机中编译
        - 在 VirtualBox/VMware 中运行 Windows
        - 使用 MASM 编译
        - 但这样生成的是 Windows 可执行文件，无法直接在 Linux 运行

        ## 5. 主要语法差异对比

        | 特性 | MASM | NASM |
        |------|------|------|
        | 段定义 | `.data`, `.code` | `section .data`, `section .text` |
        | 字符串 | `db 'Hello$'` | `db 'Hello', 0` |
        | 程序入口 | `start:` + `end start` | `global _start` + `_start:` |
        | 系统调用 | `int 21h` (DOS) | `int 0x80` (Linux) |
        | 变量引用 | `offset msg` | `msg` |

        ## 6. 实用建议

        1. **学习 NASM**: 语法与 MASM 相似，且跨平台
        2. **使用 GCC 内联汇编**: 对于需要与 C 代码混合的情况
        3. **考虑学习 AT&T 语法**: 与 GCC 工具链更好集成

        **推荐使用 NASM**，它是 Linux 下最接近 MASM 体验的汇编器，而且功能强大、文档完善。

    * 既然都是 x86 平台，为什么运行的汇编指令不一样

        这是一个很好的问题！虽然都是 x86 平台，但 Linux 和 Windows 下的汇编编程有几个关键差异：

        ## 1. 操作系统接口不同

        ### 系统调用差异
        ```nasm
        ; Linux 系统调用 (int 0x80 / syscall)
        mov eax, 1    ; sys_exit
        mov ebx, 0    ; 退出码
        int 0x80

        ; Windows API 调用
        push 0        ; 退出码
        call ExitProcess
        ```

        **原因**: 每个操作系统提供不同的系统调用接口来访问硬件资源。

        ## 2. 可执行文件格式不同

        ### Linux: ELF 格式
        ```nasm
        section .text
        global _start    ; Linux 入口点
        _start:
            ; 代码...
        ```

        ### Windows: PE 格式  
        ```masm
        .model flat, stdcall
        .code
        start:          ; Windows 入口点
            ; 代码...
        end start
        ```

        **原因**: 不同的文件格式包含不同的元数据和结构。

        ## 3. 内存管理和段结构

        ### Linux 更简单
        ```nasm
        ; Linux - 平坦内存模型
        section .data   ; 数据段
        section .text   ; 代码段
        ```

        ### Windows 更复杂
        ```masm
        ; Windows - 需要处理更多段细节
        .data           ; 数据段
        .code           ; 代码段
        .stack          ; 堆栈段
        ```

        ## 4. 函数调用约定不同

        ### Linux: cdecl 约定
        ```nasm
        ; 参数通过栈传递，从右到左
        push dword [num2]
        push dword [num1]
        call add_numbers
        add esp, 8      ; 调用者清理栈
        ```

        ### Windows: stdcall 约定
        ```masm
        ; 被调用者清理栈
        push dword [num2]  
        push dword [num1]
        call add_numbers
        ; 栈自动清理
        ```

        ## 5. 实际的硬件指令是相同的

        **核心 x86 指令集是通用的**：
        ```nasm
        ; 这些指令在两种系统上都一样
        mov eax, 10     ; 数据传送
        add ebx, ecx    ; 算术运算
        cmp eax, ebx    ; 比较
        jmp label       ; 跳转
        ```

        ## 6. 示例对比：Hello World

        ### Linux 版本
        ```nasm
        section .data
            msg db 'Hello!',0xa
            len equ $ - msg

        section .text
            global _start

        _start:
            mov eax, 4      ; sys_write
            mov ebx, 1      ; stdout
            mov ecx, msg    ; 字符串
            mov edx, len    ; 长度
            int 0x80        ; 系统调用
            
            mov eax, 1      ; sys_exit
            mov ebx, 0      ; 退出码
            int 0x80
        ```

        ### Windows 版本
        ```masm
        .386
        .model flat, stdcall
        option casemap:none

        include windows.inc
        include kernel32.inc
        includelib kernel32.lib

        .data
            msg db 'Hello!',0

        .code
        start:
            push MB_OK
            push offset caption
            push offset msg
            push 0
            call MessageBoxA
            
            push 0
            call ExitProcess

        caption db 'Hello',0
        end start
        ```

        ## 7. 根本原因总结

        1. **操作系统抽象层**: 操作系统隐藏硬件细节，提供统一的API
        2. **安全隔离**: 防止应用程序直接访问硬件
        3. **可移植性**: 相同的硬件可以运行不同的操作系统
        4. **历史发展**: 不同的开发路径和设计哲学

        ## 8. 类比理解

        可以把 x86 CPU 想象成**通用的发动机**：
        - **Linux** 像是给这个发动机配了**手动变速箱+Linux操控系统**
        - **Windows** 像是配了**自动变速箱+Windows操控系统**
        - **发动机本身（x86指令）** 是相同的，但**操控方式**不同

        这就是为什么需要为不同操作系统重新编译程序，即使硬件平台相同。

    * 那我如果在x86裸机上跑汇编，是否指令就相同的

        **是的！在 x86 裸机上跑汇编，指令是完全相同的！** 这就是最底层的硬件编程。

        ## 1. 裸机汇编的特点

        ### 直接控制硬件
        ```nasm
        ; 这是真正的通用 x86 指令
        mov ax, 0x13        ; 设置 VGA 320x200 256色模式
        int 0x10            ; BIOS 中断调用

        mov ax, 0x0003      ; 文本模式 80x25
        int 0x10
        ```

        ### 没有操作系统抽象层
        ```nasm
        ; 直接与硬件交互
        in al, 0x60         ; 直接从键盘控制器读取
        out 0x20, al        ; 直接向中断控制器写入
        ```

        ## 2. 裸机汇编示例：引导扇区程序

        ```nasm
        ; boot.asm - 简单的引导扇区程序
        org 0x7C00          ; BIOS 加载引导扇区到 0x7C00

        start:
            ; 设置文本模式
            mov ax, 0x0003
            int 0x10
            
            ; 显示字符
            mov ah, 0x0E    ; BIOS 显示字符功能
            mov al, 'H'     ; 字符 'H'
            int 0x10
            mov al, 'i'     ; 字符 'i'
            int 0x10
            
            ; 无限循环
            jmp $
            
            ; 填充引导扇区标记
            times 510-($-$$) db 0
            dw 0xAA55       ; 引导扇区标志
        ```

        编译和运行：
        ```bash
        # 编译
        nasm -f bin boot.asm -o boot.bin

        # 在 QEMU 中运行
        qemu-system-x86_64 -drive file=boot.bin,format=raw
        ```

        ## 3. 裸机编程的层次

        ### BIOS 中断层（传统方式）
        ```nasm
        ; 使用 BIOS 服务
        mov ah, 0x02        ; 读扇区功能
        mov al, 1           ; 扇区数
        mov ch, 0           ; 柱面号
        mov cl, 2           ; 扇区号
        mov dh, 0           ; 磁头号
        mov bx, buffer      ; 缓冲区地址
        int 0x13            ; 磁盘服务中断
        ```

        ### 直接硬件访问
        ```nasm
        ; 直接操作硬件端口
        ; 设置 PIC（可编程中断控制器）
        mov al, 0x11        ; ICW1: 初始化
        out 0x20, al        ; 主 PIC
        out 0xA0, al        ; 从 PIC

        mov al, 0x20        ; IRQ0-7 映射到中断 0x20-0x27
        out 0x21, al
        mov al, 0x28        ; IRQ8-15 映射到中断 0x28-0x2F
        out 0xA1, al
        ```

        ## 4. 保护模式切换示例

        ```nasm
        ; 从实模式切换到保护模式
        cli                 ; 禁用中断

        ; 加载 GDT
        lgdt [gdt_descriptor]

        ; 进入保护模式
        mov eax, cr0
        or eax, 0x1         ; 设置 PE 位
        mov cr0, eax

        ; 远跳转以清空流水线
        jmp CODE_SEG:init_pm

        [bits 32]
        init_pm:
            ; 现在在 32 位保护模式下运行
            mov ax, DATA_SEG
            mov ds, ax
            mov ss, ax
            ; ... 继续执行 32 位代码
        ```

        ## 5. 现代 x86 的复杂性

        虽然指令相同，但现代 x86 更复杂：

        ### 多核启动
        ```nasm
        ; 检测和应用处理器（AP）启动
        mov eax, 0x80000008 ; 获取物理地址大小
        cpuid
        and eax, 0xFF       ; 物理地址位数

        ; 设置 AP 启动代码
        mov edi, 0x1000     ; AP 启动地址
        mov esi, ap_start
        mov ecx, ap_end - ap_start
        rep movsb
        ```

        ### 64 位长模式
        ```nasm
        ; 切换到 64 位模式
        mov eax, cr4
        or eax, 1 << 5      ; 启用 PAE
        mov cr4, eax

        mov eax, PML4       ; 设置页表
        mov cr3, eax

        mov ecx, 0xC0000080 ; EFER MSR
        rdmsr
        or eax, 1 << 8      ; 启用长模式
        wrmsr

        mov eax, cr0
        or eax, 1 << 31     ; 启用分页
        mov cr0, eax
        ```

        ## 6. 裸机开发的工具链

        ### 编译和运行
        ```bash
        # 编译为纯二进制文件
        nasm -f bin kernel.asm -o kernel.bin

        # 使用模拟器测试
        qemu-system-x86_64 -kernel kernel.bin

        # 或写入 USB/磁盘
        sudo dd if=kernel.bin of=/dev/sdX bs=512
        ```

        ## 7. 关键理解点

        1. **指令集相同**: `mov`, `add`, `jmp` 等核心指令完全一样
        2. **编程模型不同**: 
           - 裸机：直接控制所有硬件资源
           - OS 下：通过系统调用间接访问
        3. **初始状态**: CPU 启动时总是处于 16 位实模式
        4. **没有默认服务**: 需要自己实现所有功能

        ## 总结

        **在裸机上，你看到的是真正的、通用的 x86 指令集**。操作系统只是在这个基础上增加了一个抽象层，提供了标准化的服务接口。这也是为什么操作系统开发、嵌入式系统和某些特殊应用需要直接进行裸机编程。

## Hello world

### Console 版本

file: `main.asm`

```asm
.386
.model flat, stdcall
option casemap :none
include \masm32\include\windows.inc
include \masm32\include\kernel32.inc
include \masm32\include\masm32.inc

includelib \masm32\lib\kernel32.lib
includelib \masm32\lib\masm32.lib

.data
    HelloWorld db "Hello World!", 0

.code
start:
    invoke StdOut, addr HelloWorld
    invoke ExitProcess, 0
end start
```

编译：

```
ml /c /Zd /coff main.asm
link /SUBSYSTEM:CONSOLE main.obj
```

运行：

```
./main.exe
```

输出：

```
Hello World!
```

分析：

`.386`表示使用`.386`指令集。也可以使用`.486`，`.586`等等，不过只有`.386`的兼容性最好。

`.model`是告诉编译器内存模型。`flat`表示在 windows 下 near 指针和 far 指针不再有区别。`stdcall`表示从右到左到函数的参数入栈，遵照 windows 的入栈方式。

`option casemap:none`表示强制开启大小写敏感。

`include`表示添加头文件，`includelib`表示添加库文件。

`.data`表示定义被初始化的数据。`.data?`表示定义未初始化的数据，`.const`表示定义常量。

`db`表示 define byte，`0`表示字符串以`NUL`结尾。

`.code`表示定义代码段。

`invoke`，调用函数。`addr`的意思，我猜应该就是指针了吧。


### Windows 版本

Example:

file: `main.asm`

```asm
.386
.model flat,stdcall
option casemap:none
include \masm32\include\windows.inc
include \masm32\include\kernel32.inc
includelib \masm32\lib\kernel32.lib
include \masm32\include\user32.inc
includelib \masm32\lib\user32.lib

.data 
MsgBoxCaption db "My First Win32 Assembly Program", 0
MsgBoxText db "Hello world!", 0

.code
start:
invoke MessageBox, NULL, addr MsgBoxText, addr MsgBoxCaption, MB_OK
invoke ExitProcess, NULL
end start
```

编译命令：

```
ml /c /coff /Cp main.asm
link /SUBSYSTEM:WINDOWS /LIBPATH:c:\masm32\lib main.obj
```

运行：

弹出消息框。

注：

1. `/LIBPATH:c:\masm32\lib`这个编译命令好像并不是必须的。

## Basics

通用寄存器：ax, bx, cx, dx。这些都是 16 位寄存器。它们分别是 eax, ebx, ecx, edx 的低 16 位。

汇编对寄存器名称，汇编指令，辅助关键字（比如表示十六进制的`H`）大小写不敏感。

往寄存器里写入数据：

```asm
mov ax, 1234h
```

每个 16 进制位可以代表 4 个 2 进制位，2 个 16 进制位正好是 1 个字节。

4 个 16 进制位是 2 个字节，代表 16 个二进制位。正好是寄存器`ax`的位宽。

其中`h`表示前面的数字都是 16 进制。不加`h`的话表示默认为十进制。

我们还可以往 32 位寄存器里写入数据：

```asm
mov eax, 12345678h
```

这样我们可以一次写入 4 个字节。如果数据不够 4 个字节，那么前面会自动补 0。

我还可以控制 16 位寄存器的低 8 位和高 8 位，写入单个字节：

```asm
mov al, 12h  ; 向 ax 寄存器的低 8 位写入十六进制数据 12
mov ah, 0abh  ; 写入 ax 的高 8 位。如果字面数值都是字母，那么需要在前面加一个 0，表示这是一个数字，不是一个标识符
```

`add`指令：

```asm
add ax, bx  ; 计算 ax + bx，并将结果存在 ax 中
```

现在操作系统中，`eip`寄存器指向下一条指令的内存地址，但是`cs`寄存器似乎用处不大了。`cs`寄存器在传统汇编语言中，指向代码段的起始地址。

`jmp`指令：

`jmp`可以修改`ip`寄存器的内容，从而到指定地址执行命令。

Syntax:

```asm
jmp <label>
```

Example:

```asm
mov eax, 00401005h
jmp eax  ; 语法：将某个寄存器中的内容移动到 eip 寄存器中
hello: mov eax, 12h
jmp hello  ; jump to the label "hello"
```

好像不能用`mov`命令直接修改`eip`寄存器中的内容，必须用`jmp`修改。我们也无法直接使用字面量来作为`jmp`的参数。

数据段寄存器`ds`好像也无法被直接使用。如果`mov ds, ax`，那么会有运行时错误。现代操作系统差不多已经抛弃`ds`寄存器了。

现代汇编编译器，似乎已经完全放弃`[0]`这种语法了。

常用命令：

* `mov`

    Syntax:

    ```asm
    mov <reg>, <reg>
    mov <reg>, <mem>
    mov <mem>, <reg>
    mov <reg>, <const>
    mov <mem>, <const>
    ```

    `mov`好像不再支持`mov [0], <register>`这样的格式了。会在编译时报错。

    `mov`也不支持直接操作两个内存，必须由一个寄存器做中转。

    ```asm
    mov eax, ebx  ; copy the value in ebx into eax
    mov byte ptr [var], 5  ; store the value 5 into the byte at location var
    ```

* `add`, `sub`

    Syntax:

    ```asm
    add <reg>,<reg>
    add <reg>,<mem>
    add <mem>,<reg>
    add <reg>,<con>
    add <mem>,<con>
    ```
    
    将两个数据相加，并存到第一个操作数中。

    `sub`和`add`差不多。

    Examples:

    ```asm
    add eax, 10  ; EAX ← EAX + 10
    add BYTE PTR [var], 10  ; add 10 to the single byte stored at memory address var
    ```

* `jmp`

* `j<condition>`

    Syntax:

    ```asm
    je <label> (jump when equal)
    jne <label> (jump when not equal)
    jz <label> (jump when last result was zero)
    jg <label> (jump when greater than)
    jge <label> (jump when greater than or equal to)
    jl <label> (jump when less than)
    jle <label> (jump when less than or equal to)
    ```

    Examples:

    ```asm
    cmp eax, ebx
    jle done
    ```

* `cmp`

    Syntax:

    ```asm
    cmp <reg>,<reg>
    cmp <reg>,<mem>
    cmp <mem>,<reg>
    cmp <reg>,<con>
    ```

    Examples:

    ```asm
    cmp DWORD PTR [var], 10
    jeq loop
    ```

* `push`, `pop`

    Syntax:

    ```asm
    push <reg32>
    push <mem>
    push <con32>
    ```

    `esp` (stack pointer) 指向栈顶地址，每执行一次`push`，`esp`减小 4。即`push <data>`中的`<data>`无论如何都占用 4 个字节。如果不够 4 个字节，那么低字节写入数据，高字节填 0。

    `ss`寄存器目前似乎也是无法更改的。

    `ebp` (base poiinter) 基指针。不知道干啥用的。

    Examples:

    ```asm
    push eax  ; push eax on the stack
    push [var]  ; push the 4 bytes at address var onto the stack
    ```

    Syntax:

    ```asm
    pop <reg32>
    pop <mem>
    ```

    Examples:

    ```asm
    pop edi — pop the top element of the stack into EDI.
    pop [ebx] — pop the top element of the stack into memory at the four bytes starting at location EBX.
    ```

* `dup`

    Syntax:

    `<n> dup(<val>)`

    表示`n`个相同的元素，初始化值为`<val>`。如果不初始化，那么填`?`。

* `lea`

    The lea instruction places the address specified by its second operand into the register specified by its first operand.

    汇编代码：

    ```asm
    lea eax, [ebx]
    lea eax, [buf]
    lea eax, ds:[12h]
    ```

    机器指令：

    ```
    00401010 | 8D03          | lea eax,dword ptr ds:[ebx]   | main.asm:19
    00401012 | 8D05 0D404000 | lea eax,dword ptr ds:[<buf>] | main.asm:20
    00401018 | 8D05 12000000 | lea eax,dword ptr ds:[12]    | main.asm:21
    ```

    可以看到，如果第二个操作数是寄存器，那么读取寄存器中的值。如果第二个操作数是变量，那么取变量的**内存地址**。如果第二个操作数是`ds:[constant]`，那么取`constant`这个值。

    如果将第二个操作数写成不加方括号的寄存器，那么会编译错误：

    ```asm
    lea eax, ebx  ; error
    ```

    Examples:

    ```asm
    lea edi, [ebx+4*esi]  ; the quantity EBX+4*ESI is placed in EDI.
    lea eax, [var]  ; the value in var is placed in EAX.
    lea eax, [val]  ; the value val is placed in EAX.
    ```

* `inc`, `dec`

    Syntax:

    ```asm
    inc <reg>
    inc <mem>
    ```

    Examples:

    ```asm
    dec eax  ; subtract one from the contents of EAX.
    inc DWORD PTR [var]  ; add one to the 32-bit integer stored at location var
    ```

* `imul`

    Syntax:

    ```asm
    imul <reg32>,<reg32>
    imul <reg32>,<mem>
    imul <reg32>,<reg32>,<con>
    imul <reg32>,<mem>,<con>
    ```

    Examples:

    ```asm
    imul eax, [var]  ; multiply the contents of EAX by the 32-bit contents of the memory location var. Store the result in EAX.
    imul esi, edi, 25  ; ESI → EDI * 25
    ```

* `idiv`

    The idiv instruction divides the contents of the 64 bit integer EDX:EAX (constructed by viewing EDX as the most significant four bytes and EAX as the least significant four bytes) by the specified operand value. The quotient result of the division is stored into EAX, while the remainder is placed in EDX.

    Syntax:

    ```asm
    idiv <reg32>
    idiv <mem>
    ```

    Examples:

    ```asm
    idiv ebx — divide the contents of EDX:EAX by the contents of EBX. Place the quotient in EAX and the remainder in EDX.
    idiv DWORD PTR [var] — divide the contents of EDX:EAX by the 32-bit value stored at memory location var. Place the quotient in EAX and the remainder in EDX.
    ```

* `and`, `or`, `xor`

    Bitwise logical and, or and exclusive or

    ```asm
    and <reg>,<reg>
    and <reg>,<mem>
    and <mem>,<reg>
    and <reg>,<con>
    and <mem>,<con>
    ```

    Examples:

    ```asm
    and eax, 0fH — clear all but the last 4 bits of EAX.
    xor edx, edx — set the contents of EDX to zero.
    ```

* `not`

    Bitwise Logical Not

    ```asm
    not <reg>
    not <mem>
    ```

    Example:

    ```asm
    not BYTE PTR [var] — negate all bits in the byte at the memory location var.
    ```

* `neg`

    数学取反。

    Syntax:

    ```asm
    neg <reg>
    neg <mem>
    ```

    Examples:

    ```asm
    neg eax — EAX → - EAX
    ```

* `shl`, `shr`

    ```asm
    shl <reg>,<con8>
    shl <mem>,<con8>
    shl <reg>,<cl>
    shl <mem>,<cl>
    ```

    Examples:

    ```asm
    shl eax, 1 — Multiply the value of EAX by 2 (if the most significant bit is 0)
    shr ebx, cl — Store in EBX the floor of result of dividing the value of EBX by 2n wheren is the value in CL.
    ```

* `call`, `ret`

    ```asm
    call <label>
    ret
    ```

    在调用`call`时，会先把下一条要执行的指令的地址`push`到栈中，即`esp`会减小 4。然后才跳转到 procedure 的 entry 地址。

    Example:

    ```asm
    push [var] ; Push last parameter first
    push 216   ; Push the second parameter
    push eax   ; Push first parameter last

    call _myFunc ; Call the function (assume C naming)

    add esp, 12  ; 释放内存


    .486
    .MODEL FLAT
    .CODE
    PUBLIC _myFunc
    _myFunc PROC
    ; Subroutine Prologue
    push ebp     ; Save the old base pointer value.
    mov ebp, esp ; Set the new base pointer value.
    sub esp, 4   ; Make room for one 4-byte local variable.
    push edi     ; Save the values of registers that the function
    push esi     ; will modify. This function uses EDI and ESI.
    ; (no need to save EBX, EBP, or ESP)

    ; Subroutine Body
    mov eax, [ebp+8]   ; Move value of parameter 1 into EAX
    mov esi, [ebp+12]  ; Move value of parameter 2 into ESI
    mov edi, [ebp+16]  ; Move value of parameter 3 into EDI

    mov [ebp-4], edi   ; Move EDI into the local variable
    add [ebp-4], esi   ; Add ESI into the local variable
    add eax, [ebp-4]   ; Add the contents of the local variable
                        ; into EAX (final result)

    ; Subroutine Epilogue 
    pop esi      ; Recover register values
    pop  edi
    mov esp, ebp ; Deallocate local variables
    pop ebp ; Restore the caller's base pointer value
    ret
    _myFunc ENDP
    END
    ```

程序的结构：

* `.data`段

    用来定义数据段，类似于全局变量。

    `db`, `dw`, `dd`可以表示一个字节，两个字节，四个字节。

    Examples:

    ```asm
    .DATA			
    var	DB 64  ; Declare a byte, referred to as location var, containing the value 64.
    var2 DB ?  ; Declare an uninitialized byte, referred to as location var2.
    DB 10  ; Declare a byte with no label, containing the value 10. Its location is var2 + 1.
    X DW ?  ; Declare a 2-byte uninitialized value, referred to as location X.
    Y DD 30000  ; Declare a 4-byte value, referred to as location Y, initialized to 30000.

    Z DD 1, 2, 3  ; Declare three 4-byte values, initialized to 1, 2, and 3. The value of location Z + 8 will be 3.
    bytes DB 10 DUP(?)  ; Declare 10 uninitialized bytes starting at location bytes.
    arr	DD 100 DUP(0)  ; Declare 100 4-byte words starting at location arr, all initialized to 0
    str	DB 'hello', 0  ; Declare 6 bytes starting at the address str, initialized to the ASCII character values for hello and the null (0) byte.
    ```

    定义字符串吋，使用双引号和单引号似乎都行。

    不能用`str`作为变量的名称，它似乎是一个关键字。

与内存交互数据：

一些例子：

```asm
mov eax, [ebx]	; Move the 4 bytes in memory at the address contained in EBX into EAX
mov [var], ebx	; Move the contents of EBX into the 4 bytes at memory address var. (Note, var is a 32-bit constant).
mov eax, [esi-4]	; Move 4 bytes at memory address ESI + (-4) into EAX
mov [esi+eax], cl	; Move the contents of CL into the byte at address ESI+EAX
mov edx, [esi+4*ebx]    	; Move the 4 bytes of data at address ESI+4*EBX into EDX
```

一些错误的例子：

```asm
mov eax, [ebx-ecx]	; Can only add register values
mov [eax+esi+edi], ebx    	; At most 2 registers in address computation
```

* 向内存中写入数据

    汇编代码：

    ```asm
    mov ds:[12345678h], eax
    mov ds:12345678h, eax
    mov ds:123456h, eax
    mov ds:1234h, eax
    ```

    对应的机器码：

    ```
    00401018 | A3 78563412 | mov dword ptr ds:[12345678],eax | main.asm:22
    0040101D | A3 78563412 | mov dword ptr ds:[12345678],eax | main.asm:23
    00401022 | A3 56341200 | mov dword ptr ds:[123456],eax   | main.asm:24
    00401027 | A3 34120000 | mov dword ptr ds:[1234],eax     | main.asm:25
    ```

    可以看到`ds`没有什么用，其实内存地址都是由偏移定义的，高位自动补 0。但是如果不写`ds`的话会报错：

    ```asm
    mov [12345678h], eax  ; error
    ```

    如果不想写`ds`，可以用寄存器来寻扯：

    ```asm
    mov eax, 12345678h
    mov [eax], ebx  ; OK
    ```

    对应的机器码：

    ```
    0040102C | B8 78563412 | mov eax,12345678           | main.asm:27
    00401031 | 8918        | mov dword ptr ds:[eax],ebx | main.asm:28
    ```

    另外可以看到机器码的低地址对应数据的低字节。内存模型的表示是地址左低右高，而我们写的字面量是左高右低，所以顺序正好是相反的。

    但是我们无法这样做：

    ```asm
    mov [eax], 1234h
    ```

    只能这样：

    ```asm
    mov ebx, 1234h
    mov [eax], ebx
    ```

    也就是说，内存只能和寄存器交互，立即数只能和寄存器交互。内存无法和内存交互，也无法和立即数交互。

* 从内存中取出数据

    汇编代码：

    ```asm
    mov ax, buf
    mov ax, [buf]
    ```

    对应的机器码：

    ```
    00401010 | 66:A1 0D404000 | mov ax,word ptr ds:[<buf>] | main.asm:19
    00401016 | 66:A1 0D404000 | mov ax,word ptr ds:[<buf>] | main.asm:20
    ```

    可以看到他们是完全相同的。我们也可以在他们的前面加上`ds:`。

数据大小

如果编译器无法推断出来我们需要对多少字节进行操作，那么我们需要指明数据的大小：

```asm
mov BYTE PTR [ebx], 2	; Move 2 into the single byte at the address stored in EBX.
mov WORD PTR [ebx], 2	; Move the 16-bit integer representation of 2 into the 2 bytes starting at the address in EBX.
mov DWORD PTR [ebx], 2    	; Move the 32-bit integer representation of 2 into the 4 bytes starting at the address in EBX.
```

labeL:

```asm
       mov esi, [ebp+8]
begin: xor ecx, ecx
       mov eax, [esi]
```

## Ref

* `.model`

    Syntax:

    ```asm
    .MODEL memory-model ⟦, language-type⟧ ⟦, stack-option⟧
    ```

    Parameters:

    * `memory-model`

        Required parameter that determines the size of code and data pointers.

        32-bit values: `FLAT`

        16-bit values (support for earlier 16-bit development): `TINY, SMALL, COMPACT, MEDIUM, LARGE, HUGE, FLAT`

    * `language-type`

        Optional parameter that sets the calling and naming conventions for procedures and public symbols.

        32-bit values: `C, STDCALL`

        16-bit values: `C, BASIC, FORTRAN, PASCAL, SYSCALL, STDCALL`

    * `stack-option`

        Optional parameter. stack-option is not used if memory-model is FLAT.

        32-bit values: Not used

        16-bit values: `NEARSTACK, FARSTACK`

    Ref: <https://learn.microsoft.com/en-us/cpp/assembler/masm/dot-model?view=msvc-170>

## Miscellaneous

1. 寄存器比如`eax`，`ebx`，`ecx`这些经常会被改变，尤其是在函数调用前后。所以不要将数据放在这些寄存器中，尽量将数据都放到内存中。寄存器的作用域尽量控制在它的 3，4 行上下文中。寄存器的作用尽量仅限于与内存交互数据，做算术运算，指令跳转等。

1. 如果想让函数返回一个值，可以把这值存到`eax`寄存器中。

    Ref: <https://stackoverflow.com/questions/55773868/returning-a-value-in-x86-assembly-language>

1. 调试 debug

    目前直接对汇编程序进行调试的主流调试器有 x64dbg 和 bochs。

    如果需要 C/C++ 生成的汇编程序进行调试，可以使用 gdb。

    在 visual studio 里似乎也可以调试，但我目前还不会用。<https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/debugging-in-assembly-mode>

    
