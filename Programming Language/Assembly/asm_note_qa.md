# Assembly Note QA

[unit]
[u_0]
假如一个内存地址为`12345678h`，请向这个地址的内存中移入一个字节的数据`0abh`。
[u_1]
```asm
; method 1
mov eax, 0abh
mov ds:[12345678h], eax

; method 2
mov eax, 0abh
mov ds:12345678h, eax

; method 3
mov eax, 12345678h
mov ebx, 0abh
mov [eax], ebx
```

[unit]
[u_0]
请用 masm 汇编写一个 procedure，用于反转`.data`区的字符串。
```asm
.data
m_str db "hello, world", 10, 0
```
[u_1]
(empty)

[unit]
[u_0]
请用 masm 汇编输出 0 到 9 的所有数字。
[u_1]
```asm
.386
.model flat, stdcall
include \masm32\include\windows.inc
include \masm32\include\masm32.inc
include \masm32\include\kernel32.inc
includelib \masm32\lib\masm32.lib
includelib \masm32\lib\kernel32.lib

.data
num db '0', 0
count db 0

.code
start:
mov ecx, 0
loop_start:
invoke StdOut, addr num
add [num], 1
inc [count]
cmp [count], 10
jl loop_start
invoke ExitProcess, 0
end start
end
```

[unit]
[u_0]
请用 masm 汇编输出 hello, world 字符串。
[u_1]
```asm
.386
.model flat, stdcall
option casemap:none
include \masm32\include\windows.inc
include \masm32\include\kernel32.inc
includelib \masm32\lib\kernel32.lib
include \masm32\include\masm32.inc
includelib \masm32\lib\masm32.lib

.data
hello_str db 'hello, world!', 0ah, 0

.code
start:
invoke StdOut, addr hello_str
invoke ExitProcess, 0
end start
end
```

[unit]
[u_0]
`lea`命令是什么意思？请解释它的作用，并给出几个例子。
[u_1]
(empty)

[unit]
[u_0]
请用 masm 汇编分别实现`while ()`和`do while()`。
[u_1]
(empty)

[unit]
[u_0]
请写出`.model`的 syntax，并写出 32-bit 下各个参数的可取的值。
给出一个 example。
[u_1]
```asm
.MODEL memory-model ⟦, language-type⟧ ⟦, stack-option⟧
```

`memory-model`: 32 位下可取值：`FLAT`

`language-type`: 32 位下可取值：`C, STDCALL`

`stack-option`: Not used

Example:

`.model flat, stdcall`

