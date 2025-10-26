# NASM Note

## cache

* nasm

    安装:
    
    `sudo apt install nasm`

    验证安装：

    `nasm -v`

    hello, world example:

    `hello.asm`:

    ```nasm
    section .data
        hello db 'Hello, World!', 0xa    ; 字符串和换行符
        hello_len equ $ - hello          ; 计算字符串长度

    section .text
        global _start

    _start:
        ; 写入系统调用
        mov eax, 4          ; 系统调用号 (sys_write)
        mov ebx, 1          ; 文件描述符 (stdout)
        mov ecx, hello      ; 字符串地址
        mov edx, hello_len  ; 字符串长度
        int 0x80            ; 调用内核

        ; 退出系统调用
        mov eax, 1          ; 系统调用号 (sys_exit)
        mov ebx, 0          ; 退出状态
        int 0x80            ; 调用内核
    ```

    compile:

    `nasm -f elf64 hello.asm -o hello.o`

    link:

    `ld hello.o -o hello`

    run:

    `./hello`

    output:

    ```
    Hello, World!
    ```

## topics
