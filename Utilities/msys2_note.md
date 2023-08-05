# MSYS2 Note

官网：<https://www.msys2.org/>

基本使用方法：<https://stackoverflow.com/questions/30069830/how-to-install-mingw-w64-and-msys2>

一个可能的教程：<https://wiki.openttd.org/en/Archive/Compilation%20and%20Ports/Compiling%20on%20Windows%20using%20MSYS2>

MSYS2 似乎提供了多个编译环境，原理好像是为每个编译环境建立了一个文件夹，然后文件夹里有配套的各种二进制文件。

如果我们想要用 mingw64 环境下的编译器，可以先打开 msys2 mingw64，然后

```bash
pacman -Ss gcc  # or use pacman -Ss 'mingw-w64.*gcc'
```

找到一个`mingw64/mingw-w64-x86_64-gcc`，然后使用

```bash
pacman -S mingw64/mingw-w64-x86_64-gcc
```

安装即可。

这个命令会同时装上 g++ 和 gcc。

装 gdb:

```bash
pacman -S mingw64/mingw-w64-x86_64-gdb
```

如果只用上面的命令安装，是不会安装 make 程序的。不如一步到位：`pacman -S mingw-w64-x86_64-toolchain`
