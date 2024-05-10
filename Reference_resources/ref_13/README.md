# usage

`insmod`之后，编译运行`user_prog.c`，可以看到 dmesg 的输出：

```
[98766.822998] in mod_init()...
[98766.823273] successfully create hlc dev
[98766.823274] init waitqueue
[98766.823444] wake up process
[98766.823446] start print_msg thread
[98766.823450] wait condition variable...
[98776.886899] in h_open()...
[98776.886902] in h_write()...
[98776.886911] condi is 2, hello, world
[98776.886912] rechange condi to 1
[98776.886930] in h_release()...
```