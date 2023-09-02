# ar notes

`ar cr libxxxx.a file1.o file2.o`

其中，

* `c`: create, 如果库不存在，则创建库
* `r`: replace，如果库中已存在要添加的对象文件，则旧的对象文件将被替换

ar 只是一个打包工具，是 archive 的首字母，它将一系列的目标文件首位连接在一起，并内嵌一个索引表，使得编译器能够方便地找到所需要的函数。一般来说，由于函数索引表的存在，对库的链接要比一般的对象文件的链接更快。

如果 ar 未能完成此项索引表工作，还可以手动用以下的`ranlib`命令创建索引表。

`ranlib`: generate index to archive

usage: `ranlib libxxx.a`
