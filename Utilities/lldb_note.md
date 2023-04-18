# LLDB Note

lldb 是 llvm 项目下的一个子项目，被称为下一代 gdb。

官网教程，格式化输出变量：<https://lldb.llvm.org/use/variable.html>

tutorial: <https://lldb.llvm.org/use/tutorial.html>

## Installation

官网：<https://github.com/llvm/llvm-project>

* windows

    可以在 github 项目的 release 页面找到 llvm 的 exe 安装包。直接把这个安装包下载下来安装就好了。

    llvm 是依赖 python 的，需要留意下其依赖的 python 的版本。目前依赖的 python 的版本是 3.10。

## Basic usage

1. 首先需要设置`PYTHONHOME`环境变量。

    在 cmd 下的话可以使用`set PYTHONHOME='C:\Users\wsdlh\AppData\Roaming\miniconda3\envs\llvm'`设置。在 powershell 下可以使用`$env:PYTHONHOME = 'C:\Users\wsdlh\AppData\Roaming\miniconda3\envs\llvm'`设置。

    如果`PYTHONHOME`不被设置，那么直接运行`lldb`会报错：

    ```
    lldb Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encodingPython runtime state: core initialized
    ```

1. 接下来的使用步骤和`gdb`差不多。可以用`lldb <exe_file_path>`启动，也可以直接运行`lldb`，然后使用`file <exe_file_path>`载入文件。

## 乱码问题

lldb 没有 gdb 那样的改变字符集的方法，它是使用 python 程序来处理输出的。

如果某个字符串是 UTF-8 编码，然而 terminal 输出时用的是 CP936（GBK/GB2312）编码，那么就有可能乱码。一个处理编码的方法：

1. 先创建一个文件`C:\Users\wsdlh\lldb\wcharsummary.py`

    ```py
    import lldb
    def wchar_SummaryProvider(valobj, dict):
    e = lldb.SBError()
    s = u'"'
    if valobj.GetValue() != 0:
        i = 0
        newchar = -1
        while newchar != 0:
        # read next wchar character out of memory
        data_val = valobj.GetPointeeData(i, 1)
        size = data_val.GetByteSize()
        # print('size: ', size)
        if size == 1:
            newchar = data_val.GetUnsignedInt8(e, 0)    # utf-8
        elif size == 2:
            newchar = data_val.GetUnsignedInt16(e, 0)   # utf-16
            # print('newchar: ', newchar)
        elif size == 4:
            newchar = data_val.GetUnsignedInt32(e, 0)   # utf-32
        else:
            return '<error>'
        # print('here')
        if e.fail:
            return '<error>'
        i = i + 1
        # add the character to our string 's'
        if newchar != 0:
            s = s + chr(newchar)
    s = s + u'"'
    b = s.encode('utf-8')
    c = s.encode('utf-8').decode().encode('gbk')
    # print(b)
    # print(c)
    return s.encode('utf-8').decode()
    ```

    Ref: <https://stackoverflow.com/questions/12923873/how-to-print-wchar-t-string-in-lldb>

1. 再创建 lldb 的配置文件`C:\Users\wsdlh\.lldbinit`

    ```
    command script import ~/lldb/wcharsummary.py
    type summary add -F wcharsummary.wchar_SummaryProvider "wchar_t *"
    ```

然而实际上，发现最重要的问题在 windows 上。如果不手动设置，powershell 会一直强制使用 GBK 字符集进行输出，无论是否设置`chcp 65001`，`chcp 936`，都会一律使用 GBK。

在控制面板 -> 时钟和区域 -> 区域 -> 管理 -> 更改系统区域设置 -> Beta 版：使用 Unicode UTF-8 提供全球语言支持。然后重启系统。这时候 powershell 会使用 UTF-8 进行输出。

Ref:

1. <https://stackoverflow.com/questions/69339312/why-does-debugging-with-lldb-in-vscode-garble-chinese-characters>


前面的 python 处理字符串有个我没想明白的地方。`s.encode('utf-8')`是用 utf-8 编码的字符串，如果我将其改成`s.encode('utf-8').decode().encode('gbk').decode(encoding='gbk')`，那么其实底层的数据已经变成 gbk 编码的字节了。可是把这样替换过的字符串传给 lldb 之后，输出依然是乱码，没有变化。为什么？