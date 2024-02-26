# Python Note

## cached

* python use `shutil` to copy file

    ```cpp
    import shutil

    def main():
        shutil.copyfile('./test_1.txt', './test_2.txt')

    if __name__ == '__main__':
        main()
    ```

    * <https://stackoverflow.com/questions/123198/how-to-copy-files>

    * <https://www.freecodecamp.org/news/python-copy-file-copying-files-to-another-directory/>

* python delete a file

    <https://blog.enterprisedna.co/delete-files-from-python/>

## pypi mirror

在上海使用上交的镜像比较快：<https://mirrors.sjtug.sjtu.edu.cn/docs/pypi/web/simple>

临时使用：`pip install -i https://mirror.sjtu.edu.cn/pypi/web/simple numpy`

## Miscellaneous

1. 播放 mp3 文件时，`playsound`库不好用，在 windows 下会出现无法解码 gb2312 的问题。可以用`vlc`库代替。但是`vlc`似乎不支持阻塞式播放。

1. 一个文件作为模块运行时，才能相对导入，比如`from ..package.module import some_class`。
    
    让一个文件作为模块运行有两种方法，一种是运行其他 python 文件，让其他 python 文件把这个文件作为模块或包导入；另一种是直接使用`python -m xxx.py`运行当前文件。

    相对导入也是有极限的，那就是它只能把主脚本所在的目录作为顶级包，无法再向上查找。或者说，它只能找到`__name__`中指向的顶级包。

    假如一个工程项目`proj`目录，里面有`subpack_1`和`subpack_2`两个子包，然后`subpack_1`中有一个模块文件`mod_1.py`，`subpack_2`中有一个模块文件`mod_2.py`。想直接从`mod_1`直接调用`mod_2`是不可能的。要想调用，只有一种办法，那就是在`proj`下创建一个新文件`script.py`，然后在这个文件中，使用

    ```py
    import sys
    sys.path.append('./')

    from subpack_1 import mod_1
    ```

    把当前目录加入到搜索目录中，然后再在这个文件中运行`mod_1`中的代码。

    不加`sys.path.append('./')`是不行的，因为我们直接运行的是`script.py`，所以`proj`目录被作为顶层目录。然而顶层目录并不会被作为一个包，因此`mod_1`向上找最多只能找到`subpack_1`这里，而无法看到`subpack_2`。为了让`mod_1`看到`subpack_2`，还需要将当前目录加入到搜索目录中。

1. 将 c++ 文件编译为`.pyd`文件，获取当前系统的后缀的方法：

    * linux: `python3-config --extension-suffix`

    * windows: `python -c "from distutils import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`

1. python 3.1 之后，不再有`unichr()`函数，直接使用`chr()`函数就可以了。把一个整数转换成一个 Unicode 字符。

1. `PYTHONPATH`的作用

    `PYTHONPATH`中的内容会被添加到`sys.path`中，使用冒号`:`分隔不同路径（如果是 windows 系统怎么办？比如`C:/path_1:D:/path_2`这样的）。

    我们使用`import <package_name>`时，会在`sys.path`路径下搜索。

    Ref: <https://www.simplilearn.com/tutorials/python-tutorial/python-path>

1. 有关`.pyd`文件

    <https://stackoverflow.com/questions/50278029/how-to-import-a-pyd-file-as-a-python-module>

    这个资料里说 pyd 其实就是 dll，没法直接 import。

    实际上，可以使用`importlib`这个库导入 pyd 文件。有时间了研究一下。

1. python 中 dict 类型对象的`.copy()`方法是 shallow copy

    ```python
    d = {}
    d['hello'] = 'world'
    d['1'] = '2'

    d_1 = d  # 只是创建了一个 d 的引用，并没有做值拷贝
    d_1['1'] = '3'  # 改变 d_1 的内容会影响 d 的内容
    print(d['1'])  # 3

    d['1'] = '2'
    d_2 = d.copy()  # shallow copy
    d_2['1'] = '3'  # 改变 d_2 的内容不再影响 d
    print(d['1'])  # 2

    from copy import deepcopy
    my_data = [1, 2, 3]
    d['1'] = my_data  # d['1'] 是 my_data 的一个引用，并不拥有自己的数据
    d_3 = d  # d_3 是 d 的一个引用，因此和 d 等价
    d_4 = d.copy()  # d_4 是 d 的一个浅拷贝，因此 d_4['1'] = xxx 不影响 d['1']，但是 d_1['1'][xx] = xxx 会影响 d
    d_5 = deepcopy(d)  # d_5 与 d 完全互不影响
    d_3['1'][0] = 2  # 影响 my_data 中的内容
    d_4['1'][1] = 3  # 影响 my_data 中的内容
    d_5['1'][2] = 4  # 不影响 my_data 中的内容
    print(d['1'])  # [2, 3, 3]
    ```