# pynccl note

* pynccl 的一个 example

    pynccl 只是 c 的一个 wrapper，基本就是在直接调用 c 函数，因此需要 ctypes 去做辅助。

    一个 init example:

    ```py
    import pynccl
    from ctypes import *

    def main():
        nccl = pynccl.Nccl()
        cu_nccl = nccl._nccl
        lib_nccl = nccl.api

        print(cu_nccl)
        print(lib_nccl)
        
        type_arr_2_c_int = c_int * 2
        dev_ids = type_arr_2_c_int(0, 1)
        dev_ids_ptr = cast(dev_ids, POINTER(c_int))

        type_arr_2_c_void_p = c_void_p * 2
        communicators = type_arr_2_c_void_p(0, 0)
        communicators_ptr = cast(communicators, POINTER(c_void_p))

        ret = lib_nccl.ncclCommInitAll(communicators_ptr, 2, dev_ids_ptr)
        print(ret)
        return
    ```

    output:

    ```
    <pynccl.binding.cuNccl object at 0x7f1889cbad50>
    <pynccl.binding.libnccl object at 0x7f188983a690>
    NCCL version 2.22.3+cuda12.1
    0
    ```

    目前不清楚 cu nccl 和 lib nccl 有什么区别。

    为了调用`ncclCommInitAll()`，引入了`ctypes`。看起来`c_int * 2`和`c_void_p * 2`都是定义了个类型，类似于`typedef xxx`。

    `type_arr_2_c_int(0, 1)`看起来有点像 c 语言里的变量/数组定义 + 初始化。

    `cast(dev_ids, POINTER(c_int))`有点像将数组的名字转换为指定的指针。