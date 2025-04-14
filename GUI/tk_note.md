# TK Note

## cache

* tk 本来是 tcl 语言的一个 gui 库，由于比较好用，所以移植到了其他语言上。

    tkinter 指的是 tk interface。

* 判断当前 python 环境中是否安装有 tkinter：
    
    `python -m tkinter`

* tk app: hello world

    `hello_world.py`:

    ```python
    import tkinter as tk

    def say_hello():
        label.config(text="Hello, World!")

    root = tk.Tk()

    label = tk.Label(root, text="Click the button to say hello!")
    label.pack()

    button = tk.Button(root, text="Say hello", command=say_hello)
    button.pack()

    root.mainloop()
    ```

    run: `python hello_world.py`

    result:

    ![](../../Reference_resources/ref_38/pic_1.png)

    单击 say hello 按钮后，界面变成：

    ![](../../Reference_resources/ref_38/pic_2.png)

    `root = tk.Tk()`这个可能是代表主窗口的对象。后面的逻辑比较清晰了，创建 label, button 对象的时候，先指定 parent 对象，然后通过`.pack()`方法将自己添加到 parent 中。不清楚这个`pack()`是否和 layout 相关，目前看来，新添加的元素都是从上向下垂直添加的。最后通过`root.mainloop()`开始主窗口的事件循环。

## note