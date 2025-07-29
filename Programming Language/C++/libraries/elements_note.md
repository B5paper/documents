# Elements GUI Note

## cache

* c++ elements gui

    repo: <https://github.com/cycfi/elements>

    声明式语法：Domain Specific Embedded Languages (DSEL)

    install:

    `git clone --recurse-submodules  https://github.com/cycfi/elements.git`

    linux 上 elements 依赖 GTK3

    ```bash
    sudo apt-get install libcairo2-dev
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libwebp-dev
    ```

    compile:

    ```bash
    cd elements
    mkdir build
    cd build
    cmake -G "Unix Makefiles" ../
    ```

    编译完成后，可进入`build/examples`文件夹中运行各个 example 程序。

    `hello_universe`简析：

    ```cpp
    #include <elements.hpp>

    using namespace cycfi::elements;

    int main(int argc, char* argv[])
    {
       app _app("Hello Universe");
       window _win(_app.name());
       _win.on_close = [&_app]() { _app.stop(); };

       view view_(_win);

       view_.content(
          scroller(image{"space.jpg"})
       );

       _app.run();
       return 0;
    }
    ```

    `app`看起来是做事件循环的。

    `window`看起来是只有在关闭时，才触发调用`_app.stop()`，那么 windows 是如何和 app 绑定到一起的？看起来应该是靠 app name 的字符串。（这是否意味着，不给 window 传入`_app.name()`，即使传入 c-style 字符串，window 也能正常 work？）

    `view`与 window 做了绑定，看起来应该是 view 只负责填充内容，app 的消息交给 window，window 将 app 的消息和 window 本身的消息都交给 view 处理。

    `_app.run()`进入主事件循环，没有什么好说的。

    `_win.on_close = [&_app]() { _app.stop(); };`说明 window 关闭的时候，app 不一定关闭，还可能在后台处理消息。

    compile example:

    ```bash
    cd hello_universe/build
    cmake -DELEMENTS_ROOT=/home/hlc/Documents/Projects/elements ..
    ```

    需要配置一下`ELEMENTS_ROOT`，否则会报错。

## note
