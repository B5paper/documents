# magic enum note

## cache

* `magic_enum`简介

    `magic_enum`是一个第三方 c++ 库，官网：<https://github.com/Neargye/magic_enum>

    这是个 header-only 的库，只需要 include 头文件就可以。

    `main.cpp`:

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <array>
    #include "magic_enum/magic_enum.hpp"
    using namespace std;

    enum Color {
        RED,
        YELLOW,
        GREEN,
        BLUE
    };

    int main() {
        // enum -> str
        string_view color_name = magic_enum::enum_name(GREEN);
        printf("color name: %s\n", color_name.data());

        // str -> enum
        const optional<Color> &my_color_opt =
            magic_enum::enum_cast<Color>("BLUE");
        if (!my_color_opt.has_value()) {
            printf("fail to find value\n");
            return -1;
        }
        const Color &color = my_color_opt.value();
        printf("BLUE val is %d\n", color);

        // get enum num
        size_t num_entry = magic_enum::enum_count<Color>();
        printf("Color num entry: %lu\n", num_entry);

        // traverse all enum entries
        // const array<Color, 4>&
        auto &colors = magic_enum::enum_values<Color>();
        for (int i = 0; i < colors.size(); ++i) {
            printf("%d, ", colors[i]);
        }
        putchar('\n');

        // traverse all enum entries by name str
        // const array<string_view, 4>&
        auto &entry_names = magic_enum::enum_names<Color>();
        for (int i = 0; i < entry_names.size(); ++i) {
            printf("%s, ", entry_names[i].data());
        }
        putchar('\n');

        // enum -> idx
        // const optional<size_t> &yellow_idx_opt = magic_enum::enum_index(YELLOW);
        const auto &yellow_idx_opt = magic_enum::enum_index(YELLOW);
        if (!yellow_idx_opt.has_value()) {
            printf("fail to find yellow\n");
            return -1;
        }
        printf("YELLOW idx: %lu\n", yellow_idx_opt.value());

        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    INCLUDE=../magic_enum_test/magic_enum/include

    main: main.cpp
    	g++ -g main.cpp -I${INCLUDE} -o main
    ```

    compile: `make`

    run: `./main`

    output:

    ```
    color name: GREEN
    BLUE val is 3
    Color num entry: 4
    0, 1, 2, 3, 
    RED, YELLOW, GREEN, BLUE, 
    YELLOW idx: 1
    ```

    这几个都比较常用。

    说明：

    1. `magic_enum::enum_name()`只能返回`string_view`，不能返回`string`。

    1. 将 string 转换为 enum 时，`magic_enum::enum_cast<Color>("BLUE")`必须要使用`<EnumType>`指定 enum 的类型，这点比较好理解，仅从字符串无法拿到任何类型信息。

    1. `enum_cast<>()`的返回值是个`std::optional`。可以按值返回，如果要返回引用，那么只能返回 const 左值引用。这么麻烦还不如直接用`auto`。

    1. `enum_values<Color>()`返回的是一个`const array<Color, 4>&`，但是我们想要拿到的，正是这个模板参数 4，所以这里只能用`auto`来表示返回值类型。

        如果使用`auto`,则返回值。如果使用`auto&`，则返回 const 左值引用。

    1. `enum_names<Color>()`与`enum_values<Color>()`同理，返回的是一个包含`string_view`的`array`。

    1. `const auto &yellow_idx_opt = magic_enum::enum_index(YELLOW);`这里比较奇怪，`enum_index()`返回类型，使用`const optional<size_t> &`可以通过编译，使用`const auto&`也可以通过编译，但是使用`auto&`不行。目前不清楚为什么。

    整体看来，`magic_enum`的模板复杂度与它实现的功能不匹配，到项目里容易出错。能不用尽量不要使用。

## note