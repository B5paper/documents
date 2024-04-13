# slint note

slint 是 rust 的一个 gui 框架，可以跨平台。

official site: <https://slint.dev/>

rust docs: <https://releases.slint.dev/1.5.1/docs/rust/slint/>

github repo: <https://github.com/slint-ui/slint>

## cache

* slint 下拉菜单

    slint 下拉菜单组件叫做`ComboBox`

    其属性有：

    * `current-index`: (in-out int): The index of the selected value (-1 if no value is selected)

        `current-index`从 0 开始计数。

    * `current-value`: (in-out string): The currently selected text

    * `enabled`: (in bool): Defaults to true. When false, the combobox can’t be interacted with

    * `has-focus`: (out bool): Set to true when the combobox has keyboard focus.

    * `model` (in [string]): The list of possible values

    callbacks:

    * `selected(string)`: A value was selected from the combo box. The argument is the currently selected value.

    Example:

    ```slint
    import { ComboBox } from "std-widgets.slint";
    export component Example inherits Window {
        width: 200px;
        height: 130px;
        ComboBox {
            y: 0px;
            width: self.preferred-width;
            height: self.preferred-height;
            model: ["first", "second", "third"];
            current-value: "first";
        }
    }
    ```

    自己写的一个 example:

    `main.rs`:

    ```rs
    slint::slint! {
        import { AppWindow } from "ui.slint";
    }

    fn combo_box_selected(s: slint::SharedString, idx: i32) {
        println!("selected {s}, index: {idx}");
    }

    fn main() {
        let app = AppWindow::new().unwrap();
        app.on_combo_box_selected(combo_box_selected);
        app.run().unwrap();
    }
    ```

    `ui.sint`:

    ```slint
    import { Button, VerticalBox, ComboBox } from "std-widgets.slint";

    export component AppWindow inherits Window {
        in-out property<int> counter: 42;
        callback request-increase-value();
        callback combo_box_selected(string, int);
        VerticalBox {
            Text {
                text: "Counter: \{root.counter}";
            }
            Button {
                text: "Increase value";
                clicked => {
                    root.request-increase-value();
                }
            }
            combo_box := ComboBox {
                model: ["first", "second", "third"];
                selected(val) => {
                    combo_box_selected(val, self.current-index);
                }
            }
            Rectangle {
                height: 100px;
            }
        }
    }
    ```

* slint 中 function 要求定义函数体，callback 不要求定义函数体

* slint 中 Image 不可以跨线程，只能用`SharedPixelBuffer`存储数据跨线程，然后再在 slint 所在的线程里转换成`Image`。

    这一点在官方文档里有写。

* slint 中如果 main window 所在的线程没有退出，那么 main window 在退出 event loop 后，窗口并不会消失。

    不清楚为什么。

* slint 中，如果需要在 ui 和 rs 中间传递数据，必须使用 slint 内置定义的一些数据类型，比如`SharedString`。

    通常 rust 的内置数据类型可以使用`into()`自动转换过去。比如`"hello".to_string().into()`。

* slint 中不可以将 mainwindow 直接传给其他线程，用`arc + mutex`也不行。

    但是可以使用 weak ptr + event loop 来多线程传递消息。

    官方的 example 如下：

    ```rust
    slint::slint! { export component MyApp inherits Window { in property <int> foo; /* ... */ } }
    let handle = MyApp::new().unwrap();
    let handle_weak = handle.as_weak();
    let thread = std::thread::spawn(move || {
        // ... Do some computation in the thread
        let foo = 42;
        // now forward the data to the main thread using invoke_from_event_loop
        let handle_copy = handle_weak.clone();
        slint::invoke_from_event_loop(move || handle_copy.unwrap().set_foo(foo));
    });
    handle.run().unwrap();
    ```

    ref: <https://releases.slint.dev/1.5.1/docs/rust/slint/fn.invoke_from_event_loop.html>

* slint 可以在 rs 中使用`mainwindow.invoke_<func_name>(<params>)`调用一个在 slint 中定义的函数

    也可以使用`mainwindow.on_<callback_name>(<params>)`将 slint 中声明的函数映射到 rs 文件的定义中。

* slint `HorizontalBox`与`HorizontalLayout`的关系

    A `HorizontalBox` is a `HorizontalLayout` where the spacing and padding values depend on the style instead of defaulting to 0.

    看起来`HorizontalLayout`是将 padding 和 spacing 都设置成 0，而`HorizontalBox`是动态可调的。

    那么每次只需要使用 box 就可以了。

* slint 中`_`和`-`等价

* slint 中可以使用`Colors.`访问到各种内置颜色，比如`Colors.red`

* slint 常用类型

    * struct

        ```slint
        export struct Player  {
            name: string,
            score: int,
        }
        ```

    * array

        ```slint
        export component Example {
            in-out property<[int]> list-of-int: [1,2,3];
            in-out property<[{a: int, b: string}]> list-of-structs: [{ a: 1, b: "hello" }, {a: 2, b: "world"}];
        }
        ```

        operations on an array:

        * `array.length`: One can query the length of an array and model using the builtin .length property.

        * `array[index]`: The index operator retrieves individual elements of an array.

* slint property

    ```slint
    export component Example {
        // declare a property of type int with the name `my-property`
        property<int> my-property;

        // declare a property with a default value
        property<int> my-second-property: 42;
    }
    ```

    ```slint
    export component Button {
        // This is meant to be set by the user of the component.
        in property <string> text;
        // This property is meant to be read by the user of the component.
        out property <bool> pressed;
        // This property is meant to both be changed by the user and the component itself.
        in-out property <bool> checked;

        // This property is internal to this component.
        private property <bool> has-mouse;
    }
    ```

* slint 中可以用这种方式实现相对长度

    ```slint
    export component Example inherits Window {
        preferred-width: 100px;
        preferred-height: 100px;

        background: green;
        Rectangle {
            background: blue;
            width: parent.width * 50%;
            height: parent.height * 50%;
        }
    }
    ```

    如果 parent element 的长／宽固定，那么可以简写上面的：

    ```slint
    export component Example inherits Window {
        preferred-width: 100px;
        preferred-height: 100px;

        background: green;
        Rectangle {
            background: blue;
            width: 50%;
            height: 50%;
        }
    }
    ```

* slint function

    slint 中的函数使用`function`关键字定义。

    ```slint
    export component Example {
        in-out property <int> min;
        in-out property <int> max;
        protected function set-bounds(min: int, max: int) {
            root.min = min;
            root.max = max
        }
        public pure function inbound(x: int) -> int {
            return Math.min(root.max, Math.max(root.min, x));
        }
    }
    ```

* slint callback

    callback 可以使用`=>`定义：

    ```slint
    export component Example inherits Rectangle {
        // declare a callback
        callback hello;

        area := TouchArea {
            // sets a handler with `=>`
            clicked => {
                // emit the callback
                root.hello()
            }
        }
    }
    ```

    可以带参数：

    ```slint
    export component Example inherits Rectangle {
        // declares a callback
        callback hello(int, string);
        hello(aa, bb) => { /* ... */ }
    }
    ```

    也可以有返回值：

    ```slint
    export component Example inherits Rectangle {
        // declares a callback with a return value
        callback hello(int, int) -> int;
        hello(aa, bb) => { aa + bb }
    }
    ```

* slint plotter slint 初探

    `plotter.slint`:

    ```slint
    // Copyright © SixtyFPS GmbH <info@slint.dev>
    // SPDX-License-Identifier: MIT

    import { Slider, GroupBox, HorizontalBox, VerticalBox } from "std-widgets.slint";

    export component MainWindow inherits Window {
        in-out property <float> pitch: 0.15;
        in-out property <float> yaw: 0.5;

        pure callback render_plot(/* pitch */ float, /* yaw */ float, /* amplitude */ float) -> image;

        title: "Slint Plotter Integration Example";
        preferred-width: 800px;
        preferred-height: 600px;

        VerticalBox {
            Text {
                font-size: 20px;
                text: "2D Gaussian PDF";
                horizontal-alignment: center;
            }

            Image {
                source: root.render_plot(root.pitch, root.yaw, amplitude-slider.value / 10);
                touch := TouchArea {
                    property <float> pressed-pitch;
                    property <float> pressed-yaw;

                    pointer-event(event) => {
                        if (event.button == PointerEventButton.left && event.kind == PointerEventKind.down) {
                            self.pressed-pitch = root.pitch;
                            self.pressed-yaw = root.yaw;
                        }
                    }
                    moved => {
                        if (self.enabled && self.pressed) {
                            root.pitch = self.pressed-pitch + (touch.mouse-y - touch.pressed-y) / self.height * 3.14;
                            root.yaw = self.pressed-yaw - (touch.mouse-x - touch.pressed-x) / self.width * 3.14;
                        }
                    }
                    mouse-cursor: self.pressed ? MouseCursor.grabbing : MouseCursor.grab;
                }
            }

            HorizontalBox {
                Text {
                    text: "Amplitude:";
                    font-weight: 600;
                    vertical-alignment: center;
                }

                amplitude-slider := Slider {
                    minimum: 0;
                    maximum: 100;
                    value: 50;
                }
            }
        }
    }
    ```

    `use plotters::prelude::*;`

    看来这个是绘图的重点。

    ```rs
    slint::slint! {
        import { MainWindow } from "plotter.slint";
    }
    ```

    `MainWindow`，又一个新组件。`plotter.slint`，不清楚这个是怎么来的。slint 里自带了 plotter 这个模块吗？

    `pure callback render_plot(/* pitch */ float, /* yaw */ float, /* amplitude */ float) -> image;`

    不清楚这个 pure 是什么意思。

    `title: "Slint Plotter Integration Example";`

    `title`，以及下面的`preferred-width`和`preferred-height`，看起来都是固有属性，可以直接在 slint 中赋值的。

    `source: root.render_plot(root.pitch, root.yaw, amplitude-slider.value / 10);`

    看来`source`用的是`render_plot()`的返回值，而`render_plot()`返回的是一个 image。

    `touch := TouchArea {`

    不清楚这个`:=`是什么意思，以及`TouchArea`是在哪里引入的。猜测：`Image`也是一个 container，`touch`只是给匿名对象`TouchArea`增加了一个名字。

    `pointer-event(event) => {`看来是`TouchArea`的事件处理函数，`event`中包含了事件信息。

    `if (event.button == PointerEventButton.left && event.kind == PointerEventKind.down) {`

    看来`event`是个 struct，当有事件时，会把对应的 value 赋值。

    `self.pressed-pitch = root.pitch;`

    `self`毫无疑问访问的是本组件的数据，但是`root`访问的是上一级组件，还是根组件呢？

    `moved => {`看来是在这里处理鼠标拖动事件。

    `if (self.enabled && self.pressed) {`

    `enabled`和`pressed`显然不是自定义的属性，那么它们是内置属性吗？

    `mouse-cursor: self.pressed ? MouseCursor.grabbing : MouseCursor.grab;`

    猜测：所有某个数据随别的数据的变动而变动的情况，都使用`:`来定义这个数据。

* plotter slint rs code analyze

    ```rs
    // Copyright © SixtyFPS GmbH <info@slint.dev>
    // SPDX-License-Identifier: MIT

    use plotters::prelude::*;
    use slint::SharedPixelBuffer;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen::prelude::*;

    #[cfg(target_arch = "wasm32")]
    mod wasm_backend;

    slint::slint! {
        import { MainWindow } from "plotter.slint";
    }

    fn pdf(x: f64, y: f64, a: f64) -> f64 {
        const SDX: f64 = 0.1;
        const SDY: f64 = 0.1;
        let x = x as f64 / 10.0;
        let y = y as f64 / 10.0;
        a * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
    }

    fn render_plot(pitch: f32, yaw: f32, amplitude: f32) -> slint::Image {
        let mut pixel_buffer = SharedPixelBuffer::new(640, 480);
        let size = (pixel_buffer.width(), pixel_buffer.height());

        let backend = BitMapBackend::with_buffer(pixel_buffer.make_mut_bytes(), size);

        // Plotters requires TrueType fonts from the file system to draw axis text - we skip that for
        // WASM for now.
        #[cfg(target_arch = "wasm32")]
        let backend = wasm_backend::BackendWithoutText { backend };

        let root = backend.into_drawing_area();

        root.fill(&WHITE).expect("error filling drawing area");

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_3d(-3.0..3.0, 0.0..6.0, -3.0..3.0)
            .expect("error building coordinate system");
        chart.with_projection(|mut p| {
            p.pitch = pitch as f64;
            p.yaw = yaw as f64;
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart.configure_axes().draw().expect("error drawing");

        chart
            .draw_series(
                SurfaceSeries::xoz(
                    (-15..=15).map(|x| x as f64 / 5.0),
                    (-15..=15).map(|x| x as f64 / 5.0),
                    |x, y| pdf(x, y, amplitude as f64),
                )
                .style_func(&|&v| {
                    (&HSLColor(240.0 / 360.0 - 240.0 / 360.0 * v / 5.0, 1.0, 0.7)).into()
                }),
            )
            .expect("error drawing series");

        root.present().expect("error presenting");
        drop(chart);
        drop(root);

        slint::Image::from_rgb8(pixel_buffer)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
    pub fn main() {
        // This provides better error messages in debug mode.
        // It's disabled in release mode so it doesn't bloat up the file size.
        #[cfg(all(debug_assertions, target_arch = "wasm32"))]
        console_error_panic_hook::set_once();

        let main_window = MainWindow::new().unwrap();

        main_window.on_render_plot(render_plot);

        main_window.run().unwrap();
    }
    ```

    `use slint::SharedPixelBuffer;`看起来像是导入图片缓冲区相关的功能。

    `import { MainWindow } from "plotter.slint";`

    这里的`plotter.slint`，就是那个 slint ui 界面文件的名字。

    `fn pdf(x: f64, y: f64, a: f64) -> f64 {`看来是采样计算那个高斯分布图的，不重要。

* `slint::include_modules!();`

    这个应该是把所有需要用到的 module 都包含进来。

    试了试找具体引入进来了什么，没找到。

* slint 的代码模板

    <https://github.com/slint-ui/slint-rust-template>

    按照这个说明可以创建一个代码模板，一个简单的小程序。功能是点击一个按钮，使得窗口上显示的数字加一。

* main 函数

    ```rs
    fn main() -> Result<(), slint::PlatformError> {
        let ui = AppWindow::new()?;

        ui.on_request_increase_value({
            let ui_handle = ui.as_weak();
            move || {
                let ui = ui_handle.unwrap();
                ui.set_counter(ui.get_counter() + 1);
            }
        });

        ui.run()
    }
    ```

    可以看出，`ui`是 new 出来的，`AppWindow`应该也是一个重要的 mod，背会。

    `ui`会自动找到 ui 代码里定义的一些回调函数，然后在这里对其进行定义。

    `ui.as_wea()`拿到一个 weak 指针。为什么不使用`&`或`&mut`？

    `move`表示将 ui 的 weak 指针移动到新线程里，然后让`ui`对象调用 ui 中成员的 set get 方法。

    `ui.run()`应该表示的是进入事件循环。

* ui code

    ```slint
    import { Button, VerticalBox } from "std-widgets.slint";

    export component AppWindow inherits Window {
        in-out property<int> counter: 42;
        callback request-increase-value();
        VerticalBox {
            Text {
                text: "Counter: \{root.counter}";
            }
            Button {
                text: "Increase value";
                clicked => {
                    root.request-increase-value();
                }
            }
        }
    }
    ```

    看来`std-widgets.slint`中有许多的 ui 组件。

    `Button`是按钮不必多言，`VerticalBox`看起来像一个 layout。

    `export component AppWindow inherits Window`，`Window`是基类，`AppWindow`是派生类。不清楚除了`Window`还有其他什么基类？

    `in-out property<int> counter: 42;`这个看来是定义了一个成员变量`counter`。如果是复杂的 struct 呢，该怎么写？

    `callback request-increase-value();`这个看来是声明了一个回调函数。

    `VerticalBox {`，这段用来定义 components，这些 ui 组件的写法就像一个个 struct。

    `text: "Counter: \{root.counter}";`看来 ui component 不能存储动态的变量数据。

* slint 的一个最小测试例子

    `Cargo.toml`:

    ```toml
    # ...

    [dependencies]
    slint = "1.5.0"

    # ...
    ```

    `main.rs`:

    ```rs
    slint::slint!{
        export component HelloWorld {
            Text {
                text: "hello world";
                color: green;
            }
        }
    }
    fn main() {
        HelloWorld::new().unwrap().run().unwrap();
    }
    ```

    compile:

    `cargo build`

    run:

    `cargo run`

    效果：出现一个小窗口，上面有绿色的字 hello world