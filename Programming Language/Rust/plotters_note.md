# plotters note

github repo: <https://github.com/plotters-rs/plotters>

crate doc: <https://docs.rs/plotters/latest/plotters/>

usage:

`Cargo.toml`:

```toml
[dependencies]
plotters = "0.3.3"
```

目前最新版本已经到了`0.3.5`。

## cache

* rust plotters 中，`.build_cartesian_2d(0..10, 0f32..1f32)`填的数据类型，必须和后面的 line series `LineSeries::new(zip(x_data, y_data), &RED);`中的数据类型相对应。

    如果填写不一致，会报错，无法通过编译。

* 一个 example

    `main.rs`:

    ```rust
    use plotters::prelude::*;

    fn main() {
        let bkend = BitMapBackend::new("./chart.png", (700, 500));  // 这里可以是各种格式的图片，也可以填 bytes buffer
        let root = bkend.into_drawing_area();  // 这个 root 就是整张画纸的抽象表达，我们可以将一个画布进行 split，从而画多个 chart
        root.fill(&WHITE).expect("fail to fill WHITE");  // 填底色
        let mut chart = ChartBuilder::on(&root)  // chart 是个比较独立的概念，这里指定将它画到哪个 root 画布上
            .margin(0)  // 整个 chart 最外圈是 margin
            .x_label_area_size(25)  // 往内一圈是 x label, ylabel, title 等。再往内一圈才是 xOy 内的图表
            // 如果这里设置为 0，那么就不显示 x axis 的 ticks
            .build_cartesian_2d(0i32..10i32, -1f32..1f32)?;  // 2维图表，这里也可以选择 3 维
            // 注意这里的 range 使用的类型，必须和下面的 series 的数据类型相对应，不然会编译报错
        chart.configure_mesh()  // 似乎执行完这一句才能开始配置 chart 内的内容
            .x_labels(10)  // 给出最多 10 个 x axis ticks
            .x_label_formatter(&|v| v.to_string())  // 将系统自动给的 x ticks 的 value 转换成字符串
            // 如果不想显示一些 x label，可以直接在这里输出空字符串
            .draw()?;  // 画！
        let x_data = [5, 6, 7, 8, 9];  // 由于前面在 build_cartesian_2d() 时 x range 使用了 i32，所以这里也得是 i32
        let y_data = x_data.to_owned().map(|x| (x.to_owned() as f32).sin());  // 同理，y 必须是 f32
        let ser = LineSeries::new(zip(x_data, y_data), &RED);  // LineSeries 接受 iter 给出的数据对作为绘制数据
        chart.draw_series(ser)?;  // 画！
        root.present()?;  // 这一步或者保存文件，或者输出到 bytes buffer 里
    }
    ```