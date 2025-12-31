# Zig Note

## cache

* zig

    website: <https://ziglang.org/>

    验证安装: `zig version`
    
    **基本用法**

    1. 编写第一个程序

        创建 hello.zig：
        
        ```zig
        const std = @import("std");

        pub fn main() !void {
            std.debug.print("Hello, {s}!\n", .{"World"});
        }
        ```

    2. 运行与编译

        直接运行（临时编译执行）：

        ```bash
        zig run hello.zig
        ```

        编译为可执行文件：

        ```bash
        zig build-exe hello.zig
        ./hello
        ```

        调试版本（带调试信息）：

        ```bash
        zig build-exe hello.zig -O Debug
        ```

        发布版本（优化）：

        ```bash
        zig build-exe hello.zig -O ReleaseSafe
        ```

        注：

        1. zig 没有添加 bash completion，所以`zig bui<tab>`不会自动补全子命令

        1. `-O Debug`中 Debug 的 D 必须大写，否则会报错。

    3. 使用构建系统（zig build）

        创建 build.zig：

        ```zig
        const std = @import("std");

        pub fn build(b: *std.Build) void {
            const exe = b.addExecutable(.{
                .name = "myapp",
                .root_source_file = .{ .path = "src/main.zig" },
            });
            b.installArtifact(exe);
        }
        ```

        运行构建：

        ```bash
        zig build
        ```

        注：

        1. 执行后报错

            ```
            (base) hlc@hlc-VirtualBox:~/Documents/Projects/zig_test$ zig build
            build.zig:6:10: error: no field named 'root_source_file' in struct 'Build.ExecutableOptions'
                    .root_source_file = .{ .path = "src/main.zig" },
                     ^~~~~~~~~~~~~~~~
            /home/hlc/Softwares/zig-x86_64-linux-0.16.0-dev.1859+212968c57/lib/std/Build.zig:767:31: note: struct declared here
            pub const ExecutableOptions = struct {
                                          ^~~~~~
            referenced by:
                runBuild__anon_30442: /home/hlc/Softwares/zig-x86_64-linux-0.16.0-dev.1859+212968c57/lib/std/Build.zig:2235:33
                main: /home/hlc/Softwares/zig-x86_64-linux-0.16.0-dev.1859+212968c57/lib/compiler/build_runner.zig:456:29
                5 reference(s) hidden; use '-freference-trace=7' to see all references
            ```

    **关键特性体验**

    1. 编译时计算

        ```zig
        const std = @import("std");

        fn factorial(n: u32) u32 {
            return if (n == 0) 1 else n * factorial(n - 1);
        }

        pub fn main() void {
            const result = comptime factorial(5); // 编译时计算
            std.debug.print("5! = {d}\n", .{result});
        }
        ```

    2. 交叉编译

        ```bash
        # 编译为 Windows 64 位程序
        zig build-exe hello.zig -target x86_64-windows

        # 编译为 ARM Linux 程序
        zig build-exe hello.zig -target aarch64-linux-gnu
        ```

    **开发工具**

    1. 语言服务器

        Zig 自带 LSP，支持编辑器自动补全：

        ```bash
        zig build-exe --help  # 查看所有选项
        ```

    2. 文档生成

        ```bash
        zig test hello.zig --docs  # 生成文档
        ```

        注：

        1. 这个功能好像不能用。可能是 ai 的幻觉。

    3. 测试

        在源码中添加测试：

        ```zig
        test "basic test" {
            try std.testing.expect(1 + 1 == 2);
        }
        ```

        运行测试：

        ```bash
        zig test hello.zig
        ```

    **注意事项**

    * 版本选择：Zig 更新较快，建议使用最新稳定版。

    * 标准库文档：使用 `zig std` 命令查看本地文档。

    * 包管理：目前依赖通过 Git 子模块或构建系统管理，官方包管理器在开发中。

    注：

    1. `zig std`会生成文档网页，监听 127.0.0.1 的端口，并且在浏览器中打开。

    **学习资源**

    * 官方文档：
    
        Zig Language Reference: <https://ziglang.org/documentation>

    * 《Zig 语言圣经》：
    
        zig.guide: <https://ziglearn.org>

    Zig 的设计强调直观和可控，适合系统编程、编译器开发或对性能有严苛要求的场景。

## topics

