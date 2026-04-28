# Zig Note

## cache

* zig语言如何安装？

    **Zig 语言安装指南**

    你想了解 Zig 语言的安装方法，以下是不同操作系统下的完整安装步骤，包含官方推荐方式和便捷替代方案，同时涵盖版本验证与切换技巧：

    一、官方推荐：通过 Zigup 安装（跨平台通用，优先推荐）

    `zigup` 是 Zig 官方社区推荐的工具，支持一键安装、切换多个 Zig 版本，是跨平台安装的最优选择。

    1. 安装 zigup

        （1）Linux / macOS（通过 curl 快速安装）
        打开终端，执行以下命令：
        ```bash
        # 下载并安装 zigup（自动适配你的系统架构）
        curl -fsSL https://github.com/marler8997/zigup/releases/latest/download/zigup-init.sh | sh
        ```
        安装完成后，将 `zigup` 加入系统环境变量（若终端提示未找到 zigup，执行以下命令，根据shell类型选择）：
        ```bash
        # bash 用户
        echo 'export PATH="$HOME/.zigup/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

        # zsh 用户（macOS 默认 shell，Linux 部分用户使用）
        echo 'export PATH="$HOME/.zigup/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
        ```

        （2）Windows（通过 PowerShell 安装）
        打开 **管理员模式 PowerShell**，执行以下命令：
        ```powershell
        # 下载并安装 zigup
        irm https://github.com/marler8997/zigup/releases/latest/download/zigup-init.ps1 | iex
        ```

    2. 通过 zigup 安装 Zig
        
        无论哪种系统，安装完成 zigup 后，终端/PowerShell 执行以下命令即可安装最新稳定版 Zig：
        ```bash
        # 安装最新稳定版
        zigup install stable

        # 可选：安装指定版本（例如 0.12.0）
        # zigup install 0.12.0
        ```

        zigup 会自动配置环境变量，无需手动额外配置。

    二、各操作系统原生安装方式

    1. Windows 系统
        
        方式1：手动下载安装包

        1.  访问 Zig 官方下载地址：https://ziglang.org/download/
        2.  下拉找到 **Windows** 对应版本（根据系统架构选择 `x86_64-windows` 或 `aarch64-windows`）
        3.  下载 `.zip` 压缩包，解压到自定义目录（例如 `D:\Tools\zig`）
        4.  配置环境变量：
            - 右键「此电脑」→「属性」→「高级系统设置」→「环境变量」
            - 在「系统变量」的 `Path` 中，新增一条路径：`D:\Tools\zig`（即你解压后的 zig 目录，需包含 zig.exe 文件）
        5.  重启终端生效

        方式2：通过包管理器（Chocolatey）

        若已安装 Chocolatey 包管理器，执行以下命令：
        ```powershell
        choco install zig
        ```

    2. macOS 系统

        方式1：Homebrew 包管理器（推荐，便捷高效）
        若已安装 Homebrew（macOS 常用包管理器），终端执行：
        ```bash
        # 安装最新稳定版 Zig
        brew install zig
        ```

        方式2：手动下载
        1.  访问官方下载地址：https://ziglang.org/download/
        2.  下载 macOS 对应版本（`x86_64-macos` 或 `aarch64-macos`）的 `.tar.xz` 压缩包
        3.  解压到自定义目录（例如 `~/Tools/zig`）
        4.  添加环境变量（同 zigup 安装后的配置，替换路径即可）

    3. Linux 系统
        
        方式1：包管理器安装（部分发行版支持）
        - Ubuntu / Debian（通过官方源，需先添加仓库）：
          ```bash
          # 添加 Zig 官方仓库
          curl -s https://ziglang.org/keys/zig.gpg | gpg --dearmor -o /usr/share/keyrings/zig-archive-keyring.gpg
          echo "deb [signed-by=/usr/share/keyrings/zig-archive-keyring.gpg] https://downloads.ziglang.org/dist/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/zig.list
          # 更新并安装
          apt update && apt install zig
          ```
        - Fedora：
          ```bash
          dnf install zig
          ```
        - Arch Linux（自带最新版）：
          ```bash
          pacman -S zig
          ```

        方式2：手动下载
        1.  官方下载地址获取 Linux 对应压缩包（`x86_64-linux` / `aarch64-linux`）
        2.  解压到 `/usr/local/zig` 或用户目录
        3.  将解压目录添加到系统 PATH 环境变量即可

    三、验证安装是否成功

    无论哪种安装方式，安装完成后，打开终端（Windows 打开 CMD 或 PowerShell），执行以下命令：

    ```bash
    zig version
    ```

    若终端输出类似 `0.12.0`（版本号随最新稳定版变化）的信息，说明 Zig 已成功安装；若提示「命令未找到」，请检查环境变量配置是否正确，或重启终端/电脑重试。

    四、版本切换（基于 zigup，可选）

    若你需要同时使用多个 Zig 版本（例如开发不同项目需要不同版本），可通过 zigup 快速切换：

    ```bash
    # 1. 先安装多个版本（例如稳定版和 0.11.0 版本）
    zigup install stable
    zigup install 0.11.0

    # 2. 切换到指定版本（例如 0.11.0）
    zigup default 0.11.0

    # 3. 验证切换结果
    zig version
    ```

    **总结**
    
    1.  **优先选择**：跨平台使用 `zigup` 安装，支持一键安装和版本切换，配置最简单；
    2.  **原生替代**：Windows 可手动解压+配环境变量/Chocolatey，macOS 优先 Homebrew，Linux 优先对应发行版包管理器；
    3.  **验证关键**：`zig version` 命令是检验安装成功的核心方式；
    4.  **版本管理**：通过 `zigup default <版本号>` 可快速切换 Zig 版本。

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

