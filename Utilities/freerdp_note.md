# freerdp

## cache

* 之前用 ubuntu 1 -> server -> ubuntu 2 -> win 11 rdp 3389 是正常的，可以用 rfx，但是现在在 win 11 上直接 ssh reverse tunnel 到 server，即 ubuntu 1 -> server -> win 11 rdp 却不能用 rfx 了

    这是一个非常典型的 RDP 协议“能力协商”失败的问题。在你的第一种链路（通过中间 Ubuntu 转发）中，RDP 流量表现为普通的点对点 TCP 流量；而在第二种链路（SSH 反向隧道）中，由于流量经过了本地回环接口（127.0.0.1）的转发，触发了 Windows 的安全或策略限制。

    以下是导致 RFX 无法启动的核心原因及其解决方案：

    1. 核心原因：Windows 对“回环地址”的策略限制

        当你在 Win 11 上使用反向隧道时，RDP 服务接收到的连接请求来自 `127.0.0.1`。Windows 默认认为本地回环连接不需要高性能图形加速（如 RemoteFX/高级编解码），或者出于安全考虑，在处理来自 localhost 的 RDP 请求时，某些高级功能会被降级或禁用。

    2. 解决方案：强制开启高性能策略

        你需要在 Win 11（受控端）中通过组策略明确告诉 Windows：**无论连接来自哪里，都必须使用高性能压缩和编解码。**

        步骤：

        1. 按下 `Win + R`，输入 `gpedit.msc` 并回车。
        2. 导航至：`计算机配置` -> `管理模板` -> `Windows 组件` -> `远程桌面服务` -> `远程桌面会话主机` -> `远程会话环境`。

    3. 必须配置项：

        * 启用 RemoteFX 编码：设置为 `已启用`。

        * 为远程桌面连接使用 WDDM 图形显示驱动程序：设置为 `已禁用`（有时 WDDM 驱动会干扰旧版 RFX，尝试切换此项）。

        * 配置 RemoteFX 数据的自适应图形编码：设置为 `已启用`，并选为 `优化服务器资源使用量`。

        * 为所有远程桌面会话使用硬件图形适配器：设置为 `已启用`。

    4. 执行 `gpupdate /force` 或重启电脑。

    注：

    1. 也可以使用改 ip 的方式，比如把 127.0.0.1 改成 192.168.100.1

        通过指定具体的局域网 IP（192.168.100.1），Windows 将其识别为真实的**网络路径**，从而完整激活了图形编解码加速、位图缓存和帧率优化功能。即使物理上流量还是流向自己，但逻辑上的“身份切换”绕过了回环接口的性能限制。

* (base) hlc@hlc-VirtualBox:~$ xfreerdp /v:127.0.0.1 /port:6689 /u:wsdlh /p:xxxxxx /gfx:off

    FreeRDP - A Free Remote Desktop Protocol Implementation
    See www.freerdp.com for more information

    Usage: xfreerdp [file] [options] [/v:<server>[:port]]

    Syntax:
        /flag (enables flag)

    从报错信息看，你遇到了两个主要问题：**参数语法错误**以及**底层的段错误（Segmentation Fault）**。

    1. 修正命令语法

        你使用的命令中包含 `gfx:off` 等斜杠开头的参数，这在旧版（1.x）或某些特定版本的语法中可能不被识别。新版 FreeRDP 对参数顺序和格式比较敏感。

        请尝试将所有开关参数（带 `/` 的）放在服务器地址 `/v` 之前，并加入证书忽略开关：

        ```bash
        xfreerdp /u:wsdlh /p:xxxxxx /port:6689 /gfx:off /cert:ignore /v:127.0.0.1
        ```

* freerdp 中`/`和`+`以及`-`的不同

    你提供的 `Usage` 和 `Syntax` 信息揭示了问题的关键：**你的 FreeRDP 版本对参数的解析逻辑与新版（v3）或通用文档不同。**

    仔细看这一行帮助信息：

    > `+toggle -toggle (enables or disables toggle, where '/' is a synonym of '+')`

    这句话的意思是：**`/` 等同于 `+`，只能用于“开启”功能。**

    因此，你输入 `/gfx:off` 实际上被解析器试图理解为“开启 gfx 并赋值为 off”，但这在旧版语法中是不合法的，导致了 Usage 报错；而默认情况下 `rdpgfx` 又是开启的（日志里显示 Loaded），最终导致了后续的段错误。

    核心解决方案

    要真正“禁用”引发崩溃的图形扩展，你必须使用 **`-` (减号)**。

    请尝试这条修正后的命令：

    ```bash
    xfreerdp /v:127.0.0.1 /port:6689 /u:wsdlh /p:xxxxxx -gfx -themes -wallpaper /cert-ignore
    ```

    为什么这样做有效？

    1. **`-gfx` (关键修改)**：你的版本中，`/gfx` 相当于 `+gfx` (强制开启)。只有 `-gfx` 才是强制禁用。日志显示程序在加载 `rdpgfx` 后崩溃，禁用它是解决段错误的核心。

    2. **`PDU_TYPE_DATA` 错误**：这通常是因为服务器发送了 FreeRDP 无法处理的高级图形指令（如 AVC444 或某些位图压缩）。`-gfx` 会迫使连接回退到更老、更稳定的传统 RDP 图形模式。

    3. **`/cert-ignore`**：针对 `Certificate verification failure` 警告，防止因证书交互导致连接挂起。


* /network:lan -rfx  这个参数组合没问题，可以正常跑

    太好了！既然 `/network:lan -rfx` 组合可以稳定运行，这说明你的版本在处理 RemoteFX 或 Progressive Codec 时存在特定的内存解析 Bug，而通过手动禁用它们，你既保留了 LAN 模式的高带宽优势，又避开了崩溃点。

    目前你的连接应该已经处于“稳定且较快”的状态。

    为了进一步提升你在该稳定模式下的操作流畅度和使用体验，可以尝试以下优化：

    * 开启位图缓存与数据压缩提升响应速度
    * 创建保存该稳定参数的桌面启动快捷方式
    * 配置共享剪贴板与本地磁盘重定向

