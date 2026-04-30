## AI Agent Note

## cache

* 使用 ssh 登陆到远程机器，再打开 screen，然后打开 codex 后，codex 在工作时，光标会狂闪。如何解决？

    在使用 SSH、Screen 和 Codex CLI（OpenAI 的命令行工具）这一套组合时，光标狂闪通常是因为 **Codex 的状态指示器（Spinner）触发了频繁的终端重绘**，而 Screen 在处理这种高频刷新时，会不断发送“隐藏/显示光标”的转义序列。

    这种现象在 SSH 延迟环境下会被放大，导致光标看起来像在“狂跳”或“狂闪”。你可以通过以下几个步骤解决：

    1. 关闭 Codex 的终端标题动画（最有效）
        
        Codex 默认会不断更新终端窗口的标题（Title）来显示一个旋转的加载图标（Spinner）。在 Screen 内部，这个操作会导致频繁的重绘冲突。

        * **方法**：在 Codex 交互界面中，输入 `/title` 命令来修改或禁用标题动画。
        * **或者**：在你的 Codex 配置文件中（通常是 `~/.config/codex/config.toml` 或类似路径），找到 `tui` 部分，设置：
            ```toml
            [tui]
            animated_terminal_title = false
            ```
            *注：如果该选项不可用，尝试寻找关闭 "spinner" 或 "status animation" 的开关。*

    ---

    2. 优化 Screen 的配置
        
        Screen 对高频刷新率的支持不如 Tmux。你可以通过修改 `~/.screenrc` 来减少不必要的干扰：

        1.  打开配置文件：`nano ~/.screenrc`
        2.  添加或确保有以下配置：
            ```text
            # 禁用 Screen 自己的可视铃声
            vbell off
            # 尝试开启“非阻塞”模式，减少输出等待导致的闪烁
            nonblock on
            ```
        3.  重启 Screen 会话生效。

    ---

    3. 修改光标闪烁设置
        
        如果是因为 Codex 在等待模型响应时，其 TUI（终端界面）强制光标进入某种状态，你可以尝试在进入 Screen 前先禁用系统层面的光标闪烁：

        ```bash
        # 执行此命令后再打开 screen 和 codex
        setterm -blink off
        ```

    ---

    4. 终极替代方案：使用 Tmux + 同步渲染
        
        如果上述方法无法彻底解决，建议将 `screen` 换成 **`tmux`**。Tmux 在 3.3+ 版本中支持 **“同步渲染”（Synchronized Rendering）**，可以完全消除 TUI 程序（如 Codex、Claude Code）的闪烁。

        在 `~/.tmux.conf` 中添加：
        ```text
        set-option -g terminal-features ",xterm-256color:Sync"
        ```

    **总结**

    最快的方法是进入 Codex 后查找是否有**关闭 Status Spinner（加载动画）**的选项。那个每秒几十次的标题更新通常是 SSH 环境下光标狂闪的罪魁祸首。

* 给我讲讲常用的搜索技巧，比如 grep 之类的

  * 常用思路

      先定目标，再选工具：

      - 找文件名：rg --files
      - 找文本：rg
      - 看上下文：rg -n -C 5
      - 只想知道“有没有”：rg -q
      - 按类型过滤：rg -tcpp
      - 结果太多时，先缩目录，再缩关键词，再加排除

      最常用的是 rg, 比 grep 快很多，默认还会尊重 .gitignore。

      常用例子：

      rg "workHead"
      rg -n "sipuLaunchKernel" src/
      rg -n -C 3 "remoteChip" src/
      rg -tcpp "devPeers"
      rg -g '!build' -g '!logs' "channel->devPeers"
      rg --files | rg 'CMakeLists|enqueue|channel'

      高频参数

      - -n：显示行号
      - -C 3：前后各 3 行上下文
      - -A 5 / -B 5：只看后/前文
      - -i：忽略大小写
      - -w：整词匹配
      - -F：按普通字符串匹配，不用正则
      - -g：文件路径过滤
      - -t：按文件类型过滤
      - --hidden：搜隐藏文件
      - -S：智能大小写
      - -U：跨行/不过滤二进制时才偶尔用
      - -c：只看每个文件匹配次数
      - -l：只看命中文件名
      - --files：列文件，不搜内容

      什么时候用 grep: 机器上没有 rg 时再退回去。

      grep -RIn "workHead" src
      grep -RIn --include='*.cc' "remoteChip" src
      grep -RIn --exclude-dir=build "devPeers" .

      路径过滤技巧

      只在某些目录搜：

      rg -n "ncclWork" src/enqueue.cc src/init.cc src/include

      排除目录：

      rg -n "remoteChip" . -g '!build' -g '!logs'

      只搜某类文件：

      rg -n "ptrExchange" -tcpp
      rg -n "CMAKE_" -tcmake

      **先粗后细**
      
      比如排查空指针：

      1. 先搜结构体定义

      rg -n "struct ncclDevChannelPeer|struct ncclConnInfo" src/include

      2. 再搜赋值点

      rg -n "devPeers|devPeersHostPtr|ptrExchange" src

      3. 再看 launch 点

      rg -n "sipuLaunchKernel|workHead|channelMask" src/enqueue.cc

      看日志的技巧

      看最后错误：

      tail -n 200 logs/xxx/mpi.log
      tail -n 200 logs/xxx/qemu_host_0.log

      搜关键字：

      rg -n "error|warn|fault|translation failure|iova" logs/20260327_03

      带上下文：

      rg -n -C 5 "iova=0xf8" logs/20260327_03/qemu_host_0.log

      对比两个文件
      很有用。

      diff -u a.log b.log
      vimdiff a.log b.log

      或者只抽一段再比：

      sed -n '10378,10392p' file1
      sed -n '10378,10392p' file2

      结合 shell 做批量搜索

      for f in logs/20260327_03/cmodel/inst_trace*.log; do
        rg -n "0x00000000000000f8" "$f"
      done

      按大小排序文件：

      for f in logs/20260327_03/cmodel/*.log; do
        printf "%s %s\n" "$(stat -c %s "$f")" "$f"
      done | sort -nr | head

      搜地址/十六进制时
      最好用固定字符串匹配：

      rg -n -F "0x0000c000158ba04c" logs/20260327_03/cmodel
      rg -n -F "iova=0xf8" logs/20260327_03/qemu_host_0.log

      因为正则里 . [ 这些容易误伤。

      搜不到时常见原因

      - 文件被 .gitignore 忽略了：加 --hidden 或显式传路径
      - 在 build 产物里：别用默认全局搜，直接指定目录
      - 关键词大小写不一致：加 -i 或 -S
      - 实际内容不是文本：用 strings、xxd、hexdump
      - 路径打错：先 rg --files | rg 'name'

      我自己常用的排查套路

      1. rg --files | rg '关键词'
      2. rg -n "关键词" 相关目录
      3. rg -n -C 3 "关键词" 相关文件
      4. sed -n 'start,endp' 文件
      5. 必要时 diff / 批量脚本

