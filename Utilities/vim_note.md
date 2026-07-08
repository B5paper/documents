# vim note

## cache

* 按下 `:` 进入命令行模式时，Vim 已经退出了 Visual 模式，此时 mode() 得到的并不是 v 或 V，而大概率是 n 模式。此时必须使用`visualmode()`得到上一次的 visual mode。

    它返回最近一次 Visual 模式的字符：

    * 'v' → 字符选择（v）

    * 'V' → 行选择（V）

    * '^V' → 块选择（<C-v>）

    * 空字符串 → 没有 Visual 选择

* 为什么 Vim 正则表达式常用单引号？

    在 Vim 脚本中，字符串有两种写法：

    | 写法 | 含义 | 转义 |
    |------|------|------|
    | `"双引号"` | 支持转义序列（如 `\n`、`\t`、`\\`） | `\` 会被特殊处理 |
    | `'单引号'` | 原样保留所有字符，`''` 表示一个单引号 | 几乎不转义 |

    **为什么正则常用单引号？**

    因为正则表达式里充满了 `\`（如 `\d`、`\s`、`\1`、`\.`）。如果用双引号，你需要写两个反斜杠：

    ```vim
    " 双引号写法（丑陋）
    if "\\d\\+" =~ "\\d"

    " 单引号写法（清爽）
    if '\d\+' =~ '\d'
    ```

    **结论**：用单引号是为了**避免反斜杠爆炸**，让正则表达式可读性更高。

* vim 中 `=~#` 是什么意思？

    `=~` 是 Vim 脚本中的**匹配操作符**，用来判断字符串是否匹配某个正则表达式。

    而 `=~#` 是它的变体，加了一个 `#`，表示**大小写敏感匹配**。

    **Vim 的匹配操作符家族**

    | 操作符 | 含义 |
    |--------|------|
    | `=~` | 匹配（根据当前 `'ignorecase'` 设置决定大小写是否敏感） |
    | `=~#` | 匹配，**强制大小写敏感**（case-sensitive） |
    | `=~?` | 匹配，**强制大小写不敏感**（case-insensitive） |
    | `!~` | 不匹配（根据当前设置） |
    | `!~#` | 不匹配，强制大小写敏感 |
    | `!~?` | 不匹配，强制大小写不敏感 |

    **示例**

    ```vim
    if 'Hello' =~# 'h'      " 返回 false，因为 H 不等于 h（敏感）
    if 'Hello' =~? 'h'      " 返回 true，忽略大小写
    if 'Hello' =~ 'h'       " 结果取决于 set ignorecase? 的设置
    ```

    如果你还想了解 Vim 正则里 `\%[ ]` 这种魔法语法，或者 `\v`（very magic）模式，我可以接着讲。

* 在 Vim 中，任何通过编程方式移动光标（如 setpos()、cursor()、normal! 等）都会导致 Visual 模式退出。

    通常的解决方案是

    ```vim
    let move_lines = target_line - end_line
    if move_lines > 0
        execute "normal! " . move_lines . "j"
    ```

    或者使用

    `call feedkeys("V" . target_line . "G", "n")`

    添加硬件：

    `vnoremap <leader>fn :<C-u>call FindNextSharp()<CR>`

    * `<C-u>` 用于清空命令行，防止 Vim 自动插入 '<,'> 范围标记。

    * 也可以使用 `gv`: 重新选择上一次 Visual 选择

        ```
        if target_line > 0
            " 先恢复到原来的 Visual 选择
            execute "normal! gv"
            " 然后扩展选择到目标行
            let move_lines = target_line - line("'>")
            if move_lines > 0
                execute "normal! " . move_lines . "j"
            endif
        endif
        ```

    * 或者使用范围参数重新选择

        ```vim
        " 重新选择从原始位置到目标行
        execute "normal! " . a:firstline . "GV" . target_line . "G"
        ```

    * 或者使用按键序列

        ```vim
        " 在 Visual 模式下按 :，然后输入：
        :normal gv0V/^[[:space:]]*#\+<CR>

        " 或者使用自定义函数
        :normal :call FindNextSharp()<CR>
        ```

        甚至使用 execute 嵌套两层：

        ```vim
        " 在 Visual 模式下
        :execute "normal! gv" . line("'>") . "GV" . FindTargetLine() . "G"
        ```

* 讲讲 vim 中 :save 的用法

    `:save`, 全称 `:saveas`

    功能: **“另存为（Save As）”**，同时将当前编辑的缓冲区（Buffer）切换到新文件上。

    1. 基础语法

        在命令模式下输入：

        ```vim
        :saveas <新文件名>
        # 或者缩写
        :save <新文件名>
        ```

    2. 与 `:w` 的区别

        `:w <新文件名>`（写入新文件）

        假设你当前正在编辑文件 `A.txt`：

        场景 A：使用 `:w B.txt`

        1. Vim 会把当前内容保存一份到 `B.txt`。
        2. **但是**，你当前光标所在的编辑窗口**依然停留在 `A.txt**`。你继续写的内容，依然会保存到 `A.txt`。

        场景 B：使用 `:save B.txt`

        1. Vim 会把当前内容保存一份到 `B.txt`。
        2. **关键点**：Vim 会**自动把当前窗口切换到 `B.txt**`。你接下来继续写字并执行 `:w`，内容会被保存到 `B.txt`，而原来的 `A.txt` 就被留在原地，不受后续修改的影响了。

        **总结**：`:save` 就是标准现代编辑器（如 VS Code、Word）里的 **“另存为”**；而 `:w <文件名>` 更像是 **“备份一份副本”**。

    **原文件未保存时的特殊情况**

    如果你修改了 `A.txt` 但**没有执行过 `:w` 保存**，直接运行 `:save B.txt`：

    * Vim 会把**当前修改后**的内容直接写入 `B.txt`。
    * 此时你的编辑窗口切到了 `B.txt`。
    * **注意**：原来的 `A.txt` 依然保持在你动笔修改之前的状态（未保存状态）。

    注：

    1. 使用 save 切换到新 buffer 后，旧 buffer 还在 vim 的 buffert list 里吗？

* bash 中如何展开文件路径的 tab 自动补全？

    在 Bash 中，文件路径的 Tab 自动补全功能通常是默认启用的。以下是关于如何使用和配置这个功能的详细说明：

    **基本使用方法**

    1. **输入部分路径**后按一次 `Tab`：

        ```bash
        cd /usr/loc[TAB]  # 会自动补全为 /usr/local/
        ```

    2. **按两次 `Tab`** 显示所有可能的补全选项：

        ```bash
        cd /usr/l[TAB][TAB]
        # 显示所有以 /usr/l 开头的可能路径
        lib/ local/ libexec/
        ```

    **配置选项**

    **检查是否启用补全**

    ```bash
    # 查看当前补全设置
    shopt | grep autocd
    shopt | grep cdable_vars
    ```

    **启用相关选项**

    ```bash
    # 在 ~/.bashrc 中添加以下配置
    shopt -s autocd          # 允许直接输入目录名进入该目录
    shopt -s cdspell         # 自动纠正目录名的拼写错误
    shopt -s direxpand       # 在补全时展开变量
    shopt -s dirspell        # 在补全时尝试纠正拼写错误
    ```

    **自定义补全行为**

    **修改补全设置**

    ```bash
    # 在 ~/.inputrc 中配置（影响所有 Readline 应用程序）
    set show-all-if-ambiguous on     # 按一次 Tab 就显示所有选项
    set completion-ignore-case on    # 补全时忽略大小写
    set menu-complete on             # 循环显示补全选项
    ```

    **重新加载配置**

    ```bash
    # 使配置生效
    source ~/.bashrc
    bind -f ~/.inputrc
    ```

    **特殊情况处理**

    **处理包含空格的文件名**

    Bash 会自动用反斜杠转义空格：

    ```bash
    # 对于文件 "my file.txt"
    my[TAB]  # 会自动补全为 my\ file.txt
    ```

    **使用变量补全**

    ```bash
    cd $HOME/Doc[TAB]  # 会自动补全变量和路径
    ```

    **故障排除**

    如果 Tab 补全不工作，可以检查：

    ```bash
    # 确认补全功能已安装
    complete -p  # 应该显示一系列补全命令

    # 如果缺少补全功能，可以安装 bash-completion 包
    # Ubuntu/Debian:
    sudo apt-get install bash-completion

    # CentOS/RHEL:
    sudo yum install bash-completion
    ```

    **高级技巧**

    **创建自定义补全**

    你可以为特定命令创建自定义补全规则：

    ```bash
    # 示例：为 mycommand 创建自定义补全
    complete -W "option1 option2 option3" mycommand
    ```

    **使用通配符补全**

    ```bash
    ls *.txt[TAB]  # 补全所有 .txt 文件
    ls file{1..10}[TAB]  # 使用花括号扩展
    ```

    这些设置应该能帮助你在 Bash 中更高效地使用 Tab 自动补全功能。记得在修改配置文件后重新加载它们使其生效。

    在 Vim 中编辑 bash 脚本时的 Tab 自动补全功能。以下是几种在 Vim 中实现文件路径补全的方法：

    1. 内置补全功能

        **文件路径补全**

        在插入模式下，输入部分路径后按 `Ctrl-x Ctrl-f`：

        ```bash
        # 输入 /usr/l 然后按 Ctrl-x Ctrl-f
        cd /usr/l█
        ```

    2. 插件增强补全

        **安装 coc.nvim (推荐)**

        ```bash
        # 使用 vim-plug 安装
        Plug 'neoclide/coc.nvim', {'branch': 'release'}

        # 安装 coc-bash 扩展
        :CocInstall coc-bash
        ```

        **安装 YouCompleteMe**

        ```bash
        Plug 'ycm-core/YouCompleteMe'
        ```

    3. 配置 Vim 的 bash 文件类型补全

        在 `~/.vimrc` 中添加：

        ```vim
        " 为 bash 文件启用特定补全
        autocmd FileType sh setlocal complete+=k~/.vim/bash-completion
        autocmd FileType sh setlocal iskeyword+=@-@  " 包含 @ 在关键字中

        " 设置补全菜单
        set wildmenu
        set wildmode=list:longest,full

        " 设置文件路径补全的忽略模式
        set wildignore+=*.o,*.obj,*.bak,*.exe,*.pyc,*.DS_Store

        " 使用 Tab 键触发补全
        inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
        inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
        ```

    4. 使用 omni 补全

        在 bash 文件中，可以按 `Ctrl-x Ctrl-o` 触发 omni 补全，需要安装 bash 补全插件：

        ```vim
        " 安装 bash 支持插件
        Plug 'vim-scripts/bash-support.vim'
        ```

    5. 简单键位映射

        ```vim
        " 在插入模式下快速补全文件路径
        inoremap <C-f> <C-x><C-f>

        " 快速补全单词
        inoremap <C-l> <C-x><C-n>
        ```

    6. 实用的补全命令

        在编辑 bash 脚本时可以使用这些组合键：

        | 快捷键 | 功能 |
        |--------|------|
        | `Ctrl-x Ctrl-f` | 文件路径补全 |
        | `Ctrl-x Ctrl-v` | Vim 命令补全 |
        | `Ctrl-x Ctrl-l` | 整行补全 |
        | `Ctrl-n` / `Ctrl-p` | 关键字补全 |

    7. 安装专门的 bash 插件

        ```vim
        " 安装 bash 增强插件
        Plug 'WolfgangMehner/bash-support'
        Plug 'aliou/bats.vim'
        Plug 'tmux-plugins/vim-tmux'

        " 重新加载配置后运行
        :PlugInstall
        ```

    8. 使用 snippets 插件

        ```vim
        " 安装 UltiSnips
        Plug 'SirVer/ultisnips'
        Plug 'honza/vim-snippets'

        " 在 bash 文件中使用 snippet 补全
        ```

    这样配置后，在 Vim 中编辑 bash 脚本时就可以使用丰富的自动补全功能了。推荐使用 `coc.nvim` + `coc-bash` 的组合，它能提供最智能的补全体验。

* vim 中如何映射 alt 键

	在 Vim 中映射 **Alt 键（Meta 键）** 是出了名的“老大难”问题。这是因为许多终端（Terminal）在按下 `Alt + 键` 时，默认会发送一个 `Esc` 字符加上该键的组合（比如 `Alt+j` 会变成 `^[j`），导致 Vim 无法直接识别它。

	不过别慌，解决这个问题主要有以下三种黄金法则，你可以根据自己的需求和终端类型来选择：

	---

	**方法一：直接映射法（适用于 Neovim / 现代终端）**

	如果你使用的是 **Neovim**，或者支持 `modifyOtherKeys` 的现代终端（如 Kitty, Alacritty, iTerm2 最新版，Windows Terminal），你可以像映射其他键一样直接使用 `<M-...>`（M 代表 Meta/Alt）：

	```vim
	" 正常模式下，Alt + j 向下移动一行
	nnoremap <M-j> :m .+1<CR>==

	" 插入模式下，Alt + 左右键 快速切换标签页
	inoremap <M-Left> <Esc>:tabprevious<CR>
	inoremap <M-Right> <Esc>:tabnext<CR>

	```

	---

	**方法二：利用终端转换法（最通用、最省心）**

	如果你使用的是常规终端（如 SecureCRT, Xshell, Mac 自带 Terminal），最稳妥的办法是**在终端软件的设置里，将 Alt 键配置为 Meta 键或 Escape 键**。

	* **iTerm2**: `Preferences` -> `Profiles` -> `Keys` -> 将 Left/Right Option key 改为 `Esc+`。
	* **SecureCRT / Xshell**: 在键盘映射设置中，勾选“Use Alt as Meta key”或“将 Alt 键作为 Esc 键的前缀”。

	设置好后，在 Vim 中你可以通过映射 `<Esc>` 组合来代替 Alt。例如，终端收到 `Alt+j` 会认为是 `Esc + j`：

	```vim
	" 相当于映射了 Alt + j
	nnoremap <Esc>j :m .+1<CR>==

	```

	> ⚠️ **注意**：这种方法有一个小副作用，如果你手动按下 `Esc` 键后再按 `j`，也会触发这个映射。不过在实际连击中影响不大。

	---

	**方法三：映射终端转义序列（适用于原生老版本 Vim）**

	如果你坚持在Linux服务器的原生 Vim 中使用，且不想改终端设置，可以通过在 `.vimrc` 中捕获终端发送的转义序列来“欺骗” Vim。

	你可以利用 Vim 的 `set <M-...>` 语法来手动定义。在插入模式下，按下 `Ctrl + V` 然后按下你的 `Alt + 键`，通常会打出类似 `^[j` 的字符（其中 `^[` 是真正的 Esc 字符）。

	```vim
	" 1. 先把终端的转义序列赋给 Vim 的虚拟键位
	execute "set <M-j>=\e j"

	" 2. 然后再像平常一样映射这个键
	nnoremap <M-j> :m .+1<CR>==

	```

	*(注：上面的 `\e` 代表 Esc 字符，部分 Vim 版本需要直接写入原始的 `^[j`)*

	---

	**💡 避坑总结**

	* **如果你用的是 Neovim**：直接用 `<M-j>`，绝大多数现代终端都开箱即用。
	* **如果你用的是老 Vim 且经常远程连接**：修改你本地终端软件的设置（将 Alt 映射为 `Esc+`），然后在 Vim 里用 `<Esc>j` 映射，这是最不会出幺蛾子的方案。

	注：

    1. 在映射时直接使用`<esc>`代替 alt 就好了。

        比如：`nnoremap <esc>j 2j` 就等同于 alt + j 映射为 2j.

* 终端为什么把 alt 转换成 esc

	这是一个非常经典的历史遗留问题。要理解为什么终端把 `Alt` 键转换成 `Esc`（准确地说是 `Esc` 前缀，比如把 `Alt + j` 变成 `Esc` 紧接着 `j`，即 `^[j`），我们需要把时间拨回到 **20 世纪 70-80 年代**，去看看计算机刚诞生时的窘境。

	简单来说，原因可以归结为：**ASCII 码的限制** 和 **物理键盘的演变**。

	---

	1. ASCII 编码的“坑”：装不下那么多键

        早期的计算机和终端（比如著名的 VT100 终端）使用的是 **ASCII 编码**。

        * ASCII 编码非常紧凑，总共只有 **128 个字符**（7 位二进制）。
        * 这 128 个字符里，包含了大小写字母、数字、标点符号，以及一些控制字符（比如换行 `LF`、回车 `CR`）。

        在这个标准制定的时候，**根本就没有 `Alt` 键（甚至也没有 `Ctrl` 键）**。
        后来，为了让键盘能发出更多指令，硬件厂商发明了 `Ctrl` 键和 `Alt` 键。

        * **`Ctrl` 键的解决办法**：它通过修改字母的二进制位，直接映射到了 ASCII 码前 32 个原本就存在的“控制字符”上。例如，按下 `Ctrl + M` 发送的就是回车符（`CR`）。
        * **`Alt` 键的尴尬**：轮到 `Alt` 键（又称 Meta 键）时，ASCII 码已经**全满**了，没有任何多余的空位可以分给 `Alt + 字母` 的组合。

	---

	2. 聪明的妥协：用“带头大哥” `Esc` 来组合

        既然不能发明新的单字符编码，工程师们只能想出一个折中方案：**用多个现有字符组合起来，表示一个新按键。**

        他们盯上了 `Esc` 键（Escape 字符，ASCII 码为 27）。`Esc` 的本意就是“退出当前状态、改变接下来字符的含义”。

        于是，终端协议规定：

        > 当用户按下 `Alt + X` 时，终端不发送一个单一字符，而是发送**两个连续的字符**：先发送一个 `Esc` 字符（在终端里常显示为 `^[`），紧接着发送 `X`。

        这就是大名鼎鼎的 **Escape Sequence（转义序列）**。

	---

	3. 为什么沿用至今？

        虽然现在的操作系统（Windows, macOS, Linux）和现代图形界面（GUI）早就支持了 Unicode 和更复杂的键盘事件，能够轻松识别 `Alt` 键，但**终端（Terminal）和命令行工具（如 Vim, Emacs）依然活着历史里**。

        为了保持 **向后兼容性（Backward Compatibility）**，让 40 年前的老软件依然能在今天的最新电脑上运行，现代终端（如 iTerm2、Xshell、macOS Terminal）默认依然在模拟老旧的 VT100/VT220 终端行为。因此，它们依然敬业地把 `Alt + 键` 转换成 `Esc + 键` 发送给后台。

	---

	**总结**

	终端把 `Alt` 变成 `Esc`，就像**用电报发送复杂的表情包**——因为电报机（ASCII 码）没有这个表情，所以人们只能用一串特定的标点符号（`Esc` 前缀）来拼出这个意思。 

* 详细讲讲 vim 中的 ttimeout 和 ttimeoutlen

	在 Vim 中，`ttimeout` 和 `ttimeoutlen` 是两个专门用来控制**键盘映射（Mappings）和终端转义序列（Escape Sequences）超时行为**的特殊设置。

	如果你曾经在 Vim 中遇到过“按 `Esc` 键退出插入模式时有明显的延迟”，或者“在终端里按方向键结果冒出奇怪的字母”，通常就是这两个选项没有配置好。

	下面我们逐一拆解它们的工作原理和配置方法。

	---

	1. 核心概念：为什么需要“超时”？

        在终端（Terminal）中，很多功能键（如方向键、`PageUp/PageDown`、`F1-F12` 等）在被按下时，终端并不会发送一个单一的字符，而是发送一串以 `Esc` 开头的**转义序列**。

        * 例如：按下 **方向键上 ($\uparrow$)**，终端实际发送给 Vim 的是：`\x1b[A`（其中 `\x1b` 就是 `Esc` 键）。

        这就带来了一个冲突：
        当你在 Vim 的插入模式下按下了 `Esc` 键，Vim 怎么知道你是在**单纯地想退出插入模式**，还是**刚刚按下了方向键的开头，后面还有字符没传过来**？

        为了解决这个问题，Vim 引入了**超时机制**。

	---

	2. 两个参数的详细解释

        🕒 `ttimeout` (Terminal Timeout)

        * **类型**：布尔值 (`on` 或 `off`)
        * **默认值**：在现代 Vim/Neovim 中通常默认开启 (`on`)
        * **作用**：决定 Vim 在接收到终端转义序列（如方向键、功能键等以 `Esc` 开头的代码）未完成时，**是否应用超时限制**。
        * `set ttimeout`：开启限制。如果后续字符没有在规定时间内到达，Vim 就判定你只按了 `Esc`。
        * `set nottimeout`：关闭限制。Vim 会死等下一个字符，直到确定它不是某个功能键的一部分。这就容易导致按 `Esc` 出现严重卡顿。


        ⏱️ `ttimeoutlen` (Terminal Timeout Length)

        * **类型**：数值（单位：毫秒 ms）
        * **默认值**：在传统 Vim 中默认是 `-1`（此时它会退而去同步 `timeoutlen` 的值，通常是 1000ms）
        * **作用**：当 `ttimeout` 开启时，Vim 等待**终端转义序列**下一个字符到达的**最大时间**。

        > ⚠️ **重要对比**：
        > Vim 中还有一个叫 `timeoutlen`（没有开头的 `t`）的参数。
        > * `timeoutlen`：针对你**自己设置的 multi-key 映射**（比如你设置了 `inoremap jj <Esc>`，它控制你敲完第一个 `j` 后，等第二个 `j` 的时间，默认 1000ms）。
        > * `ttimeoutlen`：专门针对**终端自身的功能键/转义序列**（比如方向键，默认 `-1` 或很小的值）。
        > 
        > 

	---

	3. 经典痛点：为什么按 `Esc` 会延迟？

        如果你发现按 `Esc` 退出插入模式要等足足 1 秒钟，原因通常是：

        1. `ttimeout` 没开，或者 `ttimeoutlen` 没有独立设置（保持了默认的 `-1`）。
        2. 此时 `ttimeoutlen` 自动套用了 `timeoutlen` 的值（1000 毫秒）。
        3. 当你按 `Esc` 时，Vim 为了确认你是不是要按方向键，硬生生等了 1000 毫秒，发现没后续，才执行 `Esc`。

        最佳实践配置

        完美解决这个问题，既让 `Esc` 秒回弹，又不影响你自己设置的复合快捷键（如 `jj` 或 `leader` 键），可以在你的 `.vimrc` 中加入以下配置：

        ``vim
        开启用户自定义映射的超时（比如你自定义的 <Leader> 键序列）
        et timeout
        用户自定义映射的等待超时时间设置为 500 毫秒（可根据手速调整）
        set timeoutlen=500

        " 开启终端转义序列（方向键、功能键等）的超时
        set ttimeout
        " 终端转义序列的等待超时时间设置为 20 毫秒（重点！）
        " 现代计算机和终端传输速度极快，20ms 足够判定是不是方向键了
        set ttimeoutlen=20
        ```

	---

	4. 总结与常见表现

        | 参数与组合 | `Esc` 键的响应速度 | 方向键/功能键的表现 | 适用场景 |
        | --- | --- | --- | --- |
        | `set nottimeout` | **极其缓慢**（需要等很久或按其他键才触发） | 极其稳定，绝对不会识别断开 | 不推荐，除非你在极度高延迟的远程 SSH 连线上 |
        | `set ttimeout` <br>

        <br> `set ttimeoutlen=1000` | **有明显 1 秒延迟** | 稳定 | 默认未优化状态，体验较差 |
        | `set ttimeout` <br>

        <br> `set ttimeoutlen=20` | **瞬间响应（推荐）** | 正常识别 | **最佳推荐配置** |
        | `set ttimeoutlen=0` | 瞬间响应 | 可能会导致方向键失效，变成输入 `A` `B` `C` `D` | 不建议设为 0，给终端留一丁点缓冲时间 |

        > 💡 **Neovim 用户注意**：如果你使用的是 **Neovim**，它在初始化时已经默认帮你把 `ttimeout` 设为了 `on`，且 `ttimeoutlen` 设为了 `50` 毫秒，所以通常不需要手动修复 `Esc` 延迟问题。但如果是原生 Vim，这几行配置属于必加项。

* vim coc.nvim 如何手动显示函数参数的悬浮窗

	在 `coc.nvim` 中，手动触发显示当前函数参数或函数签名的悬浮窗（Signature Help），默认的命令是 `:call CocActionAsync('showSignatureHelp')`。

	为了方便使用，你可以将其绑定到一个快捷键上。请将以下代码加入到你的 `init.vim` 或 `.vimrc` 中：

	```vim
	" 使用 <leader>k 手动触发函数参数悬浮窗
	nmap <silent> <leader>k <Plug>(coc-signature-help)
	imap <silent> <c-k> <Plug>(coc-signature-help)
	```

	💡 进阶技巧：让它在输入时自动显示

	如果你希望在输入括号 `(` 或逗号 `,` 时自动弹出参数提示，可以在 Vim 中运行 `:CocConfig` 打开配置文件，并添加/确认以下配置：

	```json
	{
	  "suggest.autoTrigger": "always",
	  "signature.autoTrigger": true,
	  "signature.triggerSignatureHelp": true
	}
	```

	这样通常就不需要频繁手动去按快捷键了。 

    注：

    1. 如果是正常模式下希望手动显示函数签名，那么 `call CocActionAsync('showSignatureHelp')` 这个似乎没什么用。

	    此时可以触发“悬浮窗文档”（Hover）:

        ```vim
        " 使用 K (大写) 键触发悬浮窗文档
        nnoremap <silent> K :call ShowDocumentation()<CR>

        function! ShowDocumentation()
          if CocAction('hasProvider', 'hover')
            call CocActionAsync('doHover')
          else
            call feedkeys('K', 'in')
          endif
        endfunction

        ```

        > **使用方法**：在普通模式下，把光标移到函数名上，按下大写的 **`K`**。

        `call CocActionAsync('doHover')`这个似乎是可行的。

* 是否有 vscode 快捷键，可以找到下一个以`##`或`###`开头的行？

    这个可能是在说是否有能识别 markdown 段落布局的 vscode 快捷键，可以快速在段落之间跳转。

    如果是个意思的话，vscode 没有现成的快捷键或功能。只有 ctrl + f 或 f3 的搜索功能。

* vim 映射中，alt 键如何表示

	在 Vim 的映射命令中，Alt 键最标准的表示法是 `<M-...>`，`M` 代表 Meta。

	*   **`<M-key>`**: 这是最通用和推荐的写法。
	*   **`<A-key>`**: `A` 代表 Alt，是 `<M-key>` 的同义词。

	例如，要将 Alt+j 映射为在普通模式下向下移动一行，你可以在 `.vimrc` 中这样写：

	```vim
	nnoremap <M-j> j
	nnoremap <A-j> j
	```

* `\e` 在 Vim 脚本中代表 `Esc` 字符

	未验证。

* vim buffer 的使用

	Vim 完全可以将多个文件以**隐藏缓冲区（hidden buffers）**的方式打开，并在它们之间快速切换，就像 Screen 的多个窗口一样——每次只显示一个文件，占满整个终端。

	---

	**基础操作**

	1. 打开多个文件

		```bash
		vim file1 file2 file3        # 启动时打开多个
		: e file4                    # 在当前 Vim 中再打开一个（添加到 buffer 列表）
		: badd file5                 # 添加文件到 buffer 列表但不切换过去
		```

	2. 切换缓冲区（核心命令）

		```vim
		:ls          " 列出所有缓冲区（带编号）
		:b N         " 切换到编号为 N 的缓冲区（N 是数字）
		:bn          " 下一个缓冲区（next）
		:bp          " 上一个缓冲区（previous）
		:bf          " 第一个缓冲区（first）
		:bl          " 最后一个缓冲区（last）
		:b filename  " 按文件名切换（支持 Tab 补全）
		```

		ls 的输出说明：

		```
		1 %a   "file1.txt"          line 10
		2 #h   "file2.txt"          line 25
		3      "file3.txt"          line 5
		```

		标记说明：

		%a = 当前显示的缓冲区

		# = 上一个缓冲区（可用 :e# 切换回来）

		h = 隐藏的缓冲区（有未保存修改）

		+ = 有修改但已保存的？其实是 + 表示已修改但未保存

	**配置快捷键（像 Screen 一样操作）**

	```vim
	" 切换缓冲区（类似 Screen 的 Ctrl+A n 和 Ctrl+A p）
	nnoremap <leader>n :bn<CR>
	nnoremap <leader>p :bp<CR>

	" 快速列出所有缓冲区并选择
	nnoremap <leader>l :ls<CR>:b<Space>
	```

	**高级技巧：模糊查找切换**

	**使用内置的 `:buffer` 配合通配符**

	```vim
	:buffer foo<Tab>   " 自动补全
	:buffer *bar       " 匹配包含 bar 的文件名
	:buffer /pattern   " 用正则匹配缓冲区名
	```

	**使用 fzf.vim（强烈推荐）**

	安装 [fzf](https://github.com/junegunn/fzf.vim) 后：

	```vim
	:Buffers          " 模糊搜索所有缓冲区，回车即切换
	```

	**使用 CtrlP 插件**

	```vim
	:CtrlPBuffer      " 列出所有缓冲区进行模糊匹配
	```

	---

	**管理缓冲区（删除/隐藏）**

	```vim
	:bd              " bdelete, 删除当前缓冲区（unload，关闭文件）
	:bd N            " 删除编号为 N 的缓冲区
	:bd file         " 删除指定文件的缓冲区
	:bw              " bwipeout, 彻底擦除（wipeout），比 bd 更彻底

	:bufdo bd        " 删除所有缓冲区, 谨慎使用！
	:bd hidden       " 删除所有隐藏的缓冲区
	```

	如果在 bd 时，buffer 中有未保存的改动，vim 会提示。如果需要强制退出 buffer，可以使用`bd!`

	bw 会额外删除撤销历史、标记、文件类型设置、语法高亮设置、缩进设置等。通常不会用到这个命令，等用到了再查用法。

	```vim
	set autowrite      " 切换时自动保存（不是隐藏）
	set hidden         " 切换时不保存（隐藏）
	" 这两个可以同时使用（只使用 set autowrite 会发生什么？）
	```

	查看哪些 buffer 有未保存修改:

	```vim
	:ls               " 有 + 标记的就是未保存
	:echo &modified  " 当前 buffer 是否修改
	```

	---

	**关键设置：让切换更流畅**

	在 `.vimrc` 中添加：

	```vim
	" 允许在未保存时切换缓冲区（而不是报错）
	set hidden

	" 显示缓冲区编号（方便 :b N）
	set ls=2          " 始终显示状态栏
	set statusline=%{bufnr('%')}:\ %f\ %m\ %r   " 状态栏显示编号

	" 更智能的切换：按编号直接跳转
	" 输入 5 然后 Ctrl+N 跳到缓冲区 5（需要自定义映射）
	```

	注：

	1. 即使设置了 hidden，如果有未保存的修改，直接 :q 退出，也会提示保存

	---

	**保存会话（相当于 Screen 的 session 恢复）**

	```vim
	:mksession! ~/mysession.vim   " 保存当前所有缓冲区状态
	vim -S ~/mysession.vim        " 下次启动恢复
	```

	```vim
	:wa             " wall, 保存所有缓冲区（write all）
	:wqa            " 保存所有并退出（write quit all）
	:xall           " 同 :wqa
	:qa!            " 强制退出所有，丢弃修改
	:qall!          " 同 :qa!
	```

	```vim
	" 显示缓冲区列表和编号
	set statusline=%{bufnr('%')}\ %f\ %m\ %r
	```
	---

	## 实际工作流示例

	1. **启动 Vim 打开多个文件**：
	   ```bash
	   vim main.c util.c header.h
	   ```

	2. **在 Vim 中切换**：
	   - 按 `:bn` 或 `:bp` 前后切换
	   - 按 `:ls` 查看所有文件，然后 `:b 2` 跳转到第二个

	3. **遇到新文件需要编辑**：
	   ```vim
	   :e newfile.c      " 自动加入 buffer 列表
	   ```

	4. **不想看到某个文件**：

	   ```vim
	   :bd               " 关闭当前 buffer，自动切换到下一个
	   ```

	---

	**终极技巧：结合 `arglist`（参数列表）**

	如果你打开很多文件，可以这样：
	```bash
	vim *.c                    " 打开所有 .c 文件
	```

	然后用参数列表切换：
	```vim
	:args          " 列出参数列表
	:next          " 下一个参数文件（类似 :bn）
	:prev          " 上一个
	:first         " 第一个
	```

	区别：`arglist` 是启动时指定的文件列表，`buffer list` 是 Vim 中所有打开过的文件（包括后来 `:e` 添加的）。

	---

	**总结**

	- **最简方案**：用 `:bn` 和 `:bp` 切换，设置 `set hidden` 后无需保存即可切换。
	- **效率提升**：映射 `<C-n>` 和 `<C-p>`，加上 `:ls` 查看所有 buffer。
	- **生产力飞跃**：安装 fzf.vim 用 `:Buffers` 模糊搜索。

	这样你就拥有一个比 Screen 更强大的多文件编辑环境——所有文件都在同一个 Vim 进程内，**所有寄存器（包括 `"+` 剪贴板）完全共享**，复制粘贴毫无障碍！

* 在 cygwin 中使用 ssh 连接到远程 linux server，开启 screen, 并在 screen 的多个虚拟 terminal 中打开 vim，即使远程 host 安装了 x11，多个 vim 之间也无法使用 "+ 寄存器。

    目前不清楚为什么，可能是因为 cygwin 没有 x11 server，所以 ssh 在打开 bash session 时，就没有开启 x11 以及 DISPLAY 等相关的环境变量。

    在 cygwin 中无论是直接使用 ssh 连接 remote ubuntu server，还是使用 ssh -X 连接，还是使用 ssh -Y 连接，登陆 bash 后，环境变量 DISPLAY 都没有被设置。 

* vim 中没有 enumerate

* 详细讲讲 vim 中的 if, for 等条件控制语句

	1. 核心语法通则

    	在开始之前，必须记住 Vimscript 的两个**铁律**：

    	- **没有大括号 `{}`**：Vim 使用 `endif`、`endfor`、`endfunction` 等关键词来标记代码块的结束。

    	- **布尔值逻辑**：Vim 中，**数字 `0`** 代表假（False），**非零数字（通常是 `1`）** 代表真（True）。字符串在逻辑判断中会被转换为数字。

	2. 条件语句：`if` ... `elseif` ... `else` ... `endif`

    	这是最基础的分支结构。

    	基本写法

    	```vim
    	if v:version >= 800
    		echo "你的 Vim 版本大于等于 8.0"
    	elseif v:version >= 700
    		echo "你的 Vim 版本是 7.x"
    	else
    		echo "你的 Vim 版本较老"
    	endif
    	```

	关键细节与陷阱

	- **字符串比较**：
	  使用 `==` 时默认**忽略大小写**（取决于 `'ignorecase'` 设置）。为了严谨，推荐使用：
	  - `==#` ：匹配大小写（严格）。
	  - `==?` ：忽略大小写。
	  ```vim
	  if "Vim" ==# "vim"   " 结果为假（False）
	  if "Vim" ==? "vim"   " 结果为真（True）
	  ```

	- **逻辑运算符**：
	  - `&&` （与）
	  - `||` （或）
	  - `!` （非）
	  ```vim
	  if has("python") && !has("python3")
		  echo "只有 Python2 支持"
	  endif
	  ```

	- **检查变量是否存在**：
	  直接引用未定义的变量会报错。使用 `exists()` 函数来保护：
	  ```vim
	  if exists("g:my_custom_var") && g:my_custom_var == 1
		  echo "变量存在且为 1"
	  endif
	  ```

	3. 循环语句：`for` ... `endfor`

    	Vim 的 `for` 循环**专门用于遍历列表（List）**。它不像 C 语言那样有 `for(i=0; i<10; i++)` 的写法（那需要用 `while` 循环）。

    	**遍历列表**

    	```vim
    	let fruits = ['apple', 'banana', 'cherry']
    	for fruit in fruits
    		echo "我喜欢 " . fruit
    	endfor
    	" 输出：我喜欢 apple / banana / cherry
    	```

    	**遍历行（读取缓冲区）**

    	```vim
    	for line_num in range(1, line('$'))  " range(1, 最后一行)
    		let line_content = getline(line_num)
    		if line_content =~# 'error'
    			echo "第 " . line_num . " 行有错误"
    		endif
    	endfor
    	```

    	**遍历字典（Dictionary）**

    	使用 `keys()` 或 `items()` 函数：
    	```vim
    	let my_dict = {'name': 'Tom', 'age': 30}
    	for [key, value] in items(my_dict)
    		echo key . " -> " . value
    	endfor
    	```

    	**控制循环：`break` 和 `continue`**

    	与大多数语言一致：
    	```vim
    	for i in range(1, 10)
    		if i == 3
    			continue    " 跳过 3
    		endif
    		if i > 7
    			break       " 大于 7 时停止
    		endif
    		echo i
    	endfor
    	" 输出：1, 2, 4, 5, 6, 7
    	```

	---

	4. 循环语句：`while` ... `endwhile`

    	当需要基于条件而非列表循环时（或模拟 C 风格的 `for`），使用 `while`。

    	```vim
    	let i = 0
    	while i < 5
    		echo "计数: " . i
    		let i += 1       " Vim 没有 i++ 运算符，必须用 let i += 1
    	endwhile
    	```

    	**`while` 的常见用途**：无限循环直到用户操作或文件结束，但在现代 Vimscript 中，`for` + `range()` 通常比 `while` 更受推荐，因为更简洁。

	5. 其他“控制”类语句（扩展知识）

    	虽然不完全是 `if/for`，但它们是条件逻辑中的重要补充：

    	- **三元表达式（条件运算符）**：

    	  Vim 7.0 以上支持 `条件 ? 表达式1 : 表达式2`。
    	  ```vim
    	  let result = (a > 10) ? "大于" : "小于或等于"
    	  ```

    	- **`try` ... `catch` ... `endtry`（异常处理）**：

    	  用于捕获脚本运行时的错误，避免插件崩溃。
    	  ```vim
    	  try
    		  execute "normal! gg=G"  " 尝试自动缩进全文
    	  catch /E488/               " E488 是 trailing characters 错误
    		  echo "缩进失败，请检查语法"
    	  endtry
    	  ```

	6. 高阶技巧与避坑指南（重点）

    	避坑 1：`for` 循环中修改列表

    	在 `for` 循环中直接修改列表（如 `remove()`）会改变索引顺序，导致跳过元素。推荐的做法是使用 `while` 循环并手动管理索引，或者先拷贝一份 `for item in copy(my_list)`。

    	避坑 2：`if` 判断中的“假”值

    	在 Vim 中，以下情况均为假：
    	- 数字 `0`
    	- 空字符串 `""`（因为被转为数字 0）
    	- 字符串 `"0"`（因为被转为数字 0）

    	**特别坑**：字符串 `"false"` 或 `"no"` 在 Vim 中会被转换为数字 `0`，因此它们被认为是 **真**！ 
    	```vim
    	if "false"
    		echo "这行居然会打印！"  " 因为 "false" 不是数字，转为 0，但非空字符串转为 0？注意：非数字字符串转为0，所以是假？
    	```
    	*更正*：Vim 将非数字字符串转为数字时结果为 `0`，所以 `if "hello"` 实际上是 `if 0`，即为假。只有数字非零或字符串以数字开头（如 `"1hello"` 转为 1）才为真。**最佳实践**：永远使用 `==#` 或 `==?` 比较字符串，不要直接把字符串放在 `if` 后面。

    	避坑 3：作用域修饰符

    	在条件或循环内部给变量赋值时，注意变量作用域：
    	- 全局：`g:var`
    	- 脚本局部：`s:var`
    	- 函数局部：`l:var`（不写默认也是局部）
    	- 缓冲区局部：`b:var`

    	如果不加前缀，在函数内部默认是函数局部，在函数外部默认是全局。

	**总结速查表**

	| 语句 | 结束关键词 | 适用场景 |
	| :--- | :--- | :--- |
	| `if expr` | `endif` | 单分支或多分支判断 |
	| `for var in list` | `endfor` | 遍历列表、字典、文件行 |
	| `while expr` | `endwhile` | 基于条件的循环（谨慎使用） |
	| `break` / `continue` | （无） | 用于 `for` 或 `while` 内部中断/跳转 |
	| `try` | `endtry` | 包裹可能报错的代码，增加稳定性 |

	**建议**：在编写 `.vimrc` 时，尽量多用 `if has()` 和 `if exists()` 包裹特定的配置，这样你的配置文件在不同系统、不同 Vim 版本之间就不会报错了。 

* vim 从当前光标位置选择到行尾，但是不包含行尾的 \n

    ```vim
    vg_
    ```

    详解：

    v - 进入可视模式

    g_ - 跳转到行内最后一个非空白字符（不包括换行符）

* vim 正则替换的一个 example

    * 替换`    ## text`为`**text**`

        command:

        `nnoremap <leader><S-B> :s/^\s*\zs#\+\s\+\(.*\)/**\1**/<CR>`

        如果希望替换前确认，可在 s 后加 c 标志，如 :s/.../.../gc。

        注：

        1. 使用交互式命令进行模拟

            `nnoremap <leader><S-B> ^vg_:s/#\+\s\+\(.*\)/**\1**/<CR>`

            映射解读：

            * `^vg_` 自动选中当前行 `## text`

            * `:` 进入命令行，并带上可视区域范围 `'<,'>`

                注意，这个可视区域范围是默认是针对整行内容的，并不止是选中的内容。

                另外，不需要手动写`'<,'>`，这个是输入`:`后，vim 自动加上去的。我们在映射里也不需要写这个。

            * `s/#\+\s*\(.*\)/**\1**/` 将非空白开始的整行替换为 ** + 捕获的内容 + **

                注意这里不能写成`s/^#`，因为`^`只捕获真正的行首。我们使用`^vg_`选中的区域，只是从第一个非空白字符开始，并不是行首。

            * `<CR>` 执行

* vim 中`v`模式下选中的内容，输入`:`后，vim 会自动添加`'<,'>`，但是这个是针对整行的。如果我们执行`:s/hello/world/g`，那么整行中的`hello`都会被替换成`world`。如果只想替换选中内容中的`hello`，那么可以写成`:s/\%Vhello/world`。注意这里是大写的`V`.

    `\%V` 要求匹配的文本完全位于高亮选区内部。

    即使使用`` `<,`> ``（反引号 + 选区），行为也和`'<,'>`（单引号选区）一样，都是表示整行。

    为什么 :`<,`> 也是整行？

        '< 和 `< 都表示可视模式开始位置的标记（mark）。

        在 Vim 中，Ex 命令（如 :s）的范围（range）只接受行号，不接受列号。

        当你输入 :`<,`> 时，Vim 会读取这两个标记所在的行号，忽略它们记录的列号。

        所以本质上，:`<,`> 与 :'<,'> 等价，都是“从可视区域开始行到结束行”。

    '< 和 `< 的真正区别在哪？

    它俩的区别只在普通模式跳转时才体现：

        '< 跳转到可视区域开始位置的行首第一个非空白字符（类似 ^）。

        `< 跳转到可视区域开始位置的精确列（光标当时所在的列）。

    但在 : 命令行作范围时，Vim 只取它们共同的行号，列信息被丢弃。

* 为什么 \%Vhello\%V 中 \%V 要输入两遍

    因为 **`\%V` 是一个零宽度的位置断言（类似 `^` 或 `$`），它只匹配“位置”，而不匹配“字符”本身**。

    要理解为什么要写两遍，可以把它类比成正则表达式里的行首行尾锚定：

    - **第一个 `\%V`** 断言：匹配的**起始位置**必须在可视区域内。
    - **第二个 `\%V`** 断言：匹配的**结束位置**必须在可视区域内。

    只有**起点和终点都在选区内部**，才能保证**整个 `hello` 字符串完全位于你的高亮选区中**。

    ---

    ### 如果只写一个 `\%V` 会怎样？

    假设你的文本是 `hello`，但你**只选中了前 3 个字符 `hel`**（高亮区域仅覆盖 `hel`）：

    1. **写法：`\%Vhello`**
       - 匹配引擎从 `h` 的位置开始检查，`h` 在选区内（成立）。
       - 然后继续匹配 `ello`，虽然 `ello` 在选区之外，但因为没有第二个 `\%V` 来检查结束位置，所以 **Vim 认为匹配成功**。
       - 结果：即使你只高亮了 `hel`，执行替换后**整个 `hello` 都会被替换**，超出选区范围！

    2. **写法：`\%Vhello\%V`**
       - 匹配引擎检查 `h` 的起始位置在选区内（成立）。
       - 匹配完 `hello` 后，检查结束位置（`o` 后面的位置）。因为 `o` 不在选区内，结束位置断言失败。
       - 结果：**匹配失败**，`hello` 不会被替换。这符合你的预期——“没选全就不替换”。

    ---

    ### 官方文档说明

    `:help /\%V` 的原文是：
    > Matches inside the visual area. **Can be used to limit a match to the visual area.**

    注意它的表述是“**限制匹配在可视区域内**”。为了实现“限制”，你必须在匹配文本的两端都加上这个断言，强迫引擎检查边界。

    ---

    ### 另一个直观记忆方式

    把 `\%V` 想象成一对隐形的括号：

    - `\%V`（左括号）放在要匹配的文本前面。
    - `\%V`（右括号）放在要匹配的文本后面。

    这样 Vim 才知道：“嘿，这个 `hello` 必须**从头到尾**都待在括号（选区）里！”

    ---

    ### 补充：何时只用一次 `\%V`？

    只有当你要匹配的文本**只有 1 个字符**时，起始位置和结束位置重合，这时候写一次 `\%Vx` 和写两次 `\%Vx\%V` 效果一样。

    但对于多字符字符串（如 `hello`），**必须写两遍**才能精准控制边界。

* `let g:netrw_ftpextracmd = 'ssh'`

    配置完这个之后呢，怎么打开远程文件？

    配置了 `let g:netrw_ftpextracmd = 'ssh'` 之后，你可以通过以下几种方式打开远程文件：

    ## 方法一：直接在 Vim 命令中打开

    ### 基本语法：
    ```vim
    :e scp://[user@]hostname[:port]/path/to/file
    ```

    ### 具体示例：
    ```vim
    " 使用默认用户名（当前本地用户名）
    :e scp://remote-server/home/user/project/file.txt

    " 指定用户名
    :e scp://username@remote-server/path/to/file.txt

    " 指定端口
    :e scp://username@remote-server:2222/path/to/file

    " 绝对路径
    :e scp://user@host//home/user/file.txt

    " 相对用户home的路径
    :e scp://user@host/file.txt
    ```

    ## 方法二：使用 Netrw 浏览器

    ### 打开远程目录浏览：
    ```vim
    :Explore scp://user@host/path/to/directory/
    :Vexplore scp://user@host/path/to/directory/
    :Sexplore scp://user@host/path/to/directory/
    ```

    ### 示例：
    ```vim
    " 浏览远程用户home目录
    :Explore scp://user@host/

    " 浏览远程项目目录
    :Explore scp://user@host/path/to/project/
    ```

    ## 方法三：命令行直接启动

    ### 在终端中直接使用：
    ```bash
    vim scp://user@host/path/to/file.txt
    ```

    ## 方法四：简化命令的技巧

    ### 在 .vimrc 中创建快捷命令：
    ```vim
    " 定义快捷命令
    command! -nargs=1 Rvim :e scp://user@host/<args>

    " 或者使用函数更灵活
    function! RemoteEdit(host, file)
        execute 'e scp://' . a:host . '/' . a:file
    endfunction

    command! -nargs=+ RE :call RemoteEdit(<f-args>)
    ```

    ### 使用示例：
    ```vim
    " 使用自定义命令
    :Rvim path/to/file.txt
    :RE myhost /home/user/project/file.c
    ```

    ## 实际使用示例

    假设你要编辑远程服务器上的一个 Python 文件：

    ```vim
    " 打开远程文件
    :e scp://developer@192.168.1.100/home/developer/project/app.py

    " 保存文件
    :w  " 会自动通过SCP上传

    " 打开同目录下的另一个文件
    :e scp://developer@192.168.1.100/home/developer/project/utils.py

    " 浏览远程目录
    :Explore scp://developer@192.168.1.100/home/developer/project/
    ```

    ## 注意事项

    1. **认证方式**：需要配置 SSH 密钥认证，否则会频繁输入密码
    2. **网络延迟**：操作可能会有延迟，特别是大文件
    3. **文件权限**：保存时会使用 SSH 用户的权限
    4. **临时文件**：Vim 会在本地创建临时文件，操作完成后再上传

    ## SSH 配置优化

    为了更方便使用，可以在 `~/.ssh/config` 中配置主机别名：

    ```
    Host dev-server
        HostName 192.168.1.100
        User developer
        Port 22
        IdentityFile ~/.ssh/id_rsa
    ```

    然后就可以简化为：
    ```vim
    :e scp://dev-server/path/to/file.txt
    ```

    这样配置后，你就可以像编辑本地文件一样方便地编辑远程文件了。

* 那如果不配置 let g:netrw_ftpextracmd = 'ssh'，会发生什么？

    如果不配置 `let g:netrw_ftpextracmd = 'ssh'`，Netrw 的行为会有一些变化：

    ## 默认行为

    ### 1. 仍然可以使用 SCP，但 Netrw 会尝试猜测协议
    ```vim
    :e scp://user@host/path/to/file
    ```
    Netrw 会检测到 `scp://` 前缀，仍然会使用 SSH/SCP，但可能会使用不同的默认参数。

    ### 2. 主要差异在于其他协议的处理
    没有这个配置时：
    - 对于 `ftp://` 开头的 URL，Netrw 会尝试使用 FTP 命令
    - 对于没有明确协议但看起来像主机名的路径，行为可能不同

    ## 实际测试示例

    ### 有配置的情况：
    ```vim
    let g:netrw_ftpextracmd = 'ssh'
    :e scp://user@host/file.txt
    ```
    Netrw 明确使用 `ssh` 命令进行传输。

    ### 无配置的情况：
    ```vim
    " 从 .vimrc 中移除或注释该配置
    " let g:netrw_ftpextracmd = 'ssh'
    :e scp://user@host/file.txt
    ```
    Netrw 仍然能工作，但：
    - 可能使用 `scp` 命令而不是 `ssh`
    - 参数传递可能略有不同
    - 对于某些边缘情况处理可能不一致

    ## 具体影响

    ### 1. 命令行构建差异
    **有配置时**：
    ```bash
    ssh user@host cat /path/to/file
    ```

    **无配置时**：
    ```bash
    scp user@host:/path/to/file /tmp/vimXXXXXX
    ```

    ### 2. 性能差异
    - `ssh + cat`：流式传输，内存友好
    - `scp`：需要临时文件，可能稍慢

    ### 3. 错误处理差异
    不同的命令可能有不同的错误信息和处理方式。

    ## 验证测试

    你可以通过以下方式观察差异：

    ```vim
    " 打开 verbose 模式查看实际执行的命令
    :set verbose=9
    :e scp://user@host/file.txt
    ```

    ## 实际建议

    ### 推荐配置的原因：
    ```vim
    let g:netrw_ftpextracmd = 'ssh'
    ```
    1. **一致性**：明确指定使用 SSH
    2. **可靠性**：避免 Netrw 的错误猜测
    3. **性能**：SSH 方式通常更高效
    4. **调试**：错误信息更清晰

    ### 不配置的情况：
    - 简单使用时可能感觉不到差异
    - 如果你的环境 `scp` 工作更好，可以不配置
    - 某些旧系统可能 `scp` 更稳定

    替代配置

    如果你遇到问题，也可以尝试更明确的配置：

    ```vim
    " 明确指定 scp
    let g:netrw_ftpextracmd = 'scp'

    " 或者指定完整路径
    let g:netrw_ftpextracmd = '/usr/bin/ssh'

    " 或者带参数
    let g:netrw_ftpextracmd = 'ssh -q -o ConnectTimeout=10'
    ```

    总结

    **不配置 `let g:netrw_ftpextracmd = 'ssh'` 的影响**：
    - 基本功能仍然工作
    - Netrw 会尝试自动选择传输方式
    - 可能使用 `scp` 而不是 `ssh`
    - 在某些边缘情况下可能不如明确配置稳定

    对于生产环境使用，建议明确配置以获得更可靠的行为。

    实际支持的协议

    Netrw 实际支持这些协议前缀：
    ```vim
    " 文件传输协议
    :e scp://...          " SSH 文件传输
    :e ftp://...          " FTP
    :e http://...         " HTTP
    :e fetch://...        " FreeBSD fetch

    " 本地和特殊协议
    :e file://...         " 本地文件（显式）
    :e dav://...          " WebDAV
    :e rcp://...          " RCP（已很少使用）
    ```

* 不支持 ssh://to_file，但是支持 let g:netrw_ftpextracmd = 'ssh' 对吧。除了配置成 ssh，还能配置成什么？

    对的，你的理解正确！`let g:netrw_ftpextracmd = 'ssh'` 是配置 Netrw 使用什么**命令**来处理文件传输，而不是协议前缀。

    可配置的命令选项

    1. **SSH 相关命令**
    ```vim
    " 使用 ssh（推荐）
    let g:netrw_ftpextracmd = 'ssh'

    " 使用 scp
    let g:netrw_ftpextracmd = 'scp'

    " 使用带参数的 ssh
    let g:netrw_ftpextracmd = 'ssh -q -o ConnectTimeout=10'

    " 使用特定路径的 ssh
    let g:netrw_ftpextracmd = '/usr/bin/ssh'
    ```

    2. **FTP 相关命令**
    ```vim
    " 使用 ftp 命令
    let g:netrw_ftpextracmd = 'ftp'

    " 使用其他 FTP 客户端
    let g:netrw_ftpextracmd = 'ncftp'
    let g:netrw_ftpextracmd = 'lftp'

    " 使用带认证的 FTP
    let g:netrw_ftpextracmd = 'ftp -i -n'
    ```

    3. **其他文件传输命令**
    ```vim
    " 使用 wget
    let g:netrw_ftpextracmd = 'wget'

    " 使用 curl
    let g:netrw_ftpextracmd = 'curl'

    " 使用 rsh/rcp（已不推荐）
    let g:netrw_ftpextracmd = 'rsh'
    let g:netrw_ftpextracmd = 'rcp'
    ```

    4. **协议特定的配置**
    你还可以为不同协议配置不同的命令：

    ```vim
    " 为不同协议设置不同的命令
    let g:netrw_ftp_cmd = "ftp -i -n"      " 用于 ftp://
    let g:netrw_ssh_cmd = "ssh"            " 用于 scp://
    let g:netrw_http_cmd = "wget -q -O"    " 用于 http://
    ```

    实际应用示例

    针对不同环境的优化配置：
    ```vim
    " 开发环境 - 使用 SSH 并禁用主机密钥检查
    let g:netrw_ftpextracmd = 'ssh -o StrictHostKeyChecking=no'

    " 生产环境 - 使用详细日志
    let g:netrw_ftpextracmd = 'ssh -v'

    " 慢速网络 - 增加超时时间
    let g:netrw_ftpextracmd = 'ssh -o ConnectTimeout=30 -o ServerAliveInterval=60'

    " 代理环境 - 通过跳板机
    let g:netrw_ftpextracmd = 'ssh -o ProxyCommand="ssh user@jumpbox nc %h %p"'
    ```

    多协议配置：
    ```vim
    " 完整的 Netrw 传输配置
    let g:netrw_ssh_cmd = 'ssh -q'
    let g:netrw_ftp_cmd = 'ftp -i'
    let g:netrw_http_cmd = 'curl -s -o'
    let g:netrw_https_cmd = 'curl -s -o'
    ```

    查看当前配置

    你可以通过以下命令检查 Netrw 的当前配置：
    ```vim
    " 查看所有 Netrw 变量
    :let g:netrw_

    " 查看特定变量
    :echo g:netrw_ftpextracmd
    :echo g:netrw_ssh_cmd
    :echo g:netrw_ftp_cmd
    ```

    调试配置

    如果遇到问题，可以启用详细模式查看实际执行的命令：
    ```vim
    " 设置高详细级别
    :set verbose=9

    " 然后尝试打开远程文件
    :e scp://user@host/file.txt

    " 查看 Vim 执行的底层命令
    ```

    推荐配置

    对于大多数使用场景，推荐：
    ```vim
    " 基本配置
    let g:netrw_ftpextracmd = 'ssh'

    " 或者更详细的分别配置
    let g:netrw_ssh_cmd = 'ssh'
    let g:netrw_ftp_cmd = 'ftp -i'
    ```

    所以总结来说，`g:netrw_ftpextracmd` 可以配置为任何能在命令行中执行文件传输的命令，而不仅仅是 `ssh`。

    * 查看帮助：

        ```vim
        :help netrw
        :help netrw-start
        ```

    * 查看版本：

        ```vim
        :echo g:loaded_netrw
        :echo g:netrw_version
        ```

    支持 Netrw 的快捷键（`Enter` 进入，`-` 上级目录等）

* 调研 explorer 相关的命令

    使用 `:Explore` 命令：

    ```vim
    :Explore scp://user@host/path/to/directory/
    ```

    ```vim
    :Vexplore scp://user@host/path/to/directory/    " 垂直分割
    :Sexplore scp://user@host/path/to/directory/    " 水平分割
    :Texplore scp://user@host/path/to/directory/    " 新标签页
    ```

    ```vim
    " 先打开远程文件
    :e scp://user@host/path/to/file.txt

    " 然后切换到所在目录
    :Ex    " 或者 :Explore - 打开当前文件所在目录
    :Hex   " 水平分割打开目录
    :Vex   " 垂直分割打开目录
    ```

    浏览远程项目：

    ```vim
    :Explore scp://developer@myserver/~/projects/myapp/

    " 在目录中：
    - 按 Enter 进入子目录或打开文件
    - 按 - 返回上级目录
    - 按 o 在新窗口打开
    - 按 v 垂直分割打开
    - 按 s 按不同方式排序
    ```

    多窗口浏览：

    ```vim
    " 垂直分割浏览远程目录
    :Vexplore scp://user@host/project/

    " 水平分割浏览
    :Sexplore scp://user@host/project/
    ```

    ```vim
    " 创建文件
    :new filename.txt

    " 创建目录
    :call mkdir("new_directory")

    " 删除文件（谨慎使用）
    :call delete("file_to_delete")

    " 重命名
    :call rename("oldname", "newname")
    ```

    实用技巧

    1. 创建书签命令

    ```vim
    " 在 .vimrc 中定义常用远程目录
    command! RemoteProject :Explore scp://user@host/~/project/
    command! RemoteLogs :Explore scp://user@host/var/log/
    ```

    3. 目录缓存

        Netrw 会缓存目录列表，如果需要刷新：
        ```vim
        R    " 在 Netrw 窗口中按 R 刷新目录列表
        ```

    注意事项
    
    4. **隐藏文件**：按 `a` 键切换显示隐藏文件

* vim pyright 在 normal 模式下使用`:call CocAction('showSignatureHelp')`不会看到函数原型的提示框。

    检查 Coc 是否认为有 signature help provider: `:echo CocAction('hasProvider', 'signature')`

    如果需要在 normal 模式下显示函数签名，可以使用`:call CocAction('doHover')`。这个函数调用与`K`并不绑定。

* markdown-preview.nvim 这个怎么安装

    我来详细介绍一下 `markdown-preview.nvim` 的安装和配置方法：

    安装方法

    1. 使用包管理器安装（推荐）

        **使用 packer.nvim**

        ```lua
        -- 在 plugins.lua 文件中添加
        use({
            "iamcco/markdown-preview.nvim",
            run = function() vim.fn["mkdp#util#install"]() end,
        })
        ```

        **使用 lazy.nvim**

        ```lua
        -- 在 plugins 目录下创建 markdown-preview.lua
        return {
            "iamcco/markdown-preview.nvim",
            cmd = { "MarkdownPreviewToggle", "MarkdownPreview", "MarkdownPreviewStop" },
            ft = { "markdown" },
            build = function() vim.fn["mkdp#util#install"]() end,
        }
        ```

        **使用 vim-plug**

        ```vim
        " 在 .vimrc 或 init.vim 中添加
        Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']}
        ```

    2. 安装步骤

        1. **添加插件配置**到你的 Neovim 配置文件中

        2. **重新加载配置**并安装插件：

            ```vim
            :source %
            :PackerSync    " 如果使用 packer
            :Lazy sync     " 如果使用 lazy
            :PlugInstall   " 如果使用 vim-plug
            ```

        3. **等待安装完成**，插件会自动下载并编译必要的组件

    3. 基本配置

        **Lua 配置（Neovim）**

        ```lua
        -- 在 init.lua 或 plugins/markdown-preview.lua 中
        vim.g.mkdp_theme = 'dark'
        vim.g.mkdp_browser = 'firefox'  -- 或 'chrome', 'safari' 等
        vim.g.mkdp_auto_start = 0       -- 不自动启动预览
        vim.g.mkdp_auto_close = 1       -- 切换缓冲区时自动关闭预览
        vim.g.mkdp_refresh_slow = 0     -- 快速刷新
        vim.g.mkdp_command_for_global = 0 -- 仅对 markdown 文件启用
        ```

        **Vimscript 配置**

        ```vim
        " 在 .vimrc 或 init.vim 中
        let g:mkdp_theme = 'dark'
        let g:mkdp_browser = 'firefox'
        let g:mkdp_auto_start = 0
        let g:mkdp_auto_close = 1
        let g:mkdp_refresh_slow = 0
        let g:mkdp_command_for_global = 0
        ```

    4. 常用命令

        ```vim
        :MarkdownPreview        " 启动预览
        :MarkdownPreviewStop    " 停止预览  
        :MarkdownPreviewToggle  " 切换预览状态
        ```

    5. 键位映射建议

        ```lua
        -- 在 keymaps.lua 或 init.lua 中
        vim.api.nvim_set_keymap('n', '<leader>mp', ':MarkdownPreview<CR>', {noremap = true})
        vim.api.nvim_set_keymap('n', '<leader>ms', ':MarkdownPreviewStop<CR>', {noremap = true})
        vim.api.nvim_set_keymap('n', '<leader>mt', ':MarkdownPreviewToggle<CR>', {noremap = true})
        ```

    6. 高级配置（滚动同步）

        ```lua
        vim.g.mkdp_preview_options = {
            mkit = {},
            katex = {},
            uml = {},
            maid = {},
            disable_sync_scroll = 0,  -- 启用同步滚动
            sync_scroll_type = 'relative',
            hide_yaml_meta = 1,
            sequence_diagrams = {},
            flowchart_diagrams = {},
            content_editable = false,
            disable_filename = 0,
            toc = {}
        }
        ```

    **常见问题解决**

    **如果预览没有启动：**

    - 确保已安装 Node.js（插件依赖）
    - 重新构建插件：`:call mkdp#util#install()`
    - 检查浏览器是否被正确指定

    **如果同步滚动不工作：**

    - 检查 `disable_sync_scroll` 设置为 0
    - 确保浏览器没有阻止弹出窗口

    安装完成后，你就可以享受实时的 Markdown 预览和同步滚动了！

* 安装 `markdown-preview.nvim` 时配置代理

    在内网环境下安装 `markdown-preview.nvim` 需要配置代理。以下是几种解决方案：

    方案一：配置 Git 和 npm 代理

    1. 设置 Git 代理

        ```bash
        # 设置 HTTP 代理
        git config --global http.proxy http://your-proxy-server:port
        git config --global https.proxy https://your-proxy-server:port

        # 如果需要认证
        git config --global http.proxy http://username:password@proxy-server:port
        ```

    2. 设置 npm 代理（插件依赖 Node.js）

        ```bash
        npm config set proxy http://your-proxy-server:port
        npm config set https-proxy https://your-proxy-server:port
        npm config set strict-ssl false  # 如果证书有问题
        ```

    方案二：离线安装方法

    1. 在外网环境下载插件

        ```bash
        # 克隆插件到本地
        git clone https://github.com/iamcco/markdown-preview.nvim.git

        # 进入目录并构建
        cd markdown-preview.nvim
        npm install  # 在外网环境执行
        ```

    2. 将整个插件目录打包

        ```bash
        tar -czf markdown-preview.nvim.tar.gz markdown-preview.nvim
        ```

    3. 在内网环境安装

        ```bash
        # 解压到 Neovim 插件目录
        tar -xzf markdown-preview.nvim.tar.gz -C ~/.local/share/nvim/site/pack/plugins/start/
        ```

    方案四：使用离线包管理器

    1. 使用 Packer 的离线安装

        ```lua
        -- 如果插件已经在本地目录
        use({
            "iamcco/markdown-preview.nvim",
            run = function() 
                -- 手动指定构建命令，跳过网络下载
                vim.cmd('!cd ' .. vim.fn.stdpath('data') .. '/site/pack/packer/start/markdown-preview.nvim && npm install --offline')
            end,
        })
        ```

* vim 中 markdown 实时渲染、跟随滚动的方案

    现有插件（推荐）

    1. Markdown 预览

        - **vim-preview**：专门为 Markdown 和 LaTeX 设计的预览插件
        - **markdown-preview.nvim**：支持实时预览和滚动同步
        - **vim-markdown-preview**：轻量级的 Markdown 预览方案

    2. LaTeX 预览

        - **vimtex**：功能强大的 LaTeX 插件，支持正向和反向同步
        - **vim-latex-live-preview**：实时预览 LaTeX 文档

    **自定义解决方案**

    1. 设置 Vim 自动保存和光标位置跟踪

        ```vim
        " 自动保存
        autocmd TextChanged,TextChangedI *.md,*.tex write

        " 跟踪光标位置
        function! SendCursorPosition()
            let line = line('.')
            let col = col('.')
            " 将位置信息发送给外部程序
            call system('echo "'.line.','.col.'" > /tmp/vim_cursor_position')
        endfunction

        autocmd CursorMoved *.md,*.tex call SendCursorPosition()
        ```

    2. 外部渲染程序示例（Python）

        ```python
        #!/usr/bin/env python3
        import os
        import time
        import subprocess
        from pathlib import Path

        def monitor_file_and_cursor(file_path):
            last_mtime = 0
            last_pos = None
            
            while True:
                # 检查文件修改时间
                current_mtime = os.path.getmtime(file_path)
                if current_mtime != last_mtime:
                    # 文件已修改，重新渲染
                    render_file(file_path)
                    last_mtime = current_mtime
                
                # 检查光标位置
                try:
                    with open('/tmp/vim_cursor_position', 'r') as f:
                        pos = f.read().strip()
                        if pos != last_pos:
                            # 光标位置变化，同步滚动
                            scroll_to_position(pos)
                            last_pos = pos
                except:
                    pass
                
                time.sleep(0.1)

        def render_file(file_path):
            # 调用实际的渲染逻辑
            if file_path.endswith('.md'):
                # 渲染 Markdown
                subprocess.run(['pandoc', file_path, '-o', '/tmp/preview.html'])
            elif file_path.endswith('.tex'):
                # 渲染 LaTeX
                subprocess.run(['pdflatex', file_path])

        def scroll_to_position(position):
            # 根据位置信息滚动预览窗口
            line, col = position.split(',')
            print(f"Scrolling to line {line}, column {col}")

        if __name__ == "__main__":
            import sys
            if len(sys.argv) > 1:
                monitor_file_and_cursor(sys.argv[1])
        ```

    3. 更高级的通信方式
        
        使用 socket 或命名管道进行更可靠的通信：

        ```vim
        " Vim 端设置
        function! SendToPreview(command)
            call system('echo "'.a:command.'" > /tmp/vim_preview_pipe')
        endfunction

        autocmd CursorMoved *.md,*.tex call SendToPreview('cursor:'.line('.').':'.col('.'))
        autocmd BufWritePost *.md,*.tex call SendToPreview('render:'.expand('%:p'))
        ```

    **方案三：使用现有工具组合**

    1. 对于 Markdown

        ```bash
        # 使用 inotifywait 监控文件变化
        inotifywait -m -e close_write your_file.md | while read; do
            pandoc your_file.md -o preview.html
            # 触发预览程序刷新
        done
        ```

    2. 对于 LaTeX
        
        ```bash
        # 使用 latexmk 进行持续编译
        latexmk -pvc -view=pdf your_file.tex
        ```

* 为什么使用vim使用 scp:// 打开一个远程文件，显示

    ```
    "/tmp/vvWIsyd/0.cu" 281L, 8701B
    Press ENTER or type command to continue
    但是使用 ls 看不到这个文件：
    (base) hlc@hlc-VirtualBox:~$ ls -lh /tmp/vvWIsyd/0.cu
    ls: cannot access '/tmp/vvWIsyd/0.cu': No such file or directory
    ```

    这个问题很常见，是因为你对 Vim 的 scp 功能工作原理有误解。让我详细解释一下：

    **问题原因**

    当你使用 `vim scp://user@host/path/to/file` 时，Vim 实际上是这样工作的：

    1. **不直接在本地创建文件**：Vim 不会在本地 `/tmp` 目录创建永久的文件副本
    2. **临时存储**：Vim 会在内存或临时位置处理文件内容
    3. **路径显示是虚拟的**：显示的文件路径只是一个临时的、虚拟的表示
 
* vim 中 `set completeopt=menuone,noinsert,noselect` 是什么意思？
    `completeopt=menu,preview` 又是什么意思？

    在 Vim 中，`completeopt`（complete options）是一个用于控制**自动补全菜单**行为和外观的选项。它决定了补全窗口如何显示、如何选择、以及是否预览。

    下面为你详细拆解这两组配置的含义。

    ---

    **第一组：`set completeopt=menuone,noinsert,noselect`**

    这组配置非常经典，通常用于让补全体验更接近现代 IDE（输入时不自动选中，不自动插入，只显示菜单）。

    - **`menuone`**：
      - **含义**：即使只有一个匹配项，也**显示补全菜单**。
      - **默认行为**：如果只有一个匹配项，Vim 通常不会弹出菜单，而是直接补全。
      - **效果**：加上这个选项后，即使只有一个结果，也会弹出下拉菜单让你看到它，方便你按 `<C-y>` 确认或按 `<Esc>` 取消。

    - **`noinsert`**：
      - **含义**：在弹出菜单时，**不要自动插入**当前选中的文本。
      - **默认行为**：当你打开补全菜单时，第一个匹配项的文字会自动填入光标处（高亮选中状态）。
      - **效果**：加上后，你输入的文本保持不变，菜单只是悬浮在旁边，直到你按回车或 `<C-y>` 才真正替换文本。这可以防止误操作导致文本被覆盖。

    - **`noselect`**：
      - **含义**：在弹出菜单时，**不要自动高亮选中**第一项。
      - **默认行为**：菜单弹出后，第一项默认被高亮（选中）。
      - **效果**：加上后，菜单弹出时光标停留在你原本输入的位置，菜单中的项目没有被选中（只是显示列表）。如果你按方向键下，才会开始选择。

    > **组合效果**：
    > 这种组合让你在打字时，补全菜单只是一个安静的“参考提示”。你可以继续打字缩小范围，想用的时候按 `<C-n>` 或方向键去选择，按 `<C-y>` 确认。**不会自动选中，不会自动修改你的代码**，非常安全。

    ---

    **第二组：`set completeopt=menu,preview`**

    这组配置是 Vim 的传统默认风格，加上了一个额外信息窗口。

    - **`menu`**：
      - 含义：显示下拉补全菜单（默认行为）。
      - 如果没有 `menuone`，当只有一个匹配项时菜单可能不会弹出。

    - **`preview`**：
      - 含义：在 Vim 顶部或底部打开一个**预览窗口（Preview Window）**。
      - 效果：当你在补全菜单中上下移动选择时，预览窗口会显示当前选中函数或变量的**详细声明信息**（例如函数的参数列表、返回类型、文件路径等）。
      - **缺点**：预览窗口会占用屏幕空间，且有时会自动关闭，有时需要手动用 `<C-w>z` 关闭，部分用户会觉得它干扰视线。

    > **组合效果**：
    > 这是比较朴素的补全风格。有下拉菜单，且选择时顶部会显示详细的帮助信息。适合需要看参数细节的场景，但如果你觉得窗口乱跳，通常会去掉 `preview`。

    ---

    **补充知识：常用其他值**

    - **`longest`**：只插入匹配的最长公共前缀，不自动补全完整单词（适合与 `noinsert` 配合）。
    - **`popup`**（Neovim）：使用浮动窗口替代传统下拉菜单（更美观）。

    **你的实际场景建议**

    - 如果你**不想干扰打字**，用第一组（`menuone,noinsert,noselect`）。
    - 如果你**需要看函数参数详情**，可以加上 `preview`，例如：
      ```vim
      set completeopt=menuone,noinsert,noselect,preview
      ```

    如果你想永久生效，将 `set` 命令放入 `~/.vimrc`（或 `~/.config/nvim/init.vim`）即可。

* 讲讲 vim 中 `set shortmess+=c` 的作用

    在 Vim 中，`set shortmess+=c` 的作用是**抑制补全菜单（ins-completion-menu）相关的消息提示**。

    具体来说，它会隐藏你在插入模式下触发自动补全时，命令行中出现的各种状态信息。这些信息包括但不限于：

    *   **补全模式提示**：例如 `"-- XXX completion (YYY)"`，指示当前补全正在进行。
    *   **匹配状态信息**：例如 `"match 1 of 2"`（匹配 1/2）、`"The only match"`（唯一匹配）、`"Pattern not found"`（未找到模式）等。
    *   **导航状态**：例如 `"Back at original"`（回到起始位置）。

    **原理与背景**

    `'shortmess'` 是一个用于控制 Vim 命令行消息详细程度的选项，它由一系列标志位组成，每个标志位对应抑制某类消息。其中的 `c` 标志位就是专门为补全菜单消息引入的。

    这个功能是在 **Vim 7.4.314** 版本中添加的。它最初是为了解决插件在自动补全时，这些频繁出现的消息会干扰工作流程的问题。在更新版本的 Vim（如 9.0.0738）中，这个标志位也被用于压制后台补全扫描时产生的 "scanning" 信息。

    **如何使用**

    在你的 Vim 配置文件（vimrc）中添加以下命令即可生效：

    ```vim
    set shortmess+=c
    ```

    这里的 `+=` 操作符是在 `shortmess` 选项的现有值上**追加** `c` 标志，不会影响其他的消息显示设置，是一种安全且推荐的做法。

* 讲讲 vim 中 `inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"` 这行命令的作用

    这行命令的作用是**让 `<Tab>` 键在补全菜单弹出时变为“选择下一项”，而在没有补全菜单时保持正常的“插入制表符”功能**。

    我们把它拆解开来看：

    ---

    1. 整体结构：`inoremap <expr> <Tab> ...`

        - `inoremap`：表示**插入模式**下的**非递归**映射。
        - `<expr>`：表示右边的映射是一个**表达式**，Vim 会先计算它的值，然后把计算结果作为按键输入。
        - `<Tab>`：要映射的按键，即键盘上的 Tab 键。

    ---

    2. 条件判断：`pumvisible()`

        `pumvisible()` 是 Vim 的内置函数，返回值是：
        - `1`（true）—— 如果补全菜单（Popup Menu）当前在屏幕上可见。
        - `0`（false）—— 如果补全菜单没有显示。

    ---

    3. 三目运算符：`? ... : ...`

        这是一个条件表达式，格式为 `条件 ? 结果1 : 结果2`：
        - 如果 `pumvisible()` 为真（菜单可见），返回 `"\<C-n>"`。
        - 如果为假（菜单不可见），返回 `"\<Tab>"`。

    ---

    4. 两个返回值分别代表什么

        | 返回值 | 含义 |
        |--------|------|
        | `"\<C-n>"` | 在补全菜单中**选择下一个匹配项**（next）。这和手动按 `Ctrl+N` 的效果一样。 |
        | `"\<Tab>"` | 普通 Tab 键，即在光标位置**插入一个制表符**（或根据 `expandtab` 设置插入空格）。 |

    ---

    5. 实际使用效果

        - **没有补全菜单时**：按 Tab → 正常插入制表符/空格，跟平时一样。
        - **补全菜单弹出时**（比如你按了 `Ctrl+P` 或 `Ctrl+N` 触发了补全）：按 Tab → 菜单中的高亮光标向下移动一行，选择下一个候选项。

    ---

    6. 一个常见的配套映射

        通常这行命令会和下面这行成对出现：

        ```vim
        inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<Tab>"
        ```

        它的作用是：
        - 菜单可见时，`Shift+Tab` 选择**上一个**匹配项（`Ctrl+P`，previous）。
        - 菜单不可见时，正常输入 Tab（因为 Shift+Tab 在 Vim 中默认也等同于 Tab）。

        这样你就可以用 **Tab / Shift+Tab** 在补全列表中上下浏览，而不必使用 `Ctrl+N` / `Ctrl+P`，操作更加顺手。

    ---

    7. 注意事项

        - 如果你使用 **YouCompleteMe**、**coc.nvim** 或其他补全插件，它们可能会覆盖 Tab 键的行为。这时你可能需要调整插件的设置，或者使用插件的专用 API 来实现类似功能。
        - 如果你希望在补全时按 **Enter** 直接选中当前项，通常还需要额外配置，比如：
        ```vim
        inoremap <expr> <CR> pumvisible() ? "\<C-y>" : "\<CR>"
        ```
        这会让 Enter 在菜单可见时**确认选择**（`Ctrl+Y`），否则换行。

* 再讲讲 `inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"` 这行命令的作用

    这行命令的作用是**让 `Shift+Tab` 键在补全菜单弹出时变为"选择上一项"，而在没有补全菜单时保持正常的 `Shift+Tab` 功能**。

    ---

    ### 1. 整体结构：`inoremap <expr> <S-Tab> ...`

    - `inoremap`：**插入模式**下的**非递归**映射。
    - `<expr>`：右边的映射是一个**表达式**，Vim 会先计算表达式的值，再将结果作为按键输入。
    - `<S-Tab>`：要映射的按键，即键盘上的 **Shift+Tab** 组合键。

    ---

    ### 2. 条件判断：`pumvisible()`

    `pumvisible()` 检查补全菜单是否可见：
    - 返回 `1`（true）—— 菜单当前在屏幕上显示
    - 返回 `0`（false）—— 菜单未显示

    ---

    ### 3. 三目运算符：`? ... : ...`

    - 如果菜单可见（true），返回 `"\<C-p>"`
    - 如果菜单不可见（false），返回 `"\<S-Tab>"`

    ---

    ### 4. 两个返回值分别代表什么

    | 返回值 | 含义 |
    |--------|------|
    | `"\<C-p>"` | 在补全菜单中**选择上一个匹配项**（previous）。这和手动按 `Ctrl+P` 的效果一样，高亮条会向上移动。 |
    | `"\<S-Tab>"` | 普通的 Shift+Tab 键。在 Vim 中，Shift+Tab 默认与 Tab 功能相同（插入制表符/空格），所以这里实际返回的是普通的 Tab 键行为。 |

    ---

    ### 5. 实际使用效果

    - **没有补全菜单时**：按 Shift+Tab → 正常插入制表符/空格（与普通 Tab 一样）
    - **补全菜单弹出时**：按 Shift+Tab → 菜单中的高亮光标向上移动一行，选择上一个候选项

    ---

    ### 6. 与 Tab 映射的配合使用

    这行命令通常和上一讲的 Tab 映射成对出现：

    ```vim
    " Tab 键：菜单可见时选择下一项
    inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"

    " Shift+Tab：菜单可见时选择上一项
    inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
    ```

    这样组合的效果是：
    - **Tab**：在补全列表中**向下**浏览（下一项）
    - **Shift+Tab**：在补全列表中**向上**浏览（上一项）
    - 没有补全菜单时，两个键都正常输入 Tab

    这让你可以用方向性的 Tab 键来浏览补全列表，体验上更符合直觉——Tab 向下，Shift+Tab 向上。

    ---

    ### 7. 为什么返回 `"\<S-Tab>"` 而不是 `"\<Tab>"`？

    在你提供的这行命令中，当菜单不可见时返回的是 `"\<S-Tab>"`，而不是 `"\<Tab>"`。这是因为：

    - 技术上，两者在 Vim 插入模式下的效果是一样的（都插入制表符/空格）
    - 使用 `"\<S-Tab>"` 保持了按键的原始语义，在某些终端或配置中可能更安全
    - 实际上你也可以写成 `"\<Tab>"`，效果相同

    ---

    ### 8. 一个完整的使用场景

    假设你正在写代码，触发了自动补全（比如输入 `std::` 后弹出方法列表）：

    1. 补全菜单弹出
    2. 你按 **Tab** → 高亮向下移动，选择下一个候选
    3. 按 **Shift+Tab** → 高亮向上移动，选择上一个候选
    4. 找到想要的方法后，按 **Enter**（配合之前提到的 `<CR>` 映射）确认选择

    ---

    ### 9. 注意事项

    - **终端兼容性**：在某些旧终端或通过 SSH 连接时，`Shift+Tab` 的按键码可能不被正确识别。如果发现不生效，可以检查 `:verbose imap <S-Tab>` 查看当前映射状态。
    - **插件冲突**：像 coc.nvim、YouCompleteMe 等补全插件可能接管了 Tab 键的控制权，你可能需要使用插件提供的专门配置方式（如 coc 的 `suggest.enable` 相关设置）。
    - **Vim 版本**：确保你的 Vim 版本支持 `<expr>` 映射（大多数现代版本都支持）。
    

* vim 补全的按键映射

    ```vim
    " 映射快捷键
    inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
    inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
    inoremap <expr> <CR> pumvisible() ? "\<C-y>" : "\<CR>"
    ```

* vim 中的 filetype

    ```vim
    filetype on          " 开启文件类型检测（让 Vim 能识别 .py 是 python，.c 是 c）
    filetype plugin on   " 开启文件类型插件（加载 ftplugin 下的优化设置）
    filetype indent on   " 开启文件类型缩进（加载 indent 下的缩进规则）
    " filetype plugin indent on  也可以这么写，打开两个。不知道顺序如何
    ```

    `:filetype`: 验证当前是否开启, Vim 会回显当前状态，如果显示 `detection:ON plugin:ON indent:ON`，就说明功能已激活。

    filetype plugin 指的是 Vim 自带的“文件类型专用插件”，它们存放在 Vim 安装目录的 `ftplugin/` 文件夹里。例如，当你打开 test.c 文件时，Vim 会自动加载 `ftplugin/c.vim`。这个自带脚本会帮你设置好 C 语言的注释格式、自动缩进距离（cinoptions）等。

    系统的 vim 的 ftplugin 文件夹在`/usr/local/share/vim/vim92/ftplugin`, `/usr/share/vim/vim82/ftplugin`

    **进阶提醒（避坑指南）**

    * 位置顺序：`filetype plugin on` 必须放在 `syntax on`（语法高亮）之后，否则某些语法高亮可能会被插件覆盖掉。

    * 覆盖风险：如果你自己写了自定义配置（比如 `set tabstop=4`），而某个 ftplugin 里设置了 `set tabstop=8`，插件的设置会覆盖你的全局设置。如果你不想被覆盖，可以在 `filetype plugin on` 之后再写你的自定义设置，或者在 `after/ftplugin/` 目录下创建同名文件来覆盖默认行为。

* `set omnifunc=syntaxcomplete#Complete`

    设置 Vim 的“全能补全”（Omni Completion）函数为基于语法文件的补全。

    在输入代码时，按下特定的快捷键（默认是 <C-x><C-o>），Vim 会弹出补全菜单，为你提供当前文件中已经出现过的单词或文本匹配项。

    详解：

    * `omnifunc` 是 Vim 的一个选项（Option），它专门用来指定“全能补全”调用的函数名称。

    * syntaxcomplete#Complete 是这个函数的具体路径。# 代表 Vim 脚本中的命名空间分隔符，意思是调用 syntaxcomplete 插件里的 Complete 函数。

    * 当你设置了这行命令，Vim 就把“全能补全”的功能绑定到了当前缓冲区内的语法高亮元素上。它会扫描当前文件，识别出那些被语法高亮定义为“标识符”（Identifier）、“函数”（Function）、“类型”（Type）等的词汇，然后将其作为候选项提供给你。

    在默认情况下，Vim 的 omnifunc 是空的。如果不设置，你按 <C-x><C-o> 会收到 E764: Option 'omnifunc' is not set 的错误提示。

    **与普通补全的对比**

    | 特性 | 普通补全 (<C-n> / <C-p>) | 全能补全 (<C-x><C-o>) |
    | - | - | - |
    | 依据 | 基于纯文本匹配（当前文件、打开的文件、字典） | 基于语法结构匹配（只关注代码中的变量、函数名） |
    | 过滤性 | 会弹出所有包含相同字母的单词（包括注释里的英文、字符串里的字符） | 只弹出符合当前语言语法规范的代码元素 |
    | 速度 | 在超大文件中可能较慢（扫描所有文本） | 通常更快（只查语法高亮标记的区域） |
    | 智能程度 | 低（不懂代码结构） | 中（懂一点语法，但不具备语义分析能力） |

    **注意事项（避坑）：**

    * 不要放在全局 `vimrc` 中单独使用！ 
    
        因为不同文件类型需要不同的补全函数。例如 Python 有 `pythoncomplete#Complete`，JavaScript 有 `javascriptcomplete#Complete`。如果你直接全局设置成 `syntaxcomplete#Complete`，它会覆盖语言专用的补全。

    * 正确的搭配写法：通常这行命令不会直接写在配置文件里，而是配合 `autocmd`（自动命令）针对特定文件类型设置，或者利用 `filetype plugin on` 来自动加载

        `autocmd FileType python set omnifunc=python3complete#Complete`

* vim 中 `set ff=unix`

    ff 是 fileformat 的缩写。Vim 用这个选项来判断文件以哪种换行符标准来解析内存中的缓冲区：

    * ff=unix：换行符为 LF（Line Feed，即 \n，十六进制 0x0A）。

    * ff=dos：换行符为 CRLF（Carriage Return + Line Feed，即 \r\n，十六进制 0x0D 0x0A）。

    * ff=mac（旧版）：换行符为 CR（\r，十六进制 0x0D）。

    当你执行 `:set ff=unix` 时，Vim **会**同时修改文件缓冲区里的实际字节内容。

    常用的三种转换方式：

    ```vim
    :set ff=unix
    :w
    ```

    ```vim
    :%s/\r$//g
    ```

    ```bash
    :!dos2unix %  # （% 代表当前文件名）
    ```

* exec "normal! \<C-Y>" 可以写成 exec normal! \<C-Y> 吗

    **不行，不能直接去掉引号。** 如果你写成 `exec normal! \<C-Y>`，Vim 会报错或者无法达到你预期的效果（通常会提示找不到后面的命令或把它当作普通文本解析）。

    这里有两个核心原因：

    1. `\<C-Y>` 需要被双引号转义

        在 Vimscript 中，`\<C-Y>` 这种表示快捷键的特殊字符串，**只有在双引号 `""` 中**才会被 Vim 的 `expr-quote` 机制解析为真正的键盘信号（即 `Ctrl + Y`）。

        * 如果你不用双引号，Vim 只会把它当成纯文本的 `\`、`<`、`C`、`-`、`Y`、`>` 这几个字符。

    2. `execute` (exec) 的语法要求

        `execute` 命令的作用是**将后面的表达式计算为一个字符串，然后把这个字符串当作命令执行**。

        * `exec "normal! \<C-Y>"`：Vim 先把表达式计算成 `normal! [实际的Ctrl+Y信号]`，然后执行它。
        * `exec normal! \<C-Y>`：因为没有引号，Vim 会试图去寻找名为 `normal!` 的变量或表达式，这会导致语法错误。

    **更好的替代方案**

    如果你嫌写 `exec "normal! ..."` 太麻烦，其实你**根本不需要使用 `exec**`。在绝大多数情况下，直接使用 `normal!` 配合 `\e` 或特殊处理是不行的，但在 Vimscript 脚本中，你可以直接这样写：

    ```vim
    " 使用 nnoremap 或直接在脚本中执行
    normal! 
    ```

    *(注：上面的 `` 是通过在 Vim 中按 `Ctrl+V` 然后按 `Ctrl+Y` 输入的字面量控制字符)*

    或者，如果你是在映射（Mapping）中使用，直接用 `noremap` 系列命令即可，完全不需要 `exec`：

    ```vim
    nnoremap <leader>y :normal! <C-Y><CR>
    ```

    **总结：** 只要用了 `exec`，后面的内容就必须是一个能计算出字符串的表达式，所以双引号和内部的转义是免不了的。

* vim 总是在 terminal 的最后一行输入，是否有办法让它在 terminal 的中间或者中间偏下的位置输入？

    是的，有几种方法可以调整 Vim 在终端中的显示位置，让它不在最后一行开始输入：

    ```vim
    " 总是显示状态行（即使只有一个窗口）
    set laststatus=2

    " 启用标尺显示当前行列信息
    set ruler

    " 在光标上下保留多少行（会让光标在中间）
    set scrolloff=5

    " 左右滚动保留的列数
    set sidescrolloff=10
    ```

    支持自定义布局的终端（如 iTerm2、Kitty、Alacritty）：

    使用屏幕居中插件:

    - [vim-smoothie](https://github.com/psliwka/vim-smoothie)：平滑滚动
    - [vim-peekaboo](https://github.com/junegunn/vim-peekaboo)：更好的寄存器显示
    - [goyo.vim](https://github.com/junegunn/goyo.vim)：专注模式，让内容居中

    调研这个别名的含义：

    ```bash
    alias vimcenter='clear; printf "\n%.0s" {1..10}; vim'
    ```

    ```vim
    " 启用平滑滚动（需要Vim 9.0+）
    set smoothscroll
    " 这个有啥用？

    " 设置平滑滚动的行数
    set scrolloff=8
    ```

    **插件解决方案**

    - [vim-smoothie](https://github.com/psliwka/vim-smoothie)：提供更平滑的滚动
    - [vim-scrolloff-fix](https://github.com/svermeulen/vim-scrolloff-fix)：增强的scrolloff功能

    1. **使用 `autocmd` 自动执行 `zz`**

    ```vim
    " 在特定操作后自动居中屏幕
    augroup AutoCenter
      autocmd!
      " 进入插入模式时居中（可能太频繁）
      autocmd InsertEnter * normal! zz
      
      " 或者：在特定跳转后居中
      autocmd CursorMoved * if line('.') == line('$') | normal! zz | endif
      
      " 更好的：只在到达文件末尾时居中
      autocmd CursorMoved * if line('.') >= line('$') - 1 | normal! zz | endif
    augroup END
    ```

    **使用 `CursorHold` 事件**

    ```vim
    " 当光标停留一段时间后自动居中（需要设置 updatetime）
    set updatetime=1000  " 1秒后触发 CursorHold

    augroup AutoCenter
      autocmd!
      autocmd CursorHold * if line('.') == line('$') | normal! zz | endif
    augroup END
    ```

    **专门针对插入模式尾部的居中**

    注：

    1. 这个说不定能改一下，把 zz 改成恒间隔 5 行就好了

    ```vim
    " 只在文件末尾进入插入模式时居中
    function! CenterIfAtBottomInsert()
      if line('.') == line('$')
        normal! zz
      endif
    endfunction

    augroup CenterBottomInsert
      autocmd!
      " 在插入模式开始时检查
      autocmd InsertEnter * call CenterIfAtBottomInsert()
      " 或者在离开插入模式时居中（这样不会干扰输入）
      autocmd InsertLeave * if line('.') == line('$') | normal! zz | endif
    augroup END
    ```

* vim 如何像 vscode 那样打断点？

    方法一：使用专用调试插件（最推荐，最接近 VSCode 体验）

    这类插件提供了类似现代 IDE 的调试界面，包括侧边栏断点标记、变量监视、调用栈查看等。

    1. vimspector

        这是目前功能最强大、体验最接近 VSCode 的 Vim 调试插件。它支持多种语言（C/C++, Python, Go, JavaScript, Rust 等），通过配置文件来适配不同的调试器。

        特点：

        * 有独立的图形化窗口显示变量、调用栈、断点等。
        * 支持条件断点、日志断点。
        * 需要在项目根目录创建一个 `.vimspector.json` 配置文件来告诉插件如何使用调试器。

        安装（使用 vim-plug）：

        ```vim
        Plug 'puremourning/vimspector'
        ```

        基本使用：

        1.  **安装插件**并执行 `:PlugInstall`。

        2.  **安装调试器适配器**，例如对于 Python，需要安装 `debugpy`：`pip install debugpy`。

        3.  **创建配置文件** `.vimspector.json`（以 Python 为例）:

            ```json
            {
              "configurations": {
                "Python: Launch": {
                  "adapter": "debugpy",
                  "configuration": {
                    "name": "Launch Python",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal"
                  }
                }
              }
            }
            ```

        4. 使用：

            *   `:VimspectorInstall`：安装调试适配器。
            *   `<F5>`：启动/继续调试。
            *   `<F9>`：在当前行**切换断点**。
            *   `<F10>`：单步跳过。
            *   `<F11>`：单步进入。
            *   `:VimspectorReset`：停止调试。

        官网：[https://github.com/puremourning/vimspector](https://github.com/puremourning/vimspector)

    2. nvim-dap (NeoVim 专属)

        如果你是 NeoVim 用户，`nvim-dap` 是一个类似于 vimspector 但更模块化的调试插件。它本身是核心，需要配合其他 UI 插件（如 `nvim-dap-ui`）来提供图形界面。

        特点：

        *   高度可定制和模块化。
        *   社区活跃，扩展性强。

        安装与配置示例：

        ```vim
        " 使用 packer.nvim
        use { "mfussenegger/nvim-dap" }
        use { "rcarriga/nvim-dap-ui", requires = {"mfussenegger/nvim-dap"} }
        ```

        ```lua
        -- 在 Lua 配置中
        local dap = require('dap')
        local dapui = require('dapui')

        -- 设置适配器（例如 Python）
        dap.adapters.python = {
          type = 'executable';
          command = 'path/to/your/python';
          args = { '-m', 'debugpy.adapter' };
        }
        dap.configurations.python = {
          {
            type = 'python';
            request = 'launch';
            name = "Launch file";
            program = "${file}";
            pythonPath = function() return '/usr/bin/python3' end;
          },
        }

        -- 自动打开 UI
        dap.listeners.after.event_initialized['dapui_config'] = function()
          dapui.open()
        end
        dap.listeners.before.event_terminated['dapui_config'] = function()
          dapui.close()
        end
        dap.listeners.before.event_exited['dapui_config'] = function()
          dapui.close()
        end
        ```

        **基本使用：**

        *   `<F5>`：开始调试（需要在配置中映射）。
        *   `:lua require'dap'.toggle_breakpoint()`：切换断点。
        *   `:lua require'dap'.step_over()`：单步跳过。

    方法二：使用更传统的插件（轻量级）

    **Termdebug (Vim 8.1+ / NeoVim 内置)**

    这是 Vim 自带的插件，非常轻量，适合 C/C++ 调试（使用 GDB）。

    **特点：**
    *   无需安装，开箱即用。
    *   界面传统，分为源代码、GDB 命令和程序输出三个窗口。

    **启用和使用：**
    1.  在 vim 中执行：
        ```vim
        :packadd termdebug
        :Termdebug <你的可执行文件路径>
        ```
    2.  会打开三个窗口。
    3.  在源代码窗口中，将光标移动到某行，按 `<F8>` 即可在该行**设置/取消断点**。
    4.  在 GDB 窗口中可以输入任何 GDB 命令。

    方法三：手动与 GDB/LLDB 集成（最灵活，学习曲线高）

    如果你不想用插件，可以直接在 Vim 的终端模式下运行 GDB，或者通过 `:!gdb` 执行。

    1.  **在 Terminal 模式下运行**：

        ```vim
        :term gdb your_program
        ```

        然后就可以直接使用 GDB 命令，如 `b main`（在 main 函数设断点），`run` 等。

    2.  **通过快捷键发送命令**：

        你可以映射一些 Vim 快捷键来向 GDB 发送特定命令。这种方式比较原始，但非常灵活。

    总结与建议

    | 方法 | 适用场景 | 优点 | 缺点 |
    | :--- | :--- | :--- | :--- |
    | **vimspector** | **所有 Vim/NeoVim 用户，追求接近 VSCode 的体验** | 功能强大，图形化好，支持多语言 | 需要额外配置文件 |
    | **nvim-dap** | **NeoVim 用户，喜欢高度定制** | 模块化，可扩展性强 | 配置相对复杂 |
    | **Termdebug** | **Vim 8.1+ 用户，快速调试 C/C++** | 轻量，无需安装 | 功能相对基础，界面传统 |
    | **手动 GDB** | **高手，需要最大灵活性** | 完全控制，无需依赖 | 学习曲线高，体验不友好 |

    **给你的建议：**

    *   如果你想要一个**开箱即用、功能全面、最像 VSCode** 的体验，请直接选择 **vimspector**。
    *   如果你是 **NeoVim 用户**并且喜欢折腾和定制，可以尝试 **nvim-dap + nvim-dap-ui** 的组合。
    *   如果你只是偶尔**快速调试一个 C/C++ 程序**，可以试试内置的 `Termdebug`。

    无论选择哪种，你都能在 Vim 里获得强大的调试能力，一旦熟悉，效率会非常高。



* 详细解释下`:%g/^/norm! jdd`

    这行 Vim 命令的核心作用是：**删除文本中所有的偶数行**（或者说是每隔一行删除一行）。

    我们可以把它拆解为四个部分来详细分析：`:`、`%g/^/`、`norm!` 和 `jdd`。

    ---

    1. `:` (命令行模式)

        进入 Vim 的命令行模式（Command-line mode），允许你输入底行命令。

    2. `%g/^/` (全局匹配)

        这是 Vim 中非常强大的 `global` 命令，格式通常为 `: [范围]g/模式/内容`。

        * **`%`**：表示范围是**全文**（从第一行到最后一行）。
        * **`g`**：`global` 命令的缩写，用来寻找所有匹配特定模式的行，并对这些行执行后面的操作。
        * **`/^/`**：这是正则表达式模式。`^` 代表**行首**。因为每一行都有行首，所以这个模式会**匹配文件中的每单一行**。

        > 💡 **核心机制**：`:g` 命令会分两步执行。第一步，它从上到下扫描全文，把所有匹配 `^` 的行（即所有行）都打上标记；第二步，它依次走到这些被标记的行上，执行后面的命令。

    3. `norm!` (执行普通模式命令)

        * **`norm`**：`normal` 命令的缩写。它告诉 Vim：“把我后面跟着的字符，当作是在普通模式（Normal mode）下敲击的键盘快捷键来执行”。
        * **`!`**：表示**不使用自定义映射（mappings）**。这确保了后面执行的快捷键是 Vim 原生的功能，即使你在 `.vimrc` 里把某个键改了功能，带 `!` 也能保证命令行为不受影响。

    4. `jdd` (具体的普通模式操作)

        这是真正对文本进行修改的动作。当 `:g` 命令停在某一行时，它会模拟你按下键盘上的 `j` 和 `d` `d`：

        * **`j`**：光标**向下移动一行**。
        * **`dd`**：**删除当前行**。

    ---

    **动态执行过程演示**

    假设你有一个 4 行的文件：

    ```text
    Line 1  <-- :g 命令首先标记并停在这里
    Line 2
    Line 3
    Line 4
    ```

    1. `:g` 停在 **Line 1**。
    2. 执行 `norm! jdd`：
    * `j` 让光标向下移动到 **Line 2**。
    * `dd` 删除了 **Line 2**。


    3. 此时 **Line 2** 没了，文件变成了：

        ```text
        Line 1
        Line 3  <-- 原来的第3行，现在变成了第2行
        Line 4
        ```

    4. `:g` 命令继续处理下一个被标记的行。虽然文本变了，但 `:g` 记得它下一个要访问的是“原文件的第二行”。因为原第二行已经被删了，它会直接跳到**原文件的第三行**（即现在的 Line 3）。

    5. `:g` 停在 **Line 3**，再次执行 `norm! jdd`：

        * `j` 让光标向下移动到 **Line 4**。
        * `dd` 删除了 **Line 4**。

    **最终结果**：所有的偶数行（2, 4, 6...）都被删除了，只留下了奇数行。

    ---

    **补充：更高效的替代方案**

    虽然 `:%g/^/norm! jdd` 很好理解，但因为 `:g` 在滚动时行号会发生变化，有时候处理大文件可能会遇到跳行不准的边界问题。

    如果你单纯想**删除偶数行**，Vim 里有一个更标准、更暴力的写法：

    ```vim
    :%normal jdd
    ```

    或者利用寄存器宏，甚至用更现代的 `filter` 组合。不过你给出的这行命令，可以说是理解 Vim `global` 和 `normal` 组合拳的经典教材。

    注：

    1. `jdd`之所以只删除偶数行，是因为 vim 是逐行处理的。

        并不是 multi cursor，然后并行处理。




* 详细讲讲 vim 中 snippet 的用法

    在 Vim 中，**Snippet（代码片段）** 是提高输入效率的绝对神器。简单来说，它就像是“代码宏”：你只需要输入一个简短的缩写（比如 `for`），然后按下快捷键，Vim 就会自动把它展开成一段完整的代码结构，并允许你用 `Tab` 键在不同的占位符之间快速跳跃修改。

    要在 Vim 中高效使用 Snippet，通常需要依赖**插件**。下面为你详细拆解 Vim Snippet 的核心概念、主流插件组合、配置方法以及如何编写自己的 Snippet。

    1. 核心生态：插件组合

        Vim 原生对高级 Snippet 的支持较弱，社区通常采用“引擎 + 库”的组合模式：

        * **Snippet 引擎（负责解析和跳跃）：**
        * **UltiSnips**：功能最强大、最经典的引擎（支持在 Snippet 中嵌入 Python 代码）。
        * **LuaSnip**：Neovim 用户目前的首选，速度极快，配置灵活性极高。
        * **vim-vsnip**：轻量级，完美支持 VS Code 格式的 Snippet。


        * **Snippet 语料库（负责提供现成的代码片段）：**
        * **honza/vim-snippets**：一个巨大的社区维护库，包含了几乎所有主流编程语言的常用片段。


        > 💡 **本指南以最经典的 `UltiSnips` + `vim-snippets` 组合为例进行讲解。**

    2. 基础安装与配置（以 vim-plug 为例）

        在你的 `.vimrc` 中添加以下插件：

        ```vim
        call plug#begin('~/.vim/plugged')

        " Snippet 引擎
        Plug 'SirVer/ultisnips'

        " 现成的代码片段库
        Plug 'honza/vim-snippets'

        call plug#end()

        ```

        **核心快捷键配置**

        UltiSnips 默认的展开快捷键是 `<tab>`，但这经常会和很多补全插件（如 Coc.nvim 或 YouCompleteMe）冲突。建议在 `.vimrc` 中重映射：

        ```vim
        " 触发展开的快捷键（改为 Ctrl + j）
        let g:UltiSnipsExpandTrigger="<c-j>"

        " 跳向下一个占位符的快捷键
        let g:UltiSnipsJumpForwardTrigger="<c-j>"

        " 跳向上一个占位符的快捷键
        let g:UltiSnipsJumpBackwardTrigger="<c-k>"

        " 如果你想定义自己的 snippet 存放目录
        let g:UltiSnipsSnippetDirectories=["UltiSnips", "my_snippets"]

        ```

    3. 实战用法：如何使用 Snippet？

        安装配置完成后，重启 Vim。假设你在编写一个 `main.py` 的 Python 文件：

        1. **输入触发词**：在插入模式下输入 `def`。
        2. **触发展开**：按下你设置的快捷键 `Ctrl + j`。
        3. **瞬间展开**：代码会变成：
        ```python
        def fname(arg):
            """docstring for fname"""
            pass

        ```


        4. **光标跳跃**：此时光标会停留在 `fname` 上。将其修改为 `my_function`，然后再次按下 `Ctrl + j`，光标会自动跳到 `arg`，修改完参数后再按 `Ctrl + j`，光标会跳到 `docstring`。
        5. **完成**：一气呵成，双手完全不需要离开主键盘区去按方向键。

    4. 进阶：如何编写自己的自定义 Snippet？

        现成的库虽好，但根据自己的习惯定制 Snippet 才是精髓。

        使用命令 `:UltiSnipsEdit`，Vim 会自动为你当前打开的文件类型创建一个自定义的 snippet 文件（例如 `python.snippets`）。

        **Snippet 的基本语法**

        一个标准的 Snippet 定义格式如下：

        ```text
        snippet 触发词 "描述文本" 选项
        代码内容
        endsnippet

        ```

        示例 1：基础占位符 `${N}`

        ```text
        snippet if "If condition" b
        if ${1:condition}:
            ${2:pass}
        $0
        endsnippet

        ```

        * `b` 选项：表示只有当触发词 `if` 在**行首**时才触发。
        * `${1:condition}`：第 1 个停靠点，默认文本是 `condition`。
        * `${2:pass}`：第 2 个停靠点。
        * `$0`：最终退出 Snippet 时的光标所在位置。

        示例 2：镜像占位符（修改一处，多处联动）

        如果你在写 HTML 或一些重复度高的代码：

        ```text
        snippet tag "HTML tag"
        <${1:div}>
            $0
        </${1/(\w+).*/$1/}>
        endsnippet

        ```

        * 当你在 `<` 后面把 `div` 改成 `section` 时，闭合标签 `</div>` 会**同步实时**变成 `</section>`。

        示例 3：高阶玩法（嵌入 Python/Shell 脚本）

        UltiSnips 最强悍的地方在于支持行内代码。用 ` 包裹：

        ```text
        snippet today "Insert current date"
        Today is `date +%Y-%m-%d`.
        endsnippet

        ```

        输入 `today` 并展开，它会直接调用系统命令，变成 `Today is 2026-06-05`。

    5. 常见痛点与解决建议

        * **与自动补全（Autocomplete）冲突**：这是最常见的问题。如果你使用的是 `Coc.nvim`，建议让 Coc 来管理 Snippet，安装 `coc-snippets` 插件，它能完美地把 Snippet 融入到下拉补全菜单中，直接用 `Tab` 键一并解决。
        * **不知道有哪些现成的触发词**：可以去查阅 `~/.vim/plugged/vim-snippets/snippets/` 目录下对应语言的文件，或者在 Vim 中输入触发词的前几个字母，依赖补全插件的提示。

    你目前在使用哪种补全插件（如 Coc, YCM, 或 Neovim 原生 LSP）？如果有特定插件，我可以帮你针对性地优化 `Tab` 键的冲突配置。


* vim 中内容被选中后，内容会变白色，如何取消选中？我是用鼠标选中，然后 ctrl + shift + c 复制的内容

    你现在看到的 “内容变白色” 是 **终端的鼠标选区**，并不是 Vim 的可视模式（visual mode）。

    所以你需要“取消的是终端的选中状态”，而不是 Vim 内部状态。

    **取消终端鼠标选中（不退出 Vim，因为你不是在 Vim 里选的）**

    不同 terminal emulator 的取消方式一样：

    1. 点一下空白处（最简单）

        在 Vim 窗口外、终端外的任意空白区域单击一下即可清除选区。

        但这会让光标跑出 Vim 窗口，不太优雅。

    2. 在终端内随便点击鼠标左键一次

        在终端内部但不在选区上，点击一下左键也会取消选中。

    3. 按一次 `Esc`（很多终端有效）

        部分 terminal emulator（如 gnome-terminal, alacritty）中，按 `Esc` 会清除高亮选区。

    4. 用键盘移动（左右方向键）

        按一下左右方向键
        或
        按 `j` / `k`（如果你在 vim normal mode）
        终端就会清除鼠标区域选中。

    **为什么 vim 里 :set mouse=a 时会出现这种情况？**

    因为你：

    1. 用鼠标框选 → 这不是 Vim 的 visual 模式，而是终端自己的文本选区
    2. 终端为了复制文本，会把文字反白显示
    3. 复制完成后选区仍然存在

    Vim 无法取消终端级的选区，只能通过终端自身行为来清除。

    **如果你希望使用鼠标复制，但又不想让终端选区干扰 Vim？**

    可以设置：

    **让鼠标选区不进入 Vim，而总是当作终端选区：**

    ```vim
    :set mouse=
    ```

    这样鼠标永远不会进入 Vim 模式，只作为“终端文本选择”使用。
    复制更方便，但你不能用鼠标移动光标或 resize splits。

    如果你告诉我你用的是什么终端（gnome-terminal / kitty / alacritty / iTerm2 / Windows Terminal），我可以给出你终端专属的最优操作。

* vim 如何在当前行设置一个临时标记，当在别的地方做完修改后，再跳转回这个标记处？比如：

    hello

    * 两个反引号` `` ` = jump list 的上一个位置

    * 两个单引号`' '` = jump list 的上一个位置的行的开头

    使用`m`标记时，大写标记是全局的（任何文件都能跳），小写标记（如 `ma`）只在当前文件有效。

    * Jump list（跳转列表） 

        **概念**：Vim 会把会产生“明显位置变化”的动作记录到 jump list，例如：用 `/` 或 `?` 搜索、用 `n/N`、用 `G` 跳到文件尾、使用标签跳转、用 `:tag`、使用 `%` 匹配括号跳转、用 `Ctrl-]` 跳到 tag、使用 `gf`、跳到另一个文件（`:e file`）等。

        查看 jump list：

        ```
        :jumps
        ```

        输出会像这样（简化示例）：

        ```
        jump line  col file/text
         1  10     1  file1.txt
         2  42    10  otherfile.c
         ...
        ```

        左边的序号表示 jump-list 的顺序；第一个是最旧、最后一项是最近的（显示格式和细节可参考 `:help jumps`）。

        **注意**：不是每次普通移动都会进 jump list（比如按 `h/j/k/l` 或微小移动不会），只有“明显跳跃”才记录。

    * 在跳转历史中前进 / 后退：`<C-o>` 和 `<C-i>`

        * `Ctrl-O`：回到 jump list 中**更早**的位置（后退，go older）。

        * `Ctrl-I`：回到 jump list 中**更近**的位置（前进，go newer），也就是撤销上一次 `Ctrl-O` 的效果。

        几个要点 / 常见坑：

        * `Ctrl-I` 在很多终端里被当成 `Tab`（ASCII 相同），因此终端里可能不能区分 `<C-i>`。在 GUI（gvim）或某些终端/配置下没问题。如果你发现 `<C-i>` 无效，可以用 `:normal! <C-i>` 或在 GUI 测试，或换用 jump-list 的其他手段（见下）。

        * `:jumps` 能帮助你确认当前处于 jump list 的哪一项（Vim 也会显示当前位置索引）。

        * `Ctrl-O` / `Ctrl-I` 是基于 jump list 的，所以只能在 jump list 记录的点之间跳。

    * 手动标记（marks）
    
        同时维护多个临时位置（推荐用于“书签”）

        如果你想显式地保存若干“将来要回到的位置”，用 mark 是最直接的办法。

        设置标记：

        * 小写字母 `a`–`z`：局部标记（仅当前文件有效）

            ```vim
            ma    " 在当前光标位置设置标记 a（小写）
            ```

        * 大写字母 `A`–`Z`：全局标记（可跨文件）

            ```vim
            mA    " 在当前光标位置设置全局标记 A
            ```

        跳回标记：

        * 精确位置（行和列）：`` `a ``（反引号 + 标记字母）

          ```vim
          `a    " 精确回到标记 a 的行列
          ```

        * 跳到标记所在行（列不保证）：`'a`（单引号 + 标记字母）

          ```vim
          'a    " 跳到标记 a 的行（列移动到行首或首个非空）
          ```

    **示例使用流程**：

    1. 在编辑处按 `m1` 或 `ma`（例如 `ma`）标记当前行。
    2. 跳到别处做修改。
    3. 想回到那行就按 `` `a ``（或 `'a` 如果只需要行）。

    **为什么用大写？**

    大写标记 (`mA`) 是全局的，能跨文件跳转（适合你在多个文件间工作时做书签）。小写仅对当前缓冲有意义。

    **维护多个位置**：

    可以设置多个不同字母的标记：`ma`, `mb`, `mC` 等，然后随时用对应的 `` ` `` 跳回。非常适合“先记下几处关键位置，做完一堆修改再按字母回去检查”的工作流。

* vim `：Change list`（修改列表）和一些特殊标记

    除了 jump list 和 marks，Vim 还有 change list（记录改动位置），以及一些预定义的“特殊标记”。

    **Change list**（修改点历史）：

    * 查看修改列表：

      ```
      :changes
      ```

    * 在修改历史中前后跳：

        * `g;` 跳到上一个改动处（older change）

        * `g,` 跳到下一个改动处（newer change）
        这是在你想回到“最近修改过的位置”时最方便（比如逐个查看最近修改的几处）。

    **几个内置特殊标记（常用）**：

    * `` `. `` —— 上一次修改的位置（精确到列）。例如按 `` `. `` 可跳到最后一次修改的位置。（如果只需要精准到行，则使用 ` '. `）

    * `"` —— 上次退出该文件时光标所在位置（打开文件后可 `'"` 跳回上次光标处）。
    
    * `^` —— 上次插入结束位置（last insertion end）。

    * `[` 和 `]` —— 上一次变更或文本对象的开始/结束（常用于操作后定位）。
      这些特殊标记在 debug、查看修改上下文时很有用。

    * 常见小技巧 / 推荐映射

        * 快速返回“上一跳位置”（最实用）：

            ```
            nnoremap <leader><leader> ``   " 双反引号——回到上一个精确位置
            ```

        * 使用标记作书签组（举例）：

            ```
            ma   " 在 A 处设置书签
            mB   " 在 B 处设置跨文件书签
            `a   " 回到 A（精确）
            `B   " 回到 B（精确，跨文件）
            ```

        * 如果你的终端把 `<C-i>` 变成 Tab，且你想前进（forward），可以在 Vim 中利用 `:normal` 调用某个 jump-list 索引（高级），或者直接用 marks 来避免依赖 `<C-i>`。

    * 想跳到“最近改动处”来复查修改：用 `g;` / `g,`（change list）。




* `|` (管道符)：在 Vimscript 中，竖线用来在同一行隔开多个不同的命令（类似于 Python 或 JavaScript 中的分号 ;）。

* `execute`： 把后面的字符串当作 Vim 命令来执行。

* `normal!`：模拟在普通模式（Normal Mode）下输入后面的快捷键。加感叹号 ! 的目的是为了忽略用户自定义的映射（mappings），确保执行的是 Vim 原生的、最纯粹的命令。

* vim 中，如何 visual 模式下只选取指定的行，比如隔一行选一行

    在 Vim 中，标准的 **Visual Mode**（无论是 `v` 还是 `V`）都是设计为选取**连续**块的。Vim 原生并不支持在一个选择操作中直接跳过某些行。

    注：

    1. 感觉还是得自己写个脚本，比如按`\l`进入按行选择模式，然后每按一下 s 选择一行，最后按 q 退出选择模式

    * 我想选择指定的不规律的几行，比如 1, 4, 9, 12，然后对这些行应用一些指定的宏或函数，比如在每一行的开头添加 * ，删去结尾的空格，替换行里的一些字符等。

        在 Vim 原生操作中，由于 Visual 模式本质上是**连续选择**，要处理像 1, 4, 9, 12 这样不规律的行，通常有两种策略：**“标记后批量处理”** 或 **“多光标插件”**。

        考虑到你还要执行复杂的编辑（加开头、删结尾、替换），以下是最高效的方案：

        **方案一：使用标记位（Marks）+ 全局命令（无需插件）**

        这是最硬核、最 Vim 的做法。你可以先标记这些行，然后统一执行命令。

        1. **标记这些行：**

            手动移动到第 1, 4, 9, 12 行，分别输入 `ma`, `mb`, `mc`, `md`（`m` 是标记命令，后面跟着字母标签）。

            *或者更简单的：* 移动到每一行，输入 `:sign define...`（这比较麻烦），不如直接用 **全局搜索**。

        2. **执行复合命令：**

            如果你能找到这些行的共同特征（比如都包含某个关键词 `pattern`），直接运行：

            ```vim
            :g/pattern/s/^/* / | s/\s\+$// | s/旧字符/新字符/g
            ```

        **方案三：利用 Quickfix 列表（针对复杂筛选）**

        如果你这些行是通过某种搜索得到的：

        1. `:vimgrep /关键词/ %` (这会把匹配行放入 Quickfix 列表)。

        2. 使用 `:cdo` 命令对列表中的每一行执行操作：

            ```vim
            :cdo s/^/* / | s/\s\+$// | s/foo/bar/g | update
            ```

        *注：`cdo` 会遍历 Quickfix 中的每一行并执行后面的指令。*

        **方案四：如果你已经有一个行号列表**

        如果你已经确定了行号（1, 4, 9, 12），可以直接用一个简单的循环命令：

        ```vim
        :for line in [1, 4, 9, 12] | execute line . "normal I* " | execute line . "s/\\s\\+$//" | endfor
        ```

* 详细讲讲 vimdiff 的用法

    `vimdiff` 是 Vim 自带的**文件对比与合并工具**。它不仅能在终端里用高亮清晰地标出两个（甚至多个）文件的差异，还能让你用极快的纯键盘操作把差异进行合并。

    如果你在终端里看 `git diff` 觉得头疼，或者想在服务器上快速肉眼比对两个配置文件的不同，`vimdiff` 就是终极利器。

    1. 启动命令

        最直接的启动方式是在终端输入：

        ```bash
        vimdiff file1.txt file2.txt
        # 或者（效果完全一样）
        vim -d file1.txt file2.txt
        ```

        如果你已经打开了一个文件，想拿它和另一个文件比对：
        在 Vim 底线命令模式输入 `:diffsplit file2.txt`（水平分屏比对）或 `:vert diffsplit file2.txt`（垂直分屏比对）。

    2. 认识界面与高亮含义

        打开后，你会看到屏幕被垂直平分，最神奇的是：**当你在左边滚屏时，右边会同步滚动（Scrollbind）**。

        不同的颜色代表了不同的文件差异类型（具体颜色取决于你的 Vim 主题，但逻辑一致）：

        * **整行高亮（通常为淡蓝/淡紫）**：这行文本在两个文件中**不一致**（被修改过）。
        * **行内部分高亮（通常为红色/亮色）**：这一行里**具体被修改的字符**。
        * **连续的虚线/ `-+-- 15 lines`（折叠状态）**：两个文件中**完全相同的内容**。Vim 会自动把相同的内容折叠起来，避免干扰视线。
        * **填充底色/无文字行（通常为纯蓝色或空行）**：另一个文件在这里有新加的行，而当前文件**缺失这一行**。

    3. 核心快捷键：跳转与合并

        学会这 5 组快捷键，你就掌握了 `vimdiff` 的精髓。

        * 差异点跳转（结合了上一课讲的方括号）

            * **`]c`**：跳转到**下一个**差异点（Next change）。
            * **`[c`**：跳转到**上一个**差异点（Previous change）。

        * 协同合并（核心：Put 与 Get）

            当你的光标停留在某个差异点时，你可以决定是以左边为准，还是以右边为准：

            * **`do` (Diff Obtain/Get)**：**“拿过来”**。把另一个文件的内容复制到当前光标所在的文件，覆盖当前差异。
            * **`dp` (Diff Put)**：**“推过去”**。把当前文件的内容复制到另一个文件，覆盖那边的差异。

            > **💡 记忆口诀**：
            > * `dp` = **D**iff **P**ut（把我的改动**推**给对方）
            > * `do` = **D**iff **O**btain（把对方的改动**拿**到我这）
            > 

        * 折叠控制

            Vim 默认会把相同的行折叠（Fold）起来：

            * **`zo`**：展开（Open）当前光标处的折叠，查看相同的内容。
            * **`zc`**：重新折叠（Close）当前内容。

        * 退出命令

            * **`:qa`**：退出全部（Quit All）。如果你两个文件都没改，用这个最快。
            * **`:wqa`**：保存并退出全部（Write & Quit All）。合并完成后用这个一键保存。
            * **`:qa!`**：强制退出全部，不保存任何修改。

    4. 实战场景

        * 场景一：把 `vimdiff` 设置为 Git 的默认对比工具

            Git 自带的文本 diff 有时不够直观，你可以让 Git 在执行 `git difftool` 时调用 `vimdiff`：

            ```bash
            git config --global diff.tool vimdiff
            git config --global difftool.prompt false
            ```

            以后只需要输入 `git difftool`，就能用可视化分屏看你的代码改动了。

        * 场景二：解决 Git 冲突（Merge Conflict）

            当 Git 发生冲突时，输入 `git mergetool`，`vimdiff` 会展现出极其强悍的 **三路合并（3-Way Merge）** 界面：
            屏幕会被分成 4 个窗口：

            * **LOCAL**：你本地的分支改动（左）。
            * **BASE**：冲突前两者的共同祖先版本（中）。
            * **REMOTE**：远程别人提交的改动（右）。
            * **MERGE RESULT**：最下方的窗口，是你最终需要保存的合并结果。

            你只需要在最下方的窗口里用 `[c` 和 `]c` 找到冲突，然后用 `:diffget LOCAL` 或 `:diffget REMOTE` 来决定要留谁的代码。

    **总结命令卡片**

    | 快捷键 | 动作 | 含义 |
    | --- | --- | --- |
    | **`]c`** | 跳转 | 下一个差异点 |
    | **`[c`** | 跳转 | 上一个差异点 |
    | **`dp`** | 合并 | 将当前差异**推**到对面 |
    | **`do`** | 合并 | 将对面差异**拿**到当前 |
    | **`:wqa`** | 退出 | 保存所有文件并退出 |

* 详细讲讲 vim 脚本中`\<Esc>`，`Ctrl-V Esc`有啥区别

    在编写 Vim 脚本（Vimscript）或配置 `.vimrc` 时，处理键盘上的 **Esc 键** 经常会让初学者产生困惑。你提到的 `\<Esc>` 和 `Ctrl-V Esc`（在文本中通常显示为 `^[ `）在本质上是 **“文本表示法”与“原始控制字符”** 的区别。

    虽然它们最终都能让 Vim 触发 Esc 键的功能，但它们的输入方式、存储形态以及适用场景完全不同。

    1. `\<Esc>`：键盘映射的“文本转义法”

        `\<Esc>` 并不是一个真正的 Esc 字符，而是由 6 个普通的文本字符（反斜杠、小于号、E、s、c、大于号）组成的**字符串字面量**。

        **它是如何工作的？**

        Vim 在解析脚本时，如果发现某个地方支持“键码转义（Keycode translation）”，它就会把 `\<Esc>` 自动翻译成系统底层的 Esc 信号。

        **适用场景**

        它几乎只用于 **`map` 系列命令**（如 `nnoremap`、`inoremap`）或使用了**双引号**的字符串中（如 `:execute "normal! \<Esc>"`）。

        ```vim
        " 在 .vimrc 中最常见的用法：将 jk 映射为 Esc
        inoremap jk \<Esc>
        ```

        **优点**

        * **高可读性：** 任何人看你的配置文件，都能一眼看出这是 Esc 键。
        * **高可移植性：** 它是纯文本，你可以轻松地把它复制、粘贴到网页、GitHub、微信或者其他电脑上，绝对不会格式错乱。

    2. `Ctrl-V Esc`：底层的“原始控制字符”（`^[ `）

        如果在 Vim 的插入模式下，你先按下 `Ctrl-V`，接着按下 `Esc` 键，屏幕上会神奇地跳出一个 `^[ ` 字符。

        > **注意：** 这里的 `^[ ` 是一个**单个字符**（控制字符），而不是 `^` 和 `[` 两个字符。在 Vim 中把光标移上去，你会发现它是一个整体。它的 ASCII 码是 `27`（十六进制 `0x1b`），这正是计算机底层真正的 **Escape 键码**。

        **适用场景**

        它主要用于老旧版本的 Vim 脚本，或者直接在 **`let` 变量赋值** 以及**单引号字符串**中代表真正的 Esc 字符。因为单引号 `'` 无法解析 `\<Esc>` 这种特殊文本，必须塞入真正的控制字符。

        ```vim
        " 错误的写法（单引号不会转义，t 变量存进去的只是普通文本 \<Esc>）
        let t = '\<Esc>' 

        " 正确的写法（通过 Ctrl-V Esc 输入真正的控制字符）
        let t = '^[ ' 
        ```

        **缺点**

        * **极难维护和传输：** 它是不可见的控制字符。如果你把带有 `^[ ` 的代码复制粘贴到网页或者发给同事，它往往会丢失或者变成真正的 `^` 和 `[` 两个普通字符，导致配置直接失效。
        * **可读性差：** 新手很难分清这是真正的控制字符，还是手打的 `^` 和 `[`。

    3. 核心区别对比

        为了更直观，我们可以用一张表来对比它们：

        | 特性 | `\<Esc>` (文本转义) | `Ctrl-V Esc` (`^[ ` 原始字符) |
        | --- | --- | --- |
        | **本质是什么** | 由 6 个普通英文字符组成的**纯文本标签** | 计算机底层的**单个控制字符** (ASCII 27) |
        | **如何输入** | 直接在键盘上敲 `\`、`<`、`E`、`s`、`c`、`>` | 插入模式下顺序按下 `Ctrl-V` 然后按 `Esc` |
        | **可读性** | **极佳**。一眼就能看出是 Esc 键 | **差**。容易与普通文本 `^[` 混淆 |
        | **跨设备复制** | 完美保持原样 | 极易损坏，复制到网页或聊天工具后通常会失效 |
        | **在单引号内** | 无法被 Vim 解析（单引号内只认字面量） | 可以正常工作，因为它本身就是那个字符 |

    **总结与现代配置建议**

    > 💡 **现代金科玉律：永远优先使用 `\<Esc>`。**

    在现代 Vim / Neovim 编程中，应当**尽量避免**使用 `Ctrl-V Esc` 录制控制字符。

    如果你遇到了必须在变量中存储 Esc 字符的场景（比如 `let` 赋值），请不要去按 `Ctrl-V`，而是利用**双引号字符串**配合 `\e` 或 `\x1b`，或者借助 `expr` 映射，这样既能得到底层字符，又能保持代码是 100% 纯文本，方便在网络上分享：

    ```vim
    " 现代推荐写法 1：使用双引号和十六进制转义（保持纯文本）
    let esc_char = "\<Esc>"  " 或者 "\x1b"

    " 现代推荐写法 2：在写函数时避免硬编码，用命令代替
    execute "normal! \<Esc>"
    ```

* `let @s = '^[`<i**'`

    为什么我 virual 选中 void func_1() 并执行 @s 后，变成了这样：

    ```c++
    struct A
    {
    **void func_1()
        {
            return 0;
        }
    ```

    这是一个非常经典的 Vim 宏（Macro）行为与缩进机制冲突的问题。

    你的宏 `let @s = '^[`<i'`逻辑上是：退出视觉模式 -> 跳到刚才选区的开头 -> 插入`**`**。

    导致你看到“缩进消失”且结果不理想的原因主要有以下三点：

    **更好的解决方案**

    如果你想给选中的内容（如 `void func_1()`）前后加上 ``（类似 Markdown 的加粗），建议使用以下更“Vim 化”的方法：

    **方法 A：使用 `c` (修改) 配合寄存器（最稳妥）**

    这个方法不会触发复杂的缩进重算，因为它把整个内容当作一个替换块。

    ```vim
    " 逻辑：删除选区进入插入模式 -> 输入 ** -> 粘贴刚才删的内容 -> 输入 **
    let @s = 'c****^[hp'
    ```

    注：这里的 `^[ ` 是通过 `Ctrl-V` 然后 `Esc` 输入的，或者在 .vimrc 中写 `let @s = "c\<C-r>\""*`

    **方法 B：针对你原始宏的修正（防止缩进坍塌）**

    如果你坚持用插入模式，可以强制 Vim 不去重新调整缩进：

    ```vim
    " 在 i 之前加上 :set paste (防止缩进干扰) 之后恢复
    let @s = ":set paste\<CR>`<i**\<Esc>:set nopaste\<CR>"
    ```

    **方法 C：使用插件（终极方案）**

    如果你经常需要这种操作，强烈建议安装 **`vim-surround`** 插件。

    * **操作**：选中内容后，直接按 `S*`。
    * 它会自动处理所有的边界和缩进问题，非常优雅。

    **调试建议**

    你可以尝试在执行宏之前，先执行 `:set nocindent` 和 `:set nosmartindent`。如果此时执行 `@s` 缩进没有消失，那就确认是 Vim 的自动缩进引擎在插入 `` 时“自作聪明”地把你的代码往左推了。

* `execute lnum . 'normal! :s/<Tab>/ | /g'`

    分析一下这条 vimscript 是否有问题

    这条指令在语法逻辑上是**存在问题的**，直接运行通常会报错。主要的症结在于你混淆了 **Vimscript 表达式执行**（`execute`）和 **Vim 普通模式指令**（`normal!`）的边界。

    以下是详细分析及修正建议：

    1. 核心问题分析

        A. `normal!` 后面不能直接跟命令行模式命令

        `normal!` 的作用是模拟你在键盘上按下按键。`:s/...` 是一个命令行模式（Command-line mode）命令。

        如果你想在 `normal!` 里执行冒号命令，你必须显式地写入回车键（`<CR>`），否则 Vim 只是在底行输入了字符而没有回车。

        * **错误逻辑：** `normal! :s/...` (Vim 会在底行输入文字然后停在那里)

        * **修正逻辑：** `normal! :s/...^M` (其中 `^M` 是回车)

        B. 变量 `lnum` 的连接方式

        在 `execute` 中，如果你想在特定行执行替换，通常不需要通过 `normal!` 中转。`execute` 本身就可以拼接出完美的命令行指令。

    2. 修正方案

        根据你的意图（在 `lnum` 行将 `Tab` 替换为 `|`），这里有三种更好的写法：

        方案一：最推荐（直接使用 execute）

        这是最干净、最高效的写法，直接构造 `:行号s/源/宿/` 命令。

        ```vim
        execute lnum . 's/\t/ | /g'
        ```

        * **优点**：不需要进入 `normal` 模式，直接作用于目标行。
        * **注意**：在字符串里表示 Tab，用 `\t` 比直接写 `<Tab>` 更可靠。

        方案二：如果你坚持要用 normal!

        如果你是为了练习 `normal!` 的用法，必须加上 `\n` 或 `<CR>`，并处理好转义：

        ```vim
        execute lnum . "normal! :s/\\t/ | /g\<CR>"
        ```

        * **缺点**：非常繁琐，需要处理双重转义（`\\t`）。

        方案三：使用函数式写法（现代 Vim/NeoVim）

        如果你在写插件，使用 `setline()` 和 `substitute()` 会更安全，不会污染搜索历史：

        ```vim
        let line_content = getline(lnum)
        let new_content = substitute(line_content, '\t', ' | ', 'g')
        call setline(lnum, new_content)
        ```

    3. 避坑指南：关于 `<Tab>`

        在 Vimscript 字符串中，直接写 `"<Tab>"` 往往不会被识别为制表符。

        * 如果你想匹配 **Tab 键**：请使用 `\t`。
        * 如果你想匹配 **已经存在的字符串 "<Tab>"**：请使用 `\<Tab>`。

    **总结**

    你的原指令修改为：
    `execute lnum . 's/\t/ | /g'` 即可完美运行。

* 我在 ssh 里打开了远程 host，然后又在远程 host 中 vim 打开了一个文件，然后又 :vsplit xxx 打开了第二个文件，现在我想复制第二个文件的某一行，而不复制第一个文件的相同行，而且要把复制的文件 ctrl + v 粘贴到本地的 vscode 中，该怎么办？

    这个问题是很多 SSH 和 Vim 用户都会遇到的痛点。当你使用 `vsplit`（垂直分屏）时，终端（Terminal）本身并不知道 Vim 内部的分界线，所以直接用鼠标拖选会把左右两屏的内容都选进去。

    要解决你的需求，可以根据你的环境选择以下几种方案：

    1. 方案一：最快的手动方式（不需要任何插件）

        既然你的目标是复制第二屏的内容到本地，最简单的思路是**暂时让第二屏占据整个终端窗口**。

        1. 切换到第二个文件
        
            按下 `Ctrl + w` 然后按 `l` (或者右箭头)。

        2. 全屏当前文件
        
            输入 `:tabe %`。这会在 Vim 的一个新标签页中打开当前文件，此时侧边栏消失了。

        3. 复制内容

            * 如果你在本地使用 macOS，按住 `Option` 键再用鼠标选择（这可以忽略 Vim 的行号或特殊设置）。
            * 如果你在 Windows/Linux，按住 `Shift` 键再用鼠标选择。
            * 直接右键或使用终端的复制快捷键。

        4. 回到原状
        
            复制完后，在 Vim 中输入 `:q` 关闭这个临时标签页，你就回到了之前的双分屏界面。

    2. 方案二：利用 OSC 52 协议（最推荐的“极客”方案）

        如果你经常需要从 SSH 的 Vim 复制到本地剪贴板，**OSC 52** 是终极解决方案。它允许远程服务器通过 SSH 通道直接将文本发送到你本地终端的剪贴板。

        1. **安装插件**：在远程 Vim 中安装 [vim-oscyank](https://github.com/ojroques/vim-oscyank)。
        2. **使用**：在你想复制的那一行按下 `v` 进入可视模式选中，然后输入 `:OSCYank`。
        3. **结果**：文字会直接进入你本地电脑的剪贴板。你直接在本地 VS Code 里 `Ctrl + V` 即可。
        > **注意**：你需要使用支持 OSC 52 的终端（如 iTerm2, Alacritty, Windows Terminal, 或较新版本的 Chrome SSH）。

* 只要在 remote host 上安装了`apt install vim-gtk3`增强，就可以做到使用`"+y`把内容复制到 local host

    验证：输入 `vim --version | grep clipboard`，看到 `+clipboard` 表示安装成功。

    * 开启 SSH X11 转发 (存疑，不清楚是否为必做项)

        你连接 SSH 时不能只用 `ssh user@host`，必须加上 `-X` 或 `-Y` 参数：

        ```bash
        ssh -X user@host
        # 或者更宽松的授权
        ssh -Y user@host
        ```

    为什么我还是建议你考虑 OSC 52？

    如果你觉得装 XQuartz/VcXsrv 太麻烦，**OSC 52 协议**（方案二）其实更先进。它不需要 X11 转发，也不需要安装 `vim-gtk3`，哪怕是只有 `-clipboard` 的基础版 Vim，只要装个小脚本，就能通过终端的 ANSI 转义码直接把文字“塞”给本地剪贴板。

* 我是在 windows 中使用 cygwin 调用的 vim ~/.bashrc 命令

    在 Cygwin 环境中，由于它是一个模拟 Linux 的层，文件路径和权限的处理与原生 Linux 略有不同。出现 `E212: Can't open file for writing` 通常有以下几种可能：

    1. 路径斜杠问题

        虽然 Cygwin 能够识别部分 Windows 路径，但在调用 `vim` 时，务必使用 Linux 风格的**正斜杠** (`/`)。你输入的 `\home\wsdlh\.bashrc` 使用了 Windows 反斜杠，这在 Vim 中会被视为转义字符或无效路径。

        * 正确命令：

            ```bash
            vim ~/.bashrc
            ```

        或者使用完整路径：

        ```bash
        vim /home/wsdlh/.bashrc
        ```

    2. 文件权限受限

        如果路径正确但依然报错，可能是因为该文件的所有者不是当前用户，或者文件被设置为“只读”。

        * 尝试强制保存： 在 Vim 报错时，尝试输入 `:w!` 强制写入。

        * 检查权限： 在终端运行 `ls -l ~/.bashrc`。如果权限位没有 `w`（写权限），请运行：

            ```bash
            chmod +w ~/.bashrc
            ```

* 我使用的正是 vim ~/.bashrc，是不是因为 ~ 被 cygwin 或 vim 替换成反斜杠 \home\wsdlhl\ 了？

    很有可能。这通常是由于 Cygwin 的路径转换逻辑与 Vim 的内部解析发生了冲突。

    在 Cygwin 中，如果你在命令里直接使用 `~`，Shell（Bash）会先将其展开。如果展开后的路径被误处理为 Windows 风格的物理路径（带反斜杠），Vim 在尝试写入时就会报错，因为它把反斜杠 `\` 当作了普通字符（或者文件名的一部分），而 Windows 不允许在文件名中使用这些字符，或者 Vim 无法找到对应的目录。

    核心原因排查

    1. **路径展开错误**：执行 `echo ~/.bashrc` 看看输出。如果输出的是 `\home\wsdlh\.bashrc` 而不是 `/home/wsdlh/.bashrc`，说明你的 `HOME` 环境变量可能在 Windows 系统层级被设置成了 Windows 格式。

    2. **Vim 的目录不存在**：Vim 在保存文件时会先检查目录。如果路径中包含反斜杠，Vim 会认为你要在一个名为 `\home\wsdlh\` 的**文件夹**下创建文件，而这个文件夹在 Linux 逻辑路径下是不存在的。

    解决方案

    方法二：检查并修正 HOME 变量

    在 Cygwin 终端输入：

    ```bash
    export HOME=/home/wsdlh
    vim ~/.bashrc
    ```

    如果这次成功了，说明你需要检查 Windows 的环境变量设置，确保 `HOME` 没有被硬编码为 Windows 格式。

    ```bash
    wsdlh@DESKTOP-ETVUKF3 ~
    $ echo ~/.bashrc
    /home/wsdlh/.bashrc

    wsdlh@DESKTOP-ETVUKF3 ~
    $ echo $HOME
    /home/wsdlh
    ```

    从你的 `echo` 输出结果来看，Cygwin 内部的环境变量和路径展开是非常健康的（都是标准的 Linux 正斜杠 `/`）。

    这种情况下，Vim 报错 `\home\wsdlh\.bashrc`（带反斜杠）通常是因为 **Vim 自身被配置成了 Windows 模式**，或者它在尝试通过 Windows API 写入文件时，路径转换出现了偏差。

    请尝试以下三个具体的操作来解决：

    1. 检查 Vim 的 `shellslash` 设置

        Vim 有一个选项可以控制它在 Windows 环境下使用哪种斜杠。在 Vim 报错的界面，直接输入以下命令并回车：

        ```vim
        :set shellslash?

        ```

        如果显示 `noshellslash`，请手动修复它：

        ```vim
        :set shellslash
        :w

        ```

        这会强制 Vim 在处理路径时使用 `/`，通常能解决 E212 错误。

    3. 检查是否有 `.swp` 文件残留

        如果之前的编辑异常中断，可能会留下一个交换文件，导致写入失败。尝试清理：

        ```bash
        rm -f ~/.bashrc.swp
        ```

    4. 终极尝试：强制指定路径写入

        如果 `vim ~/.bashrc` 进去后无法保存，在冒号命令行输入：

        ```vim
        :w! /home/wsdlh/.bashrc
        ```

        注意这里的 `/` 是关键。如果这仍然失败，请尝试 `sudo vim ~/.bashrc`（如果你安装了 cygutils-extra 包提供的 sudo 模拟）。

    为了彻底解决路径和权限冲突，我们可以进一步优化你的 Vim 启动配置。

    * 自动开启 shellslash 以支持正斜杠路径
    * 修复 Cygwin 目录的 Windows 访问权限
    * 检测并清除 Vim 残留的交换文件

* shellslash 是启动 vim 后才生效的吧，我这个显然是还未启动 vim 时，传给 vim 的路径有问题

    你的直觉非常准确。如果是在执行 `vim ~/.bashrc` 的那一刻就报错，说明 Bash 在传递参数给 Vim 之前，路径解析就已经出现了偏差。

    既然 `echo $HOME` 显示的是正斜杠，但报错信息里出现了反斜杠，这通常是因为你使用的 `vim` 程序可能是一个 **Windows 原生版本**（例如安装 Git for Windows 时带的 Vim），而不是 **Cygwin 编译版本**。Windows 原生程序无法直接识别 Cygwin 的虚拟路径（如 `/home/...`），会尝试将其强制转换为 Windows 格式。

    核心排查与解决方法

    1. 确认 Vim 的身份

        在终端输入以下命令：

        ```bash
        which vim
        ```

        * 如果输出 `/usr/bin/vim`，说明是 Cygwin 版本（理论上不应出现反斜杠问题）。

        * 如果输出类似 `/cygdrive/c/Program Files/Git/usr/bin/vim` 或其他 Windows 路径，说明你调用的是外部 Vim。

    2. 使用 cygpath 进行实时转换

        如果由于环境限制必须使用该 Vim，可以通过 `cygpath` 工具将 Linux 路径转换为 Windows 识别的物理路径：

        ```bash
        vim $(cygpath -m ~/.bashrc)
        ```

        `-m` 参数会强制输出带正斜杠的 Windows 路径（如 `C:/cygwin64/home/...`），这对大多数 Vim 版本都有效。*

    4. 强制修复 HOME 变量映射

        如果以上都不行，尝试在执行时临时覆盖 HOME 变量的表达方式：

        ```bash
        HOME=/home/wsdlh vim ~/.bashrc
        ```

* vim ctrl + d 向下翻页的同时，会移动光标。我想只翻页，不移动光标，怎么办

    在 Vim 中，默认的 `Ctrl + d`（向下半屏）和 `Ctrl + u`（向上半屏）确实会随着屏幕滚动而移动光标。

    如果你希望**只滚动屏幕内容，而让光标保持在当前的行号（物理位置）不动**，Vim 内置的滚动指令其实是 `Ctrl + e` 和 `Ctrl + y`。

    1. 使用内置指令（逐行滚动）

        * **`Ctrl + e`**：向下滚动一行（光标留在原行，除非该行滚出屏幕）。
        * **`Ctrl + y`**：向上滚动一行。

        如果你觉得逐行太慢，可以配合数字使用，例如 `10 Ctrl + e` 滚动 10 行。

    2. 映射 `Ctrl + d/u` 实现“视觉不动”

        如果你习惯了使用 `Ctrl + d` 的翻页幅度，但希望光标在**视觉上**看起来停留在屏幕的固定位置，可以通过 `~/.vimrc` 修改映射。

        将以下代码加入你的配置文件：

        ```vim
        " 保持光标在屏幕的相对位置不变进行翻页
        noremap <C-d> <C-d>zz
        noremap <C-u> <C-u>zz
        ```

        * **原理**：在执行翻页后，自动触发 `zz` 命令，将光标所在的行重置到屏幕正中央。这样光标在屏幕上的垂直高度就保持不变了。

    3. 实现真正的“光标不动” (高级技巧)

        如果你追求的是像现代浏览器那样，按下翻页键时，光标完全留在原有的行号上（即使翻页后该行不可见），Vim 原生并不推荐这种操作（因为 Vim 的设计逻辑是光标必须始终在屏幕可视区域内）。

        但你可以通过以下映射来模拟这种“窗口滚动”：

        ```vim
        " 向下滚动半屏，光标留在原处
        nnoremap <C-d> Lz<CR>
        " 向上滚动半屏，光标留在原处
        nnoremap <C-u> Hz<CR>
        ```

    **总结建议**

    * 如果你只想**微调视野**：请改用 `Ctrl + e`。
    * 如果你想**大范围翻页但不想找光标**：请使用 `Ctrl + d` 配合 `zz` 映射。

    需要我帮你把这些配置直接写进一个可以运行的 `.vimrc` 片段吗？

* 如果调用函数呢？在函数中，首先判断用户选了哪几行，然后 for 循环一行一行处理，如果这一行只有空白符，比如 space, tab, newline，那么跳过，否则则找到第一个非空白符，并在之前加上*空格

    通过编写 Vim 函数并结合 `range` 关键字，可以非常精确地控制每一行的处理逻辑。

    这种方法的优势在于：逻辑清晰、易于扩展（比如后续想改用 `-` 或 `1.`），并且能完美处理缩进。

    1. Vim 函数实现

        你可以将以下代码直接复制到你的 `~/.vimrc` 文件中：

        ```vim
        function! AddBulletPoints() range
            " a:firstline 和 a:lastline 是由 range 自动传入的选区起止行号
            for lnum in range(a:firstline, a:lastline)
                let line = getline(lnum)

                " 检查行是否包含非空白字符 (\S 代表非空白)
                if line =~ '\S'
                    " 找到第一个非空白字符的位置，并执行替换
                    " ^\s* 匹配行首空格，\zs 标记匹配开始位置
                    " 将第一个非空白字符 替换为 '* ' + 该字符
                    let new_line = substitute(line, '^\s*\zs\S', '* &', '')
                    call setline(lnum, new_line)
                endif
            endfor
        endfunction

        " 为 Visual 模式设置快捷键，比如按 \b (bullet) 执行
        vnoremap <leader>b :call AddBulletPoints()<CR>
        ```

    2. 代码逻辑深度解析

        * **`range` 关键字**：在定义函数时加上 `range`，Vim 会自动把 Visual 模式选中的起始行号赋给 `a:firstline`，结束行号赋给 `a:lastline`。
        * **`line =~ '\S'`**：这是判断“是否有文字”的关键。如果一行全是空格、Tab 或完全为空，它就不会匹配成功，从而实现了“跳过空行”。
        * **`substitute(..., '^\s*\zs\S', '* &', '')`**：
        * `^\s*`：匹配行首开始的所有空格。
        * `\zs`：这是 Vim 正则的“功臣”，它告诉 Vim：“虽然我前面匹配了空格，但请把匹配的‘起始点’设在这里”。
        * `\S`：匹配第一个看到的文字。
        * `* &`：`&` 代表刚才匹配到的那个文字。这样就实现了在文字前插入 `* ` 且**保留了原有缩进**。

    3. 如何使用

        1. 打开你的 `~/.vimrc`。
        2. 粘贴上面的代码并保存。
        3. 重新打开 Vim 或执行 `:source $MYVIMRC`。
        4. 进入 Visual 模式选中你想处理的行。
        5. 按下你设置的快捷键（如 `\b`，假设你的 leader 键是反斜杠）。

    **进阶：如果你希望不保留缩进，统一靠左**

    如果你希望不管原来的行缩进多少，统统在最左边加上 `* `，只需将 `substitute` 那行改为：

    ```vim
    let new_line = '* ' . trim(line)
    ```

    这样它会删掉行首行尾所有空格，然后直接在最前面加上 `* `。

    **您是否需要我为您演示如何将这个函数扩展，使其能在 `*`、`-` 或 `1.` 之间循环切换？**

* vim command 模式中如果执行多个函数，需要使用`\|`将它们串联起来：

    `:call func_1() \| call func_2()`

    否则就会报错如下：

    > E488: Trailing characters: :call setpos("'k", getpos("']"))

    不可以直接使用`|`，因为 Vim 可能会误认为你想在定义映射的同时执行第二个命令，而不是把第二个命令作为映射的一部分。

    如果连`|`也不写，那么 Vim 会将第二个冒号及其后的内容视为第一个 `call` 函数的参数，从而导致语法解析失败，报错“多余字符”。

    推荐写法：使用 `<Cmd>` 标签

    如果你使用的是 Neovim 或较新版本的 Vim（8.2+），使用 `<Cmd>` 代替 `:` 是更现代、更稳健的做法。它不需要 `<CR>` 结尾，且不会触发行列的更新或不必要的模式切换，通常能完美避开 `E488` 错误：

    ```vim
    nnoremap P "+p<Cmd>call setpos("'j", getpos("'[")) \| call setpos("'k", getpos("']"))<CR>
    ```

    * 关于`<silent>`
    
        如果执行完后，显示

        ```vim
        :call setpos("'j", getpos("'[")) | :call setpos("'k", getpos("']"))
        Press ENTER or type command to continue
        ```

        如何让它不显示`Press ENTER or type command to continue` ?

        出现 `Press ENTER or type command to continue` 的原因是 Vim 觉得输出的指令太长，占用了底部的状态栏空间，担心你没看清，所以强制停下来让你确认。

        要解决这个问题，有三种常用手段：

        1. 使用 `<silent>`

            在定义映射时加上 `<silent>` 参数，Vim 就会在后台静默执行命令，不会将指令回显在命令行。

            修改后的配置：

            ```vim
            nnoremap <silent> P "+p:call setpos("'j", getpos("'[")) \| call setpos("'k", getpos("']"))<CR>
            ```

        2. 使用 `<Cmd>`

            `<Cmd>` 专门用于在不切换模式、不显示命令行、不触发 UI 更新的情况下执行函数调用。它天生就是静默的，且不需要在末尾加 `<CR>`。

            ```vim
            nnoremap P "+p<Cmd>call setpos("'j", getpos("'[")) \| call setpos("'k", getpos("']"))<CR>
            ```

            *注：这里的 `<CR>` 是为了触发前面的 `+P` 动作，而 `<Cmd>...` 部分不需要内部再回车。*

            注：

            1. 存疑。即使是`"+p`动作，也不需要`<cr>`触发呀。

        3. 终极手段：通过脚本控制消息（如果还弹窗）

            如果是因为 `P` 操作本身产生了一些反馈（比如“5 lines pasted”）导致弹窗，可以临时调整 `shortmess`。

    注：

    1. 目前正在使用的映射：

        ```vim
        nnoremap <silent> P "+p:call setpos("'j", getpos("'[")) \| call setpos("'k", getpos("']"))<cr>
        ```

        在管道符 `|` 后面连接命令时，通常**不需要**再加冒号。

        * 错误示例： `:cmd1 | :cmd2`

        * 正确示例： `:cmd1 | cmd2`

* 详细讲讲 vim 中 :find 的用法

    在 Vim 的世界里，`:find` 是一个被严重低估的“扫地僧”级命令。很多新手习惯用 `:edit` (或 `:e`)，但 `:edit` 要求你提供精确的文件路径。而 `:find` 的核心逻辑是：**“我只给你个文件名，你自己去我的项目路径里把它搜出来。”**

    要玩转 `:find`，你必须理解它背后的“地图”——`path` 变量。

    ---

    1. 核心依赖：`path` 变量

        `:find` 能否找到文件，完全取决于 Vim 的 `path` 选项。你可以把它理解为环境变量中的 `PATH`，Vim 会按顺序在这些目录中查找。

        *   **查看当前的 path：**
            ` :set path? `
            （默认通常只包含 `.,/usr/include,,`，这意味着它只看当前文件目录和系统头文件目录。）

        *   **配置 path 以支持递归搜索：**
            这是 `:find` 变得强大的关键一步。在你的 `.vimrc` 中加入：
            ` set path+=** `
            `**` 表示递归搜索当前工作目录下的**所有子目录**。

    ---

    2. 基础用法

        当你配置好 `path` 后，`:find` 就能大显身手了：

        *   **直接查找：**
            ` :find main.c `
            Vim 会在 `path` 定义的所有目录中搜索 `main.c`，找到后直接打开。
        *   **支持 Tab 补全：**
            输入 `:find ma` 然后按 `<Tab>`，Vim 会在 `path` 中匹配所有以 `ma` 开头的文件名。
        *   **模糊匹配（某种程度上）：**
            虽然它不是真正的模糊搜索，但你可以配合通配符：
            ` :find *user*controller.py `

    ---

    3. 进阶变体：分屏与标签页

        如果你不想替换当前缓冲区，可以使用以下变体：

        | 命令 | 效果 |
        | :--- | :--- |
        | `:sfind <file>` | 在**水平分屏**中打开找到的文件 (Split find) |
        | `:tabfind <file>` | 在**新标签页**中打开找到的文件 |
        | `:vert sfind <file>` | 在**垂直分屏**中打开找到的文件 |

    ---

    4. 强力辅助：`suffixesadd`

        有时候你连后缀名都不想输入，比如在写 Java 或 Python 时。你可以通过 `suffixesadd` 让 `:find` 自动尝试补全后缀。

        *   **设置方法：**
            `set suffixesadd+=.java,.py,.c`
        *   **效果：**
            现在你只需输入 `:find Main`，Vim 会自动寻找 `Main.java`、`Main.py` 或 `Main.c`。

    ---

    5. 为什么用 `:find` 而不是插件？

        现在流行 `fzf.vim` 或 `Telescope`，为什么还要学内置的 `:find`？

        1.  **原生无依赖：** 在没有权限安装插件的服务器上，它是最强的文件跳转工具。
        2.  **配合 `gf` 命令：** 当你把光标移到代码中的文件名（如 `import "utils"`）上按下 `gf` (Go to File) 时，Vim 实际上也是通过 `path` 变量去寻找这个文件的。配置好了 `:find`，也就增强了 `gf`。
        3.  **速度：** 对于中小型项目，内置的路径查找速度极快，不需要后台索引。

    ---

    6. 避坑指南

        *   **重名文件：** 如果 `path` 中有多个同名文件，`:find` 默认打开第一个找到的。你可以按 `<Tab>` 循环选择。
        *   **性能隐患：** 如果你在根目录 (`/`) 或极大的项目目录下设置 `set path+=**`，搜索可能会变慢。建议只将 `path` 设置在具体的项目根目录下。

    **总结：**

    `:find` 的核心在于 `set path+=**`。只要这一行配置好，你就能从冗长的 `cd` 和 `../` 路径地狱中解脱出来。

* vim 中換行符相关的报错

    ```
    E492: Not an editor command: ^M
    line  127:
    Press ENTER or type command to continue
    ```

    这个错误提示说明你的 `.vimrc`（或相关的配置文件）中包含了 **Windows 风格的换行符（CRLF）** (`\r\n`)，而 Cygwin 的 Vim 期望的是 **Unix 风格的换行符（LF）** (`\n`)。

    * `^M` 其实就是多出来的 `\r`（Carriage Return）。

    * Vim 在读取配置文件时，把 `\r` 误认为是命令的一部分。例如，它本该执行 `set number`，结果却在尝试执行 `set number\r`，导致报错 "Not an editor command"。

    * 方法 1：在 Vim 内部直接修复

        1. 忽略报错进入 Vim, 多次按回车直到进入编辑器。

        2. 修改文件格式

            ```vim
            :set ff=unix
            :wq
            ```

            这会将当前文件的换行符统一转换为 Unix 格式并保存。

    * 方法 2：使用 `dos2unix` 工具

        如果你安装了 `dos2unix` 软件包，可以在 Cygwin 终端执行：

        ```bash
        dos2unix ~/.vimrc
        ```

    * 方法 3：使用 `sed` 命令

        如果没有安装工具，可以直接用 `sed` 删掉回车符：

        ```bash
        sed -i 's/\r//g' ~/.vimrc
        ```

* `nnoremap <tab> >>`

    `vnoremap <tab> >>`

    为什么我使用 <tab> 在 visual 模式下缩进几行文本后，按下 . 没有任何效果，第二次按下 . 才有缩进效果？如果你不知道原因，我该如何调试，从而知道第一次按下 . 后发生了什么？

    这是一个非常经典的 Vim 映射陷阱。你遇到的现象（第一次 `.` 没反应，第二次才有效）通常是因为你的 **Visual 模式映射触发了“操作符挂起”状态**，导致 Vim 的命令序列变得混乱。

    **为什么会发生这种情况？**

    问题的核心在于你的 `vnoremap <tab> >>`。

    1.  **模式切换的副作用**：在 Visual 模式下，单字符 `>` 的功能是“缩进选中的行”并**立刻返回 Normal 模式**。
    2.  **多余的字符**：当你映射 `<tab>` 为 `>>` 时：
          * 你按下 `<tab>`。
          * Vim 执行第一个 `>`：选中的文本缩进，**进入 Normal 模式**。
          * Vim 执行第二个 `>`：此时你已经在 Normal 模式了，`>` 在 Normal 模式下是一个**操作符（Operator）**，它在等待一个位移动作（Motion），比如 `>>` 或 `>w`。
    3.  **悬挂状态**：因为你的映射只给了两个 `>`，而在 Normal 模式下触发缩进通常需要 `>>`（即 `>` 后面再跟一个 `>`）。当你映射完并执行后，Vim 实际上可能正处于一个“等待输入位移动作”的半完成状态，或者是 `.` 记录的最后一个动作并非你想象中的整体。

    **结果**：第一次按下 `.` 时，它可能在尝试补全那个“悬挂”的命令或者重复了一个空的上下文。

    -----

    **如何修复？**

    在 Visual 模式下，你只需要一个 `>` 即可。如果你希望缩进后继续保持选中状态（这是大多数人的习惯），建议这样改：

    ```vim
    " Normal 模式：缩进当前行
    nnoremap <tab> >>
    " Visual 模式：缩进选中部分，并用 gv 重新选中刚才的区域
    vnoremap <tab> >gv
    ```

    如果你不希望保持选中，只需 `vnoremap <tab> >`。

    -----

    **如何调试这类问题？**

    如果你想亲眼看看第一次按下 `.` 后到底发生了什么，可以使用以下调试手段：

    1. 查看消息历史 (`:messages`)

        有时候 Vim 会在底部显示极其短暂的错误或提示。

        * 操作完 `<tab>` 和 `.` 后，输入 `:messages` 查看是否有 `Type  :help  indicating...` 之类的提示。

    2. 使用 `showcmd` 观察

        在你的 `vimrc` 中设置：

        ```vim
        set showcmd
        ```

        这时在 Vim 右下角会显示你输入的未完成命令。当你按下 `<tab>` 后，观察那里是否留下了一个孤零零的 `>`，如果是，说明你的映射导致 Vim 在等待下一个动作。

    3. 记录日志 (Typeahead Debugging)

        这是终极大法，可以看到 Vim 接收到的每一个原始按键：

        1.  启动 Vim 并开始记录日志：`vim -V9vim.log`
        2.  执行你的操作：进入 Visual 模式 -\> 按 `<tab>` -\> 按 `.`。
        3.  退出 Vim，打开 `vim.log` 搜索 `.`。
            你会看到 Vim 是如何解析你的映射序列的。

    4. 检查是否被其他插件接管 (`:verbose nmap .`)

        有些插件（如 `vim-repeat`）会重新映射 `.` 键。

        输入 `:verbose nmap .` 看看 `.` 是不是被谁偷偷改了。

    **建议下一步：**
    尝试将 `vnoremap <tab> >>` 修改为 `vnoremap <tab> >`，看看 `.` 是不是立刻就恢复正常了？

* 有关复制命令的几种用法

    ```vim
    yy          # 复制单行
    5yy         # 复制5行
    :10,20y     # 复制第10-20行
    ```

* vim 中，如何 visual 模式下只选取指定的行，比如隔一行选一行。我想选择指定的不规律的几行，比如 1, 4, 9, 12，然后对这些行应用一些指定的宏或函数，比如在每一行的开头添加 * ，删去结尾的空格，替换行里的一些字符等。

    感觉目前没有很好的办法，ai 推荐使用插件：

    * `vim-visual-multi`（类 VSCode 多光标）
    * `vim-multiple-cursors`（老牌）

    个人感觉可以写个 vim 脚本，按某个键（比如`\l`）进入选中模式，然后按`a`选中行，每选中一行把行号加到 vim 脚本的 list 里，最后按`\l`退出选择。在 apply 宏或函数时，可以直接使用这个 list 里的数据。另外增加 reset list 功能: `\lr`，查看已标记行号的功能：`\ll`。

* vim 中，我将 m 键映射为 2j，现在我输入 3m，想向下移动 6 行，但是实际移动的是 32j，即 32 行。该如何解决这个问题

    在 Vim 中，当你使用数字前缀（如 3）配合映射时，Vim 会将数字和映射的键组合起来执行。你的问题是 3m 被解释为 32j（因为 m 映射为 2j），所以实际执行的是 32j 而不是 3×2j。

    有几种解决方案：

    * 解决方案 1：使用表达式映射（Expression Mapping）（推荐）

        ```vim
        " 在 vimrc 中添加
        nnoremap <expr> m (v:count ? v:count * 2 : 2) . 'j'
        ```

        这个映射会完成`3x2j`。

        * `<expr>`: 它告诉 Vim：后面的字符串不是要执行的命令，而是一个需要计算的表达式。Vim 会先计算这个表达式的结果，然后再把结果当作命令执行。

            如果你在映射中去掉了 <expr>，Vim 就不会再把后面的代码当作“逻辑表达式”去计算，而是会把它当作一串普通的按键序列（Literal keys）直接输入。

        * `(v:count ? v:count * 2 : 2) . 'j'` 是一个典型的三元运算符逻辑：

            * 变量 `v:count`

                这是 Vim 的内置变量，用于存储你在按键前输入的数字前缀（Count）。

                * 如果你输入 5m，那么 v:count 就是 5。

                * 如果你直接按 m，那么 v:count 默认为 0。

            * 三元运算符 (条件 ? 结果A : 结果B)

                条件: v:count（在 VimL 中，0 为假，非 0 为真）。

                如果为真（输入了数字）: 执行 v:count * 2。

                如果为假（没输数字）: 执行 2。

            * 字符串连接符 `.`

                `. 'j'` 表示将前面的计算结果与字符 j（向下移动一行的命令）拼接在一起。

        注：

        1. 这个也可以等价映射成`nnoremap m <Cmd>execute 'normal! ' . (v:count1 * 2) . 'j'<CR>`

            为什么 <expr> 通常更好？
            
            如果你删掉 <expr> 改成 execute，主要的变化在于命令的性质：
            
            | 特性 | `<expr>` 映射 | `execute 'normal! ...'` |
            | - | - | - |
            | 本质 | 按键替换（m -> 4j） | 脚本执行（运行一段代码） |
            | 简洁性 | 高（直接返回字符串） | 低（需要写 execute, normal!, <CR>） |
            | 宏录制 | 录制宏时表现更直观 | 有时会录入完整的命令字符串 |
            | 处理 Count | 非常自然 | 需要依赖 v:count1 或手动解析 |

    * 解决方案 2：使用 `<Cmd>` 映射（Vim 8.2+/Neovim）

        ```vim
        nnoremap m <Cmd>execute 'normal! ' . (v:count1 * 2) . 'j'<CR>
        ```

        * `<Cmd> ... <CR>`

            这是现代 Vim/Neovim 推荐的执行命令方式。

            相比传统的 : 模式，`<Cmd>` 不会切换到命令行模式，也不会闪烁，执行效率更高且更安静。

            `<CR>` 代表按下回车，完成命令执行。

        * execute

            这是一个 Vim 脚本命令，用于将后面的字符串当作命令来执行。

            因为我们需要动态计算跳转的行数，所以必须用 execute 来拼接字符串。

        * `v:count1`: 这是一个内置变量。如果你在按 m 前输入了一个数字（如 5），它就是这个数字；如果你没输入数字，它的值默认为 1。

        * `.`: 在 Vim 脚本中用于连接字符串。

        * `'normal! ... j'`: 模拟按下原生的 j（向下移动）。! 确保使用的是 Vim 原生功能，不受其他插件映射的影响。

    * 解决方案 3：使用函数包装

        ```vim
        function! MoveLines() abort
            let count = v:count1 * 2
            execute 'normal! ' . count . 'j'
        endfunction

        nnoremap <silent> m :call MoveLines()<CR>
        ```

    * 最完整的解决方案

        如果你希望处理所有数字前缀（包括 0 前缀）：

        ```vim
        nnoremap <expr> m 'normal! ' . (max([v:count, 1]) * 2) . 'j'
        ```

        或者更好的版本：

        ```vim
        nnoremap <expr> m (v:count ? v:count : 1) * 2 . 'j'
        ```

    验证映射

    添加映射后，可以这样测试：

    * 保存 vimrc 文件

    * 重新打开 Vim 或执行 :source ~/.vimrc

    * 测试：

        * 按 m 应该向下移动 2 行

        * 按 3m 应该向下移动 6 行

        * 按 5m 应该向下移动 10 行

    其他考虑

    如果你还想保留 m 的原始功能（设置标记），建议使用其他按键，比如：

    ```vim
    " 使用其他不常用的键
    nnoremap <expr> <Leader>m (v:count ? v:count : 1) * 2 . 'j'
    ```

    推荐使用 解决方案 1（表达式映射），因为它最简洁且能正确处理所有情况。

* Vim 中没有直接的内置选项可以让光标在滚动时自动跟随屏幕移动

    一些相关的选项：

    * `scrolloff` - 设置在光标上下保留的最小行数

        ```vim
        set scrolloff=5  " 光标上下总是保留5行可见
        ```

    * `sidescrolloff` - 水平方向的类似选项

    * `startofline` - 某些命令是否会移动光标到行首

* 使用 ssh 登陆 remote host 后，启动 vim 打开一个文件，此时如果 remote host 没有安装`vim-gtk3`，那么使用`"+y`是无法将内容复制到 local host 的剪贴板上的。

    安装了就可以了。

    local host 应该也得安装`vim-gtk3`。

    ssh 连接时是否需要开 -X 或 -Y，不清楚。

* 大写字母`U`

    * 正常模式（Normal Mode）:

        撤销对当前行的所有修改。会一次性撤销对当前行的多次修改。

    * 可视模式（visual mode）:

        选中的文本全部转换为大写字母。

        在可视模式（Visual Mode）下选中多行或多段文本后，按 U，选中的部分会全部转为大写。

        如果想将选中的文本转为小写，则使用 gu（例如在可视模式下选中后按 gu）。

* vim 重选刚才的行

    在 Vim 中，有几种方法可以重新选中刚才选择的多行：

    方法1： gv

    这个命令会重新选择上次的 visual selection 区域。

    方法2：使用标记

    * 在 visual 模式下选择时，可以先按 m 然后设置一个标记：

        ```vim
        v  # 进入visual模式
        选中多行
        m a  # 设置标记为 a（可以是任意字母）
        ```

        稍后要重新选择时：

        ```vim
        ' a  # 跳转到标记a所在行
        V    # 进入visual line模式，选择当前行
        ` a  # 跳转到标记a的确切位置，完成选择
        ```

    方法3：使用 '> 和 '<

    Vim 会自动记录上次 visual selection 的起始位置：

    * `'<`: 跳转到上次 visual selection 的起始行

    * `'>`: 同上，跳转到结束行

    要重新选择：

    ```vim
    '<V'>  " 从起始行 visual line 模式选择到结束行
    ```

    方法4：配合移动命令

    如果你知道大概位置：

    ```vim
    :5,10V  # 选择第5-10行
    ```

    或者在normal模式下：

    ```vim
    5GV10G  # 从第5行visual选择到第10行
    ```

    推荐使用 gv，它是最简单快捷的方法，会记住你上次选择的模式和范围。

* `/w<CR>i<space><Esc>` 的作用解析

    这是一个复合命令序列，分解如下：

    * `/w` - 进入搜索模式，搜索字母 "w"

    * `<CR>` (回车) - 执行搜索，光标跳转到第一个 "w"

    * `i` - 进入插入模式（在光标前插入）

    * `<space>` - 插入一个空格字符

    * `<Esc>` - 退出插入模式，返回普通模式

    整体作用：

    搜索文件中第一个出现的 "w"，然后在该字符前插入一个空格

    补充说明：

    * 如果要插入空格到每个 "w" 前，可以用：`:%s/w/ w/g`

    * 可以在命令前加数字前缀：`3/w<CR>i<space><Esc>` 会执行三次

* vim `^=`

    在 Vim 命令 `set directory^=$HOME/.vim/swap//` 中，^= 是一个特殊的 前置追加 操作符。

    具体含义：

    * ^= 表示将指定的值添加到选项列表的开头（前置）

    * 这会将 $HOME/.vim/swap// 目录放在 Vim 交换文件搜索路径的最前面

    对比其他操作符：
    操作符	含义	示例
    =	直接设置	set directory=/path
    +=	追加到末尾	set directory+=/path
    ^=	前置到开头	set directory^=/path
    -=	从列表中移除	set directory-=/path

    实际效果：

    ```vim
    " 原始可能有默认值：directory=.,~/tmp,/var/tmp,/tmp

    set directory^=$HOME/.vim/swap//

    " 执行后变成：directory=$HOME/.vim/swap//,.,~/tmp,/var/tmp,/tmp
    ```

* Vim 搜索与替换命令

    1. 搜索

        * `/pattern` - 向下搜索

        * `?pattern` - 向上搜索

        * `n` - 跳转到下一个匹配

        * `N` - 跳转到上一个匹配

        * `*` - 搜索光标下的单词（向前）

        * `#` - 搜索光标下的单词（向后）

    2. 替换

        * `:s/old/new` - 替换当前行第一个匹配

        * `:s/old/new/g` - 替换当前行所有匹配

        * `:%s/old/new/g` - 替换整个文件中匹配的字符

        * `:%s/old/new/gc` - 替换整个文件并确认每个替换

        * `:range s/old/new/g` - 在指定范围替换

* 在 Vim 的命令行中，! 用于执行外部 shell 命令：

    ```vim
    :!ls          " 执行 ls 命令
    :!python3 script.py  " 执行 Python 脚本
    :w !sudo tee %  " 常用技巧：用 sudo 保存文件
    ```

* 在替换命令中，! 表示忽略大小写：

    ```vim
    :s/foo/bar/     " 将 foo 替换为 bar（区分大小写）
    :s/foo/bar/i    " i 表示忽略大小写
    :s/foo/bar/gi   " g 全局，i 忽略大小写
    ```

    实际上，i 标志更常用，但 ! 在 Vim 的正则表达式中有时也用于此目的。

* 在自动命令中：autocmd! 的 ! 表示清除之前的所有同类自动命令：

    ```vim
    " 清除所有 FileType 自动命令，避免重复定义
    autocmd! FileType python
    autocmd FileType python setlocal shiftwidth=4
    ```

* 在设置选项中：set !option 的 ! 表示切换/取反：

    ```vim
    :set number     " 显示行号
    :set nonumber   " 不显示行号
    :set number!    " 切换行号显示状态（显示←→不显示）
    :set invnumber  " 同上，inv = invert
    ```

* 在缓冲区命令中：b! 的 !

    ```vim
    :b 2     " 切换到缓冲区 2（如果有未保存修改会失败）
    :b! 2    " 强制切换到缓冲区 2（丢弃当前缓冲区未保存修改）
    ```

* 在模式切换中：! 用于临时进入命令模式

    在普通模式下，Q 进入 Ex 模式，而 gQ 则不同，但 ! 可以配合：

    ```vim
    :!ls      " 普通模式按 :! 然后输入命令
    ```

* vim 快速添加空格

    ```vim
    " 在普通模式下按空格键在光标后插入空格
    nnoremap <space> i<space><esc>l

    " 或者在光标前插入空格
    nnoremap <leader><space> i<space><esc>

    " 更智能的版本：在单词间插入空格（光标在 w 位置时）
    nnoremap <space> i<space><esc>w
    ```

    其他可能有用的映射：

    ```vim
    " 在单词边界智能插入空格
    nnoremap <silent> <space> :call InsertSpace()<CR>

    function! InsertSpace()
        " 如果光标在单词开头，在前面插入空格
        if col('.') == 1 || getline('.')[col('.')-2] =~# '\s'
            execute "normal! i \<esc>"
        else
            execute "normal! a \<esc>"
        endif
    endfunction
    ```

* vim 中常用的寄存器

    `"+` 寄存器：对应系统的 “Ctrl+C / Ctrl+V” 剪贴板。在大多数现代系统上，这是最常用的。

    `"*` 寄存器：在 Linux/Unix 系统上，通常对应 “鼠标中键” 或“选择”剪贴板（即你用鼠标选中文本，然后按鼠标中键粘贴的内容）。在 Windows/macOS 上，它和 "+ 通常是相同的。

    | 命令 | 描述 |
    | - | - |
    | `"+y` | 复制当前选中的文本到系统剪贴板 |
    | `"+yy` 或 `"+Y` | 复制当前行到系统剪贴板 |
    | `"+yiw` | 复制当前光标下的单词到系统剪贴板 |
    | `"+y$` | 从光标处复制到行尾到系统剪贴板 |
    | `"+d` | 剪切当前选中的文本到系统剪贴板 |
    | `"+dd` | 剪切当前行到系统剪贴板 |
    | `"+d$` | 从光标处剪切到行尾到系统剪贴板 |

    可视化模式下的操作：

    1. 按 v (字符可视模式) 或 V (行可视模式) 或 Ctrl+v (块可视模式)。

    2. 选中你要操作的文本。

    3. 按 "+y (复制) 或 "+d (剪切)。

    从系统剪贴板 粘贴 到 Vim

    在 Normal 模式下，使用 "+p 或 "*p。

    | 命令 | 描述 |
    | - | - |
    | "+p | 在光标后粘贴系统剪贴板的内容 |
    | "+P | 在光标前粘贴系统剪贴板的内容 |

    设置默认使用系统剪贴板（推荐）:

    ```vim
    " 设置默认寄存器为系统剪贴板
    set clipboard=unnamedplus " Linux, Windows (WSL)
    " 对于 macOS，有时可能需要使用 unnamed
    " set clipboard=unnamed
    ```

    解释：

    * unnamedplus：让默认寄存器 (") 与 "+ (系统剪贴板) 联通。复制 (yy)、粘贴 (p) 等命令会直接操作系统剪贴板。

    * unnamed：在 macOS 上，有时这个选项效果更好，它让默认寄存器与 "* 联通。

    **在命令行模式下粘贴**:

    如果你想在 Vim 的命令行（比如在搜索 / 或命令 : 中）粘贴系统剪贴板的内容，可以按 Ctrl+r 然后输入 +。

* 确认你的 Vim 版本是否编译了剪贴板支持

    在终端里运行以下命令：

    ```bash
    vim --version | grep clipboard
    ```

    或者直接在 Vim 内部输入：

    ```vim
    :version
    ```

    然后查找 clipboard 和 xterm_clipboard。

    * 如果看到 +clipboard 和 +xterm_clipboard：恭喜，你的 Vim 支持系统剪贴板，可以直接使用下面的所有方法。

    * 如果看到 -clipboard：说明你的 Vim 不支持。你需要安装一个带剪贴板功能的 Vim。

        * Ubuntu/Debian: sudo apt install vim-gtk3 (或者 vim-gnome, vim-gtk)

        * macOS (使用 Homebrew): brew install vim

        * CentOS/RHEL: sudo yum install vim-X11 (可能需要)

* vim help

    :help /\[]
    :help whitespace
    :help [:alnum:]

* 可视模式

    按 v 进入普通可视模式

    按 V 进入行可视模式

    按 Ctrl+V 进入块可视模式

    ```vim
    " 在 .vimrc 中修改可视模式颜色
    highlight Visual cterm=reverse ctermbg=NONE
    ```

    ```vim
    " 临时禁用高亮
    :nohlsearch
    ```

    ```vim
    " 禁用鼠标选择自动进入可视模式
    set mouse-=a
    " 或只禁用部分鼠标功能
    " set mouse=nvi  " n:普通模式, v:可视模式, i:插入模式

    " 鼠标释放后自动退出可视模式
    autocmd CursorMoved * if mode() =~ '^[vV]' | silent! execute "normal! \e" | endif
    ```

    ```vim
    " 按 Ctrl+L 清除高亮
    nnoremap <C-l> :nohlsearch<CR>:call clearmatches()<CR>
    ```

* 为什么 linux 上正常关闭 vim 后不会留下 ~ 文件，而 windows 上会

    vim 在 linux 上默认不开启 backup，但是在 windows 上开启。

* vim 的恢复功能

    * 使用 vim -r filename 恢复交换文件

    * `:recover`

* 在 Vim 中比较差异

    ```vim
    vim -d report.txt report.txt~
    # 或进入 Vim 后
    :vert diffsplit report.txt~
    ```

* `.vimrc`生效时机

    在`.vimrc`保存后，重新启动 file_1 的 vim 编辑器即可。不需要关闭系统上所有的 vim。

    配置生效通常有以下三种情况：

    * 重新启动 Vim： 这是最稳妥的方法。当你关闭并重新打开 Vim 时，它会自动加载配置文件。

    * 手动执行 Source 命令： 在不退出 Vim 的情况下，可以通过命令让当前运行的 Vim 实例立即应用新配置：

        * 在 Vim 内部输入：:source ~/.vimrc（Linux）或 :source $VIM/_vimrc（Windows）。

            或`:so ~/.vimrc`, `:source $MYVIMRC`

    * 在 .vimrc 中设置自动指令（Autocmd）： 你可以添加一段代码，让 Vim 在检测到配置文件保存时自动执行 source 命令。

    特性	Linux / macOS	Windows
    默认文件名	.vimrc	_vimrc (通常) 或 .vimrc
    主配置路径	~/.vimrc (用户家目录)	C:\Users\用户名\_vimrc 或 Vim 安装目录
    路径变量	$HOME	$VIM 或 $HOME
    换行符	LF	CRLF

    ```vim
    " 按下 \ + s 立即重新加载配置
    nnoremap <leader>s :source $MYVIMRC<cr>
    ```

    有时候你 source 了文件，但发现外观没变。这通常是因为：

    * 插件需要重启： 某些插件（如代码补全、状态栏）在初始化时加载，简单的 source 可能无法重置它们的运行状态。

    * 缓存问题： 某些配色方案（Colorscheme）在切换时，旧的颜色属性可能残留在内存中，建议重新运行 :colorscheme 方案名。

* vim 插入新行并且不进入 insert 模式

    * `:put` (简写为`:pu`)

        向下插入一行。

        向上插入一行为`:put!`或`:pu!`

    * `:call append()`

        ```vim
        :call append(line('.'), '')  " 在当前行下方插入空行
        :call append(line('.')-1, '') " 在当前行上方插入空行
        ```

    * 映射快捷键

        ```vim
        " 在 ~/.vimrc 中添加映射
        nnoremap <Leader>o o<Esc>     " 下方插入空行并返回普通模式
        nnoremap <Leader>O O<Esc>     " 上方插入空行并返回普通模式
        ```

* `grep -P`表示使用 PCRE 的 regex

    PCRE, for Perl Compatible Regular Expression

* vim 中搜索 metacharacter `.` 的帮助文档

    `:help /\.`

* vim 中有关 regex 的 help 命令

    ```
    :help pattern-searches
    :help atom
    ```

* vim 打开文件后，是否可以总是追踪最新内容？

    在传统的 Vim 逻辑中，它是一个**静态编辑器**，默认不会像 `tail -f` 那样实时滚动。不过，Vim 确实提供了几种方式来“追踪”文件的变化。

    1. 基础版：手动刷新
    
        如果你只是想偶尔同步一下磁盘上的改动：
        * 输入 `:e` 或 `:e!`（强制刷新，丢弃未保存的修改）。

    ---

    2. 进阶版：自动读取 (autoread)
        
        你可以让 Vim 在检测到文件被外部修改时自动重新读取。
        在 Vim 中输入：
        ```vim
        :set autoread
        ```
        **注意**：`autoread` 只有在 Vim 触发某些事件（比如运行了外部命令、按下某键、或者焦点切换回终端）时才会检查文件。它**不会**在你盯着屏幕看时自动更新。

    ---

    3. 高级版：模拟 `tail -f`（实时追踪）
    
        如果你想让 Vim 像 `tail -f` 一样，只要文件有新行就自动刷新并滚动到底部，可以使用以下命令组合：

        方法 A：使用内置定时器（推荐）

        在终端的 Vim 中执行以下命令，它会每秒自动检查一次文件并滚动：
        ```vim
        :set autoread
        :au CursorHold,CursorHoldI * checktime
        :let timer = timer_start(1000, {-> execute('checktime | normal! G')}, {'repeat': -1})
        ```
        * `checktime`: 强制检查文件更新。
        * `normal! G`: 将光标跳转到文件末尾。
        * `1000`: 间隔 1000 毫秒（1秒）。

        方法 B：进入“只读监控模式”

        如果你只是想看，不需要编辑，可以直接在 shell 中使用 `view` 命令并配合定时触发：
        ```bash
        vim -M +$ +"set autoread" +"au CursorHold * checktime" +"feedkeys(\"lh\")" filename
        ```
        *(这会利用 `CursorHold` 事件，通过模拟按键来不断触发文件检查。)*

    ---

    4. 终极版：使用插件

        如果你经常有这种需求，**TailBundle** 或 **vim-tail** 插件是更好的选择。它们专门为“在 Vim 中 tail 文件”设计，处理大文件和高频更新更加优雅。

    **总结**

    | 需求 | 命令 |
    | :--- | :--- |
    | **偶尔刷新** | `:e!` |
    | **后台自动读取** | `:set autoread` |
    | **实时滚动到底部** | 使用 `timer_start` 脚本 (见上文) |

    **需要我帮你把上述 `timer_start` 代码封装成一个可以快捷开启/关闭的 Vim 函数吗？**

* vim 中大写字母命令的作用

    在 Vim 中，大写字母命令通常是其对应小写字母命令的"增强版"或"相反方向版"。以下是主要大写命令的概述：

    光标移动类

    * H - High：移动到屏幕顶部（Home position）

        * h 向左移动，H 移动到屏幕最上方

    * M - Middle：移动到屏幕中间

    * L - Low：移动到屏幕底部

        * l 向右移动，L 移动到屏幕最下方

    插入模式类

    * I - Insert 在行首：在行首第一个非空白字符前插入

        * i 在当前光标前插入，I 在行首插入

    * A - Append 在行尾：在行尾追加

        * a 在当前光标后插入，A 在行尾插入

    * O - 在当前行上方新建一行

        * o 在当前行下方新建一行，O 在上方新建

    文本对象操作类

    * C - Change 到行尾：删除从光标到行尾并进入插入模式

        * c{motion} 删除指定范围并插入，C = c$

    * D - Delete 到行尾：删除从光标到行尾

        * d{motion} 删除指定范围，D = d$

    * S - 删除整行并进入插入模式（Substitute line）

        * s 删除单个字符并插入，S = ^C 或 ddO

    * Y - Yank 整行

        * y{motion} 复制指定范围，Y = yy（复制整行）

    跳转与搜索类

    * F - 向前（右）搜索字符并跳转到该字符上

        * f{char} 跳到字符，F 向左搜索

    * T - 向前（右）搜索字符并跳到该字符前

        * t{char} 跳到字符前，T 向左搜索

    * G - Go 到最后一行或指定行

        * gg 到第一行，G 到最后一行，{n}G 到第 n 行

    其他常用

    * X - 删除光标前的字符

        * x 删除光标下的字符，X 删除光标前的字符

    * P - 粘贴到光标前

        * p 粘贴到光标后，P 粘贴到光标前

    * J - 连接当前行和下一行（Join）

        * j 向下移动，J 合并行

    记忆技巧

    * 位置相关：H（高）、M（中）、L（低）

    * 行相关：I（行首）、A（行尾）

    * 反向操作：F/T、P 是其小写版本的反方向

    * 增强范围：C/D/Y 是作用到行尾的版本

    这些大写命令让 Vim 的操作更加高效，减少了模式切换和重复按键的需要。

* 我希望给 vim 加这样一个功能：normal 模式下输入 I 后，用户输入某个字符 c，然后程序在当前位置插入字符 c 后自动执行 esc 返回 normal 模式。如果输入 [N]I，则 vim 等待用户输入 N 个字符，在当前位置插入后返回到 normal 模式。

    这个需求非常实用，本质上是把 **Insert 模式** 变成了一个 **“原子化”的操作**。在 Vim 中，我们可以通过 `getcharstr()` (或旧版的 `getchar()`) 配合 `v:count1` 来完美实现这个功能。

    你可以将以下代码添加到你的 `.vimrc`（或 `init.vim`）中：

    **Vimscript 实现方案**

    ```vim
    " 定义快速插入函数
    function! QuickInsert() abort
        " 获取输入的计数（[N]I），默认为 1
        let l:count = v:count1
        let l:inserted_text = ""

        " 循环获取 N 个字符
        for i in range(l:count)
            let l:char = getcharstr()
            " 如果中途按了 Esc，则终止输入
            if l:char == "\<Esc>"
                return
            endif
            let l:inserted_text .= l:char
        endfor

        " 在当前光标处插入字符
        " 使用 feedkeys 可以确保触发原本的插入逻辑，或者直接用 execute
        execute "normal! i" . l:inserted_text
    endfunction

    " 将 I 映射到该函数
    " <C-u> 用于清除 Vim 默认在命令行展示的范围，确保 v:count1 被正确读取
    nnoremap <silent> I :<C-u>call QuickInsert()<CR>

    ```

    友情提示

    在 Vim 原生快捷键中，`I` 的功能是 **“跳转到行首第一个非空字符并进入插入模式”**。如果你想保留原生的 `I`，建议将上面的映射键改为 `ni`（取 "Next Insert" 之意）或其他不常用的组合。


* vim 中，我使用 y 复制一段文字后，想在另一个地方执行 vex 删除一个单词后，再按 p 把复制的文字粘贴到当前位置。但是这时候粘贴板上的数据已经改变了。如何处理？

    在 Vim 中，你遇到的是**寄存器（registers）**的问题。当你执行 `v e x` 时，删除的内容会进入默认寄存器（`""`），覆盖了之前复制的内容。

    有几种方法可以解决：

    方法一：使用命名寄存器（推荐）

    1. **复制时**：使用 `"ay` 将内容保存到寄存器 `a`
       - `"ay`：先按 `"a`，再按 `y`
    2. **粘贴时**：使用 `"ap` 从寄存器 `a` 粘贴
       - `"ap`：先按 `"a`，再按 `p`

    这样删除操作不会影响寄存器 `a` 的内容。

    方法二：使用系统剪贴板寄存器

    如果想让复制的内容在系统剪贴板中：

    1. **复制时**：`"+y`（Linux/macOS）或 `"*y`（Windows）

    2. **粘贴时**：`"+p` 或 `"*p`

    方法三：使用黑洞寄存器删除

    当删除不需要保存的内容时，可以使用黑洞寄存器 `_`：

    ```vim
    "_vex  " 删除内容但不保存到任何寄存器
    ```

    方法四：先删除再粘贴

    ```vim
    p    " 先粘贴之前复制的内容
    vex  " 再删除单词
    ```

    查看寄存器内容

    ```vim
    :reg    " 查看所有寄存器内容
    :reg a  " 查看寄存器a的内容
    ```

    常用寄存器说明

    - `""`：默认寄存器（你遇到问题的地方）
    - `"0`：最近一次复制的内容
    - `"a`-`"z`：命名寄存器（可自定义使用）
    - `"+`/`"*`：系统剪贴板
    - `"_`：黑洞寄存器（删除但不保存）

    **建议**：习惯使用命名寄存器（如 `"ay` 和 `"ap`）来避免这类问题，特别是当需要在多个位置进行复制粘贴操作时。



* ysiw" 为当前单词添加双引号。这个如何理解？

    ysiw" 是 vim-surround 插件的一个组合命令，分解理解如下：

    命令结构

    ```text
    ys i w "
    ┃  ┃ ┃ ┃
    ┃  ┃ ┃ └ 要添加的符号（这里是双引号）
    ┃  ┃ └─── 范围选择器（w = word，单词）
    ┃  └───── 内部文本对象（i = inner，内部的）
    └──────── 操作符（ys = you surround，添加环绕）
    ```

    详细解释

    1. ys

        * y 代表 “you”（插件中的记忆方式，表示“添加环绕”）

        * s 代表 “surround”

        * 整体表示：我要给某个东西添加环绕符号

    2. iw

        * 这是 Vim 的 文本对象 语法

        * i 表示 inner（内部，不包含周围空格）

        * w 表示 word（单词）

        * iw 合起来表示：光标所在的整个单词（不含周围空格）

    举例：

    文本 Hello, world!

    光标在 world 上时，iw 会选择 world（不包括前后的逗号或空格）

    3. "

        * 要添加的环绕符号

        * 插件会自动匹配成对的符号：" → ""，( → ()，[ → [] 等

    执行效果

    假设光标在单词 apple 上：

    ```text
    apple
         ↓ 执行 ysiw"
    "apple"
    ```

    类似用法

    * ysiw( → (apple)

    * ysiw[ → [apple]

    * ysiw' → 'apple'

    * ysiwt → <apple>（标签需要指定名称）

    记忆技巧

    * ys = “你包围”（添加环绕）

    * ds = “删除包围”

    * cs = “改变包围”

    * iw = “内部单词”（常用文本对象）

    * aw = “一个单词”（包含周围空格）

    更直观的替代

    如果觉得 ysiw" 太长，可以：

    * 先按 viw 选中单词

    * 再按 S"（Visual 模式下 S + 符号）

    两者效果相同，但 ysiw" 更快捷，不需要进入 Visual 模式。

    注：

    1. vim-surround 没法加双星号 `**`，感觉整体的功能不灵活也不好用。


* 我想像 vscode 那样显示某一行是新增的/修改的/删除的，vim 里如何做到？

    要在 Vim 中像 VS Code 那样实时显示行号旁边的 Git 状态（新增/修改/删除），有几种方案：

    方案一：使用 vim-gitgutter（推荐）

    安装

    ```vim
    " 使用 vim-plug
    Plug 'airblade/vim-gitgutter'

    " 使用 Vim 内置包管理
    " mkdir -p ~/.vim/pack/git-plugins/start
    " cd ~/.vim/pack/git-plugins/start
    " git clone https://github.com/airblade/vim-gitgutter.git
    ```

    基本配置

    ```vim
    " 在 ~/.vimrc 中添加
    let g:gitgutter_enabled = 1           " 启用
    let g:gitgutter_signs = 1             " 显示符号
    let g:gitgutter_highlight_lines = 0   " 不高亮整行（默认）

    " 更新延迟（毫秒）
    set updatetime=100
    let g:gitgutter_realtime = 1
    let g:gitgutter_eager = 1
    ```

    显示效果

    ```text
    │ 行号 │ 代码
    ├──────┼─────────────
    │  1   │ 未修改的行
    │  2 + │ 新增的行
    │  3 ~ │ 修改的行
    │  4 - │ 删除的行（在附近显示）
    ```

    方案二：使用 signify（更轻量）

    ```vim
    Plug 'mhinz/vim-signify'
    ```

    更适合大型仓库，性能更好。

    方案三：Neovim 的 gitsigns.nvim

    如果你用 Neovim：

    ```lua
    -- 使用 lazy.nvim
    {
      'lewis6991/gitsigns.nvim',
      opts = {
        signs = {
          add          = { text = '│' },
          change       = { text = '│' },
          delete       = { text = '_' },
          topdelete    = { text = '‾' },
          changedelete = { text = '~' },
          untracked    = { text = '┆' },
        },
        signcolumn = true,
        numhl      = false,
        linehl     = false,
      }
    }
    ```

    自定义符号和颜色

    修改符号样式

    ```vim
    " 使用不同的符号
    let g:gitgutter_sign_added = '+'
    let g:gitgutter_sign_modified = '~'
    let g:gitgutter_sign_removed = '-'
    let g:gitgutter_sign_removed_first_line = '^^'
    let g:gitgutter_sign_modified_removed = 'ww'

    " 或者使用 Unicode 字符
    let g:gitgutter_sign_added = '│'
    let g:gitgutter_sign_modified = '│'
    let g:gitgutter_sign_removed = '_'
    let g:gitgutter_sign_removed_first_line = '‾'
    ```

    修改颜色

    ```vim
    " 定义高亮颜色
    highlight GitGutterAdd    guifg=#009900 guibg=NONE ctermfg=2
    highlight GitGutterChange guifg=#bbbb00 guibg=NONE ctermfg=3
    highlight GitGutterDelete guifg=#ff2222 guibg=NONE ctermfg=1

    " 如果想高亮整行（像 VS Code 那样）
    let g:gitgutter_highlight_lines = 1
    " 然后自定义行高亮颜色
    highlight GitGutterAddLine    guibg=#003300
    highlight GitGutterChangeLine guibg=#333300
    highlight GitGutterDeleteLine guibg=#330000
    ```

    常用操作

    ```vim
    " 跳转到下一个/上一个变更
    ]h  " 下一个 hunk
    [h  " 上一个 hunk

    " 暂存/撤销当前区块
    <Leader>hs  " 暂存当前区块
    <Leader>hu  " 撤销当前区块

    " 预览更改
    <Leader>hp  " 预览当前区块的更改

    " 切换显示
    :GitGutterToggle     " 切换显示
    :GitGutterEnable     " 启用
    :GitGutterDisable    " 禁用
    :GitGutterSignsToggle " 切换符号显示
    ```

    完整配置示例

    ```vim
    " ~/.vimrc
    call plug#begin('~/.vim/plugged')
    Plug 'airblade/vim-gitgutter'
    call plug#end()

    " GitGutter 配置
    let g:gitgutter_map_keys = 0  " 禁用默认快捷键，自己定义
    let g:gitgutter_sign_allow_clobber = 0

    " 符号定义
    let g:gitgutter_sign_added = '│'
    let g:gitgutter_sign_modified = '│'
    let g:gitgutter_sign_removed = '_'
    let g:gitgutter_sign_removed_first_line = '‾'
    let g:gitgutter_sign_modified_removed = '~'

    " 颜色定义
    highlight GitGutterAdd          guifg=#00ff00 ctermfg=2
    highlight GitGutterChange       guifg=#ffff00 ctermfg=3
    highlight GitGutterDelete       guifg=#ff0000 ctermfg=1
    highlight GitGutterChangeDelete guifg=#ff8800 ctermfg=4

    " 更新频率
    set updatetime=100

    " 自定义快捷键
    nmap ]h <Plug>(GitGutterNextHunk)
    nmap [h <Plug>(GitGutterPrevHunk)
    nmap <Leader>hs <Plug>(GitGutterStageHunk)
    nmap <Leader>hu <Plug>(GitGutterUndoHunk)
    nmap <Leader>hp <Plug>(GitGutterPreviewHunk)
    ```

    没有插件的临时方案

    如果不想装插件，可以用内置命令：

    ```vim
    " 查看当前文件的差异
    :Gdiff

    " 或者用外部命令
    :!git diff --no-ext-diff %

    " 或者开启 Vim 的 diff 模式
    :windo diffthis   " 开启 diff 模式
    :windo diffoff    " 关闭 diff 模式
    ```

    问题排查

    如果没显示：

    * 确认在 Git 仓库中

    * 检查 :GitGutterEnable 是否开启

    * 确认 updatetime 不要太长（建议 100-500ms）

    * 查看 :messages 是否有错误

    安装配置后，Vim 就会在行号旁边显示 Git 状态，体验类似 VS Code 了！




* vim 在退出时，要求我先 :w 保存，才能 :q 退出，但是我忘了刚才修改了什么东西，是否有类似 diff 命令的方式查看 vim 编辑的文件改动了哪些？

    有几种方法可以查看未保存的修改：

    1. 查看当前文件的改动

        ```vim
        :changes
        ```

        或

        ```vim
        :undolist
        ```

        这会显示你做的修改历史。

        注：

        1. 这些显示的都只是操作历史，无法像 diff 那样看到直观的文本对比

    2. 比较已保存文件和当前版本的差异

        ```vim
        :DiffOrig
        ```

        如果 :DiffOrig 命令不存在，需要先加载 diff 功能：

        ```vim
        :vert diffsplit
        ```

        注：

        1. 简单试了下，似乎不行

    3. 使用 git diff 风格查看

        ```vim
        :w !diff % -
        ```

        这会比较已保存文件和当前未保存内容的差异。

        注：

        1. 这个好像还行, 不知道 :w 是否会直接保存文件

    4. 查看当前行或可视选择的差异

        * 查看当前行修改：g:

        * 查看整个文件的修改概要：:diffthis

    5. 如果你只是想看看修改

        先保存到临时文件，然后比较：

        ```vim
        :w /tmp/temp_file
        :vert diffpatch /tmp/temp_file
        ```

    6. 推荐配置（添加到 .vimrc）

        ```vim
        " 快速查看未保存的差异
        command DiffOrig vert new | set bt=nofile | r # | 0d_ | diffthis | wincmd p | diffthis
        ```

    7. 不想保存修改就退出

        如果你确定不需要这些修改：

        * 强制退出不保存：:q!

        * 回到上次保存的状态：:e!（然后再 :q 退出）

    8. 使用插件

        安装 vim-fugitive 等插件可以获得更好的 Git 集成和 diff 查看功能。

    最简单的方式是直接使用 :DiffOrig（如果可用），或者先执行 :w 保存，然后用 Git 等工具查看修改。

* vim 中的特殊字符

    只要一个字符满足下面任意一条，就**不能直接敲出来**，必须用 `Ctrl-V`（或等价方式）：

    * 它是 **ASCII 控制字符**（0x00–0x1F, 0x7F）
    * 它在 Vim 里有 **特殊语义**（如 `<Esc>`、`<Tab>`、`<CR>`）
    * 你希望插入的是 **“这个字符本身”**，而不是它的功能

    **典型场景：**

    * 宏 (`let @a = '...'`)
    * 映射 (`:map`)
    * 搜索模式
    * 二进制/协议文本
    * 精确复现终端行为

    常见的 Vim 内部表示 / 终端控制字符:

    * ASCII 控制字符

        | Ctrl 组合  | 名称  | ASCII | Vim 显示 | 说明      |
        | -------- | --- | ----- | ------ | ------- |
        | `Ctrl-@` | NUL | 0x00  | `^@`   | 空字符     |
        | `Ctrl-A` | SOH | 0x01  | `^A`   |         |
        | `Ctrl-B` | STX | 0x02  | `^B`   |         |
        | `Ctrl-C` | ETX | 0x03  | `^C`   | 中断      |
        | `Ctrl-D` | EOT | 0x04  | `^D`   |         |
        | `Ctrl-G` | BEL | 0x07  | `^G`   | 响铃      |
        | `Ctrl-H` | BS  | 0x08  | `^H`   | 退格      |
        | `Ctrl-I` | TAB | 0x09  | `^I`   | **Tab** |
        | `Ctrl-J` | LF  | 0x0A  | `^J`   | 换行      |
        | `Ctrl-M` | CR  | 0x0D  | `^M`   | **回车**  |
        | `Ctrl-[` | ESC | 0x1B  | `^[`   | **Esc** |
        | `Ctrl-\` | FS  | 0x1C  | `^\`   |         |
        | `Ctrl-]` | GS  | 0x1D  | `^]`   |         |
        | `Ctrl-^` | RS  | 0x1E  | `^^`   |         |
        | `Ctrl-_` | US  | 0x1F  | `^_`   |         |
        | `DEL`    | DEL | 0x7F  | `^?`   | 删除      |

        插入方法：

        ```
        Ctrl-V  Ctrl-<key>
        ```

        例如：

        * `Ctrl-V Ctrl-[`: 插入 `^[`
        * `Ctrl-V Ctrl-M`: 插入 `^M`
        * `Ctrl-V Ctrl-I`: 插入真正的 Tab（不是空格）

    * Vim 特殊键（键名 vs 字面字符）

        这些键平时你敲的是**动作**，不是字符：

        | 键         | 键名      | 字面插入方式             |
        | --------- | ------- | ------------------ |
        | Esc       | `<Esc>` | `Ctrl-V Esc`       |
        | Tab       | `<Tab>` | `Ctrl-V Tab`       |
        | Enter     | `<CR>`  | `Ctrl-V Enter`     |
        | Backspace | `<BS>`  | `Ctrl-V Backspace` |

        如果你不使用 `Ctrl-V`，Vim 会直接执行命令，或结束输入，或触发补全。

    * 非 ASCII / 特殊 Unicode

        十六进制方式（推荐）

        ```vim
        Ctrl-V u 00a0     " Unicode
        Ctrl-V x 1b       " 8-bit
        Ctrl-V o 033      " 八进制
        ```

        例子：

        | 字符            | 用途               |
        | -------------- | ----------------   |
        | `Ctrl-V u00a0` | 不断行空格 (NBSP)     |
        | `Ctrl-V x1b`   | ESC（和 `Ctrl-[` 等价） |
        | `Ctrl-V x7f`   | DEL              |

    **查看特殊按键的列表：**

    * `:help i_CTRL-V`

        插入模式下 Ctrl-V 的完整说明

        这里有：

        * 所有可插入控制字符
        * 十六进制 / Unicode 规则
        * 精确行为说明

    * `:help key-notation`

        Vim 的 `<Esc>` / `<CR>` / `<Tab>` 表示法

    * `:help ASCII`

        完整 ASCII 表（含控制字符）

    * `:help special-keys`

    **建议：**

    * 对于宏，优先使用 `<Esc>`, `<CR>`（通过 `:execute` / `:normal!`），少用字面 `^[`

    * 对于需要精确字节控制，使用`Ctrl-V`

## topics

### normal 常用 command

* vim `.`命令

    作用：重复上一次修改操作

    详细说明：

    * 重复最近一次在普通模式下执行的修改命令

    * 可以重复插入、删除、替换等操作

    * 示例：

        * dw 删除一个单词 → . 再删除下一个单词

        * ihello<Esc> 插入文本 → . 再次插入"hello"

* normal 模式下的常用命令

    * move around: `h`, `j`, `k`, `l`

    * delete a next character: `x`

    * undo the last edit: `u`

    * redo: `ctrl` + `r`

    * save and exit: `ZZ` (upper cases)

    * discard changes and exit: `:q!`

    * delete a line: `dd`

    * insert a character before the character under the cursor: `i`

    * intert text after the cursor: `a`

    * add a new line below: `o`

    * open a line above the cursor: `O` (uppercase)

* dw 与 dW 有什么区别

    在 Vim 中，`w` 和 `W` 的核心区别在于对“单词（Word）”的定义不同。这直接决定了 `dw` 和 `dW` 的删除范围。

    1. 小写 `w` (Word)

        小写 `w` 认为单词是由**字母、数字、下划线**组成的。只要遇到标点符号或空格，它就会认为一个单词结束了。

        * **定义：** 连续的序列，包含 `[a-zA-Z0-9_]`。
        * **行为：** 遇到 `.`、`#`、`-`、`(` 等特殊符号时会停下来。

    2. 大写 `W` (WORD)

        大写 `W` 认为单词是由**非空白字符**组成的。它唯一认准的边界就是**空格（Space/Tab/Newline）**。

        * **定义：** 连续的非空字符（non-blank characters）。
        * **行为：** 忽略所有标点符号，直接跳到下一个空格之后。

    注：

    1. 存疑，因为实测 dw 会删除连续的 # 号，不会因为 # 是特殊符号而只删一个。 
  
* 使用`s`删除内容的技巧

    examples:

    * `nnoremap d# :s/^\s*//<CR>`

        * `^`: 走到行首非空字符

        * `\s*`: 匹配多个空白字符

        * `//`：替换成空，即删除

    * `nnoremap d# :s/^#\+\s*//<CR>`

        命令解析

        * `^`：锁定行首。

        * `#\+`：匹配一个或多个连续的 # 号（在 Vim 正则中 + 需要转义）。

        * `\s*`：匹配 # 号后面跟随的零个或多个空格或制表符。

        * `//`：替换为空（即删除）。

        * `<CR>`：回车执行。

### g 命令

* ``` g`" ```: ``` g` ```（g 加上反引号）的意思是“跳转到某个标记，且不把这次跳转加入到跳转历史（jumplist）中”，这样可以保持你的跳转历史干净。

* 详细讲讲 vim 中 g 命令的用法

    在 Vim 中，`g` 机制可以说是最强大也最让人敬畏的“双刃剑”之一。它既是一个**普通模式下的前缀快捷键**，也是一个**命令行模式（Ex 命令）下的全局全局匹配工具**。

    为了让你彻底搞懂它，我们把 `g` 的用法拆分为两大核心版块：**命令行模式下的 `:g`（Global）** 和 **普通模式下的 `g` 前缀快捷键**。

    一、 核心杀手锏：命令行模式下的 `:g` (Global)

    如果你听别人说“Vim 的 `g` 命令强到逆天”，他们九成指的是这个。它的核心逻辑是：**在全文件中，找出所有匹配特定模式（正则表达式）的行，并对这些行执行指定的命令。**

    `g`是`global`命令的缩写。

    1. 基本语法结构

        ```vim
        :[range]g/pattern/cmd
        ```

        * `[range]`：指定范围，默认是整个文件（`%`）。
        * `pattern`：要寻找的正则表达式。
        * `cmd`：要执行的 Ex 命令（如果不写，默认是 `:p`，即打印/显示这些行）。

    2. 常用实战案例

        * **删除所有包含 "error" 的行**

            ```vim
            :g/error/d
            ```

        * **删除所有空行（包括只有空格的空行）**

            ```vim
            :g/^\s*$/d
            ```

        * **倒序排列整个文件（著名的 Vim 奇技淫巧）**

            ```vim
            :g/^/m 0
            ```

        > **原理解析：** 匹配每一行的开头（`^`），然后用 `m 0`（move）把当前行移动到文件的第 0 行（最顶部）。不断把下面的行挪到最上面，整个文件就反过来了。


        * **只在包含 "TODO" 的行里，把 "fix" 替换为 "done"**

            ```vim
            :g/TODO/s/fix/done/g
            ```

        * **把所有匹配行的内容复制到一个指定寄存器（比如寄存器 a）中**

            ```vim
            " 先清空寄存器 a
            :qa
            " 附加复制（注意 A 是大写，表示追加）
            :g/pattern/y A
            ```

        注：

        1. 还能有这种写法

            `:%g/^/norm! jdd`

            和这种写法：

            `:'<,'>g/./normal! [你的命令]`

    3. 孪生兄弟：`:v` 或 `:g!`（反向匹配）

        如果你想找**不匹配**某模式的行，可以使用 `:v`（来自 inverse video 的历史称呼）或 `:g!`。

        * **删除所有不包含 "keep" 的行（只保留含有 keep 的行）**
            
            ```vim
            :v/keep/d
            " 或者
            :g!/keep/d
            ```

    二、 普通模式下的 `g`（作为前缀快捷键）

    在普通模式下，`g` 并没有独立功能，它像是一个“第二功能键”（类似于键盘上的 Alt 或 Fn），用来扩展 Vim 有限的按键资源。

    1. 文本移动与导航（Movement）

        * **`gg`**：跳到文件的第一行（最常用）。
        * **`gd`**（Go to Definition）：跳转到当前光标下局部变量的定义处；**`gD`** 跳转到全局定义。
        * **`gf`**（Go to File）：如果光标下是一个文件名/路径（如 `src/main.py`），直接打开该文件。
        * **`gi`**：回到上一次进入插入模式（Insert）的地方，并直接进入插入模式。
        * **`gv`**：重新选中上一次在可视模式（Visual）下选中的文本。

    2. 屏幕行与实际行（处理自动折行）

        如果你的代码或文本很长，在屏幕上折成了好几行：

        * `gj` / `gk`：在屏幕显示的行中向下/向上移动，而不是逻辑上的下一行/上一行。
        * `g0` / `g$`：移动到屏幕显示行的行首/行尾。

    3. 大小写与文本转换

        `g` 后面跟特定的操作符，可以实现非常高效的文本转换：

        * **`gu` + 动作**：将选定范围转为**小写**（例如 `guw` 变当前单词为小写，`guu` 变当前整行为小写）。
        * **`gU` + 动作**：将选定范围转为**大写**（例如 `gUU` 变整行为大写）。
        * **`g~` + 动作**：**翻转**大小写（大写变小写，小写变大写）。

    4. 文本排版与信息查看

        * `gq` + 动作：对文本进行排版/自动折行（Format）。常用于把一行极长注释规范地格式化为多行。
        
        * `ga`：显示光标下字符的 ASCII 码、十进制、十六进制和八进制值。
        
        * `g` 键后再按 `Ctrl-g`：显示当前文件的详细字数统计统计（行数、字数、字节数等），写文章时极好用。

    5. 搜索与替换的辅助

        * `g*` / `g#`：类似于 `*` 和 `#`（搜索光标下的单词），但 `g*` 是**部分匹配**。例如光标在 `bar` 上，`*` 只搜独立单词 `bar`，而 `g*` 会连带搜出 `foobar`。

    **总结：记忆小窍门**

    1. 当你在**底行输入 `:g/.../**` 时，它是 **Global（全局）** 的意思，用来做大面积的代码清洗和批量操作。
    2. 当你在**普通模式下按 `g**` 时，可以把它理解为 **Go** 或者 **General**，用来触发那些 Vim 原生键盘不够放的次级高效指令。

    你目前在用 Vim 处理什么具体的任务吗？如果有特定的批量修改需求，我可以帮你写一个专属的 `:g` 命令。

* 那如果 g 后跟 mark 呢

    在 Vim 中，`g` 后面跟标记（Mark）是非常经典且高效的组合。

    通常我们跳转到某个标记（比如用 `ma` 做了标记）使用的是 **`'` (单引号)** 或 ``` ` ``` (反引号)，例如 `'a` 或 ``` `a ```。

    而把 `g` 加在前面的 **`g'`** 和 **`g``**，最大的核心作用是：**在跳转到标记时，不改变你的“跳转列表”（Jump List）。**

    下面为你详细拆解它们的用法和区别。

    一、 核心用法：保持“跳转列表”的清洁

    在 Vim 中，每当你进行远距离跳转（比如用 `/` 搜索、按 `G` 到页尾，或者用普通的 `'a` 跳到标记），Vim 都会把你的出发点记录在跳转列表（Jump List）里。这样你就可以用 `Ctrl-O` 往回跳，用 `Ctrl-I` 往前跳。

    但是，如果你用 **`g'`** 或 **``` g` ```** 组合：

    * ``` g'a ```： 使用单引号，跳到标记 `a` 所在的**行首**，**但不把当前位置存入跳转列表**。

    * ``` g`a ```： 使用反引号，跳到标记 `a` 所在的**精确行列位置，**但不把当前位置存入跳转列表**。

    > **💡 实用场景：**
    > 假设你正在写第 100 行的代码，你想临时去标记 `a`（第 10 行）看一眼某个变量定义，看完立刻就回来。
    > * 如果用 `'a` 跳过去，再用 `Ctrl-O` 回来。此时，第 100 行和第 10 行都被塞进了跳转列表，频繁这样操作会让你的 `Ctrl-O` 链路变得很乱。
    > * 如果用 `g'a` 跳过去，你的跳转列表完全没有被污染，看完后你依然可以精准地用 `Ctrl-O` 退回到你原本的浏览主线上。
    > 

    二、 两个特殊的“自动标记”组合

    Vim 会自动为你记录一些特殊的标记（无须你手动 `m` 创建），配合 `g` 使用极为强悍：

    1. `g';` 和 `g向后` (精确定位最后修改)

        * **`;` 标记**：记录了你**上一次修改文本的位置**。
        * **`g';`**：直接跳到上一次修改处的那一行。
        * **``` g`; ```**：直接跳到上一次修改处的**精确光标位置**。

        > **体验区别：** 普通的 ``;` 也可以跳过去，但 `g`;` 的好处同样是不污染跳转列表，适合频繁在“当前阅读位置”和“上一次修改位置”之间反复横跳。

    2. `g',` 和 `g重做` (回到较新的修改)

        * **`,` 标记**：如果你用 `g`;`往回跳到了更早的修改点，那么`,` 标记就代表**较新的一次修改点**。
        * **`g',`** 或 ``` g`, ```：在修改历史的前后节点中向“前”跳转，同样不影响跳转列表。

    三、 总结：`'a` vs `` `a `` vs `g'a` vs ``` g`a ```

    为了方便记忆，我们可以通过下表看清它们的细微区别：

    | 命令 | 跳转目标 | 是否精确到列？ | 是否会改变跳转列表（影响 `Ctrl-O`）？ |
    | :--- | :--- | :--- | :--- |
    | **`'a`** | 标记 `a` 所在的**行首** | ❌ 否（变行首） |  是（记录当前位置） |
    | **`` `a ``** | 标记 `a` 的**精确位置** |  是（精确行列） |  是（记录当前位置） |
    | **`g'a`** | 标记 `a` 所在的**行首** | ❌ 否（变行首） | ❌ **否（保持列表干净）** |
    | **``` g`a ```** | 标记 `a` 的**精确位置** |  是（精确行列） | ❌ **否（保持列表干净）** |

    简单来说，在你想临时跳转、不想破坏 `Ctrl-O` / `Ctrl-I` 的浏览节奏时，盲压一个 `g` 前缀就对了！

    注：

    1. ctrl + i 似乎不是向前跳跃，ai 答得不对。。。目前不知道向前跳跃是什么

### markdown

* 防止 vim 的 markdown 在渲染时，把 Precision (P) 渲染成 Precision §

    * 修改 Vim 配置

        （未试过）

        在 .vimrc 中添加：

        ```vim
        " 禁用 markdown 中的特殊符号转换
        let g:markdown_enable_conceal = 0
        ```

        或者针对特定文件类型：

        ```vim
        autocmd FileType markdown setlocal conceallevel=0
        ```

    **对于不同的 Markdown 插件**

    如果你使用的是 vim-markdown 插件：

    ```vim
    let g:vim_markdown_conceal = 0
    let g:tex_conceal = ""
    let g:vim_markdown_math = 1
    ```

    如果你使用的是 markdown-preview.nvim：

    ```vim
    let g:mkdp_markdown_css = 'custom.css' " 可以自定义 CSS 来避免这个问题
    ```

* vim 渲染 markdown 使用什么插件

    在 Vim 中渲染和预览 Markdown，通常分为 **“实时浏览器预览”** 和 **“编辑器内美化”** 两种需求。以下是目前最主流、好用的插件推荐：

    1. 实时预览：Markdown Preview Plus (推荐)

        这是目前最流行且配置最简单的方案。它会在你保存或编辑时，自动在浏览器中打开一个同步滚动的预览页面。

          * **插件名：** `iamcco/markdown-preview.nvim`
          * **核心优势：**
              * 支持**同步滚动**（你在 Vim 翻页，浏览器跟着翻）。
              * 支持 MathJax 数学公式、流程图、时序图。
              * **无需**在浏览器安装额外扩展。
          * **安装注意：** 安装后通常需要运行命令来下载预编译好的二进制文件：
            ```vim
            " 使用 plug 管理器时
            Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'pandoc'] }
            ```

    2. 编辑器内美化：vim-markdown

        如果你希望在 Vim 窗口内部看到更漂亮的格式（比如隐藏链接地址、加粗显示等），这个插件是必备的。

          * **插件名：** `preservim/vim-markdown`
          * **核心优势：**
              * **折叠功能：** 可以按标题级别折叠代码块。
              * **语法高亮：** 支持表格、栅栏代码块的高亮。
              * **隐藏链接：** 只有光标移动到链接上时才显示完整的 `[text](url)`，平时只显示 `text`。

    3. 极简主义/命令行预览：glow.nvim

        如果你不想打开浏览器，想直接在终端的浮动窗口里看渲染效果，这个插件非常酷。

          * **插件名：** `ellisonleao/glow.nvim`
          * **核心优势：**
              * 基于终端渲染引擎 **Glow**。
              * 直接在 Vim 里弹出一个漂亮的预览层。
              * 非常适合快速查看 README 文件。

    4. 符号增强：vim-polyglot

        如果你不想折腾复杂的配置，直接安装 `sheerun/vim-polyglot`。这是一个语言包集合，它内置了非常扎实的 Markdown 语法高亮和缩进支持。

    **总结建议**

    | 需求 | 推荐插件 |
    | :--- | :--- |
    | **写文档/写书 (最强预览)** | `iamcco/markdown-preview.nvim` |
    | **整理笔记 (极致排版)** | `preservim/vim-markdown` |
    | **极客/快速查看** | `ellisonleao/glow.nvim` |

    **提示：** 很多现代 Vim 用户（尤其是 Neovim 用户）还会配合 `nvim-treesitter` 来获得更精准的代码块高亮。

    你想知道如何针对其中某一个插件进行具体的配置（例如快捷键设置）吗？

* - Finishing ... Done!x Post-update hook for markdown-preview.nvim ... Vim(call):E117: Unknown function: mkdp#util#install- markdown-preview.nvim: Already up to date.

    这个错误 `E117: Unknown function: mkdp#util#install` 通常是因为在执行安装钩子（Post-update hook）时，**插件尚未被加载到 Vim 的运行路径（runtimepath）中**，导致 Vim 找不到对应的安装函数。

    这在使用了“延迟加载”（Lazy Loading）或配置了 `for: 'markdown'` 的情况下非常常见。

    **解决方法**

    方法一：手动安装（最快解决）

    由于插件文件已经下载完成，你只需手动触发安装脚本即可。

    1.  在 Vim 中随便打开一个文件。
    2.  手动输入并运行以下命令：
        ```vim
        :call mkdp#util#install()
        ```
        如果提示命令不存在，请先执行 `:packloadall`（或者对于 `vim-plug`，先打开一个 `.md` 文件触发插件加载）再尝试。

    方法二：修改 `vim-plug` 配置（长期有效）

    如果你使用的是 `vim-plug`，建议将 `do` 钩子改为直接调用 shell 脚本，或者确保安装时函数已存在。

    **方案 A：直接调用内置脚本 (推荐)**

    ```vim
    Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']}
    ```

    > **注意：** 这里加上 `'vim-plug'` 是为了让插件在插件管理器更新时也被加载，从而能找到 `install` 函数。

    **方案 B：如果你有 Node.js 和 Yarn (最稳定)**
    直接让它去编译，不使用预编译包：

    ```vim
    Plug 'iamcco/markdown-preview.nvim', { 'do': 'cd app && yarn install' }
    ```

    方法三：针对 Neovim (Lazy.nvim) 用户

    如果你使用的是 `lazy.nvim`，配置应如下：

    ```lua
    {
      "iamcco/markdown-preview.nvim",
      cmd = { "MarkdownPreviewToggle", "MarkdownPreview", "MarkdownPreviewStop" },
      ft = { "markdown" },
      build = function() vim.fn["mkdp#util#install"]() end,
    }
    ```

    -----

    为什么会报错？

    当你第一次安装插件时，Vim 还没有加载 `markdown-preview.nvim` 目录下的 `autoload` 文件夹。`vim-plug` 尝试在插件生效前执行 `mkdp#util#install()`，就像是在一个还没拆封的箱子里找说明书，自然会报“找不到函数”的错误。

    下一步建议

    你可以尝试在 Vim 中输入 `:MarkdownPreview` 看看是否能正常启动。如果依然报错，请确认你的电脑是否安装了 **Node.js**，因为该插件后台运行需要 Node 环境。

    [Vim 插件故障排除步骤](https://www.youtube.com/watch?v=wKE7dpe8czs)
    这个视频详细讲解了当 Vim 插件无法正常加载或找不到函数时的通用检查思路，非常适合解决类似的 E117 错误。

    http://googleusercontent.com/youtube_content/0

* 我安装了这个插件，但是执行 MarkdownPreview 时什么都没发生。我看到 git repo 被成功 clone 了，但是后续的安装不太对劲，我怀疑安装过程中有问题，如何检查，或者手动安装

    `MarkdownPreview` 执行后没有任何反应，最常见的原因是该插件的 **后台服务（Node.js 运行时）没有正确安装或启动失败**。

    这个插件由两部分组成：Vim 脚本和放在 `app/` 目录下的 Node.js 程序。如果 `app/` 里的依赖没装好，执行命令时 Vim 找不到可执行文件，往往会静默失败。

    请按以下步骤进行排查和手动安装：

    1. 检查 Node.js 环境

        首先，确保你的系统中安装了 Node.js。在终端运行：

        ```bash
        node -v
        ```

        如果返回版本号（建议 v16+），则环境没问题。如果没安装，请先安装 Node.js。

    2. 手动安装后台依赖 (最有效的方法)

        如果自动钩子（hook）失败了，手动编译是最稳妥的。请打开终端，进入插件所在的目录：

        **Unix/macOS 用户:**

        ```bash
        # 进入插件目录（路径根据你的插件管理器可能有所不同）
        cd ~/.vim/plugged/markdown-preview.nvim/app
        # 或者 Neovim 用户可能是：
        # cd ~/.local/share/nvim/site/pack/packer/start/markdown-preview.nvim/app

        # 执行安装
        sudo npm install
        # 或者如果你有 yarn
        yarn install
        ```

        **Windows 用户:**
        进入 `C:\Users\用户名\AppData\Local\nvim-data\site\plugged\markdown-preview.nvim\app` (路径仅供参考)，在当前目录打开 PowerShell：

        ```powershell
        npm install
        ```

    3. 检查编译产物

        安装完成后，请确认 `app` 目录下是否存在名为 **`bin`** 的文件夹，且里面有类似 `markdown-preview-linux` 或 `index.js` 的文件。

          * 如果没有 `bin` 文件夹，说明 `npm install` 过程中报错了，请查看终端输出的错误信息。

    4. 检查 Vim 内的错误日志

        有时候插件报错了但没弹窗。在执行完 `:MarkdownPreview` 没反应后，立即在 Vim 中输入：

        ```vim
        :messages
        ```

        查看最后几行输出。如果看到类似 `spawn node ENOENT` 或 `Binary file not found`，就证实了是步骤 2 的安装问题。

    5. 检查浏览器配置

        如果后台启动了但没打开浏览器，可能是 Vim 不知道该调用哪个浏览器。在你的 `.vimrc` 或 `init.lua` 中尝试手动指定浏览器路径（以 Chrome 为例）：

        ```vim
        " Vim 格式
        let g:mkdp_browser = 'google-chrome'

        " 或者在 Windows 上可能是
        " let g:mkdp_browser = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
        ```

    **验证是否成功**

    完成上述手动安装后，重启 Vim，打开一个 `.md` 文件，执行：

    ```vim
    :MarkdownPreview
    ```

    此时你应该能看到终端或后台弹出一个本地服务器地址，并自动打开浏览器。

    **如果手动执行 `npm install` 时报错了，你可以把报错信息发给我吗？我可以帮你分析是权限问题还是网络问题。**

* vim 里，对于 markdown 的列表，比如

    ```
    * hello
        * nihao
        * zaijian
        * world
        hello is a greeting message.
    ```

    **vim-markdown**（推荐）

    安装：

    ```
    Plug 'preservim/vim-markdown'
    ```

    它提供：

    ```
    ]h   跳到上一层 list 的开始
    [h   跳到上一层 list 的结束
    ```

    光标在 message → 按：

    ```
    ]h
    ```

    就直接跳到 `* hello`。

    这是最接近“树结构父节点跳转”的插件功能。

* 是否有对 `*` 无序列表生成大纲的

    一般的导航/大纲插件（如 Vista、Tagbar）都是基于标题（Heading，即 `#`、`##`）来生成的，**默认都不会对无序列表（`*`、`-`）生成大纲**。

    不过，想要实现“把无序列表当成大纲来导航”的需求，Vim 有几种非常巧妙的解决办法：

    ---

    方案一：使用原生 `Ctrl-P` 模糊搜索（免插件，最快）

    如果你想在当前文件的所有无序列表中快速跳转，不需要安装任何大纲插件，直接利用 Vim 的搜索缓冲区配合快捷键就能搞定：

    1. 在 Vim 中输入以下命令（可以临时列出所有无序列表并让你选择跳转）：

        ```vim
        :ilist /^\s*[\*+-]\s/
        ```

    2. **更高效的做法**：直接在输入模式或正常模式下，按下快捷键 `/`，输入 `^\s*[*+-] `，然后配合 `n` 或 `N` 在无序列表之间像跳表单一样切换。

    ---

    方案二：利用 Vim 的折叠功能（Folding）

    无序列表通常有嵌套结构（缩进）。你可以利用 Vim 的缩进折叠，把列表以外的内容过滤掉，只留下缩进大纲。

    在你的 `.vimrc` 中加入或者临时输入：

    ```vim
    :set foldmethod=indent
    ```

    * 此时按 `zM` 可以折叠所有有缩进的列表。
    * 整个文件会变成一个由无序列表构成的“可折叠大纲”。
    * 移动到你想去的列表项上，按 `zo` 展开，`zc` 折叠。

    ---

    方案三：如果你使用的是 `CtrlP` 插件

    如果你装了经典的模糊搜索插件 `ctrlp.vim`，它有一个内置的 `crypto-line` 功能，可以通过自定义正则来把无序列表抽出来做成一个弹窗导航。

    在 `.vimrc` 中配置一个快捷键（比如 `<leader>l`）：

    ```vim
    nnoremap <leader>l :CtrlPLine<CR>^\s*[\*+-]\s
    ```

    按下快捷键后，会弹出一个动态模糊搜索框，**里面只实时显示你的无序列表行**，上下键选择，回车直接跳转。

    ---

    方案四：如果你使用的是 `Telescope` (Neovim 用户)

    如果你使用的是 Neovim 并装了 `Telescope`，那就简单太多了。你可以直接利用 `live_grep` 或 `current_buffer_fuzzy_find` 传入正则。

    可以绑定一个快捷键：

    ```lua
    vim.keymap.set('n', '<leader>gl', function()
      require('telescope.builtin').current_buffer_fuzzy_find({ default_text = "^\\s*[\\*+-]\\s" })
    end, { desc = "无序列表大纲" })
    ```

    这会在下方弹出一个非常优雅的浮动窗，里面全是你的无序列表，支持模糊过滤和实时预览跳转。

    ---

    **总结**：由于插件作者通常认为“标题”才算大纲，因此想要给“无序列表”做导航，最完美的体验是**方案三（Vim 用户）**或**方案四（Neovim 用户）**，通过模糊搜索插件过滤出列表项。你目前使用的 Vim 核心是原生的 Vim 还是 Neovim 呢？

* vim 如何显示 markdown 文件的导航

    在 Vim 中显示 Markdown 文件的导航（大纲/目录结构），最有效的方法是使用**插件**。因为 Vim 本身没有内置的侧边栏大纲功能，但通过社区插件，你可以获得像 IDE 一样丝滑的导航体验。

    以下是目前最流行、最好用的几种解决方案：

    ---

    方案一：使用 `vim-markdown-toc`（自动生成目录）

    如果你不想在侧边栏挂一个不断渲染的窗口，而是想直接在 Markdown 文件中**插入一个可以跳转的目录（TOC）**，这是最佳选择。

    * **插件地址**：`mzlogin/vim-markdown-toc`
    * **常用命令**：
    * `:GenTocGFM`：生成 GFM 风格的目录。
    * `:UpdateToc`：手动更新目录（通常保存时会自动更新）。

    > **提示**：生成的目录中，把光标移到某个标题上按 `gd`（Go to Definition）或者如果你配合了其他跳转快捷键，就能直接跳到对应段落。

    ---

    方案二：使用 `Tagbar` 或 `Vista.vim`（侧边栏大纲）

    如果你想要一个像 VS Code 那样在左侧或右侧独立显示的**悬浮/固定大纲栏**，可以使用基于 Ctags 的插件。

    1. Vista.vim（强烈推荐，现代且美观）

        `liuchengxu/vista.vim` 是一个非常现代的大纲插件，支持异步渲染，对 Markdown 的支持极好。

        * **安装要求**：需要安装 `universal-ctags`（通过 `brew install ctags` 或 `sudo apt install ctags`）。
        * **使用方法**：
        * 在 Vim 中输入 `:Vista` 即可打开右侧大纲栏。
        * 在大纲栏中，按 `Enter` 键即可跳转到对应的标题。

    2. Tagbar（老牌经典）

        `majutsushi/tagbar` 是最经典的大纲插件。

        * **配置**：因为 Tagbar 默认对 Markdown 的支持一般，你需要在 `.vimrc` 中加上以下配置来让它识别 Markdown 标题：
        ```vim
        let g:tagbar_type_markdown = {
            \ 'ctagstype' : 'markdown',
            \ 'kinds' : [
                \ 'h:Heading_L1',
                \ 'i:Heading_L2',
                \ 'k:Heading_L3',
                \ 'u:Heading_L4',
                \ 'm:Heading_L5',
                \ 'p:Heading_L6'
            \ ]
        \ }
        ```

        **使用方法**：输入 `:TagbarToggle` 打开或关闭。

    ---

    方案三：使用 `coc-outline`（如果你用了 Coc.nvim）

    如果你是用 `coc.nvim` 作为你的补全和 LSP 引擎，你完全不需要额外安装基于 Ctags 的插件。

    * **安装命令**：在 Vim 中运行 `:CocInstall coc-outline`
    * **使用方法**：输入 `:CocOutline`，就会在侧边栏弹出一个非常精准的 Markdown 目录树。

    ---

    方案四：纯原生 Vim 技巧（无需任何插件）

    如果你在没有配置插件的服务器上，想要临时瞅一眼导航，可以利用 Vim 的**折叠（Folding）**或**搜索定位**功能：

    1. **利用折叠看大纲**

        * 输入 `:set filetype=markdown`（确保开启了 Markdown 语法）。
        * 输入 `:set foldmethod=expr`（或者有些配置里直接按 `zi` 打开折叠）。
        * 这时候所有的章节会被折叠起来，只显示标题，你可以通过 `zo`（打开折叠）和 `zc`（关闭折叠）来当成目录浏览。

    2. **快捷搜索标题**：

        * 输入 `/[#]\+ ` 然后按回车，配合 `n` 或 `N` 在各个标题之间快速跳跃。

    ---

    **总结建议**：如果你追求颜值和流畅度，首推 **`Vista.vim`**；如果你喜欢直接在文章里塞一个目录，用 **`vim-markdown-toc`**。你平时有在使用什么 Vim 插件管理器（如 vim-plug）吗？

### autocmd

* 详细讲讲 vim 中 autocmd 的用法

    在 Vim 中，`autocmd`（Automation Command，自动命令）是一个极其强大的功能。它允许你让 Vim **在特定的事件发生时（如打开文件、保存文件、切换 buffer 等），自动执行指定的命令**。

    简单来说，它就是 Vim 里的“触发器”和“自动化脚本”。

    1. `autocmd` 的基本语法

        `autocmd` 的核心结构非常固定，标准语法如下：

        ```vim
        autocmd [group] {event} {pattern} [nested] {cmd}
        ```

        * **`group`（可选）**：自动命令组。强烈建议使用（后面会详细解释原因）。

        * **`event`（必填）**：触发事件。例如 `BufRead`（读取文件）、`BufWritePost`（保存文件后）。可以同时指定多个事件，用逗号隔开。

        * **`pattern`（必填）**：文件匹配模式。例如 `*.py`（所有 Python 文件）、`*`（所有文件）。

        * **`nested`（可选）**：允许嵌套。默认情况下，`autocmd` 执行的命令不会触发其他 `autocmd`，加上 `nested` 可以打破这个限制。

        * **`cmd`（必填）**：要执行的 Vim 命令或函数。

    2. 核心三要素详解

        1. 常用事件 (`event`)

            Vim 提供了上百个事件，最常用的可以分为以下几类：

            | 事件分类 | 常用事件 | 触发时机 |
            | --- | --- | --- |
            | **读写文件** | `BufReadPre` / `BufReadPost` | 开始读取 buffer 前 / 读完文件进入 buffer 后 |
            |  | `BufWritePre` / `BufWritePost` | 开始写入（保存）文件前 / 保存成功后 |
            | **Buffer/窗口** | `BufNewFile` | 创建新文件时 |
            |  | `BufEnter` / `BufLeave` | 进入 / 离开某个 buffer 时 |
            |  | `WinEnter` / `WinLeave` | 进入 / 离开某个窗口时 |
            | **Vim 状态** | `VimEnter` / `VimLeave` | Vim 启动完成 / Vim 退出前 |
            |  | `FileType` | 当 Vim 检测到文件类型时（非常常用） |
            | **模式切换** | `InsertEnter` / `InsertLeave` | 进入插入模式 / 退出插入模式（回到普通模式）时 |

        2. 匹配模式 (`pattern`)

            用来限制哪些文件会触发该命令。支持通配符：

            * `*.py`：所有以 `.py` 结尾的文件。
            * `*`：所有文件。
            * `/tmp/*`：`/tmp` 目录下的所有文件。
            * `ginit.vim,*.html`：匹配 `ginit.vim` 或者所有 HTML 文件。

        3. 执行命令 (`cmd`)

            可以是任何合法的 Vim Ex 命令（即你在底行模式下输入的命令）。

            * `setlocal shiftwidth=4`
            * `call MyFunction()`
            * `echo "Welcome!"`

    3. 经典应用场景示例

        光看语法比较抽象，我们来看几个日常绝对用得上的实际例子。

        **示例 1：根据文件类型自动设置缩进**

        写 Python 需要 4 空格缩进，而写前端（HTML/JS）可能想要 2 空格缩进：

        ```vim
        autocmd FileType python setlocal expandtab shiftwidth=4 tabstop=4
        autocmd FileType html,javascript setlocal expandtab shiftwidth=2 tabstop=2
        ```

        **示例 2：保存时自动去除行尾多余空格**

        在编写代码时，行尾如果残留空白字符会很不优雅。可以让 Vim 在保存文件前自动清理：

        ```vim
        autocmd BufWritePre *.py,*.js,*.sh %s/\s\+$//e
        ```

        > **注**：末尾的 `e` 标志表示“如果没找到匹配的空格也不报错”。

        **示例 3：自动恢复光标上次离开的位置**

        当你重新打开一个文件时，默认光标在第一行。用下面这行命令，可以让光标自动跳到你上次关闭该文件时的位置：

        ```vim
        autocmd BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
        ```

        **示例 4：动态高亮当前行**

        只有在当前窗口时才高亮显示光标所在行，切换到其他分割窗口时自动取消高亮，避免视觉干扰：

        ```vim
        autocmd WinEnter * set local cursorline
        autocmd WinLeave * set local nocursorline
        ```

    4. 避坑指南：必须使用 `augroup`

        如果你直接把 `autocmd ...` 写在 `.vimrc` 中，会遇到一个大坑：**每次你重载 `.vimrc`（例如执行 `:source ~/.vimrc`），Vim 并不会覆盖之前的自动命令，而是会**再注册一遍****。

        久而久之，同一个事件后面会叠加几十个相同的命令，导致 Vim 运行变慢甚至卡死。

        **正确的做法：使用 `augroup`（自动命令组）**

        把你的 `autocmd` 包裹在一个组里，并在每次定义前用 `autocmd!` 清空这个组：

        ```vim
        " 定义一个名为 MyAutoCommands 的组
        augroup MyAutoCommands
            " 清空当前组内之前定义的所有自动命令
            autocmd!
            
            " 在这里放你所有的 autocmd
            autocmd BufWritePost *.sh !chmod +x %
            autocmd InsertLeave * set nocursorline
            autocmd InsertEnter * set cursorline
        augroup END
        ```

        > **原理解析**：`autocmd!` 的作用是清除当前组里已经存在的命令。这样无论你 `source ~/.vimrc` 多少次，组里永远只有最新的一套命令，干净高效。

    5. 高级技巧：管理与调试

        当你的配置越来越复杂，你可能需要查看当前生效了哪些自动命令。

        * **查看所有自动命令**：在底行模式输入 `:autocmd`。
        * **查看某个特定事件的命令**：输入 `:autocmd BufWritePost`。
        * **查看某个文件类型相关的命令**：输入 `:autocmd FileType`。
        * **手动触发某个事件**：如果你想强制触发某个事件，可以使用 `doautocmd`。例如 `:doautocmd BufReadPost`。

    掌握了 `autocmd` 和 `augroup`，你就可以根据自己的工作流，把 Vim 打造成一个完全自动化的专属 IDE。

* 详细讲讲 autocmd FileType c,cpp setlocal complete+=s 这行 vim 命令的含义

    这行 Vim 命令的作用是：**当你在 Vim 中打开或编辑 C 或 C++ 源代码文件时，自动将“书签/特殊文件（Syntax/Spell/Tags等相关的扩展补全源，这里特指 thesaurus 或 spell，稍后详解）”加入到你的代码自动补全路径中。**

    为了让你彻底明白，我们把它拆解成三个核心部分来详细分析：

    1. 触发机制：`autocmd FileType c,cpp`

        * **`autocmd` (Automated Command)**：自动命令。它是 Vim 的一种事件监听机制，允许你在特定的事件发生时，自动执行指定的命令。
        * **`FileType`**：这是 Vim 的一个内置**事件（Event）**。当 Vim 检测到当前缓冲区的的文件类型（File Type）被设置或改变时，就会触发这个事件。
        * **`c,cpp`**：这是**模式匹配（Pattern）**。意思是这个自动命令只对文件类型为 `c`（C语言）或 `cpp`（C++语言）的文件生效。

        **连起来看**：只要你打开一个 C 或 C++ 文件（或者手动执行了 `:set ft=c`），Vim 就会捕捉到这个信号，准备执行后面的命令。

    2. 作用范围：`setlocal`

        * **`set` 与 `setlocal` 的区别**：
        * `set` 会修改全局设置，影响你当前打开的以及未来打开的**所有**标签页和窗口。
        * `setlocal` 只对当前活动缓冲区（Buffer）生效。


        * **为什么要用 `setlocal**`：因为你只想在写 C/C++ 代码时改变补全行为。如果你打开了一个纯文本文件（Markdown 或 `.txt`），你显然不需要 C/C++ 相关的特定补全逻辑。`setlocal` 保证了这种隔离性。

    3. 核心功能：`complete+=s`

        这是这行命令的核心，用来控制 Vim 的 **插入模式补全（Ins-completion）** 路径。

        * **`complete` 选项**：这是一个由单字母字母组成的字符串列表，决定了当你在插入模式下按下 `Ctrl + N` 或 `Ctrl + P` 时，Vim 会去**哪些地方**寻找匹配的单词来当做补全候选。
        * **`+=`**：表示追加。在现有的补全源列表里，再添加一个新的源，而不是覆盖它们。
        * **`s`**：这是关键的字母代号。在 Vim 的 `complete` 选项中：
        * **`s` 代表 `thesaurus`（同义词词典）文件。** * *注：如果你开启了拼写检查（`spell`），在某些特定配置下它也会关联拼写检查文件，但官方定义中 `s` 主要对应 `thesaurus` 选项指定的文件。*

    **总结与实际效果**

    将它们拼装在一起，这行命令的完整逻辑是：

    > **“每当 Vim 识别到当前文件是 C 或 C++ 代码时，就针对当前文件窗口，在自动补全的搜索路径中追加‘同义词/词典文件（s）’。”**

    **补充避坑指南（为什么你可能感觉它没生效？）**

    很多人的配置文件里抄了这一行，但发现并没有什么变化。这是因为 **`s` 依赖于 `thesaurus` 选项的配置**。

    如果你没有在 `.vimrc` 中指定你的词典文件路径，例如：

    ```vim
    set thesaurus+=~/.vim/dict/c_keywords.txt
    ```

    那么追加了 `s`，Vim 也不知道去哪里找词，这个配置就会形同虚设。通常，C/C++ 开发者更常用的是 `complete+=k`（使用 `dictionary` 字典文件补全关键字）或者依赖基于 LSP（Language Server Protocol）的现代补全插件（如 coc.nvim, nvim-cmp 等）。

* autocmd FileType c,cpp setlocal dictionary+=~/.vim/cpp.dict 详细解释下这行命令的作用

    这行命令是 Vim 中非常典型且优雅的**自动化配置（Automation）**。它的核心作用是：**“当你用 Vim 打开 C 或 C++ 源代码文件时，自动为你单独加载一个专属于 C/C++ 的自定义字典文件。”**

    为了让你彻底理解，我们将这行命令拆解成 5 个部分来逐一剖析：

    ```vim
    autocmd FileType c,cpp setlocal dictionary+=~/.vim/cpp.dict
    ───┬─── ───┬──── ─┬─ ───┬──── ─────┬────  ────────┬───────
       ①       ②      ③     ④          ⑤              ⑥

    ```

    ---

    1. `autocmd` (Auto-command) —— 自动命令触发器

        这是 Vim 的自动事件监听机制。它告诉 Vim：“你在后台帮我盯着，一旦发生某个特定的**事件**（Event），就自动帮我执行后面指定的命令”。

    2. `FileType` —— 监听的事件类型

        这是 Vim 众多内置事件中的一种。每当 Vim 打开一个文件，并根据文件后缀或内容识别出该文件的“文件类型（Filetype）”时，就会触发这个事件。例如，打开 `.c` 文件时触发 `FileType c`，打开 `.py` 文件时触发 `FileType python`。

    3. `c,cpp` —— 事件的过滤条件（模式匹配）

        这限制了事件的生效范围。只有当识别出的文件类型是 `c`（C 语言）**或者** `cpp`（C++ 语言）时，后面的命令才会执行。如果打开的是 `python` 或 `markdown` 文件，这行命令直接静默跳过，不会干扰其他语言。

    4. `setlocal` —— 局部设置

        这是这行命令里非常关键的细节！

        * 如果用 `set`，那这个设置就是全局（Global）的，会影响你当前和后续打开的所有标签页和文件。
        * 用 **`setlocal`**，意味着后面修改的配置**只对当前这一个缓冲区（Buffer/文件窗口）生效**。

        > **为什么要用 `setlocal`？**
        > 这样可以避免“污染”。你想想，如果你随后又用同一个 Vim 实例打开了一个 `.txt` 文本，你肯定不希望在写普通情书或日记时，一按补全弹出来的全是 `printf`、`std::cout` 这种 C++ 字典里的词。`setlocal` 确保了 `cpp.dict` 只在 C/C++ 代码窗口里发挥作用。

    5. `dictionary+=` —— 追加字典选项

        * `dictionary` 是 Vim 内置的控制字典路径的选项。
        * **`+=`** 代表**追加（Append）**，而不是覆盖。如果你的 Vim 之前已经配置过一个通用系统字典，`+=` 就会把 C++ 字典加在后面。这样，你既能补全 C++ 术语，也能补全普通的英文单词。

    6. `~/.vim/cpp.dict` —— 你的自定义字典路径

        这是你存放 C/C++ 专属词汇的纯文本文件。

        * `~` 代表当前系统用户的家目录（Home Directory）。
        * 你需要自己在这个文件里一行一个地写上你想补全的词。例如，你可以把常用的第三方库函数、复杂的类名、结构体名写进去：
        ```text
        std::vector
        std::shared_ptr
        printf
        memcpy
        uint32_t

        ```

    最终运行效果

    当你把这行命令写入 `~/.vimrc` 后，每次你用 Vim 打开一个 `main.cpp` 文件，Vim 就会悄悄在后台执行 `setlocal dictionary+=~/.vim/cpp.dict`。

    当你在这个文件里写代码，输入 `std::v` 并按下 **`Ctrl + x` ➡️ `Ctrl + k**` 时，Vim 就会立刻去翻 `~/.vim/cpp.dict`，在下拉菜单里精准为你奉上 `std::vector`。这就实现了针对不同编程语言的**个性化、定制化文本补全**。

* autocmd FileType c,cpp setlocal iskeyword+=# 那这个呢，iskeyword+=# 是什么含义？

    这行命令在写 C 和 C++ 代码时非常实用。它的核心作用是：**把 `#` 字符也当成“单词的一部分”，而不是一个分隔符。**

    在默认情况下，Vim 会把 `#include` 拆分成两部分：一个符号 `#` 和一个单词 `include`。加了这行命令后，Vim 就会认为 **`#include` 或 `#define` 本身就是一个完整的单词**。

    我们同样用拆解的方式，带你看看它在实际写代码和补全时会带来怎样的奇妙变化：

    1. 什么是 `iskeyword`？

        `iskeyword`（通常简写为 `isk`）是 Vim 内部一个非常底层且重要的配置项。它定义了：**哪些字符可以组合成一个“单词”（Word）**。

        * **默认情况下**，Vim 认为字母（`a-z`）、数字（`0-9`）和下划线（`_`）是属于单词的字符。而像 `@`、`#`、`$`、`-` 这些特殊符号，都被当成**单词之间的分隔符**（就像空格一样）。
        * **`setlocal iskeyword+=#`** 则是把 `#` 号追加（`+=`）到了当前 C/C++ 文件的单词字符白名单里。

    2. 为什么要针对 C/C++ 设置这个？（核心痛点）

        在 C 和 C++ 中，存在大量的**预处理指令**（Preprocessing Directives），它们全部是以 `#` 开头的，比如：

        * `#include`
        * `#define`
        * `#ifndef`
        * `#pragma`

        如果你**没有**加这行命令，在写代码时会遇到以下两个非常不爽的痛点：

        痛点一：无法一次性补全预处理指令

        你想输入 `#include`。你先输入了 `#inc`，然后按下 `Ctrl + n` 尝试补全。

        * **默认情况：** 因为 `#` 是分隔符，Vim 只会把 `inc` 拿去匹配，弹出的补全菜单里也是 `include`。你选中后，虽然能拼出 `#include`，但由于中间有分隔符判断，经常会卡顿或者不符合预期。
        * **加了这行命令后：** 你直接输入 `#inc` 触发补全，Vim 会把 `#inc` 作为一个整体去搜索。它可以直接精准地帮你补全出完整的 **`#include`**、**`#ifndef`**，体验非常连贯。

        痛点二：光标快捷键会把 `#` 漏掉

        在 Vim 中，有两个非常常用的跳转神技：

        * **`*` 键**：在普通模式下，按下 `*` 会在文件中向下寻找并高亮**当前光标下的单词**。
        * **`w` 键**：光标向后跳过一个单词。

        如果光标停在 `#define` 的 `#` 上：

        * **默认情况：** 按 `*` 键，Vim 只会高亮文件中所有的 `define`，而不会带上 `#`。按 `w` 键，光标会直接从 `#` 跳到 `d`。
        * **加了这行命令后：** 按 `*` 键，Vim 会把整个 `#define` 当成一个词整体高亮和搜索！按 `w` 键，光标会把 `#define` 作为一个整体直接跳过去。

    3. 连起来解释这行命令

        ```vim
        autocmd FileType c,cpp setlocal iskeyword+=#
        ```

        > **大白话翻译：**
        > “只要我打开的文件类型是 **C** 或 **C++**（`autocmd FileType c,cpp`），就在**当前文件窗口内**（`setlocal`），把 **`#` 号**追加到单词字符定义中（`iskeyword+=#`）。这样我以后在写 C/C++ 预处理指令时，无论是用 `Ctrl+n` 补全，还是用 `*` 键搜索，Vim 都会把 `#define`、`#include` 当成一整个单词来对待。”

    这是一个能显著提升 C/C++ 原生编码流畅度的小细节！

### 配色

* 修改 vim prompt 等的配色

    在 `coc.nvim` 和 `coc-pyright` 中，LSP 弹出的提示窗口（也就是你说的 prompt、补全菜单或悬浮文档窗口）的配色，实际上是由 Vim 的高亮组（Highlight Groups）控制的。

    要修改它们的颜色，你不需要去动 `coc-settings.json`，而是直接在你的 `~/.vimrc` 中通过 `highlight`（简写 `hi`）命令来覆盖默认颜色。

    以下是控制这些提示窗口最核心的几个高亮组以及修改方法：

    ---

    1. 修改代码补全菜单（Pmenu）的颜色

        当你打字时弹出的下拉补全菜单，由 Vim 内置的 `Pmenu` 系列高亮组控制。

        ```vim
        " Pmenu: 提示菜单的背景和文字颜色（这里改成了暗灰色背景，白色文字）
        hi Pmenu ctermbg=236 ctermfg=251 guibg=#2d3139 guifg=#abb2bf

        " PmenuSel: 菜单中当前选中的那一行（这里改成了蓝色背景，白色粗体字）
        hi PmenuSel ctermbg=33 ctermfg=255 cterm=bold guibg=#61afef guifg=#ffffff gui=bold

        " PmenuSbar: 提示菜单右侧滚动条的背景
        hi PmenuSbar ctermbg=235 guibg=#21252b

        " PmenuThumb: 提示菜单右侧滚动条滑块的颜色
        hi PmenuThumb ctermbg=240 guibg=#4b5263

        ```

    2. 修改悬浮文档窗口（CocFloating）的颜色

        当你按下 `K` 查看函数文档、或者代码报错弹出的悬浮提示框，由 `CocFloating` 控制。默认情况下它会继承上面的 `Pmenu` 颜色，如果你想让它和补全菜单长得不一样，可以单独定制：

        ```vim
        " CocFloating: 悬浮提示框的背景和文字
        hi CocFloating ctermbg=235 ctermfg=250 guibg=#1e222a guifg=#abb2bf

        " CocErrorFloat: 悬浮框里错误信息的文字颜色（通常设为红色）
        hi CocErrorFloat ctermfg=167 guifg=#e06c75

        " CocWarningFloat: 悬浮框里警告信息的文字颜色（通常设为黄色）
        hi CocWarningFloat ctermfg=173 guifg=#e5c07b

        ```

    3. 修改代码内报错的下划线/前缀符号

        `coc-pyright` 在检测到语法错误时，会在代码下面画红线，或者在行号左边显示 `E` 或 `W`：

        ```vim
        " 代码下方的波浪线/下划线颜色
        hi CocErrorHighlight cterm=underline gui=underline guisp=#e06c75
        hi CocWarningHighlight cterm=underline gui=underline guisp=#e5c07b

        " 行号左侧侧边栏（SignColumn）红点或 E 的颜色
        hi CocErrorSign ctermfg=167 guifg=#e06c75
        hi CocWarningSign ctermfg=173 guifg=#e5c07b

        ```

    **怎么应用这些配置？**

    1. 打开你的 `~/.vimrc`。
    2. 将上面你想要修改的 `hi ...` 代码复制进去。
    3. **关键点**：确保这些 `hi` 命令写在你的配置文件的 **`colorscheme` 命令之后**。如果写在主题前面，主题一旦加载就会把你自定义的颜色覆盖掉。

    一个标准的配置顺序大概是这样的：

    ```vim
    " 1. 先加载你的主题
    colorscheme desert  " 或者你用的其他主题，比如 joshdick/onedark.vim

    " 2. 再覆盖提示框的颜色
    hi Pmenu ctermbg=236 ctermfg=251 guibg=#2d3139 guifg=#abb2bf
    hi PmenuSel ctermbg=33 ctermfg=255 guibg=#61afef guifg=#ffffff
    ```

    > **参数说明：**
    > * `ctermbg` / `ctermfg`：终端环境下的背景色/文字色（使用 0-255 的 ANSI 颜色编号）。
    > * `guibg` / `guifg`：GUI 环境下（如 GVim）或者开启了 `termguicolors` 后的十六进制颜色（如 `#ffffff`）。
    > 
    > 

    你可以先用上面我给出的灰色/蓝色现代化配色试试看。如果你有特定想换的颜色（比如想改成黑底绿字，或者配合你现有的主题），可以告诉我你目前用的是什么 Vim 主题。

* 控制 vim 主题配色的插件

    完全能理解，手动用 `hi` 命令去一个一个对色号确实能让人抓狂，而且换个主题就全乱了。

    其实在 Vim 生态里，最正宗的解法不是用“控制配色的插件”，而是直接安装一个**自带对 `coc.nvim`（以及各大 LSP 插件）深度支持的高质量色彩主题插件**。

    现在现代化（支持真彩、支持 tree-sitter 和 LSP 提示）的主题插件数不胜数。它们不仅把代码高亮做好了，还会一并把 `Pmenu`（补全菜单）、`CocFloating`（悬浮窗）、报错下划线等配色**全部打包设计好**，开箱即用。

    这里推荐三个目前最流行、对 LSP 提示框适配极完美的现代化主题插件：

    1. One Dark (经典的 Atom/VS Code 风格)

        这是目前最不容易踩雷、对各种 LSP 弹窗适配最和谐的暗色主题，灰色背景搭配柔和的蓝、绿、红，眼睛看着非常舒服。

        * **安装方法（vim-plug）：**

            ```vim
            Plug 'joshdick/onedark.vim'
            ```

        * **启用配置（写在 `~/.vimrc`）：**

            ```vim
            " 开启 24 位真彩色支持（非常重要，否则颜色会很难看）
            if (has("termguicolors"))
              set termguicolors
            endif

            colorscheme onedark
            ```

        注：

        1. 这个配色，对于 unused variable，根本看不清

    2. Gruvbox (复古极客、温暖舒适的黄褐色调)

        如果你喜欢复古、带一点极客范的暖色调（对眼睛刺激小），Gruvbox 是神级主题。它的 `gruvbox-material` 版本对 `coc.nvim` 的提示框做了专门的半透明和阴影质感优化。

        * **安装方法：**

            ```vim
            Plug 'sainnhe/gruvbox-material'
            ```

        * **启用配置：**

            ```vim
            if (has("termguicolors"))
              set termguicolors
            endif

            " 设置对比度：可选 'hard', 'medium' 或 'soft'
            let g:gruvbox_material_background = 'medium'
            colorscheme gruvbox-material

            ```

    3. Tokyonight (现代东京夜景，冷色调暗黑风)

        如果你喜欢偏蓝、偏紫的现代科技感冷色调，Tokyonight 是绝佳选择。它的补全菜单和提示框会用非常优雅的暗蓝色和亮青色区分开。

        * **安装方法：**

            ```vim
            Plug 'ghifarit53/tokyonight-vim'
            ```

        * **启用配置：**

            ```vim
            if (has("termguicolors"))
              set termguicolors
            endif

            let g:tokyonight_style = 'night' " 可选 'storm' 或 'night'
            colorscheme tokyonight
            ```

        注：

        1. 这个勉强还行，目前用的就是这个

    **避坑提示：一定要开启 `termguicolors`**

    换了这些现代主题后，如果发现提示框还是原汁原味的“高亮红”或“亮瞎眼黄”，通常是因为没有开启真彩支持。

    请务必确保你的终端（如 iTerm2, Alacritty, WSL2 等）支持真彩，并在 `.vimrc` 中加上这三行：

    ```vim
    if (has("termguicolors"))
      set termguicolors
    endif

    ```

    换上这些主题后，`coc-pyright` 的所有弹出框和报错提示都会自动融入整个主题的画风，你再也不需要手动去调一行 `hi` 颜色了。

### vim 基础与常识

* vim 的三种模式与切换

    vim is a modal editor, and has 3 modes:

    1. If the bottom of the screen displays the filename or is blank, you are is normal mode.

    2. If you are in insert mode, the indicator displays `--INSERT--`.

    3. if you are in visual mode, the indicator shows `--VISUAL--`.

    enter inserting mode: type `i`

    back to command mode: press `<Esc>` key.

* Vim 缓冲区

    1. 什么是缓冲区？

        缓冲区（Buffer） 是 Vim 中内存中的文本副本，代表一个打开的文件。它不等同于窗口或标签页。

        关键理解：

        * 缓冲区是文件的内存表示

        * 一个文件可以对应多个缓冲区（但通常不建议）

        * 缓冲区可以未命名（新建未保存的文件）

        * 缓冲区可以在无窗口的情况下存在

    2. 缓冲区的状态

        缓冲区有以下几种状态：

        ```vim
        " 查看缓冲区状态
        :ls    " 或 :buffers :files
        ```

        状态标志说明：

        * `a` - 激活（active）：在当前窗口中显示

        * `h` - 隐藏（hidden）：已加载但不在任何窗口显示

        * `%` - 当前缓冲区

        * `#` - 交替缓冲区（使用 Ctrl-^ 切换的缓冲区）

        * `+` - 已修改

        * `-` - 不可修改（只读模式）

        * `=` - 只读缓冲区

        * `x` - 有读取错误的缓冲区

        * `u` - 未列出的缓冲区

    3. 基本操作命令

        创建/打开缓冲区：

        ```vim
        :e file.txt      " 在新缓冲区打开文件
        :enew           " 创建新的空缓冲区
        :sp file.txt    " 水平分割窗口并打开缓冲区
        :vsp file.txt   " 垂直分割窗口并打开缓冲区
        ```

        缓冲区导航：

        ```vim
        :bn              " 下一个缓冲区
        :bp              " 上一个缓冲区
        :bf              " 第一个缓冲区
        :bl              " 最后一个缓冲区
        :b#              " 切换到交替缓冲区
        Ctrl-^           " 快速切换交替缓冲区
        ```

        按编号/名称切换：

        ```vim
        :b 2             " 切换到2号缓冲区
        :b file.txt      " 切换到包含该文件名的缓冲区
        :b <Tab>         " 补全缓冲区名称
        ```

        关闭缓冲区：

        ```vim
        :bd              " 删除当前缓冲区
        :bd 2            " 删除2号缓冲区
        :bd file.txt     " 删除指定文件缓冲区
        :%bd             " 删除所有缓冲区
        :bd!             " 强制删除（不保存修改）
        ```

    4. 缓冲区列表管理

        ```vim
        " 查看缓冲区列表
        :ls              " 简短列表
        :buffers         " 完整列表
        :files           " 同:buffers

        " 只列出某些缓冲区
        :ls!             " 列出包括未列出的缓冲区
        :filter /pattern/ ls   " 过滤显示
        ```

    5. 缓冲区选项

        每个缓冲区可以有自己的本地选项：

        ```vim
        " 设置缓冲区特定选项
        :setlocal tabstop=4
        :setlocal shiftwidth=4
        :setlocal filetype=python

        " 查看缓冲区选项差异
        :setlocal

        " 缓冲区变量
        let b:my_var = "value"  " 缓冲区局部变量
        echo b:changedtick     " 修改次数计数器
        ```

    6. 实用技巧和命令

        多文件操作：

        ```vim
        " 批量保存所有修改的缓冲区
        :wa              " write all

        " 批量放弃所有修改
        :qa!             " quit all without saving
        ```

        缓冲区导航映射：

        ```vim
        " 在 ~/.vimrc 中添加
        nnoremap <leader>bn :bn<CR>
        nnoremap <leader>bp :bp<CR>
        nnoremap <leader>bd :bd<CR>
        nnoremap <leader>bl :ls<CR>
        nnoremap <leader>b# :b#<CR>
        ```

        智能缓冲区切换：

        ```vim
        " 使用 fzf.vim 插件增强
        nnoremap <C-b> :Buffers<CR>
        ```

    7. 缓冲区 vs 窗口 vs 标签页

        概念	说明	类比
        缓冲区	内存中的文件	文件本身
        窗口	查看缓冲区的视口	查看器/窗口
        标签页	窗口的集合	工作区/桌面

        ```vim
        " 关系示例
        :tabnew file.txt  " 在新标签页创建窗口显示缓冲区
        :split file.txt   " 在新窗口显示同一缓冲区
        :vsplit file.txt  " 在垂直窗口显示同一缓冲区
        ```

    8. 缓冲区相关函数（Vim Script）

        ```vim
        " 获取当前缓冲区编号
        let bufnum = bufnr('%')
        let bufnum = bufnr()       " 同 bufnr('%')

        " 获取缓冲区名称
        let name = bufname('%')
        let fullname = expand('%:p')  " 完整路径

        " 检查缓冲区是否存在
        if buflisted(bufnum)
            echo "缓冲区在列表中"
        endif

        " 获取缓冲区信息
        let info = getbufinfo(bufnum)
        echo info[0].name         " 文件名
        echo info[0].changed      " 是否修改
        echo info[0].hidden       " 是否隐藏
        echo info[0].lastused     " 最后使用时间

        " 获取所有缓冲区
        let buflist = getbufinfo({'buflisted': 1})

        " 缓冲区选项操作
        call setbufvar(bufnum, '&modifiable', 1)  " 设置选项
        let mod = getbufvar(bufnum, '&modified')  " 获取选项

        " 创建新缓冲区
        let new_buf = bufadd('newfile.txt')
        call bufload(new_buf)
        ```

    9. 实用配置示例

        自动保存会话（包含缓冲区）：

        ```vim
        " ~/.vimrc
        set sessionoptions+=buffers  " 保存会话时包括缓冲区
        autocmd VimLeave * mksession! ~/.vim/session.vim
        autocmd VimEnter * source ~/.vim/session.vim
        ```

        缓冲区关闭时自动删除隐藏缓冲区：

        ```vim
        function! DeleteHiddenBuffers()
            let tpbl=[]
            call map(range(1, tabpagenr('$')), 'extend(tpbl, tabpagebuflist(v:val))')
            for buf in filter(range(1, bufnr('$')), 'bufexists(v:val) && index(tpbl, v:val)==-1')
                silent execute 'bwipeout' buf
            endfor
        endfunction
        command! BDH call DeleteHiddenBuffers()
        ```

        缓冲区切换增强：

        ```vim
        " 只显示已修改的缓冲区
        function! ListModifiedBuffers()
            let modified = []
            for buf in range(1, bufnr('$'))
                if getbufvar(buf, '&modified')
                    call add(modified, bufname(buf) . ' (' . buf . ')')
                endif
            endfor
            echo "已修改的缓冲区:"
            for name in modified
                echo name
            endfor
        endfunction
        ```

    10. 常见问题解决

        问题1：缓冲区太多难以管理

        ```vim
        " 只保留活动缓冲区
        :bufdo bd | tabonly  " 危险！会丢失未保存内容

        " 更好的方式：只保存重要缓冲区
        :set hidden          " 允许隐藏已修改的缓冲区
        ```

        问题2：忘记保存哪个缓冲区

        ```vim
        " 显示所有已修改缓冲区
        :ls +  " 只显示已修改的缓冲区

        " 自动保存会话
        :mksession! mysession.vim
        ```

        问题3：缓冲区编号混乱

        ```vim
        " 按使用频率排序显示
        :ls u  " 按上次使用时间排序

        " 使用插件如：
        " - fzf.vim
        " - bufexplorer
        " - miniBufExpl
        ```

    11. 最佳实践

        使用 :set hidden：允许隐藏已修改的缓冲区而不强制保存

        合理使用窗口：不要为每个缓冲区开新窗口

        定期清理：关闭不再需要的缓冲区

        使用缓冲区局部变量：避免全局污染

        结合会话管理：保存工作状态

    12. 插件推荐

        fzf.vim：模糊查找缓冲区

        vim-bufferline：在状态栏显示缓冲区

        bufexplorer：专用缓冲区管理器

        vim-startify：更好的启动界面，显示最近缓冲区

    理解缓冲区是掌握 Vim 高效编辑的关键。缓冲区让 Vim 能够同时处理多个文件而不需要多个编辑器实例。

    注：

    1. 缓冲区即打开的文件的内容。

* vim 和 neovim 有什么区别？

    Vim 和 Neovim 都是功能强大的模态文本编辑器，它们同根同源，但如今在设计哲学、目标和生态系统上已经有了显著的区别。

    可以把它们的关系理解为：Neovim 是 Vim 的一个雄心勃勃的、面向未来的分支。

    下面我们从几个维度来详细对比它们的区别。

    核心总结（一句话概括）

    * Vim: 稳定、经典、保守。追求的是在几乎所有 Unix-like 系统上开箱即用的可靠性和一致性。

    * Neovim: 现代、激进、可扩展。追求的是提供一个更易于开发和集成的现代化编辑器核心，尤其注重异步处理、嵌入性和配置体验。

    详细对比

    | 特性维度 | Vim | Neovim |
    | - | - | - |
    | 诞生与目标 | 源自更古老的 Vi，目标是提供一个强大的、普遍存在的文本编辑器。 | 2014年从 Vim 分支出来，目标是重构 Vim，解决其遗留问题，为现代开发需求而设计。 |
    | 架构与核心 | 单体架构，许多功能紧密耦合在核心中。 | 更加模块化的架构，将很多功能（如终端仿真）作为独立的、可维护的组件实现。 |
    | 异步任务 | 后期（Vim 8.0+）才引入了有限的异步任务支持（:job， :channel）。 | 从一开始就内置了强大的异步处理能力。这使得插件可以执行耗时操作（如 LSP、文件搜索）而不阻塞编辑器界面。 |
    | 配置与插件 | 使用 VimScript 作为主要配置和插件语言。 | 首选 Lua，同时完全兼容 VimScript。Lua 是一种更快、更现代、更易嵌入的脚本语言，极大地提升了插件性能和开发体验。 |
    | 终端模拟 | 内置了一个基础的终端模拟器（:term）。 | 内置了一个功能更完整、集成度更高的终端模拟器，可以很好地与键绑定和颜色主题配合。 |
    | GUI 实现 | GUI（如 gVim）是核心的一部分，与 TUI 代码紧密相关。 | 核心只提供 TUI。GUI 是通过 RPC API 实现的独立应用程序（如 Neovide, Nvy, Goneovim 等），这带来了更大的灵活性和现代化特性。 |
    | LSP 支持 | 需要通过插件（如 coc.nvim, ALE, vim-lsp）来集成 LSP。 | 内置了 LSP 客户端，可以通过 :LspInfo, :LspStart 等命令和简单的 Lua 配置直接使用 LSP 功能，无需额外插件。 |
    | 默认配置 | 非常保守，几乎为零配置，旨在保持与 Vi 的兼容性。 | 提供了一些更合理的默认设置（如启用鼠标、系统剪贴板集成、相对行号等），开箱体验更好。 |
    | 社区与生态 | 历史悠久，拥有海量的插件和庞大的用户群，是 Unix 系统的标准配备。 | 非常活跃和快速增长的社区，尤其是在 Lua 插件和现代化工具链集成方面。许多新潮、高性能的插件都优先或仅支持 Neovim。 |
    | 项目管理 | 由 Bram Moolenaar 个人主导，发展相对缓慢和保守。 | 开源社区驱动，开发流程更开放（GitHub Issues, PRs），发展速度更快，更愿意接纳新特性。 |

    **深入解析关键区别**

    1. 异步处理：这是最根本的区别之一

        * Vim 在很长一段时间里是单线程的。如果一个插件需要执行一个耗时操作（比如代码补全、语法检查），整个编辑器界面就会被“冻结”，直到操作完成。Vim 8 之后加入了异步支持，但 Neovim 的设计是从底层就为异步而生的。

        * Neovim 的异步能力使得像 Language Server Protocol (LSP)、模糊查找器 (Telescope.nvim) 这样的现代工具能够流畅运行，这是其体验远超传统 Vim 的关键。

    2. 配置语言：VimScript vs Lua

        * VimScript 是 Vim 的专属语言，语法有些怪异，性能也一般。虽然功能强大，但学习曲线较陡。

        * Neovim 将 Lua 提升为一级公民。Lua 是一种轻量级、高效、易学的语言。现在绝大部分 Neovim 的新插件都用 Lua 编写，配置起来更直观，执行速度也更快。你甚至可以用 Lua 完全重写你的 init.vim（Neovim 的配置文件）为 init.lua。

    3. API 优先与嵌入性

        * Neovim 引入了 msgpack-RPC API，这意味着任何能理解该协议的语言（Python, JavaScript, Go 等）都可以与 Neovim 核心通信。

        * 这带来了两个巨大优势：

            * 嵌入式：你可以将 Neovim 作为一个文本编辑组件嵌入到其他应用程序中（如 IDE、浏览器、笔记软件）。

            * GUI 分离：如上所述，GUI 可以成为独立的、高度定制化的前端，而核心只负责文本编辑逻辑。

    **如何选择？**

    * 你应该选择 Vim，如果：

        * 你主要在服务器环境下工作，需要保证编辑器在所有系统上都一致且可用。

        * 你讨厌变化，现有的 Vim 配置和工作流已经完美满足你的所有需求。

        * 你对 Lua 不感兴趣，也不想重构你已经用了多年的 VimScript 配置。

        * 稳定性和保守性对你来说是第一位的。

    * 你应该选择 Neovim，如果：

        * 你是一个追求效率和现代开发体验的用户。

        * 你希望获得开箱即用的 LSP、DAP 等现代工具链的无缝集成。

        * 你喜欢 Lua 或者愿意学习它来获得更好的配置和插件体验。

        * 你希望编辑器有一个更活跃、更快速的迭代周期和充满活力的社区。

        * 你可能会想尝试那些酷炫的、基于 RPC 的 GUI（如 Neovide）。

    **结论**

    * Neovim 并不是要“杀死” Vim，而是作为其一个成功的“进化”分支。它解决了 Vim 在架构上的一些历史包袱，为编辑器的未来开辟了新的可能性。

    * 对于新用户来说，Neovim 很可能是更好的起点，因为它默认配置更友好，现代插件生态更繁荣，学习资源（尤其是关于 Lua 和 LSP 的）也更贴近当前的需求。

    * 对于老 Vim 用户，迁移到 Neovim 的成本很低，因为 Neovim 几乎 100% 兼容 Vim 的配置和插件。你可以很容易地尝试一下，看看它的现代化特性是否值得你切换。事实上，Neovim 的许多创新（如异步、终端、包管理）也反过来促进了 Vim 自身的进化。

* vim 和 vim-common 有什么区别

    简单来说，`vim` 和 `vim-common` 的关系是 **“执行程序”** 与 **“支持资源”** 的关系。在 Cygwin 或 Linux 包管理器中，它们的分工如下：

    1. vim (主程序包)

        这个包包含的是 **二进制可执行文件**。

        * **核心内容**：主要是 `vim.exe`。
        * **作用**：它是你运行编辑器时真正启动的那个“引擎”。
        * **特点**：它体积较小，但如果没有 `vim-common` 提供的资源，它甚至无法正常启动或显示语法高亮。

    2. vim-common (公共资源包)

        这个包包含的是 **所有 Vim 版本共用的非执行文件**。

        * **核心内容**：
        * **帮助文档** (Runtime help files)
        * **语法高亮脚本** (Syntax files)
        * **配置文件模板** (如全局的 `vimrc` 示例)
        * **语言翻译** (各国的菜单转换和提示)
        * **工具辅助** (如 `xxd` 这种用来查看十六进制的独立小程序)


        * **作用**：为不同的 Vim 变体（比如控制台版的 `vim`、图形界面的 `gvim`、精简版的 `vim-tiny`）提供通用的支持文件。

    总结

    | 包名 | 包含什么 | 相当于 |
    | --- | --- | --- |
    | **vim** | 编辑器的主程序 (vim.exe) | 汽车的**发动机** |
    | **vim-common** | 帮助文档、语法高亮、皮肤、插件框架 | 汽车的**内饰、仪表盘和说明书** |

    **建议**：在 Cygwin 中，你应该**两个都安装**。通常当你选择安装 `vim` 时，Cygwin 的安装程序会自动勾选 `vim-common` 作为依赖项。你之所以遇到之前的路径问题，就是因为你只调用了 Windows 的“发动机”，而没有使用 Cygwin 这一套完整的“车架”。

#### vim 的不同模式

* 命令行模式（Command-line Mode）

    进入方式：

    * 在 Normal 模式下按 :（冒号）

    * 按 / 或 ? 进行搜索时

    * 按 : 后出现的状态行显示 : 提示符

    主要功能：

    * 执行 Ex 命令：`:w`（保存）、`:q`（退出）、`:s/old/new`（替换）

    * 执行 Vim 脚本：`:echo "Hello"`

    * 搜索：`/pattern` 或 `?pattern`

    * 调用外部命令：`:!ls`（执行 shell 命令）

    * 设置选项：`:set number`、`:set hlsearch`

    特点：

    * 光标移动到屏幕底部

    * 输入命令后按 Enter 执行

    * 按 Ctrl-C 或 Esc 可取消并返回 Normal 模式

    * 可以输入多行命令（使用 `\` 续行）

* Normal 模式（普通模式）

    进入方式：

    * Vim 启动时的默认模式

    * 从其他模式按 Esc 返回

    主要功能：

    * 导航文本：h j k l、w b、gg G

    * 编辑操作：d（删除）、y（复制）、p（粘贴）

    * 文本对象操作：`diw`（删除单词）、`ci"`（修改引号内内容）

    * 窗口操作：Ctrl-w + 方向键

    * 执行普通模式命令：`.`（重复上次操作）

    特点：

    * 主要工作模式

    * 按键直接执行命令，不显示在屏幕底部

    * 所有操作都是即时生效的

### 配置与信息

* vim 取消行号的方法

    `:set nonu`

    `:set nu!`

* vim 开启语法高亮

    ```vim
    syntax on
    ```

* vim 的`:source xxx.vim`是在当前环境中执行`vim`脚本，之前定义的函数会被保留。

* vim 启动后加载 Vimscript

    ```vim
    " 加载指定的 vimscript 文件
    :source ~/test_script.vim

    " 或者简写
    :so ~/test_script.vim
    ```

* `vim -u`：启动时额外加载配置文件

    example:

    `vim -u ~/test_vimrc.vim`

    完全不加载默认配置，只加载指定文件:

    `vim -u ~/test_vimrc.vim -N`

    最小化启动（不加载任何配置）:

    `vim -u NONE -N`

* 详细讲讲 vim `:source %` 的作用

    在 Vim 中，`:source %` 是一个非常强大且高频使用的命令。简单来说，它的作用是：**重新加载并执行当前正在编辑的文件中的所有 Vim 脚本命令。**

    为了让你透彻理解，我们把它拆解开来看：

    1. 语法拆解

        * `:`：进入命令行模式。
        * `source`（可缩写为 `so`）：这是 Vim 的内置命令，用于读取指定文件并像在命令行输入一样逐行执行其中的指令。
        * `%`：这是 Vim 中的一个特殊符号，代表**当前活动窗口的文件路径**。

    2. 核心作用：即时生效

        通常当你修改了 Vim 的配置文件（如 `.vimrc` 或 `init.vim`）后，更改不会自动生效。你有两种选择：

        1. 重启 Vim（比较麻烦）。
        2. 输入 `:source %`（**瞬间生效**）。

        **常见的应用场景：**

        * **更新配置**：修改了快捷键映射（map）、修改了配色方案或插件设置。
        * **脚本开发**：如果你正在编写一个 Vim 脚本（.vim 或 .lua），你可以边写边 `:so %` 来测试脚本运行结果，而无需离开编辑器。
        * **批量执行命令**：如果你在一个普通文本文件中写了一堆 Vim 命令（比如一系列复杂的替换操作），直接 source 这个文件就能批量执行它们。

    3. 工作原理

        当你执行 `:source %` 时，Vim 会：

        1. 从磁盘（或者内存缓冲区，取决于实现）读取当前文件的内容。
        2. 将文件中的每一行当作命令来解释执行。
        3. **覆盖旧设置**：如果文件中定义了变量或函数，新的定义会覆盖旧的；如果定义了快捷键，新的映射会生效。

        > **注意：** 如果你在脚本中使用了 `append` 或某些会修改缓冲区的命令，source 该文件可能会直接修改你当前正在编辑的内容。

    4. 进阶小技巧

        **自动 Source**

        如果你经常修改 `.vimrc`，可以在 `.vimrc` 中加入这样一行，实现“保存即生效”：

        ```vim
        autocmd! BufWritePost .vimrc source %
        ```

        这行代码的意思是：每当 `.vimrc` 文件写入（保存）后，自动执行 source 命令。

        **针对特定行执行**

        如果你不想执行整个文件，只想执行选中的几行，可以先进入可视模式（Visual Mode）选中代码，然后输入：

        ```vim
        :'<,'>source
        ```

        这在调试复杂的脚本片段时非常有用。

        **错误排查**

        如果 `:source %` 报错，通常是因为文件中存在语法错误。Vim 会提示出错的行号，方便你立即修正并再次 source。

    **总结**

    `:source %` 是 Vim 玩家的“刷新键”。它让你在不中断工作流的情况下，动态地调整编辑器的行为。如果你发现改了配置没反应，执行它准没错。

* vim 中，我设置了 expand tab，但是仍想保留输入 tab 的能力，于是我做了一个映射

    `nnoremap <leader><tab> i  <esc>    " i` 后面跟的是 ctrl + v，再按 tab 键输入的字符

    但是在实际测试中发现，这样输入的 tab 还是被 expand tab 展开成了 4 个空格。该如何解决这个问题？

    这是一个在 Vim 配置中非常经典的问题。产生这个现象的原因是：虽然你在编写 `.vimrc` 时使用了 `Ctrl+v` 输出了字面量 Tab，但 Vim 在**执行**映射时，会模拟你的键盘输入。由于此时 `expandtab` (et) 选项是开启的，Vim 会认为你在插入模式下“按”了一次 Tab 键，从而自动将其转换为了空格。

    要解决这个问题，有以下几种更稳妥的方法：

    方法一：在映射中使用 `<C-V>` 字符串（推荐）

    不要在 `.vimrc` 中直接输入那个不可见的 Tab 字符，而是直接写出 `<C-V><Tab>` 的字面量。这样 Vim 在执行映射时，会先触发 `Ctrl-v`（告诉 Vim 下一个字符不进行转义或缩进处理），然后再输入 Tab。

    ```vim
    " 在 Normal 模式下按下 <leader> + Tab 插入一个真实的 Tab
    nnoremap <leader><tab> i<C-v><Tab><Esc>

    ```

    方法二：利用临时取消 `expandtab` (最保险)

    如果你担心复杂的插件或缩进设置干扰，可以在映射中临时关闭 `et`，插入完后再恢复。这种方法最为“暴力”且有效：

    ```vim
    " 使用 :setlocal noet 确保该动作不受全局 expandtab 影响
    nnoremap <leader><tab> :setlocal noet <bar> execute "normal! i\t" <bar> setlocal et<CR>

    ```

    * `<bar>` 是 Vim 脚本里的管道符 `|`，用于在一行执行多条命令。
    * `\t` 在 `execute` 命令中会被解析为真正的 Tab 字符。

    方法三：使用 `R` 寄存器或字符编码插入

    你也可以通过插入字符编码（Tab 的 ASCII 码是 9）来绕过键盘映射的逻辑：

    ```vim
    " 使用 <C-q> 或 <C-v> 配合十进制编码 009 插入
    nnoremap <leader><tab> i<C-v>009<Esc>

    ```

    为什么你之前的做法失效了？

    当你写 `nnoremap <leader><tab> i<Tab字符><esc>` 时：

    1. Vim 加载配置，识别到你要在 `i` 后面输入一个 ASCII 9 (Tab)。
    2. 当你触发映射时，Vim 运行 `i` 进入插入模式。
    3. Vim 接着把 ASCII 9 送入缓冲区。
    4. **此时 `expandtab` 拦截了这个字符**，并根据当前的 `shiftwidth` 或 `tabstop` 将其替换成了空格。

    **额外的小技巧：手动插入**

    如果你在编辑过程中临时想输入一个真 Tab，而不想改配置，只需在 **插入模式** 下直接按下：
    **`Ctrl + v`** 接着按 **`Tab`**
    这在 Vim 中是“字面量插入”的通用快捷键。

    **这些方法能解决你的问题吗？或者你需要针对特定文件类型（如 Makefile）自动禁用 expandtab？**

* vim 是否可以在启动时加载指定的 .vimrc 文件？或者启动后加载指定的 vimscript？我现在想开发 vim 插件，想有个可以快速验证效果的调试环境

    1. 启动时加载指定的 `.vimrc`

        如果你想让 Vim 忽略默认的配置文件，转而加载一个你正在测试的文件，可以使用 `-u` 参数：

        ```bash
        vim -u /path/to/your/test_vimrc
        ```

        * 注意：使用 `-u` 会跳过所有默认的初始化脚本（包括插件加载）。如果你希望在加载自定义配置的同时保持环境尽量“纯净”，这是最佳选择。

        * 加载 NONE：`vim -u NONE` 可以让你在一个完全不加载任何配置的环境下启动，适合排查是否是某个插件导致的 Bug。

    2. 启动后加载指定的 Vimscript

        如果你只是想在 Vim 启动并进入界面后立即执行一段脚本（比如初始化你的插件代码），可以使用 `-S` 参数（Source）：

        ```bash
        vim -S /path/to/your/plugin_script.vim
        ```

        这相当于在 Vim 启动后手动执行了 `:source /path/to/your/plugin_script.vim`。

    3. 开发插件时的“快速验证”工作流

        为了提高开发效率，你可以组合使用上述技巧，甚至配合 **Vim 的双窗格** 或 **实时重载**：

        * 命令行组合拳

            如果你正在开发一个插件文件夹，可以这样启动：

            ```bash
            # --cmd 会在加载 .vimrc 之前执行命令
            # -S 会在加载 .vimrc 之后执行命令
            vim --cmd "set rtp+=./my_plugin_folder" -S tests/test_case.vim
            ```

        * 使用 -i NONE 避免污染

            在调试时，你可能不希望 Vim 记录这次会话的历史记录、Marks 或寄存器。可以加上 `-i NONE`：

            ```bash
            vim -u test_vimrc -i NONE
            ```

* 禁用 vim 识别鼠标动作序列

    如果你不需要在 Vim 里用鼠标点击定位、滚动页面或调整窗口大小，可以通过以下几种方式禁用鼠标支持：

    ```vim
    :set mouse=
    ```

    **如果你只想保留一部分功能**

    有时候你可能只想关掉“鼠标移动”产生的干扰，但还想保留点击功能。虽然 Vim 的 `mouse` 选项是按“模式”划分的，但你可以尝试降低鼠标支持的粒度：

    * `set mouse=n` : 仅在普通模式（Normal mode）下开启。
    * `set mouse=v` : 仅在可视模式（Visual mode）下开启。
    * `set mouse=a`: 开启鼠标支持

    禁用后的变化：

    1. **选中文本**：你无法再通过鼠标点击在 Vim 中移动光标。
    2. **右键菜单**：鼠标右键会恢复为终端的默认功能（通常是粘贴）。
    3. **滚动**：鼠标滚轮将由终端处理（通常是滚动整个终端缓存，而不是滚动 Vim 内的文件内容）。
    4. **消除杂音**：因为 Vim 不再尝试解析终端传来的鼠标坐标序列（Escape Sequences），所以那些讨厌的 alert 提示音会彻底消失。

* 为什么 vim 在重新打开一个文件时，不能自动跳转到上次关闭的位置，必须使用命令`"才可以？

    Vim **会记录**每个文件的“上次退出位置”（mark `"）, 但是 **不会自动跳转到那里** —— 必须你自己用：

    ```
    `"
    ```

    或

    ```
    '" 
    ```

* 讲讲 vim 中 :echo expand('%:p') 这个命令的作用

    `:echo expand('%:p')` 的作用是：**打印当前编辑文件的“绝对路径”**。

    1. 命令拆解

        * `:echo`: 这是 Vim 的内置命令，用于在底部的命令行显示信息（相当于编程语言中的 `print`）。
        * `expand()`: 这是 Vim 的一个内置函数，用于对字符串进行扩展或解析。
        * `%`: 在 Vim 中，`%` 是一个特殊的符号，它代表**当前文件名**。
        * `:p`: 这是一个**修饰符（modifier）**，代表 "expand to full path"（展开为完整路径）。

            如果不指定这个参数，那么 % 只会显示文件名，而不显示文件的完整路径

     3. 常见修饰符速查表

        `expand()` 函数非常强大，除了 `:p`，还有一些常用的修饰符，它们可以针对 `%`（当前文件）或其他字符串进行操作：

        | 修饰符 | 作用 | 示例 (`/home/user/project/test.txt`) |
        | --- | --- | --- |
        | **`:p`** | **转换为绝对路径** | `/home/user/project/test.txt` |
        | **`:h`** | 获取父目录（Head） | `/home/user/project` |
        | **`:t`** | 获取文件名（Tail） | `test.txt` |
        | **`:r`** | 去掉扩展名（Root） | `/home/user/project/test` |
        | **`:e`** | 仅获取扩展名（Extension） | `txt` |

    **进阶技巧：** 你可以将它们组合使用。例如，获取当前文件所在目录的绝对路径，可以输入：
    `:echo expand('%:p:h')`

    ---

    4. 快捷操作（如果不习惯输入命令）

        如果你只是想知道路径，而不一定要通过命令行输出，还可以使用 Vim 的内置快捷键：

        * **`1 + Ctrl + g`**：在屏幕底部的状态栏显示当前文件的详细路径信息。

        * **在状态栏显示**：如果你在 `~/.vimrc` 中配置了 `set statusline=%F`，Vim 会自动在窗口底部一直显示当前文件的绝对路径。

* 讲讲这个命令的作用 :scriptnames 

    在 Vim / Neovim 中，`:scriptnames` 是一个非常实用的**调试和排查**命令。

    简单来说，它的作用是：**列出当前 Vim 会话中所有已经加载（Source）过的脚本文件（如 `.vim` 或 `.lua` 文件）。**

    当你输入 `:scriptnames` 并回车后，Vim 会输出一个按加载先后顺序排列的列表，每一行都包含一个编号和对应的文件绝对路径。

    ---

    1. 输出示例

        运行该命令后，你通常会看到类似下面的内容：

        ```text
          1: /usr/share/vim/vim90/vimrc
          2: ~/.vimrc
          3: ~/.vim/pack/plugins/start/nerdtree/plugin/nerdtree.vim
          4: ~/.vim/pack/plugins/start/lightline.vim/plugin/lightline.vim
          5: /usr/share/vim/vim90/syntax/syntax.vim
        ```

        * **开头的数字（如 `1`, `2`）**：是 Vim 给该脚本分配的唯一 ID（Script ID）。
        * **后面的路径**：是该脚本在系统中的具体位置。

    2. 这个命令有什么用？（常见应用场景）

        作为开发者或 Vim 用户，你通常在以下几种情况下会用到它：

        * **排查插件冲突或报错**：

            当 Vim 报错提示类似 `Error detected while processing 行号`，或者某个变量/函数冲突时，错误信息里经常会带有一个 `<SNR>99_` 这样的标识（其中 `99` 就是 Script ID）。你可以通过 `:scriptnames` 找到 `99` 对应的是哪个插件的脚本，从而精准定位罪魁祸首。

        * **确认插件是否成功加载**：

            刚装了一个新插件却没生效？用 `:scriptnames` 搜一下（可以用 `/` 过滤）这个插件的名字。如果列表里没有它，说明 Vim 根本没有加载这个插件的脚本。

        * **查看加载顺序（性能优化）**：

            Vim 是按顺序执行这些脚本的。列表越靠前的脚本加载越早。如果你觉得 Vim 启动变慢了，可以通过这个列表大致了解哪些初始化脚本占了先机，或者配合 `:profile` 进一步分析。

    ---

    3. 进阶小技巧（在 Neovim / 新版 Vim 中）

        如果你使用的是较新版本的 Vim 或 Neovim，直接输入 `:scriptnames` 可能会因为文件太多而刷屏。你可以配合过滤命令来使用：

        * **查找特定插件**：

            ```vim
            :filter /nerdtree/ scriptnames
            ```

            这只会列出路径中包含 "nerdtree" 的脚本。

        * **直接跳转到某个 ID 的文件**：

            在较新的版本中，你甚至可以直接输入编号来查看，例如：

            ```vim
            :scriptnames 2
            ```

            这会直接显示或打开列表中编号为 2 的脚本文件。

* `:echo tempname()`

    返回一个唯一的、未使用的临时文件的路径。

* 查看 vim 识别到的当前文件的类型：`set filetype?`

* 检查 vim 中 x11 clipboard (`+` 寄存器)是否可用

    `:echo has('clipboard')`

* 详细讲讲 `:set tagrelative` 的作用

    在 Vim 中，`:set tagrelative` 是一个非常实用的设置，特别是在处理大型项目、多目录项目，或者使用 `cscope` 等标签（tags）生成工具时。

    简单来说，它的作用是：**控制 Vim 在查找标签时，如何解析标签文件中记录的文件路径。**

    ---

    **核心作用详解**

    默认情况下（即 `set tagrelative` 开启时），当你执行跳转命令（如 `Ctrl + ]`）时，Vim 会这样处理：

    1. **标签路径解析：** 如果标签文件中记录的文件路径是**相对路径**（相对于标签文件所在的目录），Vim 会将其解释为相对于**当前标签文件所在位置**的路径。
    2. **为何重要：** 当你的项目结构包含多个目录，且你为每个目录（或子项目）生成了独立的 `tags` 文件时，这一设置能确保 Vim 准确找到对应的源文件，而不会因为你在项目的不同目录下打开 Vim 而导致跳转失败。

    ---

    **举例说明**

    假设你的项目目录结构如下：

    ```text
    /project
        /src
            main.c
            tags       (记录了 main.c 的路径为 "./main.c")
        /include
            utils.h

    ```

    **情况 A：开启 `set tagrelative` (默认行为)**

    当你从 `/project` 目录下启动 Vim，并执行 `:set tags=src/tags`：

    * 当你试图跳转到 `main.c` 中的函数时，Vim 读取 `src/tags`。
    * 因为它看到 `main.c` 是相对路径，它会去 `src/` 目录下寻找 `main.c`。
    * **结果：跳转成功。**

    **情况 B：关闭 `set tagrelative` (`set notagrelative`)**

    如果你关闭了此选项：

    * Vim 会将标签文件中的相对路径解释为**相对于你当前启动 Vim 的目录**（即 `/project`）。
    * Vim 会去 `/project/main.c` 找文件，但实际上 `main.c` 在 `/project/src/main.c`。
    * **结果：报错 "file not found" 或跳转失败。**

    ---

    **什么时候需要关注它？**

    * **多级目录项目：** 如果你习惯在项目的根目录或各个子目录中分别生成 tags，开启此选项是必须的。
    * **使用 Cscope：** `cscope` 生成的数据库通常包含相对路径，保持 `tagrelative` 开启可以避免路径解析错误。
    * **团队协作/跨环境：** 如果你的项目被复制到不同路径下，只要 `tags` 文件内部的相对结构没变，`tagrelative` 就能保证标签跳转依然有效。

    ---

    最佳实践建议

    在现代 Vim 配置中，通常建议**始终开启**此选项（在 `.vimrc` 中添加 `set tagrelative`），因为它符合大多数人的项目管理习惯。

    * **如何查看状态：** 输入 `:set tagrelative?`。如果显示 `tagrelative` 则为开启；显示 `notagrelative` 则为关闭。
    * **配合使用：** 如果你发现标签跳转经常失效，通常不仅要检查 `tagrelative`，还要检查你的 `tags` 配置路径是否正确（例如 `:set tags=./tags,tags;$HOME`，这会递归向父目录查找 tags 文件）。

* `set fillchars+=eob:\ ` 详细讲讲这个的作用

    在 Vim 中，`set fillchars+=eob:\ `（注意末尾的反斜杠后面有一个**空格**）是一个非常经典的**界面美化**技巧。

    它的核心作用是：**消除文件末尾（End of Buffer）之后的波浪号（`~`）提示符，让未写满的屏幕区域保持完全空白。**

    语法深度拆解

    这行代码看起来有点古怪，特别是最后的 `:\ `，我们来逐段拆解：

    * **`set fillchars`**：`fillchars` 是 Vim 中用来设置**填充字符**的选项。它控制着各种界面边框、折叠栏、状态栏以及文件末尾的显示符号。
    * **`+=`**：追加操作符，确保不覆盖 `fillchars` 中已有的其他填充设置（比如窗口分割线等）。
    * **`eob`**：代表 **End of Buffer**（缓冲区末尾）。这就是专门控制文件结束后的那段区域的参数。
    * **`:`**：分隔符，后面接你想用来替换 `~` 的字符。
    * **`\ `（反斜杠 + 空格）**：**这是最关键的部分。** 因为在 Vimscript 中，直接写一个空格会被系统误认为是命令结束的空格，从而报错。为了告诉 Vim “我想用**空格**来填充 eob”，必须使用反斜杠 `\` 对空格进行**转义**。

    使用单引号包裹空格，效果和`\ `相同：

    ```vim
    " 用单引号包裹空格，含义完全相同，但更不容易写错
    set fillchars+=eob:' '
    ```

    或者在 Neovim (Lua) 中，直接这样写：

    ```lua
    vim.opt.fillchars:append({ eob = " " })
    ```

* `:set fixendofline` 再详细讲讲这个的作用

    在 Vim 中，`:set fixendofline`（简写为 `:set fixeol`）是一个涉及**POSIX 标准**和**文件底层规范**的硬核选项。它的默认状态是**开启**的（`set fixendofline`）。

    它的核心作用是：**在保存文件时，自动确保文件的最后一行以一个换行符（EOL，即 `\n` 或 `\r\n`）结尾。如果发现最后一行没有换行符，Vim 会自动帮你加上。**

    它与 `endofline` 选项的关系（重点防坑）

    Vim 中还有一个相关的状态叫 `[no]endofline`（简写 `[no]eol`）。它们两者的分工是这样的：

    * **`endofline`（只读状态/指示灯）**：当 Vim 打开一个文件时，会自动检测末尾有没有换行符。如果有，它就自动设为 `eol`；如果没有，它就自动设为 `noeol`（同时你会在 Vim 底部看到 `[noeol]` 的提示）。**你不需要手动去改这个选项。**

    * **`fixendofline`（行为开关）**：这是你用来控制 Vim 行为的。

        * 如果 `fixendofline` 是开启的，不管 `endofline` 初始检测到的是什么，保存时一律强行加上换行符。

        * 如果 `fixendofline` 被你关闭了（`set nofixeol`），保存时 Vim 就会去检查 `endofline` 的状态。如果原文件是 `noeol`，那就以“没有换行符”的状态保存。

* `set scrollopt=hor,ver,jump` 详细讲讲这行命令的作用

    这行命令 `set scrollopt=hor,ver,jump` 是用来**控制 Vim 在“同步滚动”（Scroll Binding）时的行为和同步范围**的。

    当你在 Vim 中同时打开多个窗口（比如分屏对比两个文件 `:vsplit`），并且开启了同步滚动（`:set scrollbind`）时，这行配置将决定它们如何“步调一致”地移动。

    下面为你逐一拆解这三个核心参数 `hor`、`ver` 和 `jump` 的具体作用：

    ---

    核心参数拆解

    1. `hor` (Horizontal - 水平同步)

        * **作用**：开启**水平方向**的同步滚动。
        * **具体表现**：当你在左边窗口向右滚动屏幕（比如向右看长代码行）时，右边分屏的窗口也会**同步向右滚动**。
        * **应用场景**：对比两份长行代码，或者看左右并排的表格/日志数据时非常有用。如果不加 `hor`，左边往右滚，右边一动不动，对比就错位了。

    2. `ver` (Vertical - 垂直同步)

        * **作用**：开启**垂直方向**的同步滚动。
        * **具体表现**：你在当前窗口往下翻页，另一个绑定的窗口也会**同步往下翻页**。
        * **地位**：这是最核心、最常用的参数。Vim 默认就包含这个参数，用来保证两个文件在行数上保持对齐。

    3. `jump` (Jump - 跳转同步)

        * 作用：控制**非滚动类型的“大跳跃”命令**是否同步。

        * 具体表现：当你执行某些会让光标和屏幕大幅度移动的操作时（例如：使用 `gg` 跳到文件头、`G` 跳到文件尾、或者通过 `/#include` 进行搜索跳转），**另一个窗口也会立刻“闪现”到对应比例的位置**。

        * 对比区别：

            * 如果**没有** `jump`：只有你用 `j/k`、`Ctrl-F/B` 这种**正常滚动**屏幕的操作时，另一个窗口才会跟着走。如果你突然 `gg` 飞到第一行，另一个窗口会原地不动，直到你再次开始正常滚动，它才会“猛拉”过去对齐。

            * 如果**加上** `jump`：不论你是滚过去的，还是瞬移过去的，另一个窗口都会死死死跟住你，保持绝对同步。

    常用场景：如何配合使用？

    光设置 `scrollopt` 是不会看到效果的，它通常和 `scrollbind`（滚动绑定）配合使用。

    经典工作流：手动代码对比 (Diff Mode 模拟)

    如果你没有使用 Vim 自带的 `vimdiff`，而是自己打开了两个窗口，想要让它们完美同步：

    ```vim
    " 1. 设置同步规则：垂直、水平、大跳转全都要同步
    :set scrollopt=hor,ver,jump

    " 2. 在第一个窗口开启绑定
    :set scrollbind

    " 3. 切换到第二个窗口（Ctrl-w w），也开启绑定
    :set scrollbind
    ```

    此时，无论你是上下滚、左右滚，还是直接 `G` 跳到文件尾，两个窗口都会像黏在一起一样绝对同步。

    ---

    补充小贴士

    * **默认值**：Vim 的默认设置通常是 `set scrollopt=ver,jump`（默认水平不同步）。加上 `hor` 可以应对更极端的宽屏对比场景。

    * **快速恢复**：如果你对比完了，想解除绑定，只需要在各自窗口运行 `:set noscrollbind` 即可。

* vim 中，详细讲讲 set display+=lastline 这行代码的作用

    在 Vim 中，`set display+=lastline` 是一个非常实用且能极大提升阅读体验的设置。

    简单来说，它的核心作用是：**当某一行文本太长，在当前窗口的最后一行显示不全时，尽可能多地显示该行的部分内容，而不是用一堆 `@` 符号直接把这一行隐藏掉。**

    为了让你彻底明白它的作用，我们可以对比一下设置前后的巨大差异：

    1. 默认情况（没有设置 `lastline`）

        当你开启了自动换行（`set wrap`），并且屏幕的最后一行放不下某段超长文本的剩余部分时，Vim 的默认行为是：

        * **现象**：整个最后一行会被一串高亮的 `@` 符号填满。
        * **原因**：Vim 认为“既然这一行放不下完整的剩余内容，那我就干脆不显示这一行的任何内容”。
        * **痛点**：这会导致屏幕底部出现大片空白（被 `@` 占满），浪费了显示空间，且你在滚动浏览时会感觉文本突然“断档”了。

    2. 开启后（`set display+=lastline`）

        当你把 `lastline` 加到 `display` 选项中后：

        * **现象**：Vim 会**尽可能多地显示**这一行的文字，直到填满屏幕的最后一个字符。如果最后还剩几个字符实在装不下了，它只会在最右下角显示三个点 `...`（或者显示 `@`），提示你后面还有内容。
        * **优点**：最大化利用屏幕空间，阅读长文本、段落或者代码时更加连贯，不会被突如其来的满行 `@` 符号打断思路。

    语法拆解

    * `+=`： 这是 Vimscript 中的追加操作符。因为 `display` 选项可以同时包含多个值（用逗号隔开），使用 `+=` 可以确保你在添加 `lastline` 的同时，不会覆盖掉 `display` 中原本已经存在的其他设置（比如 `uhex` 等）。

* vim config file: `~/.vimrc`

* vim 禁用终端响铃（Bell）

    ```vim
    set noerrorbells   " 禁用错误提示音
    set visualbell     " 使用视觉提示代替声音
    set t_vb=          " 清空视觉提示（彻底静音）
    ```

#### tab 处理

* vim 设置 tab 只对特定文件生效

    ```vim
    autocmd FileType python setlocal expandtab tabstop=4 shiftwidth=4
    autocmd FileType javascript setlocal expandtab tabstop=2 shiftwidth=2
    ```

* vim 将已经存在的文件中的 tab 转换为空格

    ```vim
    " 转换整个文件
    :%retab!

    " 只转换选中的行（先进入可视模式选择）
    :'<,'>retab!
    ```

* ai 对 softtabstop 的解释

    **softtabstop 的工作原理**

    * 场景 1：`softtabstop=4，expandtab=on`
    
        ```vim
        set softtabstop=4
        set expandtab
        ```

        按一次 Tab → 插入 4 个空格，光标移动 4 个字符

        按一次 Backspace → 删除 4 个空格，光标向左移动 4 个字符

    * 场景 2：`softtabstop=4，expandtab=off`

        ```vim
        set softtabstop=4
        set noexpandtab
        ```

        按一次 Tab：

        * 如果光标位置到下一个 tabstop 的距离 ≥ 4 → 插入 Tab 字符

        * 否则 → 插入空格补足到下一个 tabstop

    * 场景 3：`softtabstop=0（默认值）`

        ```vim
        set softtabstop=0
        ```

        Tab/Backspace 的行为完全由 `tabstop` 控制

        按 Tab 会直接跳到下一个 tabstop 边界

    使用 `:set list` 查看空格（显示为 `.`）和 Tab（显示为 `^I`）

    重要提示

    * softtabstop 只在 expandtab 开启时效果最明显

    * 如果 softtabstop > tabstop，Vim 会使用 tabstop 的值

    * 大多数现代项目中，三个值设置为相同是最佳实践

* vim 中的 `softtabstop`

    它控制按 Tab 键或 Backspace 键时光标移动的宽度。

    首先，只打开`:set softtabstop=4`，`tabstop`采用默认值`8`，不打开`expandtab`时，行为如下：

    ```
    hello, world
    ```

    光标在`h`前面，按 tab，插入 4 个空格：

    ```
        hello, world
    ```

    此时再按一下 tab，神奇的事情发生了，`h`前的 4 个空格被删掉，替换成了一个位宽为 8 的 tab 字符：

    ```
    	hello, world
    ```

    后面以此类推，交替插入空格和 tab：

    ```
    [  tab   ][....]hello, world
    [  tab   ][  tab   ]hello, world
    [  tab   ][  tab   ][....]hello, world
    ...
    ```

    按退格（backspace）时，这个顺序正好反过来：如果有完整的 tab，那么把 tab 拆成 8 个空格，然后再删掉 4 个空格；如果有 4 个空格，那么直接删掉 4 个空格。

    这个功能似乎没什么用，因为 tab 字符几乎总是会出现。对于代码，我们只需要空格，对于 makefile，我们只需要 tab。这种一会 tab 一会空格的功能，对两者都不适用。

    但是这个功能对于退格比较有用。假如我们只设置`set tabstop=4`，`set expandtab`，那么按 tab 时插入 4 个空格，但是按退格（backspace）时，只能一个一个地删空格。如果这个时候结合`set softtabstop=4`，那么前面的功能不变，退格可以一次删除 4 个空格。我们可以使用 tab 键和 backspace 键高效地控制缩进，非常方便。

* vim 将 tab 转换为 4 个空格

    ```vim
    " 将 Tab 转换为空格
    set expandtab
    " 设置 Tab 宽度为 4 个空格
    set tabstop=4
    set shiftwidth=4
    set softtabstop=4
    ```

    各选项说明：

    * expandtab：输入 Tab 时插入空格

    * tabstop：一个 Tab 显示的宽度（字符数）

    * shiftwidth：自动缩进使用的宽度

    * softtabstop：按 Tab/Backspace 时光标移动的宽度

    * 临时转换当前文件:

        ```vim
        :set expandtab
        :%retab!
        ```

        * `%retab!`会将文件中所有 Tab 转换为空格

    * 只转换特定行:

        ```vim
        :10,20retab  " 转换第10-20行
        ```

    * 文件格式配置（针对特定文件类型）:

        ```vim
        autocmd FileType python setlocal expandtab tabstop=4 shiftwidth=4
        autocmd FileType javascript setlocal expandtab tabstop=2 shiftwidth=2
        ```

    * 检查当前设置:

        ```vim
        :set expandtab? tabstop? shiftwidth? softtabstop?
        ```

    * 反向转换（空格转Tab）:

        ```vim
        :set noexpandtab
        :%retab!
        ```

    * 在打开文件时自动转换:

        ```vim
        autocmd BufRead * set expandtab | %retab!
        ```

    建议： 在团队项目中，建议使用统一的 .editorconfig 文件来保证代码风格一致。

### windows

* windows 下禁止 vim 生成 `xxx.un~` 文件和 `xxx~` 文件

    ```vim
    " 禁用备份文件（以 ~ 结尾的文件）
    set nobackup
    set nowritebackup

    " 禁用撤销文件（以 .un~ 结尾的文件）
    set noundofile
    ```

    * vim 生成的几种文件简介
    
        * 备份文件 `xxx~`： 写文件之前，先把原文件复制一份

            触发时机:

            * 执行 :w

            * backup 或 writebackup 开启时

            作用:

            * 防止写文件过程中崩溃 / 断电 / 磁盘错误

            * 写失败时，原内容还在 filename~

            写成功后：

            * backup 开启 → filename~ 会保留

            * 只开 writebackup → 写完就删

            相关配置：

            ```vim
            :set backup        " 是否保留 ~ 文件
            :set writebackup  " 是否写前临时备份
            :set nobackup
            :set nowritebackup
            ```

        * 交换文件 `xxx.swp`: 编辑过程中实时保存修改，用于崩溃恢复

            打开文件时立即创建

            作用:

            * 崩溃恢复（vim -r filename）

            * 防止同一文件被多个 Vim 实例同时修改

            特点:

            * 实时保存编辑状态

            * 正常退出 Vim 会自动删除

            * 非正常退出会残留

            看到它通常意味着:

            * 上次 Vim 崩了

            * 或该文件正在被另一个 Vim 打开

            相关配置：

            ```vim
            :set swapfile
            :set noswapfile
            :set directory?   " swap 文件存放目录
            ```

            * `.filename.swo` / `.swn` / `.swx` —— swap 冲突序号

                当 .swp 已存在：

                Vim 会尝试 .swo、.swn、.swx

        * 撤销文件 `xxx.un~`： 撤销历史记录（持久化撤销, 重启 Vim 后还可以执行`u`命令）

            需要启用 `undofile`功能，这个文件才能被创建。

            相关配置：

            ```vim
            :set undofile
            :set undodir?
            ```

    * 其它配置

        ```vim
        set backupdir=~/.vim/backup//  " 备份到特定目录
        set backupskip=/tmp/*,/private/tmp/*  " 跳过某些目录的备份

        set undodir=~/.vim/undo//
        set directory=~/.vim/swap//

        " 撤销历史（.un~文件）
        set undofile          " 持久化撤销历史到磁盘
        set undolevels=1000   " 内存中保留1000次撤销

        " 2. 配置定时清理
        autocmd VimLeave * !del /Q Z:\vim-backup\*
        " 退出时自动清理内存备份
        autocmd VimLeavePre * call CleanOldBackups(30) " 保留30天

        " 备份文件扩展名
        set backupext=.bak
        ```

    * 如果你只想对某些文件类型禁用，可以在 _vimrc 中添加：

        ```vim
        " 对特定目录禁用备份
        autocmd BufWritePre /path/to/directory/* set nobackup nowritebackup

        " 或者对特定文件类型禁用
        autocmd FileType txt,md set noundofile
        ```

    * 设置备份目录到特定位置，而不是当前目录：

        ```vim
        " 将备份文件集中到特定目录
        set backupdir=C:\vim_backups
        set directory=C:\vim_backups
        set undodir=C:\vim_undo

        " 如果目录不存在则创建
        if !isdirectory("C:\\vim_backups")
            silent !mkdir "C:\vim_backups"
        endif
        ```

    * 完全禁用所有备份相关功能

        ```vim
        " 一次性禁用所有备份相关文件
        set nobackup       " 不创建备份文件（*.~）
        set nowritebackup  " 写入时不创建备份
        set noswapfile     " 不创建.swp交换文件
        set noundofile     " 不创建.un~撤销文件
        ```

    * 折中方案

        ```vim
        " 将备份文件集中到固定目录，而不是污染当前目录
        set backupdir=~/.vim/backup//
        set directory=~/.vim/swap//
        set undodir=~/.vim/undo//

        " 确保目录存在
        if !isdirectory(expand("~/.vim/backup"))
            silent !mkdir ~/.vim/backup
        endif
        if !isdirectory(expand("~/.vim/undo"))
            silent !mkdir ~/.vim/undo
        endif
        ```

    * 自动清理脚本

        ```vim
        " 定期清理旧备份
        function! CleanOldBackups(days)
            let backup_dir = expand('~/.vim/backup')
            if isdirectory(backup_dir)
                " Windows 示例
                silent !forfiles /p backup_dir /s /m * /d -%a% /c "cmd /c del @path"
            endif
        endfunction

        autocmd VimLeave * call CleanOldBackups(7)  " 保留7天
        ```

* windows 版本的 gvim 无法识别正斜杠表示的路径，比如`/path/to/file`，只能识别反斜杠表示的路径，比如`d:\path\to\file`

    要解决这个问题，需要在 cygwin 自己的软件库里再装一次 vim。

    可以使用`which vim`查看当前 vim 的版本。

* vim 打开文件后，`"cygwin_log.md" [dos] 11L, 1144B` 全是乱码，怎么解决

    在 Vim 中看到乱码且标记为 `[dos]`，通常是因为文件的 **编码格式（Encoding）** 与 Vim 当前的解析方式不匹配（例如文件是 `GBK` 编码，但 Vim 用 `UTF-8` 打开）。

    你可以尝试在 Vim 中依次执行以下命令来解决：

    1. 强制切换编码查看（最常用）

        在不关闭文件的情况下，尝试用 `GBK`（中文 Windows 常用的编码）重新载入：

        ```vim
        :e ++enc=gbk
        ```

        如果还是乱码，可以尝试：

        ```vim
        :e ++enc=utf-8
        ```

    2. 永久配置解决

        为了让 Vim 自动识别编码，建议在你的 `~/.vimrc` 中添加以下代码。这样 Vim 会按顺序尝试解析，直到不乱码为止：

        ```vim
        set encoding=utf-8
        set fileencodings=ucs-bom,utf-8,gbk,gb18030,big5,latin1
        ```

    3. 彻底修复换行符（针对 [dos] 标记）

        如果你想把这个文件彻底转为 Unix 格式（去掉 `^M` 并移除 `[dos]` 标记），请执行：

        ```vim
        :set ff=unix
        :w
        ```

    -----

    **为什么会出现乱码？**

      * **[dos]**: 表示文件使用 Windows 的 `CRLF` 换行符。
      * **乱码**: 主要是因为 Cygwin 环境下的 Vim 默认通常是 `UTF-8`，而你在 Windows 下创建的 `.md` 文件很可能是 `GBK` (ANSI) 编码。

    为了让你的跨平台开发环境更稳健，我们可以进一步优化你的配置。

    为了确保你的 Vim 环境在处理不同来源的文件时都能保持一致，你可以尝试以下操作。

      * [一键配置支持多编码自动识别的.vimrc](http://googleusercontent.com/interactive_content/0)
      * [批量检测并转换当前目录下文件的编码格式](http://googleusercontent.com/interactive_content/1)
      * [配置Vim状态栏显示当前的编码和文件格式](http://googleusercontent.com/interactive_content/2)

    * `:e ++enc=gbk`
        ~
        E37: No write since last change (add ! to override) 

        这是因为你在执行 `:e ++enc=gbk`（以 GBK 编码重新加载）之前，当前的缓冲区（Buffer）已经有了未保存的修改。Vim 为了防止新编码载入覆盖掉你还没保存的内容，所以弹出了这个保护性警告。

        你可以根据你的需求选择以下操作：

        1. 如果你想放弃当前的修改，直接以 GBK 查看

            如果你刚才只是乱敲了几下或者不在意当前的改动，直接在命令后面加 `!` 强制执行：

            ```vim
            :e! ++enc=gbk
            ```

        2. 如果你想先保存当前修改，再切换编码

            这种操作比较少见，因为如果当前是乱码，保存可能会把乱码永久写入文件。但如果你确定要保存：

            ```vim
            :w
            :e ++enc=gbk
            ```

        3. 如果你想把当前内容“转换”并保存为 GBK

            如果你已经在终端里修正了内容（现在不是乱码了），想把它另存为 Windows 常用的 GBK 格式：

            ```vim
            :set fileencoding=gbk
            :w
            ```

        -----

        **补充小贴士：**
        如果你发现每次打开文件都要手动输入这些命令，说明你的 `~/.vimrc` 还没有生效。请确保你已经按照上一步的方法，用 `dos2unix ~/.vimrc` 修复了它的换行符问题，否则 Vim 启动时加载配置失败，就不会自动帮你识别编码了。

        为了确保你的编辑器配置能彻底告别这些编码和权限烦恼，你可以尝试以下操作。

        * [一键生成兼容 Cygwin 和 Windows 的 .vimrc 配置模板](http://googleusercontent.com/interactive_content/0)
        * [编写自动修复项目下所有文件编码和换行符的脚本](http://googleusercontent.com/interactive_content/1)

    * 试了下，还是乱码。看上去好像是 windows 的 terminal 的显示编码问题

        这确实非常有道理。如果 `vim` 内部切换了编码（`:e ++enc=gbk`）画面依然没有变化，那么问题通常就不在文件本身，而在**终端（Terminal）的渲染层**。

        Cygwin 默认使用 **Mintty** 作为终端，它对编码的处理非常严格。如果你的终端环境被锁定在 UTF-8，而你强行让 Vim 输出 GBK 字符流，终端就会因为无法识别这些字节而显示乱码（通常是方块或问号）。

        **检查与修复步骤**

        1. 修改 Mintty 终端设置（最直接）

            1.  在 Cygwin 窗口内部 **点击鼠标右键**。
            2.  选择 **Options...**。
            3.  点击左侧菜单的 **Text**。
            4.  查看右侧底部的 **Character set** 下拉框。
                * 如果当前是 `UTF-8`，请尝试改为 **`GBK`** 或 **`CP936`**（针对中文 Windows 兼容）。
                * 点击 **Apply**，观察 Vim 中的文字是否恢复正常。

        2. 检查环境变量 `LANG`

            在终端输入以下命令：

            ```bash
            echo $LANG
            ```

            * 如果输出是 `zh_CN.UTF-8`，终端会强制以 UTF-8 渲染。
            * 你可以临时切换环境编码后再打开文件：
                ```bash
                export LANG=zh_CN.GBK
                vim cygwin_log.md
                ```

        3. 终极建议：统一使用 UTF-8

            在 Windows 和 Cygwin 混用的环境下，**GBK 是万恶之源**。
            最长治久安的方法是利用工具把文件永久转为 UTF-8，并将终端固定在 UTF-8：

            1.  **转换文件：** `iconv -f gbk -t utf-8 cygwin_log.md > new_log.md`
            2.  **设置终端：** 将 Mintty 的 Character set 永远设为 `UTF-8`。
            3.  **设置 Vim：** 确保 `~/.vimrc` 中有 `set encoding=utf-8`。

        为了彻底解决你的跨平台字符显示问题，你可以尝试以下操作。

* cygwin 中 windows 版 vim 与 cygwin vim 的区别 

    ```bash
    wsdlh@DESKTOP-ETVUKF3 ~
    $ which vim
    /cygdrive/c/Program Files/Vim/vim91/vim
    ```

    找到了！问题的根源正如你所料：你正在调用 **Windows 原生版本的 Vim**，而不是 Cygwin 版本的 Vim。

    当你执行 `vim ~/.bashrc` 时，Cygwin 的 Bash 会将 `~/.bashrc` 展开为 `/home/wsdlh/.bashrc`。然而，这个 Windows 原生 Vim 无法理解 Cygwin 的虚拟路径（`/home/...`），它会尝试按照 Windows 的逻辑去解析，结果就把路径误解成了 `\home\wsdlh\.bashrc`，并因为找不到这个 Windows 路径而报错。

    解决方案

    1. 安装 Cygwin 版 Vim（推荐）

        这是最根本的解决办法。Cygwin 版的 Vim 能完美识别 `/home/` 路径。

        * 重新运行 Cygwin 的 `setup-x86_64.exe`。
        * 在软件包选择界面搜索 `vim`。
        * 确保安装了 `vim` 和 `vim-common` 及其相关组件。

    2. 使用 cygpath 进行实时转换

        如果你想继续使用这个 Windows 版 Vim，必须在传参时转换路径：

        ```bash
        /cygdrive/c/Program+Files/Vim/vim91/vim $(cygpath -m ~/.bashrc)
        ```

    3. 设置别名 (Alias)

        为了方便，你可以通过设置别名来自动完成转换。在当前终端输入（成功进入后再写入 `.bashrc`）：

        ```bash
        alias vim='/cygdrive/c/Program\ Files/Vim/vim91/vim $(cygpath -m "$@")'
        ```

        注意：由于 Windows 版 Vim 和 Cygwin 的终端交互（如退格键、颜色）可能存在兼容性问题，长远来看还是建议使用方案 1。*

    为了帮你彻底理顺开发环境，我们可以优化工具链的调用顺序。

    * 修改 PATH 变量以优先使用 Cygwin 原生工具
    * 编写自动转换路径的 Vim 启动脚本别名
    * 检查 Cygwin 环境中是否已安装原生 Vim

* 为什么我在 windows terminal 中运行 vim 打开文件时，鼠标在 terminal 上移动，会一直响 alert 的声音？

    这种现象通常是因为 **Windows Terminal 发送的鼠标移动序列（Mouse Reporting）被 Vim 识别为无效输入**，从而触发了 Vim 的“错误提示音”（Bell）。

    * 方法一：在 Vim 中禁用错误响铃（最推荐）

        如果你不需要 Vim 在操作错误时发出声音，可以直接关闭它。这不会影响鼠标功能，只是让它“闭嘴”。

        在你的 `_vimrc` (Windows) 或 `.vimrc` (Linux/WSL) 中添加：

        ```vim
        set noerrorbells  " 关闭错误响铃
        set visualbell  " 使用视觉闪烁代替声音（如果觉得闪烁也烦，配合下一行使用）
        set t_vb=  " 清空视觉响铃的序列，彻底静音
        ```

    * 方法二：正确配置 Vim 的鼠标模式

        如果你的 Vim 没有正确识别 Windows Terminal 的鼠标协议，也会导致报错。尝试在配置文件中添加：

        ```vim
        set mouse=a
        if has("mouse_sgr")
            set ttymouse=sgr
        endif
        ```

        * **SGR 模式**是现代终端（如 Windows Terminal）推荐的鼠标协议，能更好地处理大尺寸窗口的坐标，减少误码产生的报警声。

    **为什么会发生这种情况？**

    当你移动鼠标时，终端会发送类似 `^[[<0;45;12M` 的编码。如果：

    1. Vim 处于**普通模式**。
    2. 鼠标支持**未完全开启**或**协议不匹配**。
    Vim 会把这串编码解析为一系列乱码按键。由于这些按键在当前状态下是无效操作，Vim 就会触发 `beep` 警告。

    **建议方案：**

    通常结合 **方法一** 和 **方法二** 能获得最佳体验，既能正常使用鼠标点击/滚动，又不会被刺耳的声音打扰。

    注：

    1. 这个回答里的方法都未验证过。

* Windows Terminal 里使用 Vim

    既然你明确是在 **Windows Terminal** 中使用 Vim，那么问题的根源 **99% 不在 Vim 的配置，而是在 Windows Terminal 的全局快捷键拦截**。

    Windows Terminal 默认将 `Ctrl + V` 绑定到了“粘贴”操作。由于终端程序的优先级高于它内部运行的命令行程序（Vim），按键在到达 Vim 之前就被终端拦截并执行了。

    以下是彻底解决的步骤：

    1. 在 Windows Terminal 设置中取消绑定（最直接）

        这是最推荐的做法，让终端把 `Ctrl + V` 的控制权还给 Vim。

        1. 打开 Windows Terminal，点击标题栏的下拉箭头，选择 **“设置” (Settings)**（快捷键 `Ctrl + ,`）。
        2. 在左侧菜单栏最下方，点击 **“操作” (Actions)**。
        3. 在右侧的快捷键列表中找到 **“粘贴” (Paste)**。
        4. 你会看到它绑定了 `Ctrl + V`。点击该条目右侧的 **“修改” (编辑图标)** 或 **“删除” (垃圾桶图标)**。
        * **建议：** 将其修改为 `Ctrl + Shift + V`。这样既符合 Linux 终端的习惯，又避开了 Vim 的快捷键。


        5. 点击右下角的 **“保存”**。

    2. 通过 `settings.json` 快速修改（进阶版）

        如果你觉得 UI 界面找起来麻烦，可以直接修改配置文件：

        1. 在设置界面左下角点击 **“打开 JSON 文件”**。
        2. 搜索 `command": "paste"`。
        3. 将对应的快捷键从 `ctrl+v` 改为 `ctrl+shift+v`，或者直接删除该键绑定对象。
        ```json
        // 修改后示例
        { "command": "paste", "keys": "ctrl+shift+v" }

        ```

    3. 如果你不想改终端设置：使用 Vim 的替代键

        如果你已经习惯了在终端里用 `Ctrl + V` 粘贴，不想改终端配置，Vim 其实专门为这种情况准备了一个“后门”：

        * 在 Vim 中按下 **`Ctrl + q`**。
        * 在 Windows 环境下，`Ctrl + q` 的功能被定义为与 `Ctrl + v` 完全一致（进入块可视模式）。这是一个内置的官方替代方案。

    4. 最后的检查：Vim 内部是否还有干扰？

        即使你解决了终端拦截问题，如果 Vim 内部加载了 `mswin.vim`，`Ctrl + V` 依然会被映射为粘贴。

        请在 Windows Terminal 的 Vim 中输入以下命令排查：

        ```vim
        :verbose map <C-v>
        ```

        * 如果显示 **"No mapping found"**：恭喜你，现在按下 `Ctrl + V` 就能进入块模式了。
        * 如果显示 **"Last set from ... mswin.vim"**：请回到我第一个回答中的方案，在 `_vimrc` 中注释掉 `source $VIMRUNTIME/mswin.vim`。

    **总结建议：**

    最舒适的方案是将 Windows Terminal 的粘贴键改为 **`Ctrl + Shift + V`**。这不仅解决了 Vim 的冲突，还能让你在终端操作中更专业。

    需要我帮你确认你的 `_vimrc` 文件里是否有残留的快捷键冲突配置吗？

* vim 在 windows 下的所有 ctrl + v 都被默认替换成了 ctrl + q

    因为 windows 下的 ctrl + v 是粘贴功能。

    不清楚 GVim 是什么情况，可能 ctrl + v 仍保持 vim 的语义。

* 还有哪些类似 linux vim 中 ctrl + v 在 windows vim 中被默认替换成 ctrl + q 的快捷键的例子？详细列出

    在 Windows 版的 Vim（或 gVim）中，由于 Windows 系统本身有一套广为人知的全局快捷键（如 Ctrl + C 复制、Ctrl + V 粘贴），为了降低新手的门槛，Vim 官方发行版通常会加载一个名为 mswin.vim 的脚本。

    这个脚本会将许多经典的 Vim 快捷键重映射为 Windows 习惯。以下是受影响最严重的几个核心快捷键对比：

    1. 核心映射对照表

        | Windows 快捷键 | 在 Windows Vim 中的功能 | 对应的原始 Vim 快捷键 | 原始 Vim 功能说明 |
        | - | - | - | - |
        | Ctrl + V | 粘贴 (Paste) | Ctrl + V | 块可视化模式 (Visual Block) |
        | Ctrl + Q | 块可视化模式 (替换方案) | 无 | 这是为了补位被占用的 Ctrl + V |
        | Ctrl + C | 复制 (Copy) | y (yank) | 复制选中的文本到剪贴板 |
        | Ctrl + X | 剪切 (Cut) | d (delete) | 删除选中内容并存入剪贴板 |
        | Ctrl + A | 全选 (Select All) | ggVG | 选中全文 |
        | Ctrl + Z | 撤销 (Undo) | u | 撤销上一步操作 |
        | Ctrl + Y | 重做 (Redo) | Ctrl + R | 恢复被撤销的操作 |
        | Ctrl + S | 保存 (Save) | :w | 写入文件 |
        | Ctrl + F | 查找 (Find) | / | 开启搜索模式 |

    2. 详细分析受影响的逻辑

        块模式切换：Ctrl + V vs Ctrl + Q

        在 Linux 下，Ctrl + V 是进入“块选择模式”的唯一王牌。但在 Windows 中，它变成了粘贴。

        * 后果： 如果你习惯了在 Linux 下用 Ctrl + V 垂直编辑多行代码，在 Windows 下你会发现它直接吐出了一堆剪贴板里的乱码。

        * 解决： 必须改用 Ctrl + Q。

        撤销与重做：Ctrl + Z / Y

        * Vim 原生： u 是撤销，Ctrl + R 是重做。Ctrl + Z 在 Linux 终端里通常是将进程挂起到后台（Suspend）。

        * Windows 映射： 遵循 Office 逻辑，Ctrl + Z 撤销，Ctrl + Y 重做。这会导致习惯 Ctrl + R 的老用户感到困惑。

        复制粘贴：Ctrl + C / X

        * Vim 原生： Ctrl + C 通常用于中断当前的命令或回到 Normal 模式（类似 Esc）。

        * Windows 映射： 变成了标准复制。这意味着如果你想通过 Ctrl + C 强行退出某个插件的搜索或提示界面，它可能会毫无反应，甚至报错。

    3. 为什么会这样？

        这是因为在 Windows 版本的安装目录下，有一个 _vimrc 文件，它通常会自动执行：

        ```Vim Script
        source $VIMRUNTIME/mswin.vim
        behave mswin
        ```

        这两行代码就是“罪魁祸首”。mswin.vim 重新定义了上述所有快捷键。

    4. 如何恢复成 Linux 习惯？

        如果你希望在 Windows 上也拥有和 Linux 一模一样的纯粹 Vim 体验，你只需要编辑你的 _vimrc 文件（通常在 C:\Users\你的用户名\_vimrc）：

        * 删除或注释掉 source $VIMRUNTIME/mswin.vim 这一行。

        * 删除或注释掉 behave mswin。

        * 保存并重启 Vim。

        这样，Ctrl + V 就会变回块选择模式，u 依然是撤销，所有 Windows 的“入侵”映射都会消失。

### 多窗口

* vim 设置一个分窗口跟随另一个分窗口滚动

    设置 Vim 分窗口同步滚动有以下几种方法：

    1. 临时同步滚动
    
        ```vim
        :set scrollbind        " 启用同步滚动
        :set noscrollbind      " 禁用同步滚动

        " 快捷命令
        :windo set scrollbind  " 所有窗口同步
        :windo set noscrollbind" 所有窗口取消同步
        ```

    2. 一键切换同步

        在 ~/.vimrc 中添加：
        
        ```vim
        " 切换同步滚动的快捷键
        nnoremap <leader>sb :set scrollbind!<CR>
        ```

    3. 自动同步特定文件类型

        ```vim
        " 比较文件时自动同步
        autocmd FileType diff set scrollbind

        " Markdown预览时同步
        autocmd FileType markdown set scrollbind
        ```

    4. 垂直和水平分屏的配置

        ```vim
        " 垂直分屏时同步滚动
        command! VsplitWithScroll 
            \ vsp | 
            \ set scrollbind |
            \ wincmd p  " 回到原窗口

        " 水平分屏时同步滚动  
        command! SplitWithScroll
            \ sp |
            \ set scrollbind |
            \ wincmd p

        " 快捷键映射
        nnoremap <leader>vs :VsplitWithScroll<CR>
        nnoremap <leader>hs :SplitWithScroll<CR>
        ```

    5. 智能同步插件

        使用插件可以获得更好的同步体验：

        * `vim-syncscroll`（轻量级）

            安装后自动启用，支持非对称分屏, 自动检测分屏并同步滚动

        * `ScrollColors`（比较时专用）

            用于比较两个文件版本时保持同步

    6. 手动同步命令

        ```vim
        " 创建一对同步的窗口
        :diffsplit file1.txt
        :vert diffsplit file2.txt
        " 这会自动启用 scrollbind

        " 保持光标位置同步
        :set cursorbind  " 光标也一起移动
        ```

    7. 实用工作流示例

        ```vim
        " 1. 比较两个文件
        :diffthis           " 在当前窗口启用比较模式
        :vsp file2.txt      " 垂直分屏
        :diffthis           " 在新窗口启用比较模式
        " 自动同步滚动！

        " 2. 查看代码和文档
        :e main.py
        :vsp
        :e documentation.md
        :set scrollbind
        " 现在滚动代码时文档也会滚动

        " 3. 编写和预览
        :e index.html
        :sp
        :e style.css
        :windo set scrollbind
        ```

    8. 注意事项

        * 同步滚动只对相同高度的内容有效

        * 如果窗口大小不同，滚动可能不完全对齐

        * 使用 `:windo diffoff` 退出比较模式时会取消同步

        * 同步滚动时，每个窗口的折叠（fold）状态是独立的

    9. 完整配置示例

        ```vim
        " ~/.vimrc
        " 快速切换同步滚动
        nnoremap <F3> :set scrollbind!<CR>:echo '同步滚动: ' . (&scrollbind ? '开启' : '关闭')<CR>

        " 比较模式自动同步
        augroup ScrollSync
            autocmd!
            autocmd BufEnter * if &diff | set scrollbind | endif
        augroup END

        " 创建同步窗口的命令
        command! -nargs=1 VSync vertical split <args> | set scrollbind | wincmd p
        command! -nargs=1 HSync split <args> | set scrollbind | wincmd p
        ```

* vim 如何高效地调整 vsplit 子窗口的大小

    在 Vim 中高效调整 `:vsplit`（或 `:split`）子窗口的大小，主要有以下几种方法，从最推荐的开始：

    1. 使用鼠标拖动

        如果你在终端或 GUI 中启用了鼠标支持，这是最直观的。

        1.  **启用鼠标**（如果还没启用）：

            ```vim
            :set mouse=a
            ```

        2.  **操作**：直接用鼠标拖动分割窗口之间的边界线。

    2. 使用键盘命令

        Vim 提供了非常快速的水平/垂直大小调整命令。

        1. 增加或减少高度（针对 `:split` 水平分割）

            *   **增加高度**：`Ctrl + w` + `+`
            *   **减少高度**：`Ctrl + w` + `-`
            *   **设定为固定高度**：`Ctrl + w` + `_` （下划线，意思是最大化当前窗口的高度，或者恢复）

        2. 增加或减少宽度（针对 `:vsplit` 垂直分割）

            *   **增加宽度**：`Ctrl + w` + `>`
            *   **减少宽度**：`Ctrl + w` + `<`
            *   **设定为固定宽度**：`Ctrl + w` + `|` （竖线，意思是最大化当前窗口的宽度）

        3. 等分所有窗口

            *   **等宽高等**：`Ctrl + w` + `=` （无论你把窗口弄得多乱，按这个就能恢复平均分配）

        **💡 高效技巧：配合数字前缀**

            你可以先按数字，再按命令，来指定调整的大小（单位通常是行或列）。

        *   **例子**：想让当前 `vsplit` 窗口变宽 20 列。

            *   按键顺序：`20` `Ctrl + w` `>`
            *   （比一直按着 `>` 数格子快得多）

    3. 使用 `:resize` 命令（适合精确输入）

        如果你在命令模式下思考，或者需要输入非常具体的尺寸。

        *   **调整高度**：

            ```vim
            :resize 20    " 把当前窗口高度设为 20 行
            :resize +5    " 增加 5 行高度
            :resize -5    " 减少 5 行高度
            ```

        *   **调整宽度**：

            ```vim
            :vertical resize 80    " 把当前窗口宽度设为 80 列
            :vertical resize +10   " 增加 10 列宽度
            :vertical resize -10   " 减少 10 列宽度
            ```

    4. 终极技巧（映射快捷键）

        如果你经常调整大小，可以把常用的调整映射成快捷键，解放手指。

        把下面这几行加到你的 `.vimrc` 或 `init.lua`（Neovim）里：

        ```vim
        " 使用 Shift + 方向键 调整窗口大小
        nnoremap <S-Up>    <C-w>+
        nnoremap <S-Down>  <C-w>-
        nnoremap <S-Left>  <C-w><
        nnoremap <S-Right> <C-w>>
        ```

        这样你就可以按住 `Shift` 再按方向键来调整当前窗口的边框了，非常符合直觉。

* vim 进入子目录导航后的常用命令

    * `-` 返回上级目录

    * Enter 打开文件或进入子目录

    * o 在新窗口中打开

    * D 删除文件/目录

    * R 重命名

    * s 切换排序方式

    * q 或 :q 关闭目录窗口

        q 好像不行，必须使用`:q`。

    * i 切换显示格式（详细信息/简略信息）

* vim 在用 :vsplit 打开新窗口时，如何在右边打开新窗口，而不是在左边打开

    `:vsplit`或`:vs`默认会在当前窗口的左侧打开新窗口。
    
    在右侧打开新窗口的几种方法：

    1. `:rightbelow vertical split`

        ```vim
        :rightbelow vertical split
        ```

        或简写为：

        ```vim
        :rightbelow vsplit
        ```

        或更短的简写：

        ```vim
        :bo vsplit
        ```

        其中 bo 是 bottom 的缩写，在这里表示"右下方"

    2. 使用方向参数

        ```vim
        :vertical split
        ```

        然后按 Ctrl+w L 将新窗口移动到右侧

    3. 配置`:vsplit`的默认行为（添加到 `.vimrc`）

        ```vim
        set splitright
        ```

    4. 使用快捷键映射

        在 `.vimrc` 中创建自定义快捷键：

        ```vim
        " 使用空格+v在右侧垂直分割
        nnoremap <space>v :rightbelow vnew<CR>

        " 或保持原有命令但强制右侧
        nnoremap <leader>v :rightbelow vsplit<CR>
        ```

* vim 在输入 :vsplit subdir/ 后，如何显示出 subdir 中的子文件？

    1. 直接运行`:vsplit subdir`或`:vsplit subdir/`，进入子文件夹，再选择

    2. 用 tab 键补全，再按 tab 键即可滚动选择

    注：

    1. `:vsplit subdir/*`

        不会在新窗口中打开 subdir 下的所有文件，会报错:

        > E77: Too many file names

    2. 即使 subdir 是符号链接，也可以正常使用 tab 补全 subdir 下子文件的路径

    3. 如果 tab 无法正常显示子文件，可能是因为缺少权限

    4. 如果 tab 无法打开子目录，需要随便输入一点东西
    
        比如`:vsplit subdir/`，此时可以输入`:vsplit subdir/*`然后按 tab，即可显示子文件夹中内容。也可以按一下右箭头，虽然屏幕上没显示内容，但是此时按 tab 也可以显示子文件夹中内容。

* vim 中，vsplit 打开两个文件后，如何只复制一个文件的多行？

    * 使用 vim 命令复制，不能使用鼠标在 terminal 中选择，否则会两个子窗口一起选

    * 在两个子窗口之间复制内容，使用寄存器或不使用寄存器都行

* vim split（分屏）

    1. 创建分屏

        ```vim
        :vsplit [文件]  " 垂直分屏（左右分割），缩写 vsp
        :split [文件]   " 水平分屏（上下分割），缩写 sp
        :new           " 新窗口打开空白缓冲区
        ```

    2. 快捷键

        ```vim
        Ctrl+w s    " 水平分割当前窗口
        Ctrl+w v    " 垂直分割当前窗口
        Ctrl+w n    " 打开一个空的新窗口
        Ctrl+w c    " 关闭当前窗口
        Ctrl+w q    " 关闭当前窗口（同 :q, Ctrl+w c）
        Ctrl+w o    " 只保留当前窗口
        ```

    3. 窗口间导航

        ```vim
        Ctrl+w h    # 切换到左边窗口
        Ctrl+w j    # 切换到下边窗口
        Ctrl+w k    # 切换到上边窗口
        Ctrl+w l    # 切换到右边窗口
        Ctrl+w w    # 循环切换窗口
        Ctrl+w t    # 切换到顶部窗口
        Ctrl+w b    # 切换到底部窗口
        ```

    4. 调整窗口大小

        ```vim
        Ctrl+w =    # 所有窗口等宽等高
        Ctrl+w +    # 增加窗口高度
        Ctrl+w -    # 减少窗口高度
        Ctrl+w >    # 增加窗口宽度
        Ctrl+w <    # 减少窗口宽度
        :resize N   # 设置高度为 N 行
        :vertical resize N  # 设置宽度为 N 列
        ```

    5. 移动窗口

        ```vim
        Ctrl+w H    # 将当前窗口移到最左
        Ctrl+w J    # 将当前窗口移到最下
        Ctrl+w K    # 将当前窗口移到最上
        Ctrl+w L    # 将当前窗口移到最右
        Ctrl+w r    # 向右/向下旋转窗口
        Ctrl+w R    # 向左/向上旋转窗口
        Ctrl+w x    # 交换当前与下一个窗口
        ```

    6. 实用技巧

        ```vim
        # 多文件操作
        :vsp file1.txt file2.txt

        # 快速比较差异
        :diffsplit filename

        # 关闭其他所有窗口
        :only
        ```

### 远程文件

* vim 的`scp://`协议打开的文件，会在保存文件时临时把文件放到`/tmp`中，当完成 scp 传输后，会马上把这个文件删掉。这样保证打开的文件只存在于内存中，不在`/tmp`中，只有传输过程中需要用到实体文件时，才会在`/tmp`中保存一下，然后马上删掉。

* 使用 vim 的 netrw

    * 在命令行中打开远程文件

        ```bash
        vim scp://username@hostname[:port]//path/to/file
        ```

        vim 会先使用 scp 把远程文件复制到本地`/tmp`目录下，然后再进行编辑。

        注：

        1. 如果`~/.ssh/config`中已经配置了`Host nickname`，那么可以直接

            `vim scp://nickname//path/to/file`

        1. 绝对路径与相对用户目录的路径区别

            example:

            绝对路径：`vim scp//user@host//home/hlc/test.txt`

            相对用户目录的路径：`vim scp://user@host/test.txt`

            第一个`/`相当于 ssh 命令里的`:`，表示用户的 home。

        1. 不可以使用冒号`:`表示用户 home。冒号`:`只能表示 remote host 的 ssh 端口。

            比如`vim scp://nickname:2222/rel_path/to/file`，表示打开`/home/<user>/rel_path/to/file`这个文件。

        1. 如果使用从根目录开始的绝对路径，那么必须使用两个`//`

            example:

            `vim scp://hlc@54//home/hlc/Documents/Projects/c_test/main.cpp`

    * 在 vim 中使用`:e`打开文件

        ```vim
        :e scp://username@hostname/path/to/file
        ```

        ```vim
        :e scp://[user@]hostname[:port]/path/to/file
        ```

        example:

        ```vim
        " 使用默认用户名（当前本地用户名）
        :e scp://remote-server/home/user/project/file.txt

        " 指定用户名
        :e scp://username@remote-server/path/to/file.txt

        " 指定端口
        :e scp://username@remote-server:2222/path/to/file

        " 绝对路径
        :e scp://user@host//home/user/file.txt

        " 相对用户home的路径
        :e scp://user@host/file.txt
        ```

    * 在 vim 中将 ssh 配置为默认协议，效率比 scp 更高

        （存疑）

        在 ~/.vimrc 中添加：

        ```conf
        let g:netrw_ftpextracmd = 'ssh'
        ```

        配置后，底层可能会这样传输文件：

        ```bash
        ssh user@host cat /path/to/file
        ```

        不配置时，底层可能这样传输文件：

        ```bash
        scp user@host:/path/to/file /tmp/vimXXXXXX
        ```

* 我使用 vim 的 netrw 打开远程文件后，如何使用 ctags？远程 host 未安装 ctags

    `netrw` 只是通过 SSH/SCP 把远程文件 **拉到本地的临时目录** 再编辑，所以 **ctags 只需要在本地生成即可**，跟远程主机上有没有 ctags 完全无关。

    方案：用 sshfs（推荐）

    如果你常编辑远程项目，直接把远程目录挂载成本地目录就更轻松了：

    ```bash
    sshfs user@host:/path/to/proj ~/remote-proj
    cd ~/remote-proj
    ctags -R .
    ```

    然后像编辑本地项目一样使用 Vim，tag、搜索都无缝工作。

* 我使用 vim 的 netrw 打开远程文件后，如何使用 ctags？远程 host 未安装 ctags

    **方案2：在 Vim 中配置自动下载并生成 tags**

    ```vim
    function! GenerateRemoteTags()
        " 获取当前文件路径
        let l:file_path = expand('%:p')
        
        " 检查是否是远程文件（netrw 路径通常包含 scp://, ftp:// 等）
        if match(l:file_path, '^scp://') != -1
            " 提取远程路径信息
            let l:remote_path = substitute(l:file_path, '^scp://[^/]*/', '', '')
            
            " 使用 rsync 下载文件到临时目录
            let l:temp_dir = '/tmp/remote_tags/'
            let l:rsync_cmd = 'rsync -avz user@remote-host:' . l:remote_path . ' ' . l:temp_dir
            
            " 执行同步
            silent! execute '!' . l:rsync_cmd
            
            " 在本地生成 tags
            silent! execute '!ctags -R -f ' . l:temp_dir . 'tags ' . l:temp_dir
            
            " 设置 tags 路径
            set tags+=/tmp/remote_tags/tags
        endif
    endfunction

    " 打开远程文件时自动调用
    autocmd BufRead scp://* call GenerateRemoteTags()
    ```


    **方案：使用 vim-gutentags**

    安装 [vim-gutentags](https://github.com/ludovicchabant/vim-gutentags) 并配置：

    ```vim
    let g:gutentags_cache_dir = '~/.cache/tags'
    let g:gutentags_modules = ['ctags']

    " 对于远程文件，设置项目根目录标记
    let g:gutentags_project_root = ['.git', '.root']
    ```


    **方案三：抛弃 Ctags，使用 LSP (Language Server Protocol)**

    在 2026 年，`ctags` 这种基于文本匹配的静态索引已经不再是唯一选择。**LSP 才是现代方案。**

    如果你的本地环境安装了对应语言的 Language Server（如 `pyright`, `gopls`, `clangd`），你可以配合 **coc.nvim** 或 Neovim 内置的 LSP 功能：

    1.  通过 `sshfs` 挂载远程代码（同方案一）。
    2.  本地 Vim 开启 LSP。
    3.  LSP 会在后台自动扫描挂载的整个目录。
    4.  **优势**：无需手动生成 `tags` 文件，且支持“跳转到定义”、“查找引用”等更智能的操作。

### command 模式下的常用命令

* help: `:help`

* 鼠标在 vim 界面上滑动时，会向 vim 发送鼠标事件序列

    禁用鼠标序列：`:set mouse=`

    `:set mouse=nv    " 仅允许在普通模式和可视模式下使用鼠标`

* 在 command 模式下执行 normal 命令

    ```vim
    :normal I* <CR>
    ```

    * `:normal` 是一个 Ex 命令（在 : 命令行中执行）

    * 它告诉 Vim："在普通模式下执行后面的按键序列"

    * I* 是传递给 :normal 的参数

    * 最后的 `<CR>` 是结束 :normal 命令，而不是在普通模式中按回车

    执行流程解析

    ```vim
    :g/^\s*\S/normal I* <CR>
    ```

    * : 进入命令行模式

    * g/^\s*\S/ 设置匹配模式

    * normal I* 对匹配的每行执行普通模式命令 I*␣

    * <CR> 执行整个 :g 命令

    这里的 <CR> 不是给 normal I* 的，而是结束 :g 命令并执行它。

* vim 如何重载当前文件？

    在 Vim 中重载当前文件（重新加载文件内容）有以下几种方法：

    1. **基本重载命令**

        重新加载当前文件（放弃所有更改）

        ```vim
        :e!
        ```

        或者

        ```vim
        :edit!
        ```

        重新加载当前文件（如果文件在外部被修改）

        ```vim
        :checktime
        ```

        这会检查所有缓冲区，如果文件在外部被修改，会提示是否重新加载

    2. **不同情况的重载**

        如果只是想刷新文件内容（当文件在外部被修改时）

        ```vim
        :edit
        ```

        这会重新加载文件，但如果有未保存的更改，Vim 会拒绝执行

        强制重新加载（放弃未保存的更改）

        ```vim
        :e!
        ```

    3. **快捷方式**

        可以在 `.vimrc` 中添加快捷键映射：

        ```vim
        " 使用 F5 重载当前文件
        nnoremap <F5> :e!<CR>

        " 使用 <Leader>r 重载当前文件
        nnoremap <leader>r :e!<CR>
        ```

    4. **相关命令**

        查看文件状态

        ```vim
        :file
        ```

        显示文件信息，包括是否被修改

        检查文件是否在外部被修改

        ```vim
        :!ls -l %
        ```

        用 shell 命令查看文件的最后修改时间

    5. **实际应用场景**

        ```vim
        " 场景1：外部程序修改了文件（如 git pull）
        :e        " 如果确定没有本地更改
        :e!       " 如果想放弃本地更改

        " 场景2：想重新读取配置文件
        :e ~/.vimrc

        " 场景3：恢复到最后保存的状态
        :e!
        ```

    6. **自动重载设置**

        如果想在文件被外部修改时自动重载，可以在 `.vimrc` 中设置：

        ```vim
        " 自动重载被外部修改的文件
        set autoread

        " 或者定期检查文件是否被外部修改
        autocmd FocusGained,BufEnter * :checktime
        ```

    常用命令总结

    | 命令 | 作用 |
    |------|------|
    | `:e` | 重新加载当前文件（如果有更改会失败） |
    | `:e!` | 强制重新加载当前文件（放弃更改） |
    | `:checktime` | 检查并重载被外部修改的文件 |
    | `:file` | 查看文件状态 |

    **最常用的是 `:e!`**，它会强制重新加载当前文件，放弃所有未保存的更改。

* 在 vim 中如何查看寄存器的内容？

    在 Vim 的命令模式中，有以下几种方法查看寄存器的内容：

    1. **使用 `:reg` 或 `:registers` 命令**

        ```vim
        :reg           " 显示所有寄存器的内容
        :reg a         " 只显示寄存器a的内容
        :reg a b c     " 显示多个指定寄存器的内容
        :reg "         " 显示默认寄存器
        :reg /         " 显示搜索寄存器
        ```

    2. **使用 `:dis` 或 `:display` 命令**

        ```vim
        :dis           " 等同于 :reg
        :display a     " 显示寄存器a的内容
        ```

    3. **在命令中使用 `<C-r>` 插入寄存器内容**

        在命令模式下，按 `Ctrl-r` 然后输入寄存器名称，可以临时插入该寄存器的内容：

        ```vim
        :echo "<C-r>a"     " 会显示寄存器a的内容
        :echo @"           " 显示默认寄存器内容
        ```

    4. **输出示例**

        执行 `:reg` 后可能会看到类似这样的输出：

        ```
        --- Registers ---
        ""   hello world
        "a   test line
        "/   search pattern
        ":   w
        "%   /path/to/file.txt
        "+   clipboard content
        ```

    这样你就可以随时查看各个寄存器中保存的内容了。

* autoindent 相关的配置

    Vim 有多种自动缩进相关的配置，可以根据不同的编程语言和需求进行设置。以下是完整的自动缩进配置指南：

    1. **基础自动缩进设置**

        `autoindent` - 基本自动缩进

        ```vim
        set autoindent      " 新行继承上一行的缩进
        ```

        - 最简单的自动缩进

        - 只继承缩进，不智能识别代码结构

        `smartindent` - 智能缩进（已过时）

        ```vim
        set smartindent     " 在 autoindent 基础上增加一些智能
        ```

        - 会识别 `{`、`}` 等字符调整缩进

        - **不推荐使用**（对现代语言支持差）

        `cindent` - C 风格缩进

        ```vim
        set cindent         " C/C++/Java 风格的智能缩进
        ```

        - 专门为 C 家族语言设计

        - 有详细的配置选项

    2. **现代推荐：`indentexpr` 系统**

        `filetype indent` - 基于文件类型的缩进

        ```vim
        " 在 .vimrc 中启用
        filetype plugin indent on
        ```

        这是**最推荐**的方式，Vim 会根据文件类型加载对应的缩进规则。

    3. **具体语言配置示例**

        ```vim
        " 通用设置
        set autoindent
        set smartindent     " 可选，某些语言可能需要
        set expandtab       " 使用空格而非Tab

        " 按语言设置缩进宽度
        autocmd FileType python setlocal shiftwidth=4 tabstop=4 softtabstop=4
        autocmd FileType javascript setlocal shiftwidth=2 tabstop=2 softtabstop=2
        autocmd FileType java setlocal shiftwidth=4 tabstop=4 cindent
        autocmd FileType cpp setlocal shiftwidth=4 tabstop=4 cindent
        autocmd FileType html setlocal shiftwidth=2 tabstop=2
        autocmd FileType yaml setlocal shiftwidth=2 tabstop=2 expandtab
        ```

    4. **C/C++ 风格缩进详细配置**

        ```vim
        " C 风格缩进详细配置
        set cindent
        set cinoptions=:0,l1,t0,g0,(0
        ```

        - `:0` - `case` 标签不额外缩进
        - `l1` - `case` 内的代码缩进 1 级
        - `t0` - 函数返回值类型不缩进
        - `g0` - C++ 作用域声明不额外缩进
        - `(0` - 换行时括号对齐

    5. **相关辅助选项**

        `copyindent` - 复制缩进样式

        ```vim
        set copyindent      " 继承上一行的缩进样式（Tab/空格混合）
        ```

        保持上一行的缩进字符类型。

        `preserveindent` - 保留缩进

        ```vim
        set preserveindent  " 修改缩进时尽量保持原有缩进样式
        ```

    6. **缩进行为控制**

        `indentkeys` - 触发重新缩进的键

        ```vim
        " 默认包含：0{,0},0),0],:,0#,!^F,o,O,e
        " 可以自定义哪些键触发重新计算缩进
        ```

        `cinkeys` - C 缩进触发键

        ```vim
        " 控制 cindent 何时重新计算缩进
        set cinkeys=0{,0},0),0],:,0#,!^F,o,O,e
        ```

    7. **快速测试缩进**

        ```vim
        " 测试当前缩进设置
        :set autoindent? smartindent? cindent?
        :set shiftwidth? tabstop? expandtab?

        " 查看当前文件使用的缩进脚本
        :scriptnames  " 查看加载的所有脚本，寻找 indent/ 目录下的
        ```

    8. **完整配置示例**

        ```vim
        " ~/.vimrc 中的完整缩进配置
        set autoindent              " 基础自动缩进
        filetype plugin indent on   " 启用文件类型检测和缩进

        " 全局默认缩进设置
        set tabstop=4
        set shiftwidth=4
        set softtabstop=4
        set expandtab               " 使用空格

        " 特定语言覆盖
        autocmd FileType python setlocal expandtab shiftwidth=4 tabstop=4
        autocmd FileType javascript,typescript,json setlocal expandtab shiftwidth=2 tabstop=2
        autocmd FileType html,css setlocal expandtab shiftwidth=2 tabstop=2
        autocmd FileType go setlocal noexpandtab shiftwidth=4 tabstop=4
        autocmd FileType make setlocal noexpandtab shiftwidth=8 tabstop=8
        autocmd FileType yaml setlocal expandtab shiftwidth=2 tabstop=2

        " C/C++ 特殊设置
        autocmd FileType c,cpp setlocal cindent
        autocmd FileType c,cpp setlocal cinoptions=:0,l1,t0,g0,N-s,(0,w1,W4
        ```

    9. **常见问题排查**

        缩进不正常？

        ```vim
        " 1. 检查当前设置
        :set ai? si? ci? sw? ts? et?

        " 2. 关闭所有缩进
        :set noai nocin nosi inde=

        " 3. 重新启用
        :set ai

        " 4. 查看文件类型
        :set ft?

        " 5. 手动指定缩进
        :set indentexpr=  " 清空缩进表达式
        ```

        覆盖系统缩进规则

        ```vim
        " 创建 ~/.vim/indent/ 目录
        " 添加自定义缩进规则，如 ~/.vim/indent/python.vim
        ```

    10. **高级技巧**

        临时禁用自动缩进

        ```vim
        :set paste        " 进入粘贴模式（禁用所有缩进）
        " 粘贴代码...
        :set nopaste      " 恢复自动缩进
        ```

        使用插件增强

        - [vim-sleuth](https://github.com/tpope/vim-sleuth) - 自动检测缩进设置
        - [vim-indent-guides](https://github.com/nathanaelkane/vim-indent-guides) - 显示缩进指引线

    总结

    **最简单的有效配置：**

    ```vim
    " 在 ~/.vimrc 中添加
    set autoindent
    filetype plugin indent on
    set expandtab
    set shiftwidth=4 tabstop=4
    ```

    这样就能获得良好的自动缩进体验，Vim 会根据文件类型使用最适合的缩进规则。

* vim 中检查当前状态下快捷键的映射

    ```vim
    :verbose map <Tab>
    :verbose vmap j
    ```

    这会告诉你 Tab 或 j 分别在哪个文件的哪一行被定义了。

* vim 如何把一个文件的换行从 \r\n 转换成 \n

    在 Vim 中处理这个问题非常简单，因为 Vim 自带了对文件格式（File Format）的识别和转换功能。

    1. 快速转换命令
        
        在 Vim 中打开文件后，依次输入以下命令：

        1.  按下 `Esc` 确保处于普通模式。
        2.  输入冒号 `:` 进入命令行模式。
        3.  执行以下转换命令：
            ```vim
            :set ff=unix
            ```
            *(注：`ff` 是 `fileformat` 的缩写，`unix` 对应的换行符就是 `\n`)*
        4.  保存并退出：
            ```vim
            :wq
            ```

    ---

    2. 如何检查当前格式？
        
        如果你想确认文件当前的换行格式，可以输入：
        ```vim
        :set ff?
        ```
        * 如果返回 `fileformat=dos`，说明是 `\r\n` (Windows)。
        * 如果返回 `fileformat=unix`，说明是 `\n` (Linux/macOS)。

    ---

    3. 为什么会出现 `^M` 符号？
        
        如果你在 Vim 中直接看到了行尾有很多 `^M`，那是因为 Vim 已经以 `unix` 模式打开了一个 `dos` 格式的文件。这时你可以直接用替换命令删掉它们：
        ```vim
        :%s/\r//g
        ```

    ---

    4. 批量转换（不进入 Vim）
        
        如果你有大量文件需要转换，在 Cygwin 的终端里直接使用 `dos2unix` 工具是最快的：

        ```bash
        # 如果没安装，先用 Cygwin setup 安装 dos2unix 包
        dos2unix filename.txt
        ```

    **小贴士**

    由于你在 Windows Terminal 中使用 Cygwin，经常会遇到 Windows 程序（如 Notepad++ 或 VS Code）创建了 `\r\n` 文件，而 Cygwin 的工具（如 `bash`, `sshfs` 等）更喜欢 `\n`。建议在 Vim 的配置中 (`~/.vimrc`) 加入以下内容，方便自动识别：
    ```vim
    set fileformats=unix,dos,mac
    ```

### 快捷键、映射、宏

* vim 录制宏

    基本操作

    1. 开始录制

        * 按 q 键开始录制

        * 然后按一个寄存器键（a-z）来指定存储位置

        * 示例：qa 表示录制到寄存器 a

    2. 执行操作

        * 执行你想要录制的所有 Vim 操作

        * 可以包括：移动、插入、删除、替换等任何命令

    3. 停止录制

        * 按 q 键停止录制

    4. 执行宏

        * @a - 执行寄存器 a 中的宏

        * @@ - 重复执行上一次执行的宏

        * 10@a - 执行 10 次寄存器 a 中的宏

    实用技巧

    * 查看录制的宏

        ```vim
        :reg a        " 查看寄存器 a 的内容
        :reg          " 查看所有寄存器
        ```

    * 编辑宏

        ```vim
        " 将宏粘贴出来编辑
        " 1. 将寄存器内容放到当前行
        "ap            " 将寄存器 a 的内容粘贴出来

        " 2. 编辑内容

        " 3. 存回寄存器
        " 删除原有内容（如："ay$），然后
        "add            " 删除当前行到寄存器 d
        " 或
        "ayy            " 复制当前行到寄存器 a
        ```

    * 常用的宏录制模式

        ```vim
        " 在多个文件上执行宏
        1. 录制宏完成对当前文件的操作
        2. :w 保存文件
        3. :bn 跳转到下一个缓冲区
        4. 停止录制
        5. 使用 :bufdo normal @a 在所有缓冲区执行
        ```

    * 错误处理

        * 如果在录制过程中出错，可以按 q 停止，然后重新录制

        * 宏会记录所有按键，包括错误和更正

    * 追加到现有宏

        ```vim
        qA  " 大写字母会追加到寄存器 a 的宏中
        ```

* vim 中宏的持久化

    1. 方法一：保存到 vimrc 中

        先使用`:reg q`查看宏寄存器 (`:echo @q`似乎也可以？)，然后在 vimrc 中添加

        ```vim
        " 将宏保存到寄存器 q
        let @q = '宏的内容'

        " 例如：
        let @q = 'IHello ^[A World!^['
        ```

        注意：`^[` 表示 `<Esc>` 键，实际输入时，需要在 insert 模式下使用`Ctrl + V, Esc`输入。

    2. 方法二：将宏保存为函数

        ```vim
        function! MyMacro()
            " 在这里编写宏的操作
            normal! IHello 
            normal! A World!
        endfunction

        " 创建命令调用
        command! MyMacro call MyMacro()

        " 或映射快捷键
        nnoremap <leader>m :call MyMacro()<CR>
        ```

    3. 方法三：保存到单独的文件

        将宏保存到文件：

        ```vim
        " 将寄存器 q 的内容保存到文件
        :let @q = '宏内容'
        :call writefile(split(@q, "\n"), '~/.vim/macro_q.txt')
        ```

        在 vimrc 中加载：

        ```vim
        if filereadable(expand('~/.vim/macro_q.txt'))
            let @q = join(readfile('~/.vim/macro_q.txt'), "\n")
        endif
        ```

    * 方法四：使用插件

        * vim-macroswap：管理多个宏

        * vim-qmacro：提供更强大的宏管理功能

    实用技巧：快速获取宏内容

    使用以下命令可以方便地将宏转换为 vimrc 可用的格式：

    ```vim
    " 在录制宏后执行：
    :put q
    " 然后选中内容，用 :'<,'>reg 查看转义后的格式
    ```

    或者使用这个函数更方便地保存宏：

    ```vim
    function! SaveMacro(reg)
        let macro_content = getreg(a:reg)
        let macro_content = substitute(macro_content, '\n', '^M', 'g')
        let macro_content = substitute(macro_content, '\"', '\\\"', 'g')
        execute 'let @'.a:reg.' = "'.macro_content.'"'
        echo 'Macro saved to register '.a:reg
    endfunction
    ```

    转义特殊字符的完整列表：

    | 按键 | vimrc 中的表示 |
    | - | - |
    | `<Esc>` | `^[` 或 `<Esc>` |
    | `<Enter>` / `<CR>` | `^M` 或 `<CR>` |
    | `<Tab>` | `^I` 或 `<Tab>` |
    | `<Space>` | 空格字符 |
    | `<BS>` | `^H` 或 `<BS>` |
    | 普通文本 | 直接写入 |

* vim 中`<tab>`与`<C-i>`等价

    ```vim
    " 将 <leader>t 映射为插入实际的 Tab 字符
    inoremap <leader>t <Tab>

    " 使用 Ctrl+i，这与 Tab 键在插入模式下效果相同
    inoremap <leader>t <C-i>
    ```

    但是如果在字符串中，需要用`\t`表示 tab 键，不能使用`<tab>`。

* vim 中的 keycodes 表示对大小写不敏感

    ```vim
    <cr>    " 和 <cr> 等效
    <esc>   " 和 <esc> 等效
    <c-a>   " 和 <c-a> 等效（都是 ctrl+a）
    <s-a>   " 和 <s-a> 等效（都是 shift+a）
    ```

    总结：

    * 键码名称本身不区分大小写：<cr> = <cr>

    * 修饰键组合不区分大小写：<C-a> = <C-A>

    * 单独的字母键区分大小写：a ≠ A

    * 推荐使用一致的大小写风格（通常用大写，如 <CR>、<Esc>）以提高可读性

* vim 中我按 2j 向下走两行，有没有类似 . 这样的快捷键可以重复 2j 的动作？

    在 Vim 中，直接使用 `.` (dot command) 是**无法**重复 `2j` 这种移动操作的。因为在 Vim 的逻辑里，`.` 只能重复修改（Change）类的操作（比如 `dd`、`i`、`cw` 等），而单纯的移动（Motion）不属于修改。

    如果你是想**快速跳转**，建议配合以下习惯：

    * 使用 `Ctrl + d` (向下半屏) 或 `Ctrl + f` (向下一整屏)。
    * 使用 `}` 跳转到下一个空行（段落跳跃）。
    * 使用 `L` 直接跳转到当前屏幕的底部。

* vim 中宏的特殊字符的含义

    ```vim
    :reg c
    Type Name Content
      c  "c   ^[<80><fd>a`<^i```^[<80><fd>a`>o```^[<80><fd>a`<jddk
    ```
    
    这个宏中的<80>，<fd>是什么意思？

    在 Vim 中看到像 `<80><fd>a` 这样的字符序列，通常是因为你在录制宏（Macro）时使用了**鼠标操作**或是一些**特殊的键盘功能键**（如方向键、翻页键等）。

    简单来说，这些是 Vim 内部对**特殊控制序列**的原始编码。

    1. 它们代表什么？

        在 Vim 的内部二进制表示中：

        * **`<80>`**：是十六进制的 `0x80`，在 ASCII 之外，它常作为特殊键编码的**前缀（Lead byte）**。
        * **`<fd>`**：是十六进制的 `0xfd`，它紧跟在 `<80>` 之后，作为一个**子类标识符**。

        当这两个字符连在一起时（`<80><fd>`），它们通常代表 **`K_SPECIAL`** 序列。接下来的字符（比如你宏里的 `a`）决定了具体捕捉到了哪个动作。

    2. 为什么会出现在你的宏里？

        最常见的原因是你在录制宏的过程中执行了以下操作：

        * **点击了鼠标**：Vim 记录了点击位置的坐标转换。
        * **调整了窗口大小**：触发了 `ScreenResize` 信号。
        * **使用了 GUI 元素**：如果你在 gVim 中操作，点击菜单栏或滚动条也会产生这些序列。
        * **按下方向键/功能键**：虽然有时会记为 `<Left>`，但在某些终端环境下会记录为原始的转义序列。

    3. 如何读懂你的宏片段？

        以你宏中的一部分为例：
        `^[<80><fd>a`

        * `^[`: 这是 `Esc` 键（Escape）。
        * `<80><fd>a`: 这在 Vim 源码中通常对应 **`KS_MOUSE`**（鼠标位置记录）。

        这说明你在按完 `Esc` 退出插入模式后，可能**不小心点了一下鼠标**，或者你的终端在切回普通模式时发送了一个包含鼠标追踪位置的信号。

    4. 建议与解决方法

        这种宏通常**很难在其他地方复现**，因为鼠标点击的位置或特定的终端序列在不同环境下可能导致宏运行报错或行为诡异。

* vim 中，我使用 nnoremap d 2j 后，按 d 反应很慢，得等大概一秒后，光标才会向下走两行，如何解决

    在 Vim 中遇到这种情况，通常是因为你配置了**其他以 `d` 开头的快捷键**。

    当你按下 `d` 时，Vim 并不知道你是只想输入 `d`，还是想输入一个更长的组合键（比如 `dd` 或 `dw`）。为了确认你的意图，Vim 会等待一段时间（默认是 1000 毫秒，即 1 秒）。如果在这个时间内你没有按下后续按键，它才会判定你只想输入 `d`，并执行 `2j`。

    要解决这个问题，你可以采用以下几种方法：

    **方法一：检查并清除冲突的快捷键（推荐）**

    你可以排查一下到底是什么快捷键在和 `d` 抢时间。在 Vim 中输入以下命令查看：

    ```vim
    :verbose nmap d
    ```

    这会列出所有在普通模式（Normal mode）下以 `d` 开头的自定义快捷键。

    > **注意：** Vim 自带的内置命令（如 `dd`、`dw`、`diw`）**也会**导致这个延迟。如果你把 `d` 映射成了别的东西，Vim 就必须等一下，看看你是不是想执行内置的剪切/删除操作。

    **方法二：更换更安全的映射前缀（最推荐的 Vim 习惯）**

    在 Vim 中，直接占用 `d`、`c`、`y` 这种高频内置功能键作为单键快捷键，很容易引起冲突和延迟。建议使用 `<leader>` 键作为前缀。

    在你的 `.vimrc` 中修改为：

    ```vim
    " 定义你的 Leader 键（比如空格键，非常顺手）
    let mapleader = "\<space>"

    " 使用 Leader + d 来向下走两行
    nnoremap <leader>d 2j

    ```

    现在，你只需要按 `空格 + d`，它就会瞬间触发，绝不拖泥带水，而且还保留了 Vim 原生 `d`（删除）的强大功能。

    **方法三：缩短 Vim 的等待超时时间**

    如果你执意就要用单个 `d` 键，可以通过缩短 Vim 的等待时间来让它反应更快。在 `.vimrc` 中添加：

    ```vim
    set timeoutlen=200 " 将等待时间从 1000ms 缩短为 200ms

    ```

    * **优点：** 按 `d` 后的延迟明显变短（只需等 0.2 秒）。
    * **缺点：** 这是一个全局设置。如果你还有其他组合键（比如 `gT` 或自定义的 `<leader>ab`），你必须在 0.2 秒内把后续字母按完，否则快捷键就会断掉。

    ---

    **总结建议：** 除非有特殊理由，否则强烈建议使用 **方法二（使用 `<leader>` 键）**，这是规避 Vim 键位冲突、告别延迟的最优雅解法。

* vim 里我觉得按 esc 按得太频繁了，有没有什么替代方案

    确实，频繁按 Esc 很麻烦！有很多更好的替代方案：

    1. 键盘映射替代 Esc

        在 `~/.vimrc` 中添加：

        ```vim
        " 最常用的方案：jj 或 jk 替代 Esc
        inoremap jj <Esc>
        inoremap jk <Esc>
        inoremap kj <Esc>

        " Ctrl+c 替代 Esc（很多人的习惯）
        inoremap <C-c> <Esc>
        ```

    2. 使用 Leader 键组合

        ```vim
        " 设置 Leader 键为空格（很流行）
        let mapleader = " "

        " 然后可以用空格组合键
        inoremap <leader>q <Esc>
        vnoremap <leader>q <Esc>
        ```

    3. 其他便捷映射

        ```vim
        " Ctrl+[ 等同于 Esc（无需映射，系统默认支持）
        " 但位置可能不太方便

        " fd 快速退出（类似 vscode 的 ideavim）
        inoremap fd <Esc>

        " 分号快速进入命令模式
        nnoremap ; :
        vnoremap ; :
        ```

    4. 系统级修改 CapsLock

        **macOS**（使用 Karabiner-Elements）：
        ```json
        {
          "rules": [
            {
              "description": "Change caps_lock to escape",
              "manipulators": [
                {
                  "type": "basic",
                  "from": { "key_code": "caps_lock" },
                  "to": { "key_code": "escape" }
                }
              ]
            }
          ]
        }
        ```

        **Linux**：
        ```bash
        # 临时修改
        setxkbmap -option caps:escape

        # 永久修改，添加到 ~/.bashrc 或 ~/.xprofile
        setxkbmap -option caps:escape
        ```

        **Windows**（使用 AutoHotkey脚本）：
        ```autohotkey
        ; caps_lock.ahk
        CapsLock::Esc
        ```

    5. 插件解决方案

        **vim-easyescape**

        ```vim
        Plug 'zhou13/vim-easyescape'
        let g:easyescape_chars = { "j": 1, "k": 1 }
        let g:easyescape_timeout = 100
        cnoremap jk <C-c>
        cnoremap kj <C-c>
        ```

        **better-escape.vim**

        ```vim
        Plug 'jdhao/better-escape.vim'
        let g:better_escape_shortcut = 'jk'
        let g:better_escape_interval = 200
        ```

    6. 我的个人推荐配置

        ```vim
        " 我的完整配置方案：
        inoremap jk <Esc>
        inoremap kj <Esc>
        cnoremap jk <C-c>
        cnoremap kj <C-c>

        " 同时保留一些其他便捷映射
        inoremap <C-c> <Esc>
        vnoremap <C-c> <Esc>

        " 空格作为 Leader，配合其他快捷键
        let mapleader = " "
        nnoremap <leader>w :w<CR>
        nnoremap <leader>q :q<CR>
        nnoremap <leader>e :e<CR>
        ```

    7. 额外技巧

        ```vim
        " 快速保存退出组合
        nnoremap <leader>z :wq<CR>

        " 快速进入插入模式的不同位置
        nnoremap <leader>i i
        nnoremap <leader>a a
        nnoremap <leader>o o
        nnoremap <leader>O O

        " 使用 Ctrl+s 保存（需要终端设置）
        nnoremap <C-s> :w<CR>
        inoremap <C-s> <Esc>:w<CR>a
        ```

    **最推荐的方案**

    1. **首选**：`jj` 或 `jk` 映射（最自然，不需要硬件修改）
    2. **次选**：CapsLock 映射为 Esc（系统级，所有应用受益）
    3. **备选**：Ctrl+[ 或 Ctrl+c（如果习惯的话）

    我个人使用 `jk` 映射 + CapsLock 作为 Ctrl（用于其他快捷键），这样手基本不需要离开主键盘区！

* 我想查看 vim 中 ctrl + i 当前是什么作用，如何查看

    在 Vim 中，想要查看某个快捷键（包括组合键）当前被映射成了什么功能，可以使用 `:map` 系列命令。

    对于 `ctrl + i`（在 Vim 中通常表示为 `<C-i>`），你可以直接在命令模式（Normal Mode）下输入以下命令并回车：

    ```vim
    :verbose map <C-i>

    ```

    **命令拆解与技巧：**

    * **`:map <C-i>`**：这会列出所有模式下 `ctrl + i` 的快捷键映射。
    * **`verbose`**：这个前缀非常有用！它不仅能告诉你这个键被映射成了什么，还能**明确指出这个映射是在哪个配置文件（比如 `.vimrc`）的第几行被定义的**。这对于排查插件冲突引起的快捷键失效非常管用。

    ---

    **核心避坑指南：`ctrl + i` 与 `Tab` 的冲突**

    在绝大多数终端（Terminal）中，**`ctrl + i` 和 `Tab` 键发送的是完全相同的 ASCII 码**（`0x09`）。

    这意味着：

    1. 如果你查看 `<C-i>` 发现没有自定义映射，它默认触发的是 Vim 的内置功能：**在跳转列表中向前跳转（Jump forward）**。
    2. **如果你映射了 `Tab` 键，`ctrl + i` 也会跟着一起改变；反之亦然。** > **如何验证？**

    > 你可以尝试在 Vim 中输入 `:verbose map <Tab>`，看看是不是有插件（比如自动补全插件）把 `Tab` 给占用了，这通常就是 `ctrl + i` 行为异常的罪魁祸首。

* 进入 visual 模式后，进入命令窗口总会自动弹出这个 :'<,'>， 此时如果我想调用函数 call MyFunc()，会导致每一行都会调用 MyFunc()，从而数据被重复处理，有没有什么好的解决办法？

    这是 Vim 的常见问题。当你在 Visual 模式下选中文本后，命令行会自动添加 `:'<,'>` 范围标记，这表示 "对选中的每一行执行命令"。有几种解决方法：

    1. **临时取消范围（最简单）**

        按 `<C-u>` 或 `<C-w>` 删除命令行中的 `:'<,'>`：
        ```vim
        :'<,'>  " 自动出现
        <C-u>   " 删除 :'<,'
        call MyFunc()  " 正常调用
        ```

    2. **在函数内部处理范围**

        ```vim
        function! MyFunc() range
            " 使用 a:firstline 和 a:lastline 获取范围
            for lnum in range(a:firstline, a:lastline)
                let line = getline(lnum)
                " 处理每一行
                call setline(lnum, '* ' . line)
            endfor
        endfunction
        ```

        然后在 Visual 模式下：

        ```vim
        :'<,'>call MyFunc()  " 这时会正确处理每一行，不会重复
        ```

    3. **使用 `<C-r>=` 进入表达式寄存器**

        ```vim
        :  " 先按冒号
        <C-r>=  " 按 Ctrl+r 再按 =，进入表达式输入
        MyFunc()<CR>  " 输入函数调用并回车
        ```

        这样不会带入 `:'<,'>`。

    4. **创建映射避免此问题**

        在 `.vimrc` 中添加：

        ```vim
        " 映射调用函数，不传递范围
        vnoremap <leader>f :<C-u>call MyFunc()<CR>
        ```

        使用时：

        1. 选中文本（Visual 模式）
        2. 按 `<leader>f`
        3. 函数被调用一次，而不是每行调用

    5. **判断并处理重复调用**

        在函数内部检测和处理：

        ```vim
        function! MyFunc()
            " 检查是否在范围内调用
            if exists('b:myfunc_called') && b:myfunc_called
                return
            endif
            
            " 标记已调用
            let b:myfunc_called = 1
            
            " 你的逻辑...
            
            " 处理完后清除标记
            let b:myfunc_called = 0
        endfunction
        ```

    6. **使用 `normal` 命令代替范围**

        ```vim
        function! ProcessSelection()
            " 获取可视模式选择的范围
            let start_line = line("'<")
            let end_line = line("'>")
            
            " 对范围内的每一行处理
            for lnum in range(start_line, end_line)
                " 处理每一行...
            endfor
        endfunction
        ```

    最佳实践推荐：

    方案A：使用 `range` 函数（最规范）

    ```vim
    function! AddStar() range
        for lnum in range(a:firstline, a:lastline)
            let line = getline(lnum)
            call setline(lnum, '* ' . line)
        endfor
    endfunction

    " 在 Visual 模式下使用：
    :'<,'>call AddStar()
    ```

    方案B：使用映射（最方便）

    ```vim
    " 在 .vimrc 中添加
    vnoremap <leader>* :<C-u>call AddStarToSelection()<CR>

    function! AddStarToSelection()
        let start = line("'<")
        let end = line("'>")
        
        for lnum in range(start, end)
            call setline(lnum, '* ' . getline(lnum))
        endfor
    endfunction
    ```

    方案C：直接编辑时用 `<C-u>`

    临时需要调用函数时：
    1. `:'<,'>` 出现
    2. 按 `<C-u>` 删除
    3. 输入 `call MyFunc()`

    总结：

    - **如果函数需要处理多行**：使用 `function! MyFunc() range`
    - **如果函数只需执行一次**：用 `<C-u>` 删除范围前缀，或使用映射
    - **避免重复处理**：在函数内部合理设计逻辑，不依赖调用次数

    这样就能避免 Visual 模式下 `:'<,'>` 导致的重复调用问题了。

* 关于 noremap! 的特殊性

    当使用 noremap! 时，映射同时生效于：

    * 插入模式（输入文本时）

    * 命令行模式（输入命令时）

    但 不生效于 Normal 模式。如果你想要 Normal 模式的映射，需要使用 nnoremap。

* 在 visual 模式下，使用快捷键调用自定义函数 function MyFunc()，会自动添加 :'<,'> 吗？

    是的，在 Visual 模式下使用快捷键调用自定义函数时，Vim 会自动添加 :'<,'> 前缀。让我详细解释一下：

    Visual 模式下的行为

    ```vim
    " 示例映射
    vnoremap <leader>f :call MyFunc()<CR>
    ```

    在 Visual 模式下按下 <leader>f 时，命令行会显示：

    ```text
    :'<,'>call MyFunc()
    ```

    这个 :'<,'> 表示 Visual 模式下选中的行范围：

    * '< - 可视选择的起始行

    * '> - 可视选择的结束行

    * , - 范围分隔符

    为什么会有这个行为？

    这是 Vim 的历史遗留特性。在 Visual 模式下，当你按 : 进入命令行时，Vim 会自动插入 :'<,'> 前缀，表示接下来的命令会作用于选中的行范围。

    实际影响

    在你的函数中，你需要正确处理这个行范围：

    ```vim
    function! MyFunc() range
        " range 关键字让函数接收行范围参数
        let start_line = a:firstline
        let end_line = a:lastline
        
        for line_num in range(start_line, end_line)
            let content = getline(line_num)
            echo "处理第 " . line_num . " 行: " . content
        endfor
    endfunction

    " 映射（range 会自动传递）
    vnoremap <leader>f :call MyFunc()<CR>
    ```

    如何控制这个行为？

    方法1：使用 range 参数

    ```vim
    function! MyFunc() range
        " 函数会自动接收可视选择的范围
        echo "从 " . a:firstline . " 到 " . a:lastline
    endfunction
    ```

    方法2：不使用范围（需要在映射中清除）

    ``````vim
    vnoremap <leader>f :<C-u>call MyFunc()<CR>

    function! MyFunc()
        " 现在这个函数只针对当前行，忽略可视选择
        echo "当前行: " . line('.')
    endfunction
    ```
    ```

    方法3：同时支持两种调用方式

    ```vim
    function! MyFunc() range
        if a:0 > 0
            " 被 range 调用（有行范围）
            echo "处理范围: " . a:firstline . "-" . a:lastline
        else
            " 普通调用（无行范围）
            echo "处理当前行: " . line('.')
        endif
    endfunction

    " 两种映射方式
    nnoremap <leader>f :call MyFunc()<CR>    " 正常模式
    vnoremap <leader>f :call MyFunc()<CR>    " 可视模式（带范围）
    ```

    最佳实践建议

    * 明确设计意图：

        * 如果函数设计为处理多行，使用 range

        * 如果只处理当前行，在映射中用 <C-u> 清除范围

    * 示例：处理选中的多行

        ```vim
        function! ProcessSelection() range
            let total_chars = 0
            for lnum in range(a:firstline, a:lastline)
                let total_chars += strlen(getline(lnum))
            endfor
            echo "选中 " . (a:lastline - a:firstline + 1) . " 行，共 " . total_chars . " 字符"
        endfunction

        vnoremap <leader>c :call ProcessSelection()<CR>
        ```

    * 示例：只处理当前可视模式位置

        ```vim
        function! ProcessCursor()
            echo "光标在行: " . line('.') . "，列: " . col('.')
        endfunction

        " <C-u> 清除 '<,'> 前缀
        vnoremap <leader>p :<C-u>call ProcessCursor()<CR>
        ```

    测试技巧

    你可以用以下方式测试：

    ```vim
    " 临时测试映射
    :vnoremap <leader>t :echo "测试"<CR>
    ```

    在 Visual 模式下按 <leader>t，你会看到命令行显示 :'<,'>echo "测试"，然后执行。

    总结

    * 默认行为：Visual 模式下会自动添加 :'<,'>

    * 处理方式：

        * 接受范围：函数加 range 参数

        * 拒绝范围：映射中加 <C-u>

    * 设计建议：根据函数用途选择合适的方式，保持行为一致性

* vim 中 noremap! 同时作用于 insert 模式和 command 模式，那么是否有同时作用于 normal 模式和 visual 模式的？

    是的，Vim 中确实有同时作用于 Normal 模式和 Visual 模式的映射命令。以下是相关命令：

    1. noremap

        最常用的就是 noremap 本身，它默认作用于 Normal、Visual 和 Operator-pending 模式：

        ```vim
        :noremap j gj    " 在 Normal 和 Visual 模式都有效
        ```

    2. nnoremap + vnoremap

        如果你想明确指定只用于 Normal 和 Visual 模式，可以分别设置：

        ```vim
        :nnoremap j gj    " 只在 Normal 模式
        :vnoremap j gj    " 只在 Visual 模式
        ```

    3. xnoremap

        专门用于 Visual 模式（包括 character-wise、line-wise 和 block-wise visual 模式）：

        ```vim
        :xnoremap <Tab> >
        ```

    4. 特殊模式组合

        你可以用模式前缀来组合多个模式：

        ```vim
        " Normal + Visual 模式
        :map j gj          " 等同于 noremap，但不建议使用（会递归映射）

        " 使用模式标识符
        :map <silent> <expr> j mode() =~ "[Vv]" ? "gj" : "j"
        ```

    5. 推荐的实践

        通常建议分开设置，这样更清晰：

        ```vim
        " 相同的映射应用到多个模式
        nnoremap <leader>c :Comment<CR>
        vnoremap <leader>c :Comment<CR>
        ```

        或者使用 augroup 批量设置：

        ```vim
        augroup my_mappings
          autocmd!
          autocmd ModeChanged *:[vV\x16]* mapclear <buffer>
          autocmd FileType python nnoremap <buffer> <leader>c I#<Esc>
          autocmd FileType python vnoremap <buffer> <leader>c :s/^/#/<CR>
        augroup END
        ```

    6. 模式说明

        * Normal 模式：普通浏览模式

        * Visual 模式：包括 v（字符可视）、V（行可视）、Ctrl+v（块可视）

        * Operator-pending 模式：比如输入 d 后等待动作

    注意：noremap 默认已经包含了 Normal、Visual 和 Operator-pending 模式，所以如果你想要 Normal + Visual 模式的效果，直接用 noremap 就可以（Operator-pending 模式通常也适用相同的映射）。

* `noremap!`

    在 Vim 中，noremap! 用于 非递归地 定义插入模式和命令行模式的键盘映射。它的作用如下：

    核心特点：

    * 非递归 - 不会展开映射中的其他映射，避免循环或意外的多层映射

    * 双重模式 - 同时影响插入模式和命令行模式

    * 安全映射 - 防止映射间的相互干扰

    基本语法：

    ```vim
    :noremap! {lhs} {rhs}
    ```

    与相关命令对比：

    | 命令 | 递归 | 影响模式 |
    | - | - | - |
    | map! | 是 | 插入 + 命令行模式 |
    | noremap! | 否 | 插入 + 命令行模式 |
    | inoremap | 否 | 仅插入模式 |
    | cnoremap | 否 | 仅命令行模式 |

    使用示例：

    ```vim
    " 非递归映射：按 jk 退出插入模式（安全方式）
    :noremap! jk <Esc>

    " 比下面这个更安全（避免递归问题）：
    :map! jk <Esc>  " 可能有递归风险
    ```

    适用场景：

    * 定义模式切换快捷键（如 jk 退出插入模式）

    * 在插入/命令行模式中使用，且需要避免映射冲突时

    * 当同一映射需要在两种模式下生效时

    注意事项：

    * 通常推荐使用更具体的 inoremap（仅插入模式）和 cnoremap（仅命令行模式）

    * 只有当确实需要在两种模式下使用相同映射时才用 noremap!

    总的来说，noremap! 提供了在插入和命令行模式中定义安全、非递归映射的方式。

* `:'<,'>normal I* `解析

    这个命令的作用是：在当前选中的每一行前面插入一个星号 *。

    具体解析：

    * :'<,'>：

        * : 进入命令模式

        * '<,'> 是代表上次可视模式（visual mode）选中的范围

        * '< 表示选中区域的开始行

        * '> 表示选中区域的结束行

    * normal：

        * 表示在指定范围内执行普通模式命令

        * 可以使用缩写 norm

    * I* ：

        * I 是普通模式的命令，表示在当前行的行首（第一个非空白字符前）进入插入模式

        * * 是要插入的字符（星号加空格）

    使用步骤：

    * 用可视模式（V 或 Ctrl+V）选中多行文本

    * 按 : 进入命令模式，会自动出现 :'<,'>

    * 输入 normal I* 并回车

* 关于 vim 中`:s/foo/bar<CR>`的解析机制

    * 在 Vim 的命令行模式下，`<CR>` 是特殊的控制字符表示回车执行

    * 输入`<cr>`字符串需要转义：`:s/foo/bar\<CR>/g`

        注意：

        1. 只需要转义左尖括号就可以了，足以破坏原义

        1. 在映射中要双重转义：

            ```vim
            " 如果在映射定义中：
            nnoremap <leader>x :s/foo/bar\\<CR>/g<CR>
            " 需要 \\ 因为映射本身会解析一次
            ```

        1. 使用表达式寄存器更安全：

            ```vim
            :s/foo/\='bar<CR>'/g
            " 在表达式中，字符串字面量不需要转义特殊键名
            ```

    * 如果要输入换行，可以使用`\r`

        ```vim
        " 将 foo 替换为 bar + 回车 + 换行
        :s/foo/bar\r/g

        " 效果：将 "foo" 替换为：
        " bar
        " （后面内容在新行）
        ```

        也可以使用表达式寄存器:

        ```vim
        " 使用 Ctrl-V + Enter 输入真正的回车符
        :s/foo/bar^M/g
        " ^M 的输入方法：在 replacement 部分按 Ctrl-V 然后按 Enter

        " 或者使用表达式：
        :s/foo/\="bar" . nr2char(13)/g
        ```

    * 重要区别：\r vs \n

        ```vim
        " \r 在 replacement 中表示换行符
        " \n 在 pattern 中表示换行符，在 replacement 中表示 NUL 字符（ASCII 0）

        :s/foo/bar\n/g    " 错误：会插入 NUL 字符（显示为 ^@）
        :s/foo/bar\r/g    " 正确：插入换行符
        ```

    **注意事项**

    * `\r` 在搜索模式中表示回车（`^M`），在替换文本中表示换行

    * 如果要替换为真正的 `^M` 字符（CR），需要用 `\x0d` 或表达式寄存器

    * 某些情况下，可能需要设置 `:set ff=unix` 确保换行符正确

* vim 中的`<Leader>`键

    Leader 键是 Vim 中的一个自定义前缀键，用于创建用户自定义快捷键映射

    默认情况下，Vim 的 Leader 键是反斜杠`\`。

    查看当前 Leader 键:

    ```vim
    :echo mapleader
    :echo g:mapleader
    ```

    设置 Leader 键:

    ```vim
    " 最常见的设置：逗号 ,
    let mapleader = ","      " 全局 Leader 键
    let maplocalleader = "\\"  " 本地 Leader 键（用于文件类型特定映射）

    " 其他常用选择
    let mapleader = ";"      " 分号（也很方便）
    let mapleader = "<Space>" " 空格键（需要先按空格，再按其他键）
    let mapleader = "\\"     " 保持默认的反斜杠

    " 空格键作为 Leader（现在很流行）
    let mapleader = " "
    nnoremap <Space> <Nop>  " 禁用空格键的默认行为
    ```

    设置了 Leader 键后，配合映射使用：

    ```vim
    " 在 .vimrc 中添加映射
    nnoremap <Leader>w :w<CR>        " \w 保存文件（如果 Leader 是 \）
    nnoremap <Leader>q :q<CR>        " \q 退出
    nnoremap <Leader>o o<Esc>        " 下方插入空行并返回普通模式
    nnoremap <Leader>O O<Esc>        " 上方插入空行并返回普通模式

    " 如果是空格作为 Leader，那么就是：
    " 按空格，再按 w = 保存
    " 按空格，再按 o = 下方插入空行
    ```

    与 Local Leader 的区别

    * `<Leader>`：全局快捷键前缀

    * `<LocalLeader>`：文件类型特定的快捷键前缀

    ```vim
    " 设置 Local Leader
    let maplocalleader = "\\"

    " 只在特定文件类型中有效的映射
    autocmd FileType python nnoremap <buffer> <LocalLeader>c I#<Esc>
    " 在 Python 文件中，按 \c 在行首添加注释
    ```

    examples:

    ```vim
    " ~/.vimrc 中建议这样设置
    let mapleader = " "          " 空格作为 Leader
    let maplocalleader = "\\"    " 反斜杠作为 Local Leader

    " 一些实用映射
    nnoremap <Leader>w :w<CR>
    nnoremap <Leader>q :q<CR>
    nnoremap <Leader>e :e $MYVIMRC<CR>  " 编辑 vimrc
    nnoremap <Leader>s :source $MYVIMRC<CR>  " 重新加载 vimrc
    nnoremap <Leader>o o<Esc>k  " 下方插入空行，光标移到新行
    nnoremap <Leader>O O<Esc>j  " 上方插入空行，光标移到原行
    ```

* vim `nnoremap`

    nnoremap 是 Vim 中用于定义快捷键映射（key mapping） 的核心命令之一。以下是它的详细用法介绍：

    基本语法

    ```vim
    nnoremap <新按键> <执行的操作>
    ```

    关键特点

    1. 非递归映射

        * `nnoremap` 中的 `n` 表示 Normal 模式，`nore` 表示 非递归

        * 如果映射右侧包含其他映射键，不会再次触发映射

        * 对比：`nmap` 是递归映射，可能导致无限循环

        ```vim
        " 安全：不会再次触发映射
        nnoremap ; :

        " 危险：可能造成无限循环
        nmap ; :
        nmap : ;
        ```

    2. 模式限定

        * 只在 Normal 模式 下生效

        其他常用变体：

        ```vim
        inoremap    " Insert 模式
        vnoremap    " Visual 模式
        cnoremap    " Command-line 模式
        onoremap    " Operator-pending 模式
        noremap!    " 在 插入模式（Insert mode） 和 命令行模式（Command-line mode） 中创建非递归的键映射。
        ```

    **常用示例**

    * 基础用法

        ```vim
        " 将空格键设为 leader 键（常用前缀键）
        nnoremap <Space> <Nop>
        let mapleader = " "

        " 使用 leader 键的组合映射
        nnoremap <leader>w :w<CR>        " 保存文件
        nnoremap <leader>q :q<CR>        " 退出
        nnoremap <leader>fs :w<CR>       " 快速保存

        " 窗口导航
        nnoremap <C-h> <C-w>h            " 切换到左侧窗口
        nnoremap <C-j> <C-w>j            " 切换到下方窗口
        nnoremap <C-k> <C-w>k            " 切换到上方窗口
        nnoremap <C-l> <C-w>l            " 切换到右侧窗口
        ```

    * 特殊按键

        ```vim
        " 使用特殊键
        nnoremap <Esc> :nohlsearch<CR>   " 按 Esc 清除搜索高亮
        nnoremap <CR> o<Esc>             " 回车在当前行下方插入新行
        nnoremap <BS> X                  " 退格键删除前一个字符

        " 功能键
        nnoremap <F2> :set invpaste paste?<CR>  " F2切换粘贴模式
        nnoremap <F5> :source ~/.vimrc<CR>      " F5重新加载配置
        ```

    **实用技巧**

    ```vim
    " 快速编辑配置文件
    nnoremap <leader>ev :vsplit $MYVIMRC<CR>
    nnoremap <leader>sv :source $MYVIMRC<CR>

    " 缓冲区操作
    nnoremap <leader>bn :bnext<CR>   " 下一个缓冲区
    nnoremap <leader>bp :bprevious<CR> " 上一个缓冲区
    nnoremap <leader>bd :bdelete<CR> " 删除缓冲区

    " 快速移动
    nnoremap H ^                     " H 移动到行首
    nnoremap L $                     " L 移动到行尾
    nnoremap J 5j                    " J 向下移动5行
    nnoremap K 5k                    " K 向上移动5行

    " 大小写转换
    nnoremap <leader>u viwU          " 将当前单词转为大写
    nnoremap <leader>l viwu          " 将当前单词转为小写
    ```

    最佳实践

    1. 始终使用非递归映射

        除非有特殊需求，否则优先使用 `nnoremap` 而不是 `nmap`，避免意外递归。

    2. 使用 <leader> 前缀

        ```vim
        let mapleader = ","  " 设置 leader 键为逗号
        nnoremap <leader>s :w<CR>
        ```   

    3. 查看现有映射
    
        ```vim
        :map           " 查看所有映射
        :nmap          " 查看 Normal 模式映射
        :verbose nmap  " 查看映射及其定义位置
        ```

    4. 取消映射

        ```vim
        :nunmap <按键>  " 取消 Normal 模式映射
        ```

    5. 条件映射

        ```vim
        " 只在特定文件类型生效
        autocmd FileType python nnoremap <buffer> <leader>r :!python %<CR>
        ```

    注意事项

    * 映射覆盖：映射会覆盖默认功能，确保不会影响常用操作

    * 兼容性：在不同终端中，特殊键的表示可能不同

    * 可读性：复杂的映射建议加上注释说明

    * 冲突检查：使用 :nmap 检查是否已有映射

    通过合理使用 nnoremap，可以极大提升 Vim 的工作效率，打造个性化的编辑环境。

* 查看 nnoremap 支持的所有特殊按键

    在 Vim 中查看支持的特殊按键（key codes），有几种方法：

    1. 查看官方文档

        最全面的方法是查看 Vim 的官方帮助文档：
        
        ```vim
        :help key-notation      " 查看所有按键表示法
        :help <>                " 查看特殊键列表
        :help keycodes          " 详细按键代码说明
        ```

    2. 主要特殊按键列表

        基础控制键

        ```vim
        <CR>       回车（Carriage Return）
        <Esc>      退出键
        <Tab>      Tab 键
        <BS>       退格键（Backspace）
        <Del>      删除键
        <Space>    空格键
        <Bar>      竖线符号（|）
        <Bslash>   反斜杠（\）
        ```

        方向键和功能键

        ```vim
        <Up>       上箭头
        <Down>     下箭头
        <Left>     左箭头
        <Right>    右箭头

        <F1> - <F12>     功能键 F1 到 F12
        <F13> - <F37>    扩展功能键（如有）
        <S-F1> - <S-F12> Shift + 功能键
        ```

        修饰键组合

        ```vim
        <C-...>    Ctrl 组合键，如：<C-a>, <C-b>, <C-Space>
        <A-...>    Alt 键（在终端中可能表示为 <M-...>）
        <M-...>    Meta 键（通常与 Alt 相同）
        <S-...>    Shift 组合键，如：<S-Tab>, <S-F1>
        <D-...>    Command 键（macOS）

        <C-S-...>   Ctrl+Shift 组合，如：<C-S-a>
        <A-S-...>   Alt+Shift 组合
        ```

        特殊符号键

        ```vim
        <lt>       小于号（<），用于避免被解析为按键开始
        <gt>       大于号（>）
        <Bslash>   反斜杠
        <Bar>      竖线
        ```

        鼠标按键

        ```vim
        <LeftMouse>     鼠标左键
        <MiddleMouse>   鼠标中键
        <RightMouse>    鼠标右键
        <2-LeftMouse>   双击左键
        <3-LeftMouse>   三击左键
        ```

        其他特殊键

        ```vim
        <Insert>        Insert 键
        <Home>          Home 键
        <End>           End 键
        <PageUp>        Page Up
        <PageDown>      Page Down

        <Help>          Help 键
        <Undo>          Undo 键
        <Redo>          Redo 键
        <Print>         Print Screen
        <Pause>         Pause/Break
        ```

    3. 实用查看方法

        方法一：使用 showkey 命令（Linux）

        ```bash
        # 在终端中查看按键代码
        showkey -a
        # 按键后会显示十进制和十六进制代码
        ```

        方法二：在 Vim 中测试按键

        ```vim
        " 1. 插入模式查看按键代码
        i<C-v>然后按特殊键

        " 2. 使用 :echo 测试
        :echo getchar()    " 按下一个键，显示其代码
        :echo keytrans(getchar())  " 转换为可读形式

        " 3. 查看最后按下的键
        :echo v:termresponse     " 显示终端响应
        ```

        方法三：查看当前终端支持的键码

        ```vim
        " 查看终端键码设置
        :set termcap

        " 查看所有 termcap 条目
        :set termcap?
        ```

    4. 诊断按键问题

        ```vim
        " 1. 设置 verbose 模式查看键码
        :set verbose=15
        " 然后尝试按键，Vim 会显示详细日志

        " 2. 检查 terminfo/termcap
        :echo &term          " 查看终端类型
        :echo $TERM          " 查看环境变量

        " 3. 使用捕获模式
        :map <F13>           " 如果不知道键名，可以先映射一个不存在的键
        " 然后按下你想测试的键，Vim 会显示其名称
        ```

    5. 常见问题解决

        终端中的特殊键问题

        ```vim
        " 在 ~/.vimrc 中添加终端兼容性设置
        if !has('gui_running')
          " 确保方向键正常工作
          set term=$TERM
          set ttyfast
          
          " 某些终端需要明确的键码设置
          if &term =~ '^screen' || &term =~ '^tmux'
            " tmux/screen 兼容性
            execute "set <xUp>=\e[1;*A"
            execute "set <xDown>=\e[1;*B"
            execute "set <xRight>=\e[1;*C"
            execute "set <xLeft>=\e[1;*D"
          endif
        endif
        ```

        检查特定按键的映射

        ```vim
        " 查看某个按键是否已被映射
        :map <Tab>           " 查看 Tab 键的所有映射
        :nmap <Tab>          " 只查看 Normal 模式的 Tab 映射

        " 查看映射来自哪里
        :verbose map <Tab>
        ```

    6. 创建测试映射来识别未知键

        ```vim
        " 方法：将未知键映射到可识别的操作
        nnoremap <F20> :echo "特殊键被按下"<CR>
        " 然后按下你想识别的键，如果看到消息，说明键名是 <F20>

        " 或者使用这个技巧：
        :nnoremap \| :echo "按键代码:" getchar()<CR>
        ```

    7. 实用脚本：显示按键信息

        ```vim
        " 在 vimrc 中添加以下函数
        function! ShowKey()
          let c = getchar()
          echom "按键代码:" c
          echom "按键名称:" keytrans(c)
          echom "字符表示:" nr2char(c)
        endfunction

        " 映射一个快捷键来测试
        nnoremap <leader>k :call ShowKey()<CR>
        ```

    注意事项：

    * GUI Vim 和终端 Vim 的键码支持可能不同

    * 不同终端（iTerm2, GNOME Terminal, tmux, screen）可能有差异

    * 组合键在不同环境中的表示可能不同（如 <C-Space> 在某些终端中不可用）

    * 使用 :help 总是最可靠的信息来源

    最常用的特殊键包括：<Space>, <CR>, <Esc>, <Tab>, <C-字母>, 方向键和功能键。

* 在插入模式（Insert mode）下，ctrl + h 作用为删除前一个字符（等同于 Backspace 退格键）

    这是因为在早期的终端中，Ctrl+h 被用作退格键的替代（ASCII 码为 8）。Vim 保留了这一传统支持，尤其是在终端环境中。

    在 Vim 分屏（split window）环境下，<C-w> h 用于向左切换窗口，但注意这不是 <C-h>（后者没有 w 前缀）。

* vim 中常见的映射模式

    命令	作用模式	是否递归	说明
    map	普通、可视、选择、操作符等待	递归	通用映射（已弃用，建议用模式专用命令）
    nmap	普通模式	递归	Normal mode
    imap	插入模式	递归	Insert mode
    cmap	命令行模式	递归	Command-line mode
    noremap!	插入模式和命令行模式	非递归	Insert + Command-line，非递归
    inoremap	插入模式	非递归	Insert mode only
    cnoremap	命令行模式	非递归	Command-line mode only
    nnoremap	普通模式	非递归	Normal mode only

* noremap! 的 ! 表示映射适用于插入模式和命令行模式

    noremap → 普通、可视、选择模式（无 !）

    noremap! → 插入和命令行模式（有 !）

* vim 如何表示 shift + enter 这个快捷键

    在 vim 中，shift+enter 没有直接的默认映射，但可以通过不同的方式来表示和绑定：

    1. 终端中的键码表示

        在大多数终端中，shift+enter 发送的键码通常与普通 enter 不同：

        ```vim
        " 可以尝试这些映射，具体取决于你的终端
        <c-m>      " ctrl+m（有时等同于回车）
        <s-cr>     " shift+enter（可能在某些环境中有效）
        <cr>       " 普通的回车（enter）

        " 更常用的表示方式
        <nl>       " new line (shift+enter)
        <c-j>      " ctrl+j（在某些终端中代表 shift+enter）
        ```

    2. 查找 shift+enter 的实际键码

        要确定你的终端/vim 中 shift+enter 发送什么键码：

        * 方法1：使用 :help 查看

        ```vim
        :h keycodes
        ```

        * 方法2：在插入模式下查看

            * 进入插入模式：i

            * 按 ctrl+v，然后按 shift+enter

                * 会显示实际的转义序列，如 ^m 或 ^j

        * 方法3：使用脚本检测

            ```vim
            " 在 normal 模式下，将 shift+enter 映射到显示键码
            :nnoremap <key> :echo "key code: " . getchar()<cr>
            " 然后用 shift+enter 替换上面的 <key>

            " 或者使用更简单的方法
            :map <s-cr>
            ```

    3. 实际映射示例

        根据不同的终端/环境，可能需要这样映射：

        ```vim
        " 尝试1：使用 <s-cr>
        :nnoremap <s-cr> :echo "shift+enter pressed"<cr>
        :inoremap <s-cr> <esc>:echo "shift+enter in insert mode"<cr>

        " 尝试2：如果 <s-cr> 无效，尝试其他表示
        :nnoremap <c-j> :echo "ctrl+j / shift+enter"<cr>

        " 尝试3：使用 <nl>（new line）
        :nnoremap <nl> :echo "new line key"<cr>
        ```

    4. gui vim (gvim)

        在 gui 版本的 vim 中，表示更直接：

        ```vim
        <s-enter>  " gui vim 中的 shift+enter
        ```

    5. 实际应用场景

        常见的 shift+enter 用法：

        ```vim
        " 在 normal 模式下，shift+enter 快速插入新行
        nnoremap <s-cr> o<esc>

        " 在 insert 模式下，shift+enter 在当前行下方插入新行并保持插入模式
        inoremap <s-cr> <esc>o

        " 或者映射到特定功能
        nnoremap <s-cr> :make<cr>      " 编译
        nnoremap <s-cr> :w<cr>         " 保存
        ```

    建议：

    * 先用 ctrl+v 然后按 shift+enter 查看实际键码

    * 根据显示的字符进行映射

    * 如果显示 ^m，使用 <c-m> 映射

    * 如果显示 ^j，使用 <c-j> 映射

    大多数现代终端模拟器和 gui vim 都支持 <s-cr> 表示法，但具体取决于你的 vim 版本和终端环境。

### vim 脚本、字符串、函数

* vim 可以使用单引号作为字符串，也可以使用双引号

* `line("$")`：获取当前文件的最后一行行号（$ 在这里代表最后一行）。

    注：

    1. 为什么这里没有单引号`'`？

* vim 3 秒后自动从 insert 模式转换为 normal 模式。但是现在这套方案只能在 vim 为前台是才开始计时。我希望 vim 在后台时仍在计时，怎么解决？ 

    这是一个非常经典的问题。你当前使用的 `CursorHoldI` 事件依赖于 Vim 的内部事件循环，而 Vim 在后台（失去焦点）时，为了节省 CPU 资源，它的内部定时器默认会“挂起”或者不再触发光标停顿事件。这就导致了“只有 Vim 在前台时才开始计时”的现象。

    要实现在后台也能精准计时并强制退回 Normal 模式，有两种主流的高效解决方案。

    方案一：使用 Vim 8+ / Neovim 的内置异步定时器（推荐）

    从 Vim 8.0 开始引入了 `timer_start()` API，它是**系统级别**的异步定时器，即使 Vim 处于后台，定时器依然会在后台倒计时。

    你可以通过监听进入插入模式（`InsertEnter`）、插入模式下有输入（`InsertCharPre` / `CursorMovedI`）来重置定时器，并在时间到期时强制执行 `stopinsert`。

    把以下代码替换你原有的配置：

    ```vim
    augroup AsyncBackToNormal
        autocmd!
        " 进入插入模式时启动定时器
        autocmd InsertEnter * call s:StartInsertTimer()
        " 在插入模式下每次按键或移动光标，刷新定时器（重新计算3秒）
        autocmd CursorMovedI,InsertCharPre * call s:StartInsertTimer()
        " 离开插入模式时，必须要清除定时器，防止在 Normal 模式下误触发
        autocmd InsertLeave * call s:StopInsertTimer()
    augroup END

    let g:insert_timer_id = -1

    function! s:StartInsertTimer()
        " 先清除上一个定时器
        call s:StopInsertTimer()
        " 开启新定时器：3000毫秒后执行 s:TimeoutHandler
        let g:insert_timer_id = timer_start(3000, function('s:TimeoutHandler'))
    endfunction

    function! s:StopInsertTimer()
        if g:insert_timer_id != -1
            call timer_stop(g:insert_timer_id)
            let g:insert_timer_id = -1
        endif
    endfunction

    function! s:TimeoutHandler(timer_id)
        " 确保当前确实还在插入模式或替换模式，然后强制退出
        if mode() =~# '[iR]'
            stopinsert
            " 可选：如果希望切回 Normal 后屏幕能立刻刷新，可以加上 redraw
            redraw
        endif
    endfunction
    ```

    **为什么这个方案能解决你的问题？**

    1. **后台持续计时**：`timer_start` 由操作系统内核或 Vim 异步事件循环驱动，不受 Vim 是否处于前台（Focus）的影响。

    2. **体验更合理**：原方案的 `CursorHoldI` 只要你停顿就会计时，而这个方案在你**每次敲击键盘输入内容时都会重置 3 秒**。只有当你彻底停手（无论是在前台停手，还是直接切到后台）3 秒后，才会触发退回。

    方案二：利用 `FocusGained` 事件瞬间判定（轻量级补丁）

    如果你不想用复杂的定时器脚本，只想在“回到 Vim 前台的一瞬间”检查在后台是否已经超过了 3 秒，可以用时间戳比对的方式。

    ```vim
    augroup FocusBackToNormal
        autocmd!
        " 每次离开焦点时，记录当前时间戳
        autocmd FocusLost * if mode() =~# '[iR]' | let g:lost_focus_time = localtime() | endif
        
        " 重新获得焦点时，比对时间差
        autocmd FocusGained * call s:CheckFocusTimeout()
    augroup END

    function! s:CheckFocusTimeout()
        if mode() =~# '[iR]' && exists('g:lost_focus_time')
            " 如果当前时间距离离开焦点的时间超过了 3 秒
            if localtime() - g:lost_focus_time >= 3
                stopinsert
            endif
            " 用完后清除变量
            unlet g:lost_focus_time
        endif
    endfunction
    ```

    **两种方案怎么选？**

    * **选方案一**：如果你希望 Vim 在后台时**切切实实地已经退回了 Normal 模式**（例如你写了其他自动化脚本依赖 Vim 的模式状态）。

    * **选方案二**：如果你不在乎它在后台时是什么模式，只要**重新点开 Vim 的一瞬间**发现超过 3 秒了就自动变回 Normal 模式即可。这个方案最省资源。

    * 这两种方案我都试了，不行，你再想想

        这两个方案都不行，给我提供了一个非常关键的线索！这说明问题出在 **Vim 的事件循环阻塞** 或者 **终端环境的限制** 上。

        我们来剖析一下为什么刚才的方案会失效，并针对性地给出终极解法。

        **为什么之前的方案会失效？**

        1. **针对方案一（定时器失效的原因）：** Vim 的 `timer_start` 确实在后台计时，但 `stopinsert` 这个命令有时候是“惰性”的。在 Vim 失去焦点或没有任何键盘输入时，Vim 的主事件循环会进入“休眠”状态。虽然定时器时间到了，触发了 `stopinsert`，但 Vim 觉得“反正你现在没看着我，我等你有下一个操作时再刷新界面/状态”。

        2. **针对方案二（FocusGained 失效的原因）：** 如果你使用的是**终端里的 Vim**（而不是图形界面的 gVim 或 MacVim），终端默认情况下是**不会向 Vim 发送焦点丢失/获得信号的**！也就是说，Vim 根本不知道你把它切到后台了，`FocusLost` 和 `FocusGained` 事件压根就没有触发。

        针对这两个痛点，我们来进行“暴力升级”。

        **升级版方案 A：强制唤醒事件循环（推荐）**

        既然 `stopinsert` 过于温和，甚至会被休眠的事件循环忽略，那我们在定时器触发时，直接给 Vim **模拟发送一个真实的 `<Esc>` 按键**。这能强制打断 Vim 的休眠并强制退回 Normal 模式。

        请清空之前的尝试，将以下代码放入配置：

        ```vim
        let g:insert_timer_id = -1

        function! s:StartInsertTimer()
            call s:StopInsertTimer()
            " 3秒后触发
            let g:insert_timer_id = timer_start(3000, function('s:TimeoutHandler'))
        endfunction

        function! s:StopInsertTimer()
            if g:insert_timer_id != -1
                call timer_stop(g:insert_timer_id)
                let g:insert_timer_id = -1
            endif
        endfunction

        function! s:TimeoutHandler(timer_id)
            if mode() =~# '[iR]'
                " 弃用 stopinsert，改用 feedkeys 强行发送 ESC 按键
                " 'n' 表示不触发映射，确保发送最纯粹的 Escape
                call feedkeys("\<Esc>", 'n')
                
                " 强制刷新屏幕，防止状态切换了但 UI 没更新
                redrawstatus
            endif
        endfunction

        augroup AsyncBackToNormal
            autocmd!
            autocmd InsertEnter * call s:StartInsertTimer()
            " 每次输入或移动光标都会重新计时
            autocmd CursorMovedI,InsertCharPre * call s:StartInsertTimer()
            autocmd InsertLeave * call s:StopInsertTimer()
        augroup END

        ```

        **为什么这次能行：** `feedkeys()` 会把按键直接塞进 Vim 的输入队列，这会让 Vim 误以为是你真实敲击了键盘，从而强制唤醒事件循环并退出 Insert 模式。

        **升级版方案 B：打通终端的 Focus 神经**

        如果你依然想通过“切换后台”来作为触发逻辑，那你必须先教会终端 Vim 识别“什么是后台”。

        **第一步：在你的 Vim 配置中强行开启终端焦点上报机制**（把这段加到配置最前面）：

        ```vim
        " 开启终端的 Focus 报告功能 (Neovim 默认已开启，Vim 需要手动加)
        if !has('nvim') && !has('gui_running')
            let &t_fe = "\<Esc>[?1004h"
            let &t_fd = "\<Esc>[?1004l"
            execute "set t_fe=" . &t_fe
            execute "set t_fd=" . &t_fd
        endif

        augroup FocusBackToNormal
            autocmd!
            " 只要失去焦点，立刻强制退出 Insert 模式并保存当前时间
            autocmd FocusLost * if mode() =~# '[iR]' | call feedkeys("\<Esc>", 'n') | endif
        augroup END

        ```

        **第二步：如果你使用了 Tmux**

        如果你是在 `tmux` 里面运行的 Vim，光靠上面的配置还不够，你必须在 `~/.tmux.conf` 中加入这行代码，让 tmux 允许传递焦点事件：

        ```tmux
        set -g focus-events on
        ```

        *(加完后别忘了跑一遍 `tmux source-file ~/.tmux.conf`)*

        ---

        **诊断排查**

        如果升级版方案 A 仍然不行，可能是你的操作系统（比如 macOS 的 App Nap 功能）在终端后台时直接挂起了整个进程的 CPU 调度。

        你目前的运行环境是怎样的？使用的是普通的 Vim 还是 Neovim？是在 Tmux 中运行，还是直接在某个特定的终端（如 iTerm2 / Windows Terminal）里？

    * 这次方案 A 可以了，你总结一下经验

        搞定就好！这是一个非常典型且深刻的 Vim 脚本避坑案例。我们可以把这次的经验总结为以下三点，这对于未来你编写任何 Vim 自动化脚本都非常有帮助：

        1. 核心症结：`stopinsert` 与 `feedkeys` 的底层差异

            这是导致前几次方案失败的罪魁祸首。

            * **`stopinsert` 是“惰性”的**：它只是在 Vim 的内部状态机中打了一个“准备退出插入模式”的标记（Flag）。如果此时 Vim 处于后台，主事件循环处于半休眠状态，这个标记就会被搁置。直到你重新把 Vim 切回前台、或者敲击键盘时，Vim 才会去检查这个标记并执行退出。这就导致了“在后台不生效”的假象。
            * **`feedkeys("\<Esc>", 'n')` 是“强制唤醒”的**：它直接向 Vim 的底层输入缓冲区（Input Buffer）塞入了一个真实的 `<Esc>` 信号。对于 Vim 来说，这和人类物理按下了键盘上的 ESC 键一模一样。这个动作会立刻惊醒处于休眠或后台状态的 Vim 事件循环，强制它在后台完成模式切换。

        2. 异步时代的定时器思维（Vim 8+ / Neovim）

            * **旧时代（你的原方案）**：依赖 `updatetime` 和 `CursorHoldI`。这种方式是“同步”且“阻塞”的，它高度依赖光标在**前台**的静止状态。一旦 Vim 失去焦点，光标状态停滞，整个计时机制直接停摆。

            * **新时代（方案A）**：利用 `timer_start()`。这是系统级的异步定时器。它的计时不依赖 Vim 内部的光标事件，而是由操作系统或 Vim 的异步底层内核驱动。只要时间一到，无论 Vim 在前台还是后台，回调函数必定会被执行。

        3. 防抖（Debounce）设计模式的运用

            在最终成功的方案 A 中，我们引入了类似前端开发中的“防抖”机制：

            ```vim
            autocmd CursorMovedI,InsertCharPre * call s:StartInsertTimer()
            ```

            每次你在插入模式下打字（`InsertCharPre`）或移动光标（`CursorMovedI`），代码都会**先销毁上一个定时器，再重新开启一个 3 秒定时器**。

            * **原方案的缺点**：只要时间到了就退回，不管你当时是不是正在疯狂打字，容易打断思路。
            * **新方案的优点**：只有当你**彻底停下手来 3 秒钟**（无论此时是在前台发呆，还是切到了后台微信聊天），它才会触发退回。这种体验更符合人类的直觉。

        **结语**

        编写 Vim 脚本时，**“改变状态”**（如 `stopinsert`、`set` 等）往往是温和且容易被终端环境忽略的；而“模拟输入”（如 `feedkeys`）则是强力且无法被忽略的。当以后遇到“脚本在后台不执行”或者“状态没有及时刷新”的问题时，优先考虑用 `feedkeys` 或 `redraw` 去强制唤醒它。

* 我想让 vim 在进入 insert 模式后，5 秒内如果没有键盘的输入，则自动切换到 normal 模式，怎么办

    在 Vim 中实现“Insert 模式下 5 秒内无输入自动退回 Normal 模式”的功能，可以通过结合使用 `CursorHoldI` 自动命令（autocmd）和 `updatetime` 参数来实现。

    你可以将以下配置直接复制到你的 `~/.vimrc`（或 Neovim 的 `init.vim`）中：

    ```vim
    " 设置等待时间为 5000 毫秒（5秒）
    set updatetime=5000

    " 当在 Insert 模式下触发 CursorHoldI 事件时，执行 stopinsert 退出到 Normal 模式
    augroup AutoBackToNormal
        autocmd!
        autocmd CursorHoldI * stopinsert
    augroup END

    ```

    💡 核心原理解析与注意事项

    有几个在使用这个功能前需要了解的细节（避免你后续用起来觉得体验奇怪）：

    1. **什么是 `updatetime`？**

        Vim 没有一个专门只针对“输入超时”的独立定时器。它是通过 `updatetime` 这个全局参数来控制的。当你停止输入达到 `updatetime` 设定的时间后，Vim 会触发 `CursorHold`（Normal模式下）或 `CursorHoldI`（Insert模式下）事件。

    2. **潜在的副作用：**

        * **全局影响**：`updatetime` 的默认值是 4000 毫秒（4秒）。很多 Vim 插件（比如代码补全插件、语法检查插件、Git 符号显示插件）都依赖这个参数来决定什么时候刷新界面。如果你把它改成 5000 毫秒，这些插件的响应可能会**变慢一点**（如果你之前有插件把它改成了 100 毫秒，现在改成 5000 毫秒，感官会很明显）。

        * **交换文件（Swap file）**：Vim 也会在 `updatetime` 到期时将缓冲区写入交换文件。


    3. **如果只想在特定文件类型生效：**

        如果你不想让这个功能全局生效，只想在写普通文本（比如 Markdown）时启用，可以这样写：

        ```vim
        autocmd FileType markdown autocmd CursorHoldI <buffer> stopinsert
        ```

    * 我将按下 i 后，按 alt + tab 将 vim 切换到后台，然后再 alt + tab 把 vim 切换到前台，此时不触发这个 autocmd 了，如何解决

        这是因为 Vim 的 `CursorHoldI` 事件有一个触发前提：**用户必须在 Vim 内部有过至少一次按键移动或输入**，然后停止输入，定时器才会开始倒计时。

        当你按下 `i` 进入 Insert 模式，接着立刻按 `Alt + Tab` 切走再切回来，对 Vim 来说，你在 Insert 模式下**从未在 Vim 窗口内产生过有效输入**，因此它的内部定时器根本没有被激活。

        要解决这个问题，我们需要让 Vim 在“**重新获得焦点**”（FocusGained）或者“**进入窗口**”（WinEnter）时，强制重新计算时间或触发状态检查。

        请将之前的配置替换为以下**升级版配置**：

        ```vim
        " 设置等待时间为 5000 毫秒（5秒）
        set updatetime=5000

        augroup AutoBackToNormal
            autocmd!
            " 1. 正常的输入停顿 5 秒自动退出
            autocmd CursorHoldI * stopinsert

            " 2. 当 Vim 重新获得焦点时，如果处于 Insert 模式，重新触发一次检测
            "    通过 feedkeys 模拟一个安全的无感操作（比如向右移动再向左移动光标 \<Left>\<Right>）
            "    来强行激活 Vim 的 CursorHoldI 定时器
            autocmd FocusGained * if mode() =~# '[iR]' | call feedkeys("\<Left>\<Right>", 'n') | endif
        augroup END
        ```

        🔍 为什么这样能解决？

        * `FocusGained` 是 Vim 的一个内置事件，当你的窗口从后台（比如 `Alt + Tab` 回来）切回前台时会触发。
        * `mode() =~# '[iR]'` 用于判断当前是否处于 Insert 模式或 Replace 模式。
        * `call feedkeys("\<Left>\<Right>", 'n')` 是核心魔法。由于 Vim 必须“动一下”才会开始计算 `updatetime`，我们通过代码模拟用户按了一次左方向键和右方向键。因为位置没变，对你没有任何视觉影响，但它成功**欺骗并激活了 Vim 的倒计时机制**。

        > ⚠️ **注意**：
        > `FocusGained` 事件在**图形界面（GUI）的 Vim**（如 GVim, MacVim）中百分之百完美生效。
        > 如果你使用的是**终端（Terminal）里的 Vim**，终端可能默认不会把焦点事件传递给 Vim。如果你发现切回来还是没生效，请确保你的终端开启了 Focus Reporting。
        > * **Neovim 用户**：默认已原生支持终端焦点。
        > * **原版 Vim 用户**：如果是在 tmux 或普通终端里，可能需要安装插件 `tmux-plugins/vim-tmux-focus-events` 或 `chriskempson/vim-alternate-toggles` 才能让终端支持 `FocusGained`。
        > 
        > 

        你可以在你的环境中试一下这个新配置，看看 `Alt + Tab` 切回来后停顿 5 秒是否能正常退回 Normal 模式？

* vim 中我想使用 `vnoremap <leader>b` 对选中的文件左右各添加两个星号 `**`，

    `vnoremap <leader>b mboma<esc>`ai**<esc>`ba**<esc>` 为什么不行？正确方法是什么？

    你原来的思路非常有趣，通过标记（marks）来定位选区的两端是一个很极客的想法。

    不过，这个映射之所以无法如愿工作，主要是因为 **Vim 模式切换的特性** 以及 **光标位置的微妙变化**。

    ---

    为什么你原来的方案不行？

    我们来拆解一下你的命令 `mboma<esc>`ai`ba<esc>`：

    1. **`mboma` 的逻辑死胡同**：在 Visual 模式下，当你按下 `m` 时，Vim 会**立刻退出 Visual 模式**并把当前光标所在的那*一个*字符标记为 `b`。接着你按 `o`（在正常模式下是向下新开一行），然后又按 `m`、`a`。这时候你原本的选区早就消失了，光标也已经跑偏了。
    2. **光标位置会发生偏移**：即使你在 Normal 模式下手动用 ``a` 和 ``b` 找到了位置，当你插入左侧的 `` 后，右侧标记 ``b` 的相对字符位置就会向后偏移 2 个字符。这时候再直接跳到 ``b`，加星号的位置就会错位（变成加在倒数第二个字符前面）。

    ---

    正确且高效的方法

    在 Vim 中，处理选区文本最地道、最常用的方法有两种：

    方法一：利用寄存器（最推荐，简单且丝滑）

    按 `c` 剪切选区，然后利用 `<C-r>"` 瞬间把剪切的内容连同星号一起塞回去。

    ```vim
    vnoremap <leader>b c**<C-r>"**<esc>
    ```

    * 原理解析：

        * `c`：删除选中的文本，并进入插入模式（此时选中的文本自动存入了默认寄存器 `"` ）。
        * ``：输入左边的星号。
        * `<C-r>"`：在插入模式下，直接粘贴默认寄存器里的文本。
        * ``：输入右边的星号。
        * `<esc>`：返回正常模式。

    方法二：利用内置的选区标记 `<` 和 `>`

    如果你非要用类似“标记跳转”的思路，Vim 其实自带了代表选区开头和结尾的内置标记：`<` 和 `>`。

    ```vim
    vnoremap <leader>b <esc>`.>a**<esc>`.`<i**<esc>
    ```

    * **原理解析**：必须先用 `<esc>` 退出可视模式，然后利用 ``>`（选区末尾）和 ``<`（选区开头）进行精准定位。为了防止先加左边导致右边偏移，这里**先加右边，再加左边**。

    终极白嫖方案：使用插件

    如果你经常需要给文本加括号、引号、星号，强烈推荐装一个 Vim 神级插件 **`vim-surround`**。

    装好后，你甚至不需要自己写映射。在 Visual 模式下，直接输入：
    `S*` 就可以包裹一层星号（*text*）
    再按一次 `S*` 就可以包裹两层星号（**text**）

* 那如果使用 s 命令进行替换呢 

    如果你想用 `s` 命令（即 `:s` 替换命令）来实现，思路会稍微有些不同。在 Vim 的 Visual 模式下按下 `:` 时，命令行会自动补全为 `:'<,'>`，这意味着接下来的替换命令**仅针对当前选中的区域**。

    但是，使用 `s` 命令时有一个非常重要的细节需要注意：**你是选中了一行中的“部分单词”，还是选中了“整行/多行”？**

    这里为你提供两种基于 `s` 命令的实现方式：

    ---

    方法一：针对“部分单词”或单行内的选区（最常用）

    如果你只是在某一行里用 `v` 选中了几个词，想给它们前后加 ``，可以利用正则表达式的 `\%V` 匹配元字符（它代表“仅匹配选区内的内容”）：

    ```vim
    vnoremap <leader>b :s/\%V.*\%V/**&**/<CR>
    ```

    * **原理解析**：
    * `:`：进入命令行模式，Vim 自动补全为 `:'<,'>`（代表选区范围）。
    * `s/`：开始替换。
    * `\%V.*\%V`：极其关键！`\%V` 告诉 Vim 匹配必须在选区内开始，也在选区内结束。`.*` 匹配选区里的所有文本。
    * `/&/`：`&` 代表刚才匹配到的原文本，我们在它左右各加上 ``。
    * `<CR>`：敲回车执行。

    方法二：针对“多行”或“整行”的选区

    如果你是用 `V`（行可视模式）选中了一整行或多行，想让这几行的最前端和最后端加上 ``，可以使用下面这个更简单的正则：

    ```vim
    vnoremap <leader>b :s/^.\+$/\*\*&\*\*/<CR>
    ```

    或者直接用替代符（省去转义）：

    ```vim
    vnoremap <leader>b :s#.*#**&**#<CR>
    ```

    * **注意**：这种方法会把选中的**每一行**都分别用 `` 包裹起来。

    为什么更推荐上一轮回答的 `c<C-r>"<esc>` ？

    虽然用 `s` 命令（方法一）看起来很高级，但在实际盲打时它有一个小小的痛点：
    当你执行完 `:s/.../` 替换后，光标通常会**跳转到当前行的行首**，而不是停留在你刚刚操作的单词旁边。

    而使用 `c<C-r>"<esc>` 这种“修改+寄存器”的方式，光标在执行完后会**精准停留在右侧星号的旁边**，更符合丝滑输入的直觉。

* 下面是我的自定义函数：

    ```vim
    function AddStar()
        let start_line = line("'<")
        let end_line = line("'>")
        echo "选中的行范围: " . start_line . " 到 " . end_line
        for lnum in range(start_line, end_line)
            let line = getline(lnum)
            if line !~ '\S'
                continue
            endif
            execute lnum . 'normal! ^i* '
        endfor
        return 1
    endfunction
    ```

    该如何修改这个函数，避免重复处理数据？

    你需要修改函数来避免重复处理。有几种方法：

    方法1：添加 `range` 关键字（推荐）

    ```vim
    function! AddStar() range  " 注意：添加 range 关键字
        echo "选中的行范围: " . a:firstline . " 到 " . a:lastline
        for lnum in range(a:firstline, a:lastline)
            let line = getline(lnum)
            if line !~ '\S'  " 跳过空行或只有空白字符的行
                continue
            endif
            execute lnum . 'normal! ^i* '
        endfor
        return 1
    endfunction
    ```

    **使用方法**：

    ```vim
    :'<,'>call AddStar()  " 只会执行一次，正确处理所有行
    ```

    方法2：在 Visual 模式下使用特殊调用方式

    ```vim
    function! AddStar()
        " 检查是否是从命令行调用的（避免重复）
        if exists('s:in_addstar') && s:in_addstar
            return 1
        endif
        
        let s:in_addstar = 1  " 设置标记
        
        let start_line = line("'<")
        let end_line = line("'>")
        echo "选中的行范围: " . start_line . " 到 " . end_line
        
        for lnum in range(start_line, end_line)
            let line = getline(lnum)
            if line !~ '\S'
                continue
            endif
            execute lnum . 'normal! ^i* '
        endfor
        
        let s:in_addstar = 0  " 清除标记
        return 1
    endfunction
    ```

    方法3：使用不同的函数名映射（最实用）

    ```vim
    " 在 .vimrc 中添加映射
    vnoremap <leader>* :<C-u>call AddStarVisual()<CR>

    function! AddStarVisual()
        let start_line = line("'<")
        let end_line = line("'>")
        echo "选中的行范围: " . start_line . " 到 " . end_line
        
        for lnum in range(start_line, end_line)
            let line = getline(lnum)
            if line !~ '\S'
                continue
            endif
            execute lnum . 'normal! ^i* '
        endfor
        
        return 1
    endfunction
    ```

    方法4：检查命令行内容（通用解决方案）

    ```vim
    function! AddStar()
        " 获取当前命令行内容
        let cmdline = getcmdline()
        
        " 如果命令行以 :'<,'> 开头，说明是范围调用
        if cmdline =~ "^'<,'>"
            " 我们只需要处理一次，所以直接返回
            return 1
        endif
        
        " 正常处理逻辑
        let start_line = line("'<")
        let end_line = line("'>")
        echo "选中的行范围: " . start_line . " 到 " . end_line
        
        for lnum in range(start_line, end_line)
            let line = getline(lnum)
            if line !~ '\S'
                continue
            endif
            execute lnum . 'normal! ^i* '
        endfor
        
        return 1
    endfunction
    ```

    最佳解决方案：使用 `range` + 改进逻辑

    ```vim
    function! AddStar() range
        echo "选中的行范围: " . a:firstline . " 到 " . a:lastline
        
        " 保存光标位置
        let save_pos = getpos('.')
        
        " 遍历选中的行
        for lnum in range(a:firstline, a:lastline)
            let line = getline(lnum)
            
            " 跳过空行或只有空白字符的行
            if line !~ '\S'
                continue
            endif
            
            " 检查是否已经以星号开头（避免重复添加）
            if line =~ '^\s*\*'
                echo "跳过第 " . lnum . " 行（已以星号开头）"
                continue
            endif
            
            " 添加星号
            execute lnum . 'normal! ^i* '
            echo "已处理第 " . lnum . " 行"
        endfor
        
        " 恢复光标位置
        call setpos('.', save_pos)
        
        echo "处理完成！"
        return 1
    endfunction

    " 同时添加一个非 range 版本供普通调用
    function! AddStarSingle()
        " 获取当前行
        let lnum = line('.')
        let line = getline(lnum)
        
        if line !~ '\S'
            echo "当前行为空，跳过"
            return
        endif
        
        if line =~ '^\s*\*'
            echo "当前行已以星号开头"
            return
        endif
        
        execute lnum . 'normal! ^i* '
        echo "已为当前行添加星号"
    endfunction

    " 创建映射
    nnoremap <leader>* :call AddStarSingle()<CR>
    vnoremap <leader>* :call AddStar()<CR>
    ```

    使用建议：

    1. **如果你希望用 `:'<,'>call AddStar()` 调用**：使用 `range` 版本（方法1）
    2. **如果你希望用映射调用**：使用方法3的映射方式
    3. **如果你希望函数更智能**：使用方法4，但注意获取命令行内容只在命令行模式下有效

    推荐**使用方法1的 `range` 版本**，这是 Vim 的标准做法，语义清晰，不易出错。

* 在 if 语句中进行正则匹配

    * 匹配非空白字符

        匹配非空白字符（即行中至少有一个非空白字符），有几种正确的方法：

        1. 使用单引号字符串（推荐）

            ```vim
            if line =~ '\S'      " \S 匹配非空白字符
                echo 'selected'
            endif
            ```

        2. 使用双引号字符串并转义

            ```vim
            if line =~ "\\S"     " 需要双反斜杠
                echo 'selected'
            endif
            ```

        3. 使用更明确的模式

            ```vim
            " 匹配至少一个非空白字符
            if line =~ '\S\+'
                echo 'selected'
            endif

            " 或者匹配任何非空白字符
            if line =~ '[^[:space:]]'
                echo 'selected'
            endif
            ```

        4. 使用 Vim 内置函数

            ```vim
            " 如果不需要正则，可以使用 trim 函数
            if trim(line) != ''
                echo 'selected'
            endif

            " 或者检查是否全是空白
            if line !~ '^\s*$'
                echo 'selected'
            endif
            ```

        正则表达式说明

        * `\S`：匹配非空白字符（非空格、制表符等）

        * `\s`：匹配空白字符

        * `^\s*$`：匹配空行或只有空白字符的行（使用 !~ 取反即可匹配非空行）

* vimscript 中如何实现 if xxx != yyy，或者 if not xxx == yyy 这种效果？

    在 Vimscript 中，有多种方式可以实现不等于判断或取反逻辑：

    1. 不等于运算符

        ```vim
        " 使用 != 运算符（最常用）
        if xxx != yyy
            echo "not equal"
        endif
        ```

    2. 取反运算符

        ```vim
        " 使用 ! 取反运算符
        if !(xxx == yyy)
            echo "not equal"
        endif

        " 或者
        if not xxx == yyy
            echo "not equal"
        endif
        ```

    3. 具体示例

        ```vim
        let name = "vim"
        let value = 10

        " 字符串不等于
        if name != "emacs"
            echo "This is vim"
        endif

        " 数字不等于  
        if value != 5
            echo "Value is not 5"
        endif

        " 取反方式
        if !(name == "emacs")
            echo "Not emacs"
        endif

        if not name == "emacs"
            echo "Also not emacs"
        endif
        ```

    4. 对于模式匹配的取反

        ```vim
        let line = "hello world"

        " 匹配包含 "hello" 的行
        if line =~ "hello"
            echo "contains hello"
        endif

        " 取反：不包含 "hello" 的行
        if line !~ "hello"
            echo "does not contain hello"
        endif

        " 或者使用取反运算符
        if !(line =~ "hello")
            echo "also does not contain hello"
        endif
        ```

    5. 复杂逻辑组合

        ```vim
        let a = 1
        let b = 2
        let c = 3

        " 多个条件组合
        if a != b && b != c
            echo "a, b, c are all different"
        endif

        " 使用 not 运算符
        if not (a == b || b == c)
            echo "no equal pairs"
        endif
        ```

    6. 运算符总结

    | 运算符 | 描述 | 示例 |
    | - | - | - |        
    | != | 不等于 | if a != b |
    | !~ | 不匹配模式 | if str !~ "pattern" |
    | ! | 逻辑非 | if !condition |
    | not | 逻辑非（更可读） | if not condition |

    7. 注意事项

        ```vim
        " 字符串比较注意类型
        if "10" != 10          " 字符串和数字比较，通常为 true
        if "10" != "10"        " false
        if 10 != 10.0          " false（数字类型比较）

        " 使用 is 和 isnot 进行严格比较（比较对象引用）
        let list1 = [1, 2, 3]
        let list2 = [1, 2, 3]
        if list1 isnot list2   " true，不同对象
            echo "different objects"
        endif
        ```

    8. 实际应用示例

        ```vim
        " 检查文件类型
        if &filetype != "python"
            echo "Not a Python file"
        endif

        " 检查是否在插入模式
        if mode() != "i"
            echo "Not in insert mode"
        endif

        " 检查行是否不为空
        let line = getline(lnum)
        if line != ""
            echo "Line is not empty"
        endif

        " 更严谨的检查空行（包含空白字符）
        if line !~ '^\s*$'
            echo "Line has content"
        endif
        ```

    推荐使用：对于大多数情况，直接使用 != 运算符最清晰简单。如果需要取反复杂的表达式，使用 !() 或 not 可以提高可读性。

* vimscript 中，如何获取当前行的行号，以及当前行的内容？

    在 Vimscript 中，可以使用以下方式获取当前行号和当前行内容：

    获取当前行号

    ```vim
    " 方法1: line() 函数
    let current_line = line('.')
    echo "当前行号: " . current_line

    " 方法2: line('.') 的简写
    echo "当前行号: " . line('.')
    ```

    获取当前行内容

    ```vim
    " 方法1: getline() 函数
    let current_content = getline('.')
    echo "当前行内容: " . current_content

    " 方法2: 直接使用 getline('.')
    echo "当前行内容: " . getline('.')
    ```

    结合使用的示例

    ```vim
    " 获取并显示当前行号和内容
    let line_num = line('.')
    let line_content = getline('.')

    echo "第 " . line_num . " 行: " . line_content

    " 或者更简洁地
    echo "第 " . line('.') . " 行: " . getline('.')
    ```

    其他相关函数

    ```vim
    " 获取指定行号的内容
    let line_10 = getline(10)  " 获取第10行的内容

    " 获取可视模式下选中的行范围
    let start_line = line("'<")  " 可视模式起始行
    let end_line = line("'>")    " 可视模式结束行

    " 获取总行数
    let total_lines = line('$')  " 文件总行数
    ```

    实际应用示例

    ```vim
    " 示例1: 显示当前行信息
    function! ShowCurrentLineInfo()
        let lnum = line('.')
        let content = getline('.')
        echo "行号: " . lnum . " | 长度: " . strlen(content) . " | 内容: " . content
    endfunction

    " 示例2: 在当前行前面添加行号
    function! AddLineNumber()
        let lnum = line('.')
        let content = getline('.')
        call setline('.', lnum . ': ' . content)
    endfunction

    " 示例3: 复制当前行并添加行号
    function! CopyWithLineNumber()
        let lnum = line('.')
        let content = getline('.')
        let @" = lnum . ': ' . content  " 复制到无名寄存器
        echo "已复制: " . lnum . ': ' . content
    endfunction
    ```

    注意事项

    * . 表示当前光标位置

    * getline() 返回的是字符串，包含行尾的换行符会被去掉

    * 如果要修改当前行，可以使用 setline() 函数：

        ```vim
        call setline('.', '新的内容')
        ```

    这些函数在编写 Vim 插件或自定义映射时非常有用。

* Vimscript 中双引号字符串的转义规则：

    ```vim
    if line =~ "\w"
    ```

    "\w" 在双引号字符串中，\w 不是一个标准转义序列

    Vim 会将它解释为：反斜杠 + 字母 w

    所以实际匹配的是字面字符串 "w"，而不是正则表达式 \w

    如果要匹配单词字符，应该用：

    ```vim
    if line =~ '\w'      " 单引号字符串，不转义
    " 或者
    if line =~ "\\w"     " 双引号字符串中需要双反斜杠
    ```

* 函数 return 0 和 return 1 的区别

    这个区别主要体现在 条件判断 和 调用者如何处理返回值 上。让我详细解释：

    ```vim
    function! TestZero()
        return 0
    endfunction

    function! TestOne()
        return 1
    endfunction
    ```

    1. 在条件判断中的含义

        Vimscript 的真值判断规则：

        * 0 是假（false）

        * 1 是真（true）

        * 任何非零数值都是真

        ```vim
        if TestZero()
            echo "这会执行吗？不会！因为返回 0 是假"
        else
            echo "这个会执行"
        endif

        if TestOne()
            echo "这个会执行（返回 1 是真）"
        endif

        " 同样，返回 2, 3, -1 等也都是真"
        function! TestTwo()
            return 2
        endfunction

        if TestTwo()
            echo "这个也会执行"
        endif
        ```

    2. 表示函数的成功与失败

        模式1：成功/失败指示

        ```vim
        function! SaveFileIfModified()
            if &modified
                write
                return 1    " 成功保存"
            else
                echo "文件未修改，无需保存"
                return 0    " 未执行保存操作"
            endif
        endfunction

        " 使用示例
        if SaveFileIfModified()
            echo "文件已保存"
        else
            echo "文件未保存"
        endif
        ```

        模式2：查找/检查函数

        ```vim
        function! HasTodoComment()
            " 检查当前行是否有 TODO 注释
            if getline('.') =~ 'TODO'
                return 1    " 有 TODO"
            else
                return 0    " 没有 TODO"
            endif
        endfunction

        " 使用示例
        if HasTodoComment()
            echo "当前行有 TODO 注释"
        endif
        ```

    **实际应用场景**

    1. 作为条件使用

        ```vim
        function! IsValidFilename(filename)
            " 检查文件名是否有效
            if a:filename =~ '[/\\:*?"<>|]'
                return 0    " 无效
            else
                return 1    " 有效
            endif
        endfunction

        let fname = "test.txt"
        if IsValidFilename(fname)
            echo fname . " 是有效的文件名"
        endif
        ```

    2. 链式调用判断

        ```vim
        function! CheckAllConditions()
            if !Condition1()
                return 0
            endif
            if !Condition2()
                return 0
            endif
            if !Condition3()
                return 0
            endif
            return 1    " 所有条件都满足
        endfunction
        ```

    **vim script 中数值与布尔值之间的弱类型自动转换**

    ```vim
    function! VariousReturns()
        " 不同返回值的真值测试
        return 0    " 假
        return 1    " 真
        return -1   " 真
        return ""   " 真（空字符串在Vim中是真！）
        return "0"  " 真（字符串"0"也是真！）
        return []   " 真
        return {}   " 真
    endfunction

    " 重要：Vimscript 中只有 0 是假，其他都是真！
    ```

    **除了使用布尔值作为函数返回值，还可以考虑使用字符串或更丰富的返回值：**

    ```vim
    " 有时简单的 0/1 不够用
    function! ValidateInput(input)
        if empty(a:input)
            return "EMPTY"
        elseif a:input !~ '^\w\+$'
            return "INVALID_CHARS"
        elseif strlen(a:input) < 3
            return "TOO_SHORT"
        else
            return "OK"    " 成功
        endif
    endfunction

    let result = ValidateInput("ab")
    if result == "OK"
        echo "输入有效"
    else
        echo "错误：" . result
    endif
    ```

    **总结**

    | 返回值 | 布尔值 | 典型含义 | 使用场景 |
    | - | - | - | - |
    | return 0 | 假 (false) | 失败、否、错误 | 条件判断中表示假，操作失败 |
    | return 1 | 真 (true) | 成功、是、正确 | 条件判断中表示真，操作成功 |


* vim 字符串切片

    基础切片语法

    ```vim
    let str = "Hello World"

    " 基本形式：str[start:end] 或 str[start:end:step]
    " 索引从 0 开始，包含 start，不包含 end

    echo str[0:4]      " 输出: Hello
    echo str[6:11]     " 输出: World
    echo str[:5]       " 输出: Hello (从开头到索引5)
    echo str[6:]       " 输出: World (从索引6到末尾)
    echo str[-5:-1]    " 输出: Worl  (负索引表示从末尾倒数)
    echo str[-5:]      " 输出: World
    ```

    切片步长（stride）

    ```vim
    let str = "abcdefghij"

    " str[start:end:step]
    echo str[::2]      " 输出: acegi (每隔一个字符)
    echo str[1::2]     " 输出: bdfhj
    echo str[2:8:3]    " 输出: cf  (从索引2到8，步长为3)
    ```

    特殊用法和注意事项

    1. 越界处理

        ```vim
        let str = "Vim"

        echo str[0:10]     " 输出: Vim (end超出范围时取到末尾)
        echo str[5:10]     " 输出: ""  (start超出范围返回空字符串)
        echo str[-10:-5]   " 输出: ""  (负索引超出范围返回空)
        ```

    2. 处理 Unicode/多字节字符

        ```vim
        " 注意：切片按字节位置，不是按字符！
        let str = "中文测试"  " 每个中文字符占3字节
        echo str[0:3]       " 输出: 中 (正确，刚好3字节)
        echo str[0:4]       " 输出: 中 加上乱码！(截断了UTF-8字符)

        " 要按字符切片，需要先转为列表
        let char_list = split(str, '\zs')
        echo join(char_list[0:2], '')  " 输出: 中文 (正确按字符切片)
        ```

    3. 与 split() 结合使用

        ```vim
        let line = "vim is awesome"
        let words = split(line)  " 按空白分割
        echo words[0:2]          " 输出: ['vim', 'is', 'awesome']
        echo join(words[1:], ' ') " 输出: is awesome
        ```

    实际应用示例

    示例1：提取文件扩展名

    ```vim
    function! GetExtension(filename)
        let dot_pos = strridx(a:filename, '.')
        if dot_pos >= 0
            return a:filename[dot_pos + 1:]
        endif
        return ''
    endfunction

    echo GetExtension("script.py")  " 输出: py
    ```

    示例2：去除括号

    ```vim
    let text = "(Hello World)"
    echo text[1:-2]  " 输出: Hello World (去掉首尾括号)

    " 更安全的方式：
    if text[0] == '(' && text[-1] == ')'
        echo text[1:-2]
    endif
    ```

    示例3：每行处理

    ```vim
    function! TrimLeadingSpaces(line)
        let first_non_space = match(a:line, '\S')
        if first_non_space >= 0
            return a:line[first_non_space:]
        endif
        return ''  " 全空白行
    endfunction

    echo TrimLeadingSpaces("    indented")  " 输出: indented
    ```

    与列表切片的对比

    ```vim
    " 字符串和列表切片语法相同
    let str = "abcde"
    let lst = ['a', 'b', 'c', 'd', 'e']

    echo str[1:3]    " 输出: bcd
    echo lst[1:3]    " 输出: ['b', 'c', 'd']

    echo str[::2]    " 输出: ace
    echo lst[::2]    " 输出: ['a', 'c', 'e']
    ```

    性能提示

    ```vim
    " 对于大字符串的频繁切片，考虑使用函数：
    " 方法1：使用内置函数（最快）
    echo strpart(str, start, len)  " 相当于 str[start:start+len]

    " 方法2：使用正则表达式（适合复杂提取）
    echo matchstr(str, 'pattern')  " 提取匹配的部分

    " 方法3：直接切片（可读性好）
    echo str[start:end]
    ```

    重要限制

    * 索引不可变：不能通过切片赋值修改字符串

        ```vim
        let str = "hello"
        " str[0:2] = "xxx"  " 错误！不支持切片赋值
        let str = "xxx" . str[3:]  " 正确方式
        ```

    * : 字符的歧义：在 Vim 命令模式下 : 是命令前缀

        ```vim
        " 在脚本中正常使用
        let sliced = str[1:3]

        " 在命令模式下需要转义或使用函数
        :echo str[1:3]  " 可能被解释为命令
        :echo str[1\:3] " 正确
        :echo strpart(str, 1, 2)  " 替代方案
        ```

    替代方案比较

    | 场景 | 推荐方法 |
    | - | - |
    | 简单子串提取 | str[start:end] |
    | 已知长度的提取 | strpart(str, start, len) |
    | 基于模式的提取 | matchstr(str, pattern) |
    | 去除首尾字符 | str[1:-1] |
    | 处理多字节字符 | split(str, '\zs')[start:end] |

    Vimscript 的切片语法虽然源自 Python，但经过了简化和适应 Vim 的生态环境。掌握这些技巧能让你的 Vim 脚本更加简洁高效！

* vim 的函数可以返回多个值？

    不，Vim 函数不能直接返回多个值。这里涉及的是 Vim 脚本的列表解构赋值特性，不是真正的多返回值。

    让我详细解释：

    1. Vim 函数返回单个值

        Vim 函数总是返回单个值，这个值可以是：

        * 数字、字符串

        * 列表（List）

        * 字典（Dictionary）

    2. getpos() 实际上返回一个列表

        ```vim
        " getpos() 返回的是包含 4 个元素的列表
        let pos = getpos("'>")
        echo pos  " 输出类似: [0, 15, 3, 0]

        " 列表的 4 个元素分别是:
        " pos[0] - 缓冲区号 (0 表示当前缓冲区)
        " pos[1] - 行号
        " pos[2] - 列号 (从 1 开始)
        " pos[3] - 屏幕列偏移 (用于处理制表符等)
        ```

    3. 列表切片和多重赋值

        ```vim
        " 方法1: 先获取整个列表，再分别提取
        let pos = getpos("'>")
        let lnum2 = pos[1]
        let col2 = pos[2]

        " 方法2: 使用列表切片获取子列表
        let sublist = getpos("'>")[1:2]  " 返回 [行号, 列号]

        " 方法3: 列表解构赋值 (你例子中的写法)
        let [lnum2, col2] = getpos("'>")[1:2]
        " 这其实是两步:
        " 1. getpos("'>")[1:2] 返回一个列表，比如 [15, 3]
        " 2. let [a, b] = [15, 3] 将列表元素分别赋给变量
        ```

    4. 模拟多返回值的常见模式

        虽然不能真正返回多个值，但可以通过这些方式模拟：

        模式1：返回列表

        ```vim
        function! GetPosition()
            let pos = getpos("'>")
            return [pos[1], pos[2]]  " 返回包含两个值的列表
        endfunction

        let [line, col] = GetPosition()
        ```

        模式2：返回字典

        ```vim
        function! GetPositionDict()
            let pos = getpos("'>")
            return {'line': pos[1], 'col': pos[2]}
        endfunction

        let pos = GetPositionDict()
        echo pos.line
        echo pos.col
        ```

        模式3：通过参数引用修改（类似输出参数）

        ```vim
        function! GetPositionRef(line_var, col_var)
            let pos = getpos("'>")
            execute "let " . a:line_var . " = " . pos[1]
            execute "let " . a:col_var . " = " . pos[2]
        endfunction

        call GetPositionRef("line_num", "col_num")
        echo line_num
        echo col_num
        ```

    5. 其他支持列表解构的场景

        ```vim
        " 多个变量从列表赋值
        let [a, b, c] = [1, 2, 3]

        " 配合 range() 函数
        let [first, second] = range(1, 10)[0:1]

        " 字符串分割后解构
        let [name, version] = split('vim-8.2', '-')
        ```

    总结：

    你的代码 let [lnum2, col2] = getpos("'>")[1:2] 实际上是：

    * getpos() 返回一个列表

    * [1:2] 切片提取部分元素得到另一个列表

    * 使用列表解构语法将列表元素分别赋给两个变量

    这不是真正的多返回值，而是 Vim 脚本的语法糖，让列表赋值更简洁。Python 等语言也有类似的解构赋值特性。


* 简述 vim 中 `let [lnum1, lnum2] = [lnum2, lnum1]` 的作用

    在 Vim 脚本中，let [lnum1, lnum2] = [lnum2, lnum1] 的作用是交换两个变量的值。

    详细解释：

    * 赋值解构：[lnum1, lnum2] 是 Vim 脚本的列表解构语法，允许一次性将列表中的值赋给多个变量。

    * 临时列表：[lnum2, lnum1] 会先创建一个包含两个元素的临时列表，第一个元素是 lnum2 的当前值，第二个是 lnum1 的当前值。

    * 同时交换：赋值完成后，lnum1 获得原 lnum2 的值，lnum2 获得原 lnum1 的值，实现无中间变量的交换。

    示例：

    ```vim
    let lnum1 = 10
    let lnum2 = 20
    let [lnum1, lnum2] = [lnum2, lnum1]
    " 现在 lnum1 == 20, lnum2 == 10
    ```

    对比传统方法：

    传统交换需要临时变量：

    ```vim
    let temp = lnum1
    let lnum1 = lnum2
    let lnum2 = temp
    ```

    而 Vim 的这种语法更简洁，且可扩展（如交换更多变量）。

    注意事项：

    * 需要 Vim 7.4.710 或更高版本支持解构赋值。

    * 确保两侧列表长度一致，否则会报错。


* getpos("'<") 与 getpos("<") 有什么不同？

    getpos("'<") 和 getpos("<") 在 Vim 中有重要的区别：

    1. getpos("'<") - 可视模式选择的起始位置

        * '< 是 标记（mark），表示最近一次可视模式选择的起始位置

        * 这个标记会 持久保存，直到被新的可视选择覆盖

        * 即使在可视模式退出后，仍然可以通过 '< 访问这个位置

        * 返回的是缓冲区位置，包括缓冲区编号

        ```vim
        " 进入可视模式选择一些文本，然后退出可视模式
        " 仍然可以获取之前的可视选择起始位置
        let start_visual = getpos("'<")  " 例如：[bufnum, line, col, off]

        ```

    2. getpos("<") - 当前可视选择的起始位置

        * < 是 光标位置符号，表示 当前 可视模式选择的起始位置

        * 只在 可视模式激活时 有效

        * 如果没有激活可视模式，使用这个会报错或返回无效值

        * 返回的是窗口位置（相对当前窗口）

        ```vim
        " 必须在可视模式中执行
        vmap <leader>p :call ProcessCurrentVisual()<CR>

        function! ProcessCurrentVisual()
            " 获取当前可视选择的起始位置
            let current_start = getpos("<")  " 只在可视模式中有效
            let current_end = getpos(">")    " 当前可视选择的结束位置
        endfunction
        ```

    关键区别总结

    | 特性 | '<（带单引号的标记） | <（不带单引号的位置） |
    | - | - | - |
    | 持久性 | 持久保存，直到新的可视选择 | 只在可视模式激活时有效 |
    | 作用域 | 缓冲区范围 | 窗口范围 |
    | 使用时机 | 随时可用，即使不在可视模式 | 必须在可视模式中 |
    | 返回内容 | [bufnum, lnum, col, off] | [0, lnum, col, off]（bufnum为0） |

    实际应用示例

    ```vim
    " 示例1：获取最近的可视选择（即使不在可视模式）
    function! GetLastVisualRange()
        if getpos("'<")[1] == 0  " 检查是否有有效的可视选择历史
            echo "No previous visual selection"
            return []
        endif
        
        let start = getpos("'<")
        let end = getpos("'>")
        return [start, end]
    endfunction

    " 示例2：在当前可视选择中操作（必须在可视模式）
    vnoremap <leader>r :<C-U>call ProcessVisual()<CR>

    function! ProcessVisual() range
        " 使用 '< 和 '> 获取选择范围（这是安全的）
        let start_line = line("'<")
        let end_line = line("'>")
        
        " 或者使用 a:firstline, a:lastline（由 range 参数提供）
        for lnum in range(a:firstline, a:lastline)
            " 处理每一行
        endfor
    endfunction
    ```

    最佳实践建议

    * 通常使用 '< 和 '>：

        * 更安全，不会因为不在可视模式而报错

        * 可以获取最近的可视选择历史

        * 在可视模式命令中推荐使用

    * 使用 line() 和 col() 函数作为替代：

        ```vim
        " 更简洁的获取行号方式
        let start_line = line("'<")
        let start_col = col("'<")
        let end_line = line("'>")
        let end_col = col("'>")
        ```

    * 在可视模式映射中，Vim 会自动设置 '< 和 '> 标记，所以可以直接使用它们。

    结论：在大多数情况下，你应该使用 getpos("'<") 而不是 getpos("<")，因为它更可靠且不会依赖当前模式。

* 简述 `let save_reg = @"` 这个 vim 代码的作用

    在 Vim 脚本中，`let save_reg = @"` 这行代码的作用是**保存当前寄存器中的内容**。

    具体解释如下：

    - `@"` 是 Vim 中**默认寄存器**的表示方式（即未指定寄存器时复制的内容）
    - 这行代码将默认寄存器的内容保存到变量 `save_reg` 中

    **使用场景：** 

    通常在需要修改寄存器内容（如执行复制、删除操作）但又不想丢失原有内容的场景中使用。比如：

    ```vim
    " 保存当前寄存器内容
    let save_reg = @"

    " 执行某些会修改寄存器的操作
    normal dd

    " 恢复之前保存的寄存器内容
    let @" = save_reg
    ```

    这样可以在不影响用户原有剪切板内容的情况下执行操作，提高脚本的友好性。

    如果你想保存其他寄存器的内容，只需要将 `@"` 替换为对应的寄存器表示即可。

    **常用寄存器的表示方式：**

    ```vim
    " 命名寄存器 a-z
    let save_reg_a = @a    " 保存寄存器a的内容
    let save_reg_b = @b    " 保存寄存器b的内容

    " 其他特殊寄存器
    let save_reg_slash = @/    " 保存最后一次搜索的模式
    let save_reg_colon = @:    " 保存最后一次执行的命令行命令
    let save_reg_dot = @.      " 保存最后一次插入的文本
    let save_reg_star = @*     " 保存系统剪贴板 (X11 primary selection)
    let save_reg_plus = @+     " 保存系统剪贴板 (clipboard)
    let save_reg_percent = @%  " 保存当前文件名
    let save_reg_hash = @#     " 保存交替文件名
    ```

    实际使用示例：

    ```vim
    " 保存多个寄存器
    let save_reg_a = @a
    let save_reg_plus = @+

    " 执行某些操作
    normal "ayy        " 复制当前行到寄存器a
    normal "+p         " 从系统剪贴板粘贴

    " 恢复寄存器
    let @a = save_reg_a
    let @+ = save_reg_plus
    ```

    这样你就可以保存和恢复任意寄存器的内容，确保脚本操作不会破坏用户已有的寄存器数据。

* 简述 vim 中列表 [] 的用法

    在 Vim 脚本（Vimscript）中，列表（List）是一个非常重要的数据结构。它本质上是一个**有序的、可变**的元素集合，可以包含任意类型的数据（数字、字符串、甚至是另一个列表）。

    以下是 Vim 中列表的几种核心用法：

    1. 定义列表

        使用方括号 `[]` 定义列表，元素之间用逗号分隔。

        ```vim
        let empty_list = []
        let numbers = [1, 2, 3, 4]
        let mixed = [10, "hello", [5, 6]]  " 支持嵌套
        let names = ["Alice", "Bob", "Charlie"]
        ```

    2. 访问元素

        Vim 列表的索引从 **0** 开始。可以使用负数从末尾开始索引（-1 表示最后一个元素）。

        ```vim
        echo numbers[0]      " 输出 1
        echo names[-1]       " 输出 Charlie
        echo mixed[2][1]     " 输出 6 (访问嵌套列表)
        ```

    3. 切片

        可以使用 `[a:b]` 的形式获取子列表。切片操作会生成一个新的列表。

        ```vim
        let nums = [0, 1, 2, 3, 4, 5]
        echo nums[2:4]       " 输出 [2, 3, 4]
        echo nums[3:]        " 输出 [3, 4, 5] (从3到末尾)
        echo nums[:2]        " 输出 [0, 1, 2] (从开头到2)
        ```

    4. 添加元素

        - **`add(list, item)`**：在列表末尾添加一个元素。

          ```vim
          let fruits = ["apple"]
          call add(fruits, "banana")
          echo fruits   " 输出 ["apple", "banana"]
          ```

        - **`extend(list, list2)`**：将一个列表追加到另一个列表末尾。

          ```vim
          let list1 = [1, 2]
          let list2 = [3, 4]
          call extend(list1, list2)
          echo list1   " 输出 [1, 2, 3, 4]
          ```

        - **`insert(list, item, index)`**：在指定位置插入元素。

          ```vim
          let list = ["a", "c"]
          call insert(list, "b", 1)
          echo list   " 输出 ["a", "b", "c"]
          ```

    5. 删除元素

        - **`remove(list, index)`**：删除指定索引的元素。

          ```vim
          let list = [10, 20, 30]
          call remove(list, 1)   " 删除索引1的元素（20）
          echo list              " 输出 [10, 30]
          ```

        - **`remove(list, start, end)`**：删除范围内的元素。

          ```vim
          let list = [1, 2, 3, 4]
          call remove(list, 1, 2)   " 删除索引1到2的元素（2,3）
          echo list                 " 输出 [1, 4]
          ```

    6. 常用函数

        - **`len(list)`**：获取列表长度。

            ```vim
            echo len([1, 2, 3])   " 输出 3
            ```

        - **`empty(list)`**：检查列表是否为空。

            ```vim
            if empty(mylist)
              echo "列表是空的"
            endif
            ```

        - **`index(list, value)`**：查找值的索引。

            ```vim
            echo index(["a", "b", "c"], "b")   " 输出 1
            ```

        - **`join(list, separator)`**：将列表拼接成字符串。

            ```vim
            echo join(["vim", "is", "fun"], "-")   " 输出 vim-is-fun
            ```

        - **`reverse(list)`**：反转列表。

            ```vim
            echo reverse([1, 2, 3])   " 输出 [3, 2, 1]
            ```

    7. 循环遍历

        结合 `for` 循环遍历列表是常见的操作：

        ```vim
        let colors = ["red", "green", "blue"]
        for item in colors
            echo "颜色: " . item
        endfor
        ```

    8. 字符串与列表互转

        这在处理文本或命令行参数时非常有用。

        - 字符串转列表：`split(str, pattern)`

            ```vim
            let words = split("one,two,three", ",")   " 返回 ['one', 'two', 'three']
            ```

        - 列表转字符串：`join(list, sep)`

            ```vim
            let str = join(['a', 'b', 'c'], ':')      " 返回 'a:b:c'
            ```

    总结

    在 Vim 中，`[]` 不仅仅是创建列表的符号，它结合上述函数可以实现数据的存储、遍历、修改和转换。无论是写复杂的 Vim 配置，还是编写插件，掌握列表的用法都能让脚本变得更加简洁和高效。


* 在 Vimscript 中，点号 . 是字符串连接运算符。

    1. 基本字符串连接

        ```vim
        let str1 = "Hello"
        let str2 = "World"
        let result = str1 . " " . str2  " => "Hello World"
        ```

    2. 不同数据类型的连接规则

        * 字符串 + 字符串（正常连接）

            ```vim
            echo "Hello" . "World"    " => HelloWorld"
            echo "Age: " . "25"       " => Age: 25"
            ```

        * 字符串 + 数字（数字自动转换）

            ```vim
            echo "Number: " . 42      " => Number: 42"
            echo "Result: " . 3.14    " => Result: 3.14"
            ```

        * 数字 + 数字（先连接成字符串，不是数学加法！）

            ```vim
            echo 10 . 20              " => 1020" （字符串）
            echo 10 + 20              " => 30"   （数字相加）
            ```

    注意：Vimscript 使用 . 而不是 + 来避免歧义！

    **重要区别：**

    * +：数学加法（尝试转换为数字）

    * .：字符串连接（保持字符串）

    **连接列表/字典时需要转换**

    ```vim
    let list = [1, 2, 3]
    " echo "List: " . list  " 错误！不能直接连接列表

    " 需要先转换为字符串
    echo "List: " . string(list)  " => List: [1, 2, 3]"
    echo "List: " . join(list, ", ")  " => List: 1, 2, 3"

    let dict = {'a': 1, 'b': 2}
    echo "Dict: " . string(dict)  " => Dict: {'a': 1, 'b': 2}"
    ```

    **性能提示**

    对于大量字符串连接，使用列表更高效：

    ```vim
    " 低效：每次连接都创建新字符串
    let result = ""
    for i in range(1000)
        let result = result . i . ","
    endfor

    " 高效：使用列表收集，最后连接
    let parts = []
    for i in range(1000)
        call add(parts, i)
    endfor
    let result = join(parts, ",")
    ```

* vim 中，visual 模式选中几行后，如何在 vim script 中得到当前选中了哪几行，以及每行的内容？

    在 Vim Script 中，你可以通过以下方式获取 visual 模式下选中的行范围和内容：

    1. 获取选中行的范围

        ```vim
        " 获取起始行和结束行
        let start_line = line("'<")  " 视觉模式起始行
        let end_line = line("'>")    " 视觉模式结束行

        " 示例使用
        function! GetVisualSelectionRange()
            let start_line = line("'<")
            let end_line = line("'>")
            echo "选中的行范围: " . start_line . " 到 " . end_line
            return [start_line, end_line]
        endfunction
        ```

    2. 获取选中行的内容

        ```vim
        " 方法1: 使用 getline() 函数
        function! GetVisualSelectionLines()
            let start_line = line("'<")
            let end_line = line("'>")
            let lines = []
            
            for lnum in range(start_line, end_line)
                call add(lines, getline(lnum))
            endfor
            
            return lines
        endfunction

        " 方法2: 更简洁的方式
        function! GetVisualSelectionLines2()
            let [start_line, end_line] = [line("'<"), line("'>")]
            return map(range(start_line, end_line), 'getline(v:val)')
        endfunction
        ```

    3. 完整示例：获取选中行信息

        ```vim
        function! PrintVisualSelectionInfo()
            " 保存当前寄存器
            let old_reg = @"
            
            " 复制选中内容到无名寄存器
            normal! gv"xy
            
            " 获取行范围
            let start_line = line("'<")
            let end_line = line("'>")
            
            " 获取行数
            let line_count = end_line - start_line + 1
            
            " 获取内容（通过寄存器）
            let selected_text = @x
            
            " 恢复寄存器
            let @" = old_reg
            
            " 输出信息
            echo "起始行: " . start_line
            echo "结束行: " . end_line
            echo "总行数: " . line_count
            echo "内容:"
            echo selected_text
            
            return {
                \ 'start': start_line,
                \ 'end': end_line,
                \ 'count': line_count,
                \ 'text': selected_text,
                \ 'lines': split(selected_text, '\n')
                \ }
        endfunction
        ```

    4. 使用建议

        基础使用：

        ```vim
        " 在 visual 模式下选中的行上执行
        vnoremap <leader>p :<C-u>call PrintVisualSelectionInfo()<CR>
        ```

        更实用的函数：

        ```vim
        function! ProcessVisualSelection() range
            " a:firstline 和 a:lastline 自动包含选中范围
            let lines = []
            for lnum in range(a:firstline, a:lastline)
                let line_content = getline(lnum)
                " 对每行进行处理
                " ...
                call add(lines, line_content)
            endfor
            
            return lines
        endfunction

        " 映射
        vnoremap <leader>s :call ProcessVisualSelection()<CR>
        ```

    5. 注意事项

        * '< 和 '> 标记：只在 visual 模式退出后有效

        * 列选择模式：如果是在 visual block 模式，上面的方法会获取整行

        * 跨行选择：如果选择不是整行，getline() 会返回整行内容

        * 性能：对于大量行的选择，考虑性能影响

    6. 获取精确选择内容（包括部分行）

        ```vim
        function! GetExactVisualSelection()
            " 保存寄存器
            let old_reg = @"
            
            try
                normal! gv"xy
                let selected = @x
            finally
                let @" = old_reg
            endtry
            
            return selected
        endfunction
        ```

    这样你就可以在 Vim Script 中获取并处理 visual 模式下选中的行了。

* Vimscript

    Vimscript（Vim Script）是 Vim 编辑器的内置脚本语言，用于配置、自定义和扩展 Vim。以下是 Vimscript 的核心写法要点：

    1. 基础语法

        注释：以 " 开头

        ```vim
        " 这是一行注释
        ```

        变量：

        * 全局变量：`g:var_name`

        * 局部变量：`l:var_name`（函数内）

        * 选项变量：`&option_name`（如 `&tabstop`）

        * 环境变量：`$PATH`

        ```vim
        let g:my_var = 10
        let s:local_var = "hello"  " 脚本局部变量
        ```

    2. 数据类型

        * 字符串：`"string"` 或 `'string'`

        * 数字：整数或浮点数（如 42、3.14）

        * 列表：`[1, 2, 'three']`

        * 字典：`{'key': 'value', 'num': 42}`

        * 特殊类型：`v:true`、`v:false`、`v:null`

    3. 控制结构

        ```vim
        " 条件判断
        if condition
          echo "yes"
        elseif another_condition
          echo "maybe"
        else
          echo "no"
        endif

        " 循环
        for i in range(1, 5)
          echo i
        endfor

        while condition
          echo "looping"
        endwhile
        ```

    4. 函数定义

        ```vim
        function! MyFunction(arg1, arg2)
          echo a:arg1 . " " . a:arg2  " 参数前缀 a:
          return 1
        endfunction
        ```

        * 函数名首字母通常大写（避免与内置函数冲突）。

        * 用 ! 覆盖同名函数。

    5. 常用命令

        * echo：输出信息

        * execute：执行字符串形式的命令

        * normal：执行普通模式命令
        vim

        * normal! ggdd  " 跳转到首行并删除

        * command：自定义命令
        vim

        * command! Hello echo "Hello, Vim!"

    6. 自动命令（Autocmd）

        在特定事件触发时执行命令：

        ```vim
        autocmd BufNewFile *.txt echo "新文本文件已创建"
        autocmd BufWritePre *.js :call FormatFile()
        ```

    7. 映射（Key Mapping）

        普通模式映射：`nnoremap <Leader>w :w<CR>`

        插入模式映射：`inoremap jk <Esc>`

        可视化模式映射：`vnoremap <C-c> "+y`

        使用 `noremap` 避免递归映射。

    8. 插件开发示例

        简单插件结构：
        
        ```vim
        " plugin/myplugin.vim
        if exists('g:loaded_myplugin')
          finish
        endif
        let g:loaded_myplugin = 1

        function! s:MyPluginFunction()
          echo "插件已调用"
        endfunction

        command! -nargs=0 MyPluginCall call s:MyPluginFunction()
        ```

    9. 调试与测试

        查看错误：:`messages`

        调试模式：启动 Vim 时加 `-D` 参数，或使用 `:debug` 命令。

        脚本局部变量可通过` s:` 前缀隔离作用域。

    10. 注意事项

        * Vimscript 对空格敏感（赋值时等号两侧可加空格，但部分命令格式严格）。

        * 使用 `set nocompatible` 确保 Vim 模式。

        * 新版 Vim（8.0+）支持更多现代语法（如 Lambda 表达式）。

    **快速入门建议**

    * 从修改 .vimrc 开始，逐步添加自定义函数。

    * 参考 :help usr_41.txt（Vimscript 用户手册）。

    * 阅读现有插件源码（如 vim-airline）学习实践。

    掌握基础后，可进一步学习 `:help eval.txt`（完整语言参考）。

* vim 在 visual 下选择多行，进入命令模式时会自动添加`:'<,'>`，表示对每一行都调用一次后续的命令

    如果我们的函数按`:'<,'>call MyFunc()`方式调用时，对于每一行都会调用一次`MyFunc()`函数。

    可以在进入命令模式后，按`Ctrl` + `u`清除`'<,'>`。

* vim 可视模式下对 md 段落中有文字的行添加星号

    ```vim
    function AddAsterisk()
        let line = getline('.')
        if line !~ '\S'
            return 0
        endif
        let lnum = line('.')
        execute lnum . 'normal! ^i* '
        return 0
    endfunction

    vnoremap <leader>a :call AddAsterisk()<CR>
    ```

    可以按`\`, `a`触发函数调用。

* vim script 显示 visual 模式下选中的内容

    ```vim
    function ShowLines()
        let start_line = line("'<")
        let end_line = line("'>")
        echo "选中的行范围: " . start_line . " 到 " . end_line
        for lnum in range(start_line, end_line)
            let line = getline(lnum)
            echo 'cur line: ' . line
        endfor
        return 0
    endfunction
    ```

* vim function 不需要`function!`也能覆盖之前自己自定义的函数。

    不清楚如果不加`!`能不能覆盖 vim 内置函数。

* vim 中连接字符串时，`.`左右的空格可省略

    `echo 'line: '.line`

    似乎点`.`本身也可以被省略：

    `echo 'cur line: 'line`

    不清楚原因。

* vim 函数规则

    > E128: Function name must start with a capital or "s:": add_star()

    注：

    1. 可以看出，如果用`s:`作为函数名前缀，那么有点像 C 语言中的`private`函数了。

#### 常用函数介绍

* `setline()`

    在 Vim 脚本中，setline() 函数用于修改缓冲区中的指定行。

    函数原型

    ```vim
    :call setline(行号, 文本)
    ```

    或处理多行：

    ```vim
    :call setline(行号, 文本列表)
    ```

    作用

    setline() 的主要作用是替换缓冲区中指定行的内容。

    基本用法

    1. 替换单行

        ```vim
        " 将第 3 行替换为 "Hello World"
        :call setline(3, "Hello World")

        " 将当前行替换为新内容
        :call setline('.', "New line content")
        ```

    2. 替换多行

        ```vim
        " 从第 5 行开始替换 3 行内容
        :call setline(5, ['Line 1', 'Line 2', 'Line 3'])
        ```

    3. 实际示例

        ```vim
        " 示例 1：在文件开头添加标题
        function! AddTitle()
            " 在第一行插入标题
            call setline(1, "# My Document")
            " 在第二行插入空行
            call setline(2, "")
        endfunction

        " 示例 2：修改当前行
        nnoremap <leader>uc :call setline('.', toupper(getline('.')))<CR>
        ```

    注意事项

    * 行号有效性：行号必须在缓冲区有效范围内（1 到 line('$')）

    * 返回值：成功返回 0，失败返回 1

    * 性能：比执行 Ex 命令（如 :3s/old/new/）更快

    * 撤销：每次 setline() 调用会创建一个撤销块

    常见搭配

    ```vim
    " 与 getline() 配合使用
    let old_line = getline(5)
    call setline(5, "Modified: " . old_line)

    " 批量修改
    for i in range(1, line('$'))
        let line_content = getline(i)
        if line_content =~ 'pattern'
            call setline(i, substitute(line_content, 'old', 'new', 'g'))
        endif
    endfor
    ```

    对比其他方法

    ```vim
    " 使用 setline() - 更高效
    :call setline(10, "new text")

    " 使用 Ex 命令 - 会触发重绘
    :10s/.*/new text/

    " 使用 append()/setline() 组合添加行
    :call append(5, "Inserted line")
    :call setline(5, "Actual content")  " 如果需要精确位置
    ```

    setline() 是 Vim 脚本编程中操作缓冲区内容的核心函数之一，特别适合在需要精确控制行内容时使用。

* `match()`

    match() 函数在 Vim 中用于在字符串中查找正则表达式的匹配位置。

    基本语法

    ```vim
    match({字符串}, {模式} [, {起始位置} [, {计数}]])
    ```

    主要作用

    1. 查找匹配位置

        ```vim
        " 返回第一个数字的位置
        echo match("hello 123 world", '\d')  " 输出: 6

        " 查找第一个非空白字符
        echo match("   text", '\S')  " 输出: 3
        ```

    2. 从指定位置开始查找

        ```vim
        let s = "abc abc abc"
        echo match(s, 'abc', 4)  " 从位置4开始查找，输出: 4
        ```

    3. 查找第N次匹配

        ```vim
        let s = "one two three two one"
        echo match(s, 'two', 0, 2)  " 查找第2次出现的'two'，输出: 14
        ```

    返回值特点

    * 找到匹配：返回匹配开始的字符位置（从0开始计数）

    * 未找到匹配：返回 -1

    * 只返回第一个匹配的位置，要获取所有匹配需使用其他方法

    实际应用示例

    ```vim
    " 1. 检查字符串是否包含特定模式
    if match(line, '^#') != -1
        echo "这是一个注释行"
    endif

    " 2. 提取匹配位置后的内容
    let line = "Name: John Doe"
    let colon_pos = match(line, ':')
    if colon_pos != -1
        let name = strpart(line, colon_pos + 1)
        echo trim(name)  " 输出: John Doe
    endif

    " 3. 在函数中处理匹配
    function! FindEmail(text)
        let email_pos = match(a:text, '\v\w+@\w+\.\w+')
        if email_pos != -1
            return strpart(a:text, email_pos)
        endif
        return ""
    endfunction
    ```

    与相似函数的对比

    * match()：返回位置

    * matchstr()：返回匹配的字符串

    * matchend()：返回匹配结束后的位置

    * matchlist()：返回包含所有匹配组的列表

    注意事项

    * 默认使用 Vim 的魔术模式，建议使用 '\v' 前缀启用 very magic 模式

    * 位置索引从 0 开始

    * 支持 Vim 的所有正则表达式特性

    example:

    * `match(line_content, '\S')`

        这段 Vim 代码的作用是在行内查找第一个非空白字符，并移动光标到该位置。

        具体解析：

        * match(line_content, '\S') 调用 Vim 的 match() 函数

        * line_content：表示当前行的内容

        * '\S'：正则表达式模式，匹配非空白字符（等价于 [^ \t\r\n\v\f]）

        * 返回值：匹配到的位置索引（从0开始），如果没找到则返回 -1

        实际应用场景：

        ```vim
        " 示例：查找当前行第一个非空白字符的位置
        let pos = match(getline('.'), '\S')
        if pos != -1
            echo "第一个非空白字符在位置: " . pos
        endif

        " 常用组合：将光标移动到行首第一个非空白字符处（类似 0 命令）
        normal! ^
        ```

        类似功能的其他写法：

        * 普通模式：按 ^ 键

        * 命令行：:normal! ^

        * 搜索：/\S 然后按 n 查找

* `substitute()`

    substitute() 是 Vim 中的一个内置函数，用于对字符串执行替换操作。

    原型

    ```vim
    substitute({字符串}, {模式}, {替换内容}, {标志})
    ```

    参数说明

    * {字符串}：要进行替换的源字符串

    * {模式}：要匹配的正则表达式模式

    * {替换内容}：替换后的文本，可以包含特殊替换符

    * {标志}：控制替换行为的标志字符（可选）

    常用标志

    * g：全局替换（一行中的所有匹配）

    * i：忽略大小写

    * I：区分大小写

    * c：每次替换前确认

    用法示例

    ```vim
    " 基本替换
    echo substitute('hello world', 'world', 'Vim', '')
    " 返回: hello Vim

    " 全局替换
    echo substitute('foo bar foo', 'foo', 'baz', 'g')
    " 返回: baz bar baz

    " 使用正则表达式分组
    echo substitute('2024-01-15', '\(\d\+\)-\(\d\+\)-\(\d\+\)', '\2/\3/\1', '')
    " 返回: 01/15/2024

    " 在函数中使用
    let text = 'apple orange banana'
    let new_text = substitute(text, 'apple', 'pear', '')
    ```

    与 :substitute 命令的区别

    ```vim
    " substitute() 函数 vs :substitute 命令
    echo substitute('text', 'pattern', 'replacement', 'g')  " 函数
    :s/pattern/replacement/g                               " 命令

    " 主要区别：
    " 1. substitute() 操作字符串，不修改缓冲区
    " 2. :s 命令直接修改缓冲区内容
    ```

    实际应用

    ```vim
    " 在脚本中处理字符串
    function! CleanText(text)
        " 移除多余空格
        let cleaned = substitute(a:text, '\s\+', ' ', 'g')
        " 移除行首行尾空格
        let cleaned = substitute(cleaned, '^\s*\|\s*$', '', 'g')
        return cleaned
    endfunction

    " 批量处理列表
    let list = ['vim1', 'vim2', 'vim3']
    call map(list, "substitute(v:val, 'vim', 'neovim', 'g')")
    ```

    substitute() 是 Vim 脚本编程中处理字符串的重要工具，特别适合在不修改缓冲区的情况下进行文本转换。

    **`substitute(line_content, '^\s*\zs\S', '* &', '')` 简述这个函数的作用**

    这个函数的作用是：在每行第一个非空白字符前添加星号和空格（* ）

    详细解析：

    参数分解：

    * line_content：输入的字符串（通常是行内容）

    * '^\s*\zs\S'：正则表达式模式

        * ^：匹配行首

        * \s*：匹配0个或多个空白字符（空格、制表符等）

        * \zs：匹配开始标记，表示从这里开始才是真正要匹配的内容

        * \S：匹配一个非空白字符

    * '* &'：替换内容

        * * ：星号加空格

        * &：代表整个匹配到的内容（即第一个非空白字符）

    * ''：无标志，只替换第一个匹配项

    匹配逻辑：

    * ^\s*\zs\S 的意思是：找到行首的空白字符（不匹配它们），然后在第一个非空白字符处开始匹配

    效果示例：

    ```vim
    输入 "  hello world"  → 输出 "  * hello world"
    输入 "    vim"        → 输出 "    * vim"
    输入 "no spaces"      → 输出 "* no spaces"
    ```

    关键点：

    * \zs 是关键，它让前面的 ^\s* 只作为定位条件，而不作为匹配内容的一部分

    * 因此替换时只在第一个非空白字符前插入 * ，保留原有的缩进

    类似功能对比：

    ```vim
    " 这个会破坏缩进（匹配包括空白字符）：
    substitute(line_content, '^\s*\S', '* &', '')
    " 输入 "  hello" → 输出 "* hello"（丢失了两个空格）

    " 而使用 \zs 的版本保持缩进：
    substitute(line_content, '^\s*\zs\S', '* &', '')
    " 输入 "  hello" → 输出 "  * hello"（保持两个空格缩进）
    ```

    这个函数常用于创建带项目符号的列表，同时保持原有的文本缩进格式。

* `getpos()`

    **原型**

    ```text
    getpos({expr})
    ```

    * `{expr}`：字符串，指定位置标识符或标记

    **作用**

    获取指定位置在缓冲区中的坐标信息，返回一个包含位置信息的列表。

    **返回值**

    返回包含 4 个元素的列表 `[bufnum, lnum, col, off]`：

    * bufnum：缓冲区编号（总是当前缓冲区编号）

    * lnum：行号（从 1 开始）

    * col：列号（从 1 开始，字节索引）

    * off：屏幕列偏移量（用于制表符等宽字符）

    位置标识符

    * "."：当前光标位置

    * "'" 或 ""`：上一次跳转位置

    * "<" 或 ">"：最近可视模式选择的开始/结束位置

    * "0"：上一次修改的位置

    * "^"：上一次插入模式结束的位置

    * "$"：缓冲区最后一行

    * 标记：如 "'a"（小写标记仅当前缓冲区）、"'A"（大写标记全局）

    用法示例

    ```vim
    " 获取当前光标位置
    let pos = getpos(".")
    echo pos  " 输出类似 [1, 10, 5, 0]

    " 获取标记 'a 的位置
    let mark_pos = getpos("'a")

    " 获取上一次跳转位置
    let jump_pos = getpos("'")

    " 获取可视模式选择结束位置
    let visual_end = getpos(">")

    " 分解使用位置信息
    let [bufnum, lnum, col, off] = getpos(".")
    echo "行号: " . lnum
    echo "列号: " . col

    " 与 setpos() 配合使用
    let saved_pos = getpos(".")
    " ... 其他操作 ...
    call setpos(".", saved_pos)  " 恢复位置

    " 检查标记是否存在
    if getpos("'m")[1] != 0
        echo "标记 m 已设置"
    endif
    ```

    注意事项

    * getpos() 返回的列号是字节索引（从 1 开始），不是字符索引

    * 对于多字节字符（如中文），一个字符可能占用多个字节

    * 使用 getcurpos() 可获取字符索引的列位置

    * 未设置的标记返回 [0, 0, 0, 0]

    相关函数

    * setpos()：设置位置

    * getcurpos()：获取当前光标位置（字符索引）

    * col()：获取列号

    * line()：获取行号

* `add()`

    在 Vim 脚本中，add(list, item) 函数用于向列表末尾添加一个元素。

    基本语法

    ```vim
    let new_list = add(list, item)
    ```

    参数说明

    * list：要添加元素的目标列表

    * item：要添加到列表的元素（可以是任意类型）

    * 返回值：返回添加元素后的列表（会修改原列表）

    用法示例

    1. 添加字符串

        ```vim
        let fruits = ['apple', 'banana']
        let fruits = add(fruits, 'orange')
        " fruits 现在是 ['apple', 'banana', 'orange']
        ```

    2. 添加数字

        ```vim
        let numbers = [1, 2, 3]
        call add(numbers, 4)
        " numbers 现在是 [1, 2, 3, 4]
        ```

    3. 添加列表（嵌套）

        ```vim
        let matrix = [[1, 2], [3, 4]]
        call add(matrix, [5, 6])
        " matrix 现在是 [[1, 2], [3, 4], [5, 6]]
        ```

    4. 实际应用场景

        ```vim
        " 收集搜索结果
        let results = []
        for line in range(1, line('$'))
            if getline(line) =~ 'TODO'
                call add(results, line)
            endif
        endfor

        " 构建选项列表
        let options = ['Option 1', 'Option 2']
        call add(options, 'Option 3')
        let choice = inputlist(options)
        ```

    注意事项

    * add() 会直接修改原列表，不需要重新赋值

    * 如果要添加多个元素，可以使用 extend() 函数

    * 与 + 操作符不同，add() 是原地修改，效率更高

    * 如果需要在列表开头添加元素，使用 insert(list, item, 0)

    对比其他方法

    ```vim
    " add() - 原地添加，效率高
    call add(list, item)

    " + 操作符 - 创建新列表
    let list = list + [item]

    " extend() - 添加多个元素
    call extend(list, [item1, item2, item3])
    ```

    add() 是 Vim 脚本中操作列表时最常用的函数之一，特别适合在循环中动态构建列表。

* `setpos()`

    setpos() 是 Vim 中用于设置缓冲区中任意位置的函数，比 cursor() 更灵活。

    原型

    ```vim
    setpos({expr}, {list})
    ```

    * {expr}：位置标识符，可以是以下之一：

        * '.' 当前光标位置

        * '$' 最后一行

        * "'x" 标记 x 的位置

        * 缓冲区名称或编号

    * {list}：四元素列表 [bufnum, lnum, col, off]

    作用

    精确设置光标位置，包括：

    * 指定缓冲区

    * 指定行号

    * 指定列号（字节索引）

    * 虚拟编辑偏移

    用法示例

    1. 设置当前光标位置

        ```vim
        " 移动到第10行第5列
        call setpos('.', [0, 10, 5, 0])
        ```

    2. 设置标记位置

        ```vim
        " 设置标记a到第5行第10列
        call setpos("'a", [0, 5, 10, 0])
        ```

    3. 在特定缓冲区设置位置

        ```vim
        " 在缓冲区3的第15行第20列设置标记b
        call setpos("'b", [3, 15, 20, 0])
        ```

    4. 使用虚拟偏移

        ```vim
        " 第8行，第25列，虚拟偏移30（用于特殊编辑）
        call setpos('.', [0, 8, 25, 30])
        ```

    参数说明

    * bufnum：缓冲区编号，0表示当前缓冲区

    * lnum：行号（1-based）

    * col：列号（字节索引，1-based，0表示第1列前）

    * off：虚拟偏移（通常为0）

    注意事项

    * 列号是字节索引，不是字符索引（对多字节字符需要注意）

    * 使用 getpos() 获取当前位置

    * 比 cursor() 更底层，可以操作非当前缓冲区

    * 设置标记时，标记会自动移动到新位置

    实用组合

    ```vim
    " 保存并恢复光标位置
    let save_pos = getpos('.')
    " ... 其他操作 ...
    call setpos('.', save_pos)

    " 在两个位置间跳转
    function! TogglePosition()
        if !exists('g:alt_pos')
            let g:alt_pos = getpos('.')
            call setpos('.', [0, 1, 1, 0])
        else
            let temp = getpos('.')
            call setpos('.', g:alt_pos)
            let g:alt_pos = temp
        endif
    endfunction
    ```

* `getpos()`

    原型

    ```vim
    getpos({expr})
    ```

    * `{expr}`：字符串，指定位置标识符或标记

    作用

    获取指定位置在缓冲区中的坐标信息，返回一个包含位置信息的列表。

    返回值

    返回包含 4 个元素的列表 [bufnum, lnum, col, off]：

    * bufnum：缓冲区编号（总是当前缓冲区编号）

    * lnum：行号（从 1 开始）

    * col：列号（从 1 开始，字节索引）

    * off：屏幕列偏移量（用于制表符等宽字符）

    位置标识符

    * "."：当前光标位置

    * "'" 或 ""`：上一次跳转位置

    * "<" 或 ">"：最近可视模式选择的开始/结束位置

    * "0"：上一次修改的位置

    * "^"：上一次插入模式结束的位置

    * "$"：缓冲区最后一行

    * 标记：如 "'a"（小写标记仅当前缓冲区）、"'A"（大写标记全局）

    用法示例

    ```vim
    " 获取当前光标位置
    let pos = getpos(".")
    echo pos  " 输出类似 [1, 10, 5, 0]

    " 获取标记 'a 的位置
    let mark_pos = getpos("'a")

    " 获取上一次跳转位置
    let jump_pos = getpos("'")

    " 获取可视模式选择结束位置
    let visual_end = getpos(">")

    " 分解使用位置信息
    let [bufnum, lnum, col, off] = getpos(".")
    echo "行号: " . lnum
    echo "列号: " . col

    " 与 setpos() 配合使用
    let saved_pos = getpos(".")
    " ... 其他操作 ...
    call setpos(".", saved_pos)  " 恢复位置

    " 检查标记是否存在
    if getpos("'m")[1] != 0
        echo "标记 m 已设置"
    endif
    ```

    注意事项

    * `getpos()` 返回的列号是字节索引（从 1 开始），不是字符索引

    * 对于多字节字符（如中文），一个字符可能占用多个字节

    * 使用 getcurpos() 可获取字符索引的列位置

    * 未设置的标记返回 [0, 0, 0, 0]

    相关函数

    * setpos()：设置位置

    * getcurpos()：获取当前光标位置（字符索引）

    * col()：获取列号

    * line()：获取行号

* 常用的内置函数

    Vim 有许多强大的内置函数，以下分类介绍常用的函数：

    1. 缓冲区相关函数

        ```vim
        " 获取当前缓冲区编号
        let bufnum = bufnr('%')

        " 获取缓冲区名称
        let bufname = bufname('%')

        " 获取所有缓冲区列表
        let buflist = getbufinfo()

        " 判断缓冲区是否修改过
        if getbufinfo(bufnr('%'))[0].changed
            echo "缓冲区已修改"
        endif

        " 设置缓冲区选项
        call setbufvar(bufnr('%'), '&modifiable', 1)

        " 获取行数
        let total_lines = line('$')
        ```

    2. 窗口和标签页相关

        ```vim
        " 获取当前窗口编号
        let winnum = winnr()

        " 获取窗口列表
        let winlist = getwininfo()

        " 获取当前标签页编号
        let tabnum = tabpagenr()

        " 获取标签页列表
        let tablist = gettabinfo()

        " 窗口高度和宽度
        let winheight = winheight(0)
        let winwidth = winwidth(0)
        ```

    3. 文本操作函数

        ```vim
        " 获取指定行
        let line_content = getline(5)

        " 设置指定行内容
        call setline(5, "新内容")

        " 添加行到缓冲区
        call append(10, ["新行1", "新行2"])

        " 删除行
        call deletebufline('%', 10, 15)  " 删除10-15行

        " 获取光标位置
        let [row, col] = [line('.'), col('.')]

        " 设置光标位置
        call cursor(10, 5)

        " 搜索文本
        let match_line = search('pattern', 'n')  " 不移动光标

        " 获取匹配位置
        let [match_line, match_col] = searchpos('pattern', 'n')
        ```

    4. 字符串处理函数

        ```vim
        " 字符串长度
        let len = strlen("string")

        " 子字符串
        let sub = strpart("hello world", 6, 5)  " world

        " 分割字符串
        let parts = split("a,b,c", ',')  " ['a','b','c']

        " 连接字符串
        let joined = join(['a','b','c'], '-')  " a-b-c

        " 转换大小写
        let upper = toupper("hello")
        let lower = tolower("HELLO")

        " 替换字符串
        let new_str = substitute("hello world", "world", "vim", "")

        " 匹配正则表达式
        if "hello" =~ '^h'
            echo "以h开头"
        endif

        " 格式化字符串
        let formatted = printf("行号: %d, 内容: %s", 10, getline(10))
        ```

    5. 列表和字典函数

        ```vim
        " 列表操作
        let list = [1, 2, 3]
        call add(list, 4)           " 添加元素
        let item = remove(list, 0)  " 删除元素
        let idx = index(list, 3)    " 查找索引
        let len = len(list)         " 长度
        call reverse(list)          " 反转
        call sort(list)             " 排序

        " 字典操作
        let dict = {'key': 'value'}
        let val = get(dict, 'key', 'default')  " 安全获取
        let keys = keys(dict)                  " 所有键
        let values = values(dict)              " 所有值
        let has_key = has_key(dict, 'key')     " 检查键是否存在

        " 映射函数
        let doubled = map([1,2,3], 'v:val * 2')  " [2,4,6]
        let filtered = filter([1,2,3,4], 'v:val > 2')  " [3,4]
        ```

    6. 文件系统函数

        ```vim
        " 检查文件是否存在
        if filereadable('/path/to/file')
            echo "文件可读"
        endif

        " 获取文件大小
        let size = getfsize('/path/to/file')

        " 读取文件内容
        let content = readfile('/path/to/file')

        " 写入文件
        call writefile(['line1', 'line2'], '/path/to/file')

        " 获取当前文件目录
        let dir = expand('%:p:h')

        " 文件名相关
        let full_path = expand('%:p')     " 完整路径
        let filename = expand('%:t')      " 仅文件名
        let extension = expand('%:e')     " 扩展名
        let directory = expand('%:p:h')   " 目录
        ```

    7. 时间和日期函数

        ```vim
        " 获取当前时间戳
        let timestamp = localtime()

        " 格式化时间
        let time_str = strftime('%Y-%m-%d %H:%M:%S')

        " 时间计算
        let future_time = localtime() + 3600  " 1小时后

        " 测量执行时间
        let start = reltime()
        " 执行代码...
        let elapsed = reltime(start)
        echo "耗时: " . reltimestr(elapsed)
        ```

    8. 正则表达式和模式匹配

        ```vim
        " 匹配函数
        let matched = match("hello world", "world")  " 返回匹配位置
        let list = matchlist("abc123def", '\([a-z]\+\)\d\+')  " 返回匹配列表

        " 替换
        let new_str = substitute("hello world", '\l\+', '\U\0', 'g')  " HELLO WORLD
        ```

    9. 用户交互函数

        ```vim
        " 获取用户输入
        let name = input("请输入姓名: ")

        " 确认对话框
        let choice = confirm("是否继续?", "&Yes\n&No\n&Cancel", 2)

        " 弹出选择列表
        let selection = inputlist(['选项1', '选项2', '选项3'])

        " 显示消息
        echo "普通消息"
        echomsg "带历史的消息"
        echoerr "错误消息"
        ```

    10. 系统相关函数

        ```vim
        " 执行系统命令
        let result = system('ls -la')

        " 获取环境变量
        let path = $PATH

        " 检查特性支持
        if has('python3')
            echo "支持Python3"
        endif

        " 获取Vim版本
        let version = v:version
        ```

    11. 常用函数组合示例

        ```vim
        " 1. 读取文件并在每行前添加行号
        function! AddLineNumbers()
            let lines = getline(1, '$')
            let numbered = map(range(1, len(lines)), 'printf("%4d: %s", v:val, lines[v:val-1])')
            call setline(1, numbered)
        endfunction

        " 2. 查找并高亮所有匹配
        function! HighlightAll(pattern)
            let matches = []
            let line = 1
            while line <= line('$')
                let col = match(getline(line), a:pattern)
                while col != -1
                    call add(matches, [line, col + 1])
                    let col = match(getline(line), a:pattern, col + 1)
                endwhile
                let line += 1
            endwhile
            return matches
        endfunction

        " 3. 批量替换
        function! BatchReplace(pattern, replacement)
            let start_line = line("'<")
            let end_line = line("'>")
            
            for lnum in range(start_line, end_line)
                let line_content = getline(lnum)
                let new_line = substitute(line_content, a:pattern, a:replacement, 'g')
                call setline(lnum, new_line)
            endfor
        endfunction
        ```

    12. 特殊变量

        ```vim
        " 当前文件名
        let fname = expand('%')

        " 寄存器内容
        let reg_content = @a      " 寄存器a
        let clipboard = @*        " 系统剪贴板

        " 上次搜索模式
        let last_search = @/

        " 最近执行的命令
        let last_cmd = @:

        " Vim信息变量
        echo v:version      " Vim版本
        echo v:servername   " 服务器名
        echo v:count        : 命令前的计数
        echo v:errmsg       " 错误消息
        ```

    这些函数组合使用可以创建强大的 Vim 脚本功能。建议通过 :help function-list 查看完整函数列表。

#### 变量

* `s:local_var` 和 `l:var_name` 有什么区别

    1. `l:`（局部变量）

        * 作用域：仅在当前函数内部有效

        * 用途：函数内的局部变量，类似于其他语言的局部变量

        * 生命周期：函数调用时创建，函数返回后销毁

        示例：

        ```vim
        function! MyFunc()
            let l:count = 10      " 局部变量，只能在函数内访问
            let g:global = 20     " 全局变量，随处可访问
            echo l:count          " 输出 10
        endfunction

        call MyFunc()
        echo l:count             " 错误！l:count 在此不可访问
        echo g:global            " 输出 20
        ```

    2. `s:`（脚本局部变量）

        作用域：仅在当前脚本文件内有效（跨函数共享）

        用途：脚本内部共享数据，但对其他脚本隐藏（类似模块私有变量）

        生命周期：Vim 会话期间持续存在

        示例：

        ```vim
        " myplugin.vim
        let s:initialized = 0     " 脚本局部变量

        function! s:Init()
            if s:initialized == 0
                let s:initialized = 1
                echo "首次初始化"
            endif
        endfunction

        function! MyPluginFunc()
            call s:Init()         " 可以访问 s:initialized
            echo s:initialized    " 输出 1
        endfunction
        ```

    example:

    ```vim
    " 示例：简单的缓存系统
    let s:cache = {}  " 脚本级缓存字典

    function! s:GetCachedData(key)
        if has_key(s:cache, a:key)
            echo "从缓存读取"
            return s:cache[a:key]
        endif
        
        " 模拟计算
        let l:result = a:key . "_processed"  " l: 局部变量
        let s:cache[a:key] = l:result        " s: 脚本变量，跨调用保持
        
        echo "首次计算"
        return l:result
    endfunction

    function! MyFunction()
        echo s:GetCachedData("test1")  " 首次调用：计算
        echo s:GetCachedData("test1")  " 第二次：从 s:cache 读取
        echo s:GetCachedData("test2")  " 新键：计算
    endfunction
    ```

    **重要细节**

    * 前缀省略：在函数内，局部变量有时可以省略 l: 前缀：

        ```vim
        function! Test()
            let count = 5    " 隐式使用 l:count
            echo count       " 等同于 echo l:count
        endfunction
        ```

        但为了清晰，建议显式使用 l:。

    * `a:` 前缀：函数参数必须使用 `a:` 前缀：

        ```vim
        function! Add(x, y)
            return a:x + a:y  " 正确
            return x + y      " 错误！找不到变量 x
        endfunction
        ```

    * 跨脚本访问 `s:`（不推荐但可能）：

        ```vim
        " 脚本 A: let s:secret = 42
        " 脚本 B: 无法直接访问 scriptA#secret
        " 但可通过 <SID> 和函数间接访问
        ```

    **最佳实践**

    * 函数内部：

        * 临时计算用 l:（或省略前缀）

        * 参数用 a:

        * 修改选项用 &

            注：

            1. 没看明白这个是什么意思

    * 脚本开发：

        * 内部状态用 s:

        * 用户配置用 g:

        * 用 s: 函数作为私有函数：

            ```vim
            function! s:Helper()  " 私有函数，只在本脚本内可用
            endfunction
            ```

    * 避免污染全局：

        * 除非必要，不用 g: 存储临时数据

        * 用 s: 替代全局变量，避免命名冲突

* s 变量必须在函数外部创建吗，可以在函数内部创建吗

    `s:` 变量可以在函数内部创建，但是有重要的注意事项：

    1. 可以在函数内创建 s: 变量

        ```vim
        function! s:Initialize()
            " 在函数内部创建 s: 变量是完全合法的
            let s:counter = 0
            let s:config = {'auto_save': 1}
        endfunction

        function! s:Increment()
            " 这里 s:counter 可能不存在，需要处理
            if !exists('s:counter')
                let s:counter = 0
            endif
            let s:counter += 1
            return s:counter
        endfunction
        ```

    2. 关键区别：何时初始化

        情况A：在脚本顶层初始化（推荐）

        ```vim
        " 脚本顶部初始化 - 清晰明确
        let s:initialized = 0
        let s:cache = {}

        function! s:DoSomething()
            " 直接使用，无需检查是否存在
            if s:initialized == 0
                " 初始化代码
                let s:initialized = 1
            endif
        endfunction
        ```

        情况B：在函数内初始化（惰性初始化）

        ```vim
        " 不在顶部初始化
        function! s:LazyInit()
            " 第一次调用时才创建
            if !exists('s:data_loaded')
                let s:data_loaded = 0
                let s:cache = {}
                echo "执行惰性初始化"
            endif
            
            if s:data_loaded == 0
                " 加载数据...
                let s:data_loaded = 1
            endif
        endfunction
        ```

    总结

    * 可以在函数内创建 s: 变量，但要小心访问顺序问题

    * 最佳实践：在脚本顶部初始化主要变量，在函数内只做惰性初始化

    * 总是用 exists() 检查变量是否存在（如果在函数内初始化）

* 全局变量指的是可以跨脚本使用的变量吗？

    Vimscript 中的全局变量确实可以跨脚本使用，但有不同的前缀和行为：

    1. 真正的全局变量：g:

        ```vim
        " 在任何脚本中定义
        let g:my_global = "可在任何地方访问"

        " 在其他脚本中使用
        echo g:my_global  " 输出 "可在任何地方访问"

        " 甚至可以在命令行直接设置
        " :let g:my_setting = 1
        ```

        特点：

        * 整个 Vim 实例中可见

        * 跨所有脚本、函数、插件

        * 会话间不持久（除非保存到 viminfo）

    **不同前缀的作用域对比**

    | 前缀 | 作用域 | 跨脚本访问 | 示例 |
    | - | - | - | - |
    | g: | 全局 | ✅ 完全可访问 | let g:plugin_enabled = 1 |
    | s: | 脚本内部 | ❌ 对其他脚本隐藏 | let s:cache = {} |
    | l: | 函数内部 | ❌ 仅限当前函数 | let l:temp = 42 |
    | a: | 函数参数 | ❌ 仅限当前函数 | function! Foo(a:arg) |
    | b: | 缓冲区 | ✅ 对访问该缓冲区的脚本可见 | let b:filetype = 'python' |
    | w: | 窗口 | ✅ 对访问该窗口的脚本可见 | let w:scroll_pos = 100 |
    | t: | 标签页 | ✅ 对访问该标签页的脚本可见 | let t:custom_title = 'Edit' |
    | v: | Vim 内置 | ✅ 全局只读变量 | echo v:version |

    **实际跨脚本使用示例**

    * 场景1：插件配置

        ```vim
        " 用户 vimrc 中设置
        let g:myplugin_theme = 'dark'

        " 在插件脚本中读取
        function! myplugin#LoadTheme()
            if exists('g:myplugin_theme')
                echo "使用主题：" . g:myplugin_theme
            else
                let g:myplugin_theme = 'light'  " 设置默认值
            endif
        endfunction
        ```

    * 场景2：脚本间通信

        ```vim
        " script1.vim
        let g:shared_data = {
            \ 'count': 0,
            \ 'users': ['Alice', 'Bob']
        \ }

        " script2.vim（完全不同的脚本）
        function! ProcessSharedData()
            if exists('g:shared_data')
                let g:shared_data.count += 1
                echo "当前计数：" . g:shared_data.count
            endif
        endfunction
        ```

    * 场景3：缓冲区局部变量跨脚本

        ```vim
        " 在 python 文件类型插件中
        autocmd FileType python let b:python_indent = 4

        " 在其他插件中也可以访问
        function! CheckPythonIndent()
            if exists('b:python_indent')
                echo "Python缩进：" . b:python_indent . "空格"
            endif
        endfunction
        ```

    **命名空间最佳实践**

    * 避免污染全局命名空间

        ```vim
        " 不好的做法：容易冲突
        let g:debug = 1        " 可能与其他插件冲突
        let g:count = 0        " 太通用了

        " 好的做法：使用前缀
        let g:myplugin_debug = 1
        let g:myplugin_counter = 0
        let g:myplugin#config = {}  " 或使用 # 分隔
        ```

    * 插件推荐的模式

        ```vim
        " 在插件初始化时
        if !exists('g:myplugin_config')
            let g:myplugin_config = {
                \ 'enable': 1,
                \ 'timeout': 1000,
                \ 'colors': ['red', 'green', 'blue']
            \ }
        endif

        " 在函数中使用默认值
        function! myplugin#GetConfig(key)
            if has_key(g:myplugin_config, a:key)
                return g:myplugin_config[a:key]
            else
                return s:defaults[a:key]  " 回退到脚本默认值
            endif
        endfunction
        ```

    **特殊作用域变量的跨脚本访问**

    * 缓冲区变量 b: 的跨脚本

        ```vim
        " 脚本A：设置缓冲区变量
        autocmd BufEnter *.py let b:python_mode = 'strict'

        " 脚本B：读取（需要处于同一缓冲区）
        function! GetPythonMode()
            if &filetype == 'python' && exists('b:python_mode')
                return b:python_mode
            endif
            return 'default'
        endfunction
        ```

    * 窗口变量 w: 的跨脚本

        ```vim
        " 记录窗口滚动位置
        autocmd CursorMoved * let w:last_line = line('.')

        " 另一个脚本恢复位置
        function! RestoreLastPosition()
            if exists('w:last_line')
                execute 'normal! ' . w:last_line . 'G'
            endif
        endfunction
        ```

    **重要的注意事项**

    * 变量是否存在检查：

        ```vim
        " 总是检查全局变量是否存在
        if exists('g:myplugin_initialized')
            " 已初始化
        else
            " 首次运行
        endif
        ```

    * 避免循环依赖：

        ```vim
        " script1.vim
        if !exists('g:shared')
            let g:shared = {}
        endif
        let g:shared.value1 = ComputeValue()  " 可能依赖 script2

        " script2.vim
        if !exists('g:shared')
            let g:shared = {}
        endif
        let g:shared.value2 = ComputeValue2()  " 可能依赖 script1
        ```

    * 性能考虑：

        ```vim
        " 频繁访问的全局变量，可缓存到局部
        function! OptimizedFunction()
            let l:config = g:myplugin_config  " 缓存到局部变量
            for i in range(1000)
                " 使用 l:config 而不是每次 g:myplugin_config
                if l:config.enable
                    " ...
                endif
            endfor
        endfunction
        ```


### 插入、删除、修改、替换

* `gU`

    gU 是一个操作符，需要配合移动命令来使用。

    examples:

    * `gUU` 或 `gUgU`: 将当前行转为大写。

    * `gUw`: 将从光标处到下一个单词结尾的文本转为大写。

* normal 模式下进入 insert 模式的方法

    * 常用插入命令

        * `I`: 在当前行的**第一个非空白字符**处开始插入。

        * `A`: 在当前行的**末尾**开始插入。

            等价于直接按 `$` 移动到行尾再按 `a`。

    * 常用替换命令

        * `s` (Substitute): 删掉光标所在位置的**字符（Character）**，并进入插入模式。

            等价于`x`（删除字符） +  `i`（插入）。

            * `[数字]s`: 删除光标下的指定数量的字符，并进入插入模式。

            * **`S` (大写)**：删除**整行**内容，并进入插入模式（同时会保留该行的缩进）。

                * 等价于 `cc` 命令。

            注：

            1. 个人感觉`s`没啥用，因为完全可以用`[n]xi`或者`xi`完成同样的功能。

        * `c` (Change)

            `c` 针对的是**动作（Motion）或文本对象（Text Object）**

            `c` 是**操作符（Operator）**之一。在 Vim 的“动词 + 名词”语法中，`c` 是动词（更改），它必须配合一个表示范围的名词（动作或文本对象）才能发挥作用。

            * 常用操作（配合移动/动作）：

                * `cw` (change word)： 从光标位置删除到单词结尾（包含标点），并进入插入模式。

                * `ce` (change end)： 从光标位置删除到单词结尾（不包含单词后的空格），并进入插入模式。
                
                * `c$`： 从光标位置删除到**行尾**，并进入插入模式。

                    * 等价于`C` (大写)

                * `c0`： 从光标位置删除到**行首**，并进入插入模式。

                * `cc`： 删除**整行**，保留缩进，并进入插入模式。

            * 高级操作（配合文本对象）

                这是 `c` 命令最能提升效率的地方。`i` 代表 inner（内部），`a` 代表 around（包含周围空格或符号）。

                * `ciw` (change inner word)： 无论光标在单词的哪个位置，都会**更改整个单词**。

                * `ci"`： 更改双引号 `""` **里面**的内容（写代码时修改字符串的神技）。

                * `ca"`： 更改双引号以及引号里面的内容（连同引号一起删掉并修改）。

                * `ci(` 或 `cib`： 更改括号 `()` 内部的内容。

                * `cit`： 更改 HTML/XML 标签（Tag）内部的内容。

#### `d`删除

* `d`命令删除字符的常见用法

    * `dtx`: 删除直到字符 'x' 前

    * `dfx`: 删除直到字符 'x'（包括 'x'）

    * `dTx`: 向后删除直到字符 'x' 前

    * `dFx`: 向后删除直到字符 'x'（包括 'x'）

    * 文本对象操作

        * `diw`: 删除当前单词（光标在单词任意位置）

        * `daw`: 删除当前单词（包括周围空格）

        * `di"`: 删除引号内的内容

        * `da"`: 删除引号及内容

        * `di(` 或 `dib`: 删除括号内的内容

        * `da(` 或 `dab`: 删除括号及内容

        * `dit`: 删除 HTML/XML 标签内的内容

        * `dat`: 删除整个 HTML/XML 标签

* 常用技巧

    * 搜索和删除

        * d/pattern - 删除直到匹配 pattern 的位置

        * d?pattern - 向后删除直到匹配 pattern 的位置

    * 结合数字使用

        * d2t. - 删除直到第二个 '.' 前

        * 3df - 删除直到第三个空格（包括空格）

    实际例子：

    假设文本：hello.world foo.bar test.example end

    * 光标在 foo 的 'f'：dt → 删除 foo.bar

    * 光标在 foo 的 'f'：df → 删除 foo.bar

    * 光标在引号内：di" → 删除 "hello world" 中的 hello world

    * 光标在单词任意位置：daw → 删除整个单词

    这些命令中的 d 都可以换成 c（删除并进入插入模式）或 y（复制），非常灵活！

#### `:s`替换

* vim 替换命令的基本结构

    `:s/pattern/replace/flags`

* vim 替换中的`\zs`和`&`

    `\zs` - 设置匹配开始位置

    作用：在匹配模式中标记“真正匹配开始的位置”，前面的部分只作为匹配条件但不被替换。

    示例：

    ```vim
    :%s/\d\+\zs\.\d\+/,/
    ```

    * 匹配模式：\d\+\.\d\+（如 123.456）

    * \zs 前的 \d\+ 只是匹配条件

    * 实际替换的是 \zs 后的 \.\d\+

    * 结果：123,456（只替换小数点后的 .456 为 ,456）

    另一个例子：

    ```vim
    :%s/\w\+\zs\s\+//g
    ```

    * 删除每个单词后的空格，但保留单词本身

    & - 引用匹配的整个文本

    作用：在替换部分引用整个匹配的模式。

    示例：

    ```vim
    :%s/\w\+/[&]/g
    ```

    * 把每个单词用方括号括起来

    * hello world → [hello] [world]

    ```vim
    :%s/\d\+/&%/g
    ```

    * 在每个数字后加百分号

    * 123 test 456 → 123% test 456%

    组合使用示例

    ```vim
    :%s/\d\+\zs\.\d\+/&0/g
    ```

    * 匹配：数字.数字

    * \zs 前只作为条件

    * & 引用匹配部分（.数字）

    * 结果：给小数点后的数字加 0

    * 12.3 → 12.30

    这两个标记让 Vim 的替换更加灵活精确，可以精确控制匹配和替换的范围。


* vim 替换中的 normal 模式

    * 配合 `:global` 命令: `:g/pattern/normal I*`

        含义：在所有匹配到 pattern 的行上，执行“在行首插入星号”的操作。

        效果：这常用于批量给某些行加注释或列表标记。

    * 在替换式中使用表达式: `s/pattern/\=.../`，这属于高级用法。例如： `:%s/pattern/\=execute('normal I*')/n`

        含义：找到匹配项后，不直接通过替换文本来修改，而是调用 execute 函数去触发一段普通模式指令。

* vim 替换命令的 example: 为每个非空行行首添加星号

    `vnoremap <leader>l :s/^\s*\zs\S.*$/* &/<CR>`

    这行 Vim 命令的作用是 在可视模式下，给选中的每一行添加 Markdown 风格的列表标记。

    具体解析：

    * `vnoremap`：在可视模式下创建映射

    * `<leader>l`：当按下 <leader> 键（通常是 \）和 l 键时触发

    * `:s/^\s*\zs\S.*$/* &/`：替换命令

        * `^\s*`：匹配行首的空白字符

        * `\zs`：设置匹配开始位置（不替换前面的空白）

        * `\S.*$`：匹配第一个非空白字符到行尾的内容

        * `* &`：替换为 * 加上原匹配内容（& 代表匹配的内容）

    * `<CR>`：回车执行命令

* Vim 替换命令中其他有用的标记

    **替换部分特殊标记**

    1. `\1`, `\2`, ..., `\9` - 引用捕获组

        ```vim
        :%s/\(\w\+\) \(\w\+\)/\2 \1/g
        " 交换两个单词的位置
        " hello world → world hello

        :%s/\(\d\+\)-\(\d\+\)/\2.\1/g
        " 交换用连字符分隔的数字
        " 123-456 → 456.123
        ```

    2. `\0` - 引用整个匹配（与 & 相同）

        ```vim
        :%s/\w\+/item: \0/g
        " 每个单词前加 "item: "
        ```

    3. `~` - 使用上次替换的字符串

        ```vim
        " 第一次替换：
        :%s/foo/bar/g
        " 第二次替换可以使用 ~ 代表 "bar"：
        :%s/baz/~/g  " 实际替换为 baz → bar
        ```

    模式部分的特殊标记

    4. \ze - 设置匹配结束位置（与 \zs 对应）

        ```vim
        :%s/foo\zebar/BAR/g
        " 匹配 "foobar"，但实际匹配结束在 foo 后
        " 只替换 "foo"，不替换 "bar"
        " foobar → BARbar
        ```

    5. \zs 和 \ze 组合使用

        ```vim
        :%s/\w\+\zs\d\+\ze\w\+/###/g
        " 匹配：单词+数字+单词，但只替换数字部分
        " abc123def → abc###def
        ```

    6. `\@<=` - 正向回顾（positive lookbehind）

        ```vim
        :%s/\(\w\)\@<=x/X/g
        " 替换前面是单词字符的 x
        " axb cxd → aXb cXd
        " 但 xyz 中的 x 不变（前面不是单词字符）
        ```

    7. `\@<!` - 负向回顾（negative lookbehind）

        ```vim
        :%s/\(\d\)\@<!\./点/g
        " 替换前面不是数字的点
        " a.b → a点b
        " 但 1.5 中的 . 不变（前面是数字）
        ```

    8. `\@=` - 正向预览（positive lookahead）

        ```vim
        :%s/foo\(bar\)\@=/FOO/g
        " 替换后面是 bar 的 foo
        " foobar → FOObar
        " foobaz 不变
        ```

    9. `\@!` - 负向预览（negative lookahead）

        ```vim
        :%s/foo\(bar\)\@!/FOO/g
        " 替换后面不是 bar 的 foo
        " foobaz → FOObar
        " foobar 不变
        ```

    替换标志（flags）

    10. \c 和 \C - 忽略/区分大小写

        ```vim
        :%s/foo\c/bar/g  " 忽略大小写，匹配 foo, FOO, Foo 等
        :%s/foo\C/bar/g  " 区分大小写，只匹配 foo
        ```

    11. \v - 使用非常魔法模式

        ```vim
        :%s/\v(\w+)\s+(\w+)/\2 \1/g
        " 不需要转义括号
        " 对比：%s/\(\w\+\)\s\+\(\w\+\)/\2 \1/g
        ```

    12. \V - 使用原义模式

        ```vim
        :%s/\Va.b/a.b/g
        " 点就是字面意义的点，不是正则的 .
        " 匹配 "a.b" 而不是 "a任意字符b"
        ```

    其他有用标记

    13. \e - 结束替换字符串

        ```vim
        :%s/foo/\=toupper(submatch(0))\e bar/g
        " \e 后面的 " bar" 是普通文本
        " 结果：在匹配的 foo 转为大写后加 " bar"
        " foo → FOO bar
        ```

    14. \U, \L, \E - 大小写转换

        ```vim
        :%s/foo/\U&/g      " foo → FOO
        :%s/FOO/\L&/g      " FOO → foo
        :%s/foo\w*/\U&\Ebar/g  " \U 开始，\E 结束
        " foobar → FOOBARbar
        ```

    15. \u, \l - 转换下一个字符大小写

        ```vim
        :%s/foo/\u&/g      " foo → Foo
        :%s/FOO/\l&/g      " FOO → fOO
        ```

    16. \= - 使用 Vim 脚本表达式

        ```vim
        :%s/\d\+/\=(submatch(0)*2)/g
        " 所有数字乘以 2

        :%s/\w\+/\=toupper(submatch(0))/g
        " 所有单词转大写

        :%s/.*/\=printf("%03d", submatch(0))/g
        " 格式化为 3 位数字
        ```

    实用组合示例

    ```vim
    " 1. 在每行行首添加行号
    :%s/^/\=printf("%3d: ", line('.'))/

    " 2. 将下划线命名转为驼峰
    :%s/_\(\w\)/\u\1/g

    " 3. 提取 email 用户名
    :%s/^\(\w\+\)@.*$/\1/

    " 4. 智能缩进调整
    :%s/^\(\s*\)/\=repeat(' ', len(submatch(1)) + 4)/

    " 5. 转义 HTML 特殊字符
    :%s/[&<>"]/\={'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[submatch(0)]/g
    ```

* 替换命令`:s`里常用的特殊符号含义

    在 Vim 的 `:substitute` (简写为 `:s`) 命令中，除了 `&` 之外，确实还有一系列非常有用的特殊符号。它们能极大地简化复杂的文本重组工作。

    我们将这些符号分为 **“匹配到的文本”** 和 **“特殊格式控制”** 两类：

    1. 引用匹配内容的符号

        除了 `&` 表示“匹配到的全部文本”外，最常用的是**捕获组**引用：

        * `\0`: 效果等同于 `&`，代表整个匹配项。

        * `\1`, `\2` ... `\9`: 引用正则表达式中用 `\(` `\)`（或扩展模式下的 `()`）括起来的第 n 个分组。

            * 例子：`:s/\(\w+\) \(\w+\)/\2 \1/` 会把 "hello world" 交换顺序变成 "world hello"。

        * `~`: 代表**上一次**替换命令中使用的替换字符串。

            * 例子： 如果你刚才执行了 `:s/A/B/`，现在执行 `:s/C/~/`，则 C 会被替换成 B。

        * replace the first two digits of our years

            `:%s/\v\d\d(\d\d)/19\1`

    2. 改变大小写的控制符

        这些符号通常放在替换字符串的前面，用来转换匹配内容的大小写：

        * **`\u`**: 将下一个字符转为**大写**。
        * **`\l`**: 将下一个字符转为**小写**。
        * **`\U`**: 将其后的所有字符转为**大写**，直到遇到 `\e` 或 `\E`。
        * **`\L`**: 将其后的所有字符转为**小写**，直到遇到 `\e` 或 `\E`。
        * **`\e` 或 `\E`**: 结束 `\U` 或 `\L` 的转换范围。

        用法示例：

        * `:s/\w\+/\u&/g`：将匹配到的单词首字母大写。
        * `:s/hello/\U&/`：将 "hello" 变成 "HELLO"。

    3. 特殊字符与表达式

        * **`\r`**: 插入一个**换行符**（注意：在替换部分用 `\n` 通常会插入一个空字符 NULL，换行必须用 `\r`）。
        * **`\t`**: 插入一个制表符 (Tab)。
        * **`\\`**: 插入一个反斜杠原义字符。
        * **`\n`**: 在替换部分代表一个 Null 字符（但在搜索部分代表换行）。
        * **`\=`**: **（非常强大）** 后接一个 Vim 脚本表达式。
            * *例子*：`:s/\d\+/\=submatch(0) + 1/` 会将匹配到的数字全部加 1。

    **总结对照表**

    | 符号 | 含义 |
    | :--- | :--- |
    | **`&`** 或 **`\0`** | 整个匹配到的文本 |
    | **`\1` - `\9`** | 第 1 到第 9 个捕获组的内容 |
    | **`\u` / `\U`** | 下个字符大写 / 之后全部大写 |
    | **`\l` / `\L`** | 下个字符小写 / 之后全部小写 |
    | **`\r`** | 换行符 (CR) |
    | **`\=`** | 执行 Vim 脚本表达式 |

    **小贴士**：如果你在执行替换时发现某些特殊符号（比如 `/` 或 `&`）太多导致写起来很麻烦，你可以更换替换命令的分隔符，例如使用 `:s#http://#https://#g`，这样就不需要给斜杠加反斜杠转义了。

    你是在处理大批量的日志格式化，还是在进行代码重构？如果涉及到复杂的逻辑，`\=` 表达式配合 `submatch()` 简直是黑科技般的存在。

* vim 中，`s` 前的百分号与其他表示范围的特殊符号

    `:%s` 前面的 % 表示在整个文件中执行替换，即对所有行进行操作。

    % 的含义

    % 是 Vim 中的一个范围符号，代表整个文件或当前缓冲区。

    ```vim
    :%s/pattern/replacement/g
    " 在整个文件中进行替换
    ```

    其他常用范围符号

    1. % - 整个文件

        ```vim
        :%s/foo/bar/g     " 替换文件中所有 foo
        ```

    2. 行号范围

        ```vim
        :10,20s/foo/bar/g  " 只在第10-20行替换
        :1,$s/foo/bar/g    " 从第1行到最后一行（同 %）
        :.s/foo/bar/g      " 只在当前行替换
        ```

    3. 'm - 标记位置

        ```vim
        :'a,'bs/foo/bar/g  " 在标记 a 到标记 b 之间替换
        ```

    4. 模式匹配范围

        ```vim
        :/^Start/,/^End/s/foo/bar/g
        " 从匹配 "Start" 的行到匹配 "End" 的行
        ```

    5. . - 当前行

        ```vim
        :.s/foo/bar/g      " 仅当前行
        :.,+5s/foo/bar/g   " 当前行及后面5行
        ```

    6. $ - 最后一行

        ```vim
        :.,$s/foo/bar/g    " 从当前行到最后一行
        ```

    实用示例对比

    ```vim
    " 不同范围的替换示例
    :%s/old/new/g        " 整个文件
    :1,50s/old/new/g     " 第1-50行
    :/begin/,/end/s/o/n/g # 从包含"begin"到包含"end"的行
    :'<,'>s/old/new/g    " 在可视模式选中的行（自动生成）
    :.,+3s/old/new/g     " 当前行及后面3行
    ```

    如果省略 % 会怎样？

    如果只写 :s 而不指定范围：

    ```vim
    :s/foo/bar/g
    ```

    这只会在当前行进行替换，不是整个文件！

    记忆技巧

    * % 像 "百分百" → 100% 的行都要处理

    * % 也像 "全部" (All) 的意思

    * :%s 是全局替换的标准写法

    所以你的例子：

    ```vim
    :%s/\d\+\zs\.\d\+/,/
    ```

    会在整个文件中查找匹配 \d\+\.\d\+（数字.数字）的模式，但只替换 \zs 之后的部分（即 .数字 部分），将其替换为 ,（保持 \zs 前的数字不变）。

* `:s`中的范围选择

    example: `:'<,'>s/.*/```&```/`

    解释：

    * `'<,'>`：表示选中的范围

    * `s/.*/```&```/`：将整行替换为三个反引号 + 原内容 + 三个反引号

    * `&`: 表示匹配到的整个文本

### 搜索、正则表达式、替换

* vim 中的 regex 构建 group 时，括号需要加`\`(parentheses)：`\(key-words\)`，但是其它常用的 regex 都不需要。

    在 regex 前加`\v`表示 very magic，即所有可能被认为是 metacharacter 的字符 ，都会被判定为 metacharacter。

    这样上述的 regex 就可以写成`\v(key-worlds)`。此时如果我们需要匹配`(`和`)`，那么我们需要对它们进行转义：`\v\(key-words\)`。

* 讲讲 vim 中 * 和 {-} 的匹配含义，Match the preceding (meta)character 0 or more time.

    在 Vim 的正则表达式中，你提到的 `*` 和 `{-}` 确实都表示“匹配前面的字符 0 次或多次”。

    但它们之间有一个至关重要的区别：**`*` 是“贪婪的”（Greedy），而 `{-}` 是“非贪婪的”（Non-greedy / Lazy）**。一个是“大胃王”，能吃多少吃多少；另一个是“点到为止”，够用就行。

    1. `*` —— 贪婪匹配（尽可能多）

        `*` 会吞掉尽可能多的字符，直到不满足匹配条件为止。它会一路向后狂奔，直到整行的最后一个匹配点。

        * **语法**：在 Vim 中直接写 `*` 即可。
        * **特点**：从左边开始匹配，一直匹配到**最右边**符合条件的终点。

    2. `{-}` —— 非贪婪匹配（尽可能少）

        `{-}` 相当于其他现代正则引擎（如 Python、JavaScript、PCRE）中的 `*?`。它非常懒惰，只要一达到匹配的最低要求（哪怕只有 0 次或 1 次），它就会立刻停下来。

        * 语法：

            * 在 Vim 的**默认模式**下，你需要加反斜杠：`\{-}`
            * 如果在 **Very Magic 模式**（即加了 `\v`）下，可以像你写的那样直接写 `{-}`

        * 特点：从左边开始匹配，只要遇到**第一个**符合条件的终点就立刻收工。

    **实战对比：一眼看懂区别**

    假设我们有下面这行 HTML 文本：

    ```html
    <h6>坚持学习</h6><h6>天天向上</h6>
    ```

    我们想要匹配 `<h6>...</h6>` 标签及其内部的内容。

    **表现 1：使用贪婪匹配 `*`**

    * **模式**：`<h6>.*</h6>`
    * **匹配结果**：
    `<strong><h6>坚持学习</h6><h6>天天向上</h6></strong>`
    * **为什么？**
    `.*` 看到整行最后还有一个 `</h6>`，于是它直接把中间的 `</h6><h6>` 也当成普通字符给“吞”掉了，直接一路匹配到了大后期。

    **表现 2：使用非贪婪匹配 `{-}`**

    * **模式**：`<h6>.\{-}</h6>` （或 `\v<h6>.{-}</h6>`）
    * **匹配结果**（会匹配到两个独立的结果）：
    1. `<strong><h6>坚持学习</h6></strong>`
    2. `<strong><h6>天天向上</h6></strong>`

    * **为什么？**

    `.\{-}` 只要一看到紧接着出现了 `</h6>`，就立刻停止当前的匹配。所以它能精准地把两个标签分别揪出来。

    **总结与避坑指南**

    | 特性 | `*` | `{-}` |
    | --- | --- | --- |
    | **匹配次数** | 0 次或多次 | 0 次或多次 |
    | **匹配策略** | 贪婪（尽可能多） | 非贪婪（尽可能少） |
    | **现代正则对应** | `*` | `*?` |
    | **Vim 默认写法** | `*` | `\{-}` (别漏了反斜杠) |

    > **Vim 小贴士**： 在 Vim 中用非贪婪匹配时，很多人容易漏掉反斜杠写成 `{-}`，导致 Vim 把它当成了普通字符串处理。如果不喜欢写反斜杠，记得在正则开头加上 `\v`（Very Magic 模式），例如：`/\v.{-}`。

    vim 中贪婪和非贪婪匹配的对照表：

    | Greedy quantifier | Non-greedy quantifier | Description |
    | - | - | - |
    | `*` | `{-}` | Match the preceding (meta)character 0 or more time. |
    | `+` | `{-1,}` | Match the preceding (meta)character 1 or more time. |
    | `=` | `{-0,1}` | Match the preceding (meta)character 0 or 1 time. |
    | `{n,m}` | `{-n,m}` | Match the preceding (meta)character from n to m times. |
    | `{n}` | `{-n}` | Match the preceding (meta)character exactly n times. |
    | `{,m}` | `{-,m}` | Match the preceding (meta)character from 0 to m. |

* vim 中，我希望匹配 ## hello world，我使用 #+ +hello 为什么不行

    这个困惑非常经典！你在 Vim 中使用 `#+ +hello` 无法匹配 `## hello world`，主要有两个核心原因：**Vim 默认的正则表达式模式（Magic）** 以及 **对空格匹配的误解**。

    我们直接来看看为什么不行，以及怎么改。

    **原因拆解**

    1. `+` 没有被当成“一个或多个”，而被当成了字面量

        Vim 默认的正则模式叫做 **Magic 模式**。在这个模式下，大部分特殊符号（比如 `+`、`?`、`{`、`|`）都必须加反斜杠 `\` 转义，才能获得它们在现代正则（如 PCRE/Python）中的特殊含义。

        * 在你的写法中，`#+` 意味着：匹配一个井号 `#`，后面**紧跟一个真正的加号 `+**`。

    2. 空格的连写变成了单纯的字符匹配

        在正则中，如果你连写两个空格（或者像你写的 `+ ` 加号后面接空格），Vim 只会死板地去匹配对应的空格数量。

        * 你的第二段 ` +hello`（前面有个空格）意味着：匹配一个空格，后面**紧跟一个加号 `+**`，再紧跟 `hello`。

        所以，你的表达式 `#+ +hello` 实际上在寻找一段长得像 **`#++hello`** 的文本，这当然找不到 `## hello world`。

    **正确的解决方法**

    如果你想实现“匹配一个或多个 `#`，后面接一个或多个空格，再接 `hello`”，你有两种最简单好用的改法：

    **方法 A：使用 Vim 的“超级魔法”模式 `\v`（强烈推荐）**

    在正则最前面加上 `\v`（Very Magic），可以让 Vim 的正则变得和大多数现代编程语言一样，`+` 不需要加反斜杠就能表示“一个或多个”。

    ```text
    /\v#+ +hello
    ```

    * `#+`：匹配一个或多个 `#`
    * ` +`：匹配一个或多个空格
    * `hello`：匹配单词 `hello`

    **方法 B：在标准模式下给 `+` 加反斜杠**

    如果你不想用 `\v`，就必须告诉 Vim 这两个 `+` 是特殊符号，写作 `\+`。

    ```text
    /#\+ \+hello
    ```

    **进阶小贴士**

    在实际写 Markdown 或代码时，空格往往不止是空格键，还可能包含制表符（Tab）。为了更稳妥地匹配“任意空白字符”，建议使用 `\s`：

    * **极其优雅的写法（Very Magic）：** `/\v#+\s+hello`
    * **传统写法：** `/#\+\s\+hello`

* 简述 vimscript 中正则表达式的用法

    在 Vimscript 中，正则表达式有**两种模式**，用法有重要区别：

    1. **两种模式区分**

        **魔术模式（Magic）**

        ```vim
        /hello\s\+world  " \+ 表示1个或多个空白字符
        ```
        - 大多数元字符有特殊含义：`. * ^ $ [ ] \`
        - 默认模式（搜索命令中使用）

        **字面模式（Literal/Very Magic）**

        ```vim
        \vhello\s+world  " 使用 \v 前缀，+ 不需要转义
        ```
        - `\v` 开启 "very magic" 模式
        - 几乎所有标点符号都有特殊含义（类似 Perl 正则）
        - 推荐使用，更清晰

    2. **常用语法**

        **基础匹配**

        ```vim
        " 字符串开头和结尾
        if str =~ '^start'     " 以 start 开头
        if str =~ 'end$'       " 以 end 结尾

        " 任意字符和数量
        'x.y'      " x任意单个字符y
        'xy*'      " x 后跟0个或多个 y
        'xy\+'     " x 后跟1个或多个 y (魔术模式)
        'xy+'      " x 后跟1个或多个 y (\v模式)
        'xy\?'     " x 后跟0个或1个 y
        'xy{2,4}'  " x 后跟2到4个 y
        ```

        **字符类**

        ```vim
        '[aeiou]'     " 任意元音字母
        '[0-9]'       " 任意数字
        '[a-zA-Z]'    " 任意字母
        '[^0-9]'      " 非数字（[^] 表示否定）
        '\d'          " 数字，同 [0-9]
        '\w'          " 单词字符 [a-zA-Z0-9_]
        '\s'          " 空白字符（空格、制表符等）
        ```

    3. **在 Vimscript 中的使用场景**

        **模式匹配操作符**

        ```vim
        " 匹配
        if str =~ pattern    " 匹配成功返回 1
        if str !~ pattern    " 不匹配时返回 1

        " 示例
        let str = "hello world"
        if str =~ '^hello'   " 真
        if str !~ 'goodbye$' " 真
        ```

        **字符串函数中使用**

        ```vim
        " substitute() - 替换
        let new = substitute(str, '\s\+', ' ', 'g')  " 多个空格变一个

        " matchstr() - 提取匹配部分
        let matched = matchstr(str, 'hel\?lo')  " 返回匹配的子串

        " match() - 返回匹配位置
        let pos = match(str, 'world')  " 返回索引（从0开始），未找到返回-1
        ```

    4. **转义规则示例**

        ```vim
        " 魔术模式（默认）需要转义：
        'hello\.world'    " 匹配 hello.world（. 需要转义）
        'example\d\+'     " example后跟1个或多个数字

        " \v 模式更简洁：
        \vhello\.world    " 匹配 hello.world（. 仍需转义，因为是字面点）
        \vexample\d+      " example后跟1个或多个数字（+ 不用转义）
        ```

    5. **实际应用示例**

        ```vim
        " 1. 验证邮箱格式
        if email =~ '\v^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            echo "有效邮箱"
        endif

        " 2. 提取数字
        let numbers = matchstr(text, '\v\d+')

        " 3. 分割字符串
        let parts = split(text, '\s*,\s*')  " 按逗号分割，忽略周围空格

        " 4. 替换操作
        let cleaned = substitute(text, '\v\s+', ' ', 'g')  " 多个空格合并为一个
        ```

    6. **重要提示**

        - 在 Vimscript 中**单引号字符串**更安全（双引号会解释 `\` 转义）
        - 推荐**总是使用 `\v` 模式**，更直观易读
        - 搜索时用 `/` 命令，脚本中用字符串函数
        - 使用 `:help pattern` 查看详细文档

    记住这个规则：**脚本中用 `\v`，搜索时可选，单引号更安全**。

* 简述 vim 中 if "hello" =~ '^h' 是什么意思

    在 Vim 脚本中，`if "hello" =~ '^h'` 是一个条件判断语句，它的意思是：

    **检查字符串 "hello" 是否匹配正则表达式 `'^h'`**

    具体解释：

    - `"hello"`：要检查的字符串
    - `=~`：Vim 中的模式匹配操作符（类似其他语言中的 `=~` 或 `match`）
    - `'^h'`：正则表达式，`^` 表示字符串开头，`h` 表示字符 'h'
    - 整个表达式判断字符串是否以 'h' 开头

    这个表达式会返回 **真（1）**，因为：

    - `"hello"` 确实以 'h' 开头

    注：

    1. 为什么这里以单引号开头？

    **其他相关操作符：**

    - `!~`：不匹配操作符
    - 例如：`if "hello" !~ '^x'` 返回真，因为 "hello" 不以 'x' 开头

    **简单示例：**

    ```vim
    if "hello" =~ '^h'
        echo "以 h 开头"
    else
        echo "不以 h 开头"
    endif
    ```
    
    输出：`以 h 开头`

    这是一个 Vim 脚本中常用的模式匹配方式，用于字符串验证和条件判断。

* vim 使用 / 搜索时，如何按 raw string 搜索？即我想要搜索的字符串就是字面意思，不是正则表达式的 pattern?

    在 Vim 中，你可以通过以下几种方式按字面意思（raw string）进行搜索，而不是将搜索字符串解释为正则表达式：

    方法1：使用 `\V`（非常神奇模式）

    在搜索模式前加上 `\V`：

    ```vim
    /\V要搜索的字面字符串
    ```

    例如：`/\Vfoo.bar` 会搜索字面值 "foo.bar"，而不会将 `.` 解释为正则表达式中的通配符。

    方法2：转义特殊字符

    在正则表达式中特殊字符前加上 `\`：

    ```vim
    /foo\.bar
    ```

    方法3：使用 `nomagic` 模式（旧方法）

    使用 `\M` 进行不神奇搜索（比 `\V` 少一些限制，但比普通正则更字面）：

    ```vim
    /\M要搜索的字面字符串
    ```

    方法4：使用 `search()` 函数（在脚本中）

    ```vim
    :call search('\V字面字符串')
    ```

    **特殊字符转义参考**

    以下字符在正则表达式中有特殊含义，需要转义（使用 `\`）：

    - `.`、`*`、`[`、`]`、`^`、`$`、`\`、`?`、`+`、`(`、`)`、`{`、`}`、`|`

    **实用技巧**

    如果你想永久使用字面搜索，可以在 `.vimrc` 中设置：

    ```vim
    " 设置搜索时默认使用字面匹配
    nnoremap / /\V
    nnoremap ? ?\V
    ```

    或者使用快捷键快速切换：

    ```vim
    " 按 \v 切换字面搜索
    nnoremap \v /\V
    ```

    **最推荐的方法是使用 `\V`**，因为它最直观且功能完整。

* 仅匹配单词

    `grep -P '\bvim\b' example.html`

    vim: `/\v<vim>`

* vim 中正则表达式使用替代项（alternatives）

    `/\vclass|href`

* vim 中特殊的正则表达式匹配规则

    | Metacharacter | Description |
    | - | - |
    | * | Matches the preceding (meta)character 0 or more time. |
    | + | Matches the preceding (meta)character 1 or more time. |
    | = | Matches the preceding (meta)character 0 or 1 time. |
    | {n,m} | Matches the preceding (meta)character from n to m times. |
    | {n} | Matches the preceding (meta)character exactly n times. |
    | {,m} | Matches the preceding (meta)character from 0 to m. |

    Note that the quantifier = doesn’t exist in the PCRE world.

* vim 中使用 / 搜索时，如何 实现类似 grep 的 \bxxx\b 的效果？

    在 Vim 中，实现类似 `grep` 或其他正则引擎中 `\b`（单词边界）的效果，主要有以下两种最常用的方式：

    1. 使用 `\<` 和 `\>` (最常用)
        
        Vim 使用尖括号配合转义符来表示词首和词尾。这是最标准的方法：

        * **`\<`**：匹配单词的**开头**。
        * **`\>`**：匹配单词的**结尾**。

        **操作示例：**
        如果你想搜索独立的单词 `cat`（不匹配 `category` 或 `copycat`），在 Vim 中输入：
        `/\<cat\>`

    ---

    2. 使用快捷键 `*` 或 `#` (最快技巧)
        
        如果你要搜索的目标单词已经在屏幕上出现了，你根本不需要手动输入这些符号：

        * 将光标移动到该单词上。
        * 按下 **`*`** (星号)：向下**全字匹配**搜索该单词。
        * 按下 **`#`** (井号)：向上**全字匹配**搜索该单词。

        > **小贴士**：按下 `*` 后，你可以输入 `/` 然后按“上方向键”，你会发现 Vim 自动为你补全了 `/\<单词\>` 的格式。

    ---

    3. 使用 `\v` (非常模式)
        
        如果你觉得输入反斜杠和尖括号太麻烦，可以使用 **Very Magic** 模式，它让 Vim 的正则语法更接近 Perl/Python/grep：

        * 输入：`/\v<cat>`
            * 在 `\v` 模式下，`<` 和 `>` 直接代表单词边界，不需要加反斜杠。

    ---

    **总结对比**

    | 工具/模式 | 语法 | 备注 |
    | :--- | :--- | :--- |
    | **grep / Standard Regex** | `\bcat\b` | 通用标准 |
    | **Vim 默认模式** | `/\<cat\>` | **推荐使用** |
    | **Vim 快捷键** | 光标停在词上按 `*` | **最高效** |
    | **Vim Very Magic** | `/\v<cat>` | 适合复杂正则 |

    你是想在编写代码时快速跳转到变量定义，还是在进行大范围的文本替换？

* 正则表达式中的 common POSIX character classes

    | Character class | Description | Equivalent |
    | - | - | - |
    | `[:alnum:]` | Uppercase and lowercase letters, as well as digits | `A-Za-z0-9` |
    | `[:alpha:]` | Uppercase and lowercase letters | `A-Za-z` |
    | `[:digit:]` | Digits from 0 to 9 | `0-9` |
    | `[:lower:]` | Lowercase letters | `a-z` |
    | `[:upper:]` | Uppercase letters | `A-Z` |
    | `[:blank:]` | Space and tab | `[ \t]` |
    | `[:punct:]` | Punctuation characters (all graphic characters except letters and digits)` | - |
    | `[:space:]` | Whitespace characters (space, tab, new line, return, NL, vertical tab, and form feed) | `[ \t\n\r\v\f]` |
    | `[:xdigit:]` | Hexadecimal digits | `A-Fa-f0-9` |

* `/\v[vim]`

    表示匹配 v, i, m 三个其中的一个。

* vim 输入 /xx 后立即开始搜索（增量搜索）

    **开启增量搜索**

    在 ~/.vimrc 中添加：

    ```vim
    " 开启增量搜索
    set incsearch
    ```

    开启后，输入 / 后每输入一个字符，Vim 会立即高亮显示匹配的位置。

    **高亮显示所有匹配项**

    ```vim
    " 高亮所有匹配项
    set hlsearch
    ```

    **相关设置**

    ```vim
    " 完整搜索相关配置
    set ignorecase    " 搜索忽略大小写
    set smartcase     " 搜索包含大写时区分大小写
    set incsearch     " 输入搜索模式时实时显示匹配
    set hlsearch      " 高亮显示所有匹配
    ```

* vim 实用的搜索技巧

    **清除高亮**

    ```vim
    :noh           " 清除搜索高亮
    :set nohlsearch " 禁用高亮
    :set hlsearch   " 重新启用高亮
    ```

    **搜索时跳转**

    * n：下一个匹配

    * N：上一个匹配

    * / + ↑/↓：浏览搜索历史

    **搜索特殊字符**

    ```text
    /特殊字符     " 使用反斜杠转义
    /\/path       " 搜索 /path
    /\<word\>     " 精确匹配单词
    ```

    **替换命令也适用**

    ```vim
    :%s/pattern/replace/gi  " g:全局，i:忽略大小写
    ```

    Vim 8.2+ 新增功能

    ```vim
    " 输入搜索词时显示匹配数
    set shortmess-=S
    ```

    我的推荐配置

    ```vim
    " ~/.vimrc 中搜索相关设置
    set ignorecase
    set smartcase
    set incsearch
    set hlsearch

    " 快速清除高亮快捷键
    nnoremap <silent> <C-l> :<C-u>nohlsearch<CR><C-l>
    " 按 Ctrl+l 清除高亮并重绘屏幕
    ```

    这样配置后，在 Vim 中：

    * 默认搜索大小写不敏感

    * 输入搜索词时立即显示匹配结果

    * 所有匹配项都会高亮显示

    * 可以快速清除高亮

* vim 实现搜索时大小写不敏感

    1. 搜索时大小写不敏感

        **临时设置**

        在搜索时添加 \c 前缀：

        ```text
        /\cpattern
        ```

        * 使用 \c：忽略大小写

        * 使用 \C：强制区分大小写

        **在搜索模式中设置**

        ```text
        /pattern\c
        ```

        效果相同，\c 放在模式前后都可以。

        **永久设置（推荐）**

        在 ~/.vimrc 中添加：

        ```vim
        " 设置搜索时默认忽略大小写
        set ignorecase
        set smartcase   " 智能大小写：如果搜索词包含大写字母，则区分大小写
        ```

        **切换命令**

        在 Vim 命令模式：

        ```vim
        :set ignorecase   " 忽略大小写
        :set noignorecase " 区分大小写
        :set ic           " 简写（ignorecase）
        :set noic         " 简写（noignorecase）
        ```

* vim `\v`

    \v 在 Vim 搜索中表示使用 "very magic" 模式，这是 Vim 正则表达式的一种特殊模式。

    Vim 正则表达式的四种模式：

    ```vim
    /pattern          " magic 模式（默认，有些字符有特殊含义）
    /\vpattern        " very magic 模式（大多数字符都有特殊含义）
    /\Vpattern        " very nomagic 模式（几乎不特殊，字面匹配）
    /\mpattern        " nomagic 模式（折中方案）
    ```

    `\v` 的作用：

    ```vim
    " 普通 magic 模式（默认）
    /\(\d\{3}\)-\d\{4}    " 匹配 (123)-4567
    " 需要转义很多特殊字符：\( \) \{ \}

    " very magic 模式
    /\v(\d{3})-\d{4}      " 匹配 (123)-4567
    " 几乎不需要转义，像其他语言的正则表达式
    ```

    特殊字符对比表:

    | 元字符 | magic 模式 | very magic 模式 | 说明 |
    | - | - | - | - |
    | `(`, `)` | 需要转义：`\(` `\)` | 不需要转义 | 分组 |
    | `{` `}` | 需要转义：`\{` `\}` | 不需要转义 | 重复次数 |
    | `+` | 需要转义：`\+` | 不需要转义 | 一个或多个 |
    | `?` | 需要转义：`\?` | 不需要转义 | 零个或一个 |
    | `\|` | 需要转义： `\\|` | 不需要转义 | 或 |
    | `^`, `$` | 不需要转义 | 不需要转义 | 行首/行尾 |
    | `.`, `*` | 不需要转义 | 不需要转义 | 任意字符/零个或多个 |

    注：

    1. 直接使用`/pattern`匹配，想要实现分组功能时，必须给括号加`\`：

        `/\(hello\).*\(world\)`

        其他的处理方式类似。

    examples:

    ```vim
    " 1. 匹配邮箱
    /\v\w+@\w+\.\w+                  " 简单邮箱匹配
    /\v[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}  " 更复杂的邮箱

    " 2. 匹配时间 (HH:MM)
    /\v\d{2}:\d{2}                   " 24小时制时间

    " 3. 匹配括号内的内容
    /\v\([^)]+\)                     " 匹配 (任意内容)

    " 4. 匹配 Markdown 标题
    /\v^#{1,6}\s+.+$                 " 匹配 # 标题

    " 5. 匹配 IP 地址
    /\v(\d{1,3}\.){3}\d{1,3}        " 匹配 192.168.1.1
    ```

    * 与其他模式的对比

        ```vim
        " 场景：匹配 "function(arg1, arg2)"

        " 1. very magic 模式（最简洁）
        /\vfunction\([^)]+\)

        " 2. magic 模式（默认，需要转义）
        /function\([^)]\+\)

        " 3. very nomagic 模式（字面匹配，需要转义特殊字符）
        /\Vfunction(arg1, arg2)          " 只能匹配这个具体字符串
        ```

    * tricks

        ```vim
        " 快速搜索替换中使用
        :%s/\v(\d+)-(\d+)/\2-\1/g       " 交换 123-456 为 456-123

        " 在搜索模式中使用变量
        let pattern = '\v\d{3}-\d{4}'
        execute '/' . pattern

        " 结合其他标志
        /\vpattern/i                     " 忽略大小写
        /\vpattern\c                     " 强制忽略大小写
        /\vpattern\C                     " 强制区分大小写
        ```

    * 建议

        * 推荐使用 \v：写起来更自然，与其他编程语言的正则表达式习惯一致

        * 特殊场景用 \V：当需要字面搜索包含特殊字符的字符串时

        * 保持一致性：在整个文件中使用相同的模式

* vim 中的范围匹配

    * `/\v[a-z]`

        匹配`a`到`z`中的一个字符。

    * `/[a-]`

        匹配`a`或`-`。

        `/[-z]`同理。

    * `/\v[0-9A-Z]`

        匹配多个范围。

    * `/\v[^abc]`

        匹配除了 a, b, c 外的所有字符中的一个

    * `[a^bc]`

### 导航、跳转、marker

* vim 打开文件后，跳转到上次关闭时候的位置：

    * 反引号 + 双引号：`` ` `` + `"`

    * 单引号 + 双引号：`'` + `"`

* vim 如何将光标移动到当前页的最上面/最下面？

    在 Vim 中，将光标移动到当前页的最上面或最下面有以下几种常用方法：

    1. 移动到当前屏（可见区域）

        * H → 移动到当前屏的最上面一行（High）

        * M → 移动到当前屏的中间一行（Middle）

        * L → 移动到当前屏的最下面一行（Low）

    2. 移动到当前页面（基于分页符）

        如果你说的"页"是指按分页符（\f）分隔的逻辑页：

        * { → 向前移动到上一个空白行（段落开头），通常是上一页

        * } → 向后移动到下一个空白行（段落开头），通常是下一页

    3. 滚动并保持光标位置

        * Ctrl + f → 向前翻一屏，光标相对位置不变

        * Ctrl + b → 向后翻一屏，光标相对位置不变

        * Ctrl + d → 向下翻半屏

        * Ctrl + u → 向上翻半屏

    4. 百分比定位

        * gg → 移动到文件最开头

        * G → 移动到文件最末尾

        * 50% → 移动到文件的50%位置

    实用技巧

    * 如果你经常需要操作屏幕顶部/底部的文本，可以配合数字使用：3H 移动到屏幕从上往下第3行

    * 使用 zt 将当前行置于屏幕顶部，zb 置于底部，zz 置于中间

    根据你的需求选择合适的命令：

    * 如果是当前可见屏幕：用 H、M、L

    * 如果是按分页的逻辑页：用 {、}

    * 如果是文件开头/结尾：用 gg、G

* 详细讲讲 vim 中 mark 的用法

    在 Vim 中，**Mark（标记）** 就像是书签，能让你在长文本中快速跳回特定的位置。对于处理大型代码文件或复杂文档，这是提高效率的“神技”。

    ---

    1. 基础操作：标记与跳转

        标记的基本语法非常简单，由 `m` 指令加上一个字母组成。

        **设置标记**

        * **格式：** `m{a-zA-Z}`
        * **操作：** 在普通模式下，按下 `m` 键，紧接着按下一个字母（如 `a`）。
        * **示例：** `ma` 会在当前光标位置设置一个名为 `a` 的标记。

        **跳转到标记**

        * **反引号 `a**：精确跳回到标记 `a` 的**行和列**（原位还原）。
        * **单引号 'a**：跳回到标记 `a` 所在行的**第一个非空字符**（更常用）。

    ---

    2. 标记的分类：局部 vs 全局

        Vim 的标记区分大小写，这是一个非常关键的设计。

        | 标记类型 | 范围 | 说明 |
        | --- | --- | --- |
        | **小写字母 (`a-z`)** | **单个缓冲区 (Buffer)** | 仅在当前文件内有效。每个文件都可以有一套独立的 `a-z`。 |
        | **大写字母 (`A-Z`)** | **跨文件 (Global)** | 全局标记。即使你切换了文件或重启了 Vim，`'A` 也会带你回到当初标记的那个文件的特定位置。 |

        > **提示：** 全局标记（大写字母）的数据存储在 `~/.viminfo`（或 `~/.shada`）中，因此它们是持久化的。

    ---

    3. Vim 自动生成的“特殊标记”

        除了手动设置，Vim 还会自动记录一些位置，这些标记非常实用：

        * **`` (双反引号)**：跳转回**上一次跳转前**的位置。在两次位置间快速切返。
        * **`. (点号)**：跳转到**最后一次修改**发生的位置。
        * **`^ (脱字符)**：跳转到**最后一次插入模式停止**的位置。
        * **`[** 和 **`]**：跳转到上一次修改或粘贴内容的**起始**和**结束**位置。
        * **`<** 和 **`>**：跳转到上一次**视觉模式 (Visual Mode)** 选中区域的起始和结束位置。
        * **`" (双引号)**：当你重新打开一个文件时，跳转到上次**关闭该文件时**光标所在的位置。

    ---

    4. 标记的管理与查看

        如果你忘记了设置过哪些标记，可以使用以下命令：

        * **`:marks`**：列出当前所有的标记（包括位置和对应的文本内容）。
        * **`:marks a b`**：仅查看标记 `a` 和 `b` 的信息。
        * **`:delmarks a`**：删除标记 `a`。
        * **`:delmarks a-z`**：删除所有小写标记。
        * **`:delmarks!`**：删除当前缓冲区的所有标记（不包括全局和数字标记）。

    ---

    5. 高阶用法：结合范围操作

        标记不仅可以用来“看”，还可以配合 **Operator（操作符）** 来精确控制编辑范围。

        * **`d'a`**：删除从当前行到标记 `a` 所在行的所有内容。
        * **`y'a`**：从当前位置复制到标记 `a`。
        * **`v'a`**：直接选中从当前位置到标记 `a` 的区域。

    ---

    **总结**

    1. **快速标记**：用 `ma` 记住这行。
    2. **快速跳回**：用 `'a` 回来。
    3. **跨文件操作**：用 `mA` 标记重要配置，在任何地方输入 `'A` 都能秒回。
    4. **神来之笔**：改完代码后跑去别处看逻辑，想回来继续改？直接 `'.`。

    掌握了 Mark，你在文件间的穿梭速度将从“步行”升级为“瞬移”。

* 那我如果想把一个 mark 赋值给另一个呢，比如复制粘贴后，把 [ 赋值给 a，把 ] 赋值给 b

    在 Vim 中，标记（Mark）本身并不支持像变量那样直接使用 `a = b` 这种语法进行“赋值”。

    **跳转到目标标记位置，然后再重新设置标记**的方式来实现这种效果：

    ```vim
    `[ma`]mb
    ```

    注意： 跳转时建议使用 **反引号 `** 而不是单引号 `'`，因为反引号能精确对齐到那一行的具体**列**，而单引号只对齐到行首。

    * 通过 Vim 脚本函数 `getpos()` 和 `setpos()`

        在 Vim 中，每个标记其实就是一个包含四个元素的列表：
        `[bufnum, lnum, col, off]`

        * **bufnum**：缓冲区编号（当前文件通常为 0）。
        * **lnum**：行号。
        * **col**：列号。
        * **off**：在使用 'virtualedit' 时的偏移量。

        实现代码:

        * 把 `[` 赋值给 `a`：

            ```vim
            :call setpos("'a", getpos("'["))
            ```

        * 把 `]` 赋值给 `b`：

            ```vim
            :call setpos("'b", getpos("']"))
            ```

        关键点： 在 `setpos()` 中使用标记名称时，必须带上**单引号**（例如 `"'a"`），这告诉 Vim 你要设置的是一个标记位置。

    如果需要拿到标记所在的缓冲区编号，可以使用 `getmarkpos()`（在较新版本的 Vim/Neovim 中可用）：

    ```vim
    " 获取标记 a 的位置信息，返回 [bufnum, lnum, col, off]
    let pos = getmarkpos("'[")
    call setpos("'a", pos)
    ```

    小提醒：`getpos()` 获取的是当前缓冲区的坐标，如果你在处理跨文件的全局标记（大写字母 A-Z），确保你在正确的 buffer 中操作，或者使用 `getmarkpos()` 来获取包含缓冲区信息的完整坐标。

* vim 中，visual 模式下选中 void func_1() 后，我希望快速跳转到 v 和 ) 怎么办？

    ```c
    struct A
    {
        void func_1()
        {
            return 0;
        }
    }
    ```

    在 Vim 的 Visual 模式下，当你已经选中了 `void func_1()`，最快的方式是利用 **`o`** 键。

    **核心操作：`o` (Go to Other end)**

    在 Visual 模式（或 Visual Line 模式）下，按下 **`o`** 会将光标切换到选中区域的**另一端**。

    1. **跳转到 `v**`：如果光标当前在 `)`，按下 `o`，光标会立刻跳回起始点 `v`。
    2. **跳转到 `)**`：如果光标当前在 `v`，按下 `o`，光标会跳到结束点 `)`。

    **其他进阶技巧**

    如果你想在选中状态下进行更细微的调整，可以使用以下方法：

    * 精确移动：

        * 按下 `o` 切换到你想调整的那一端。
        * 使用 `f` 或 `t` 进行行内快速定位。例如，按下 `o` 回到开头后，再按 `f(` 就能精准定位到左括号。

    * 重新选择（更符合 Vim 思路的方式）：

        如果你经常需要选中函数名或括号内容，可以尝试使用 **Text Objects**，这比手动涂抹选中更高效：
        * `viw`：选中光标下的单词（如 `func_1`）。
        * `vi(`：选中括号内的内容。
        * `va(`：选中括号及其内部内容。

    * 那`[和`]呢？

        在 Vim 中，**`[`** 和 **`]`**（特别是配合反引号或撇号使用的 ``[` 和 ``]`）是非常强大的工具，它们专门用于定位**上一次操作（修改或拉取）的边界**。

        在 Visual 模式下，它们的具体作用如下：

        1. 精准跳转到选区的起点和终点

            如果你已经在 Visual 模式下选中了一段内容，这两个组合键可以让你瞬间到达边界：

            * **``[`**：跳转到上一次选区（或修改）的**起始位置**（即 `v`）。
            * **``]`**：跳转到上一次选区（或修改）的**结束位置**（即 `)`）。

            > **注意：** 使用反引号 ``` 会定位到具体的行列，而使用单引号 `'[` 则只会定位到那一行的行首。

        2. 与 `o` 的区别

            虽然它们都能在选区两端跳转，但逻辑逻辑略有不同：

            | 快捷键 | 行为特征 |
            | --- | --- |
            | **`o`** | **切换端点**。如果光标在头，按 `o` 去尾；如果光标在尾，按 `o` 去头。非常适合临时调整选区范围。 |
            | **``[` / ``]**` | **绝对定位**。无论光标当前在哪，``[` 永远回起点，``]` 永远去终点。 |

        3. 实际应用场景（不仅限于 Visual 模式）

            这两个标记在普通模式下极其有用，因为 Vim 会自动在以下操作后更新它们：

            * **粘贴（Put）后**：如果你刚按 `p` 粘贴了一大段代码，按 **``]`** 可以立刻跳到粘贴内容的末尾，方便你继续编写。
            * **修改（Change）后**：刚用 `cw` 或 `c/pattern` 改完代码，想要跳回改动的地方，可以用这两个标记。
            * **重新选中**：如果你退出了 Visual 模式，想再次选中刚才那段代码，可以输入 **`gv`**，它的底层逻辑其实就是引用了这两个边界标记。

        **总结**

        对于你提到的 `void func_1()`：

        1. 按 **`v`** 进入选中。
        2. 移动到 **`)`** 结束。
        3. 此时想回 **`v`**：按 **`o`** 或者 **``[`**。
        4. 此时想回 **`)`**：按 **`o`** 或者 **``]`**。

    * 但是`[和`]定位的是“上一次的”，不是当前 visual 模式下选中的。你再想想

        你观察得非常敏锐，确实是我之前的描述不够精确。

        你说得对： **在 Vim 正在进行的 Visual 模式中，`[` 和 `]`（及其对应的标记 ``[` 和 ``]`）定位的是“上一次”操作的边界，而不是“当前正在变动中”的选区边界。**

        为了说清楚这个问题，我们需要拆解一下 Vim 的底层逻辑：

        1. 正在选中时：`o` 是唯一的“跳转键”

            当你输入 `v` 选中 `void func_1()` 时，Vim 内部维护了两个点：

            * **锚点 (Anchor)**：你开始按 `v` 的地方（`v`）。
            * **光标点 (Cursor)**：你当前移动到的地方（`)`）。

            在 **Visual 模式活跃期间**，除了通过 `o` (other end) 在锚点和光标点之间切换，并没有其他内置的快捷键能直接跳到“当前选区的开头”。

        2. `[` 和 `]` 到底什么时候更新？

            正如你所指出的，它们是“延迟更新”的。

            * 当你**结束** Visual 模式（比如按 `y` 复制了，或者按 `Esc` 退出了），Vim 才会把刚刚这段区域的起始和终点赋值给 ``[` 和 ``]`。
            * 所以，如果你正在选中，按 ``[` 实际上会跳到你**上一次**复制或修改的代码块起点，这会导致你直接脱离当前的 Visual 模式，飞到别处去。

        3. 一个容易混淆的标记：`<` 和 `>`

            既然提到了边界，不得不提 ``<` 和 ``>`：

            * 它们专门指向**上一次选区**的开头和结尾。
            * 同样，它们也只在 Visual 模式**结束后**才更新。
            * **gv** 命令之所以能重选上次区域，本质上就是执行了 `v`<`> `。

        **总结：在 Visual 模式下操作的“真相”**

        如果你已经选中了 `void func_1()`，光标在 `)`：

        | 目标 | 操作 | 备注 |
        | --- | --- | --- |
        | **跳转到 `v` 并继续选区** | 按 **`o`** | 这是最快、且唯一保持选区状态的办法。 |
        | **跳转到 `v` 但退出选区** | 按 **`Esc`** 然后 **`o`** (非标准) | 通常在 Vim 里不这么做。 |
        | **跳到上一次操作处** | 按 **``[`** | 会直接中断当前的 Visual 模式。 |

        **补充一个极端技巧：**

        如果你是因为选区太长（比如选了好几屏），想看一眼开头再回来，你会发现 `o` 非常好用。按一下 `o` 回到开头看一眼，再按一下 `o` 回到结尾继续往下选。

        既然你对 Vim 的内部逻辑这么细致，那我们直接拆解这两对最容易混淆的“边界标记”。它们就像 Vim 的**自动快照**，但在不同的时刻按下快门。

        * `[` / `]`  vs  `<` / `>` ：深度对比

            | 标记对 | 名称 | **什么时候“按下快门”？** (更新时机) | 主要用途 |
            | --- | --- | --- | --- |
            | **``[` 和 ``]**` | **操作边界** | 当你执行了 **Operator**（如 `y` 复制, `d` 删除, `p` 粘贴, `c` 修改）之后。 | 跳到刚粘贴内容的末尾，或刚改动代码的起点。 |
            | **``<` 和 ``>**` | **选区边界** | 当你**退出** Visual 模式的那一瞬间。 | 重新选中（`gv`）、针对上次选区执行命令（`:'<,'>s/...`）。 |

        * 场景模拟：它们是如何工作的？

            假设你在编辑这段代码：

            ```cpp
            void func_1() { return 0; }
            ```

            * 第一阶段：正在选中 (Visual Mode)

                1. 你按下 `v` 并移动到 `)`。

                2. **此时此刻：** * ``[` 和 ``]` 还是**旧的**（指向你上一次复制或粘贴的地方）。

                    * ``<` 和 ``>` 也是**旧的**（指向你上一次 Visual 选区）。
                    * **唯一**能实时跳到 `v` 的只有 **`o`**。

            * 第二阶段：退出选区 (Exit Visual)

                1. 你按下 `Esc` 退出。

                2. **快门按下：** ``<` 瞬间更新为 `v` 的位置，``>` 更新为 `)` 的位置。

                3. **此时：** 你输入 **`gv`**，Vim 其实就是去读这两个标记，帮你把区域重新刷黑。

            * 第三阶段：执行操作 (After Operator)

                1. 你重新选中区域并按下 **`y`** (Yank)。

                2. **快门再次按下：** * ``<` 和 ``>` 依然保留（因为选区结束了）。
                * **``[` 和 ``]` 被更新**，记录下被复制内容的范围。

                3. **应用：** 假如你执行的是 `p` (Paste) 了一大段文档，按 **``]`** 就能让你瞬间从第一行跳到粘贴完的最后一行。

        * 一个极实用的进阶技巧：自定义快捷键

            如果你觉得在 Visual 模式下只能用 `o` 来回跳，而无法像普通模式那样精准定位边界，你可以在 `.vimrc` 中利用这些标记实现“**回到选区开头**”：

            ```vim
            " 在 Visual 模式下，按 \ [ 跳到当前选区的开头（虽然还没更新快门，但可以强制跳转）
            vnoremap <leader>[ <Esc>`<v`>o
            ```

            *注：这类映射通常比较 hack，最推荐的依然是习惯 `o` 的“旋转”逻辑。*

        **总结**

        * **正在选：** 用 `o`。
        * **选完了想回味：** 用 ``<` 或 `gv`。
        * **动了代码（删/改/贴）后：** 用 ``[`。

* `'"` (单引号 + 双引号)：这是 Vim 中的一个特殊标记（Mark）。在 Vim 中，`"` 标记专门用来记录上一次退出该文件时光标所在的位置。单引号`'`表示接下来是一个标记。如果不使用单引号，还可以使用反引号 ``` ` ```

* vim 如何在打开一个文件时，恢复上次关闭时的位置

    Vim 有几种方法可以在打开文件时恢复上次关闭时的位置：

    1. 自动恢复位置（内置功能）

        Vim 默认会在你打开之前编辑过的文件时自动恢复光标位置。这个功能由 `'viminfo'` 选项控制：

        ```vim
        " 检查当前 viminfo 设置
        :set viminfo?

        " 通常包含 '100 或类似设置，表示记住 100 个文件的位置
        ```

    2. 手动配置增强

        在 `.vimrc` 中添加以下配置来确保位置恢复：

        ```vim
        " 自动恢复到上次编辑位置
        autocmd BufReadPost *
            \ if line("'\"") >= 1 && line("'\"") <= line("$") && &ft !~# 'commit'
            \ |   exe "normal! g`\""
            \ | endif
        ```

        这个配置会：

        - 检查上次位置是否有效（在文件范围内）
        - 排除 git commit 文件
        - 跳转到上次位置

    3. 更完整的配置

        ```vim
        " 恢复光标位置、折叠状态等
        augroup restore_position
            autocmd!
            " 恢复光标位置
            autocmd BufReadPost *
                \ if line("'\"") > 1 && line("'\"") <= line("$") && &ft !~# 'commit'
                \ |   execute "normal! g'\""
                \ | endif
            
            " 恢复折叠状态（如果有的话）
            autocmd BufWinEnter * if &ft !~# 'commit' | silent! loadview | endif
            autocmd BufWrite * if &ft !~# 'commit' | mkview | endif
        augroup END
        ```

    4. 检查 viminfo 设置

        确保你的 viminfo 设置正确：

        ```vim
        " 在 .vimrc 中设置合适的 viminfo
        set viminfo='100,<500,s10,h,n~/.vim/viminfo

        " 解释：
        " '100 - 记住 100 个文件的标记
        " <500 - 每行最多 500 个字符
        " s10  - 大小超过 10KB 的寄存器不保存
        " h    - 在搜索时高亮关闭（避免安全风险）
        " n~/.vim/viminfo - viminfo 文件路径
        ```

    5. 手动跳转到上次位置

        如果自动恢复没生效，可以使用这些命令：

        ```vim
        ``   " 跳转到上次位置（当前文件内）
        `.   " 跳转到上次修改的位置
        `"   " 跳转到上次退出时的位置
        ```

    6. 查看位置历史

        ```vim
        :marks   " 查看所有标记，其中 '" 是上次退出时的位置
        ```

    7. 问题排查

        如果位置恢复不工作：

        1. **检查 viminfo 文件**：
           ```bash
           ls -la ~/.viminfo    # 或 ~/.vim/viminfo
           ```

        2. **检查权限**：
           ```bash
           chmod 600 ~/.viminfo
           ```

        3. **重新生成 viminfo**：
           ```vim
           :wviminfo!   " 强制保存当前状态
           ```

    8. 推荐的完整配置

        ```vim
        " 在 ~/.vimrc 中添加
        if has("autocmd")
            " 恢复光标位置
            au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") && &ft !~# 'commit'
                \ | exe "normal! g`\""
                \ | endif
            
            " 恢复折叠
            au BufWinLeave * if expand("%") != "" && &ft !~# 'commit' | mkview | endif
            au BufWinEnter * if expand("%") != "" && &ft !~# 'commit' | silent! loadview | endif
        endif

        " 设置 viminfo
        set viminfo='100,<1000,s100,h,n~/.vim/viminfo
        ```

    这样配置后，每次打开文件时 Vim 都会自动将光标定位到上次编辑的位置。

* vim 在一个函数体内，如何快速跳转到函数开头

    在 Vim 中有多种方法可以快速跳转到函数开头：

    1. 使用内置的移动命令

        **最常用：**

        ```vim
        [[   " 跳转到上一个函数的开头
        ]]   " 跳转到下一个函数的开头
        ```

        **在函数体内跳转到当前函数开头：**

        ```vim
        [{   " 跳转到当前代码块的开头（包括函数）
        []   " 跳转到当前代码块的开头（另一种方式）
        ```

    2. 使用百分比跳转
        
        ```vim
        50%  " 跳转到文件 50% 位置（不精确）
        ```

    3. 使用标记（推荐）

        先标记当前位置，然后快速跳转：

        ```vim
        ma    " 在当前光标位置标记为 a
        # 编辑其他位置...
        `a    " 跳回标记 a 的位置
        ```

    4. 使用搜索跳转
        
        ```vim
        ?^{   " 向上搜索函数开头（适用于 C/C++ 风格）
        ?^function   " 向上搜索 function 关键字（适用于 JavaScript/Python）
        ```

    5. 使用插件增强

        **安装 vim-scripts 插件：**

        ```vim
        " 在 .vimrc 中
        Plugin 'vim-scripts/argtextobj.vim'
        Plugin 'vim-scripts/vim-matchit'
        ```

        然后可以使用：

        ```vim
        [a    " 跳转到参数开头
        ]a    " 跳转到参数结尾
        ```

    6. 针对不同语言的特定方法

        C/C++：

        ```vim
        [m    " 跳转到成员函数的开头
        [M    " 跳转到成员函数的结尾
        ```

        Python：

        ```vim
        [/^def<CR>   " 搜索 def 开头的行
        ?^class<CR>  " 搜索 class 开头的行
        ```

    7. 自定义映射

        在 `.vimrc` 中添加自定义快捷键：

        ```vim
        " 跳转到当前函数开头
        nnoremap <leader>f ?^[{}]*[[:space:]]*function\\|^[{}]*[[:space:]]*def\\|^{<CR>

        " 跳转到函数结尾
        nnoremap <leader>F /^}\\|^}<CR>
        ```

    8. 使用文本对象

        安装插件后可以使用文本对象：

        ```vim
        vif   " 选择整个函数
        dif   " 删除整个函数
        cif   " 修改整个函数
        ```

    9. 最实用的组合方法

        对于快速在函数内跳转，推荐：

        1. **标记当前位置**：

           ```vim
           ma    " 标记当前位置为 a
           ```

        2. **跳转到函数开头**：

           ```vim
           [[    " 跳转到函数开头
           ```

        3. **跳回原位置**：

           ```vim
           `a    " 跳回标记 a 的位置
           ''    " 跳回上次跳转前的位置
           ```

    10. 查看当前位置信息
        
        ```vim
        :function   " 显示当前函数名（需要插件支持）
        CTRL-g      " 显示当前文件名和位置
        ```

    对于日常使用，`[[` 和 `]]` 是最直接有效的方法，它们能准确识别大多数编程语言中的函数边界。

* 讲讲 z. 和 zt 的区别

    在 Vim 中，`zt` 和 `z.` 都是用来**滚动屏幕（Redraw）**的快捷键，它们的作用都是将**当前光标所在的行**移动到屏幕的特定位置，但它们移动的目标位置不同。

    可以通过下面这个表格和形象的对比来快速理解：

    **核心区别一览**

    | 快捷键 | 光标所在行的去向 | 助记（Memory Hook） |
    | --- | --- | --- |
    | **`zt`** | 滚到屏幕的**顶部** (Top) | **t** 代表 **T**op (顶部) |
    | **`z.`** | 滚到屏幕的**中央** (Center) | **`.`** 点在中间，代表 **Center** (中央) |

    ---

    **详细对比与行为**

    1. `zt` (Scroll Line to **T**op)

        * **动作**：滚动屏幕，使当前光标所在行成为屏幕的**第一行**（最顶部）。
        * **光标位置**：光标依然停留在这一行，它的**列位置（左右位置）不会改变**。
        * **使用场景**：当你开始读某一个函数，或者进入了一个新章节，想让接下来的所有内容尽可能多地显示在屏幕下方时，用 `zt` 把它顶上去。

    2. `z.` (Scroll Line to **Center**)

        * **动作**：滚动屏幕，使当前光标所在行成为屏幕的**正中间那一行**。
        * **光标位置**：与 `zt` 有个小小的不同——`z.` 除了把行居中外，还会将光标移动到该行**第一个非空白字符**上（相当于自动按了 `^`）。
        * **使用场景**：当你通过 `/` 搜索（Search）跳转到一个匹配项时，该行可能刚好在屏幕的最边缘。按下 `z.` 可以瞬间把这一行移到视野中央，方便你看清它的上下上下文。

    **拓展：另外两个同族兄弟**

    既然提到了 `zt` 和 `z.`，就不得不提它们同一家族的另外两个高频快捷键，它们正好凑成一套：

    * **`zb`**：将当前行滚到屏幕**底部** ( **B**ottom )。
    * **`z<Enter>`**：将当前行滚到屏幕**顶部** (Top)，但与 `zt` 不同的是，它会把光标移到该行的第一个非空白字符上（类似于 `z.` 的光标行为）。

    你现在是在写某种特定的自动化滚动脚本，还是单纯在优化自己的日常编辑工作流呢？

* 详细讲讲 vim 中 [ 和 ] 的作用

    在 Vim 中，`[` 和 `]`（方括号）本身在普通模式下**不是独立的移动命令**，而是作为**前缀组合键**。它们就像是 Vim 里的“快捷键修饰符”（类似于 Ctrl 或 Alt），专门用来配合其他字符，实现各种极其强大的**双击流/成对跳转**。

    理解这两个键的逻辑非常简单：

    * **以 `[` 开头**：通常表示**向上**、**向前**（Previous / Backward）搜索或跳转。
    * **以 `]` 开头**：通常表示**向下**、**向后**（Next / Forward）搜索或跳转。

    以下是 `[` 和 `]` 在 Vim 中最常用、最核心的几大场景：

    1. 代码块与大括号跳转（程序员必备）

        虽然上一次讲到 `{` 和 `}` 是按空行跳段落，但如果你想在 C 语言风格的代码中，直接在外层的 `{` 和 `}`（函数体）之间跳转，就要用到方括号了。

        * **`[[`**：跳转到**上一个**位于行首的 `{`（通常是上一个函数的开头）。
        * **`]]`**：跳转到**下一个**位于行首的 `{`（通常是下一个函数的开头）。
        * **`[]`**：跳转到**上一个**位于行首的 `}`（通常是上一个函数的结尾）。
        * **`][`**：跳转到**下一个**位于行首的 `}`（通常是下一个函数的结尾）。

        > **💡 小提示**：这些命令对格式有要求（大括号必须在行首）。如果你的代码是大括号不换行的风格（如 Java/JavaScript 的 `function() {`），它们可能不太好使。

        注：

        1. 这几个命令意义不大，且不好记。后面可以把这几个命令 map 成别的跳转命令。对于 C / C++ 我们可以使用 lsp 插件，更靠谱。

    2. 语法与方法跳转（跨语言）

        为了解决不同语言排版不同的问题，Vim 提供了更通用的函数/方法跳转快捷键：

        * **`[m`**：跳转到**上一个**方法的开始（Method start）。
        * **`]m`**：跳转到**下一个**方法的开始。
        * **`[M`**：跳转到**上一个**方法的结束（Method end）。
        * **`]M`**：跳转到**下一个**方法的结束。

        这对 Java、Python (类中的 `def`)、C++ 等面向对象语言的开发者来说，是极其高效的切函数工具。

        注：

        1. 这个似乎可以识别`void func() { `这种风格的写法。不过不清楚是否是因为当前环境中已经安装了 lsp

    3. 诊断与修改点跳转

        在编写和调试代码时，你经常需要回到刚才报错的地方或者刚才改动过的位置：

        * **`[c`** 和 **`]c`**：在代码对比模式（Vimdiff）或者配合 Git 插件时，用来在上一个/下一个修改差异点（Change）之间跳转。

        * **`[` 加特定插件**：如果你安装了语法检查插件（如 ALE 或 Coc.nvim），通常可以通过 `[d` 和 `]d` 在上一个/下一个错误/警告（Diagnostic）之间跳转。

    4. 文本对象选择（Visual 模式下的特殊用法）

        当你进入视觉模式（Visual Mode）想选中括号里的内容时，方括号和圆括号/大括号的使用逻辑是一样的。配合 `i`（inner，内部）或 `a`（around，包含括号本身）：

        * **`vi]`**：选中当前光标所在方括号 `[ ]` **内部**的所有文本。
        * **`va]`**：选中整个方括号 `[ ]`，**包括方括号本身**。

    5. 究极进化：Unimpaired 插件（几乎成为行业标准）

        谈到 `[` 和 `]`，就绝对不能不提 Vim 殿堂级大牛 Tim Pope 写的插件 **`vim-unimpaired`**。这个插件把 `[` 和 `]` 的哲学发挥到了极致，几乎每个重度 Vim 用户都会安装它。

        它扩展了以下超级好用的成对快捷键：

        * **空行插入**：

            * **`[<Space>`**：在当前行**上方**快速插入一个空行（不进入插入模式）。
            * **`]<Space>`**：在当前行**下方**快速插入一个空行。

        * **行移动（相当于拖动行）**：

            * **`[e`**：把当前行**往上移动**一行（Bubble up）。
            * **`]e`**：把当前行**往下移动**一行（Bubble down）。

        * **配置开关**：

            * **`[o n`** / **`]o n`**：快速开启/关闭行号（`set number`）。
            * **`[o s`** / **`]o s`**：快速开启/关闭拼写检查（`set spell`）。

        * **列表跳转**：

            * **`[q`** / **`]q`**：在 Quickfix 列表（编译错误列表）的上一个/下一个条目间跳转。

* jump to tag: `ctrl` + `]`

* ctrl + f

    无论光标在何处，都跳转到当前页的倒数第 2 行，并把这一行置顶

* vim 中 gj / gk 的作用

    gj 和 gk 是 Vim 中用于 按屏幕行（visual line）移动 的命令，主要用于处理长行换行显示的情况。

    主要作用对比

    | 命令 | 普通模式移动 | 作用 |
    | j | 下一实际行 | 按文件中的换行符移动 |
    | k | 上一实际行 | 按文件中的换行符移动 |
    | gj | 下一屏幕行 | 按屏幕上显示的行移动（即使长行被折行显示） |
    | gk | 上一屏幕行 | 按屏幕上显示的行移动（即使长行被折行显示） |

    使用场景示例

    假设有一个很长的行（例如 200 个字符），在 Vim 中因为窗口宽度限制被自动折行显示为多行：

    ```text
    这是一段非常非常非常...（省略）...长的文本行，
    在屏幕上显示为多行，但实际上文件中只有一行。
    ```

    * 按 j/k：光标会跳到下一个实际行（文件中的换行处）

    * 按 gj/gk：光标会在同一个长行内的不同屏幕行之间上下移动

    相关配置

    ```vim
    " 让 j/k 在换行时按屏幕行移动（类似 gj/gk 的默认行为）
    nnoremap j gj
    nnoremap k gk

    " 但这样会失去按实际行移动的功能，可改用：
    nnoremap <expr> j v:count ? 'j' : 'gj'
    nnoremap <expr> k v:count ? 'k' : 'gk'
    ```

    其他相关命令

    * g0/g^/g$：按屏幕行的行首/非空白字符/行尾移动

    * :set wrap：启用自动折行（默认开启）

    * :set nowrap：禁用折行，长行会水平滚动

    gj/gk 在处理 Markdown、文档、日志等包含长行的文本时特别有用。

* 在 vim 中使用 p 粘贴了一段内容后，如何快速定位到粘贴内容的开头和结尾？

    1. 使用标记（marks）

        vim 在粘贴时自动设置了两个标记：

        ```vim
        `[  " 粘贴内容的开头
        `]  " 粘贴内容的结尾
        ```

        注：

        1. 也可以使用单引号`'`代替反引号``` ` ```

    2. 使用可视模式重选

        ```vim
        gv
        ```

        `gv`可以重新选择上次可视模式选中的区域。

        如果是在 visual 模式下使用 p 粘贴 东西后再按 gv，会选中刚粘贴的内容。

        注：

        1. `gv`重选后，光标会跳转到选中内容的开头还是结尾？

            目前看来是结尾。

    3. 使用选择器查看

        ```vim
        :marks [ ]   " 查看 [ 和 ] 标记的位置
        ```

    快速操作粘贴内容：

    ```vim
    `[v`]       " 选中整个粘贴内容（行模式）
    `[v`]       " 选中整个粘贴内容（字符模式）
    c`[         " 从当前位置修改到粘贴开头
    d`]         " 删除从当前位置到粘贴结尾
    ```

    注：

    1. 这里的行模式应该改成单引号`'`吧。可能是 ai 有幻觉

    相关标记：

    * 按 `. 回到上次编辑位置

    * `^：上次插入模式退出位置

    * `"：上次退出文件时的位置

* 行内移动和操作

    * 0 - 移动到行首

    * ^ - 移动到行首第一个非空白字符

    * $ - 移动到行尾

    * d$ 或 D - 删除到行尾

    * d0 - 删除到行首

    * d^ - 删除到行首第一个非空白字符

* vim 纯滚动不移动光标

    ```vim
    " 纯滚动屏幕，不移动光标
    nnoremap <C-e> <C-e>j  " 向下滚动一行
    nnoremap <C-y> <C-y>k  " 向上滚动一行
    ```

* vim 中，使用 "+p 粘贴内容后，如何跳转到粘贴内容的开关和结尾

    * gv 

        * 粘贴后直接按 gv 可以重新选择刚才粘贴的内容

        * 然后按 o 可以在选择模式的开头和结尾之间切换

    * 使用标记 (Marks)

        Vim 在粘贴时会自动设置一些标记：

        ```vim
        `[     " 粘贴内容的开头位置
        `]     " 粘贴内容的结尾位置
        ```

    * 查看粘贴区域的内容

        ```vim
        :echo @[    " 查看最后修改/粘贴的开头位置
        :echo @]    " 查看最后修改/粘贴的结尾位置
        ```

    example：

    ```vim
    "+p    " 粘贴系统剪贴板内容
    `[     " 跳转到粘贴内容的第一个字符
    ``     " 跳转回之前的位置
    `]     " 跳转到粘贴内容的最后一个字符
    ```

* vim 相对行号

    `set relativenumber`

* vim 保持光标位置翻页

    ```vim
    " 保持光标位置翻页
    nnoremap <C-d> <C-d>zz
    nnoremap <C-u> <C-u>zz
    ```

* vim 移动光标

    ```vim
    zz    " 将当前行移到屏幕中间
    zb    " 将当前行移到屏幕底部
    zt    " 将当前行移到屏幕顶部
    ```

* vim 使用标记（mark）

    * 先标记当前位置：ma（标记到 a）

    * 翻页：Ctrl+f（向下翻整页）或 Ctrl+d

    * 返回标记：`a 或 'a

* 详细讲讲 vim 中 { 和 } 的作用

    在 Vim 中，`{` 和 `}` 是非常高效的**段落移动快捷键**。它们可以让你在代码或文本中以“段落”为单位进行大跨度的快速跳转。

    理解这两个键的核心，在于理解 Vim 是如何定义“段落”的。

    **核心功能：段落跳转**

    在普通模式（Normal Mode）下：

    * **`}` (右大括号)**：跳转到**下一个**段落的开头（其实就是光标移动到下一个空行）。
    * **`{` (左大括号)**：跳转到**上一个**段落的开头（光标移动到上一个空行）。

    > **💡 什么是 Vim 眼中的“段落”？**
    > 在默认情况下，Vim 把**完全空白的行（或者只包含空白字符的行）**当作段落的分隔符。也就是说，连续的有文字的行是一个段落，一旦遇到空行，就意味着跳转的终点。

    **高级进阶用法**

    这两个键绝不仅仅用来“赶路”，配合 Vim 的操作符（Operators）和计数器，它们能发挥出强大的威力。

    1. 结合操作符（删除、复制、高亮）

        Vim 的经典公式是：`操作符 + 位移命令`。因为 `{` 和 `}` 是位移命令，所以你可以这样组合：

        * **`d}`**：删除从当前光标位置一直到**当前段落结束**的所有内容。
        * **`y}`**：复制从当前光标位置一直到**当前段落结束**的所有内容。
        * **`v}`**：从当前光标开始，**视觉高亮选中**到当前段落结束。
        * **`c}`**：修改当前光标到段落结束的内容（删除并直接进入插入模式）。

    2. 结合数字（多段跳跃）

        如果你想一口气跳过好几个段落，可以在前面加上数字：

        * **`3}`**：向下猛跳 3 个段落（跳到第 3 个空行处）。
        * **`2{`**：向上猛跳 2 个段落。

    **写代码时的“特殊变通”**

    虽然 `{` 和 `}` 的本意是写文章时跳段落，但在现代编程中，它们成了**在代码块之间跳转**的神器。

    因为程序员习惯用空行来分隔不同的函数、类或者逻辑块，所以：

    * 在 Python 中，可以用它们在不同的函数（`def`）之间快速穿梭。
    * 在 JSON 或 CSS 中，可以用它们快速跳过整个配置块。

    **注意一个盲区（针对 C / C++ / Java / Go 等大括号党）**

    如果你在写 C++ 或 Java，你可能会误以为 `{` 和 `}` 是用来跳转到代码里的 `if { ... }` 的大括号对齐处的。

    **其实不是。**

    * `{` 和 `}` 依然只认**空行**。
    * 如果你想在配对的大括号 `{` 和 `}` 之间跳转，你应该使用 **`%`** 键（把光标放在任意一个大括号上，按 `%` 就会跳到对应的另一个大括号）。

    **总结口诀**

    * 想下跳，按 `}`（像个向下的漏斗，装满接下来的内容）。
    * 想上跳，按 `{`（像个向上的盖子）。
    * 配合 `d` 或 `y`，一秒清理/复制整块代码。

* `+`

    作用：移动到下一行的第一个非空白字符

    详细说明：

    * 相当于 j + ^ 的组合

    * 直接定位到下一行有文本内容的位置

    * 数字前缀可用：3+ 向下移动3行并定位

    * 反义命令是 -（移动到上一行的第一个非空白字符）

* 跳转历史

    * `Ctrl+o`: 跳转到上一个位置

        返回到光标之前的位置（向后浏览跳转历史）

    * `Ctrl+i`: 跳转到下一个位置

        向前跳转到光标之后的位置（向前浏览跳转历史）

    触发跳转的操作包括：

    * 使用 G、gg、/搜索、%匹配括号等

    * 使用标签跳转 `Ctrl-]`

    * 使用 `'m` 标记跳转等

    小技巧：

    * 查看完整的跳转列表：:jumps

    * Ctrl-o 和 Ctrl-i 在普通模式和插入模式下都有效

* vim 中`%`: 在匹配的括号间跳转

    基本用法

    * 在 `(`、`)`、`[`、`]`、`{`、`}` 等符号上按 %，光标会跳转到匹配的对应符号

    * 支持 C/C++ 风格的 #if、#ifdef、#else、#endif 等预处理指令

    高级用法

    * 视觉模式：在视觉模式下，% 会选中从当前位置到匹配括号之间的所有内容

    * 配合操作符：

        * d%：删除从当前位置到匹配括号之间的内容

        * c%：修改从当前位置到匹配括号之间的内容

        * y%：复制从当前位置到匹配括号之间的内容

    配置增强

    可以通过配置增强 % 的匹配能力：

    ```vim
    " 扩展匹配的字符对
    set matchpairs+=<:>  " 添加尖括号匹配
    ```

    注意事项

    * % 命令会查找最近的匹配符号

    * 如果当前位置不在符号上，会向前查找最近的符号

    * 可以通过 :match 命令高亮显示匹配的括号对

* vim 中，`[[`表示向上搜索行开头的`{`，等价于正则表达式`^{`

    ```cpp
    int main()
    {  // 可以匹配到这个
        return 0;
    }
    ```

    在`return 0;`处，按`[[`，可以匹配到上述例子中的`{`。

    但是下面的例子无法匹配：

    ```cpp
    int main() {  // 无法匹配到这个
        return 0;
    }

    int main()
      {  // 这样也无法匹配
        return 0;
    }
    ```

    没有插件的 vim 无法理解编程语言，只能按固定位置匹配字符。

    相关命令

    * `]]` - 跳转到下一个函数开始

    * `[]` - 跳转到上一个函数结束

    * `][` - 跳转到下一个函数结束

    可以通过设置 `:help 'define'` 选项来改变 `[[` 的识别模式：

    `:set define=^\\s*def  " 对于 Python，识别 def 开头的函数`

* vim 中，`[{`表示向上搜索`{`，但必须是上一级的`{`

    example:

    ```cpp
    int main() {  // 第二步，跳转到这里。在这里按 [{，则无法跳转到上一个函数，因为这里的 { 已是顶级
        int a = 1;
        if (a == 1) {
            printf("a == 1\n");
        } else {  // 第一步，跳转到这里。在这里再按 [{
            printf("a != 1\n");  // 在这里按 [{
        }
        return 0;
    }
    ```

* vim 中的`[m`

    搜索 class / struct 中 method 的开头，或者 class / struct 的开头或结尾。

    example:

    ```cpp
    struct A
    {
        void func_1()
        {
            return 0;
        }

        void func_2() {
            return 0;
        }

        void func_3()
        {
            if (aaa) {
                printf("hello\n");
            } else {
                for (int i = 0; i < 3; ++i) {
                    printf("world\n");
                }
            }
            return 0;
        }
    }
    ```

    对于上面的例子，光标在 struct A 的某个位置中，按`[m`可以跳转到当前 method 或者上一个 method 的开头。如果已经在第一个 method 开头，那么会跳转到 struct 的开头。如果已经在 struct 开头，那么会跳转到上一个 struct 的开头。

    同理，下面几个相关命令：

    * `[M`：跳转到上一个 method 的结尾，或者 struct / class 的开头或结尾

    * `]m`：跳转到下一个 method 的开头，或者 struct / class 的开头或结尾

    * `]M`：跳转到下一个 method 的结尾，或者 struct / class 的开头或结尾

    注：

    1. 这种跳转明显对 c++ / java 有效，但是不清楚是否对 python 有效。

    1. 跳转命令前可以加数字，表示向上重复几次。`[N][m`

        example: `3[m`

### 自动补全

* `:syntax list` 中的 `cInclude`

    ```vim
    cInclude       xxx match /^\s*\zs\(%:\|#\)\s*include\>\s*["<]/  display contains
    =cIncluded
                       links to Include
    cDefine        xxx start=/^\s*\zs\(%:\|#\)\s*\(define\|undef\)\>/ skip=/\\$/ end
    =/$/  keepend contains=ALLBUT,@cPreProcGroup,@Spell
    ...
    ```

    你看到：

    ```
    cInclude  xxx match /^\s*\zs\(%:\|#\)\s*include\>\s*["<]/  display contains
    ```

    说明 Vim 的 C 语法规则是：

    > `#include` 只在匹配完整的 `#include` + `<...>` 或 `"..."` 才会被识别为 cInclude。

    也就是说：

    ```
    #include <xxx>
    ```

    才是被识别的语法 token。

    **`cInclude` 语法匹配的作用**

    你看到的：

    ```
    cInclude  xxx match /^\s*\zs\(%:\|#\)\s*include\>\s*["<]/
    ```

    它的功能主要是 **语法高亮和显示效果**，而不是补全：

    * 匹配行首的 `#include`（或者 `%:`）
    * 后面跟 `<` 或 `"`
    * 高亮显示这一整行为 include 指令

    作用包括：

    1. **颜色高亮**：让 `#include <vector>` 显示成蓝色（或主题指定颜色）
    2. **折叠**（folding）或其他语法相关操作
    3. **提供语法相关的 Vim 命令**（如 `:help cInclude` 提到的 `contains`/`display`）

    **总结一句话**

    > **Vim 的 `cInclude` 只是高亮和识别写好的 `#include`，它不提供补全；补全必须靠 dictionary/snippet/tags/插件。**

* 详细讲讲 vim 中 Ctrl + n 和 ctrl + p 的功能

    在 Vim 中，`Ctrl + n` 和 `Ctrl + p` 是一对极其强大的组合键。它们最核心的功能是**关键词自动补全（Keyword Completion）**，但根据你当前所处的**模式**，它们的作用会有所不同。

    下面为你详细拆解它们在**插入模式**和**命令行模式**下的具体功能和高级用法。

    1. 插入模式（Insert Mode）：代码与文本自动补全

        这是这对组合键最常用的场景。当你在输入代码或文字时，Vim 会在后台扫描当前文件、其他缓冲区（Buffers）甚至标签页，帮你自动补全单词。

        * **`Ctrl + n` (Next)：** 向下（向前）查找匹配的单词。
        * **`Ctrl + p` (Previous)：** 向上（向后）查找匹配的单词。

        **核心操作流程**

        1. 在插入模式下，输入一个单词的前几个字母（例如 `myfu`）。
        2. 按下 `Ctrl + n` 或 `Ctrl + p`，Vim 会弹出一个下拉菜单（Pum - Popup menu）。
        3. **继续按 `Ctrl + n**`：在弹出的补全列表中**向下**移动光标。
        4. **继续按 `Ctrl + p**`：在弹出的补全列表中**向上**移动光标。
        5. 选中想要的单词后，直接**继续输入其他字符**或者按 **`Ctrl + y`**（Yes）确认选择。如果想取消补全，按 **`Ctrl + e`**（Exit）。

        **Vim 默认从哪里查找匹配项？**

        默认情况下，Vim 的补全搜索范围非常广（由 `complete` 选项控制）。它会查找：

        * 当前当前文件（Current buffer）
        * 其他处于打开状态的缓冲区（Other buffers）
        * 已经关闭但载入过窗口的缓冲区（Unloaded buffers）
        * 当前文件的包含文件（如 C/C++ 的 `#include` 头文件）
        * Tags 标签文件

        > **冷知识：`Ctrl + n` 和 `Ctrl + p` 的初始搜索方向不同**
        > * `Ctrl + n` 会先从当前光标**往后（下方）**搜索，找不到再往前搜。
        > * `Ctrl + p` 会先从当前光标**往前（上方）**搜索。如果你刚刚定义了一个变量，马上在下面用到它，用 `Ctrl + p` 通常能一击必中！
        > 
        > 

    2. 命令行模式（Command-line Mode）：历史命令补全

        当你按下 `:` 进入命令行模式，或者按下 `/`、`?` 进入搜索模式时，这组快捷键就变成了**历史记录导航利器**。

        * **`Ctrl + n` (Next)：** 显示**下一条**更晚执行的命令（向未来的时间线移动）。
        * **`Ctrl + p` (Previous)：** 显示**上一条**更早执行的命令（向过去的时间线移动）。

        **进阶技巧：前缀过滤搜索**

        如果你只按 `:` 然后不停按 `Ctrl + p`，它会机械地遍历你输入过的所有命令。但如果你**先输入几个字母，再按 `Ctrl + p**`，它就会进行前缀匹配。

        * **操作示例：**
        1. 输入 `:set`
        2. 按 `Ctrl + p`
        3. Vim 只会帮你找出历史记录中所有以 `set` 开头的命令（如 `:set number`、`:set paste` 等），自动过滤掉其他无关命令。这比用方向键方便得多，手不需要离开主键盘区。

    **总结与对比**

    | 模式 | `Ctrl + n` (Next) | `Ctrl + p` (Previous) | 实用场景 |
    | --- | --- | --- | --- |
    | **插入模式** | 补全菜单：**向下**选择 / 向后搜索单词 | 补全菜单：**向上**选择 / 向前搜索单词 | 写代码、写文档时快速补全长单词、变量名 |
    | **命令行模式** | 历史记录：切换到**较新**的命令（向下） | 历史记录：切换到**较旧**的命令（向上） | 快速重复执行之前输入过的复杂命令 |

    **高效盲打建议：** 在 Vim 中频繁使用方向键（↑ ↓）会让你的手离开主键区，降低输入效率。无论是在插入模式选补全，还是在命令行翻历史，肌肉记忆养成 **`Ctrl + n` / `Ctrl + p**` 绝对能让你的操作流畅度翻倍！

* Ctrl + x  Ctrl + i 再详细讲讲这个

    看来你已经开始探索 Vim 极其强大的 **`Ctrl + x` 子模式（Omni-completion / Sub-mode）** 了！

    在 Vim 的插入模式下，单纯按 `Ctrl + n` 或 `Ctrl + p` 属于**通用补全**（什么都混在一起搜）。而先按 **`Ctrl + x`**，就会激活 Vim 的“特定类型补全”模式。

    接下来接一个 **`Ctrl + i`**，其核心功能是：**针对“包含文件”（Include Files）进行关键词自动补全。**

    1. 什么是 `Ctrl + x` 加上 `Ctrl + i`？

        * **`Ctrl + x`**：进入补全子模式（状态栏会显示 `-- ^X mode (` 提示）。
        * **`Ctrl + i`**：代表 **Include**。它会专门去搜索当前文件**以及所有被 `include`（引入）进来的头文件或库文件**中的单词。

        **举个例子：**

        假设你在写一个 C++ 或 C 语言程序，文件开头写了：

        ```c
        #include <stdio.h>
        #include "my_project_header.h"

        ```

        当你写到一半，想用 `my_project_header.h` 里面定义的一个结构体或 `stdio.h` 里的某个冷门函数，但当前 `.c` 文件里还没出现过这个词。

        这时候，普通的 `Ctrl + n` 可能找不到。但如果你输入前几个字母，然后按下：

        > **`Ctrl + x` 然后按下 `Ctrl + i**`

        Vim 就会瞬间顺着 `#include` 路线，飞速爬进 `stdio.h` 和 `my_project_header.h` 的源码里，把匹配的关键词抓取出来做成下拉列表供你选择！

    2. 补全触发后的后续操作

        按下 `Ctrl + x` `Ctrl + i` 弹出了菜单后，你该怎么选择呢？

        * **继续按 `Ctrl + i**` 或按 **`Ctrl + n`**：在弹出的补全列表中**向下**移动光标。
        * **按 `Ctrl + p**`：在弹出的补全列表中**向上**移动光标。
        * 确认和退出的方式与普通补全一致（`Ctrl + y` 确认，`Ctrl + e` 取消）。

    3. 它对哪些语言有效？（Vim 怎么知道什么是 "Include"？）

        Vim 之所以这么聪明，是因为它内部有一个叫 **`include`** 的配置项。Vim 会根据你当前文件的语言（Filetype），自动用正则表达式去匹配“引入文件”的语法。

        它绝对不仅限于 C/C++ 的 `#include`：

        * **Python:** 识别 `import os` 或 `from datetime import datetime`，会去搜这些库。
        * **Java:** 识别 `import java.util.*;`。
        * **Ruby:** 识别 `require 'blueprint'`。

        > ⚠️ **注意（避坑指南）：**
        > Vim 默认是基于文本和路径去查找的。它能否成功深入到库文件内部，取决于 Vim 的 `path` 变量配置。如果是一些需要复杂编译环境才能找到的深层第三方库，原生的 `Ctrl + x` `Ctrl + i` 可能会因为找不到文件路径而抓取失败。

    **顺便赠送：`Ctrl + x` 家族的其他神技**

    既然学了 `Ctrl + x` `Ctrl + i`，顺便记一下这几个亲兄弟，它们在特定场景下比普通补全好用 10 倍：

    * **`Ctrl + x` ➡️ `Ctrl + f` (File)：** 文件路径补全。输入 `./src/` 然后按这个，直接弹出该目录下的所有文件名，写路径再也不怕写错。
    * **`Ctrl + x` ➡️ `Ctrl + l` (Line)：** 整行补全。如果想写一句和上面差不多长的代码/注释，按这个直接复制一整行。
    * **`Ctrl + x` ➡️ `Ctrl + k` (Dictionary)：** 字典补全。写英文文档或注释时，帮你补全英文单词。
    * **`Ctrl + x` ➡️ `Ctrl + o` (Omni)：** 全能补全（类似于现代 IDE 的智能提示）。需要开启代码语义补全插件或内置的 `omnicomplete`。

    你平时主要用 Vim 写哪种编程语言呢？针对不同的语言，`Ctrl + x` 家族的侧重点会很不一样。

* 详细讲讲 vim 中的 dictionary 补全

    在 Vim 中，**Dictionary（字典）补全**是一个针对“纯文本”或“特定领域术语”非常有用的高级补全功能。

    普通的 `Ctrl + n` 只能补全你**已经打开的文件**里出现过的单词。但如果你在写一篇全英文的 Markdown 报告、在代码中写大段的英文注释，或者在写特定行业文档，很多单词是第一次出现，普通补全就无能为力了。

    这时候，**Dictionary 补全**（快捷键：`Ctrl + x` ➡️ `Ctrl + k`）就能让 Vim 调用一个外部的“单词列表文件”（即字典），直接帮你补全你从未输入过的单词。

    1. 核心操作方法

        在插入模式（Insert Mode）下：

        1. 输入单词的前几个字母（例如：`hypot`）。

        2. 按下组合键：**`Ctrl + x`**，紧接着按下 **`Ctrl + k`**（你可以把 $k$ 想象成 **K**eyword 或 Dictionary 的第二个音节）。

        3. Vim 会弹出的下拉菜单，展示字典中所有匹配的单词（如 `hypothesis`、`hypothetical` 等）。

        4. 切换和选定方式与普通补全完全一样（`Ctrl + n` / `Ctrl + p` 上下翻页，`Ctrl + y` 确认）。

    2. 关键前提：Vim 的“字典”从哪来？

        要想让 `Ctrl + x` `Ctrl + k` 工作，Vim 必须知道你的“字典文件”存在哪里。这需要通过配置 Vim 的 **`dictionary`** 选项来实现。

        如果未做任何配置，直接按 `Ctrl + x` `Ctrl + k`，Vim 通常会报错：`E142: File not found: "dictionary"`。

        如何配置你的字典？

        你可以将字典配置写入你的 `~/.vimrc`（或 Neovim 的 `init.lua`）中：

        方法 A：利用 Linux / macOS 自带的系统字典（最省心）

        类 Unix 系统（如 Ubuntu, macOS）通常自带一个巨大的英文单词列表，一般位于 `/usr/share/dict/words`。

        ```vim
        " 在 .vimrc 中添加这一行
        set dictionary+=/usr/share/dict/words
        ```

        方法 B：创建你自己的“专属术语库”

        如果你在写医学、法律、金融文档，或者某个特定项目的开发文档，你可以自己创建一个纯文本文件（比如 `~/.vim/my_terms.txt`），里面**每行写一个单词/术语**。

        ```vim
        " 在 .vimrc 中指定你自己的术语字典
        set dictionary+=~/.vim/my_terms.txt
        ```

    3. 进阶高阶技巧

        技巧 1：让普通 `Ctrl + n` 自动融合字典

        每次都按 `Ctrl + x` `Ctrl + k` 嫌麻烦？你可以修改 Vim 的 `complete` 选项，把字典（代号为 `k`）直接塞进普通的 `Ctrl + n` 搜索路径里。

        ```vim
        " 在 .vimrc 中配置：让普通补全也去搜字典
        set complete+=k
        ```

        这样配置后，你以后只需要输入前缀并按普通的 **`Ctrl + n`**，Vim 在搜完当前文件后，会自动去搜你配置的字典文件，一步到位！

        技巧 2：分文件类型（Filetype）加载不同字典

        如果你既写英文博客，又写特定的行业代码，你肯定不希望写代码时弹出一堆英文日常词汇。你可以利用 Vim 的 `autocmd` 针对不同文件类型加载不同的字典：

        ```vim
        " 只有在写 Markdown 或纯文本时，才加载系统英文词典
        autocmd FileType markdown,text setlocal dictionary+=/usr/share/dict/words
        " 写 C 语言时，加载你的 C 语言 API 专用术语表
        autocmd FileType c setlocal dictionary+=~/.vim/c_keywords.txt
        ```

    **总结与常见误区区分**

    有很多初学者会把 **Dictionary 补全 (`Ctrl + k`)** 和 **Thesaurus（同义词）补全 (`Ctrl + t`)** 搞混：

    * **Dictionary 补全 (`Ctrl + x` ➡️ `Ctrl + k`)**：根据你输入的**前缀**，补全**完整的单词**（比如输入 `aut` 补全 `automatic`）。
    
    * **Thesaurus 补全 (`Ctrl + x` ➡️ `Ctrl + t`)**：根据你当前光标下的**完整单词**，推荐它的**同义词**（比如光标在 `big` 上，按完弹出来 `large`, `huge`, `enormous`）。它需要另外配置 `thesaurus` 文本文件。

    如果你经常需要在 Vim 中编写大量非代码的文本，花 2 分钟给 Vim 配一个 `/usr/share/dict/words` 字典，绝对会让你的文本输入效率大幅提升！

* vim c++ 文件如何补全关键字，比如 `#include`？

    `:help cscope` 和 `:help include-search` 对你没用

    * 使用 UltiSnips 或 LuaSnip（自动扩展）

        如果你想输入 `#i` 自动变成 `#include <…>`，可以用 snippet：

        例如用 UltiSnips：

        ```
        snippet inc "include directive"
        #include <${1}>
        endsnippet
        ```

        然后输入：

        ```
        inc + Tab
        ```

        自动变成：

        ```
        #include <>
        ```

        查询：

        ```
        :setl iskeyword?
        :setl complete?
        :set ft?
        ```

    **查看当前可以补全哪些关键字**

    Vim **没有一个“一条命令列出所有可补全关键字”的官方接口**，因为补全来源有多种（buffer、dictionary、tags、syntax 等）。

    * 查看当前 buffer 中的所有“单词”（最基础的补全来源）

        Vim 会从 buffer 里的“单词”补全，所以最直接：

        ```
        :g/\k\+/
        ```

        但这个太多了，最好的方式：

        ```
        :%s/\k\+/\r&/gn
        ```

        不过更简单的方式是：

        ```
        :echo join(sort(keys(wordlist#WordList())),"\n")
        ```

        但这个需要插件。

        所以**最推荐你这样：**

        ```
        :lopen
        ```

        再输入：

        ```
        Ctrl + n
        ```

        你会看到补全窗口里 *所有匹配的“单词”*。

    * 查看 dictionary（字典补全）的关键词

        ```
        :echo &dictionary
        ```

        如果你想直接看内容：

        ```
        :tabnew ~/.vim/dict/cpp.dict
        ```

    * 查看当前启用的 `iskeyword`（决定 Vim 把什么当成词）

        ```
        :setl iskeyword?
        ```

        如果没有 `#`，补全不会包含 `#include`。

        你要看到：

        ```
        iskeyword=@,48-57,_,192-255,#
        ```

        否则你补不了带 `#` 的关键字。

    * 查看补全来源（complete option）

        ```
        :setl complete?
        ```

        常用来源：

        * `k` → dictionary
        * `s` → syntax keyword
        * `w` → buffers
        * `b` → current buffer
        * `t` → tags

        你需要至少看到：

        ```
        complete=.,w,b,u,t,i,s,k
        ```

        如果没有 `s`，语法关键字不会参与补全
        如果没有 `k`，字典补全不会参与。

    * 查看 syntax 关键字（很多 C/C++ 关键字在这里）

        语法关键字在：

        ```
        :syntax list
        ```

        你会看到：

        * cInclude
        * cStatement
        * cType
        * cPreProc
        * cConditional

        等

        但是注意：

        > syntax 的名字不是补全词，补全词是它们的 token。

        如果你开启了：

        ```vim
        set complete+=s
        ```

        这里的关键字就会加入补全词库。

    * 查看 tags（如果你有 ctags 文件）

        如果你使用 ctags：

        ```
        :ts
        ```

        或者列出当前 tags 文件：

        ```
        :echo &tags
        ```

        tags 文件里的所有符号都可补全。

    如果你想“完整列出所有可补关键字”给我看, 你可以执行：

    ```
    :messages clear
    Ctrl+n
    :messages
    ```

* vim ctags 如何补全一个 symbol

    1. 使用 ctags
    
        在 Vim 中使用 ctags 进行符号补全，主要有以下几种方法：

        1. 使用 `<C-x><C-]>` 补全

            这是最直接的方式，在插入模式下：

            ```vim
            <C-x><C-]>  " 先按 Ctrl+x，然后按 Ctrl+]
            ```

            * Tagbar 插件

                ```vim
                " 安装 Tagbar
                Plugin 'preservim/tagbar'

                " 快捷键映射
                nmap <F8> :TagbarToggle<CR>
                ```

        1. 实用的快捷键和命令

            ```vim
            " 跳转到符号定义
            <C-]>        " 跳转到定义
            <C-t>        " 跳回
            <C-o>        " 后退
            <C-i>        " 前进

            " 在多个匹配项中选择
            g<C-]>       " 显示所有匹配的定义供选择

            " 查看当前光标下的符号
            :ts <symbol> " 列出所有匹配的标签
            :tn          " 下一个匹配
            :tp          " 上一个匹配
            ```

        1. 完整的工作流程示例

            1. **生成 tags**：

               ```bash
               ctags -R --languages=python --python-kinds=-i .
               ```

            2. **在 Vim 中配置**：

               ```vim
               set tags=./tags,tags
               set autochdir
               ```

            3. **使用补全**：

               - 输入部分符号名
               - 按 `<C-x><C-]>` 或 `<C-x><C-o>`
               - 从列表中选择完成

        这样就能高效地使用 ctags 进行符号补全了。

        **ctrl + x, ctrl + `]` 似乎没法补全函数里定义的变量**

        对于局部变量，使用 `<C-n>` 或 `<C-p>`：

    2. 配置更智能的补全引擎

        **YouCompleteMe**：

        ```vim
        let g:ycm_min_num_of_chars_for_completion = 2
        let g:ycm_seed_identifiers_with_syntax = 1
        ```

        **coc.nvim**：

        ```vim
        " 安装 coc.nvim 后
        :CocInstall coc-snippets
        :CocInstall coc-clangd  " 对于 C/C++
        ```

    3. 使用其他插件

        **Supertab**：

        ```vim
        Plug 'ervandew/supertab'
        let g:SuperTabDefaultCompletionType = '<C-n>'
        ```

        **asyncomplete.vim**：

        ```vim
        Plug 'prabirshrestha/asyncomplete.vim'
        Plug 'prabirshrestha/asyncomplete-buffer.vim'
        ```

* vim c++ 文件如何补全关键字，比如 #include？

    在 Vim 中补全 C++ 关键字（如 `#include`）有几种方法，我来为你详细介绍：

    1. 使用 Vim 内置补全

        **基本关键字补全**

        ```vim
        " 在插入模式下，输入 #inc 后按 Ctrl+P 或 Ctrl+N
        #inc <Ctrl+P>  " 会补全为 #include
        ```

        **具体补全方式：**

        - `Ctrl` + `P` - 向前搜索补全
        - `Ctrl` + `N` - 向后搜索补全
        - `Ctrl` + `X`, `Ctrl` + `F` - 文件名补全（对 #include 特别有用）
        - `Ctrl` + `X`, `Ctrl` + `I` - 包含文件补全

    2. 使用插件（推荐）

        **YouCompleteMe**

        ```vim
        " 安装后自动提供智能补全
        #inc <Tab>  " 自动补全为 #include
        ```

        **coc.nvim**

        ```vim
        " 配合 coc-clangd 提供 C++ 智能补全
        #inc <Tab>  " 自动补全
        ```

        **clang_complete**

        ```vim
        " 专门用于 C/C++ 补全
        Plugin 'Rip-Rip/clang_complete'
        ```

    3. 配置 .vimrc 优化补全

        ```vim
        " 启用文件类型检测和插件
        filetype plugin on

        " 设置自动补全
        autocmd FileType c,cpp setlocal omnifunc=ccomplete#Complete

        " 设置 include 路径，帮助补全
        set path+=/usr/include/c++/**

        " 映射快捷键
        inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
        inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
        ```

    4. 使用 omni 补全

        ```vim
        " 在插入模式下输入 #include < 后按 Ctrl+X Ctrl+O
        #include <Ctrl+X Ctrl+O>  " 会显示可用的头文件列表
        ```

    5. 简单自动补全配置

        ```vim
        " 在 .vimrc 中添加自定义补全
        function! CppComplete()
            if getline('.') =~ '^#inc'
                return "#include <>"
            else
                return ""
            endif
        endfunction

        inoremap #i #include <><Esc>i
        ```

* vim 中如何使用 ctags 对 python 文件中的符号自动补全

    在 Vim 中使用 ctags 对 Python 文件进行自动补全，需要以下几个步骤：

    1. 安装 ctags

        首先确保系统已安装 ctags：

        ```bash
        # Ubuntu/Debian
        sudo apt-get install exuberant-ctags

        # CentOS/RHEL
        sudo yum install ctags

        # macOS (使用 Homebrew)
        brew install ctags
        ```

    2. 生成 ctags 文件

        为你的 Python 项目生成 tags 文件：

        ```bash
        # 在当前目录生成
        ctags -R .

        # 为 Python 文件生成（更精确）
        ctags -R --fields=+l --languages=python --python-kinds=-iv -f ./tags .

        # 包含系统 Python 库
        ctags -R --fields=+l --languages=python --python-kinds=-iv -f ./tags /usr/lib/python*/ ./
        ```

    3. 配置 Vim

        在 `~/.vimrc` 中添加以下配置：

        ```vim
        " 设置 tags 文件路径
        set tags=./tags,tags,./.tags;,.tags;

        " 自动补全相关配置
        set completeopt=menu,menuone,preview

        " 使用 Ctrl+] 跳转到定义
        nnoremap <C-]> g<C-]>

        " 使用 Ctrl+t 返回
        nnoremap <C-t> <C-t>

        " 自动生成 tags 的命令
        command! MakeTags !ctags -R --fields=+l --languages=python --python-kinds=-iv -f ./tags .
        ```

    4. 使用自动补全

        在 Vim 编辑 Python 文件时，可以使用以下快捷键：

        - **Ctrl+x Ctrl+]** - 使用 tags 补全
        - **Ctrl+n** - 普通关键字补全（也会使用 tags）
        - **Ctrl+]** - 跳转到符号定义
        - **Ctrl+t** - 返回跳转前的位置
        - **Ctrl+x Ctrl+o** - Omni 补全（需要开启文件类型检测）

    5. 增强配置

        为了更好的 Python 开发体验，可以添加更多配置：

        ```vim
        " 启用文件类型检测和自动缩进
        filetype plugin indent on

        " 设置 Python 特定的配置
        autocmd FileType python setlocal completeopt=menu,menuone,preview,noselect

        " 自动在保存时更新 tags（可选）
        autocmd BufWritePost *.py silent! !ctags -R --fields=+l --languages=python --python-kinds=-iv -f ./tags . &
        ```

    6. 使用插件增强（推荐）

        安装以下插件可以获得更好的体验：

        **Tagbar**

        ```vim
        " 使用 vim-plug 安装
        Plug 'preservim/tagbar'

        " 配置
        nmap <F8> :TagbarToggle<CR>
        let g:tagbar_type_python = {
            \ 'ctagstype' : 'python',
            \ 'kinds'     : [
                \ 'c:classes',
                \ 'f:functions',
                \ 'm:members',
                \ 'v:variables',
                \ 'i:imports'
            \ ]
        \ }
        ```

        **Gutentags（自动管理 tags）**

        ```vim
        Plug 'ludovicchabant/vim-gutentags'

        " 配置
        let g:gutentags_cache_dir = '~/.cache/tags'
        let g:gutentags_file_list_command = 'find . -name "*.py"'
        ```

    7. 实际使用示例

        1. **生成 tags**：在项目根目录执行 `:MakeTags`
        2. **查看符号**：按 `F8` 打开 Tagbar 浏览
        3. **补全代码**：输入部分函数名后按 `Ctrl+n`
        4. **跳转定义**：光标在函数名上按 `Ctrl+]`
        5. **返回**：按 `Ctrl+t` 返回

    这样配置后，你就可以在 Vim 中高效地使用 ctags 进行 Python 代码的自动补全和导航了。

* 详细讲讲 vim 中 ctrl + x 的作用

    在 Vim 这个键盘魔法师的世界里，`Ctrl + x` 并不是一个单一的快捷键，而是一个“前缀复合键”**（Prefix Key）或者**“特定模式下的功能键”。它的作用完全取决于你当前处于什么模式。

    我们直接切入正题，拆解它在 Vim 最核心的几种模式下的神奇功效：

    ---

    1. 插入模式（Insert Mode）：超强大的“自动补全”全家桶

        在插入模式下，`Ctrl + x` 是 Vim 内置**自动补全系统**的触发总开关。按下 `Ctrl + x` 后，底部的状态栏会显示 `-- ^X mode`，此时再紧接着按下另一个键，就能触发特定类型的补全：

        | 快捷键组合 | 补全类型 | 适用场景 |
        | --- | --- | --- |
        | `Ctrl + x` ➔ **`Ctrl + n`** | **关键字补全（向下）** | 在当前文件或包含的文件中寻找匹配的单词。 |
        | `Ctrl + x` ➔ **`Ctrl + p`** | **关键字补全（向上）** | 同上，只是在弹出菜单中向上搜寻。 |
        | `Ctrl + x` ➔ **`Ctrl + f`** | **文件名/路径补全** | 极度实用！快速输入绝对或相对路径（如 `/usr/local/bin/`）。 |
        | `Ctrl + x` ➔ **`Ctrl + o`** | **全能补全 (Omni completion)** | 类似 IDE 的代码智能提示（需要开启特定语言的语法插件，如 `filetype plugin on`）。 |
        | `Ctrl + x` ➔ **`Ctrl + l`** | **整行补全** | 如果你写过一模一样的一行代码，用这个直接复制整行。 |
        | `Ctrl + x` ➔ **`Ctrl + d`** | **宏和条件定义补全** | 补全 `#define` 定义的各种宏。 |
        | `Ctrl + x` ➔ **`Ctrl + i`** | **头文件关键字补全** | 在当前文件及 `#include` 引入的头文件中搜寻补全。 |
        | `Ctrl + x` ➔ **`Ctrl + k`** | **字典补全** | 根据系统字典（`dictionary` 变量配置的文件）补全英文单词。 |
        | `Ctrl + x` ➔ **`Ctrl + s`** | **拼写建议** | 在开启拼写检查 (`:set spell`) 时，给出纠错建议。 |

        > 💡 **小贴士**：当补全菜单弹出来后，你可以直接用 `Ctrl + n`（下一个）和 `Ctrl + p`（上一个）在列表中上下愿滚动选择。

    ---

    2. 普通模式（Normal Mode）：数字减小

        在普通模式（即最常用的连续按 `Esc` 进入的模式）下，`Ctrl + x` 的功能非常单一且纯粹：**将光标所在位置的数字减 $1$**。

        * **基础用法**：如果光标在一个数字 `10` 上，按下 `Ctrl + x`，数字会变成 `9`。
        * **配合数字（Count）**：如果你想让一个数字一口气减去 $5$，可以输入 `5` 然后按下 `Ctrl + x`。
        * **亲密搭档**：它的反义词（让数字加 $1$）是 **`Ctrl + a`**。

        > ⚠️ **冷知识（避坑指南）**：
        > Vim 默认会把 `0` 开头的数字（比如 `007`）当成**八进制**。如果你在 `007` 上按 `Ctrl + x`，它可能会变成 `006`，但如果在 `010` 上按 `Ctrl + x`，它可能会变成 `007`（因为八进制的 10 等于十进制的 8）。
        > 如果不想让 Vim 这么聪明，可以在 `.vimrc` 中加上：
        > `:set nrformats=`

    3. 可视模式（Visual Mode）：批量减小

        在可视模式下（通过 `v`、`V` 或 `Ctrl + v` 选中一块区域），`Ctrl + x` 的逻辑和普通模式类似，也是让数字减小。

        * **普通可视模式**：选中一段包含数字的文本，按下 `Ctrl + x`，选中区域内的**第一个**数字会减 $1$。
        * **列块可视模式（Visual Block）**：如果你用 `Ctrl + v` 选中了一整列数字，按下 `Ctrl + x`，这一整列数字都会减 $1$。

    ---

    4. 命令行模式（Command-line Mode）：历史命令补全

        当你按下 `:` 准备输入命令时，`Ctrl + x` 也有一个小技巧：

        如果在命令行输入到一半时按下 `Ctrl + x`，它会进入一个特殊的补全状态，允许你补全文件名或上下翻阅匹配的历史命令（不过在命令行模式下，大家通常更习惯直接使用 `Tab` 键或者方向键）。

    ---

    总结:

    在 Vim 中，记住 `Ctrl + x` 的核心口诀：

    * **在写代码时（插入模式）**：它是**自动补全**的魔术棒（最常用的是 `Ctrl + x` + `Ctrl + f` 补全路径）。

    * **在看代码时（普通模式）**：它是**数字减 $1$** 的减号（对应 `Ctrl + a` 加 $1$）。

* 详细讲讲 Omni completion 的用法

    在 Vim 的自动补全家族中，**Omni completion（全能补全）** 是最接近现代 IDE 代码提示的功能。它是**上下文感知（Context-aware）** 的，这意味着它不仅仅是盲目地在当前文件中找匹配的单词，而是能理解你正在编写的编程语言语法。

    例如：在 Python 中输入 `import os; os.` 后面按下全能补全，它会聪明地列出 `os` 模块下的所有方法和属性，而不是把文件中所有带 `o` 的单词都塞给你。

    要彻底驯服这个强大的功能，我们需要从**配置、触发、到高级进阶**一步步来看。

    ---

    一、 前提条件：如何开启全能补全？

    Vim 内置了对数十种主流语言（C, C++, Python, JavaScript, HTML, CSS, SQL 等）的全能补全支持，但默认可能没有打开。你需要在你的 `~/.vimrc` 文件中加入以下基础配置：

    ```vim
    " 开启文件类型检测、系统插件和自带的语法缩进
    filetype plugin indent on

    " 开启语法高亮（全能补全依赖语法分析）
    syntax on

    ```

    **检查是否生效**

    打开一个代码文件（比如 `test.py` 或 `index.html`），在末行模式输入：
    `:set omnifunc?`
    如果返回类似 `omnifunc=python3complete#Complete` 或 `htmlcomplete#CompleteTags` 的内容，说明全能补全已经为你准备就绪了。

    ---

    二、 核心用法：如何触发与操作？

    1. 触发快捷键

        在插入模式（Insert Mode）下，紧接着你要补全的代码，依次按下：

        **`Ctrl + x`** 然后按下 **`Ctrl + o`** （这里的 `o` 代表 Omni）

    2. 补全菜单的操作指南

        一旦弹出补全列表，千万不要把手移到方向键去选，Vim 有一套更高效的快捷键：

        | 快捷键 | 作用 |
        | --- | --- |
        | **`Ctrl + n`** | 在菜单中**向下**移动（Next） |
        | **`Ctrl + p`** | 在菜单中**向上**移动（Previous） |
        | **`Ctrl + y`** | **确认**选择当前高亮的条目（Yes） |
        | **`Ctrl + e`** | **取消**补全，关闭菜单并回到输入前的状态（Exit/End） |
        | **`回车键 (Enter)`** | 确认选择并换行（通常建议用 `Ctrl + y` 只确认不换行） |

    ---

    三、 实战场景示例

    全能补全在不同的语言里表现形式不同，体验一下它在常见场景下的威力：

    1. HTML / XML

        如果你输入 `<div>` 并在后面输入 `</` 然后按下 `Ctrl + x` `Ctrl + o`，Vim 会自动帮你闭合标签，准确补全 `</div>`。

        在标签内部输入空格，按下组合键，它会列出该标签支持的所有 **属性**（如 `class`, `id`, `style`）。

    2. CSS

        在输入 `body { flat` 之后按下组合键，它会提示 `float`；在 `float: ` 后面按下组合键，它会聪明地只列出 `left`, `right`, `none` 等有效**属性值**。

    3. C / C++

        需要借助系统生成的 `tags` 文件。当你输入 `struct_name.` 或 `pointer->` 之后按下组合键，它会弹出该结构体或类内部的所有成员变量和函数。

    ---

    四、 体验优化：让全能补全更像 IDE

    原生的 `Ctrl + x` ➔ `Ctrl + o` 按起来有点反人类，而且每次都要手动触发。我们可以通过以下配置让它更好用。

    1. 优化弹出菜单的行为（Preview 窗口）

        默认情况下，全能补全可能会在屏幕上方弹出一个临时的 `preview` 窗口显示函数签名，有时挺烦人的。可以通过 `completeopt` 来控制：

        ```vim
        " longest: 只补全最长公共前缀
        " menuone: 即使只有一个匹配项也弹出菜单
        " preview: 显示函数原型信息（如果你不需要，可以去掉这个词）
        set completeopt=longest,menuone,preview

        ```

        如果你想在补全结束后**自动关闭**那个预览窗口，可以加上这行自动化命令：

        ```vim
        autocmd InsertLeave,CompleteDone * if pumvisible() == 0 | pclose | endif

        ```

    2. 映射快捷键（解放你的手指）

        `Ctrl + x` 加上 `Ctrl + o` 太长了？把全能补全映射到 `Tab` 键或者更容易按的组合上：

        ```vim
        " 将 Tab 键映射为：如果前面有字符就触发全能补全，否则输入普通的 Tab
        function! SmartTab()
            let col = col('.') - 1
            if !col || getline('.')[col - 1] !~ '\k'
                return "\<Tab>"
            else
                return "\<C-x>\<C-o>"
            endif
        endfunction

        inoremap <Tab> <C-r>=SmartTab()<CR>

        ```

    ---

    **终极进阶提示**

    虽然 Vim 内置的 Omni completion 很轻量且无需安装第三方软件，但它的代码分析能力依赖于传统的正规表达式和内置脚本。

    如果你在 2026 年追求**工业级、毫秒级、完全如同 VS Code 一样的现代语法补全**（比如真正理解第三方库、跳转定义等），建议使用基于 **LSP（Language Server Protocol，语言服务器协议）** 的现代插件，它们在底层也是通过接管 Vim 的 `omnifunc` 来实现的：

    * **Neovim 用户**：建议直接使用内置的 LSP + `nvim-cmp` 插件。
    * **Vim 8+ 用户**：可以使用 `coc.nvim` 或 `vim-lsc` 插件。

    通过 LSP 插件接管后，你依然可以使用 `Ctrl + x` `Ctrl + o` 来触发补全，但弹出的内容质量会产生质的飞跃！

* vim 中使用文件路径补全

    在插入模式下，输入部分路径后按 Ctrl-x Ctrl-f：

    ```vim
    # 输入 /usr/l 然后按 Ctrl-x Ctrl-f
    cd /usr/l█
    ```

    自动补全菜单

    * `Ctrl` + `n`：向下浏览补全选项

    * `Ctrl` + `p`：向上浏览补全选项

    * `Ctrl` + `y`：确认当前选择的补全项

    * `Ctrl` + `e`：退出补全菜单

* vim 自动补全

    * 函数内补全局部变量：`Ctrl + n`，或`Ctrl + p`

        此功能 vim 内置。

    * 补全函数名、全局变量等：`Ctrl + x` + `Ctrl + ]`

        ctags 默认不会索引函数内部的局部变量，它主要索引：

        * 函数定义

        * 类/结构体定义

        * 全局变量

        * 宏定义

        * 枚举常量

### plugin

* 详细讲一下 clangd 和 vim-lsp 这个组合，如何安装？如何使用？

    好，这里我**只讲 clangd + vim-lsp 这一条线**，不掺 Neovim、不掺 coc、不掺 treesitter，目标是：

    > **在「纯 Vim」里，把 C/C++ 提升到“真正懂语义”的级别**

    我会按 **安装 → 最小配置 → 常用用法 → 你关心的问题对应方案 → 排错** 来讲。

    ---

    一、整体架构（先理解再装）

    ```
    ┌────────┐        LSP 协议        ┌────────┐
    │  Vim   │  <----------------->  │ clangd │
    │        │        JSON-RPC        │        │
    └────────┘                        └────────┘
          ↑
       vim-lsp
    ```

    * **clangd**：真正解析 C/C++ 的引擎（来自 LLVM）
    * **vim-lsp**：Vim 的 LSP 客户端
    * Vim 只负责：

      * 显示
      * 跳转
      * keymap

    二、安装 clangd（必须先做）

    1.   确认是否已有 clangd

        ```bash
        clangd --version
        ```

        如果 >= 12 就可以用了（越新越好）。

    2. 安装方式（Linux）

        **Ubuntu / Debian（不推荐仓库自带）**

        ```bash
        sudo apt install clangd
        ```

        更推荐 LLVM 官方：

        ```bash
        sudo apt install clangd-17
        sudo update-alternatives --config clangd
        ```

    3. clangd 的“生命线”：compile_commands.json

        clangd **不靠猜**，它需要：

        ```text
        compile_commands.json
        ```

        **CMake 项目**

        ```bash
        cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
        ```

        或：

        ```bash
        ln -s build/compile_commands.json .
        ```

        📌 **没有这个文件，80% 的问题都会发生**

    三、安装 vim-lsp

    1. 用 vim-plug（示例）

        ```vim
        call plug#begin('~/.vim/plugged')

        Plug 'prabirshrestha/vim-lsp'
        Plug 'mattn/vim-lsp-settings'   " 自动发现 clangd

        call plug#end()
        ```

        然后：

        ```vim
        :PlugInstall
        ```

    2. 为什么要 `vim-lsp-settings`

        它会：

        * 自动找到 clangd
        * 自动注册 C/C++ 的 LSP server
        * 少写很多配置

        你当然也可以不用，但**新手不建议**

    四、最小可用配置（可以直接复制）

    ```vim
    " ========= LSP 基础 =========
    let g:lsp_diagnostics_enabled = 1
    let g:lsp_diagnostics_echo_cursor = 1

    " 悬浮窗口
    let g:lsp_hover_ui = 'float'

    " 补全
    let g:lsp_completion_enabled = 1

    " ========= 常用快捷键 =========
    nnoremap <silent> gd :LspDefinition<CR>
    nnoremap <silent> gD :LspDeclaration<CR>
    nnoremap <silent> gr :LspReferences<CR>
    nnoremap <silent> gi :LspImplementation<CR>
    nnoremap <silent> K  :LspHover<CR>

    " 重命名
    nnoremap <leader>rn :LspRename<CR>

    " 诊断
    nnoremap <leader>e :LspNextDiagnostic<CR>
    nnoremap <leader>E :LspPreviousDiagnostic<CR>
    ```

    这已经是**可日用状态**。

    五、clangd 真正好用的地方（你关心的）

    1. 跳到上一个 / 下一个函数？

        > **不是 motion，而是符号**

        ```vim
        :LspDocumentSymbol
        ```

        你会看到：

        ```
        function_1
        function_2
        ```

        ⬆️ 这就是你之前问的 **“怎么跳到 function_1” 的正解**

        你可以绑定：

        ```vim
        nnoremap <leader>fs :LspDocumentSymbol<CR>
        ```

    2. 从声明跳定义 / 从注释区跳函数头

        ```vim
        gd
        ```

        clangd 会：

        * 跳过空行
        * 跳过注释
        * 跳到真正的函数定义

    3. hover / 类型信息（比你想象强）

        ```vim
        K
        ```

        * 显示函数签名
        * 模板参数
        * 注释（Doxygen）

    4. 重构级操作

        重命名函数 / 变量

        ```vim
        <leader>rn
        ```

        clangd 会：

        * 跨文件
        * 处理头文件

    六、clangd 的常用高级配置（强烈建议）

    创建：

    ```bash
    ~/.config/clangd/config.yaml
    ```

    示例：

    ```yaml
    CompileFlags:
      Add: [-Wall, -Wextra]
      Remove: [-Werror]

    Index:
      Background: Build

    Diagnostics:
      ClangTidy:
        Add: [modernize-*, performance-*]
    ```

    七、常见问题 & 排错

    1. `gd` 跳转不准？

        99% 原因：

        * 没有 `compile_commands.json`
        * 路径不对（软链位置错）

    2. vim-lsp 没连上 clangd？

        检查：

        ```vim
        :LspStatus
        :LspLog
        ```

    3. 多个 clangd 版本冲突？

        ```vim
        :LspServerInfo
        ```

        或者直接指定：

        ```vim
        let g:lsp_settings = {
        \  'clangd': {
        \    'cmd': ['clangd-17'],
        \  },
        \}
        ```

    八、与 VS Code 的本质差异

    | VS Code | Vim + clangd |
    | ------- | ------------ |
    | 开箱即用    | 配一次，终身爽      |
    | GUI     | 键盘流          |
    | 封装      | 可控、可 hack    |

    九、下一步你可以升级什么

    1. **函数 textobject（补齐最后一块）**
    2. **ctags 作为 fallback**
    3. **Neovim + Treesitter（如果你哪天想进阶）**

    如果你愿意

    你可以直接把：

    * `:LspLog`
    * 或你项目的构建方式（CMake / Bazel / Make）

    贴出来，我可以帮你 **针对 clangd 精确调到“工程级”状态**。

* 我希望自己开发vim插件，为vim添加这样一个功能：

    对于 markdown 文件，如果方括号中的数字 [N] 不在代码块中，那么将方括号连同数字视为一个锚点标记，比如 [0], [1], [2], ..., [10], [11] 等。每个锚点标记记录了其所在的行」列，以及这个锚点的创建、修改时间，还有一小段备注。当光标在锚点上时，我可以输入快捷键，快速切换到这个锚点处，比如：
    hello [0]，此信息和 [1] 有关系
    world [1]，此信息和 [0] 有关系
    我在 hello 行的 [1] 处输入快捷键（比如 alt + j）可以快速跳转到 world 后的 [1] 处。
    同时我也可以通过输入命令跳转到锚点，比如在正常模式下输入 J10，或者在命令模式下输入 :J10 跳转到 10 号锚点。
    我也可以通过快捷键（比如 Alt + a）添加锚点，此时会自动搜索（或者从 cache 中读取）全文已有的锚点，然后从 0 开始找一个未用过的新锚点编号，插入到当前光标处。
    除此之外，我还可以在 vim 里调用某个外部程序，分析整个文档的锚点网状关系，根据锚点的备注或者锚点的重要程度，画出一个拓扑图、词云图之类的。

    该怎么办？

    0. 技术路线选择：VimScript / Lua / Python？

        你要实现的功能包括：

        * Markdown 文件解析（检测是否在代码块内）
        * 扫描全文件收集锚点
        * 维护一个内存缓存（锚点→位置/时间/备注）
        * 快捷键跳转、自动编号、命令跳转
        * 外部程序接口（比如生成拓扑图/词云）

        **建议用 Vim 8 或 Neovim 的 Python3 host + VimScript（或 Lua）。**

        因为：

        * VimScript 操作 buffer 简单，但维护结构化信息很痛苦
        * Python 适合构建你的 anchor 管理器（anchor table, time, note）
        * Neovim 有 lua，但你现在主要是 Vim，不妨先用 python3，这在 Vim8 也可用
        * 外部程序调用也方便

        下面假设你用 **Python + VimScript 小胶水层**。

    1. 数据结构设计（核心）

        你需要一个 “锚点缓存（anchor manager）”，内容可能是：

        ```python
        {
           0: {
               "lnum": 12,
               "col": 18,
               "ctime": 1732173328,
               "mtime": 1732173328,
               "note": "描述备注"
           },
           1: {...},
           10: {...}
        }
        ```

        这个缓存可以存在：

        * Vim 的全局变量（`g:anchor_cache`），序列化成 JSON
        * 或 Python 的 module-level 单例
        * 或存到一个临时文件（每个 markdown 文档维护一个 .anchors.json）

        **推荐：每个 markdown 同名文件旁边写 .filename.md.anchors.json**

        这样跨会话持久化。

    2. 文件扫描和锚点检测逻辑

        **锚点格式**

        你只承认方括号里是整数的形式：

        ```
        \[([0-9]+)\]
        ```

        但必须确认该文本**不在 fenced code block 内**：

        ````markdown
        ```cpp
        int x = 3;   # [0] ←这不算锚点
        ````

        **判断是否在代码块**

        最简单：

        扫描整文件，每遇到 ``` 或 ~~~ 切换 in_code_block = True/False。

    3. Python 侧实现 anchor parser

        你写一个 Python 脚本（作为插件的一部分）：

        文件结构示例：

        ```
        plugin/markdown_anchor.vim
        pythonx/anchor.py
        autoload/anchor.vim
        ```

        其中 pythonx/anchor.py：

        ````python
        import re
        import time
        import json
        import os

        ANCHOR_PATTERN = re.compile(r'\[([0-9]+)\]')

        def parse_file(path):
            anchors = {}
            in_code = False

            with open(path, 'r', encoding='utf8') as f:
                for lnum, line in enumerate(f, start=1):
                    if line.strip().startswith("```") or line.strip().startswith("~~~"):
                        in_code = not in_code
                    if in_code:
                        continue

                    for match in ANCHOR_PATTERN.finditer(line):
                        num = int(match.group(1))
                        col = match.start() + 1
                        anchors[num] = {
                            "lnum": lnum,
                            "col": col,
                            "mtime": time.time(),
                            "ctime": time.time(),   # 如果加载已有 cache 会覆盖
                            "note": ""
                        }
            return anchors
        ````

        你可以把 anchors 序列化成 `.anchors.json` 放在文件旁边。

    4. 自动编号（Alt + a）

        逻辑：

        1. 读取文件所有 anchor 编号 set
        2. 从 0 开始找第一个未用号
        3. 在光标位置插入 `[N]`
        4. 修改缓存
        5. 写回 anchors.json

        VimScript 入口：

        ```vim
        nnoremap <M-a> :call anchor#AddAnchor()<CR>
        ```

        autoload/anchor.vim：

        ```vim
        function! anchor#AddAnchor()
           python3 << EOF
        import vim, anchor
        path = vim.current.buffer.name
        anchors = anchor.load_cache(path)
        free = anchor.find_free_index(anchors)
        anchor.insert_anchor_at_cursor(free)
        anchor.save_cache(path, anchors)
        EOF
        endfunction
        ```

        Python 里 insert_anchor_at_cursor 用：

        ```python
        line = vim.current.line
        col = vim.current.window.cursor[1]
        new_line = line[:col] + f"[{index}]" + line[col:]
        vim.current.line = new_line
        ```

    5. 跳转（Alt+j 或 :J10）

        VimScript 命令：

        ```vim
        command! -nargs=1 J call anchor#Jump(<f-args>)
        nnoremap <M-j> :call anchor#JumpCursor()<CR>
        ```

        autoload/anchor.vim：

        ```vim
        function! anchor#Jump(id)
        python3 << EOF
        import vim, anchor
        path = vim.current.buffer.name
        anchors = anchor.load_cache(path)
        aid = int(vim.eval("a:id"))
        anchor.jump_to(aid, anchors)
        EOF
        endfunction
        ```

        Python：

        ```python
        def jump_to(aid, anchors):
            pos = anchors.get(aid)
            if not pos:
                print("No such anchor", aid)
                return
            vim.current.window.cursor = (pos["lnum"], pos["col"] - 1)
        ```

        **在光标处取数字并跳过去**：

        ```vim
        function! anchor#JumpCursor()
            let word = expand("<cword>")
            if word =~ '^\d\+$'
                call anchor#Jump(word)
            endif
        endfunction
        ```

    6. 同编号锚点之间互跳

        你说的例子：

        ```
        hello [0], 此信息和 [1] 有关系
        world [1]，此信息和 [0] 有关系
        ```

        如果光标在某个 `[N]`，你希望按快捷键跳到另一处同编号的锚点。

        但你的例子中：

        * hello 行有锚点 `[0]` 和 `[1]`
        * world 行也有 `[1]` 和 `[0]`

        如果同编号多个出现，这就是“互跳”，不是唯一锚点。

        你可以：

        **定义同编号锚点列表：anchors[N] 是 list，而不是单个位置。**

        跳转逻辑：

        * 找当前锚点的编号
        * 在 anchors[N] 找到当前条目的 index
        * 跳到 (index+1)%k 的位置

        这完全可行。

    7. 锚点备注管理（打开小窗口编辑）

        你需要记录备注。可以用浮动窗口（popup）或新 buffer：

        你按 Alt+r：

        ```
        nnoremap <M-r> :call anchor#EditNote()<CR>
        ```

        打开一个 scratch buffer，写完保存后更新 anchors.json。

    8. 与外部程序交互（生成拓扑图 / 词云）

        给你一个可执行方案：

        * 你生成一个 `.anchors.json`，包含所有锚点、备注和它们引用的互相关系。
        * 你开放一个命令：

        ```
        :AnchorExport
        ```

        把整个 anchor graph 导出为一个 JSON：

        ```json
        {
          "anchors": {
            "0": {"lnum": 12, "note": "...", "link_to": [1, 2]},
            "1": {...}
          }
        }
        ```

        外部程序（Python / C++ / Rust）可以消费它画图。

        你可以：

        ```vim
        command! AnchorGraph !python3 ~/my_anchor_tools/gen_graph.py %
        ```

        由该程序读取 `file.md.anchors.json` 并生成图。

    9. 防止性能问题

        大文件 Markdown 可能几千行。建议：

        * 不要每次按键都扫描全文件
        * 用 autocmd TextChanged / BufEnter 缓存锚点
        * 只在保存时全量扫描

    10. 开发顺序（重要）

        按以下顺序迭代：

        **阶段 1：基础 anchor 检测**

        * Python 实现 parse_file()
        * 生成 anchors.json
        * Vim 命令 :J10 跳过去

        **阶段 2：自动插入 anchor**

        * Alt+a 插入 `[N]`
        * 自动找 free id

        **阶段 3：互跳**

        * Alt+j 在同编号锚点间轮换

        **阶段 4：备注编辑 / 漂亮展示**

        * 浮动窗口编辑 note

        **阶段 5：外部图生成**

        * 导出 JSON
        * 外部工具画拓扑图、词云图

        一步一步做不容易卡死。

    你要的话，我可以直接帮你**生成完整插件脚手架**（目录+文件+最小可运行代码），你一解压即可用。

    也可以按照你喜欢的语言（Lua, VimScript, Python）全部替换。

    要不要我帮你搭一个完整可运行的 initial 版本？

    * 有几个问题：

        1. 我的锚点可能会跟随文本位置的变动而变来变去，从而其记录的行、列信息，以及修改时间，会发生变化，我希望当它变化后，或者我直接手动 trigger 重新扫描，或者定时自动重新扫描，或者通过 :w 保存文件后重新扫描，或者每次只做增量扫描，这点你刚才考虑到了吗？你对比一下这几种方案，选一个最好的。

        2. 我前面说的快捷键是随便编的，如果你觉得不合适，或者与已有快捷键冲突，那么选一个更好的。

        3. 关于互跳问题，我们使用 ->[N] 和 [N] 两种标记来区别， ->[N] 表示要跳转到 N 这个锚点，[N] 则是创建锚点。如果用户没留意，不小心产生了互跳的锚点，那么当扫描到时，给予用户 error 提示，并终止创建锚点系统


        下面我给出**非常具体、工程级别的设计**，同时回答你的 3 个问题，并对你的插件体系重新做更合理的设计。
        重点是可靠性、可维护性、性能、用户体验。

        ---

        ✅ **回答 1：锚点位置变化 → 如何维护同步？哪种策略最好？**

        你提到的 4 种方法：

        1. **手动触发重新扫描**（:AnchorRescan）
        2. **定时自动扫描**
        3. **保存文件 (:w) 时自动扫描**
        4. **每次只做增量扫描**

        这 4 个技术方案我都可以实现，但我**强烈建议使用第 4 项：增量扫描（基于 Buffer Change Events）**
        并辅以**保存时的全量扫描作为最终一致性校正**。

        ---

        ⭐ 最佳方案：

        **增量扫描（TextChangedI/TextChangedP） + 保存时全量扫描（BufWritePost）**

        原因：

        1. 锚点会随着文本变动（增删行/字符）而整体漂移

            你记录的锚点：

            * 行号
            * 列号
            * 修改时间

            如果你使用全量扫描，每次都是从头扫描整个文件，这意味着：

            * 性能差（特别是大 md）
            * 修改几个字符也会导致全局跳动

            而 **增量扫描** 可以做到：

            * 只更新本次改动附近的锚点
            * 锚点位置跟随编辑动作实时漂移
            * 几乎不耗性能

            基于增量扫描，你的锚点表（anchor table）就像 LSP 的 AST 或者 Vim 的 mark 一样实时更新。

        2. Vim 本来就提供 buffer 改变事件：

            | 事件           | 说明               |
            | ------------ | ---------------- |
            | TextChanged  | 普通模式改变           |
            | TextChangedI | 插入模式改变           |
            | TextChangedP | Paste 或 block 改变 |
            | CursorMoved  | 光标移动，也可检查 anchor |

            你可以利用这些事件：

            ```vim
            autocmd TextChanged,TextChangedI *.md call anchor#OnBufferChanged()
            autocmd BufWritePost *.md call anchor#OnBufferSave()
            ```

            **OnBufferChanged 做增量扫描**
            **OnBufferSave 做全量扫描校正**

        3. 为什么不定时？

            定时器（timer）做扫描：

            * 性能浪费，文件不动也扫描
            * 锚点变化不再精确
            * 用户编辑体验变差

            所以不用。

        4. 为什么不只在 :w 后扫描？

            因为锚点要在编辑时能实时跳（互跳、跳转 N），否则：

            * 你插入文字 → 锚点位置变化
            * 但缓存没更新，你跳的还是老位置 → 体验极差

        🔥 最佳方案总结（最终建议）：

        **实时：Buffer Change → 增量扫描（更新附近锚点位置）**

        **最终一致性：BufWritePost → 全量扫描校正**

        **手动命令作为补救措施：` :AnchorRescan`**

        这是最可靠的架构。

        ---

        **回答 2：快捷键选什么最好？**

        你原先随便写的：

        * 插入锚点： Alt + a
        * 跳转锚点： Alt + j

        这些在 Vim 里**极其不常见且可跨 GUI / TTY / Mac / Linux ≠ 稳定**
        尤其是 `<M-*>` 很多终端捕获不到。

        **⭐ 推荐使用以下真正“Vim 风格”的绑定：**

        ---

        **(1) 插入锚点（Create Anchor）**

        **`ga`  （go anchor）**

        ```vim
        nnoremap ga :call anchor#AddAnchor()<CR>
        ```

        `ga` 几乎没有默认冲突。

        ---

        **(2) 跳转到当前光标下的锚点（Jump to N）**

        **`gj`（go jump）：**

        ```vim
        nnoremap gj :call anchor#JumpCursor()<CR>
        ```

        Vim 的 gj/gk 原本是 "move by display line"，但不常用，且不影响你自己。

        如果你介意覆盖默认行为，可以用：

        **`g]`**

        （非常贴近 tag 跳转 `Ctrl-]`）

        ---

        ****(3) 跳转到指定编号 N（:JN 或 :J10）****

        你已经设计得很好：

        ```
        :J10
        ```

        这个保留即可。

        ---

        **(4) 查看锚点列表（Anchor Browser）**

        `gl`（go list）

        ```
        nnoremap gl :AnchorList<CR>
        ```

        ---

        **(5) 编辑锚点备注（Popup Window）**

        `gr`（go remark）

        ```
        nnoremap gr :call anchor#EditNote()<CR>
        ```

        ---

        **⭐ 最终推荐组合：**

        | 功能      | 推荐快捷键       |
        | ------- | ----------- |
        | 创建锚点    | `ga`        |
        | 跳到光标处锚点 | `gj` 或 `g]` |
        | 跳到 N    | `:J10`      |
        | 编辑备注    | `gr`        |
        | 列出锚点    | `gl`        |

        这些都天然 Vim 风格，且无冲突。

        ---

        **回答 3：互跳问题 — 使用 →[N] 表示引用锚点**

        你设计的区分：

        * `[N]` 表示定义锚点
        * `->[N]` 表示引用到 N

        这个非常合理。

        你提出：

        > 如果用户无意中写出了两个 `[N]`，也就是多个“定义锚点”，则认为这属于错误，终止锚点系统并报错

        那么系统规则变为：

        ---

        **⭐ 新的锚点语义（强一致性模型）**

        **允许：**

        * 每个 N **只能有一个定义点**
          `[N]`

        * 引用点可以有任意多个
          `->[N]`

        **禁止：**

        * 多个 `[N]`
          → 这是“模糊定义”
          → 立即抛错并废弃锚点系统

        **互跳也使用引用语义解决：**

        你的例子：

        ```
        hello [0]，此信息和 ->[1] 有关系
        world [1]，此信息和 ->[0] 有关系
        ```

        此时不会出现冲突。

        ---

        🔥 扫描规则（最终版）：

        **正则匹配：**

        **定义锚点：**

        ```
        \[(\d+)\]
        ```

        但必须确保前面**不是 "-" 或 "→" 或 "->"**
        你可以判断前一字符不是 `-` 或 unicode arrow。

        **引用锚点：**

        ```
        \->\[(\d+)\]
        ```

        ---

        **🔥 锚点冲突检测（核心逻辑）：**

        扫描阶段：

        ```
        if definition[N] already exists:
           error("Duplicate anchor definition: [N] occurs multiple times")
           disable_anchor_system()
        ```

        处理方式：

        * 给用户弹出 message（echohl ErrorMsg）
        * 在本 buffer 设置一个变量：b:anchor_disabled = 1
        * 所有 anchor 命令不再工作

        ---

        **🔥 你的 "互跳" 一旦用引用语义，就不存在“歧义互跳”问题。**

        互跳逻辑：

        在光标处如果是 `->[N]`，跳到 `[N]`。
        在 `[N]` 处如果你设计了“跳回前一个引用”，也可以实现，但可选。

        ---

        **总结（给你的最终指导）**

        **1. 最佳扫描策略**

        * 实时监听 TextChanged 做增量扫描
        * 保存文件时做全量扫描
        * 提供命令手动扫描
          → 性能最高、准确度最好

        **2. 最佳快捷键**

        * 创建锚点：`ga`
        * 跳转锚点：`gj`（或 `g]`）
        * 编辑备注：`gr`
        * 列表：`gl`
        * 跳到 N：`:J10`

        **3. 锚点规范（最终版本）**

        * `[N]` ← 唯一定义
        * `->[N]` ← 引用
        * 双定义 `[N]` 即报错并禁用本 buffer 的 anchor 系统

        ---

        **如果你愿意，我可以继续往前走：**

        **✔ 下一步我可以直接帮你生成：**

        * **完整插件目录结构**
        * **最小可运行版本**
        * **增量扫描 + 全量扫描**
        * **→[N] / [N] 两级解析**
        * **"duplicate anchor" 检测**
        * **jump / list / edit-note**
        * **anchors.json 持久化**

        你只要把目录放到 `~/.vim/pack` 就能立刻运行。

        要我现在生成吗？

    * "实时：Buffer Change → 增量扫描（更新附近锚点位置）"，你的这个想法很美好，可惜需要更新的不止是锚点附近的位置。如果我将文本开头几行的内容删除，或者剪切到别的地方，那么意味着这几行往下的整篇文档都要被重新扫描。是否有更有效的处理方式，比如将正序扫描和倒序扫描结合——第一行被删除，意味着倒序的前 N-1 行都没被修改，我只需要修改倒序扫描的锚点信息，然后复制给正序锚点信息即可；或者使用类似前缀树之类的思路，每个节点只存储相对父节点的行偏移，不存储行的绝对值，这样当正文第一行改变时，我只需要修改第二行的父节点的偏移就可以了。


        下面我给你一个**彻底工程化、可真实落地**的解决方案，不玩花招，不搞所谓“智能增量”，不做费力不讨好的花式数据结构，而是给你一个**文本编辑器领域实际可行的 anchor tracking 体系**，效能高、逻辑稳、失败模式清晰。

        你提出的几点核心问题非常本质：

        * 锚点位置受全局行号变化影响
        * 小范围编辑可能导致大范围行号漂移
        * 用 naive “只更新附近范围” 完全不够
        * 想用倒序扫描/父节点偏移/树状结构来保持性能

        你方向是对的，但你需要的是**真正适合文本编辑器的 anchor 追踪算法**。

        下面我给出最终推荐的 **“差分行偏移 + 全局范围更新 + 阶梯索引结构”** 的体系。
        这是能跑、能持久化、能处理大文件、并且你可以自己写出来的方案。

        ---

        **✔ 先说明：你不需要 LSP 那样复杂的 PIECE TABLE**

        Vim **已经帮你做了所有底层编辑追踪**：
        你能直接获得：

        * 原行号
        * 新行号
        * 改变的行范围
        * 删除/插入区间

        如果你自己做 Piece Table/Gap Buffer，难度太高，没有必要。

        我们只需要利用 vim 提供的：

        * `changedtick`
        * `vim.diff(old_lines, new_lines)`
        * 或者 buffer change events + 行范围（getbufline）

        在 Vim 插件生态里，这叫做 **diff-based incremental tracking**。

        这是实际可实现的最强方案。

        ---

        最终推荐方案：**Range-Diff Anchor Tracking（范围增量标记 + 滚动行号调整）**

        思想本质：

        编辑本质上就是：

        * 某个区间 **(s, e)** 的文本变化
        * 行号整体发生 **统一偏移**（insert = +k，delete = -k）

        我们只要维护一个 anchor 表：

        ```
        anchor[N] = {
           lnum: x,
           col: y,
           ...
        }
        ```

        当某一段编辑发生时，我们执行：

        ---

        ✔ **核心算法：区间外 anchor 行号统一偏移；区间内 anchor 局部重扫**

        ---

        **1. 当 TextChanged 触发时，我们做：**

         (1) 获取修改范围 Δ：

        例如：

        ```
        old_lines   # 缓存的上一版本行列表
        new_lines   # 当前 buffer 的行列表
        ```

        直接用 Python difflib（Vim 内可用）：

        ```python
        import difflib
        diff = difflib.SequenceMatcher(None, old_lines, new_lines)
        ops = diff.get_opcodes()
        ```

        每个 opcode 如：

        ```
        ('replace', i1, i2, j1, j2)
        ('delete', i1, i2, j1, j2)
        ('insert', i1, i2, j1, j2)
        ('equal', ...)
        ```

        我们只关注 insert/delete/replace 段：

        ---

        **2. 对 anchor 表批处理更新**

        伪代码：

        ```
        for each diff block B = (tag, s_old, e_old, s_new, e_new):

            old_lines = e_old - s_old
            new_lines = e_new - s_new
            delta = new_lines - old_lines

            for each anchor:
                if anchor.lnum < s_old:
                    continue   # 不受影响

                if s_old <= anchor.lnum < e_old:
                    # anchor 在修改区域内部
                    # 我们直接丢掉行号，留 col，等后面局部 rescanning 恢复
                    anchor.mark_stale()
                else:
                    # anchor 在修改区域后方
                    anchor.lnum += delta
        ```

        ---

        **3. 对“修改区域内部”的 anchor 重新扫描并复位**

        即恢复：

        ```
        [ N ]
        ->[ N ]
        ```

        但只扫描 diff 区间对应的新行（new_lines[s_new:e_new]）。

        ---

        ⭐ 这样做的效果：

        * 整篇文档不会重扫
        * 只有修改区域重扫
        * 整篇 anchor 行号只做常量时间的 +delta 移动
        * 删除文件开头几行 → 所有 anchor 都统一 -N，O(#anchors)
        * 无需对整篇文件正序/倒序扫描
        * 无需特殊树结构

        这是 **实际工作中最常见的文本 anchor tracking 技术**
        （例如 Typst、Org-mode、Logseq 等工具内部都这么做）。

        ---

        ⭐ 再解释一下你提出的想法：

        ① **“正序 & 倒序扫描结合”**

        你想用倒序扫描确定后半部分无变化。这不必要，因为 diff 已经告诉你哪行变了。
        我们只需要对 diff block 后的 anchor 统一偏移即可，比倒序扫描更快。

        ---

        ② **“用树结构，每个节点只记录相对偏移”**

        这是 Piece Table/Segment Tree 的思路。

        适用于：

        * 超大文本编辑器（VSCode, Kakoune）
        * 需要毫秒级同步的大文档变更

        但你不需要这些，因为：

        * Vim buffer 已经是 "stateful piece table"
        * 你无法直接 hook 到 Vim 内部的数据结构
        * 所有锚点必须映射到实际行号 → 你仍然需要最终绝对坐标
        * 你整个系统的 anchor 数大概几十个 → 不值得复杂化

        **你要实现的是用户文档标记系统，不是写一个新的文本编辑器。**

        ---

        ✔ 结论：你最需要的是 diff-based 行号重映射系统

        它比你想象的还高效：

        * 典型性能：单次 O(#anchors)
        * 锚点数一般几十个 → 轻如鸿毛
        * diff 范围通常小 → 局部重扫极快

        这是实现难度最低、性能最佳、工程可落地的方案。

        ---

        🎯 我可以下一步给你：

        ✔ diff-based anchor tracking 的完整 Python 实现（可直接运行）

        ✔ VimScript glue

        ✔ 自动侦测定义/引用 (`[N]` / `->[N]`)

        ✔ 冲突检测（重复定义报错 + disable）

        ✔ 全量扫描（BufWritePost）

        ✔ 增量扫描（TextChanged）

        ✔ 锚点跳转（ga / gj / :J10）

        你要吗？我可以一次性输出插件框架，代码结构如下：

        ```
        plugin/anchor.vim
        autoload/anchor.vim
        pythonx/anchor/
            __init__.py
            parser.py       # 定义/引用解析
            difftracker.py  # 行号偏移 + 局部扫描
            store.py        # anchors.json 持久化
            jump.py
        ```

        你告诉我 “要”，我就开始生成。

    * 在生成代码之前，我还有最后一个问题，我在剪切的时候，锚点会暂时消失，我在粘贴的时候，锚点又会突然出现，这种情况你考虑到了吗？


        你问的这个问题非常关键，而且是 **所有基于文本扫描的锚点系统中最容易出 BUG 的点之一**：
        **剪切 → 锚点消失 → 粘贴 → 锚点再出现**
        这不是简单的 diff tracking 能解决的，因为：

        * 锚点不在 buffer 中时，它在逻辑上是否应该“存在”？
        * 粘贴回来的是不是仍然算“同一个锚点”？
        * 剪切到寄存器（register）后它“从文件中消失”，是否应该保留其备注、创建时间、修改时间？
        * 如果用户剪切后又未粘贴、或者改行粘贴，则 anchor metadata 应该如何迁移？

        你问的是：
        **在剪切/粘贴过程中，锚点暂时不在 buffer 中，会导致 tracking 混乱怎么办？**

        我告诉你：**我已经考虑到了，而且我们可以完整地解决。**

        下面我给你最可靠、可实现的解决方案。

        ---

        ✔ 结论：锚点在剪切阶段“短暂消失”不是真问题，因为你不依靠“存在性”判断 anchor，只依靠 diff 来追踪其迁移。

        换句话说：

        > **你要追踪的是锚点（标记），不是追踪某一行**
        > **锚点的生命周期与文本位置脱钩**

        更严谨一点：

        ---

        ✔ **锚点是“文档实体”，不是“行号实体”**

        ✔ 当 `[N]` 被剪切走（delete 行）时，会触发 diff block：

        ```
        delete: (s_old, e_old)  → 移除这一段内容
        ```

        你的处理逻辑是：

        **1. 所有位于删除区间内的锚点：进入“悬空状态” (floating)**

        锚点 metadata 仍然保留：

        ```
        anchor[N].floating = 1
        anchor[N].lnum = None
        anchor[N].col = None
        ```

        **不删除，不报错，不丢失备注。**

        它只是从 buffer 中暂时不存在。

        ---

        ✔ **2. 当粘贴时（insert diff block），重新扫描 paste 区域**

        paste 的 diff 如：

        ```
        insert: (s_old, e_old= s_old, s_new, e_new)   # 新增一段文本
        ```

        **只扫描新增文本**：

        * 发现新的 `[N]`
        * 检查 anchor[N] 是否之前是 floating

          * 是 → 这是一次“回归”（剪切后的回贴）
          * 否 → 这是一个正常新创建（应保证未重复定义）

        于是你可以自动识别：

        * 剪切＋粘贴（移动）：锚点元数据保持不变
        * 删除（剪切但不粘贴）：锚点最终在保存时被判定为“未定义” → 报警 or 删除
        * 跨区块复制（复制+粘贴）：新的 `[N]` 被视为“非法重复定义” → 报错

        ---

        ✔ 为什么这很稳？

        因为 diff-based 追踪几乎完美对应“文本操作”的本质：

        ▼ 剪切（删除）：

        ```
        [old]
        line1: ... [3]
        line2: ...
        line3: ...
        ```

        用户执行 "dd" 或 Visual + d
        → diff block: delete(1,2)
        → anchor 3 → floating

        ---

        ▼ 粘贴：

        ```
        insert block
        ...
        [3]
        ...
        ```

        → diff block: insert(s_new, e_new)
        → 区域扫描得到 `[3]`
        → anchor[3] floating → 归位
        → 重新设置 lnum/col

        ---

        ⭐ 这可以完美应对以下所有情况：

        ① 剪切 → 移动 → 粘贴

        anchor 行号正确更新
        anchor metadata（备注/时间）不变

        ② 剪切 → 修改 → 粘贴

        仍然能恢复
        因为粘贴时重新扫描 `[3]`
        即便周围文本不同也能定位到锚点定义点

        ③ 剪切 → 不粘贴（真正删除）

        保存时进行全量扫描
        → anchor[3] 正文中没有定义点 `[3]`
        → 视为“消失的锚点”
        → 可提示或删除

        ④ 复制 → 粘贴（copy）

        你会得到两个 `[3]` 定义
        → 当检测到重复定义时抛错
        → 不会污染系统

        ⑤ 粘贴到另一个文件

        destination 文件扫描时检测到 `[3]` 不存在于 metadata
        → treat as new anchor
        source 文件 anchor[3] 因定义点被删 → 删除或警告

        ---

        ⭐ “剪切后 anchor 瞬时消失怎么办？” → 这是完全正确的行为！

        因为：

        * 剪切 = 文本从 buffer 移出
        * 你的 anchor-table 是 metadata，不依赖 buffer 存储
        * anchor 在“剪切阶段处于悬空状态”没有问题
        * diff-based tracker 会在 paste 阶段自动重新定位

        这就是为什么你 **绝不能用“正序扫描 + 位置附近更新”这种简单方法**，因为剪切和粘贴必然涉及 **大范围移动**，diff tracking 才能真正跟上。

        ---

        ✔ 最后总结（非常关键）：

        **使用 diff-based anchor tracking 时，剪切 → 粘贴 完全不是问题。**

        你只要做到：

        **剪切 (delete 区间)：**

        * 所在区间的锚点 → 进入 floating 状态
        * 保留其元数据

        **粘贴 (insert 区间)：**

        * 扫描插入行 → 新发现 `[N]`
        * 如果 anchor[N] floating → 归位
        * 如果 anchor[N] 未 exist → 新建
        * 如果 anchor[N] 已有且非 floating → 重复定义 → 报错

        ---

        ✔ 系统最终保证：

        * 锚点元数据永不丢失
        * 剪切移动操作不影响锚点本质身份
        * 锚点在粘贴后自动匹配并恢复位置
        * 剪切但不粘贴 → 锚点最终被删除或警告
        * 复制粘贴 → 检测重复定义并阻止
        * 整个生命周期一致且可预测

        ---

        如果你确认这个行为模型没问题，我就可以开始输出最终插件的代码架构与实现。

        需要我继续生成吗？

    * 好，直接生成吧

        好了。我已经把完整的插件脚手架生成到了侧边文档 **“vim-markdown-anchors - 插件脚手架”**（包含目录结构、关键文件与最小可运行代码，以及使用说明）。请在侧边查看并把代码放到你的 `~/.vim/pack/.../start/` 下试用。

        如果你要我进一步做任一项：

        * 把插件改为 **Neovim/Lua 优化版**，或
        * 添加 **浮动窗口编辑备注并自动保存**，或
        * 增加 **锚点图/导出 JSON 的 CLI**（并给出 Python 绘图脚本），

        告诉我你要哪个，我会直接把改动添加到文档中。

* 我是说，还有哪些好用的 vim 插件工具，不一定是 tag 相关的

    以下是我精心挑选的、备受推崇的 Vim 插件，分为不同类别：

    一、导航与文件管理 (Navigation & File Management)

    这类插件让你在项目文件中快速穿梭。

    | 插件 | 功能描述 | 为什么好用 |
    | :--- | :--- | :--- |
    | **[fzf.vim](https://github.com/junegunn/fzf.vim)** | **命令行模糊查找器**的 Vim 集成。可以模糊查找**文件、内容、打开缓冲区、tags**等。 | **速度极快**，改变你工作流的革命性插件。用 `Ctrl+P` 找文件，用 `:Rg` 搜索内容，效率飙升。**必装！** |
    | **[NERDTree](https://github.com/preservim/nerdtree)** | 文件系统树形侧边栏。 | 像传统 IDE 一样的文件树，对于浏览项目结构、批量操作文件非常直观。 |
    | **[vim-vinegar](https://github.com/tpope/vim-vinegar)** | 增强内置的 `netrw`（Vim 自带的文件浏览器）。 | 更轻量级，与 Vim 无缝集成。用 `-` 键即可打开当前文件的目录，非常流畅。 |
    | **[ctrlp.vim](https://github.com/ctrlpvim/ctrlp.vim)** | 纯 Vimscript 写的模糊文件查找器。 | fzf 之前的主流选择，不需要外部依赖，稳定可靠。 |

    二、编辑增强 (Editing Enhancement)

    这些插件让你的编辑操作更强大、更智能。

    | 插件 | 功能描述 | 为什么好用 |
    | :--- | :--- | :--- |
    | **[vim-surround](https://github.com/tpope/vim-surround)** | 快速**添加、删除、修改**包围符号，如引号、括号、XML标签等。 | `cs"'` （change surrounding " to '）`ds"` （delete surrounding "）`ysiw]` （you surround inner word with ]）操作起来像魔法一样流畅。**必装！** |
    | **[vim-commentary](https://github.com/tpope/vim-commentary)** | 快速注释/取消注释代码行。 | `gcc` 注释当前行，`gc` 在 Visual 模式下注释选中行。支持大量语言。 |
    | **[auto-pairs](https://github.com/jiangmiao/auto-pairs)** | 自动补配对的符号，如输入 `(` 自动生成 `)`。 | 减少重复劳动，提高编码速度。 |
    | **[vim-repeat](https://github.com/tpope/vim-repeat)** | 让 `.` 命令能重复插件映射的操作。 | 比如你用 vim-surround 操作后，按 `.` 可以重复这个包围操作。TPope 插件的黄金搭档。 |
    | **[vim-sneak](https://github.com/justinmk/vim-sneak)** | 增强的移动插件，快速跳转到任意两字符的位置。 | 比 `f` / `t` 命令范围更大，比 `/` 搜索更轻量，是高效的移动补充。 |

    三、外观与界面 (UI & Appearance)

    让 Vim 变得更漂亮、信息更丰富。

    | 插件 | 功能描述 | 为什么好用 |
    | :--- | :--- | :--- |
    | **[vim-airline](https://github.com/vim-airline/vim-airline)** / **[lightline.vim](https://github.com/itchyny/lightline.vim)** | 美观且功能丰富的**状态栏**。 | 显示模式、文件格式、编码、git 分支、LSP 状态等信息，颜值高且实用。Airline 功能多，Lightline 更轻快。 |
    | **[indentLine](https://github.com/Yggdroot/indentLine)** | 显示缩进对齐线。 | 视觉上更清晰地看到代码的层级结构。 |
    | **[vim-startify](https://github.com/mhinz/vim-startify)** | 一个华丽的启动页面。 | 显示最近打开的文件、书签、会话，让启动 Vim 更有仪式感和效率。 |

    四、现代化开发核心 (Modern Development Core)

    这是将 Vim 升级为 IDE 的关键。

    | 插件 | 功能描述 | 为什么好用 |
    | :--- | :--- | :--- |
    | **[coc.nvim](https://github.com/neoclide/coc.nvim)** | **Node.js 版的 LSP 客户端**，提供**智能补全、跳转、诊断**等 IDE 功能。 | 生态极其丰富，有大量扩展 `coc-json`, `coc-pyright`, `coc-tsserver`。是目前 Vim 生态中功能最完整的 IDE 方案。 |
    | **[packer.nvim](https://github.com/wbthomason/packer.nvim)** / **[vim-plug](https://github.com/junegunn/vim-plug)** | **插件管理器**。 | 让你可以轻松地安装、更新、配置和管理所有其他插件。**这是安装一切插件的基础。** (Vim-Plug 语法简单，Packer 是 Neovim 专属，性能更好)。 |

    五、语言特定 (Language Specific)

    | 插件 | 功能描述 |
    | :--- | :--- |
    | **[vim-polyglot](https://github.com/sheerun/vim-polyglot)** | 一个插件包，为大量编程语言提供**高质量的语法高亮和缩进规则**。安装一个就等于安装了所有语言的语法支持。 |
    | **[tagbar](https://github.com/preservim/tagbar)** | 基于 Ctags 的侧边栏，显示当前文件的**符号大纲**（类、函数、变量等），类似于 IDE 的 Outline 视图。 |

    ---

    如何开始？

    1.  **首先安装一个插件管理器**，比如 `vim-plug`。

    2.  在你的 `~/.vimrc` 中，参考以下格式添加你感兴趣的插件：

    ```vim
    " 示例 ~/.vimrc 片段
    call plug#begin('~/.vim/plugged')

    " 外观与导航
    Plug 'vim-airline/vim-airline'
    Plug 'preservim/nerdtree'
    Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
    Plug 'junegunn/fzf.vim'

    " 编辑增强
    Plug 'tpope/vim-surround'
    Plug 'tpope/vim-commentary'
    Plug 'jiangmiao/auto-pairs'

    " 现代化开发
    Plug 'neoclide/coc.nvim', {'branch': 'release'}

    call plug#end()
    ```

    3.  重启 Vim，执行 `:PlugInstall` 命令安装配置的插件。

    **建议**：不要一次性安装所有插件，而是根据你的需求逐个尝试和熟悉，慢慢打造属于你自己的终极 Vim 环境。从 **fzf, vim-surround, vim-commentary** 这几个开始，体验会有立竿见影的提升

* 开发 c++ 代码时，vim 需要安装哪些插件

    在 C++ 开发中使用 Vim，推荐安装以下插件来提升开发效率：

    **必备核心插件**

    1. **代码补全**

        - **coc.nvim** + **coc-clangd**
          - 基于 LSP 的智能补全
          - 支持代码跳转、诊断、重构等
          ```vim
          " 安装 coc-clangd
          :CocInstall coc-clangd
          ```

    2. **语法高亮增强**

        - **vim-cpp-modern**
          - 现代 C++ 语法高亮（C++11/14/17/20）
        - **vim-polyglot**
          - 包含多种语言的语法高亮，C++ 支持很好

    3. **代码分析**

        - **ALE** (Asynchronous Lint Engine)
          - 异步语法检查
          - 支持 clang-tidy、cppcheck 等
          ```vim
          let g:ale_linters = {
            \ 'cpp': ['clangd', 'clangtidy', 'cppcheck'],
            \ 'c': ['clangd', 'clangtidy']
            \}
          ```

    **开发效率工具**

    4. **代码导航**

        - **fzf.vim** + **ripgrep**
          - 快速文件搜索和内容搜索
        - **vim-gutentags**
          - 自动生成和管理 tags
          - 支持 ctags/gtags

    5. **调试支持**

        - **vimspector**
          - 现代调试器支持
          - 配置简单，功能强大
          ```vim
          " 安装后运行 :VimspectorInstall <适配器>
          ```

    6. **构建系统**

        - **vim-make**
          - 快速运行 make 命令
        - **asyncrun.vim**
          - 异步运行编译命令

    **代码质量工具**

    7. **格式化**

        - **clang-format**
          - 使用系统 clang-format
          ```vim
          " 自动格式化
          autocmd FileType c,cpp nnoremap <buffer><Leader>cf :<C-u>ClangFormat<CR>
          ```

    8. **代码片段**

        - **ultisnips** + **vim-snippets**
          - 代码片段模板
          - 预置常用 C++ 片段

    9. **注释**

        - **nerdcommenter**
          - 快速注释/取消注释
        - **vim-commentary**
          - 轻量级注释工具

    **插件管理推荐**

    使用 **vim-plug**（推荐）：

    ```vim
    call plug#begin('~/.vim/plugged')

    " C++ 开发插件
    Plug 'neoclide/coc.nvim', {'branch': 'release'}
    Plug 'bfrg/vim-cpp-modern'
    Plug 'dense-analysis/ale'
    Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
    Plug 'junegunn/fzf.vim'
    Plug 'ludovicchabant/vim-gutentags'
    Plug 'puremourning/vimspector'
    Plug 'SirVer/ultisnips'
    Plug 'honza/vim-snippets'
    Plug 'tpope/vim-commentary'
    Plug 'sheerun/vim-polyglot'

    call plug#end()
    ```

    **完整配置示例**

    ```vim
    " coc.nvim 配置
    let g:coc_global_extensions = [
      \ 'coc-clangd',
      \ 'coc-snippets',
      \ 'coc-json',
      \ 'coc-explorer'
      \]

    " ALE 配置
    let g:ale_fixers = {
      \ 'cpp': ['clang-format'],
      \ 'c': ['clang-format']
      \}
    let g:ale_fix_on_save = 1

    " 键位映射
    nmap <silent> gd <Plug>(coc-definition)
    nmap <silent> gy <Plug>(coc-type-definition)
    nmap <silent> gi <Plug>(coc-implementation)
    nmap <silent> gr <Plug>(coc-references)
    ```

    **安装建议**

    初学者精简套装：

    1. vim-polyglot（语法高亮）
    2. coc.nvim + coc-clangd（补全）
    3. fzf.vim（搜索）

    专业开发完整套装：

    全部上述插件 + 自定义配置

    **注意事项**

    1. **依赖工具需要先安装**：

       - clangd、clang-tidy、cmake、gdb 等
       - ripgrep、fd-find（搜索工具）

    2. **性能考虑**：

       - 大型项目建议使用 bear 生成 compile_commands.json
       - 适当调整 gutentags 的递归深度

    3. **调试配置**：

       - vimspector 需要额外下载调试器适配器

    这些插件组合可以提供类似 IDE 的开发体验，同时保持 Vim 的高效和灵活性。建议根据实际需求逐步添加插件。

    2. **Vundle**（经典）

    ```bash
    git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
    ```

    **方案B：YouCompleteMe**（传统方案）

    ```vim
    Plug 'ycm-core/YouCompleteMe'
    " 需要编译安装，支持 C++ 语义补全
    " 安装后运行：cd ~/.vim/bundle/YouCompleteMe && python3 install.py --clangd-completer
    ```

    **方案C：deoplete.nvim** + **deoplete-clang**（轻量级）

    ```vim
    Plug 'Shougo/deoplete.nvim'
    Plug 'deoplete-plugins/deoplete-clang'
    " 需要 vim 8.0+ 和 python3 支持
    let g:deoplete#enable_at_startup = 1
    ```

    2. **语法高亮**

        ```vim
        " C++ 现代语法高亮
        Plug 'bfrg/vim-cpp-modern'

        " 或使用综合方案
        Plug 'sheerun/vim-polyglot'
        ```

    3. **语法检查**

        ```vim
        " ALE - 异步语法检查
        Plug 'dense-analysis/ale'

        let g:ale_linters = {
        \   'cpp': ['clangtidy', 'cppcheck', 'gcc'],
        \   'c': ['clangtidy', 'cppcheck', 'gcc']
        \}

        let g:ale_cpp_clangtidy_options = '-std=c++17 -I./include'
        let g:ale_cpp_gcc_options = '-std=c++17 -Wall -I./include'
        ```

    4. **标签导航**

        ```vim
        " 传统 tags 系统增强
        Plug 'ludovicchabant/vim-gutentags'
        Plug 'skywind3000/gutentags_plus'

        " 配置
        let g:gutentags_project_root = ['.root', '.svn', '.git', '.hg', '.project']
        let g:gutentags_ctags_tagfile = '.tags'
        let g:gutentags_modules = []
        if executable('ctags')
            let g:gutentags_modules += ['ctags']
        endif
        if executable('gtags-cscope') && executable('gtags')
            let g:gutentags_modules += ['gtags_cscope']
        endif
        ```

    5. **文件浏览**

        ```vim
        " NERDTree - 经典文件树
        Plug 'scrooloose/nerdtree'

        " 快速文件搜索
        Plug 'kien/ctrlp.vim'
        " 或
        Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
        Plug 'junegunn/fzf.vim'
        ```

    6. **调试支持**

        ```vim
        " vimspector - 调试支持
        Plug 'puremourning/vimspector'

        " 或使用传统 vimgdb
        " Plug 'vim-scripts/vimgdb'
        ```

    7. **代码格式化**

        ```vim
        " 使用系统的 clang-format
        Plug 'rhysd/vim-clang-format'

        " 配置
        let g:clang_format#auto_format = 1
        autocmd FileType c,cpp nnoremap <buffer><Leader>cf :<C-u>ClangFormat<CR>
        ```

    8. **代码片段**

        ```vim
        " UltiSnips
        Plug 'SirVer/ultisnips'
        Plug 'honza/vim-snippets'

        " 触发设置
        let g:UltiSnipsExpandTrigger="<tab>"
        let g:UltiSnipsJumpForwardTrigger="<c-b>"
        let g:UltiSnipsJumpBackwardTrigger="<c-z>"
        ```

    **完整 .vimrc 配置示例**

    ```vim
    " ==================== 插件管理 ====================
    call plug#begin('~/.vim/plugged')

    " 外观
    Plug 'vim-airline/vim-airline'
    Plug 'vim-airline/vim-airline-themes'

    " 语法高亮
    Plug 'sheerun/vim-polyglot'

    " 代码补全
    Plug 'neoclide/coc.nvim', {'branch': 'release'}

    " 语法检查
    Plug 'dense-analysis/ale'

    " 标签系统
    Plug 'ludovicchabant/vim-gutentags'
    Plug 'skywind3000/gutentags_plus'

    " 文件浏览
    Plug 'scrooloose/nerdtree'
    Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
    Plug 'junegunn/fzf.vim'

    " 调试
    Plug 'puremourning/vimspector'

    " 代码格式化
    Plug 'rhysd/vim-clang-format'

    " 代码片段
    Plug 'SirVer/ultisnips'
    Plug 'honza/vim-snippets'

    " 注释工具
    Plug 'tpope/vim-commentary'

    " Git 集成
    Plug 'tpope/vim-fugitive'
    Plug 'airblade/vim-gitgutter'

    call plug#end()

    " ==================== C++ 特定配置 ====================
    " 编译运行快捷键
    autocmd FileType cpp nnoremap <F5> :w<CR>:!g++ -std=c++17 % -o %:r && ./%:r<CR>
    autocmd FileType cpp nnoremap <F9> :w<CR>:make<CR>

    " 设置 makeprg
    autocmd FileType c set makeprg=gcc\ -o\ %<\ %
    autocmd FileType cpp set makeprg=g++\ -std=c++17\ -o\ %<\ %

    " 错误导航
    autocmd FileType c,cpp nnoremap <Leader>n :cn<CR>
    autocmd FileType c,cpp nnoremap <Leader>p :cp<CR>

    " ==================== 插件配置 ====================
    " coc.nvim 配置
    set hidden
    set updatetime=300
    set shortmess+=c
    set signcolumn=yes

    " 快捷键
    nmap <silent> gd <Plug>(coc-definition)
    nmap <silent> gy <Plug>(coc-type-definition)
    nmap <silent> gi <Plug>(coc-implementation)
    nmap <silent> gr <Plug>(coc-references)

    " 使用 Tab 选择补全
    inoremap <silent><expr> <TAB>
          \ pumvisible() ? "\<C-n>" :
          \ <SID>check_back_space() ? "\<TAB>" :
          \ coc#refresh()
    inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"

    " ALE 配置
    let g:ale_fix_on_save = 1
    let g:ale_fixers = {
    \   'cpp': ['clang-format'],
    \   'c': ['clang-format']
    \}

    " NERDTree 配置
    map <C-n> :NERDTreeToggle<CR>

    " ==================== 基础设置 ====================
    syntax on
    set number
    set relativenumber
    set cursorline
    set tabstop=4
    set shiftwidth=4
    set expandtab
    set smartindent
    set mouse=a
    set hlsearch
    set incsearch
    set encoding=utf-8
    set fileencodings=utf-8,gbk
    ```

    🔧 **系统依赖安装**

    ```bash
    # Ubuntu/Debian
    sudo apt update
    sudo apt install -y \
        clang \
        clangd \
        clang-tidy \
        cppcheck \
        universal-ctags \
        global \
        fzf \
        ripgrep \
        fd-find \
        nodejs npm \
        bear  # 用于生成 compile_commands.json

    # 安装 coc.nvim 扩展
    vim +'CocInstall coc-clangd coc-snippets coc-json' +qall
    ```

    **精简配置方案（如果担心性能）**

    ```vim
    call plug#begin('~/.vim/plugged')

    " 基本功能
    Plug 'sheerun/vim-polyglot'          " 语法高亮
    Plug 'dense-analysis/ale'            " 语法检查
    Plug 'ludovicchabant/vim-gutentags'  " 标签
    Plug 'junegunn/fzf.vim'              " 文件搜索
    Plug 'tpope/vim-commentary'          " 注释

    call plug#end()

    " 使用纯 vim 内置补全
    set completeopt=menuone,noinsert,noselect
    set complete=.,w,b,u,t,i
    ```

    **Vim 8.2 注意事项**

    1. **Python 支持**：确保编译时启用了 Python3
       ```bash
       vim --version | grep python
       ```

    2. **终端支持**：如果使用终端 Vim，确保支持真彩色
       ```vim
       set termguicolors
       ```

    3. **性能优化**：
       - 大型项目避免使用太多同步插件
       - 使用 `:syntax off` 临时关闭语法高亮
       - 考虑使用 `ctags` 而不是 `gtags`

    这个配置在传统 Vim 8.2 上运行良好，提供了现代 C++ 开发所需的核心功能。

* 使用 apt 安装的 vim 插件

    虽然 vim-plug 不能通过 apt 安装，但有些插件管理器可以：

    1. Vundle (通过 apt)

        ```bash
        sudo apt install vim-vundle
        ```

        然后在 `~/.vimrc` 中配置：
        ```vim
        set nocompatible
        filetype off

        set rtp+=/usr/share/vim-vundle/
        call vundle#begin()

        Plugin 'gmarik/Vundle.vim'
        " 添加其他插件...

        call vundle#end()
        filetype plugin indent on
        ```

    2. Pathogen (通过 apt)

        ```bash
        sudo apt install vim-pathogen
        ```

        配置：
        ```vim
        execute pathogen#infect()
        syntax on
        filetype plugin indent on
        ```

* vim-vundle 与 vim-plug 有什么区别？

    Vundle 和 vim-plug 都是 Vim 插件管理器，但它们在设计理念、功能和性能上有显著区别：

    **主要区别对比**

    | 特性 | Vundle | vim-plug |
    |------|--------|----------|
    | **安装方式** | 可通过 apt 安装 | 需要手动下载 |
    | **异步支持** | ❌ 不支持 | ✅ 完全支持 |
    | **并行安装** | ❌ 顺序安装 | ✅ 并行安装插件 |
    | **按条件加载** | ❌ 有限支持 | ✅ 强大条件加载 |
    | **性能** | 较慢 | 很快 |
    | **维护状态** | 维护较少 | 活跃维护 |
    | **配置语法** | 相对复杂 | 简洁直观 |

    **技术细节对比**

    1. 配置语法差异

        **Vundle 配置：**
        ```vim
        set nocompatible
        filetype off

        set rtp+=~/.vim/bundle/Vundle.vim
        call vundle#begin()

        Plugin 'VundleVim/Vundle.vim'  " 管理器自身
        Plugin 'scrooloose/nerdtree'   " 插件
        Plugin 'tpope/vim-fugitive'

        call vundle#end()
        filetype plugin indent on
        ```

        **vim-plug 配置：**
        ```vim
        call plug#begin('~/.vim/plugged')

        Plug 'scrooloose/nerdtree'     " 更简洁的语法
        Plug 'tpope/vim-fugitive'
        Plug 'neoclide/coc.nvim', {'branch': 'release'}  " 支持分支和选项

        call plug#end()
        ```

    2. 条件加载能力

        **vim-plug 的高级功能：**

        ```vim
        " 按文件类型延迟加载
        Plug 'dense-analysis/ale', { 'for': 'python' }

        " 按命令延迟加载
        Plug 'scrooloose/nerdtree', { 'on': 'NERDTreeToggle' }

        " 按事件延迟加载
        Plug 'tpope/vim-fugitive', { 'on': [] }

        " 使用特定分支或标签
        Plug 'neoclide/coc.nvim', { 'branch': 'release' }

        " 使用固定版本
        Plug 'junegunn/fzf', { 'tag': '0.24.0' }
        ```

        **Vundle 的条件加载相对有限**

    3. 性能对比

        **安装/更新速度：**
        - Vundle：顺序安装，速度较慢
        - vim-plug：并行安装，速度很快

        **启动时间：**
        - vim-plug 的延迟加载能显著减少启动时间

    常用命令对比

    | 功能 | Vundle 命令 | vim-plug 命令 |
    |------|-------------|---------------|
    | 安装插件 | `:PluginInstall` | `:PlugInstall` |
    | 更新插件 | `:PluginUpdate` | `:PlugUpdate` |
    | 清理插件 | `:PluginClean` | `:PlugClean` |
    | 查看状态 | `:PluginList` | `:PlugStatus` |

    实际使用示例

    **vim-plug 的并行安装：**

    ```bash
    # 安装多个插件时，vim-plug 会并行下载
    :PlugInstall
    # 输出显示多个插件同时下载进度
    ```

    **Vundle 的顺序安装：**

    ```bash
    :PluginInstall
    # 插件一个一个顺序下载安装
    ```

    🎯 选择建议

    选择 Vundle 如果：

    - 你希望使用 apt 包管理器安装
    - 你需要一个简单稳定的解决方案
    - 你对性能要求不高
    - 你使用较老的 Vim 版本（不支持异步）

    选择 vim-plug 如果：

    - 你追求最佳性能和体验
    - 你需要异步安装和更新
    - 你想要强大的条件加载功能
    - 你使用较新的 Vim/Neovim（支持异步）

    🔄 迁移示例

    从 Vundle 迁移到 vim-plug：

    **原来的 Vundle 配置：**

    ```vim
    Plugin 'scrooloose/nerdtree'
    Plugin 'tpope/vim-fugitive'
    Plugin 'vim-airline/vim-airline'
    ```

    **迁移到 vim-plug：**

    ```vim
    Plug 'scrooloose/nerdtree'
    Plug 'tpope/vim-fugitive'
    Plug 'vim-airline/vim-airline'
    ```

    💡 总结

    **推荐使用 vim-plug**，因为：

    1. 更好的性能和用户体验
    2. 强大的条件加载功能
    3. 活跃的维护和社区支持
    4. 更简洁的配置语法
    5. 并行安装节省时间

    除非你有特定的兼容性需求，否则 vim-plug 是现代 Vim 用户的更好选择。

    一个参考的 .vimrc 配置：

    ```vim
    " 基本设置
    set number
    set relativenumber
    syntax on
    filetype plugin indent on
    set encoding=utf-8

    " vim-plug 插件管理
    call plug#begin('~/.vim/plugged')

    " 异步语法检查和补全
    Plug 'dense-analysis/ale'                         " 异步语法检查
    Plug 'neoclide/coc.nvim', {'branch': 'release'}   " LSP 补全引擎

    " Python 开发增强
    Plug 'jiangmiao/auto-pairs'                       " 自动括号补全
    Plug 'alvan/vim-closetag'                         " HTML/XML 标签补全
    Plug 'tpope/vim-commentary'                       " 快速注释
    Plug 'Yggdroot/indentLine'                        " 缩进指南线

    " 界面美化
    Plug 'vim-airline/vim-airline'                    " 状态栏
    Plug 'vim-airline/vim-airline-themes'             " 状态栏主题
    Plug 'scrooloose/nerdtree'                        " 文件浏览器
    Plug 'ryanoasis/vim-devicons'                     " 文件图标

    " 代码导航
    Plug 'preservim/tagbar'                           " 代码结构侧边栏
    Plug 'ctrlpvim/ctrlp.vim'                         " 文件模糊搜索

    call plug#end()

    " ==================== coc.nvim 配置 ====================
    " 使用 Tab 触发补全
    inoremap <silent><expr> <TAB>
          \ pumvisible() ? "\<C-n>" :
          \ <SID>check_back_space() ? "\<TAB>" :
          \ coc#refresh()

    inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"

    function! s:check_back_space() abort
      let col = col('.') - 1
      return !col || getline('.')[col - 1]  =~# '\s'
    endfunction

    " 回车确认补全
    inoremap <expr> <cr> pumvisible() ? "\<C-y>" : "\<C-g>u\<CR>"

    " 代码导航快捷键
    nmap <silent> gd <Plug>(coc-definition)
    nmap <silent> gy <Plug>(coc-type-definition)
    nmap <silent> gi <Plug>(coc-implementation)
    nmap <silent> gr <Plug>(coc-references)

    " ==================== ALE 配置 ====================
    let g:ale_linters = {
    \   'python': ['flake8', 'pylint'],
    \   'javascript': ['eslint'],
    \}

    let g:ale_fixers = {
    \   'python': ['black', 'isort'],
    \   'javascript': ['prettier'],
    \}

    let g:ale_sign_error = '❌'
    let g:ale_sign_warning = '⚠️'
    nmap <silent> <C-k> <Plug>(ale_previous_wrap)
    nmap <silent> <C-j> <Plug>(ale_next_wrap)

    " ==================== NERDTree 配置 ====================
    map <C-n> :NERDTreeToggle<CR>
    let NERDTreeShowHidden=1

    " ==================== 其他优化 ====================
    " 自动安装插件
    autocmd VimEnter * if len(filter(values(g:plugs), '!isdirectory(v:val.dir)'))
      \| PlugInstall --sync | source $MYVIMRC
    \| endif
    ```

    🎯 针对 Python 开发的特别优化

    在 `~/.vimrc` 中添加：
    ```vim
    " Python 特定设置
    autocmd FileType python setlocal expandtab tabstop=4 shiftwidth=4 softtabstop=4
    autocmd FileType python setlocal commentstring=#\ %s

    " 保存时自动格式化
    autocmd BufWritePre *.py :call CocAction('format')
    ```

* coc.nvim 的 pyright 加载失败原因排查

    1. 现象：py 文件中，pyright 无法弹出提示

    2. 检查 coc 插件是否正常加载：`:CocList extensions` 

        如果插件前面是`+`，表示已安装，但未激活。

        如果插件前面是`*`，表示已激活。

        可以使用方向键选择，然后按 tab 键，根据提示，按 t 键手动激活插件

    3. 如果手动激活失败，那么可以使用`:CocInfo`看 coc 的日志

        通常原因是 vim 版本太老，或者其它版本相关的问题。

    4. 更新 coc.nvim

        先把`.vimrc`中的 coc.nvim 版本改成：

        `Plug 'neoclide/coc.nvim', {'branch': 'release'}`

        然后更新：

        `PlugUpdate coc.nvim`

        清除用不到的插件的缓存：`:PlugClean`

    5. 如果提示 vim 版本过低（可能当前的 vim 版本是 8.2，而最新的 coc.nvim 需要 9.1 以上的 vim），那么可以去 vim 官网更新 vim 版本

        vim git repo: <https://github.com/vim/vim>

        clone: `git clone --depah 1 https://github.com/vim/vim`

        编译：`make`

        安装：`sudo make install`

        此时会把 vim 安装在`/usr/local/bin/`下，这个目录的优先级默认比`/usr/bin/`高

        还可以强制手动指定版本：`alias vim='/usr/local/bin/vim'`

    6. 如果`vim --version | head 3`仍显示 vim 8.2 版本，那么有可能是 bash 中的缓存没有刷新

        可以使用`hash -r`刷新缓存

        再次查看`vim --version | head 3`，可以看到 vim 9.2，版本已经正常。

        此时 coc.nvim, pyright 应该都可以正常使用了。

* vim 配置 python lsp 插件

    1. 安装 node.js

    2. 使用 vim plug 安装 coc: <https://github.com/neoclide/coc.nvim> 

    3. 使用 coc 安装 pyright: `:CocInstall coc-pyright`

    4. ai 建议的配置快捷键

        将以下经典配置复制到你的 `~/.vimrc` 中，它能让你像在 VS Code 中一样，使用快捷键进行代码跳转和查看文档：

        ```vim
        " --- coc.nvim 快捷键配置 ---

        " 使用 Tab 键触发补全
        inoremap <silent><expr> <TAB>
              \ coc#pum#visible() ? coc#pum#next(1) :
              \ CheckBackspace() ? "\<Tab>" :
              \ coc#refresh()
        inoremap <expr><S-TAB> coc#pum#visible() ? coc#pum#prev(1) : "\<C-h>"

        " 使用 Enter 回车键确认补全
        inoremap <silent><expr> <CR> coc#pum#visible() ? coc#pum#confirm()
                                      \: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"

        function! CheckBackspace() abort
          let col = col('.') - 1
          return !col || getline('.')[col - 1]  =~# '\s'
        endfunction

        " 跳转到定义 (Go To Definition)
        nmap <silent> gd <Plug>(coc-definition)

        " 跳转到类型定义
        nmap <silent> gy <Plug>(coc-type-definition)

        " 跳转到实现
        nmap <silent> gi <Plug>(coc-implementation)

        " 查看引用过该变量/函数的地方
        nmap <silent> gr <Plug>(coc-references)

        " 使用 K 查看文档提示 (Hover)
        nnoremap <silent> K :call ShowDocumentation()<CR>

        function! ShowDocumentation()
          if CocAction('hasProvider', 'hover')
            call CocActionAsync('doHover')
          else
            call feedkeys('K', 'in')
          endif
        endfunction

        " 重命名变量/函数 (Rename)
        nmap <leader>rn <Plug>(coc-rename)

        ```

    **进阶：如何让 LSP 识别虚拟环境（venv）？**

    Python 项目经常使用虚拟环境。`coc-pyright` 非常智能，它会自动寻找项目根目录下的 `.venv` 或 `venv` 文件夹。

    如果你的虚拟环境在别的地方（比如 `~/.virtualenvs`），你可以在项目根目录下创建一个 `.vim` 文件夹，并在其中新建 `coc-settings.json` 文件：

    ```json
    {
      "python.venvPath": "~/.virtualenvs",
      "python.formatting.provider": "black"
    }
    ```

    或者，你也可以在激活虚拟环境后（例如运行 `source venv/bin/activate`），**直接在同一个终端窗口里启动 Vim**，LSP 会自动使用当前激活的 Python 环境。

    你目前使用的是原生的 Vim 还是 Neovim？如果是 Neovim 的话，我们还可以用内置的 Lua LSP 来做一套更现代、不需要 Node.js 依赖的配置。

* coc.pyright 中，红色的`>>`表示 error，使用大写`K`键可以看到详细信息。

    光标移动到下划线的地方也可以看到 error 信息的悬浮窗。

    `:CocList diagnostics`可以打开 error 列表。里面会按行列出当前文件的所有诊断信息，包括错误级别、行号、列号和详细描述。你可以在这个列表中进行浏览和跳转.

    查看 coc.nvim 的帮助：

    `:help coc`

* 查看 coc 中 pyright 的输出：

    `:CocCommand workspace.showOutput Pyright`

    这里能看到当前使用的是哪个 python 版本。

    如果一个 shell 中既有 conda, 又有 pyenv （使用`/home/hlc/.vim/coc-settings.json`配置`"python.pythonPath": "/home/hlc/.pyenv/shims/python"`），会优先使用 conda 环境，只有在 `conda deactivate` 后，才会使用 pythonPath.

* pyright 还支持 pyrightconfig.json，

    <https://github.com/microsoft/pyright/blob/main/docs/configuration.md>

    但是没仔细研究。这个支持 pythonPath 吗？回头可以让 ai 研究一下这个配置。

* vim 中 lsp 插件

    * C/C++: clangd, ccls
    
    * Python: pyright, jedi-language-server
    
    * Rust: rust-analyzer

    * go: gopls

    * TypeScript/JS: tsserver

* vim 有关滚动的插件

    * vim-smoothie：平滑滚动

        <https://github.com/psliwka/vim-smoothie>

    * neoscroll.nvim（Neovim）

        <https://github.com/karb94/neoscroll.nvim>


* vim surround repo: <https://github.com/tpope/vim-surround>

* 简述 vim-surround 的用法

    vim-surround 插件是 Vim 中处理“环绕符号”的强大工具。以下是其核心用法：

    基本操作

    * `ds + 符号`：删除环绕符号

        例如：`ds"` 删除双引号。

        注：
        
        1. 这个不适用中文符号，比如`"你好"`，输入`ds"`，无法删除`"`。

    * `cs + 旧符号 + 新符号`：替换环绕符号

        例如：`cs"'` 将双引号替换为单引号。

    * `ys + 范围 + 符号`：添加环绕符号

        例如：`ysiw"` 为当前单词添加双引号。

    * S（Visual 模式） + 符号：为选中文本添加环绕符号

        例如：选中文本后按 `S(` 添加括号。

    常用符号

    * (、)、[、]、{、}

    * "、'、`

    * HTML/XML 标签：t（如 cst<p> 替换标签）

    * 自定义符号（如 #、_ 等）

    高级用法

    * 点操作符 .：重复上一次 surround 操作。

    * 嵌套修改：cs 和 ds 支持多层符号。

    * HTML 标签：

        * cst + 标签名：修改标签类型。

        * dst：删除标签，保留内容。

    * 可视块模式：选中多行后按 S 批量添加符号。

    示例

    ```vim
    " Hello world!"  
    光标在文本内，按 `cs"!` → "Hello world!" 替换为 !Hello world!  

    选中单词后按 `S(` → (Hello)
    ```

    安装

    * Vim 8+ / Neovim 插件管理器：

        `Plug 'tpope/vim-surround'`（vim-plug）

        `packadd! vim-surround`（内置包管理）

    掌握后可以极大提升编辑配对符号（引号、括号、标签等）的效率。


* vim-gutentags

    Vim-Gutentags 是一个 Vim 插件，它的核心功能是自动化管理 Vim 的标签文件（tags files）。

    在没有 Gutentags 之前，开发者通常需要手动运行 ctags -R . 来生成标签文件，并且在项目代码更新后，还需要重新运行该命令来更新标签，否则索引就会过时。这个过程非常繁琐且容易忘记。

    Gutentags 的解决方案:

    * 自动生成：当你用 Vim 在项目根目录（通过 .git, .hg, .svn 等版本控制目录识别）打开一个文件时，Gutentags 会自动在后台为你运行 ctags 命令来生成标签文件（通常是 ./tags 或 ./.git/tags）。

    * 自动更新：当你保存（write）一个文件后，Gutentags 会在后台静默地、异步地只更新刚才修改的那个文件的标签，而不是重新生成整个项目。这极大地提升了效率，避免了大型项目生成标签时造成的 Vim 卡顿。

    * 自动管理：你完全无需手动干预整个过程。它“Just Works”。

    主要特点:

    * 后台异步运行：使用 Vim 的 job 功能（或其它兼容插件）在后台运行 ctags，不会阻塞你的编辑操作。

    * 增量更新：只更新改变的文件，速度极快。

    * 智能项目管理：自动识别项目根目录，并为每个项目单独管理标签文件。

    * 高度可定制：你可以配置使用哪种 ctags 工具、标签文件存放位置、哪些文件需要被索引等。

    * 支持多种标签生成工具：默认支持 ctags 和 etags，通过配置也可以支持其它工具。

    安装：

    * 方法一，使用 Vim-Plug

        在`~/.vimrc`文件中添加：

        ```vim
        Plug 'ludovicchabant/vim-gutentags'
        ```

        重启 Vim 并执行：

        ```
        :PlugInstall
        ```

    * 方法二，使用 Vundle

        在`~/.vimrc`文件中添加：

        ```vim
        Plugin 'ludovicchabant/vim-gutentags'
        ```

        重启 Vim 并执行：

        ```
        :PluginInstall
        ```

#### markdown

* vim 安装 markdown preview 插件

    插件：`'iamcco/markdown-preview.nvim'`

    安装：

    1. 在`.vimrc`中添加

        ```vim
        call plug#begin()

        " List your plugins here
        " Plug 'tpope/vim-sensible'
        "
        " Plug 'plasticboy/vim-markdown'

        Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']}

        call plug#end()
        ```

        （需提前安装 VimPlug）

    2. 打开一个 vim，输入命令：`:PlugInstall`

        此时会 clone github repo，然后尝试编译和安装。

        这个 repo 是一个 js 项目，需要 npm 安装一些包，如果环境没有挂境外代理，可能会失败。即使 repo 编译安装失败，VimPlug 也会显示插件安装成功。实际上此时并没有安装成功。

        可以打开一个 md 文件，输入命令`:MarkdownPreview`试一下，如果什么都没有发生，说明没有安装成功。

    3. 如果 repo 已经 clone 了，但是没有安装成功，可以用下面的方法

        进入`~/.vim/plugged/markdown-preview.nvim/app`, 执行

        `sudo npm install`

        如果需要设置代表，先配置好`http_proxy`, `https_proxy`再执行 npm install。
        
        如果 app 目录中有`node_modules`，并且 npm 没有什么报错，那么大概率安装成功了。找一个 md 文件测试一下即可。

* vim 渲染 markdown 使用什么插件

    在Vim中渲染Markdown有多种选择，以下是最常用的插件：

    **主流选择**

    1. Markdown预览插件

        * markdown-preview.nvim - 最流行的实时预览插件

            ```vim
            " 安装
            Plug 'iamcco/markdown-preview.nvim'
            ```

            支持实时预览

            自动刷新

            支持数学公式、代码高亮

        * glow.nvim - 使用glow渲染

            ```vim
            Plug 'ellisonleao/glow.nvim'
            ```

    2. 语法高亮增强

        * vim-markdown - 增强Markdown语法高亮

            ```vim
            Plug 'plasticboy/vim-markdown'
            ```

            折叠支持

            更好的语法高亮

            TOC生成

        * vim-markdown-toc - 生成目录

            ```vim
            Plug 'mzlogin/vim-markdown-toc'
            ```

    **配置示例**

    ```vim
    " 基本配置
    call plug#begin()
    Plug 'iamcco/markdown-preview.nvim', { 'do': 'cd app && yarn install' }
    Plug 'plasticboy/vim-markdown'
    call plug#end()

    " Markdown预览设置
    let g:mkdp_auto_start = 0
    let g:mkdp_auto_close = 1
    let g:mkdp_refresh_slow = 0
    let g:mkdp_command_for_global = 0
    let g:mkdp_open_to_the_world = 0

    " 快捷键映射
    nmap <silent> <leader>mp <Plug>MarkdownPreview
    nmap <silent> <leader>ms <Plug>MarkdownPreviewStop
    nmap <silent> <leader>mt <Plug>MarkdownPreviewToggle
    ```

    **使用方式**

    * 实时预览：打开Markdown文件，按<leader>mp

    * 停止预览：按<leader>ms

    * 切换预览：按<leader>mt

    💡 其他选择

    * 终端内渲染

        * mdv - 终端内查看Markdown

        * glow - 美观的终端渲染

        * mdcat - cat的Markdown版本

    * Neovim专属

        * markdown-preview.nvim（也支持Neovim）

        * neovim内置LSP + markdown语言服务器

    📋 推荐组合

    对于大多数用户，我推荐：

    ```text
    markdown-preview.nvim + vim-markdown
    ```

    这个组合提供了完整的Markdown编写体验。


* markdown-preview.nvim

    markdown-preview.nvim 详细使用指南

    📦 安装配置

    1. 安装插件

        ```vim
        " 使用 vim-plug
        Plug 'iamcco/markdown-preview.nvim', { 'do': 'cd app && yarn install' }

        " 使用 lazy.nvim
        {
            'iamcco/markdown-preview.nvim',
            cmd = { 'MarkdownPreviewToggle', 'MarkdownPreview', 'MarkdownPreviewStop' },
            ft = { 'markdown' },
            build = 'cd app && yarn install',
        }
        ```

    2. 基础配置

        ```vim
        " 设置默认选项
        let g:mkdp_auto_start = 0           " 不自动开启预览
        let g:mkdp_auto_close = 1           " 切换buffer时自动关闭预览
        let g:mkdp_refresh_slow = 0         " 实时刷新（1=只在保存时刷新）
        let g:mkdp_command_for_global = 0   " 0=仅markdown文件，1=所有文件
        let g:mkdp_open_to_the_world = 0    " 0=本地，1=允许外部访问

        " 浏览器选项
        let g:mkdp_browser = ''             " 空=默认浏览器，或指定 'chrome', 'firefox'
        let g:mkdp_browserfunc = ''         " 自定义浏览器打开函数

        " 预览选项
        let g:mkdp_preview_options = {
            \ 'mkit': {},
            \ 'katex': {},
            \ 'uml': {},
            \ 'maid': {},
            \ 'disable_sync_scroll': 0,
            \ 'sync_scroll_type': 'middle',
            \ 'hide_yaml_meta': 1,
            \ 'sequence_diagrams': {},
            \ 'flowchart_diagrams': {},
            \ 'content_editable': v:false,
            \ 'disable_filename': 0
            \ }

        " 主题选项
        let g:mkdp_theme = 'dark'           " 'dark' 或 'light'

        " 文件路径
        let g:mkdp_filetypes = ['markdown'] " 启用预览的文件类型
        ```

    🚀 基本使用

    * 快捷键映射（推荐配置）

        ```vim
        " 正常模式快捷键
        nmap <silent> <C-m> <Plug>MarkdownPreview        " 开启/刷新预览
        nmap <silent> <C-s> <Plug>MarkdownPreviewStop    " 关闭预览
        nmap <silent> <C-p> <Plug>MarkdownPreviewToggle  " 切换预览

        " 或使用 leader 键
        nmap <leader>mp <Plug>MarkdownPreview
        nmap <leader>ms <Plug>MarkdownPreviewStop
        nmap <leader>mt <Plug>MarkdownPreviewToggle

        " 插入模式也可以映射
        imap <C-m> <esc><Plug>MarkdownPreview<cr>a
        ```

    * 命令行命令

        ```vim
        :MarkdownPreview          " 启动预览
        :MarkdownPreviewStop      " 停止预览
        :MarkdownPreviewToggle    " 切换预览状态
        ```

    ⚙️ 高级配置

    * 自定义浏览器

        ```vim
        " 使用 Chrome
        let g:mkdp_browser = 'chrome'

        " 或指定浏览器路径
        let g:mkdp_browser = '/usr/bin/google-chrome-stable'

        " 自定义打开函数（Linux示例）
        let g:mkdp_browserfunc = 'OpenMarkdownPreview'
        function! OpenMarkdownPreview(url)
        silent exec "!xdg-open " . a:url
        endfunction
        ```

    * 同步滚动配置

        ```vim
        " 同步滚动类型
        " 'relative' - 相对位置同步
        " 'middle'   - 光标在中间
        " 'absolute' - 绝对位置同步
        let g:mkdp_preview_options = {
            \ 'sync_scroll_type': 'middle',
            \ 'disable_sync_scroll': 0
            \ }
        ```

    * 端口和地址配置

        ```vim
        " 指定端口（避免冲突）
        let g:mkdp_port = '8080'

        " 指定地址
        let g:mkdp_host = '127.0.0.1'

        " 允许外部访问（远程服务器用）
        let g:mkdp_open_to_the_world = 1
        let g:mkdp_open_ip = '0.0.0.0'  " 监听所有地址
        ```

    * 主题和样式

        ```vim
        " 主题设置
        let g:mkdp_theme = 'dark'

        " 自定义CSS
        let g:mkdp_markdown_css = expand('~/.config/markdown.css')
        let g:mkdp_highlight_css = expand('~/.config/highlight.css')

        " 自动根据系统主题切换
        function! SetMarkdownTheme()
        if strftime('%H') < 18 && strftime('%H') > 6
            let g:mkdp_theme = 'light'
        else
            let g:mkdp_theme = 'dark'
        endif
        endfunction
        autocmd BufEnter *.md call SetMarkdownTheme()
        ```

    🔍 调试和问题排查

    * 查看预览URL

        ```vim
        :echo g:mkdp_browserfunc
        :echo g:mkdp_port
        :echo g:mkdp_host
        ```

    * 检查进程

        ```bash
        # 查看预览服务器进程
        ps aux | grep markdown
        netstat -tlnp | grep 8080
        ```

    常见问题解决

    * 预览不更新
    
        ```vim
        " 设置实时刷新
        let g:mkdp_refresh_slow = 0
        ```

    * 浏览器不打开

        ```vim
        " 明确指定浏览器
        let g:mkdp_browser = 'firefox'
        ```

    * 端口被占用
        
        ```vim
        " 更换端口
        let g:mkdp_port = '8888'
        ```

    * 数学公式不显示

        ```vim
        " 确保启用katex
        let g:mkdp_preview_options = {
            \ 'katex': {},
            \ }
        ```

    🎨 实用技巧

    * 自动模式

        ```vim
        " 自动开启预览（进入markdown文件时）
        autocmd FileType markdown let g:mkdp_auto_start = 1

        " 离开时自动关闭
        autocmd BufLeave *.md :MarkdownPreviewStop
        ```

    * 与文件树插件配合

        ```vim
        " 使用NERDTree时
        autocmd BufWinEnter *.md if &ft == 'markdown' | NERDTreeClose | endif
        autocmd BufWinLeave *.md if &ft == 'markdown' | NERDTreeToggle | endif
        ```

    * 性能优化

        ```vim
        " 大文件时禁用实时刷新
        autocmd BufReadPost *.md 
            \ if line('$') > 1000 |
            \   let g:mkdp_refresh_slow = 1 |
            \ endif
        ```

    * 多标签页支持

        ```vim
        " 每个markdown文件独立预览
        let g:mkdp_page_title = '「${name}」'
        ```

    📋 完整配置示例

    ```vim
    " ~/.vimrc 或 ~/.config/nvim/init.vim
    call plug#begin()
    Plug 'iamcco/markdown-preview.nvim', { 'do': 'cd app && yarn install' }
    call plug#end()

    " Markdown预览配置
    let g:mkdp_auto_start = 0
    let g:mkdp_auto_close = 1
    let g:mkdp_refresh_slow = 0
    let g:mkdp_command_for_global = 0
    let g:mkdp_browser = 'chrome'
    let g:mkdp_theme = 'dark'
    let g:mkdp_port = '8888'

    " 快捷键
    nmap <leader>mp <Plug>MarkdownPreview
    nmap <leader>ms <Plug>MarkdownPreviewStop
    nmap <leader>mt <Plug>MarkdownPreviewToggle

    " 自动命令
    autocmd FileType markdown nnoremap <buffer> <leader>p :MarkdownPreview<CR>
    autocmd FileType markdown nnoremap <buffer> <leader>s :MarkdownPreviewStop<CR>
    ```

    🎪 工作流程

    * 编辑文件：用Vim编辑 .md 文件

    * 开启预览：按 <leader>mp 或 :MarkdownPreview

    * 实时编辑：编辑内容，预览自动更新

    * 关闭预览：按 <leader>ms 或关闭Vim时自动关闭

    * 切换主题：修改 g:mkdp_theme 或使用CSS自定义

#### C++

* 我用 vim 8.2，开发 c++ 需要安装哪些插件

    使用**LSP (Language Server Protocol)**：

    1. LSP 引擎

        * `coc.nvim`: 目前最流行的插件，生态极其丰富。它通过 Node.js 运行，配置简单，补全非常智能。

            * 配套安装：在 Vim 中运行 `:CocInstall coc-clangd` 来支持 C++。

        * `vim-lsp`: 如果你追求更轻量、纯粹的 Vim 脚本实现，这是个好选择。

    2. 代码解析引擎 (后端)，必装

        * `clangd`: 它是 LLVM 提供的语言服务器，负责语法检查、自动导入头文件、重构和精准补全。

            `coc.nvim` 只是一个客户端，它需要 **clangd** 才能理解 C++。在终端执行：

            ```bash
            sudo apt update
            sudo apt install clangd
            ```

            安装完成后，在 Vim 中执行 `:CocInstall coc-clangd`。

    3. 项目管理与导航

        * **LeaderF** 或 **fzf.vim**: 极速的模糊查找工具。通过文件名、函数名或字符串快速定位代码。

        * **NERDTree** 或 **fern.vim**: 侧边栏文件浏览器，方便管理复杂的 C++ 项目目录。

        * **Tagbar**: 基于 ctags，在右侧展示当前文件的类、变量、函数列表。

    4. 辅助增强

        * **vim-cpp-enhanced-highlight**: 增强 C++11/14/17 的语法高亮，让代码更易读。

        * **vim-cmake-lists**: 如果你的项目使用 CMake，这个插件必不可少。

        * **ale**: 异步语法检查，在你输入时实时标注语法错误（如果不用 coc.nvim 的话必装）。

    **vim-plug**配置如下：

    ```vim
    call plug#begin()
        " 智能补全核心
        Plug 'neoclide/coc.nvim', {'branch': 'release'}
        " 模糊搜索
        Plug 'Yggdroot/LeaderF', { 'do': ':LeaderfInstallQtPy' }
        " 语法高亮增强
        Plug 'octol/vim-cpp-enhanced-highlight'
        " 侧边栏
        Plug 'preservim/nerdtree'
    call plug#end()
    ```

* coc.nvim requires at least Vim 9.0.0438 or Neovim 0.8.0, but you're using an older version.

    看来你在 Vim 8.2 上遇到了 **coc.nvim** 的版本兼容性问题。由于 coc.nvim 近期的更新大量使用了 Vim 9 引入的新脚本语法（Vim9script），在旧版本上直接运行会触发 `E1043` 等错误。

    针对你的情况，有以下三种解决方案，按推荐程度排序：

    方案一：切换到兼容旧版的 coc.nvim 分支（最快解决）

    coc.nvim 官方维护了一个 `v0.0.82` 分支，专门用于支持旧版本的 Vim。你只需要修改 `.vimrc` 中的插件声明：

    ```vim
    " 修改这一行
    Plug 'neoclide/coc.nvim', {'branch': 'v0.0.82'}
    ```

    修改后，在 Vim 中运行 `:PlugUpdate` 即可。

    方案二：使用非 LSP 架构的补全插件（最轻量）

    如果你无法升级 Vim 且不需要极其复杂的 IDE 功能，**YouCompleteMe (YCM)** 是 Vim 8.2 时代的经典选择。

    * **优点**：原生 C++ 支持极佳，且在 Vim 8.x 上非常稳定。
    * **安装**：`Plug 'ycm-core/YouCompleteMe'`，然后运行 `./install.py --clangd-completer`。

    方案三：升级 Vim 环境（长久之计）

    如果你的系统环境允许（如 Ubuntu 22.04+ 或使用 Homebrew/PPA），建议升级到 Vim 9.x。Vim 9 的执行效率比 8.2 有质的提升，尤其是运行 LSP 插件时会更加流畅。

    如果你决定继续使用旧版 Vim，务必配合 **clangd** 使用。你可以通过以下方式检查系统是否已有 `clangd`：

    ```bash
    clangd --version
    ```

    如果没有，建议通过系统的包管理器（如 `apt install clang-tools`）安装。

    **`coc.nvim`相关的报错**

    从你的 `scriptnames` 输出看，`coc.nvim` 的相关脚本（第 17-20 行以及 33-36 行）确实已经被加载了。报错 `E1043` 主要是因为 Vim 8.2 解析不了脚本中的 `:export` 命令，这是 Vim 9 专属的语法。

    既然你已经处于 `v0.0.82` 标签，但依然报错，最可能的原因是 coc.nvim 的 **自动化安装脚本** 并没有完全清理掉 master 分支残留的 `.vim` 或 `.js` 文件。

* 简介 coc-clangd 如何使用

    要让它正常工作，你主要需要关注**配置文件**、**编译数据库**和**常用快捷键**这三点：

    1. 配置文件 (`coc-settings.json`)

        在 Vim 中输入 `:CocConfig` 会打开一个 JSON 文件。这是配置 `clangd` 行为的地方。建议加入以下基础配置：

        ```json
        {
          "clangd.path": "clangd",
          "clangd.arguments": [
            "--background-index",     // 后台索引整个项目
            "--clang-tidy",           // 开启代码检查
            "--header-insertion=never" // 禁止自动导入头文件（可选）
          ]
        }
        ```

    2. 编译数据库 (`compile_commands.json`)

        这是 `clangd` 能够精准跳转的关键。`clangd` 需要知道你的代码是怎么编译的（包含哪些头文件路径）。

        * CMake 项目：在执行 cmake 时加上 `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`。

        * Makefile/其他项目：安装 `bear` 工具，通过 `bear -- make` 编译来生成该文件。

        * 单文件：如果只是写简单的 `.cpp`，`clangd` 会默认查找系统头文件，通常不需要此文件。

    3. 核心快捷键 (建议配置)

        将以下代码加入你的 `.vimrc`，这样你就能通过快捷键实现 IDE 功能：

        * `gd`: 跳转到定义 (Go Definition)
        * `K`: 显示悬浮文档 (Hover Documentation)
        * `<leader>rn`: 变量重命名 (Rename)
        * `[g` 和 `]g`: 在语法错误间跳转

    **验证使用**

    1. 打开一个 `.cpp` 文件。
    2. 输入 `:CocInfo`，确认 `clangd` 服务处于 `Service: clangd ... running` 状态。
    3. 试着把光标移动到一个函数名上，按下 `gd`，看是否能跳转。

    为了让你的开发环境更加完整，我们可以进一步完善具体的配置细节。

    通过以下操作可以进一步优化你的 C++ 编码体验：

    * 生成 C++ 开发专用的 coc-settings.json 完整模板
    * 配置 coc.nvim 的 C++ 常用跳转与重构快捷键
    * 编写针对不同构建系统的编译数据库生成指南

#### vim-plug

* 如何验证vim Plug是否安装成功？

    验证Vim Plug是否安装成功有以下几种方法：

    🔍 基本检查方法

    1. 查看插件目录结构

        ```bash
        # 检查Plug是否已克隆
        ls -la ~/.vim/autoload/
        # 应该能看到 plug.vim 文件
        ```

    2. 在Vim中测试命令

        ```vim
        :PlugStatus    " 查看插件状态
        :PlugInstall   " 尝试安装插件（如果没有错误，说明Plug正常工作）
        :PlugUpdate    " 更新插件
        :PlugClean     " 清理未使用的插件
        ```

    3. 检查Vim版本和功能

        ```vim
        " 在Vim中执行
        :version
        " 确认支持 +eval 和 +python3（部分插件需要）
        ```

    📝 创建测试配置验证

    步骤1：创建测试配置

    ```vim
    " ~/.vimrc 中添加测试代码
    call plug#begin('~/.vim/plugged')

    " 安装一个简单插件测试
    Plug 'junegunn/vim-easy-align'

    call plug#end()
    ```

    步骤2：重新加载并安装

    ```bash
    # 重新加载配置
    vim +'source ~/.vimrc'

    # 或在Vim中
    :source ~/.vimrc
    :PlugInstall
    ```

    步骤3：验证安装

    ```bash
    # 检查插件是否下载
    ls ~/.vim/plugged/
    # 应该能看到 vim-easy-align 目录
    ```

    ✅ 完整的验证流程

    方法一：使用内置命令

    ```vim
    " 1. 检查Plug命令是否存在
    :command Plug

    " 2. 检查插件列表
    :scriptnames
    " 在列表中应该能看到 plug.vim

    " 3. 测试插件功能
    " 安装测试插件后，验证其功能
    :EasyAlign     " 如果vim-easy-align安装成功，这个命令应该可用
    ```

    方法二：查看帮助文档

    ```vim
    :help plug.txt
    " 如果能看到Plug的帮助文档，说明安装成功
    ```

    方法三：创建测试脚本

    ```bash
    #!/bin/bash
    # test_vim_plug.sh

    echo "检查Vim Plug状态..."

    # 检查文件是否存在
    if [ -f ~/.vim/autoload/plug.vim ]; then
        echo "✅ Plug.vim 文件存在"
    else
        echo "❌ Plug.vim 文件不存在"
        exit 1
    fi

    # 在Vim中测试
    echo "在Vim中测试Plug命令..."
    vim -c 'PlugStatus' -c 'qa!' > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ PlugStatus 命令执行成功"
    else
        echo "❌ PlugStatus 命令执行失败"
    fi

    echo ""
    echo "验证完成！"
    ```

    🔧 常见问题排查

    * 如果Plug未安装：

        ```bash
        # 手动安装Vim Plug
        curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
            https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
        ```

    * 如果命令不可用：

        * 检查Vim版本：确保Vim版本 ≥ 7.0

        * 检查文件权限：
    
            ```bash
            chmod 755 ~/.vim/autoload/
            chmod 644 ~/.vim/autoload/plug.vim
            ```

        * 检查配置语法：

            ```vim
            " 确保.vimrc中正确配置
            call plug#begin('~/.vim/plugged')
            " 插件列表
            call plug#end()
            ```

    测试完成后，验证一个真实插件：

    ```vim
    " 安装并验证一个真实插件
    call plug#begin('~/.vim/plugged')
    Plug 'tpope/vim-surround'  " 一个常用插件
    call plug#end()

    " 在Vim中
    :PlugInstall
    " 测试插件功能
    :help surround  " 应该能看到帮助文档
    ```

    📊 验证结果示例

    成功的安装会显示：

    ```text
    ✅ ~/.vim/autoload/plug.vim 文件存在
    ✅ :PlugStatus 显示插件列表
    ✅ :PlugInstall 能正常下载插件
    ✅ 插件目录 (~/.vim/plugged/) 中有下载的插件
    ```

    最简单的快速验证命令：

    ```bash
    vim +PlugStatus +qa
    ```

    如果没有任何错误信息，说明Plug安装成功！

* vim-plug

    official site: <https://github.com/junegunn/vim-plug>

    下载和安装：

    ```bash
    curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
        https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
    ```

    使用：

    编辑`~/.vimrc`文件：

    ```vim
    call plug#begin()

    " List your plugins here
    Plug 'tpope/vim-sensible'

    call plug#end()
    ```

    进入`vim`，执行命令`:PlugInstall`，此时会开始安装插件`vim-sensible`。若安装成功，则会提示插件`vim-sensible`已经安装成功。此时说明 vim-plug 已经成功安装。

* windows 中，vim 如何安装 vim-plug ?

    在 Windows 中为 Vim 安装 vim-plug 有几种方法，以下是详细的步骤：

    方法一：使用 PowerShell 或 CMD（推荐）

    1. 确保 Vim 已正确安装

        首先确认你的 Vim 版本（建议 8.0 以上）：

        ```cmd
        vim --version
        ```

    2. 创建必要的目录

        打开 PowerShell 或 CMD，运行：

        ```powershell
        # 创建 vim 的 autoload 目录（如果不存在）
        mkdir -Force $HOME\vimfiles\autoload
        ```

    3. 下载 vim-plug

        ```powershell
        # 使用 PowerShell 下载
        $uri = 'https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
        (New-Object Net.WebClient).DownloadFile($uri, $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath("$HOME\vimfiles\autoload\plug.vim"))
        ```

        或者使用 curl（如果已安装）：

        ```powershell
        curl -fLo "$HOME/vimfiles/autoload/plug.vim" --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
        ```

    方法二：手动下载

    1. 创建目录结构

        ```text
        C:\Users\你的用户名\vimfiles\
                        └── autoload\
        ```

    2. 下载 plug.vim

        访问以下 URL，将内容保存为 plug.vim：

        ```text
        https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
        ```

        将下载的文件放入：

        ```text
        C:\Users\你的用户名\vimfiles\autoload\plug.vim
        ```

    方法三：使用 Git Bash（如果有安装）

    打开 Git Bash，运行：

    ```bash
    curl -fLo ~/vimfiles/autoload/plug.vim --create-dirs \
        https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
    ```

    配置 vim-plug

    1. 编辑 Vim 配置文件

        Windows 上的 Vim 配置文件通常是：

        ```text
        C:\Users\你的用户名\_vimrc
        ```

        或（较新版本）：

        ```text
        C:\Users\你的用户名\vimfiles\vimrc
        ```

    2. 添加插件配置

        在 _vimrc 文件中添加：

        ```vim
        call plug#begin('~/vimfiles/plugged')

        " 在这里添加插件，例如：
        Plug 'junegunn/vim-easy-align'
        Plug 'https://github.com/junegunn/vim-github-dashboard.git'

        call plug#end()
        ```

    3. 安装插件

        * 重新打开 Vim

        * 运行命令：

            ```vim
            :PlugInstall
            ```

    验证安装

    在 Vim 中运行以下命令检查是否安装成功：

    ```vim
    :echo exists('g:plugs')
    :PlugStatus
    ```

    常见问题解决

    1. 如果遇到网络问题

        * 可能需要设置代理（如果有）：

            ```vim
            let g:plug_shallow = 0
            let g:plug_threads = 1
            ```

    2. 如果目录权限问题

        * 以管理员身份运行 PowerShell 或 CMD

    3. 如果使用的是 Neovim

        Neovim 的配置路径不同：

        ```powershell
        # Neovim (Windows)
        mkdir -Force $HOME\AppData\Local\nvim\autoload
        $uri = 'https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
        (New-Object Net.WebClient).DownloadFile($uri, "$HOME\AppData\Local\nvim\autoload\plug.vim")
        ```

    常用命令

    * :PlugInstall - 安装插件

    * :PlugUpdate - 更新插件

    * :PlugClean - 删除未使用的插件

    * :PlugUpgrade - 升级 vim-plug 自身

    这样就完成了 vim-plug 在 Windows 上的安装。


#### git 相关

* vim 有哪些 git 相关的插件？

    Vim 有很多优秀的 Git 相关插件，以下是主要分类和推荐：

    核心 Git 集成插件

    1. vim-fugitive ⭐ 最流行

        * 作者：Tim Pope

        * 功能：全面的 Git 集成

        * 特色：

            * :G 命令查看状态

            * :Gdiff 查看差异

            * :Gblame 查看提交历史

            * :Gcommit 提交

            * :Gpush / :Gpull 推送/拉取

    2. gv.vim

        * 作者：Junegunn Choi

        * 功能：git commit 浏览器

        * 特色：可视化的提交历史，可查看提交详情和差异

    Git 状态显示插件

    3. vim-gitgutter

        * 功能：在侧边栏显示 git 差异标记

        * 特色：

            * 实时显示修改状态（+/-/~）

            * 支持阶段/取消阶段单个区块

            * 性能优化好

    4. signify / gitgutter.nvim（Neovim）

        * 轻量级替代方案，性能更优

    Git 差异查看插件

    5. vimdiff（内置）

        * 基本的 diff 功能：vimdiff file1 file2

    6. diffview.nvim（Neovim）

        * 现代的分屏差异查看器

    Git 集成增强

    7. rhubarb

        * 配合 vim-fugitive 使用

        * 支持在 GitHub 等平台打开文件链接

    8. committia.vim

        * 改进的 commit 信息编辑界面

        * 在 commit 时显示 diff 预览

    懒加载插件管理器推荐

    如果你使用插件管理器（如 vim-plug, packer.nvim, lazy.nvim），推荐配置：

    ```vim
    " 使用 vim-plug 示例
    Plug 'tpope/vim-fugitive'
    Plug 'tpope/vim-rhubarb'
    Plug 'airblade/vim-gitgutter'
    Plug 'junegunn/gv.vim'
    ```

    Neovim 专属插件

    9. neogit

        * 类似 Magit 的现代 Git 界面

        * 功能丰富，界面美观

    10. lazygit.nvim

        * lazygit 的终端 UI 集成

    使用建议

    * 新手组合：vim-fugitive + vim-gitgutter

    * Neovim 用户：neogit + gitgutter.nvim

    * 追求简洁：只安装 vim-fugitive

    这些插件可以大大提高你在 Vim 中处理 Git 操作的效率！

* vim-fugitive 用法

    常见命令速查

    ```vim
    :Gstatus       " 查看状态（常用）
    :Gcommit       " 提交
    :Gpush         " 推送
    :Gpull         " 拉取
    :Gblame        " 查看谁改了什么
    :Gdiff         " 查看差异
    :Gwrite        " 暂存文件（git add）
    :Gread         " 恢复文件（git checkout --）
    :Gmove         " 移动/重命名文件
    :Gdelete       " 删除文件
    ```

    ```vim
    " vim-fugitive 的快捷键设置（可选）
    nmap <leader>gs :Gstatus<CR>
    nmap <leader>gc :Gcommit<CR>
    nmap <leader>gp :Gpush<CR>
    nmap <leader>gl :Gpull<CR>
    nmap <leader>gd :Gdiff<CR>
    nmap <leader>gb :Gblame<CR>
    ```

    更新插件

    要更新 vim-fugitive 到最新版本：

    ```vim
    :PlugUpdate vim-fugitive
    ```

    或者更新所有插件：

    ```vim
    :PlugUpdate
    ```

    卸载插件

    如果不需要了：

    * 从 .vimrc 中删除 Plug 'tpope/vim-fugitive'

    * 执行 :PlugClean

    如果遇到问题，查看 `:messages` 获取错误信息

    注：

    1. 这个插件的地址为：<https://github.com/tpope/vim-fugitive>

        目前看来这个版本已经比较新了，前面提到的命令有的有点旧了，最新的版本已经不支持。

### ctags

* 详细讲讲 ctags 可以跳转哪些符号，不能跳转哪些符号。对于这些不能跳转的符号，又该如何跳转（如果不使用 lsp，只使用静态符号的话）

    要彻底搞懂 `ctags` 的边界，我们需要先理清它的底层逻辑：`ctags` 是一个**基于语法标签的静态索引工具**。它能否跳转某个符号，取决于这个符号是否具有“全局（或类/结构体级别）的唯一性或可识别性”。

    下面为你详细拆解 `ctags`（以目前最主流的 Universal Ctags 为准）能跳转与不能跳转的符号，以及在**不使用 LSP、只靠静态符号工具**的情况下，如何解决那些它无法跳转的痛点。

    一、 Ctags 可以跳转哪些符号？

    `ctags` 擅长捕捉那些**有固定声明语法、生命周期长、作用域较广**的“骨架型”符号。

    * **函数与方法（Functions & Methods）：** 全局函数、静态函数、类成员函数。
    * **类与结构体（Classes, Structures, Unions）：** `class`、`struct`、`union` 的定义。
    * **接口与协议（Interfaces & Protocols）：** 如 Java 的 interface，Go 的 interface。
    * **宏定义（Macros）：** C/C++ 中的 `#define` 宏。
    * **全局变量与静态变量（Global & Static Variables）：** 文件级或命名空间级的变量。
    * **枚举（Enums）：** 枚举类型名以及枚举内部的常量值（Enumerator）。
    * **类型定义（Typedefs / Type Aliases）：** 如 `typedef int int32_t;`。
    * **命名空间/包（Namespaces / Packages / Modules）：** 如 C++ 的 `namespace`，Java 的 `package`。

    > **💡 核心特征：** 这些符号在代码中通常有明显的关键字前缀（如 `void`、`class`、`enum`），且它们的生命周期通常伴随整个程序或整个文件。

    ---

    二、 Ctags 无法（或极难）跳转哪些符号？

    `ctags` 无法跳转的符号，通常具有“动态性”**、**“局部性”**或**“高度依赖上下文语义”的特点。

    1. **函数的局部变量（Local Variables）：**
    
        原因： 作用域仅限于函数内部，`ctags` 默认不为其生成标签（前文已解释原因）。

    2. **匿名对象/ lambda 表达式内部的临时符号：**
        
        原因： 没有固定的符号名称或完全动态生成。


    3. **通过父类指针/接口调用的多态方法（Polymorphic Calls）：**

        * *虚函数调用：* 当你在一行代码 `ptr->doSomething();` 上按跳转时，如果 `ptr` 是父类指针，`ctags` 只会傻傻地带你到**父类的虚函数声明**处，而**无法**根据运行时的实际类型带你跳转到子类的具体实现。


    4. **动态语言中的动态属性/反射符号：**

        * *原因：* 比如 Python 中通过 `setattr()` 动态绑定的属性，或者 JavaScript 中动态组装的对象键值。静态扫描工具根本无法在不运行代码的情况下预知这些符号。


    5. **宏展开后生成的符号：**

        * *原因：* 如果一个函数名是用宏拼接出来的（例如 `void MAKE_FUNC(init)()`），`ctags` 只能识别到宏，很难直接索引到最终生成的 `init_func`。

    三、 不用 LSP，如何解决这些“无法跳转”的符号？

    如果你由于各种原因（如服务器资源受限、大型老旧库、无法编译等）无法部署 LSP，只想依靠**轻量级的静态符号/文本工具**，可以通过以下这套“传统组合拳”来解决：

    1. 解决【局部变量】跳转：利用编辑器的内置文本搜索

        局部变量的生命周期非常短，通常就在当前文件或当前函数内。不需要全局索引，只需要**上下文文本匹配**。

        * **Vim 用户的绝招：**
        * `gd` (Go to Definition)：在 Vim 中，将光标移到局部变量上，按下 `gd`。Vim 会自动从当前函数的开头向下搜索该符号的第一次出现（通常就是定义处）。
        * `gD`：如果是全局变量但在本文件内，`gD` 会从当前文件的第一行开始搜索定义。
        * `*` 和 `#`：快速在当前文件中向前/向后寻找同名变量。

        * **VS Code / 现代编辑器：**
        * 即使关闭 LSP，编辑器内置的轻量级语法高亮引擎（如 Tree-sitter）也能识别当前文件内的变量引用，直接使用 `Ctrl + F` 或 `Cmd + F` 即可瞬间定位。

    2. 解决【虚函数实现/多态/全局引用】跳转：引入 GNU Global (Gtags)

        这是不用 LSP 时**最强大的替代方案**。`ctags` 只能告诉你“定义在哪（Definition）”，但不能告诉你“谁引用了我（Reference）”。**GNU Global (gtags)** 完美弥补了这一缺陷。

        * **它能做什么：** gtags 除了生成定义索引（GRTAGS），还会生成一个**引用索引（GTAGS）**。
        * **如何解决多态跳转：** 当你遇到父类指针 `ptr->toString()` 无法精准跳转时，你可以在 `toString` 上查找“所有引用该符号的地方”**或**“所有定义了该符号的地方”。Gtags 会列出一个清单，包含所有子类的 `toString` 实现，你只需在清单中肉眼筛选即可。
        * **配合工具：** 在 Vim 中配合 `ggtags.vim` 或 `gutentags` 插件，可以体验到接近 LSP 的引用查找功能。

    3. 解决【极端复杂符号/宏拼接/动态符号】：重型文本检索 (Ripgrep / Ag)

        当符号完全脱离了语法规则（比如宏拼接出来的符号，或者配置文件里配置的动态函数名），不管是 `ctags` 还是普通的 LSP 可能都会失效。这时候只能靠**速度纯粹到极致的文本检索**。

        * **推荐工具：** `ripgrep` (rg) — 目前世界上最快的文本搜索工具。
        * **使用方法：** 直接在工程根目录下暴力搜索字符串。
        ```bash
        rg "void .*my_macro_built_func" 
        # 或者直接搜变量名
        rg -w "target_variable_name"
        # -w 表示精确匹配单词

        ```

        * **集成体验：** 在 Vim 中配合 `fzf.vim` 或 `telescope.nvim`，可以实现一键全工程模糊搜索。虽然它不理解语义，但因为速度极快（几毫秒遍历数万文件），在死角情况下往往能救命。

    4. 强行让 Ctags 支持局部变量（不推荐，但可行）

        如果你有强迫症，非要用 `ctags` 跳转局部变量，Universal Ctags 其实留了后门（仅限部分语言如 C/C++）。

        你可以在生成 tags 时，强行开启 `l` (local variables) 标签：

        ```bash
        ctags --c-kinds=+l -R .

        ```

        *警告：这会导致 `tags` 文件体积暴增数倍甚至数十倍，且编辑器在遇到同名常见变量（如 `i`, `ret`）时会频繁弹出让人崩溃的冲突选择列表，请谨慎使用。*

    总结：传统的静态高效工作流

    在没有 LSP 的时代，资深开发者通常是这样搭配的：

    1. **跨文件找函数/结构体定义：** 用 `ctags` (快捷键 `Ctrl + ]`)。
    2. **函数内找局部变量：** 用 Vim 内置的 `gd`。
    3. **找虚函数实现/谁调用了这个函数：** 用 `GNU Global (gtags)` 查看引用列表。
    4. **遇到诡异的宏定义或动态符号：** 用 `ripgrep` 全局暴力模糊搜索。

* vim 如何 ctag 跳转？

    好的，在 Vim 中使用 Ctags 进行跳转是一个非常强大和高效的功能。下面我将为你提供一个从安装到使用的完整指南。

    **核心步骤概览**

    1.  **安装 Ctags** 工具。
    2.  在**项目根目录**生成标签文件（`tags`）。
    3.  在 Vim 中使用快捷键进行跳转。

    **详细步骤**

    1. 安装 Ctags

        首先，你需要在你的操作系统上安装 `ctags` 工具。

        * Ubuntu/Debian

            ```bash
            sudo apt-get install universal-ctags
            ```
            > 注意：推荐安装 `universal-ctags`（功能更强的新一代维护版本），而不是古老的 `exuberant-ctags`。

        * CentOS/RHEL/Fedora

            ```bash
            # 对于较新版本的 Fedora/CentOS，可能也叫 ctags
            sudo dnf install universal-ctags
            # 或者
            sudo yum install ctags
            ```

        * Windows (with WSL or MSYS2)

            建议在 WSL (Windows Subsystem for Linux) 环境中使用上述 Linux 安装方法，或者通过 MSYS2 的包管理器安装。

        安装完成后，可以在终端输入 `ctags --version` 确认安装成功。

    2. 生成 Tags 文件

        进入你的项目源代码的根目录，然后运行 `ctags` 命令来解析你的代码并生成一个名为 `tags` 的文件。

        ```bash
        # 进入你的项目目录
        cd /path/to/your/project

        # 递归地为当前目录及所有子目录中的文件生成 tags
        ctags -R .

        # 如果你只想为特定类型的文件生成 tags（例如只想要 C++ 和头文件），可以使用 --languages 选项
        ctags -R --languages=C,C++ .

        # 一个更常用的强大命令：排除不需要的目录（如 node_modules, build, .git）
        ctags -R --exclude=node_modules --exclude=build --exclude=.git .
        ```

        执行后，你会在当前目录下看到一个名为 `tags` 的文件。这就是 Vim 用来查找符号定义的数据库。

        **最佳实践：** 将 `tags` 文件添加到你的 `.gitignore` 中，因为它可以根据本地环境重新生成，不需要纳入版本控制。

    3. 在 Vim 中跳转

        现在，你可以在 Vim 中打开项目中的任何一个文件，并使用以下命令进行跳转：

        | 快捷键 / 命令 | 作用 |
        | :--- | :--- |
        | `Ctrl-]` | **跳转到光标下符号的定义处**（最常用的命令） |
        | `g Ctrl-]` | 如果有多个匹配的定义，此命令会列出所有候选，让你选择跳转到哪一个 |
        | `Ctrl-t` | **跳回**到跳转之前的位置（类似于“后退”按钮）。可以多次按它来回溯跳转历史。 |
        | `:ts <tag>` 或 `:tselect <tag>` | 列出所有匹配 `<tag>` 的标签定义，供你选择。 |
        | `:tjump <tag>` | 跳转到 `<tag>`。如果只有一个匹配则直接跳转，有多个则列出列表。 |

    **示例：**
    1.  在 Vim 中打开一个 C 文件。
    2.  将光标移动到一个函数名（如 `main`）或者一个变量名上。
    3.  按下 `Ctrl-]`，Vim 会立刻跳转到该函数或变量的定义处。
    4.  查看完定义后，按下 `Ctrl-t` 即可返回到原来的位置。

    ---

    高级配置和技巧

    为了让体验更好，你可以在 `~/.vimrc` 文件中添加一些配置。

    1. 自动在上级目录查找 tags 文件

        大型项目可能有多级目录，你不一定总是在项目根目录打开文件。这个配置让 Vim 自动向上递归查找父目录中的 `tags` 文件，非常有用。

        ```vim
        " 在 ~/.vimrc 中添加
        set tags=./tags;,tags;
        ```
        *   `./tags;`：从当前文件所在目录开始查找名为 `tags` 的文件，`;` 代表“如果没找到，继续向上递归到父目录查找”，直到找到为止。
        *   `tags;`：同时也在当前工作目录（`:pwd` 显示的目录）下查找 `tags` 文件。

    2. 自动生成 Tags

        对于大型项目，每次手动生成 tags 很麻烦。你可以：
        *   将 `ctags -R .` 命令写入项目的 `Makefile` 或构建脚本。
        *   使用插件（如 **vim-gutentags**）来**自动管理** tags 文件的生成和更新，在你保存文件时自动在后台更新 tags，体验非常好。

    3. 使用多个 Tags 文件

        如果你需要引用系统库（如 `/usr/include`）的标签，可以指定多个 tags 文件。

        ```vim
        set tags=./tags,./TAGS,tags,TAGS,/path/to/other/tags
        ```

    总结

    1.  **安装**: `sudo apt-get install universal-ctags`
    2.  **生成**: 在项目根目录运行 `ctags -R .`
    3.  **跳转**:
        *   `Ctrl-]` - 跳转到定义
        *   `Ctrl-t` - 跳回
    4.  **配置**: 在 `~/.vimrc` 中添加 `set tags=./tags;,tags;` 以便自动查找标签文件。

    掌握 Ctags 能极大提升你在 Vim 中阅读和导航代码的效率，是 Vim 使用者必备的技能之一。


    你可以在终端使用以下命令查看 **你的 ctags 版本** 所支持的全部语言列表：

    ```bash
    # 查看支持的语言列表
    ctags --list-languages

    # 查看对某种语言（如 Python）支持的具体符号类型（kinds）
    ctags --list-kinds=Python

    # 查看所有可用的符号类型（全局）
    ctags --list-kinds-full
    ```

    基于语义理解的强大工具

    这些工具对代码的理解深度远超基于正则匹配的 Ctags。

    | 工具 | 描述 | 优点 |
    | :--- | :--- | :--- |
    | **Scope** | 类 Ctags，但使用 Clang 的 AST（抽象语法树）来解析 C/C++/Obj-C，精度极高。 | **绝对准确**，不会因为宏或复杂的条件编译而出错。 |
    | **rg (ripgrep)** | 超快的代码搜索工具。 | 虽然不生成 tags，但用它来**全局搜索符号** `rg 'def my_function'` 速度极快，很多时候可以临时替代跳转功能。 |
    | **fd** | 简单的文件查找工具，是 `find` 命令的现代化替代。 | 快速找到文件，配合模糊查找插件可以快速跳转。 |

    ---

    4. 项目管理与模糊查找

    这些工具帮助你快速导航项目文件结构，是代码跳转的重要补充。

    | 工具 | 描述 | 优点 |
    | :--- | :--- | :--- |
    | **fzf (模糊查找)** | 命令行模糊查找器，有强大的 Vim 插件集成。 | 可以模糊查找文件、tags、符号、git commits 等。按几下键就能找到任何东西，速度极快。 |
    | **vim-gutentags** | **不是 tags 生成器，而是 tags 管理插件**。 | 它在你背后自动运行 Ctags/Gtags，**自动在后台更新 tags**，你无需再手动运行 `ctags -R .`。体验提升巨大！ |

* vim + ctags 如何给出函数的参数提示？

    Vim + ctags 本身对函数参数提示的支持有限，但可以通过以下几种方案来增强参数提示功能：

    1. 使用 Omni 补全

        Vim 内置的 Omni 补全可以对 Python 提供一定的参数提示：

        ```vim
        " 在 ~/.vimrc 中配置
        filetype plugin indent on

        " 启用 Omni 补全
        autocmd FileType python setlocal omnifunc=python3complete#Complete

        " 使用快捷键触发
        inoremap <C-space> <C-x><C-o>
        ```

        使用方法：在函数名后输入 `(`，然后按 `Ctrl+x Ctrl+o` 或 `Ctrl+空格`

    2. 使用 Jedi-vim 插件（推荐）

        这是最完善的 Python 参数提示解决方案：

        ```vim
        " 安装
        Plug 'davidhalter/jedi-vim'

        " 配置
        let g:jedi#completions_enabled = 1
        let g:jedi#show_call_signatures = 1  " 显示函数签名
        let g:jedi#popup_on_dot = 1          " 输入 . 时自动提示
        let g:jedi#use_tabs_not_buffers = 1

        " 禁用 Jedi 的自动补全，使用其他补全插件
        " let g:jedi#completions_enabled = 0
        ```

        安装后，在函数调用处会实时显示函数签名。

    3. 使用 YouCompleteMe + Jedi

        ```vim
        Plug 'ycm-core/YouCompleteMe'

        " 配置 YCM 使用 Jedi
        let g:ycm_python_interpreter_path = 'python'
        let g:ycm_python_binary_path = 'python'
        let g:ycm_auto_trigger = 1
        let g:ycm_show_diagnostics_ui = 0

        " 自动触发补全
        let g:ycm_min_num_of_chars_for_completion = 2
        let g:ycm_seed_identifiers_with_syntax = 1
        ```

    4. 使用 coc.nvim

        ```vim
        Plug 'neoclide/coc.nvim', {'branch': 'release'}

        " 安装 coc-python 扩展
        :CocInstall coc-python

        " 配置
        nmap <silent> gd <Plug>(coc-definition)
        nmap <silent> gy <Plug>(coc-type-definition)
        nmap <silent> gi <Plug>(coc-implementation)
        nmap <silent> gr <Plug>(coc-references)

        " 显示文档
        nnoremap <silent> K :call ShowDocumentation()<CR>

        function! ShowDocumentation()
          if CocAction('hasProvider', 'hover')
            call CocActionAsync('doHover')
          else
            call feedkeys('K', 'in')
          endif
        endfunction
        ```

    5. 纯 ctags 的有限方案

        如果坚持只用 ctags，可以这样优化：

        ```bash
        # 生成更详细的 tags
        ctags -R --fields=+l --languages=python --python-kinds=-iv --extras=+q .
        ```

        然后在 Vim 中：

        ```vim
        " 查看函数定义
        nnoremap <leader>d :tag <C-r><C-w><CR>

        " 在命令行预览
        nnoremap <leader>s :ptag <C-r><C-w><CR>
        ```

    6. 使用参数提示插件

        **echodoc.vim**

        ```vim
        Plug 'Shougo/echodoc.vim'
        set noshowmode  " 如果状态栏已显示模式，可以关闭重复显示
        ```

        **ALE**

        ```vim
        Plug 'dense-analysis/ale'
        let g:ale_completion_enabled = 1
        let g:ale_hover_to_floating_preview = 1
        ```

    7. 完整配置示例

        这是我的 Python 开发配置：

        ```vim
        " ~/.vimrc
        call plug#begin('~/.vim/plugged')
        Plug 'davidhalter/jedi-vim'
        Plug 'Shougo/echodoc.vim'
        call plug#end()

        " 基本设置
        filetype plugin indent on
        syntax enable

        " Jedi-vim 配置
        let g:jedi#show_call_signatures = 2  " 在命令行显示签名
        let g:jedi#popup_on_dot = 1
        let g:jedi#use_splits_not_buffers = "right"

        " echodoc 配置
        set cmdheight=2  " 为 echodoc 留出空间

        " 快捷键映射
        inoremap <silent> <C-space> <C-x><C-o>  " 手动触发补全
        nnoremap <leader>d :JediGoToDefinition<CR>
        nnoremap <leader>s :JediShowDocumentation<CR>

        " ctags 后备
        set tags=./tags,tags;
        command! MakeTags !ctags -R --fields=+l --languages=python --python-kinds=-iv --extras=+q .
        ```

    8. 使用技巧

        1. **自动显示参数**：安装 Jedi-vim 后，输入函数名加 `(` 会自动显示参数
        2. **查看文档**：光标在函数上按 `K` 查看文档
        3. **手动触发**：`Ctrl+x Ctrl+o` 手动触发补全
        4. **跳转定义**：`gd` 或 `Ctrl+]`

    **推荐方案**

    对于 Python 开发，**Jedi-vim** 是最简单有效的参数提示方案。如果追求更现代化的体验，可以考虑 **coc.nvim** 或 **YouCompleteMe**。

    纯 ctags 的方案功能有限，建议配合上述插件使用。

* `set tags=./tags,./TAGS,tags,TAGS,/path/to/other/tags`

    设置 Vim 查找 tags 文件的搜索路径列表，用逗号分隔多个路径。

    解释：

    * `./tags` - 当前文件所在目录的 tags 文件

    * `./TAGS` - 当前文件所在目录的 TAGS 文件（大写版本）

    * `tags` - 当前工作目录的 tags 文件

    * `TAGS` - 当前工作目录的 TAGS 文件（大写版本）

    * `/path/to/other/tags` - 指定的绝对路径下的 tags 文件

* ctags 扩展用法

    ```bash
    # 进入你的项目目录
    cd /path/to/your/project

    # 递归地为当前目录及所有子目录中的文件生成 tags
    ctags -R .

    # 如果你只想为特定类型的文件生成 tags（例如只想要 C++ 和头文件），可以使用 --languages 选项
    ctags -R --languages=C,C++ .

    # 一个更常用的强大命令：排除不需要的目录（如 node_modules, build, .git）
    ctags -R --exclude=node_modules --exclude=build --exclude=.git .
    ```

    * 自动在上级目录查找 tags 文件

        大型项目可能有多级目录，你不一定总是在项目根目录打开文件。这个配置让 Vim 自动向上递归查找父目录中的 tags 文件，非常有用。

        ```vim
        " 在 ~/.vimrc 中添加
        set tags=./tags;,tags;
        ```

        * `./tags;`：从当前文件所在目录开始查找名为 tags 的文件，; 代表“如果没找到，继续向上递归到父目录查找”，直到找到为止。

        * `tags;`：同时也在当前工作目录（:pwd 显示的目录）下查找 tags 文件。

    * tips

        * 将`tags`文件添加到你的`.gitignore`中，因为它可以根据本地环境重新生成，不需要纳入版本控制。

        * 将`ctags -R .`命令写入项目的 Makefile 或构建脚本。

* ctags 基本用法

    install: `sudo apt install universal-ctags`

    进入工程目录，执行`ctags -R .` (递归地为当前目录及所有子目录中的文件生成 tags)，执行完后会生成`tags`文件。

    进入 vim，导入 ctags：`:set tags=./tags`

    常用快捷键：

    * `Ctrl-]`: 跳转到光标下符号的定义处

    * `g Ctrl-]`: 如果有多个匹配的定义，此命令会列出所有候选，让你选择跳转到哪一个

    * `Ctrl-t`: 跳回到跳转之前的位置（类似于“后退”按钮）。可以多次按它来回溯跳转历史。

    * `:ts <tag>`或`:tselect <tag>`: 列出所有匹配`<tag>`的标签定义，供你选择。

    * `:tjump <tag>`: 跳转到`<tag>`。如果只有一个匹配则直接跳转，有多个则列出列表。

## note

go back: `ctrl` + `t` (pop tag, pops a tag off the tag stack)

* commonly used help commands

    * `:help x`: get help on the `x` command

    * `:help deleting`: find out how to delete text

    * `:help index`: get a complete index of what is available

    * `:help CTRL-A`: get help for a control character command, for example, `CTRL-A`

        here the `CTRL` doesn't mean press `ctrl` key, but to type `C`, `T`, `R` and `L` four keys.

    * `:help CTRL-H`: displays help for the normal-mode CTRL-H command

    * `:help i_CTRL-H`: get the help for the insert-mode version of this command

    * find meaning of vim build-in options, for example, `number`: `:help 'number'`. (quote the `number` option with single quote)

help prefixes:

| What | Prefix | Example |
| - | - | - |
| Normal-mode commands | (nothing) | `:help x` |
| Control character | `CTRL-` | `:help CTRL-u` |
| Visual-mode commands | `v` | `:help v_u` |
| Insert-mode commands | `i` | `:help i_<Esc>` |
| ex-mode commands | `:` | `:help :quit` |
| Command-line editing | `c` | `:help c_<Del>` |
| Vim command arguments | `-` | `:help -r` |
| Options | `'` (both ends) | `:help 'textwidth'` |

for special key, use angle brackets `<>`, for example: `:help <Up>`: find help on the up-arrow key.

start vim with a `-t` option: `vim -t`

check what does `-t` mean: `:help -t`

看到 P13 using a count to edit faster
