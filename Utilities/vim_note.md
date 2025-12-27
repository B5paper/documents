# vim note

## cache

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

* vim help

    :help /\[]
    :help whitespace
    :help [:alnum:]

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

* vim 模式

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

* vim 中`}`命令

    移动到下一个空行的第一个非空白字符（段落移动）

    注意事项：

    * 配合 { 命令（向上跳转到上一个空行）使用

    * 计数前缀可用：3} 向下跳转3个段落

* vim `+`命令

    作用：移动到下一行的第一个非空白字符

    详细说明：

        相当于 j + ^ 的组合

        直接定位到下一行有文本内容的位置

        数字前缀可用：3+ 向下移动3行并定位

        反义命令是 -（移动到上一行的第一个非空白字符）

* vim `.`命令

    作用：重复上一次修改操作

    详细说明：

    * 重复最近一次在普通模式下执行的修改命令

    * 可以重复插入、删除、替换等操作

    * 示例：

        * dw 删除一个单词 → . 再删除下一个单词

        * ihello<Esc> 插入文本 → . 再次插入"hello"

* vim 快捷键

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

    临时转换当前文件:

    ```vim
    :set expandtab
    :%retab!
    ```

    * `%retab!`会将文件中所有 Tab 转换为空格

    只转换特定行:

    ```vim
    :10,20retab  " 转换第10-20行
    ```

    文件格式配置（针对特定文件类型）:

    ```vim
    autocmd FileType python setlocal expandtab tabstop=4 shiftwidth=4
    autocmd FileType javascript setlocal expandtab tabstop=2 shiftwidth=2
    ```

    检查当前设置:

    ```vim
    :set expandtab? tabstop? shiftwidth? softtabstop?
    ```

    反向转换（空格转Tab）:

    ```vim
    :set noexpandtab
    :%retab!
    ```

    在打开文件时自动转换:

    ```vim
    autocmd BufRead * set expandtab | %retab!
    ```

    建议： 在团队项目中，建议使用统一的 .editorconfig 文件来保证代码风格一致。

* vim 打开文件后，跳转到上次关闭时候的位置：

    * 反引号 + 双引号：`` ` `` + `"`

    * 单引号 + 双引号：`'` + `"`

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

* vim 取消行号的方法

    `:set nonu`

    `:set nu!`

* vim 中的 regex 构建 group 时，括号需要加`\`(parentheses)：`\(key-words\)`，但是其它常用的 regex 都不需要。

    在 regex 前加`\v`表示 very magic，即所有可能被认为是 metacharacter 的字符 ，都会被判定为 metacharacter。

    这样上述的 regex 就可以写成`\v(key-worlds)`。此时如果我们需要匹配`(`和`)`，那么我们需要对它们进行转义：`\v\(key-words\)`。

* `grep -P`表示使用 PCRE 的 regex

* vim 中搜索 metacharacter `.` 的帮助文档

    `:help /\.`

* PCRE, for Perl Compatible Regular Expression

* vim 中有关 regex 的 help 命令

    ```
    :help pattern-searches
    :help atom
    ```

## topics

### ctags

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

vim config file: `~/.vimrc`

vim is a modal editor, and has 3 modes:

1. If the bottom of the screen displays the filename or is blank, you are is normal mode.

2. If you are in insert mode, the indicator displays `--INSERT--`.

3. if you are in visual mode, the indicator shows `--VISUAL--`.

enter inserting mode: type `i`

back to command mode: press `<Esc>` key.

move around: `h`, `j`, `k`, `l`

delete a character: type `x`

undo the last edit: `u`

redo: `ctrl` + `r`

undo line: `U`, press again to redo

save and exit: `ZZ` (upper cases)

discard changes and exit: `:q!`

insert a character before the character under the cursor: `i`

intert text after the cursor: `a`

delete a line: `dd`

add a new line: `o`

open a line above the cursor: `O` (uppercase)

help: `:help`

jump to tag: `ctrl` + `]`

go back: `ctrl` + `t` (pop tag, pops a tag off the tag stack)

commonly used help commands:

* `:help x`: get help on the `x` command

* `:help deleting`: find out how to delete text

* `:help index`: get a complete index of what is available

* `:help CTRL-A`: get help for a control character command, for example, `CTRL-A`

    here the `CTRL` doesn't mean press `ctrl` key, but to type `C`, `T`, `R` and `L` four keys.

* `:help CTRL-H`: displays help for the normal-mode CTRL-H command

* `:help i_CTRL-H`: get the help for the insert-mode version of this command

start vim with a `-t` option: `vim -t`

check what does `-t` mean: `:help -t`

find meaning of vim build-in options, for example, `number`: `:help 'number'`. (quote the `number` option with single quote)

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

看到 P13 using a count to edit faster
