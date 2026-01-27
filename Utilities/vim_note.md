# vim note

## cache

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

* vim 替换命令的 example: 为每个非空行行首添加星号

    `vnoremap <leader>l :s/^\s*\zs\S.*$/* &/<CR>`

    这行 Vim 命令的作用是 在可视模式下，给选中的每一行添加 Markdown 风格的列表标记。

    具体解析：

    * `vnoremap`：在可视模式下创建映射

    * `<leader>l`：当按下 <leader> 键（通常是 \）和 l 键时触发

    * :s/^\s*\zs\S.*$/* &/：替换命令

        * `^\s*`：匹配行首的空白字符

        * `\zs`：设置匹配开始位置（不替换前面的空白）

        * `\S.*$`：匹配第一个非空白字符到行尾的内容

        * `* &`：替换为 * 加上原匹配内容（& 代表匹配的内容）

    * `<CR>`：回车执行命令

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

* 关于 vim 中`:s/foo/bar<CR>`的解析机制

    Vim 不会把 :s/foo/bar<CR> 解析为 "把 foo 替换为 bar<CR>"，原因如下：

    Vim 的命令行解析机制:

    * <CR> 是特殊的控制字符

    * 在 Vim 的命令行模式下，<CR> 表示 回车执行

    * 它不是普通文本字符，而是 命令结束符

    * 当 Vim 遇到 <CR> 时，它会立即：

        * 结束当前命令的输入

        * 解析已输入的内容

        * 执行解析后的命令

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

    注意事项

    * \r 在搜索模式中表示回车（^M），在替换文本中表示换行

    * 如果要替换为真正的 ^M 字符（CR），需要用 \x0d 或表达式寄存器

    * 某些情况下，可能需要设置 :set ff=unix 确保换行符正确

* normal 命令

    ```vim
    :normal I* <CR>
    ```

    * :normal 是一个 Ex 命令（在 : 命令行中执行）

    * 它告诉 Vim："在普通模式下执行后面的按键序列"

    * I* 是传递给 :normal 的参数

    * 最后的 <CR> 是结束 :normal 命令，而不是在普通模式中按回车

    执行流程解析

    ```vim
    :g/^\s*\S/normal I* <CR>
    ```

    * : 进入命令行模式

    * g/^\s*\S/ 设置匹配模式

    * normal I* 对匹配的每行执行普通模式命令 I*␣

    * <CR> 执行整个 :g 命令

    这里的 <CR> 不是给 normal I* 的，而是结束 :g 命令并执行它。

* vim 中，vsplit 打开两个文件后，如何只复制一个文件的多行？

    在 Vim 中使用 vsplit 分割窗口后，要只复制其中一个文件的多行内容，有以下几种方法：

    1. 基本方法：先进入对应窗口

        ```vim
        <C-w>h      # 先切换到左侧窗口
        <C-w>l      # 先切换到右侧窗口
        ```

        然后在该窗口中使用正常的复制命令：

        ```vim
        yy          # 复制单行
        5yy         # 复制5行
        :10,20y     # 复制第10-20行
        ```

    2. 不切换窗口直接操作

        如果你知道当前光标在哪个窗口，可以直接：

        ```vim
        " 如果当前窗口是你要复制的窗口
        v           # 进入可视模式
        选择多行
        y           # 复制

        " 或者
        :'<,'>y     # 复制已选择的行
        ```

    3. 使用窗口局部寄存器（推荐）

        如果你需要频繁在两个窗口间复制，可以：

        * 在源窗口选择文本：

            ```vim
            v           # 进入可视模式
            选择多行
            "ay         # 复制到寄存器 a
            ```

        * 切换到目标窗口：

            ```vim
            <C-w>w      # 切换到另一个窗口
            ```

        * 粘贴：

            ```vim
            "ap         # 从寄存器 a 粘贴
            ```

    4. 使用全局寄存器

        如果你想复制的文本在所有窗口间共享：

        ```vim
        "+y         # 复制到系统剪贴板
        "+p         # 从系统剪贴板粘贴
        ```

    5. 快速切换窗口的快捷键

        * <C-w>h - 切换到左侧窗口

        * <C-w>l - 切换到右侧窗口

        * <C-w>w - 在窗口间循环切换

        * <C-w>j - 切换到下方窗口

        * <C-w>k - 切换到上方窗口

    实际示例

    假设你想复制左侧文件的第10-20行到右侧文件：

    ```vim
    <C-w>h      # 切换到左侧窗口
    :10,20y a   # 将10-20行复制到寄存器a
    <C-w>l      # 切换到右侧窗口
    "ap         # 粘贴寄存器a的内容
    ```

    这样就能准确地在分割窗口中只复制一个文件的内容了。

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

* vim 替换中的`\zs`和`&`

    \zs - 设置匹配开始位置

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

* Vim 替换命令中其他有用的标记

    **替换部分特殊标记**

    1. \1, \2, ... \9 - 引用捕获组

        ```vim
        :%s/\(\w\+\) \(\w\+\)/\2 \1/g
        " 交换两个单词的位置
        " hello world → world hello

        :%s/\(\d\+\)-\(\d\+\)/\2.\1/g
        " 交换用连字符分隔的数字
        " 123-456 → 456.123
        ```

    2. \0 - 引用整个匹配（与 & 相同）

        ```vim
        :%s/\w\+/item: \0/g
        " 每个单词前加 "item: "
        ```

    3. ~ - 使用上次替换的字符串

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

    6. \@<= - 正向回顾（positive lookbehind）

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

    8. \@= - 正向预览（positive lookahead）

        ```vim
        :%s/foo\(bar\)\@=/FOO/g
        " 替换后面是 bar 的 foo
        " foobar → FOObar
        " foobaz 不变
        ```

    9. \@! - 负向预览（negative lookahead）

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

    替换中使用表达式 \=

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

    掌握这些标记可以让你在 Vim 中进行非常强大和灵活的文本处理！

* vim 中，s 前的百分号是什么意思

    在 Vim 中，:%s 前面的 % 表示在整个文件中执行替换，即对所有行进行操作。

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

* vim 替换命令的基本结构

    `:s/pattern/replace/flags`

* vim 替换中的 normal 模式

    * 配合 `:global` 命令: `:g/pattern/normal I*`

        含义：在所有匹配到 pattern 的行上，执行“在行首插入星号”的操作。

        效果：这常用于批量给某些行加注释或列表标记。

    * 在替换式中使用表达式: `s/pattern/\=.../`，这属于高级用法。例如： :%s/pattern/\=execute('normal I*')/n

        含义：找到匹配项后，不直接通过替换文本来修改，而是调用 execute 函数去触发一段普通模式指令。

* vim 禁用终端响铃（Bell）

    ```vim
    set noerrorbells   " 禁用错误提示音
    set visualbell     " 使用视觉提示代替声音
    set t_vb=          " 清空视觉提示（彻底静音）
    ```

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

* 鼠标在 vim 界面上滑动时，会向 vim 发送鼠标事件序列

    禁用鼠标序列：`:set mouse=`

    `:set mouse=nv    " 仅允许在普通模式和可视模式下使用鼠标`

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

* 什么是命令行模式，它和 normal 模式一样吗？

    不，命令行模式和 Normal 模式完全不一样。这是 Vim 中两个不同的模式，让我详细解释它们的区别：

    命令行模式（Command-line Mode）

    进入方式：

    * 在 Normal 模式下按 :（冒号）

    * 按 / 或 ? 进行搜索时

    * 按 : 后出现的状态行显示 : 提示符

    主要功能：

    * 执行 Ex 命令：:w（保存）、:q（退出）、:s/old/new（替换）

    * 执行 Vim 脚本：:echo "Hello"

    * 搜索：/pattern 或 ?pattern

    * 调用外部命令：:!ls（执行 shell 命令）

    * 设置选项：:set number、:set hlsearch

    特点：

    * 光标移动到屏幕底部

    * 输入命令后按 Enter 执行

    * 按 Ctrl-C 或 Esc 可取消并返回 Normal 模式

    * 可以输入多行命令（使用 \ 续行）

    Normal 模式（普通模式）

    进入方式：

    * Vim 启动时的默认模式

    * 从其他模式按 Esc 返回

    主要功能：

    * 导航文本：h j k l、w b、gg G

    * 编辑操作：d（删除）、y（复制）、p（粘贴）

    * 文本对象操作：diw（删除单词）、ci"（修改引号内内容）

    * 窗口操作：Ctrl-w + 方向键

    * 执行普通模式命令：.（重复上次操作）

    特点：

    * 主要工作模式

    * 按键直接执行命令，不显示在屏幕底部

    * 所有操作都是即时生效的

    对比表格

    | 特性 | 命令行模式 | Normal 模式 |
    | - | - | - |
    | 提示符 | : / ? | 无 |
    | 光标位置 | 屏幕底部 | 文本任意位置 |
    | 输入方式 | 输入完整命令后按 Enter | 按键立即执行 |
    | 主要用途 | 执行复杂命令、设置选项 | 导航、编辑文本 |
    | 退出方式 | Enter 执行，Esc 取消 | 按 i 等进入其他模式 |
    | 可见性 | 命令显示在底部 | 操作不显示 |

    关系示意图

    ```text
    Normal 模式
        ↓ 按 `:`
    命令行模式 (输入命令)
        ↓ 按 Enter 执行
    Normal 模式 (执行结果生效)
    ```

    示例对比

    Normal 模式操作：

    ```text
    光标在文本中 → 按 `dd` → 立即删除当前行
    ```

    命令行模式操作：

    ```text
    在 Normal 模式 → 按 `:` → 输入 `:%s/foo/bar/g` → 按 Enter
    → 全文替换所有 "foo" 为 "bar"
    ```

    关键区别：命令行模式需要输入完整的命令字符串然后执行，而 Normal 模式是每个按键立即对应一个操作。

    关于 noremap! 的特殊性

    当使用 noremap! 时，映射同时生效于：

    * 插入模式（输入文本时）

    * 命令行模式（输入命令时）

    但 不生效于 Normal 模式。如果你想要 Normal 模式的映射，需要使用 nnoremap。

    这就是为什么了解这些模式区别很重要——不同模式的映射是完全独立的！

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

* vim 在用 :vsplit 打开新窗口时，如何在右边打开新窗口，而不是在左边打开

    在 Vim 中使用 :vsplit 时，默认会在当前窗口的左侧打开新窗口。如果你希望在右侧打开新窗口，有以下几种方法：

    方法一：使用 :rightbelow vertical split（最常用）

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

    方法二：使用方向参数

    ```vim
    :vertical split
    ```

    然后按 Ctrl+w L 将新窗口移动到右侧

    方法三：配置为默认行为（添加到 .vimrc）

    如果你希望 :vsplit 默认在右侧打开，可以在 ~/.vimrc 中添加：

    ```vim
    set splitright
    ```

    这样设置后，所有的垂直分割都会在右侧打开。

    方法四：使用快捷键映射

    在 .vimrc 中创建自定义快捷键：

    ```vim
    " 使用空格+v在右侧垂直分割
    nnoremap <space>v :rightbelow vnew<CR>
    " 或保持原有命令但强制右侧
    nnoremap <leader>v :rightbelow vsplit<CR>
    ```

    快速参考：

    * :vsplit 或 :vs - 在左侧垂直分割

    * :rightbelow vsplit 或 :rightbelow vs - 在右侧垂直分割

    * :vertical split - 垂直分割，然后可以用 Ctrl+w L 移动到右侧

    最实用的是 方法一（使用 :rightbelow vsplit）或 方法三（设置 splitright 全局选项）。

* vim 在输入 :vsplit subdir/ 后，如何显示出 subdir 中的子文件？

    在 Vim 中，当你使用 :vsplit subdir/ 命令分割窗口打开目录时，它会显示该目录的文件列表（类似文件管理器）。要查看 subdir 中的子文件，有几种方法：

    1. 直接进入子目录（推荐）

        * 在目录列表中，将光标移动到 subdir 上

        * 按 enter 键即可进入该子目录

        * 或者直接使用命令：:vsplit subdir（不加末尾的 /）

    2. 在命令中使用路径通配符

        ```vim
        :vsplit subdir/*
        ```

        这会在新窗口中打开 subdir 下的所有文件（如果有多个文件会显示列表）。

    3. 使用 netrw（vim 内置文件浏览器）

        ```vim
        :explore subdir/   " 在当前位置打开文件浏览器
        :vexplore subdir/  " 垂直分割打开文件浏览器
        :sexplore subdir/  " 水平分割打开文件浏览器
        ```

    4. 使用 :find 命令 + 补全

        ```vim
        :find subdir/      " 输入后按 Tab，会显示子目录中的文件
        :vsplit `tab补全的结果`
        ```

    5. 进入子目录后导航

        一旦进入 subdir 目录视图：

        * - 返回上级目录

        * Enter 打开文件或进入子目录

        * o 在新窗口中打开

        * D 删除文件/目录

        * R 重命名

        * s 切换排序方式

        * q 或 :q 关闭目录窗口

        * i 切换显示格式（详细信息/简略信息）

    注意：如果 subdir 显示为普通文件而非目录，可能是因为它是符号链接或者你缺少该目录的读取权限。

    注：

    1. 可以使用 tab 打开子目录，但是需要随便输入一点东西
    
        比如`:vsplit subdir/`，此时可以输入`:vsplit subdir/*`然后按 tab，即可显示子文件夹中内容。也可以按一下右箭头，虽然屏幕上没显示内容，但是此时按 tab 也可以显示子文件夹中内容。

* vim 中，我将 m 键映射为 2j，现在我输入 3m，想向下移动 6 行，但是实际移动的是 32j，即 32 行。该如何解决这个问题

    在 Vim 中，当你使用数字前缀（如 3）配合映射时，Vim 会将数字和映射的键组合起来执行。你的问题是 3m 被解释为 32j（因为 m 映射为 2j），所以实际执行的是 32j 而不是 3×2j。

    有几种解决方案：

    解决方案 1：使用表达式映射（推荐）

    ```vim
    " 在 vimrc 中添加
    nnoremap <expr> m (v:count ? v:count * 2 : 2) . 'j'
    ```

    这个映射会：

    * 当输入 m 时 → 执行 2j

    * 当输入 3m 时 → 执行 6j（3×2）

    * 当输入 5m 时 → 执行 10j（5×2）

    解决方案 2：使用 <Cmd> 映射（Vim 8.2+/Neovim）

    ```vim
    nnoremap m <Cmd>execute 'normal! ' . (v:count1 * 2) . 'j'<CR>
    ```

    解决方案 3：使用函数包装

    ```vim
    function! MoveLines() abort
        let count = v:count1 * 2
        execute 'normal! ' . count . 'j'
    endfunction

    nnoremap <silent> m :call MoveLines()<CR>
    ```

    解决方案 4：使用递归映射（较简单）

    ```vim
    " 这个方法有点取巧，但可以工作
    nnoremap m 2j
    nnoremap 1m 2j
    nnoremap 2m 4j
    nnoremap 3m 6j
    nnoremap 4m 8j
    " ... 可以继续添加更多数字
    ```

    最完整的解决方案

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

* Vim 中没有直接的内置选项可以让光标在滚动时自动跟随屏幕移动

    一些相关的选项：

    * 'scrolloff' - 设置在光标上下保留的最小行数

        ```vim
        set scrolloff=5  " 光标上下总是保留5行可见
        ```

    * 'sidescrolloff' - 水平方向的类似选项

    * 'startofline' - 某些命令是否会移动光标到行首

* 使用 ssh 登陆 remote host 后，启动 vim 打开一个文件，此时如果 remote host 没有安装`vim-gtk3`，那么使用`"+y`是无法将内容复制到 local host 的剪贴板上的。

    安装了就可以了。

    local host 应该也得安装`vim-gtk3`。

    ssh 连接时是否需要开 -X 或 -Y，不清楚。

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

* 匹配非空白字符

    在 Vimscript 中，要匹配非空白字符（即行中至少有一个非空白字符），有几种正确的方法：

    方法1：使用单引号字符串（推荐）

    ```vim
    if line =~ '\S'      " \S 匹配非空白字符
        echo 'selected'
    endif
    ```

    方法2：使用双引号字符串并转义

    ```vim
    if line =~ "\\S"     " 需要双反斜杠
        echo 'selected'
    endif
    ```

    方法3：使用更明确的模式

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

    方法4：使用 Vim 内置函数

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

    * \S：匹配非空白字符（非空格、制表符等）

    * \s：匹配空白字符

    * ^\s*$：匹配空行或只有空白字符的行（使用 !~ 取反即可匹配非空行）

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

* 函数 return 0 和 return 1 有区别吗？

    在 Vimscript 中，函数 return 0 和 return 1 有区别，但这个区别主要体现在 条件判断 和 调用者如何处理返回值 上。让我详细解释：

    1. 基本区别

        ```vim
        function! TestZero()
            return 0
        endfunction

        function! TestOne()
            return 1
        endfunction
        ```

    2. 在条件判断中的不同行为

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

    3. 常见的返回模式

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

    4. 实际应用场景

        场景1：作为条件使用

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

        场景2：链式调用判断

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

    5. 与其他值的对比

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

    6. 最佳实践建议

        建议1：明确约定
        
        ```vim
        " 文档注释说明返回值的意义
        "
        " @return 1 表示成功，0 表示失败
        function! ProcessFile()
            " ... 处理逻辑
            return success ? 1 : 0
        endfunction
        ```

        建议2：使用命名常量提高可读性

        ```vim
        let s:SUCCESS = 1
        let s:FAILURE = 0
        let s:TRUE = 1
        let s:FALSE = 0

        function! SaveSettings()
            try
                " ... 保存逻辑
                return s:SUCCESS
            catch
                return s:FAILURE
            endtry
        endfunction
        ```

        建议3：考虑使用字符串或更丰富的返回值

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

    7. 特殊用途：命令的成功/失败

        ```vim
        " 有些 Vim 命令会设置 v:shell_error
        function! RunCommand(cmd)
            let output = system(a:cmd)
            if v:shell_error
                echo "命令失败，错误码：" . v:shell_error
                return 0
            else
                echo "命令成功"
                return 1
            endif
        endfunction
        ```

    总结

    | 返回值 | 布尔值 | 典型含义 | 使用场景 |
    | - | - | - | - |
    | return 0 | 假 (false) | 失败、否、错误 | 条件判断中表示假，操作失败 |
    | return 1 | 真 (true) | 成功、是、正确 | 条件判断中表示真，操作成功 |

    关键点：

    * 在 if 语句中，0 表示假，1 表示真

    * 这符合大多数编程语言的约定（C/C++风格）

    * 在 Vimscript 中，只有数值 0 是假，其他所有值都是真

    * 设计函数时，考虑调用者如何使用返回值来决定返回 0 还是 1

* vim ctrl + f

    无论光标在何处，都跳转到当前页的倒数第 2 行，并把这一行置顶

* `gU`

    gU 是一个操作符，需要配合移动命令来使用。

    examples:

    * `gUU` 或 `gUgU`: 将当前行转为大写。

    * `gUw`: 将从光标处到下一个单词结尾的文本转为大写。

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





* vim 有关滚动的插件

    * vim-smoothie：平滑滚动

        <https://github.com/psliwka/vim-smoothie>

    * neoscroll.nvim（Neovim）

        <https://github.com/karb94/neoscroll.nvim>

* vim 技巧

    4. 搜索和删除

        * d/pattern - 删除直到匹配 pattern 的位置

        * d?pattern - 向后删除直到匹配 pattern 的位置

    5. 结合数字使用

        * d2t. - 删除直到第二个 '.' 前

        * 3df - 删除直到第三个空格（包括空格）

    实际例子：

    假设文本：hello.world foo.bar test.example end

    * 光标在 foo 的 'f'：dt → 删除 foo.bar

    * 光标在 foo 的 'f'：df → 删除 foo.bar

    * 光标在引号内：di" → 删除 "hello world" 中的 hello world

    * 光标在单词任意位置：daw → 删除整个单词

    这些命令中的 d 都可以换成 c（删除并进入插入模式）或 y（复制），非常灵活！

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


* vim split

    Vim Split（分屏）基本用法：

    1. 创建分屏

        ```vim
        :vsplit      # 垂直分屏（左右分割）
        :split       # 水平分屏（上下分割）
        :new         # 新窗口打开空白缓冲区
        :vsp [文件]  # 垂直分屏并打开文件
        :sp [文件]   # 水平分屏并打开文件
        ```

    2. 快捷键

        ```vim
        Ctrl+w s    # 水平分割当前窗口
        Ctrl+w v    # 垂直分割当前窗口
        Ctrl+w n    # 新窗口打开新文件
        Ctrl+w c    # 关闭当前窗口
        Ctrl+w q    # 关闭当前窗口（同 :q）
        Ctrl+w o    # 只保留当前窗口
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
        # 在分屏间复制粘贴
        # 普通模式下可以互相复制粘贴

        # 多文件操作
        :vsp file1.txt file2.txt

        # 快速比较差异
        :diffsplit filename

        # 关闭其他所有窗口
        :only
        ```

    示例工作流：

    * vim main.py 打开主文件

    * :vsp utils.py 垂直分屏打开工具文件

    * Ctrl+w l 切换到右边窗口

    * :sp config.py 水平分屏打开配置文件

    * Ctrl+w k 回到主编辑区

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
        " 例如按 \sb 切换（默认leader是\）
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

        vim-syncscroll（轻量级）

        ```vim
        " 安装后自动启用，支持非对称分屏
        " 自动检测分屏并同步滚动
        ```

        ScrollColors（比较时专用）

        ```vim
        " 用于比较两个文件版本时保持同步
        ```

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

        * 使用 :windo diffoff 退出比较模式时会取消同步

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

    这样设置后，你可以很方便地让分窗口跟随滚动，特别适合：

    * 代码和文档对照

    * 不同版本文件比较

    * 多语言文件对照查看

* `vim -u`：启动时额外加载配置文件

    example:

    `vim -u ~/test_vimrc.vim`

    完全不加载默认配置，只加载指定文件:

    `vim -u ~/test_vimrc.vim -N`

    最小化启动（不加载任何配置）:

    `vim -u NONE -N`

* vim 启动后加载 Vimscript

    ```vim
    " 加载指定的 vimscript 文件
    :source ~/test_script.vim

    " 或者简写
    :so ~/test_script.vim
    ```

* vim 中`<tab>`与`<C-i>`等价

    ```vim
    " 将 <leader>t 映射为插入实际的 Tab 字符
    inoremap <leader>t <Tab>

    " 使用 Ctrl+i，这与 Tab 键在插入模式下效果相同
    inoremap <leader>t <C-i>
    ```

    但是如果在字符串中，需要用`\t`表示 tab 键，不能使用`<tab>`。


* vim 的`:source xxx.vim`是在当前环境中执行`vim`脚本，之前定义的函数会被保留。

* vim 可以使用单引号作为字符串，也可以使用双引号

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


* vim 开启语法高亮

    ```vim
    syntax on
    ```

* vim 的三种模式与切换

    vim is a modal editor, and has 3 modes:

    1. If the bottom of the screen displays the filename or is blank, you are is normal mode.

    2. If you are in insert mode, the indicator displays `--INSERT--`.

    3. if you are in visual mode, the indicator shows `--VISUAL--`.

    enter inserting mode: type `i`

    back to command mode: press `<Esc>` key.

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

* vim `.`命令

    作用：重复上一次修改操作

    详细说明：

    * 重复最近一次在普通模式下执行的修改命令

    * 可以重复插入、删除、替换等操作

    * 示例：

        * dw 删除一个单词 → . 再删除下一个单词

        * ihello<Esc> 插入文本 → . 再次插入"hello"



* vim 打开文件后，跳转到上次关闭时候的位置：

    * 反引号 + 双引号：`` ` `` + `"`

    * 单引号 + 双引号：`'` + `"`

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

### 快捷键与映射

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

* s:local_var 和 l:var_name 有什么区别

    1. l:（局部变量）

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

    2. s:（脚本局部变量）

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

    重要细节

    * 前缀省略：在函数内，局部变量有时可以省略 l: 前缀：

        ```vim
        function! Test()
            let count = 5    " 隐式使用 l:count
            echo count       " 等同于 echo l:count
        endfunction
        ```

        但为了清晰，建议显式使用 l:。

    * a: 前缀：函数参数必须使用 a: 前缀：

        ```vim
        function! Add(x, y)
            return a:x + a:y  " 正确
            return x + y      " 错误！找不到变量 x
        endfunction
        ```

    * 跨脚本访问 s:（不推荐但可能）：

        ```vim
        " 脚本 A: let s:secret = 42
        " 脚本 B: 无法直接访问 scriptA#secret
        " 但可通过 <SID> 和函数间接访问
        ```

    **最佳实践建议**

    * 函数内部：

        * 临时计算用 l:（或省略前缀）

        * 参数用 a:

        * 修改选项用 &

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

* 还有哪些常用的内置函数，像 getline() 这种

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

### 插入、删除与修改

* 删除/更改到字符

    * dtx - 删除直到字符 'x' 前

    * dfx - 删除直到字符 'x'（包括 'x'）

    * dTx - 向后删除直到字符 'x' 前

    * dFx - 向后删除直到字符 'x'（包括 'x'）

* 文本对象操作

    * diw - 删除当前单词（光标在单词任意位置）

    * daw - 删除当前单词（包括周围空格）

    * di" - 删除引号内的内容

    * da" - 删除引号及内容

    * di( 或 dib - 删除括号内的内容

    * da( 或 dab - 删除括号及内容

    * dit - 删除 HTML/XML 标签内的内容

    * dat - 删除整个 HTML/XML 标签

* 从当前位置删除到指定字符

    `dt<c>`: 删除从当前位置到`<c>`之前的所有字符，`<c>`不被删。

    `df<c>`: 删除从当前位置到`<c>`的所有字符，`<c>`被删。

### 搜索与正则表达式

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

### 导航与跳转

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

        `gv`可以重新选择上次可视模式选中的区域。如果是粘贴后立即使用，会选中刚粘贴的内容。

        注：

        1. `gv`重选后，光标会跳转到选中内容的开头还是结尾？

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

* `}`

    移动到下一个空行的第一个非空白字符（段落移动）

    注意事项：

    * 配合 { 命令（向上跳转到上一个空行）使用

    * 计数前缀可用：3} 向下跳转3个段落

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

### tab 处理

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

### 自动补全

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

    🌟 主流选择

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

    🔧 配置示例

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

    🚀 使用方式

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

help: `:help`

jump to tag: `ctrl` + `]`

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
