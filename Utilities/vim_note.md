# vim note

## cache

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

* `set tags=./tags,./TAGS,tags,TAGS,/path/to/other/tags`

    设置 Vim 查找 tags 文件的搜索路径列表，用逗号分隔多个路径。

    解释：

    * `./tags` - 当前文件所在目录的 tags 文件

    * `./TAGS` - 当前文件所在目录的 TAGS 文件（大写版本）

    * `tags` - 当前工作目录的 tags 文件

    * `TAGS` - 当前工作目录的 TAGS 文件（大写版本）

    * `/path/to/other/tags` - 指定的绝对路径下的 tags 文件

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

* ctags 基本用法

    install: `sudo apt install universal-ctags`

    进入工程目录，执行`ctags -R .` (递归地为当前目录及所有子目录中的文件生成 tags)，执行完后会生成`tags`文件。

    常用快捷键：

    * `Ctrl-]`: 跳转到光标下符号的定义处

    * `g Ctrl-]`: 如果有多个匹配的定义，此命令会列出所有候选，让你选择跳转到哪一个

    * `Ctrl-t`: 跳回到跳转之前的位置（类似于“后退”按钮）。可以多次按它来回溯跳转历史。

    * `:ts <tag>`或`:tselect <tag>`: 列出所有匹配`<tag>`的标签定义，供你选择。

    * `:tjump <tag>`: 跳转到`<tag>`。如果只有一个匹配则直接跳转，有多个则列出列表。

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