# vim note

## cache

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