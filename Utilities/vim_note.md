# vim note

## cache

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