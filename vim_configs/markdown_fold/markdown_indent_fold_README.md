# Markdown heading + indent folding experiment

Open the demo in your normal Vim, then source the script:

```vim
:edit ~/Documents/Projects/vim_test/markdown_indent_fold_demo.md
:source ~/Documents/Projects/vim_test/markdown_indent_fold.vim
```

Use `zc` in the indented details to close the indentation fold. Use `zc` again
to close the containing heading's body (or the next enclosing indentation level).
The heading itself stays visible when its body is folded. Child headings are
inside the parent's body, so folding a parent hides its child headings as well.
Use `zo` to reopen a level and `zR` to open everything.

Rules:

- ATX headings (`#` through `######`, followed by whitespace or end of line)
  form outer body folds, excluding their own heading lines. Up to three leading
  spaces are allowed. To close a body fold, place the cursor in the body.
- Actual heading nesting determines the base fold level. Skipping from `#` to
  `###` does not introduce a phantom intermediate heading.
- Each `shiftwidth()` of indentation adds one inner fold level. With
  `shiftwidth=4`, four spaces add one level. Tabs use Vim's indentation rules.
- List markers do not start folds; only their indented contents do.
- Each blank run gets the lower level of its surrounding nonblank lines.
  All trailing blanks before the next sibling/ancestor heading, or at EOF,
  remain outside the heading's body fold, preserving their visible count.
- Backtick/tilde fenced code blocks do not introduce headings, but their
  indentation still contributes inner folds.
- Before any heading, only indentation contributes fold levels.

This is a lightweight folding rule, not a full Markdown parser: Setext headings
and headings prefixed by blockquote/list markers are not recognized. Four-space
indented `#` lines count as indented content. Code blocks do not reset indentation.

The script registers a `FileType markdown` autocmd and also applies immediately
when sourced from a Markdown buffer. Other filetypes are unaffected. New
Markdown windows start with all folds open (`foldlevel=99`). Keep vim-markdown's
folding disabled if you re-enable that plugin. The main `.vimrc` is unchanged.

After trying it, to enable permanently add this line at the end of `.vimrc`:

```vim
source ~/Documents/Projects/vim_test/markdown_indent_fold.vim
```

Run automated checks from this directory:

```sh
vim -X -Nu NONE -i NONE -n -es -V1 -S test_markdown_indent_fold.vim
```

## `markdown_indent_fold.vim` 逐行解释

下面对应当前脚本的第 1～99 行。空白行只用于分隔代码，没有执行效果。阅读时可以在另一个 Vim 窗口打开脚本，用 `:set number` 显示行号。

### 先理解：脚本交给 Vim 的是什么

脚本不直接执行 `zc`，也不删除或改写文件里的任何文字。它为每一行计算一个“折叠层级”，再把这些数字交给 Vim。

- 层级 `0`：不在任何折叠里。
- 层级 `1`：在第一层折叠里。
- 层级 `2`：还在第一层内部的第二层折叠里。
- 连续的、层级至少为 `1` 的行组成外层折叠；其中层级至少为 `2` 的连续行又组成内层折叠。

例如下面这份文件，假设 `shiftwidth=4`：

| 文本行 | 层级 | 为什么 |
| --- | --- | --- |
| `# 第一章` | 0 | 标题保留在自己的正文折叠之外 |
| `正文` | 1 | 属于第一章 |
| `* 列表 A` | 1 | 没有缩进，不增加层级 |
| `    说明一` | 2 | 章节层级 1，加缩进层级 1 |
| `    说明二` | 2 | 同上，与上一行组成内层折叠 |
| `## 第一节` | 1 | 属于第一章，但不属于自己的正文折叠 |
| `节的正文` | 2 | 属于第一章下的第一节 |
| 空行 | 0 | 下一行是新章，取前后较低层级 |
| `# 第二章` | 0 | 同级新章，结束第一章正文的折叠 |
| `第二章正文` | 1 | 开始第二章正文的折叠 |

因此，在“说明一”中按 `zc`，先关闭两行说明所在的第二层折叠；再按一次 `zc`，关闭第一章正文的第一层折叠。`# 第一章` 本身层级是 0，所以仍然显示。

注意：“标题层级”在这个脚本里指实际的标题嵌套深度，不一定等于 `#` 的数量。例如从 `#` 直接跳到 `###`，实际只有两层标题。

### Vimscript 中会反复用到的语法

| 写法 | 含义 |
| --- | --- |
| `" 注释` | 双引号在此处表示注释的开始，后面的说明不执行 |
| `let 名称 = 值` | 创建变量或给变量赋值 |
| `s:name` | 脚本内部的变量或函数；`s:` 表示 script |
| `l:name` | 当前函数调用中的局部变量；`l:` 表示 local |
| `b:name` | 当前 buffer 的变量，不同文件各自保存一份 |
| `a:lnum` | 函数参数，表示调用者传入的行号 |
| `v:lnum` | Vim 内置变量；计算折叠表达式时，表示正在询问的行号 |
| `&l:shiftwidth` | 读取当前局部选项 `shiftwidth` 的值；这里的 `l:` 与变量 `l:name` 的用途不同 |
| `[]`、`[1, 2]` | 列表，类似数组；下标从 0 开始 |
| `list[-1]` | 列表的最后一个元素 |
| `{'key': value}` | 字典，按名称保存数据；可用 `.key` 读取 |
| `call Func(...)` | 调用函数，不使用它的返回值 |
| `!条件` | 逻辑“非”；`!empty(x)` 就是“x 不为空” |
| `&&`、`\|\|` | 逻辑“并且”“或者”，可短路求值，左侧已决定结果时不计算右侧 |
| `==#`、`!=#` | 区分大小写的相等、不相等比较，不受 `ignorecase` 影响；也能用于列表等数据 |
| `=~#`、`!~#` | 区分大小写地匹配、不匹配正则表达式 |
| `条件 ? A : B` | 条件成立取 A，否则取 B |
| 行首 `\` | 将这一物理行接到上一行，组成一条较长的命令 |

这里的字符串大多使用单引号，例如 `'\s*'`。单引号字符串保留正则里的反斜杠，比较适合书写 Vim 正则。

### 第 1～9 行：准备计算折叠层级

```vim
" Markdown headings contain indent folds. Source this file to try it.
" Only ATX headings (# ... through ###### ...) are recognized.
let s:cache_version = 3

function! s:BuildLevels() abort
    let l:levels = []
    let l:headings = []
    let l:fence_char = ''
    let l:fence_width = 0
```

- 第 1 行：注释，说明标题折叠内部还有缩进折叠，可以通过 `:source` 加载脚本。
- 第 2 行：注释，说明只识别 `#` 到 `######` 这种 ATX 标题，不识别下一行写 `===`、`---` 的标题形式。
- 第 3 行：设置脚本的缓存版本号。修改算法后增加这个数字，可以避免重新加载脚本时继续使用旧算法生成的缓存。它不是 Vim 的版本号。
- 第 5 行：定义内部函数 `s:BuildLevels()`，它负责扫描整份文件。`function!` 中的 `!` 允许重新定义同名函数，方便反复 `:source`；`abort` 表示函数出错时停止执行该函数。
- 第 6 行：创建空列表 `levels`，依次保存第 1 行、第 2 行……的折叠层级。文件行号从 1 开始，列表下标从 0 开始，因此第 N 行的结果保存在 `levels[N - 1]`。
- 第 7 行：创建标题路径列表 `headings`。例如读到 `# 章`、再读到 `### 节` 后，列表是 `[1, 3]`。列表保存的是标题的 `#` 数量，而列表长度 2 才是实际嵌套深度。
- 第 8 行：`fence_char` 保存当前代码围栏使用的字符：反引号或 `~`。空字符串表示目前不在围栏代码块内。
- 第 9 行：`fence_width` 保存打开围栏时的字符数量。例如四个反引号开启的代码块，必须用至少四个反引号关闭。

### 第 11～14 行：逐行读取，检查代码围栏

```vim
    for l:lnum in range(1, line('$'))
        let l:text = getline(l:lnum)
        let l:inside_fence = !empty(l:fence_char)
        let l:fence = matchlist(l:text, '^\s*\(`\{3,}\|\~\{3,}\)\(.*\)$')
```

- 第 11 行：`line('$')` 返回文件最后一行的行号。`range(1, 最后行号)` 包含两端，所以循环会依次访问文件中的每一行。
- 第 12 行：`getline(lnum)` 读取该行的文本，不包括行尾换行符。
- 第 13 行：根据 `fence_char` 是否为空，得到“当前是否在代码块里”。逻辑值通常用 `0` 表示假、`1` 表示真。
- 第 14 行：尝试匹配代码围栏。`matchlist()` 匹配成功会返回一个列表，失败返回 `[]`。成功时，`fence[0]` 是整个匹配，`fence[1]` 是围栏符号，`fence[2]` 是围栏后面的文本。

第 14 行正则拆开如下：

| 片段 | 含义 |
| --- | --- |
| `^` | 行首 |
| `\s*` | 零个或多个空白字符，允许围栏前有缩进 |
| `\(` 与 `\)` | 捕获一组匹配结果 |
| `` `\{3,} `` | 至少三个连续反引号 |
| `\|` | 或者 |
| `\~\{3,}` | 至少三个连续的波浪号；转义 `~` 避免其 Vim 正则特殊含义 |
| `\(.*\)` | 将围栏后剩余内容保存为第二组 |
| `$` | 行尾 |

例如一行是三个反引号加 `python`，第一组是三个反引号，第二组是 `python`。

### 第 16～29 行：维护“代码块内/外”状态

```vim
    if l:inside_fence
        if !empty(l:fence) && strpart(l:fence[1], 0, 1) ==# l:fence_char
                    \ && strlen(l:fence[1]) >= l:fence_width
                    \ && l:fence[2] =~# '^\s*$'
            let l:fence_char = ''
        endif
    elseif !empty(l:fence)
        " Backtick fence info strings cannot themselves contain backticks.
        if strpart(l:fence[1], 0, 1) !=# '`' || l:fence[2] !~# '`'
            let l:fence_char = strpart(l:fence[1], 0, 1)
            let l:fence_width = strlen(l:fence[1])
            let l:inside_fence = 1
        endif
    endif
```

- 第 16 行：如果本来就在代码块里，优先检查当前行是不是关闭围栏。
- 第 17 行：要求当前行匹配到了围栏，而且第一个围栏字符与开启时相同。`strpart(字符串, 0, 1)` 取第一个字节；反引号和 `~` 都是单字节字符。
- 第 18 行：还要求关闭围栏的长度不短于开启围栏。`strlen()` 返回字符串的字节数，这里也就是符号数量。
- 第 19 行：还要求围栏后只有空白。关闭围栏后不能再跟 `python` 这样的说明。
- 第 20 行：满足全部条件，清空 `fence_char`，让下一行回到代码块外。当前行的 `inside_fence` 仍保持真，因此关闭围栏本身仍按代码块处理。
- 第 21 行：结束关闭围栏的条件判断。
- 第 22 行：如果本来不在代码块里，但当前行匹配到了围栏，就尝试开启代码块。
- 第 23 行：注释，说明反引号围栏后面的说明文字不能包含反引号。
- 第 24 行：如果是波浪号围栏，可以继续；如果是反引号围栏，则要求后面的文本没有反引号。`||` 表示两种情况满足一种即可。
- 第 25 行：记录开启围栏使用的字符。
- 第 26 行：记录开启围栏的长度。
- 第 27 行：将当前行也标记为在代码块内。
- 第 28 行：结束开启围栏的合法性判断。
- 第 29 行：结束整个代码块状态判断。

这样代码块里的 `# 注释` 不会被当成章节标题。但代码块内部的缩进仍会贡献缩进折叠层级。

### 第 31～41 行：识别标题，维护标题路径

```vim
    let l:heading = l:inside_fence ? '' :
                \ matchstr(l:text, '^ \{0,3}\zs#\{1,6}\ze\(\s\|$\)')
    if !empty(l:heading)
        let l:rank = strlen(l:heading)
        while !empty(l:headings) && l:headings[-1] >= l:rank
            call remove(l:headings, -1)
        endwhile
        call add(l:headings, l:rank)
        " Keep the heading outside its own fold, but inside its parent.
        " The following content rises one level and starts the body fold.
        call add(l:levels, len(l:headings) - 1)
```

- 第 31 行：如果在代码块内，直接把 `heading` 设为空字符串；否则执行下一行的标题匹配。
- 第 32 行：提取标题的 `#` 序列。`matchstr()` 返回匹配到的字符串，失败返回 `''`。`^ \{0,3}` 允许行首有 0～3 个普通空格；`#\{1,6}` 要求有 1～6 个 `#`；`\(\s\|$\)` 要求后面有空白或直接结束。`\zs` 和 `\ze` 限定实际返回的范围，因此只返回 `#`，不返回前后的空格。
- 第 33 行：如果 `heading` 不为空，当前行就是标题。
- 第 34 行：计算 `#` 的数量，得到标题的等级 `rank`。
- 第 35 行：如果旧标题路径的最后一项，与新标题同级或比新标题更深，就应退出那个旧标题。
- 第 36 行：删除标题路径最后一项。`remove(list, -1)` 表示移除列表末项。
- 第 37 行：回到 `while` 判断，必要时继续删除，直到最后一项是新标题的父级，或者列表为空。
- 第 38 行：把新标题加到路径末尾。
- 第 39 行：注释，说明标题行在自己的折叠外，但仍在父标题正文的折叠内。
- 第 40 行：注释，说明后续正文比标题行高一级，从而形成正文折叠。
- 第 41 行：将“标题路径长度减 1”作为标题行的折叠层级，追加到结果列表。

标题路径的变化可以这样理解：

| 读到的标题 | 处理前 | 删除哪些旧项 | 处理后 |
| --- | --- | --- | --- |
| `# 第一章` | `[]` | 无 | `[1]` |
| `### 第一节` | `[1]` | 无 | `[1, 3]` |
| `#### 子节` | `[1, 3]` | 无 | `[1, 3, 4]` |
| `### 第二节` | `[1, 3, 4]` | 4 和旧的 3 | `[1, 3]` |
| `# 第二章` | `[1, 3]` | 3 和旧的 1 | `[1]` |

第 41 行的减 1 正是“保留标题行”的关键。例如路径 `[1, 3]` 长度是 2，当前标题行层级为 1，它下面不缩进的正文层级为 2。所以关闭第二层时，正文消失，标题保留；关闭父级第一层时，这个子标题才一起隐藏。

### 第 42～48 行：处理空行和普通内容

```vim
    elseif l:text =~# '^\s*$'
        " Resolve each blank run explicitly after scanning the document.
        call add(l:levels, -1)
    else
        call add(l:levels, len(l:headings) + indent(l:lnum) / shiftwidth())
    endif
endfor
```

- 第 42 行：当前行不是标题，再检查它是否为空行或仅有空白字符。`^\s*$` 要求从头到尾都是空白，也允许长度为零。
- 第 43 行：注释，说明连续空行会在整份文件扫描后统一处理。
- 第 44 行：暂时保存 `-1`，表示“待定”。它只是当前计算过程中的占位符，后面会全部替换成具体层级，不会直接交给 Vim。
- 第 45 行：剩下的都是普通内容，包括正文、列表项和代码块中的文本。
- 第 46 行：计算“标题嵌套深度 + 缩进层级”。`indent(lnum)` 返回行首缩进占用的显示列数，会考虑 Tab；`shiftwidth()` 返回实际生效的缩进宽度，如果选项 `shiftwidth=0`，会采用 `tabstop`。两者在这里进行整数除法，例如 `7 / 4 = 1`，`8 / 4 = 2`。
- 第 47 行：结束标题/空行/普通内容的分支判断。
- 第 48 行：结束逐行扫描。

这里没有任何专门识别 `*` 列表的逻辑。`* 列表` 和普通正文一样计算，只有它下面的缩进内容才增加层级。这就是为什么列表项本身不包含在自己的缩进折叠内。

### 第 50～69 行：给每一段连续空行分配明确层级

这一段解决“多个末尾空行在折叠后没有全部保留显示”的问题。原则是：一整段空行统一取前后非空行的较低层级。文件开头或结尾缺少一侧时，那一侧按 0 处理。

```vim
let l:index = 0
while l:index < len(l:levels)
    if l:levels[l:index] != -1
        let l:index += 1
        continue
    endif
    let l:start = l:index
    let l:before = l:start > 0 ? l:levels[l:start - 1] : 0
    while l:index < len(l:levels) && l:levels[l:index] == -1
        let l:index += 1
    endwhile
    let l:after = get(l:levels, l:index, 0)
    let l:blank_level = min([l:before, l:after])
    for l:blank in range(l:start, l:index - 1)
        let l:levels[l:blank] = l:blank_level
    endfor
endwhile
```

- 第 50 行：注释，说明整段空行统一采用前后较低层级。
- 第 51 行：注释，特别强调章节末尾、文件末尾的空行应排除在正文折叠之外。
- 第 52 行：注释，说明不把占位符 `-1` 留给 Vim 自动计算。
- 第 53 行：从列表下标 0，也就是文件第 1 行，开始检查。
- 第 54 行：只要还没走到列表末尾，就继续。
- 第 55 行：如果当前层级不是 `-1`，说明它不是待处理空行。
- 第 56 行：下标加 1，移动到下一行。
- 第 57 行：`continue` 跳过本次循环余下部分，回到第 54 行。
- 第 58 行：结束这个判断。
- 第 59 行：运行到这里就意味着找到了空行，记录这段空行的起始下标。
- 第 60 行：读取这段空行前一行的层级；如果空行位于文件开头，没有前一行，就取 0。
- 第 61 行：向后寻找这段连续空行的结束位置。先检查下标没有越界，再读取列表元素，避免访问不存在的位置。
- 第 62 行：只要遇到 `-1` 就继续向后移动。
- 第 63 行：结束寻找。此时 `index` 指向空行段之后的第一行，或者刚好指向列表末尾之外。
- 第 64 行：读取后面那行的层级。`get(列表, 下标, 默认值)` 在下标不存在时返回默认值 0，因此文件末尾也能正常处理。
- 第 65 行：`min([前面层级, 后面层级])` 取两者较小值。
- 第 66 行：遍历从 `start` 到 `index - 1` 的全部空行下标。注意 `range()` 包含结束下标。
- 第 67 行：把这些空行的占位符全部改成同一个明确层级。
- 第 68 行：结束给这一段空行赋值。
- 第 69 行：继续扫描剩余部分；`index` 已经在空行段后面，所以不会重复处理同一段。

例如原始层级为 `[1, -1, -1, -1, 0]`，处理后是 `[1, 0, 0, 0, 0]`。关闭层级 1 的正文时，这三行空行都保持显示，不会被包进折叠。

子标题之间的空行可能是层级 1：它们在子标题正文的层级 2 折叠之外，但仍在父标题的层级 1 折叠内。因此折叠父章节时，这些内部空行仍会随父章节隐藏。父章节末尾的空行则保持在父章节折叠之外。

### 第 71～75 行：保存计算结果，避免反复扫描文件

```vim
let b:markdown_indent_fold_cache = {
            \ 'key': [s:cache_version, b:changedtick, &l:shiftwidth, &l:tabstop, &l:vartabstop],
            \ 'levels': l:levels,
            \ }
endfunction
```

- 第 71 行：在当前 buffer 中保存缓存字典。不同 Markdown 文件的缓存互不混用。
- 第 72 行：`key` 保存决定计算结果的条件：脚本算法版本、文本修改计数，以及三个缩进相关选项。`b:changedtick` 是 Vim 维护的修改计数，编辑文本时会变化；`tabstop` 决定 Tab 宽度；`vartabstop` 可指定可变 Tab 宽度。任一条件改变，都需要重新计算。
- 第 73 行：`levels` 保存完整的逐行折叠层级列表。
- 第 74 行：结束字典。行首的反斜杠表示这些物理行属于第 71 行开始的同一条赋值命令。
- 第 75 行：结束 `s:BuildLevels()` 函数。

为什么要缓存？Vim 会分别询问每一行的层级。如果每问一行都重新扫描整份文件，大文件会变慢。现在同一份未修改文本，只在需要时扫描一次，其余查询直接查列表。

### 第 77～84 行：回答 Vim“这一行是什么层级”

```vim
function! MarkdownIndentFold(lnum) abort
    let l:key = [s:cache_version, b:changedtick, &l:shiftwidth, &l:tabstop, &l:vartabstop]
    if !exists('b:markdown_indent_fold_cache')
                \ || b:markdown_indent_fold_cache.key !=# l:key
        call s:BuildLevels()
    endif
    return get(b:markdown_indent_fold_cache.levels, a:lnum - 1, 0)
endfunction
```

- 第 77 行：定义供折叠表达式调用的函数，参数 `lnum` 是 Vim 正在询问的行号。没有 `s:` 前缀，所以这是一个全局函数，可以在 `foldexpr` 里按名称调用。
- 第 78 行：根据当前状态生成新的缓存条件列表。
- 第 79 行：检查当前 buffer 是否还没有缓存。`exists()` 接受变量名字符串，返回该变量是否存在。
- 第 80 行：或者，虽然有缓存，但它的条件与当前条件不同。`||` 的短路求值很重要：如果缓存不存在，就不会尝试访问不存在的 `.key`。
- 第 81 行：上述任一情况成立，都调用内部函数重新生成全部层级。
- 第 82 行：结束缓存检查。
- 第 83 行：返回指定行的层级。函数参数需要通过 `a:lnum` 读取；减 1 是为了把 Vim 行号转换为列表下标。第三个参数 0 是防止下标越界时出错的默认值。
- 第 84 行：结束查询函数。

例如 `MarkdownIndentFold(5)` 就是在问“第 5 行的层级是多少”。函数会返回 `levels[4]`。

### 第 86～90 行：让当前 Markdown 窗口采用这个算法

```vim
function! s:Setup() abort
    setlocal foldexpr=MarkdownIndentFold(v:lnum)
    setlocal foldmethod=expr
    setlocal foldlevel=99
endfunction
```

- 第 86 行：定义内部设置函数。
- 第 87 行：设置 `foldexpr`，告诉 Vim 计算某一行折叠层级时，调用 `MarkdownIndentFold(v:lnum)`。这里是登记表达式，不是在设置时立即对所有行执行一次。
- 第 88 行：将折叠方法设为 `expr`，正式使用上面的表达式。之前全局的 `foldmethod=indent` 不再决定这个窗口的折叠。
- 第 89 行：设置 `foldlevel=99`，使普通情况下的折叠初始全部展开。`foldlevel` 控制显示开合，不负责计算每行层级；之后仍可用 `zc`、`zo` 手动开合。它也不改变 `foldnestmax` 对最大折叠深度的限制。
- 第 90 行：结束设置函数。

`setlocal` 只设置当前局部值。这里的折叠选项是窗口局部选项，因此不会把所有其他文件窗口都改成这个算法。再次加载脚本并执行设置函数，也会把当前窗口的 `foldlevel` 设回 99。

### 第 92～99 行：自动启用，以及手动加载时立即生效

```vim
augroup MarkdownIndentFolding
    autocmd!
    autocmd FileType markdown call s:Setup()
augroup END

if &l:filetype ==# 'markdown'
    call s:Setup()
endif
```

- 第 92 行：进入名为 `MarkdownIndentFolding` 的自动命令组，用来管理本脚本的自动命令。
- 第 93 行：清除当前组内旧的自动命令，防止每次 `:source` 都重复登记。因为已经进入了专用组，不会清除其他组的自动命令。
- 第 94 行：登记规则：发生 `FileType` 事件且文件类型是 `markdown` 时，调用设置函数。通常打开 Markdown 文件、Vim 识别出文件类型时会触发。
- 第 95 行：退出自动命令组。
- 第 97 行：检查当前窗口的文件类型是否已经是 Markdown。
- 第 98 行：如果是，立即执行设置函数。这样你在已经打开的 Markdown 文件里 `:source` 时，不用等待下一次 `FileType` 事件。
- 第 99 行：结束判断。

第 94 行负责以后打开的 Markdown 文件，第 97～99 行负责加载脚本时已经打开的当前文件。判断依据是 `filetype`，不是直接检查文件名后缀。

### 重新加载与排查

修改脚本后，在 Markdown 窗口执行：

```vim
:source ~/Documents/Projects/vim_test/markdown_indent_fold.vim
:normal! zx
```

第一行重新定义函数、登记自动命令，并对当前 Markdown 窗口应用设置。第二行用 Vim 原生的 `zx` 重新计算折叠，并按当前 `foldlevel` 更新开合状态、展开光标所在位置。它不会修改文件文字，但可能重置之前手动关闭的折叠。

可以用下面的命令查看实际状态：

```vim
:setlocal filetype? foldmethod? foldexpr? shiftwidth? tabstop?
:echo MarkdownIndentFold(line('.'))
:echo b:markdown_indent_fold_cache.levels
:verbose setlocal foldmethod?
```

依次用于：检查相关选项、查询当前行计算出的层级、查看全部缓存层级、检查最后是谁设置了 `foldmethod`。`levels` 里的数字只是折叠计算结果，不代表那些折叠此刻一定处于关闭状态。
