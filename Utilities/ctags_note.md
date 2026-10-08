# ctags note

* {postrun}{scope=intervaltab}

    先让内建 Markdown parser 建立各级标题的 `line/end` 区间，再把列表项归入包含它的最内层标题 scope

    在 `~/.ctags.d/markdown-list.ctags` 写入：

    ```
    --kinddef-Markdown=i,listItem,list items
    --regex-Markdown=/^[ \t]*[*][ \t]+(.+)$/\1/i/{postrun}{scope=intervaltab}
    ```

    验证：

    ```
    ctags --fields=+nKse --sort=no -o - msg.md
    ```

* ctags --list-kinds=Markdown

    Ctags 会输出它内置的 Markdown Kind 定义

* 详细讲讲这两行是什么意思

    --kinddef-Markdown=i,listItem,list items
    --regex-Markdown=/^[ \t]*[*][ \t]+(.+)$/\1/i/

    结论：这两行是 **Universal Ctags 的 optlib 正则解析器定义语法**。第一行定义了一个新的 tag 种类（kind），第二行用正则规则告诉 ctags 如何从 Markdown 文件中识别这种 kind 并生成 tag。

    ---

    ## 第一行：定义 Kind

    ```
    --kinddef-Markdown=i,listItem,list items
    ```

    这是 ctags 的 `--kinddef-<LANG>` 选项，用于为某种语言定义一个自定义的 tag 种类（kind）。

    **格式拆解：**

    | 部分 | 值 | 含义 |
    |------|-----|------|
    | `Markdown` | 语言名 | 这个 kind 定义属于 Markdown 解析器 |
    | `i` | 单字母缩写 | 该 kind 的 letter 形式，后续在 regex 中引用 |
    | `listItem` | 长名称 | kind 的内部名称 |
    | `list items` | 描述 | 人类可读的说明 |

    **作用：** 在 Markdown 语言中注册一个名为 `listItem` 的 tag 种类，缩写为 `i`。这样 ctags 就知道 Markdown 文件里可以提取出“列表项”这种类型的 tag。

    ---

    ## 第二行：定义正则匹配规则

    ```
    --regex-Markdown=/^[ \t]*[*][ \t]+(.+)$/\1/i/
    ```

    这是 ctags 的 `--regex-<LANG>` 选项，用于通过正则表达式提取 tag。

    **格式拆解：**

    ```
    --regex-Markdown=/正则表达式/替换/flag/
    ```

    | 部分 | 内容 | 含义 |
    |------|------|------|
    | 正则 | `^[ \t]*[*][ \t]+(.+)$` | 匹配以可选空白开头、后跟 `*` 和空白、再跟内容的行 |
    | 替换 | `\1` | 用捕获组 1（即列表项文字）作为 tag 名 |
    | flag | `i` | 指定生成的 tag 种类为 `listItem`（对应第一行的缩写） |

    **正则细节：**

    - `^[ \t]*` — 允许行首有缩进（空格或制表符），支持嵌套列表
    - `[*]` — 字面量星号（用方括号避免被当作量词）
    - `[ \t]+` — `*` 后必须至少有一个空白
    - `(.+)` — 捕获组，匹配列表项的文本内容
    - `$` — 行尾

    **匹配示例：**

    | 源代码行 | 生成的 tag 名 |
    |----------|---------------|
    | `* 苹果` | `苹果` |
    | `  * 香蕉` | `香蕉` |
    | `- 橙子` | **不匹配**（只认 `*`） |

    ---

    ## 整体效果

    这两行组合起来的效果是：

    让 ctags 在解析 Markdown 文件时，把每一行 `* 内容` 格式的列表项提取出来，以“内容”作为 tag 名，并标记 tag 种类为 `listItem`。

    生成 tags 文件后，在 Vim 中把光标放在某个列表项文字上按 `Ctrl-]`，就能跳转到该列表项的定义位置。

* git repo

   <https://github.com/universal-ctags/ctags> 

   build and install:

   <https://docs.ctags.io/en/latest/autotools.html#gnu-linux-distributions>

* optlib doc

    <https://docs.ctags.io/en/latest/man/ctags-optlib.7.html>

    Extending ctags with Regex parser (optlib):

    <https://docs.ctags.io/en/latest/optlib.html>

* kind字母只能是单个字母吗

    **是的，ctags 的 kind 字母严格限定为单个字符。**

    ### 📌 官方定义

    根据 Universal Ctags 的官方手册，kind 字母必须是一个**字母字符**（`[0-9a-zA-Z]`），但排除了大写 `F`，因为 `F` 被保留用于表示“文件”（file）。

    ### 🔤 单字母 vs 长名称

    虽然 ctags 也支持使用**长名称**作为 kind 的替代标识，但在底层，每个 kind 仍然必须绑定一个唯一的单字母标志：

    - **单字母标志（one-letter flag）**：这是核心标识，在 `tags` 文件中用于过滤和分类。
    - **长名称（long-name flag）**：可以作为单字母的替代，在启用 `K` 字段时，输出中会使用长名称而非单字母。

    例如，在 C 语言中，`--kinds-C=+'{local}'` 和 `--kinds-C=+l` 是等价的，`local` 就是 `l` 这个单字母的长名称。

    ### 💡 对你的 Markdown 配置意味着什么

    回到你之前那段 Tagbar 配置，里面用了 `c`、`s`、`S`、`t`、`T`、`u` 这些单字母来区分不同级别的标题。这是完全符合 ctags 规范的。

    **需要注意的是**：ctags 的 kind 字母**区分大小写**，所以 `s` 和 `S` 可以同时存在，代表不同的 kind 类型。这也是你的 Markdown 配置里能用 `s` 表示 section、用 `S` 表示 subsection 的原因。

    如果未来你想给 Markdown 增加更多种类的标签（比如“代码块”或“链接”），你仍然需要为它们分配**单个字母**，而不能用多字母的字符串。

* 那如果单个字母不够用怎么办

    单个字母不够用时，Universal Ctags 本身提供了一套**长名称（long-name flag）** 机制，但在 Vim/Tagbar 的配置层面，你通常需要**换个思路**来解决。

    ### 🧩 Ctags 层面的“长名称”

    Ctags 的 kind 标识实际上有两种形式：
    *   **单字母标志**：如 `c`、`s`、`S`，用于内部区分和过滤。
    *   **长名称标志**：如 `chapter`、`section`、`subsection`，是给人类看的完整描述。

    例如，Markdown 解析器内部实际上是这样定义的：
    ```c
    { true, 'c', "chapter",       "chapters"},
    { true, 's', "section",       "sections" },
    { true, 'S', "subsection",    "level 2 sections" },
    ```
    当你启用 `--fields=+K` 选项输出时，标签文件里显示的就是长名称 `chapter`、`section`，而不是单字母 `c`、`s`。

    **但这里有个关键限制**：长名称是解析器**内部固定**的，你**无法在 Tagbar 配置里直接引用长名称来替代单字母**。`kind2scope` 和 `scope2kind` 的键值必须使用单字母。

    ### 🛠️ Tagbar 配置层面的实际解法

    既然不能用长名称，当 `g:tagbar_type_markdown` 里的 26 个单字母（区分大小写共 52 个）不够用时，**实际做法是：把多个相似的语义合并到同一个 kind 字母下**。

    在你的 Markdown 配置里，本来有 6 级标题（`c`、`s`、`S`、`t`、`T`、`u`）。如果还要新增“代码块”和“链接”两种类型，理论上只剩 `l`、`i` 等少数空闲字母。

    **推荐做法：复用现有的单字母，用 kind 的长名称来区分语义。**

    比如，让四级标题 `t`（subsubsection）和“代码块”共用同一个 kind 字母 `t`，但在 Tagbar 的 `kinds` 数组里，通过不同的“显示名”来区分：

    ```vim
    'kinds' : [
      \ 'c:chapter:0:1',
      \ 's:section:0:1',
      \ 'S:subsection:0:1',
      \ 't:subsubsection:0:1',
      \ 'T:l4subsection:0:1',
      \ 'u:l5subsection:0:1',
      \ 't:codeblock:0:1',  " 与 subsubsection 共用字母 t
    \ ],
    ```

    这样 Tagbar 侧边栏会显示两个不同的类别，虽然它们在 ctags 内部都标记为 `t`。

    **更彻底的方案是：为 Markdown 写一个自定义的 ctags 解析器（Optlib）**，就像你之前问的 `myfoo` 那样。你可以自由分配 52 个单字母，用 `--kinddef-markdown` 定义每个字母对应的长名称，然后在 Tagbar 配置里为每个字母设置对应的显示名。这能从根本上解决字母不够用的问题。

    ### 💡 核心原则

    Ctags 的 **单字母是硬限制**，因为 tag 文件的格式设计和内部过滤机制都依赖它。长名称只是“显示层”的便利。在 Tagbar 集成中，你通过**合并语义到同一个字母**或**自定义解析器**来绕开这个限制。

* 详细讲讲 universal ctags 的用法

	Universal Ctags 是一个从 Exuberant Ctags 演化而来的、正在积极维护的代码索引工具。它的核心作用是扫描源代码，生成一个名为 `tags` 的索引文件，让你的编辑器（如 Vim、Neovim、Emacs 或 VS Code）能够实现“跳转到定义”功能。

	### 🚀 快速上手

	对于大多数项目，最常用的命令就是在项目根目录执行递归扫描：
	```bash
	ctags -R .
	```
	这条命令会扫描当前目录下的所有文件，并生成一个 `tags` 文件。它的速度非常快，对于几万行的代码库通常只需几十毫秒，你可以放心地在每次 `git pull` 或切换分支后重建索引。

	### ⚙️ 核心选项与定制

	虽然默认设置已经够用，但 Universal Ctags 提供了丰富的选项来满足特定需求。

	**1. 递归与排除**
	`-R` 或 `--recurse` 用于递归扫描子目录。你可以通过 `--exclude` 来排除不需要索引的目录（如 `.git`、`node_modules`），如果只想排除某个目录下的部分文件，可以使用 `--exclude-exception`。
	```bash
	ctags -R --exclude=.git --exclude=node_modules --exclude-exception=src/important.js .
	```

	**2. 语言与种类控制**
	使用 `--languages` 可以指定只扫描特定语言的文件。如果你只想索引函数和类，不关心变量或宏，可以用 `--kinds-<LANG>` 来精细控制。例如，对于 C 语言，`--kinds-c=f` 表示只包含函数（function）。
	```bash
	ctags -R --languages=Python,JavaScript .
	```

	**3. 输出字段与格式**
	默认生成的 `tags` 文件包含了定位所需的基本信息（标签名、文件、搜索模式）。你可以通过 `--fields` 添加更多信息，比如行号（`line`）或结束位置（`end`）。如果你担心与其他工具生成的文件冲突，可以使用 `--put-field-prefix` 为 Universal Ctags 特有的字段加上 `UCTAGS` 前缀。
	```bash
	ctags --fields=+line --put-field-prefix -R .
	```

	**4. 指定输出文件**
	使用 `-f` 或 `-o` 选项可以指定输出文件名，而不是默认的 `tags`。这在多级目录结构中很有用，例如为每个子目录生成独立的标签文件，或者将标签文件输出到标准输出（`-f -`）以便管道处理。
	```bash
	ctags -R -f ./mytags .
	```

	### 💡 进阶用法与场景

	**处理宏的定位问题**
	对于宏定义（`#define`），Ctags 默认记录的是行号。一旦宏所在的文件行数发生变化，旧标签就可能失效。为了避免这个问题，可以使用 `--excmd=p` 强制对宏也使用搜索模式来定位，这样只要宏的代码内容没变，位置就是准确的。

	**大型项目的索引策略**
	对于庞大的代码库，不要在子目录中反复执行 `ctags -a`（追加模式），因为每次追加都会触发一次全文件排序，导致效率极低。最推荐的做法是在根目录执行一次 `ctags -R`。如果确实需要分步构建，可以先用 `--sort=no` 快速生成所有标签，最后再手动或由最后一次 `ctags` 调用统一排序。

	**与编辑器集成**
	生成 `tags` 文件后，只需在编辑器中配置好标签文件路径，就可以使用内置快捷键进行跳转。在 Vim/Neovim 中，`Ctrl-]` 跳转到定义，`Ctrl-t` 跳回。需要注意的是，Ctags 的核心是“定义索引”，它不记录函数调用关系，因此无法直接用于“查找引用”（Find References）。

* 详细讲讲 ctags 如何自定义解析一个文件类型

	为自定义文件类型编写 ctags 解析器，最直接的方式是使用 **Optlib（选项库）** 机制。你只需通过命令行选项或 `.ctags` 配置文件，用正则表达式描述文件中的关键结构（如函数、变量），ctags 就会在扫描时应用这些规则。

	下面以解析一个虚构的 `.foo` 文件为例，演示从零开始定义一个解析器的完整步骤。

	### 📝 定义步骤

	假设 `.foo` 文件中有以下结构，我们需要提取 `def` 定义的函数和 `let` 定义的变量：

	```text
	def my_function {
		...
	}
	let my_variable = 10
	```

	**1. 定义语言名称**
	使用 `--langdef` 创建一个新语言，名字建议用小写，以便区分内置解析器。
	```bash
	--langdef=myfoo
	```

	**2. 关联文件扩展名**
	用 `--map-<LANG>` 告诉 ctags，遇到什么后缀的文件时调用这个新解析器。
	```bash
	--map-myfoo=+.foo
	```

	**3. 定义“种类”（Kind）**
	Kind 用于区分标签的类型（如函数、变量）。使用 `--kinddef-<LANG>` 为每种类型分配一个字母和描述。
	```bash
	--kinddef-myfoo=f,function,functions
	--kinddef-myfoo=v,variable,variables
	```

	**4. 编写正则表达式规则**
	这是核心步骤。使用 `--regex-<LANG>`，格式类似 `sed` 的替换命令：`/匹配模式/提取名称/种类/[标志]`。

	对于函数定义 `def my_function {`：
	```bash
	--regex-myfoo=/^def[ \t]+([a-zA-Z0-9_]+)/\1/f/
	```
	*   `^def[ \t]+([a-zA-Z0-9_]+)`：匹配行首的 `def` 和后面的函数名，用 `()` 捕获函数名。
	*   `\1`：引用捕获组的内容作为标签名。
	*   `f`：指定 Kind 为函数。

	对于变量定义 `let my_variable = 10`：
	```bash
	--regex-myfoo=/^let[ \t]+([a-zA-Z0-9_]+)/\1/v/
	```

	### 💻 使用方式

	将上述所有选项写入一个文件（例如 `myfoo.ctags`），然后在运行 ctags 时通过 `--options` 加载它。

	```bash
	ctags --options=myfoo.ctags -R .
	```
	如果你希望每次自动生效，可以将这些选项写入 `~/.ctags.d/myfoo.ctags` 文件（具体路径取决于你的系统配置）。

	### ⚙️ 进阶技巧

	*   **忽略注释行以提升效率**：如果文件以 `#` 开头是注释，可以在正则规则前加一条“排他性”规则。当匹配到注释行时，ctags 会跳过后续规则，不再尝试提取标签，从而节省解析时间。
		```bash
		--regex-myfoo=/^#.*//{exclusive}
		```
	*   **区分大小写**：默认情况下正则表达式是区分大小写的。如果需要忽略大小写，可以在规则末尾加上 `{icase}` 标志。
	*   **多行匹配**：虽然大部分情况用单行正则即可，但如果结构跨越多行（例如函数定义和名字不在同一行），可以使用 `--mline-regex-<LANG>` 选项来处理。

	通过组合这些简单的正则规则，你就能快速为 ctags 添加对任意文本格式的索引能力。

* 详细讲讲 .ctags 配置文件的规则

    `.ctags` 配置文件的规则，核心是 **Universal Ctags 的 Optlib 系统**。它的作用是不用修改 Ctags 源代码，仅通过配置正则表达式，就能定义新语言的解析器或扩展已有解析器。

    ### 📝 基本语法格式

    一条正则规则的标准写法是：

    ```
    --regex-<LANG>=/pattern/name/[kind/][flags]
    ```

    各部分含义如下：

    *   **`<LANG>`**：语言名称，必须先通过 `--langdef` 或 `--langmap` 定义。
    *   **`pattern`**：用于匹配目标文本的正则表达式。
    *   **`name`**：提取出的标签名，通常用反向引用（如 `\1`）指向 `pattern` 中的捕获组。
    *   **`kind`**：可选，标签的类型标识，可以是单字母（如 `f`）或完整定义（如 `{function}`）。
    *   **`flags`**：可选，控制匹配行为的标志，用大括号 `{}` 包裹，可组合使用。

    ### 🔧 常用标志（Flags）

    标志可以改变正则的匹配方式或行为，以下是最常用的几个：

    **正则语法控制**
    *   **`{icase}` 或 `i`**：忽略大小写。
    *   **`{exclusive}` 或 `x`**：**独占匹配**。如果某一行被这条规则匹配，就跳过其他所有规则。非常适合用来跳过注释行，避免误匹配。
    *   **`{basic}` 或 `b`**：使用 POSIX 基本正则语法。
    *   **`{extend}` 或 `e`**：使用 POSIX 扩展正则语法（默认）。

    **作用域追踪（Scope）**
    这是实现层级结构（如类下的方法）的关键。Ctags 内部维护一个“作用域栈”。
    *   **`{scope=push}`**：将当前匹配的标签压入栈顶，成为后续标签的父作用域。
    *   **`{scope=ref}`**：当前标签作为栈顶标签的子项（仅引用，不修改栈）。
    *   **`{scope=pop}`**：弹出栈顶，表示作用域结束。
    *   **`{scope=set}`**：清空栈后，将当前标签压入。
    *   **`{scope=clear}`**：清空整个作用域栈。
    *   **`{placeholder}`**：生成一个占位标签但不输出到最终文件。常配合 `{scope=push}` 使用，用来追踪没有名字的作用域（如 C 的 `{`）。

    ### 🧩 进阶用法：多行与状态机

    当单行正则不够用时，Ctags 提供了更强大的工具：

    *   **`--mline-regex-<LANG>`**：**跨行匹配**。将整个文件内容视为一个大字符串来匹配，必须配合 `{mgroup=N}` 标志指定使用哪个捕获组来确定标签所在的行号。
    *   **`--_mtable-regex-<LANG>`**：**多表状态机匹配**。这是最强大的功能，类似 lex/yacc。它允许你定义多个“表”（状态），通过规则在不同的表之间跳转，非常适合解析嵌套结构、处理注释块或上下文相关的语法。

    ### 💡 一个完整的例子

    下面这个配置定义了一个简单的 `foo` 语言，它用 `{` 和 `}` 来界定作用域：

    ```ctags
    --langdef=foo
    --map-foo=+.foo
    --kinddef-foo=c,class,classes
    --kinddef-foo=d,definition,definitions

    --regex-foo=/^class[ \t]+([[:alpha:]]+)/\1/c/{scope=set}
    --regex-foo=/^[ \t]+def[ \t]+([[:alpha:]]+)/\1/d/{scope=ref}
    ```

    *   当匹配到 `class foo` 时，生成 `c` 类型的标签 `foo`，并用 `{scope=set}` 将其设为当前作用域。
    *   当匹配到 `def bar` 时，生成 `d` 类型的标签 `bar`，并用 `{scope=ref}` 将其记录为 `class:foo` 的子项。

    要调试复杂的正则，可以先用 `ctags --options=你的.ctags文件 -o - 测试文件` 命令，直接查看输出的标签是否符合预期。

* 怎么指定 tag 的层级呢

    在 `.ctags` 配置文件中指定 tag 的层级，核心机制是**作用域（Scope）**。Ctags 通过内部维护一个“作用域栈”来追踪当前解析的位置处于哪个父级之下，从而为每个 tag 记录其父级信息。

    ### 🏗️ 作用域的实现方式

    对于正则表达式自定义的语言，层级完全由你在 `.ctags` 规则中手动控制。对于 Ctags 内置支持的语言（如 C、Java、Markdown），Ctags 的解析器会自动处理作用域，你通常不需要手动干预。

    在正则规则中，层级通过 **`{scope=...}` 标志**来控制。以下是几种核心的 scope 操作：

    *   **`{scope=push}`**：将当前匹配的 tag **压入**作用域栈，使它成为后续 tag 的父级。
    *   **`{scope=ref}`**：当前 tag 仅**引用**栈顶的 tag 作为父级，但**不改变**栈的状态。
    *   **`{scope=pop}`**：弹出栈顶元素，表示一个作用域结束。
    *   **`{scope=set}`**：清空栈后，将当前 tag 压入，作为新的根作用域。
    *   **`{scope=clear}`**：清空整个作用域栈。

    ### 📝 一个具体的层级配置示例

    假设有一种简单的 `foo` 语言，用 `begin` 和 `end` 来划分作用域，用 `def` 定义函数。我们可以这样配置层级：

    ```ctags
    --langdef=foo
    --map-foo=+.foo
    --kinddef-foo=b,block,blocks
    --kinddef-foo=f,function,functions

    --regex-foo=/^begin[ \t]+([[:alpha:]]+)/\1/b/{scope=set}
    --regex-foo=/^[ \t]+def[ \t]+([[:alpha:]]+)/\1/f/{scope=ref}
    --regex-foo=/^end/\//b/{scope=pop}
    ```

    **逻辑解读**：
    1.  遇到 `begin Alpha` 时，生成一个 `b` 类型的 tag `Alpha`，并用 `{scope=set}` 将它设为当前作用域的**根**。
    2.  遇到 `def foo` 时，生成一个 `f` 类型的 tag `foo`，并用 `{scope=ref}` 声明它是**当前栈顶作用域（即 Alpha）的子项**。
    3.  遇到 `end` 时，用 `{scope=pop}` 弹出栈顶，表示 `Alpha` 这个作用域结束了。

    这样生成的 tag 文件里，`foo` 的 scope 字段就会显示 `block:Alpha`，表明它的父级是 `Alpha`。

    ### 💡 关于复杂层级的提示

    对于像 Markdown 标题那种**天然嵌套**（一级标题 > 二级标题 > 三级标题）的结构，单纯用 `push/ref/pop` 来手动管理可能会比较繁琐。Universal Ctags 对于内置支持的 Markdown 解析器，内部有一套更智能的“嵌套层栈”机制：它会比较新标题和栈顶标题的级别，自动弹出所有级别不低于新标题的元素，从而精确还原出 `# a > ## b > ### c` 这样的层级链。

    如果你是为自定义语言配置层级，建议从简单的 `set/ref/pop` 组合开始尝试；如果结构复杂，可能需要考虑使用更强大的多表状态机（`--_mtable-regex`）来模拟这种智能弹栈逻辑。

* universal ctags 和 exuberant ctags 有什么区别？

    Universal Ctags 是 Exuberant Ctags 的活跃延续版本，可以理解为它的“精神续作”与现代化替代品。两者的核心区别在于**维护状态**，由此延伸出功能、语言支持和兼容性上的一系列差异。

    ### 🧬 本质区别：活跃 vs. 停滞

    *   **Exuberant Ctags**：由 Darren Hiebert 开发，最后一次重大更新是在 2009 年（版本 5.8），项目早已**停止维护**。它曾是 Vim/Emacs 用户最主流的选择。
    *   **Universal Ctags**：最初是 Exuberant Ctags 的一个 fork，后来发展为一个独立项目，目标是**继续开发**并建立一个更友好的协作空间。它目前处于**活跃维护**状态。

    ### 🆚 主要差异详解

    虽然同源，但 Universal Ctags 为了向前发展，做出了一些有意的、不兼容的改动。如果你从 Exuberant 迁移，需要留意以下几点：

    **1. 配置文件加载机制（最常见的变化）**
    这是迁移时最容易遇到问题的地方。
    *   **Exuberant Ctags**：自动加载 `~/.ctags` 文件。
    *   **Universal Ctags**：**不再自动加载** `~/.ctags`。取而代之的是，它会加载 `~/.ctags.d/*.ctags` 目录下的所有文件。你需要把旧的配置**迁移**到新目录中。

    **2. 命令行参数的顺序**
    *   **Exuberant Ctags**：比较宽松，允许选项出现在源文件**之后**。
    *   **Universal Ctags**：遵循 POSIX 惯例，要求选项必须在源文件**之前**。如果顺序错了，它会把后面的选项当作文件名，并报错“cannot open input file”。

    **3. 语言与 Kind 定义的严格性**
    Universal Ctags 对自定义语言（optlib）的定义要求更严格，目的是减少冲突。
    *   **Kind 字母**：必须使用**字母字符** `[a-zA-EG-Z]`（字母 `F` 被保留给“文件”这个 kind）。
    *   **Kind 名称**：必须以**字母开头**，后面可以跟字母或数字 `[a-zA-Z][a-zA-Z0-9]*`。
    这意味着，你为 Exuberant 写的、使用数字或其他符号作为 kind 标识的 `.ctags` 文件，可能需要修改才能在 Universal Ctags 中正常工作。

    **4. 文件映射的优先级**
    当文件名同时匹配多个解析器时，两者的选择逻辑不同。
    *   **Exuberant Ctags**：先看**扩展名**，再看**文件名模式**（e-map-order）。
    *   **Universal Ctags**：先看**文件名模式**，再看**扩展名**（u-map-order）。这个改动是为了处理像 `build.xml` 这样的文件——它应该被 Ant 解析器（基于文件名）处理，而不是 XML 解析器（基于扩展名）。

    ### ✨ Universal Ctags 带来的新特性

    除了兼容性调整，Universal Ctags 还引入了不少 Exuberant 没有的功能：
    *   **更广泛的现代语言支持**：对 C/C++ 的解析器进行了重写，对现代语言（如 Go、Rust 等）的支持也更好。
    *   **实验性交互模式**：提供了一个交互式的命令行界面，方便调试和探索。
    *   **增强的 optlib**：在 Exuberant 正则表达式定义语言的基础上，提供了更强大的多表状态机匹配（mtable）等功能。

    ### 💡 给你的建议

    如果你正在开始一个新的 Vim 配置，或者还在使用 Exuberant Ctags，**直接切换到 Universal Ctags 是更好的选择**。它更活跃，对新语言的支持更好，并且是未来的发展方向。

    迁移时，重点检查你的 `~/.ctags` 配置文件是否需要移动到 `~/.ctags.d/` 目录，以及其中自定义的 Kind 定义是否符合新的命名规则。

* 在哪看 universal ctags 是如何处理各个类型的文件的

    Universal Ctags 处理不同类型文件的核心机制在于**语言解析器（Parsers）**，每种语言对应一个专门的解析器。要看它如何处理特定类型文件，最直接的方法是查阅其官方文档中关于解析器的说明，或使用命令行工具动态查询。

    ### 🔍 查看官方文档中的解析器说明
    Universal Ctags 官网（docs.ctags.io）有针对特定语言的解析器说明页面，会详细介绍该语言的处理逻辑、标签类型（Kinds）和实现细节。例如：
    - **Python 解析器**：文档说明了新旧解析器的区别，新解析器将词法分析与语法分析分离，能更好地处理多行结构、装饰器等。
    - **HTML 解析器**：文档提到新解析器能正确处理跨行标签和注释，并支持在 `<script>` 和 `<style>` 标签内调用 JavaScript 和 CSS 解析器。
    - **V 解析器**：文档展示了如何通过 `-d` 选项调试解析器，并说明了其设计规则（如不构建 AST，并行读写 token）。

    ### 💻 使用命令行工具查询
    对于已安装的 Universal Ctags，可以通过以下命令直接获取信息：

    **查看支持的语言列表**：
    ```bash
    ctags --list-languages
    ```

    **查看特定语言识别的标签类型（Kinds）**：
    ```bash
    # 查看 Python 识别的所有标签类型
    ctags --list-kinds-full=Python
    ```
    输出的结果会显示每种 Kind 的字母标识、名称、描述以及是否默认启用。

    **查看文件扩展名与语言的映射关系**：
    ```bash
    # 查看所有映射
    ctags --list-maps
    # 查看特定语言的映射，比如 C++
    ctags --list-maps=C++
    ```
    这能告诉你哪些文件扩展名（如 `.h` 可能同时映射到 C、C++ 和 ObjectiveC）会被关联到哪个语言解析器。

    ### 📂 查看解析器源码
    如果需要深入了解实现原理，可以查看 Universal Ctags 的源码。在 GitHub 仓库的 `parsers/` 目录下，每种语言通常有一个对应的 C 源文件（如 `parsers/python.c`、`parsers/html.c`）。文档“Writing a parser in C”也提供了编写解析器的指南，有助于理解其内部工作流程。

    ### 🧩 特殊机制：多解析器与子解析器
    Universal Ctags 还支持更复杂的解析场景：
    - **子解析器（Subparser）**：例如 Autoconf 解析器建立在 m4 解析器之上。当解析 `configure.ac` 时，会先启用 Autoconf 解析器，它再调用底层的 m4 解析器。选择方式可以是基于文件名的“自顶向下”，也可以是由基础解析器在解析过程中动态检测的“自底向上”。
    - **宿主与访客解析器**：HTML 解析器可以作为宿主，在解析过程中遇到 `<script>` 时，动态调用 JavaScript 解析器来处理嵌入的代码。

    如果你对某个特定语言的处理方式感兴趣，可以告诉我，我帮你查找更具体的资料。

* 我想用自己的方式处理 markdown 文件，如何自定义配置？

    想用自己的方式处理 Markdown 文件，主要有两条路径：**自定义正则解析（Optlib）** 和 **扩展内置解析器**。前者适合完全按你的规则生成标签，后者适合在内置 Markdown 解析的基础上补充信息。

    ### 🛠️ 方法一：自定义正则解析（Optlib）

    这是最灵活的方式，你可以用正则表达式定义完全属于自己的解析规则，不受内置 Markdown 解析器的限制。核心是创建一个 `.ctags` 配置文件，然后用 `--options` 加载。

    **1. 创建一个自定义解析器**

    假设你想把 Markdown 中所有以 `TODO:` 开头的行标记为 `todo` 类型，可以新建文件 `my_markdown.ctags`：

    ```ctags
    --langdef=myMarkdown
    --map-myMarkdown=+.md
    --kinddef-myMarkdown=t,todo,TODO items
    --regex-myMarkdown=/^TODO:\s*(.+)/\1/t/
    ```

    *   `--langdef=myMarkdown`：定义一个名为 `myMarkdown` 的新语言。
    *   `--map-myMarkdown=+.md`：将 `.md` 文件映射到这个新语言。注意这里用的是 `--map-<LANG>`，它比 `--langmap` 更精细，支持多个语言映射到同一扩展名。
    *   `--kinddef-myMarkdown=t,todo,TODO items`：定义一个标签类型。`t` 是单字母标识，`todo` 是名称，后面是描述。
    *   `--regex-myMarkdown=...`：正则表达式。`^TODO:\s*(.+)` 匹配行，`\1` 捕获括号里的内容作为标签名，`t` 表示标签类型。

    **2. 使用自定义解析器**

    运行时通过 `--options` 加载你的配置：

    ```bash
    ctags --options=./my_markdown.ctags -R .
    ```

    **3. 重要细节：处理内置解析器的冲突**

    如果你定义的 `myMarkdown` 和内置的 `Markdown` 语言都映射了 `.md` 文件，ctags 会**同时使用两个解析器**，可能产生重复标签。如果你想完全用自己的规则替代内置解析器，可以在 `--map-myMarkdown=+.md` 之前，先用 `--map-Markdown=-.md` 解除内置 Markdown 对 `.md` 的映射。

    ### 🔗 方法二：扩展内置 Markdown 解析器

    如果你只想在内置解析器的基础上**补充**标签，而不是替代它，可以用 `--regex-Markdown=` 向内置解析器添加规则。

    这种方式下，你的规则会和内置规则同时生效。如果发现内置解析器产生了你不想要的标签，可以用 `--kinds-Markdown=-<letter>` 来禁用特定的内置标签类型。你可以先用 `ctags --list-kinds-full=Markdown` 查看所有内置类型及其字母标识。

    ### 📝 进阶技巧：正则标志

    `--regex-<LANG>` 支持多种标志来精细控制匹配行为：

    *   `{icase}`：忽略大小写。
    *   `{exclusive}`：匹配到此行后，跳过其他规则，避免重复处理。
    *   `{placeholder}`：不生成标签，仅用于匹配和跳过某些行。
    *   `{scope=...}`：用于维护层级关系（如章节嵌套），但这属于较复杂的用法。

    自定义正则解析的核心是“**定义语言 → 映射文件 → 定义标签类型 → 编写正则**”这四个步骤。你可以先从简单的标签（如 `TODO`、`FIXME`）开始尝试，熟悉后再逐步增加复杂规则。

* ctags --list-languages

	ctags --version

