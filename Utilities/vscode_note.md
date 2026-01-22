# VSCode Note

## cache

* vscode 的设置中，如何只显示已修改的

    在 VS Code 中，可以通过以下几种方式只显示已修改的设置：

    方法一：使用设置编辑器

    * 打开设置（Ctrl+, 或 Cmd+,）

    * 在搜索框中输入 @modified

    * 这样只会显示你已经修改过的设置项

    方法二：使用配置文件

    * 打开用户设置文件：

        * 命令面板（Ctrl+Shift+P 或 Cmd+Shift+P）

        * 输入 "Preferences: Open Settings (JSON)"

    * 这会直接打开 settings.json 文件，里面只包含你自定义的设置

    方法三：设置界面筛选

    在设置界面顶部，点击右上角的“更多操作”（⋯ 图标），然后选择：

    * 显示修改的设置：只显示已更改的设置

    * 重置设置：可以查看所有已修改的设置

    方法四：使用命令面板

    * 打开命令面板

    * 搜索并执行 "Preferences: Open User Settings"

    * 默认就会显示所有用户自定义设置

    方法五：查看设置差异

    * 打开命令面板

    * 输入 "Preferences: Open Default Settings (JSON)"

    * 这会在一个编辑器中显示默认设置，另一个编辑器中显示你的自定义设置，便于对比

    小贴士

    * 设置标识：已修改的设置前面会有一个蓝色的竖线标记

    * 搜索语法：

        * @modified - 已修改的设置

        * @tag:usesOnlineService - 按标签筛选

        * window.title - 搜索特定设置

    最常用的方法是直接在设置搜索框输入 @modified，这样最简单快捷！

* git log --all 是什么意思？

    git log --all 是一个 Git 命令，用于显示所有分支的提交历史，而不仅仅是当前分支。

    主要功能：

    * 显示所有分支的提交 - 包括本地分支和远程跟踪分支

    * 展示完整的项目历史 - 而不仅仅是当前分支的线性历史

    使用场景：

    对比没有 --all 的情况：

    ```bash
    # 只显示当前分支的提交历史
    git log

    # 显示所有分支的提交历史
    git log --all
    ```

    常用组合：

    ```bash
    # 以图形化方式显示所有分支历史
    git log --all --oneline --graph

    # 显示所有分支的历史，包含统计信息
    git log --all --stat

    # 查看所有分支中某个文件的修改历史
    git log --all -- path/to/file
    ```

    示例输出：

    ```text
    *   a1b2c3d (main) Merge branch 'feature/login'
    |\  
    | * e4f5g6h (feature/login) Add login validation
    | * i7j8k9l Add login form
    |/  
    * m1n2o3p Initial commit
    ```

    与其他选项的对比：

    | 命令 | 作用 |
    | - | - |
    | git log | 仅当前分支的历史 |
    | git log --all | 所有分支的历史 |
    | git log --branches | 所有本地分支的历史 |
    | git log --remotes | 所有远程分支的历史 |

    简单来说，git log --all 让你看到项目的完整历史画卷，而不仅仅是当前所在分支的局部视图。这在查看合并历史、理解分支结构时特别有用。

* 简述 git log --decorate 的作用

    git log --decorate 的作用是在提交历史中显示引用信息，让分支、标签等指针更直观地展示。

    主要作用：

    1. 显示引用位置

        * 在每个提交旁显示它所在的分支、标签

        * 用不同颜色区分不同类型的引用

    2. 查看分支拓扑关系

        * 清楚看到哪些提交属于哪个分支

        * 了解分支的合并点和起点

    示例对比：

    没有 --decorate：

    ```text
    commit abc123
    Author: user
    Date:   ...
        Add feature
    ```

    使用 --decorate：

    ```text
    commit abc123 (HEAD -> main, origin/main, tag: v1.0)
    Author: user
    Date:   ...
        Add feature
    ```

    常用组合：

    ```bash
    # 基础用法
    git log --decorate

    # 简洁的单行显示 + 图形化 + 引用
    git log --oneline --graph --decorate

    # 所有分支的引用信息
    git log --all --decorate
    ```

    显示的内容：

    * HEAD - 当前检出的位置

    * 分支名 - 本地分支（如 main, feature/login）

    * 远程分支 - 远程跟踪分支（如 origin/main）

    * 标签 - 版本标签（如 tag: v1.0）

    现代 Git 的默认行为：

    在较新的 Git 版本中（2.13+），--decorate 通常是默认启用的，可以通过以下配置查看：

    ```bash
    git config --get log.decorate  # 查看当前设置
    ```

    如果想永久启用，可以设置：

    ```bash
    git config --global log.decorate auto
    ```

    实际价值：

    通过 --decorate，你可以一目了然地：

    * 知道当前在哪个分支（HEAD 指向）

    * 看到哪些提交已经推送到了远程

    * 识别重要的版本标签

    * 理解分支的合并和分离状态

    这个选项让 Git 历史不再是冷冰冰的提交哈希列表，而是变成了一个有上下文、有位置信息的项目演化地图。

* vscode github copilot插件如何使用

    好的，VS Code 中的 GitHub Copilot 插件是一个非常强大的 AI 编程助手，可以极大地提高编码效率。下面我将为您提供一个从安装到使用的完整指南。

    一、安装与激活

    1. 安装插件：

        * 打开 VS Code。

        * 进入扩展市场 (Ctrl+Shift+X 或 Cmd+Shift+X)。

        * 搜索 “GitHub Copilot”。

        * 点击 “Install” 进行安装。

    2. 激活订阅：

        * 你需要一个 GitHub Copilot 的订阅。学生和热门开源项目的维护者可以免费使用，否则需要按月或按年付费。

        * 安装后，VS Code 左下角会出现一个 Copilot 图标，或者会弹出提示引导你登录 GitHub 账户进行授权。

        * 按照提示在浏览器中完成 GitHub 账户的登录和授权即可。

    3. 检查状态：

        * 授权成功后，你可以在 VS Code 左下角看到 Copilot 图标，将鼠标悬停在上方会显示 “GitHub Copilot is enabled”。

    二、核心功能与使用方法

    GitHub Copilot 主要有三种使用方式：

    1. 代码自动补全

        这是最常用的功能。当你打字时，Copilot 会自动分析你的代码上下文，并给出灰色的代码建议。

        * 使用方法：

            * 正常敲代码，当看到灰色建议时：

                * 接受建议：按 Tab 键。

                * 接受下一个单词：按 Alt + ] (Windows/Linux) 或 Option + ] (Mac)。

                * 拒绝建议：继续打字，或者按 Esc 键。

                * 查看下一个建议：按 Alt + ] 或 Option + ]。

                * 查看上一个建议：按 Alt + [ 或 Option + [。

    2. 代码注释生成代码

        你可以通过编写详细的注释（自然语言）来让 Copilot 生成整段代码。

        * 使用方法：

            * 在代码文件中，先写一行注释，描述你想要实现的功能。

            * 按 Enter 换行，Copilot 就会给出代码建议。

        * 示例：

            * 输入注释：// 函数：计算斐波那契数列的第n项

            * 按 Enter 后，Copilot 可能会生成：

                ```javascript
                function fibonacci(n) {
                    if (n <= 1) return n;
                    return fibonacci(n - 1) + fibonacci(n - 2);
                }
                ```

    3. Copilot Chat（聊天功能）

        这是一个更强大的交互式功能，允许你像与同事一样与 AI 对话，进行代码解释、重构、调试、生成测试等操作。注意：此功能可能需要你额外安装 “GitHub Copilot Chat” 扩展。

        * 打开方式：

            * 快捷键：Ctrl + I (Windows/Linux) 或 Cmd + I (Mac)。

            * 侧边栏：点击 VS Code 左侧活动栏的 Copilot 图标。

            * 内联聊天：在代码编辑器中选中一段代码，右键选择 “Copilot” -> “Start Inline Chat”。

        * 常用场景：

            * 解释代码：选中一段复杂的代码，在 Chat 中输入 “解释这段代码”。

            * 重构代码：输入 “帮我重构这段代码，提高可读性”。

            * 生成测试：输入 “为这个函数生成单元测试”。

            * 修复错误：将错误信息粘贴到 Chat 中，问 “这个错误是什么意思？如何修复？”。

            * 寻找漏洞：输入 “检查这段代码是否有安全漏洞”。

    三、实用技巧与最佳实践

    * 提供清晰的上下文：Copilot 非常依赖上下文。确保你的文件命名、已有的代码和注释足够清晰，这样它才能给出更准确的建议。

    * 善用注释：把你想要的功能用详细的英文或中文注释写出来，越具体越好。例如，写 // 使用快速排序算法对数组进行升序排序 比 // 排序 效果要好得多。

    * 循序渐进：对于复杂功能，可以尝试分步骤写注释，让 Copilot 一步步生成代码。

    * 审查生成的代码：Copilot 不是万能的，它生成的代码可能不是最优的，甚至可能存在错误或安全漏洞。 一定要仔细检查和测试它提供的代码。

    * 使用 @workspace 指令：在 Copilot Chat 中，使用 @workspace 可以让 AI 了解你整个项目的工作区上下文，从而给出更相关的答案。例如：“@workspace 这个项目的主要功能是什么？”

    四、常用命令与快捷键

    你可以在命令面板 (Ctrl+Shift+P 或 Cmd+Shift+P) 中搜索以下命令：

    * GitHub Copilot: Toggle Copilot：全局开启或关闭 Copilot。

    * GitHub Copilot: Toggle Completions：在当前编辑器中开启或关闭自动补全。

    * GitHub Copilot: Open Completions Panel：打开一个面板，查看更多的代码建议（在自动补全不出现时有用）。

    五、故障排除

    * 没有代码建议：

        * 检查 VS Code 左下角 Copilot 图标是否正常（不是带叉号或叹号）。

        * 检查你的订阅是否有效。

        * 尝试在命令面板中执行 GitHub Copilot: Toggle Completions 确保补全功能是开启的。

        * 检查网络连接，因为 Copilot 需要联网。

    * 聊天功能不工作：

        * 确保你安装了 “GitHub Copilot Chat” 扩展。

    希望这份详细的指南能帮助你快速上手 GitHub Copilot，享受 AI 辅助编程带来的乐趣和效率提升！

* remote host 无法联网时，vscode 安装 remote ssh 插件的解决方案

    目前 vscode 版本在安装 remote ssh 插件时，会

    1. 在 remote host 上下载 cli 文件

        cli 文件本质是一个小型的 bootstrap 程序。

        此时可以在本地的 vscode 设置里，找到`Remote.SSH: Local Server Download`，将其改为`always`，表示在本地下载完 cli 文件后，通过 ssh 上传到 remote host

    2. remote host 运行 cli 组件的功能，开始下载 vscode server 文件

        vscode server 才是真正具有 remote ssh 功能的文件。

        此时可以在本地的 vscode 设置里，找到`Remote.SSH: Http Proxy`和`Remote.SSH: Https Proxy`，点击`Edit in settings.json`后，将其配置为

        ```json
        "remote.SSH.httpsProxy": "http://127.0.0.1:8823",
        "remote.SSH.httpProxy": "http://127.0.0.1:8823"
        ```

        然后在本地启动 v2ray `http-to-freedom.json`，监听 8823 端口。

        再在本地执行`ssh -NR 8823:127.0.0.1:8823 remote_host`打开远程隧道即可。

        这时 vscode 会使用这个代理下载 vscode server。

* 简述 vscode 中 Snippets，macros 的用法

    **VSCode Snippets（代码片段）**

    基本用法

    * 触发方式：输入前缀 + Tab

    * 预置片段：很多语言自带（如 for、if、clg 等）

    创建自定义Snippet

    * 打开命令面板：Ctrl+Shift+P

    * 搜索：Configure User Snippets

    * 选择语言 或 新建全局片段文件

    JSON格式示例

    ```json
    {
      "Print to console": {
        "prefix": "log",
        "body": [
          "console.log('$1', $1);",
          "$2"
        ],
        "description": "Log output to console"
      }
    }
    ```

    特殊语法

    * $1, $2：制表位位置

    * ${1:label}：带默认值的制表位

    * $TM_FILENAME：当前文件名

    * $BLOCK_COMMENT_START：语言相关变量

    范围限定

    ```json
    // 仅在特定文件类型生效
    {
      "[javascript]": {
        "My Snippet": { ... }
      }
    }
    ```

    **VSCode Macros（宏）**

    1. 内置宏录制

        VSCode没有官方宏录制功能，但可通过扩展实现

    2. 使用 macros 插件

        ```json
        // settings.json 配置
        {
          "macros": {
            "deleteToEnd": [
              "cursorEndSelect",
              "deleteRight"
            ],
            "duplicateLine": [
              "editor.action.copyLinesDownAction",
              "cursorUp"
            ]
          }
        }

        // 绑定快捷键
        {
          "key": "ctrl+shift+d",
          "command": "macros.duplicateLine"
        }
        ```

    3. 使用 multi-command 插件

        更强大的宏功能：

        ```json
        // settings.json
        {
          "multiCommand.commands": [
            {
              "command": "multiCommand.saveAndFormat",
              "sequence": [
                "workbench.action.files.save",
                "editor.action.formatDocument"
              ]
            },
            {
              "command": "multiCommand.surroundWithDiv",
              "interval": 100, // 延迟（毫秒）
              "sequence": [
                "editor.action.insertSnippet",
                { "snippet": "<div>$TM_SELECTED_TEXT</div>" }
              ]
            }
          ]
        }

        // keybindings.json
        {
          "key": "ctrl+shift+f",
          "command": "extension.multiCommand.saveAndFormat"
        }
        ```

    4. 宏录制扩展

        * macros-re：录制和回放操作

        * multi-command：手动定义复杂序列

    **宏的典型用途**

    * 重复操作自动化

        * 格式化 + 保存

        * 注释切换 + 折叠代码

    * 文本处理

        * 包围选中文本

        * 批量修改格式

    * 工作流优化

        * 打开终端 + 运行命令

        * 切换侧边栏视图

    **实用技巧对比**

    | 功能 | Snippets | Macros |
    | - | - | - |
    | 最佳用途 | 代码模板 | 操作序列 |
    | 复杂度 | 简单文本替换 | 复杂动作流 |
    | 触发方式 | 前缀+Tab | 快捷键/命令 |
    | 动态性 | 有限变量 | 可包含条件逻辑 |

    **结合使用示例**

    创建一个“新建React组件”的宏：

    * Snippet 生成组件代码

    * Macro 执行：

        * 插入Snippet

        * 创建新文件

        * 保存到指定目录

* vscode 中与 vim 相关的插件

    1. VSCode Vim插件（已支持部分脚本功能）

        VSCodeVim/Vim：最流行的Vim模拟插件

        * 支持部分Vim脚本功能，如：

            * .vimrc 配置文件支持

            * 基本的映射（mapping）和缩写（abbreviations）

            * 一些自定义命令

    2. 专门扩展插件

        * VimL

            * 提供对Vim脚本语言的更多支持

            * 支持更复杂的Vim脚本功能

        * Neo Vim集成

            * vscode-neovim：使用真实的Neovim实例

            * 完全支持Vim脚本，因为它在后台运行真正的Neovim

            * 配置方式与Vim/Neovim相同

    3. 替代方案

        * VSCode原生扩展

            * 任务（Tasks）：自动化重复操作

            * 代码片段（Snippets）：自定义代码模板

            * 键盘快捷键（Keybindings）：JSON配置文件

        * 宏录制插件

            * macros：录制和回放操作序列

            * multi-command：创建复杂的命令序列

    4. 配置示例

        ```json
        // keybindings.json 示例
        {
          "key": "ctrl+shift+b",
          "command": "extension.multiCommand.execute",
          "args": {
            "sequence": [
              "editor.action.selectAll",
              "editor.action.copy",
              "editor.action.deleteLines"
            ]
          }
        }
        ```

* vscode 中，如果 tab 修改原代码的对齐格式，可以把

    `Editor: Use Tab Stops`

    这个配置取消。

    修改 indent / tab 相关的配置后，可能需要 reload window 才能生效。

* 如何让 vscode 显示所有隐藏的 unicode 字符

    插件：`Render Special Characters`

    未测试：

    * `Highlight Bad Chars`

* vscode 插件

    Code Runner - 运行代码片段

    GitLens - 增强代码提示功能

    安装 Settings Cycler 或 Settings Watcher 扩展来管理设置变更

    扩展：Compareit 来比较文件

* vscode 中查看哪些配置被改过

    * 使用设置界面（推荐）

        打开设置：Ctrl+,（Windows/Linux）或 Cmd+,（Mac）

        在搜索框中输入 @modified

    * 使用命令面板

        按 Ctrl+Shift+P

        输入 Preferences: Open Settings (JSON)

    * 文件

        用户设置: %APPDATA%\Code\User\settings.json（Windows）或 ~/.config/Code/User/settings.json（Linux/Mac）

        工作区设置: .vscode/settings.json（项目根目录）

* vscode 中 debug 时自动加载环境变量

    * 指定 envfile

        `"envFile": "${workspaceFolder}/.env"`

        注：

        1. `.env`是默认情况下就支持的吧？写`envFile`的时机是给`.env`換名字。

    * 使用 preLaunchTask

        适用于在 shell 脚本中配置环境变量。

        1. 创建 task (tasks.json)：

            ```json
            {
                "version": "2.0.0",
                "tasks": [
                    {
                        "label": "load-env",
                        "type": "shell",
                        "command": "source ${workspaceFolder}/env.sh && env > ${workspaceFolder}/.tmp.env",
                        "isBackground": false
                    }
                ]
            }
            ```

        2. 修改 launch.json：

            ```json
            {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Debug",
                        "type": "python",
                        "request": "launch",
                        "program": "${file}",
                        "preLaunchTask": "load-env",
                        "envFile": "${workspaceFolder}/.tmp.env"
                    }
                ]
            }
            ```

* vscode 快捷键

    * `Ctrl + Shift + \`：跳转到匹配的括号/列表项

    * `Ctrl + Shift + [`：折叠当前内容

    * `Ctrl` + `Home`：跳转到文件开头

* vscode 插件 CJK Word Handler

    按 Ctrl+Left/Right 移动光标时，能正确处理中英文混合的情况（例如“VS Code是一个编辑器”）

* vscode 插件 Cursor Align

    按光标对齐多行文本，功能有点像 latex 中的`&`。

    example:

    ```
    name = "John"
    age = 25
    city = "New York"
    ```

    使用 alt 键选中三个`=`前面的位置，然后按 alt + A，即可对齐，效果如下：

    ```
    name = "John"
    age  = 25
    city = "New York"
    ```

    主要功能：

    1. 对齐多行光标位置的文本 —— 在多行同时编辑时，自动将每行光标所在的列对齐到同一位置。

    2. 按特定字符对齐文本 —— 可以按照 =、:、# 等符号对齐选中的多行文本。

    3. 快速格式化选中的代码或文本 —— 无需手动添加空格，一键对齐，提升可读性。

    上面展示的是功能 1。功能 2 和 3 待探索。

* vscode 的 ctrl + f 搜索功能有正则表达式选项

* vscode 增加路径提示功能

    安装插件`Path Intellisense`，输入一半路径后，使用 Ctrl + Space 喚出路径自动补全。

* vscode 中，可以在工程目录中创建`.env`文件，静态地配置环境变量

    比如 python 工程，如果想临时添加一些 module 搜索目录，在代码里如果写成

    ```py
    import sys
    sys.path.append('xxx')
    ```

    那么这段代码是动态执行的，无法用于 vscode 的静态代码分析。

    我们可以在`.env`文件中添加：

    ```conf
    PYTHONPATH=xxx
    ```

    添加搜索路径，此时再在代码里

    ```py
    import my_module
    ```

    便可以直接 ctrl + 鼠标左键点进去了。

    注意，这个方法只能用于 vscode 的静态代码分析和调试（F5 运行）。
    
    如果要在实际运行中使环境变量生效，必须按照 bash 的规则执行`export PYTHONPATH=xxx`。仅仅`source ./.env`是不行的。

* vscode 里，可以使用 ctrl + up / down 实现向上／下滚动一行，不改变光标位置

* c++ 中的 string & 在 vscode 中，debug 断点模式下，鼠标悬停不显示内容

    下面是实测结果：

    ```cpp
    #include <string>
    #include <unordered_map>
    using namespace std;

    int main() {
        string str = "hello, world";  // 显示
        string &str_2 = str;  // 不显示
        const string &str_3 = str;  // 不显示
        const string &str_4 = "hello, world";  // 不显示

        unordered_map<string, string> umap {
            {"hello", "world"},
            {"nihao", "zaijian"}
        };
        string &str_val = umap["hello"];  // 不显示
        const string &con_str_val = umap["nihao"];  // 不显示
        return 0;
    }
    ```

* vscode 中 gdb pretty print

    ```json
    "configurations": [{
        "name": "(gdb) Launch",
        // ...
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
            },
            // ...
        ]
    }]
    ```

    必须设置`-enable-pretty-printing`这个才可以直接显示 string 的内容。否则显示的都是原始的指针。

* vscode sync scroll

    vscode 中同时滚动左右两个分栏

    extension name: `Sync Scroll`

    安装完后，`Ctrl` + `Shift` + `P`打开命令栏，搜索 sync scroll，会看到一个 change sync scroll mode，选择之后，会有三个选项:

    * normal

        保持左右两栏行数相同。

    * offset

        保持左右两栏滚动相同的距离

    * off

    在 vscode 右下角也会有当前 sync scroll 模式的状态。

* vscode 快捷键

    * 移动到下一个词的开头	Ctrl + →

    * 移动到上一个词的开头	Ctrl + ←

    * 选择到下一个词的开头	Ctrl + Shift + →

    * 选择到上一个词的开头	Ctrl + Shift + ←

    * 删除前一个词	Ctrl + Backspace

    * 删除后一个词	Ctrl + Delete

    * 打开设置  Ctrl + ,

* vscode 在`launch.json`中没有设置`cwd`时，程序中的`./`表示用户目录。比如`/share_data/users/hlc`

* 关于 vscode 中 gdb 调试 c++ 时，无法鼠标悬停显示`const string &str`的调研

    1. ds 的输出为 GDB 显示的是 std::string 在 libstdc++（GCC 的标准库实现）中的内部结构，而非直接显示字符串内容。这是因为 GDB 默认以“结构体/类成员”的形式显示对象，而没有自动调用 std::string 的字符串解码逻辑。

    2. 只有 local 变量窗口和 watch 变量窗口可以正确显示`const string &str`的内容，鼠标悬停无法直接显示。试了下默认的 lldb，比 gdb 更差，鼠标悬停时根本不解析`const string&`，只解析一些基本的 C 语言的数据结构。

        重新看了下，只有`const string &str = xxx;`这一行里，`str`无法正常显示，在其他行里`str`在鼠标悬停时可以正常显示。

* vscode 中，使用`-exec p`打印字符串数组，会把`\000`也打印出来

    ```cpp
    #include <stdio.h>

    int main()
    {
        char str[16] = "hello";
        printf("%s\n", str);
        return 0;
    }
    ```

    在`return 0;`前下断点，在 vscode debug console 中调用`-exec p str`，输出如下：

    ```
    $1 = "hello\000\000\000\000\000\000\000\000\000\000"
    ```

    而正常的 terminal 中，`printf()`的输出如下：

    ```
    hello
    ```

* vscode 可以使用 ctrl + shift + B 编译，使用 F5 启动 debug。

* `\bold`无法被 vscode 和 markdown preview 渲染插件识别，可以使用`\mathbf`来指定正粗体。

* vscode 中，取消了 tab stop 后，还是会有 tab 缩进 2 个空格的现象，这时候还需要取消 Detect Indentation

* 如果 vscode 中在编辑 makefile 时，tab 键总是插入 4 个空格而不是 tab，可以在 vscode setting 里把 Detect indentation 选项关了再打开一次就好了

* vscode 的 pipe 模式

    vscode 可以在`launch.json`中设置 pipe 模式，通过 ssh 在 remote host 上调用 gdb，再把 gdb 的输入输出重定向到 local host，从而不需要在 remote host 上安装 vscode，也不需要安装任何 vscode 插件，即可远程调试程序。

    配置方法：

    1. 在 remote host 上新建一个工程文件夹`mkdir -p /home/hlc/Documents/Projects/vscode_test`

        创建程序文件`main.c`:

        ```c
        #include <stdio.h>

        int main()
        {
            printf("hello world\n");
            return 0;
        }
        ```

        编译：`gcc -g main.c -o main`

    2. remote host 上还需要安装 ssh server 和 gdb，此时即可满足最小调试要求。

    3. 在 local host 上创建`launch.json`：

        ```json
        {
            // Use IntelliSense to learn about possible attributes.
            // Hover to view descriptions of existing attributes.
            // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "(gdb) Pipe Launch",
                    "type": "cppdbg",
                    "request": "launch",
                    "program": "/remote_host/path/to/main",
                    "args": [],
                    "stopAtEntry": false,
                    "cwd": "${fileDirname}",
                    "environment": [],
                    "externalConsole": false,
                    "pipeTransport": {
                        "debuggerPath": "/usr/bin/gdb",
                        "pipeProgram": "/usr/bin/ssh",
                        "pipeArgs": [
                            // "-pw",
                            // "password",
                            "hlc@10.0.2.4"
                        ],
                        "pipeCwd": "/usr/bin"  // this line
                    },
                    "MIMode": "gdb",
                    "setupCommands": [
                        {
                            "description": "Enable pretty-printing for gdb",
                            "text": "-enable-pretty-printing",
                            "ignoreFailures": true
                        },
                        {
                            "description": "Set Disassembly Flavor to Intel",
                            "text": "-gdb-set disassembly-flavor intel",
                            "ignoreFailures": true
                        }
                    ]
                }
            ]
        }
        ```

        标注为`// this line`的那一行可以为空字符串，不影响。

        `-pw password`实测不支持。不清楚为什么 vscode 官网写了这个用法。

    4. 在 local host 上创建一份和 remote host 上一模一样的`main.c`文件，并打上断点

        按 F5 运行，此时会显示要求安装一个交互式输入 ssh 密码的小工具`ssh-askpass`。执行`sudo apt install ssh-askpass`。

        按照提示输入 ssh fingerprint 和密码后，即可正常 hit 断点。如果嫌麻烦可以在 local host 上运行`ssh-copy-id <user>@<remote_host_ip>`把 local host 的 ssh public key 添加到 remote host。

        debug console 的输出如下：

        ```
        =thread-group-added,id="i1"
        GNU gdb (Ubuntu 15.0.50.20240403-0ubuntu1) 15.0.50.20240403-git
        Copyright (C) 2024 Free Software Foundation, Inc.
        License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
        This is free software: you are free to change and redistribute it.
        There is NO WARRANTY, to the extent permitted by law.
        Type "show copying" and "show warranty" for details.
        This GDB was configured as "x86_64-linux-gnu".
        Type "show configuration" for configuration details.
        For bug reporting instructions, please see:
        <https://www.gnu.org/software/gdb/bugs/>.
        Find the GDB manual and other documentation resources online at:
            <http://www.gnu.org/software/gdb/documentation/>.

        For help, type "help".
        Type "apropos word" to search for commands related to "word".
        Warning: Debuggee TargetArchitecture not detected, assuming x86_64.
        =cmd-param-changed,param="pagination",value="off"
        [Thread debugging using libthread_db enabled]
        Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

        Breakpoint 1, main () at main.c:5
        5	    printf("hello from ubuntu 2404\n");
        Loaded '/lib64/ld-linux-x86-64.so.2'. Symbols loaded.
        Loaded '/lib/x86_64-linux-gnu/libc.so.6'. Symbols loaded.
        Warning: Source file '/home/hlc/Documents/Projects/vscode_test/main.c' is newer than module file '/home/hlc/Documents/Projects/vscode_test/main'.
        Execute debugger commands using "-exec <command>", for example "-exec info registers" will list registers in use (when GDB is the debugger)
        ```

    说明：

    1. local host 的`main.c`文件与 remote host 不一致不影响远程调试，因为 remote gdb 在打断点时只关心文件名和第几行。

        如下图所示：

        <div style='text-align:center'>
        <img src='../../Reference_resources/ref_30/pic_1.png'>
        </div>

    refs:

    1. Pipe transport

        <https://code.visualstudio.com/docs/cpp/pipe-transport>

    2. VS Code Remote Development

        <https://code.visualstudio.com/docs/remote/remote-overview>

    3. How to debug with VSCode and pipe command

        <https://stackoverflow.com/questions/54052909/how-to-debug-with-vscode-and-pipe-command>

    4. How to specify password in ssh command

        <https://superuser.com/questions/1605215/how-to-specify-password-in-ssh-command>

* vscode `Ctrl` + `Shift` + `O` 可以跳转到 symbol，也可以使用 `Ctrl` + `T` 跳转。

    `Ctrl` + `Shift` + `O`只搜索当前文件，`ctrl` + `t`会搜索 work space 下的所有文件。

    ref: <https://code.visualstudio.com/docs/editor/editingevolved>

* vscode 中，取消了 tab stop 后，还是会有 tab 缩进 2 个空格的现象，这时候还需要取消 Detect Indentation

* 防止 vscode 里 tab 键做太多段落自动对齐的工作：

    取消 use tab stops

* vscode 关闭自动补全引号：

    auto closing quotes 设置成 never

* vscode 关闭自动补全括号：

    auto closing brackets 设置成 never

* 使用 vscode 调试 sudo 程序

    核心是需要 gdb 由 sudo 启动。

    可以在`launch.json`里加一行：

    `"miDebuggerPath": "/home/hlc/.local/bin/sudo_gdb.sh"`

    `sudo_gdb.sh`里只要写一行：

    ```bash
    #!/bin/bash
    sudo gdb "$@"
    ```

    然后`sudo chmod +x sudo_gdb.sh`。

    接着在 vscode 的 intergrated terminal 里输入`sudo echo 1`，正常输入密码。此时这个 terminal 里，root 权限会持续开启一段时间，使用`sudo`运行其他程序不需要再输入密码。

    这个时候就可以在 vscode 里运行 F5 调试程序了。

* vscode attach 不需要输入 root 的方法

    ```bash
    echo 0 | sudo tee /proc/sys/kernel/ya/ptrace_scope
    ```

    sudo 和 tee 连用，就可以让 echo 写字符到需要 root 权限的文件里？

* cuda vscode debug

    ```cu
    #include <stdio.h>
    #include <cuda_runtime.h>

    __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) 
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < numElements)
        {
            C[i] = A[i] + B[i] + 0.0f;
        }
    }

    int main()
    {
        int numElements = 1024;
        size_t size = numElements * sizeof(float);
        float *h_A = (float *) malloc(size);
        float *h_B = (float *) malloc(size);
        float *h_C = (float *) malloc(size);
        for (int i = 0; i < numElements; ++i)
        {
            h_A[i] = rand() / (float) RAND_MAX;
            h_B[i] = rand() / (float) RAND_MAX;
        }

        float *d_A = NULL;
        cudaMalloc((void**) &d_A, size);
        float *d_B = NULL;
        cudaMalloc((void**) &d_B, size);
        float *d_C = NULL;
        cudaMalloc((void**) &d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i)
        {
            printf("%.2f + %.2f = %.2f\n", h_A[i], h_B[i], h_C[i]);
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);

        return 0;
    }
    ```

    compile:

    ```bash
    nvcc -g main.cu -o main
    ```

    run:

    ```
    ./main
    ```

    output:

    ```
    0.84 + 0.39 = 1.23
    0.78 + 0.80 = 1.58
    0.91 + 0.20 = 1.11
    0.34 + 0.77 = 1.10
    ```

    vscode debug:

    1. install vscode extension: `Nsight Visual Studio Code Edition`

    2. add debug type: cuda-gdb

    3. press F5 start debugging

    也可以直接使用`gdb <exe_file>`去调试。

* vscode 可以使用`ctrl + p`搜索文件名

1. invoke call stack window: <https://stackoverflow.com/questions/1607065/how-to-show-call-stack-immediates-and-other-hidden-windows-in-visual-studio>

1. <https://learn.microsoft.com/en-us/visualstudio/debugger/how-to-use-the-call-stack-window?view=vs-2022>

* vscode 使用 lldb 调试的技巧：<https://www.codenong.com/cs105826090/>

    感觉有用的内容不多，有时间了看看。

* ssh 多级跳转

    ```ssh
    Host jump1
    HostName ip of jump1
    Port port id
    User user name
    IdentityFile key_path

    Host jump2
    HostName ip of jump2
    Port port id
    User user name
    IdentityFile key_path
    # command line,setting jump1
    ProxyCommand ssh -q -W %h:%p jump1


    # Target machine with private IP addressHost target
    Host target
    HostName ip_of target
    Port port id
    User user name
    IdentityFile key_pathProxyCommand ssh -q -W %h:%p jump2
    ```

    比如一台 server 要经过 jump1 连到 jump2，再通过 jump2 连到 target，那么就可以用上面的配置。

    Ref: <https://www.doc.ic.ac.uk/~nuric/coding/how-to-setup-vs-code-remote-ssh-with-a-jump-host.html>

    <https://support.cs.wwu.edu/home/survival_guide/tools/VSCode_Jump.html>

    也可以直接用`proxyjump`，example:

    ```s
    Host 10.211.10.67
    HostName 10.211.10.67
    ProxyJump liucheng@172.18.25.248
    User test

    Host 192.168.122.75
    HostName 192.168.122.75
    ProxyJump liucheng@172.18.25.248
    User ubuntu
    ```

* vscode debug with a root privilege

    一种方式是创建一个`gdb`的脚本：

    ```bash
    #!/bin/bash
    pkexec /usr/bin/gdb "$@"
    ```

    然后在 vscode 中将`miDebuggerPath`设置成 gdb 脚本的 path 就好了。

    另一种方法是设置权限：`user_name ALL=(ALL) NOPASSWD:/usr/bin/gdb`，可以直接让 gdb 获得 root 权限。

    Ref: <https://stackoverflow.com/questions/40033311/how-to-debug-programs-with-sudo-in-vscode>

## note

