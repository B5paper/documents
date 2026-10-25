# neo vim note

## cache

* 详细讲讲 telescope.nvim

    `telescope.nvim` 是 Neovim 生态中事实上的标准模糊查找器。它用 Lua 编写，提供了一个可高度扩展的交互式界面，把“查找文件、搜索文本、切换缓冲区、浏览 Git 记录、调用 LSP 功能”等操作统一到相似的 UI 中，极大提升了在项目里导航和检索的效率。

    ### 核心架构：理解它的四个部件

    Telescope 的设计很清晰，一个交互界面由四个协作的部分组成，理解它们有助于你排查问题和进行深度定制：

    *   **Finder（查找器）**：决定“候选池里有什么”。例如，`find_files` 的 Finder 会调用 `fd` 或 `rg --files` 列出所有文件；`live_grep` 的 Finder 则会在你每次输入时重新运行 `rg` 来搜索文件内容。
    *   **Sorter（排序器）**：负责“怎么排、怎么筛”。默认是 Lua 实现的模糊排序，如果你装了 `fzf-native` 扩展，排序会交由 C 语言实现，速度会快几个数量级。它决定了你输入“tstr”时，`t_string.c` 能排到最前面的匹配逻辑。
    *   **Picker（选择器）**：负责“交互与呈现”。它把 Finder 和 Sorter 组合成一个浮窗，并管理你的按键操作（如 `<CR>` 打开、`<C-v>` 分屏、`<Tab>` 多选）。
    *   **Previewer（预览器）**：右侧的实时预览窗口。默认使用 `vim_buffer` 预览器，能利用 Tree-sitter 或正则进行语法高亮，让你不打开文件就能确认内容。

    ### 常用内置 Picker

    Telescope 内置了大量开箱即用的 Picker，覆盖日常高频操作。你可以通过 `:Telescope` 后跟 Tab 补全来查看全部列表，以下是最常用的几个：

    | 功能类别 | 命令 | 用途 |
    | :--- | :--- | :--- |
    | **文件查找** | `find_files` | 按文件名模糊查找项目内文件 |
    | | `oldfiles` | 查找最近打开过的文件 |
    | **文本搜索** | `live_grep` | 实时全文搜索（依赖 `rg`） |
    | | `grep_string` | 直接搜索光标下的单词 |
    | **缓冲区** | `buffers` | 在已打开的缓冲区之间切换 |
    | **LSP 集成** | `lsp_references` | 查找光标下符号的所有引用 |
    | | `lsp_document_symbols` | 列出当前文件的符号（函数、变量等） |
    | | `diagnostics` | 列出所有诊断信息（错误、警告） |
    | **Git 集成** | `git_commits` | 浏览提交记录，带 diff 预览 |
    | | `git_status` | 查看当前仓库的变更文件 |
    | | `git_branches` | 管理分支（切换、创建、删除等） |

    此外，还有 `treesitter`（基于 Tree-sitter 的符号列表）、`keymaps`（查看键位映射）、`help_tags`（搜索帮助文档）等实用 Picker。

    ### 默认快捷键与操作逻辑

    Telescope 的默认映射设计得比较符合直觉，进入浮窗后（Insert 模式）常用操作如下：

    *   **导航**：`<C-n>` / `<C-p>`（或 `<Down>` / `<Up>`）上下移动选择。
    *   **打开**：`<CR>` 在当前窗口打开；`<C-x>` 水平分屏；`<C-v>` 垂直分屏；`<C-t>` 新标签页打开。
    *   **预览滚动**：`<C-u>` / `<C-d>` 在预览窗内上下滚动。
    *   **多选与批量操作**：`<Tab>` 选中当前项并移到下一项；`<C-q>` 把所有未被过滤的项发送到 quickfix 列表，方便批量跳转或重构。
    *   **求助**：在 Insert 模式下按 `<C-/>`（或 Normal 模式下按 `?`），会弹出当前 Picker 可用的所有操作映射，这是发现高级功能的好方法。

    ### 安装与基础配置

    Telescope 依赖 `plenary.nvim`，强烈建议同时安装 `telescope-fzf-native.nvim` 来获得更好的排序性能。

    **依赖清单**：
    *   `nvim-lua/plenary.nvim`（必须）
    *   `nvim-telescope/telescope-fzf-native.nvim`（强烈推荐，需执行 `make` 编译）
    *   `nvim-tree/nvim-web-devicons`（可选，提供文件图标）

    **配置示例**（Lua）：
    ```lua
    local telescope = require('telescope')
    local actions = require('telescope.actions')

    telescope.setup({
      defaults = {
        -- 你可以在这里修改全局默认映射
        mappings = {
          i = {
            ["<C-j>"] = actions.move_selection_next,
            ["<C-k>"] = actions.move_selection_previous,
            ["<C-q>"] = actions.send_selected_to_qflist + actions.open_qflist,
          },
        },
        -- 智能路径显示：长路径会截断中间部分，保留尾部文件名
        path_display = { "smart" },
      },
    })

    -- 加载 fzf-native 扩展（如果安装了的话）
    pcall(telescope.load_extension, 'fzf')
    ```

    **建议的键位映射**（按你的习惯调整）：
    ```lua
    vim.keymap.set('n', '<leader>ff', '<cmd>Telescope find_files<cr>', { desc = '查找文件' })
    vim.keymap.set('n', '<leader>fg', '<cmd>Telescope live_grep<cr>', { desc = '全文搜索' })
    vim.keymap.set('n', '<leader>fb', '<cmd>Telescope buffers<cr>', { desc = '切换缓冲区' })
    vim.keymap.set('n', '<leader>fr', '<cmd>Telescope oldfiles<cr>', { desc = '最近文件' })
    vim.keymap.set('n', '<leader>fd', '<cmd>Telescope diagnostics<cr>', { desc = '诊断信息' })
    ```

    ### 实用技巧与注意点

    *   **模糊匹配心智模型**：Telescope 的模糊匹配不要求字符连续。输入 `tstr` 能匹配 `t_string.c`，因为字符 `t`、`s`、`t`、`r` 是按顺序出现的。利用这个特性，打几个“特征字母”就能快速定位。
    *   **善用 `<C-/>`**：这是最被低估的键。每当你对一个 Picker 能做什么感到好奇时，按一下它，所有可用操作和对应键位都会列出来。
    *   **`live_grep` 与 `find_files` 的分工**：前者搜**内容**，后者搜**文件名**。找“哪里调用了 `parse()`”用 `live_grep`，找“叫 parser 的文件”用 `find_files`，不要混用。
    *   **性能问题定位**：如果 Telescope 响应变慢，首先检查 `fzf-native` 是否正确加载（`:checkhealth telescope`）。如果是在超大项目或超大文件上预览卡顿，可能是 Previewer 在读文件，可以尝试关闭预览或调整 `preview` 配置。

* 详细讲讲  neo vim 中的 nvim-tree

    `nvim-tree` 是 Neovim 生态中最主流的文件树插件之一，用 Lua 编写，用来替代 Neovim 自带的 netrw。它提供了一个侧边栏式的树形文件浏览器，集成了文件操作、Git 状态显示和 LSP 诊断等功能。

    ### 核心功能概览

    `nvim-tree` 的主要能力包括：

    - **树形文件浏览**：带图标的层级目录展示，可自定义宽度和位置（左/右）。
    - **完整的文件操作**：在树中直接创建、删除、重命名、复制、剪切、粘贴文件或文件夹。
    - **Git 集成**：在文件旁显示 Git 状态标记（如未跟踪、已修改、已暂存）。
    - **LSP 诊断集成**：在树中直接显示文件的错误、警告图标。
    - **过滤与搜索**：可切换显示隐藏文件、被 gitignore 忽略的文件，支持实时过滤。
    - **书签功能**：可标记常用文件并快速过滤查看。

    ### 安装与基础配置

    **前提**：Neovim 版本需 ≥ 0.10（较新版本的要求；旧版本可尝试兼容标签，但不再受支持）。可选安装 `nvim-web-devicons` 来显示文件图标，并需要 Nerd Font 字体配合。

    **关键一步：禁用 netrw**。nvim-tree 官方强烈建议在配置最开头禁用 Neovim 自带的 netrw，避免冲突：

    ```lua
    vim.g.loaded_netrw = 1
    vim.g.loaded_netrwPlugin = 1
    vim.opt.termguicolors = true -- 可选，启用真彩色让图标和配色更正常
    ```

    然后调用 setup 进行初始化。以下是一个常用配置示例：

    ```lua
    require("nvim-tree").setup({
      view = {
        width = 35,           -- 侧边栏宽度
        side = "left",        -- 显示在左侧
      },
      renderer = {
        indent_markers = { enable = true }, -- 缩进参考线
        group_empty = true,   -- 合并空目录
      },
      filters = {
        dotfiles = false,     -- 是否隐藏点文件，按需调整
        custom = { ".DS_Store" }, -- 额外过滤的文件
      },
      git = {
        ignore = false,       -- 是否隐藏被 gitignore 的文件
      },
    })
    ```

    ### 常用命令与快捷键

    **全局命令**（在普通 Neovim 模式下使用）：

    | 命令 | 作用 |
    |---|---|
    | `:NvimTreeToggle` | 打开/关闭文件树 |
    | `:NvimTreeFindFile` | 在树中定位当前正在编辑的文件 |
    | `:NvimTreeCollapse` | 折叠所有目录 |
    | `:NvimTreeRefresh` | 刷新文件树 |

    **文件树内部快捷键**（焦点在 nvim-tree 窗口时使用）：

    | 按键 | 操作 |
    |---|---|
    | `o` 或 `<CR>` | 打开文件 / 展开或折叠目录 |
    | `a` | 创建新文件（名称以 `/` 结尾则创建目录） |
    | `d` | 删除 |
    | `r` | 重命名 |
    | `c` / `x` / `p` | 复制 / 剪切 / 粘贴 |
    | `y` / `Y` | 复制文件名 / 相对路径 |
    | `H` | 切换显示隐藏文件 |
    | `I` | 切换显示 gitignore 文件 |
    | `W` | 全部折叠 |
    | `g?` | 显示所有可用快捷键帮助 |
    | `q` | 关闭文件树窗口 |
    | `<C-v>` / `<C-x>` | 在垂直 / 水平分屏中打开文件 |

    ### 使用建议

    初次使用时，建议先执行 `require("nvim-tree").setup()`（空配置，使用默认值），进入树后按 `g?` 查看完整的默认快捷键列表，熟悉后再按需在 `setup` 中调整宽度、过滤规则等选项。如果使用 `lazy.nvim` 等插件管理器，确保禁用 netrw 的代码写在加载 nvim-tree **之前**。

## topics
