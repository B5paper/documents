# markdown note

* <https://stackedit.io/app#>

    一个在线渲染 markdown 的网站，支持数学公式。

    有临时需求可以试一试这个网站。

## cache

* markdown 的表格中，如何输入竖线 |？

    在 Markdown 表格中，如果要输入竖线 | 符号，你需要使用 HTML 实体 或 Markdown 转义，因为 | 是表格列的分隔符，直接输入会被解析为列的分界。

    方法 1：使用 HTML 实体

    用 &#124; 或 &verbar; 代替 |：

    ```markdown
    | 符号 | 说明         |
    |------|--------------|
    | \|   | 竖线（管道符） |
    | &#124; | 使用HTML实体 |
    ```

    效果：

    | 符号 | 说明 |
    | - | - |
    | | | 竖线（管道符） |
    | | | 使用HTML实体 |

    方法 2：使用反斜杠转义（在某些解析器中有效）

    在 | 前加反斜杠：\|

    ```markdown
    | 列1      | 列2        |
    |----------|------------|
    | 竖线：\| | 另一个竖线 |
    ```

    注意：并非所有 Markdown 解析器都支持在表格内转义 |，例如 GitHub Flavored Markdown 就不支持。因此更推荐使用 HTML 实体 方式。

    方法 3：使用代码块

    如果内容较多，可以在表格单元格中使用行内代码（用反引号包裹），这样 | 就不会被解析为分隔符：

    ```markdown
    | 示例 | 说明 |
    |------|------|
    | `a | b` | 用代码块包裹，竖线原样显示 |
    ```

    效果：

    | 示例 | 说明 |
    | - | - |
    | a | b | 用代码块包裹，竖线原样显示 |

    总结

    * 最通用可靠：使用 HTML 实体 &#124;

    * 简单情况：可尝试用反斜杠转义 \|（先确认你的解析器是否支持）

    * 代码相关：使用行内代码 `a | b`

    根据你的 Markdown 渲染环境选择合适的方法即可。

* 常见的 markdown 渲染工具

    📝 编辑器和笔记软件

        Typora - 实时预览的经典编辑器

        Obsidian - 本地知识库，支持丰富插件

        VS Code + Markdown 插件 - 开发者的选择

        Notion - 在线协作笔记

        语雀 - 国内优秀的文档工具

    🌐 网页渲染

        marked.js - JavaScript 库，轻量快速

        Showdown - 另一款流行的 JS 转换库

        Marked.it - 在线即时预览工具

    📦 静态网站生成器

        Hexo - 基于 Node.js 的博客框架

        Hugo - Go 语言编写，速度极快

        Jekyll - GitHub Pages 原生支持

        VuePress / VitePress - Vue 生态的文档工具

    🔧 开发工具

        Pandoc - 文档格式转换的“瑞士军刀”

        Markdown Preview Enhanced - VS Code 的强大插件

        GitHub/GitLab - 代码仓库自带渲染

    📱 移动端

        MWeb - iOS/macOS 的优秀编辑器

        纯纯写作 - Android 上的专注写作工具

    💡 选择建议

        日常写作：Typora 或 Obsidian

        技术文档：VS Code 或 VitePress

        博客写作：Hexo 或 Hugo

        团队协作：Notion 或 语雀

## note
