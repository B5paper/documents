## cache


* 详细讲讲 python 中 jinja2 的作用和用法

    **Jinja2** 是 Python 中最流行、功能最强大的**模板引擎**（Template Engine）。它的核心作用是**将“文本模板”与“数据”动态结合，生成最终的文本文件**。

    虽然它最常用于 Web 开发（如 Flask、FastAPI）中生成 HTML 页面，但它本质上是一个文本生成工具，因此也可以用来动态生成配置文件（Nginx、Docker）、Markdown、甚至自动化代码。

    ---

    一、 Jinja2 的核心作用

    在没有模板引擎之前，如果你想用 Python 生成一个动态 HTML 页面，你可能得这么写：

    ```python
    html = "<html><body><h1>" + user_name + "</h1></body></html>"

    ```

    这种代码不仅难看、难维护，而且极易引发安全漏洞（如 XSS 攻击）。

    **Jinja2 的到来解决了这些痛点：**

    1. **关注点分离（SoC）：** 后端负责处理业务逻辑和数据，Jinja2 模板负责页面的结构和展现。
    2. **提高代码复用性：** 支持模板继承和包含，避免到处复制粘贴 HTML。
    3. **安全性：** 默认开启 HTML 自动转义，有效防止 XSS（跨站脚本）攻击。

    ---

    二、 Jinja2 的核心语法

    Jinja2 的语法非常直观，主要由以下三种“标签”组成：

    1. 变量占位符：`{{ ... }}`

        用于在模板中插入 Python 传过来的变量或表达式的结果。

        * **基本变量：** `{{ username }}`
        * **访问字典或对象属性：** `{{ user.name }}` 或 `{{ user['name'] }}`
        * **调用方法：** `{{ username.upper() }}`

    2. 控制流/逻辑语句：`{% ... %}`

        用于编写条件判断、循环等逻辑控制。

        * **条件判断 (If-Else)：**
        ```html
        {% if user.is_logged_in %}
            <p>欢迎回来，{{ user.name }}！</p>
        {% else %}
            <a href="/login">请登录</a>
        {% endif %}

        ```


        * **循环 (For-Loop)：**
        在循环内部，Jinja2 还提供了一个非常有用的 `loop` 对象（例如 `loop.index` 获取当前索引）。
        ```html
        <ul>
        {% for item in item_list %}
            <li>{{ loop.index }} - {{ item }}</li>
        {% endfor %}
        </ul>

        ```

    3. 注释：`{# ... #}`

        用于在模板中写注释，这些注释**不会**被渲染到最终的文本中。

        ```html
        {# 这一行是注释，前端查看源代码也看不到我 #}

        ```

    ---

    三、 进阶高级特性

    1. 过滤器 (Filters)

        过滤器用于对变量进行修改或格式化，使用管道符 `|` 连接。

        * `{{ text | upper }}`：将文本转为大写。
        * `{{ text | default('默认值') }}`：如果变量不存在，则使用默认值。
        * `{{ html_content | safe }}`：告诉 Jinja2 这段文本是安全的，**不需要**自动转义（常用于渲染富文本）。

    2. 模板继承 (Inheritance)——最强大的功能

        你可以写一个公共的母版（`base.html`），然后让其他子模板继承它，这样只需要修改多变的部分。

        **母版 `base.html`：**

        ```html
        <!DOCTYPE html>
        <html>
        <head>
            <title>{% block title %}我的网站{% endblock %}</title>
        </head>
        <body>
            <nav>公共导航栏</nav>
            
            <main>
                {% block content %}{% endblock %}
            </main>
            
            <footer>公共页脚</footer>
        </body>
        </html>

        ```

        **子模板 `index.html`：**

        ```html
        {% extends "base.html" %}

        {% block title %}首页 - 我的网站{% endblock %}

        {% block content %}
            <h1>这是首页的独特内容</h1>
            <p>欢迎来到我的世界。</p>
        {% endblock %}

        ```

    四、 在 Python 中如何使用 Jinja2（代码示例）

    如果你不在 Flask 等框架中，而是在纯 Python 脚本中独立使用 Jinja2，可以按照以下步骤：

    1. 安装

        ```bash
        pip install Jinja2

        ```

    2. 独立运行代码示例

        ```python
        from jinja2 import Environment, FileSystemLoader

        # 1. 创建一个环境对象，并指定模板文件存放的目录（假设在当前目录的 templates 文件夹下）
        env = Environment(loader=FileSystemLoader('templates'))

        # 2. 加载特定的模板文件
        template = env.get_template('hello.html')

        # 3. 准备要传入的数据
        data = {
            "title": "Jinja2 教程",
            "user": {"name": "张三", "is_vip": True},
            "skills": ["Python", "Web 开发", "Jinja2"]
        }

        # 4. 渲染模板，将数据注入
        result = template.render(data)

        # 5. 打印或写入文件
        print(result)

        ```

        假设你的 `templates/hello.html` 内容如下：

        ```html
        <!DOCTYPE html>
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <h1>你好，{{ user.name }}！</h1>
            {% if user.is_vip %}
                <p style="color: gold;">尊贵的 VIP 用户</p>
            {% endif %}
            
            <h3>掌握技能：</h3>
            <ul>
            {% for skill in skills %}
                <li>{{ skill }}</li>
            {% endfor %}
            </ul>
        </body>
        </html>

        ```

        运行 Python 脚本后，它就会完美的输出一段打满数据的标准 HTML 文本。

    五、 总结

    | 特性 | 语法/说明 |
    | --- | --- |
    | **输出变量** | `{{ variable }}` |
    | **逻辑控制** | `{% if/for ... %} ... {% endif/endfor %}` |
    | **代码注释** | `{# comment #}` |
    | **模板继承** | 骨架用 `{% block xxx %}`，子类用 `{% extends "base.html" %}` |
    | **核心优势** | 语法简单、效率极高、防 XSS 安全性好、解耦前后端 |
