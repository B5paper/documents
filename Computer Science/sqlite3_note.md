# sqlite3 note

## cache

* 简述 sqlite 用法、如何对接 python，以及学习资源

    SQLite 用法、Python对接及学习资源

    一、SQLite简介

    SQLite是一个轻量级、零配置、无服务器的嵌入式关系型数据库，将整个数据库存储为单个文件。

    二、SQLite基本用法

    1. 基本命令

        ```sql
        -- 创建表
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        );

        -- 插入数据
        INSERT INTO users (name, email, age) 
        VALUES ('张三', 'zhangsan@example.com', 25);

        -- 查询数据
        SELECT * FROM users;
        SELECT name, email FROM users WHERE age > 20;

        -- 更新数据
        UPDATE users SET age = 26 WHERE name = '张三';

        -- 删除数据
        DELETE FROM users WHERE id = 1;
        ```

    2. 数据类型

        - NULL: 空值
        - INTEGER: 整数
        - REAL: 浮点数
        - TEXT: 文本
        - BLOB: 二进制数据

    三、Python对接SQLite

    1. 使用内置sqlite3模块

        ```python
        import sqlite3

        # 连接数据库（不存在则创建）
        conn = sqlite3.connect('mydatabase.db')
        cursor = conn.cursor()

        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER
            )
        ''')

        # 插入数据
        cursor.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                       ('李四', 'lisi@example.com', 30))

        # 批量插入
        users_data = [
            ('王五', 'wangwu@example.com', 28),
            ('赵六', 'zhaoliu@example.com', 35)
        ]
        cursor.executemany("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", users_data)

        # 查询数据
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

        # 使用字典返回（更易读）
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        for row in rows:
            print(dict(row))

        # 提交并关闭
        conn.commit()
        conn.close()
        ```

    2. 使用上下文管理器

        ```python
        import sqlite3

        # 自动提交和关闭连接
        with sqlite3.connect('mydatabase.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            print(cursor.fetchall())
        ```

    3. 使用SQLAlchemy（ORM）

        ```python
        from sqlalchemy import create_engine, Column, Integer, String
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker

        Base = declarative_base()

        class User(Base):
            __tablename__ = 'users'
            id = Column(Integer, primary_key=True)
            name = Column(String)
            email = Column(String)
            age = Column(Integer)

        # 创建数据库引擎
        engine = create_engine('sqlite:///mydatabase.db')
        Base.metadata.create_all(engine)

        # 创建会话
        Session = sessionmaker(bind=engine)
        session = Session()

        # 添加数据
        new_user = User(name='钱七', email='qianqi@example.com', age=40)
        session.add(new_user)
        session.commit()

        # 查询数据
        users = session.query(User).filter(User.age > 25).all()
        for user in users:
            print(user.name, user.email)
        ```

    四、最佳实践

    1. 错误处理

        ```python
        import sqlite3

        try:
            conn = sqlite3.connect('mydatabase.db')
            cursor = conn.cursor()
            # 数据库操作
        except sqlite3.Error as e:
            print(f"数据库错误: {e}")
        finally:
            if conn:
                conn.close()
        ```

    2. 创建实用函数

        ```python
        import sqlite3
        from contextlib import contextmanager

        @contextmanager
        def get_db_connection(db_path):
            """数据库连接上下文管理器"""
            conn = None
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                yield conn
                conn.commit()
            except sqlite3.Error as e:
                if conn:
                    conn.rollback()
                raise e
            finally:
                if conn:
                    conn.close()

        # 使用示例
        with get_db_connection('mydatabase.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
        ```

    五、学习资源

    1. 官方文档

        - **SQLite官网**: https://www.sqlite.org/
        - **SQLite教程**: https://www.sqlite.org/docs.html
        - **Python sqlite3文档**: https://docs.python.org/3/library/sqlite3.html

    2. 在线教程

        - **SQLite Tutorial**: https://www.sqlitetutorial.net/
        - **W3Schools SQLite教程**: https://www.w3schools.com/sql/
        - **菜鸟教程SQLite**: https://www.runoob.com/sqlite/sqlite-tutorial.html

    3. 书籍推荐

        - 《SQLite权威指南》
        - 《Using SQLite》（O'Reilly）
        - 《Python编程：从入门到实践》（含SQLite章节）

    4. 视频课程

        - **YouTube**: SQLite Database for Beginners
        - **慕课网**: SQLite数据库实战
        - **B站**: Python+SQLite完整项目实战

    5. 实践项目

        1. **个人博客系统**: 使用SQLite存储文章和评论
        2. **任务管理应用**: 待办事项的增删改查
        3. **小型库存管理系统**: 商品信息管理
        4. **联系人管理**: 存储联系人信息

    6. 工具推荐

        - **DB Browser for SQLite**: 图形化管理工具
        - **SQLite Studio**: 功能丰富的SQLite管理工具
        - **VS Code插件**: SQLite by alexcvzz

    六、学习建议

    1. **从基础开始**: 先掌握基本的SQL语法
    2. **边学边练**: 通过实际项目巩固知识
    3. **理解原理**: 了解SQLite的ACID特性和事务机制
    4. **性能优化**: 学习索引、查询优化等高级话题
    5. **安全实践**: 防止SQL注入攻击

    SQLite是学习数据库和Python数据处理的绝佳起点，它简单易用但又功能齐全，适合小型项目、原型开发和移动应用。

* 不使用 Python 处理 SQLite 数据库文件的方法

    一、图形化界面工具

    1. **DB Browser for SQLite** (推荐)

        - **下载**: https://sqlitebrowser.org/

        - **特点**:

          - 免费开源，跨平台
          - 直观的图形界面
          - 支持数据浏览、编辑、查询

        **基本操作**:

        1. 打开数据库文件 (.db)
        2. 浏览数据库结构
        3. 执行 SQL 查询
        4. 编辑表格数据
        5. 导入/导出数据 (CSV, JSON, SQL)

    2. **SQLite Studio**

        - **下载**: https://sqlitestudio.pl/

        - **特点**:

          - 功能更丰富
          - 支持插件扩展
          - 多标签页管理

    3. **TablePlus** (部分免费)

        - **下载**: https://tableplus.com/
        - **特点**:
          - 现代 UI 设计
          - 支持多种数据库
          - 数据筛选和排序方便

    二、命令行工具

    1. **SQLite 命令行工具**

        ```bash
        # 下载 SQLite 命令行工具
        # Windows: 从官网下载 sqlite-tools
        # macOS: brew install sqlite
        # Linux: sudo apt-get install sqlite3

        # 基本使用
        sqlite3 mydatabase.db  # 打开数据库
        ```

        **常用命令**:

        ```sql
        -- 在 sqlite3 命令行中
        .help                    -- 查看帮助
        .tables                  -- 显示所有表
        .schema [table_name]     -- 查看表结构
        .mode column             -- 设置显示模式
        .headers on              -- 显示列名

        SELECT * FROM users;     -- 执行查询
        .quit                    -- 退出
        ```

    2. **示例会话**

        ```bash
        $ sqlite3 test.db
        SQLite version 3.37.0 2021-12-09 01:34:53
        Enter ".help" for usage hints.

        sqlite> .tables
        users   products  orders

        sqlite> .schema users
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        );

        sqlite> .mode column
        sqlite> .headers on
        sqlite> SELECT * FROM users LIMIT 5;
        id          name        email
        ----------  ----------  -----------------
        1           Alice       alice@email.com
        2           Bob         bob@email.com

        sqlite> .quit
        ```

    三、浏览器扩展

    1. **SQLite Viewer** (Chrome/Firefox)

        - 直接在浏览器中打开 SQLite 文件
        - 支持基本查询和数据浏览
        - 无需安装桌面应用

    2. **SQLite Manager** (Firefox 旧版)

        - 注意: 新版 Firefox 已移除

    四、代码编辑器插件

    1. **VS Code 插件**

        - **SQLite**: 直接在 VS Code 中浏览 SQLite 数据库
        - **SQLite Viewer**: 查看和查询数据库

        **安装后使用**:

        1. 打开 .db 文件
        2. 在资源管理器中浏览表结构
        3. 右键执行查询

    2. **Sublime Text 插件**

        - **SQLite Completions**: SQLite 语法支持

    五、在线工具

    1. **SQLite Online**

        - https://sqliteonline.com/
        - 直接在浏览器中操作
        - 支持上传 .db 文件
        - 无需安装任何软件

        **使用方法**:

        1. 访问网站
        2. 点击 "File" → "Open DB"
        3. 上传你的 .db 文件
        4. 执行查询和操作

    2. **SQLite Playground**

        - 练习 SQL 语法的好地方
        - 预设示例数据库

    六、移动端应用

    1. **iOS: SQLite Flow**

        - 在 iPhone/iPad 上查看 SQLite 文件
        - 支持 AirDrop 导入

    2. **Android: SQLite Editor**

        - 查看和编辑设备上的 SQLite 数据库
        - 需要 root 权限访问系统数据库

    七、实用脚本和批处理

    1. **Shell 脚本自动化**

        ```bash
        #!/bin/bash
        # backup_database.sh

        DB_FILE="mydatabase.db"
        BACKUP_DIR="backups"

        # 创建备份
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_file="${BACKUP_DIR}/backup_${timestamp}.db"

        # 使用 SQLite 备份命令
        sqlite3 $DB_FILE ".backup '$backup_file'"
        echo "数据库已备份到: $backup_file"
        ```

    2. **Windows 批处理**

        ```batch
        @echo off
        REM query_database.bat
        set DB_FILE=mydatabase.db

        echo 正在查询数据库...
        sqlite3 %DB_FILE% "SELECT * FROM users;"
        pause
        ```

    八、数据导入导出

    1. **导出为 CSV**

        ```bash
        # 命令行导出
        sqlite3 mydatabase.db <<EOF
        .headers on
        .mode csv
        .output data.csv
        SELECT * FROM users;
        .quit
        EOF
        ```

    2. **从 CSV 导入**

        ```bash
        # 创建表后导入
        sqlite3 mydatabase.db <<EOF
        .mode csv
        .import users.csv users
        EOF
        ```

    3. **导出为 SQL**

        ```bash
        # 导出整个数据库结构
        sqlite3 mydatabase.db .dump > backup.sql

        # 只导出特定表
        sqlite3 mydatabase.db .dump users > users.sql
        ```

    九、高级工具

    1. **SQLite Analyzer**

        ```bash
        # 分析数据库信息
        sqlite3_analyzer mydatabase.db > analysis.txt
        ```

    2. **数据库修复**

        ```bash
        # 修复损坏的数据库
        sqlite3 corrupted.db ".recover" | sqlite3 new.db
        ```

    十、最佳实践建议

    1. **日常查看和简单编辑**

        - **推荐**: DB Browser for SQLite
        - **理由**: 直观易用，功能齐全

    2. **自动化处理**

        - **推荐**: 命令行工具 + 脚本
        - **理由**: 可批量处理，适合任务自动化

    3. **快速查看**

        - **推荐**: VS Code 插件或在线工具
        - **理由**: 无需额外安装，快速便捷

    4. **手机查看**

        - **推荐**: 相应平台的 SQLite 查看器应用

    示例：完整工作流程

    场景：分析用户数据

    1. **查看数据库结构**

       ```bash
       sqlite3 app.db ".tables"
       sqlite3 app.db ".schema users"
       ```

    2. **使用图形工具深入分析**

       - 用 DB Browser 打开 app.db
       - 查看数据分布
       - 执行复杂查询

    3. **导出报告**

       ```bash
       # 导出活跃用户
       sqlite3 app.db <<EOF
       .headers on
       .mode csv
       .output active_users.csv
       SELECT * FROM users WHERE last_login > '2024-01-01';
       EOF
       ```

    4. **备份数据库**

       ```bash
       sqlite3 app.db ".backup 'app_backup_$(date +%Y%m%d).db'"
       ```

    注意事项

    1. **备份重要数据**：操作前先备份 .db 文件
    2. **注意文件权限**：确保有读写权限
    3. **版本兼容性**：SQLite 3.x 文件格式基本兼容
    4. **文件锁问题**：避免多个程序同时写入

    快速开始建议

    1. 先安装 **DB Browser for SQLite**
    2. 学习基本的 **sqlite3 命令行**操作
    3. 根据需求选择其他工具

    这样你可以完全脱离 Python，使用各种工具来处理 SQLite 数据库文件。

* sqlite3 如何取消当前语句的输入？

    在 SQLite3 命令行中取消当前语句的输入有以下几种方法：

    一、**常用取消方法**

    1. **输入分号结束语句**（最常用）

        ```sql
        sqlite> CREATE TABLE users (
           ...> id INTEGER PRIMARY KEY,
           ...> name TEXT
           ...> ;  -- 输入分号执行
        ```

    2. **输入空行加分号**

        ```sql
        sqlite> CREATE TABLE test (
           ...> id INT,
           ...> name
           ...>   -- 直接按 Enter（空行）
           ...> ;  -- 再输入分号
        ```

    3. **使用点命令中断**

        ```sql
        sqlite> CREATE TABLE unfinished (
           ...> id INTEGER,
           ...> .exit  -- 使用点命令中断
        ```

    4. **输入无效命令后加分号**

        ```sql
        sqlite> CREATE TABLE bad (
           ...> id INTEGER,
           ...> cancel  -- 输入无效内容
           ...> ;       -- 输入分号会报错，但退出输入模式
        Error: near "cancel": syntax error
        ```

    二、**特殊取消方法**

    1. **Ctrl+C**（部分终端有效）

        ```
        sqlite> SELECT * FROM 
        (press Ctrl+C)
        sqlite>  -- 回到正常提示符
        ```

    2. **输入单个点号**

        ```sql
        sqlite> CREATE TABLE test (
           ...> id INTEGER,
           ...> .  -- 输入单个点号
        sqlite>  -- 回到正常提示符
        ```

    3. **使用 .nullvalue 命令**

        ```sql
        sqlite> INSERT INTO users VALUES (
           ...> 1, 'John',
           ...> .nullvalue  -- 点命令中断
        sqlite>
        ```

    三、**不同情况下的取消示例**

    **情况1：创建表时取消**

    ```sql
    sqlite> CREATE TABLE temp_table (
       ...> id INTEGER PRIMARY KEY,
       ...> .quit  -- 使用点命令
    ```

    **情况2：复杂查询时取消**

    ```sql
    sqlite> SELECT * FROM users WHERE 
       ...> name LIKE '%John%' AND 
       ...> age > 20 AND
       ...> ;  -- 直接输入分号执行（可能出错）
    ```

    **情况3：插入数据时取消**

    ```sql
    sqlite> INSERT INTO users VALUES 
       ...> (1, 'Alice', 25),
       ...> (2, 'Bob', 30),
       ...> .exit  -- 中断并退出
    ```

    四、**预防措施和最佳实践**

    1. **使用文件编写SQL**

        ```bash
        # 先写 SQL 文件
        cat > create_table.sql << 'EOF'
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        );
        EOF

        # 再执行
        sqlite3 mydb.db < create_table.sql
        ```

    2. **使用 -init 选项**

        ```bash
        # 创建初始化文件
        echo ".tables" > init.sql
        echo "SELECT * FROM users LIMIT 5;" >> init.sql

        # 启动时自动执行
        sqlite3 mydb.db -init init.sql
        ```

    3. **交互模式下使用 .read**

        ```sql
        sqlite> .read create_table.sql  -- 从文件读取
        ```

    4. **设置提示符更容易识别**

        ```sql
        sqlite> .prompt "SQLITE> "  -- 设置简单提示符
        SQLITE> CREATE TABLE test (
           ...>   -- 更容易识别
        ```

    五、**特殊情况处理**

    1. **卡在引号中**

        ```sql
        sqlite> INSERT INTO users (name) VALUES ('John
           ...> ');  -- 必须补全引号
        ```

    2. **括号不匹配**

        ```sql
        sqlite> SELECT * FROM (SELECT id FROM users
           ...> );  -- 必须补全括号
        ```

    3. **命令历史问题**

        ```sql
        sqlite> -- 输入 .history 查看历史
        sqlite> .history  -- 查看之前输入
        ```

    六、**常用快捷键参考**

    | 按键 | 功能 | 说明 |
    |------|------|------|
    | `;` + Enter | 结束语句 | **最常用** |
    | `.exit` | 退出 SQLite | 同时取消当前语句 |
    | `.quit` | 退出 SQLite | 同 .exit |
    | `Ctrl+C` | 中断 | 部分终端支持 |
    | 空行 + `;` | 结束多行输入 | 安全方法 |

    七、**实用技巧**

    1. **批量操作时使用事务**

        ```sql
        sqlite> BEGIN TRANSACTION;
        sqlite> INSERT INTO users VALUES (1, 'Alice');
        sqlite> -- 发现错误想取消
        sqlite> ROLLBACK;  -- 回滚所有操作
        ```

    2. **使用 .echo 查看执行的命令**

        ```sql
        sqlite> .echo on
        sqlite> CREATE TABLE test (
           ...> id INTEGER
           ...> ;
        -- 会显示执行的完整命令
        ```

    3. **保存会话记录**

        ```bash
        # 启动时记录所有操作
        sqlite3 mydb.db -init /dev/null | tee session.log
        ```

    八、**快速取消的黄金法则**

    1. **首选**：直接输入 `;`（分号）
    2. **备用**：输入 `.exit` 或 `.quit`
    3. **紧急**：关闭终端窗口（不推荐）

    九、**示例：完整的工作流程**

    ```bash
    # 1. 启动 SQLite
    sqlite3 test.db

    # 2. 开始创建表
    sqlite> CREATE TABLE employees (
       ...> id INTEGER,
       ...> name TEXT,
       ...> -- 哎呀，想取消！
       ...> .exit  # 取消并退出

    # 3. 重新进入
    sqlite3 test.db

    # 4. 用文件方式
    sqlite> .read create_emp.sql
    ```

    **总结**：最安全可靠的方法是**输入分号 `;`** 来结束当前语句（即使它不完整），SQLite 会报语法错误但你会回到正常提示符。如果连分号都不起作用，就用 `.exit` 或 `.quit` 退出后重新进入。

* sqlite 常用语法

    以下是 SQLite3 常用语法总结：

    一、数据库操作

    ```sql
    -- 连接/创建数据库
    sqlite3 database.db

    -- 查看数据库信息
    .databases
    ```

    二、表操作

    1. 创建表

        ```sql
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            email TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        ```

    2. 修改表

        ```sql
        -- 添加列
        ALTER TABLE users ADD COLUMN phone TEXT;

        -- 重命名表
        ALTER TABLE users RENAME TO customers;
        ```

    3. 删除表

        ```sql
        DROP TABLE IF EXISTS users;
        ```

    三、数据操作（CRUD）

    1. 插入数据

        ```sql
        -- 插入完整数据
        INSERT INTO users (name, age, email) 
        VALUES ('张三', 25, 'zhangsan@example.com');

        -- 插入多条数据
        INSERT INTO users (name, age) VALUES 
        ('李四', 30),
        ('王五', 28);
        ```

    2. 查询数据

        ```sql
        -- 基础查询
        SELECT * FROM users;

        -- 选择特定列
        SELECT name, email FROM users;

        -- 条件查询
        SELECT * FROM users WHERE age > 25;

        -- 排序
        SELECT * FROM users ORDER BY age DESC;

        -- 限制结果
        SELECT * FROM users LIMIT 10;

        -- 分页
        SELECT * FROM users LIMIT 10 OFFSET 20;

        -- 去重
        SELECT DISTINCT age FROM users;

        -- 聚合函数
        SELECT COUNT(*) FROM users;
        SELECT AVG(age) FROM users;
        SELECT MAX(age), MIN(age) FROM users;

        -- 分组
        SELECT age, COUNT(*) FROM users GROUP BY age;
        ```

    3. 更新数据

        ```sql
        UPDATE users 
        SET age = 26, email = 'new@example.com'
        WHERE id = 1;
        ```

    4. 删除数据

        ```sql
        DELETE FROM users WHERE id = 1;
        -- 清空表
        DELETE FROM users;
        ```

    四、约束

    ```sql
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL CHECK (price > 0),
        category_id INTEGER REFERENCES categories(id)
    );
    ```

    五、连接查询

    ```sql
    -- 内连接
    SELECT users.name, orders.amount 
    FROM users 
    INNER JOIN orders ON users.id = orders.user_id;

    -- 左连接
    SELECT users.name, orders.amount 
    FROM users 
    LEFT JOIN orders ON users.id = orders.user_id;
    ```

    六、索引

    ```sql
    -- 创建索引
    CREATE INDEX idx_users_email ON users(email);

    -- 查看索引
    .indices users

    -- 删除索引
    DROP INDEX idx_users_email;
    ```

    七、事务

    ```sql
    BEGIN TRANSACTION;
    -- 执行多个SQL语句
    COMMIT;
    -- 或回滚
    ROLLBACK;
    ```

    八、视图

    ```sql
    -- 创建视图
    CREATE VIEW adult_users AS
    SELECT * FROM users WHERE age >= 18;

    -- 使用视图
    SELECT * FROM adult_users;

    -- 删除视图
    DROP VIEW adult_users;
    ```

    九、实用命令

    1. 命令行工具命令

        ```sql
        -- 显示所有表
        .tables

        -- 显示表结构
        .schema users

        -- 设置输出格式
        .mode column
        .headers on

        -- 导出数据
        .output data.sql
        .dump

        -- 导入数据
        .read data.sql

        -- 退出
        .quit
        ```

    2. 常用函数

        ```sql
        -- 字符串函数
        SELECT LENGTH(name), UPPER(name), LOWER(name) FROM users;

        -- 日期时间
        SELECT DATE('now'), TIME('now'), DATETIME('now');

        -- 类型转换
        SELECT CAST(age AS TEXT) FROM users;

        -- 条件判断
        SELECT name, CASE 
            WHEN age < 18 THEN '未成年'
            WHEN age >= 65 THEN '老年'
            ELSE '成年'
        END AS age_group FROM users;
        ```

    十、示例综合

    ```sql
    -- 创建数据库和表
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT,
        salary REAL,
        hire_date DATE
    );

    -- 插入数据
    INSERT INTO employees (name, department, salary, hire_date)
    VALUES 
        ('张三', '技术部', 15000, '2023-01-15'),
        ('李四', '销售部', 12000, '2022-08-20');

    -- 查询示例
    SELECT 
        department,
        COUNT(*) as count,
        AVG(salary) as avg_salary
    FROM employees
    WHERE hire_date > '2023-01-01'
    GROUP BY department
    ORDER BY avg_salary DESC;
    ```

    这些是 SQLite3 最常用的语法，涵盖了日常开发中的大多数需求。SQLite 语法简洁且高度兼容标准 SQL，学习成本较低。

* sqlite 查看表结构

    在 SQLite3 中，可以通过以下命令查看表的数量、表结构以及行数：

    1. 查看所有表

        ```sql
        .tables
        ```
        或者
        ```sql
        SELECT name FROM sqlite_master WHERE type='table';
        ```

    2. 查看表结构（列名）

        ```sql
        .schema 表名
        ```

        或者

        ```sql
        PRAGMA table_info(表名);
        ```

        例如：

        ```sql
        PRAGMA table_info(users);
        ```

        这会返回：

        - cid: 列ID
        - name: 列名
        - type: 数据类型
        - notnull: 是否允许NULL
        - dflt_value: 默认值
        - pk: 是否是主键

    3. 查看表行数

        ```sql
        SELECT COUNT(*) FROM 表名;
        ```

    4. 批量查看所有表信息

        **查看所有表名和行数：**

        ```sql
        SELECT 
            name as 表名,
            (SELECT COUNT(*) FROM sqlite_master m WHERE m.name = t.name AND m.type='table') as 是否表,
            (SELECT COUNT(*) FROM t.name) as 行数
        FROM sqlite_master t 
        WHERE type='table';
        ```

        或者分步骤查看：

        **查看所有表数量：**

        ```sql
        SELECT COUNT(*) FROM sqlite_master WHERE type='table';
        ```

        查看每个表的详细信息（表名、列信息、行数）：

        ```sql
        -- 创建临时表存储结果
        CREATE TEMP TABLE IF NOT EXISTS table_info AS
        SELECT 
            m.name as table_name,
            p.cid as column_id,
            p.name as column_name,
            p.type as data_type,
            p.notnull,
            p.dflt_value,
            p.pk,
            (SELECT COUNT(*) FROM (SELECT m.name) t) as row_count
        FROM sqlite_master m
        LEFT JOIN pragma_table_info(m.name) p
        WHERE m.type='table'
        ORDER BY m.name, p.cid;
        ```

    5. 实用脚本

        以下是一个实用的一键查询脚本：

        ```sql
        -- 查看所有表基本信息
        SELECT '=== 数据库中的表 ===' as '';
        SELECT type, name FROM sqlite_master WHERE type='table';

        -- 查看每个表的详细信息
        SELECT '=== 表详细信息 ===' as '';
        SELECT 
            m.name as '表名',
            COUNT(p.cid) as '列数',
            (SELECT COUNT(*) FROM (SELECT m.name) t) as '行数'
        FROM sqlite_master m
        LEFT JOIN pragma_table_info(m.name) p
        WHERE m.type='table'
        GROUP BY m.name;

        -- 查看所有表的列信息
        SELECT '=== 所有表的列结构 ===' as '';
        SELECT 
            m.name as '表名',
            p.name as '列名',
            p.type as '数据类型',
            CASE WHEN p.pk = 1 THEN '是' ELSE '' END as '主键',
            CASE WHEN p.notnull = 1 THEN '是' ELSE '' END as '非空'
        FROM sqlite_master m
        LEFT JOIN pragma_table_info(m.name) p
        WHERE m.type='table'
        ORDER BY m.name, p.cid;
        ```

    6. 使用 SQLite3 命令行工具的快捷方式

        在 SQLite3 命令行中，可以使用以下点命令：

        ```bash
        # 进入数据库
        sqlite3 your_database.db

        # 显示当前数据库中的所有表
        .tables

        # 显示某个表的完整结构
        .schema 表名

        # 显示表的创建语句和索引
        .schema

        # 退出 SQLite3
        .quit
        ```

    这些命令能帮你全面了解 SQLite 数据库的结构和数据量。

* sqlite3 查数据时，select xxx from some_table; 的输出太多太长，我希望只看 head 或 tail 的几个，怎么办

    在 SQLite 中有几种方法可以限制查询结果的数量：

    1. **使用 LIMIT 子句（最常用）**

        ```sql
        -- 查看前 10 条记录（类似 head）
        SELECT * FROM some_table LIMIT 10;

        -- 查看前 5 条记录
        SELECT * FROM some_table LIMIT 5;
        ```

    2. **查看最后几条记录（类似 tail）**

        ```sql
        -- 查看最后 10 条记录（需要按某个字段排序）
        SELECT * FROM some_table ORDER BY id DESC LIMIT 10;

        -- 如果表有自增 ID 或时间字段
        SELECT * FROM some_table ORDER BY created_at DESC LIMIT 10;
        ```

    3. **分页查看**

        ```sql
        -- 查看第 11-20 条记录
        SELECT * FROM some_table LIMIT 10 OFFSET 10;
        ```

    4. **在 SQLite 命令行中的快捷方式**

        ```sql
        -- 在 sqlite3 命令行中，可以设置输出格式
        .headers on      -- 显示列名
        .mode column     -- 列模式，更易读
        .width 20 10     -- 设置列宽度

        -- 然后使用 LIMIT
        SELECT * FROM some_table LIMIT 5;
        ```

    5. **使用 .shell 命令结合系统工具**

        ```sql
        -- 如果你在 Linux/Mac 的 sqlite3 命令行中
        .shell echo "SELECT * FROM some_table;" | sqlite3 your_database.db | head -20

        -- 或者先保存到文件再查看
        .output temp.txt
        SELECT * FROM some_table;
        .output stdout
        .shell head -20 temp.txt
        ```

    6. **创建视图查看样本数据**

        ```sql
        -- 创建一个只包含前 N 条记录的视图
        CREATE VIEW sample_view AS 
        SELECT * FROM some_table LIMIT 50;

        -- 然后查询这个视图
        SELECT * FROM sample_view;
        ```

    **推荐用法：**

    对于日常使用，最简单的就是：

    ```sql
    -- 查看开头
    SELECT * FROM some_table LIMIT 10;

    -- 查看结尾（假设有 id 字段）
    SELECT * FROM some_table ORDER BY id DESC LIMIT 10;
    ```

    如果表没有合适的排序字段，但你想了解数据结构：

    ```sql
    -- 先看表结构
    .schema some_table

    -- 然后看几条样本数据
    SELECT * FROM some_table LIMIT 5;
    ```

* sqlite3 如何查看指定行的数据

    在 SQLite3 中查看指定行的数据，主要有以下几种方法：

    1. 使用 `LIMIT` 和 `OFFSET` 子句

        ```sql
        -- 查看第 1 行（从第 0 行开始算）
        SELECT * FROM 表名 LIMIT 1 OFFSET 0;

        -- 查看第 5 行
        SELECT * FROM 表名 LIMIT 1 OFFSET 4;

        -- 查看第 10-15 行
        SELECT * FROM 表名 LIMIT 6 OFFSET 9;
        ```

    2. 使用行号（ROW_NUMBER()）

        ```sql
        -- SQLite 3.25.0+ 版本支持
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY 某列) AS row_num
            FROM 表名
        ) WHERE row_num = 5;
        ```

    3. 通过主键或唯一标识符

        ```sql
        -- 如果知道具体的 ID 或主键值
        SELECT * FROM 表名 WHERE id = 10;

        -- 使用多个条件定位
        SELECT * FROM 表名 
        WHERE 列1 = '值1' AND 列2 = '值2';
        ```

    4. 获取特定位置的行

        ```sql
        -- 获取第一行
        SELECT * FROM 表名 ORDER BY 某列 LIMIT 1;

        -- 获取最后一行（假设有自增ID）
        SELECT * FROM 表名 ORDER BY id DESC LIMIT 1;
        ```

    5. 实用示例

        ```sql
        -- 假设有一个 users 表
        -- 1. 查看第 3 行数据
        SELECT * FROM users LIMIT 1 OFFSET 2;

        -- 2. 查看第 3-5 行
        SELECT * FROM users LIMIT 3 OFFSET 2;

        -- 3. 结合排序查看
        SELECT * FROM users ORDER BY created_at DESC LIMIT 1 OFFSET 4;

        -- 4. 使用子查询（如果版本支持）
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY id) AS rn 
            FROM users
        ) WHERE rn = 7;
        ```

    注意事项：

    - **行号从 0 开始**：`OFFSET 0` 表示第一行
    - **SQLite 没有内置行号**：除非使用窗口函数（需要较新版本）
    - **结果可能不确定**：如果没有 `ORDER BY`，返回的行顺序可能不固定
    - **性能考虑**：`OFFSET` 在处理大数据时可能较慢

    完整示例：

    ```sql
    -- 进入 SQLite
    sqlite3 database.db

    -- 查看表结构
    .schema users

    -- 查看第 5 行数据
    SELECT * FROM users LIMIT 1 OFFSET 4;

    -- 退出 SQLite
    .quit
    ```

    最简单常用的方法就是使用 `LIMIT 1 OFFSET n-1` 来获取第 n 行数据。

## topics
