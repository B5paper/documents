# sqlite3 note

## cache

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
