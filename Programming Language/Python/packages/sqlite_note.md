# sqlite note

## cache

* python 如何处理 SQLite format 3 这种数据的文件？

    Python 内置了 `sqlite3` 模块，可以直接处理 SQLite format 3 数据库文件。以下是几种常见操作：

    1. **基本连接和操作**

        ```python
        import sqlite3

        # 连接到数据库文件（如果不存在会自动创建）
        conn = sqlite3.connect('example.db')  # 你的 SQLite 文件

        # 创建游标对象
        cursor = conn.cursor()

        # 执行 SQL 语句
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER
            )
        ''')

        # 插入数据
        cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Alice', 30))
        cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Bob', 25))

        # 提交事务
        conn.commit()

        # 查询数据
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        for row in rows:
            print(row)  # (1, 'Alice', 30), (2, 'Bob', 25)

        # 关闭连接
        conn.close()
        ```

    2. **使用上下文管理器（推荐）**

        ```python
        import sqlite3

        with sqlite3.connect('example.db') as conn:
            cursor = conn.cursor()
            
            # 使用 execute 执行单条语句
            cursor.execute("SELECT * FROM users")
            
            # 使用 executemany 批量插入
            data = [('Charlie', 28), ('David', 35), ('Eve', 22)]
            cursor.executemany("INSERT INTO users (name, age) VALUES (?, ?)", data)
            
            # 获取所有结果
            results = cursor.fetchall()
            
            # 逐行获取
            cursor.execute("SELECT * FROM users")
            while row := cursor.fetchone():
                print(row)
        ```

    3. **使用 Pandas 处理（适合数据分析）**

        ```python
        import sqlite3
        import pandas as pd

        # 使用 pandas 读取 SQLite 数据
        conn = sqlite3.connect('example.db')

        # 读取整个表到 DataFrame
        df = pd.read_sql_query("SELECT * FROM users", conn)
        print(df)

        # 将 DataFrame 写入数据库
        new_data = pd.DataFrame({
            'name': ['Frank', 'Grace'],
            'age': [40, 27]
        })
        new_data.to_sql('users', conn, if_exists='append', index=False)

        # 关闭连接
        conn.close()
        ```

    4. **查看数据库信息**

        ```python
        import sqlite3

        def inspect_database(db_file):
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                
                # 获取所有表名
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print("Tables:", tables)
                
                # 查看表结构
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    print(f"\nTable: {table_name}")
                    for col in columns:
                        print(f"  Column: {col[1]}, Type: {col[2]}, Nullable: {col[3]}")
                
                # 查看数据
                cursor.execute("SELECT * FROM sqlite_master;")
                schema = cursor.fetchall()
                print("\nDatabase schema:", schema)

        inspect_database('example.db')
        ```

    5. **处理大型数据集**

        ```python
        import sqlite3

        def process_large_dataset(db_file):
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # 使用 fetchmany 分批处理
            cursor.execute("SELECT * FROM large_table")
            
            batch_size = 1000
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                # 处理每一批数据
                for row in rows:
                    process_row(row)
            
            conn.close()

        # 或者使用迭代器
        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row  # 以字典形式访问
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users")
            for row in cursor:
                print(dict(row))  # {'id': 1, 'name': 'Alice', 'age': 30}
        ```

    6. **实用技巧**

        ```python
        import sqlite3

        # 设置连接参数
        conn = sqlite3.connect(
            'example.db',
            timeout=10,  # 连接超时时间
            isolation_level=None,  # 自动提交模式
            detect_types=sqlite3.PARSE_DECLTYPES  # 自动解析日期时间类型
        )

        # 启用外键约束
        conn.execute("PRAGMA foreign_keys = ON")

        # 创建自定义函数
        def my_function(x):
            return x.upper()

        conn.create_function("UPPERCASE", 1, my_function)
        cursor = conn.cursor()
        cursor.execute("SELECT UPPERCASE(name) FROM users")

        # 备份数据库
        def backup_database(src_db, dst_db):
            with sqlite3.connect(src_db) as src:
                with sqlite3.connect(dst_db) as dst:
                    src.backup(dst)

        # 内存数据库（临时处理）
        memory_conn = sqlite3.connect(':memory:')
        ```

    7. **完整示例：创建和管理数据库**

        ```python
        import sqlite3
        from contextlib import closing

        class SQLiteManager:
            def __init__(self, db_file):
                self.db_file = db_file
            
            def __enter__(self):
                self.conn = sqlite3.connect(self.db_file)
                self.conn.row_factory = sqlite3.Row  # 返回字典样式的行
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.conn.close()
            
            def execute_query(self, query, params=None):
                with closing(self.conn.cursor()) as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    return cursor.fetchall()
            
            def execute_commit(self, query, params=None):
                with closing(self.conn.cursor()) as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    self.conn.commit()
                    return cursor.rowcount

        # 使用示例
        with SQLiteManager('test.db') as db:
            # 创建表
            db.execute_commit('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    price REAL,
                    stock INTEGER DEFAULT 0
                )
            ''')
            
            # 插入数据
            db.execute_commit(
                "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
                ('Laptop', 999.99, 10)
            )
            
            # 查询数据
            results = db.execute_query("SELECT * FROM products")
            for row in results:
                print(dict(row))
        ```

    重要提示：

    1. **文件路径**：SQLite 数据库是单个文件，可以直接指定路径
    2. **事务管理**：记得使用 `commit()` 保存更改，或使用上下文管理器自动处理
    3. **参数化查询**：始终使用 `?` 占位符防止 SQL 注入
    4. **连接关闭**：及时关闭连接释放资源
    5. **数据类型映射**：
       - NULL → None
       - INTEGER → int
       - REAL → float
       - TEXT → str
       - BLOB → bytes

    这些方法应该能帮助你处理大多数 SQLite format 3 数据库的操作需求。

* 我想快速获得 example.db 的概览/摘要，怎么办

    有几种快速获取 SQLite 数据库概览的方法：

    1. **使用 Python 快速概览脚本**

        ```python
        import sqlite3
        import pandas as pd

        def get_db_summary(db_file):
            """快速获取数据库概览"""
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            print(f"📊 数据库概览: {db_file}")
            print("=" * 50)
            
            # 1. 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"📁 共有 {len(tables)} 张表:")
            print("-" * 30)
            
            for i, (table_name,) in enumerate(tables, 1):
                # 2. 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                # 3. 获取行数
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                # 4. 获取示例数据
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_rows = cursor.fetchall()
                
                print(f"{i}. 表名: {table_name}")
                print(f"   📈 行数: {row_count:,}")
                print(f"   🏗️  列结构 ({len(columns)}列):")
                
                for col in columns:
                    col_id, col_name, col_type, notnull, default, pk = col
                    pk_flag = " 🔑" if pk else ""
                    null_flag = " NOT NULL" if notnull else ""
                    default_flag = f" DEFAULT {default}" if default else ""
                    print(f"     - {col_name}: {col_type}{null_flag}{default_flag}{pk_flag}")
                
                # 5. 显示示例数据（如果有数据）
                if row_count > 0:
                    print(f"   📋 示例数据 (前3行):")
                    
                    # 获取列名
                    column_names = [desc[0] for desc in cursor.description]
                    
                    # 显示列名
                    print(f"     {', '.join(column_names)}")
                    
                    # 显示数据
                    for row in sample_rows:
                        # 截断长字符串
                        display_row = []
                        for cell in row:
                            if isinstance(cell, str) and len(cell) > 20:
                                display_row.append(f"{cell[:20]}...")
                            else:
                                display_row.append(str(cell))
                        print(f"     {tuple(display_row)}")
                
                print()
            
            # 6. 数据库大小
            import os
            if os.path.exists(db_file):
                size_bytes = os.path.getsize(db_file)
                size_mb = size_bytes / (1024 * 1024)
                print(f"📦 数据库文件大小: {size_mb:.2f} MB ({size_bytes:,} 字节)")
            
            conn.close()

        # 使用
        get_db_summary('example.db')
        ```

    2. **更简洁的版本**

        ```python
        import sqlite3
        import pandas as pd

        def quick_db_overview(db_file):
            """快速概览 - 只显示关键信息"""
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                
                print("🔍 数据库快速概览")
                print("=" * 40)
                
                # 获取所有表
                cursor.execute("""
                    SELECT name 
                    FROM sqlite_master 
                    WHERE type='table' 
                    AND name NOT LIKE 'sqlite_%'
                """)
                
                tables = cursor.fetchall()
                
                summary_data = []
                
                for table_name, in tables:
                    # 行数
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    # 列信息
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    summary_data.append({
                        '表名': table_name,
                        '行数': row_count,
                        '列数': len(columns),
                        '列名': ', '.join(column_names[:3]) + 
                               ('...' if len(column_names) > 3 else '')
                    })
                
                # 使用 pandas 漂亮打印
                df_summary = pd.DataFrame(summary_data)
                print(df_summary.to_string(index=False))
                print()
                
                # 如果有表，显示第一张表的预览
                if tables:
                    first_table = tables[0][0]
                    print(f"📄 {first_table} 表数据预览:")
                    df_preview = pd.read_sql_query(f"SELECT * FROM {first_table} LIMIT 5", conn)
                    print(df_preview.to_string(index=False))

        quick_db_overview('example.db')
        ```

    3. **交互式探索工具**

        ```python
        import sqlite3
        from tabulate import tabulate  # pip install tabulate

        def interactive_db_explorer(db_file):
            """交互式数据库探索器"""
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            while True:
                print("\n" + "="*60)
                print("📁 SQLite 数据库交互式探索器")
                print("="*60)
                
                # 显示所有表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                print("\n可用的表:")
                for i, table in enumerate(tables, 1):
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  {i}. {table} ({count} 行)")
                
                print("\n命令:")
                print("  s [表名] - 查看表结构")
                print("  p [表名] - 预览表数据")
                print("  q - 查询SQL")
                print("  x - 退出")
                
                choice = input("\n输入命令: ").strip().lower()
                
                if choice == 'x':
                    break
                elif choice.startswith('s '):
                    table = choice[2:].strip()
                    if table in tables:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = cursor.fetchall()
                        print(f"\n{table} 表结构:")
                        print(tabulate(columns, headers=['ID', '列名', '类型', 'Not Null', '默认值', '主键']))
                
                elif choice.startswith('p '):
                    table = choice[2:].strip()
                    if table in tables:
                        cursor.execute(f"SELECT * FROM {table} LIMIT 10")
                        rows = cursor.fetchall()
                        if rows:
                            # 获取列名
                            column_names = [desc[0] for desc in cursor.description]
                            print(f"\n{table} 前10行数据:")
                            print(tabulate(rows, headers=column_names))
                
                elif choice == 'q':
                    sql = input("输入SQL查询: ").strip()
                    try:
                        cursor.execute(sql)
                        rows = cursor.fetchall()
                        if rows:
                            column_names = [desc[0] for desc in cursor.description]
                            print(tabulate(rows, headers=column_names))
                        else:
                            print("✓ 执行成功，无返回数据")
                    except Exception as e:
                        print(f"❌ 错误: {e}")
            
            conn.close()

        # 使用
        interactive_db_explorer('example.db')
        ```

    4. **使用现成工具（无需写代码）**

        **SQLite 命令行工具**（系统自带）

        ```bash
        # 进入 SQLite 交互模式
        sqlite3 example.db

        # 常用命令
        .tables                 # 显示所有表
        .schema [表名]          # 显示表结构
        SELECT * FROM 表名 LIMIT 5;  # 预览数据
        .quit                   # 退出
        ```

        **使用 DB Browser for SQLite**（图形界面）

        1. 下载安装：https://sqlitebrowser.org/
        2. 打开 `example.db` 文件
        3. 直接查看表结构、数据、索引等

        **使用 Python 单行命令快速查看**

        ```bash
        # 查看所有表
        python -c "import sqlite3; conn=sqlite3.connect('example.db'); print(conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\";').fetchall())"

        # 查看第一个表的结构
        python -c "import sqlite3; conn=sqlite3.connect('example.db'); cursor=conn.cursor(); cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\" LIMIT 1'); table=cursor.fetchone()[0]; print(f'Table: {table}'); print(cursor.execute(f'PRAGMA table_info({table})').fetchall())"
        ```

    5. **生成 HTML 报告**

        ```python
        import sqlite3
        from datetime import datetime

        def generate_html_report(db_file, output_file='db_report.html'):
            """生成HTML格式的数据库报告"""
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            html = f"""
            <html>
            <head>
                <title>数据库报告 - {db_file}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .table {{ margin: 20px 0; border-collapse: collapse; width: 100%; }}
                    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; }}
                    .table th {{ background-color: #f2f2f2; }}
                    .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
                    .code {{ font-family: monospace; background-color: #f5f5f5; padding: 2px 4px; }}
                </style>
            </head>
            <body>
                <h1>📊 数据库报告: {db_file}</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table_name, in tables:
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # 获取行数
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # 获取示例数据
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                sample_data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if sample_data else []
                
                html += f"""
                <div class="summary">
                    <h2>📁 表: {table_name}</h2>
                    <p><strong>行数:</strong> {row_count}</p>
                    <p><strong>列数:</strong> {len(columns)}</p>
                    
                    <h3>表结构:</h3>
                    <table class="table">
                        <tr>
                            <th>列名</th><th>类型</th><th>Not Null</th><th>默认值</th><th>主键</th>
                        </tr>
                """
                
                for col in columns:
                    html += f"""
                        <tr>
                            <td><span class="code">{col[1]}</span></td>
                            <td>{col[2]}</td>
                            <td>{'✓' if col[3] else ''}</td>
                            <td>{col[4] or ''}</td>
                            <td>{'✓' if col[5] else ''}</td>
                        </tr>
                    """
                
                html += "</table>"
                
                if row_count > 0:
                    html += f"""
                    <h3>数据预览 (前5行):</h3>
                    <table class="table">
                        <tr>
                            {' '.join(f'<th>{name}</th>' for name in column_names)}
                        </tr>
                    """
                    
                    for row in sample_data:
                        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                    
                    html += "</table>"
                
                html += "</div>"
            
            html += "</body></html>"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            conn.close()
            print(f"✅ 报告已生成: {output_file}")

        # 使用
        generate_html_report('example.db')
        ```

    推荐方法：

    1. **最快**：使用第一个 `get_db_summary()` 函数
    2. **最方便**：使用 SQLite 命令行工具
    3. **最直观**：使用 DB Browser for SQLite（图形界面）
    4. **最详细**：生成 HTML 报告

    选择哪种方法取决于你的具体需求和偏好！

## topics
