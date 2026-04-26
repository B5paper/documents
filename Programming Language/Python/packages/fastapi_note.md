* 详细讲讲 python 中 fastapi 的用法

    FastAPI 是目前 Python 领域最受关注的 Web 框架之一。它的核心卖点一如其名：**快**。不仅是运行速度快（基于 Starlette 和 Pydantic），开发速度也极快。

    以下是 FastAPI 的核心用法指南：

    ---

    1. 快速上手：最简代码
        
        要运行 FastAPI，你需要安装框架本身和一个 ASGI 服务器（通常是 `uvicorn`）。

        ```bash
        pip install fastapi uvicorn
        ```

        创建一个 `main.py` 文件：
        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/")
        def read_root():
            return {"Hello": "World"}
        ```
        **运行命令：** `uvicorn main:app --reload`

    ---

    2. 核心特性：自动文档
        
        FastAPI 最让开发者省心的地方在于它会自动生成交互式 API 文档。项目启动后，你可以通过以下地址访问：
        * **Swagger UI:** `http://127.0.0.1:8000/docs`（直接在线测试接口）
        * **ReDoc:** `http://127.0.0.1:8000/redoc`（更整洁的规范文档）

    ---

    3. 参数处理
        
        FastAPI 对参数的处理非常直观，利用了 Python 的类型提示（Type Hints）。

        **路径参数 (Path Parameters)**

        ```python
        @app.get("/items/{item_id}")
        def read_item(item_id: int): # 自动进行类型转换和校验
            return {"item_id": item_id}
        ```

        **查询参数 (Query Parameters)**

        ```python
        @app.get("/users/")
        def get_user(name: str, age: int = 20): # age 有默认值，为可选参数
            return {"name": name, "age": age}
        ```

    ---

    4. 请求体与 Pydantic 模型
        
        对于 POST 请求，FastAPI 使用 **Pydantic** 来定义数据结构。这不仅能自动解析 JSON，还能自动验证数据的合法性。

        ```python
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            price: float
            is_offer: bool = None

        @app.post("/items/")
        def create_item(item: Item):
            return {"message": f"商品 {item.name} 已创建", "total": item.price}
        ```

    ---

    5. 异步编程 (Async/Await)
        
        FastAPI 原生支持异步。如果你的操作涉及 IO 密集型任务（如数据库查询、调用其他 API），使用 `async def` 可以显著提升并发性能。

        ```python
        @app.get("/data")
        async def fetch_data():
            results = await some_db_call()
            return results
        ```

    ---

    6. 依赖注入 (Dependency Injection)
        
        这是 FastAPI 处理插件、数据库连接和认证的核心机制。

        ```python
        from fastapi import Depends

        def common_parameters(q: str = None, skip: int = 0, limit: int = 10):
            return {"q": q, "skip": skip, "limit": limit}

        @app.get("/items/")
        async def read_items(commons: dict = Depends(common_parameters)):
            return commons
        ```

    ---

    7. 性能对比
        
        FastAPI 的性能在 Python 框架中处于第一梯队，这主要归功于它底层的设计：

        | 特性 | FastAPI | Flask | Django |
        | :--- | :--- | :--- | :--- |
        | **异步支持** | 原生支持 (ASGI) | 较弱 (WSGI) | 较弱 (WSGI/ASGI) |
        | **开发速度** | 极快 (类型提示) | 快 | 中等 (重型框架) |
        | **性能** | 高 (接近 NodeJS/Go) | 中 | 低 |
        | **自动文档** | 内置 | 需插件 | 需插件 |

    ---

    **总结与建议**

    * **初学者：** 重点掌握 Pydantic 模型和装饰器路由。
    * **进阶：** 深入研究 `Depends` 依赖注入系统，这是构建复杂后端逻辑的灵魂。
    * **部署：** 生产环境下通常结合 `Gunicorn` 和 `Uvicorn` 工作。

    你目前是打算将 FastAPI 用于构建高性能的微服务，还是想快速给现有的数据分析模型套一个 API 外壳？

