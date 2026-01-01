# click note

## cache

* 如果同一个文件中用 @click.command() 装饰了多个函数，每个函数都会成为一个独立的命令行命令，但你需要使用 命令组（Command Groups） 来组织它们，否则只能直接调用其中一个。

    错误示例（有问题）

    ```python
    import click

    @click.command()
    def hello():
        """问候命令"""
        click.echo("Hello!")

    @click.command()  
    def goodbye():
        """告别命令"""
        click.echo("Goodbye!")

    if __name__ == '__main__':
        # 问题：应该调用哪个？
        hello()  # 只能调用一个
        # goodbye()  # 这个不会执行
    ```

    正确方式：使用命令组

    * 方法1：使用 @click.group()

        ```python
        import click

        @click.group()
        def cli():
            """主命令组"""
            pass

        @cli.command()
        def hello():
            """问候命令"""
            click.echo("Hello!")

        @cli.command()
        def goodbye():
            """告别命令"""
            click.echo("Goodbye!")

        if __name__ == '__main__':
            cli()  # 运行命令组
        ```

        运行：

        ```bash
        python script.py hello      # 输出: Hello!
        python script.py goodbye    # 输出: Goodbye!
        python script.py --help     # 显示所有可用命令
        ```

    * 方法2：多个独立命令（不推荐）

        ```python
        import click

        @click.command()
        def hello():
            click.echo("Hello!")

        @click.command()  
        def goodbye():
            click.echo("Goodbye!")

        if __name__ == '__main__':
            # 手动选择要执行的命令
            import sys
            if len(sys.argv) > 1:
                command = sys.argv[1]
                if command == 'hello':
                    hello()
                elif command == 'goodbye':
                    goodbye()
                else:
                    print("未知命令")
            else:
                print("请指定命令: hello 或 goodbye")
        ```

        运行：

        ```bash
        python script.py hello
        python script.py goodbye
        ```

        完整示例：功能丰富的命令组

        ```python
        import click

        @click.group()
        def cli():
            """一个多功能命令行工具"""
            pass

        @cli.command()
        @click.option('--name', default='World', help='问候的对象')
        @click.option('--count', default=1, help='重复次数')
        def hello(name, count):
            """问候某人"""
            for i in range(count):
                click.echo(f"Hello {name}!")

        @cli.command()
        @click.option('--name', default='World', help='告别的对象')
        def goodbye(name):
            """告别某人"""
            click.echo(f"Goodbye {name}!")

        @cli.command()
        @click.option('--verbose', is_flag=True, help='显示详细信息')
        def status(verbose):
            """显示系统状态"""
            if verbose:
                click.echo("详细状态信息...")
            else:
                click.echo("系统状态正常")

        if __name__ == '__main__':
            cli()
        ```

        运行示例：

        ```bash
        python script.py --help
        python script.py hello --name Alice --count 3
        python script.py goodbye --name Bob
        python script.py status --verbose
        ```

    **嵌套命令组**

    对于更复杂的应用，还可以创建嵌套的命令组：

    ```python
    import click

    @click.group()
    def cli():
        pass

    @cli.group()
    def db():
        """数据库操作"""
        pass

    @db.command()
    def init():
        """初始化数据库"""
        click.echo("数据库初始化完成")

    @db.command()
    def migrate():
        """数据库迁移"""
        click.echo("数据库迁移完成")

    @cli.group()
    def user():
        """用户管理"""
        pass

    @user.command()
    @click.option('--name', required=True, help='用户名')
    def create(name):
        """创建用户"""
        click.echo(f"创建用户: {name}")

    @user.command()
    def list():
        """列出用户"""
        click.echo("用户列表...")

    if __name__ == '__main__':
        cli()
    ```

    运行：

    ```bash
    python script.py db init
    python script.py db migrate
    python script.py user create --name Alice
    python script.py user list
    ```

    总结

    * 单个 @click.command()：创建一个独立的命令

    * 多个命令需要组织：使用 @click.group() 创建命令组

    * 子命令：用 @group.command() 装饰器将命令添加到组中

    * 执行：调用命令组的函数来启动整个CLI应用

    这样就能创建出像 git、docker 那样具有丰富子命令的专业命令行工具！

* @click.command() 装饰的函数在当前进程的当前线程中执行，不会创建子进程。

    执行流程详解

    ```python
    import click

    @click.command()
    def hello():
        """这是一个命令行命令"""
        print(f"进程ID: {os.getpid()}")
        print(f"线程ID: {threading.current_thread().ident}")
        print("Hello World!")

    if __name__ == '__main__':
        hello()  # 直接函数调用，在当前进程执行
    ```

    运行这个程序，你会看到：

    * 进程ID就是当前Python解释器的进程ID

    * 线程ID就是主线程的ID

    实际执行过程

    1. 装饰器准备阶段
    
        ```python
        # @click.command() 实际上做了类似这样的事情：
        def command_decorator(func):
            def wrapper(*args, **kwargs):
                # 1. 解析命令行参数
                parsed_args = parse_command_line()
                # 2. 调用原始函数，传入解析后的参数
                return func(**parsed_args)
            return wrapper

        # 所以：
        # hello = command_decorator(hello)
        # hello() 实际上调用的是 wrapper()
        ```

    2. 执行时

        ```python
        import os
        import threading
        import click

        @click.command()
        @click.option('--name', default='World')
        def hello(name):
            print(f"当前进程: {os.getpid()}")
            print(f"当前线程: {threading.current_thread().name}")
            print(f"Hello {name}!")

        if __name__ == '__main__':
            print(f"调用前进程: {os.getpid()}")
            hello()  # 就是普通的函数调用！
            print("函数执行完毕")
        ```

    **与子进程的对比**

    * 当前进程执行（click的方式）

        ```python
        import click
        import os

        @click.command()
        def show_pid():
            print(f"PID: {os.getpid()} - 这是同一个进程")

        if __name__ == '__main__':
            print(f"主程序PID: {os.getpid()}")
            show_pid()  # 输出相同的PID
        ```

    * 子进程执行（明确创建子进程）

        ```python
        import subprocess
        import os

        def run_in_subprocess():
            print(f"主进程PID: {os.getpid()}")
            
            # 明确创建子进程
            result = subprocess.run(['python', '-c', 'import os; print(f"子进程PID: {os.getpid()}")'])
            
            print("回到主进程")

        if __name__ == '__main__':
            run_in_subprocess()
        ```

    "命令行命令"的真正含义

    @click.command() 的意思是：

    * "这个函数可以从命令行调用，并且会自动处理命令行参数的解析，但执行仍然在当前Python进程中"

    完整示例说明

    ```python
    import click
    import os
    import time

    @click.command()
    @click.option('--count', default=1, help='循环次数')
    def demo(count):
        """演示程序"""
        print(f"执行进程: {os.getpid()}")
        
        for i in range(count):
            print(f"循环 {i+1}/{count}")
            time.sleep(0.5)
        
        print("执行完成")

    if __name__ == '__main__':
        # 这行代码：
        demo()
        
        # 相当于：
        # 1. 解析 sys.argv
        # 2. 提取 --count 参数  
        # 3. 调用 demo(count=解析到的值)
        # 4. 在当前进程执行函数体
    ```

    运行：

    ```bash
    python script.py --count 3
    ```

    总结

    * 当前进程：@click.command() 装饰的函数在当前Python进程执行

    * 当前线程：默认在主线程执行

    * 不是子进程：不会创建新的进程

    * 自动参数解析：主要价值是自动处理命令行参数解析

    * 普通函数调用：本质上还是调用Python函数

    所以"作为一个命令行命令来执行"指的是提供命令行接口，而不是在子进程中执行。

    被装饰的函数的参数直接就是 input arg，click 已经帮你解析好了

* @click.command() 这个是什么意思？

    `@click.command()` 是 装饰器（decorator） 语法，用于将普通 Python 函数转换为命令行接口。让我详细解释一下：

    **基本概念**

    * 装饰器是什么？

        装饰器是 Python 的一种语法糖，用于修改或增强函数的行为。

    * 在 click 中的用途

        @click.command() 告诉 click："下面的函数应该作为一个命令行命令来执行"

    **基本用法**

    ```python
    import click

    @click.command()
    def hello():
        """简单的问候命令"""
        click.echo("Hello World!")

    if __name__ == '__main__':
        hello()
    ```

    运行：

    ```bash
    python script.py
    # 输出: Hello World!
    ```

    **添加参数和选项**

    * 带参数的命令

        ```python
        import click

        @click.command()
        @click.option('--name', default='World', help='要问候的人')
        @click.option('--count', default=1, help='问候次数')
        def hello(name, count):
            """问候命令"""
            for i in range(count):
                click.echo(f"Hello {name}!")

        if __name__ == '__main__':
            hello()
        ```

    * 运行：

        ```bash
        python script.py --name Alice --count 3
        # 输出: 
        # Hello Alice!
        # Hello Alice!
        # Hello Alice!
        ```

    **多层命令结构**

    click 支持创建复杂的多级命令：

    ```python
    import click

    @click.group()
    def cli():
        """主命令组"""
        pass

    @cli.command()
    def init():
        """初始化项目"""
        click.echo("初始化项目...")

    @cli.command()
    @click.option('--port', default=8000, help='端口号')
    def serve(port):
        """启动服务器"""
        click.echo(f"服务器运行在端口 {port}")

    if __name__ == '__main__':
        cli()
    ```

    运行：

    ```bash
    python script.py init
    python script.py serve --port 8080
    ```

    实际完整示例

    ```python
    import click

    @click.command()
    @click.option('--name', prompt='你的名字', help='问候的对象')
    @click.option('--count', default=1, help='重复次数')
    @click.option('--uppercase', is_flag=True, help='大写输出')
    @click.option('--color', type=click.Choice(['red', 'green', 'blue']), 
                  help='输出颜色')
    def hello(name, count, uppercase, color):
        """
        一个友好的问候程序
        
        示例:
            python hello.py --name Alice --count 3 --uppercase
        """
        message = f"Hello {name}!"
        
        if uppercase:
            message = message.upper()
        
        for i in range(count):
            if color:
                click.secho(message, fg=color)
            else:
                click.echo(message)

    if __name__ == '__main__':
        hello()
    ```

    运行：

    ```bash
    python script.py --name Alice --count 3 --uppercase --color red
    ```

    装饰器链的执行顺序

    注意：装饰器从下往上执行

    ```python
    @click.command()                    # 3. 最后执行
    @click.option('--count', default=1) # 2. 然后执行  
    @click.option('--name')             # 1. 先执行
    def hello(name, count):
        # 函数体
    ```

    这相当于：

    ```python
    def hello(name, count):
        pass

    hello = click.option('--name')(hello)
    hello = click.option('--count', default=1)(hello) 
    hello = click.command()(hello)
    ```

    **与传统 argparse 对比**

    * argparse 方式

        ```python
        import argparse

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument('--name', default='World')
            parser.add_argument('--count', type=int, default=1)
            args = parser.parse_args()
            
            for i in range(args.count):
                print(f"Hello {args.name}!")

        if __name__ == '__main__':
            main()
        ```

    * click 方式

        ```python
        import click

        @click.command()
        @click.option('--name', default='World')
        @click.option('--count', default=1)
        def hello(name, count):
            for i in range(count):
                click.echo(f"Hello {name}!")

        if __name__ == '__main__':
            hello()
        ```

    优势

    * 声明式：参数定义就在函数上方，清晰直观

    * 类型安全：自动类型转换和验证

    * 帮助生成：自动生成帮助文档

    * 功能丰富：支持提示、颜色、文件操作等

    * 易于测试：函数可以像普通函数一样测试

    总结

    `@click.command()` 是 click 库的核心，它：

    * 将 Python 函数转换为 CLI 命令

    * 自动处理参数解析

    * 生成帮助文档

    * 提供丰富的命令行功能

    这种装饰器方式让创建命令行工具变得非常简单和直观！

* python click

    install: `pip install click`

    ```py
    import click

    @click.command()
    @click.option('--name', prompt='你的名字', help='你的名字')
    @click.option('--age', default=18, help='你的年龄')
    @click.option('--verbose', is_flag=True, help='详细模式')
    @click.argument('input_file')
    def main(name, age, verbose, input_file):
        """一个简单的命令行程序"""
        print(f"你好 {name}, 年龄 {age}")
        if verbose:
            print("详细模式已开启")
        print(f"处理文件: {input_file}")

    if __name__ == '__main__':
        main()
    ```

    run: `python main.py [commands]`

    可以看到，使用了 click 后，`main()`函数不需要再处理 argc 和 argv 了，命令行的 arg 直接被 click 处理好，作为参数传入 main() 函数中。

## topics
