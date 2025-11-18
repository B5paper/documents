# click note

## cache

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
