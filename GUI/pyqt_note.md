# PyQt Note

## cache

* PyQt 安装与使用指南

    1. 安装问题：需要原版 Qt 吗？

        答案是：不需要。

        你只需要通过 pip install 安装 PyQt 库即可。这是因为 PyQt 的 Wheel 安装包（.whl 文件）已经包含了运行 PyQt 所必需的 Qt 库的二进制文件。

        * pip install PyQt5 或 pip install PyQt6 会同时安装：

            * PyQt 本身（即 Python 的绑定和模块）。

            * 一份精简版的 Qt 运行时库（包括核心库、GUI 组件等）。

        所以，你不需要去 Qt 官网下载和安装庞大的原版 Qt SDK。这极大地简化了部署过程。

        一个例外情况：

        如果你计划使用 Qt Designer（一个用于通过拖拽方式设计 GUI 界面的可视化工具），那么你需要额外安装它。有两种方式：

        * 安装完整的 Qt SDK（从 Qt 官网下载），其中就包含了 Qt Designer。

        * （更推荐）直接安装 pyqt5-tools 包，它包含了为 PyQt5 定制的 designer.exe 和其他实用工具。

            ```bash
            pip install pyqt5-tools
            ```

            注意：对于 PyQt6，官方没有提供对应的 pyqt6-tools，你需要单独安装 Qt SDK 来获取 Designer。社区有替代方案，但 pyqt5-tools 是目前最方便的选择。

    2. PyQt 基本用法简述

        PyQt 的使用遵循一个典型的模式：创建应用、创建部件、设置布局、连接信号与槽、启动事件循环。

        核心概念：

        * QApplication： 管理整个应用程序的控制流和主要设置。每个 PyQt 程序都必须有且只有一个 QApplication 实例。

        * 窗口部件（Widgets）： 所有用户界面元素的基类，例如按钮（QPushButton）、标签（QLabel）、文本框（QLineEdit）等。

        * 布局（Layouts）： 自动管理窗口部件的排列方式，如水平布局（QHBoxLayout）、垂直布局（QVBoxLayout）等。

        * 信号与槽（Signals & Slots）： PyQt 的核心机制，用于对象之间的通信。当某个事件发生时（如按钮被点击），一个信号会被发射；而这个信号可以被连接到一个槽（一个普通的 Python 函数）上，从而触发相应的操作。

        一个极简的示例代码（PyQt5）

        ```python
        import sys
        from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel

        # 1. 创建应用实例
        app = QApplication(sys.argv)

        # 2. 创建主窗口
        window = QWidget()
        window.setWindowTitle('我的第一个 PyQt 程序')
        window.resize(300, 200)

        # 3. 创建部件和布局
        label = QLabel('你好，世界！')
        button = QPushButton('点击我')

        # 创建一个垂直布局，并添加部件
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(button)

        # 将布局设置到窗口上
        window.setLayout(layout)

        # 4. 定义槽函数（处理按钮点击）
        def on_button_clicked():
            label.setText('按钮被点击了！')

        # 5. 连接信号和槽：当按钮被点击时，调用 on_button_clicked 函数
        button.clicked.connect(on_button_clicked)

        # 6. 显示窗口
        window.show()

        # 7. 进入应用程序的主事件循环
        sys.exit(app.exec_())
        ```

        开发流程总结：

        * 设计界面：可以直接手写代码，或者先用 Qt Designer 拖拽生成一个 .ui 文件。

        * 转换界面文件（如果用了 Designer）：使用 pyuic5 命令行工具将 .ui 文件转换为 .py 文件。

            ```bash
            pyuic5 main_window.ui -o main_window.py
            ```

        * 编写业务逻辑：在你的主程序代码中，导入生成的界面类，并为其上的部件（如按钮）连接信号和槽函数，实现具体功能。

        * 运行：像运行普通 Python 脚本一样运行你的程序。

    版本选择：PyQt5 vs PyQt6

    * PyQt5： 基于 Qt5，非常成熟稳定，社区资源丰富，有 pyqt5-tools 提供便利的工具。

    * PyQt6： 基于 Qt6，是未来的方向。它对一些 API 进行了调整以更符合 Python 的风格，并且默认不支持一些已被标记为废弃的 API。如果你是新手，从 PyQt6 开始也可以，但网上 PyQt5 的教程和解决方案更多。

    建议：对于初学者，从 PyQt5 开始学习曲线会更平缓，因为参考资料更多。

    总结

    * 安装：只需 pip install PyQt5，无需单独安装 Qt。

    * 核心：理解 QApplication、窗口部件、布局、信号与槽。

    * 工具：使用 Qt Designer（通过 pip install pyqt5-tools 获得）可以极大地提高界面设计效率。

    * 起步：从一个简单的例子开始，逐步添加功能，是学习 PyQt 的最佳方式。

## topics
