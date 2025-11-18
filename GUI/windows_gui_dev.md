# gui dev note

## cache

* windows gui dev

    * Win32 API (C/C++)

    * Windows Forms (WinForms) - .NET Framework/.NET

        .NET 平台上的拖拽式快速开发框架。它封装了 Win32 API，提供了大量可视化的控件，通过事件驱动模型开发。

    * Windows Presentation Foundation (WPF) - .NET Framework/.NET

        Microsoft 推出的更现代、更强大的 .NET GUI 框架。它使用 DirectX 进行渲染，实现了界面（XAML）与逻辑（C#/VB.NET）的分离。

        有很多优秀的第三方控件库（如 DevExpress, Telerik）

        学习曲线比 WinForms 稍陡，尤其是 XAML 和 MVVM 模式。

    * Windows App SDK (WinUI 3) - 现代 Windows 原生开发的首选

        这是微软最新的原生 UI 开发平台，旨在统一和取代 Win32 和 UWP 的开发模型。它的 UI 框架是 WinUI 3，是 Fluent Design 系统的最新、最原生的体现。

        Windows App SDK 运行时与操作系统解耦，可以通过 NuGet 包独立更新。

    * Avalonia UI (.NET)

        一个受 WPF 启发的、开源的、跨平台的 .NET UI 框架。使用 XAML，语法和开发体验与 WPF 非常相似。

    * Electron (Web 技术)

        跨平台，使用 JavaScript, HTML 和 CSS 来构建桌面应用。其核心是 Chromium 渲染引擎和 Node.js 运行时。

        开发速度快，可利用海量的 Web 生态（React, Vue, Angular 等），UI 非常灵活。

        内存占用高，打包体积大，性能（尤其是原生操作性能）不如原生框架。

    * Tauri (Rust + Web 技术)

        被认为是 Electron 的替代品。前端使用任何 Web 技术（HTML, JS, CSS），但后端使用 Rust，并且使用操作系统的原生 WebView（在 Windows 上是 WebView2），而不是打包 Chromium。

    * python

        * Tkinter: Python 标准库内置，简单易用但界面老旧。

        * PyQt/PySide: Qt 的 Python 绑定，就是你提到的 Qt 的另一种使用方式。

        * wxPython: 封装了原生控件，外观原生。

        * Kivy: 开源跨平台，适合触摸屏应用，但外观不原生。

    * Rust:

        * egui, Iced: 即时模式（Immediate Mode）GUI，正在快速发展。

        * Slint: 声明式 UI，专为嵌入式和桌面设计。

        * Tauri（上文已提及）：用于构建应用程序外壳，UI 是 Web 技术。

    * Go:

        * Fyne, Walk, GIU: 新兴的 Go 语言 GUI 库，生态在逐步完善。

    * Qt

        是 C++ 跨平台 GUI 的事实标准，非常强大

        库体积较大， licensing 需要注意（LGPL 要求动态链接或购买商业许可）。

    * wxWidgets

        一个成熟的开源 C++ 跨平台框架。哲学是在任何平台上都使用原生控件。

        真正原生外观: 在 Windows 上调用 Win32 API，在 macOS 上调用 Cocoa，在 Linux 上调用 GTK+，因此外观和行为与本地应用完全一致。

        许可友好: 使用宽松的 wxWindows License，可自由用于闭源商业项目。

    * Dear ImGui

        一个非常独特的即时模式 GUI 库，而非传统的保留模式。

        非常适合工具、调试器、游戏编辑器等需要快速迭代UI的内部工具。

        非常轻量，只需要一个 OpenGL/DirectX/Vulkan 的渲染后端。

        不适合通用应用程序: 它的设计初衷不是用来做带有复杂窗口管理、菜单栏、标准对话框的普通桌面应用。


## topics