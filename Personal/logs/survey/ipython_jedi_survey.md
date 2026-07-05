* ipython 如何提示函数中的参数

    在IPython中，有几种方法可以提示函数参数：

    ## 1. **Tab 补全**（最常用）
    在函数名后输入左括号，然后按 Tab 键：
    ```python
    # 输入以下内容后按 Tab
    def my_function(name, age, city):
        pass

    my_function(  # 在这里按 Tab
    ```

    ## 2. **Shift + Tab**（查看文档字符串）
    - 在函数名后输入括号，然后按 **Shift + Tab**
    - 或输入完整的函数调用后，光标在括号内按 **Shift + Tab**
    - 按多次可以展开更多信息

    ```python
    my_function(  # 光标在这里，按 Shift+Tab
    ```

    ## 3. **使用问号查看帮助**
    ```python
    my_function?  # 显示函数签名和文档字符串
    my_function??  # 显示函数源代码
    ```

    ## 4. **%pdef 魔术命令**
    ```python
    %pdef my_function  # 只显示函数签名
    ```

    ## 5. **使用 inspect 模块**
    ```python
    import inspect
    print(inspect.signature(my_function))  # 显示函数签名
    print(inspect.getdoc(my_function))     # 显示文档字符串
    ```

    ## 6. **Jupyter Notebook 中的特殊提示**
    在 Jupyter 中，输入函数名后会自动显示工具提示：
    ```python
    my_function(  # 会自动弹出参数提示
    ```

    ## 7. **配置自动提示**
    在 IPython 配置中启用更丰富的自动提示：
    ```python
    %config IPCompleter.greedy = True  # 贪婪模式自动提示
    ```

    ## 示例演示
    ```python
    def calculate_bmi(weight: float, height: float, unit: str = 'metric') -> float:
        """
        计算身体质量指数 (BMI)
        
        Parameters:
        -----------
        weight : float
            体重
        height : float
            身高
        unit : str, optional
            单位系统 ('metric' 或 'imperial')
        
        Returns:
        --------
        float
            BMI 值
        """
        if unit == 'metric':
            return weight / (height ** 2)
        else:
            return (weight / (height ** 2)) * 703

    # 尝试以下操作：
    calculate_bmi(  # 输入这行后按 Tab
    # 或
    calculate_bmi?  # 查看完整文档
    ```

    ## 实用技巧
    1. **类型注解增强提示**：使用类型注解可以让提示更清晰
    2. **写好文档字符串**：规范的 docstring（如 NumPy/Google 格式）会显示得更好
    3. **参数默认值**：在函数定义中设置默认值会自动显示

    这样你就可以在编写代码时轻松查看函数参数信息了！

* 但是我的显示出来是这样：

    ```
    In [14]: plt.plot(d[:2])
    abs()                     ascii()                   bin()                     BrokenPipeError
    all()                     AssertionError            BlockingIOError           BufferError
    any()                     AttributeError            bool                      bytearray                 >
    ArithmeticError           BaseException             breakpoint()              bytes
    ```

    看起来你的 Tab 补全显示的是全局名称空间中的对象列表，而不是函数的参数提示。这是因为 IPython 的自动补全在错误的上下文中被触发了。

    ## **正确使用 Tab 补全的方法：**

    ### 1. **确保光标位置正确**
    ```python
    # 错误的方式：
    plt.plot(  # 输入完括号后，光标还在这里
      # 这时按 Tab 会列出所有变量/函数

    # 正确的方式：
    plt.plot(d[:2],  # 输入部分参数后，光标在逗号后面
      # 这时按 Tab 会提示下一个参数
    ```

    ### 2. **使用 Shift+Tab（更可靠）**
    ```python
    plt.plot(  # 输入 plt.plot( 然后将光标保持在括号内
               # 按一次 Shift+Tab：显示简短提示
               # 按两次 Shift+Tab：显示详细文档
               # 按三次 Shift+Tab：显示更大窗口
    )
    ```

    ### 3. **清理你的环境**
    有时 IPython 缓存会导致问题：
    ```python
    %reset -f  # 清除所有变量（谨慎使用）
    # 或者重新启动 IPython
    ```

    ### 4. **检查 IPython 配置**
    ```python
    # 查看当前配置
    %config IPCompleter

    # 设置贪婪模式（可能改善补全）
    %config IPCompleter.greedy=True

    # 设置自动提示延迟
    %config TerminalInteractiveShell.timeoutlen=0.5
    ```

    ### 5. **使用参数名提示**
    ```python
    # 先输入参数名，然后按 Tab 补全
    plt.plot(x=d[:2], y=  # 在等号后按 Tab
    ```

    ### 6. **替代方法：使用 help()**
    ```python
    # 在 IPython 中
    help(plt.plot)  # 显示完整的函数文档
    ```

    ### 7. **在 Jupyter Notebook 中的特殊用法**
    如果你在用 Jupyter：
    - 鼠标悬停在函数名上
    - 或者使用 `plt.plot?` 单元格魔术命令

    ### 8. **检查 matplotlib 是否正常安装**
    ```python
    import matplotlib
    print(matplotlib.__version__)  # 查看版本
    print(dir(plt))  # 查看可用的方法
    ```

    ### **示例：正确操作流程**
    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    d = np.random.randn(100)

    # 方法1：使用 Shift+Tab
    plt.plot(  # ← 在这里按 Shift+Tab（光标在括号内）

    # 方法2：输入部分参数后
    plt.plot(d[:10],  # ← 输入逗号后按 Tab

    # 方法3：使用 help
    help(plt.plot)  # 显示完整文档

    # 方法4：IPython 的特殊语法
    plt.plot?      # 简短帮助
    plt.plot??     # 源代码（如果有）
    ```

    ### **如果还是不行，尝试：**
    1. 更新 IPython：
    ```bash
    pip install --upgrade ipython
    ```

    2. 使用更新版本的 matplotlib：
    ```bash
    pip install --upgrade matplotlib
    ```

    3. 尝试在干净的 Python 环境中测试：
    ```python
    # 启动新的 IPython
    ipython --no-banner

    # 测试基本函数
    def test_func(a, b=10):
        """测试函数"""
        return a + b

    test_func(  # 这里按 Tab 或 Shift+Tab
    ```

    最常见的问题是在函数名后的括号内按 Tab，而不是在参数输入位置按 Tab。确保你的光标在正确的位置！

* 输出如下：

    ```
    In [14]: %config IPCompleter
    IPCompleter(Completer) options
    ----------------------------
    IPCompleter.auto_close_dict_keys=<Bool>
        Enable auto-closing dictionary keys.
        When enabled string keys will be suffixed with a final quote (matching the
        opening quote), tuple keys will also receive a separating comma if needed,
        and keys which are final will receive a closing bracket (``]``).
        Current: False
    IPCompleter.auto_import_method=<DottedObjectName>
        Provisional:
            This is a provisional API in IPython 9.3, it may change without warnings.
        A fully qualified path to an auto-import method for use by completer. The
        function should take a single string and return `ModuleType` and can raise
        `ImportError` exception if module is not found.
        The default auto-import implementation does not populate the user namespace
        with the imported module.
        Current: 'importlib.import_module'
    IPCompleter.backslash_combining_completions=<Bool>
        Enable unicode completions, e.g. \alpha<tab> . Includes completion of latex
        commands, unicode names, and expanding unicode characters back to latex
        commands.
        Current: True
    IPCompleter.debug=<Bool>
        Enable debug for the Completer. Mostly print extra information for
        experimental jedi integration.
        Current: False
    IPCompleter.disable_matchers=<list-item-1>...
        List of matchers to disable.
                The list should contain matcher identifiers (see
        :any:`completion_matcher`).
        Current: []
    IPCompleter.evaluation=<Enum>
        Policy for code evaluation under completion.
                Successive options allow to enable more eager evaluation for better
                completion suggestions, including for nested dictionaries, nested lists,
                or even results of function calls.
                Setting ``unsafe`` or higher can lead to evaluation of arbitrary user
                code on :kbd:`Tab` with potentially unwanted or dangerous side effects.
                Allowed values are:
                - ``forbidden``: no evaluation of code is permitted,
                - ``minimal``: evaluation of literals and access to built-in namespace;
                no item/attribute evaluation, no access to locals/globals,
                no evaluation of any operations or comparisons.
                - ``limited``: access to all namespaces, evaluation of hard-coded methods
                (for example: :py:meth:`dict.keys`, :py:meth:`object.__getattr__`,
                :py:meth:`object.__getitem__`) on allow-listed objects (for example:
                :py:class:`dict`, :py:class:`list`, :py:class:`tuple`, ``pandas.Series``),
                - ``unsafe``: evaluation of all methods and function calls but not of
                syntax with side-effects like `del x`,
                - ``dangerous``: completely arbitrary evaluation; does not support auto-import.
                To override specific elements of the policy, you can use
        ``policy_overrides`` trait.
        Choices: any of ['forbidden', 'minimal', 'limited', 'unsafe', 'dangerous']
        Current: 'limited'
    IPCompleter.greedy=<Bool>
        Activate greedy completion.
                .. deprecated:: 8.8
                    Use :std:configtrait:`Completer.evaluation` and :std:configtrait:`Completer.auto_close_dict_keys` instead.
                When enabled in IPython 8.8 or newer, changes configuration as
        follows:
                - ``Completer.evaluation = 'unsafe'``
                - ``Completer.auto_close_dict_keys = True``
        Current: False
    IPCompleter.jedi_compute_type_timeout=<Int>
        Experimental: restrict time (in milliseconds) during which Jedi can compute types.
                Set to 0 to stop computing types. Non-zero value lower than 100ms may hurt
                performance by preventing jedi to build its cache.
        Current: 400
    IPCompleter.limit_to__all__=<Bool>
        DEPRECATED as of version 5.0.
        Instruct the completer to use __all__ for the completion
        Specifically, when completing on ``object.<tab>``.
        When True: only those names in obj.__all__ will be included.
        When False [default]: the __all__ attribute is ignored
        Current: False
    IPCompleter.merge_completions=<Bool>
        Whether to merge completion results into a single list
                If False, only the completion results from the first non-empty
                completer will be returned.
                As of version 8.6.0, setting the value to ``False`` is an alias for:
                ``IPCompleter.suppress_competing_matchers = True.``.
        Current: True
    IPCompleter.omit__names=<Enum>
        Instruct the completer to omit private method names
                Specifically, when completing on ``object.<tab>``.
                When 2 [default]: all names that start with '_' will be excluded.
                When 1: all 'magic' names (``__foo__``) will be excluded.
                When 0: nothing will be excluded.
        Choices: any of [0, 1, 2]
        Current: 2
    IPCompleter.policy_overrides=<key-1>=<value-1>...
        Overrides for policy evaluation.
                For example, to enable auto-import on completion specify:
                .. code-block::
                    ipython --Completer.policy_overrides='{"allow_auto_import":
        True}' --Completer.use_jedi=False
        Current: {}
    IPCompleter.profile_completions=<Bool>
        If True, emit profiling data for completion subsystem using cProfile.
        Current: False
    IPCompleter.profiler_output_dir=<Unicode>
        Template for path at which to output profile data for completions.
        Current: '.completion_profiles'
    IPCompleter.suppress_competing_matchers=<Union>
        Whether to suppress completions from other *Matchers*.
        When set to ``None`` (default) the matchers will attempt to auto-detect
        whether suppression of other matchers is desirable. For example, at the
        beginning of a line followed by `%` we expect a magic completion to be the
        only applicable option, and after ``my_dict['`` we usually expect a
        completion with an existing dictionary key.
        If you want to disable this heuristic and see completions from all matchers,
        set ``IPCompleter.suppress_competing_matchers = False``. To disable the
        heuristic for specific matchers provide a dictionary mapping:
        ``IPCompleter.suppress_competing_matchers = {'IPCompleter.dict_key_matcher':
        False}``.
        Set ``IPCompleter.suppress_competing_matchers = True`` to limit completions
        to the set of matchers with the highest priority; this is equivalent to
        ``IPCompleter.merge_completions`` and can be beneficial for performance, but
        will sometimes omit relevant candidates from matchers further down the
        priority list.
        Current: None
    IPCompleter.use_jedi=<Bool>
        Experimental: Use Jedi to generate autocompletions. Default to True if jedi
        is installed.
        Current: True
    ```

    看起来你的 IPython 配置是正常的。让我帮你诊断一下具体问题。

    ## **问题诊断和解决方案**

    ### 1. **首先，验证最基本的功能**
    ```python
    # 创建一个简单的测试函数
    def test_func(a, b=10, c=None):
        """这是一个测试函数"""
        return a + b

    # 测试1：在括号内按 Shift+Tab
    test_func(  # ← 光标在这里按 Shift+Tab（不要按 Tab）
    ```

    ### 2. **检查你的实际 matplotlib 问题**
    ```python
    # 先确认 matplotlib 的 plot 函数是否正常
    import matplotlib.pyplot as plt

    # 方法1：使用问号
    plt.plot?  # 这会显示什么？

    # 方法2：检查 plot 的类型
    print(type(plt.plot))  # 应该显示 <function ...>

    # 方法3：使用 help()
    help(plt.plot)
    ```

    ### 3. **可能是 Jedi 补全的问题**
    尝试禁用 Jedi：
    ```python
    %config IPCompleter.use_jedi = False
    # 然后重试
    plt.plot(  # Shift+Tab
    ```

    如果禁用 Jedi 后正常了，说明是 Jedi 的问题，可以：
    ```bash
    pip install --upgrade jedi  # 更新 jedi
    # 或
    pip install 'jedi>=0.18.0'  # 安装特定版本
    ```

    ### 4. **检查你的 IPython 版本**
    ```python
    import IPython
    print(IPython.__version__)  # 应该 >= 7.0

    # 如果是旧版本，升级：
    # !pip install --upgrade ipython
    ```

    ### 5. **尝试不同的输入方式**
    ```python
    import numpy as np
    d = np.random.randn(100)

    # 方式 A：直接输入完整函数
    plt.plot(d[:10])  # 先正常执行

    # 方式 B：然后重新尝试
    plt.plot(  # ← 现在再试 Shift+Tab

    # 方式 C：使用完整路径
    from matplotlib.axes._axes import Axes
    help(Axes.plot)
    ```

    ### 6. **临时修复：使用 inspect 模块**
    ```python
    import inspect
    import matplotlib.pyplot as plt

    # 获取函数签名
    sig = inspect.signature(plt.plot)
    print("函数签名:", sig)

    # 获取参数详细信息
    params = inspect.signature(plt.plot).parameters
    for name, param in params.items():
        print(f"{name}: {param}")
    ```

    ### 7. **创建快捷函数**
    在你的 IPython 配置文件中添加：
    ```python
    # ~/.ipython/profile_default/ipython_config.py
    # 或启动时运行
    def show_params(func):
        """显示函数参数"""
        import inspect
        sig = inspect.signature(func)
        print(f"函数: {func.__name__}")
        print(f"签名: {sig}")
        print("\n参数详情:")
        for name, param in sig.parameters.items():
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
            print(f"  {name}{annotation}{default}")

    # 使用
    show_params(plt.plot)
    ```

    ### 8. **如果以上都不行，尝试：**
    ```python
    # 1. 清理 matplotlib 缓存
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # 2. 重新导入
    import importlib
    importlib.reload(matplotlib.pyplot)

    # 3. 使用原始的 pyplot 函数
    from matplotlib import pyplot as plt_orig
    plt_orig.plot(  # Shift+Tab
    ```

    ### 9. **最后的手段：检查终端/IDE 设置**
    如果你在使用终端：
    ```bash
    # 检查 TERM 环境变量
    echo $TERM

    # 尝试不同的终端
    # 比如使用 xterm 或 gnome-terminal
    ```

    如果你在使用 IDE（如 VS Code、PyCharm）：
    - 确保 IPython 集成正常
    - 检查 IDE 的 Python 解释器设置
    - 尝试在纯终端中运行 IPython

    ### **告诉我你的具体情况：**
    1. 你使用的 IPython 版本是多少？
    2. 是在终端还是 Jupyter 中？
    3. 执行 `plt.plot?` 的输出是什么？
    4. 执行 `print(type(plt.plot))` 的输出是什么？

    这样我能更好地帮你定位问题！

* 禁用 jedi 后正常了，但是 jedi 已经是最新版本：

    ```
    $ pip install --upgrade jedi -i https://mirrors.aliyun.com/pypi/simple
    Looking in indexes: https://mirrors.aliyun.com/pypi/simple
    Requirement already satisfied: jedi in d:\miniconda\envs\sci\lib\site-packages (0.19.2)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in d:\miniconda\envs\sci\lib\site-packages (from jedi) (0.8.5)
    ```

    好的，既然禁用 Jedi 后正常，说明是 Jedi 与你的环境兼容性问题。Jedi 0.19.2 在某些配置下可能会有问题。让我给你几种解决方案：

    ## **解决方案**

    ### **方案1：降级 Jedi 版本**
    某些情况下，Jedi 0.19.x 版本有问题，降级到 0.18.x 可能解决：
    ```bash
    pip install jedi==0.18.2 -i https://mirrors.aliyun.com/pypi/simple
    ```
    然后重新启用 Jedi：
    ```python
    %config IPCompleter.use_jedi = True
    ```

    ### **方案2：配置 Jedi 参数**
    ```python
    # 在 IPython 启动时配置
    %config IPCompleter.use_jedi = True
    %config IPCompleter.jedi_compute_type_timeout = 100  # 降低超时时间
    ```

    ### **方案3：创建 IPython 配置文件**
    1. 生成配置文件：
    ```bash
    ipython profile create
    ```

    2. 编辑配置文件 `~/.ipython/profile_default/ipython_config.py`，添加：
    ```python
    c = get_config()

    # Jedi 配置
    c.Completer.use_jedi = True
    c.Completer.jedi_compute_type_timeout = 200
    c.Completer.evaluation = 'limited'  # 或 'minimal'

    # 对于 Windows 环境，可能需要额外的设置
    import sys
    if sys.platform == 'win32':
        c.TerminalInteractiveShell.windows_enable_virtual_terminal = True
    ```

    ### **方案4：临时修复 - 启用 Jedi 但调整设置**
    ```python
    # 在 IPython 会话中运行
    from IPython.core.completer import IPCompleter
    from jedi import settings

    # 调整 Jedi 设置
    settings.case_insensitive_completion = False
    settings.autocompletion_in_modules = {'matplotlib': False}

    # 重新配置 completer
    %config IPCompleter.use_jedi = True
    %config IPCompleter.debug = True  # 开启调试查看问题
    ```

    ### **方案5：使用混合模式**
    创建一个自定义的解决方案，只在需要时使用 Jedi：
    ```python
    # 在 IPython 启动脚本中 (~/.ipython/profile_default/startup/00-custom.py)
    import IPython

    def jedi_aware_completion():
        """智能切换 Jedi 完成"""
        try:
            # 尝试使用 Jedi
            IPython.get_ipython().Completer.use_jedi = True
            print("Jedi 补全已启用")
        except Exception as e:
            IPython.get_ipython().Completer.use_jedi = False
            print(f"Jedi 补全禁用，使用默认补全: {e}")

    jedi_aware_completion()
    ```

    ### **方案6：检查环境特定问题**
    对于 Windows/MinGW/Conda 环境，可能有路径问题：
    ```python
    import os
    import sys

    # 检查 Python 路径
    print(f"Python 路径: {sys.executable}")
    print(f"Jedi 路径: {os.path.dirname(os.path.abspath(__file__))}")

    # 临时修复：添加额外路径
    sys.path.insert(0, r'D:\miniconda\envs\sci\Lib\site-packages')
    ```

    ### **方案7：替代方案 - 使用其他补全引擎**
    如果你主要需要 matplotlib 的补全，可以：
    ```python
    # 1. 创建 matplotlib 特定的补全提示
    def mpl_complete():
        """matplotlib 函数参数帮助"""
        import inspect
        import matplotlib.pyplot as plt
        
        mpl_funcs = {
            'plot': plt.plot,
            'scatter': plt.scatter,
            'bar': plt.bar,
            'hist': plt.hist,
        }
        
        def show_params(func_name):
            if func_name in mpl_funcs:
                sig = inspect.signature(mpl_funcs[func_name])
                print(f"{func_name}{sig}")
        
        return show_params

    mpl_help = mpl_complete()
    mpl_help('plot')  # 显示 plot 的参数
    ```

    ### **方案8：永久禁用 Jedi 但保持功能**
    如果你决定永久禁用 Jedi，可以创建启动脚本：
    ```python
    # ~/.ipython/profile_default/ipython_config.py
    c = get_config()
    c.Completer.use_jedi = False
    c.Completer.evaluation = 'limited'
    c.Completer.auto_close_dict_keys = True
    ```

    ### **方案9：尝试开发版 Jedi**
    有时开发版修复了问题：
    ```bash
    # 从 GitHub 安装开发版
    pip install git+https://github.com/davidhalter/jedi.git

    # 或使用预发布版
    pip install --pre jedi -i https://mirrors.aliyun.com/pypi/simple
    ```

    ### **方案10：诊断脚本**
    运行这个诊断脚本来找出具体问题：
    ```python
    def diagnose_jedi():
        """诊断 Jedi 问题"""
        import jedi
        import matplotlib.pyplot as plt
        
        print(f"Jedi 版本: {jedi.__version__}")
        print(f"Python 版本: {sys.version}")
        
        try:
            # 测试 Jedi 基本功能
            script = jedi.Interpreter('plt.plot(', [{'plt': plt}])
            completions = script.complete()
            print(f"Jedi 补全数量: {len(completions)}")
            for comp in completions[:5]:
                print(f"  - {comp.name}: {comp.type}")
        except Exception as e:
            print(f"Jedi 错误: {e}")
        
        # 测试 matplotlib 的具体问题
        print("\n测试 matplotlib.pyplot.plot:")
        print(f"函数对象: {plt.plot}")
        print(f"模块: {plt.plot.__module__}")

    diagnose_jedi()
    ```

    ### **推荐方案**
    对于你的情况，我建议按顺序尝试：
    1. **先降级 Jedi**：`pip install jedi==0.18.2`
    2. **如果不行，创建 IPython 配置文件**（方案3）
    3. **最后考虑永久禁用 Jedi**，但使用 `inspect` 模块作为补充

    ### **对于 matplotlib 特定问题**
    你还可以使用 matplotlib 自带的帮助：
    ```python
    # matplotlib 有自己的参数帮助系统
    import matplotlib as mpl
    print(mpl.rcParams.keys())  # 查看可用配置

    # 使用 matplotlib 的在线文档
    import webbrowser
    webbrowser.open("https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html")
    ```

    ### **快捷命令**
    在你的 IPython 配置中添加：
    ```python
    # ~/.ipython/profile_default/startup/00-mpl_helper.py
    def mplsig(func_name):
        """快速查看 matplotlib 函数签名"""
        import inspect
        import matplotlib.pyplot as plt
        
        if hasattr(plt, func_name):
            func = getattr(plt, func_name)
            print(f"plt.{func_name}{inspect.signature(func)}")
        else:
            print(f"函数 plt.{func_name} 不存在")
    ```

    这样你可以随时使用 `mplsig('plot')` 查看参数。

    选择最适合你工作流的方案吧！

* jedi 是干嘛用的

    **Jedi** 是一个 Python 的**自动补全和静态分析库**。让我详细解释它的作用和重要性：

    ## **Jedi 的主要功能**

    ### **1. 智能代码补全**
    ```python
    # 传统补全（无 Jedi）：
    # 只能补全当前命名空间中存在的名称

    # Jedi 补全：
    import numpy as np
    np.array(  # Jedi 可以：
        # 1. 补全 array 函数的参数
        # 2. 根据上下文提供类型感知的补全
        # 3. 补全复杂表达式中的方法
    )
    ```

    ### **2. 函数签名提示**
    ```python
    # 无 Jedi：可能只显示函数名
    # 有 Jedi：显示完整签名，包括参数名、默认值、类型注解
    def process_data(data: pd.DataFrame, normalize=True, axis=0):
        pass

    process_data(  # Jedi 会显示：
        # process_data(data: pandas.core.frame.DataFrame, normalize: bool = True, axis: int = 0)
    )
    ```

    ### **3. 转到定义（Go to Definition）**
    ```python
    # 可以跳转到函数/类的定义处
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()  # Ctrl+点击可以跳转到源码
    ```

    ### **4. 代码重构支持**
    ```python
    # 重命名符号时，Jedi 可以找到所有引用
    class MyClass:  # 重命名这个类时，Jedi 会找到所有使用的地方
        pass
    ```

    ## **Jedi 的工作原理**

    ### **静态分析 vs 动态分析**
    ```python
    # 传统 IPython 补全（动态）：
    x = some_function()  # 需要实际执行代码才知道类型
    x.  # 按 Tab：需要运行 some_function() 才能知道 x 的方法

    # Jedi（静态）：
    # 通过分析代码结构，不需要执行就能推断类型
    # 即使 some_function() 会报错，也能提供补全
    ```

    ### **类型推断**
    ```python
    import pandas as pd

    df = pd.read_csv('data.csv')
    # Jedi 能推断出 df 是 DataFrame，从而提供正确的补全
    df.gr  # 可以补全为 groupby, not gratings
    ```

    ## **Jedi 在 IDE/编辑器中的应用**

    ### **支持的编辑器**
    1. **VS Code**：Python 扩展使用 Jedi
    2. **PyCharm**：有自己的实现，但原理类似
    3. **Sublime Text**：通过插件支持
    4. **Vim/Neovim**：jedi-vim 插件
    5. **Emacs**：jedi.el

    ### **在 IPython/Jupyter 中的优势**
    ```python
    # 复杂场景下的优势：
    class DataProcessor:
        def __init__(self):
            self.data = pd.DataFrame()
        
        def process(self):
            return self.data.groupby('category')

    processor = DataProcessor()
    result = processor.process()
    result.  # Jedi 能推断出 result 是 DataFrameGroupBy
    ```

    ## **Jedi 的局限性**

    ### **1. 动态特性的挑战**
    ```python
    # Jedi 难以处理的情况：
    # 动态属性设置
    obj = type('Dynamic', (), {})()
    setattr(obj, 'dynamic_method', lambda x: x)
    obj.  # Jedi 可能不知道 dynamic_method 存在

    # 元编程
    class Meta(type):
        def __getattr__(self, name):
            return lambda: None

    class Dynamic(metaclass=Meta):
        pass

    Dynamic.  # Jedi 可能无法补全动态生成的方法
    ```

    ### **2. 性能考虑**
    ```python
    # 大型代码库中，Jedi 的初始化可能较慢
    # 这就是为什么有 jedi_compute_type_timeout 设置
    %config IPCompleter.jedi_compute_type_timeout = 400  # 毫秒
    ```

    ### **3. 第三方库支持**
    ```python
    # 某些库的复杂结构可能让 Jedi 困惑
    import tensorflow as tf
    # TensorFlow 的动态图机制可能让 Jedi 难以准确推断类型
    ```

    ## **Jedi 的替代方案**

    ### **1. Microsoft Python Language Server**
    ```python
    # VS Code 默认使用 Pylance（基于 Pyright）
    # 更快的补全，更好的类型检查
    ```

    ### **2. Rope**
    ```python
    # 另一个 Python 重构库
    # 在某些场景下可能比 Jedi 更快
    ```

    ### **3. IPython 的内置补全**
    ```python
    # 你刚刚体验的：禁用 Jedi 后的补全
    # 基于运行时 introspection，更准确但需要代码可执行
    ```

    ## **为什么你的环境有问题？**

    ### **可能的原因**
    ```python
    # 1. Windows 路径问题
    # Jedi 可能对 Windows 的反斜杠路径处理有问题

    # 2. Conda 环境隔离
    # Jedi 可能没有正确识别环境中的包

    # 3. matplotlib 的特殊结构
    # matplotlib 使用大量的 __getattr__ 和动态导入
    # Jedi 可能难以解析这种结构

    # 4. Jedi 0.19.x 的 bug
    # 特定版本的兼容性问题
    ```

    ## **如何决定是否使用 Jedi？**

    ### **使用 Jedi 的情况**
    ```python
    # 适合使用 Jedi：
    # 1. 大型项目，需要代码导航
    # 2. 写库代码，需要查看函数签名
    # 3. 学习新库，需要探索 API
    # 4. 代码重构需求多
    ```

    ### **不使用 Jedi 的情况**
    ```python
    # 适合禁用 Jedi：
    # 1. 主要做数据分析和探索
    # 2. 使用大量动态特性的库（如 matplotlib, tensorflow）
    # 3. 性能要求高，需要快速补全
    # 4. 遇到兼容性问题时
    ```

    ## **实用建议**

    ### **对于数据分析工作流**
    ```python
    # 推荐配置：
    %config IPCompleter.use_jedi = False  # 禁用 Jedi
    %config IPCompleter.greedy = True     # 启用贪婪补全
    %config IPCompleter.evaluation = 'limited'  # 有限评估

    # 使用辅助函数
    def sig(func):
        """快速查看函数签名"""
        import inspect
        print(f"{func.__name__}{inspect.signature(func)}")

    sig(plt.plot)  # 手动查看签名
    ```

    ### **对于开发工作流**
    ```python
    # 使用支持 Jedi 的 IDE（如 VS Code）
    # 在 IPython 中禁用 Jedi，避免冲突
    # 利用 IDE 的更好集成
    ```

    **总结**：Jedi 是一个强大的静态分析工具，但在某些环境（特别是 Windows + Conda + 特定库组合）下可能有问题。对于你的情况，如果主要做数据科学/分析，禁用 Jedi 并使用 IPython 的内置补全可能更稳定。
