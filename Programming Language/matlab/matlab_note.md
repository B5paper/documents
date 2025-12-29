# Matlab Note

* matlab 2025b x64 破解版下载

    <https://www.puresys.net/8739.html>

## cache

* matlab 匿名函数

    主要用于创建简单的单行函数。

    基本语法

    ```matlab
    函数句柄 = @(输入参数) 表达式
    ```

    **特点**

    * 无需函数文件：直接在工作空间或脚本中定义

    * 单表达式：通常只包含一个表达式（复杂逻辑需用括号组合）

    * 无函数名：通过函数句柄调用

    * 可捕获变量：可以访问定义时的局部变量

    **示例**

    * 基础示例

        ```matlab
        % 平方函数
        square = @(x) x^2;
        result = square(5);  % 返回 25

        % 多参数函数
        add = @(a, b) a + b;
        sum = add(3, 4);  % 返回 7

        % 无参数函数
        getPi = @() pi;
        value = getPi();  % 返回 3.1416
        ```

    * 数组操作

        ```matlab
        % 向量化操作
        scaleVector = @(v, factor) v * factor;
        scaled = scaleVector([1, 2, 3], 2);  % 返回 [2, 4, 6]

        % 元素级操作
        cube = @(x) x.^3;  % 注意使用 .^ 进行元素级运算
        ```

    * 在函数中使用

        ```matlab
        % 作为其他函数的参数
        data = [1, 2, 3, 4, 5];
        squared = arrayfun(@(x) x^2, data);  % 对每个元素平方

        % 在积分函数中使用
        f = @(x) sin(x).^2;
        integral(f, 0, pi)  % 计算积分
        ```

    * 捕获工作空间变量

        ```matlab
        a = 10;
        b = 5;
        linearFunc = @(x) a*x + b;  % 捕获 a 和 b
        result = linearFunc(2);  % 10*2 + 5 = 25
        ```

    **高级用法**

    * 多行匿名函数（需要显式返回）
    
        ```matlab
        % 使用括号和逗号分隔多个表达式
        complexFunc = @(x) ( ...
            y = x^2;       ...
            z = y + 10;    ...
            z);            % 最后一条表达式作为返回值

        result = complexFunc(3);  % 3^2 + 10 = 19
        ```

    * 函数组合

        ```matlab
        f = @(x) x^2;
        g = @(x) x + 1;
        compose = @(x) f(g(x));  % 组合函数 (x+1)^2
        ```

    * 返回多个值

        可以通过返回数组或结构体实现。

        ```matlab
        % 返回数组
        getMinMax = @(x) [min(x), max(x)];
        result = getMinMax([1, 5, 3, 8, 2]);  % 返回 [1, 8]
        [minVal, maxVal] = deal(result(1), result(2));

        % 返回元胞数组
        getStats = @(x) {mean(x), std(x), length(x)};
        stats = getStats([1:10]);
        [avg, stdev, n] = deal(stats{:});

        % 返回结构体
        getInfo = @(x) struct('sum', sum(x), 'product', prod(x));
        info = getInfo([2, 3, 4]);
        info.sum      % 9
        info.product  % 24
        ```

    **注意事项**

    * 性能：对于简单操作，匿名函数通常比普通函数更快

    * 调试：匿名函数较难调试，复杂逻辑建议使用普通函数

    * 作用域：匿名函数会捕获定义时的变量快照

    * 内存：函数句柄是变量，会占用工作空间内存

    **实际应用场景**

    * 快速测试简单数学表达式

    * 作为回调函数传递给其他函数

    * 在 GUI 中定义事件处理函数

    * 函数式编程中的临时函数

    * 优化和数值计算中的目标函数

* MATLAB 字符串

    一、字符串类型

    1. 字符数组（传统）

    ```matlab
    str1 = 'Hello World';  % 单引号，字符数组
    str2 = ['Hello' ' World'];  % 拼接
    ```

    2. 字符串数组（R2016b+）

    ```matlab
    str3 = "Hello World";  % 双引号，字符串类型
    str_array = ["apple", "banana", "cherry"];  % 字符串数组
    ```

    example:

    ```matlab
    >> a = 'hello, world'

    a =

        'hello, world'

    >> class(a)

    ans =

        'char'

    >> b = "hello, world"

    b = 

        "hello, world"

    >> class(b)

    ans =

        'string'
    ```

    二、基本操作

    1. 创建与拼接

        ```matlab
        % 拼接
        str = strcat('Hello', ' ', 'World');
        str = ['Hello' ' ' 'World'];  % 水平拼接
        str = "Hello" + " " + "World";  % R2016b+

        % 格式化
        str = sprintf('Value: %.2f', 3.14159);  % Value: 3.14
        ```

    2. 索引与访问

        ```matlab
        str = 'MATLAB Programming';
        first_char = str(1);      % 'M'
        first_word = str(1:6);    % 'MATLAB'
        last_char = str(end);     % 'g'
        ```

    三、常用字符串函数

    1. 查找与替换

        ```matlab
        str = 'The quick brown fox';

        % 查找
        idx = strfind(str, 'quick');  % 返回起始位置
        idx = contains(str, 'fox');    % 返回逻辑值

        % 替换
        new_str = strrep(str, 'brown', 'red');

        % 正则表达式
        tokens = regexp(str, '(\w+)\s(\w+)', 'tokens');
        ```

    2. 分割与连接

        ```matlab
        % 分割
        parts = strsplit('a,b,c,d', ',');  % {'a','b','c','d'}

        % 连接
        str = strjoin({'a','b','c'}, '-');  % 'a-b-c'

        % 按行分割
        lines = splitlines("Line1\nLine2");
        ```

    3. 转换与比较

        ```matlab
        % 大小写转换
        upper_str = upper('hello');  % 'HELLO'
        lower_str = lower('HELLO');  % 'hello'

        % 类型转换
        num_str = num2str(123);      % '123'
        str_num = str2double('123'); % 123

        % 比较
        is_equal = strcmp('text1', 'text2');      % 严格比较
        is_similar = strcmpi('HELLO', 'hello');   % 忽略大小写
        ```

    四、字符串数组操作

    1. 批量处理

        ```matlab
        % 创建字符串数组
        files = ["data1.csv", "data2.csv", "log.txt"];

        % 批量判断
        is_csv = endsWith(files, ".csv");  % [true, true, false]

        % 批量提取
        names = extractBefore(files, ".");  % ["data1", "data2", "log"]
        ```

    2. 模式匹配

        ```matlab
        str = ["test_001.mat", "data_002.csv", "log.txt"];

        % 提取数字部分
        nums = extract(str, digitsPattern);  % ["001", "002", ""]

        % 匹配特定模式
        matches = contains(str, ["test", "data"]);  % [true, true, false]
        ```

    五、实用技巧

    1. 多行字符串

        ```matlab
        % 使用双引号（R2017a+）
        multi_str = "First line" + newline + "Second line";

        % 使用字符数组
        multi_str = sprintf('Line 1\nLine 2\nLine 3');
        ```

    2. 路径操作

        ```matlab
        % 构建路径
        full_path = fullfile('folder', 'subfolder', 'file.txt');

        % 提取部分
        [filepath, name, ext] = fileparts('/path/to/file.txt');
        ```

    3. 性能优化

        ```matlab
        % 避免在循环中拼接（慢）
        result = '';
        for i = 1:1000
            result = [result num2str(i)];  % 每次重新分配内存
        end

        % 使用字符串数组或 cell（快）
        strs = strings(1000, 1);
        for i = 1:1000
            strs(i) = string(i);
        end
        result = strjoin(strs, '');
        ```

    六、重要注意事项

    * 字符数组 vs 字符串类型

        * 字符数组：'text'，适用于单个字符串

        * 字符串类型："text"，支持数组操作和 Unicode

    * 空字符串的不同表示

        ```matlab
        empty_char = '';      % 字符数组
        empty_str = "";       % 字符串类型
        missing_str = missing; % 缺失值
        ```

    * 中文字符处理

        ```matlab
        str = '中文测试';  % 正确
        length(str)        % 返回字符数，不是字节数
        ```

    MATLAB 字符串功能强大，建议在 R2016b 及以上版本中使用字符串类型（双引号），它提供了更现代、更安全的API，并支持向量化操作。

* MATLAB 函数跳转

    * Ctrl + 点击函数名

        有时候会失效，不清楚原因

    * 光标移动到函数名上，ctrl + D

    * 右键点击函数名 -> 选择 "打开函数名" 或 "转至定义"

    * F12 - 跳转到定义, Ctrl + F12 - 返回（跳回原位置）

    * `Ctrl` + `=`: 跳转到定义, `Ctrl` + `-`: 返回

    * 使用命令

        ```matlab
        open functionName  % 在当前编辑器打开
        edit functionName  % 在编辑器中打开
        type functionName  % 在命令窗口显示
        ```

    注意事项：

    * 函数必须在MATLAB路径中才能跳转

    * 对于内置函数，会打开MATLAB文档

    * 匿名函数和局部函数也可以跳转

    * 可以跳转到类方法、属性等

* matlab `split()`

    将字符串数组（String Array）或字符向量（Character Vector）按分隔符拆分为多个子串（字符串列向量）。

    syntax:

    ```matlab
    % 默认拆分：按空白字符（空格、换行、制表符）进行拆分。
    newStr = split(str)

    % 指定分隔符：按照你定义的 delimiter（如逗号、分号或特定单词）拆分。
    newStr = split(str, delimiter)

    % 多重分隔符：同时匹配多个不同的分隔符进行拆分。
    newStr = split(str, [d1, d2])
    ```

    example:

    * 默认按空格拆分

        ```matlab
        str = "MATLAB is a powerful tool";
        words = split(str)
        % 输出:
        % "MATLAB"
        % "is"
        % "a"
        % "powerful"
        % "tool"
        ```

    * 指定分隔符

        ```matlab
        str = "apple,banana,orange";
        fruits = split(str, ",")
        % 输出:
        % "apple"
        % "banana"
        % "orange"
        ```

    * 使用多个分隔符

        如果你有一串复杂的地址信息，包含多种符号：

        ```matlab
        str = "Red;Blue,Green";
        colors = split(str, [";", ","])
        % 输出:
        % "Red"
        % "Blue"
        % "Green"
        ```

    注意事项

    * 返回类型：无论输入是字符向量 '' 还是字符串 ""，split() 通常都会返回一个字符串数组。

    * 维度变化：如果输入是一个字符串矩阵（多行），split 会在第三维度上进行拆分。

    * 与 strsplit 的区别：strsplit 是较老版本中常用的函数，主要针对字符向量并返回元胞数组（Cell Array）；而 split 是为了配合现代 MATLAB 的 string 数据类型设计的，效率更高且用法更直观。

* `set(findall(ax, '-property', 'PickableParts'), 'PickableParts', 'none');`可以降低 matlab figure 在大量数据时 cpu 占用率高的问题

    但是`set(findall(ax, '-property', 'HitTest'), 'HitTest', 'off');`这个不太有用。不清楚为什么。

* matlab legend()

    legend() 函数用于在 MATLAB 图形中添加图例，解释不同数据系列的含义。

    基本语法

    ```matlab
    legend('label1', 'label2', ..., 'labelN')
    legend({'label1', 'label2', ..., 'labelN'})  % 使用元胞数组
    ```

    常见用法示例

    1. 基础用法
    
        ```matlab
        x = 0:0.1:2*pi;
        y1 = sin(x);
        y2 = cos(x);

        plot(x, y1, 'r-', x, y2, 'b--');
        legend('sin(x)', 'cos(x)');
        ```

    2. 指定位置

        ```matlab
        legend('sin(x)', 'cos(x)', 'Location', 'best');           % 自动选择最佳位置
        legend('sin(x)', 'cos(x)', 'Location', 'northwest');      % 左上角
        legend('sin(x)', 'cos(x)', 'Location', 'northeast');      % 右上角
        legend('sin(x)', 'cos(x)', 'Location', 'southoutside');   % 底部外侧
        ```

    3. 高级选项

        ```matlab
        legend('sin(x)', 'cos(x)', ...
            'FontSize', 12, ...              % 字体大小
            'TextColor', 'blue', ...         % 文本颜色
            'Box', 'off', ...                % 关闭边框
            'Orientation', 'horizontal');    % 水平排列
        ```

    4. 自动获取句柄

        ```matlab
        h1 = plot(x, sin(x), 'r-');
        hold on;
        h2 = plot(x, cos(x), 'b--');
        legend([h1, h2], {'正弦函数', '余弦函数'});
        ```

    5. 动态更新

        ```matlab
        % 创建图形后动态修改
        lgd = legend('sin(x)', 'cos(x)');
        lgd.Title.String = '三角函数';          % 添加图例标题
        lgd.NumColumns = 2;                     % 分两列显示
        ```

    重要参数说明

    | 参数 | 说明 |
    | - | - |
    | `'Location'` | 位置：'best', 'north', 'south', 'east', 'west' 等 |
    | `'NumColumns'` | 列数 |
    | `'Box'` | 边框：'on'（默认）或 'off' |
    | `'FontSize'` | 字体大小 |
    | `'Orientation'` | 方向：'vertical'（默认）或 'horizontal' |

    注意事项

    * 图例顺序与绘图顺序一致

    * 使用 legend('off') 或 legend([]) 移除图例

    * 在多个子图中，每个子图需要单独添加图例

    * R2016b+ 支持 'AutoUpdate' 控制是否自动更新

    最佳实践：在完成所有绘图操作后再添加图例，确保所有数据系列都被正确标注。

* matlab 中，设置 3d figure 的长宽高比例

    * axis equal 和相关命令

        ```matlab
        % 创建3D图形
        [X,Y,Z] = peaks(30);
        surf(X,Y,Z);

        % 设置不同比例选项
        axis equal;        % 三个坐标轴等比例
        axis square;       % 使坐标框呈正方形
        axis tight;        % 紧凑模式，贴合数据范围
        axis vis3d;        % 保持比例不变，避免旋转时变形
        ```

    * daspect 函数 - 最常用方法
    
        ```matlab
        % 设置数据单位的比例
        daspect([1 2 3]);  % x:y:z = 1:2:3，z轴是x轴的3倍长度

        % 示例：使z轴拉伸2倍
        [x,y,z] = sphere(20);
        surf(x,y,z);
        daspect([1 1 2]);  % x和y轴1:1，z轴拉伸2倍

        % 获取当前比例
        current_ratio = daspect;
        ```

    * pbaspect 函数 - 设置绘图框比例
    
        ```matlab
        % 设置绘图框的相对比例
        pbaspect([1 1 1]);   % 立方体形状的绘图框
        pbaspect([2 1 1]);   % x方向占2份，y和z各占1份

        % 示例：宽高比2:1:1
        surf(peaks);
        pbaspect([2 1 1]);    % 绘图框比例，不是数据比例
        ```

    * axis 函数综合设置
    
        ```matlab
        % 设置坐标轴范围和比例
        [x,y,z] = peaks(30);
        surf(x,y,z);

        % 方法1：设置坐标范围
        axis([-3 3 -3 3 -10 10]);  % [xmin xmax ymin ymax zmin zmax]

        % 方法2：组合使用
        axis equal tight;          % 等比例且紧凑
        ```

    * view 函数调整视角

        ```matlab
        surf(peaks);
        daspect([1 1 3]);          % z轴拉伸3倍
        view(30, 30);              % 方位角30°，俯仰角30°
        view(3);                   % 默认3D视角
        ```

    * 完整示例

        ```matlab
        % 创建3D数据
        [X,Y] = meshgrid(-2:0.2:2);
        Z = X.*exp(-X.^2 - Y.^2);

        % 绘制曲面
        figure('Position', [100 100 800 600]);
        surf(X,Y,Z);
        title('调整3D图形比例示例');

        % 设置数据比例：使z轴更明显
        daspect([1 1 0.5]);  % z轴压缩为原来的一半

        % 设置绘图框比例
        pbaspect([1.5 1 1]);  % x方向绘图框更宽

        % 调整视角
        view(45, 30);

        % 添加坐标轴标签
        xlabel('X轴');
        ylabel('Y轴');
        zlabel('Z轴');

        % 添加网格和光照
        grid on;
        light;
        lighting gouraud;
        shading interp;
        ```

    * 重要区别

        daspect: 控制数据单位的显示比例

        pbaspect: 控制绘图框的形状比例

        axis equal: 使三个坐标轴的数据单位长度相等

    * 实际应用技巧

        ```matlab
        % 技巧1：恢复默认比例
        daspect('auto');
        pbaspect('auto');

        % 技巧2：保存和恢复设置
        original_daspect = daspect;
        original_pbaspect = pbaspect;
        % ... 进行修改 ...
        daspect(original_daspect);  % 恢复原始设置

        % 技巧3：结合subplot使用
        figure;
        subplot(1,2,1);
        surf(peaks);
        daspect([1 1 2]);  % 左图

        subplot(1,2,2);
        surf(peaks);
        daspect([1 1 0.5]);  % 右图，z轴压缩
        ```

    * 推荐使用流程：

        1. 先用 daspect 调整数据比例

        2. 再用 pbaspect 调整绘图框形状

        3. 最后用 view 调整最佳视角

        4. 使用 axis tight 确保图形充分利用空间

* matlab 画图时，由于鼠标可以和图片中的数据点交互，当数据量很大时，即使鼠标从图片上滑过也会很卡，如何禁用图片交互，减少卡顿？

    ```matlab
    % 禁用坐标轴交互
    ax = gca()
    disableDefaultInteractivity(ax);

    % 清空所有交互
    ax.Interactions = [];

    % 移除工具栏
    ax.Toolbar = [];

    % 只保留缩放和平移
    ax.Interactions = [zoomInteraction, panInteraction];
    ```

    旧版本 MATLAB 的解决方案

    ```matlab
    % 对于R2020b及更早版本
    fig = figure;
    ax = gca;

    % 方法1：禁用数据光标
    ax.Toolbar = [];  % 移除工具栏
    dcm_obj = datacursormode(fig);
    set(dcm_obj, 'Enable', 'off');  % 禁用数据光标

    % 方法2：关闭所有交互模式
    zoom off;    % 禁用缩放
    pan off;     % 禁用平移
    rotate3d off; % 禁用3D旋转
    datacursormode off; % 禁用数据光标

    % 方法3：设置图形属性
    set(fig, 'WindowButtonDownFcn', '');     % 清空鼠标点击回调
    set(fig, 'WindowButtonUpFcn', '');       % 清空鼠标释放回调
    set(fig, 'WindowButtonMotionFcn', '');   % 清空鼠标移动回调
    ```

    * 性能优化组合方案

        ```matlab
        function createNonInteractivePlot(x, y)
            % 创建非交互式图形以提高性能
            
            % 1. 创建图形并禁用交互
            fig = figure('Interactions', [], ...
                        'ToolBar', 'none', ...
                        'MenuBar', 'none', ...
                        'IntegerHandle', 'off');
            
            % 2. 创建坐标轴
            ax = axes('Parent', fig, ...
                    'Interactions', [], ...
                    'XTickMode', 'manual', ...
                    'YTickMode', 'manual', ...
                    'ZTickMode', 'manual');
            
            % 3. 绘制数据（使用性能优化选项）
            h = plot(x, y, '-', ...
                    'LineWidth', 1, ...
                    'Marker', 'none', ...      % 无标记点
                    'MarkerSize', 0.1);        % 最小化标记
            
            % 4. 进一步禁用交互
            disableDefaultInteractivity(ax);
            
            % 5. 设置渲染器
            set(fig, 'Renderer', 'opengl');     % 使用OpenGL渲染
            
            % 6. 关闭不必要的功能
            set(h, 'HitTest', 'off');           % 禁用图形对象点击检测
            set(h, 'PickableParts', 'none');    % 完全不可选取
            
            % 7. 对3D图形特别优化
            if ~isempty(findobj(ax, 'Type', 'surface'))
                axis(ax, 'vis3d');              % 保持3D视角不变
                set(ax, 'CameraViewAngleMode', 'manual');
            end
            
            % 8. 设置回调函数为空（防止意外交互）
            set(fig, 'WindowButtonMotionFcn', '');
            set(fig, 'WindowButtonDownFcn', '');
            set(fig, 'WindowButtonUpFcn', '');
        end
        ```

    6. 针对大数据量的特别优化

        ```matlab
        % 大数据量示例
        x = randn(1e6, 1);
        y = cumsum(x);

        % 方法1：使用快速绘制函数
        fig = figure('Interactions', []);
        scatter(x(1:100:end), y(1:100:end), 1);  % 降采样
        disableDefaultInteractivity(gca);

        % 方法2：设置对象属性
        fig = figure;
        ax = gca;
        h = plot(x, y);
        h.HitTest = 'off';            % 禁用点击测试
        h.PickableParts = 'none';     # 完全不可交互
        ax.PickableParts = 'none';    # 坐标轴也不可交互

        % 方法3：使用graphics.smoothline（R2023b+）
        % 对于曲线，使用平滑线对象
        ```

    7. 交互模式管理函数

        ```matlab
        % 快速切换函数
        function toggleInteractivity(state)
            % state: 'on' 或 'off'
            fig = gcf;
            ax = gca;
            
            if strcmpi(state, 'off')
                % 禁用交互
                zoom off;
                pan off;
                rotate3d off;
                datacursormode off;
                
                if verLessThan('matlab', '9.10')  % R2021a之前
                    set(fig, 'WindowButtonMotionFcn', '');
                else
                    disableDefaultInteractivity(fig);
                end
            else
                % 启用交互
                zoom on;
                pan on;
                rotate3d on;
            end
        end
        ```

    8. 最佳实践建议

        绘图前准备：

        ```matlab
        % 在绘制大数据前设置
        set(0, 'DefaultFigureInteractions', []);  % 设置默认不交互
        ```

        批量绘图时：

        ```matlab
        for i = 1:10
            fig = figure('Visible', 'off', 'Interactions', []);  % 先隐藏
            plot(bigData{i});
            % ... 其他设置
            disableDefaultInteractivity(gca);
            set(fig, 'Visible', 'on');  # 最后显示
        end
        ```

        临时禁用/启用：

        ```matlab
        % 需要交互时临时启用
        ax.Interactions = [zoomInteraction, panInteraction];

        % 完成后再次禁用
        ax.Interactions = [];
        ```

        最有效的单行解决方案：

        ```matlab
        disableDefaultInteractivity(gcf);  % R2021a+
        % 或
        set(gca, 'Interactions', []);      % R2021a+
        ```

        对于旧版本，使用：

        ```matlab
        zoom off; pan off; rotate3d off; datacursormode off;
        set(gcf, 'WindowButtonMotionFcn', '');
        ```

        这些方法可以显著提升大数据绘图时的响应速度。

* 假如 `a` 是一个 struct，`b = a.ages`，那么 `b` 是按值传递，还是按引用传递？

    在 MATLAB 中，`b = a.ages` 是值传递（Copy-on-Write 机制）

    详细解释：

    1. MATLAB 的基本传递规则

        ```matlab
        a = struct('ages', [25, 30, 35], 'names', {'Tom', 'Jerry', 'Spike'});
        b = a.ages;  % 此时创建了一个副本（但不是立即复制）
        ```

    2. Copy-on-Write（写时复制）机制

        ```matlab
        % 初始情况：b 和 a.ages 可能共享数据（内部优化）
        b = a.ages;  % 表面上复制，实际上可能只是引用（内部优化）

        % 但是一旦修改 b，就会触发真正的复制
        b(1) = 26;   % 此时 MATLAB 才会真正复制数据
        disp(a.ages(1));  % 输出: 25（a.ages 未被修改）
        ```

    3. 验证实验

        ```matlab
        % 测试1：基本赋值
        a = struct('ages', [25, 30, 35]);
        b = a.ages;
        disp('初始内存地址可能相同（内部优化）');

        % 测试2：修改验证
        b(1) = 100;  % 触发实际复制
        disp(['a.ages(1) = ', num2str(a.ages(1))]);  % 仍是 25
        disp(['b(1) = ', num2str(b(1))]);            % 变为 100

        % 测试3：使用内存函数验证（需要自定义函数）
        % memBefore = memory_usage();  % 伪代码，实际需要自己实现
        % b = a.ages;
        % b(1) = 26;  % 此时内存会增加
        % memAfter = memory_usage();
        ```

    4. 与引用传递的对比

        ```matlab
        % 对比：真正的引用传递（使用 handle 类）
        classdef Person < handle
            properties
                ages
            end
        end

        p = Person();
        p.ages = [25, 30, 35];
        b_ref = p.ages;  % 通过 handle，b_ref 是引用
        b_ref(1) = 26;
        disp(p.ages(1));  % 输出: 26（原数据被修改）
        ```

    5. 特殊情况：元胞数组和结构体数组

        ```matlab
        % 结构体数组的情况
        a(1).ages = 25;
        a(2).ages = 30;
        b = [a.ages];  % 这也会创建副本

        % 元胞数组
        a = struct('ages', {{25, 30, 35}});
        b = a.ages{1};  % 对元胞内容也是值传递
        b = 26;
        disp(a.ages{1});  % 仍是 25
        ```

    6. 性能考虑

        ```matlab
        % 大型数据时的优化策略
        a = struct('ages', randn(10000, 10000));  % 大型矩阵

        % 不好的做法：无意识复制
        b = a.ages;  % 表面上只是赋值
        % ...很多操作...
        % 如果不需要修改，这个复制是浪费的

        % 好的做法：明确是否需要修改
        if needToModify
            b = a.ages;  % 接受可能的复制开销
        else
            % 如果只是读取，可以考虑其他方式
            processData(a.ages);  % 直接使用原数据
        end
        ```

    7. 如何实现真正的引用传递

        ```matlab
        % 方法1：使用嵌套函数（共享工作区）
        function processLargeData()
            data = struct('ages', randn(1000));
            
            function innerFunc()
                % 可以直接访问和修改 data.ages
                data.ages(1) = 100;  % 修改的是原始数据
            end
            innerFunc();
        end

        % 方法2：使用 handle 类（面向对象）
        classdef DataHolder < handle
            properties
                ages
            end
        end

        holder = DataHolder();
        holder.ages = randn(1000);
        b = holder;  % b 是引用，不是副本
        b.ages(1) = 100;  % 修改的是原始数据
        ```

    8. 总结要点

        | 情况 | 传递方式 | 说明 |
        | - | - | - |
        | `b = a.ages` | 值传递（Copy-on-Write） | 默认行为，修改时复制 |
        | handle 类对象 | 引用传递 | 修改会影响原数据 |
        | 嵌套函数内访问 | 引用传递 | 共享父函数工作区 |
        | 大型数据操作 | 需谨慎 | 无意识复制可能导致内存问题 |

        核心结论：

        * MATLAB 中结构体字段的赋值是值传递

        * 但有 Copy-on-Write 优化：只在修改时才真正复制

        * 这既保证了安全性（默认不修改原数据），又在只读场景下有性能优化

        * 如果需要引用语义，需要使用 handle 类或嵌套函数

    9. 实际编程建议

        ```matlab
        % 如果确定不需要修改原数据
        function result = processData(inputStruct)
            % 直接使用，让MATLAB决定是否复制
            data = inputStruct.ages;  
            result = mean(data);  % 只读操作，可能不会复制
        end

        % 如果需要修改但不影响原数据
        function modified = modifyCopy(inputStruct)
            modified = inputStruct.ages;  % 可能复制（Copy-on-Write）
            modified(1) = newValue;       % 此时一定复制
        end

        % 如果需要修改原数据（引用传递）
        function modifyInPlace(dataHolder)  % dataHolder 是 handle 对象
            dataHolder.ages(1) = newValue;  % 直接修改原数据
        end
        ```

* 复数

    ```matlab
    a = 1 + 2i;
    complex(2, 4)
    ```

* 常见的与复数相关的函数

    ```matlab
    conj(c)  % 计算 c 的共轭复数
    real(c)  % 返回复数 c 的实部
    imag(c)  % 返回复数 c 的虚部
    isreal(c)  % 如果数组中全是实数，则返回 1，否则返回 0
    abs(c)  % 返回复数 c 的模
    angle(c)  % 返回复数 c 的幅角
    ```

* 预置变量

    * `eps`：计算机的最小正数

    * `pi`：圆周率 pi 的近似值 3.14159265358979

    * `inf`或`Inf`：无穷大

    * `NaN`：不定量

    * `i, j`：虚数单位定义`i`

    * `flops`：浮点运算次数，用于统计计算量

* `summary()`

    显示 table 类型数据的摘要。

    ```matlab
    % 对于 table 类型
    if istable(data)
        summary(data)  % 显示统计摘要
    end

    % 对于timetable类型
    if istimetable(data)
        summary(data)
    end
    ```

* 快速查看函数集合

    可以创建自定义函数方便使用：

    ```matlab
    function quickview(var, n)
        % 快速查看变量部分内容
        if nargin < 2
            n = 5;
        end
        disp(['Size: ', num2str(size(var))])
        disp(['Type: ', class(var)])
        disp('--- Head ---')
        if istable(var)
            disp(var(1:min(n, height(var)), :))
        elseif ismatrix(var)
            disp(var(1:min(n, size(var,1)), :))
        else
            disp(var)
        end
    end
    ```

* 查看稀疏矩阵类型数据的信息

    ```matlab
    % 稀疏矩阵
    spy(data)  % 可视化稀疏模式
    nnz(data)  % 非零元素个数
    ```

* 对于超大数组，考虑先抽取样本查看：

    ```matlab
    sample_idx = randsample(size(data,1), 1000);  % 随机抽取1000行
    sample = data(sample_idx, :);
    ```

* matlab 使用尾递归优化

    ```matlab
    function file_list = walk_fast(root_folder)
        % 性能优化的递归遍历
        
        file_list = {};
        folder_queue = {root_folder};
        
        % 使用队列而非递归栈
        while ~isempty(folder_queue)
            current_folder = folder_queue{1};
            folder_queue(1) = [];
            
            % 获取当前文件夹内容
            items = dir(current_folder);
            
            for i = 1:length(items)
                item = items(i);
                
                if strcmp(item.name, '.') || strcmp(item.name, '..')
                    continue;
                end
                
                item_path = fullfile(current_folder, item.name);
                
                if item.isdir
                    folder_queue{end+1} = item_path;
                else
                    file_list{end+1} = item_path;
                end
            end
        end
    end
    ```

* matlab 并行处理（大文件夹）

    ```matlab
    function process_folder_parallel(root_folder)
        % 并行处理文件夹中的文件
        
        % 获取所有文件
        all_files = find_files(root_folder, '');
        
        % 并行处理
        parfor i = 1:length(all_files)
            process_single_file(all_files{i});
        end
    end

    function process_single_file(filepath)
        % 处理单个文件
        fprintf('处理: %s\n', filepath);
        % ... 具体处理逻辑 ...
    end
    ```

* matlab 递归处理目录的注意事项

    ```matlab
    % 1. 处理符号链接（可能需要额外检查）
    items = dir(folder);
    % dir() 会跟随符号链接，可能导致无限循环

    % 2. 权限问题
    try
        items = dir(folder);
    catch ME
        warning('无法访问: %s, 原因: %s', folder, ME.message);
    end

    % 3. 内存管理（大文件夹）
    % 考虑使用分批处理或datastore
    ```

* matlab 错误处理

    ```matlab
    try
        fid = fopen('nonexistent.txt', 'r');
        data = fscanf(fid, '%f');
        fclose(fid);
    catch ME
        fprintf('错误: %s\n', ME.message);
        % 错误恢复逻辑...
    end
    ```

* matlab 编程基础

    * 脚本文件 (.m)

        ```matlab
        % myscript.m
        x = linspace(0, 2*pi, 100);
        y = sin(x);
        plot(x, y);
        title('正弦曲线');
        ```

        在 command windows 直接输入脚本文件的名字，即可运行。比如

        `>>> myscript`

    * 函数文件

        ```matlab
        % myfunction.m
        function [output1, output2] = myfunction(input1, input2)
            % 函数说明
            output1 = input1 + input2;
            output2 = input1 * input2;
        end
        ```

    * 流程控制

        ```matlab
        % 条件判断
        if x > 0
            disp('正数');
        elseif x < 0
            disp('负数');
        else
            disp('零');
        end

        % 循环
        for i = 1:10
            disp(i);
        end

        while x < 100
            x = x * 2;
        end
        ```

* 基本计算

    ```matlab
    >> 3 + 4 * 2  % 直接计算
    ans = 11

    >> x = 5;      % 赋值（分号抑制输出）
    >> y = x^2 + 3*x - 2
    y = 38
    ```

* 向量和矩阵

    ```matlab
    >> A = [1 2 3; 4 5 6; 7 8 9]  % 创建矩阵
    >> v = 1:0.5:3                % 创建向量 [1, 1.5, 2, 2.5, 3]
    >> B = zeros(3, 2)            % 3×2零矩阵
    >> C = ones(2, 3)             % 2×3全1矩阵
    ```

    `A`中的空格也可以为逗号。

    `1:3`默认步长为 1，即生成`1 2 3`

* 常用运算

    ```matlab
    >> A'          % 转置
    >> A * B       % 矩阵乘法
    >> A .* B      % 元素对应相乘
    >> inv(A)      % 求逆
    >> size(A)     % 矩阵大小
    ```

    注：

    1. `size(A)`得到的也是一个向量，或者说，一维矩阵

## topics

### 可视化

* 基本绘图

    ```matlab
    x = linspace(0, 2 * pi)
    y = sin(x)
    figure;                       % 新建图形窗口
    plot(x, y, 'r--', 'LineWidth', 2);  % 红色虚线
    xlabel('X轴');
    ylabel('Y轴');
    grid on;                      % 显示网格
    hold on;                      % 保持图形
    plot(x, cos(x), 'b-');       % 继续绘图
    legend('sin', 'cos');        % 添加图例
    ```

    注：

    1. `hold on`和`grid on`也可以写在 plot 第一幅图之前

* 子图

    ```matlab
    subplot(2, 2, 1);
    plot(x, sin(x));
    subplot(2, 2, 2);
    plot(x, cos(x));
    subplot(2, 2, 3);
    bar([1 2 3 4]);
    subplot(2, 2, 4);
    histogram(randn(1000,1));
    ```

    注：

    1. 还可以写成`subplot(221)`，`subplot(222)`等。

### 打印与字符串

* matlab 中，单引号`'`和双引号`"`都可以表示字符串。

* `disp()`

    基本显示。不支持格式化打印。

    ```matlab
    disp('Hello World');          % 显示字符串
    disp(['x = ', num2str(5)]);   % 显示变量

    % 显示矩阵
    A = [1 2 3; 4 5 6];
    disp('矩阵A:');
    disp(A);
    ```

    注：

    1. `disp()`不把`\n`识别为转义字符

        ```matlab
        >> disp('hello\n')
        hello\n
        ```

    1. `disp('xxx')`和`disp('xxx');`效果相同，都没有`ans = xxx`的输出。

* `fprintf()`

    格式化输出（最常用）

    ```matlab
    % 基本格式
    fprintf('格式字符串', 变量1, 变量2, ...);

    % 常用格式符
    % %d - 整数
    % %f - 浮点数
    % %e - 科学计数法
    % %g - 自动选择 %f 或 %e
    % %s - 字符串
    % %c - 字符
    % %% - 百分号本身
    ```

    example:

    ```matlab
    % 整数
    age = 25;
    fprintf('年龄: %d 岁\n', age);          % 年龄: 25 岁

    % 浮点数（控制小数位数）
    pi_value = pi;
    fprintf('π = %.2f\n', pi_value);        % π = 3.14
    fprintf('π = %.4f\n', pi_value);        % π = 3.1416
    fprintf('π = %8.4f\n', pi_value);       % π =   3.1416（总宽度8）

    % 科学计数法
    speed = 299792458;
    fprintf('光速: %.2e m/s\n', speed);     % 光速: 3.00e+08 m/s

    % 字符串
    name = '张三';
    fprintf('姓名: %s\n', name);            % 姓名: 张三

    % 多个变量
    x = 10; y = 3.1416; z = '结果';
    fprintf('%s: x = %d, y = %.2f\n', z, x, y);  % 结果: x = 10, y = 3.14

    % 对齐输出
    fprintf('%-10s %10s %10s\n', '姓名', '年龄', '分数');
    fprintf('%-10s %10d %10.1f\n', '张三', 20, 85.5);
    fprintf('%-10s %10d %10.1f\n', '李四', 22, 92.0);
    % 输出：
    % 姓名             年龄        分数
    % 张三               20       85.5
    % 李四               22       92.0
    ```

* `sprintf()`

    格式化字符串（不直接显示）

    ```matlab
    % 创建格式化的字符串，不直接输出
    str = sprintf('结果: %.3f', pi);
    disp(str);  % 结果: 3.142

    % 构建复杂字符串
    name = '小明';
    score = 95.5;
    date_str = datestr(now, 'yyyy-mm-dd');
    report = sprintf('成绩报告\n姓名: %s\n分数: %.1f\n日期: %s\n', ...
                    name, score, date_str);
    disp(report);
    ```

* 表格形式输出

    ```matlab
    % 创建表格数据
    names = {'张三', '李四', '王五'};
    ages = [20; 22; 21];
    scores = [85.5; 92.0; 88.5];

    % 表头
    fprintf('\n========== 学生成绩表 ==========\n');
    fprintf('%-10s %-8s %-10s\n', '姓名', '年龄', '成绩');
    fprintf('%s\n', repmat('-', 1, 30));

    % 数据行
    for i = 1:length(names)
        fprintf('%-10s %-8d %-10.1f\n', names{i}, ages(i), scores(i));
    end

    fprintf('\n总人数: %d\n', length(names));
    ```

    output:

    ```
    ========== 学生成绩表 ==========
    姓名         年龄       成绩        
    ------------------------------
    张三         20       85.5      
    李四         22       92.0      
    王五         21       88.5      

    总人数: 3
    ```

* 进度条和动态显示

    ```matlab
    % 进度条
    total = 100;
    fprintf('进度: [');
    for i = 1:total
        % 每10%显示一个#
        if mod(i, 10) == 0
            fprintf('#');
        end
        pause(0.01); % 模拟计算
    end
    fprintf('] 完成！\n');

    % 动态更新单行
    for i = 1:20
        fprintf('处理中: %d/%d\r', i, 20);
        pause(0.1);
    end
    fprintf('\n完成！\n');
    ```

* 与打印相关的综合 example

    ```matlab
    clear; clc;  % 清理环境

    % 程序开始
    fprintf('%s\n', repmat('=', 1, 50));
    fprintf('           数据分析报告\n');
    fprintf('%s\n\n', repmat('=', 1, 50));

    % 计算并显示结果
    data = randn(100, 1);
    mean_val = mean(data);
    std_val = std(data);

    fprintf('统计结果:\n');
    fprintf('%-15s: %8.4f\n', '平均值', mean_val);
    fprintf('%-15s: %8.4f\n', '标准差', std_val);
    fprintf('%-15s: %8d\n', '样本数', length(data));

    % 用表格显示前5个数据
    fprintf('\n前5个样本:\n');
    for i = 1:min(5, length(data))
        fprintf('样本 %2d: %8.4f\n', i, data(i));
    end
    ```

### 高性能与大数据

* matlab 内存管理

    ```matlab
    % 大文件分块读取
    chunk_size = 10000;
    fid = fopen('large_file.txt', 'r');
    while ~feof(fid)
        chunk = textscan(fid, '%f', chunk_size);
        % 处理chunk...
    end
    fclose(fid);

    % 使用datastore处理超大文件
    ds = datastore('large_file.csv');
    while hasdata(ds)
        chunk = read(ds);
        % 处理chunk...
    end
    ```

* 预分配内存的大数据应用

    ```matlab
    % 不推荐：多次动态扩展
    result = [];
    for i = 1:1000
        result = [result; repmat(i, 10, 1)];  % 每次循环都复制
    end

    % 推荐：一次性预分配
    result = zeros(10000, 1);
    for i = 1:1000
        idx = (i-1)*10+1 : i*10;
        result(idx) = repmat(i, 10, 1);
    end

    % 更推荐：向量化（无循环）
    result = repmat((1:1000)', 10, 1);
    result = sort(result);  % 如果需要排序
    ```

### 文件 IO 与序列化

* matlab 文本文件读写

    * 读取文本文件

        ```matlab
        % 方法1: fscanf - 格式化读取
        fid = fopen('data.txt', 'r');
        data = fscanf(fid, '%f');  % 读取浮点数
        fclose(fid);

        % 方法2: textscan - 灵活读取混合数据
        fid = fopen('data.txt', 'r');
        C = textscan(fid, '%s %f %f', 'Delimiter', ',');
        fclose(fid);
        names = C{1}; values1 = C{2}; values2 = C{3};

        % 方法3: readmatrix (R2019a+)
        data = readmatrix('data.txt');

        % 方法4: readtable - 读取为表格
        T = readtable('data.csv');
        T = readtable('data.txt', 'Delimiter', '\t');  % 制表符分隔
        ```

    * 写入文本文件

        ```matlab
        % 方法1: fprintf
        fid = fopen('output.txt', 'w');  % 'w'写入，'a'追加
        fprintf(fid, '姓名: %s, 年龄: %d, 分数: %.2f\n', '张三', 20, 85.5);
        fclose(fid);

        % 方法2: writematrix (R2019a+)
        A = [1 2 3; 4 5 6];
        writematrix(A, 'matrix.txt');
        writematrix(A, 'matrix.txt', 'Delimiter', '\t');  % 制表符分隔

        % 方法3: writetable - 写入表格
        data = {'张三', 20, 85.5; '李四', 22, 92.0};
        T = cell2table(data, 'VariableNames', {'Name','Age','Score'});
        writetable(T, 'students.csv');
        ```

* matlab 读写CSV

    ```matlab
    % 读取CSV
    data = csvread('data.csv');        % 纯数值数据
    T = readtable('data.csv');         % 包含表头
    [M, ~, ~] = xlsread('data.csv');   % 读取Excel格式CSV

    % 写入CSV
    csvwrite('output.csv', data);      % 写入数值数据
    writetable(T, 'output.csv');       % 写入表格（带表头）

    % 自定义分隔符
    data = dlmread('data.txt', ',');   % 指定分隔符为逗号
    dlmwrite('output.txt', data, 'delimiter', '\t');  % 制表符分隔
    ```

* matlab 文件与目录管理

    ```
    % 文件操作
    exist('filename.txt')   % 检查文件是否存在（返回0-7）
    copyfile('源','目标')   % 复制文件
    movefile('源','目标')   % 移动/重命名
    delete('filename.txt')  % 删除文件

    % 创建目录
    mkdir('newfolder')      % 创建文件夹
    rmdir('foldername')     % 删除空文件夹
    ```

* 文件常用函数速查表

    | 操作类型 | 读取函数 | 写入函数 | 适用格式 |
    | - | - | - | - |
    | 文本文件 | fscanf, textscan | fprintf | .txt, .dat |
    | 表格数据 | readtable | writetable | .csv, .txt, .xlsx |
    | 数值矩阵 | readmatrix | writematrix | .txt, .csv |
    | Excel | xlsread | xlswrite | .xlsx, .xls |
    | 二进制 | load | save | .mat |
    | 图像 | imread | imwrite | .jpg, .png, .tif |
    | 音频 | audioread | audiowrite | .wav, .mp3 |

* matlab 读写 Excel

    ```matlab
    % 读取 Excel
    [num, txt, raw] = xlsread('data.xlsx');      % 读取整个工作表
    [num, txt, raw] = xlsread('data.xlsx', 'Sheet2');  % 指定工作表
    [num, txt, raw] = xlsread('data.xlsx', 'A1:C10');  % 指定区域

    % 写入 Excel
    data = {'Name','Age','Score'; '张三',20,85.5; '李四',22,92.0};
    xlswrite('output.xlsx', data);               % 写入数据
    xlswrite('output.xlsx', data, 'Sheet1', 'A1');  % 指定位置

    % 新版方法 (R2019a+)
    T = readtable('data.xlsx');                  % 读取为表格
    writetable(T, 'output.xlsx', 'Sheet', 'Results');  % 写入表格
    ```

* matlab 保存和加载变量

    ```matlab
    % 保存变量
    x = rand(10, 10);
    y = magic(5);
    z = 'test string';
    save('data.mat');                    % 保存所有变量
    save('data.mat', 'x', 'y');         % 只保存x和y
    save('data.mat', '-append');        % 追加保存
    save('data.mat', '-v7.3');          % 保存大型数据（>2GB）

    % 加载变量
    load('data.mat');                   % 加载所有变量
    data = load('data.mat');            % 加载到结构体
    x = data.x;                         % 访问具体变量

    % 检查MAT文件内容
    whos('-file', 'data.mat');          % 查看文件中的变量
    ```

* matlab 读写图像

    ```matlab
    % 读取图像
    img = imread('image.jpg');          % 读取JPEG
    img = imread('image.png');          % 读取PNG
    img = imread('image.tif', 3);       % 读取TIFF第3帧

    % 显示图像
    imshow(img);
    image(img); colormap(gray);

    % 保存图像
    imwrite(img, 'output.jpg', 'Quality', 90);      % JPEG质量90%
    imwrite(img, 'output.png', 'Transparency', [0 0 0]);  % PNG透明
    ```

* matlab 读写音频

    ```matlab
    % 读取音频
    [y, Fs] = audioread('audio.wav');   % 读取WAV
    [y, Fs] = audioread('audio.mp3');   % 读取MP3
    info = audioinfo('audio.wav');      % 获取音频信息

    % 播放音频
    sound(y, Fs);                       % 立即播放
    player = audioplayer(y, Fs);        % 创建播放器对象
    play(player);

    % 保存音频
    audiowrite('output.wav', y, Fs);                    % WAV格式
    audiowrite('output.mp3', y, Fs, 'BitRate', 128);    % MP3格式
    ```

* matlab 遍历文件夹文件

    ```matlab
    % 获取文件列表
    files = dir('*.txt');               % 所有txt文件
    files = dir('data/*.csv');          % 子文件夹中的CSV

    % 遍历处理
    for i = 1:length(files)
        filename = files(i).name;
        fullpath = fullfile(files(i).folder, filename);
        fprintf('处理文件: %s\n', filename);
        
        % 读取文件内容
        data = readmatrix(fullpath);
        
        % 处理数据...
    end
    ```

* matlab 文件路径操作

    ```matlab
    % 构建完整路径
    fullpath = fullfile('folder', 'subfolder', 'file.txt');

    % 路径分解
    [pathstr, name, ext] = fileparts('C:\data\test.txt');
    % pathstr = 'C:\data', name = 'test', ext = '.txt'

    % 相对路径转绝对路径
    abs_path = which('myfunction.m');    % 查找文件绝对路径
    ```

* matlab 交互式文件选择

    ```matlab
    % 选择文件
    [filename, pathname] = uigetfile(...
        {'*.txt;*.csv', '数据文件 (*.txt, *.csv)';
        '*.xlsx;*.xls', 'Excel文件 (*.xlsx, *.xls)';
        '*.*', '所有文件 (*.*)'}, ...
        '选择数据文件');

    if ~isequal(filename, 0)
        fullpath = fullfile(pathname, filename);
        % 读取文件...
    end

    % 选择文件夹
    folder = uigetdir('C:\', '选择数据文件夹');
    ```

* matlab 完整的数据处理流程 example

    ```matlab
    function process_data_files()
        clear; clc; close all;
        
        % 1. 选择输入文件
        [infile, inpath] = uigetfile({'*.csv;*.xlsx', '数据文件'}, '选择输入文件');
        if isequal(infile, 0)
            error('未选择文件');
        end
        input_file = fullfile(inpath, infile);
        
        % 2. 读取数据
        fprintf('正在读取: %s\n', infile);
        if endsWith(infile, '.csv')
            data = readtable(input_file);
        elseif endsWith(infile, '.xlsx')
            data = readtable(input_file);
        end
        
        % 3. 处理数据
        fprintf('数据大小: %d行 × %d列\n', size(data));
        
        % 统计分析
        stats.mean = mean(data{:, 2:end});
        stats.std = std(data{:, 2:end});
        stats.min = min(data{:, 2:end});
        stats.max = max(data{:, 2:end});
        
        % 4. 保存结果
        % 保存为MAT文件
        save('analysis_results.mat', 'data', 'stats');
        
        % 保存为Excel
        output_table = array2table([stats.mean; stats.std; stats.min; stats.max]');
        output_table.Properties.VariableNames = {'Mean', 'Std', 'Min', 'Max'};
        output_table.Properties.RowNames = data.Properties.VariableNames(2:end);
        
        writetable(output_table, 'statistics.xlsx', 'WriteRowNames', true);
        
        % 5. 生成报告
        generate_report(data, stats, infile);
        
        fprintf('处理完成！\n');
    end

    function generate_report(data, stats, filename)
        report_file = 'analysis_report.txt';
        fid = fopen(report_file, 'w');
        
        fprintf(fid, '数据分析报告\n');
        fprintf(fid, '%s\n', repmat('=', 50));
        fprintf(fid, '输入文件: %s\n', filename);
        fprintf(fid, '分析时间: %s\n', datestr(now));
        fprintf(fid, '\n统计结果:\n\n');
        
        fprintf(fid, '%-15s %10s %10s %10s %10s\n', ...
            '变量', '平均值', '标准差', '最小值', '最大值');
        fprintf(fid, '%s\n', repmat('-', 55));
        
        vars = data.Properties.VariableNames(2:end);
        for i = 1:length(vars)
            fprintf(fid, '%-15s %10.2f %10.2f %10.2f %10.2f\n', ...
                vars{i}, stats.mean(i), stats.std(i), stats.min(i), stats.max(i));
        end
        
        fclose(fid);
        fprintf('报告已生成: %s\n', report_file);
    end
    ```

* matlab 批量处理图像 example

    ```matlab
    function batch_process_images()
        % 批量处理文件夹中的所有图像
        input_folder = uigetdir('', '选择图像文件夹');
        output_folder = fullfile(input_folder, 'processed');
        
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        
        % 获取所有图像文件
        image_files = dir(fullfile(input_folder, '*.jpg'));
        image_files = [image_files; dir(fullfile(input_folder, '*.png'))];
        
        fprintf('找到 %d 个图像文件\n', length(image_files));
        
        for i = 1:length(image_files)
            % 读取图像
            img_path = fullfile(image_files(i).folder, image_files(i).name);
            img = imread(img_path);
            
            % 处理图像（示例：转为灰度并调整大小）
            if size(img, 3) == 3
                img_gray = rgb2gray(img);
            else
                img_gray = img;
            end
            
            img_resized = imresize(img_gray, [256 256]);
            
            % 保存处理后的图像
            [~, name, ext] = fileparts(image_files(i).name);
            output_path = fullfile(output_folder, [name '_processed' ext]);
            imwrite(img_resized, output_path);
            
            fprintf('已处理: %s\n', image_files(i).name);
        end
        
        fprintf('批量处理完成！\n');
    end
    ```

* matlab 文件编码问题

    ```matlab
    % 处理中文等特殊字符
    fid = fopen('file.txt', 'r', 'n', 'UTF-8');  % 指定UTF-8编码
    % 或者使用
    fid = fopen('file.txt', 'r', 'native', 'UTF-8');
    ```

* matlab 查找特定类型文件

    ```matlab
    function file_list = find_files(root_folder, pattern)
        % 递归查找匹配模式的文件
        % pattern: 文件模式，如 '*.m', '*.mat', '*.txt'
        
        file_list = {};
        
        % 获取当前文件夹内容
        items = dir(root_folder);
        
        for i = 1:length(items)
            item = items(i);
            
            % 跳过 . 和 .. 目录
            if strcmp(item.name, '.') || strcmp(item.name, '..')
                continue;
            end
            
            item_path = fullfile(root_folder, item.name);
            
            if item.isdir
                % 递归遍历子文件夹
                sub_files = find_files(item_path, pattern);
                file_list = [file_list, sub_files];
            else
                % 检查文件是否匹配模式
                if isempty(pattern) || contains(item.name, pattern) || ...
                (~isempty(regexp(pattern, '^\*\.', 'once')) && ...
                    endsWith(item.name, pattern(2:end)))
                    file_list{end+1} = item_path;
                end
            end
        end
    end

    % 使用示例
    m_files = find_files('C:\MyProject', '*.m');
    mat_files = find_files('C:\MyProject', '*.mat');
    all_files = find_files('C:\MyProject', '');  % 所有文件
    ```

* matlab 面向对象实现（类版本）递归搜索

    ```matlab
    classdef DirectoryWalker < handle
        % DirectoryWalker - 目录遍历器类
        
        properties
            RootFolder      % 根目录
            FileExtension   % 文件扩展名过滤
            ExcludePatterns % 排除模式
            MaxDepth        % 最大深度
        end
        
        properties (SetAccess = private)
            Results         % 遍历结果
            FileCount       % 文件总数
            FolderCount     % 文件夹总数
        end
        
        methods
            function obj = DirectoryWalker(root_folder, varargin)
                % 构造函数
                obj.RootFolder = root_folder;
                
                % 解析可选参数
                p = inputParser;
                addParameter(p, 'FileExtension', {}, @iscell);
                addParameter(p, 'ExcludePatterns', {}, @iscell);
                addParameter(p, 'MaxDepth', Inf, @isnumeric);
                parse(p, varargin{:});
                
                obj.FileExtension = p.Results.FileExtension;
                obj.ExcludePatterns = p.Results.ExcludePatterns;
                obj.MaxDepth = p.Results.MaxDepth;
                
                obj.Results = [];
                obj.FileCount = 0;
                obj.FolderCount = 0;
            end
            
            function walk(obj)
                % 执行遍历
                obj.Results = struct('path', {}, 'files', {}, 'subfolders', {});
                [obj.Results, obj.FileCount, obj.FolderCount] = ...
                    obj.walk_recursive(obj.RootFolder, 0);
            end
            
            function print_summary(obj)
                % 打印摘要信息
                fprintf('=== 目录遍历摘要 ===\n');
                fprintf('根目录: %s\n', obj.RootFolder);
                fprintf('总文件夹数: %d\n', obj.FolderCount);
                fprintf('总文件数: %d\n', obj.FileCount);
                
                if ~isempty(obj.FileExtension)
                    fprintf('文件过滤: %s\n', strjoin(obj.FileExtension, ', '));
                end
                
                if ~isempty(obj.ExcludePatterns)
                    fprintf('排除模式: %s\n', strjoin(obj.ExcludePatterns, ', '));
                end
            end
            
            function plot_file_types(obj)
                % 绘制文件类型统计图
                if isempty(obj.Results)
                    error('请先执行walk()方法');
                end
                
                % 收集所有文件扩展名
                exts = {};
                for i = 1:length(obj.Results)
                    for j = 1:length(obj.Results(i).files)
                        [~, ~, ext] = fileparts(obj.Results(i).files{j});
                        if isempty(ext)
                            ext = '无扩展名';
                        end
                        exts{end+1} = ext;
                    end
                end
                
                % 统计
                [unique_exts, ~, idx] = unique(exts);
                counts = histcounts(idx, 1:length(unique_exts)+1);
                
                % 绘制饼图
                figure;
                pie(counts, unique_exts);
                title('文件类型分布');
            end
        end
        
        methods (Access = private)
            function [results, file_count, folder_count] = walk_recursive(obj, ...
                    current_path, current_depth)
                % 私有递归方法
                
                results = struct('path', {}, 'files', {}, 'subfolders', {});
                file_count = 0;
                folder_count = 0;
                
                % 检查深度限制
                if current_depth > obj.MaxDepth
                    return;
                end
                
                % 获取当前文件夹内容
                items = dir(current_path);
                
                files = {};
                subfolders = {};
                
                for i = 1:length(items)
                    item = items(i);
                    
                    if strcmp(item.name, '.') || strcmp(item.name, '..')
                        continue;
                    end
                    
                    item_fullpath = fullfile(current_path, item.name);
                    
                    if item.isdir
                        % 检查是否排除
                        if ~obj.should_exclude(item.name)
                            subfolders{end+1} = item.name;
                            folder_count = folder_count + 1;
                        end
                    else
                        % 检查文件扩展名
                        if obj.should_include(item.name)
                            files{end+1} = item.name;
                            file_count = file_count + 1;
                        end
                    end
                end
                
                % 创建当前层结果
                current_result.path = current_path;
                current_result.files = files;
                current_result.subfolders = subfolders;
                results = [results, current_result];
                
                % 递归遍历子文件夹
                for i = 1:length(subfolders)
                    subfolder_path = fullfile(current_path, subfolders{i});
                    [sub_results, sub_file_count, sub_folder_count] = ...
                        obj.walk_recursive(subfolder_path, current_depth + 1);
                    
                    results = [results, sub_results];
                    file_count = file_count + sub_file_count;
                    folder_count = folder_count + sub_folder_count;
                end
            end
            
            function include = should_include(obj, filename)
                % 检查文件是否应该包含
                include = true;
                
                if isempty(obj.FileExtension)
                    return;
                end
                
                [~, ~, ext] = fileparts(filename);
                
                for i = 1:length(obj.FileExtension)
                    if strcmpi(ext, obj.FileExtension{i})
                        return;
                    end
                end
                
                include = false;
            end
            
            function exclude = should_exclude(obj, foldername)
                % 检查文件夹是否应该排除
                exclude = false;
                
                if isempty(obj.ExcludePatterns)
                    return;
                end
                
                for i = 1:length(obj.ExcludePatterns)
                    if contains(foldername, obj.ExcludePatterns{i})
                        exclude = true;
                        return;
                    end
                end
            end
        end
    end

    % 使用示例
    walker = DirectoryWalker('C:\MyProject', ...
        'FileExtension', {'.m', '.mat'}, ...
        'ExcludePatterns', {'temp', '.git'}, ...
        'MaxDepth', 3);

    walker.walk();
    walker.print_summary();
    walker.plot_file_types();
    ```

* MATLAB 递归遍历文件夹（实现 os.walk() 功能）

    Python 函数	MATLAB 对应函数	说明
    os.walk()	dir() + 递归	无直接对应，需自定义
    os.listdir()	dir(), ls()	列出文件夹内容
    os.path.join()	fullfile()	构建完整路径

    * 简单递归实现

        ```matlab
        function [files, folders] = walk_directory(root_folder)
            % 递归遍历文件夹，返回所有文件和子文件夹
            % files: 所有文件的完整路径
            % folders: 所有子文件夹的完整路径
            
            files = {};
            folders = {};
            
            % 获取当前文件夹内容
            items = dir(root_folder);
            
            for i = 1:length(items)
                item = items(i);
                
                % 跳过 . 和 .. 目录
                if strcmp(item.name, '.') || strcmp(item.name, '..')
                    continue;
                end
                
                % 构建完整路径
                item_path = fullfile(root_folder, item.name);
                
                if item.isdir
                    % 是文件夹：添加到列表并递归
                    folders{end+1} = item_path;
                    
                    % 递归遍历子文件夹
                    [sub_files, sub_folders] = walk_directory(item_path);
                    
                    % 合并结果
                    files = [files, sub_files];
                    folders = [folders, sub_folders];
                else
                    % 是文件：添加到列表
                    files{end+1} = item_path;
                end
            end
        end

        % 使用示例
        [root_files, root_folders] = walk_directory('C:\MyProject');
        ```

    * 类似 os.walk() 的生成器风格

        ```matlab
        function walk(root_folder)
            % 类似Python的os.walk()，打印遍历结果
            % root_folder: 起始路径
            % dirpath: 当前目录
            % dirnames: 子目录名列表
            % filenames: 文件名列表
            
            % 获取当前文件夹内容
            items = dir(root_folder);
            
            % 分离文件和文件夹
            dirnames = {};
            filenames = {};
            
            for i = 1:length(items)
                item = items(i);
                
                % 跳过 . 和 .. 目录
                if strcmp(item.name, '.') || strcmp(item.name, '..')
                    continue;
                end
                
                if item.isdir
                    dirnames{end+1} = item.name;
                else
                    filenames{end+1} = item.name;
                end
            end
            
            % 输出当前层结果
            fprintf('目录: %s\n', root_folder);
            fprintf('  子文件夹: %s\n', strjoin(dirnames, ', '));
            fprintf('  文件: %s\n\n', strjoin(filenames, ', '));
            
            % 递归遍历子文件夹
            for i = 1:length(dirnames)
                subfolder = fullfile(root_folder, dirnames{i});
                walk(subfolder);
            end
        end

        % 使用示例
        walk('C:\MyProject');
        ```

    * 返回结构体数组版本

        ```matlab
        function walk_results = matlab_walk(root_folder)
            % 完全模拟Python os.walk()功能
            % 返回结构体数组，每个元素包含：
            %   dirpath: 当前目录路径
            %   dirnames: 子目录名（元胞数组）
            %   filenames: 文件名（元胞数组）
            
            walk_results = struct('dirpath', {}, 'dirnames', {}, 'filenames', {});
            walk_results = walk_recursive(root_folder, walk_results);
        end

        function walk_results = walk_recursive(current_path, walk_results)
            % 递归辅助函数
            
            % 获取当前文件夹内容
            items = dir(current_path);
            
            % 初始化当前层的目录和文件列表
            dirnames = {};
            filenames = {};
            
            % 分离文件和文件夹
            for i = 1:length(items)
                item = items(i);
                
                % 跳过 . 和 .. 目录
                if strcmp(item.name, '.') || strcmp(item.name, '..')
                    continue;
                end
                
                if item.isdir
                    dirnames{end+1} = item.name;
                else
                    filenames{end+1} = item.name;
                end
            end
            
            % 添加到结果
            idx = length(walk_results) + 1;
            walk_results(idx).dirpath = current_path;
            walk_results(idx).dirnames = dirnames;
            walk_results(idx).filenames = filenames;
            
            % 递归遍历子文件夹
            for i = 1:length(dirnames)
                subfolder = fullfile(current_path, dirnames{i});
                walk_results = walk_recursive(subfolder, walk_results);
            end
        end

        % 使用示例
        results = matlab_walk('C:\MyProject');

        % 遍历结果
        for i = 1:length(results)
            fprintf('目录: %s\n', results(i).dirpath);
            fprintf('  子文件夹数: %d\n', length(results(i).dirnames));
            fprintf('  文件数: %d\n', length(results(i).filenames));
            
            % 列出前几个文件
            if ~isempty(results(i).filenames)
                fprintf('  文件示例: %s\n', strjoin(results(i).filenames(1:min(3, end)), ', '));
            end
            fprintf('\n');
        end
        ```

    * 带过滤功能的增强版

        ```matlab
        function walk_results = walk_with_filter(root_folder, varargin)
            % 带过滤功能的递归遍历
            % 可选参数:
            %   'FileExt', {'.m', '.mat'} - 文件扩展名过滤
            %   'ExcludeDirs', {'temp', 'test'} - 排除的文件夹
            %   'MaxDepth', 3 - 最大递归深度
            
            % 解析可选参数
            p = inputParser;
            addParameter(p, 'FileExt', {}, @iscell);
            addParameter(p, 'ExcludeDirs', {}, @iscell);
            addParameter(p, 'MaxDepth', Inf, @isnumeric);
            parse(p, varargin{:});
            
            file_ext = p.Results.FileExt;
            exclude_dirs = p.Results.ExcludeDirs;
            max_depth = p.Results.MaxDepth;
            
            % 初始化结果结构
            walk_results = struct('dirpath', {}, 'dirnames', {}, 'filenames', {});
            
            % 开始递归遍历
            walk_results = walk_filter_recursive(root_folder, walk_results, ...
                file_ext, exclude_dirs, max_depth, 0);
        end

        function walk_results = walk_filter_recursive(current_path, walk_results, ...
                file_ext, exclude_dirs, max_depth, current_depth)
            % 递归辅助函数
            
            % 检查深度限制
            if current_depth > max_depth
                return;
            end
            
            % 获取当前文件夹内容
            items = dir(current_path);
            
            % 初始化当前层的目录和文件列表
            dirnames = {};
            filenames = {};
            
            for i = 1:length(items)
                item = items(i);
                
                % 跳过 . 和 .. 目录
                if strcmp(item.name, '.') || strcmp(item.name, '..')
                    continue;
                end
                
                item_fullpath = fullfile(current_path, item.name);
                
                if item.isdir
                    % 检查是否在排除列表中
                    if ~is_matched(item.name, exclude_dirs)
                        dirnames{end+1} = item.name;
                    end
                else
                    % 文件：检查扩展名
                    if isempty(file_ext) || is_matched(item.name, file_ext)
                        filenames{end+1} = item.name;
                    end
                end
            end
            
            % 添加到结果
            idx = length(walk_results) + 1;
            walk_results(idx).dirpath = current_path;
            walk_results(idx).dirnames = dirnames;
            walk_results(idx).filenames = filenames;
            
            % 递归遍历子文件夹
            for i = 1:length(dirnames)
                subfolder = fullfile(current_path, dirnames{i});
                walk_results = walk_filter_recursive(subfolder, walk_results, ...
                    file_ext, exclude_dirs, max_depth, current_depth + 1);
            end
        end

        function matched = is_matched(name, patterns)
            % 检查名称是否匹配模式
            matched = false;
            
            if isempty(patterns)
                matched = true;
                return;
            end
            
            for j = 1:length(patterns)
                pattern = patterns{j};
                
                % 如果是扩展名模式（以.开头）
                if pattern(1) == '.'
                    [~, ~, ext] = fileparts(name);
                    if strcmpi(ext, pattern)
                        matched = true;
                        return;
                    end
                else
                    % 普通字符串匹配（包含关系）
                    if contains(name, pattern)
                        matched = true;
                        return;
                    end
                end
            end
        end

        % 使用示例
        % 1. 查找所有.m文件
        results1 = walk_with_filter('C:\MyProject', 'FileExt', {'.m'});

        % 2. 排除某些文件夹
        results2 = walk_with_filter('C:\MyProject', ...
            'ExcludeDirs', {'temp', 'backup', '.git'});

        % 3. 限制递归深度
        results3 = walk_with_filter('C:\MyProject', 'MaxDepth', 2);

        % 4. 组合条件
        results4 = walk_with_filter('C:\MyProject', ...
            'FileExt', {'.m', '.mat'}, ...
            'ExcludeDirs', {'test', 'old'}, ...
            'MaxDepth', 3);
        ```

* matlab 计算文件夹大小

    ```matlab
    function [total_size, file_count] = get_folder_size(root_folder)
        % 递归计算文件夹总大小
        
        total_size = 0;
        file_count = 0;
        
        items = dir(root_folder);
        
        for i = 1:length(items)
            item = items(i);
            
            if strcmp(item.name, '.') || strcmp(item.name, '..')
                continue;
            end
            
            item_path = fullfile(root_folder, item.name);
            
            if item.isdir
                % 递归计算子文件夹
                [sub_size, sub_count] = get_folder_size(item_path);
                total_size = total_size + sub_size;
                file_count = file_count + sub_count;
            else
                % 累加文件大小
                total_size = total_size + item.bytes;
                file_count = file_count + 1;
            end
        end
    end

    % 使用示例
    [sz, cnt] = get_folder_size('C:\MyProject');
    fprintf('总大小: %.2f MB，文件数: %d\n', sz/1024/1024, cnt);
    ```

* `fullfile()`

    用于构建完整文件路径的函数，它能自动处理不同操作系统之间的路径分隔符差异。

    比较像 python 中的`os.path.join()`。

    syntax:

    ```matlab
    fullpath = fullfile(filepart1, filepart2, ..., filepartN)
    ```

    usage:

    ```matlab
    % 基本路径拼接
    folder = 'C:\Users';
    subfolder = 'Documents';
    filename = 'data.txt';

    % 自动处理分隔符
    fullpath = fullfile(folder, subfolder, filename)
    % 输出（在Windows上）: 'C:\Users\Documents\data.txt'
    % 输出（在Linux/macOS上）: 'C:/Users/Documents/data.txt'

    % 拼接多个部分
    path1 = 'home';
    path2 = 'user';
    path3 = 'projects';
    path4 = 'code';
    path5 = 'main.m';

    result = fullfile(path1, path2, path3, path4, path5)
    % 输出: 'home/user/projects/code/main.m' (Linux/macOS)
    ```

    examples:

    ```matlab
    %% 示例1：构建文件路径
    data_dir = 'data';
    year = '2024';
    month = '03';
    filename = 'experiment.csv';

    filepath = fullfile(data_dir, year, month, filename)
    % 输出: 'data/2024/03/experiment.csv'

    %% 示例2：与 dir、ls 等函数配合使用
    folder = 'images';
    files = dir(fullfile(folder, '*.png'));  % 查找 images 文件夹下所有 PNG 文件

    %% 示例3：创建新目录
    new_dir = fullfile('results', 'analysis', 'plots');
    mkdir(new_dir);  % 创建 results/analysis/plots 目录

    %% 示例4：保存文件
    output_dir = 'output';
    output_file = 'results.mat';
    save(fullfile(output_dir, output_file), 'data');

    %% 示例5：处理绝对路径和相对路径
    root = '/home/user';
    relative_path = 'docs/report.txt';
    abs_path = fullfile(root, relative_path)
    % 输出: '/home/user/docs/report.txt'

    %% 示例6：单元格数组输入
    parts = {'usr', 'local', 'bin', 'matlab'};
    path_cell = fullfile(parts{:})
    % 输出: 'usr/local/bin/matlab'
    ```

    注意事项

    * 不会验证路径是否存在：fullfile() 只负责构建路径字符串，不检查路径是否真实存在

    * 不解析相对路径：不会将 .. 或 . 解析为上级目录或当前目录

    * 与 fileparts() 互补：fileparts() 用于拆分路径，fullfile() 用于合并路径

    * 输入可以是字符串、字符向量或字符串数组

* `fileparts()`

    将完整的文件名或路径分解为路径、文件名和扩展名三个部分，便于单独处理各个组件。

    syntax:

    ```matlab
    [pathstr, name, ext] = fileparts(filename)
    [pathstr, name, ext] = fileparts(filename, 'PeriodicExtension', value)
    ```

    可选参数

    * 'PeriodicExtension', value：控制如何处理周期性扩展名（如 .tar.gz）

        * true（默认）：将 .tar.gz 视为一个扩展名

        * false：只取最后一个扩展名（.gz）

    usage:

    * 基本分解

        ```matlab
        % 分解完整路径
        [pathstr, name, ext] = fileparts('C:\Users\John\Documents\report.pdf');

        % 结果:
        % pathstr = 'C:\Users\John\Documents'
        % name = 'report'
        % ext = '.pdf'
        ```

    * 分解相对路径

        ```matlab
        [pathstr, name, ext] = fileparts('../data/input.csv');

        % 结果:
        % pathstr = '../data'
        % name = 'input'
        % ext = '.csv'
        ```

    * 只获取特定部分

        ```matlab
        % 只需要路径部分
        path_only = fileparts('C:\project\src\main.m');

        % 只需要文件名（不含扩展名）
        [~, filename] = fileparts('data/experiment_results.xlsx');

        % 只需要扩展名
        [~, ~, extension] = fileparts('/home/user/image.jpg');
        ```

    examples:

    * 创建输出文件路径

        ```matlab
        % 基于输入文件生成输出文件路径
        input_file = 'C:\data\input\experiment_data.mat';
        [input_path, input_name, ~] = fileparts(input_file);

        % 创建输出文件路径
        output_dir = fullfile(input_path, 'output');
        output_file = fullfile(output_dir, [input_name '_processed.mat']);
        ```

    * 获取文件类型信息

        ```matlab
        % 根据扩展名分类文件
        files = {'report.pdf', 'data.xlsx', 'script.m', 'notes.txt'};

        for i = 1:length(files)
            [~, ~, ext] = fileparts(files{i});
            
            switch lower(ext)
                case {'.pdf'}
                    disp('PDF document');
                case {'.xlsx', '.xls'}
                    disp('Excel file');
                case {'.m'}
                    disp('MATLAB script');
                case {'.txt', '.csv'}
                    disp('Text file');
                otherwise
                    disp('Unknown file type');
            end
        end
        ```

    特殊情况和注意事项

    * 处理无扩展名的文件

        ```matlab
        [pathstr, name, ext] = fileparts('C:\files\README');
        % pathstr = 'C:\files'
        % name = 'README'
        % ext = ''  (空字符串)
        ```

    * 处理只有扩展名的文件

        ```matlab
        [pathstr, name, ext] = fileparts('.gitignore');
        % pathstr = ''
        % name = ''
        % ext = '.gitignore'
        ```

    * 处理文件夹路径

        ```matlab
        [pathstr, name, ext] = fileparts('C:\Users\John\Documents\');
        % pathstr = 'C:\Users\John'
        % name = 'Documents'
        % ext = ''
        ```

    * 处理网络路径

        ```matlab
        [pathstr, name, ext] = fileparts('\\server\share\file.txt');
        % pathstr = '\\server\share'
        % name = 'file'
        % ext = '.txt'
        ```

    常见错误处理

    * 常见错误处理

        ```matlab
        % 检查文件是否存在
        filename = 'somefile.dat';
        if isfile(filename)
            [pathstr, name, ext] = fileparts(filename);
        else
            error('文件不存在: %s', filename);
        end

        % 处理空输入
        filename = '';
        if ~isempty(filename)
            [pathstr, name, ext] = fileparts(filename);
        else
            % 处理空文件名的情况
        end
        ```

    扩展应用:

    ```matlab
    % 1. 构建新文件名
    [~, name, ~] = fileparts(original_file);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    new_file = sprintf('%s_%s%s', name, timestamp, '.mat');

    % 2. 提取路径层次
    full_path = 'C:\a\b\c\d\file.txt';
    parts = strsplit(fileparts(full_path), filesep);
    % parts = {'C:', 'a', 'b', 'c', 'd'}

    % 3. 获取父目录
    full_path = 'C:\project\src\utils\helper.m';
    parent_dir = fileparts(fileparts(full_path));
    % parent_dir = 'C:\project\src'
    ```

* `mfilename()`

    返回当前正在执行的 MATLAB 函数或脚本的文件名和路径信息。

    syntax:

    ```matlab
    function_name = mfilename('fullpath')
    function_name = mfilename()
    [name, path] = mfilename('fullpath')
    ```

    usage:

    * 无参数调用 - 返回当前文件名

        ```matlab
        % 在 myFunction.m 中调用
        name = mfilename();
        % 返回: 'myFunction' (不带扩展名 .m)
        ```

    * 'fullpath' 参数 - 返回完整路径

        ```matlab
        % 返回包含完整路径的文件名
        full_name = mfilename('fullpath');
        % 示例返回值: 'C:\Users\name\Documents\MATLAB\myScript.m'
        ```

    * 获取路径和文件名分离

        ```matlab
        % 方法1: 使用 fileparts
        [filepath, name, ext] = fileparts(mfilename('fullpath'));

        % 方法2: 直接分割
        full_name = mfilename('fullpath');
        [path_str, name_str, ext_str] = fileparts(full_name);
        ```

        注：

        1. 这两个方法不是一样的吗？

    examples:

    * 获取当前脚本所在目录

        ```matlab
        % 获取当前.m文件所在目录，用于相对路径操作
        current_folder = fileparts(mfilename('fullpath'));
        data_file = fullfile(current_folder, 'data', 'input.csv');
        ```

    重要特性

    * 在命令窗口调用：返回空字符串 `''`

    * 在脚本中调用：返回脚本文件名

    * 在函数中调用：返回函数文件名

    * 嵌套调用时：始终返回当前执行的文件名

    * 返回不带扩展名：mfilename() 不包含 .m 扩展名

    * mfilename() 在调试和部署时行为一致，适用于需要根据文件位置动态构建路径的应用场景。

    与相关函数的比较
    函数	返回值	特点
    mfilename()	当前文件名	不含路径和扩展名
    mfilename('fullpath')	完整路径+文件名	包含完整路径
    which(mfilename)	完整路径+文件名	与 mfilename('fullpath') 相似
    dbstack	调用栈信息	更详细的调用链信息

    * 实用技巧

        ```matlab
        % 获取当前文件所在目录的最佳实践
        current_file_path = mfilename('fullpath');
        [current_dir, ~, ~] = fileparts(current_file_path);

        % 添加当前目录到MATLAB路径（临时）
        addpath(current_dir);

        % 使用相对路径访问同级目录文件
        config_file = fullfile(current_dir, 'config', 'settings.json');
        ```

### struct 与 cell

* 创建 struct

    ```matlab
    % 方法1：直接赋值
    person.name = '张三';
    person.age = 25;
    person.scores = [90, 85, 88];
    person.isStudent = true;

    % 方法2：使用 struct 函数
    person = struct('name', '张三', 'age', 25, 'scores', [90, 85, 88]);

    % 方法3：创建结构体数组
    students(1) = struct('name', '张三', 'age', 25);
    students(2) = struct('name', '李四', 'age', 23);
    ```

    注：

    1. 使用数组创建 struct 还可以写成

        ```matlab
        arr(1).name = 'zhangsan';
        arr(1).age = 25;
        arr(2).name = 'lisi';
        arr(2).age = 26;
        ```

        如果`arr`这个变量名已经被占用，那么无法通过这样新创建数组`arr`。

    1. 在创建元素为 struct 的数组时，如果数据不全，那么会被自动补上空数组

        ```matlab
        >> arr(1).name = 'zhangsan';
        >> arr(2).age = 35;
        >> arr(1)

        ans = 

            struct with fields:

            name: 'zhangsan'
                age: []

        >> arr(2)

        ans = 

            struct with fields:

            name: []
                age: 35
        ```

*  struct（结构体）

    matlab 中 struct (结构体) 是一种将不同类型数据组织在一起的容器。

    * 访问和操作

        ```matlab
        % 访问字段
        name = person.name;
        age = person.age;

        % 修改字段
        person.age = 26;

        % 添加新字段
        person.gender = '男';

        % 删除字段
        person = rmfield(person, 'isStudent');

        % 获取所有字段名
        fields = fieldnames(person);

        % 检查字段是否存在
        hasField = isfield(person, 'name');

        % 遍历结构体数组
        for i = 1:length(students)
            disp(students(i).name);
        end
        ```

    * 常用操作

        ```matlab
        % 嵌套结构体
        company.department.engineering.manager = '王工';
        company.department.engineering.employeeCount = 50;

        % 结构体转为表格（如果结构一致）
        T = struct2table(students);

        % 从表格转为结构体
        S = table2struct(T);
        ```

* cell（元胞数组）

    元胞数组可以存储不同类型和大小的数据，每个元素称为一个"元胞"。

    * 创建元胞数组

        ```matlab
        % 方法1：使用花括号 {}
        C = {'字符串', 123, [1,2,3;4,5,6], true};

        % 方法2：使用cell函数
        C = cell(2, 3);  % 创建2×3的空元胞数组
        C{1,1} = 'MATLAB';
        C{1,2} = 2023;
        C{2,1} = magic(3);

        % 方法3：混合创建
        data{1} = '文本数据';
        data{2} = rand(3,4);
        data{3} = struct('a', 1, 'b', 2);
        ```

    * 访问元素（重要区别！）

        ```matlab
        % 花括号 {} 用于访问内容
        content = C{1,1};  % 返回 'MATLAB' 字符串

        % 圆括号 () 用于访问元胞本身
        cellElement = C(1,1);  % 返回包含 'MATLAB' 的元胞

        % 示例对比
        C = {'A', 100};
        value1 = C{1};     % 'A' (字符)
        value2 = C(1);     % {'A'} (1×1 cell)

        % 多元素访问
        subset = C(1:2);   % 返回包含前两个元素的元胞数组
        contents = C{1:2}; % 错误！不能这样批量获取内容
        ```

    * 常用操作

        ```matlab
        % 获取信息
        size(C)      % 元胞数组维度
        numel(C)     % 元素总数
        iscell(C)    % 检查是否为元胞数组

        % 转换为其他类型
        % 当所有元胞内容类型一致时
        cell2mat(C)      % 转换为矩阵
        cell2table(C)    % 转换为表格

        % 从其他类型转换
        mat2cell(A, [2,2], [3,3])  % 矩阵分块转为元胞
        num2cell(A)                % 矩阵每个元素转为独立元胞

        % 删除元胞
        C(3) = [];      % 删除第三个元胞
        C(:,2) = [];    % 删除第二列
        ```

    * 特殊用法

        ```matlab
        % 存储函数句柄
        funcs = {@sin, @cos, @tan};
        result = funcs{1}(pi/2);  % 计算 sin(pi/2)

        % 存储不同长度的向量
        data{1} = 1:10;
        data{2} = 1:100;
        data{3} = 1:5;

        % 存储不同类型数据
        mixedData = {'文本', 123.45, struct(), [1,2,3], @plot};
        ```

* matlab struct 和 cell 的组合使用

    ```matlab
    % 结构体字段包含元胞数组
    student.info = {'张三', 'CS101', 2023};
    student.grades = {[90,85], [88,92]};

    % 元胞数组包含结构体
    cellArray{1} = struct('name','A','value',1);
    cellArray{2} = struct('name','B','value',2);

    % 常见的表格式数据存储
    for i = 1:100
        data{i}.id = i;
        data{i}.value = rand();
        data{i}.timestamp = datetime('now');
    end
    ```

    选择建议

    | 场景 | 推荐 | 理由 |
    | - | - | - |
    | 数据有明确字段名 | struct | 字段名自解释，代码可读性强 |
    | 数据是异构的 | cell | 灵活存储不同类型数据 |
    | 需要按名称访问 | struct | 直接使用字段名访问 |
    | 需要按索引访问 | cell | 索引访问更自然 |
    | 存储函数集合 | cell | 方便批量操作函数句柄 |
    | 配置参数 | struct | 层次清晰，易于修改 |

    实用技巧

    ```matlab
    % 批量处理结构体数组的字段
    ages = [students.age];              % 提取所有age字段到数组
    names = {students.name};            % 提取所有name字段到元胞数组

    % 元胞数组的便捷操作
    % 对每个元胞应用函数
    results = cellfun(@mean, dataCells);  % 对每个元胞计算均值

    % 条件筛选
    textCells = C(cellfun(@ischar, C));   % 筛选出所有文本元胞

    % 转换嵌套元胞
    flatCell = [C{:}];  % 展开嵌套的元胞数组（如果内容维度一致）
    ```

* matlab cell 的历史

    MATLAB 最初（1970s）就设计了 cell 来处理异构数据，大量遗留代码依赖它。

    * 处理真实世界数据的需求:

        ```matlab
        % 实际问题中确实需要存储混合数据
        experiment_data{1} = '2023-01-01';  % 日期字符串
        experiment_data{2} = sensor_readings;  % 数值矩阵
        experiment_data{3} = experiment_notes;  % 长文本
        experiment_data{4} = @processing_function;  % 处理函数

        % 表格不能直接存储函数句柄或任意类型
        ```

    * 灵活性的优势

        ```matlab
        % 动态数据结构
        % 链表、树等结构在MATLAB中只能用cell实现
        tree_node = {value, left_child, right_child};

        % 变长向量数组
        data{1} = 1:10;
        data{2} = 1:1000;  % 每个向量长度不同
        data{3} = 1:5;

        % 如果用矩阵，需要统一长度或用NaN填充
        ```

    * 特定场景的实用性

        ```matlab
        % 函数参数传递
        varargin和varargout必须是cell

        % GUI开发
        handles结构经常包含cell存储回调函数等

        % 文件处理
        file_list = dir('*.txt');
        filenames = {file_list.name};  % 提取所有文件名
        ```

    * 更好的替代方案

        ```matlab
        % 1. table类型（R2013b引入）
        T = table();
        T.Name = {'张三'; '李四'};
        T.Age = [25; 23];
        T.Score = {[90,85]; [88,92]};  % 注意：table中仍可用cell列

        % 更规范，支持SQL式查询，显示友好

        % 2. 结构体数组
        students = struct('name', {}, 'score', {});
        % 比cell数组更结构化

        % 3. 面向对象编程
        classdef ExperimentData
            properties
                date
                readings
                notes
                processFunc
            end
        end
        % 类型安全，封装性好
        ```

    * Cell仍在某些场景不可替代

        ```matlab
        % 函数式编程风格
        operations = {@sin, @(x)x.^2, @exp};
        results = cellfun(@(f) f(data), operations, 'UniformOutput', false);

        % 快速原型开发
        % 不想定义class或struct时，cell最方便

        % 与旧代码交互
        % 很多MATLAB工具箱函数返回cell
        ```

    实际使用建议

    * 尽量避免使用cell的情况

        * 存储同类型数值数据 → 用矩阵

        * 结构化记录数据 → 用table或struct

        * 需要类型安全 → 用class

    * 可以合理使用cell的情况

        * 真正的异构数据

        * 函数句柄集合

        * 字符串数组（直到R2016b才引入string类型）

        * 实现动态数据结构

    * 新代码尽量用table、struct、matrix、自定义类

* 查看 struct 类型数据的信息

    ```matlab
    % 结构体
    fieldnames(mystruct)  % 查看字段名
    ```

* 查看 cell 类型数据的信息

    ```matlab
    % 元胞数组
    celldisp(data, 'data')  % 显示元胞内容
    ```

### 矩阵运算与索引

* 使用 `head()` / `tail()` 查看数据头部/尾部

    ```matlab
    head(data)      % 显示前 8 行
    tail(data)      % 显示后 8 行

    % 指定行数
    head(data, 10)  % 显示前 10 行
    tail(data, 10)  % 显示后 10 行
    ```

* matlab repmat() 函数

    repmat() 是 "repeat matrix" 的缩写，用于复制和拼接数组/矩阵，创建更大的数组。

    syntax:

    ```matlab
    B = repmat(A, m, n)        % 将A复制m×n次（二维）
    B = repmat(A, [m n])       % 同上，使用向量参数
    B = repmat(A, [m n p ...]) % 多维复制
    ```

    usage:

    * 二维复制（最常用）

        ```matlab
        % 复制单个元素
        A = 5;
        B = repmat(A, 2, 3);
        % B = [5 5 5
        %      5 5 5]

        % 复制向量
        A = [1 2 3];
        B = repmat(A, 3, 1);    % 垂直复制3次
        % B = [1 2 3
        %      1 2 3
        %      1 2 3]

        C = repmat(A, 1, 3);    % 水平复制3次
        % C = [1 2 3 1 2 3 1 2 3]

        D = repmat(A, 2, 2);    % 2×2块复制
        % D = [1 2 3 1 2 3
        %      1 2 3 1 2 3]
        ```

    * 矩阵复制

        ```matlab
        A = [1 2; 3 4];
        B = repmat(A, 2, 3);
        % B = [1 2 1 2 1 2
        %      3 4 3 4 3 4
        %      1 2 1 2 1 2
        %      3 4 3 4 3 4]
        ```

    * 三维及多维复制

        ```matlab
        % 三维数组复制
        A = [1 2; 3 4];
        B = repmat(A, [2, 3, 2]);  % 2行×3列×2页
        size(B)  % 返回 [4, 6, 2]

        % 创建三维网格
        [X, Y, Z] = meshgrid(1:3, 1:2, 1:2);
        % 等价于：
        X = repmat([1 2 3], [2, 1, 2]);
        Y = repmat([1; 2], [1, 3, 2]);
        ```

    example:

    * 创建测试数据

        ```matlab
        % 创建全零模板
        template = zeros(3, 4);
        data = repmat(template, 2, 3);  % 6×12的零矩阵

        % 创建棋盘格
        black = 0; white = 1;
        pattern = [black white; white black];
        chessboard = repmat(pattern, 4, 4);  % 8×8棋盘
        imshow(chessboard);
        ```

    * 向量化运算（广播机制替代）

        ```matlab
        % 传统方法：每个元素加常数
        A = [1 2 3; 4 5 6];
        constant = 10;
        % 需要循环或repmat
        B = A + repmat(constant, size(A));

        % 现代MATLAB：广播机制（更高效）
        B = A + constant;  % R2016b+ 自动广播

        % 矩阵每行加上不同常数
        row_constants = [10; 20];
        C = A + repmat(row_constants, 1, size(A, 2));
        % C = [1+10 2+10 3+10
        %      4+20 5+20 6+20]
        ```

    * 图像处理

        ```matlab
        % 创建彩色图像
        red_channel = 0.8 * ones(100, 100);
        green_channel = 0.5 * ones(100, 100);
        blue_channel = 0.2 * ones(100, 100);

        % 方法1：使用cat
        rgb_image = cat(3, red_channel, green_channel, blue_channel);

        % 方法2：使用repmat
        single_channel = 0.5 * ones(100, 100);
        rgb_image = repmat(single_channel, [1, 1, 3]);
        rgb_image(:, :, 1) = 0.8;  % 红色通道
        rgb_image(:, :, 2) = 0.5;  % 绿色通道
        rgb_image(:, :, 3) = 0.2;  % 蓝色通道
        ```

    * 数值计算

        ```matlab
        % 计算网格点的距离
        x = 1:3;
        y = 1:2;

        % 创建坐标网格
        X = repmat(x, length(y), 1);      % [1 2 3; 1 2 3]
        Y = repmat(y', 1, length(x));     % [1 1 1; 2 2 2]

        % 计算到原点的距离
        distance = sqrt(X.^2 + Y.^2);
        % distance = [sqrt(2) sqrt(5) sqrt(10);
        %             sqrt(5) sqrt(8) sqrt(13)]
        ```

* repmat 高级技巧

    * 创建结构体数组

        ```matlab
        % 创建空结构体模板
        template.name = '';
        template.value = 0;
        template.valid = false;

        % 复制创建结构体数组
        n = 5;
        data = repmat(template, 1, n);

        % 批量初始化
        for i = 1:n
            data(i).name = sprintf('Item%d', i);
            data(i).value = i * 10;
            data(i).valid = true;
        end
        ```

    * 处理单元数组

        ```matlab
        % 创建单元数组模板
        cell_template = {'', [], false};

        % 复制创建
        cell_array = repmat(cell_template, 3, 2);
        % 3×2的单元数组，每个元素是 {'', [], false}

        % 填充数据
        for i = 1:size(cell_array, 1)
            for j = 1:size(cell_array, 2)
                cell_array{i, j} = {sprintf('Cell(%d,%d)', i, j), i*j, mod(i*j,2)==0};
            end
        end
        ```

    * 生成重复序列

        ```matlab
        % 创建重复模式
        pattern = [1 2 3 4];
        repeats = 3;
        sequence = repmat(pattern, 1, repeats);
        % sequence = [1 2 3 4 1 2 3 4 1 2 3 4]

        % 创建带间隔的序列
        base = [1 0 0 0];  % 1后面跟3个0
        sequence = repmat(base, 1, 4);
        % sequence = [1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0]
        ```

* repmat 常见错误和注意事项

    * 维度匹配

        ```matlab
        % 错误：维度不匹配
        A = [1 2 3];
        B = repmat(A, 2.5, 3);  % 错误：2.5不是整数

        % 正确：使用整数
        B = repmat(A, 2, 3);    % 正确
        ```

    * 内存限制

        ```matlab
        % 大矩阵复制可能导致内存不足
        A = rand(10000, 10000);  % ~800MB
        B = repmat(A, 2, 2);     % ~3.2GB，可能内存不足

        % 解决方案：使用稀疏矩阵或分块处理
        A_sparse = sparse(A);
        B_sparse = repmat(A_sparse, 2, 2);  % 更节省内存
        ```

    * 替代函数

        ```matlab
        % 对于简单重复，考虑使用：
        A = [1 2 3];

        % 方法1: 使用ones
        B = A(ones(3,1), :);     % 垂直重复

        % 方法2: 使用索引
        B = A(repmat(1:size(A,1), 3, 1), :);

        % 方法3: 使用kron（克罗内克积）
        B = kron(A, ones(3,1));  % 垂直重复
        ```

    * 性能提示

        ```matlab
        % 测试不同方法的性能
        A = rand(1000, 1000);
        n = 100;

        % 方法1: repmat
        tic;
        for i = 1:n
            B = repmat(A, 2, 2);
        end
        t1 = toc;

        % 方法2: 使用ones索引（可能更快）
        tic;
        for i = 1:n
            B = A(ones(2,1), ones(2,1), :);  % 需要调整维度
        end
        t2 = toc;

        fprintf('repmat: %.4f秒，索引: %.4f秒\n', t1, t2);
        ```

* matlab repmat 与 python repeat dim 对比

    | 功能 | MATLAB | NumPy | 说明 |
    | - | - | - | - |
    | 块复制 | `repmat(A, m, n)` | `np.tile(A, (m, n))` | 复制整个数组 |
    | 元素复制 | 需用 kron 或索引 | `np.repeat(A, repeats, axis)` | 复制单个元素 |
    | 多维复制 | `repmat(A, [m n p])` | `np.tile(A, (m, n, p))` | 多维度扩展 |

    **详细对比分析**

    1. 块复制（整个数组复制）

        MATLAB repmat:

        ```matlab
        A = [1 2; 3 4];
        B = repmat(A, 2, 3);
        % B = [1 2 1 2 1 2
        %      3 4 3 4 3 4
        %      1 2 1 2 1 2
        %      3 4 3 4 3 4]
        ```

        NumPy tile (对应功能):

        ```python
        import numpy as np
        A = np.array([[1, 2], [3, 4]])
        B = np.tile(A, (2, 3))
        # B = [[1 2 1 2 1 2]
        #      [3 4 3 4 3 4]
        #      [1 2 1 2 1 2]
        #      [3 4 3 4 3 4]]
        ```

    2. 元素复制（单个元素重复）

        MATLAB 没有直接对应函数，需要变通：

        ```matlab
        % 方法1: 使用 kron (克罗内克积)
        A = [1 2; 3 4];
        B = kron(A, ones(2, 2));
        % B = [1 1 2 2
        %      1 1 2 2
        %      3 3 4 4
        %      3 3 4 4]
        % 每个元素变成2×2块

        % 方法2: 使用索引
        A = [1 2 3];
        B = A(ones(3,1), :);  % 每行重复3次
        % B = [1 2 3
        %      1 2 3
        %      1 2 3]
        ```

        NumPy repeat (直接支持):

        ```python
        import numpy as np
        A = np.array([[1, 2], [3, 4]])

        # 沿axis=0重复（行方向）
        B = np.repeat(A, 2, axis=0)
        # B = [[1 2]
        #      [1 2]
        #      [3 4]
        #      [3 4]]

        # 沿axis=1重复（列方向）
        C = np.repeat(A, 2, axis=1)
        # C = [[1 1 2 2]
        #      [3 3 4 4]]

        # 每个元素重复不同次数
        D = np.repeat(A, [2, 3], axis=0)  # 第1行重复2次，第2行重复3次
        ```

    3. 三维及多维数组复制

        MATLAB:

        ```matlab
        A = reshape(1:8, [2, 2, 2]);
        % A(:,:,1) = [1 3; 2 4]
        % A(:,:,2) = [5 7; 6 8]

        B = repmat(A, [2, 1, 2]);  % 行×2，列×1，页×2
        size(B)  % [4, 2, 4]
        ```

        NumPy:

        ```python
        import numpy as np
        A = np.arange(1, 9).reshape(2, 2, 2)

        # tile: 整个数组复制
        B = np.tile(A, (2, 1, 2))  # 对应MATLAB的repmat
        print(B.shape)  # (4, 2, 4)

        # repeat: 元素复制
        C = np.repeat(A, 2, axis=2)  # 沿第三维度复制
        print(C.shape)  # (2, 2, 4)
        ```

    三、功能映射表

    * 从 NumPy 到 MATLAB:

        ```python
        # NumPy 代码
        np.tile(A, (2, 3))      → repmat(A, 2, 3)      # MATLAB
        np.repeat(A, 3, axis=0) → A(ones(3,1), :, :)   # MATLAB (3D时需要调整)
        np.repeat(A, 3, axis=1) → A(:, ones(3,1), :)   # MATLAB (3D时需要调整)
        np.repeat(A, 3, axis=2) → A(:, :, ones(3,1))   # MATLAB
        ```

    * 从 MATLAB 到 NumPy:

        ```matlab
        % MATLAB 代码
        repmat(A, 2, 3)     → np.tile(A, (2, 3))          # NumPy
        kron(A, ones(2,2))  → np.repeat(np.repeat(A, 2, axis=0), 2, axis=1)  # NumPy
        A(ones(3,1), :)     → np.repeat(A, 3, axis=0)     # NumPy
        ```

    四、实际应用场景对比

    * 场景1：创建网格数据

        MATLAB (常用meshgrid):

        ```matlab
        % 创建2D网格
        x = 1:3;
        y = 1:2;
        [X, Y] = meshgrid(x, y);

        % 用repmat实现
        X_rep = repmat(x, length(y), 1);
        Y_rep = repmat(y', 1, length(x));
        ```

        NumPy (常用meshgrid):

        ```python
        import numpy as np
        x = np.arange(1, 4)
        y = np.arange(1, 3)
        X, Y = np.meshgrid(x, y)

        # 用tile实现
        X_tile = np.tile(x, (len(y), 1))
        Y_tile = np.tile(y.reshape(-1, 1), (1, len(x)))
        ```

    * 场景2：批量向量运算

        MATLAB (广播机制R2016b+):

        ```matlab
        % 旧方法：需要repmat
        A = rand(3, 4);
        row_vector = [1 2 3 4];
        B = A + repmat(row_vector, size(A,1), 1);

        % 新方法：自动广播
        B = A + row_vector;  % R2016b+ 支持
        ```

        NumPy (天然支持广播):

        ```python
        import numpy as np
        A = np.random.rand(3, 4)
        row_vector = np.array([1, 2, 3, 4])
        B = A + row_vector  # 自动广播
        ```

    * 场景3：图像处理中的通道复制

        MATLAB:

        ```matlab
        % 灰度图转RGB伪彩色
        gray_img = imread('gray.jpg');
        height = size(gray_img, 1);
        width = size(gray_img, 2);

        % 方法1: 使用cat
        rgb_img = cat(3, gray_img, gray_img, gray_img);

        % 方法2: 使用repmat
        rgb_img = repmat(gray_img, [1, 1, 3]);
        ```

        NumPy:

        ```python
        import numpy as np
        import cv2

        gray_img = cv2.imread('gray.jpg', cv2.IMREAD_GRAYSCALE)
        height, width = gray_img.shape

        # 方法1: 使用stack
        rgb_img = np.stack([gray_img, gray_img, gray_img], axis=2)

        # 方法2: 使用tile
        rgb_img = np.tile(gray_img[:, :, np.newaxis], (1, 1, 3))

        # 方法3: 使用repeat
        rgb_img = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
        ```

    五、性能对比
    
    测试代码：
    
    MATLAB:

    ```matlab
    function test_performance()
        A = rand(1000, 1000);
        
        % 测试repmat
        tic;
        for i = 1:100
            B = repmat(A, 2, 2);
        end
        repmat_time = toc;
        
        % 测试索引方法
        tic;
        for i = 1:100
            B = A(ones(2,1), ones(2,1), :);
        end
        index_time = toc;
        
        fprintf('repmat: %.4f秒\n', repmat_time);
        fprintf('索引: %.4f秒\n', index_time);
    end
    ```
    
    Python/NumPy:

    ```python
    import numpy as np
    import time

    A = np.random.rand(1000, 1000)

    # 测试tile
    start = time.time()
    for i in range(100):
        B = np.tile(A, (2, 2))
    tile_time = time.time() - start

    # 测试repeat
    start = time.time()
    for i in range(100):
        B = np.repeat(np.repeat(A, 2, axis=0), 2, axis=1)
    repeat_time = time.time() - start

    print(f"tile: {tile_time:.4f}秒")
    print(f"repeat: {repeat_time:.4f}秒")
    ```

    六、关键差异总结
    
    1. 设计哲学差异
    
        MATLAB：repmat 专注于块复制，设计简单统一

        NumPy：区分 tile (块复制) 和 repeat (元素复制)，功能更精细

    2. 语法差异
    
        ```matlab
        % MATLAB - 参数顺序不同
        B = repmat(A, m, n, p)      % 直接列出各维度重复次数
        ```

        ```py
        # NumPy - 使用元组
        B = np.tile(A, (m, n, p))   # 重复次数放在元组中
        ```

    3. 广播机制

        MATLAB：从 R2016b 开始支持自动广播，减少对 repmat 的依赖

        NumPy：从一开始就支持广播，更自然地处理维度扩展

    4. 内存布局考虑

        ```matlab
        % MATLAB：列优先（Fortran风格）
        A = [1 2 3; 4 5 6];  % 内存顺序：1, 4, 2, 5, 3, 6
        ```

        ```py
        # NumPy：默认行优先（C风格），可指定
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 内存顺序：1, 2, 3, 4, 5, 6
        ```

    七、迁移建议

    * 从 NumPy 迁移到 MATLAB:

        ```py
        np.tile(A, reps) → repmat(A, reps)（直接对应）

        np.repeat(A, n, axis) → 需要手动实现：
        ```

        ```matlab
        % axis=0 (行方向)
        B = A(ones(n,1), :, :, ...);  % 根据维度调整

        % axis=1 (列方向)  
        B = A(:, ones(n,1), :, ...);

        % 或用 kron 实现元素级复制
        B = kron(A, ones(n, 1));  % 特定情况
        ```

    * 从 MATLAB 迁移到 NumPy:

        `repmat(A, m, n)` → `np.tile(A, (m, n))`

        `kron(A, ones(m,n))` → `np.repeat(np.repeat(A, m, axis=0), n, axis=1)`

        `A(ones(m,1), :)` → `np.repeat(A, m, axis=0)`

    八、现代替代方案
    
    * MATLAB 广播机制（R2016b+）

        ```matlab
        % 不再需要 repmat 的场景
        A = rand(3, 4);
        v = [1 2 3 4];

        % 旧方法
        B = A + repmat(v, size(A,1), 1);

        % 新方法（更简洁高效）
        B = A + v;  % 自动广播
        ```

    * NumPy 广播机制

        ```python
        # NumPy 一直支持广播
        A = np.random.rand(3, 4)
        v = np.array([1, 2, 3, 4])
        B = A + v  # 自动广播
        ```

    九、结论

    repmat 与 NumPy 的对应关系：

    * `repmat ≈ np.tile`：都是块复制，复制整个数组

    * `repmat ≠ np.repeat`：repmat 不直接支持元素级复制

    主要差异原因：

    * 设计目标不同：MATLAB 强调数学计算简洁性，NumPy 提供更细粒度控制

    * 历史原因：MATLAB 先有 repmat，后来 NumPy 设计时区分了 tile 和 repeat

    * 使用场景：MATLAB 用户更多进行矩阵运算，NumPy 用户需要更灵活的数据操作

    实用建议：

    * 熟悉两种工具的差异，避免直接移植代码时的误解

    * 了解各自的最优实践（如 MATLAB 的广播、NumPy 的向量化）

    * 根据具体需求选择合适的方法，而非强行对应

* matlab repmat 与其他函数的对比

    * repmat vs. kron

        ```matlab
        % repmat: 块复制
        A = [1 2; 3 4];
        B_repmat = repmat(A, 2, 2);
        % [A A; A A]

        % kron: 克罗内克积（元素级复制）
        B_kron = kron(A, ones(2, 2));
        % [1*ones(2) 2*ones(2); 3*ones(2) 4*ones(2)]
        % 即每个元素都变成2×2块
        ```

    * repmat vs. 索引复制

        ```matlab
        A = [1 2 3];

        % 方法1: repmat
        B = repmat(A, 3, 1);

        % 方法2: 索引（更高效）
        B = A(ones(3,1), :);

        % 方法3: 使用ones创建索引
        B = A(repmat(1:size(A,1), 3, 1), :);
        ```

    * repmat vs. meshgrid/ndgrid

        ```matlab
        % 创建2D网格
        x = 1:3; y = 1:2;

        % meshgrid (主要用于绘图)
        [X1, Y1] = meshgrid(x, y);

        % 用repmat实现
        X2 = repmat(x, length(y), 1);
        Y2 = repmat(y', 1, length(x));

        % 验证
        isequal(X1, X2)  % true
        isequal(Y1, Y2)  % true
        ```

* matlab 广播机制（R2016b+）

    ```matlab
    % 旧方法：需要repmat
    A = rand(1000, 1000);
    row_mean = mean(A, 2);
    A_centered = A - repmat(row_mean, 1, size(A, 2));

    % 新方法：自动广播（更高效、更简洁）
    A_centered = A - row_mean;  % 自动扩展维度
    ```

* 获取变量信息

    ```matlab
    size(data)      % 显示形状
    class(data)     % 数据类型
    length(data)    % 长度（向量）
    ndims(data)     % 维度数
    ```

    注：

    1. `length(data)`显示的是第 1 个维度的长度

    1. 这些信息是只有矩阵才有吧？struct / cell / table 有吗？

* 直接使用索引查看部分数据

    ```matlab
    % 查看前 5 行
    data(1:5, :)  % 对于矩阵
    data(1:5)     % 对于向量

    % 查看后 5 行
    data(end-4:end, :)  % 矩阵
    data(end-4:end)     % 向量
    ```

    注：

    1. matlab 不支持`data(:5)`或`data(5:)`这种省略部分索引的写法

    1. 对于二维矩阵，如果只写了一维的索引，比如`data(1:10)`，那么它指的是将原矩阵做取消第一个维度的 flatten 操作后，再索引。

        即从第 1 列开始，下面跟第 2 列，第 3 列…… 以此类推。

        example:

        ```
        a = [ 1 3 5
              2 4 6 ]
        ```

        使用`a(1:3)`索引时，

        ```matlab
        % 先将 a 变成
        a_1 = [ 1
                2
                3
                4
                5
                6 ]

        % 再取转置
        a_2 = [ 1 2 3 4 5 6 ]
        
        % 最后拿到结果
        a_3 = [ 1 2 3 ]
        ```

        注意，最终拿到的结果是行向量。

    1. 索引的后端都是包含在内的，即`data(ind_1:ind_2)`中的`ind_2`是 included 的，整体为闭区间。

* 测试一个数组是否为空数组：`isempty()`

* 将 nan 转换为 0

    ```matlab
    i = find(isnan(a));
    a(i) = zeros(size(i));  % changes NaN into zeros
    ```

* 查看维度：`size()`，查看长度：`length()`（行数或列数的最大值），元素的总数：`numel()`

* `reshape`

    syntax:

    * `reshape(A, sz)`
    * `reshape(A, sz1, ..., szN)`

* `repmat`

    对于一个矩阵`A`，`repmat`可以将其维度进行重复。对于一个向量`A`，`repmat`可以先为其增加一个维度，然后再按矩阵做重复。对于一个纯量`A`，`repmat`可以构建具有重复元素的数组。

    其实无论输入的是矩阵，向量还是纯量，`repmat`都是先把其变换成至少两个维度的矩阵，然后再处理。shape 为`(n, )`的一维向量，会先变成`(1, d)`，shape 为`(1, )`的纯量，会先变成`(1, 1)`。

    `repmat`的处理方式很奇怪，如果待处理矩阵`A`后只有一个参数`r1`，那么会把`A`的前两个维度翻`r1`倍。如果`A`后有大于 1 个参数`(r1, r2, ....)`，`repmat`则会按对应位置对维度进行翻倍。

    syntax:

    * `repmat(A, n)`
    * `repmat(A, r1, ..., rN)`
    * `repmat(A, r)`

    其中`r1`，...，`rN`表示在这些维度上重复几遍。

    Examples:

    ```matlab
    a = 1
    b = zeros(3)
    ```

* `cat`

    syntax:

    * `C = cat(dim, A, B)`
    * `C = cat(dim, A1, A2, ..., An)`

* `squeeze`

    syntax:

    * `squeeze(A)`

    删除矩阵中大小为 1 的维度。
    
    具体实现的话不同情况挺复杂的，看例子理解吧。

    Examples:

    ```matlab
    a = 1
    size(squeeze(a))  % (1, 1)

    b = zeros(1, 3)
    size(squeeze(b))  % (1, 3)

    c = zeros(1, 3, 1)
    size(squeeze(c))  % (1, 3)

    d = zeros(1, 3, 1, 2)
    size(squeeze(d))  % (3, 2)
    ```

* `sub2ind`

    syntax:

    * `ind = sub2ind(sz, row, col)`
    * `ind = sub2ind(sz, I1, I2, ..., In)`

    将多维索引拉伸成一维索引。

    `ind2sub()`

* `permute`

    syntax:

    * `B = permute(A, dimorder)`

    重排维度。相当于 numpy 中的`transpose`。

    `ipermute()`

* `size`

    syntax:

    * `sz = size(A)`
    * `szdim = size(A, dim)`
    * `szdim = size(A, dim1, dim2, ..., dimN)`
    * `[sz1, ..., szN] = size(___)`

* `N = ndims(A)`

    获取维度的数量

* 其他

    * `cat()`

    * `flipdim()`

    * `shiftdim()`

* 矩阵信息

    * `length(A)`

        一个数组的行数和列数的最大值。

    * `numel(A)`

        数组元素总数。

    * `[a, b] = size(A)`
    
        数组的行数和列数。

* 矩阵测试

    * `isempty()`

        用于检测某个数组是否为空数组。

        ```matlab
        TF = isempty(A)
        ```

    * `isscalar()`

        检测某个数组是否为单元素的标量数组。

        ```matlab
        TF = isscalar(A)
        ```

    * `isvector()`

        检测某个数组是否为只有一行元素或一列元素。

    * `issparse()`

        检测某个数组是否为稀疏数组。

* 排序

    ```matlab
    a = rand(1, 10)
    b = sort(a)
    [b, index] = sort(a)
    ```

    排序二维数组时可以指定维度：

    ```matlab
    [b, index] = sort(A, dim, mode)  % 其中 mode 可以取 'ascend', 'descend'
    ```

    默认情况下，matlab 会对`dim=1`维度进行排序，而 numpy 中的`sort`则会对`axis=-1`维度进行排序。matlab 与 numpy 的相同点是，第二个维度指的都是行，第一个维度指的都是列。

* 找到为`1`或`true`的索引：`find()`

* 整数类型：

    `int8()`, `int16()`, `int32()`, `int64()`, `uint8()`, `uint16()`, `uint32()`, `uint64()`

* 整数的溢出会变成最小值和最大值：

    ```matlab
    k = cast('hellothere', 'uint8');  % k = 104 101 108 108 111 116 104 101 114 101

    double(k) + 150;  % ans = 254 251 258 261 266 254 251 264 251

    k + 150;  % ans = 254 251 255 255 255 255 254 251 255 251

    k - 110;  % and = 0 0 0 0 1 6 0 0 4 0
    ```

* 浮点类型：

    `single()`, `double()`

    ```matlab
    a = zeros(1, 5, 'single')

    d = cast(6:-1:0, 'single')  % 转换单精度与双精度
    ```

* 单精度与双精度浮点数之间的运算结果是单精度浮点数。

* 判断是否为`nan`或`inf`：`isnan()`, `isinf()`。在 matlab 中，不同的`NaN`互不相等。不可以使用`a == nan`判断。

* 可以使用赋空操作删除某一行或列：

    ```matlab
    A(:, 2:3, :) = []
    ```

* 运算符：除法为`\`或`/`，乘方为`^`。

* 矩阵扩展

    ```matlab
    % 直接使用索引扩展
    a = reshape(1:9, 3, 3)
    a(5, 5) = 111
    a(:, 6) = 222
    aa = a(:, [1:6, 1:6])

    % 使用分号扩展行
    b = ones(2, 6)
    ab_r = [a; b]

    % 使用逗号扩展列
    ab_c = [a, b(:, 1:5)']
    ```

* 逻辑运算

    ```matlab
    A & B
    and(A, B)  % 若两个数均非 0 值，则结果为 1

    A | B
    or(A, B)  % 若两个数有一个不为 0，则结果为 1

    ~A
    not(A)  % 若待运算矩阵的元素为 0，则结果元素为 1

    xor(A, B)  % 若一个为 0，一个不为 0，则结果为 1
    ```

* 数值运算

    ```matlab
    cumsum(a) 
    sum(a)
    dot(a, b)
    cross(a, b)  % 叉乘运算
    prod(a)
    cumprod(a)
    triu(a, k)  % 提取上三角矩阵
    tril(a, k)
    flipud()  % 矩阵翻转
    fliplr()
    rot90()
    ```

* 直接使用`a * b`做的是矩阵乘法，想做逐元素相乘可以用`a .* b`。

* 使用`sub2ind()`计算二维索引在拉伸为一维的数组中的索引：

    ```matlab
    b = sub2ind(size(a), [2, 3], [2, 1])
    a(b)
    ```

* 数组索引

    ```matlab
    a = [1, 3, 4, 5, 6, 7]
    a(5)
    a([1,3,5])  % 访问多个元素
    a(1:2:5)  % 访问多个元素
    a(find(x>3))  % 按条件访问多个元素

    a = zeros(2, 6)
    a(:) = 1:12  % 按照一维方式访问
    a(2, 4)
    a(8)
    a(:, [1,3])
    a([1, 2, 5, 6]')
    a(:, 4:end)
    a(2, 1:2:5)
    a([1, 2, 2, 2], [1, 3, 5])
    ```

* 转置

    `a'` 或 `transpose(a)`
    
    这两种方法只适用二维数组（包含向量），如果维度超过二维会报错。

* 可以使用`a([2, 3, 4])`进行多元素索引，也可以使用`a(1:2:10)`这样的方式。

### 矩阵创建

* matlab 中的`1:10`包含 1 和 10。即 end included

* 输入矩阵元素创建矩阵

    ```matlab
    a = [1, 2, 3; 4, 5, 6; 7, 8, 9]

    a = [1 2 3 4
        5 6 7 8
        0 1 2 3]
    ```

* 指定索引元素创建矩阵

    ```matlab
    B(1,2) = 3;
    B(4,4) = 6;
    B(4,2) = 11;
    ```

    matlab 会自动构建一个剩下元素都是 0 的矩阵：

    ```matlab
    B = 
    0 3 0 0
    0 0 0 0
    0 0 0 0
    0 11 0 6
    ```

* 创建高维矩阵

    ```matlab
    % 直接创建
    A = zeros(2, 3)
    % 使用索引创建
    A(:, :, 4) = zeros(2, 3)
    ```

* 常用的二维数组生成函数

    ```matlab
    zeros(2, 4)
    ones(2, 4)
    randn('state', 0)  % 把正态随机发生器置 0
    randn(2, 3)  % 产生正态随机矩阵
    D = eye(3)  % 产生 3 x 3 的单位矩阵
    diag(D)  % 取 D 矩阵的对角元
    diag(diag(D))  % 外 diag() 利用一维数组生成对角矩阵
    randsrc(3, 10, [-3, -1, 1, 3], 1)  % 在[-3, -1, 1, 3]中产生 3 x 10 的均匀分布随机数组，随机发生器的状态设置为 1
    ```

* `linspace()`：线性向量生成

    `linspace(x1, x2)`默认生成 100 个点，也可以用`linspace(x1, x2, n)`指定生成的采样点数量。

    `logspace`：等比数列生成

* `start:step:end`: 按间隔生成向量

    `start:end`按步长为 1 生成向量，比如`1:5`生成向量`[1 2 3 4 5]`，注意`end`是包括在区间内的。

    `start:step:end`可以按指定步长生成向量。

* 直接创建向量

    ```matlab
    a = [1 2 3]  % shape: (1, 3)
    a = [1, 2, 3]  % shape: (1, 3)
    a = [1; 2; 3]  % shape: (3, 1)
    ```

    matlab 里没有纯量和一维向量，只有维度从二起始的矩阵。
    
    为了方便，这里把 shape 为`(1, n)`或`(n, 1)`的矩阵称为向量，把 shape 为`(1, 1)`的矩阵称为数字或纯量。把 shape 为`(m, n)`的二维矩阵或三维以上的矩阵称为矩阵或张量。

    可以用`size(a)`查看矩阵`a`的 shape。

    example:

    ```matlab
    >> a = [1 2 3]

    a =

        1     2     3

    >> size(a)

    ans =

        1     3

    >> b = [1; 2; 3]

    b =

        1
        2
        3

    >> size(b)

    ans =

        3     1
    ```

### 快捷键

* 注释/取消注释

    ```matlab
    % 注释选中行：Ctrl + R
    % 取消注释：Ctrl + T

    % 示例：
    % 选中下面三行，按 Ctrl+R
    x = 1;
    y = 2;
    z = x + y;

    % 再按 Ctrl+T 取消注释
    ```

    块注释：

    ```matlab
    %{
    这是一个多行注释块
    可以跨越多行
    这里面的代码不会执行
    x = 1;
    y = 2;
    %}
    ```

    代码段折叠:

    ```matlab
    %% 节标题（双百分号）
    % 可以折叠代码段
    x = linspace(0, 10, 100);
    y = sin(x);

    %% 另一个节
    plot(x, y);
    title('正弦波');
    ```

* clear

    ```matlab
    clear        % 清除工作区所有变量
    clear x y    % 只清除变量x和y
    clear all    % 清除所有变量、函数、MEX文件等（最彻底）
    clear classes % 清除类定义
    clear functions % 清除编译的函数

    % 示例
    >> a = 1; b = 2; whos  % 查看当前变量
    >> clear a            % 只清除a
    >> clear              % 清除所有变量
    ```

* clc

    ```matlab
    clc  % Clear Command Window - 清空命令窗口显示
    % 只清除显示内容，不影响变量和程序运行

    % 示例
    >> disp('这行文字会显示')
    >> clc  % 清屏，但变量仍然存在
    >> a = 5;  % a变量没有被清除
    ```

    example:

    ```matlab
    % 典型用法：脚本开头
    clear; clc; close all;  % 黄金三连
    % clear: 清除旧变量，避免冲突
    % clc: 清屏，让输出更清晰
    % close all: 关闭所有图形窗口

    % 实际应用
    clear; clc;
    fprintf('========== 程序开始 ==========\n\n');
    ```

### 函数

* matlab 函数类型和调用示例

    * 单输出函数

        ```matlab
        % circle_area.m
        function area = circle_area(radius)
            area = pi * radius^2;
        end

        % 调用
        area = circle_area(5);  % 计算半径为5的圆面积
        ```

        注：

        1. 函数文件中只能写函数，不能写调用。

            调用只能写在脚本文件中。

    * 多输出函数

        ```matlab
        % stats.m
        function [mean_val, std_val] = stats(data)
            mean_val = mean(data);
            std_val = std(data);
        end

        % 调用方式1：获取所有输出
        [avg, deviation] = stats([1 2 3 4 5]);

        % 调用方式2：只获取第一个输出
        avg_only = stats([1 2 3 4 5]);

        % 调用方式3：使用波浪线忽略输出
        [~, std_only] = stats([1 2 3 4 5]);
        ```

    * 无输出函数

        ```matlab
        % plot_data.m
        function plot_data(x, y)
            figure;
            plot(x, y, 'b-', 'LineWidth', 2);
            xlabel('X'); ylabel('Y');
            title('数据图');
            grid on;
        end

        % 调用
        x = 0:0.1:10;
        y = sin(x);
        plot_data(x, y);  % 只执行绘图，不返回数据
        ```

    * 带可选参数的函数

        ```matlab
        % calculate.m
        function result = calculate(a, b, operation)
            if nargin < 3  % 检查输入参数数量
                operation = 'add';
            end
            
            switch operation
                case 'add'
                    result = a + b;
                case 'subtract'
                    result = a - b;
                case 'multiply'
                    result = a * b;
                otherwise
                    error('不支持的操作');
            end
        end

        % 调用
        result1 = calculate(3, 5);          % 默认加法
        result2 = calculate(3, 5, 'multiply');  % 指定乘法
        ```

        注：

        1. matlab 不支持函数的默认参数

* matlab 嵌套函数和子函数

    * 嵌套函数

        ```matlab
        % outer.m
        function outer(x)
            disp('外层函数');
            nested_func(x);
            
            function nested_func(y)
                disp(['嵌套函数接收到: ', num2str(y)]);
            end
        end

        % 调用
        outer(10);
        ```
        
        注：

        * `nested_func()`可以写在调用前，也可以写在调用后

    * 子函数（同一文件多个函数）

        ```matlab
        % mainfile.m
        function mainfile()
            data = [1 2 3 4 5];
            result1 = subfunc1(data);
            result2 = subfunc2(data);
            disp(['结果: ', num2str(result1), ', ', num2str(result2)]);
        end

        % 子函数1
        function res1 = subfunc1(d)
            res1 = mean(d);
        end

        % 子函数2
        function res2 = subfunc2(d)
            res2 = sum(d);
        end
        ```

        注：

        1. 子函数必须写在主函数的下面。

            即文件的第一个函数必须是以文件名命名的函数。

* matlab 函数

    * 文件命名规则

        ```matlab
        % 文件名必须与函数名相同！
        % myfunction.m 文件内容：
        function output = myfunction(input1, input2)
            % 函数体
            output = input1 + input2;
        end
        ```

    * 直接调用（同目录）

        ```matlab
        % 当函数文件在当前目录或MATLAB路径中时
        result = myfunction(3, 5);
        disp(result);  % 输出 8
        ```

    * 通过函数句柄调用

        ```matlab
        % 创建函数句柄
        myfunc = @myfunction;

        % 使用句柄调用
        result = myfunc(3, 5);

        % 匿名函数句柄（适用于简单函数）
        square = @(x) x^2;
        result = square(4);  % 返回 16
        ```

    * 带路径调用

        ```matlab
        % 如果函数在不同文件夹
        result = C:\MyFunctions\myfunction(3, 5);

        % 或先添加到路径
        addpath('C:\MyFunctions');
        result = myfunction(3, 5);
        ```

* matlab 私有函数

    创建私有文件夹

    ```matlab
    project_folder/
    ├── main.m
    └── private/
        └── helper.m  % 私有函数，只能被父文件夹中的函数调用
    ```

    ```matlab
    % private/helper.m
    function result = helper(input)
        result = input * 2;
    end

    % main.m
    function main()
        x = 5;
        y = helper(x);  % 可以调用私有函数
        disp(y);
    end
    ```

* matlab 函数调试和问题排查

    * 常见错误及解决

        ```matlab
        % 错误：未定义的函数
        % 解决方法：检查路径或添加路径
        which myfunction  % 查看函数位置
        addpath('函数所在文件夹路径');

        % 错误：函数名与内置函数冲突
        % 解决方法：重命名函数或使用which检查
        which myfunction  % 查看是否被内置函数覆盖
        ```

    * 查看函数信息

        ```matlab
        % 查看函数代码
        type myfunction

        % 查看函数帮助
        help myfunction

        % 查看函数输入输出
        nargin('myfunction')   % 输入参数数量
        nargout('myfunction')  % 输出参数数量
        ```

* matlab 可变参数函数

    * 使用 nargin 判断参数数量

        ```matlab
        function result = myfunction(a, b, c)
            % 设置默认值
            if nargin < 3 || isempty(c)
                c = 10;  % c的默认值为10
            end
            if nargin < 2 || isempty(b)
                b = 5;   % b的默认值为5
            end
            
            result = a + b + c;
        end

        % 调用
        result1 = myfunction(1);        % a=1, b=5, c=10
        result2 = myfunction(1, 2);     % a=1, b=2, c=10
        result3 = myfunction(1, 2, 3);  % a=1, b=2, c=3
        ```

    * 使用 varargin 处理可变参数

        ```matlab
        function result = myfunction(varargin)
            % 解析参数
            p = inputParser;  % 创建参数解析器
            
            % 定义参数和默认值
            addRequired(p, 'a', @isnumeric);
            addOptional(p, 'b', 5, @isnumeric);      % b默认=5
            addOptional(p, 'c', 10, @isnumeric);     % c默认=10
            addParameter(p, 'mode', 'fast', @ischar); % 键值对参数
            
            parse(p, varargin{:});
            
            % 使用参数
            a = p.Results.a;
            b = p.Results.b;
            c = p.Results.c;
            mode = p.Results.mode;
            
            if strcmp(mode, 'fast')
                result = a + b + c;
            else
                result = a * b * c;
            end
        end

        % 调用
        result1 = myfunction(1);                    % a=1, b=5, c=10, mode='fast'
        result2 = myfunction(1, 2, 'mode', 'slow'); % a=1, b=2, c=10, mode='slow'
        result3 = myfunction(1, 2, 3);              % a=1, b=2, c=3, mode='fast'
        ```

    * 使用 inputParser 类（最灵活强大）

        ```matlab
        function result = process_data(data, varargin)
            % 创建参数解析器
            p = inputParser;
            
            % 定义必需参数
            addRequired(p, 'data', @isnumeric);
            
            % 定义可选参数及其默认值
            addParameter(p, 'method', 'mean', @(x) ismember(x, {'mean', 'median', 'sum'}));
            addParameter(p, 'normalize', false, @islogical);
            addParameter(p, 'threshold', 0, @isnumeric);
            addParameter(p, 'verbose', true, @islogical);
            
            % 解析参数
            parse(p, data, varargin{:});
            
            % 获取参数值
            method = p.Results.method;
            normalize = p.Results.normalize;
            threshold = p.Results.threshold;
            verbose = p.Results.verbose;
            
            % 处理数据
            if normalize
                data = (data - min(data)) / (max(data) - min(data));
            end
            
            switch method
                case 'mean'
                    result = mean(data(data > threshold));
                case 'median'
                    result = median(data(data > threshold));
                case 'sum'
                    result = sum(data(data > threshold));
            end
            
            if verbose
                fprintf('方法: %s, 结果: %.2f\n', method, result);
            end
        end

        % 各种调用方式
        data = randn(100, 1);
        r1 = process_data(data);  % 使用所有默认值
        r2 = process_data(data, 'method', 'median', 'normalize', true);
        r3 = process_data(data, 'threshold', 0.5, 'verbose', false);
        ```

    * 从 R2019b 开始，MATLAB引入了参数块（Argument Blocks），更接近其他语言的默认参数

        ```matlab
        function result = new_function(a, b, options)
            % 参数声明块
            arguments
                a (1,1) double
                b (1,1) double = 5      % 默认值
                options.mode (1,1) string = "fast"    % 名称-值参数
                options.scale (1,1) double = 1.0
            end
            
            % 函数体
            if options.mode == "fast"
                result = (a + b) * options.scale;
            else
                result = a * b * options.scale;
            end
        end

        % 调用
        r1 = new_function(2, 3);  % b=3, 使用其他默认值
        r2 = new_function(2);     % b=5（默认值）
        r3 = new_function(2, 'mode', "slow", 'scale', 2.0);
        ```

### 常用命令与工程环境

* 注释：使用`%`进行单行注释。

* 在脚本中，两个`%`可以标志一个 block：

    ```matlab
    a = 1

    %% new block
    b = 2
    ```

    使用`Ctrl + Enter`可以运行 block 中的内容。

* 块注释

    ```matlab
    %{

        comments

    %}
    ```

* 文件与目录

    ```matlab
    pwd                     % 显示当前工作目录
    cd 'C:\MyFolder'        % 更改目录
    ls                      % 列出当前目录文件
    dir                     % 详细列表
    ```

* `whos`

    ```matlab
    % 查看工作区所有变量信息
    whos

    % 查看指定变量信息
    whos variable_name
    ```

* `help func_name`

* 查找函数：`lookfor keyword`

    `lookfor`搜索所有函数的第一行注释行，找到对应的文件或函数。

* `who`：列出当前工作空间中的变量

* `what`：列出当前文件夹或指定目录下的 M 文件、MAT 文件和 MEX 文件

* `which`：显示指定函数或文件的路径

* `whos`：列出当前工作空间中变量的更多信息

* `exist`：检查指定变量或文件的存在性

* `doc`：直接查询在线文档。通常更详细

* `echo`：直接运行时，切换是否显示 m 文件中的内容。也可以`echo on`，`echo off`来指定状态。

* 显示运算符优先顺序：`help precedence`

* 环境 -> 设置路径 可以为 Matlab 添加或删除搜索路径。

* 命令行中的快捷键

    * `Up`, `Ctrl + P`：调用上一行
    * `Down`, `Ctrl + N`：调用下一行
    * `Left`,  `Ctrl + B`：光标左移一个字符
    * `Right`, `Ctrl + F`：光标右移一个字符
    * `Ctrl + Left`, `Ctrl + L`：光标左移一个单词
    * `Ctrl + Right`, `Ctrl + R`：光标右移一个单词
    * `Home`, `Ctrl + A`：光标置于当前行开头
    * `End`, `Ctrl + E`：光标置于当前行末尾
    * `Esc`, `Ctrl + U`：清除当前输入行
    * `Del`, `Ctrl + D`：删除光标处的字符
    * `Backspace`, `Ctrl + H`：删除光标前的字符
    * `Alt + Backspace`：恢复上一次删除

* 数值的显示格式

    | 格式命令 | 作用 |
    | - | - |
    | `format short` | 5 位有效数字 |
    |`format long`|15 位有效数字|
    |`format short e`|5位有效数字 + 科学计数法|
    |`format long e`|15 位有效数字 + 科学计数法|
    |`format short g`|短紧缩格式|
    |`format long g`|长紧缩格式|
    |`format hex`|十六进制，浮点数|
    |`format bank`|2 位小数|
    |`format +`|正，负或 0|
    |`format rat`|有理数近似|
    |`format debug`|短紧缩格式的内部存储信息|

* 常用的控制命令

    * `cd`：显示或改变当前文件夹
    * `dir`：显示当前文件夹或指定文件夹下的文件
    * `clc`：清除工作窗口中的所有显示内容
    * `home`：将光标移至命令行窗口的最左上角
    * `clf`：清除图形窗口
    * `type`：显示文件内容
    * `clear`：清理内存变量
    * `echo`：工作窗口信息显示开关
    * `disp`：显示变量或文字内容
    * `load`：加载指定文件的变量
    * `diary`：日志文件命令
    * `!`：调用 dos 命令
    * `exit`，`quit`：退出 
    * `pack`：收集内存碎片
    * `hold`：图形保持开关
    * `path`：显示搜索目录
    * `save`：将内存变量保存到指定文件中

* `clear`

    删除工作区中的所有变量。

    `clear var1, var2`：清除指定变量

* `pack`

    用于整理内存。将内存中的数据先存储到磁盘上，再从磁盘将数据读入到内存中。

## note

* 与数据类型相关的函数

    ```matlab
    double
    single
    int8, int16, int32, int64
    uint8, uint16, uint32, uint64
    isnumeric
    isinteger
    isfloat
    isa(x, 'type')  % 其中 type 可以是 'numeric', `integer` 和 'float'，当 x 的类型为 type 时，返回 true
    cast(x, 'type')  % 将 x 类型置为 type
    intmin('type')  % type 类型的最小整数值
    realmax('type')  % type 类型的最大浮点实数值
    realmin('type')  % type 类型的最小浮点实数值
    eps('type')  % type 数据类型的 eps 值（浮点值）
    eps(x)  % x 的 eps 值
    zeros(..., 'type')
    ones(..., 'type')
    eye(..., 'type')
    ```

常用的初等函数：

```matlab
sin
sind  % 正弦，输入以度为单位
sinh  % 双曲正弦
asin  % 反正弦
asind
asinh
cos
cosd
cosh
acos
acosd
acosh
tan
tand
tanh
atan
atand
atan2  % 四象限反正切
sec  % 正割
secd
asec  % 反正割
asecd
asech
csc  % 余割
cscd
csch  % 双曲余割
acsc  % 反余割
acscd
acsch
cot  % 余切
cotd
coth
acot  % 反余切
acotd
acoth

exp
expm1  % 准确计算 exp(x)-1 的值
log
log1p  % 准确计算 log(1+x) 的值
log10
log2
realpow  % 对数，若结果是复数则报错
reallog  % 自然对数，若输入不是正数则报错
realsqrt  % 开平方根，若输入不是正数则报错
sqrt
nthroot  % 求 x 的 n 次方根
nextpow2  % 返回满足 2^P >= abs(N) 的最小正整数 P，其中 N 为输入

fix  % 向零取整
floor  % 向负无穷方向取整
ceil  % 向正无穷方向取整
round  % 四舍五入
mod  % 除法求余（与除数同号）
rem  % 除法求余（与被除数同号）
sign  % 符号函数
```

关系运算符：

`<`, `<=`, `>`, `>=`, `==`, `~=`

逻辑运算符：`&`, `|`, `~`

关系函数和逻辑函数：

```matlab
xor(x, y)  % 异或
any(x)  % 若 x 时向量，有任意一个不为 0 时返回 true，否则返回 false。若 x 是数组，则对于 x 的任意一列，如果有一个元素不为 0，则在该列返回 true，否则返回 false。
all(x)

ismember  % 检测一个值是否是某个集合中的元素
isglobal  % 检测一个变量是否是全局变量
mislocked  % 检测一个 M 文件是否被锁定
isempty  % 判断数组是否为空
isequal  % 检测两个数组是否相等
isequalwithequalNaN  % 检测两个数组是否相等，若数组中含有 NaN，则认为所有的 NaN 是相等的
isfinte  % 检测数组中各元素是否为有限值
isfloatpt  % 检测数组中各元素是否为浮点值
isscalar  % 检测一个变量是否为标量
isinf  % 检测数组中各元素是否为无穷大
islogical  % 检测一个数组是否为逻辑数组
isnan
isnumeric
isreal
isprime  % 检测一个数是否为素数
issorted  % 检测一个数组是否接序排列
automesh  % 如果输入参数是不同方向的向量，则返回 true
inpolygon  % 检测一个点是否位于一个多边形区域内
isvarname  % 检测一个变量名是否是一个合法的变量名
iskeyword  % 检测一个变量名是否是 matlab 的关键词或保留字
issparse  % 检测一个矩阵是否为稀疏矩阵
isvector  % 检测一个数组是否是一个向量
isappdata  % 检测应用程序定义的数据是否存在
ishandle  % 检测是否为图形句柄
ishold  % 检测一个图形是否为 hold 状态
figflag  % 检测一个图形是否是当前屏幕上显示的图形
iscellstr % 检测一个数组是否为字符串单元数组
ischar  % 检测一个数组是否为字符串数组
isletter  % 检测一个字符是否是英文字母
isspace  % 检测一个字符是否是空格字符
isa  % 检测一个对象是否为指定类型
iscell  % 检测一个数组是否为单元数组
isfield  % 检测一个名称是否是结构体中的域
isjava  % 检测一个数组是否是 java 对象数组
isobject  % 检测一个名称是否是一个对象
isstrcut  % 检测一个名称是否是一个结构体
isvalid  % 检测一个对象是否可以连接到硬件的串行端口对象
```

### 字符串

* 字符串是一个`1 x n`的`char`型数组，每个字符占 2 字节。

    ```matlab
    str = 'I am a great person ';  % 使用单引号直接创建

    str = ['second'; 'string']  % 字符串数组，要求字符串长度必须一致

    c = char('first', 'second')  % 使用 char 创建字符串数组时，如果字符串长度不同，char() 会自动在较短的字符串后加空格，使所有字符串长度相等
    ```

* 字符串常用函数：

    * `c = strcat(a, b)`

        删去字符串末尾的空格`' '`后进行拼接

    * `c = [a b]`

        不删除空格进行字符串拼接

    * `c = strvcat('name', 'string')`

        与`char()`作用类似。

    * `celldata = cellstr(c)`

        删除字符串末尾的空格，然后将字符串数组转换为字符串单元数组。

    * `chararray = char(celldata)`

        把一个字符串单元数组转换成一个字符数组。

    * `new_str = deblank(str)`

        删除字符串末尾的空格并返回。

    * 字符串比较

        `strcmp()`, `strcmpi()`, `strncmp()`, `strncmpi()`

        可以使用关系运算符对单个字符进行比较。

    * `isletter()`

        判断字符串的每个字符是否为一个字母。

    * `isspace()`

        判断字符串中的每个字符是否为空白字符（空格，制表符，换行符）

    * 查找和替换

        * `str = strrep(str1, str2, str3)`，把`str1`中的`str2`替换成`str3`

        * `k = findstr(str1, str2)`，查找输入中较长字符串中较短字符串的位置

        * `k = strfind(str, pattern)`，查找`str`中`pattern`出现的位置

        * `k = strfind(cellstr, pattern)`，查找单元字符串`cellstr`中`pattern`出现的位置

        * `strtok`：获得第一个分隔符之前的字符串

            `token = strtok('str')`，以空格符作为分隔符

            `token = strtok('str', delimiter)`，指定分隔符

            `[token, rem] = strtok(...)`，返回值`rem`为第1个分隔符之后的字符串（包含分隔符）

        * `strmatch`：在字符串数组中匹配指定的字符串

            * `x = strmatch('str', STRS)`，在字符串数组`STRS`中匹配字符串`str`，返回匹配上的字符串所在行的指标

            * `x = strmatch('str', STRS, 'exact')`：精确匹配，要求完全一致才算匹配上。

    * `upper()`，`lower()`，把整个字符串转换大小写

    * `eval()`：把字符串转换成数字

        `value = sscanf(string, format)`

        example:

        ```matlab
        v1 = sscanf('3.141593', '%g')  % 浮点数
        v2 = sscanf('3.141593', '%d')  % 整数
        ```

    * 数字转换成字符串

        `num2str()`, `int2str()`, `dex2hex()`，`hex2num()`，`hex2dec()`，`bin2dex()`，`dec2bin()`，`base2dec()`, `dec2base`

        `mat2str()`可以把一个数组转换成字符串。

        `sprintf()`，`fprintf()`，格式化字符串。

        `char()`可以按 ascii 码将数字转换成字符。

    * `str2num`, `uintN`, `str2double`, `hex2num`, `hex2dec`, `bin2dec`, `base2dec`

    * `double`

        把字符串转换成 ascii 形式

    * `blanks(n)`, `evalc(s)`

    * 其他的一些函数

        `isstrprop()`, `strtrim()`, `strjust()`

正则表达式：

* `regexp(str, pattern)`

    `regexpi()`

    `regexprep()`：使用正则表达式替换字符串

    查找单个字符的字符串表达式：

    ```matlab
    .  % 任意单个字符
    [abcd35]  % 查找方框中任意一个字符
    [a-zA-Z]  % 查找指定范围字母
    [^aeiou]  % 取反
    \s  % 任意空白符
    \S  % 任意非空白符
    \w  % 任意文字字符，字母，数字，下划线
    \W  % 任意非文字字符，相当于[^a-zA-Z_0-9]
    \d  % 任意数字，相当于[0-9]
    \D  % 任意非数字字符，[^0-9]
    \xN 或 \x{N}  % 查找十六进制为 N 的字符
    \oN 或 \o{N}  % 查找八进制为 N 的字符
    \a  % 查找警告
    \b  % 退格 
    \t  % 横向制表符
    \n  % 换行符
    \v  % 纵向制表符
    \f  % 换页符
    \r  % 回车符
    \e  % 退出符
    \., \*, \?, \\ 等  % 转义字符

    (p)  % 限制后面修饰符的作用范围
    (?:p)  % 不懂 
    (?>p)  % 不懂
    (?#A Comment)  % 插入注释
    \N  % 与该表达式中第 N 个标记相同，不懂
    $N  % 不懂
    (?<name>p)  % 
    \k<name>  %
    (?(T)p)  %  if then 结构
    (?(T)p|q)  % if then else 结构
    ```

### 结构体（structures）

创建：

```matlab
a.b = 1
a.c = [1, 2, 3]

s = struct('Name', 'John', 'Score', 85.5, 'Salary', [4500 4200])

repmat(struct('Name', 'John', 'Score', 85.5, 'Salary', [4500, 4200]), 1, 3)

struct('Name', {'klj', 'Dana', 'John'}, 'Score', {98, 92, 85.5}, 'Salary', {[4500 4200], [], []})
```

将属性作为数组返回：

```matlab
[obj.attr]
{obj.attr}
```

### 单元数组（cell array）

cell array 可看成 Python 中保持矩阵形状的 list，可以存储任何类型的值。

cell 通过直接赋值的方式创建：

```matlab
a = {1, [1, 2]; 3, 'str'};

a(1, 1) = {[1, 2, 3]};
a(1, 2) = {'abc'};
a(2, 1) = {1};
```

注意，此时`a`会变成 size 为`(4, 4)`的 cell，其中`a(2, 2)`是 size 为`(0, 0)`的矩阵。

正是因为对 cell 矩阵化，所以大部分对维度的操作对 cell 同样也适用。

如果使用花括号进行索引，那么等号右侧的值就不需要再加花括号了：

```matlab
a{1, 1} = [1, 2, 3]
```

可以使用`celldisp(A)`完整地显示一个 cell 中的内容。

cell array，使用`()`进行索引，只能得到一个 cell，而使用`{}`进行索引，可以得到 cell 中的内容。

可以使用`deal()`取多个单元元素的内容。

### 语句

`if`语句：

```matlab
if condition_1
    statement_1
elseif condition_2
    statement_2
else
    statement_3
end
```

对于`condition`，当其为空数组`[]`，空字符串`''`或全零矩阵（`0`，`[0, 0]`，`[0 0; 0 0]`等）时，`condition`为假，其余情况全为真。空 cell 无法判断真假，会报错。

`matlab`对缩进和换行没有要求，因此`if`必须以`end`结尾。

如果不想在`if`后空格，也可以使用`if(expr)`的形式。

`switch`语句：

```matlab
switch expr
    case {val1, val2, val3},
        statement_1
    case val4,
        statement_2
    otherwise,
        statement_3
end
```

`while`语句：

```matlab
while expr
    statements
end
```

`for`语句：

```matlab
for index = expr
    statements
end
```

`expr`会首先被 reshape 成`(d1, n)`的二维数组，然后按`expr(:, 1)`，`expr(:, 2)`，`...`的方式赋值给`index`。

```matlab
try
    % ...
catch
    % ...
end
```

**与脚本文件相关的函数**

* `beep`：蜂鸣器
* `disp(variable_name)`：只显示结果，不显示变量名
* `echo`：控制 M 脚本文件的内容是否在 command 窗口中显示
* `input`：提示用户输入数据
* `keyboard`：临时终止 M 脚本文件的执行，让键盘获得控制权。
* `pause`，`pause(n)`：暂停，等待用户按键。或暂停 n 秒后继续执行
* `waitforbuttonpress`：暂停，直到用户按下鼠标或其他按键

特殊变量：

* `ans`
* `i(j)`：虚数单位
* `pi`
* `eps`：最小浮点数精度
* `inf`
* `NaN`
* `nargin`：函数的输入变量数目
* `nargout`：函数的输出变量数目
* `realmin`：最小可用正实数
* `realmax`：最大可用正实数

将函数中的某个变量声明为全局变量：`global var1 var2;`

计时器：

```matlab
% 使用计时器延迟启动一个脚本文件
my_timer = timer('TimerFcn', 'script_name', 'StartDelay', 100)  % 延迟 100 秒
start(my_timer)  % 100 秒后执行脚本 script_name

% 设置启动函数，终止函数等
my_timer = timer('TimerFcn', 'Mfile1', ...
    'StartFcn', 'Mfile2', ...
    'StopFcn', 'Mfile3', ...
    'ErrorFcn', 'Mfile4');
start(my_timer)  % 执行 Mfile2，然后循环执行 Mfile1
stop(my_timer)  % 执行 Mfile3
```

matlab 在启动时，会默认执行`MATLABrc.m`和`startup.m`这两个文件。

输入`exit`或`quit`可退出 matlab，在退出之前，会执行`finish.m`脚本。

### 函数

可以用`error()`终止函数执行，并返回命令行窗口。也可以使用`warning()`函数打印警告信息。

在函数内部调用的 M 脚本文件，创建的变量都作为函数的内部变量。

在函数文件中可以创建子函数。可以调用`helpwin func/subfunc`查看子函数的帮助文档。

放在某个函数子目录下的函数为私有函数，只有父目录下的主函数才能调用。

查看已经被编译在内存中的函数：`inmem()`。可以用`mlock`将某个函数锁定在内存中，此时使用`clear`命令不会清除编译好的函数。可以使用`munlock()`解除锁定，使用`mislocked()`检查一个函数是否被锁定。

使用`pcode`可以将编译好后的伪码存储在硬盘中。

matlab 会在启动时对`toolbox`目录下的所有函数做一次缓存，后续不再读取这些文件。可以使用`rehash toolbox`强制刷新缓存。也可以使用`clear`清除掉旧的缓存。

`clear functions`：清除所有未锁定的函数。

`depfunc()`可以检查一个函数与其他文件之间的关联性。

检查参数：`nargchk()`，`nargoutchk()`

* `varargin`, `varargout`

* `nargin()`, `nargout()`

`persistent`变量称为永久变量。类似 C 语言中的静态变量。

`evalin()`，`eval()`。不知道这俩有啥区别。

其他有关工作区的函数：

`assignin()`, `inputname()`

当前正在被执行的函数文件名：`mfilename`

`mlint`，`mlint()`可以检查脚本文件的语法。

### 画图

直接用`plot(y)`画图，如果`y`的 shape 为`(1, d2)`，那么会画一条曲线。如果`y`的 shape 为`(d1, d2)`，那么会画`d2`条曲线，每一列作为一条曲线的数据。

不过通常都用`plot(x, y)`画吧，这样更清晰。如果`x`的 shape 为`(1, dx2)`，`y`的 shape 为`(dy1, dy2)`，那么会画`dy1`条曲线。

剩下的用法都不怎么直观。

```matlab
plot(x, y, s)
```

其中`s`可以为：

* 颜色

    `b`：蓝色，`g`：绿色，`r`：红色，`c`：青色，`m`：洋红，`y`：黄色，`k`：黑色，`w`：白色

* 标记

    `.`, `o`, `x`, `+`, `*`, `s`（方形）, `d`（菱形）, `^`

* 线型

    `-`, `:`（点线）, `-.`, `--`, `v`（向下三角形）, `p`（五角星）, `h`（六角星）, `<`（向左三角形）

```matlab
plot(x, y, 's', 'PropertyName', PropertyValue, ...)
```

常用的属性：

* `Color`：`[r, g, b]`，取值范围为`0 ~ 1`。

* `LineStyle`：4 种线型

* `LineWidth`：正实数。默认线宽为 0.5。

* `Marker`：14 种点型

* `MarkerSize`：正实数。默认大小为 6.0。

* `MarkerEdgeColor`：`[r, g, b]`

* `MarkerFaceColor`：`[r, g, b]`

常用函数：

* `axis([xmin xmax ymin ymax])`

    设置当前图形的坐标范围

* `V = axis`

    返回包含当前坐标范围的一个行向量

* `axis auto`

    将坐标轴刻度恢复为自动的默认设置

* `axis manual`

    冻结坐标轴刻度。如果`hold`被设置为`on`，那么后面的图形将使用与前面相同的坐标轴刻度范围。

* `axis tight`

    将坐标范围设置为被绘制的数据范围

* `axis fill`

    设置坐标范围和屏幕高宽比，使坐标轴可以包含整个绘制区域。

    该选项只在`PlotBoxAspectRatio`或`DataAspectRatioMode`被设置为`manual`模式时才有效。

* `axis ij`

    将坐标轴设置为矩阵模式。此时水平坐标轴从左到右取值，垂直坐标轴从上到下取值。

* `axis xy`

    将坐标轴设置为笛卡尔模式。此时水平坐标轴从左到右取值，垂直坐标轴从下到上取值。

* `axis equal`

    设置屏幕高宽比，使每个坐标轴具有均匀的刻度间隔。

* `axis image`

    设置坐标范围，使其与被显示的图形相适应

* `axis square`

    将坐标轴框设置为正方形

* `axis normal`

    将当前的坐标轴恢复为全尺寸，并将单位刻度的所有限制取消。

* `axis vis3d`

    冻结屏幕高宽比，使一个三维对象的旋转不会改变坐标轴的刻度显示

* `axis off`, `axis on`

    关闭/打开所有的坐标轴标签、刻度和背景

可以同时给出多个`axis`命令：`axis auto on xy`

图片中文本的特殊字符：

`\alpha`, `\beta`, `\gamma`, `\delta`, `\epsilon`, `\zeta`, `\eta`, `\theta`, `\vartheta`, `\iota`, `\kappa`, `\lambda`, `\mu`, `\nu`, `\xi`, `\pi`, `\rho`, `\rfloor`, `\lfloor`, `\perp`, `\wedge`, `\rceil`, `vee`, `\langle`, `\upsilon`, `\phi`, `\chi`, `\psi`, `\omega`, `\Gamma`, `\Delta`, `\Theta`, `\Lambda`, `\Xi`, `\cong`, `\approx`, `\Re`, `\oplus`, `\cup`, `\subseteq`, `\in`, `\lceil`, `\cdot`, `\neg`, `\times`, `\surd`, `\varpi`, `\rangle`, `\sim`, `\leq`, `\infty`, `\sigma`, `\varsigma`, `\tau`, `\equiv`, `\Im`, `\otimes`, `\cap`, `\supset`, `\int`, `\circ`, `\pm`, `\geq`, `\propto`, `\partial`, `\bullet`, `\div`, `\Pi`, `\Sigma`, `\Upsilon`, `\Phi`, `\Psi`, `\Omega`, `\forall`, `\exists`, `\ni`, `\neq`, `\aleph`, `\wp`, `\oslash`, `\supseteq`, `\subset`, `\o`, `\clubsuit`, `\diamondsuit`, `\heartsuit`, `\spadesuit`, `\leftrightarrow`, `\leftarrow`, `\uparrow`, `\rightarrow`, `\downarrow`, `\nabla`, `\ldots`, `\prime`, `\0`, `\mid`, `\copyright`

字体控制：

* `\bf`：黑体
* `\it`：斜体
* `\sl`：透视
* `\rm`：标准形式
* `\fontname{fontname}`
* `\fontsize{fontsize}`

标注：

```matlab
text(x, y, 'string')
text(x, y, z, 'string')
text(...'PropertyName', PropertyValue...)

gtext('string')
gtext({'string1', 'string2', 'string3', ...})
gtext({'string1'; 'string2'; 'string3'; ...})
```

图例：

```matlab
legend('string1', 'string2', ...)
legend(..., 'Location', location)
```

其中`location`可以是一个向量`[left bottom width height]`或任意一个字符串。

常用的字符串：

`'North'`, `'East'`, `'NorthEast'`, `'SouthEast'`, `'NorthOutside'`, `'EastOutside'`, `'NorthEastOutside'`, `'SouthEastOutside'`, `'South'`, `'West'`, `'NorthWest'`, `'SouthWest'`, `'SouthOutside'`, `'WestOutside'`, `'NorthWestOutside'`, `'SouthWestOutside'`, `'BestOutside'`（绘图区外占用最小面积）, `'Best'`（标注与图形的重叠最小处）

`hold on`可不刷新继续画图，`hold off`表示刷新后画图。

双纵坐标图：

```matlab
plotyy(X1, Y1, X2, Y2)
plotyy(X1, Y1, X2, Y2, 'FUN')
plotyy(X1, Y1, X2, Y2, 'FUN1', 'FUN2')
```

多子图：

```matlab
subplot(m, n, k)
```

```matlab
fplot(function, limits)
fplot(function, limits, LineSpec)
fplot(function, limits, tol)
fplot()
```