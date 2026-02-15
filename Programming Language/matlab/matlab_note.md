# Matlab Note

* matlab 允许在函数中定义一个函数，并函数这个函数的指针（句柄）

    example 1:

    ```matlab
    function [funcHandle] = createMultiReturnFunc()
        % 创建返回多个值的函数句柄
        funcHandle = @multiReturn;
        
        function [a, b, c] = multiReturn(x)
            a = x^2;
            b = x^3;
            c = sqrt(x);
        end
    end

    % 使用
    myFunc = createMultiReturnFunc();
    [val1, val2, val3] = myFunc(4);  % 16, 64, 2
    ```

    example 2:

    ```matalb
    function [funcHandle] = createMultiReturnFunc(power)
        % 可以返回不同幂次的计算函数
        funcHandle = @(x) multiReturn(x, power);
        
        function [a, b, c] = multiReturn(x, p)
            a = x^p;
            b = x^(p+1);
            c = sqrt(x);
        end
    end

    % 使用
    myFunc = createMultiReturnFunc(2);
    [val1, val2, val3] = myFunc(4);  % 16, 64, 2
    ```

    **为什么 funcHandle 定义在 function multiReturn 的前面？**

    这是一个很好的观察！让我解释一下MATLAB中函数定义的顺序问题：

    MATLAB的解析机制

    在MATLAB中，**函数定义的顺序通常不影响其可用性**，因为MATLAB会在解析整个文件时预先加载所有函数定义。这意味着：

    ```matlab
    function [funcHandle] = createMultiReturnFunc()
        % 这里先创建函数句柄，引用后面定义的函数
        funcHandle = @multiReturn;  % ✅ 没问题，multiReturn在后面定义
        
        function [a, b, c] = multiReturn(x)  % 这个函数定义在后面
            a = x^2;
            b = x^3;
            c = sqrt(x);
        end
    end
    ```

    为什么这样写是可行的？

    1. **编译时解析**：MATLAB在执行代码前会先扫描整个文件，找到所有函数定义

    2. **词法作用域**：在同一个文件内，函数名在整个作用域都是可见的

    3. **嵌套函数的特殊性**：嵌套函数在定义之前就可以被引用

    对比其他语言

    这种写法在MATLAB中是特有的，对比其他语言：

    ```python
    # Python - 必须先定义后使用
    def multi_return(x):
        return x**2, x**3, x**0.5

    def create_func():
        return multi_return  # ✅ 必须在引用前定义
    ```

    ```javascript
    // JavaScript - 函数提升让这也可以工作
    function createFunc() {
        return multiReturn;  // ✅ 函数声明会被提升
        
        function multiReturn(x) {  // 函数声明会被提升
            return [x**2, x**3, Math.sqrt(x)];
        }
    }
    ```

    代码风格考虑

    虽然MATLAB允许这种写法，但常见的最佳实践是：

    1. **主函数在前，辅助函数在后**：便于阅读代码的主要逻辑

    2. **先定义后使用**：虽然技术上不必要，但更符合直觉

    3. **保持一致性**：在项目中统一风格

        ```matlab
        function [funcHandle] = createMultiReturnFunc()
            % 定义内部函数
            function [a, b, c] = multiReturn(x)
                a = x^2;
                b = x^3;
                c = sqrt(x);
            end
            
            % 创建函数句柄（先定义后使用）
            funcHandle = @multiReturn;  % 这样写更符合阅读习惯
        end
        ```

    所以，你观察到的代码顺序在MATLAB中是完全可以工作的，这是由MATLAB的语言特性决定的！

* matlab `deal()`

    deal() 函数详解

    1. 函数原型

        ```matlab
        [Y1, Y2, ..., Yn] = deal(X)
        [Y1, Y2, ..., Yn] = deal(X1, X2, ..., Xn)
        ```

    2. 基本作用

        deal() 的主要作用是分配输入到多个输出，特别是在需要同时给多个变量赋值的场景。

    3. 两种主要用法

        用法1：复制相同的值到多个输出

        ```matlab
        % 所有输出变量获得相同的值
        [a, b, c] = deal(10);
        % 结果：a = 10, b = 10, c = 10

        % 在结构体或元胞数组中很有用
        S = struct('name', {'Alice', 'Bob'}, 'age', {25, 30});
        [name1, name2] = deal(S.name);
        % name1 = 'Alice', name2 = 'Bob'
        ```

        用法2：分配不同的值到对应的输出

        ```matlab
        % 输入输出一一对应
        [x, y, z] = deal(1, 2, 3);
        % 结果：x = 1, y = 2, z = 3

        % 相当于：
        % x = 1;
        % y = 2;
        % z = 3;
        ```

    4. 在匿名函数中的特殊用途

        在匿名函数中，deal() 是实现多输出匿名函数的唯一方式：

        ```matlab
        % 普通匿名函数只能返回一个值
        singleOut = @(x) x^2;  % 只能返回一个值

        % 使用 deal() 实现多输出
        multiOut = @(x) deal(x^2, x^3, sqrt(x));
        [a, b, c] = multiOut(4);
        % a = 16, b = 64, c = 2
        ```

    5. 实际应用示例

        示例1：交换变量值

        ```matlab
        a = 1; b = 2;
        [a, b] = deal(b, a);  % 交换 a 和 b
        % 结果：a = 2, b = 1
        ```

        示例2：结构体字段批量提取

        ```matlab
        students(1) = struct('name', 'Alice', 'score', 85);
        students(2) = struct('name', 'Bob', 'score', 92);

        % 提取所有学生的分数
        [score1, score2] = deal(students.score);
        % score1 = 85, score2 = 92
        ```

        示例3：处理函数输出

        ```matlab
        % 某些函数返回多输出，但中间需要处理
        [minVal, maxVal] = minmax([1 5 3 8 2]);

        % 如果需要存储所有极值信息
        [val, idx] = min([1 5 3 8 2]);
        [minInfo, maxInfo] = deal([val, idx], max([1 5 3 8 2]));
        ```

        示例4：元胞数组操作

        ```matlab
        C = {'apple', 'banana', 'cherry'};
        [first, second, third] = deal(C{:});
        % first = 'apple', second = 'banana', third = 'cherry'
        ```

    6. 注意事项

        * 输入输出数量必须匹配（用法2时）：

            ```matlab
            % 错误：输入输出数量不匹配
            [a, b, c] = deal(1, 2);  % 报错
            ```

        * 在匿名函数中是必要的：

            * 没有 deal()，匿名函数只能返回一个值

        * 对于简单赋值，直接赋值更高效：

            ```matlab
            % 使用 deal()
            [a, b, c] = deal(1, 2, 3);

            % 更高效的直接赋值
            a = 1; b = 2; c = 3;
            ```

    7. 替代方案对比

        ```matlab
        % 方法1：使用 deal()
        [x, y, z] = deal(a, b, c);

        % 方法2：直接赋值（更推荐，性能更好）
        x = a;
        y = b;
        z = c;

        % 方法3：使用元胞扩展（特定场景）
        results = num2cell([a, b, c]);
        [x, y, z] = results{:};
        ```

    总结

    deal() 主要在两个场景中特别有用：

    * 匿名函数中实现多输出

    * 批量处理结构体或元胞数组数据

    在其他大多数情况下，直接赋值是更简单高效的选择。

    **简述 matlab 中 getTwoValues = @(x) deal(x^2, x^3); 的含义**

    在 MATLAB 中，getTwoValues = @(x) deal(x^2, x^3); 是一个匿名函数的定义，具体含义如下：

    1. 整体含义

        创建一个名为 getTwoValues 的函数句柄，该函数：

        * 接受一个输入参数 x

        * 同时返回两个输出：x² 和 x³

    2. 关键元素解析

        @(x)

        * 定义匿名函数的语法

        * x 是输入参数

        deal(x^2, x^3)

        * deal() 是 MATLAB 函数，用于将多个输出分配给不同的变量

        * 这里指定函数返回两个值：x^2 和 x^3

    3. 使用示例

        ```matlab
        % 调用函数并接收两个输出
        [a, b] = getTwoValues(3)

        % 结果：
        % a = 9  (3²)
        % b = 27 (3³)

        % 如果只接收一个输出，会得到第一个值
        c = getTwoValues(2)  % c = 4

        % 调用示例
        [x2, x3] = getTwoValues(5);  % x2=25, x3=125
        ```

    4. 相当于传统函数定义

        ```matlab
        function [y1, y2] = getTwoValues(x)
            y1 = x^2;
            y2 = x^3;
        end
        ```

        但匿名函数更简洁，适合简单的单行函数定义。

    5. 应用场景

        * 需要快速定义简单函数

        * 作为参数传递给其他函数（如 fplot、arrayfun）

        * 临时计算，无需创建单独的 .m 文件

    这种语法在 MATLAB 中常用于创建简洁的多输出函数。

* 简述 matlab 中 shading() 的原型、作用及用法

    shading() 是 MATLAB 中用于控制曲面和补片图形颜色着色的函数。

    原型

    ```matlab
    shading(style)
    ```

    主要作用

    控制曲面 (surf, mesh 等) 和补片 (patch) 图形的着色方式，即如何用颜色填充图形表面。

    三种着色样式

    1. shading flat

        * 特点：每个网格面片使用单一均匀颜色

        * 颜色依据：使用面片最后一个顶点的颜色值

        * 效果：面片间有明显边界，没有平滑过渡

    2. shading faceted（默认）

        * 特点：flat 的基础上添加黑色网格线

        * 效果：清晰显示网格结构，最常用

    3. shading interp

        * 特点：在网格点之间进行颜色插值

        * 效果：颜色平滑过渡，外观最光滑

    用法示例

    ```matlab
    % 创建示例曲面
    [X,Y,Z] = peaks(30);

    % 绘制曲面
    subplot(1,3,1)
    surf(X,Y,Z)
    shading flat
    title('flat 着色')

    subplot(1,3,2)
    surf(X,Y,Z)
    shading faceted
    title('faceted 着色（默认）')

    subplot(1,3,3)
    surf(X,Y,Z)
    shading interp
    title('interp 着色')
    ```

    高级用法

    ```matlab
    % 结合光照效果
    figure
    surf(peaks)
    shading interp
    light
    lighting gouraud  % 配合 shading interp 效果更好

    % 对不同子图应用不同着色
    figure
    ax1 = subplot(1,2,1);
    surf(peaks(20))
    shading(ax1, 'flat')

    ax2 = subplot(1,2,2);
    surf(peaks(20))
    shading(ax2, 'interp')
    ```

    注意事项

    * 生效对象：只影响当前坐标区 (gca)

    * 与 colormap 关系：配合使用，colormap 定义颜色映射，shading 控制着色方式

    * 性能考虑：interp 计算量最大，flat 和 faceted 较轻量

    * 透明度支持：如果设置了 FaceAlpha，所有着色方式都支持透明度

    典型应用场景

    * 科学可视化：faceted 清晰显示数据网格

    * 渲染效果：interp 用于平滑表面渲染

    * 快速预览：flat 用于快速数据检查

    shading 是 MATLAB 三维可视化中调整图形外观的重要工具，可根据需要选择不同平衡（清晰度 vs 平滑度 vs 性能）。

* matlab spy() 函数

    原型

    ```matlab
    spy(S)
    spy(S, markersize)
    spy(S, LineSpec)
    spy(S, LineSpec, markersize)
    spy(ax, ___)
    ```

    作用

    可视化稀疏矩阵的非零元素模式，常用于分析矩阵结构、填充模式、连接性等。

    用法示例

    ```matlab
    % 创建稀疏矩阵
    S = bucky;  % Bucky 球邻接矩阵

    % 基本用法
    figure
    subplot(2,2,1)
    spy(S)
    title('默认显示')

    % 控制标记大小
    subplot(2,2,2)
    spy(S, 10)  % 标记大小为10
    title('标记大小=10')

    % 使用线条规范
    subplot(2,2,3)
    spy(S, 'r*', 8)  % 红色星号，大小8
    title('红色星号标记')

    % 获取额外信息
    subplot(2,2,4)
    spy(S)
    [nz, rows, cols] = find(S);
    title(sprintf('非零元素数: %d', nnz(S)))

    % 分析矩阵结构
    A = delsq(numgrid('S', 30));  % 有限差分矩阵
    figure
    spy(A)
    title('有限差分矩阵结构')
    ```

    典型应用

    * 矩阵结构分析

        ```matlab
        % 带状矩阵分析
        B = diag(ones(50,1)) + diag(ones(49,1),1) + diag(ones(49,1),-1);
        spy(B)
        title('三对角矩阵')
        ```

    * 稀疏模式比较

    ```matlab
    % 比较不同填充策略
    load west0479
    figure
    subplot(1,3,1)
    spy(west0479)
    title('原始矩阵')

    subplot(1,3,2)
    spy(chol(west0479))
    title('Cholesky分解后')

    subplot(1,3,3)
    [p,~,~] = colamd(west0479);
    spy(west0479(p,p))
    title('列重排序后')
    ```

* histogram() 函数

    原型

    ```matlab
    histogram(X)
    histogram(X, nbins)
    histogram(X, edges)
    histogram(ax, ___)
    h = histogram(___)
    ```

    作用

    创建数据的直方图，用于可视化数据分布、频率统计。

    主要属性控制

    ```matlab
    % 基本用法示例
    figure

    % 1. 默认直方图
    subplot(2,3,1)
    data = randn(1000,1);
    histogram(data)
    title('默认直方图')

    % 2. 指定箱数
    subplot(2,3,2)
    histogram(data, 30)  % 30个箱子
    title('30个箱子')

    % 3. 指定边界
    subplot(2,3,3)
    edges = -4:0.5:4;
    histogram(data, edges)
    title('指定边界')

    % 4. 归一化
    subplot(2,3,4)
    histogram(data, 'Normalization', 'pdf')
    title('概率密度')

    % 5. 显示样式
    subplot(2,3,5)
    histogram(data, 'FaceColor', 'g', 'EdgeColor', 'k', 'FaceAlpha', 0.7)
    title('自定义颜色')

    % 6. 水平直方图
    subplot(2,3,6)
    histogram(data, 'Orientation', 'horizontal')
    title('水平直方图')
    ```

    高级用法

    ```matlab
    % 1. 多个数据集比较
    figure
    data1 = randn(1000,1);
    data2 = randn(1000,1) * 0.8 + 1;
    histogram(data1, 'FaceAlpha', 0.5, 'Normalization', 'probability')
    hold on
    histogram(data2, 'FaceAlpha', 0.5, 'Normalization', 'probability')
    legend('数据集1', '数据集2')
    title('分布比较')

    % 2. 获取统计信息
    h = histogram(data);
    counts = h.Values;      % 每个箱的计数
    edges = h.BinEdges;     % 箱边界
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

    % 3. 累积分布
    figure
    histogram(data, 'Normalization', 'cdf')
    title('累积分布函数')

    % 4. 分组数据
    figure
    load carsmall
    histogram(MPG, 'BinMethod', 'auto')
    xlabel('MPG')
    ylabel('频数')
    title('汽车MPG分布')

    % 5. 箱宽控制
    figure
    subplot(1,2,1)
    histogram(data, 'BinWidth', 0.2)
    title('箱宽=0.2')

    subplot(1,2,2)
    histogram(data, 'BinLimits', [-2, 2])
    title('限制范围[-2,2]')
    ```

    属性选项总结

    | 属性 | 说明 | 示例值 |
    | - | - | - |
    | NumBins | 箱数 | 20 |
    | BinWidth | 箱宽度 | 0.5 |
    | BinEdges | 箱边界 | [-3:0.5:3] |
    | Normalization | 归一化方式 | 'count', 'probability', 'pdf', 'cdf' |
    | FaceColor | 填充颜色 | 'r', [0.5 0.5 0.5] |
    | EdgeColor | 边界颜色 | 'k', 'none' |
    | FaceAlpha | 透明度 | 0.5 |
    | Orientation | 方向 | 'vertical', 'horizontal' |
    | DisplayStyle | 显示样式 | 'bar', 'stairs' |

    典型应用场景

    * 数据分布分析：查看数据是否符合正态分布等

    * 异常值检测：识别分布边缘的异常点

    * 数据比较：多个数据集分布对比

    * 质量控制：监控过程数据的稳定性

    * 图像处理：图像灰度直方图分析

* camlight() 函数

    原型

    ```matlab
    camlight
    camlight(az, el)
    camlight(light_handle, ...)
    camlight(..., style)
    light_handle = camlight(...)
    ```

    作用

    创建与相机位置相关联的光源，使三维图形在不同视角下都有良好的照明效果。

    用法示例

    ```matlab
    % 创建三维曲面
    [x,y,z] = sphere(50);
    figure
    surf(x,y,z, 'FaceAlpha', 0.8)
    axis equal

    % 1. 默认相机灯
    subplot(2,2,1)
    surf(x,y,z)
    camlight  % 默认：右上方45度
    title('默认 camlight')

    % 2. 指定方位角和高程角
    subplot(2,2,2)
    surf(x,y,z)
    camlight(30, 60)  % az=30°, el=60°
    title('az=30°, el=60°')

    % 3. 不同样式
    subplot(2,2,3)
    surf(x,y,z)
    camlight('left')  % 左侧光
    title('左侧光')

    subplot(2,2,4)
    surf(x,y,z)
    camlight('right', 'infinite')  % 右侧无限远光
    title('右侧无限远光')
    ```

    光源样式

    ```matlab
    figure
    % local 样式：点光源，随距离衰减
    subplot(1,2,1)
    surf(peaks)
    camlight(45, 30, 'local')
    lighting gouraud
    title('local 点光源')

    % infinite 样式：平行光，无衰减
    subplot(1,2,2)
    surf(peaks)
    camlight(45, 30, 'infinite')
    lighting gouraud
    title('infinite 平行光')
    ```

    高级用法

    ```matlab
    % 动态更新光源
    figure
    surf(peaks)
    camlight('headlight')  % 头灯光
    title('使用方向键旋转，光源跟随相机')

    % 多个光源
    figure
    [x,y,z] = sphere;
    surf(x,y,z, 'FaceColor', 'y', 'EdgeColor', 'none')
    camlight('left')
    camlight('right')
    lighting gouraud
    material shiny
    ```

* set() 函数

    原型

    ```matlab
    set(H, Name, Value)
    set(H, NameArray, ValueArray)
    set(H, S)
    set(H)
    A = set(H)
    ```

    作用

    设置图形对象属性，是MATLAB图形系统的核心函数之一。

    用法示例

    ```matlab
    % 创建图形
    x = 0:0.1:2*pi;
    y = sin(x);

    figure
    h_plot = plot(x, y);

    % 1. 设置单个属性
    set(h_plot, 'LineWidth', 2)

    % 2. 设置多个属性
    set(h_plot, 'Color', 'r', 'LineStyle', '--', 'Marker', 'o')

    % 3. 使用结构体设置
    props.LineWidth = 3;
    props.MarkerSize = 8;
    props.MarkerFaceColor = 'g';
    set(h_plot, props)

    % 4. 获取可用属性列表
    prop_list = set(h_plot);

    % 5. 设置坐标轴属性
    set(gca, 'FontSize', 12, 'GridLineStyle', '-')
    ```

    常见图形对象属性设置

    ```matlab
    figure

    % 设置图形窗口
    set(gcf, 'Name', '我的图形', 'NumberTitle', 'off', ...
             'Position', [100 100 800 600], 'Color', [0.9 0.9 0.9])

    % 创建子图
    subplot(2,2,1)
    h1 = plot(rand(10,1));
    set(h1, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 8)

    % 设置坐标轴
    subplot(2,2,2)
    plot(rand(10,1))
    set(gca, 'XGrid', 'on', 'YGrid', 'on', ...
             'FontSize', 10, 'Box', 'on')

    % 设置文本
    subplot(2,2,3)
    plot(rand(10,1))
    h_title = title('随机数据');
    set(h_title, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red')

    % 设置图例
    subplot(2,2,4)
    h2 = plot(rand(10,2));
    h_legend = legend('数据1', '数据2');
    set(h_legend, 'Location', 'best', 'FontSize', 10)
    ```

    批量设置技巧

    ```matlab
    % 批量设置多个对象
    figure
    hold on
    h_lines = zeros(5,1);
    colors = {'r', 'g', 'b', 'm', 'c'};

    for i = 1:5
        h_lines(i) = plot(rand(10,1) + i);
    end

    % 批量设置线条属性
    set(h_lines, {'Color'}, colors', 'LineWidth', 2)

    % 设置不同标记
    markers = {'o', '+', '*', '.', 'x'};
    for i = 1:5
        set(h_lines(i), 'Marker', markers{i})
    end
    ```

* view() 函数

    原型

    ```matlab
    view(az, el)
    view([az, el])
    view([x, y, z])
    view(2)
    view(3)
    view(ax, ...)
    [az, el] = view
    ```

    作用

    设置三维图形的观察视角（相机位置）。

    视角参数解释

    * az：方位角，绕z轴旋转的角度（度）

        * 0° = 从负y轴方向看

        * 90° = 从正x轴方向看

    * el：高程角，相对于xy平面的仰角（度）

        * 0° = 在xy平面

        * 90° = 正上方垂直看

    用法示例

    ```matlab
    % 创建测试曲面
    [x,y,z] = peaks(30);

    figure

    % 1. 标准三维视图
    subplot(2,3,1)
    surf(x,y,z)
    view(3)  % 默认三维视图 az=-37.5, el=30
    title('view(3) 默认三维')

    % 2. 二维视图
    subplot(2,3,2)
    surf(x,y,z)
    view(2)  % 二维俯视图
    title('view(2) 俯视')

    % 3. 指定角度
    subplot(2,3,3)
    surf(x,y,z)
    view(45, 60)  % az=45°, el=60°
    title('view(45,60)')

    % 4. 使用向量
    subplot(2,3,4)
    surf(x,y,z)
    view([-10, -20, 5])  % [x, y, z] 方向向量
    title('view([-10,-20,5])')

    % 5. 不同视角比较
    subplot(2,3,5)
    surf(x,y,z)
    view(0, 90)  % 正上方
    title('正上方 view(0,90)')

    subplot(2,3,6)
    surf(x,y,z)
    view(0, 0)  % 侧面
    title('侧面 view(0,0)')
    ```

    视角预设

    ```matlab
    figure
    surf(peaks)

    % 常用预设视角
    view_angles = {
        '默认3D', [-37.5, 30];
        '正上方', [0, 90];
        '正前方', [0, 0];
        '左前方', [-45, 30];
        '右前方', [45, 30];
        '底部', [-37.5, -90]
    };

    for i = 1:6
        subplot(2,3,i)
        surf(peaks)
        view(view_angles{i,2})
        title(view_angles{i,1})
    end
    ```

    高级用法与动画

    ```matlab
    % 1. 旋转动画
    figure
    surf(peaks)
    colormap(jet)
    shading interp

    for az = 0:2:360
        view(az, 30)
        pause(0.01)
    end

    % 2. 获取当前视角
    [az, el] = view;
    fprintf('当前视角: az=%.1f°, el=%.1f°\n', az, el)

    % 3. 交互式视角设置
    figure
    surf(peaks)
    rotate3d on  % 启用交互旋转

    % 4. 视角与光照配合
    figure
    surf(peaks)
    camlight  % 创建相机灯
    view(45, 30)  % 设置视角
    lighting gouraud

    % 5. 多子图统一视角
    figure
    for i = 1:4
        subplot(2,2,i)
        surf(peaks + i)
        view(30, 45)  % 统一视角
        title(sprintf('曲面 %d', i))
    end
    ```

* 简述 matlab inputParser 的用法

    inputParser 是 MATLAB 中用于解析和验证函数输入参数的强大工具，特别适用于处理可变数量参数、参数-值对（名称-值对）等复杂输入情况。

    基本使用步骤

    1. 创建 inputParser 对象

        ```matlab
        p = inputParser;
        ```

    2. 设置参数属性

        ```matlab
        p.FunctionName = 'myFunction';  % 可选，用于错误信息
        p.KeepUnmatched = true;         % 是否允许未定义的参数
        p.PartialMatching = false;      % 是否允许参数名部分匹配
        ```

    3. 添加参数

        有三种类型的参数：

        必需参数

        ```matlab
        addRequired(p, 'paramName', validationFunc);
        ```

        * 必须提供，按位置传递

        * 验证函数示例：@isnumeric, @ischar, 自定义函数

        可选参数

        ```matlab
        addOptional(p, 'paramName', defaultValue, validationFunc);
        ```

        * 可以不提供，使用默认值

        * 必须位于必需参数之后

        参数-值对

        ```matlab
        addParameter(p, 'paramName', defaultValue, validationFunc);
        ```

        * 使用名称-值对形式：'Name', value

        * 可以任意顺序

    4. 解析参数

        ```matlab
        parse(p, varargin{:});
        ```

    5. 获取解析结果

        ```matlab
        results = p.Results;
        paramValue = p.Results.paramName;
        ```

    完整示例

    ```matlab
    function result = myFunction(a, b, varargin)
        % 创建解析器
        p = inputParser;
        
        % 添加必需参数
        addRequired(p, 'a', @isnumeric);
        addRequired(p, 'b', @(x) x > 0);  % b 必须为正数
        
        % 添加可选参数
        addOptional(p, 'c', 10, @isnumeric);  % 默认值 10
        
        % 添加参数-值对
        addParameter(p, 'Scale', 1.0, @isnumeric);
        addParameter(p, 'Method', 'linear', ...
            @(x) any(validatestring(x, {'linear', 'nearest', 'spline'})));
        addParameter(p, 'Display', false, @islogical);
        
        % 解析输入
        parse(p, a, b, varargin{:});
        
        % 使用解析结果
        params = p.Results;
        disp(params);
        
        % 访问具体参数
        result = params.a * params.b * params.c * params.Scale;
        
        if params.Display
            fprintf('Result: %f\n', result);
        end
    end
    ```

    使用示例

    ```matlab
    % 调用函数
    myFunction(5, 2)  % 使用默认值
    myFunction(5, 2, 3)  % 指定可选参数 c=3
    myFunction(5, 2, 'Scale', 2.0, 'Method', 'spline', 'Display', true)
    ```

    处理未匹配参数

    ```matlab
    p.KeepUnmatched = true;
    parse(p, varargin{:});

    % 获取未匹配的参数
    unmatchedParams = p.Unmatched;
    % unmatchedParams 是一个结构体，包含未定义的参数名值对
    ```

    验证函数示例

    ```matlab
    % 内置验证函数
    @isnumeric      % 必须是数值
    @ischar         % 必须是字符
    @islogical      % 必须是逻辑值
    @isvector       % 必须是向量
    @(x) x > 0      % 必须大于0
    @(x) ismember(x, {'opt1', 'opt2'})  % 必须在集合中

    % 自定义验证函数
    validateFcn = @(x) validateattributes(x, {'numeric'}, ...
        {'scalar', 'positive', 'integer'});
    ```

    优点

    * 参数管理：清晰地区分必需、可选和名称-值参数

    * 验证：自动验证输入类型和范围

    * 默认值：为可选参数提供默认值

    * 错误处理：统一的错误信息

    * 灵活性：支持多种参数传递方式

    替代方案

    * 对于简单函数：直接使用 nargin, varargin

    * MATLAB R2019b+：考虑使用参数块（Arguments Block）

        ```matlab
        function result = myFunction(a, b, options)
            arguments
                a double
                b (1,1) double {mustBePositive}
                options.Scale (1,1) double = 1.0
                options.Method string {mustBeMember(options.Method, ["linear","nearest"])} = "linear"
            end
            % 函数体
        end
        ```

    inputParser 特别适合需要向后兼容旧版本 MATLAB 的代码，或者需要精细控制参数解析逻辑的情况。

    inputParser 是一个类（class），通过创建该类的实例（对象）来使用。

* 简述 matlab strcmpi() 的原型、作用与用法

    strcmpi() 是 MATLAB 中用于字符串比较（不区分大小写） 的函数。

    原型

    ```matlab
    TF = strcmpi(str1, str2)
    ```

    参数说明

    | 参数 | 说明 |
    | - | - |
    | str1, str2 | 要比较的字符串、字符向量、字符向量元胞数组或字符串数组 |
    | TF | 逻辑数组，比较结果为真时对应元素为 true，否则为 false |

    作用

    * 不区分大小写比较两个字符串是否完全相同

    * 比较时忽略大小写差异：'Hello'、'HELLO'、'hello' 被视为相同

    * 返回逻辑值（真/假）指示是否匹配

    基本用法

    1. 比较两个字符串

        ```matlab
        strcmpi('Hello', 'HELLO')    % 返回 true
        strcmpi('MATLAB', 'matlab')  % 返回 true
        strcmpi('text', 'test')      % 返回 false
        ```

    2. 比较字符串与字符向量元胞数组

        ```matlab
        % 单个字符串与元胞数组比较
        C = {'apple', 'Banana', 'ORANGE', 'Grape'};
        TF = strcmpi('banana', C)  % 返回 [false, true, false, false]

        % 查找匹配项
        idx = find(strcmpi('orange', C))  % 返回 3
        ```

    3. 比较两个元胞数组

        ```matlab
        C1 = {'John', 'Mary', 'Peter'};
        C2 = {'JOHN', 'mary', 'PAUL'};
        TF = strcmpi(C1, C2)  % 返回 [true, true, false]
        ```

    4. 比较字符串数组（R2016b+）

        ```matlab
        % 字符串数组
        str1 = ["Apple", "Banana", "Cherry"];
        str2 = ["APPLE", "BANANA", "Date"];
        TF = strcmpi(str1, str2)  % 返回 [true, true, false]
        ```

    高级用法

    1. 查找列表中的特定字符串

        ```matlab
        files = {'readme.txt', 'README.TXT', 'data.csv', 'DATA.CSV'};
        isReadme = strcmpi('readme.txt', files);  % 不区分大小写查找
        ```

    2. 验证用户输入

        ```matlab
        userInput = 'YES';
        validOptions = {'yes', 'no', 'maybe'};
        if any(strcmpi(userInput, validOptions))
            disp('Valid input');
        else
            disp('Invalid input');
        end
        ```

    3. 处理混合大小写数据

        ```matlab
        data = {'ProductA', 'PRODUCTB', 'productC', 'ProductD'};
        productA_indices = strcmpi('producta', data);  % 找到所有变体
        ```

    与相关函数的比较

    | 函数 | 区分大小写 | 功能 | 返回类型 |
    | - | - | - | - |
    | strcmpi() | 否 | 字符串相等比较 | 逻辑值 |
    | strcmp() | 是 | 字符串相等比较 | 逻辑值 |
    | strncmpi() | 否 | 比较前 N 个字符 | 逻辑值 |
    | strncmp() | 是 | 比较前 N 个字符 | 逻辑值 |
    | contains() | 可选 | 检查是否包含子串 | 逻辑值 |
    | strfind() | 是 | 查找子串位置 | 位置索引 |

    ```matlab
    % 比较示例
    strcmp('Hello', 'HELLO')   % false（区分大小写）
    strcmpi('Hello', 'HELLO')  % true（不区分大小写）

    strncmp('Hello', 'HELLOworld', 5)   % false
    strncmpi('Hello', 'HELLOworld', 5)  % true
    ```

    实际应用示例

    示例 1：处理文件类型

    ```matlab
    files = {'image.JPG', 'document.PDF', 'photo.jpg', 'data.TXT'};

    % 查找所有 JPG 文件（不区分大小写）
    isJpg = strcmpi('.jpg', cellfun(@(x) x(end-3:end), files, 'UniformOutput', false));
    % 或者更简单的方式
    isJpg = endsWith(files, '.jpg', 'IgnoreCase', true);  % R2016b+
    ```

    示例 2：命令解析器

    ```matlab
    function processCommand(cmd)
        switch lower(cmd)  % 转换为小写再比较
            case 'start'
                disp('Starting...');
            case 'stop'
                disp('Stopping...');
            case 'pause'
                disp('Pausing...');
            otherwise
                % 或者使用 strcmpi
                if strcmpi(cmd, 'help')
                    disp('Help information');
                else
                    disp('Unknown command');
                end
        end
    end
    ```

    示例 3：数据库查询处理

    ```matlab
    names = {'John Smith', 'jane doe', 'JOHN SMITH', 'Bob Johnson'};
    query = 'john smith';

    % 不区分大小写查找所有匹配项
    matches = strcmpi(query, names);
    matchedNames = names(matches);  % 返回 {'John Smith', 'JOHN SMITH'}
    ```

    注意事项

    * 输入类型必须兼容：两个输入必须是相同类型（都是字符串、都是字符向量、都是元胞数组等）

    * 元胞数组大小需匹配：当比较两个元胞数组时，它们必须具有相同的大小

    * 空值处理：

        ```matlab
        strcmpi('', '')        % true
        strcmpi('text', '')    % false
        strcmpi([], [])        % 错误，不支持空矩阵
        ```

    * 性能考虑：对于大型字符串比较，strcmpi 通常比先转换为小写再比较更高效

        ```matlab
        % 不推荐（低效）：
        TF = strcmp(lower(str1), lower(str2));

        % 推荐（高效）：
        TF = strcmpi(str1, str2);
        ```

    strcmpi() 在处理用户输入、文件操作、数据匹配等需要忽略大小写的场景中非常实用。

* 简述 matlab 中 path 命令的作用

    path 命令在 MATLAB 中用于显示或设置 MATLAB 的搜索路径。搜索路径是 MATLAB 查找和执行文件（如脚本、函数、类等）的一系列目录列表。

    主要功能

    1. 查看当前搜索路径

        ```matlab
        path
        % 或
        path()
        ```

        显示 MATLAB 当前的所有搜索路径，按优先级顺序排列。

    2. 设置新路径（替换当前路径）

        ```matlab
        path(newPath)
        ```

        用 newPath（字符串或字符向量）完全替换当前搜索路径。

    3. 添加路径

        更常用的操作是使用 addpath 和 rmpath 来动态修改路径，但 path 也可用于组合路径：

        ```matlab
        path(path, 'newDirectory')        % 将 newDirectory 添加到路径末尾
        path('newDirectory', path)        % 将 newDirectory 添加到路径开头（优先级更高）
        ```

    示例

    ```matlab
    % 查看当前路径
    path

    % 添加新目录到路径开头（优先搜索）
    path('/home/user/myfunctions', path)

    % 设置全新的路径列表
    newPath = '/home/user/libs:/home/user/projects:/matlab/toolbox';
    path(newPath)
    ```

    相关常用命令

    * addpath：向路径添加目录（更常用）

    * rmpath：从路径移除目录

    * savepath：保存当前路径设置到 pathdef.m，使下次启动 MATLAB 时生效

    * pathtool：打开图形界面管理路径

    * which：查看函数文件所在路径（帮助确定搜索顺序）

    注意事项

    * 路径优先级：MATLAB 按路径顺序从上到下搜索，找到第一个匹配文件即停止。

    * 修改建议：通常使用 addpath 和 pathtool 管理路径更为方便和安全。

    * 持久化：直接使用 path 命令修改的路径在退出 MATLAB 后会丢失，需配合 savepath 保存。

    通过合理设置路径，可以确保 MATLAB 找到自定义函数或第三方工具箱，并管理不同项目的文件组织。

* 简述 savepath 的作用

    savepath 的作用是将当前 MATLAB 的搜索路径设置保存为永久性配置，使这些路径设置在未来的 MATLAB 会话中自动生效。

    核心功能：

    将 内存中的临时路径设置 → 保存为 永久的启动路径配置

    详细作用：

    1. 永久化路径更改

        ```matlab
        % 临时添加路径（仅当前会话有效）
        addpath('/path/to/my/functions');

        % 保存路径设置（永久生效）
        savepath;
        % 下次启动 MATLAB 时，该路径仍然存在
        ```

    2. 更新 pathdef.m 文件

        * 默认保存到：matlabroot/toolbox/local/pathdef.m

        * 此文件在 MATLAB 启动时自动加载

        * 定义了 MATLAB 的默认搜索路径顺序

    典型工作流程：

    ```matlab
    % 1. 添加新路径（临时）
    addpath('/project/code');

    % 2. 测试功能是否正常
    my_function();  % 测试调用

    % 3. 确认无误后保存
    savepath;

    % 4. 验证保存成功
    which pathdef  % 查看路径定义文件位置
    ```

    主要应用场景：

    场景1：安装第三方工具箱

    ```matlab
    % 添加工具箱路径
    addpath(genpath('/toolboxes/image_toolbox'));

    % 测试工具箱功能
    imtool test_image.jpg;

    % 确认可用后永久保存
    savepath;
    ```

    场景2：开发自定义函数库

    ```matlab
    % 添加个人代码库
    addpath('/Users/name/matlab_lib');
    addpath('/Users/name/matlab_lib/utils');
    addpath('/Users/name/matlab_lib/plotting');

    % 保存配置
    savepath;
    ```

    场景3：解决权限问题

    ```matlab
    % 如果默认保存位置无写入权限
    savepath('/alternative/path/pathdef.m');
    % 然后需要设置 MATLAB 启动选项来加载此文件
    ```

    与相关命令的关系：

    添加路径 → 保存路径

    ```matlab
    addpath('new/path');    % 临时添加
    rmpath('old/path');     % 临时移除
    savepath;               % 永久保存当前状态
    ```

    恢复默认 → 重新配置

    ```matlab
    restoredefaultpath;     % 清除所有自定义路径
    addpath('essential/path');  % 重新添加必要路径
    savepath;               % 保存精简后的配置
    ```

    重要注意事项：

    ⚠️ 保存位置问题

    * 需要写入 pathdef.m 文件的权限

    * 在多用户系统中可能遇到权限错误

    * 解决方案：

        ```matlab
        % 保存到用户有权限的位置
        savepath('~/my_pathdef.m');
        ```

    ⚠️ 路径冲突风险

    * 保存前确保路径顺序正确

    * 错误的路径顺序可能导致函数调用错误版本

    * 建议检查：

        ```matlab
        path  % 查看完整路径列表
        which function_name -all  % 查看函数所有版本
        ```

    ✅ 最佳实践

    ```matlab
    % 1. 批量添加路径后统一保存
    addpath(genpath('lib1'));
    addpath(genpath('lib2'));
    % ... 测试功能 ...
    savepath;  % 一次性保存

    % 2. 定期清理路径
    pathtool;  % 使用图形界面管理路径
    savepath;  % 保存清理后的结果

    % 3. 备份原配置
    copyfile(which('pathdef.m'), 'pathdef_backup.m');
    ```

    相关文件和命令：

    | 项目 | 说明 |
    | - | - |
    | pathdef.m | 路径定义文件，savepath 的保存目标 |
    | pathtool | 图形化路径管理工具，可替代命令操作 |
    | userpath | 用户个人路径，默认自动添加 |
    | startup.m | 启动脚本，可包含 addpath 命令 |

    总结：

    savepath 是 MATLAB 路径管理的"保存按钮"：

    * 作用：将当前会话的路径设置永久化

    * 时机：在确认所有路径配置正确后使用

    * 位置：默认保存到 pathdef.m 文件

    * 重要性：避免每次启动 MATLAB 都重新配置路径

    记住：addpath 是临时修改，savepath 是永久保存。两者配合使用，才能高效管理 MATLAB 的工作环境。

* `restoredefaultpath`

    restoredefaultpath 的作用是将 MATLAB 的搜索路径恢复为出厂默认设置，这是解决 MATLAB 路径相关问题的“终极”手段。

    主要作用：

    * 清除所有自定义路径

        * 移除所有用户手动添加的路径

        * 删除所有第三方工具箱的路径

        * 只保留 MATLAB 安装时自带的工具箱路径

    * 修复路径冲突和损坏

        * 解决因路径顺序错误导致的函数调用问题

        * 修复因路径损坏导致的 MATLAB 启动或运行错误

    典型应用场景：

    1. 路径冲突问题

        ```matlab
        % 当出现以下情况时使用：
        % - 调用函数时提示错误版本
        % - 同名函数冲突（用户函数覆盖了系统函数）
        % - MATLAB 行为异常
        restoredefaultpath
        savepath  % 保存默认路径
        ```

    2. MATLAB 启动失败

        ```matlab
        % 如果MATLAB启动时崩溃或卡住
        % 可以在启动时按住Shift键，进入安全模式
        % 然后在命令窗口执行：
        restoredefaultpath
        savepath
        ```

    3. 工具箱加载问题

        ```matlab
        % 第三方工具箱导致的问题
        restoredefaultpath  % 先恢复默认
        % 然后重新逐一手动添加必要的工具箱路径
        addpath('/path/to/your/toolbox')
        savepath
        ```

    工作流程：

    执行前：

    ```text
    MATLAB路径 = 默认路径 + 用户自定义路径 + 第三方工具箱路径
    ```

    执行后：

    ```text
    MATLAB路径 = 仅默认路径
    （所有用户添加的内容都被移除）
    ```

    重要注意事项：

    ⚠️ 后果严重

    * 所有自定义路径设置都会丢失

    * 需要重新配置所有第三方工具箱

    * 用户编写的函数如果不在默认路径下将无法直接调用

    ✅ 恢复后的操作

    ```matlab
    restoredefaultpath;  % 恢复默认路径
    matlabrc;            % 运行启动脚本
    savepath;            % 保存当前路径设置

    % 然后需要重新添加必要的路径
    addpath(genpath('/Users/name/matlab'));
    addpath('/path/to/important/toolbox');
    savepath;  % 再次保存
    ```

    与相关命令对比：

    | 命令 | 作用 | 破坏性 |
    | - | - | - |
    | rehash toolboxcache | 刷新工具箱缓存 | 无破坏 |
    | rehash path | 刷新所有路径缓存 | 无破坏 |
    | restoredefaultpath | 完全重置路径为出厂设置 | 高破坏 |
    | rmpath | 移除单个路径 | 局部影响 |

    最佳实践建议：

    * 先用温和方法：

        ```matlab
        % 首先尝试
        rehash toolboxcache
        rehash path
        ```

    * 备份当前路径：

        ```matlab
        % 在执行 restoredefaultpath 前备份
        p = path;
        save('my_path_backup.mat', 'p');
        ```

    * 作为最后手段：

        * 当其他方法都无效时

        * MATLAB 频繁崩溃或无法正常工作时

        * 怀疑路径被严重破坏时

    总结：restoredefaultpath 是 MATLAB 路径问题的"核选项"——非常有效，但代价是失去所有自定义配置，需谨慎使用。

* matlab `rehash toolboxcache`

    在 MATLAB 中，rehash toolboxcache 的作用是刷新 MATLAB 的工具箱缓存，主要用于解决 MATLAB 找不到新添加的或已修改的函数、文件或工具箱的问题。

    主要应用场景：

    * 手动添加了新的工具箱或函数文件

        * 当你在 MATLAB 搜索路径中添加了新的 .m 文件或工具箱文件，但 MATLAB 无法识别时

    * 更新了现有工具箱

        * 修改了工具箱中的函数，但 MATLAB 仍使用旧版本

    * 工具箱安装后不显示

        * 安装了新的工具箱，但在命令窗口中无法调用其函数

    * MATLAB 提示找不到函数

        * 出现 "Function not found" 错误，即使文件确实存在

    工作机制：

    MATLAB 会缓存工具箱和函数的信息以提高加载速度。rehash toolboxcache 强制 MATLAB：

    * 重新扫描所有工具箱目录

    * 更新内部函数缓存

    * 使新添加或修改的文件立即可用

    使用方法：

    ```matlab
    % 基本用法
    rehash toolboxcache

    % 更彻底的刷新（如果 toolboxcache 无效）
    rehash path
    ```

    类似命令比较：

    * rehash toolboxcache：专门刷新工具箱缓存

    * rehash path：刷新整个 MATLAB 路径的缓存

    * restoredefaultpath：恢复默认路径设置（更彻底，但会清除所有自定义路径）

    注意：执行此命令后，可能需要重新启动 MATLAB 才能使某些更改完全生效。

* matlab `jsondecode()`

    函数原型

    ```matlab
    S = jsondecode(txt)
    ```

    输入参数：

    * txt：包含JSON格式文本的字符串标量或字符向量

    输出参数：

    * S：MATLAB数据结构（结构体、数组、元胞数组等）

    作用

    将JSON（JavaScript Object Notation）格式的文本字符串解码为MATLAB可识别的数据结构。它是MATLAB与JSON数据交换的核心函数，常用于：

    * 读取API返回的JSON数据

    * 解析配置文件

    * 处理Web数据

    * 与JavaScript/Python等其他语言进行数据交换

    用法示例

    基本示例

    ```matlab
    % JSON字符串
    jsonStr = '{"name":"张三","age":25,"scores":[85,92,78]}';

    % 解码为MATLAB结构体
    data = jsondecode(jsonStr);

    disp(data.name)     % 输出：张三
    disp(data.age)      % 输出：25
    disp(data.scores)   % 输出：[85 92 78]
    ```

    处理嵌套结构

    ```matlab
    jsonStr = '{"employee":{"name":"李四","department":"研发部"},"projects":["A","B"]}';
    data = jsondecode(jsonStr);

    disp(data.employee.name)  % 输出：李四
    disp(data.projects{2})    % 输出：B（注意：数组转为元胞数组）
    ```

    从文件读取JSON

    ```matlab
    % 假设有data.json文件
    fid = fopen('data.json', 'r');
    jsonStr = fread(fid, '*char')';
    fclose(fid);

    data = jsondecode(jsonStr);
    ```

    与jsonencode()配对使用

    ```matlab
    % 创建MATLAB结构体
    matlabStruct = struct('x', [1,2,3], 'y', 'text');

    % 编码为JSON字符串
    jsonStr = jsonencode(matlabStruct);
    % jsonStr = '{"x":[1,2,3],"y":"text"}'

    % 解码回MATLAB
    decoded = jsondecode(jsonStr);
    ```

    重要特性

    1. 数据类型映射

    | JSON类型 | MATLAB类型 |
    | - | - |
    | 对象 | 结构体 |
    | 数组（同类型） | 矩阵/数组 |
    | 数组（混合类型） | 元胞数组 |
    | 字符串 | 字符向量 |
    | 数字 | double（默认） |
    | true/false | logical |
    | null | []（空双精度） |

    2. 选项设置（R2021b+）

        ```matlab
        % 保留数字精度（默认转换为double）
        data = jsondecode(jsonStr, 'MakeOutputs', false);
        % 返回未转换的字符串，保留原始数字格式
        ```

    3. 特殊字符处理

        ```matlab
        % JSON中的Unicode转义序列会自动转换
        jsonStr = '{"text":"\u4E2D\u56FD"}'  % "\u4E2D\u56FD" = "中国"
        data = jsondecode(jsonStr);
        disp(data.text)  % 输出：中国
        ```

    4. 错误处理

        ```matlab
        try
            data = jsondecode(invalidJsonStr);
        catch ME
            disp('JSON解析错误:');
            disp(ME.message);
        end
        ```

    实际应用场景

    ```matlab
    % 场景：调用Web API
    url = 'https://api.example.com/data';
    options = weboptions('ContentType', 'json');
    response = webread(url, options);  % response已是结构体

    % 如果API返回原始JSON字符串
    jsonResponse = urlread(url);
    data = jsondecode(jsonResponse);

    % 处理复杂嵌套数据
    % 假设返回：{"status":"success","data":[{"id":1,"value":10},{"id":2,"value":20}]}
    parsed = jsondecode(jsonResponse);
    for i = 1:length(parsed.data)
        fprintf('ID: %d, Value: %d\n', parsed.data(i).id, parsed.data(i).value);
    end
    ```

    注意事项

    * 大文件处理：对于大JSON文件，考虑使用fileread()或fread()分块读取

    * 性能：jsondecode()比loadjson()等第三方函数更快（R2016b+）

    * 内存：解析大型JSON时注意内存使用

    * 向后兼容：需要MATLAB R2016b或更高版本

    替代方案

    * MATLAB R2016b之前：使用loadjson()（File Exchange中的第三方函数）

    * 简单解析：正则表达式（不推荐，容易出错）

    这个函数极大简化了MATLAB与Web服务、配置文件和跨平台应用的数据交换。


* 简述 matlab `fread()`, `fopen()`, `fclose()` 的原型、作用及用法

    在 MATLAB 中，fopen()、fread() 和 fclose() 是用于文件 I/O 操作的核心函数。

    1. fopen() - 打开文件

        原型：

        ```matlab
        fid = fopen(filename, permission)
        [fid, message] = fopen(filename, permission, machineformat)
        ```

        作用：

        打开文件用于读取或写入，返回文件标识符。

        参数：

        * filename：文件名（可包含路径）

        * permission：访问模式

            * 'r'：只读（默认）

            * 'w'：写入（覆盖）

            * 'a'：追加

            * 'r+'：读写

            * 'W', 'A'：无自动刷新的写/追加（用于磁带）

        * machineformat（可选）：字节顺序，如 'ieee-le'（小端）

        示例：

        ```matlab
        % 打开文件用于读取
        fid = fopen('data.bin', 'r');
        if fid == -1
            error('文件打开失败');
        end

        % 或获取错误信息
        [fid, errmsg] = fopen('data.bin', 'r');
        ```

    2. fread() - 读取二进制数据

        原型：

        ```matlab
        A = fread(fid)
        A = fread(fid, size)
        A = fread(fid, size, precision)
        A = fread(fid, size, precision, skip)
        ```

        作用：

        从二进制文件中读取数据。

        参数：

        * fid：文件标识符

        * size：读取数量

            * N：读取 N 个元素

            * [M,N]：读取 M×N 矩阵（按列填充）

            * Inf：读取到文件末尾

        * precision：数据类型

            * 'uint8'、'int16'、'float32' 等

            * '*datatype'：保持原始类型

            * N*datatype：读取 N 个值作为一个元素

        * skip：跳过的字节数

        示例：

        ```matlab
        % 读取整个文件作为 uint8
        data = fread(fid);

        % 读取 100 个 double 值
        data = fread(fid, 100, 'double');

        % 读取 10x20 的 int16 矩阵
        matrix = fread(fid, [10, 20], 'int16');

        % 每跳过 4 字节读取一个 double
        data = fread(fid, 100, 'double', 4);
        ```

    3. fclose() - 关闭文件

        原型：

        ```matlab
        status = fclose(fid)
        status = fclose('all')
        ```

        作用：

        关闭已打开的文件。

        参数：

        * fid：文件标识符

        * 'all'：关闭所有打开的文件

        返回值：

        * 0：成功

        * -1：失败

        示例：

        ```matlab
        % 关闭单个文件
        status = fclose(fid);
        if status == 0
            disp('文件关闭成功');
        end

        % 关闭所有打开的文件
        fclose('all');
        ```

        完整使用示例：

        ```matlab
        % 1. 打开文件
        fid = fopen('data.bin', 'r');
        if fid == -1
            error('无法打开文件');
        end

        % 2. 读取文件头（2个int32）
        header = fread(fid, 2, 'int32');

        % 3. 根据头信息读取数据
        rows = header(1);
        cols = header(2);
        data = fread(fid, [rows, cols], 'double');

        % 4. 关闭文件
        fclose(fid);
        ```

    注意事项：

    * 文件标识符检查：fopen 失败时返回 -1

    * 资源管理：使用 fclose 避免文件描述符泄漏

    * 二进制格式：这些函数主要用于二进制文件，文本文件建议使用 textscan、fgetl 等

    * 精度匹配：确保 precision 参数与文件实际存储格式一致

    * 错误处理：总是检查函数返回值以确保操作成功

* matlab `fileread()`

    fileread() 是 MATLAB 中用于读取文本文件内容的函数。

    函数原型

    ```matlab
    str = fileread(filename)
    ```

    参数说明

    * filename：字符串或字符向量，指定要读取的文件路径

    * 返回值：包含整个文件内容的字符串

    主要作用

    将整个文本文件的内容读取为一个字符串，适用于：

    * 读取配置文件、日志文件

    * 处理源代码文件

    * 读取XML、JSON等文本格式数据（通常配合解析函数使用）

    * 小到中等大小的文本文件处理

    使用示例

    基本用法

    ```matlab
    % 读取文本文件
    content = fileread('data.txt');

    % 显示内容
    disp(content);
    ```

    读取不同类型文件

    ```matlab
    % 读取配置文件
    config = fileread('config.ini');

    % 读取JSON文件（配合jsondecode）
    jsonStr = fileread('data.json');
    data = jsondecode(jsonStr);

    % 读取CSV文件头信息
    csvContent = fileread('data.csv');
    firstLine = strtok(csvContent, newline);
    ```

    处理相对/绝对路径

    ```matlab
    % 相对路径
    content1 = fileread('../data/input.txt');

    % 绝对路径
    content2 = fileread('C:\Users\Name\Documents\file.txt');

    % 当前目录文件
    content3 = fileread('myfile.txt');
    ```

    常见配合操作

    ```matlab
    % 分割为行
    content = fileread('log.txt');
    lines = strsplit(content, {'\r\n', '\n', '\r'});

    % 搜索特定内容
    content = fileread('script.m');
    if contains(content, 'function')
        disp('该文件包含函数定义');
    end

    % 计算文件字符数
    content = fileread('document.txt');
    charCount = length(content);
    ```

    注意事项

    * 文件大小：适合读取中小型文件，大文件建议使用fread或textscan

    * 编码：默认使用系统编码，MATLAB R2018b+ 可指定编码：

        ```matlab
        % R2018b及以后版本
        content = fileread(filename, 'Encoding', 'UTF-8');
        ```

    * 二进制文件：不适合读取二进制文件，会尝试将内容转为字符串

    * 内存占用：整个文件内容会加载到内存中

    * 文件不存在：会抛出错误，建议先检查文件存在性：

        ```matlab
        if exist(filename, 'file')
            content = fileread(filename);
        else
            error('文件不存在');
        end
        ```

    替代方案

    * 大文件：使用 textscan, fread, readlines (R2017b+)

    * 表格数据：使用 readtable

    * 需要逐行处理：使用 fgetl 或 fgets

    fileread() 因其简单易用，是小文件读取场景的首选函数。

* 简述 matlab which() 的原型、作用及用法

    which() 是 MATLAB 中用于定位函数、文件、变量等对象位置的函数。

    函数原型

    ```matlab
    % 基本形式
    path = which(item)

    % 多种变体
    path = which(item, '-all')          % 查找所有同名项
    path = which(item, '-all', '-f')    % 查找所有内置函数
    [path, errmsg] = which(item)        % 获取错误信息
    ```

    主要作用

    * 定位文件路径：确定 MATLAB 如何解析特定名称

    * 调试工具：检查函数遮蔽（shadowing）问题

    * 路径管理：验证文件是否在搜索路径中

    * 类型识别：区分内置函数、工具箱函数、MEX 文件等

    使用示例

    基本定位

    ```matlab
    % 查找函数位置
    which plot            % 显示 plot 函数的路径
    which mean            % 显示 mean 函数的路径
    which('sin')          % 字符串参数形式

    % 查找当前目录文件
    which myfunction.m    % 显示自定义函数的完整路径

    % 查找类方法
    which handle/set
    ```

    查找所有同名项（解决遮蔽问题）

    ```matlab
    % 查找所有名为 'test' 的项
    paths = which('test', '-all')

    % 输出示例：
    % paths = 
    %   'C:\work\test.m'          % 用户自定义
    %   'matlabroot\toolbox\matlab\general\test.m'  % MATLAB 内置
    ```

    获取详细信息

    ```matlab
    % 使用 -f 选项获取更多信息
    which mean -all -f

    % 输出会显示完整的类型信息
    ```

    检查特定类型

    ```matlab
    % 检查是否为内置函数
    path = which('sin', 'builtin')

    % 检查是否为变量
    path = which('myVariable')
    % 如果是变量，返回空字符串或提示不存在
    ```

    带错误信息的调用

    ```matlab
    % 获取错误信息
    [path, errmsg] = which('nonexistent_function')
    % path = ''
    % errmsg = '函数不存在'
    ```

    常见应用场景

    ```matlab
    %% 1. 调试函数遮蔽问题
    % 如果自定义函数与内置函数同名
    which test -all
    % 可以查看哪个版本被优先调用

    %% 2. 验证工具箱安装
    % 检查特定工具箱函数是否可用
    if ~isempty(which('svmclassify'))
        disp('统计和机器学习工具箱已安装')
    end

    %% 3. 检查文件位置
    scriptPath = which('myscript.m')
    if isempty(scriptPath)
        error('脚本不在MATLAB路径中')
    end

    %% 4. 获取文件所在目录
    [filePath, ~] = fileparts(which('myfunction.m'));
    disp(['函数位于：', filePath]);
    ```

    返回类型判断

    which() 的输出可以判断对象类型：

    | 返回路径特征 | 类型说明 |
    | - | - |
    | 'built-in' | MATLAB 内置函数 |
    | 以 .m 结尾 | M 文件函数 |
    | 以 .mex 结尾 | MEX 文件 |
    | 以 .p 结尾 | P 代码文件 |
    | 空字符串 | 变量或不存在的项 |
    | 目录路径 | 类或包目录 |

    实用技巧

    判断函数类型

    ```matlab
    function type = checkFunctionType(funcName)
        path = which(funcName);
        
        if isempty(path)
            type = '不存在或变量';
        elseif strcmp(path, 'built-in')
            type = '内置函数';
        elseif endsWith(path, '.m')
            type = 'M文件函数';
        elseif endsWith(path, '.mex')
            type = 'MEX文件';
        else
            type = '其他类型';
        end
    end
    ```

    查找类定义

    ```matlab
    % 查找类的定义位置
    which handle          % 查找 handle 类
    which containers.Map  % 查找特定类
    ```

    批量检查

    ```matlab
    % 检查多个函数是否存在
    functions = {'plot', 'myFunc', 'toolboxFunc'};
    for i = 1:length(functions)
        if ~isempty(which(functions{i}))
            fprintf('%s: 可用\n', functions{i});
        else
            fprintf('%s: 不可用\n', functions{i});
        end
    end
    ```

    注意事项

    * 搜索顺序：按照 MATLAB 路径顺序搜索

    * 变量优先：工作区变量优先于函数名

    * 性能：对于频繁调用，可缓存结果

    * 私有函数：在父目录外调用 which 可能找不到私有函数

    * 方法：需要指定完整方法名：className/methodName

    which() 是 MATLAB 开发和调试中的重要工具，特别是在处理大型项目或调试路径问题时非常有用。

* matlab 2025b x64 破解版下载

    <https://www.puresys.net/8739.html>

## cache

* matlab `butter()`

    butter() 是 MATLAB 中设计 Butterworth 滤波器（一种最平坦通带响应的滤波器）的核心函数。

    一、函数原型

    ```matlab
    % 完整形式
    [b, a] = butter(n, Wn, ftype, s)
    % 或使用数字滤波器对象
    [z, p, k] = butter(n, Wn, ftype, s)
    sos = butter(n, Wn, ftype, s)
    ```

    二、参数说明

    | 参数 | 说明 |
    | - | - |
    | n | 滤波器阶数 |
    | Wn | 归一化截止频率 (0~1，1对应奈奎斯特频率) |
    | ftype | 滤波器类型：'low'(低通)、'high'(高通)、'stop'(带阻)、'bandpass'(带通) |
    | s | 可选，用于设计模拟滤波器，如 's' |
    | b, a | 返回的传递函数系数（分子、分母） |
    | z, p, k | 零极点增益形式 |
    | sos | 二阶节形式，数值更稳定 |

    三、主要作用

    设计 Butterworth 滤波器，其特点：

    * 通带内具有最大平坦的幅度响应

    * 过渡带衰减平缓（相比椭圆、切比雪夫滤波器）

    * 无纹波（通带和阻带都单调）

    四、基本用法示例

    1. 低通滤波器设计

        ```matlab
        fs = 1000;            % 采样率 1000 Hz
        fc = 50;              % 截止频率 50 Hz
        Wn = fc/(fs/2);       % 归一化频率
        n = 6;                % 6阶滤波器

        [b, a] = butter(n, Wn, 'low');
        freqz(b, a);          % 查看频率响应
        ```

    2. 高通滤波器

        ```matlab
        fc = 200;             % 截止频率 200 Hz
        Wn = fc/(fs/2);
        [b, a] = butter(4, Wn, 'high');
        ```

    3. 带通滤波器

        ```matlab
        f_low = 100;          % 通带下限 100 Hz
        f_high = 200;         % 通带上限 200 Hz
        Wn = [f_low, f_high]/(fs/2);
        [b, a] = butter(4, Wn, 'bandpass');
        ```

    4. 带阻滤波器（陷波器）

        ```matlab
        Wn = [48, 52]/(fs/2); % 阻带 48-52 Hz
        [b, a] = butter(4, Wn, 'stop');
        ```

    5. 使用 sos 形式（推荐）

        ```matlab
        [sos, g] = butter(6, Wn, 'low');
        filtObj = dfilt.df2sos(sos, g);  % 创建滤波器对象
        ```

    6. 模拟滤波器设计

        ```matlab
        % 设计模拟低通滤波器
        [b, a] = butter(4, 2*pi*100, 's');  % 100 Hz 截止
        freqs(b, a);                        % 查看模拟频率响应
        ```

    五、滤波应用

    ```matlab
    % 设计滤波器
    [b, a] = butter(6, 0.4, 'low');

    % 滤波操作
    t = 0:0.001:1;
    x = sin(2*pi*10*t) + 0.5*sin(2*pi*100*t);  % 混合信号
    y = filter(b, a, x);                       % 滤波

    % 或使用 filtfilt 实现零相位滤波
    y_zero_phase = filtfilt(b, a, x);
    ```

    六、设计技巧

    * 确定阶数：可先用 buttord() 计算最小所需阶数

        ```matlab
        [n, Wn] = buttord(Wp, Ws, Rp, Rs);
        ```

    * 归一化频率计算：

        * 数字滤波器：Wn = 频率/(fs/2)

        * 模拟滤波器：Wn = 2π*频率

    * 稳定性：高阶滤波器建议使用 sos 形式避免数值问题

    七、注意事项

    * Butterworth 滤波器阶数越高，过渡带越陡，但相位非线性越严重

    * 对于要求线性相位的应用，考虑使用 FIR 滤波器（如 fir1()）

    * 采样频率至少为信号最高频率的 2 倍（满足奈奎斯特采样定理）

    这个函数是 MATLAB 数字信号处理工具箱的基础，广泛用于信号去噪、频率分离等应用场景。

* 感觉 toml.read() 用法比较奇怪，因为在执行前不知道有哪些 fields，执行后才知道。比较适合一行一行执行，做实验，不适合正式开发场景

    您指出的这一点非常关键和准确！这确实是 MATLAB 中 toml.read() 这类动态解析函数在正式开发中的主要痛点。让我深入分析这个问题并提供更健壮的解决方案。

    问题的本质：动态字段访问的隐患

    ```matlab
    % 问题代码示例
    config = toml.read('config.toml');
    value = config.some_field;  % ⚠️ 运行时才知道是否存在

    % 如果配置文件被修改或字段名拼写错误：
    value = config.database.hostname;  % 实际是 config.database.host
    % → 运行时错误：引用了不存在的字段
    ```

    更稳健的正式开发方案

    方案1：验证层模式（推荐）

    ```matlab
    function config = loadConfig(filename)
        % 步骤1：读取原始数据
        raw = toml.read(filename);
        
        % 步骤2：验证和转换
        config = validateConfig(raw);
    end

    function cfg = validateConfig(raw)
        % 定义期望的字段结构
        expectedFields = {
            'database.host', 'string';
            'database.port', 'numeric';
            'settings.timeout', 'numeric';
            'user.name', 'string';
        };
        
        cfg = struct();
        
        % 逐个验证字段
        for i = 1:size(expectedFields, 1)
            fieldPath = strsplit(expectedFields{i, 1}, '.');
            expectedType = expectedFields{i, 2};
            
            try
                value = getNestedField(raw, fieldPath);
                
                % 类型检查
                switch expectedType
                    case 'string'
                        if ~ischar(value)
                            error('类型错误: %s 应为字符串', expectedFields{i, 1});
                        end
                    case 'numeric'
                        if ~isnumeric(value)
                            error('类型错误: %s 应为数字', expectedFields{i, 1});
                        end
                end
                
                % 设置到输出结构体
                cfg = setNestedField(cfg, fieldPath, value);
                
            catch ME
                % 提供有意义的错误信息
                error('配置验证失败: %s - %s', expectedFields{i, 1}, ME.message);
            end
        end
    end
    ```

    方案2：配置模式定义（更正式）

    ```matlab
    classdef AppConfig < handle
        % 明确定义所有配置字段和默认值
        properties
            % 数据库配置
            db_host char = 'localhost'
            db_port double = 3306
            db_enabled logical = true
            
            % 用户配置
            user_name char = '默认用户'
            user_tags cell = {'default'}
            
            % 设置
            timeout double = 30.0
            max_retries double = 3
        end
        
        methods
            function obj = AppConfig(tomlFile)
                if nargin > 0
                    obj.loadFromToml(tomlFile);
                end
            end
            
            function loadFromToml(obj, filename)
                raw = toml.read(filename);
                
                % 安全赋值，有类型检查
                obj.db_host = getFieldSafe(raw, 'database.host', obj.db_host);
                obj.db_port = getFieldSafe(raw, 'database.port', obj.db_port);
                % ... 其他字段
            end
        end
    end

    % 使用方式
    config = AppConfig('config.toml');
    % 现在有代码补全和类型安全
    fprintf('主机: %s\n', config.db_host);  % ✅ 有代码提示
    ```

    方案3：配置工厂模式

    ```matlab
    function config = createConfig(filename)
        % 1. 读取原始数据
        raw = toml.read(filename);
        
        % 2. 转换为强类型配置对象
        config = struct();
        
        % 数据库配置组
        config.database = struct(...
            'host', ensureString(getNested(raw, 'database.host', 'localhost')), ...
            'port', ensureNumber(getNested(raw, 'database.port', 3306), 1, 65535), ...
            'enabled', ensureBoolean(getNested(raw, 'database.enabled', true)) ...
        );
        
        % 用户配置组
        config.user = struct(...
            'name', ensureString(getNested(raw, 'user.name', 'anonymous')), ...
            'tags', ensureCellArray(getNested(raw, 'user.tags', {})) ...
        );
        
        % 设置默认值，确保结构完整
        config = applyDefaults(config);
    end

    function value = getNested(data, path, default)
        % 安全的嵌套字段访问
        parts = strsplit(path, '.');
        value = data;
        for i = 1:length(parts)
            if isfield(value, parts{i})
                value = value.(parts{i});
            else
                value = default;
                return;
            end
        end
    end
    ```

    方案4：使用 MATLAB 的输入解析器（适合函数配置）

    ```matlab
    function processData(inputFile, varargin)
        % 定义可配置参数
        p = inputParser;
        p.addRequired('inputFile');
        p.addParameter('Host', 'localhost', @ischar);
        p.addParameter('Port', 3306, @(x) isnumeric(x) && x > 0);
        p.addParameter('Timeout', 30, @isnumeric);
        p.addParameter('Enabled', true, @islogical);
        
        % 从 TOML 读取配置并合并
        if exist('config.toml', 'file')
            tomlConfig = toml.read('config.toml');
            % 转换为参数对
            params = structToParams(tomlConfig);
            p.parse(inputFile, params{:}, varargin{:});
        else
            p.parse(inputFile, varargin{:});
        end
        
        results = p.Results;
        % 使用 results.Host, results.Port 等
    end
    ```

    最佳实践建议

    1. 开发阶段 vs 生产阶段

        ```matlab
        % 开发时：使用完整配置
        if isDevelopment()
            config = toml.read('config_dev.toml');
        else
            % 生产时：使用验证过的配置
            config = loadValidatedConfig('config_prod.toml');
        end
        ```

    2. 配置版本控制

        ```matlab
        function config = loadVersionedConfig(filename)
            raw = toml.read(filename);
            
            % 检查配置版本
            if ~isfield(raw, 'config_version')
                error('配置文件缺少版本信息');
            end
            
            % 根据版本使用不同的验证规则
            switch raw.config_version
                case '1.0'
                    config = validateV1Config(raw);
                case '2.0'
                    config = validateV2Config(raw);
                otherwise
                    error('不支持的配置版本: %s', raw.config_version);
            end
        end
        ```

    3. 配置文档生成

        ```matlab
        % 从验证代码生成配置模板
        function generateConfigTemplate()
            template = struct();
            template.database = struct(...
                'host', 'localhost  # 数据库主机地址', ...
                'port', '3306       # 数据库端口', ...
                'enabled', 'true    # 是否启用数据库' ...
            );
            
            toml.write('config_template.toml', template);
        end
        ```

    总结对比

    | 方法 | 适合场景 | 优点 | 缺点 |
    | - | - | - | - |
    | 直接 toml.read() | 快速原型、实验 | 简单直接 | 无类型安全、无代码补全 |
    | 验证层模式 | 中小型项目 | 有错误检查、可维护 | 需要额外代码 |
    | 配置类模式 | 大型正式项目 | 类型安全、代码补全 | 需要设计类结构 |
    | 工厂模式 | 需要默认值的场景 | 鲁棒性强 | 配置重复定义 |

    建议：对于正式开发，至少使用方案1（验证层）。对于团队协作或长期维护的项目，方案2（配置类） 是最佳选择。TOML 作为配置文件格式本身很好，关键在于不要让动态解析的数据直接进入业务逻辑，而应该通过一层验证和转换。

* 简述 matlab `toml.read()` 的原型、作用及用法

    toml.read() 是 MATLAB 中用于读取 TOML 文件的函数，通常来自第三方工具箱。

    函数原型

    ```matlab
    data = toml.read(filename)
    ```

    * 输入：filename - TOML 文件路径（字符串）

    * 输出：data - 包含 TOML 数据的 MATLAB 结构体

    主要作用

    将 TOML（Tom's Obvious, Minimal Language）格式的配置文件解析为 MATLAB 数据结构，便于在 MATLAB 中访问和使用配置参数。

    基本用法

    1. 读取 TOML 文件

        ```matlab
        % 读取 TOML 配置文件
        config = toml.read('config.toml');

        % 访问数据
        disp(config.database.host);  % 访问嵌套字段
        disp(config.settings.timeout);
        ```

    2. 示例 TOML 文件 (config.toml)

        ```toml
        title = "项目配置"

        [database]
        host = "localhost"
        port = 5432
        enabled = true

        [user]
        name = "张三"
        tags = ["工程师", "研究员"]

        [settings]
        timeout = 30.5
        retry_attempts = 3
        ```

    3. 完整示例

        ```matlab
        % 读取配置文件
        config = toml.read('config.toml');

        % 访问不同类型的数据
        % 字符串
        fprintf('标题: %s\n', config.title);

        % 数字
        fprintf('端口: %d\n', config.database.port);

        % 布尔值
        if config.database.enabled
            disp('数据库已启用');
        end

        % 数组/元胞数组
        fprintf('用户标签: %s\n', strjoin(config.user.tags, ', '));

        % 浮点数
        fprintf('超时时间: %.1f秒\n', config.settings.timeout);
        ```

    注意事项

    * 安装要求：需要先安装第三方 TOML 工具箱

        ```matlab
        % 通过 File Exchange 或手动安装
        % 常用的有：
        % - tomlab: https://github.com/g-s-k/matlab-toml
        % - tomllib: https://github.com/Percontarix/tomllib
        ```

    * 数据类型转换：

        * TOML 数组 → MATLAB 元胞数组

        * TOML 表 → MATLAB 结构体

        * TOML 布尔值 → MATLAB logical

        * TOML 时间 → MATLAB datetime

    * 错误处理：

        ```matlab
        try
            config = toml.read('config.toml');
        catch ME
            fprintf('读取文件失败: %s\n', ME.message);
            config = struct();  % 返回空结构体
        end
        ```

    优势

    * 人类可读：相比 JSON 和 XML，TOML 更易于手动编辑

    * 类型支持：完整支持日期、时间等复杂类型

    * 层级清晰：适合存储分层配置数据

    常用工具箱

    * tomlab：功能完整的 MATLAB TOML 实现

    * tomllib：另一个轻量级替代方案

    * 配置管理：常用于科学计算项目的参数配置

    这个函数特别适合需要灵活配置参数的 MATLAB 项目，如实验设置、算法参数调优等场景。

* matlab load()

    MATLAB load() 函数

    1. 原型

        ```matlab
        load(filename)
        load(filename, variables)
        load(filename, '-mat')
        load(filename, '-ascii')
        S = load(___)
        ```

    2. 作用

        * 从磁盘文件加载变量到MATLAB工作空间

        * 支持多种文件格式：MAT文件、ASCII文件、Excel文件等

        * 默认加载MAT格式文件（扩展名通常为.mat）

    3. 基本用法

        ```matlab
        % 1. 加载整个MAT文件（所有变量）
        load('data.mat')

        % 2. 加载指定变量
        load('data.mat', 'x', 'y')

        % 3. 将加载的变量保存到结构体（不直接放入工作空间）
        S = load('data.mat')

        % 4. 加载ASCII文件
        load('data.txt', '-ascii')
        ```

    4. load(nirsFileName,'-mat'); 的作用

        ```matlab
        load(nirsFileName, '-mat')
        ```

        作用解释：

        * nirsFileName：包含近红外光谱(NIRS)数据的文件名（字符串）

        * '-mat'：明确指定以MAT文件格式加载

        特殊含义：

        * 强制MAT格式：即使文件扩展名不是.mat，也按MAT格式解析

        * 防止歧义：当文件名没有标准扩展名或扩展名不明确时，确保正确加载

        * NIRS数据场景：近红外光谱数据文件有时可能使用非标准扩展名（如.nirs、.data等）

        示例场景：

        ```matlab
        % 假设文件是NIRS数据，扩展名为 .nirs 但内部是MAT格式
        nirsFileName = 'subject01_data.nirs';
        load(nirsFileName, '-mat');  % 明确按MAT格式加载

        % 等价于（如果扩展名是 .mat）：
        load('subject01_data.mat');
        ```

        实际应用中的典型NIRS文件：

        ```matlab
        % NIRS数据通常包含多个变量：
        % - d: 光强度数据
        % - s: 刺激标记
        % - t: 时间向量
        % - aux: 辅助信号

        load('experiment.nirs', '-mat');
        whos  % 查看加载的变量
        % 可能显示：d, s, t, aux, ml, etc.
        ```

    5. 注意事项

        ```matlab
        % 如果文件已经是 .mat 扩展名，'-mat' 是可选的
        load('data.mat')           % 默认就是MAT格式
        load('data.mat', '-mat')   % 明确指定，结果相同

        % 但如果扩展名不标准或缺失，'-mat' 很重要
        load('data', '-mat')       % 无扩展名，强制按MAT格式
        load('data.nirs', '-mat')  % 特殊扩展名，强制按MAT格式
        ```

    6. 与NIRS工具箱结合使用

        ```matlab
        % 在fNIRS分析中常见用法
        function processNIRSData(nirsFile)
            % 加载NIRS数据
            load(nirsFile, '-mat');
            
            % 访问典型NIRS变量
            % d: 原始光强度数据 [时间点 × 通道]
            % s: 刺激标记矩阵 [时间点 × 条件]
            % t: 时间向量 [时间点 × 1]
            
            % 进一步处理...
            [oxy, deoxy] = convertToHb(d, ...);  % 转换为血红蛋白浓度
        end
        ```

    这种用法确保了即使NIRS数据文件使用非标准扩展名，也能正确加载MAT格式的数据结构。

* matlab saveas()

    saveas() 是 MATLAB 中用于将图形窗口保存为图像文件的函数。

    函数原型

    ```matlab
    saveas(fig, filename, format)
    ```

    或简化版本：

    ```matlab
    saveas(fig, filename)
    ```

    参数说明

    * fig：图形窗口句柄或图形对象数组

    * filename：保存的文件名（可包含路径）

    * format（可选）：文件格式字符串

    **主要作用**

    将 MATLAB 图形窗口的内容保存为各种格式的图像文件，支持多种常见图像格式。

    **支持的主要格式**

    * 图像格式：'png', 'jpg'/'jpeg', 'tif'/'tiff', 'bmp', 'gif'

    * 矢量格式：'pdf', 'eps', 'emf', 'svg'

    * MATLAB 格式：'fig'（MATLAB 专有格式，可重新编辑）

    用法示例

    基本用法

    ```matlab
    % 创建图形
    fig = figure;
    plot(1:10, rand(1,10));

    % 保存为PNG格式（默认）
    saveas(fig, 'myplot.png');

    % 指定格式
    saveas(fig, 'myplot.jpg', 'jpg');
    saveas(fig, 'myplot.pdf', 'pdf');
    ```

    指定路径

    ```matlab
    % 保存到特定目录
    saveas(fig, 'C:\plots\figure1.png');
    saveas(fig, '../results/plot.pdf', 'pdf');
    ```

    批量保存多个图形

    ```matlab
    % 创建多个图形
    fig1 = figure(1);
    plot(sin(0:0.1:2*pi));
    fig2 = figure(2);
    plot(cos(0:0.1:2*pi));

    % 分别保存
    saveas(fig1, 'sine_wave.png');
    saveas(fig2, 'cosine_wave.png');
    ```

    使用gca保存当前坐标区

    ```matlab
    % 保存当前坐标区
    plot(1:10, 1:10);
    saveas(gca, 'current_axes.png', 'png');
    ```

    注意事项

    * 分辨率控制：saveas() 默认使用屏幕分辨率，对于高质量输出，建议使用 print() 函数：

        ```matlab
        print(fig, 'myplot.png', '-dpng', '-r300');  % 300 DPI
        ```

    * 大小调整：保存前可使用 set(fig, 'Position', ...) 调整图形大小

    * 文件扩展名：如果文件名中包含扩展名且与格式参数一致，MATLAB 会使用该格式

    * 推荐替代：对于更高级的图像输出控制，推荐使用：

        ```matlab
        exportgraphics(fig, filename)       % R2020a+
        exportfig(fig, filename)             % 第三方工具
        print(fig, filename, formatoptions)  % 传统但功能强大
        ```

    总结

    saveas() 是 MATLAB 中最简单的图形保存方式，适合快速保存图形。对于需要控制分辨率、裁剪、背景透明等高级需求，建议使用 print() 或 exportgraphics() 函数。

* matlab exist()

    exist() 是 MATLAB 中用于检查变量、函数、文件或文件夹是否存在的函数。

    原型

    ```matlab
    status = exist(name, kind)
    status = exist(name)
    ```

    参数说明

    * name: 要检查的名称（字符串或字符向量）

    * kind (可选): 指定检查的类型，可以是：

        * 'var': 检查变量

        * 'builtin': 检查内置函数

        * 'file': 检查文件或文件夹

        * 'dir': 检查文件夹

        * 'class': 检查类

        * 'all': 检查所有类型（默认）

    返回值 status

    * 0: 不存在

    * 1: 变量

    * 2: 文件（M文件、P文件等）

    * 3: MEX文件

    * 4: MDL文件（Simulink模型）

    * 5: 内置函数

    * 6: P文件

    * 7: 文件夹

    * 8: 类

    主要用法示例

    ```matlab
    % 1. 检查变量是否存在
    x = 10;
    if exist('x', 'var')
        disp('变量 x 存在');
    end

    % 2. 检查文件是否存在
    if exist('myfunction.m', 'file')
        disp('myfunction.m 文件存在');
    end

    % 3. 检查文件夹是否存在
    if exist('myfolder', 'dir')
        disp('文件夹存在');
    end

    % 4. 不指定类型（自动检测）
    status = exist('plot');  % 返回 5（内置函数）
    status = exist('sin');   % 返回 5（内置函数）

    % 5. 检查用户自定义函数
    if exist('myCustomFunc') == 2
        disp('这是一个M文件函数');
    end
    ```

    实用技巧

    ```matlab
    % 检查并创建变量
    if ~exist('data', 'var')
        data = load('data.mat');
    end

    % 检查文件并加载
    filename = 'results.mat';
    if exist(filename, 'file')
        load(filename);
    else
        error('文件不存在');
    end

    % 检查文件夹并创建
    folder = 'output';
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    ```

    注意事项

    * exist 会搜索 MATLAB 路径，包括当前目录

    * 当不指定类型时，会按优先级返回第一个匹配项

    * 对于重名的情况（如变量和文件同名），指定类型参数可以避免歧义

    * 比直接使用 isfile() 或 isfolder() 更通用，但性能稍差

    exist() 是 MATLAB 编程中常用的存在性检查函数，特别适用于：

    * 避免重复定义变量

    * 检查依赖文件是否存在

    * 条件性加载数据或创建目录

* matlab `reshape()`

    reshape() 是 MATLAB 中用于改变数组维度而不改变数据总量的核心函数。

    1. 函数原型

        ```matlab
        B = reshape(A, sz)
        B = reshape(A, sz1, sz2, ..., szN)
        ```

        * A: 输入数组（任意维度）

        * sz: 新维度大小的向量，如 [m, n, p, ...]

        * sz1, sz2, ..., szN: 各维度的具体大小

        * B: 重塑后的输出数组

    2. 核心作用

        重新排列数组的维度布局，保持元素总数不变（即 numel(A) == numel(B)）。

    3. 基本用法示例

        示例1：矩阵维度转换

        ```matlab
        A = 1:12;            % 1×12 行向量
        B = reshape(A, 3, 4) % 重塑为 3×4 矩阵
        % B = 
        %    1    4    7   10
        %    2    5    8   11
        %    3    6    9   12
        ```

        示例2：多维度数组

        ```matlab
        A = 1:24;
        B = reshape(A, [2, 3, 4])  % 三维数组：2×3×4
        ```

        示例3：自动推断维度

        ```matlab
        A = rand(4, 6);           % 4×6 矩阵
        B = reshape(A, 3, [])     % 3×8 矩阵（自动计算第二维为8）
        ```

    4. 关键特性

        4.1 按列优先顺序

        MATLAB 按列优先（column-major）顺序重塑：

        ```matlab
        A = [1 2 3; 4 5 6];  % 2×3 矩阵
        % A = 
        %     1  2  3
        %     4  5  6

        B = reshape(A, 3, 2)
        % B = 
        %     1  3  5
        %     2  4  6
        % 按列读取：1→4→2→5→3→6
        ```

        4.2 保持元素总数不变

        ```matlab
        A = rand(5, 7);
        % numel(A) = 35，新维度乘积必须为35
        B = reshape(A, 7, 5);    % √ 7×5=35
        % C = reshape(A, 6, 6);  % × 会报错：6×6≠35
        ```

        4.3 与转置的区别

        ```matlab
        A = [1 2 3; 4 5 6];
        B1 = reshape(A, 3, 2)   % 改变数据布局
        B2 = A'                 % 转置：行列交换
        % B1 = [1 3 5; 2 4 6]
        % B2 = [1 4; 2 5; 3 6]
        ```

    5. 实用技巧

        5.1 向量化矩阵

        ```matlab
        A = magic(3);
        v = reshape(A, [], 1)   % 展成列向量，等价于 A(:)
        ```

        5.2 批量图像处理

        ```matlab
        % 将多个 28×28 图像展平为向量
        images = rand(28, 28, 1000);      % 1000张灰度图
        flattened = reshape(images, 784, 1000);  % 784×1000
        ```

        5.3 维度顺序调整（结合 permute）

        ```matlab
        % 将通道数放在第三维（适合图像处理）
        data = rand(224, 224, 3, 100);  % 100张RGB图像
        data_reshaped = reshape(data, 224, 224, 3, 100);
        ```

    6. 常见应用场景

        * 数据预处理：统一输入维度

        * 矩阵运算：改变形状以适应算法要求

        * 图像处理：展平图像为特征向量

        * 神经网络：调整数据维度以适应网络层

    注意事项

    * reshape 不改变内存中的数据顺序，仅改变索引方式

    * 新维度的元素总数必须与原数组相同

    * 与 squeeze() 结合可去除单一维度：reshape(squeeze(A), ...)

    * 高维数组重塑时，建议用向量形式 [dim1, dim2, ...] 提高可读性

* matlab `dir()`

    dir() 是 MATLAB 中用于获取目录内容信息的核心函数。

    1. 函数原型

        ```matlab
        files = dir
        files = dir(name)
        ```

        * 无输入参数: 获取当前目录内容

        * name: 指定目录路径或文件通配符

        * files: 返回结构体数组，包含文件/文件夹信息

    2. 核心作用

        列出目录中的文件和子目录信息，包括文件名、日期、大小、是否目录等属性。

    3. 输出结构体字段

        ```matlab
        files = dir;
        % 每个元素包含以下字段：
        files(1).name      % 文件/文件夹名称（字符串）
        files(1).folder    % 所在文件夹完整路径（字符串）
        files(1).date      % 修改日期时间（字符串）
        files(1).bytes     % 文件大小（字节数）
        files(1).isdir     % 是否为目录（逻辑值：1=是，0=否）
        files(1).datenum   % 日期序列号（可用于排序）
        ```

    4. 基本用法示例

        示例1：列出当前目录所有内容

        ```matlab
        files = dir;  % 获取当前目录内容
        disp(files)   % 显示所有文件和文件夹
        ```

        示例2：列出指定目录

        ```matlab
        % 列出特定目录
        files = dir('C:\Users\Documents\MATLAB');

        % 列出当前目录的子目录
        files = dir('./data');
        ```

        示例3：使用通配符过滤

        ```matlab
        % 所有 .m 文件
        m_files = dir('*.m');

        % 所有以 test 开头的文件
        test_files = dir('test*');

        % 特定扩展名
        csv_files = dir('*.csv');

        % 子目录中的特定文件
        data_files = dir('data/*.mat');
        ```

        示例4：获取文件详细信息

        ```matlab
        files = dir('*.mat');
        for i = 1:length(files)
            fprintf('文件名: %s\n', files(i).name);
            fprintf('大小: %.2f KB\n', files(i).bytes/1024);
            fprintf('修改日期: %s\n', files(i).date);
            fprintf('是否为目录: %d\n', files(i).isdir);
            fprintf('---\n');
        end
        ```

    5. 实用技巧与常见操作

        5.1 过滤掉 "." 和 ".." 目录

        ```matlab
        files = dir;
        % 去除当前目录(.)和父目录(..)
        files = files(~ismember({files.name}, {'.', '..'}));

        % 或者只保留文件（排除目录）
        files = dir;
        files = files(~[files.isdir]);  % 只保留文件
        ```

        5.2 按特定条件筛选文件

        ```matlab
        % 按日期筛选（最近7天内修改的）
        files = dir('*.m');
        file_dates = [files.datenum];
        recent_idx = file_dates > (now - 7);
        recent_files = files(recent_idx);

        % 按大小筛选（大于1MB的文件）
        files = dir('*.mat');
        large_files = files([files.bytes] > 1024^2);
        ```

        5.3 文件排序

        ```matlab
        files = dir('*.csv');
        % 按文件名排序
        [~, idx] = sort({files.name});
        files = files(idx);

        % 按修改日期排序（从新到旧）
        [~, idx] = sort([files.datenum], 'descend');
        files = files(idx);

        % 按文件大小排序
        [~, idx] = sort([files.bytes]);
        files = files(idx);
        ```

        5.4 递归获取所有子目录文件

        ```matlab
        function all_files = getAllFiles(dir_path)
            % 获取目录及所有子目录中的文件
            files = dir(dir_path);
            files = files(~ismember({files.name}, {'.', '..'}));
            
            all_files = {};
            
            for i = 1:length(files)
                file_path = fullfile(files(i).folder, files(i).name);
                
                if files(i).isdir
                    % 递归处理子目录
                    sub_files = getAllFiles(file_path);
                    all_files = [all_files; sub_files];
                else
                    % 添加文件
                    all_files = [all_files; {file_path}];
                end
            end
        end
        ```

        5.5 批量处理文件

        ```matlab
        % 批量读取和处理图像
        image_files = dir('images/*.jpg');
        for i = 1:length(image_files)
            filename = fullfile(image_files(i).folder, image_files(i).name);
            img = imread(filename);
            % 处理图像...
        end
        ```

        5.6 与 fullfile() 结合使用

        ```matlab
        files = dir('data/*.txt');
        for i = 1:length(files)
            % 构建完整文件路径
            full_path = fullfile(files(i).folder, files(i).name);
            
            % 读取文件
            data = readtable(full_path);
            % 处理数据...
        end
        ```

    6. 注意事项

        * 隐藏文件：dir 会显示隐藏文件（如 .gitignore）

        * 性能考虑：处理大量文件时，考虑使用通配符过滤

        * 路径分隔符：使用 fullfile() 处理跨平台路径问题

        * 符号链接：会返回符号链接本身的信息

        * 空目录：空目录返回空的结构体数组

    7. 常见应用场景

        * 数据批量导入：读取目录中所有数据文件

        * 文件管理：查找、筛选、组织文件

        * 自动化处理：批量处理图像、文本等文件

        * 项目结构分析：获取项目文件列表

    8. 相关函数

        * ls: 简单列出文件名（返回字符数组）

        * what: 列出MATLAB特定文件（.m, .mat, .mlx等）

        * exist: 检查文件或文件夹是否存在

        * cd: 更改当前目录

        * pwd: 显示当前目录

* matlab `min()`, `max()`

    原型

    单输出：

    ```matlab
    M = min(A)          % 返回数组的最小元素
    M = min(A, [], dim) % 沿维度dim返回最小值
    ```

    多输出（获取值和位置）：

    ```matlab
    [M, I] = min(A)          % 同时返回最小值及其索引
    [M, I] = min(A, [], dim) % 沿维度dim返回最小值和索引
    ```

    元素级比较：

    ```matlab
    C = min(A, B)  % 返回A和B逐元素比较的较小值
    ```

    主要功能

    * 查找最小值：返回数组中的最小元素

    * 沿指定维度查找：可指定操作维度

    * 元素级比较：比较两个数组的对应元素

    * 获取索引：可同时返回最小值的位置

    基本用法示例

    ```matlab
    % 1. 向量最小值
    A = [3, 1, 4, 1, 5];
    min_val = min(A)  % 返回 1

    % 2. 同时获取值和索引
    [val, idx] = min(A)  % val=1, idx=2（第一个最小值的位置）

    % 3. 矩阵每列最小值
    B = [3, 7, 1; 2, 9, 5];
    col_min = min(B)  % 返回 [2, 7, 1]

    % 4. 沿指定维度
    row_min = min(B, [], 2)  % 每行最小值： [1; 2]

    % 5. 元素级比较
    C = [4, 2, 6];
    D = [1, 5, 3];
    result = min(C, D)  % 返回 [1, 2, 3]
    ```

    注意事项

    * 默认情况下，对于矩阵，min(A)返回每列的最小值

    * 忽略NaN值（除非所有元素都是NaN）

    * 可处理复数（按幅值比较）

    * max()用法完全相同，只是查找最大值

    * `[]`是占位的，无实际意义

        MATLAB 中许多函数使用相同的占位符模式：

        * sum(A, dim) ✓ 不需要占位符（没有两个数组相加的歧义）

        * min(A, [], dim) ✓ 需要占位符

        * max(A, [], dim) ✓ 需要占位符

        * mean(A, dim) ✓ 不需要占位符

    max()函数具有完全相同的语法和选项，只是功能相反（查找最大值）。

* 简述 matlab gpuarray 的用法

    gpuArray 是 MATLAB 中用于 GPU 并行计算的核心工具，它将数据从 CPU 内存传输到 GPU 显存，从而利用 GPU 进行并行计算加速。以下是其主要用法：

    1. 基本操作

        创建 gpuArray

        ```matlab
        % 将现有数据上传到 GPU
        A = rand(1000, 'single');  % 单精度浮点数更适合 GPU
        gpuA = gpuArray(A);

        % 直接在 GPU 上创建数据
        gpuB = rand(1000, 'gpuArray');          % 双精度
        gpuC = rand(1000, 'single', 'gpuArray'); % 单精度

        % 其他初始化函数
        gpuOnes = ones(1000, 'gpuArray');
        gpuZeros = zeros(1000, 'gpuArray');
        gpuEye = eye(1000, 'gpuArray');
        ```

        检查 GPU 可用性

        ```matlab
        gpuDeviceCount()      % 查看 GPU 数量
        d = gpuDevice()       % 查看当前 GPU 信息
        gpuAvailable = gpuDeviceCount() > 0  % 判断 GPU 是否可用
        ```

    2. 运算规则

        支持的操作

        * 大多数内置函数和运算符自动支持：

            ```matlab
            gpuA = gpuArray(rand(1000));
            gpuB = gpuArray(rand(1000));

            % 自动在 GPU 上执行
            C = gpuA * gpuB;      % 矩阵乘法
            D = sin(gpuA);        % 逐元素运算
            E = fft(gpuA);        % 快速傅里叶变换
            ```

        混合运算

        * 允许：gpuArray 与标量/小矩阵的混合运算（小矩阵会自动上传）

        * 不允许：gpuArray 与大型 CPU 数组的直接运算

    3. 数据传输

        ```matlab
        % CPU → GPU
        cpuData = rand(100);
        gpuData = gpuArray(cpuData);

        % GPU → CPU（显式操作）
        resultGPU = gpuData * 2;
        resultCPU = gather(resultGPU);  % 关键函数：将数据取回 CPU

        % 错误示例
        % cpuResult = double(gpuData);  % 不会自动传输数据！
        ```

    4. 常用函数支持

        数学运算

        ```matlab
        abs, sin, exp, log, sqrt  % 逐元素运算
        sum, mean, max, min       % 规约操作（支持维度参数）
        fft, ifft, filter         % 信号处理
        ```

        线性代数

        ```matlab
        mtimes (*)    % 矩阵乘法
        inv, pinv     % 矩阵求逆
        eig, svd      % 特征值/奇异值分解（部分支持）
        ```

        数组操作

        ```matlab
        reshape, permute, transpose
        cat, repmat, bsxfun
        ```

    5. 编程技巧

        预分配 GPU 内存

        ```matlab
        gpuArray.zeros(1000, 'single');  % 避免多次内存分配
        ```

        使用 arrayfun 自定义函数

        ```matlab
        % 定义标量函数
        func = @(x, y) x.^2 + sin(y);

        % 应用于 gpuArray
        gpuX = gpuArray(rand(1000));
        gpuY = gpuArray(rand(1000));
        gpuZ = arrayfun(func, gpuX, gpuY);  % 在 GPU 上并行执行
        ```

        性能优化

        ```matlab
        % 1. 使用单精度（速度更快，显存占用减半）
        gpuSingle = rand(1000, 'single', 'gpuArray');

        % 2. 避免 CPU-GPU 频繁传输
        % 错误做法：在循环内反复调用 gather()
        % 正确做法：所有计算完成后一次性 gather()

        % 3. 检查内核执行时间
        tic; gpuResult = fft(gpuData); wait(gpuDevice); toc;
        % wait() 确保计时准确
        ```

    6. 限制与注意事项

        * 显存限制：数据大小不能超过 GPU 显存

        * 函数支持：并非所有 MATLAB 函数都支持 gpuArray（查看文档确认）

        * 精度差异：GPU 计算结果可能与 CPU 有微小差异

        * 数据类型：大多数操作需要相同类型的 gpuArray

    7. 简单示例

        ```matlab
        % 示例：矩阵乘法加速比较
        N = 2000;

        % CPU 计算
        A = rand(N);
        B = rand(N);
        tic; C_cpu = A * B; t_cpu = toc;

        % GPU 计算
        gpuA = gpuArray(single(A));  % 转为单精度
        gpuB = gpuArray(single(B));
        tic; gpuC = gpuA * gpuB; wait(gpuDevice); t_gpu = toc;

        % 取回结果并比较
        C_gpu = gather(gpuC);

        fprintf('CPU: %.3f s, GPU: %.3f s, 加速比: %.2f\n', ...
                t_cpu, t_gpu, t_cpu/t_gpu);
        ```

    重要提示

    * 使用前确保安装正确版本的 GPU 驱动和 CUDA Toolkit

    * 小规模计算可能因数据传输开销而变慢

    * 使用 gpuDevice 管理 GPU 内存（reset(gpuDevice) 可清空显存）

    通过合理使用 gpuArray，MATLAB 程序通常可获得数倍到数十倍的加速效果，特别适合大规模数值计算和矩阵运算。

* MATLAB pause() 函数

    1. 函数原型

        ```matlab
        pause                   % 暂停直到用户按任意键
        pause(n)                % 暂停 n 秒
        pause(state)            % 启用/禁用暂停功能
        oldState = pause(state) % 设置新状态并返回旧状态
        ```

    2. 主要作用

        * 程序暂停：暂停代码执行

        * 时间控制：实现固定时间延迟

        * 用户交互：等待用户操作（如按键）

        * 调试辅助：查看中间结果

    3. 基本用法

        (1) 暂停直到按键

        ```matlab
        disp('程序开始...');
        pause;  % 暂停，按任意键继续
        disp('继续执行');
        ```

        (2) 暂停指定时间

        ```matlab
        disp('开始处理');
        pause(2.5);  % 暂停2.5秒
        disp('2.5秒后继续');
        ```

    4. 高级用法

        (1) 控制暂停功能开关

        ```matlab
        % 禁用所有pause()调用
        pause('off');
        pause(5);  % 此pause()不会暂停！

        % 重新启用
        pause('on');

        % 查询当前状态
        currentState = pause('query');
        ```

        (2) 临时禁用pause()

        ```matlab
        % 保存旧状态，设置新状态
        oldState = pause('off');

        % 执行不需要暂停的代码
        for i = 1:100
            fprintf('迭代 %d\n', i);
        end

        % 恢复原状态
        pause(oldState);
        ```

    5. 使用场景示例

        示例1：动态显示进度

        ```matlab
        for i = 1:5
            fprintf('处理第 %d 项...\n', i);
            pause(1);  % 模拟处理时间
        end
        ```

        示例2：用户确认继续

        ```matlab
        disp('请检查数据是否正确');
        input('按回车键继续，Ctrl+C取消: ', 's');
        ```

        示例3：动画效果

        ```matlab
        for t = 0:0.1:2*pi
            plot(t, sin(t), 'ro');
            pause(0.05);  % 控制刷新率
        end
        ```

    6. 注意事项

        * 精度限制：实际暂停时间可能略长于指定时间，精度受系统负载影响

        * Ctrl+C中断：暂停期间可按 Ctrl+C 中断程序

        * 图形界面影响：暂停期间仍可操作图形界面

        * 替代函数：对于精确计时，考虑 drawnow 或 tic/toc

    7. 常见问题

        ```matlab
        % ❌ 错误：pause内不能有表达式
        pause(2+3);  % 正确！可以计算表达式
        pause 2;     % 错误！缺少括号

        % ✅ 正确用法
        pause(2 + 3);  % 暂停5秒
        pause(0.001);  % 毫秒级暂停
        ```

    8. 相关函数

        * waitforbuttonpress：等待鼠标或键盘事件

        * tic/toc：精确计时

        * drawnow：强制刷新图形而不暂停

* matlab `norm()`

    MATLAB norm() 函数

    原型

    ```matlab
    n = norm(X)
    n = norm(X, p)
    n = norm(X, p, 'fro')
    n = norm(X, 'fro')
    ```

    作用

    计算向量或矩阵的范数（测量大小/长度的数学概念）

    主要用法

    1. 向量范数

        ```matlab
        v = [1, 2, 3];

        % 2-范数（欧几里得长度，默认）
        n2 = norm(v)           % √(1² + 2² + 3²) = √14 ≈ 3.7417

        % 1-范数（绝对值之和）
        n1 = norm(v, 1)        % |1| + |2| + |3| = 6

        % 无穷范数（最大绝对值）
        ninf = norm(v, inf)    % max(|1|, |2|, |3|) = 3

        % p-范数
        p = 3;
        np = norm(v, p)        % (|1|^p + |2|^p + |3|^p)^(1/p)
        ```

    2. 矩阵范数

        ```matlab
        A = [1 2; 3 4];

        % 弗罗贝尼乌斯范数（默认）
        f = norm(A, 'fro')     % √(Σ|a_ij|²) = √30 ≈ 5.4772

        % 2-范数（最大奇异值）
        n2 = norm(A)           % 最大奇异值 ≈ 5.4650

        % 1-范数（最大列和）
        n1 = norm(A, 1)        % max(|1|+|3|, |2|+|4|) = 6

        % 无穷范数（最大行和）
        ninf = norm(A, inf)    % max(|1|+|2|, |3|+|4|) = 7
        ```

    常见范数总结表

    | 输入类型 | p值 | 含义 | 计算公式 |
    | - | - | - | - |
    | 向量 | 2（默认） | 欧几里得长度 | √(Σ|x_i|²) |
    | 向量 | 1 | 绝对值之和 | Σ|x_i| |
    | 向量 | inf | 最大绝对值 | max(|x_i|) |
    | 矩阵 | 2（默认） | 最大奇异值 | σ_max(A) |
    | 矩阵 | 1 | 最大列和 | max(Σ|a_ij|) |
    | 矩阵 | inf | 最大行和 | max(Σ|a_ij|) |
    | 矩阵 | 'fro' | 弗罗贝尼乌斯范数 | √(Σ|a_ij|²) |

    应用示例

    ```matlab
    % 计算误差范数
    x_true = [1; 2; 3];
    x_approx = [1.1; 1.9; 3.05];

    error = norm(x_true - x_approx)  % 误差的2-范数 ≈ 0.1581
    rel_error = norm(x_true - x_approx) / norm(x_true)  % 相对误差

    % 矩阵条件数估计
    A = hilb(5);  % 5阶希尔伯特矩阵
    cond_est = norm(A) * norm(inv(A))  % 条件数估计

    % 检查向量是否单位化
    v = randn(4,1);
    if abs(norm(v) - 1) < 1e-10
        disp('向量已单位化')
    end
    ```

    注意：norm() 对于复数输入同样有效，此时使用模长计算。对于稀疏矩阵，部分范数计算可能不支持。

* matlab 中，我看到有的函数内部使用的变量并不是在当前函数内部创建的，为什么？

    你观察到的现象在 MATLAB 中很常见，通常是由于以下几种情况：

    1. 嵌套函数与共享变量

        在嵌套函数中，内层函数可以直接访问外层函数的变量：

        ```matlab
        function outer()
            sharedVar = 10;  % 外层函数变量

            function inner()
                % 可以直接使用 sharedVar
                result = sharedVar * 2;
            end

            inner()
        end
        ```

    2. 函数句柄捕获工作区变量

        ```matlab
        function createHandle()
            baseValue = 5;

            % 匿名函数捕获了 baseValue
            myHandle = @(x) x + baseValue;

            % 即使 baseValue 不在当前工作区，myHandle 仍记得它
            result = myHandle(3);  % 返回 8
        end
        ```

    3. 持久变量 (persistent)

        ```matlab
        function counter()
            persistent count  % 跨函数调用保持值

            if isempty(count)
                count = 0;
            end

            count = count + 1;
            disp(count);
        end
        % 每次调用 counter()，count 都会递增
        ```

    4. 全局变量 (global)

        ```matlab
        % 文件1: setGlobal.m
        function setGlobal(val)
            global GLOBAL_VAR
            GLOBAL_VAR = val;
        end

        % 文件2: getGlobal.m
        function result = getGlobal()
            global GLOBAL_VAR  % 访问同一全局变量
            result = GLOBAL_VAR;
        end
        ```

    5. 隐式参数传递

        某些函数可能依赖 MATLAB 路径中的其他变量：

        ```matlab
        % config.m
        CONFIG_VALUE = 42;

        % myFunction.m
        function result = myFunction()
            % 如果 CONFIG_VALUE 在路径中已定义
            result = CONFIG_VALUE * 2;  % 使用外部定义的变量
        end
        ```

    6. 调试或开发时的临时情况

        在开发过程中，可能临时使用工作区变量进行测试：

        ```matlab
        function testFunc(x)
            % 如果在调用前，workspace 中有 testVariable
            y = x + testVariable;  % testVariable 来自基础工作区
            disp(y);
        end
        ```

    最佳实践建议：

    * 避免隐式依赖：明确传递所有需要的参数

    * 文档说明：如果必须使用非局部变量，请充分注释

    * 优先使用参数传递：

        ```matlab
        % 推荐方式
        function result = myFunc(x, config)
            result = x * config.value;
        end

        % 而不是
        function result = myFunc(x)
            global config  % 或使用其他隐式方式
            result = x * config.value;
        end
        ```

    检查变量来源的方法：

    ```matlab
    % 1. 使用 which 查看变量定义位置
    which variableName

    % 2. 使用 dbstack 查看调用栈
    dbstack

    % 3. 使用 exist 检查变量作用域
    exist('varName', 'var')
    ```

    总结：虽然 MATLAB 允许这些用法，但在正式代码中，为了可维护性和避免意外错误，建议尽量减少对非局部变量的依赖，明确通过参数传递所需数据。

* 简述 matlab 函数 diff() 的原型、作用及用法

    diff() 是 MATLAB 中用于计算差分和近似导数的核心函数。

    一、函数原型

    ```matlab
    % 基本语法
    Y = diff(X)
    Y = diff(X, n)
    Y = diff(X, n, dim)
    ```

    二、参数说明

    * X：输入数组（向量、矩阵或多维数组）

    * n（可选）：差分阶数，默认为 1

    * dim（可选）：沿指定维度操作，默认为第一个非单一维度

    三、主要作用

    1. 计算相邻元素的差值

        ```matlab
        x = [1, 3, 6, 10, 15];
        y = diff(x);  % 结果: [2, 3, 4, 5]
        ```

    2. 近似数值微分

        ```matlab
        t = 0:0.1:1;
        x = sin(t);
        dx_dt = diff(x) / 0.1;  % 近似计算导数
        ```

    3. 高阶差分

        ```matlab
        x = [1, 4, 9, 16];
        d1 = diff(x);      % 一阶差分: [3, 5, 7]
        d2 = diff(x, 2);   % 二阶差分: [2, 2] (等价于 diff(diff(x)))
        ```

    四、常用用法示例

    示例1：向量差分

    ```matlab
    A = [1 2 3 5 8];
    B = diff(A);  % 结果: [1 1 2 3]
    ```

    示例2：矩阵按行/列差分

    ```matlab
    M = [1 2 3; 4 5 6; 7 8 9];

    % 按列差分（默认）
    col_diff = diff(M);  % 结果: [3 3 3; 3 3 3]

    % 按行差分
    row_diff = diff(M, 1, 2);  % 结果: [1 1; 1 1; 1 1]
    ```

    示例3：与梯度计算对比

    ```matlab
    x = linspace(0, 2*pi, 100);
    y = sin(x);

    % diff 计算（长度减1）
    dy_diff = diff(y) / (x(2)-x(1));

    % gradient 计算（长度不变）
    dy_grad = gradient(y, x(2)-x(1));
    ```

    五、重要特性

    * 输出长度：diff(X) 输出的元素比 X 少 n 个

    * 数值微分精度：一阶前向差分，精度为 O(h)

    * 与梯度区别：diff 是前向差分，gradient 是中心差分（更精确）

    六、实际应用场景

    ```matlab
    % 1. 检测信号突变
    signal = [ones(1,50), 10, ones(1,50)];
    changes = find(abs(diff(signal)) > 5);  % 找到突变点位置

    % 2. 计算速度（位置差分）
    time = 0:0.1:10;
    position = sin(time);
    velocity = diff(position) / 0.1;

    % 3. 多阶差分消除趋势
    data = 0.1*(1:100) + randn(1,100);  % 线性趋势+噪声
    detrended = diff(data, 2);           % 二阶差分消除线性趋势
    ```

    七、注意事项

    * 边界处理：diff 会减少数据长度，需注意索引对应关系

    * 精度限制：对于高精度微分需求，建议使用 gradient 或样条插值

    * 高维数据：明确指定 dim 参数以避免意外行为

    diff() 是数值分析和信号处理中的基础工具，简洁高效但需理解其差分本质。

* sortrows()

    sortrows() 是 MATLAB 中用于按行排序的函数，特别适用于对矩阵、表格等数据进行排序。

    一、函数原型

    ```matlab
    B = sortrows(A)
    B = sortrows(A, column)
    B = sortrows(A, column, direction)
    B = sortrows(___, 'ComparisonMethod', method)
    [B, index] = sortrows(___)
    ```

    二、主要作用

    按行对数据进行排序，保持行的完整性。即整行数据作为一个整体进行排序。

    三、基本用法

    1. 默认排序（按第一列升序）

        ```matlab
        A = [3, 2, 1;
             1, 3, 2;
             2, 1, 3];

        B = sortrows(A)
        % 结果：
        %   1   3   2
        %   2   1   3
        %   3   2   1
        % 按第一列 [3;1;2] 升序排列
        ```

    2. 指定排序列

        ```matlab
        A = [3, 2, 1;
             1, 3, 2;
             2, 1, 3];

        % 按第二列排序
        B = sortrows(A, 2)
        % 结果：
        %   2   1   3  ← 第二列值最小（1）
        %   3   2   1  ← 第二列值次之（2）
        %   1   3   2  ← 第二列值最大（3）
        ```

    3. 多列排序

        ```matlab
        A = [3, 2, 1;
             1, 2, 3;
             2, 1, 3;
             1, 2, 1];

        % 先按第二列排序，再按第一列排序
        B = sortrows(A, [2, 1])
        % 结果：
        %   2   1   3  ← 第二列=1
        %   1   2   1  ← 第二列=2，第一列=1
        %   1   2   3  ← 第二列=2，第一列=1（与前一行第二列相同）
        %   3   2   1  ← 第二列=2，第一列=3
        ```

    4. 指定排序方向

        ```matlab
        A = [3, 2;
             1, 4;
             2, 1];

        % 按第一列降序
        B = sortrows(A, 1, 'descend')
        % 结果：
        %   3   2
        %   2   1
        %   1   4

        % 不同列不同方向
        B = sortrows(A, [1, 2], {'ascend', 'descend'})
        ```

    5. 获取排序索引

        ```matlab
        A = [3, 2, 1;
             1, 3, 2;
             2, 1, 3];

        [B, idx] = sortrows(A, 1)
        % idx = [2; 3; 1]  ← 原行号对应新位置
        ```

    6. 对表格（table）排序

        ```matlab
        % 创建表格
        T = table([3;1;2], {'A';'C';'B'}, 'VariableNames', {'Num', 'Char'});

        % 按 'Num' 列排序
        T_sorted = sortrows(T, 'Num')

        % 按多列排序
        T_sorted = sortrows(T, {'Char', 'Num'})
        ```

    7. 对元胞数组排序

        ```matlab
        C = {3, 'A';
             1, 'C';
             2, 'B'};

        % 按第一列数值排序
        C_sorted = sortrows(C, 1)
        ```

    四、关键特性

    * 稳定性：相等元素的相对顺序保持不变

    * 方向控制：每列可单独指定升序（'ascend'）或降序（'descend'）

    * 比较方法：可通过 'ComparisonMethod' 指定

        * 'auto'（默认）：自动选择

        * 'real'：按实部比较复数

        * 'abs'：按模值比较复数

    五、常见应用场景

    ```matlab
    % 1. 数据预处理：按时间排序
    data = [2023, 1, 15, 100;
            2023, 1, 10, 150;
            2023, 1, 12, 120];
    sorted_data = sortrows(data, [1, 2, 3]);  % 年、月、日排序

    % 2. 多条件排序
    students = {'张三', 85, 'A';
                '李四', 90, 'B';
                '王五', 85, 'A'};
    % 先按成绩降序，再按姓名升序
    sorted_students = sortrows(students, [2, 1], {'descend', 'ascend'});
    ```

    sortrows() 是 MATLAB 数据排序的常用工具，特别适合需要保持行数据完整性的排序需求。

* arrayfun()

    一、核心作用

    将函数应用于数组中的每个元素，实现向量化操作，避免显式使用循环。

    二、基本语法

    ```matlab
    B = arrayfun(func, A)
    B = arrayfun(func, A1, ..., An)
    [B1, ..., Bm] = arrayfun(func, ___)
    ```

    三、主要用途

    1. 替代循环，简化代码

        ```matlab
        % 传统循环方式
        A = 1:5;
        result = zeros(size(A));
        for i = 1:length(A)
            result(i) = A(i)^2 + sin(A(i));
        end

        % 使用 arrayfun
        result = arrayfun(@(x) x^2 + sin(x), A)
        ```

    2. 对数组每个元素应用复杂函数

        ```matlab
        A = [1, 2, 3; 4, 5, 6];

        % 对每个元素计算
        B = arrayfun(@(x) x^2 + 2*x + 1, A)
        % 等效于：[1^2+2*1+1, 4, 9; 25, 36, 49]
        ```

    3. 处理多个输入数组

        ```matlab
        A = [1, 2, 3];
        B = [4, 5, 6];

        % 对每对元素操作
        C = arrayfun(@(x, y) x*y + x + y, A, B)
        % C = [1*4+1+4=9, 2*5+2+5=17, 3*6+3+6=27]
        ```

    4. 返回多个输出

        ```matlab
        A = [1, 2, 3, 4];

        % 返回商和余数
        [quotients, remainders] = arrayfun(@(x) deal(fix(x/2), mod(x,2)), A)
        ```

    5. 处理非数值数据（结合'UniformOutput', false）

        ```matlab
        A = [1, 3, 2, 4];

        % 返回不同长度的输出（如重复字符串）
        B = arrayfun(@(n) repmat('*', 1, n), A, 'UniformOutput', false)
        % B = {'*', '***', '**', '****'}
        ```

    四、关键参数

    1. 'UniformOutput'

        * true（默认）：输出必须具有相同类型和大小

        * false：允许输出为元胞数组，可处理不同类型/大小的输出

        ```matlab
        % UniformOutput = true（默认）
        A = 1:4;
        B = arrayfun(@(x) x*2, A)  % 返回数组: [2, 4, 6, 8]

        % UniformOutput = false
        B = arrayfun(@(x) 1:x, A, 'UniformOutput', false)
        % B = {[1], [1,2], [1,2,3], [1,2,3,4]}
        ```

    2. 'ErrorHandler'

        指定错误处理函数

        ```matlab
        A = [-2, -1, 0, 1, 2];

        % 自定义错误处理
        B = arrayfun(@sqrt, A, 'ErrorHandler', @(err, x) NaN)
        % B = [NaN, NaN, 0, 1, 1.4142]
        ```

    五、与循环的对比

    | 特性 | arrayfun() | for 循环 |
    | - | - | - |
    | 代码简洁性 | 高 | 低 |
    | 可读性 | 函数式风格，直观 | 传统，详细 |
    | 性能 | 通常接近循环，有时更快 | 稳定 |
    | 调试难度 | 较高 | 较低 |
    | 灵活性 | 中等 | 高 |

    六、性能注意事项

    * 并非总是比循环快

        ```matlab
        % 对于简单操作，向量化通常更快
        A = 1:1e6;

        % 方法1: arrayfun
        tic
        B1 = arrayfun(@(x) x^2, A);
        toc

        % 方法2: 向量化（最快）
        tic
        B2 = A.^2;
        toc

        % 方法3: 循环
        tic
        B3 = zeros(size(A));
        for i = 1:length(A)
            B3(i) = A(i)^2;
        end
        toc
        ```

    * 适用场景：

        * 函数较复杂，难以向量化

        * 需要代码简洁性和可读性

        * 处理多个输入数组

    七、实际应用示例

    1. 批量处理文件

        ```matlab
        files = {'data1.mat', 'data2.mat', 'data3.mat'};
        % 批量加载文件
        data = arrayfun(@(f) load(f{1}), files, 'UniformOutput', false);
        ```

    2. 结构数组处理

        ```matlab
        students = struct('name', {'Alice', 'Bob', 'Charlie'}, ...
                          'score', {85, 92, 78});

        % 提取所有分数
        scores = arrayfun(@(s) s.score, students)
        ```

    3. 条件处理

        ```matlab
        A = [-3, 0, 2, -1, 4];

        % 对正数开方，负数返回NaN
        B = arrayfun(@(x) ifelse(x>0, sqrt(x), NaN), A)
        ```

    八、相关函数对比

    | 函数 | 作用 | 区别 |
    | - | - | - |
    | arrayfun | 对数组元素应用函数 | 通用，可处理多种情况 |
    | cellfun | 对元胞数组元素应用函数 | 专门用于元胞数组 |
    | structfun | 对结构体字段应用函数 | 专门用于结构体 |
    | spfun | 对稀疏矩阵应用函数 | 保持稀疏性 |

    总结

    arrayfun() 是 MATLAB 中实现函数式编程的重要工具，主要优势在于：

    * 代码简洁：一行替代多行循环

    * 可读性强：明确表达"对每个元素应用函数"

    * 灵活性高：支持多种输出格式和错误处理

    最佳实践：在简单向量化操作不可行，且代码可读性比极致性能更重要时使用。

* unique()

    一、核心作用

    找出数组中的唯一值（去除重复元素），并可返回相关索引信息。

    二、基本语法

    ```matlab
    C = unique(A)
    C = unique(A, setOrder)
    [C, ia, ic] = unique(___)
    [C, ia, ic] = unique(A, occurrence)
    ```

    三、主要功能

    1. 基本去重

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];

        C = unique(A)
        % 结果：C = [1, 2, 3, 4]
        % 升序排列，去除了重复的1,2,3
        ```

    2. 保持原顺序

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];

        C = unique(A, 'stable')
        % 结果：C = [3, 1, 2, 4]
        % 保持首次出现的顺序，不排序
        ```

    3. 返回相关索引

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];

        [C, ia, ic] = unique(A)
        % C = [1, 2, 3, 4]  (唯一值，升序)
        % ia = [2; 3; 1; 6] (C中元素在A中首次出现的位置)
        % ic = [3; 1; 2; 3; 1; 4; 2] (A中每个元素在C中的索引)

        % 验证：
        % A(ia) = [1; 2; 3; 4] 等于 C
        % C(ic) = [3; 1; 2; 3; 1; 4; 2] 等于 A
        ```

    4. 处理行唯一性

        ```matlab
        A = [1, 2, 3;
             2, 3, 4;
             1, 2, 3;  % 重复行
             3, 4, 5];

        % 默认按列处理
        C = unique(A)  % 返回所有唯一元素：[1;2;3;4;5]

        % 按行处理
        C = unique(A, 'rows')
        % 结果：
        %   1   2   3
        %   2   3   4
        %   3   4   5
        % 去除了重复的第三行
        ```

    四、参数详解

    1. setOrder 排序顺序

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];

        % 'sorted' - 默认，升序排列
        C1 = unique(A, 'sorted')      % [1, 2, 3, 4]

        % 'stable' - 保持原顺序
        C2 = unique(A, 'stable')      % [3, 1, 2, 4]
        ```

    2. occurrence 出现选项

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];

        % 'first' - 默认，返回首次出现的索引
        [C, ia] = unique(A, 'stable', 'first')
        % ia = [1; 2; 3; 6] (对应值3,1,2,4首次出现的位置)

        % 'last' - 返回最后出现的索引
        [C, ia] = unique(A, 'stable', 'last')
        % ia = [4; 7; 3; 6] (对应值3,1,2,4最后一次出现的位置)
        ```

    3. 'legacy' 兼容模式

        ```matlab
        % 用于保持与R2012b之前版本的兼容性
        C = unique(A, 'legacy')
        ```

    五、支持的数据类型

    1. 数值数组

        ```matlab
        A = [1.2, 3.4, 1.2, 5.6];
        C = unique(A)  % [1.2, 3.4, 5.6]
        ```

    2. 字符数组和字符串

        ```matlab
        % 字符数组
        str = ['a', 'b', 'a', 'c', 'b'];
        C = unique(str)  % 'abc'

        % 字符串数组
        strArray = ["apple", "banana", "apple", "cherry"];
        C = unique(strArray)  % ["apple", "banana", "cherry"]
        ```

    3. 元胞数组

        ```matlab
        C = {'apple', 'banana', 'apple', 'cherry'};
        U = unique(C)  % {'apple', 'banana', 'cherry'}
        ```

    4. 分类数组

        ```matlab
        categories = categorical({'small', 'large', 'medium', 'small'});
        C = unique(categories)  % [large, medium, small]
        ```

    5. 表格

        ```matlab
        T = table([1;2;1;3], {'A';'B';'A';'C'}, 'VariableNames', {'ID', 'Name'});
        T_unique = unique(T)  % 去除重复行
        ```

    六、高级用法

    1. 多输出应用 - 重建原始数组

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];
        [C, ia, ic] = unique(A, 'stable');

        % 用索引重建A
        A_reconstructed = C(ic)  % 等于原数组A
        ```

    2. 统计元素出现次数

        ```matlab
        A = [3, 1, 2, 3, 1, 4, 2];
        [C, ~, ic] = unique(A);

        % 统计每个唯一值的出现次数
        counts = accumarray(ic, 1);
        % 结果：counts = [2; 2; 2; 1] 对应值1,2,3,4

        % 显示统计结果
        for i = 1:length(C)
            fprintf('值 %d 出现了 %d 次\n', C(i), counts(i));
        end
        ```

    3. 分组操作

        ```matlab
        % 根据分组计算统计量
        data = [10, 20, 15, 25, 30];
        groups = [1, 2, 1, 2, 1];

        [uniqueGroups, ~, groupIdx] = unique(groups);

        % 计算每组的平均值
        for i = 1:length(uniqueGroups)
            groupMean = mean(data(groupIdx == i));
            fprintf('组 %d 的平均值: %.2f\n', uniqueGroups(i), groupMean);
        end
        ```

    4. 处理NaN值

        ```matlab
        A = [1, NaN, 2, NaN, 3, 1];

        % unique默认将每个NaN视为唯一值
        C = unique(A)
        % 结果：[1, 2, 3, NaN, NaN] 
        % 注意：不同的NaN被视为不同的值
        ```

    七、性能考虑

    1. 大数据量处理

        ```matlab
        % 对于大型数组，unique可能消耗较多内存
        A = rand(1e6, 1);  % 100万个随机数

        tic
        C = unique(A);
        toc
        ```

    2. 与sort()的关系

    unique() 内部使用了排序算法，因此：

    * 时间复杂度：O(n log n)

    * 默认'sorted'输出已排序的唯一值

    * 'stable'选项不使用排序，保持首次出现顺序

    八、实际应用场景

    1. 数据清洗

        ```matlab
        % 去除重复数据点
        data = [1.2, 2.3, 1.2, 3.4, 2.3, 4.5];
        clean_data = unique(data);
        ```

    2. 类别提取

        ```matlab
        % 从数据中提取所有类别
        labels = {'cat', 'dog', 'cat', 'bird', 'dog', 'fish'};
        categories = unique(labels);
        ```

    3. 查找公共元素

        ```matlab
        % 使用intersect()更合适，但可配合unique使用
        A = [1, 2, 3, 4, 5];
        B = [3, 4, 5, 6, 7];

        % 方法：先合并再找唯一值
        common = unique([A(ismember(A, B)), B(ismember(B, A))]);
        ```

    4. 生成索引映射

        ```matlab
        % 为分类数据创建索引映射
        colors = {'red', 'blue', 'green', 'blue', 'red', 'yellow'};
        [uniqueColors, ~, colorIdx] = unique(colors);
        % colorIdx可用于机器学习中的分类编码
        ```

    九、相关函数对比

    | 函数 | 作用 | 与unique()的区别 |
    | - | - | - |
    | unique() | 找唯一值 | 基本功能 |
    | ismember() | 检查元素是否在集合中 | 可配合使用 |
    | intersect() | 找交集 | 找多个集合的共同元素 |
    | union() | 找并集 | 合并多个集合并去重 |
    | setdiff() | 找差集 | 找一个集合有而另一个没有的元素 |
    | histcounts() | 统计频数 | 提供更详细的统计信息 |

    总结

    unique() 是 MATLAB 中数据预处理和集合操作的核心函数，主要特点：

    * 核心功能：高效去除重复元素，提取唯一值

    * 灵活性：支持多种排序方式和索引输出

    * 通用性：适用于多种数据类型（数值、字符、元胞、表格等）

    * 实用性：返回的索引可用于重建原始数据、分组统计等操作

    最佳实践：在需要数据去重、类别提取、创建索引映射等场景时优先考虑使用 unique()。

* matlab 尝试不打开窗口

    `matlab -nodesktop`会启动一个 command line window

    `matlab -nodisplay`无效，仍会启动 matlab 整体。

    * `matlab -batch "disp('Hello World')"`

        有效。直接在 terminal 里输出

        ```
        Hello World
        ```

    * `matlab -batch "your_script"`: 未测试

    * `matlab -r "run('myscript.m')"`: 未测试

    * `matlab -nodisplay -nosplash -r "myscript; exit"`: 未测试

    * Jupyter Notebook 支持

        通过 imatlab 内核在 Jupyter 中使用：

        ```bash
        pip install imatlab
        python -m imatlab install
        ```

        未测试

    * MATLAB Engine API

        未测试

        从 Python 调用 MATLAB：

        ```python
        import matlab.engine
        eng = matlab.engine.start_matlab()
        result = eng.sqrt(4.0)
        ```

        常用参数说明：

        * -nodesktop：无桌面界面

        * -nodisplay：无图形显示

        * -nosplash：不显示启动画面

        * -batch "command"：执行命令后退出

        * -r "command"：执行命令（旧版方式）

* MATLAB Engine API

    安装步骤：

    1. 确保 MATLAB 已安装

        首先确认你的系统已经安装了 MATLAB。

    2. 找到 MATLAB 的安装位置

        ```bash
        # Linux/Mac
        which matlab

        # Windows
        where matlab
        ```

    3. 从 MATLAB 安装 Python 引擎

        * 方法 A：在 MATLAB 中安装

            启动 MATLAB，然后运行：

            ```matlab
            cd (fullfile(matlabroot,'extern','engines','python'))
            system('python setup.py install')
            ```

        * 方法 B：在命令行中安装

            ```bash
            # 切换到 MATLAB 的引擎目录
            cd "matlabroot/extern/engines/python"

            # 安装到当前 Python 环境
            python setup.py install

            # 或者安装到特定用户
            python setup.py install --user

            # 或者使用 pip
            pip install -e .
            ```

        * 方法 C：使用 pip 直接安装（推荐）

            ```bash

            # 找到 MATLAB 安装路径后
            cd "C:\Program Files\MATLAB\R2023a\extern\engines\python"
            python -m pip install .

            # 或指定具体版本
            python setup.py build --build-base="builddir" install
            ```

    4. 验证安装

        ```bash
        python -c "import matlab.engine; print('Success!')"
        ```

    5. 常见问题解决

        * 问题 1：Python 版本不匹配

            MATLAB Engine 需要与 MATLAB 兼容的 Python 版本。检查：

            ```matlab
            % 在 MATLAB 中查看支持的 Python 版本
            pyversion
            ```

        * 问题 2：权限问题

            ```bash
            # 使用管理员/root权限
            sudo python setup.py install

            # 或安装到用户目录
            python setup.py install --user
            ```

        * 问题 3：多个 Python 环境

            ```bash
            # 确认使用的是正确的 Python
            which python
            python --version

            # 使用绝对路径
            /usr/bin/python3 setup.py install

            # 或创建虚拟环境
            python -m venv matlab-env
            source matlab-env/bin/activate
            python setup.py install
            ```

        * 问题 4：Windows 特定问题

            ```cmd
            # 以管理员身份运行命令提示符
            cd "C:\Program Files\MATLAB\R2023a\extern\engines\python"
            python setup.py install

            # 如果系统有多个 Python，使用完整路径
            "C:\Python39\python.exe" setup.py install
            ```

    6. 快速测试

        安装成功后，测试基本功能：

        ```python
        import matlab.engine

        # 启动引擎
        eng = matlab.engine.start_matlab()

        # 执行 MATLAB 命令
        result = eng.sqrt(4.0)
        print(f"sqrt(4) = {result}")

        # 创建 MATLAB 数组
        a = matlab.double([1, 2, 3, 4, 5])
        print(f"MATLAB array: {a}")

        # 调用 MATLAB 函数
        b = eng.sum(a)
        print(f"sum: {b}")

        # 关闭引擎
        eng.quit()
        ```

    替代方案：

    ```py
    # 替代方案：通过子进程调用 MATLAB
    import subprocess
    import json

    # 创建 MATLAB 脚本
    script = """
    data = rand(3, 3);
    disp(jsonencode(data))
    """

    # 执行
    result = subprocess.run(['matlab', '-batch', script], 
                           capture_output=True, text=True)
    ```

    注意：MATLAB Engine API 需要 MATLAB 许可证，并且 MATLAB 版本和 Python 版本必须兼容。

* matlab find()

    find() 函数是 MATLAB 中用于查找数组非零元素位置的核心函数。

    函数原型

    基本语法

    ```matlab
    ind = find(X)              % 查找所有非零元素
    ind = find(X, k)           % 查找前k个非零元素
    ind = find(X, k, 'first')  % 查找前k个（默认）
    ind = find(X, k, 'last')   % 查找最后k个
    ```

    多输出语法

    ```matlab
    [row, col] = find(X, ...)        % 二维数组的行列索引
    [row, col, v] = find(X, ...)     % 同时返回值
    [I1, I2, ..., In] = find(X, ...) % 多维数组的n维索引
    ```

    主要作用

    * 定位非零元素：找出数组中所有非零元素的索引位置

    * 条件查找：结合逻辑表达式，查找满足特定条件的元素

    * 稀疏索引：获取需要处理的数据位置，避免循环

    用法示例

    1. 基本查找

        ```matlab
        A = [0 2 3; 0 5 0; 7 0 9];
        ind = find(A)  % 返回所有非零元素的线性索引： [2; 4; 5; 7; 9]
        ```

        注：

        1. `A`不是逻辑矩阵也支持查找非零元素索引

    2. 使用条件表达式

        ```matlab
        A = [1 2 3; 4 5 6; 7 8 9];
        ind = find(A > 5)       % 大于5的元素位置： [6; 7; 8; 9]
        ind = find(mod(A,2)==0) % 偶数的位置： [2; 4; 6; 8]
        ```

        注：

        1. matlab 中不使用`%`取模，因为`%`是注释

    3. 获取前/后k个结果

        ```matlab
        A = [0 3 0 1 4 0 2];
        ind = find(A, 3)          % 前3个非零： [2, 4, 5]
        ind = find(A, 2, 'last')  % 最后2个非零： [5, 7]
        ```

        注：

        1. 搜索顺序按第 1 列，第 2 列，……

            python numpy 里是按行的，即 flatten 后的排列顺序是第 1 行，第 2 行，……

    4. 获取行列索引（适用于矩阵）

        ```matlab
        A = [0 2 0; 4 0 6; 0 0 9];
        [row, col] = find(A)
        % row = [2; 1; 2; 3]
        % col = [1; 2; 3; 3]
        ```

    5. 实际应用示例

        ```matlab
        % 示例1：查找峰值位置
        x = [1 2 5 3 4 2];
        peak_idx = find(x == max(x));  % 找到最大值位置：3

        % 示例2：数据筛选
        data = [10, 25, 3, 48, 15];
        valid_idx = find(data >= 10 & data <= 30);  % [1, 2, 5]

        % 示例3：稀疏矩阵处理
        [row, col] = find(A > threshold);
        ```

    重要特性

    * 逻辑输入：当输入是逻辑数组时，find 返回 true 值的位置

    * 空结果：如果没有找到符合条件的元素，返回空数组 []

    * 线性索引：单输出时返回线性索引（按列优先顺序）

    * 性能：通常比循环快，特别是在大数据集上

    替代方案

    对于逻辑索引，有时直接使用逻辑索引比 find 更高效：

    ```matlab
    % 使用 find
    idx = find(A > 0.5);
    B = A(idx);

    % 直接逻辑索引（更简洁高效）
    B = A(A > 0.5);
    ```

    find() 在数据处理、条件筛选和矩阵操作中非常实用，是 MATLAB 编程中的常用函数之一。

* matlab 中，函数可以通过`nargout`判断返回值的个数

    example:

    ```matlab
    function [row, col, val] = myFind(X)
        % 函数内部可以根据 nargout 判断调用方式
        switch nargout
            case 0
                error('至少需要一个输出参数');
            case 1
                % 调用者只请求一个输出：find(X)
                row = computeLinearIndices(X);
            case 2
                % 调用者请求两个输出：[row, col] = find(X)
                [row, col] = computeRowColIndices(X);
            case 3
                % 调用者请求三个输出：[row, col, val] = find(X)
                [row, col, val] = computeRowColVal(X);
        end
    end
    ```

* matlab 中，`disp()`的原型是什么？有多个输入时，中间是否有空格间隔？

* matlab `mean()`

    原型

    ```matlab
    M = mean(A)
    M = mean(A, dim)
    M = mean(___, nanflag)
    M = mean(___, outtype)
    ```

    * dim：指定维度（1-按列，2-按行，'all'-所有元素）

    * nanflag：'omitnan'忽略NaN，'includenan'包含NaN（默认）

    * outtype：输出数据类型（'default', 'double', 'native'）

    作用

    计算数组元素的平均值

    用法示例

    ```matlab
    % 基本用法
    A = [1 2 3; 4 5 6];
    mean_val = mean(A)         % 返回 [2.5 3.5 4.5]，默认按列计算
    mean_row = mean(A, 2)      % 按行计算，返回 [2; 5]
    mean_all = mean(A, 'all')  % 计算所有元素的平均值，返回 3.5

    % 处理NaN值
    B = [1 2 NaN; 4 5 6];
    mean_with_nan = mean(B, 'omitnan')  % 忽略NaN计算
    mean_with_nan = mean(B, 1, 'omitnan') % 按列忽略NaN

    % 指定输出类型
    C = single([1 2 3]);
    mean_double = mean(C, 'native')     % 返回single类型
    mean_default = mean(C)              % 返回double类型
    ```

* matlab `std()`

    原型

    ```matlab
    S = std(A)
    S = std(A, w)
    S = std(A, w, dim)
    S = std(___, nanflag)
    ```

    * w：权重（0-使用N-1，1-使用N，或指定权重向量）

    * dim：指定计算维度

    * nanflag：同上

    作用

    计算数组元素的标准差

    用法示例

    ```matlab
    % 基本用法
    A = [1 2 3; 4 5 6];
    std_val = std(A)           % 默认按列计算，返回 [2.1213 2.1213 2.1213]
    std_row = std(A, 0, 2)     % 按行计算，w=0表示使用N-1标准化

    % 权重参数w
    B = [1 2 3 4];
    std_n = std(B, 1)          % w=1，使用N标准化（总体标准差）
    std_n_1 = std(B, 0)        % w=0，使用N-1标准化（样本标准差，默认）

    % 处理NaN值
    C = [1 2 NaN 4 5];
    std_omit = std(C, 'omitnan')  % 忽略NaN计算
    std_includenan = std(C)       % 默认包含NaN（结果为NaN）

    % 多维数组
    D = randn(3, 4, 5);
    std_dim3 = std(D, 0, 3)    % 沿第3维计算标准差
    ```

    example:

    ```matlab
    % 数据分析示例
    data = randn(100, 3);  % 100个样本，3个特征

    % 计算均值和标准差
    data_mean = mean(data);      % 每个特征的平均值
    data_std = std(data);        % 每个特征的标准差

    % 数据标准化（z-score标准化）
    data_normalized = (data - data_mean) ./ data_std;

    % 验证标准化结果
    norm_mean = mean(data_normalized)    % 应接近 [0 0 0]
    norm_std = std(data_normalized)      % 应接近 [1 1 1]
    ```

* matlab 查找函数对应的文件

    `which functionName`

    exist 命令（检查是否存在）
    
    ```matlab
    % 检查函数是否存在并返回类型
    exist('functionName', 'builtin')  % 内建函数
    exist('functionName', 'file')     % 文件或函数
    exist('functionName', 'var')      % 变量
    ```

    `pathtool`: 打开搜索路径设置

    光标停在函数上按 f1: 打开函数帮助文档

    查找文件命令:

    ```matlab
    % 在搜索路径中查找包含特定字符串的文件
    lookfor keyword

    % 在所有.m文件中搜索文本
    grep('pattern', '*.m')
    ```

    dbtype 查看函数内容:

    ```matlab
    % 查看函数内容（如果知道位置）
    dbtype functionName
    ```

    编辑器中的跳转

    在MATLAB编辑器中：

    * F12：跳转到函数定义

    * Ctrl+单击：点击函数名跳转

    * 右键 → 打开 functionName：打开函数文件

* matlab 中，`fprintf()`既可以输出到文件，也可以输出到屏幕

    如果第一个参数是 file hander，那么输出到文件。

* matlab 中 & 和 && 有什么不同

    在 MATLAB 中，& 和 && 都是逻辑运算符，但它们有重要的区别：

    主要区别总结

    | 特性 | & (按位与) | && (短路与) |
    | - | - | - |
    | 运算对象 | 数组（逐元素运算） | 标量（单个布尔值） |
    | 短路求值 | ❌ 无，总是计算所有操作数 | ✅ 有，第一个为假就停止 |
    | 输入维度 | 必须相同或可广播 | 必须是标量 |
    | 执行速度 | 较慢（计算所有元素） | 较快（可能提前终止） |
    | 主要用途 | 数组/矩阵的逻辑运算 | 条件判断和控制流 |

    详细说明

    1. & - 逐元素逻辑与

        ```matlab
        % 示例 1：数组运算
        A = [true, false, true];
        B = [true, true, false];
        result = A & B  % 输出: [true, false, false]

        % 示例 2：矩阵运算
        M1 = [1 0; 0 1] > 0;  % [true false; false true]
        M2 = [1 1; 0 0] > 0;  % [true true; false false]
        M1 & M2  % 输出: [true false; false false]
        ```

    2. && - 短路逻辑与

        ```matlab
        % 示例 1：短路行为
        x = 0;
        y = 10;

        % 使用 &&：第一个为假，第二个不计算
        if (x > 5) && (y/x > 2)  % 安全，不会除零
            disp('条件成立');
        else
            disp('条件不成立');  % 输出这个
        end

        % 如果使用 & 会报错
        if (x > 5) & (y/x > 2)  % 错误！除以0
            % ...
        end
        ```

    更多对比示例

    示例 1：性能差异

    ```matlab
    % 假设有一个计算量很大的函数
    function result = expensiveComputation()
        disp('执行昂贵计算...');
        pause(1);  % 模拟耗时操作
        result = false;
    end

    % 使用 && - 更快（短路）
    condition1 = false;
    if condition1 && expensiveComputation()
        % 不会执行expensiveComputation()
    end

    % 使用 & - 更慢
    if condition1 & expensiveComputation()
        % 会执行expensiveComputation()，即使不需要
    end
    ```

    示例 2：维度要求

    ```matlab
    % & 可以处理数组
    A = [1 2 3];
    B = [0 2 4];
    (A > 1) & (B < 3)  % 有效：逐元素比较

    % && 只能处理标量
    a = 2;
    b = 3;
    (a > 1) && (b < 5)  % 有效：都是标量

    % 这会报错
    (A > 1) && (B < 3)  % 错误：操作数必须是标量

    示例 3：在循环和条件中的典型用法

    ```matlab
    % 正确使用 && 在if语句中
    value = 10;
    threshold = 5;

    if ~isempty(value) && value > threshold
        disp('值有效且大于阈值');
    end

    % 正确使用 & 在数组操作中
    data = [1, 5, 8, 3, 9];
    indices = (data > 2) & (data < 8);  % 找出 2 < data < 8 的元素
    % indices = [false, true, false, true, false]
    filtered = data(indices);  % [5, 3]
    ```

    选择指南

    使用 && 当：

    * 在 if、while 等条件语句中

    * 操作数是标量

    * 需要短路求值来提高性能或避免错误

    * 第二个表达式可能有副作用或计算成本高

    使用 & 当：

    * 对数组或矩阵进行逐元素逻辑运算

    * 需要创建逻辑索引数组

    * 操作数不是标量

    * 总是需要计算所有操作数

    重要注意事项

    * 优先级：& 的优先级比 && 高

    * 非逻辑输入：对于非逻辑输入，两者都会进行隐式转换（非零为true）

    * | 和 ||：同样的区别也适用于或运算符

        ```matlab
        % 优先级示例
        a = 1; b = 0; c = 1;
        result1 = a & b || c   % 相当于 (a & b) || c
        result2 = a && b | c   % 相当于 a && (b | c)
        ```

    记住这个简单的经验法则：

    * 条件语句（if/while） → 用 && 或 ||

    * 数组/矩阵运算 → 用 & 或 |

* `format compact`

    减少命令行窗口中的空行输出。

    使用效果对比：
    
    默认模式（未设置）：

    ```matlab
    % 默认情况下：
    A = [1 2; 3 4];
    B = [5 6; 7 8];
    ```

    输出：

    ```text
    A =
         1     2
         3     4

    B =
         5     6
         7     8
    ```

    注意：每个变量输出后都有额外的空行

    紧凑模式：

    ```matlab
    format compact  % 设置紧凑显示
    A = [1 2; 3 4];
    B = [5 6; 7 8];
    ```

    输出：

    ```text
    A =
         1     2
         3     4
    B =
         5     6
         7     8
    ```

    相关命令：

    ```matlab
    format compact     % 启用紧凑显示
    format loose       % 恢复默认显示（带空行）
    format short       % 短格式（默认，小数点后4位）
    format long        % 长格式（更多小数位）
    format bank        % 货币格式（两位小数）
    format rat         % 有理分数格式
    format short e     % 短科学计数法
    ```

    注意事项：

    * format compact 设置是会话级别的，即设置后一直有效，直到 MATLAB 关闭或更改设置

* matlab `length()`

    获取数组的最大维度长度。

    与其他函数对比：

    * size(A)：返回各维度尺寸（如 [m, n]）

    * numel(A)：返回元素总数

    * length(A)：返回最大维度长度

    注意事项：

    ```matlab
    % 特殊情况
    empty = [];
    len = length(empty);  % 返回 0

    % 标量被视为 1×1 数组
    scalar = 42;
    len = length(scalar);  % 返回 1
    ```

    建议：

    * 当需要特定维度长度时，使用 size(A, dim)

    * 当需要元素总数时，使用 numel(A)

    * length() 主要用于快速获取数组最大维度的长度，特别是在不确定数组维度但需要循环遍历时

* matlab 命令行

    * 通过 `matlab -nodisplay -nosplash -r "command"` 在终端中运行命令（适用于 Linux/macOS）

        windows 下尝试了一下，这个会打开新的 matlab 窗口，并在 command line 窗口中执行。

    * `octave --eval "command"` 在终端中执行单行命令。

* matlab table

    table 类型是MATLAB中用于存储异构数据的表格型数据结构，类似于电子表格或数据库表。

    1. 创建table

        ```matlab
        % 从变量创建
        names = {'Alice'; 'Bob'; 'Charlie'};
        ages = [25; 30; 35];
        scores = [85.5; 92.0; 78.5];
        T = table(names, ages, scores, ...
                  'VariableNames', {'Name', 'Age', 'Score'});

        % 从文件导入
        T = readtable('data.csv');
        ```

        注：

        1. 为什么 names 使用的是`{}`，而其他的使用的是`[]`？

            可以写成`names = ['Alice'; 'Bob'; 'Charlie']`吗？

            2025/12/30/00: 不可以。报错如下：`Error using vertcat`, `Dimensions of arrays being concatenated are not consistent.`

        1. names 可以使用字符串数组，写成

            `names = ["Alice"; "Bob"; "Charlie"]`

    2. 访问数据

        ```matlab
        % 按列名访问
        T.Age        % 返回数值向量
        T.('Name')   % 返回元胞数组

        % 索引访问
        T(1:2, :)              % 前两行
        T(:, {'Name', 'Age'})  % 特定列
        T{1:2, 'Age'}          % 提取数值数据
        T{1, 'Name'}           % 单个元素
        ```

        注：

        1. `T.('Name')`返回的是列向量，并不一定都是元胞数组

            比如`T.("Age")`输出如下：

            ```
            >> a = T.("Age")

            a =

                25
                30
                35

            >> class(T.("Age"))

            ans =

                'double'
            ```

    3. 修改和添加

        ```matlab
        % 添加新列
        T.Gender = {'F'; 'M'; 'M'}';

        % 修改数据
        T.Age(2) = 31;

        % 重命名变量
        T.Properties.VariableNames = {'FullName', 'Years', 'Grade'};
        ```

        注：

        1. table 中只能用列向量添加新列

    4. 常用操作

        ```matlab
        % 筛选数据
        young = T(T.Age < 30, :);  % 年龄小于30的行

        % 排序
        sortedT = sortrows(T, 'Age');

        % 分组统计
        groupsummary(T, 'Gender', 'mean', 'Score')  % 按性别分组求平均分

        % 连接表格
        T2 = [T; newRows];       % 垂直连接
        T3 = [T, additionalCol]; % 水平连接
        ```

    5. 实用属性

        ```matlab
        T.Properties.VariableNames    % 查看列名
        T.Properties.Description = '学生数据表'  % 添加描述
        T.Properties.RowNames = {'S1','S2','S3'}  % 设置行名
        ```

        注：

        1. 列名返回的是 cell array

            ```
            >> T.Properties.VariableNames

            ans =

            1×4 cell array

            Columns 1 through 3

                {'Name'}    {'Age'}    {'Score'}

            Column 4

                {'Gender'}
            ```

    6. 类型转换

        ```matlab
        % table 转矩阵（数值列）
        M = T{:, {'Age', 'Score'}};

        % table 转元胞数组
        C = table2cell(T);

        % 转置
        T_transpose = rows2vars(T);
        ```

    特点总结：

    * 异构数据：每列可包含不同类型数据（数值、字符、逻辑等）

    * 列名访问：支持通过变量名直接访问列数据

    * 数据完整性：保持数据与元数据的关联

    * 集成分析：与统计、机器学习工具箱良好兼容

    table 类型特别适合处理结构化数据，是替代 dataset 类型的现代推荐方式。

    注：

    1. dataset 类型是 matlab 的旧类型，已被抛弃

    1. table 有点像 pandas DataFrame 的感觉

* matlab 垂直拼接字符串

    代码与报错如下：

    ```
    >> names = ['Alice'; 'Bob'; 'Charlie']
    Error using vertcat
    Dimensions of arrays being concatenated are
    not consistent.
    ```

    解释：

    这个错误是因为你使用分号 ; 进行垂直拼接时，三个字符串的长度不一致。

    在 MATLAB 中：

    * 使用方括号 [] 加 分号 ; 表示垂直拼接（vertcat）

    * 垂直拼接要求所有行具有相同的列数（字符串长度）

    * 'Alice' 有5个字符，'Bob' 有3个字符，'Charlie' 有7个字符

    解决方案：

    方案1：使用字符数组（要求填充空格使长度一致）

    ```matlab
    names = ['Alice  '; 'Bob    '; 'Charlie'];
    % 或者使用 char 函数自动填充
    names = char('Alice', 'Bob', 'Charlie');
    ```

    方案2：使用元胞数组（推荐，更灵活）

    ```matlab
    names = {'Alice'; 'Bob'; 'Charlie'};
    % 访问：names{1}, names{2}, names{3}
    ```

    方案3：使用字符串数组（MATLAB R2016b+）

    ```matlab
    names = ["Alice"; "Bob"; "Charlie"];
    % 访问：names(1), names(2), names(3)
    ```

    方案4：转置为水平拼接（如果不需要垂直排列）

    ```matlab
    names = ['Alice', 'Bob', 'Charlie'];  % 水平拼接
    ```

    推荐使用元胞数组或字符串数组，因为它们可以处理不同长度的字符串，而且更符合现代 MATLAB 编程习惯。

* matlab disp 技巧

    使用拼接的字符串显示稍微复杂些的内容。

    ```matlab
    % num2str() 转换数值
    x = 3.1415926;
    disp(['π的值是：', num2str(x, '%.4f')]);

    % 使用字符串数组（MATLAB R2016b+）
    name = "李四";
    age = 30;
    disp(name + "，" + age + "岁");
    ```

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

* `split()`

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
