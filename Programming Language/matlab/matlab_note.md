# Matlab Note

* matlab 2025b x64 破解版下载

    <https://www.puresys.net/8739.html>

## cache

* matlab 读写Excel

    ```matlab
    % 读取Excel
    [num, txt, raw] = xlsread('data.xlsx');      % 读取整个工作表
    [num, txt, raw] = xlsread('data.xlsx', 'Sheet2');  % 指定工作表
    [num, txt, raw] = xlsread('data.xlsx', 'A1:C10');  % 指定区域

    % 写入Excel
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

* matlab 常用函数速查表

    | 操作类型 | 读取函数 | 写入函数 | 适用格式 |
    | - | - | - | - |
    | 文本文件 | fscanf, textscan | fprintf | .txt, .dat |
    | 表格数据 | readtable | writetable | .csv, .txt, .xlsx |
    | 数值矩阵 | readmatrix | writematrix | .txt, .csv |
    | Excel | xlsread | xlswrite | .xlsx, .xls |
    | 二进制 | load | save | .mat |
    | 图像 | imread | imwrite | .jpg, .png, .tif |
    | 音频 | audioread | audiowrite | .wav, .mp3 |

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

* matlab 打印

    * disp() - 基本显示

        ```matlab
        disp('Hello World');          % 显示字符串
        disp(['x = ', num2str(5)]);   % 显示变量

        % 显示矩阵
        A = [1 2 3; 4 5 6];
        disp('矩阵A:');
        disp(A);
        ```

    * fprintf() - 格式化输出（最常用）

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

    * sprintf() - 格式化字符串（不直接显示）

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

    * 综合 example

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

* matlab 中，单引号`'`和双引号`"`都可以表示字符串。

* matlab 中，`disp('xxx')`和`disp('xxx');`效果相同，都没有`ans = xxx`的输出。

* matlab 中的`1:10`包含 1 和 10。即 end included

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

* matlab 数据可视化

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

### 命令与快捷键

* 文件与目录

    ```matlab
    pwd                     % 显示当前工作目录
    cd 'C:\MyFolder'        % 更改目录
    ls                      % 列出当前目录文件
    dir                     % 详细列表
    ```

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

### 内置函数

## note

`help func_name`

查找函数：`lookfor keyword`

`lookfor`搜索所有函数的第一行注释行，找到对应的文件或函数。

在脚本中，两个`%`可以标志一个 block：

```
a = 1

%% new block
b = 2
```

使用`Ctrl + Enter`可以运行 block 中的内容。

块注释：

```matlab
%{

    comments

%}
```

其他常用命令：

* `who`：列出当前工作空间中的变量
* `what`：列出当前文件夹或指定目录下的 M 文件、MAT 文件和 MEX 文件
* `which`：显示指定函数或文件的路径
* `whos`：列出当前工作空间中变量的更多信息
* `exist`：检查指定变量或文件的存在性
* `doc`：直接查询在线文档。通常更详细
* `echo`：直接运行时，切换是否显示 m 文件中的内容。也可以`echo on`，`echo off`来指定状态。

预置变量：

* `eps`：计算机的最小正数
* `pi`：圆周率 pi 的近似值 3.14159265358979
* `inf`或`Inf`：无穷大
* `NaN`：不定量
* `i, j`：虚数单位定义`i`
* `flops`：浮点运算次数，用于统计计算量

环境 -> 设置路径 可以为 Matlab 添加或删除搜索路径。

`clear`：删除工作区中的所有变量。

`clear var1, var2`：清除指定变量

`pack`：用于整理内存。将内存中的数据先存储到磁盘上，再从磁盘将数据读入到内存中。

数值的显示格式：

|格式命令|作用|
|-|-|
|`format short`|5 位有效数字|
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

常用的控制命令：

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

命令行中的快捷键：

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

运算符：除法为`\`或`/`，乘方为`^`。

显示运算符优先顺序：`help precedence`

输入矩阵：

```matlab
a = [1, 2, 3;, 4, 5, 6; 7, 8, 9]
a = [1 2 3 4
    5 6 7 8
    0 1 2 3]
```

还可以直接输入矩阵元素：

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

注释：使用`%`进行单行注释。

整数类型：

`int8()`, `int16()`, `int32()`, `int64()`, `uint8()`, `uint16()`, `uint32()`, `uint64()`

整数的溢出会变成最小值和最大值：

```matlab
k = cast('hellothere', 'uint8');  % k = 104 101 108 108 111 116 104 101 114 101

double(k) + 150;  % ans = 254 251 258 261 266 254 251 264 251

k + 150;  % ans = 254 251 255 255 255 255 254 251 255 251

k - 110;  % and = 0 0 0 0 1 6 0 0 4 0
```

浮点类型：

`single()`, `double()`

```matlab
a = zeros(1, 5, 'single')

d = cast(6:-1:0, 'single')  % 转换单精度与双精度
```

单精度与双精度浮点数之间的运算结果是单精度浮点数。

判断是否为`nan`或`inf`：`isnan()`, `isinf()`。在 matlab 中，不同的`NaN`互不相等。不可以使用`a == nan`判断。

找到为`1`或`true`的索引：`find()`

查看维度：`size()`，查看长度：`length()`（行数或列数的最大值），元素的总数：`numel()`

```matlab
i = find(isnan(a));
a(i) = zeros(size(i));  % changes NaN into zeros
```

测试一个数组是否为空数组：`isempty()`

复数：

```matlab
a = 1 + 2i;
complex(2, 4)
```

常见的与复数相关的函数：

```matlab
conj(c)  % 计算 c 的共轭复数
real(c)  % 返回复数 c 的实部
imag(c)  % 返回复数 c 的虚部
isreal(c)  % 如果数组中全是实数，则返回 1，否则返回 0
abs(c)  % 返回复数 c 的模
angle(c)  % 返回复数 c 的幅角
```

与数据类型相关的函数：

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

## 矩阵运算

**向量生成**

1. 直接创建

    ```matlab
    a = [1 2 3]  % shape: (1, 3)
    a = [1, 2, 3]  % shape: (1, 3)
    a = [1; 2; 3]  % shape: (3, 1)
    ```

    matlab 里没有纯量和一维向量，只有维度从二起始的矩阵。为了方便，这里把 shape 为`(1, n)`或`(n, 1)`的矩阵称为向量，把 shape 为`(1, 1)`的矩阵称为数字或纯量。把 shape 为`(m, n)`的二维矩阵或三维以上的矩阵称为矩阵或张量。

    可以用`size(a)`查看矩阵`a`的 shape。

1. `linspace`：线性向量生成

    `linspace(x1, x2)`默认生成 100 个点，也可以用`linspace(x1, x2, n)`指定生成的采样点数量。

    `logspace`：等比数列生成

1. `start:step:end`按间隔生成向量

    `start:end`按步长为 1 生成向量，比如`1:5`生成向量`[1 2 3 4 5]`，注意`end`是包括在区间内的。

    `start:step:end`可以按指定步长生成向量。

转置：`a'`或`transpose(a)`。这两种方法只适用二维数组（包含向量），如果维度超过二维，那么会报错。

可以使用`a([2, 3, 4])`进行多元素索引，也可以使用`a(1:2:10)`这样的方式。

常用的二维数组生成函数：

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

数组寻址：

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

使用`sub2ind()`计算二维索引在拉伸为一维的数组中的索引：

```matlab
b = sub2ind(size(a), [2, 3], [2, 1])
a(b)
```

排序：

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

数组检测：

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

数组结构：

* `length(A)`

    一个数组的行数和列数的最大值。

* `numel(A)`

    数组元素总数。

* `[a, b] = size(A)`：数组的行数和列数。

直接使用`a * b`做的是矩阵乘法，想做逐元素相乘可以用`a .* b`。

逻辑运算：

```matlab
A & B
and(A, B)  % 若两个数均非 0 值，则结果为 1

A | B
or(A, B)  % 若两个数有一个不为 0，则结果为 1

~A
not(A)  % 若待运算矩阵的元素为 0，则结果元素为 1

xor(A, B)  % 若一个为 0，一个不为 0，则结果为 1
```

数组常用的运算：

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

矩阵扩展：

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

高维数组：

```matlab
% 直接创建
A = zeros(2, 3)
% 使用索引创建
A(:, :, 4) = zeros(2, 3)
```

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

可以使用赋空操作删除某一行或列：

```matlab
A(:, 2:3, :) = []
```

## 字符串

字符串是一个`1 x n`的`char`型数组，每个字符占 2 字节。

```matlab
str = 'I am a great person ';  % 使用单引号直接创建

str = ['second'; 'string']  % 字符串数组，要求字符串长度必须一致

c = char('first', 'second')  % 使用 char 创建字符串数组时，如果字符串长度不同，char() 会自动在较短的字符串后加空格，使所有字符串长度相等
```

常用函数：

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

## 结构体（structures）

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

## 单元数组（cell array）

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

## 语句

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

## 函数

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

## 画图

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