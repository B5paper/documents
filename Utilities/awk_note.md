# awk note

## cache

* 详细讲讲 awk 的用法

    **awk 详解**

    awk 是 Linux/Unix 下强大的文本处理工具，适合按行、按列处理结构化文本。下面从基础到进阶系统讲解。

    **一、基本工作模式**

    awk 逐行读取输入，对每一行执行 `pattern { action }` 规则：

    ```bash
    awk 'pattern { action }' file
    ```

    - **pattern**：匹配条件（可省略，表示每行都执行）

    - **action**：要执行的操作（可省略，默认打印整行）

    ```bash
    awk '{ print }' file.txt          # 打印每行（类似 cat）
    awk '/error/ { print }' file.txt  # 打印含 error 的行
    awk '/error/' file.txt            # pattern 省略 action，默认打印
    ```

    **二、字段与内置变量**

    awk 按**分隔符**把每行拆成字段，用 `$1, $2, ...` 引用：

    | 变量 | 含义 |
    |------|------|
    | `$0` | 整行内容 |
    | `$1, $2...` | 第 1、2... 个字段 |
    | `$NF` | 最后一个字段 |
    | `NF` | 当前行的字段数 |
    | `NR` | 当前处理的行号（跨文件累计） |
    | `FNR` | 当前文件内的行号 |
    | `FS` | 输入字段分隔符（默认空格/Tab） |
    | `OFS` | 输出字段分隔符（默认空格） |
    | `RS` | 输入记录分隔符（默认换行） |
    | `ORS` | 输出记录分隔符（默认换行） |
    | `FILENAME` | 当前文件名 |

    ```bash
    awk '{ print $1, $3 }' file.txt        # 打印第1、3列
    awk '{ print $NF }' file.txt           # 打印最后一列
    awk '{ print NF }' file.txt            # 打印每行字段数
    awk '{ print NR": "$0 }' file.txt      # 带行号打印
    ```

    **三、分隔符设置**

    ```bash
    # -F 指定分隔符
    awk -F: '{ print $1 }' /etc/passwd
    awk -F'\t' '{ print $2 }' file.tsv
    awk -F'[,;]' '{ print $1 }' file.txt   # 支持正则作为分隔符

    # 用 FS 变量
    awk 'BEGIN{FS=":"} { print $1 }' /etc/passwd

    # 输出分隔符 OFS
    awk 'BEGIN{FS=":";OFS="-"}{ print $1,$3 }' /etc/passwd
    ```

    **四、BEGIN / END 块**

    ```bash
    awk 'BEGIN{ print "开始" } { print $1 } END{ print "结束" }' file.txt
    ```

    - `BEGIN`：处理输入前执行一次（常用于初始化、打印表头）

    - `END`：处理完所有输入后执行一次（常用于汇总统计）

    ```bash
    # 统计行数
    awk 'END{ print NR }' file.txt

    # 求和
    awk '{ sum += $1 } END{ print sum }' nums.txt

    # 求平均
    awk '{ sum += $1 } END{ print sum/NR }' nums.txt
    ```

    **五、条件与模式**

    ```bash
    # 数值比较
    awk '$3 > 100 { print $1, $3 }' file.txt

    # 字符串匹配
    awk '$1 == "root" { print }' /etc/passwd
    awk '$1 ~ /^ro/ { print }' /etc/passwd      # 正则匹配
    awk '$1 !~ /^ro/ { print }' /etc/passwd     # 不匹配

    # 逻辑运算
    awk '$3 > 50 && $3 < 100 { print }' file.txt
    awk '$1 == "a" || $1 == "b" { print }' file.txt

    # 范围模式
    awk '/START/,/END/ { print }' file.txt      # 打印两个标记之间的行

    # 行号条件
    awk 'NR==1 { print }' file.txt              # 第一行
    awk 'NR>1 { print }' file.txt               # 除第一行外
    ```

    注：

    1. `NR == 1`等号左右可以有空格，也可以没有。

    2. 范围模式，更像是选中多个匹配模式，不像是范围。没看明白这个。

    **六、变量与运算**

    awk 变量无需声明，默认为 0 或空字符串：

    ```bash
    awk '{ count++ } END{ print count }' file.txt
    awk '{ sum += $2 } END{ print sum }' file.txt
    awk '{ max = ($1 > max) ? $1 : max } END{ print max }' nums.txt
    ```

    支持算术 `+ - * / %`、比较、逻辑、三目运算符。

    **七、流程控制**

    ```bash
    # if-else
    awk '{ if ($1 > 60) print "及格"; else print "不及格" }' scores.txt

    # for 循环
    awk '{ for (i=1; i<=NF; i++) print $i }' file.txt

    # while
    awk '{ i=1; while (i<=NF) { print $i; i++ } }' file.txt

    # 数组遍历
    awk '{ count[$1]++ } END{ for (k in count) print k, count[k] }' file.txt

    # break / continue / next / exit
    awk '{ if ($1 == "skip") next; print }' file.txt   # next 跳过本行
    awk 'NR==5 { exit } { print }' file.txt            # exit 提前结束
    ```

    **八、数组**

    awk 的数组是关联数组（哈希表），是统计的利器：

    ```bash
    # 统计每个单词出现次数
    awk '{ for (i=1; i<=NF; i++) count[$i]++ } END{ for (w in count) print w, count[w] }' file.txt

    # 判断键是否存在
    awk '{ if ($1 in seen) print "重复:", $1; seen[$1]=1 }' file.txt

    # 删除数组元素
    awk '{ delete arr[$1] }' file.txt

    # 多维数组（用 SUBSEP 连接）
    awk '{ a[$1,$2]++ } END{ for (k in a) print k, a[k] }' file.txt
    ```

    **九、常用内置函数**

    **字符串函数：**

    | 函数 | 说明 |
    |------|------|
    | `length(s)` | 字符串长度 |
    | `substr(s,m,n)` | 从 m 取 n 个字符 |
    | `index(s,t)` | t 在 s 中的位置 |
    | `split(s,a,fs)` | 按 fs 分割 s 到数组 a |
    | `sub(r,t,s)` | 替换第一个匹配 |
    | `gsub(r,t,s)` | 替换所有匹配 |
    | `match(s,r)` | 返回匹配位置 |
    | `tolower(s)` / `toupper(s)` | 大小写转换 |
    | `sprintf(fmt,...)` | 格式化字符串 |

    ```bash
    awk '{ print length($0) }' file.txt
    awk '{ print substr($1,1,3) }' file.txt
    awk '{ gsub(/foo/,"bar"); print }' file.txt
    awk '{ n=split($0,arr,","); print n, arr[1] }' file.txt
    ```

    **数学函数：**

    `int()`, `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `rand()`, `srand()`

    ```bash
    awk '{ print int($1) }' nums.txt
    awk 'BEGIN{ srand(); print rand() }'
    ```

    **格式化输出 printf：**

    ```bash
    awk '{ printf "%-10s %5d\n", $1, $2 }' file.txt
    awk '{ printf "%.2f\n", $1 }' nums.txt
    ```

    格式符：`%s` 字符串、`%d` 整数、`%f` 浮点、`%x` 十六进制、`%c` 字符。

    **十、实战示例**

    ```bash
    # 1. 统计文件行数、字数、字符数
    awk '{ chars+=length($0)+1; words+=NF } END{ print NR, words, chars }' file.txt

    # 2. 打印 /etc/passwd 的用户名和 shell
    awk -F: '{ print $1, $7 }' /etc/passwd

    # 3. 求某列总和与平均
    awk '{ s+=$3 } END{ printf "sum=%d avg=%.2f\n", s, s/NR }' data.txt

    # 4. 按第一列分组求和
    awk '{ sum[$1]+=$2 } END{ for (k in sum) print k, sum[k] }' data.txt

    # 5. 找出重复行
    awk 'seen[$0]++' file.txt

    # 6. 去重（保留首次出现）
    awk '!seen[$0]++' file.txt

    # 7. 打印第 10-20 行
    awk 'NR>=10 && NR<=20' file.txt

    # 8. 交换前两列
    awk '{ t=$1; $1=$2; $2=t; print }' file.txt

    # 9. 统计日志中各 IP 的访问次数并排序
    awk '{ ip[$1]++ } END{ for (i in ip) print ip[i], i }' access.log | sort -rn

    # 10. 处理 CSV 某列求和
    awk -F, 'NR>1 { sum+=$3 } END{ print sum }' data.csv
    ```

    **十一、进阶技巧**

    **多个规则组合：**

    ```bash
    awk 'NR==1 { print "表头:", $0 } $3>100 { print "大值:", $1 }' file.txt
    ```

    **变量传递：**

    ```bash
    awk -v threshold=100 '$3 > threshold { print }' file.txt
    ```

    **多文件处理：**

    ```bash
    awk '{ print FILENAME, FNR, $0 }' a.txt b.txt
    ```

    **使用外部脚本文件：**

    ```bash
    # script.awk
    BEGIN { FS=":" }
    $3 > 100 { print $1 }
    END { print "done" }

    awk -f script.awk /etc/passwd
    ```

    **gawk 扩展（GNU awk 特有）：**

    ```bash
    awk 'BEGIN{ PROCINFO["sorted_in"]="@ind_str_asc"; } ...'  # 数组排序遍历
    awk '{ a[NR]=$0 } END{ ... }'                              # 排序文本
    ```

    **十二、awk vs 其他工具**

    | 场景 | 推荐 |
    |------|------|
    | 按列提取/简单过滤 | awk |
    | 复杂正则替换 | sed |
    | 结构化统计/汇总 | awk |
    | JSON 处理 | jq |

    **核心记忆点：**
    1. `pattern { action }` 是基本结构
    2. `$0` 整行，`$1..$NF` 字段，`NR/NF/FS/OFS` 常用变量
    3. `BEGIN/END` 做初始化和汇总
    4. 关联数组是统计核心
    5. 条件、循环、函数齐备，几乎是一门小语言

    掌握以上内容，日常文本处理基本够用。需要某个方向（如日志分析、性能优化）深入可以再展开。

* 一种提取路径`/path/to/target`路径中`target`的方法

    `awk -F'/' '{print $NF}' <<< "/tmp/dir/target"`

* 简述 awk 的用法

	`awk` 是一种强大的文本处理工具，适用于对结构化文本数据（如列式数据）进行提取、转换和统计操作。以下是其核心用法简述：

	1. 基本结构

    	```bash
    	awk '模式 {动作}' 文件名
    	```

    	- **模式**：可选，用于筛选行（如 `/正则/`、`条件表达式`）。
    	- **动作**：对匹配的行执行的操作（如 `print`、计算）。

	2. 常用内置变量

    	- `$0`：整行内容。
    	- `$1, $2...`：第1、2...列字段。
    	- `NF`：当前行的字段数。
    	- `NR`：当前行号。
    	- `FS`：输入字段分隔符（默认为空格/制表符）。
    	- `OFS`：输出字段分隔符（默认为空格）。

	3. 常见用法示例

    	1. 打印指定列

            ```bash
            awk '{print $1, $3}' file.txt      # 打印第1列和第3列
            ```

            注：

            1. 如果文件最后有一行空行，那么空行也会被输出

            2. 逗号不会被输出，只在 $1 和 $3 之间输出空格

    	2. 条件过滤

            ```bash
            awk '$2 > 100 {print $0}' data.txt  # 打印第 2 列大于 100 的行
            awk '/error/ {print NR, $0}' log.txt  # 打印包含 "error" 的行及其行号
            ```

	4. 高级功能

    	- **数组操作**：支持关联数组，用于计数、去重等。

            ```bash
            awk '{count[$1]++} END {for (k in count) print k, count[k]}' log.txt
            ```

    	- **字符串函数**：`length()`、`substr()`、`gsub()` 等。
    	- **自定义变量**：直接在动作中定义变量（如 `awk '{total=$2+$3; print total}'`）。

	5. 组合其他命令

    	```bash
    	ps aux | awk '$3 > 1.0 {print $2, $3}'   # 找出CPU使用率>1%的进程
    	```

	总结

	`awk` 的核心是通过**模式匹配**和**字段处理**高效操作文本，适用于日志分析、数据提取、报表生成等场景。结合正则表达式、变量和逻辑控制，可以解决复杂的文本处理需求。

* 'awk '$2 > 100 {print $0}' data.txt'
    如果文本中第二行有非数字的，awk会如何处理？

	当`awk '$2 > 100 {print $0}' data.txt`遇到第二列包含非数字内容时，**awk会将非数字字符串当作 0 来处理**。

	* 具体行为示例

        假设`data.txt`内容为：

        ```
        abc 150
        def twohundred
        ghi 50
        jkl 200
        mno "123"
        ```

        **执行结果：**
        ```
        abc 150
        jkl 200
        ```

	* 处理逻辑

        1. **`def twohundred`**：`$2 = "twohundred"` → 转换为数字 `0` → `0 > 100` 为假 → 不打印
        2. **`ghi 50`**：`$2 = "50"` → 转换为数字 `50` → `50 > 100` 为假 → 不打印  
        3. **`mno "123"`**：`$2 = "\"123\""` → 转换为数字 `0`（因为有引号）→ `0 > 100` 为假 → 不打印

    * 验证测试

        ```bash
        # 测试awk对非数字的转换
        echo "abc twohundred" | awk '{print $2, ($2+0)}'
        # 输出：twohundred 0

        echo "abc 123abc" | awk '{print $2, ($2+0)}'  
        # 输出：123abc 123  (部分转换)
        ```

	* 解决方案

        如果需要更严格的数值检查，可以：

        1. 明确检查是否为数字

            ```bash
            awk '$2+0 == $2 && $2 > 100 {print $0}' data.txt
            ```

        2. 使用正则表达式验证

            ```bash
            awk '$2 ~ /^[0-9]+$/ && $2 > 100 {print $0}' data.txt
            ```

        3. 处理非数字时给出警告

            ```bash
            awk '{
                if ($2+0 != $2) 
                    print "警告: 第" NR "行第2列不是纯数字: " $2 > "/dev/stderr"
                else if ($2 > 100) 
                    print $0
            }' data.txt
            ```

	* 总结

        - **非数字字符串在数值比较时会被当作0**
        - **混合字符串**（如"123abc"）会**部分转换**（提取开头的数字部分）
        - 在需要精确处理的场景中，建议**先验证字段是否为纯数字**

* awk 命令基本结构：
    
    `awk '模式 {动作}' 文件名`

    * 模式：可选，用于筛选行（如 `/正则/`、`条件表达式`）。

    * 动作：对匹配的行执行的操作（如 `print`、计算）。

* 常用内置变量:

    * `$0`：整行内容。

    * `$1, $2...`：第1、2...列字段。

    * `NF`：当前行的字段数。

    * `NR`：当前行号。

    * `FS`：输入字段分隔符（默认为空格/制表符）。

    * `OFS`：输出字段分隔符（默认为空格）。

## topics

### 常见用法

* 打印指定列

    `msg.txt`:

    ```
    hello world nihao zaijian
    haha hehe haihai huaihuai
    1 2 3 4
    ```

    ```bash
    awk '{print $1, $3}' msg.txt      # 打印第1列和第3列
    ```

    output:

    ```
    hello nihao
    haha haihai
    1 3
    ```

* 条件过滤

    `data.txt`:

    ```
    1 2 3
    4 5 6
    7 8 9
    10 11hehe 12
    ```

    ```bash
    awk '$2 >= 5 {print $0}' data.txt # 打印第 2 列大于等于 5 的行
    ```

    output:

    ```
    4 5 6
    7 8 9
    ```

    如果在比较时，发现有非数字项，那么会被当作`0`处理。所以`11hehe`没有被输出。数据中的小数也可以被正确处理。

    如果需要做“等于”比较，那么可以使用两个等号`==`。

    ```bash
    awk '/error/ {print NR, $0}' log.txt # 打印包含"error"的行及其行号
    ```

* 设置分隔符

    ```bash
    awk -F',' '{print $2}' file.csv    # 输入分隔符为逗号
    awk -F: '{print $1, $3}' /etc/passwd # 以冒号分隔，打印用户名和UID
    ```

    注：

    1. `-F,`也是可以的，逗号不需要加引号

* 数值统计

    ```bash
    awk '{sum += $1} END {print sum}' data.txt  # 对第1列求和
    awk '{if ($1 > max) max=$1} END {print max}' data.txt  # 求最大值
    ```

* 使用 BEGIN/END

    ```bash
    awk 'BEGIN {FS=":"; OFS="|"} {print $1, $3} END {print "处理完成"}' file.txt
    ```

    - `BEGIN`：在处理前执行（如初始化变量）。
    - `END`：在处理后执行（如输出结果）。

    注：

    1. 必须是大写的 END。awk 对大小写敏感
