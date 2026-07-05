# fail2ban note

## cache

* fail2ban 如何测试指定一个字符串是否被 filter 匹配？

    在fail2ban中，你可以使用`fail2ban-regex`工具来测试指定的字符串是否被filter匹配。以下是几种测试方法：

    1. **基本测试语法**

        ```bash
        fail2ban-regex "要测试的日志字符串" /path/to/filter.conf
        ```

    2. **测试示例**

        **示例1：测试单行日志**
        ```bash
        # 测试单行日志是否匹配
        fail2ban-regex "2023-10-01 12:34:56 Failed password for root from 192.168.1.100" /etc/fail2ban/filter.d/sshd.conf
        ```

        **示例2：测试多行日志**
        ```bash
        # 创建测试文件
        echo "2023-10-01 12:34:56 Failed password for root from 192.168.1.100" > test.log
        echo "2023-10-01 12:35:01 Failed password for admin from 192.168.1.101" >> test.log

        # 测试文件
        fail2ban-regex test.log /etc/fail2ban/filter.d/sshd.conf
        ```

    3. **使用不同格式的输入**

        **从标准输入测试：**

        ```bash
        echo "Invalid user test from 192.168.1.100" | fail2ban-regex stdin /etc/fail2ban/filter.d/sshd.conf
        ```

        **从变量测试：**

        ```bash
        LOG_LINE="Connection closed by invalid user [preauth]"
        echo "$LOG_LINE" | fail2ban-regex stdin /etc/fail2ban/filter.d/sshd.conf
        ```

    4. **详细输出模式**

        **显示详细信息：**
        ```bash
        # -v 显示详细信息
        fail2ban-regex -v "测试字符串" /etc/fail2ban/filter.d/sshd.conf

        # --print-all-matched 显示所有匹配
        fail2ban-regex --print-all-matched "测试字符串" /path/to/filter.conf

        # --print-ignored 显示被忽略的行
        fail2ban-regex --print-ignored "测试字符串" /path/to/filter.conf
        ```

    5. **常用选项**

        | 选项 | 说明 |
        |------|------|
        | `-v` | 详细输出，显示匹配过程 |
        | `--print-all-matched` | 显示所有匹配的行 |
        | `--print-ignored` | 显示被忽略的行 |
        | `--print-all-missed` | 显示所有未匹配的行 |
        | `-d <DATE>` | 设置日期模式 |
        | `--full-trace` | 显示完整的调试信息 |

    6. **实际应用示例**

        ```bash
        # 测试SSH失败登录
        fail2ban-regex "Jan 1 12:34:56 server sshd[1234]: Failed password for invalid user test from 192.168.1.100 port 22 ssh2" /etc/fail2ban/filter.d/sshd.conf

        # 测试自定义filter
        fail2ban-regex "ERROR - Authentication failed for user=admin" /path/to/custom-filter.conf

        # 显示详细匹配信息
        fail2ban-regex -v --print-all-matched "测试字符串" /etc/fail2ban/filter.d/apache-auth.conf
        ```

    7. **在脚本中使用**

        你可以将`fail2ban-regex`整合到脚本中进行自动化测试：

        ```bash
        #!/bin/bash

        TEST_STRING="Failed password for root from 192.168.1.100"
        FILTER="/etc/fail2ban/filter.d/sshd.conf"

        if fail2ban-regex "$TEST_STRING" "$FILTER" > /dev/null 2>&1; then
            echo "字符串匹配成功！"
        else
            echo "字符串不匹配。"
        fi
        ```

    8. **调试filter规则**

        如果你需要调试或测试filter的某个特定正则表达式：

        ```bash
        # 直接测试正则表达式
        fail2ban-regex "测试字符串" 'failregex = ^.*Failed password.*<HOST>'
        ```

    **注意事项：**

    1. 确保fail2ban服务已安装：`apt install fail2ban` 或 `yum install fail2ban`
    2. filter文件路径通常为：`/etc/fail2ban/filter.d/`
    3. 测试时会考虑filter中的`failregex`和`ignoreregex`配置
    4. 输出结果会显示匹配的主机IP、时间和匹配次数等信息

    通过`fail2ban-regex`工具，你可以快速验证filter规则是否正确匹配特定的日志字符串，这对于调试和优化fail2ban配置非常有帮助。

* 如何查看 fail2ban 提取出的 <HOST> 具体是多少？

    你可以使用 `--print-all-matched` 选项来查看提取的 `<HOST>` 值。具体方法如下：

    1. **查看匹配的HOST值**

        ```bash
        # 使用 --print-all-matched 显示匹配详情
        fail2ban-regex --print-all-matched \
            "12月 02 20:52:30 iZuf63p69virk2qhn3bl4jZ sshd[1753315]: Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" \
            /etc/fail2ban/filter.d/sshd.conf
        ```

    2. **显示更详细的信息**

        ```bash
        # 使用详细模式查看提取过程
        fail2ban-regex -v --print-all-matched \
            "12月 02 20:52:30 iZuf63p69virk2qhn3bl4jZ sshd[1753315]: Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" \
            /etc/fail2ban/filter.d/sshd.conf
        ```

    3. **如果上面的不显示，尝试这个调试方法**

        创建一个简化的测试配置文件：

        ```bash
        # 创建测试配置
        cat > /tmp/debug_host.conf << 'EOF'
        [Definition]
        failregex = Connection closed by authenticating user root <HOST> port \d+ \[preauth\]$
        ignoreregex =
        EOF

        # 测试并显示HOST
        fail2ban-regex --print-all-matched \
            "Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" \
            /tmp/debug_host.conf
        ```

    4. **使用grep查看匹配过程**

        ```bash
        # 首先确保能匹配
        echo "Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" | \
            fail2ban-regex stdin /etc/fail2ban/filter.d/sshd.conf -v
        ```

    5. **如果上面方法都不行，使用正则表达式测试**

        ```bash
        # 直接测试正则表达式提取
        echo "Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" | \
            python3 -c "
        import re
        import sys
        line = sys.stdin.read().strip()
        pattern = r'Connection closed by authenticating user root (\d+\.\d+\.\d+\.\d+) port \d+ \[preauth\]\$'
        match = re.search(pattern, line)
        if match:
            print('提取的HOST:', match.group(1))
        else:
            print('未匹配')
        "
        ```

    6. **检查fail2ban的调试输出**

        ```bash
        # 启用更详细的调试
        fail2ban-regex --full-trace \
            "Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" \
            /etc/fail2ban/filter.d/sshd.conf 2>&1 | grep -i host
        ```

    7. **使用fail2ban-client测试**

        ```bash
        # 另一种方法是通过fail2ban服务测试
        sudo fail2ban-client -vvv start sshd 2>&1 | grep -A5 -B5 "167.172.36.246"
        ```

    8. **创建测试脚本**

        ```bash
        cat > test_host_extraction.sh << 'EOF'
        #!/bin/bash

        # 测试字符串
        LOG_LINE="12月 02 20:52:30 iZuf63p69virk2qhn3bl4jZ sshd[1753315]: Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]"

        # 创建临时filter文件
        cat > /tmp/test_extract.conf << 'CONFEOF'
        [Definition]
        failregex = ^.*Connection closed by authenticating user root <HOST> port \d+ \[preauth\]$
        ignoreregex =
        CONFEOF

        echo "测试日志: $LOG_LINE"
        echo "----------------------------------------"

        # 运行测试
        fail2ban-regex --print-all-matched "$LOG_LINE" /tmp/test_extract.conf

        # 清理
        rm -f /tmp/test_extract.conf
        EOF

        chmod +x test_host_extraction.sh
        ./test_host_extraction.sh
        ```

    9. **查看实际的匹配结果**

        如果`--print-all-matched`不显示具体值，可能根本没有匹配成功。你可以先确认是否有匹配：

        ```bash
        # 检查是否有任何匹配
        echo "Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]" | \
            fail2ban-regex stdin 'failregex = Connection closed by authenticating user root <HOST> port \d+ \[preauth\]$' | \
            grep -E "(Matched|HOST|IP)"
        ```

    10. **使用Python直接调试**

        ```bash
        python3 << 'EOF'
        import re

        # 日志行
        line = "12月 02 20:52:30 iZuf63p69virk2qhn3bl4jZ sshd[1753315]: Connection closed by authenticating user root 167.172.36.246 port 56310 [preauth]"

        # 尝试不同的模式
        patterns = [
            r'Connection closed by authenticating user root (\d+\.\d+\.\d+\.\d+) port \d+ \[preauth\]$',
            r'root (\d+\.\d+\.\d+\.\d+) port \d+ \[preauth\]$',
            r'(\d+\.\d+\.\d+\.\d+) port \d+ \[preauth\]$',
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, line)
            if match:
                print(f"模式{i+1}匹配成功!")
                print(f"提取的IP: {match.group(1)}")
                break
        else:
            print("所有模式都未匹配")
        EOF
        ```

    如果还是看不到HOST值，可能是因为：

    1. **根本没有匹配成功**：查看输出中的 `matched: 0`
    2. **HOST标记没有正确识别**：确保正则表达式中有 `<HOST>` 标记
    3. **IP地址格式不被识别**：确认IP地址是有效的IPv4格式

    请运行上面的命令，把输出结果发给我，我可以帮你分析具体问题所在。

* fail2ban 手动往 sshd jail 里添加一个 ip

    ```bash
    # 永久禁止 IP（直到手动移除）
    sudo fail2ban-client set sshd banip 192.168.1.100

    # 禁止带时间限制（例如 1 天）
    sudo fail2ban-client set sshd banip --time 86400 192.168.1.100
    ```

    查看和管理已禁止的 IP:

    ```bash
    # 查看 sshd jail 状态
    sudo fail2ban-client status sshd

    # 查看已禁止的 IP 列表
    sudo fail2ban-client get sshd banned

    # 解禁 IP
    sudo fail2ban-client set sshd unbanip 192.168.1.100
    ```

## topics
