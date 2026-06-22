* windows下是否有类似 systemd 这样的功能

    Windows 没有完全等同于 Linux systemd 的组件，但有多个类似功能的机制：

    ## 1. **服务管理（最接近 systemd）**
    - **Services.msc** - 图形界面服务管理器
    - **PowerShell**：
      ```powershell
      # 查看服务
      Get-Service
      
      # 启动/停止服务
      Start-Service -Name "服务名"
      Stop-Service -Name "服务名"
      
      # 设置开机启动
      Set-Service -Name "服务名" -StartupType Automatic
      
      # 查看依赖关系（类似 systemd 的依赖链）
      Get-Service -Name "服务名" -DependentServices
      ```

    - **命令行（sc.exe）**：
      ```cmd
      sc query 服务名
      sc start 服务名
      sc stop 服务名
      sc config 服务名 start=auto
      ```

    ## 2. **任务计划程序（类似 systemd 定时器）**
    ```powershell
    # 查看计划任务
    Get-ScheduledTask

    # 创建定时任务（类似 systemd timer）
    schtasks /create /tn "任务名" /tr "程序路径" /sc daily /st 09:00
    ```

    ## 3. **Windows 子系统（类似 init 系统）**
    - **Service Control Manager (SCM)** - Windows 的服务控制系统
    - **Windows 启动顺序**：
      1. 内核启动
      2. Session Manager (smss.exe)
      3. 服务控制管理器启动
      4. Winlogon 启动
      5. 用户会话启动

    ## 4. **现代替代方案**

    ### **Windows Services for Linux (WSL2)**
    在 WSL2 中可以直接使用 systemd：
    ```bash
    # 编辑 WSL 配置文件
    sudo nano /etc/wsl.conf
    ```
    添加：
    ```ini
    [boot]
    systemd=true
    ```

    ### **第三方工具**
    1. **NSSM** - Windows 服务管理器
       ```cmd
       nssm install 服务名 程序路径
       nssm start 服务名
       ```

    2. **WinSW** - 将任何程序包装为 Windows 服务
       ```xml
       <!-- 配置文件示例 -->
       <service>
         <id>myapp</id>
         <name>My Application</name>
         <executable>java.exe</executable>
         <arguments>-jar myapp.jar</arguments>
       </service>
       ```

    ## 5. **PowerShell 自动化（类似 systemd unit 文件）**
    创建 PowerShell 脚本管理服务：
    ```powershell
    # 类似 systemd unit 的脚本
    $service = @{
        Name        = "MyService"
        BinaryPath  = "C:\app\myapp.exe"
        Description = "My custom service"
        DependsOn   = @("NetLogon", "EventLog")
    }

    # 安装为服务
    New-Service @service
    ```

    ## 6. **关键差异对比**

    | Linux systemd 功能 | Windows 对应方案 |
    |-------------------|-----------------|
    | `systemctl start` | `Start-Service` / `sc start` |
    | `systemctl enable` | `Set-Service -StartupType Automatic` |
    | `systemctl status` | `Get-Service` |
    | `journalctl` | `Get-EventLog` / `Event Viewer` |
    | `systemd timer` | `Task Scheduler` |
    | `systemd unit 文件` | `服务注册表项` / `NSSM` |
    | 依赖管理 | 服务依赖关系（有限） |
    | cgroups | Job Objects / Process Groups |

    ## 7. **日志管理（类似 journalctl）**
    ```powershell
    # 查看应用日志
    Get-EventLog -LogName Application -Newest 50

    # 实时监控日志（类似 journalctl -f）
    Get-WinEvent -LogName System -MaxEvents 10 -Wait

    # 使用 Windows Event Viewer（图形界面）
    eventvwr.msc
    ```

    ## 推荐方案：
    1. **简单服务管理**：使用原生 `Services.msc` + PowerShell
    2. **复杂服务需求**：使用 NSSM 或 WinSW
    3. **需要 systemd 完整功能**：在 WSL2 中启用 systemd
    4. **定时任务**：使用 Task Scheduler

    Windows 的服务管理虽然不如 systemd 统一，但通过组合使用这些工具，可以实现大多数 systemd 的功能。

* 我希望 windows 开机执行 ssh -N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 user@remote_server，如果执行失败则等待 120 秒后重新执行，怎么办

    在 Windows 上实现开机自动执行 SSH 反向隧道并自动重试，有几种方法：

    ## 方法 1：使用计划任务（推荐）

    创建 PowerShell 脚本 `C:\Scripts\ssh-tunnel.ps1`：
    ```powershell
    # ssh-tunnel.ps1
    $maxRetries = 10
    $retryDelay = 120  # 秒
    $retryCount = 0

    function Start-SSHTunnel {
        param(
            [string]$SSHPath = "C:\Windows\System32\OpenSSH\ssh.exe",
            [string]$RemoteServer = "user@remote_server",
            [string]$RemotePort = "xxxx",
            [string]$LocalPort = "xxxx"
        )
        
        $sshArgs = "-N", "-R", "${RemotePort}:127.0.0.1:${LocalPort}", 
                   "-o", "ServerAliveInterval=60", 
                   "-o", "ExitOnForwardFailure=yes",
                   "-o", "ConnectTimeout=30",
                   $RemoteServer
        
        Write-Host "启动 SSH 隧道到 $RemoteServer..."
        $process = Start-Process -FilePath $SSHPath -ArgumentList $sshArgs -NoNewWindow -PassThru
        
        # 等待进程结束
        $process.WaitForExit()
        
        return $process.ExitCode
    }

    while ($retryCount -lt $maxRetries) {
        $retryCount++
        Write-Host "尝试 #$retryCount"
        
        $exitCode = Start-SSHTunnel
        
        if ($exitCode -eq 0) {
            Write-Host "SSH 隧道正常退出"
            break
        } else {
            Write-Host "SSH 隧道异常退出 (代码: $exitCode)"
            if ($retryCount -lt $maxRetries) {
                Write-Host "等待 $retryDelay 秒后重试..."
                Start-Sleep -Seconds $retryDelay
            }
        }
    }

    if ($retryCount -eq $maxRetries) {
        Write-Host "达到最大重试次数 ($maxRetries)，停止尝试"
    }
    ```

    ## 方法 2：使用任务计划程序

    1. **创建基本任务**：
       ```powershell
       # 打开任务计划程序
       taskschd.msc
       ```

    2. **设置任务**：
       - 触发器：`启动时` 或 `用户登录时`
       - 操作：`启动程序`
         - 程序/脚本：`powershell.exe`
         - 参数：`-WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Scripts\ssh-tunnel.ps1"`

    3. **配置任务属性**：
       - 勾选 `使用最高权限运行`
       - `如果任务失败，按以下频率重新启动`：每 2 分钟
       - `尝试重启最多`：10 次
       - `如果正在运行，则停止现有实例`

    ## 方法 3：使用 NSSM 创建 Windows 服务

    ```powershell
    # 1. 下载 NSSM：https://nssm.cc/download
    # 2. 创建服务
    nssm install SSH-Tunnel

    # 设置路径
    服务路径：C:\Windows\System32\OpenSSH\ssh.exe
    启动参数：-N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 user@remote_server

    # 3. 配置失败重试
    nssm set SSH-Tunnel AppExit Default Restart
    nssm set SSH-Tunnel AppRestartDelay 120000  # 120秒 = 120000毫秒
    nssm set SSH-Tunnel AppThrottle 300000

    # 4. 启动服务
    nssm start SSH-Tunnel
    ```

    ## 方法 4：使用批处理脚本 + 计划任务

    创建 `C:\Scripts\ssh-tunnel.bat`：
    ```batch
    @echo off
    :start
    echo [%date% %time%] 启动 SSH 隧道...
    C:\Windows\System32\OpenSSH\ssh.exe -N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 user@remote_server

    echo [%date% %time%] SSH 隧道断开，等待 120 秒后重试...
    timeout /t 120 /nobreak >nul
    goto start
    ```

    创建 `C:\Scripts\start-tunnel.vbs`（隐藏运行）：
    ```vbscript
    Set WshShell = CreateObject("WScript.Shell")
    WshShell.Run "cmd /c C:\Scripts\ssh-tunnel.bat", 0, False
    ```

    设置计划任务执行 `.vbs` 文件。

    ## 方法 5：使用 WinSW（Windows 服务包装器）

    1. **下载 WinSW**：https://github.com/winsw/winsw/releases
    2. **创建 `ssh-tunnel.xml`**：
    ```xml
    <service>
      <id>ssh-tunnel</id>
      <name>SSH Reverse Tunnel</name>
      <description>SSH 反向隧道到远程服务器</description>
      
      <executable>C:\Windows\System32\OpenSSH\ssh.exe</executable>
      <arguments>-N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 user@remote_server</arguments>
      
      <!-- 失败重试配置 -->
      <onfailure action="restart" delay="120 sec"/>
      <resetfailure>3600</resetfailure>
      
      <log mode="roll-by-size">
        <sizeThreshold>10240</sizeThreshold>
        <keepFiles>8</keepFiles>
      </log>
    </service>
    ```

    3. **安装服务**：
    ```cmd
    winsw install ssh-tunnel.xml
    winsw start ssh-tunnel.xml
    ```

    ## 推荐方案对比：

    | 方案 | 优点 | 缺点 |
    |------|------|------|
    | **计划任务 + PowerShell** | 原生支持，配置灵活，日志清晰 | 需要 PowerShell 脚本知识 |
    | **NSSM** | 简单，直接，服务管理方便 | 第三方工具，依赖外部程序 |
    | **WinSW** | 功能强大，配置文件灵活 | 配置相对复杂 |

    ## 额外建议：使用密钥认证

    为了避免密码输入，配置 SSH 密钥：
    ```powershell
    # 1. 生成密钥（如果还没有）
    ssh-keygen -t ed25519 -f C:\Users\用户名\.ssh\tunnel_key

    # 2. 复制公钥到远程服务器
    type C:\Users\用户名\.ssh\tunnel_key.pub | ssh user@remote_server "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

    # 3. 修改脚本使用密钥
    $sshArgs = "-N", "-R", "${RemotePort}:127.0.0.1:${LocalPort}", 
               "-o", "ServerAliveInterval=60",
               "-o", "ExitOnForwardFailure=yes",
               "-i", "C:\Users\用户名\.ssh\tunnel_key",
               "-o", "StrictHostKeyChecking=accept-new",
               $RemoteServer
    ```

    ## 验证配置是否生效：

    ```powershell
    # 查看服务状态
    Get-Service -Name "SSH-Tunnel" -ErrorAction SilentlyContinue

    # 查看进程
    Get-Process -Name ssh -ErrorAction SilentlyContinue

    # 手动重启服务
    Restart-Service -Name "SSH-Tunnel" -Force

    # 查看日志（如果是NSSM/WinSW）
    Get-EventLog -LogName Application -Source "SSH-Tunnel" -Newest 10
    ```

    **推荐使用方案1（计划任务+PowerShell）**，这是最稳定且易于维护的 Windows 原生解决方案。

* ssh tunnel 进程会结束吗？开机后一直会保持连接吧？

    ```bash
    # 等待进程结束
    $process.WaitForExit()
    ```

    **你说得对！我的脚本逻辑有错误。** SSH 隧道进程在正常情况下**不会结束**，会一直保持连接。`WaitForExit()` 会一直阻塞，不会触发重试逻辑。

    需要改为**监控进程状态**，而不是等待退出。以下是修正方案：

    ## 方案 1：改进的 PowerShell 监控脚本

    ```powershell
    # ssh-tunnel-monitor.ps1
    $maxRetries = 9999  # 几乎无限重试
    $retryDelay = 120   # 秒
    $retryCount = 0
    $sshProcess = $null

    # SSH 配置
    $sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
    $remoteServer = "user@remote_server"
    $remotePort = "xxxx"
    $localPort = "xxxx"
    $sshKeyPath = "C:\Users\用户名\.ssh\tunnel_key"  # 可选

    function Start-SSHTunnel {
        $sshArgs = "-N",
                   "-R", "${remotePort}:127.0.0.1:${localPort}",
                   "-o", "ServerAliveInterval=60",
                   "-o", "ExitOnForwardFailure=yes",
                   "-o", "ConnectTimeout=30",
                   "-o", "TCPKeepAlive=yes",
                   "-o", "StrictHostKeyChecking=accept-new"
        
        # 如果有密钥，添加 -i 参数
        if (Test-Path $sshKeyPath) {
            $sshArgs += "-i", $sshKeyPath
        }
        
        $sshArgs += $remoteServer
        
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 启动 SSH 隧道到 $remoteServer..."
        
        # 启动进程但不等待退出
        $processInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processInfo.FileName = $sshPath
        $processInfo.Arguments = $sshArgs
        $processInfo.RedirectStandardOutput = $true
        $processInfo.RedirectStandardError = $true
        $processInfo.UseShellExecute = $false
        $processInfo.CreateNoWindow = $true
        
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $processInfo
        $process.EnableRaisingEvents = $true
        
        # 添加退出事件处理（进程异常退出时重试）
        Register-ObjectEvent -InputObject $process -EventName Exited -SourceIdentifier "SSHProcessExited" -Action {
            $exitCode = $event.SourceEventArgs.ExitCode
            $time = Get-Date -Format "HH:mm:ss"
            Write-Host "[$time] SSH 进程异常退出，代码: $exitCode"
            # 这里可以触发重试逻辑
        }
        
        $null = $process.Start()
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] SSH 隧道进程已启动 (PID: $($process.Id))"
        
        return $process
    }

    function Monitor-SSHTunnel {
        param([System.Diagnostics.Process]$process)
        
        while ($true) {
            # 检查进程是否存在
            if ($process.HasExited) {
                $exitCode = $process.ExitCode
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] SSH 隧道进程已退出，代码: $exitCode"
                return $false
            }
            
            # 可选：检查网络连接状态（需要更复杂的检测）
            # 这里简单检查，实际可以添加端口检查等
            
            # 等待一段时间再检查
            Start-Sleep -Seconds 60
        }
        
        return $true
    }

    # 主循环
    while ($true) {
        $retryCount++
        Write-Host "`n========== 隧道连接尝试 #$retryCount =========="
        
        # 检查是否已有 ssh 进程在运行
        $existingProcess = Get-Process ssh -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -like "*$remoteServer*"
        }
        
        if ($existingProcess) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 发现已存在的 SSH 隧道进程 (PID: $($existingProcess.Id))"
            $sshProcess = $existingProcess
        } else {
            # 启动新隧道
            $sshProcess = Start-SSHTunnel
        }
        
        # 等待一小段时间，让进程稳定
        Start-Sleep -Seconds 10
        
        # 检查进程是否还在运行
        if (-not $sshProcess.HasExited) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] SSH 隧道运行正常，开始监控..."
            
            # 等待进程退出（这会阻塞直到进程退出）
            $sshProcess.WaitForExit()
        }
        
        # 进程已退出，等待重试延迟
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 等待 $retryDelay 秒后重试..."
        Start-Sleep -Seconds $retryDelay
    }
    ```

    ## 方案 2：简化的守护进程脚本

    ```powershell
    # ssh-tunnel-simple.ps1
    $sshPath = "ssh.exe"  # 如果已添加 PATH，可以直接用 ssh
    $sshArgs = "-N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 -o ServerAliveCountMax=3 user@remote_server"

    while ($true) {
        Write-Host "$(Get-Date -Format '[yyyy-MM-dd HH:mm:ss]') 启动 SSH 隧道..."
        
        # 启动 ssh 隧道
        $process = Start-Process -FilePath $sshPath `
                   -ArgumentList $sshArgs `
                   -NoNewWindow `
                   -PassThru
        
        # 等待进程退出
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        
        Write-Host "$(Get-Date -Format '[yyyy-MM-dd HH:mm:ss]') SSH 隧道断开，退出码: $exitCode"
        
        # 等待 120 秒后重试
        Write-Host "$(Get-Date -Format '[yyyy-MM-dd HH:mm:ss]') 等待 120 秒后重试..."
        Start-Sleep -Seconds 120
    }
    ```

    ## 方案 3：使用 PowerShell 作业（更优雅）

    ```powershell
    # ssh-tunnel-job.ps1
    $maxRetries = 9999
    $retryDelay = 120

    function Test-SSHTunnel {
        # 简单的隧道检测（检查本地端口或远程连接）
        # 这里只是示例，需要根据实际情况实现
        try {
            # 例如：检查本地是否有 SSH 进程
            $sshProcess = Get-Process ssh -ErrorAction SilentlyContinue
            return ($sshProcess -ne $null)
        }
        catch {
            return $false
        }
    }

    for ($i = 1; $i -le $maxRetries; $i++) {
        Write-Host "隧道连接尝试 #$i"
        
        if (-not (Test-SSHTunnel)) {
            # 启动 SSH 作业
            $job = Start-Job -Name "SSHTunnel" -ScriptBlock {
                ssh -N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 user@remote_server
            }
            
            Write-Host "SSH 隧道作业已启动"
        }
        
        # 检查作业状态
        $jobStatus = Get-Job -Name "SSHTunnel" -ErrorAction SilentlyContinue
        
        if ($jobStatus.State -eq "Failed" -or $jobStatus.State -eq "Completed") {
            Write-Host "SSH 作业状态异常: $($jobStatus.State)"
            Remove-Job -Name "SSHTunnel" -Force -ErrorAction SilentlyContinue
        }
        
        # 等待下次检查
        Start-Sleep -Seconds $retryDelay
    }
    ```

    ## 方案 4：Windows 服务方案（推荐用于生产）

    使用 WinSW 或 NSSM 创建 Windows 服务，它们自带进程监控和重启功能：

    ```xml
    <!-- WinSW 配置文件 -->
    <service>
      <id>ssh-tunnel</id>
      <name>SSH Reverse Tunnel</name>
      
      <executable>C:\Windows\System32\OpenSSH\ssh.exe</executable>
      <arguments>-N -R xxxx:127.0.0.1:xxxx -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes user@remote_server</arguments>
      
      <!-- 进程退出后自动重启 -->
      <onfailure action="restart" delay="120 sec"/>
      
      <!-- 30分钟后重启（防止内存泄漏） -->
      <stoptimeout>30 sec</stoptimeout>
      <startmode>Automatic</startmode>
      
      <logpath>C:\Logs\ssh-tunnel</logpath>
    </service>
    ```

    ## 任务计划程序设置：

    1. **创建基本任务**
    2. **触发器**：`计算机启动时`
    3. **操作**：
       - 程序：`powershell.exe`
       - 参数：`-WindowStyle Hidden -ExecutionPolicy Bypass -File "C:\Scripts\ssh-tunnel-simple.ps1"`
    4. **条件**：
       - 取消勾选 `只有在计算机使用交流电源时才启动此任务`
       - 勾选 `如果使用电池，则启动`

    ## 最佳实践建议：

    1. **使用 SSH 配置简化**：
       ```ssh
       # 在 ~/.ssh/config 中添加
       Host tunnel-server
           HostName remote_server
           User user
           IdentityFile ~/.ssh/tunnel_key
           ServerAliveInterval 60
           ServerAliveCountMax 3
           ExitOnForwardFailure yes
       
       # 然后脚本中只需：
       ssh -N -R xxxx:127.0.0.1:xxxx tunnel-server
       ```

    2. **添加健康检查**：
       ```powershell
       # 定期检查隧道是否真正工作
       function Test-TunnelWorking {
           # 尝试通过隧道连接本地服务
           # 或者检查端口状态
       }
       ```

    3. **日志记录**：
       ```powershell
       # 重定向输出到日志文件
       Start-Transcript -Path "C:\Logs\ssh-tunnel-$(Get-Date -Format 'yyyyMMdd').log" -Append
       ```

    **最简单可靠的是方案 2**，它会保持 SSH 隧道运行，断开后自动重连。

* 为什么 msc 配置 service 需要如此复杂的 powershell 脚本？似乎所有的状态监测都由 powershell 脚本完成。而 systemd 只需要写核心 command，其余的状态监测以及重启控制全由 systemd 完成

    **你说得非常对！这正是 Windows 服务管理的一个痛点。**

    ## Windows 服务管理的核心问题

    ### 1. **理念差异**
    - **Systemd**：统一的服务管理器，提供完整的生命周期管理
    - **Windows Services**：更轻量级的服务管理器，只负责启动/停止/重启

    ### 2. **Windows 原生服务限制**
    ```xml
    <!-- 这是 Windows 服务能做到的极限 -->
    <service>
        <id>myservice</id>
        <executable>myapp.exe</executable>
        <arguments>--arg value</arguments>
        
        <!-- 原生只有这些有限的控制 -->
        <onfailure action="restart" delay="10 sec"/>  <!-- 延迟只能是固定值 -->
        <resetfailure>1 hour</resetfailure>           <!-- 重置失败计数 -->
    </service>
    ```

    **原生 Windows 服务缺少：**
    - ❌ 灵活的重试逻辑（如：指数退避）
    - ❌ 健康检查（进程存在≠服务健康）
    - ❌ 依赖关系管理（有限）
    - ❌ 详细的日志结构化输出
    - ❌ 服务重启前的清理工作

    ## 为什么需要复杂脚本？— Systemd vs Windows 对比

    | 功能 | Systemd (简单) | Windows (复杂) |
    |------|----------------|----------------|
    | **基本服务** | `ExecStart=/usr/bin/ssh ...` | 需要配置服务或计划任务 |
    | **失败重试** | `Restart=always` + `RestartSec=120` | 需要 PowerShell 脚本循环 |
    | **健康检查** | `ExecStartPost=/bin/test-connection` | 需要脚本实现检查逻辑 |
    | **日志管理** | 自动到 journal | 需要手动重定向到文件 |
    | **环境变量** | `Environment=VAR=value` | 需要脚本设置或配置文件 |
    | **工作目录** | `WorkingDirectory=/path` | 需要脚本 cd 或配置 |

    ## Windows 的 "Systemd-like" 解决方案

    ### 方案 A：使用第三方工具（最接近 systemd）

    ```yaml
    # 使用 WinSW (类似 systemd unit)
    service:
      id: ssh-tunnel
      name: SSH Tunnel Service
      description: SSH Reverse Tunnel with auto-restart
      
      # 核心命令（类似 systemd 的 ExecStart）
      executable: ssh.exe
      arguments: >
        -N -R xxxx:127.0.0.1:xxxx 
        -o ServerAliveInterval=60 
        user@remote_server
      
      # 失败处理（类似 systemd 的 Restart）
      onfailure:
        - action: restart
          delay: 10 sec
        - action: restart  
          delay: 30 sec
        - action: restart
          delay: 120 sec  # 指数退避需要配置多组
      
      # 日志（类似 journald）
      log:
        mode: roll-by-size
        sizeThreshold: 10MB
        keepFiles: 10
      
      # 环境变量
      env:
        - name: SSH_AUTH_SOCK
          value: /tmp/ssh-agent.sock
        
      # 前置检查（类似 ConditionPathExists）
      prerequisites:
        - check: "file"
          path: "C:\Users\user\.ssh\id_rsa"
          action: log-warning
    ```

    ### 方案 B：Windows 内置的最佳方案

    ```powershell
    # 使用 Scheduled Tasks + PowerShell 实现 systemd 功能
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $action = New-ScheduledTaskAction `
        -Execute "pwsh.exe" `
        -Argument "-File C:\Scripts\ssh-service.ps1" `
        -WorkingDirectory "C:\Scripts"

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RestartInterval (New-TimeSpan -Minutes 2) `
        -RestartCount 3 `
        -MultipleInstances Parallel

    # 注册任务
    Register-ScheduledTask `
        -TaskName "SSH-Tunnel-Systemd" `
        -Trigger $trigger `
        -Action $action `
        -Settings $settings `
        -User "SYSTEM" `
        -RunLevel Highest
    ```

    ## 为什么 Microsoft 不提供类似 systemd 的功能？

    1. **向后兼容**：Windows 服务模型从 NT 时代延续至今
    2. **设计哲学**：Windows 偏爱 GUI 配置，Linux 偏爱配置文件
    3. **第三方生态**：有 NSSM、WinSW 等工具填补空白
    4. **容器化趋势**：现代应用推荐用 Docker/容器，而不是系统服务

    ## 真正的简化方案：使用 Docker Desktop

    ```dockerfile
    # 在 Windows 上使用 Docker（最接近 Linux 体验）
    version: '3.8'
    services:
      ssh-tunnel:
        image: alpine/openssh
        command: >
          ssh -N -R xxxx:127.0.0.1:xxxx 
          -o ServerAliveInterval=60 
          user@remote_server
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "nc", "-z", "localhost", "22"]
          interval: 30s
          timeout: 10s
          retries: 3
        volumes:
          - ~/.ssh:/root/.ssh:ro
    ```

    ```powershell
    # 启动服务（类似 systemctl start）
    docker-compose up -d

    # 查看状态（类似 systemctl status）
    docker-compose ps

    # 查看日志（类似 journalctl）
    docker-compose logs -f
    ```

    ## 推荐的实际做法

    ### 对于简单服务：使用 **WinSW**
    ```xml
    <!-- ssh-tunnel.xml -->
    <service>
      <id>ssh-tunnel</id>
      <executable>ssh</executable>
      <arguments>-N -R xxxx:127.0.0.1:xxxx user@remote_server</arguments>
      <onfailure action="restart" delay="120 sec"/>
      <logpath>C:\Logs</logpath>
    </service>
    ```

    安装：
    ```cmd
    winsw install ssh-tunnel.xml
    winsw start ssh-tunnel.xml
    ```

    ### 对于复杂服务：使用 **Docker**

    ### 对于临时方案：使用 **计划任务**

    ## 总结

    | 需求 | 推荐方案 | 复杂度 |
    |------|----------|--------|
    | **简单 SSH 隧道** | WinSW | ⭐⭐ |
    | **有健康检查的服务** | Docker + docker-compose | ⭐⭐⭐ |
    | **企业级服务管理** | PowerShell DSC | ⭐⭐⭐⭐ |
    | **最像 systemd** | WSL2 + 启用 systemd | ⭐⭐ |

    **核心问题**：Windows 确实缺乏像 systemd 那样的**统一服务管理框架**。PowerShell 脚本本质上是手动实现 systemd 已经内置的功能。对于 SSH 隧道这种简单服务，WinSW 是最接近 systemd 体验的选择。

