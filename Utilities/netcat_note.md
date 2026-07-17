# Netcat note

## cache

* 如何使用 nc 传输文件

	好的，使用 `nc`（netcat）传输文件是一种简单直接的网络文件传输方法，特别适用于临时性的任务或在没有其他工具（如 `scp`, `rsync`）的环境下。其核心原理是：一端作为**接收方**监听一个端口，另一端作为**发送方**向该端口发送数据。

	### 核心概念

	*   **接收方**：启动一个监听模式，将收到的网络数据重定向到一个文件。
	*   **发送方**：连接到接收方的指定端口，并将文件的内容发送过去。

	---

	### 使用方法

	这里介绍两种最常见的情景：传输单个文件和传输整个目录。

	#### 场景一：传输单个文件

	假设你有两台机器：
	*   **接收文件方**：IP 为 `192.168.1.100`
	*   **发送文件方**：IP 为 `192.168.1.200`
	*   **要传输的文件**：`backup.tar.gz`
	*   **选择端口**：`12345`（你可以使用任何未被占用的端口，注意防火墙设置）

	**步骤 1：在接收方启动监听**

	在 `192.168.1.100` 上执行命令。这个命令会监听 `12345` 端口，并将收到的所有数据写入 `received_backup.tar.gz` 文件。

	```bash
	# 在接收方 (192.168.1.100) 执行
	nc -l -p 12345 > received_backup.tar.gz
	```
	*   `-l`：监听模式。
	*   `-p 12345`：指定监听端口。
	*   `>`：重定向符号，将 `nc` 接收到的标准输出保存到文件。

	**步骤 2：在发送方发送文件**

	在 `192.168.1.200` 上执行命令。这个命令会将 `backup.tar.gz` 文件的内容发送到接收方的 `12345` 端口。

	```bash
	# 在发送方 (192.168.1.200) 执行
	nc 192.168.1.100 12345 < backup.tar.gz
	```
	*   `<`：重定向符号，将文件的内容作为 `nc` 的标准输入发送出去。

	**传输过程**
	*   发送方命令执行后，文件开始传输。
	*   传输期间不会有任何进度提示。
	*   当文件传输完毕，两端的 `nc` 进程会自动退出。你可以通过检查接收方文件的大小或使用校验和（如 `md5sum`）来验证文件完整性。

	---

	#### 场景二：传输整个目录（使用 tar）

	`nc` 本身只能传输数据流，不能直接传输目录结构。但我们可以结合 `tar` 命令来打包和解包目录。

	**目标**：将发送方的 `/path/to/some_directory` 目录传输到接收方。

	**步骤 1：在接收方启动监听并解包**

	在 `192.168.1.100` 上执行。`nc` 接收到的数据流直接通过管道 `|` 传递给 `tar` 进行解压。

	```bash
	# 在接收方 (192.168.1.100) 执行
	nc -l -p 12345 | tar -xzvf -
	```
	*   `tar -xzvf -`：
		*   `-x`：解压。
		*   `-z`：使用 gzip 解压缩（如果发送方用了 `-z`，这里也必须用）。
		*   `-v`：显示解压过程（可选，便于观察进度）。
		*   `-f -`：从标准输入读取数据。

	**步骤 2：在发送方打包目录并发送**

	在 `192.168.1.200` 上执行。`tar` 命令将目录打包成一个数据流，然后通过管道 `|` 发送给 `nc`。

	```bash
	# 在发送方 (192.168.1.200) 执行
	tar -czf - /path/to/some_directory | nc 192.168.1.100 12345
	```
	*   `tar -czf - /path/to/some_directory`：
		*   `-c`：创建归档文件。
		*   `-z`：使用 gzip 压缩（可选，但能显著减少传输量）。
		*   `-f -`：将归档文件输出到标准输出，而不是一个文件。
		*   `/path/to/some_directory`：要打包的目录。

	**传输过程**
	*   发送方开始打包和压缩，数据流通过网络发送到接收方。
	*   接收方接收到数据流后立即开始解包，你会在接收方终端看到 `tar` 解压出来的文件列表（因为用了 `-v` 参数）。
	*   传输完成后，两端进程自动退出。

	---

	### 注意事项与技巧

	1.  **防火墙**：确保接收方机器上的防火墙开放了你所使用的端口（如 `12345`）。
	2.  **网络安全**：`nc` 传输是**明文**的，不加密！不要在不可信的网络上用此方法传输敏感数据。对于敏感数据，请使用 `scp` 或 `rsync over SSH`。
	3.  **速度与进度**：`nc` 传输没有进度条。对于大文件，你可以使用 `pv`（Pipe Viewer）命令来监视进度。
		*   **发送方**：`tar -czf - /path/to/dir | pv | nc 192.168.1.100 12345`
		*   **接收方**：`nc -l -p 12345 | pv | tar -xzf -`
	4.  **保持监听**：有时发送方连接后，接收方会立即退出。可以使用 `-k`（在某些版本的 `nc` 中）参数让接收方保持监听，以接收多个文件，但通常更简单的做法是每次传输都重新执行命令。
	5.  **版本差异**：不同系统上的 `nc` 版本（如传统的 `netcat` 和 `ncat`）参数可能略有不同。上述命令在大多数 Linux 发行版的 `netcat-openbsd` 或 `netcat-traditional` 包中测试通过。如果遇到问题，可以查看 `man nc`。

	### 总结

	| 场景 | 接收方命令 | 发送方命令 |
	| :--- | :--- | :--- |
	| **单个文件** | `nc -l -p <端口> > <输出文件名>` | `nc <接收方IP> <端口> < <要发送的文件>` |
	| **整个目录** | `nc -l -p <端口> \| tar -xzvf -` | `tar -czf - <目录路径> \| nc <接收方IP> <端口>` |

	这是一种强大而快速的“瑞士军刀”式解决方案，非常适合在受信任的网络环境中进行临时文件传输。

	* 不需要使用-w吗

		您提出了一个很好的问题！**在大多数情况下，传输文件时不需要使用 `-w` 参数。**

		让我详细解释一下 `-w` 参数的作用和何时需要使用它：

		## `-w` 参数的作用

		`-w` 参数用于设置连接的超时时间（单位：秒）。例如：
		```bash
		nc -w 30 192.168.1.100 12345 < file.txt
		```
		这表示如果连接在30秒内没有完成传输，`nc` 会自动退出。

		## 为什么文件传输通常不需要 `-w`

		1. **传输完成自动退出**：当文件传输完毕时，发送方的 `nc` 会检测到文件结束(EOF)并自动关闭连接，接收方也会相应退出。

		2. **避免意外中断**：如果设置 `-w 30`，但传输一个大文件需要60秒，那么传输会在30秒时被强制中断，导致文件损坏。

		3. **默认行为更合适**：大多数 `nc` 版本的默认行为是等待直到传输完成，这正好符合文件传输的需求。

		## 需要使用 `-w` 的情况

		只有在特定场景下才需要考虑使用 `-w`：

		1. **网络诊断**：测试端口是否开放时，你希望快速得到结果而不是一直等待：
		   ```bash
		   nc -w 3 192.168.1.100 80
		   ```

		2. **防止挂起**：当网络连接可能出现问题，你希望设置一个最大等待时间。

		3. **自动化脚本**：在脚本中不希望 `nc` 进程无限期挂起。

		## 文件传输的最佳实践

		对于文件传输，推荐的做法是**不使用 `-w`**：

		```bash
		# 接收方
		nc -l -p 12345 > received_file

		# 发送方  
		nc 192.168.1.100 12345 < source_file
		```

		或者如果确实需要超时保护，设置一个足够长的时间：
		```bash
		# 设置300秒（5分钟）超时，对于大多数文件都足够了
		nc -w 300 192.168.1.100 12345 < large_file.iso
		```

		## 总结

		**您是对的 - 在标准的文件传输场景中，不需要使用 `-w` 参数。** 让 `nc` 在传输完成后自然退出是最简单可靠的方式。只有在特殊需求（如网络测试、脚本自动化）时才需要考虑使用超时设置。

	* 我一个用的是winfows上的nc，一个用的是linux上的nc，windows上的是否支持>或<？不支持该怎么办？

		您指出了一个很重要的跨平台兼容性问题！**Windows 上的 `nc` 通常不支持 `>` 或 `<` 这种重定向语法**，因为 Windows 的命令行重定向机制与 Linux 不同。

		## 问题分析

		- **Linux `nc`**：通常基于 Unix 版本，完美支持重定向
		- **Windows `nc`**：可能是 Cygwin、Nmap 的 `ncat` 或其他移植版本，重定向行为不一致

		## 解决方案

		### 方案1：使用管道（推荐，兼容性最好）

		这种方法在 Windows 和 Linux 上都适用：

		**传输单个文件：**

		```bash
		# 接收方（Linux或Windows）：监听并将输出保存到文件
		nc -l -p 12345 > received_file.txt

		# 发送方（Windows）：使用 type 命令和管道
		type file.txt | nc 192.168.1.100 12345
		```

		**传输二进制文件：**

		```bash
		# 接收方（Linux或Windows）：
		nc -l -p 12345 > received_image.jpg

		# 发送方（Windows）：
		type image.jpg | nc 192.168.1.100 12345
		```

		### 方案2：使用 `-q` 参数（如果您的 Windows nc 支持）

		某些 `nc` 版本支持 `-q` 参数在传输完成后自动退出：

		```bash
		# 接收方（Linux）：
		nc -l -p 12345 > received_file

		# 发送方（Windows）：
		nc -q 1 192.168.1.100 12345 < file.txt
		```

		### 方案3：使用 PowerShell（如果 Windows 有 PowerShell）

		PowerShell 的重定向语法不同但功能更强：

		```powershell
		# 接收方（Linux）：
		nc -l -p 12345 > received_file

		# 发送方（Windows PowerShell）：
		Get-Content file.txt -Raw | nc 192.168.1.100 12345

		# 对于二进制文件，使用：
		Get-Content file.bin -AsByteStream | nc 192.168.1.100 12345
		```

		### 方案4：传输目录（跨平台）

		**发送方（Windows）：**
		```cmd
		# 打包目录
		tar -czf - directory_name | nc 192.168.1.100 12345
		```

		**接收方（Linux）：**
		```bash
		nc -l -p 12345 | tar -xzvf -
		```

		*注意：Windows 需要安装 tar 工具（Windows 10 1809+ 自带，或者安装 Git Bash、Cygwin 等）*

		## 完整示例

		假设：
		- Linux 接收方 IP: `192.168.1.100`
		- Windows 发送方要发送 `document.pdf`

		**步骤1：在 Linux 上启动接收**
		```bash
		nc -l -p 12345 > document.pdf
		```

		**步骤2：在 Windows 上发送**
		```cmd
		type document.pdf | nc 192.168.1.100 12345
		```

		## 故障排查技巧

		1. **检查 Windows `nc` 版本**：
		   ```cmd
		   nc -h
		   ```

		2. **测试连接**（先确认网络通畅）：
		   ```cmd
		   # Windows 测试连接
		   nc -zv 192.168.1.100 12345
		   ```

		3. **如果管道方法失败**，尝试使用其他工具如 `ncat`（Nmap 的一部分），它通常有更好的 Windows 支持。

		## 推荐做法

		**对于跨平台文件传输，我强烈推荐使用管道方法**，因为它：
		- 在几乎所有 `nc` 版本上都工作
		- 语法一致
		- 支持文本和二进制文件

		所以记住这个模式：
		- **接收方**：`nc -l -p [端口] > [文件名]`
		- **发送方**：`type [文件名] | nc [IP] [端口]`

	* type是干嘛用的

		好的，这是一个很好的问题！`type` 是 Windows 命令提示符（CMD）中的一个内置命令，它的作用类似于 Linux 中的 `cat` 命令。

		## `type` 命令的作用

		**`type` 的主要功能是：在命令行中显示文本文件的内容。**

		### 基本用法
		```cmd
		type filename.txt
		```
		这会在命令提示符窗口中显示 `filename.txt` 的全部内容。

		## 为什么我们在 `nc` 传输中使用 `type`

		在文件传输的上下文中，我们利用 `type` 来**读取文件内容并通过管道传递给 `nc`**：

		```cmd
		type file.txt | nc 192.168.1.100 12345
		```

		这个命令的分解：
		1. `type file.txt` - 读取 `file.txt` 的内容
		2. `|` - 管道符号，将前一个命令的输出作为后一个命令的输入
		3. `nc 192.168.1.100 12345` - 将接收到的数据发送到指定地址和端口

		## `type` vs Linux 的 `<` 重定向

		| 操作 | Windows | Linux |
		|------|---------|-------|
		| 显示文件内容 | `type file.txt` | `cat file.txt` |
		| 发送文件内容 | `type file.txt \| nc ...` | `nc ... < file.txt` |
		| 重定向文件 | 不支持 `<` 输入重定向 | 支持 `<` 输入重定向 |

		## `type` 的局限性

		**重要提醒：`type` 主要设计用于文本文件！**

		对于二进制文件（如图片、压缩包、可执行文件等），`type` 可能会遇到问题：
		- 可能无法正确处理二进制数据
		- 在某些编码下可能损坏文件

		## 更好的跨平台方案

		### 如果您的 Windows `nc` 是 `ncat`（来自 Nmap）
		```cmd
		ncat 192.168.1.100 12345 < file.bin
		```

		### 使用 PowerShell（推荐用于二进制文件）
		```powershell
		Get-Content file.bin -AsByteStream | nc 192.168.1.100 12345
		```

		### 如果传输二进制文件且 `type` 失败
		可以考虑使用其他工具：
		- 安装 `ncat`（Nmap 的一部分）
		- 使用 Git Bash 或 Cygwin 提供的 Unix-like 环境
		- 使用 PowerShell

		## 实际测试建议

		1. **先测试文本文件**：
		   ```cmd
		   type test.txt | nc 192.168.1.100 12345
		   ```

		2. **如果传输二进制文件出现问题**，切换到其他方法。

		## 总结

		`type` 是 Windows CMD 中查看文本文件内容的命令，我们在文件传输中借用它来替代 Linux 的 `<` 重定向操作。对于文本文件它工作良好，但对于二进制文件可能需要替代方案。

		**简单记法：**
		- **Windows CMD**: `type 文件名 | nc IP 端口`
		- **Linux**: `nc IP 端口 < 文件名`

	* 没有-asbytestrram

		您说得对！`-AsByteStream` 参数是在 PowerShell 6.0+ 中引入的，如果您使用的是旧版本的 PowerShell（比如 Windows 自带的 PowerShell 5.x），这个参数不可用。

		## 旧版本 PowerShell 的解决方案

		### 方法1：使用 `-Encoding byte`（PowerShell 5.x 及更早版本）

		```powershell
		# 对于文本文件
		Get-Content file.txt -Encoding byte | nc 192.168.1.100 12345

		# 对于二进制文件（如图片、压缩包等）
		Get-Content file.bin -Encoding byte -ReadCount 0 | nc 192.168.1.100 12345
		```

		### 方法2：使用 `.NET` 方法（最可靠）

		```powershell
		# 读取二进制文件并传输
		[System.IO.File]::ReadAllBytes("file.bin") | nc 192.168.1.100 12345
		```

		或者更完整的写法：
		```powershell
		$bytes = [System.IO.File]::ReadAllBytes("C:\path\to\file.bin")
		$bytes | ForEach-Object { [byte]$_ } | nc 192.168.1.100 12345
		```

		### 方法3：使用 `System.IO.FileStream`

		```powershell
		$stream = [System.IO.File]::OpenRead("file.bin")
		$buffer = New-Object byte[] 4096
		while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
			$buffer[0..($read-1)] | nc 192.168.1.100 12345
		}
		$stream.Close()
		```

		## 完整的工作示例

		### 传输文本文件：
		```powershell
		# 方法1：使用 Get-Content
		Get-Content file.txt -Encoding byte | nc 192.168.1.100 12345

		# 方法2：使用 .NET
		[System.IO.File]::ReadAllBytes("file.txt") | nc 192.168.1.100 12345
		```

		### 传输二进制文件（如图片、ZIP等）：
		```powershell
		# 最推荐的方法
		[System.IO.File]::ReadAllBytes("image.jpg") | nc 192.168.1.100 12345
		```

		### 传输整个目录：
		```powershell
		# 打包并传输目录
		tar -czf - mydirectory | nc 192.168.1.100 12345
		```

		## 最简单的推荐方案

		**对于大多数情况，我推荐使用这个简单的 .NET 方法：**

		```powershell
		[System.IO.File]::ReadAllBytes("你的文件") | nc IP地址 端口号
		```

		例如：
		```powershell
		[System.IO.File]::ReadAllBytes("document.pdf") | nc 192.168.1.100 12345
		```

		## 验证 PowerShell 版本

		如果您想知道自己的 PowerShell 版本，可以运行：
		```powershell
		$PSVersionTable.PSVersion
		```

		## 总结

		对于没有 `-AsByteStream` 参数的旧版 PowerShell：

		1. **文本文件**：`Get-Content file.txt -Encoding byte | nc ...`
		2. **二进制文件**：`[System.IO.File]::ReadAllBytes("file.bin") | nc ...`（推荐）
		3. **目录**：使用 `tar` 命令打包后传输

		`.NET` 方法是最可靠且兼容性最好的解决方案，适用于所有版本的 PowerShell。

	* -Encoding byte 传输的是char的int数字，不是二进制。你再想想

		您完全正确！我犯了一个错误。`-Encoding byte` 在 PowerShell 中确实会将每个字节作为字符的整数值输出，而不是原始二进制数据。感谢您的纠正！

		让我提供正确的解决方案：

		## 正确的 PowerShell 二进制文件传输方法

		### 方法1：使用 `.NET` 方法（最可靠）

		```powershell
		# 正确的方法：直接写入标准输出流
		[System.IO.File]::ReadAllBytes("file.bin") | ForEach-Object { $_.ToString() } | Out-Null
		```

		但更好的方法是直接使用文件流：

		```powershell
		# 使用 FileStream 直接读取和传输二进制数据
		$fileStream = [System.IO.File]::OpenRead("file.bin")
		$buffer = New-Object byte[] 8192
		while (($bytesRead = $fileStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
			# 将字节直接写入标准输出
			[Console]::OpenStandardOutput().Write($buffer, 0, $bytesRead)
			[Console]::OpenStandardOutput().Flush()
		}
		$fileStream.Close()
		```

		### 方法2：使用 `certutil`（Windows 内置工具）

		```powershell
		# 将文件编码为十六进制，然后解码并传输
		certutil -encode file.bin temp.b64 && type temp.b64 | nc 192.168.1.100 12345 && del temp.b64
		```

		接收方需要相应地解码：
		```bash
		# Linux 接收方
		nc -l -p 12345 | base64 -d > received_file.bin
		```

		### 方法3：使用 PowerShell 5.1+ 的 `Get-Content -AsByteStream`

		如果您有较新版本的 PowerShell：
		```powershell
		Get-Content file.bin -AsByteStream | nc 192.168.1.100 12345
		```

		### 方法4：最简单的跨平台解决方案

		**使用 Python（如果系统上有安装）：**

		发送方（Windows）：
		```python
		python -c "import sys; [sys.stdout.buffer.write(open(sys.argv[1], 'rb').read())]" file.bin | nc 192.168.1.100 12345
		```

		接收方（Linux）：
		```bash
		nc -l -p 12345 > received_file.bin
		```

		### 方法5：使用 `ncat`（如果可用）

		如果您的 Windows 系统有 `ncat`（Nmap 的一部分）：
		```cmd
		ncat 192.168.1.100 12345 < file.bin
		```

		## 推荐的实用解决方案

		考虑到兼容性和可靠性，我建议：

		### 对于一次性传输：
		```powershell
		# 发送方（Windows PowerShell）
		$filePath = "file.bin"
		$ip = "192.168.1.100"
		$port = 12345

		$fileStream = [System.IO.File]::OpenRead($filePath)
		$buffer = New-Object byte[] 8192
		while (($bytesRead = $fileStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
			[Console]::OpenStandardOutput().Write($buffer, 0, $bytesRead)
			[Console]::OpenStandardOutput().Flush()
		}
		$fileStream.Close() | nc $ip $port
		```

		### 对于频繁使用，创建一个 PowerShell 函数：
		```powershell
		function Send-File {
			param(
				[string]$FilePath,
				[string]$RemoteHost,
				[int]$Port = 12345
			)
			
			$fileStream = [System.IO.File]::OpenRead($FilePath)
			$buffer = New-Object byte[] 8192
			while (($bytesRead = $fileStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
				[Console]::OpenStandardOutput().Write($buffer, 0, $bytesRead)
				[Console]::OpenStandardOutput().Flush()
			}
			$fileStream.Close()
		}

		# 使用示例
		Send-File -FilePath "file.bin" -RemoteHost "192.168.1.100" -Port 12345 | nc 192.168.1.100 12345
		```

		## 总结

		正确的做法是使用 `.NET` 的文件流 API 直接读取二进制数据并写入标准输出流，而不是依赖 PowerShell 的文本处理管道。感谢您指出我的错误！

## note

windows exe: <https://joncraton.org/blog/46/netcat-for-windows/>

netcat for windows: <https://github.com/diegocr/netcat>


传输二进制文件：

在接收端使用命令：`nc -l 1234 > filetxt`

在发送端使用命令：`nc <ip_addr> 1234 < file.txt` 

文件完成传输后，接收文件的 nc 进程会自动退出。

scanning:

`nc -v -w 2 -z target 20-30`

* `-z`：Only scan for listening daemons, without sending any data to them. 

* `-i`可以用于增加端口之间的扫描间隔。

Banner grabbing:

`nc -v -n 192.168.1.90 80`

* `-d`：禁止从 stdin 输入。此时无论从 terminal 输入，还是使用`<`进行重定向输入，都是不可以的。

文件传输：

可以使用`<`向 nc 通过 stdin 传输文件：

server: `nc -l -p 12345 < hello.txt`

client: `nc 127.0.0.1 > hello.txt`

* Use netcat behind a proxy

    <https://unix.stackexchange.com/questions/38783/how-to-make-netcat-use-an-existing-http-proxy>

    Example:

    ```bash
    nc -X connect -x <proxy_address> <connect_ip> <port>
    ```

    `connect` means HTTP proxy.

* port forwarding

    netcat can only listen a port or connect to a port, but can't do port forwarding. You can use `ncat` (from `nmap`) or `socat` instead.

    Ref: <https://unix.stackexchange.com/questions/293304/using-netcat-for-port-forwarding>
