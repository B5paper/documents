## cache

* 我自己写了个 ./main <subcommand> 程序，现在希望输入 ./main some-half-command 后，按 tab 键能补全 subcommand，该怎么办

	你需要为你的程序实现 **Shell 补全** 功能。有几种方法可以实现，我推荐使用最标准的 `bash-completion` 方式：

	* 方法一：使用 bash-completion（推荐）

        1. 创建补全脚本

            创建一个补全脚本文件，比如 `/etc/bash_completion.d/main` 或 `~/.local/share/bash-completion/completions/main`：

            ```bash
            # 补全脚本：main
            _main_completion() {
                local cur prev words cword
                _init_completion || return

                # 定义所有可用的子命令
                local subcommands="init build test deploy clean help version"

                # 如果当前是第一个参数，补全子命令
                if [[ $cword -eq 1 ]]; then
                    COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
                fi
            }

            # 注册补全函数
            complete -F _main_completion main
            ```

        2. 更高级的补全脚本

            如果你的子命令有参数，可以这样实现：

            ```bash
            _main_completion() {
                local cur prev words cword
                _init_completion || return

                local subcommands="init build test deploy clean help version"
                local build_opts="--debug --release --verbose --help"
                local deploy_opts="--production --staging --dry-run --help"

                case ${prev} in
                    build)
                        COMPREPLY=($(compgen -W "$build_opts" -- "$cur"))
                        ;;
                    deploy)
                        COMPREPLY=($(compgen -W "$deploy_opts" -- "$cur"))
                        ;;
                    main)
                        COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
                        ;;
                    *)
                        if [[ $cword -eq 1 ]]; then
                            COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
                        fi
                        ;;
                esac
            }

            complete -F _main_completion main
            ```

        3. 安装补全脚本

            ```bash
            # 系统级安装（需要root权限）
            sudo cp main /etc/bash_completion.d/

            # 或用户级安装
            mkdir -p ~/.local/share/bash-completion/completions/
            cp main ~/.local/share/bash-completion/completions/

            # 重新加载补全
            source ~/.bashrc
            # 或
            exec bash
            ```

	* 方法二：使用 compgen 内嵌在 .bashrc 中

        如果你不想创建单独的文件，可以直接在 `~/.bashrc` 中添加：

        ```bash
        # main 命令补全
        _main_completion() {
            local cur prev words cword
            _init_completion 2>/dev/null
            
            local subcommands="init build test deploy clean help version"
            
            if [[ $cword -eq 1 ]]; then
                COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
            fi
        }

        complete -F _main_completion main
        ```

        然后执行 `source ~/.bashrc`

	* 方法三：使用 complete 命令直接定义

        最简单的方式（但功能有限）：

        ```bash
        complete -W "init build test deploy clean help version" main
        ```

        把这个命令添加到 `~/.bashrc` 中。

    * 方法四：为不同 Shell 提供支持

        ### 对于 Zsh

        创建 `~/.zsh/completions/_main`：

        ```zsh
        #compdef main

        _main() {
            local -a subcommands
            subcommands=(
                'init:Initialize project'
                'build:Build project'
                'test:Run tests'
                'deploy:Deploy application'
                'clean:Clean build artifacts'
                'help:Show help'
                'version:Show version'
            )
            
            _describe 'command' subcommands
        }

        _main "$@"
        ```

        然后在 `~/.zshrc` 中添加：

        ```zsh
        fpath=(~/.zsh/completions $fpath)
        autoload -Uz compinit
        compinit
        ```

        ### 对于 Fish

        创建 `~/.config/fish/completions/main.fish`：

        ```fish
        complete -c main -f
        complete -c main -n __fish_use_subcommand -a init -d "Initialize project"
        complete -c main -n __fish_use_subcommand -a build -d "Build project"
        complete -c main -n __fish_use_subcommand -a test -d "Run tests"
        complete -c main -n __fish_use_subcommand -a deploy -d "Deploy application"
        complete -c main -n __fish_use_subcommand -a clean -d "Clean build artifacts"
        complete -c main -n __fish_use_subcommand -a help -d "Show help"
        complete -c main -n __fish_use_subcommand -a version -d "Show version"
        ```

	测试补全

	1. 保存并加载补全配置
	2. 在终端中输入：`main `（注意后面的空格）
	3. 按 Tab 键，应该能看到所有子命令的列表
	4. 输入部分命令如 `main bu` 然后按 Tab，应该能补全为 `main build`

* local cur prev words cword 是干嘛用的

	这些是 `bash-completion` 框架中的**内置变量**，用于获取补全时的上下文信息。让我详细解释每个变量的作用：

	**变量说明**

	| 变量 | 类型 | 描述 |
	|------|------|------|
	| `cur` | 字符串 | 当前正在输入的单词（光标前的部分） |
	| `prev` | 字符串 | 前一个单词 |
	| `words` | 数组 | 命令行中所有的单词 |
	| `cword` | 整数 | 当前单词在 `words` 数组中的索引 |

	**实际例子**

	假设用户输入了：

	```bash
	main build --debug some_pa[TAB]
	```

	这些变量的值会是：

	- `cur` = `"some_pa"` （当前正在输入的部分）
	- `prev` = `"--debug"` （前一个参数）
	- `words` = `("main" "build" "--debug" "some_pa")`
	- `cword` = `3` （当前是第4个单词，索引从0开始）

	**详细解释**

	* `cur` (current word)

        当前正在输入的单词。如果用户输入了一部分然后按 Tab，这个变量包含已输入的部分。

        ```bash
        # 用户输入: main ini[TAB]
        cur = "ini"

        # 用户输入: main build --[TAB]  
        cur = "--"
        ```

	* `prev` (previous word)

        前一个完整的单词。常用于判断上下文。

        ```bash
        # 用户输入: main build --[TAB]
        prev = "build"
        cur = "--"

        # 用户输入: main deploy --production [TAB]
        prev = "--production"
        cur = ""  # 空，因为光标在空格后
        ```

	* `words` (words array)

        整个命令行的单词数组。

        ```bash
        # 用户输入: main build --debug src/
        words = ("main" "build" "--debug" "src/")
        ```

	* `cword` (current word index)

        当前单词在数组中的位置。

        ```bash
        # 用户输入: main build --debug src/
        #                    ^ 光标在这里
        cword = 2  # --debug 是第3个单词（索引2）
        ```

	**实际使用示例**

	```bash
	_main_completion() {
		local cur prev words cword
		_init_completion || return

		echo "=== 调试信息 ===" >&2
		echo "cur: '$cur'" >&2
		echo "prev: '$prev'" >&2
		echo "cword: $cword" >&2
		echo "words: ${words[@]}" >&2
		echo "=================" >&2

		local subcommands="init build test deploy"
		local build_opts="--debug --release --help"
		local deploy_opts="--production --staging --help"

		# 根据上下文提供不同的补全
		case "${prev}" in
			build)
				# 前一个单词是 build，补全 build 的选项
				COMPREPLY=($(compgen -W "$build_opts" -- "$cur"))
				;;
			deploy)
				# 前一个单词是 deploy，补全 deploy 的选项
				COMPREPLY=($(compgen -W "$deploy_opts" -- "$cur"))
				;;
			--production|--staging)
				# 前一个单词是这些选项，可以补全环境名等
				COMPREPLY=($(compgen -W "prod staging dev" -- "$cur"))
				;;
			*)
				if [[ $cword -eq 1 ]]; then
					# 第一个参数，补全子命令
					COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
				elif [[ $cword -eq 2 && "$prev" == "build" ]]; then
					# build 后的第一个参数，补全目标
					COMPREPLY=($(compgen -W "all clean package" -- "$cur"))
				fi
				;;
		esac
	}
	```

	**`_init_completion` 函数**

	这个函数是 `bash-completion` 框架提供的，它：

	- 解析命令行，设置 `cur`, `prev`, `words`, `cword` 变量

	- 处理引号和转义字符

	- 返回 0 表示成功，非 0 表示不需要补全

	```bash
	_init_completion || return
	```

	这行代码确保变量正确初始化，如果失败就退出函数。

	**更复杂的实际案例**

	```bash
	_myapp_completion() {
		local cur prev words cword
		_init_completion || return

		local subcommands="server client config"
		local server_opts="start stop restart --port --host --help"
		local client_opts="connect disconnect send --server --timeout --help"

		case ${prev} in
			server)
				COMPREPLY=($(compgen -W "$server_opts" -- "$cur"))
				;;
			client)
				COMPREPLY=($(compgen -W "$client_opts" -- "$cur"))
				;;
			--port|--timeout)
				# 数字参数的补全
				COMPREPLY=($(compgen -W "80 443 8080 3000" -- "$cur"))
				;;
			--host|--server)
				# 主机名补全
				COMPREPLY=($(compgen -A hostname -- "$cur"))
				;;
			*)
				if [[ $cword -eq 1 ]]; then
					COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
				elif [[ "$cur" == -* ]]; then
					# 如果当前输入以 - 开头，补全通用选项
					local general_opts="--help --version --verbose --debug"
					COMPREPLY=($(compgen -W "$general_opts" -- "$cur"))
				fi
				;;
		esac
	}
	```

	这些变量让 Shell 补全变得非常强大，你可以根据用户在命令行中的位置和之前输入的内容，提供智能的补全建议。

* local 是干嘛用的

	`local` 是 shell 中的关键字，用于在函数内部声明变量，这些变量的作用域仅限于该函数内部。

	语法

	```bash
	local variable_name=value
	local var1 var2 var3
	```

	具体作用

	1. **限制变量作用域**

        ```bash
        #!/bin/bash

        my_function() {
            local local_var="我在函数内部"
            global_var="我是全局变量"
            
            echo "函数内: local_var=$local_var"
            echo "函数内: global_var=$global_var"
        }

        my_function

        echo "函数外: local_var=$local_var"      # 空，因为 local_var 是局部变量
        echo "函数外: global_var=$global_var"    # 有值，因为 global_var 是全局变量
        ```

        输出：

        ```
        函数内: local_var=我在函数内部
        函数内: global_var=我是全局变量
        函数外: local_var=
        函数外: global_var=我是全局变量
        ```

	2. **避免变量污染**

        在补全函数中使用 `local` 可以避免影响外部的全局变量：

        ```bash
        _main_completion() {
            # 这些变量只在函数内部有效
            local cur prev words cword
            local subcommands="init build test"
            local options="--help --version"
            
            _init_completion || return
            
            # 使用局部变量进行计算
            COMPREPLY=($(compgen -W "$subcommands $options" -- "$cur"))
        }

        # 函数外部无法访问 cur, prev, subcommands 等变量
        ```

	3. **在补全函数中的具体应用**

        ```bash
        _main_completion() {
            # 声明局部变量，避免与外部变量冲突
            local cur prev words cword
            
            # 初始化这些变量（由 _init_completion 设置）
            _init_completion || return
            
            # 定义只在当前补全函数中使用的变量
            local subcommands="init build test deploy"
            local build_opts="--debug --release"
            local deploy_opts="--prod --staging"
            
            # 根据上下文提供补全建议
            case $prev in
                build)
                    COMPREPLY=($(compgen -W "$build_opts" -- "$cur"))
                    ;;
                deploy)
                    COMPREPLY=($(compgen -W "$deploy_opts" -- "$cur"))
                    ;;
                *)
                    if [[ $cword -eq 1 ]]; then
                        COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
                    fi
                    ;;
            esac
        }
        ```

	**如果不使用 `local` 会怎样？**

	```bash
	#!/bin/bash

	# 全局变量
	global_cur="original"

	dangerous_function() {
		# 没有使用 local，会修改全局变量！
		cur="modified"
		echo "函数内: cur=$cur"
		echo "函数内: global_cur=$global_cur"
	}

	echo "调用前: global_cur=$global_cur"
	dangerous_function
	echo "调用后: global_cur=$global_cur"  # 被意外修改了！
	```

	输出：

	```
	调用前: global_cur=original
	函数内: cur=modified
	函数内: global_cur=modified
	调用后: global_cur=modified  # 全局变量被污染！
	```

	**在补全脚本中为什么必须用 `local`**

	1. **避免命名冲突**

        补全框架本身使用了很多变量，使用 `local` 可以避免冲突。

	2. **线程安全**

        虽然 Shell 通常是单线程的，但多个补全函数可能同时运行，使用局部变量可以避免相互干扰。

	3. **可重入性**

        确保每次调用补全函数时都有干净的变量环境。

	4. **内存管理**

        局部变量在函数退出时自动清理。

	**实际对比示例**

	**不好的写法（没有使用 local）：**

	```bash
	_main_completion() {
		# 这些变量可能会污染全局命名空间
		cur=$1
		prev=$2
		# ... 其他代码
	}
	```

	**好的写法（使用 local）：**

	```bash
	_main_completion() {
		# 这些变量只在函数内部有效
		local cur prev words cword
		_init_completion || return
		# ... 其他代码
	}
	```

	**其他 Shell 的类似功能**

	不同的 Shell 有不同的局部变量声明方式：

	| Shell | 局部变量声明 | 备注 |
	|-------|-------------|------|
	| Bash | `local var` | 最常用 |
	| Zsh | `local var` 或 `typeset var` | 类似 Bash |
	| Ksh | `typeset var` | |
	| Fish | 默认所有变量都是局部的 | |

	**总结**

	在 Shell 补全函数中使用 `local` 是为了：

	1. **封装性**：变量只在函数内部可见
	2. **安全性**：避免意外修改全局变量
	3. **清洁性**：函数退出时自动清理变量
	4. **可维护性**：明确变量的作用范围

	这就是为什么在补全函数开头总是看到：
	```bash
	local cur prev words cword
	_init_completion || return
	```

	这样的代码结构确保了补全函数的健壮性和可靠性。

* 所以 _init_completion 会捕捉到 local 声明的变量，并初始化他们？

	不，你的理解正好反了。实际情况是：**`_init_completion` 函数设置了一些变量，然后我们用 `local` 来捕获这些变量，使其成为当前函数的局部变量。**

	让我详细解释这个过程：

	## 实际执行流程

	```bash
	_main_completion() {
		# 1. 先声明局部变量（此时值为空）
		local cur prev words cword
		
		# 2. 调用 _init_completion，这个函数会设置全局变量
		_init_completion || return
		
		# 3. 现在我们的局部变量 cur, prev, words, cword 
		#    包含了补全所需的信息
	}
	```

	_init_completion` 的内部逻辑大致如下：

	```bash
	# 这是 bash-completion 框架中的函数（简化版）
	_init_completion() {
		# 这个函数会设置全局变量：
		# COMP_WORDS, COMP_CWORD, COMP_LINE, COMP_POINT 等
		
		# 然后它根据这些全局变量来设置：
		words=("${COMP_WORDS[@]}")    # 设置为全局的 words 数组
		cword=$COMP_CWORD             # 设置为全局的 cword
		cur="${COMP_WORDS[COMP_CWORD]}"  # 设置当前单词
		prev="${COMP_WORDS[COMP_CWORD-1]}"  # 设置前一个单词
		
		# 返回成功
		return 0
	}
	```

	**如果我们不用 `local` 会怎样？**

	```bash
	# 假设在某个地方有全局变量
	cur="important data"

	_bad_completion() {
		# 没有用 local！
		# _init_completion 会修改全局的 cur 变量！
		_init_completion || return
		
		# 现在全局的 cur 被覆盖了！
		echo "补全中的 cur: $cur"
	}

	echo "调用前全局 cur: $cur"
	_bad_completion
	echo "调用后全局 cur: $cur"  # 被破坏了！
	```

	**`_init_completion` 会捕捉到 `local` 声明的变量并初始化它们**。

	```bash
	_main_completion() {
		local cur prev words cword     # 声明局部变量
		_init_completion || return     # 这个函数会初始化这些局部变量
	}
	```

	在 Bash 中，当你在函数内使用 `local` 声明变量后，这些变量在当前函数及其调用的子函数中都是可见的。

* 如果我想补全 <program> <subcommand> <args> 中的 program 本身呢？

	补全 `<program>` 本身（而不仅仅是它的子命令）需要为你的程序创建**独立的补全脚本**。这样当用户在命令行中输入程序名并按 Tab 时，系统就能找到并执行对应的补全函数。

	## 方法一：创建独立的补全脚本（推荐）

	### 1. 创建补全脚本文件

	创建 `/etc/bash_completion.d/your_program` 或 `~/.local/share/bash-completion/completions/your_program`：

	```bash
	# your_program 的补全脚本
	_your_program_completion() {
		local cur prev words cword
		_init_completion || return

		# 定义所有可用的子命令
		local subcommands="init build test deploy clean help version"

		case $cword in
			1)
				# 第一个单词：补全子命令
				COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
				;;
			2)
				# 第二个单词：根据第一个子命令提供不同的补全
				case ${words[1]} in
					build)
						COMPREPLY=($(compgen -W "--debug --release --verbose --help" -- "$cur"))
						;;
					deploy)
						COMPREPLY=($(compgen -W "--production --staging --dry-run --help" -- "$cur"))
						;;
					*)
						# 其他子命令可以补全文件名等
						COMPREPLY=($(compgen -f -- "$cur"))
						;;
				esac
				;;
			*)
				# 更多参数的情况
				COMPREPLY=($(compgen -f -- "$cur"))
				;;
		esac
	}

	# 关键：为你的程序名注册补全函数
	complete -F _your_program_completion your_program
	```

	### 2. 安装并测试

	```bash
	# 安装补全脚本
	sudo cp your_program /etc/bash_completion.d/

	# 或者用户级安装
	mkdir -p ~/.local/share/bash-completion/completions/
	cp your_program ~/.local/share/bash-completions/completions/

	# 重新加载
	source ~/.bashrc

	# 测试补全
	your_program [TAB]        # 应该显示子命令列表
	your_program bu[TAB]      # 应该补全为 build
	your_program build [TAB]  # 应该显示 build 的选项
	```

	## 方法二：在 .bashrc 中直接定义

	如果你不想创建单独的文件，可以在 `~/.bashrc` 中添加：

	```bash
	# your_program 补全
	_your_program_completion() {
		local cur prev words cword
		_init_completion || return

		local subcommands="init build test deploy clean help version"
		local build_opts="--debug --release --help"
		local deploy_opts="--production --staging --help"

		case $cword in
			1)
				COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
				;;
			2)
				case ${words[1]} in
					build) COMPREPLY=($(compgen -W "$build_opts" -- "$cur")) ;;
					deploy) COMPREPLY=($(compgen -W "$deploy_opts" -- "$cur")) ;;
					*) COMPREPLY=($(compgen -f -- "$cur")) ;;
				esac
				;;
			*)
				COMPREPLY=($(compgen -f -- "$cur"))
				;;
		esac
	}

	complete -F _your_program_completion your_program
	```

	## 方法三：使用动态发现的可执行文件补全

	如果你希望补全系统能够自动发现所有可用的程序，可以创建一个更通用的补全脚本：

	```bash
	# 在 /etc/bash_completion.d/completers 中
	# 为多个程序定义补全

	# 你的程序
	_your_program_completion() {
		# ... 同上 ...
	}
	complete -F _your_program_completion your_program

	# 其他程序的补全
	_other_tool_completion() {
		# ... 其他工具的补全 ...
	}
	complete -F _other_tool_completion other_tool
	```

	## 验证补全是否工作

	```bash
	# 检查补全是否已注册
	complete -p | grep your_program

	# 应该输出：
	# complete -F _your_program_completion your_program

	# 测试补全功能
	your_program [TAB][TAB]  # 按两次Tab应该显示所有子命令
	```

	## 为不同安装位置的处理

	如果你的程序可能安装在不同的路径，可以使用这个更健壮的版本：

	```bash
	_your_program_completion() {
		local cur prev words cword
		_init_completion || return

		# 检查程序是否在PATH中
		if ! command -v your_program >/dev/null 2>&1; then
			COMPREPLY=()
			return
		fi

		local subcommands="init build test deploy clean help version"
		
		case $cword in
			1) COMPREPLY=($(compgen -W "$subcommands" -- "$cur")) ;;
			*) 
				# 根据已输入的参数提供智能补全
				case ${words[1]} in
					build) COMPREPLY=($(compgen -W "--debug --release --help" -- "$cur")) ;;
					deploy) COMPREPLY=($(compgen -W "--production --staging --help" -- "$cur")) ;;
					init) COMPREPLY=($(compgen -d -- "$cur")) ;;  # 补全目录
					*) COMPREPLY=($(compgen -f -- "$cur")) ;;
				esac
				;;
		esac
	}

	complete -F _your_program_completion your_program
	```

	## 关键点总结

	1. **`complete -F _func_name program_name`** 将补全函数绑定到程序名
	2. 补全脚本需要放在正确的位置让 bash 自动加载
	3. 通过 `$cword` 判断当前是第几个参数
	4. 通过 `${words[n]}` 获取之前输入的内容来提供上下文相关的补全

	这样设置后，用户在任何位置输入 `your_program` 后按 Tab，都能获得智能补全建议了！

* 我的意思是 $your_pro[tab] 可以补全为 $your_program

	啊，明白了！你是想让 **程序名本身** 能够被补全，比如输入 `your_pro` 按 Tab 能补全为 `your_program`。这是通过 **可执行文件补全** 实现的。

	## 方法一：让系统自动补全（最简单）

	Bash 默认就会补全 PATH 中的可执行文件。确保：

	1. 你的程序在 `$PATH` 环境变量中的某个目录里
	2. 程序有可执行权限

	```bash
	# 检查程序是否在 PATH 中
	which your_program
	# 或
	command -v your_program

	# 检查执行权限
	ls -l $(which your_program)
	```

	如果程序在 PATH 中且有执行权限，输入 `your_pro[Tab]` 应该就能自动补全。

	## 方法二：如果自动补全不工作

	如果系统没有自动补全，可以在 `~/.bashrc` 中显式启用：

	```bash
	# 启用命令补全
	complete -c your_program
	```

	或者为所有命令启用补全：
	```bash
	# 在 ~/.bashrc 中添加
	complete -c command
	```

	## 方法三：自定义程序名补全

	如果你想要更精细的控制，可以创建补全脚本：

	```bash
	# /etc/bash_completion.d/your_program
	_your_program_command_completion() {
		local cur prev words cword
		_init_completion || return

		# 当还没有输入任何命令时，补全程序名
		if [[ $cword -eq 0 ]] || [[ -z $prev ]]; then
			# 补全你的程序名（支持前缀匹配）
			COMPREPLY=($(compgen -c "your_program" -- "$cur"))
		else
			# 如果已经输入了程序名，则补全子命令
			local subcommands="init build test deploy clean help version"
			COMPREPLY=($(compgen -W "$subcommands" -- "$cur"))
		fi
	}

	# 为程序名注册补全
	complete -F _your_program_command_completion your_program
	```

	## 方法四：测试和调试 PATH 补全

	如果程序名补全不工作，可以测试：

	```bash
	# 1. 检查程序是否在 PATH 中
	echo $PATH
	which your_program

	# 2. 检查程序权限
	chmod +x /path/to/your_program

	# 3. 测试 Bash 补全
	ls /usr/share/bash-completion/completions/ | head  # 查看现有补全

	# 4. 手动重新生成补全缓存（某些系统需要）
	sudo update-bash-completion
	```

	## 最可能的原因和解决方案

	如果程序名本身不能补全，通常是：

	1. **程序不在 PATH 中**

	   ```bash
	   # 解决方案：添加到 PATH
	   echo 'export PATH="/path/to/your/program/dir:$PATH"' >> ~/.bashrc
	   source ~/.bashrc
	   ```

	2. **程序没有执行权限**

	   ```bash
	   chmod +x /path/to/your_program
	   ```

	3. **Bash 补全功能未启用**

	   ```bash
	   # 在 ~/.bashrc 中确保有
	   if [ -f /etc/bash_completion ]; then
		   . /etc/bash_completion
	   fi
	   ```

