# Socat Note

## cache

* how to use socat to forward tcp link from 10.0.2.5:7860 to 127.0.0.1:7860, use chinese to answer

	可以使用以下 socat 命令来将 TCP 连接从 10.0.2.5:7860 转发到 127.0.0.1:7860：

	## 基本转发命令

	```bash
	socat TCP-LISTEN:7860,bind=10.0.2.5,fork TCP:127.0.0.1:7860
	```

	## 命令参数说明

	- `TCP-LISTEN:7860` - 在7860端口监听连接
	- `bind=10.0.2.5` - 绑定到指定的IP地址（10.0.2.5）
	- `fork` - 为每个新连接创建子进程，支持多连接
	- `TCP:127.0.0.1:7860` - 将流量转发到本地主机的7860端口

	## 其他有用的选项

	```bash
	# 添加更多控制选项
	socat TCP-LISTEN:7860,bind=10.0.2.5,fork,reuseaddr TCP:127.0.0.1:7860

	# 后台运行
	socat TCP-LISTEN:7860,bind=10.0.2.5,fork TCP:127.0.0.1:7860 &

	# 记录日志
	socat -d -d TCP-LISTEN:7860,bind=10.0.2.5,fork TCP:127.0.0.1:7860
	```

	## 验证转发

	可以使用 telnet 或 curl 测试转发是否正常工作：

	```bash
	# 测试连接
	telnet 10.0.2.5 7860

	# 或者使用curl（如果是HTTP服务）
	curl http://10.0.2.5:7860
	```

	这样配置后，所有发送到 `10.0.2.5:7860` 的请求都会被自动转发到 `127.0.0.1:7860`。

## note

1. 创建一个 tunnel: <https://unix.stackexchange.com/questions/10428/simple-way-to-create-a-tunnel-from-one-local-port-to-another>

1. port forwarding: <https://unix.stackexchange.com/questions/293304/using-netcat-for-port-forwarding>

1. tcp port forwarding: <https://www.cyberciti.biz/faq/linux-unix-tcp-port-forwarding/>
