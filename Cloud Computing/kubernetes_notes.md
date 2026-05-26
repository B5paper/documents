# kubernetes notes

## cache

* k8s 中，如果我在 pod 里起 qemu，并用 cni  插件使得 pod 在同一 ip 网段内，那么如何在 pod 1  中的 qemu 访问到 pod 2 中的 qemu ?

    在大规模虚拟化平台（例如 KubeVirt）的设计中，这是一个非常经典的网络嵌套（Nested Networking）场景。

    要让 Pod 1 中的 QEMU 虚拟机（假设为 VM 1）直接访问 Pod 2 中的 QEMU 虚拟机（假设为 VM 2），核心痛点在于：**默认情况下，Kubernetes CNI 只知道 Pod 的 IP，不知道你塞在 Pod 里面的 VM 的 IP，更不知道怎么为 VM 做路由。**

    要打通这个网络，主要有以下三种主流的解决方案，取决于你对**网络性能**和**架构复杂度**的要求：

    * 方案一：桥接模式（Bridge）+ 独立子网（最推荐，最符合 K8s 规范）

        如果你的 CNI 本身支持在 Pod 内创建网桥（比如使用 **Multus CNI** 接入了一个二层网络，或者使用支持多网卡的 CNI 插件），你可以把 Pod 容器的网卡和 QEMU 的 `tap` 设备绑定在同一个虚拟网桥上。

        **核心网络拓扑**

        ```
        [ Pod 1 (IP: 10.244.1.2) ]              [ Pod 2 (IP: 10.244.2.3) ]
          ├── eth0 (CNI 分配)                     ├── eth0 (CNI 分配)
          ├── br0 (容器内网桥)                     ├── br0 (容器内网桥)
          └── tap0 (QEMU 绑定)                    └── tap0 (QEMU 绑定)
               └── [ VM 1 (IP: 192.168.10.10) ]        └── [ VM 2 (IP: 192.168.10.20) ]
        ```

        **实现步骤**

        1. **容器启动脚本中创建网桥**：

            在 Pod 启动时，不要直接把流量给 QEMU，而是在容器内用 `ip link` 创建一个 `br0`。
        
        2. **把 CNI 网卡挂载到网桥**：

            将 Pod 的主网卡（如 `eth0`）的 IP 清除，并把 `eth0` 作为 slave 接口挂载到 `br0` 上，把原本属于 `eth0` 的 IP 配置给 `br0`。
        
        3. **QEMU 使用 TAP 模式启动**：

            配置 QEMU 使用 `tap` 网络，并将创建的 `tap0` 也挂载到 `br0` 上。
            ```bash
            qemu-system-x86_64 -netdev tap,id=net0,ifname=tap0,script=no,downscript=no -device virtio-net-pci,netdev=net0
            ```

        4. **IP 分配**：

            * **情况 A（同网段）**：如果 CNI 本身是 Macvlan/IPvlan 或者 Flat 二层网络，VM 1 和 VM 2 可以直接向物理网络申请和 Pod 同网段的独立 IP。它们之间可以直接通过 ARP 广播找到对方。
            * **情况 B（独立非重叠网段）**：给 VM1 分配 `192.168.10.10`，VM2 分配 `192.168.10.20`。此时需要在 K8s 节点上或者 CNI 路由表中**配置静态路由**，告诉集群：去往 `192.168.10.0/24` 的下一跳（Next Hop）是对应的 Pod IP。

    * 方案二：利用 K8s Service / NodePort + QEMU 端口映射（最简单，不改网络）

        如果你不想大动干戈去改动底层 CNI 网络，可以把 QEMU 虚拟机当成一个普通的“容器内应用”，通过端口转发（Port Forwarding）来打通。

        **实现步骤**

        1. **QEMU 启动时配置端口转发（User 模式网络）**：

            假设 VM 2 开放了 SSH（22端口）和某个业务端口（8080）。在 Pod 2 启动 QEMU 时，加入 `hostfwd` 参数，将 Pod 2 的端口映射到 VM 2 内部：

            ```bash
               # 在 Pod 2 中运行，将 Pod 的 2222 映射到 VM 的 22，8080 映射到 VM 的 80
               qemu-system-x86_64 -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8080-:80 -device virtio-net-pci,netdev=net0
            ```

        2. **Pod 1 访问 Pod 2**：

            由于 Pod 1 和 Pod 2 在同一个 CNI 插件的 IP 网段内，Pod 1 可以直接通过 Pod 2 的 IP 访问其映射出来的端口。
            * VM 1 访问 VM 2 的 SSH：在 VM 1 内部访问 `Pod2_IP:2222`。
            * 如果想更稳定，可以为 Pod 2 创建一个 K8s Service，VM 1 直接访问 `Service_IP:Mapped_Port`。

    * 方案三：Macvtlan / IPvlan 直通模式（性能最高）

        如果你的底层 CNI 本身就是基于 `macvlan` 或 `ipvlan` 实现的（或者是带有多网卡插件如 Multus 的物理网络直通），你可以直接让 QEMU 共享或复制容器的 MAC 地址。

        **实现步骤**

        1. 在 Pod 内利用 `ip link` 基于 `eth0` 创建一个 `macvtlan` 设备。
        2. 将该设备直接作为文件描述符（FD）传递给 QEMU，或者使用传统配置挂载。
        3. 这样 VM 1 和 VM 2 就像是直接接在同一个物理交换机上的两台独立服务器，它们拥有和 Pod 同级别的网络地位，直接通过二层/三层网络互通。

    **行业成熟方案：看看 KubeVirt 是怎么做的**

    如果你是在生产环境或者打算长期维护这个架构，强烈建议参考 **KubeVirt** 的源码或直接使用 KubeVirt。KubeVirt 是 Kubernetes 官方专门用来在 Pod 里跑 QEMU/KVM 的项目。

    KubeVirt 的默认网络（Pod 模式）实现逻辑如下：

    1. 它利用了一个叫 `virt-launcher` 的原生容器去接管 CNI 分配给 Pod 的 IP。

    2. 它把这个 IP “剥夺”下来，通过容器内的 `dnsmasq`（一个轻量级 DHCP 服务）或者静态 ARP 绑定，**无缝传递给 QEMU 虚拟机内部**。

    3. 此时，Pod 本身变成了“透明透明的管道”，**VM 2 直接占用了 Pod 2 的 IP**。

    4. **最终效果**：Pod 1 中的 VM 1 如果想访问 Pod 2 中的 VM 2，**直接 ping Pod 2 的 IP 即可**。

    > **避坑提示：**
    > 如果选择方案一（桥接），请务必注意 CNI 插件是否开启了 **反 IP/MAC 欺骗（Anti-Spoofing）**。很多 CNI（如 Calico, Cilium）默认开启了严格的校验，如果发现从 Pod 出来的流量源 IP/MAC 不是 CNI 分配的那一个，流量会被直接丢弃。这种情况下，使用**方案二（端口映射）**或者 **KubeVirt 模式（让 VM 彻底继承 Pod IP）** 是最稳妥的选择。

* 详细讲讲 KubeVirt 的用法

    KubeVirt 是一个将虚拟化能力带入 Kubernetes 的开源项目。它允许你在 Kubernetes 集群中像管理容器一样管理虚拟机（VM），从而实现**容器与虚拟机在同一个编排平台上的统一管理**。

    1. 核心架构与逻辑

        KubeVirt 通过 **Custom Resource Definitions (CRDs)** 扩展了 Kubernetes API。它并没有在容器内部运行一个完整的虚拟机，而是：

        * **Virt-launcher**: 这是一个 Kubernetes Pod，它在内部通过 `libvirtd` 启动一个 QEMU 进程，从而运行虚拟机。
        * **Control Plane**: 提供 `virt-controller` 和 `virt-handler`，负责监听 API 事件并控制虚拟机的生命周期。

    2. 基本工作流程

        第一步：安装 KubeVirt

        通常使用 `virtctl` 命令行工具和 Kubernetes Operator 进行安装。

        ```bash
        # 检查是否满足安装条件
        virtctl version
        # 安装 KubeVirt operator
        kubectl create -f https://github.com/kubevirt/kubevirt/releases/download/.../kubevirt-operator.yaml
        ```

        第二步：定义虚拟机 (VirtualMachine)

        在 KubeVirt 中，虚拟机通过 `VirtualMachine` (VM) 对象定义。这类似于 Deployment 定义 Pod。

        ```yaml
        apiVersion: kubevirt.io/v1
        kind: VirtualMachine
        metadata:
          name: test-vm
        spec:
          running: false # 是否立即启动
          template:
            spec:
              domain:
                devices:
                  disks:
                  - name: containerdisk
                    disk:
                      bus: virtio
                resources:
                  requests:
                    memory: 64M
              volumes:
              - name: containerdisk
                containerDisk:
                  image: kubevirt/cirros-registry-disk-demo
        ```

        第三步：启动与连接

        * **启动/停止**: 
        
            修改 `running` 字段，或使用 `virtctl start test-vm`。

        * **访问控制台**:

            ```bash
            virtctl console test-vm
            ```

    3. KubeVirt 的核心用法场景

        A. 存储集成 (DataVolumes)

        KubeVirt 使用 `DataVolume` 来简化虚拟机磁盘的导入和管理。它支持从 URL、Registry 或其他 PVC 导入镜像，并自动将其转换为虚拟机可用的磁盘格式。

        B. 网络模型

        KubeVirt 虚拟机可以直接使用 Kubernetes 的网络 (Pod Network)，也可以通过 **Multus CNI** 将虚拟机接入额外的物理网络或 VLAN，满足电信级或企业级网络需求。

        C. 热迁移 (Live Migration)

        这是 KubeVirt 的核心优势之一。你可以将运行中的虚拟机从一个节点迁移到另一个节点，而不会导致服务中断。

        * **用法**: `kubectl virt migrate <vm-name>`

    4. 关键命令行工具：`virtctl`

        `virtctl` 是 KubeVirt 的“瑞士军刀”，主要用于：

        * **管理**: `start`, `stop`, `restart`, `pause`
        * **监控/调试**: `vnc` (远程桌面), `console` (串口登录)
        * **数据操作**: `image-upload` (将本地镜像上传到集群的 PVC 中)

    5. 什么时候使用 KubeVirt？

        | 场景 | 推荐原因 |
        | --- | --- |
        | **混合工作负载** | 需要在一个集群中同时运行传统遗留应用（需 VM）和现代微服务（容器）。 |
        | **基础设施统一化** | 希望使用一套 CI/CD 流水线（如 ArgoCD/Flux）同时管理虚拟机和容器。 |
        | **边缘计算** | 在有限的硬件资源上实现计算虚拟化。 |

    **建议的学习路径**

    1. **快速部署**: 在 Minikube 或 Kind 集群中部署 KubeVirt 体验一下。
    2. **镜像准备**: 学习如何使用 `virt-builder` 或 `qemu-img` 制作 cloud-init 兼容的磁盘镜像。
    3. **高级网络**: 研究如何配置 Multus CNI 以实现虚拟机对复杂网络拓扑的接入。

    你目前是希望在现有的 Kubernetes 集群中部署虚拟机，还是想了解它在特定业务场景下的实现方案？
* Get a Shell to a Running Container

	`kubectl exec --stdin --tty shell-demo -- /bin/bash`

	`--`是防止把参数`/bin/bash`传给`kubectl`。

	如果一个 pod 有多个 container，可以使用这个：

	`kubectl exec -i -t my-pod --container main-app -- /bin/bash`

## note

Learning materials:

1. <https://www.bluematador.com/blog/safely-removing-pods-from-a-kubernetes-node>

1. <https://computingforgeeks.com/deploy-ubuntu-pod-in-kubernetes-openshift/>

1. <https://kubernetes.io/docs/concepts/workloads/controllers/deployment/>


cheatsheet: <https://kubernetes.io/docs/reference/kubectl/cheatsheet/>

k8s 官网上的资料比较散，比较乱。可以作为 reference，但不建议作为 tutorial。

安装教程：

* <https://adamtheautomator.com/installing-kubernetes-on-ubuntu/>

    这个教程上面写有`hostnamectl set-hostname master-node`，可能是这个操作造成了 journal 里提到的`master-node not found`。不知道是不是这个原因。

* <https://phoenixnap.com/kb/install-kubernetes-on-ubuntu>

    这个教程和上面那个挺像的。

command cheat sheet: <https://kubernetes.io/docs/reference/kubectl/cheatsheet/>

Kubernetes can largely be divided into Master and Node components. There are also some add ons such as the Web UI and DNS that are provided as a service by managed Kubernetes offerings (e.g. GKE, AKS, EKS).

Master components globally monitor the cluster and respond to cluster events. These can include scheduling, scaling, or restarting and unhealthy pod.

Five components make up the Master components: kube-apiserver, etcd, kube-scheduler, kube-controller-manager, and cloud-controller-manager.

* kube-apiserver: REST API endpoint to serve as the frontend for the Kubernetes control plane.

* etcd: Key value store for the cluster data (regarded as the single source of truth)

* kube-scheduler: Watches new workloads/pods and assigns them to a node based on several scheduling factors (resource constraints, anti-affinity rules, data locality, etc.)

* kube-controller-manager: Central controller that watches the node, replication set, endpoints (services), and service accounts

* cloud-controller-manager: Interacts with the underlying cloud provider to manager resources

Node Components:

* kubelet: Agent running on the node to inspect the container health and report to the master as well as listening to new commands from the kube-apiserver.

* kube-proxy: Maintains the network rules.

* container runtime: Software for running the containers (e.g. Docker, rkt, runc)

## 常用命令

启动 cluster: `minikube start`

关闭 cluster: `minikube stop`

删除 local cluster: `minikube delete`

启动 dashboard: `minikube dashboard`

创建一个 deployment: `kubectl create deployment hello-minikube --image=k8s.gcr.io/echoserver:1.4`

Exposing a service as a NodePort:

`kubectl expose deployment hello-minikube --type=NodePort --port=8080`

（不懂这一步是什么意思）

在浏览器中访问服务页面：`minikube service hello-minikube`

列出 addons: `minikube addons list`

启动一个 addon: `minikube addons enable <name>`

在集群启动时就启动 addons: `minikube start --addons <name1> --addons <name2>`

For addons that expose a browser endpoint, you can quickly open them with: `minikube addons open <name>` （不懂啥意思）

禁用 addons: `minikube addons disable <name>`

当启动`minikube start`时，`kubectl`会被 minikube 自动配置好。如果没有配置好的话，需要用 minikube 内置的 kubectl: `minikube kubectl -- <kubectl commands>`。

查看正在运行的 pods: `minikube kubectl -- get pods`

查看 pod 的详细信息：`kubectl get pods -o wide`（主要是显示每个 pod 的 ip 地址）

kubernetes 有两个主要的 services:

* NodePort

* LoadBalancer

NodePort, as the name implies, opens a specific port, and any traffic that is sent to this port is forwarded to the service.

启动 service，并且列出 service 的 ip 和 port:

`minikube service <service-name> --url`

实际上这个命令会建立一个 tunnel，把 cluster 内网的端口映射到 host 的 ip 和某个端口上。

Example:

`minikube service hello-minikube --url`

输出： `http://192.168.59.100:32463`

注：

1. `192.168.59.100`这个网段似乎用是`vboxnet0`这个虚拟网卡。

列出网络映射关系：`kubectl get svc`

minikube 是通过 host-only IP address 把地址映射到主机上的，可以通过`minikube ip`看到 ip 地址。

`minikube tunnel`是一个 LoadBalancer，在终端的前台运行，使用`Ctrl + C`来终止。

Create a Kubernetes service with type LoadBalancer：

`kubectl expose deployment hello-minikube1 --type=LoadBalancer --port=8080`

Note that without minikube tunnel, Kubernetes will show the external IP as “pending”.

如果不正常退出了 tunnel，可以使用`minikube tunnel --cleanup`清除孤儿路由（orphaned routes）。

get all existing namespaces:

`kubectl get namespaces`

check all deployments:

`kubectl get deploy`

delete a deployment: 

`kubectl delete deploy <deployment name>` (`deploy = deployment = deployments`)

print datailed information about deployments:

`kubectl describe deployments`

delete a pod:

`kubectl delete pod <podname>`

check the output of a pod:

`kubectl logs <pod name>`

## addons

### headlamp

启动：`minikube addons enable headlamp`

启动 web ui: `minikube service headlamp -n headlamp`

获得有关 nodes 的信息：`kubectl get nodes`

启动集群时设置 runtime: `minikube start --container-runtime=docker`

可以使用`minikube ssh`进入集群中。

### calico

coredns pod 需要在安装完 calico 之后才能启动。

calico 安装教程：<https://projectcalico.docs.tigera.io/getting-started/kubernetes/quickstart>

注意`custom-resources.yaml`这个资源里面有个`pod cidr`的网段地址，这个必须要和`kubeadm init`时设置的`pod cidr`网段地址一致。

## runtime class

这篇文章介绍得很好，也有一个详细的例子：<https://devopslearners.com/different-container-runtimes-and-configurations-in-the-same-kubernetes-cluster-fed228e1853e>

1. runtimeclass 似乎是用于对 device 进行虚拟化。这个 runtimeclass 只是一个名字，最后实际 work 的是 containerd 里的一个配置。这个配置似乎是个 toml 文件。目前这个文件是怎么写的，怎么起作用的，我还没弄明白。

1. runtimeclass 作为一种 k8s 的 resource，不知道是给每个 container 单独虚拟化一组设备，还是虚拟一套设备，然后让多个 containers 共享？

## 有关 container 的底层机制介绍

<https://faun.pub/kubernetes-story-linux-namespaces-and-cgroups-what-are-containers-made-from-d544ac9bd622>

## yaml 配置文件

hello-world pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:
  containers:
  - name: hello-world
    image: docker.io/library/hello-world:latest
    ports:
    - containerPort: 80
```

这个 example 运行完就会`CrashLoopBackOff`。

指定 command 和 arguments:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: command-demo
  labels:
    purpose: demonstrate-command
spec:
  containers:
  - name: command-demo-container
    image: debian
    command: ["printenv"]
    args: ["HOSTNAME", "KUBERNETES_PORT"]
  restartPolicy: OnFailure
```

运行一个 ubuntu:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ubuntu
  labels:
    app: ubuntu
spec:
  containers:
  - name: ubuntu
    image: ubuntu:latest
    command: ["/bin/sleep", "3650d"]
    imagePullPolicy: IfNotPresent
  restartPolicy: Always
```

**Get a shell to a running container**

* there is only one container inside a pod

    `kubectl exec --stdin --tty ubuntu -- /bin/bash`

* there are more than one container inside a pod

    If a Pod has more than one container, use --container or -c to specify a container in the kubectl exec command. For example, suppose you have a Pod named my-pod, and the Pod has two containers named main-app and helper-app. The following command would open a shell to the main-app container.

    `kubectl exec -i -t my-pod --container main-app -- /bin/bash`

    Note: The short options -i and -t are the same as the long options --stdin and --tty

Ref: <https://kubernetes.io/docs/tasks/debug/debug-application/get-shell-running-container/>

**Other resources**

官方文档：<https://kubernetes.io/docs/tasks/manage-kubernetes-objects/declarative-config/>，里面有一些简单的说明，有时间了看看。

一些k8s的命令参考手册：<https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#apply>

delete a pod: <https://www.bluematador.com/blog/safely-removing-pods-from-a-kubernetes-node>，可以看看这个 blog 里面还有什么有价值的东西

## k8s networking model

Materials:

1. <https://matthewpalmer.net/kubernetes-app-developer/articles/kubernetes-networking-guide-beginners.html>

1. <https://www.edureka.co/blog/kubernetes-networking/>



1. Containers in the same pod have the same ip address, and the same ethernet interface. Their MAC addresses of eth0 are the same one. Containers can commicate with each other direcly using local lookback (127.0.0.1).

    Example:

    Generate a yaml file with two containers inside one pod:

    `ubuntu_pod.yaml`:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
    name: ubuntu
    labels:
        app: ubuntu
        
    spec:
    containers:
    - name: ubuntu_1
        image: ubuntu:latest
        command: ["/bin/sleep", "3650d"]
        imagePullPolicy: IfNotPresent

    - name: ubuntu_2
        image: ubuntu:latest
        command: ["/bin/sleep", "3650d"]
        imagePullPolicy: IfNotPresent

    restartPolicy: Always
    ```

    create the pod:

    ```bash
    kubectl apply -f ubuntu_pod.yaml
    ```

    Enter the first container and second container respectively:

    ```bash
    kubectl exec -i -t ubuntu --container ubuntu-1 -- /bin/bash
    ```

    ```bash
    kubectl exec -i -t ubuntu --container ubuntu-2 -- /bin/bash
    ```

    Check their network configs:

    In `ubuntu-1`, execute `ifconfig`, output:

    ```
    eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.10  netmask 255.255.0.0  broadcast 172.17.255.255
            ether 02:42:ac:11:00:0a  txqueuelen 0  (Ethernet)
            RX packets 9472  bytes 63414951 (63.4 MB)
            RX errors 0  dropped 0  overruns 0  frame 0
            TX packets 7753  bytes 435806 (435.8 KB)
            TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

    lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
            inet 127.0.0.1  netmask 255.0.0.0
            loop  txqueuelen 1000  (Local Loopback)
            RX packets 0  bytes 0 (0.0 B)
            RX errors 0  dropped 0  overruns 0  frame 0
            TX packets 0  bytes 0 (0.0 B)
            TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
    ```

    In `ubuntu-2`, execute `ifconfig`, output:

    ```
    eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.10  netmask 255.255.0.0  broadcast 172.17.255.255
            ether 02:42:ac:11:00:0a  txqueuelen 0  (Ethernet)
            RX packets 9552  bytes 63640961 (63.6 MB)
            RX errors 0  dropped 0  overruns 0  frame 0
            TX packets 7834  bytes 441702 (441.7 KB)
            TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

    lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
            inet 127.0.0.1  netmask 255.0.0.0
            loop  txqueuelen 1000  (Local Loopback)
            RX packets 9  bytes 496 (496.0 B)
            RX errors 0  dropped 0  overruns 0  frame 0
            TX packets 9  bytes 496 (496.0 B)
            TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
    ```

    In `ubuntu-1`, use `nc` to open a tcp connection listening at localhost:

    `nc -l -s 127.0.0.1 -p 12345`

    In `ubuntu-2`, use `nc` to connect to the server:

    `nc 127.0.0.1 12345`

    Then these two containers can send messages to each other.

1. inspect the k8s networking architecture

    <https://www.alibabacloud.com/blog/how-to-inspect-kubernetes-networking_594236>

    <https://platform9.com/kb/kubernetes/how-to-identify-the-virtual-interface-of-a-pod-in-the-root-name>

    <https://www.digitalocean.com/community/tutorials/how-to-inspect-kubernetes-networking>

1. cluster ip meaning

    <https://stackoverflow.com/questions/33407638/what-is-the-cluster-ip-in-kubernetes>

## pod communication

Besides socket network communication, IPC message queue is also a available way.

Ref: <https://www.mirantis.com/blog/multi-container-pods-and-container-communication-in-kubernetes/>

## run containers in a container

Ref:

1. <https://devopscube.com/run-docker-in-docker/>

## Operators

Ref:

1. <https://developers.redhat.com/blog/2020/08/21/hello-world-tutorial-with-kubernetes-operators#>

1. <https://developers.redhat.com/articles/2021/09/07/build-kubernetes-operator-six-steps#>

1. <https://www.techtarget.com/searchitoperations/tutorial/How-to-build-a-Kubernetes-operator>

1. <https://betterprogramming.pub/build-a-kubernetes-operator-in-10-minutes-11eec1492d30>

1. <https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/kubernetes-operators-example-tutorial-k8s-docker-mariadb>

1. <https://zhuanlan.zhihu.com/p/246550722>

1. <https://zhuanlan.zhihu.com/p/515524518>

## debugging with crictl

1. <https://kubernetes.io/docs/tasks/debug/debug-cluster/crictl/>

## custom resources

<https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/>

## refs

* 有关`kubectl create`和`kubectl apply`的区别

    <https://www.containiq.com/post/kubectl-apply-vs-create>

* how to get the shell access of a pod

    <https://computingforgeeks.com/deploy-ubuntu-pod-in-kubernetes-openshift/>

    问题：如果一个 pod 中有多个 containers 该怎么办？

* 让 container 默认运行一个命令

  <https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/>

* 登陆，login

  <https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/>

  `docker login`似乎会在`$HOME`目录下生成一个类似的 cookie 的，写有登陆信息的文件，然后 k8s 会去找这个文件，并用这里的登陆信息去 pull image。

## Problem shooting

1. `0/1 nodes are available: 1 node(s) had taints that the pod didn't tolerate.`

    <https://github.com/calebhailey/homelab/issues/3>

1. 自己写 operator:

    <https://zhuanlan.zhihu.com/p/246550722>

1. 理解 runtimeclass

    <https://www.alibabacloud.com/blog/getting-started-with-kubernetes-%7C-understanding-kubernetes-runtimeclass-and-using-multiple-container-runtimes_596341>

    <https://kubernetes.io/docs/concepts/containers/runtime-class/>

    <https://itnext.io/kubernetes-running-multiple-container-runtimes-65220b4f9ef4>

    <https://devopslearners.com/different-container-runtimes-and-configurations-in-the-same-kubernetes-cluster-fed228e1853e>
