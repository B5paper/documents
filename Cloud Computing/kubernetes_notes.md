# kubernetes notes

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