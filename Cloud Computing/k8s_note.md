# Kubernetes Note

materials:

1. github awesome: <https://github.com/caicloud/kube-ladder#%E7%9B%AE%E6%A0%87-5>

1. github asesome: <https://github.com/0voice/k8s_awesome_document>

1. hand book: <https://www.bookstack.cn/read/kubernetes-handbook/ch05>

1. dive into kubeadm: <https://zhuanlan.zhihu.com/p/74922114>

1. kubectl commands: <https://zhuanlan.zhihu.com/p/145268395>

1. k8s hand book: <https://github.com/Leverege/kubernetes-book>

有个 k8s api 的教程，写得还挺好：<https://www.containiq.com/post/kubernetes-api>

Ubuntu 20.04 安装过程中遇到的问题:

* `/proc/sys/net/bridge/bridge-nf-call-iptables not found`

    Ref: <https://github.com/weaveworks/weave/issues/2789>

* `proc/sys/net/ipv4/ip_forward`

    Ref: <https://netref.soe.ucsc.edu/node/19>

## k8s 中 yaml 文件的格式

required fileds:

* `apiVersion`

    作用：
    
    > Which version of the Kubernetes API you’re using to create this object.

* `kind`

    作用：
    
    > What kind of object you want to create.

    常见的`kind`有`Pod`, `Deployment`。

* `metadata`

    用于唯一地定位资源：
    
    > Data that helps uniquely identify the object, including a `name` string, `UID`, and optional `namespace`.

    常见的可选字段：

    * `name`
    * `UID`
    * `namespace`
    * `labels`

* `spec`

    定义 pod 类型的资源时，使用的一个 spec 例子：

    ```yaml
    spec:

    containers:

        - name: front-end

        image: nginx

        ports:

            - containerPort: 80

        - name: rss-reader
    ```

    spec 里常用的字段有：

    * `name`
    * `image`
    * `command`
    * `args`
    * `workingDir`
    * `ports`
    * `env`
    * `resources`
    * `volumeMounts`
    * `livenessProbe`
    * `readinessProbe`
    * `lifecycle`
    * `terminationMessagePath`
    * `imagePullPolicy`
    * `securityContext`
    * `stdin`
    * `stdinOnce`
    * `tty`

### deployment

```yaml
---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: rss-site
  labels:
    app: web
spec:
  replicas: 2
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: front-end
          image: nginx
          ports:
            - containerPort: 80
        - name: rss-reader
          image: nickchase/rss-php-nginx:v1
          ports:
            - containerPort: 88
```

## runtimeclass

有关 runtimeclass 的资料：

* <https://devopslearners.com/different-container-runtimes-and-configurations-in-the-same-kubernetes-cluster-fed228e1853e>

## Frequently used commands

* `kubectl wait`

    Ref:

    1. <https://vadosware.io/post/so-you-need-to-wait-for-some-kubernetes-resources/>

    1. <https://stackoverflow.com/questions/71858436/kubectl-wait-until-pod-is-gone-terminating>

    Example:

    ```bash
    kubectl wait --for=condition=ready pod -l app=blog --timeout=60s
    ```

## Miscellaneous

* Why do pod names have 5 random alphanumeric characters

    Ref: <https://stackoverflow.com/questions/54489128/why-do-pod-names-have-5-random-alphanumeric-characters-appended-to-their-name-wh>

* 
