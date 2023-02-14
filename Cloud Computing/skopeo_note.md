# Skopeo Note

skopeo 用于在不拉取镜像的条件下，获取镜像的哈希值，layer 等信息，也可以对镜像进行复制和删除。

安装: `apt install skopeo`

Home page: <https://github.com/containers/skopeo>

* Show properties

    display the digest (sha256) of the image (maybe this is the sha256 of manifest), all tags of the image, and layer sha256.

    Example:

    `skopeo inspect docker://registry.fedoraproject.org/fedora:latest`

    `skopeo inspect docker://registry.fedoraproject.org/fedora:latest | jq '.Digest'`

    `skopeo inspect --config docker://registry.fedoraproject.org/fedora:latest`

* list tags

    `skopeo list-tags docker://quay.io/kata-containers/confidential-containers`

Notice:

1. <quay.io> 的 repo 的 url 是这样的：<https://quay.io/repository/kata-containers/confidential-containers>，不能把它直接复制成 skopeo 的镜像地址，因为其中有一个 repository。

