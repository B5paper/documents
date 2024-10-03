# NFS Note

## cache

* 如果 nfs server 在 export 目录软链接其他路径的目录/文件，那么 client 的 nfs 目录里的软链接会链到 client 的文件目录上，不会读取 server 的软链接的内容

## note

(empty)