# tar note

## cache

* tar 在打包目录时，要求目录中的文件不能有改动。如果还没来得及打包的文件被修改，那么就会报错。

    `tar: file changed as we read it`

## note

解压时指定目录：

`tar -xvf articles.tar --directory /tmp/my_articles/`

这个好像也行：

`tar -zvxf documents.tgz -C /tmp/tgz/ `
