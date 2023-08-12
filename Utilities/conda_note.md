# Conda

搜索某个包：`conda search -f <package_name>`

列出已经安装的所有的包：`conda list`

列出所有可安装的包：`conda search`

安装一个特定版本的包：`conda install pandas=1.0.2`

1. python 3.4 安装不上

    似乎是因为较老的包都放到了 conda-forge channel 里。

    Ref: <https://stackoverflow.com/questions/56850972/why-conda-cannot-create-environment-with-python-3-4-installed-in-it>