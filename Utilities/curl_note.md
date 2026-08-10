# Curl Note

* curl 默认读取的是小写的 http_proxy 和 https_proxy，而不是大写的 HTTP_PROXY

发送 post 请求：

`curl --data "post1=value1&post2=value2&etc=valetc" http://host/resource`

或使用 RESTful API：

`curl -X POST -d @file http://host/resource`
