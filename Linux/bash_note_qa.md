# bash note

[u_0]
遍历输出 1 到 10，每行一个数字
[u_1]
```bash
counter=1
while [ $counter -le 10 ]
do
    echo $counter
    ((counter++))
done
```

[u_0]
使用`{}`遍历输出 1 到 10，每行一个数字
[u_1]
```bash
for value in {1..10}
do
    echo $value
done
```