# python note qa

[unit]
[u_0]
使用 python 中`re`模块的`search`函数提取字符串`hello, world`中的`world`字符串。
[u_1]
```python
import re
txt = 'hello, world'
pat_1 = re.compile('world')
m = pat_1.search(txt)
start_pos = m.start()
end_pos = m.end()
selected_txt = txt[start_pos:end_pos]
print(selected_txt)  # world
```

[unit]
[u_0]
python 的`re`模块中，`search()`和`match()`有什么异同？
[u_1]
`search()`表示从指定位置开始匹配，`match()`表示从头开始匹配。
`search()`和`match()`都会返回一个`Match`类型的对象作为匹配结果。

[unit]
[u_0]
使用 python 中`re`模块的`finditer()`函数匹配字符串`abcbacaccba`中`a.{2}`的不重叠子串，即`a`后跟两个非换行任意字符。
[u_1]
```python
txt = 'abcbacaccba'
pat_2 = re.compile('a.{2}')
for m in pat_2.finditer(txt):
	start_pos = m.start()
	end_pos = m.end()
	selected_txt = txt[start_pos:end_pos]
	print(selected_txt)  # [abc, aca]
```
