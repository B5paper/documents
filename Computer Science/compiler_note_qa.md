# compiler note qa

[unit]
[u_0]
请说出正则表达式的 5 种组成形式。
[u_1]
* 符号（symbol）

    symbol $\bold a$ 可以匹配任意包含$\bold a$的字符串。

* 可选（alternation）

    对于两个正则表达式$M$，$N$，$M | N$可以形成一个新的正则表达式。只要一个字符串包含$M$或$N$其中的一个，就可以匹配得上。

* 联结（concatenation）

    $M \cdot N$表示$M$后紧跟$N$。

* $\epsilon$（epsilon）

    表示空字符串。

* 重复（repetition）

    对于给定的正则表达式$M$，它的克林（Kleene）闭包是$M^*$。

    如果一个字符串是由$M$中的字符串经零至多次联结运算的结果，则该字符串属于$M^*$。

[unit]
[u_0]
写一个 flex 程序，统计输入中的 number 数量。
[u_1]
```lex
%{
int digit_count = 0;
%}

%%
[0-9]+ {digit_count++;}
. {}
\n {return 0;}
%%

int yywrap(){}

int main()
{
	yylex();
	printf("total %d numbers.\n", digit_count);
	return 0;
}
```

compile:

```bash
flex count_num.lex
gcc lex.yy.c -o count_num
```

run:

`./count_num`

input example:

```
123 345 hehe #@!
```

output:

```
total 2 numbers.
```