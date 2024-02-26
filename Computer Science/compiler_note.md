# compiler note

## cached

* 编译器习题 1.1 的一个实现

    ```cpp
    #include <iostream>
    #include <string.h>
    using namespace std;

    typedef const char* String;
    #define TRUE 1
    #define FALSE 0

    typedef struct tree *T_tree;
    struct tree {
        T_tree left;
        String key;
        T_tree right;
    };

    T_tree Tree(T_tree l, String k, T_tree r)
    {
        T_tree t = (T_tree) malloc(sizeof(*t));
        t->left = l;
        t->key = k;
        t->right = r;
        return t;
    }

    T_tree insert(String key, T_tree t)
    {
        if (t == NULL)
            return Tree(NULL, key, NULL);
        else if (strcmp(key, t->key) < 0)
            return Tree(insert(key, t->left), t->key, t->right);
        else if (strcmp(key, t->key) > 0)
            return Tree(t->left, t->key, insert(key, t->right));
        else
            return Tree(t->left, key, t->right);
    }

    bool member(String key, T_tree root)
    {
        if (root == NULL)
            return FALSE;

        if (strcmp(key, root->key) == 0)
            return TRUE;
        else if (strcmp(key, root->key) < 0)
            return member(key, root->left);
        else
            return member(key, root->right);
    }

    T_tree insert(String key, void *binding, T_tree t);
    void *lookup(String key, T_tree t);

    int main()
    {
        T_tree root = NULL;
        root = insert("hello", root);
        root = insert("world", root);
        root = insert("nihao", root);
        root = insert("zaijian", root);

        String key = "zaijian";
        bool found = member(key, root);
        if (found)
            printf("member %s exists.\n", key);
        else
            printf("member %s doesn't exist.\n", key);

        key = "haha";
        found = member(key, root);
        if (found)
            printf("member %s exists.\n", key);
        else
            printf("member %s doesn't exist.\n", key);

        return 0;
    }
    ```

    输出：

    ```
    member zaijian exists.
    member haha doesn't exist.
    ```

    这段代码，每次`insert()`，都会返回一个全新的树，所以习题上才说旧的树还可继续用于查找。

    这个`insert()`不会插入重复的元素，如果某个元素已经存在，那么就返回原树的一个副本。

    `typedef struct tree *T_tree;`实际上就是指定了一个新类型的指针，有些代码可能没有`struct tree`的定义，纯粹是为了区分类型。这样的操作常见于句柄。

    `T_tree t = (T_tree) malloc(sizeof(*t));`这种写法还是第一次见，看来等号左边写出来的变量，右边就可以直接用了。或者说，这个语句，其实是声明和赋值的结合体。

    b 小题没看懂。

* 非确定有限自动机（NFA）是一种需要对从一个状态出发的多条标有相同符号的边进行选择的自动机。

    比如对于初始状态$s_0$，它向外有 2 条边，每条边的条件都是字母`a`，由此可以得到两个完全不同的终止条件。

    标有$\epsilon$的边可以在不接收输入字符的情况下进行状态转换。

## notes

使用 bison 和 flex 创建一个简易计算器。

创建一个工程文件夹，在文件夹中创建一个新文件`calc.y`，写入以下内容：

`calc.y`:

```y
%{
    #include <stdio.h>
    #include <assert.h>
    static int Pop();
    static int Top();
    static void Push(int val);
%}

%token T_Int

%%

S : S E '\n' { printf("= %d\n", Top()); }
  |
  ;
E : E E '+' { Push(Pop() + Pop()); }
  | E E '-' { int op2 = Pop(); Push(Pop() - op2); }
  | E E '*' { Push(Pop() * Pop()); }
  | E E '/' { int op2 = Pop(); Push(Pop() / op2); }
  | T_Int   { Push(yylval); }
  ;

%%

static int stack[100], count = 0;
static int Pop() {
    assert(count > 0);
    return stack[--count];
}
static int Top() {
    assert(count > 0);
    return stack[count-1];
}
static void Push(int val) {
    assert(count < sizeof(stack)/sizeof(*stack));
    stack[count++] = val;
}
int main() {
    return yyparse();
}
```

再创建一个`calc.l`，输入以下内容：

`calc.l`:

```l
%{
    #include "y.tab.h"
%}
%%
[0-9]+   { yylval = atoi(yytext); return T_Int;}
[-+*/\n] { return yytext[0];}
.        { /* ignore everything else */ }
```

创建一个 makefile 文件：

`Makefile`:

```makefile
calc: lex.yy.o y.tab.o
	gcc -o calc lex.yy.o y.tab.o -ly -ll

lex.yy.c: calc.l y.tab.c
	flex calc.l

y.tab.c: calc.y
	bison -vdty calc.y

clean:
	rm -f lex.yy.c y.tab.c y.tab.h lex.yy.o y.tab.o calc
```

在生成解析器前，需要先安装`bison`和`flex`：

```bash
sudo apt install bison flex
```

最后执行 makefile 就可以了：

```bash
make
```

此时会生成一个`calc`文件，运行它：

```bash
./calc
```

然后输入：

```
1 2 +
```

按回车后，可以看到程序的输出为

```
= 3
```

表示编译器运行成功。

