# SQL Note

首次进入 mysql：`sudo mysql -u root -p`

常用命令：

`show databases;`

`source /path/to/xxx.sql`

select: `select select_list from table_name;`

启动：

`sudo service mysql start`

`sudo /etc/init.d/mysql start`

`sudo systemctl start mysqld`

重启：

`service mysql restart`或`service mysqld restart`

`/etc/init.d/mysqld restart`

用户与权限管理：

`create user account_name identified by 'password';`

`account_name`有两部分：`username@hostname`，其中`hostname`表示从哪里连接到 server 上，如果被省略，则表示可以从任意地方连接。被省略的`hostname`也可以写成：`username@%`。

如果`username`和`hostname`中有特殊字符，还需要把它们用引号或反引号或双引号括起来：`'username'@'hostname'`。

`create user`只能创建出没有权限的用户，如果需要权限，还需要使用`grant`命令。

显示当前的所有 users：`select user from mysql.user`

给用户赋予权限：`grant privilege [,privilege],.. on privilege_level to account_name;`

例子：

1. `grant select on employees to bob@localhost;`：将当前 database 的`employees`表的`select`权限赋给`bob`。

1. `grant insert, update, delete on employees to bob@localhost;`

权限等级：

1. global privileges: `*.*`，所有权限
1. database privileges: `database_name.*`，数据库权限
1. table privileges: `database_name.table_name`，表权限。如果不写`database_name`，那么 mysql 使用默认的 database，如果没有默认的 database，mysql 会报错。
1. column privileges: 必须用命令指定 columns

    ```sql
    grant select (employeeNumber, lastName, firstName, email), update (lastName) on employees to bob@localhost;
    ```

1. stored routine privileges: 用于过程和函数

    ```sql
    grant execute on procedure CheckGredit to bob@localhost;
    ```

1. proxy user privileges: allow one user to be a proxy for another. The proxy user gets all privileges of the proxied user.

    ```sql
    grant proxy on root to alice@localhost;
    ```

    In this example, `alice@localhost` assumes all privileges of `root`.

如果想授予别人权限，需要满足两个条件：

1. 首先自己必须有`grant option`权限
1. 自己要有待授予的权限

如果`read_only`系统变量启动，那么还必须有`super`权限才能执行`grant`语句。

显示一个用户的权限：`show grants for super@localhost;`

`usage`权限表示用户可以 log in database，但是没有权限。

详细的权限列表可以参考这篇文章的末尾表格：<https://www.mysqltutorial.org/mysql-grant.aspx>

用新用户登录：`mysql -u hlc -p`

## 数据相关操作

`select`可以从 table 中查询查询 columns：

`select select_list from table_name;`

Examples:

```sql
select lastName from employees;
select lastName, firstName, jobTitle from employees;
select * from employees;
```

（如果是在其他编程语言中调用 mysql，那就不能再使用星号`*`了）

### functions

```sql
select 1 + 1;
select now();
select concat('John', ' ', 'Doe');
```

`select select_list from dual;`，`dual`是一个语法补充，这条语句并不生效，只在不得不用到`select`的地方用。

mysql 会将 result set 的名字定为`select`后面的语句，如果我们想定另外个名字，可以这样：

```sql
select expression as column_alias;
select expression column_alias;
```

Examples:

```sql
select concat('John', ' ', 'Doe') as name;
select concat('Jane', ' ', 'Doe') as 'Full name';
```

### order by

```sql
select
    select_list
from
    table_name
order by
    column1 [ASC|DESC],
    column2 [ASC|DESC],
    ...;
```

`order by`默认使用`asc`。

还可以给结果定一个别名：

```sql
select
    orderNumber,
    orderLineNumber,
    quantityOrdered * priceEach as subtotal
from
    orderdetails
order by subtotal desc;
```

**`field`**

```sql
field(str, str1, str2, ...)
```

The `field()` function returns the position of the str in the `str1`, `str2`, ... list. If the `str` is not in the list, the `field()` function returns 0.

例子：

```sql
select field('B', 'A', 'B', 'C');
```

这条语句返回`2`。

这样我们就可以按指定的顺序排序了：

```sql
select
    orderNumber, status
from
    orders
order by field(status,
    'In Process',
    'On Hold',
    'Cancelled',
    'Resolved',
    'Disputed',
    'Shipped');
```

如果数据中有`null`值，那么`null`会出现在升序顺序的最前面。

### where

```sql
select
    select_list
from
    table_name
where
    search_condition;
```

优先级：

`from -> where -> select -> order by`

Examples:

```sql
select
    lastname,
    firstname,
    jobtitle
from
    employees
where
    jobtitle = 'sales rep';

select
    lastname,
    firstname,
    jobtitle,
    officeCode
from
    employees
where
    jobtitle = 'Sales Rep' and
    officeCode = 1;

select
    lastName,
    firstName,
    jobTitle,
    officeCode
from
    employees
where
    jobtitle = 'Sales Rep' or
    officeCode = 1
order by
    officeCode,
    jobTitle;
```

还可以用`between`表示一个范围：

`expression between low and high`

example:

```sql
select
    firstName,
    lastName,
    officeCode
from
    employees
where
    officeCode between 1 and 3
order by officeCode;
```

还可以用`like`来匹配 pattern。其中`%`匹配 any string of zero or more characters，`_`匹配 ang single character。

```sql
select
    firstName,
    lastName
from
    employees
where
    lastName like '%son'
order by firstName;
```

还可以用`in`来匹配列表中的项：

`value in (value1, value2, ...)`

example:

```sql
select
    firstName,
    lastName,
    officeCode
from
    employees
where
    officeCode in (1, 2, 3)
order by
    officeCode;
```

检测一个值是否为`null`时，需要用`is null`运算符，而不是`=`：

`value IS NULL`

一些常用的运算符：

`=`, `<>`或`!=`, `<`, `>`, `<=`, `>=`

### distinct

`distinct`可以消除重复的行。

```sql
select distinct
    select_list
from
    table_name
where
    search_condition
order by
    sort_expression;
```

如果只有单个 column，那么按单列去重；如果有多个 columns，那么按多列去重。

执行顺序：

`from -> where -> select -> distinct -> order by`

`distinct`会认为所有的`null`是同一个值。

### operators

`and`：如果两个操作数（operands）或表达式都是`1`，则返回`1`，若其中有一个`null`，则返回`null`，其余情况返回`0`。

`and`还有 short-circuit evaluation 的特性。

`or`也有类似的特性。

对于`value in (value1, value2, value3, ...)`，当`value`是`null`时，`in`返回`null`。当 value doesn't equal any value in the list and one of values in the list is `NULL` 时，也返回`NULL`。

注意`null`和`null`并不相等。

`value not in (value1, value2, value3)`表示不在列表中则返回 1。当`value`是`null`时，`not in`返回`null`。

有关`between`的语法：

`value between low and high;`

当`value`, `low`, or `high` is `NULL`, the `BETWEEN` operator returns `NULL`.

`value not between low and high`

一些查询日期的例子：

```sql
select
    orderNumber,
    requiredDate,
    status
from
    orders
where
    requireddate between
        cast('2003-01-01' as date) and
        cast('2003-01-31' as date);
```

有关`like`：

`expression LIKE pattern ESCAPE escape_character`

如果想将`pattern`中含有的特殊字符视为正常的查询字符，可以使用`escape`。默认的 excape letter 是`\`。

`not like pattern`

**`limit`**

```sql
select
    select_list
from
    table_name
limit [offset,] row_count;
```

其中`offset` specifies the offset of the first row to return. The `offset` of the first row is `0`, not `1`. The `row_count` specifies the maximum number of rows to return.

也可以用下面的语法：

`LIMIT row_count OFFSET offset`

通常`select`返回的值是没有特定顺序的，所以我们将`limit`用在`order by`后：

```sql
select
    select_list
from
    table_name
order by
    sort_expression
limit offset, row_count;
```

如果将`limit`和`distint`联合使用，那么当找到`limit`指定数量的 unique rows 后，就会停止搜索。

`value IS NULL`判断一个值是否为`NULL`, `value IS NOT NULL`判断一个值是否不是`NULL`。

`is null`还可以识别日期`0000-00-00`。

### Joining tables

**alias**

MySQL supports two kinds of aliases: column alias and table alias.

column alias:

```sql
select
    [column_1 | expression] as descriptive_name
from table_name;
```

其中`as`可以省略。如果`descriptive_name`中有空格，需要将其用单引号括起来。

examples:

```sql
select
    concat_ws(', ', lastName, firstName) as 'Full name'
from
    employees;
```

`concat_ws`会选择表中的`lastName`和`firstName`两列，用逗号拼接成一列。`as`会给这一列起一个别名`Full name`。


```sql
select
    concat_ws(', ', lastName, firstName) 'Full name'
from
    employees
order by
    'Full name';


select
    orderNumber 'Order no.',
    sum(priceEach * quantityOrdered) total
from
    orderDetails
group by
    'Order no.'
having
    total > 60000;
```

注意我们无法在`where`中用 alias，因为`where`先于`select`执行。

alias for tables:

`table_name as table_alias`

example:

```sql
select * from employees e;
```

一旦一个`table`被赋以别名，就可以用`table_alias.column_name`访问了：

```sql
select
    e.firstName,
    e.lastName
from
    employees e
order by e.firstName;
```

当两个 table 中都含有同一个 column name 时，就需要用 table name 来加以区分了：

```sql
select
    customerName,
    count(o.orderNumber) total
from
    customers c
inner join orders o on c.customerNumber = o.customerNumber
group by
    customerName
order by
    total desc;
```

通常一个 column 可能分开保存在多个 table 中，为了查询某个 column 所有的信息，需要用`join`。

inner join:

```sql
select column_list
from table_1
inner join table_2 on join_condition;
```

The inner join clause compares each row from the first table with every row from the second table. In other words, the inner join clause includes only matching rows from both tables.

example:

```sql
select
    m.member_id,
    m.name as member,
    c.committee_id,
    c.name as committee
from
    members m
inner join committees c on c.name = m.name;
```

If the join condition uses the equality operator (`=`) and the column names in both tables used for matching are the same, and you can use the `using` clause instead.

```sql
select column_list
from table_1
inner join table_2 using (column_name);
```

example:

```sql
select
    m.member_id,
    m.name as member,
    c.committee_id,
    c.name as committee
from
    members m
inner join committees using(name);
```

**`left join`**

```sql
select column_list
from table_1
left join table_2 on join_condition;
```

If the values in the two rows are not matched, the left join clause still creates a new row whose columns contain columns of the row in the left table and `NULL` for columns of the row in the right table.

In other words, the left join selects all data from the left table whether there are matching rows exist in the right table or not.

The left join also supports the `using` clause.

```sql
select column_list
from table_1
left join table_2 using (column_name);
```

To find member who are not the committee members, you add a `WHERE` clause and `IS NULL` operator as follows:

```sql
select
    m.member_id,
    m.name as member,
    c.committee_id,
    c.name as committee
from
    members m
left join committees c using(name)
where c.committee_id is null;
```

**`right join`**

The `right join` clause selects all rows from the right table and matches rows in the left table.

```sql
select column_list
from table_1
right join table_2 on join_condition;

select column_list
from table_1
right join table_2 using (column_name);
```

**`cross join`**

`cross join` clause does not have a join condition. The cross join makes a Cartersian product of rows from the joined tables. The cross join combines each row from the first table with every row from the right table to make the result set.

```sql
select select_list
from table_1
cross join table_2;
```

example:

```sql
select
    m.member_id,
    m.name as member,
    c.committee_id,
    c.name as committee
from
    members m
cross join committees c;
```

`cross join`不可以用`on`或`using`。如果使用`where`的话，`cross join`就像`inner join`一样了：

```sql
select * from t1
cross join t2
where t1.id = t2.id;
```

**inner join**

```sql
select
    select_list
from t1
inner join t2 on join_condition1
inner join t3 on join_condition2
...;
```

The `inner join` clause compares each row in the `t1` table with every row in the `t2` table based on the join condition.

```sql
select
    orderNumber,
    productNumber,
    msrp,
    priceEach
from
    products p
inner join orderdetails o
    on p.productcode = o.productcode
        and p.msrp > o.priceEach
where
    p.productcode = 'S10_1678';
```

For `inner join` clause, the condition in the `on` clause is equivalent to the condition in the `where` clause.

**self join**

The self join is often used to query hierarchical data or to compare a row with other rows within the same table.

在 self join 时必须要给 table 创建一个别名。如果没有别名，引用一个 table 两次是会报错的。

example:

```sql
select
    concat(m.lastName, ', ', m.firstName) as Manager,
    concat(e.lastName, ', ', e.firstName) as 'Direct report'
from
    employees e
inner join employees m on
    m.employeeNumber = e.reportsTo
order by
    manager;
```

self join 还可以使用`left join`和`right join`来实现。

另外一个例子，显示出居住在同一个城市的居民：

```sql
select
    c1.city,
    c1.customerName,
    c2.customerName
from
    customer c1
inner join customer c2 on
    c1.city = c2.city
    and c1.customerName > c2.customerName
order by
    c1.city;
```

其中`c1.customerName > c2.customerName`保证不会有相同的 customer。

### group by

```sql
select
    c1, c2, ..., cn, aggregate_function(ci)
from
    table_name
where
    where_conditions
group by c1, c2, ..., cn;
```

执行顺序：

`from -> where -> group by -> select -> distinct -> order by -> limit`

Example:

```sql
select
    status
from
    orders
group by status;
```

其效果类似于：

```sql
select distinct
    status
from
    orders;
```

using group by with aggregate functions:

use `count` function with the `group by` clause to know the number of orders in each status:

```sql
select
    status, count(*)
from
    orders
group by status;
```

other examples which joining two tables together to query more information:

```sql
select
    status,
    sum(quantifyOrdered * priceEach) as amount
from
    orders
inner join orderdetails
    using (orderNumber)
group by
    status;
```

```sql
select
    year(orderData) as year,
    sum(quantityOrdered * priceEach) as total
from
    orders
inner join orderdetails
    using (orderNumber)
where
    status = 'Shipped'
group by
    year(orderDate);
```

Note that the expression which appears in the `select` clause must be the same as the one in the `GROUP BY` clause.

To filter the groups returned by `GROUP BY`, we can use a `HAVING` clause.

```sql
select
    year(orderDate) as year,
    sum(quantityOrdered * priceEach) as total
from
    orders
inner join orderdetails
    using (orderNumber)
where
    status = 'Shipped'
group by
    year
having
    year > 2003;
```

The SQL standard does not allow you to use an alias in the `GROUP BY` clause whereas MySQL supports this:

```sql
select
    year(orderDate) as year,
    count(orderNumber)
from
    orders
group by
    year;
```

MySQL also allows you to sort the groups in ascending or descending orders. The default sorting order is ascending.

```sql
select
    status,
    count(*)
from
    orders
group by
    status desc;
```

If you use the `GROUP BY` clause in the `SELECT` statement without using aggregate functions, the `GROUP BY` clause behaves like the `DISTINCT` clause.

```sql
select
    state
from
    customers
group by state;


select distinct
    state
from
    customers;
```

Generally speaking, the `DISTINCT` clauses is a special case of the `GROUP BY` clause. The difference between `DISTINCT` clause and `GROUP BY` clauses is that the `GROUP BY` clause sorts the result set, whereas the `DISTINCT` clause does not. (Notice that MySQL 8.0+ removed the implicit sorting for the `group by` clause.)

**having**

The `having` clause is used in the `select` statement to specify filter conditions for a group of rows or aggregates. If you omit the `group by` clause, the `having` clause behaves like the `where` clause.

```sql
select
    select_list
from
    table_name
where
    search_condition
group by
    group_by_expression
having
    group_condition;
```

Notice that the `having` clause applies a filter condition to each group of rwos, while the `where` clause applies the filter condition to each individual row.

processing flow:

```
from -> where -> group by -> having -> select -> distinct -> order by -> limit
```

example:

```sql
select
    orderNumber,
    sum(quantityOrdered) as itemsCount,
    sum(priceeach * quantityOrdered) as total
from
    orderDetails
group by
    orderNumber
having
    total > 1000;
```

还可以使用逻辑运算连接多个语句：

```sql
having
    total > 1000 and
    itemsCount > 600;
```

还可以和`join`合起来使用：

```sql
select
    a.orderNumber,
    status,
    sum(priceEach * quantityOrdered) total
from
    orderDetails a
inner join orders b
    on b.orderNumber = a.orderNumber
group by
    orderNumber,
    status
having
    status = 'Shipped' and
    total > 1500;
```

我们可以使用`union all`把两个 grouping sets 合并成一个：

```sql
select
    productline,
    sum(orderValue) totalOrderValue
from
    sales
group by
    productline;


select
    sum(orderValue) totalOrderValue
from
    sales;


select
    productline,
    sum(orderValue) totalOrderValue
from
    sales
group by
    productline
union all
select
    null,
    sum(orderValue) totalOrderValue
from
    sales;
```

但是`union all`需要列的数量对齐，所以加了个`null`。

`union all`的效率比较低，所以使用`rollup`

```sql
select
    select_list
from
    table_name
group by
    c1, c2, c3 with rollup;
```

The `rollup` generates multiple grouping sets based on the columns or expressions specified in the `GROUP BY` clause.

```sql
select
    productLine,
    sum(orderValue) totalOrderValue
from
    sales
group by
    productline with rollup;
```

The `rollup` clause generates not only the subtotals but also the grand total of the order values.

If you have more than one column specified in the `group by` clause, the `rollup` clause assumes a hierarchy among the input columns.