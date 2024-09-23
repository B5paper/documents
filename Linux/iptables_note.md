# iptables note

* iptables relatives

    * chain

        chain 是一系列匹配规则（rule）的集合。

        问题：

        * 是否一条 chain 没有处理的包会被传递给下一条 chain？

    * 猜想：所有的 iptables 命令都需要 sudo 权限

    * 列出当前的规则：`iptables --list`

    * 如何启用一个指定的 table？

    * 如何列出所有的 table ?

        table 的一共就 5 个，且是固定的，不可添加，不可删除，因此没有动态列出所有 table 的需求。

        可用的 table 有：

        `filter`, `nat`, `mangle`, `raw`, `security`

    * 默认进入的 table 是`filter`

    * `filter` table 中可以加入 prerouting, postrouting 这些 chain 吗？

    * 既然有 custom defined chain，那么说明 built-in chain 并不具有特殊功能

    * 所有的 table 都共同在工作吗？是否存在“启用”某个 table，禁用某个 table 的操作？