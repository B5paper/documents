# iptables note

* iptables 向指定 chain 中添加/删除 rule

    * `iptables -t filter --append INPUT -j DROP`

        向`filter` table 中的`INPUT` chain 添加`DROP` rule，效果如下：

        ```
        (base) hlc@hlc-VirtualBox:~$ sudo iptables --list
        Chain INPUT (policy ACCEPT)
        target     prot opt source               destination         
        DROP       all  --  anywhere             anywhere            

        Chain FORWARD (policy DROP)
        target     prot opt source               destination         
        DOCKER-USER  all  --  anywhere             anywhere            
        DOCKER-ISOLATION-STAGE-1  all  --  anywhere             anywhere            
        ACCEPT     all  --  anywhere             anywhere             ctstate RELATED,ESTABLISHED
        DOCKER     all  --  anywhere             anywhere            
        ACCEPT     all  --  anywhere             anywhere            
        ACCEPT     all  --  anywhere             anywhere            

        Chain OUTPUT (policy ACCEPT)
        target     prot opt source               destination         

        Chain DOCKER (1 references)
        target     prot opt source               destination         

        Chain DOCKER-ISOLATION-STAGE-1 (1 references)
        target     prot opt source               destination         
        DOCKER-ISOLATION-STAGE-2  all  --  anywhere             anywhere            
        RETURN     all  --  anywhere             anywhere            

        Chain DOCKER-ISOLATION-STAGE-2 (1 references)
        target     prot opt source               destination         
        DROP       all  --  anywhere             anywhere            
        RETURN     all  --  anywhere             anywhere            

        Chain DOCKER-USER (1 references)
        target     prot opt source               destination         
        RETURN     all  --  anywhere             anywhere            
        ```

    * `iptables -t filter --delete INPUT 1`

        删除`filter` table 中的`INPUT` chain 中的第 1 条 rule。

        ```
        (base) hlc@hlc-VirtualBox:~$ sudo iptables -t filter --delete INPUT 1
        (base) hlc@hlc-VirtualBox:~$ sudo iptables --list
        Chain INPUT (policy ACCEPT)
        target     prot opt source               destination         

        Chain FORWARD (policy DROP)
        target     prot opt source               destination         
        DOCKER-USER  all  --  anywhere             anywhere            
        DOCKER-ISOLATION-STAGE-1  all  --  anywhere             anywhere            
        ACCEPT     all  --  anywhere             anywhere             ctstate RELATED,ESTABLISHED
        DOCKER     all  --  anywhere             anywhere            
        ACCEPT     all  --  anywhere             anywhere            
        ACCEPT     all  --  anywhere             anywhere            

        Chain OUTPUT (policy ACCEPT)
        target     prot opt source               destination         

        Chain DOCKER (1 references)
        target     prot opt source               destination         

        Chain DOCKER-ISOLATION-STAGE-1 (1 references)
        target     prot opt source               destination         
        DOCKER-ISOLATION-STAGE-2  all  --  anywhere             anywhere            
        RETURN     all  --  anywhere             anywhere            

        Chain DOCKER-ISOLATION-STAGE-2 (1 references)
        target     prot opt source               destination         
        DROP       all  --  anywhere             anywhere            
        RETURN     all  --  anywhere             anywhere            

        Chain DOCKER-USER (1 references)
        target     prot opt source               destination         
        RETURN     all  --  anywhere             anywhere
        ```

    * 列出每条 rule 的索引

        `iptables --list --line-number`

    * iptables 的 man page，有时间了看看

        <https://sites.uclouvain.be/SystInfo/manpages/man8/iptables.8.html>

* iptables 中的 policy 指的是当没有 rule 可以匹配时的默认 target (action)

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