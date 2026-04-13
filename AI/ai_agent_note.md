* 给我讲讲常用的搜索技巧，比如 grep 之类的

  * 常用思路

      先定目标，再选工具：

      - 找文件名：rg --files
      - 找文本：rg
      - 看上下文：rg -n -C 5
      - 只想知道“有没有”：rg -q
      - 按类型过滤：rg -tcpp
      - 结果太多时，先缩目录，再缩关键词，再加排除

      最常用的是 rg, 比 grep 快很多，默认还会尊重 .gitignore。

      常用例子：

      rg "workHead"
      rg -n "sipuLaunchKernel" src/
      rg -n -C 3 "remoteChip" src/
      rg -tcpp "devPeers"
      rg -g '!build' -g '!logs' "channel->devPeers"
      rg --files | rg 'CMakeLists|enqueue|channel'

      高频参数

      - -n：显示行号
      - -C 3：前后各 3 行上下文
      - -A 5 / -B 5：只看后/前文
      - -i：忽略大小写
      - -w：整词匹配
      - -F：按普通字符串匹配，不用正则
      - -g：文件路径过滤
      - -t：按文件类型过滤
      - --hidden：搜隐藏文件
      - -S：智能大小写
      - -U：跨行/不过滤二进制时才偶尔用
      - -c：只看每个文件匹配次数
      - -l：只看命中文件名
      - --files：列文件，不搜内容

      什么时候用 grep: 机器上没有 rg 时再退回去。

      grep -RIn "workHead" src
      grep -RIn --include='*.cc' "remoteChip" src
      grep -RIn --exclude-dir=build "devPeers" .

      路径过滤技巧

      只在某些目录搜：

      rg -n "ncclWork" src/enqueue.cc src/init.cc src/include

      排除目录：

      rg -n "remoteChip" . -g '!build' -g '!logs'

      只搜某类文件：

      rg -n "ptrExchange" -tcpp
      rg -n "CMAKE_" -tcmake

      **先粗后细**
      
      比如排查空指针：

      1. 先搜结构体定义

      rg -n "struct ncclDevChannelPeer|struct ncclConnInfo" src/include

      2. 再搜赋值点

      rg -n "devPeers|devPeersHostPtr|ptrExchange" src

      3. 再看 launch 点

      rg -n "sipuLaunchKernel|workHead|channelMask" src/enqueue.cc

      看日志的技巧

      看最后错误：

      tail -n 200 logs/xxx/mpi.log
      tail -n 200 logs/xxx/qemu_host_0.log

      搜关键字：

      rg -n "error|warn|fault|translation failure|iova" logs/20260327_03

      带上下文：

      rg -n -C 5 "iova=0xf8" logs/20260327_03/qemu_host_0.log

      对比两个文件
      很有用。

      diff -u a.log b.log
      vimdiff a.log b.log

      或者只抽一段再比：

      sed -n '10378,10392p' file1
      sed -n '10378,10392p' file2

      结合 shell 做批量搜索

      for f in logs/20260327_03/cmodel/inst_trace*.log; do
        rg -n "0x00000000000000f8" "$f"
      done

      按大小排序文件：

      for f in logs/20260327_03/cmodel/*.log; do
        printf "%s %s\n" "$(stat -c %s "$f")" "$f"
      done | sort -nr | head

      搜地址/十六进制时
      最好用固定字符串匹配：

      rg -n -F "0x0000c000158ba04c" logs/20260327_03/cmodel
      rg -n -F "iova=0xf8" logs/20260327_03/qemu_host_0.log

      因为正则里 . [ 这些容易误伤。

      搜不到时常见原因

      - 文件被 .gitignore 忽略了：加 --hidden 或显式传路径
      - 在 build 产物里：别用默认全局搜，直接指定目录
      - 关键词大小写不一致：加 -i 或 -S
      - 实际内容不是文本：用 strings、xxd、hexdump
      - 路径打错：先 rg --files | rg 'name'

      我自己常用的排查套路

      1. rg --files | rg '关键词'
      2. rg -n "关键词" 相关目录
      3. rg -n -C 3 "关键词" 相关文件
      4. sed -n 'start,endp' 文件
      5. 必要时 diff / 批量脚本

