# Programming insights

This note is used to summarize tricks in C++ language.

## cache

* 写递归函数时，我总觉得自己是凑出来的，是否有更清晰的分析方法或思考方式

    这是非常普遍的困惑！很多人写递归时都有“凑”的感觉。让我分享一套系统性的分析方法，帮你建立清晰的递归思维框架。

    1. 递归的本质：数学归纳法

        递归本质上就是数学归纳法的编程实现：

        * 基础情况（Base Case）：相当于证明 n=1 时成立

        * 归纳步骤（Inductive Step）：假设 n=k 时成立，证明 n=k+1 时成立

    2. 四步分析法（最核心的方法）

        每次写递归函数前，问自己这四个问题：

        **第一步：明确函数的定义**

        用自然语言描述这个函数是干什么的

        ```python
        # 错误想法：“这个函数会计算阶乘...然后调用自身”
        # 正确想法：“factorial(n) 返回 n 的阶乘结果”
        ```

        **第二步：找出基础情况**

        什么情况下可以直接得出答案，无需递归？

        ```python
        # 阶乘：n = 0 或 n = 1 时，直接返回 1
        # 斐波那契：n = 0 时返回 0，n = 1 时返回 1
        ```

        第三步：分解问题（关键）

        假设子问题已经解决，如何构建原问题的解？

        这是最核心的一步，需要「信任递归」：

        ```python
        # 对于 factorial(n):
        # 如果我「已经知道」factorial(n-1) 的结果，
        # 那么 factorial(n) = n * factorial(n-1)

        # 对于二叉树高度：
        # 如果我「已经知道」左子树和右子树的高度，
        # 那么树高 = max(左高, 右高) + 1
        ```

        第四步：确保收敛

        每次递归调用是否向基础情况靠近？

        ```python
        def factorial(n):
            if n <= 1:          # 基础情况
                return 1
            return n * factorial(n-1)  # n-1 确保收敛
        ```

    3. 具体示例：二叉树遍历

        问题：计算二叉树节点数

        第一步：函数定义

        ```python
        # count_nodes(root) 返回以 root 为根的树的节点总数
        ```

        第二步：基础情况

        ```python
        if root is None:    # 空树
            return 0
        ```

        第三步：分解问题

        ```python
        # 假设我已经知道：
        left_count = count_nodes(root.left)   # 左子树节点数
        right_count = count_nodes(root.right) # 右子树节点数

        # 那么总节点数 = 左子树节点 + 右子树节点 + 根节点
        return left_count + right_count + 1
        ```

        第四步：验证收敛

        * 每次递归处理子树，树的高度减少

        * 最终会到达 None（叶子节点的子节点）

        完整代码：

        ```python
        def count_nodes(root):
            if root is None:            # 基础情况
                return 0

            # 分解问题（信任递归）
            left_count = count_nodes(root.left)
            right_count = count_nodes(root.right)

            # 组合结果
            return left_count + right_count + 1
        ```

    4. 常见递归模式

        模式1：递推型（自顶向下）

        ```python
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n-1)  # 先递归，后计算
        ```

        模式2：分治型（树形）

        ```python
        def max_depth(root):
            if not root:
                return 0
            left_depth = max_depth(root.left)    # 解决子问题
            right_depth = max_depth(root.right)  # 解决子问题
            return max(left_depth, right_depth) + 1  # 合并结果
        ```

        模式3：回溯型（尝试所有可能）

        ```python
        def permutations(nums, path=[], result=[]):
            if not nums:                    # 基础情况
                result.append(path.copy())
                return

            for i in range(len(nums)):      # 做选择
                path.append(nums[i])
                permutations(nums[:i] + nums[i+1:], path, result)  # 递归
                path.pop()                  # 撤销选择
        ```

    5. 调试技巧：递归树可视化

        在纸上画出递归调用树：

        ```text
        factorial(4)
        ├── 4 * factorial(3)
        │   ├── 3 * factorial(2)
        │   │   ├── 2 * factorial(1)
        │   │   │   └── return 1
        │   │   └── return 2
        │   └── return 6
        └── return 24
        ```

    6. 避免「凑」的练习方法

        * 先写注释，后写代码：先完成四步分析的注释

        * 从小开始：先处理 n=0,1 的情况，再考虑 n=2

        * 信任递归：写递归调用时，假设它已经正确工作

        * 画图辅助：对树、图等问题一定要画图

        * 测试小例子：手动模拟 n=2,3 的情况验证

    7. 检查清单

        写完后问：

        * 基础情况覆盖完整吗？

        * 每次递归是否向基础情况靠近？

        * 递归调用是否在解决「更小」的同类问题？

        * 结果组合是否正确？

    8. 进阶：递归 → 迭代的思维

        理解递归后，可以思考如何改写为迭代。这能加深理解：

        ```python
        # 递归
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n-1)

        # 迭代
        def factorial_iter(n):
            result = 1
            for i in range(2, n+1):
                result *= i  # 逐步构建结果
            return result
        ```

    总结

    摆脱「凑」的感觉的关键是：

    * 严格遵循四步分析法

    * 明确函数定义（用自然语言描述）

    * 信任递归（假设子问题已解决）

    * 画图验证（特别是树形结构）

    开始时会觉得这些步骤繁琐，但形成思维习惯后，写递归就会变得自然和自信。就像学数学归纳法一样，先按部就班，熟练后就能一眼看出递归结构。

* 什么时候需要开一个新函数？

    当变量名混乱有冲突，不好起新名字的时候可以尝试开一个新函数。

* 有关 graph 数据结构的渐进构造

    图由点（vertex）和边（edge）构成，点与点之间用有向边相连，只要能让这个信息存储起来，就可以处理图。

    如果基于点去构造，我们可以这样存储：

    ```cpp
    struct Vertex
    {
        vector<Vertex*> edges;
    };

    struct Graph
    {
        // 这里不能使用 vertor<Vertex> verts;
        // 因为 verts 的大小改变时，Vertex* 会改变，此时 vertex 中存储的 edges 都会失效
        vector<Vertex*> verts;
        Vertex* add_vert() {
            Vertex *vert = new Vertex;
            verts.push_back(vert);
            return vert;
        }
        ~Graph() {
            for (Vertex *vert_ptr: verts) {
                delete vert_ptr;
            }
        }
    };

    int main()
    {
        Graph graph;
        Vertex *vert = graph.add_vert();
        Vertex *vert_2 = graph.add_vert();
        vert.edges.push_back(vert_2);
        return 0;
    }
    ```

    这样的话，新增加 vert，只能按 edge 的依赖添加，比较麻烦。

    ```cpp
    int main()
    {
        Graph graph;
        for (int i = 0; i < 10; ++i)
        {
            graph.add_vert();
        }
        graph.verts[0]->edges.push_back(graph.verts[1]);
        return 0;
    }
    ```

    像这样，把所有 vert 都构建好，然后再添加 edge，就没有依赖的问题了。

    但是我们进一步想一下，既然都用索引了，我们存指针是否没有必要。

    以上是基于 vertex 构建的图，后面我们讨论基于 edge 构建的图。

* 不可能使用单个 path 存储 bfs 搜索到的路径

    由于 bfs 是由内层向外层一层一层搜索，并且外层不保存内层的信息，所以不可能使用单个 path 记录 bfs 的搜索结果。

    ```
    0 -> 1 -> 2
      -> 3 -> 4
    ```

    假如 0 有两条边，分别指向 1 和 3，我们想搜索`0 -> ... -> 2`的 path。首先把 0 的下一层节点放到 queue 里：`1, 3`，然后我们遍历 1 和 3 的下一个节点，注意，当 path 为`0 -> 3`时，新的 queue 为`2, 4`，此时我们再搜索 2，已经无法知道 2 的上一个节点是什么了，path 无法从`0 -> 3`修正到`0 -> 1 -> 2`。

    如果我们每次往 queue 里存储`pair(prev_vert, vert)`是否可以解决这个问题？不可以。例子：

    ```
    0 -> 1 -> 2 -> 3
      -> 4 -> 5 -> 6
    ```

    当 path 为 0 -> 4 -> 5 时，开始搜索 3，我们根据 queue 中的`pair(prev_vert, vert)`数据，得知 3 的上一个节点是 2，但是无法知道 2 的上一个节点是什么。path 中的 4 无论如何无法变成 1.

    由此可见，bfs 搜索时，要么只记录最短路径的节点数，要么就记录所有搜索的 path，这样才能返回具体的 path。

* 基于 vert + 指针的 graph，可读性太差

    ```cpp
    #include <string>
    #include <stdio.h>
    #include <iostream>
    #include <vector>
    #include <unordered_map>
    using namespace std;

    struct Vertex
    {
        vector<Vertex*> edges;
    };

    struct Graph
    {
        vector<Vertex*> verts;

        Vertex* add_vert() {
            Vertex *vert_ptr = new Vertex;
            verts.push_back(vert_ptr);
            return vert_ptr;
        }
        
        ~Graph() {
            for (Vertex *vert_ptr : verts) {
                delete vert_ptr;
            }
        }

        struct VertexPtrHash {
            size_t operator()(const pair<Vertex*, Vertex*> &src_dst) const
            {
                return hash<void*>()(src_dst.first) ^ hash<void*>()(src_dst.second);
            }
        };

        // <<src, dst>, path>
        unordered_map<pair<Vertex*, Vertex*>, vector<Vertex*>, VertexPtrHash> paths;
        int search_path_bfs(vector<Vertex*> **path_ptr, Vertex *src_vert, Vertex *dst_vert) {
            auto iter = paths.find({src_vert, dst_vert});
            if (iter != paths.end()) {
                *path_ptr = &iter->second;
                return 0;
            }
            vector<Vertex*> vert_queue_cur, vert_queue_nex;
            vert_queue_cur.push_back(src_vert);
            while (!vert_queue_cur.empty()) {
                for (int i = 0; i < vert_queue_cur.size(); ++i) {
                    Vertex *cur_vert = vert_queue_cur[i];
                    for (int j = 0; j < cur_vert->edges.size(); ++j) {
                        Vertex *nex_vert = cur_vert->edges[j];
                        if (paths.find({src_vert, nex_vert}) != paths.end())
                            continue;
                        vector<Vertex*> &path_src_to_cur = paths[{src_vert, cur_vert}];
                        vector<Vertex*> &path_src_to_nex = paths[{src_vert, nex_vert}];
                        path_src_to_nex = path_src_to_cur;
                        path_src_to_nex.push_back(nex_vert);
                        if (nex_vert == dst_vert) {
                            *path_ptr = &path_src_to_nex;
                            return 0;
                        }
                        vert_queue_nex.push_back(nex_vert);
                    }
                }
                vert_queue_cur = vert_queue_nex;
                vert_queue_nex.clear();
            }
            *path_ptr = nullptr;
            return -1;
        }
    };

    int main()
    {
        Graph graph;

        for (int i = 0; i < 7; ++i)
        {
            graph.add_vert();
        }
        graph.verts[0]->edges.push_back(graph.verts[1]);
        graph.verts[0]->edges.push_back(graph.verts[2]);
        graph.verts[0]->edges.push_back(graph.verts[3]);
        graph.verts[1]->edges.push_back(graph.verts[2]);
        graph.verts[1]->edges.push_back(graph.verts[4]);
        graph.verts[2]->edges.push_back(graph.verts[5]);
        graph.verts[3]->edges.push_back(graph.verts[5]);
        graph.verts[4]->edges.push_back(graph.verts[5]);
        graph.verts[5]->edges.push_back(graph.verts[6]);

        Vertex *src_vert = graph.verts[0];
        Vertex *dst_vert = graph.verts[6];
        vector<Vertex*> *path_ptr;
        int ret = graph.search_path_bfs(&path_ptr, src_vert, dst_vert);
        if (ret != 0) {
            printf("fail to find a path\n");
            return -1;
        }

        vector<Vertex*> &path = *path_ptr;
        printf("%p", src_vert);
        for (int i = 0; i < path.size(); ++i)
        {
            printf(" -> %p", path[i]);
        }
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/cpp_test$ ./main
    0x6041d45d7eb0 -> 0x6041d45d7ed0 -> 0x6041d45d7fd0 -> 0x6041d45d7ff0
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/cpp_test$ ./main
    0x5da664fc1eb0 -> 0x5da664fc1ed0 -> 0x5da664fc1fd0 -> 0x5da664fc1ff0
    ```

    可以看到，如果不存储额外信息，我们只能拿到 ptr，并且每次运行程序时会变化，对人类不友好。

    代码实现里的一些设计：

    * 使用`vector<Vertex*> edges`存储 edges，而不是`vector<Vertex>`，是为了使得 vertex 的指针由我们控制，不会随 vector 的扩容而改变。代价是我们需要手动 new 和 delete Vertex，好在工作量并不大，逻辑也简单。

        另外，这里的 edge 直接指向了下一个节点，如果我们需要存储一些边的权重之类的信息，那么这样的结构是做不到的。

    * 使用`vector<Vertex*> verts;`存储所有的 vert 指针。我们必须要有一个 container 来存储所有的指针，图不是树，不可能使用单个 root 就遍历所有节点。

    * 由于前面的数据结构是 vertex 指针，所以我们索性不使用数组`paths[i][j]`来快速定位 path 了，因为如果使用索引，我们还不如从一开始就对所有 vertex 使用索引。

        `unordered_map<pair<Vertex*, Vertex*>, vector<Vertex*>, VertexPtrHash> paths;`

        问题是如果使用了 vertex ptr，该如何存储能快速查找的 path 呢？似乎只有哈希表或红黑树了。这里我们选择哈希表。

        由于`pair<Vertex*, Vertex*>`是我们自定义的类型，所以我们还需要实现一个配套的 hash 函数：

        ```cpp
        struct VertexPtrHash {
            size_t operator()(const pair<Vertex*, Vertex*> &src_dst) const
            {
                return hash<void*>()(src_dst.first) ^ hash<void*>()(src_dst.second);
            }
        };
        ```

        `unordered_map<>`的尖括号内需要填类型，所以我们使用 struct 实现一个仿函数，作为类型。

    * `auto iter = paths.find({src_vert, dst_vert});`

        因为 bfs 搜索需要不断记录 path，所以我们使用记忆化搜索，如果之前搜索过了这条路径，就直接返回。

    * `vector<Vertex*> vert_queue_cur, vert_queue_nex;`

        使用`queue<>`并不会更好，queue 不支持索引，所以 debug 时不好定位，而且 queue 占的内存并不比 vector 小。

        或许使用 queue 的唯一好处是只需要一个 queue 就可以完成 bfs:

        ```cpp
        queue<Vertex*> que;
        que.push(src_vert);
        while (!que.empty())
        {
            Vertex *cur_vert = que.front();
            que.pop();
            for (auto nex_vert: cur_vert->edges)
            {
                // ...
                // if not searched and not in que
                que.push(nex_vert);
            }
        }
        ```

        如果不介意每次都从尾部开始搜索的话，其实 vector 也能做到这一点：

        ```cpp
        vector<Vertex*> que;
        que.push_back(src_vert);
        while (!que.empty())
        {
            Vertex *cur_vert = que.back();
            que.pop_back();
            for (auto nex_vert: cur_vert->edges)
            {
                // ...
                // if not searched and not in que
                que.push_back(nex_vert);
            }
        }
        ```

    * que 中有可能出现重复的 entry，因为只判断了是否 processed。但是不影响算法收敛

        ```cpp
        if (paths.find({src_vert, nex_vert}) != paths.end())
            continue;
        ```

    * `vector<Vertex*> &path_src_to_cur = paths[{src_vert, cur_vert}];`

        pair`{src_vert, cur_vert}`竟然也能作为 key，放在`[]`里索引。用习惯了 string 作 key，这种方式初看有点邪门。

    * 如果搜索到 dst_vert 了，那么我们就提前停止

        ```cpp
        if (nex_vert == dst_vert) {
            *path_ptr = &path_src_to_nex;
            return 0;
        }
        ```

        这样其实是 lazy search 的方式。如果实际场景是 init 时对时间不敏感，但是在 query 时对时间敏感，那么还不如在 init 时 search all paths，这样 query 单条 path 时就没有负担了。

        如果 graph 中无意义的顶点比较多，有意义的顶点只有有限个，那么可以使用 lazy search 的方式。

    * 由于返回值可能为空，所以不能返回引用，我们使用指针的指针 + return code 的方式来返回搜索结果

        ```cpp
        *path_ptr = nullptr;
        return -1;
        ```

    * `graph`的初始化比较繁琐，这个过程完全可以索引化

        ```cpp
        for (int i = 0; i < 7; ++i)
        {
            graph.add_vert();
        }
        graph.verts[0]->edges.push_back(graph.verts[1]);
        graph.verts[0]->edges.push_back(graph.verts[2]);
        graph.verts[0]->edges.push_back(graph.verts[3]);
        graph.verts[1]->edges.push_back(graph.verts[2]);
        graph.verts[1]->edges.push_back(graph.verts[4]);
        graph.verts[2]->edges.push_back(graph.verts[5]);
        graph.verts[3]->edges.push_back(graph.verts[5]);
        graph.verts[4]->edges.push_back(graph.verts[5]);
        graph.verts[5]->edges.push_back(graph.verts[6]);
        ```

        经过索引化后为：

        ```cpp
        vector<pair<int, int>> tmp_edges {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 4},
            {2, 5}, {3, 5}, {4, 5}, {5, 6}
        };

        for (pair<int, int> &edge: tmp_edges)
        {
            graph.verts[edge.first]->edges.push_back(graph.verts[edge.second]);
        }
        ```

        看起来，在初始化时，使用索引更方便。

    * 这种 print path　的方式挺好的，也不用额外判断是否到了最后一个 vertex:

        ```cpp
        vector<Vertex*> &path = *path_ptr;
        printf("%p", src_vert);
        for (int i = 0; i < path.size(); ++i)
        {
            printf(" -> %p", path[i]);
        }
        putchar('\n');
        ```

* bfs 中，重复搜索点的判断

    只有一个 entry 为 not searched (processed) and not in current queue and not in next queue 才能精确判断这个 entry 应该被 push 到 next queue 里。如果我们只判断了 not processed，queue 中会有重复，但是不影响结果的收敛性。

    example:

    ```
    0 -> 1
      -> 2

    1 -> 3
      -> 4

    2 -> 3
      -> 4
    ```

    现在我们 bfs 搜索`0 -> 3`和`0 -> 4`的 path，当搜索到 1 节点时，我们将 3, 4 加入 next queue。接下来搜索 2 节点，由于 3, 4 还没有被 processed，所以它们仍会被加入 next queue。这样 next queue 中的数据就变成了`[3, 4, 3, 4]`。

    ```cpp
    #include <vector>
    #include <iostream>
    using namespace std;

    int main()
    {
        // i means the i-th vertex
        // edges = graph[i] mean the edges of i-th vertex
        // nex_vert = edges[j] means the j-th edge is pointer to nex_vert-th vertex
        vector<vector<int>> graph {
            {1, 2},  // 0 -> 1, 0 -> 2
            {3, 4},  // 1 -> 3, 1 -> 4
            {3, 4},  // 2 -> 3, 2 -> 4
            {},
            {}
        };

        vector<int> cur_que, nex_que;
        vector<bool> vis(graph.size(), false);
        cur_que.push_back(0);
        int round_idx = 0;
        while (!cur_que.empty())
        {
            // print cur_que
            printf("round idx: %d\n", round_idx);
            for (int i = 0; i < cur_que.size(); ++i)
            {
                printf("%d, ", cur_que[i]);
            }
            putchar('\n');
            round_idx++;

            for (int i = 0; i < cur_que.size(); ++i)
            {
                int vert = cur_que[i];
                for (int j = 0; j < graph[vert].size(); ++j)
                {
                    int nex_vert = graph[vert][j];
                    if (!vis[nex_vert])
                    {
                        nex_que.push_back(nex_vert);
                    }
                }
                vis[vert] = true;
            }
            cur_que = nex_que;
            nex_que.clear();
        }

        return 0;
    }
    ```

    output:

    ```
    round idx: 0
    0, 
    round idx: 1
    1, 2, 
    round idx: 2
    3, 4, 3, 4,
    ```

* vertex based + pointer + vertex id, implement

    ```cpp
    #include <string>
    #include <stdio.h>
    #include <iostream>
    #include <vector>
    #include <unordered_map>
    using namespace std;

    struct Vertex
    {
        int id;
        vector<Vertex*> edges;
    };

    struct Graph
    {
        vector<Vertex*> verts;

        Vertex* add_vert() {
            Vertex *vert_ptr = new Vertex;
            // by default the id of vert is the same with idx
            vert_ptr->id = verts.size();
            verts.push_back(vert_ptr);
            return vert_ptr;
        }

        // id 必须存在，这一点由用户保证
        void del_vert(int id) {
            Vertex *vert_for_del = NULL;
            int idx_for_del = -1;
            for (int i = 0; i < verts.size(); ++i) {
                if (verts[i]->id == id) {
                    vert_for_del = verts[i];
                    idx_for_del = i;
                    break;
                }
            }

            for (int i = 0; i < verts.size(); ++i) {
                Vertex *vert = verts[i];
                vector<Vertex*> new_edges;
                int num_edges = verts[i]->edges.size();
                for (int j = 0; j < num_edges; ++j) {
                    if (vert->edges[j]->id != id) {
                        new_edges.push_back(vert->edges[j]);
                    }
                }
                vert->edges = move(new_edges);
            }

            decltype(paths) new_paths;
            for (auto &path : paths) {
                if (path.first.first == vert_for_del ||
                    path.first.second == vert_for_del) {
                    continue;
                }
                new_paths.insert(path);
            }
            paths = new_paths;

            verts.erase(verts.begin() + idx_for_del);
        }

        void add_edge(int vert_id, int nex_vert_id) {
            for (int i = 0; i < verts.size(); ++i) {
                if (verts[i]->id != vert_id) {
                    continue;
                }
                for (int j = 0; j < verts.size(); ++j) {
                    if (verts[j]->id == nex_vert_id) {
                        verts[i]->edges.push_back(verts[j]);
                        break;
                    }
                }
                break;
            }
        }

        // 边是没有顺序的，因此不能使用 int vert_idx 来删除边
        // 通过 Vertex *nex_vert 也可以唯一地定位到边，但是用户无法通过
        // 指针来定位，因此也不能使用 Vertex *nex_vert
        // 剩下的就只有使用 int nex_vert_id 了
        void del_edge(int vert_id, int nex_vert_id) {
            Vertex *vert = NULL;
            for (int i = 0; i < verts.size(); ++i) {
                if (verts[i]->id == vert_id) {
                    vert = verts[i];
                }
            }
            for (int i = 0; i < vert->edges.size(); ++i) {
                if (vert->edges[i]->id == nex_vert_id) {
                    vert->edges.erase(vert->edges.begin() + i);
                    break;
                }
            }
        }

        ~Graph() {
            for (Vertex *vert_ptr : verts) {
                delete vert_ptr;
            }
        }

        struct VertexPtrHash {
            size_t operator()(const pair<Vertex*, Vertex*> &src_dst) const
            {
                return hash<void*>()(src_dst.first) ^ hash<void*>()(src_dst.second);
            }
        };

        // <<src, dst>, path>
        unordered_map<pair<Vertex*, Vertex*>, vector<Vertex*>, VertexPtrHash> paths;
        int search_path_bfs(vector<Vertex*> **path_ptr, Vertex *src_vert, Vertex *dst_vert) {
            auto iter = paths.find({src_vert, dst_vert});
            if (iter != paths.end()) {
                *path_ptr = &iter->second;
                return 0;
            }
            vector<Vertex*> vert_queue_cur, vert_queue_nex;
            vert_queue_cur.push_back(src_vert);
            while (!vert_queue_cur.empty()) {
                for (int i = 0; i < vert_queue_cur.size(); ++i) {
                    Vertex *cur_vert = vert_queue_cur[i];
                    for (int j = 0; j < cur_vert->edges.size(); ++j) {
                        Vertex *nex_vert = cur_vert->edges[j];
                        if (paths.find({src_vert, nex_vert}) != paths.end())
                            continue;
                        vector<Vertex*> &path_src_to_cur = paths[{src_vert, cur_vert}];
                        vector<Vertex*> &path_src_to_nex = paths[{src_vert, nex_vert}];
                        path_src_to_nex = path_src_to_cur;
                        path_src_to_nex.push_back(nex_vert);
                        if (nex_vert == dst_vert) {
                            *path_ptr = &path_src_to_nex;
                            return 0;
                        }
                        vert_queue_nex.push_back(nex_vert);
                    }
                }
                vert_queue_cur = vert_queue_nex;
                vert_queue_nex.clear();
            }
            *path_ptr = nullptr;
            return -1;
        }

        void print() {
            if (verts.empty()) {
                printf("(empty)\n");
                return;
            }

            for (int i = 0; i < verts.size(); ++i) {
                printf("vert %d -> ", verts[i]->id);
                for (int j = 0; j < verts[i]->edges.size(); ++j) {
                    printf("%d, ", verts[i]->edges[j]->id);
                }
                putchar('\n');
            }
        }
    };

    int main()
    {
        Graph graph;

        for (int i = 0; i < 7; ++i) {
            graph.add_vert();
        }
        graph.verts[0]->edges.push_back(graph.verts[1]);
        graph.verts[0]->edges.push_back(graph.verts[2]);
        graph.verts[0]->edges.push_back(graph.verts[3]);
        graph.verts[1]->edges.push_back(graph.verts[2]);
        graph.verts[1]->edges.push_back(graph.verts[4]);
        graph.verts[2]->edges.push_back(graph.verts[5]);
        graph.verts[3]->edges.push_back(graph.verts[5]);
        graph.verts[4]->edges.push_back(graph.verts[5]);
        graph.verts[5]->edges.push_back(graph.verts[6]);

        Vertex *src_vert = graph.verts[0];
        Vertex *dst_vert = graph.verts[6];
        vector<Vertex*> *path_ptr;
        int ret = graph.search_path_bfs(&path_ptr, src_vert, dst_vert);
        if (ret != 0) {
            printf("fail to find a path\n");
            return -1;
        }

        vector<Vertex*> &path = *path_ptr;
        printf("%d", src_vert->id);
        for (int i = 0; i < path.size(); ++i) {
            printf(" -> %d", path[i]->id);
        }
        putchar('\n');
        putchar('\n');

        graph.print();
        putchar('\n');

        int num_verts = graph.verts.size();
        for (int i = 0; i < num_verts; ++i) {
            graph.del_vert(i);
        }
        
        graph.print();

        return 0;
    }
    ```

    output:

    ```
    0 -> 2 -> 5 -> 6

    vert 0 -> 1, 2, 3, 
    vert 1 -> 2, 4, 
    vert 2 -> 5, 
    vert 3 -> 5, 
    vert 4 -> 5, 
    vert 5 -> 6, 
    vert 6 -> 

    (empty)
    ```

    可以看到，使用 vertex id 后，输出对人类比较友好了。

    下面是一些细节说明：

    * `Vertex *vert_ptr = new Vertex;`

        vert 的内存是手动分配，而 vert 的指针交给 vector，这样保证了不会因为 vector 的自动扩容而导致 vert 的地址变化。

        由于 va 和进程是强绑定的，所以跨进程、通过网络传输时，需要额外的序列化和反序列化。

    * `vert_ptr->id = verts.size();`

        默认使用和索引一样的 id，这表示对 add vertex 的顺序是有要求的。

        由于每次都是按 size 分配 id，如果先删了几个，又增添几个 vert，那么 id 就比较混乱了。

    * 通过线性查找找到 id，

        ```cpp
        Vertex *vert_for_del = NULL;
        int idx_for_del = -1;
        for (int i = 0; i < verts.size(); ++i) {
            if (verts[i]->id == id) {
                vert_for_del = verts[i];
                idx_for_del = i;
                break;
            }
        }
        ```

        `vert_for_del`留着后面给 del path 的时候用。其实后面的 path 也可以使用`path.first.first->id == id`, `path.first.second->id == id`来判断，效果一样的。

        `idx_for_del`留着后面给`verts.erase(verts.begin() + idx_for_del);`时使用。

        这一段也可以改成

        ```cpp
        void del_vert(int id) {
            for (int i = 0; i < verts.size(); ++i) {
                if (verts[i]->id == id) {
                    verts.erase(verts.begin() + i);
                    break;
                }
            }

            for (int i = 0; i < verts.size(); ++i) {
                Vertex *vert = verts[i];
                vector<Vertex*> new_edges;
                int num_edges = verts[i]->edges.size();
                for (int j = 0; j < num_edges; ++j) {
                    if (vert->edges[j]->id != id) {
                        new_edges.push_back(vert->edges[j]);
                    }
                }
                vert->edges = move(new_edges);
            }

            decltype(paths) new_paths;
            for (auto &path : paths) {
                if (path.first.first->id == id ||
                    path.first.second->id == id) {
                    continue;
                }
                new_paths.insert(path);
            }
            paths = new_paths;
        }
        ```

        反正都是线性搜索，使用 id 和使用 ptr 区别不大。

    * `vector<Vertex*> new_edges;`

        连续删除 vector 中的元素比较麻烦，这里采用将有效数据放到一个新的 vector 里，最后再使用新 vector 取代旧 vector 的方式。

        由于 add_vert, del_vert 都是不频繁的操作，所以这个方式还算 ok。

    * `vert->edges = move(new_edges);`

        交换数据时，只需要交换 payload 指针就可以了。不知道使用 move 会不会快一点。

    * `void add_edge(int vert_id, int nex_vert_id)`

        verts 内部存储的是指针，但是用户在操作时，只需要提供 vert id，这样用户只要操作正确的 id，就能得到正确的结果，不需要关心 ptr。

        这样看来`Vertex* add_vert()`应该同时返回 vert ptr 和 vert id 比较好。

    * `void del_edge(int vert_id, int nex_vert_id)`

        当删除节点时，要删除所有与其关联的 edge 和 path。删除 edge 也同理，我们需要删除所有与之关联的 path。显然这里没有做到这一点。

    整个代码中使用了大量的线性查找，效率很低，构建一个 pointer -> id 的查找表是一个不可忽视的需求。

* vertex + graph 的功能拓展探索

    ```cpp
    #include <iostream>
    #include <unordered_map>
    #include <vector>
    #include <unordered_set>
    #include <string>
    #include <cstdio>
    using namespace std;

    template<typename T>
    struct BaseVertex {
        int id;
        vector<T*> edges;
    };

    template<typename T>
    struct BaseGraph {
        vector<T*> verts;
        void print_verts() {
            for (int i = 0; i < verts.size(); ++i) {
                printf("%d, ", verts[i]->id);
            }
            putchar('\n');
        }
    };

    struct Vertex: public BaseVertex<Vertex> {

    };

    struct Graph: public BaseGraph<Vertex> {

    };

    struct MyVertex: public BaseVertex<MyVertex> {
        int vert_type;
        string vert_name;
    };

    struct MyGraph: public BaseGraph<MyVertex> {
        void print_verts() {
            for (int i = 0; i < verts.size(); ++i) {
                printf("%d type: %d, ", verts[i]->id, verts[i]->vert_type);
            }
            putchar('\n');
        }

        void filter_my_vertex_type(int type) {
            for (int i = 0; i < verts.size(); ++i) {
                if (verts[i]->vert_type == type) {
                    printf("%d, ", verts[i]->id);
                }
            }
            putchar('\n');
        }
    };

    int main() {
        Graph graph;
        for (int i = 0; i < 5; ++i) {
            graph.verts.emplace_back(new Vertex);
            graph.verts[i]->id = i;
        }
        graph.print_verts();

        MyGraph my_graph;
        for (int i = 0; i < 5; ++i) {
            my_graph.verts.emplace_back(new MyVertex);
            my_graph.verts[i]->id = i;
            my_graph.verts[i]->vert_type = 1;
        }
        my_graph.print_verts();
        my_graph.verts[2]->vert_type = 2;
        my_graph.verts[4]->vert_type = 2;
        my_graph.filter_my_vertex_type(2);
        return 0;
    }
    ```

    output:

    ```
    0, 1, 2, 3, 4, 
    0 type: 1, 1 type: 1, 2 type: 1, 3 type: 1, 4 type: 1, 
    2, 4,
    ```

    说明

    * `BaseVertex`中`T`类型的指针，我们只知道`T`类型会继承自`BaseVertex`，也可能就是`BaseVertex`。此时我们发现`BaseVertex`无法实例化：

        假如`T`就是`BaseVertex`，那么

        ```cpp
        BaseVertex<BaseVertex> vert;
        ```

        无法通过编译。因为尖括号内的`BaseVertex`，仍然是一个模板类型，编译器会提示模板参数缺失。我们再填一级也不行，`BaseVertex<BaseVertex<BaseVertex<T>>>`，这样递归填下去，`T`还是无法被确定。

        此时我们必须借助另一个类来完成实例化：

        ```cpp
        struct Vertex: public BaseVertex<Vertex> {

        };

        int main() {
            Vertex vert;  // OK
            return 0;
        }
        ```

        此时`Vertex`继承自`BaseVertex`，而`BaseVertex`的模板类型又是`Vertex`。这样通过一个额外的类型，将递归变成了两种类型的循环调用，从而可以通过编译。

    * `BaseGraph`被设计成只依赖`BaseVertex`内成员的类，即`T`一定继承自`BaseVertex`

        ```cpp
        template<typename T>
        struct BaseGraph {
            vector<T*> verts;
            void print_verts() {
                for (int i = 0; i < verts.size(); ++i) {
                    printf("%d, ", verts[i]->id);
                }
                putchar('\n');
            }
        };
        ```

        如果能显式地约束`T`一定继承自`BaseVertex`，那么在输入`verts[i]->`时就能自动显示成员了。可惜到 c++20 才支持类型约束。

        `BaseGraph`可以使用`Vertex`实例化，但是不能用`BaseVertex`实例化：

        ```cpp
        BaseGraph<Vertex> graph;
        ```

        但是我们想了想，如果使`Graph`类型默认对应`Vertex`类型，`BaseGraph`默认对应`BaseVertex`类型，视觉效果更好一点：

        ```cpp
        // BaseGraph<BaseVertex> base_graph;  // not allowed
        Graph graph;  // graph.verts is Vertex*
        ```

    * 是否可以将`BaseGraph`里的函数实现放到`Graph`里？不可以，因为`Graph`和`Vertex`绑定在一起，后面的 derived graph 如果继承了 Graph，那么就连带着 Vertex 一起绑定了。我们还是希望 derived graph 和 derived Vertex 绑定。

    * `MyVertex`增加了几个字段，

        ```cpp
        struct MyVertex: public BaseVertex<MyVertex> {
            int vert_type;
            string vert_name;
        };
        ```

        后面与`struct MyGraph: public BaseGraph<MyVertex>`绑定在一起，`MyGraph`通过处理`Graph`增加的字段来增加高级功能（比如使得节点有 type 的属性）。

    整体看来。代码的对称性足够，实际使用时简洁，可扩展性强，唯一的不足地方是`BaseGraph`中，IDE 无法给出成员的编程提示，这个问题只能等到 c++ 20 再修复了。

* 不能使用 edge 去初始化 vertex

    一个使用 edge 去 init 的 example 如下：

    ```cpp
    explicit StaticGraph(const initializer_list<pair<int, int>> &init_list) {
        int max_vert_idx = -1;
        for (auto &&iter = init_list.begin(); iter != init_list.end(); ++iter) {
            max_vert_idx = std::max(max_vert_idx,
                std::max(iter->first, iter->second));
        }
        verts.resize(max_vert_idx + 1);
        for (int i = 0; i <= max_vert_idx; ++i) {
            verts[i] = new Vertex;
            Vertex &vert = *verts[i];
            vert.id = i;
        }
        for (auto &&iter = init_list.begin(); iter != init_list.end(); ++iter) {
            verts[iter->first]->edges.push_back(verts[iter->second]);
        }
    };

    int main() {
        // 不能使用 edge 去初始化 vertex，因为 edge 只有索引信息，没有 vertex 的属性信息
        StaticGraph graph {
            {0, 1}, {0, 2}, {0, 3},
            {1, 2}, {1, 4},
            {2, 5},
            {3, 5},
            {4, 5},
            {5, 6}
        };
        return 0;
    }
    ```

    我们必须先 init vertex，再添加 edges:

    ```cpp
    struct Graph: public BaseGraph<Vertex> {
        void init_verts(const initializer_list<DevType> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                Vertex *vert_ptr = new Vertex;
                Vertex &vert = *vert_ptr;
                vert.id = this->verts.size();
                vert.dev_type = *iter;
                this->verts.push_back(vert_ptr);
            }
        }

        void init_edges(const initializer_list<pair<int, int>> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                this->verts[iter->first]->edges.push_back(
                    this->verts[iter->second]);
            }
        }
    }

    int test() {
        Graph graph;

        graph.init_verts({GPU, GPU, CPU, NIC, CPU, GPU, NIC});

        // 目前这里的做法是先 init vert，然后再将 edge 连接上，这样应避免了无法录入 vertex info 的问题
        graph.init_edges({
            {0, 1}, {0, 2}, {0, 3},
            {1, 2}, {1, 4},
            {2, 5},
            {3, 5},
            {4, 5},
            {5, 6}
        });

        return 0;
    }
    ```

* extended static graph

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <iostream>
    #include <vector>
    #include <unordered_map>
    #include <algorithm>
    #include <utility>
    #include <initializer_list>
    using namespace std;

    template<typename T>
    struct BaseVertex {
        int id;
        vector<T*> edges;
    };

    // 约束：T 必须继承自 BaseVertex
    template<typename T>
    struct BaseGraph {
        vector<T*> verts;

        ~BaseGraph() {
            for (T* vert_ptr : verts) {
                delete vert_ptr;
            }
        }

        void init_verts(const initializer_list<int> &init_list) {
            for (int i = 0; i < init_list.size(); ++i) {
                verts.push_back(new T);
                T *vert = verts[i];
                vert->id = *(init_list.begin() + i);
            }
        }

        void init_edges(const initializer_list<pair<int, int>> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                verts[iter->first]->edges.push_back(verts[iter->second]);
            }
        }

        void print_graph() {
            for (int i = 0; i < verts.size(); ++i) {
                T* vert = verts[i];
                printf("idx %d: vert id %d, num edges: %lu\n",
                    i, vert->id, vert->edges.size());
                printf("\t-> ");
                for (int j = 0; j < vert->edges.size(); ++j) {
                    printf("%d, ", vert->edges[j]->id);
                }
                putchar('\n');
            }
        }
    };

    struct PrimeVertex: public BaseVertex<PrimeVertex> {};

    struct PrimeGraph: public BaseGraph<PrimeVertex> {};

    enum DevType {
        GPU,
        CPU,
        NIC,
        NET
    };

    template<typename KeyType, typename ValType>
    struct LookupTable {
        unordered_map<KeyType, ValType> lut;
        explicit LookupTable(const initializer_list<
            pair<KeyType, ValType>> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                lut.emplace(iter->first, iter->second);
            }
        }
        const ValType& operator[](const KeyType &key) const {
            return lut.at(key);
        }
    };

    const LookupTable<DevType, string> Dev_Type_Id_To_Str {
        {GPU, "gpu"},
        {CPU, "cpu"},
        {NIC, "nic"},
        {NET, "net"}
    };

    struct Vertex: public BaseVertex<Vertex> {
        DevType dev_type;
    };

    struct Graph: public BaseGraph<Vertex> {
        void init_verts(const initializer_list<DevType> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                Vertex *vert_ptr = new Vertex;
                Vertex &vert = *vert_ptr;
                vert.id = this->verts.size();
                vert.dev_type = *iter;
                this->verts.push_back(vert_ptr);
            }
        }

        void init_edges(const initializer_list<pair<int, int>> &init_list) {
            for (auto iter = init_list.begin(); iter != init_list.end(); ++iter) {
                this->verts[iter->first]->edges.push_back(
                    this->verts[iter->second]);
            }
        }

        void print_graph() {
            for (int i = 0; i < verts.size(); ++i) {
                Vertex &vert = *verts[i];
                printf("idx: %d, vert id: %d, dev type: %s, num edges: %lu\n",
                    i, vert.id, Dev_Type_Id_To_Str[vert.dev_type].c_str(),
                    vert.edges.size());
                printf("\t-> ");
                for (int j = 0; j < vert.edges.size(); ++j) {
                    printf("%d, ", vert.edges[j]->id);
                }
                putchar('\n');
            }
        }

        void collect_verts(vector<Vertex*> &out_verts, DevType dev_type) {
            out_verts.clear();
            for (int i = 0; i < verts.size(); ++i) {
                if (verts[i]->dev_type == dev_type) {
                    out_verts.push_back(verts[i]);
                }
            }
        }
    };

    // 要求输入按顺序，vert 的编号必须从 0 开始，并且与位置保持一致
    struct StaticGraph: public Graph {
        struct VertexPtrHash {
            size_t operator()(const pair<Vertex*, Vertex*> &src_dst) const {
                return std::hash<Vertex*>()(src_dst.first) ^
                    std::hash<Vertex*>()(src_dst.second);
            }
        };

        unordered_map<pair<Vertex*, Vertex*>, vector<int>, VertexPtrHash> paths;
        // DynamicGraph 下才会考虑 paths 是否为 valid
        // staitc 模式下，paths 经初始化后一直有效
        // bool is_paths_valid = false;
        // path 应该是一个 edge list，不应该是 vert list
        // 因为给定 edge 可以很快查到 next vert，但是给定两个 vert 不容易查到 edge
        // path is an edge idx list
        int search_path_bfs(vector<int> **path_ptr, int src_id, int dst_id) {
            Vertex *src_vert = verts[src_id];
            Vertex *dst_vert = verts[dst_id];
            auto iter = paths.find({src_vert, dst_vert});
            if (iter != paths.end()) {
                *path_ptr = &iter->second;
                return 0;
            }

            vector<Vertex*> que_cur, que_nex;
            que_cur.push_back(src_vert);
            while (!que_cur.empty()) {
                for (int i = 0; i < que_cur.size(); ++i) {
                    Vertex *cur_vert = que_cur[i];
                    for (int j = 0; j < cur_vert->edges.size(); ++j) {
                        Vertex *nex_vert = cur_vert->edges[j];
                        if (paths.find({src_vert, nex_vert}) != paths.end()) {
                            continue;
                        }
                        vector<int> &path_src_to_cur = paths[{src_vert, cur_vert}];
                        vector<int> &path_src_to_nex = paths[{src_vert, nex_vert}];
                        path_src_to_nex = path_src_to_cur;
                        path_src_to_nex.push_back(j);
                        if (nex_vert == dst_vert) {
                            *path_ptr = &path_src_to_nex;
                            return 0;
                        }
                        que_nex.push_back(nex_vert);
                    }
                }
                que_cur = que_nex;
                que_nex.clear();
            }
            *path_ptr = nullptr;
            return -1;
        }

        void print_path(int src_id, vector<int> &path) {
            Vertex &src_vert = *verts[src_id];
            printf("%d", src_vert.id);
            Vertex *vert = &src_vert, *vert_nex;
            for (int i = 0; i < path.size(); ++i) {
                vert_nex = vert->edges[path[i]];
                printf(" -> %d", vert_nex->id);
                vert = vert_nex;
            }
            putchar('\n');
        }
    };

    int main() {
        printf("-------- prime graph --------\n");
        PrimeGraph p_graph;
        p_graph.init_verts({0, 1, 2, 3, 4});
        p_graph.init_edges({
            {0, 1}, {0, 2}, {0, 4},
            {1, 3}, {1, 4},
            {2, 1}, {2, 3}, {2, 0},
            {4, 0}, {4, 2}
        });
        p_graph.print_graph();
        putchar('\n');

        printf("-------- static graph --------\n");
        StaticGraph graph;
        graph.init_verts({GPU, GPU, CPU, NIC, CPU, GPU, NIC});
        graph.init_edges({
            {0, 1}, {0, 2}, {0, 3},
            {1, 2}, {1, 4},
            {2, 5},
            {3, 5},
            {4, 5},
            {5, 6}
        });

        graph.print_graph();

        vector<int> *path_ptr;
        int ret = graph.search_path_bfs(&path_ptr, 0, 6);
        if (ret != 0) {
            printf("fail to get path\n");
            return -1;
        }
        graph.print_path(0, *path_ptr);

        vector<Vertex*> gpu_verts;
        graph.collect_verts(gpu_verts, GPU);
        printf("gpu vert ids: ");
        for (Vertex *vert : gpu_verts) {
            printf("%d, ", vert->id);
        }
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    -------- prime graph --------
    idx 0: vert id 0, num edges: 3
    	-> 1, 2, 4, 
    idx 1: vert id 1, num edges: 2
    	-> 3, 4, 
    idx 2: vert id 2, num edges: 3
    	-> 1, 3, 0, 
    idx 3: vert id 3, num edges: 0
    	-> 
    idx 4: vert id 4, num edges: 2
    	-> 0, 2, 

    -------- static graph --------
    idx: 0, vert id: 0, dev type: gpu, num edges: 3
    	-> 1, 2, 3, 
    idx: 1, vert id: 1, dev type: gpu, num edges: 2
    	-> 2, 4, 
    idx: 2, vert id: 2, dev type: cpu, num edges: 1
    	-> 5, 
    idx: 3, vert id: 3, dev type: nic, num edges: 1
    	-> 5, 
    idx: 4, vert id: 4, dev type: cpu, num edges: 1
    	-> 5, 
    idx: 5, vert id: 5, dev type: gpu, num edges: 1
    	-> 6, 
    idx: 6, vert id: 6, dev type: nic, num edges: 0
    	-> 
    0 -> 2 -> 5 -> 6
    gpu vert ids: 0, 1, 5, 
    ```

## note

* If we want to verify if two numbers are the same sign, we can use XOR: `sign_1 ^ sign_2`.

    This trick can get the product sign of two multipliers.

* the absolute value of `INT32_MIN` is greater by 1 than `INT32_MAX`

    So don't convert negative value to positive. Turn positive value to negative to avoid overflow.

* 首先让最底层的函数暴露尽量多的信息和细节，然后再在上面套一层封装，适应不同的调用需求

    比如对射线和空间三维物体交点的计算，底层的计算函数需要输出是否相交，相交的话`t`是多少，交点在哪里，相交的是哪个物体，交点处的法向量，颜色等等，然后上层函数再对这个函数进行一次封装，有些需求是只需要判断是否相交就可以，有些需求是要得到交点位置。我们使用第二层函数来对接这些需求。

* 如果想要实现复杂的功能，可以用 c/c++ 写许多小程序，然后把它们组合起来使用。

    这个建议的原理同上面相同。让每个 c/c++ 程序专注一个功能，比如搜索字符串，做内存拷贝，等等，然后让上层的程序组合多个底层的程序，实现一个功能。

    让 c/c++ 写复杂的交互效率很低，还不好写。

    不要让一个程序变得很臃肿。

* 在写多线程的代码时，由于负载不均衡是个很大的问题，所以子线程函数的颗粒需要尽可能细，细到无法再分。然后在上层再写一层调度，进行任务的分配。

* 写代码时，前面写过的代码，实现过的函数，经常会被忘记，因此写代码也需要进行 qa 记忆巩固。

* 引入一个头文件，就代表了引入一种功能。头文件与头文件之间尽量独立

* array 类似于一个数组，而 vector 类似一个指针

    array 是占用栈，而 vector 占用的是堆。

* 每个函数代码尽量控制在 30 以内。人每次只能处理很短的一段逻辑，太长的逻辑就算当前能正常处理，过段时间遗忘后再加载也会浪费时间。

    不一定是 30 行，但目前看来，超过 90 行一定是不容易处理的。

* 对于一些文本或格式化的文件进行数据处理时，通常用有限状态机解决。使用状态机处理是一种很自然的想法。

    可以设置一些 bool 值来控制状态。有时间了系统学一下这个编程思想。

* 编程时，如果变量名称没有明显的歧义或者冗余，那么就不需要优化

    比如 direction 不需要刻意写成 dir

    如果感到变量名太长，确实影响了效率，再使用缩写。比如大片的代码长度超过了 2/3 行。

* 复杂项目的渐进编程法

    不要想着一开始就实现一个接口灵活，非常好用的框架。因为我们编程时面对的是当前的项目，所谓的“灵活”是针对大量的项目而言的，接手的项目比较少时，无法抽象出需要在什么地方灵活。

    比如在渲染物体表面时，一开始可能只用到`Kd`，`Ks`这两个材质参数，后面慢慢又增加了折射，色散之类的效果，又增加了透明度，折射率，吸收率，衰减系统，brdf 之类的材质参数。刚开始的时候不可能考虑到后面的这么多东西，因此过度考虑代码的灵活性也是在浪费时间。

    一个比较好的想法是，对于每个项目，都只使用已有的知识，想办法最快地实现目标，记录下其中学到的东西。然后再开启下一个项目，同样以最快的方式达到目标。一直这样循环下去。

    比如在给函数传参数`vec3`时，既可以使用引用，也可以按值传递。如果要传引用，需要区分左值引用，右值引用，为了统一这两种引用，还要写模板，为了使模板只对`vec3`类型生效，还要写模板元编程`enable_if`，非常麻烦。如果我们的目标仅仅是渲染，那么直接按值传递，可以节省很多时间。只有当按值传递确实成为性能瓶颈时，再考虑将值换成引用。

    如果希望对引用和模板展开调研，可以另开一个项目，专门研究引用，模板，完美转发。但在当前项目中去研究这些东西，是不值得推荐的。

* 按格式解析文件

    目前我觉得比较好的方法是使用有限状态机。有空了刷刷相关的题目。不知道有什么更好的方法。

* c/c++ 的程序开发效率低

    c/c++ 写程序原型不容易写得复杂。c/c++ 适合写单一功能，或者架构已经大概规划好的程序。

    c/c++ 不适合写有大量交互的程序，因为交互的程序逻辑复杂，走走停停，而且需求变动太，使用c/c++开发效率太低。

* 如果一个底层的资源不在上层的控制中，那么这个资源其实就相当于泄漏的内存了，越积越多，就会导致电脑放不下。

    由于各种复杂因素的限制，我们无法像 rust 或 c++ 那样精确处理泄漏内存，一个折中的办法是模仿 java，隔一段时间就去遍历一遍底层资源，看是否有什么资源失去了控制。

    联想：java 的 gc 其实可以做到少次多量，与分配内存形成动态平衡，未来的程序应该这样比较好。

* 有关c++框架的调试

    有些程序是 c++ 的框架程序，代码分为上下层两部分，上层以是抽象类和接口，下层是接口的实现。整个代码中用到了大量的虚函数。

    这种程序调试时的一些经验如下：

    * 对于一个基类对象，在调用它的虚函数时，使用静态分析代码+跳转可能找不到正确的虚函数，可以在程序运行时，使用 F11 step in 到虚函数里，就能找到正确的虚函数了。

    * 想要将下层实现的头文件包含到上层，然后强制将基类对象指针转换成继承类的指针，很有可能会编译失败。通常的解决方法为在基类中写个虚函数，然后自己在派生类里实现一下。

        如果基类里是纯虚函数，那么需要在所有的派生类里实现一遍。

        最后用基类指针去调用虚函数就可以了，c++会自动帮你找到派生类的虚函数。

    * 如果不熟悉项目的编译系统，可以尝试在`.cpp`文件里引入`.hpp`文件，把实现写到头文件里。
