# Algorithms

致力于线性直觉思考 + 非线性联想的分析方式，拒绝记忆模板化的答案。

分析代码中的任何细节，对常见的写法进行不常见的提问：为什么这样是对的？为什么不可以那样？对的只有这一种写法吗？

## cache

* tsp problem greedy algorithm

    python version:

    ```python
    #!/home/hlc/miniconda3/envs/torch/bin/python

    import numpy as np
    import matplotlib.pyplot as plt

    # Greedy TSP算法
    def greedy_tsp(distance_matrix, start_point):
        n = len(distance_matrix)
        visit_mask = [False] * n
        path = [start_point]
        visit_mask[start_point] = True
        current_point = start_point
        total_distance = 0  # 初始化总路径长度
        
        while len(path) < n:
            next_point = None
            min_dist = float('inf')
            # n 定义为顶点的数量，由于每个顶点都与其他顶点有边，所以 n 又拿来作为边数
            for i in range(n):
                # 这里并没有用 i != self_idx 来跳过自身顶点，而是提前把自身顶点
                # 放到了 visit_mask 里，然后根据 visit_mask 去跳过顶点，
                # 这样也连带把自身顶点也跳过了
                if not visit_mask[i] and distance_matrix[current_point][i] < min_dist:
                    min_dist = distance_matrix[current_point][i]
                    next_point = i
            # 每次都从当前顶点的所有边中，选出到未经过顶点的最短的边
            path.append(next_point)
            visit_mask[next_point] = True
            total_distance += min_dist  # 累加路径长度
            current_point = next_point
        
        # 返回起点，并累加最后一段的距离
        total_distance += distance_matrix[current_point][start_point]
        path.append(start_point)  
        
        return path, total_distance

    # 绘制路径
    def draw_tsp_path_with_distances(path, distance_matrix, start_point):
        # 生成节点的坐标，均匀分布在圆上
        n = len(distance_matrix)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 10
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        plt.figure(figsize=(12, 12))  # 图像放大至12x12
        
        # 绘制节点，扩大节点大小
        plt.scatter(x, y, c='lightblue', s=3000, edgecolors='black', zorder=2)  # 节点扩大3倍
        
        # 标注节点编号，扩大字号
        for i in range(n):
            plt.text(x[i], y[i], str(i), fontsize=36, ha='center', va='center', color='black')  # 字号扩大3倍
        
        # 绘制路径并标注距离
        for i in range(len(path) - 1):
            start, end = path[i], path[i+1]
            plt.plot([x[start], x[end]], [y[start], y[end]], 'r-', lw=6, zorder=1)  # 线宽扩大3倍
            
            # 计算线的中点，标注距离
            mid_x = (x[start] + x[end]) / 2
            mid_y = (y[start] + y[end]) / 2
            dist = distance_matrix[start][end]
            plt.text(mid_x, mid_y, f'{dist}', fontsize=30, color='blue', ha='center', va='center')  # 距离标注字号扩大3倍
        
        # 标注起点，扩大节点大小
        plt.scatter(x[start_point], y[start_point], c='red', s=3000, edgecolors='black', zorder=3)  # 起点扩大3倍
        
        plt.title("TSP Path with Distances (Scaled)", fontsize=24)  # 标题字号扩大
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')  # 关闭坐标轴
        plt.show()

    def main():
        # 距离矩阵
        distance_matrix = [
            [0, 11, 8.4, 5.2, 4.8, 2.6],
            [11, 0, 7.4, 4.2, 10.2, 12],
            [8.4, 7.4, 0, 3.2, 6.4, 11],
            [5.2, 4.2, 3.2, 0, 6, 7.8],
            [4.8, 10.2, 6.4, 6, 0, 10.2],
            [2.6, 12, 11, 7.8, 10.2, 0]
        ]

        # 开始点为0
        start_point = 0
        tour_path, tour_length = greedy_tsp(distance_matrix, start_point)

        print("访问路径:", tour_path)
        print("路径总长度:", tour_length)

        # 绘制TSP路径
        draw_tsp_path_with_distances(tour_path, distance_matrix, start_point)  # 包括最后一个回到起点的0

    if __name__ == '__main__':
        main()

    ```

    output:

    ```
    访问路径: [0, 5, 3, 2, 4, 1, 0]
    路径总长度: 41.2
    ```

    另外还输出一张 graph 图片。

    说明：

    * `draw_tsp_path_with_distances()`画图函数比较有意思，没用到第三方的 layout 工具，是作者自己实现的一个简单圆形 layout。

    * 如果在 greedy 时不选最短的，而选前 2 最短的，该如何写代码？是否可以达到比原来更好的效果？

* 旅行商问题

    Travelling salesman problem

    给定一系列城市和每对城市之间的距离，求解访问每一座城市一次并回到起始城市的最短回路。

    等价约束：

    对于图$G = (V, A)$，其中$\lvert A \rvert = \lvert V \rvert \times (\lvert V \rvert - 1)$，这个看上像是从$V$中一个顶点出发，向着出了自身外的所有顶点构建边（edge）。

    $c_{ij}$为顶点$i$到顶点$j$的边的成本（cost），其中$i, j \in A$。

    根据以上定义，旅行商问题可以等价定义为如下公式：

    $$\min \sum_{i, j \in A} c_{ij} x_{ij}$$

    subject to:

    $$\sum_{i \in V \setminus \{j\}} x_{ij} = 1, j \in V$$

    (定义闭合回路)

    $$\sum_{j \in V \setminus \{i\}} x_{ij} = 1, i \in V$$

    (定义闭合回路，即任意顶点 i ，解回路中必有且仅有一条弧以其作为起点的同时有且仅有一条弧以其作为终点)

    $$\sum_{i \in S} \sum_{j \in S} x_{ij} \geq 1, S \subsetneq V, \lvert S \rvert = 2$$

    (保证解仅含一条闭合回路)

    $$x_{ij} \in \{ 0, 1 \}, (i, j) \in A$$

    (变量$x_{ij}$为二进制变量, 其等于 1 当仅当边$(i, j)$属于解回路)

    目前不明白上面的条件是如何转化为旅行商问题的，也不清楚将上述条件输入到求解器里，是否可以求解出和回溯法得到的相同的解。

    * 精确算法

        * 暴力搜索（Brute Force Search） 通过计算所有可能的城市访问顺序，检查每一条路径，选择最短的路径。

        * 动态规划（Dynamic Programming） 该方法通过子问题的递归求解，避免重复计算，利用如Held-Karp算法等，逐步构建全局最优解。

        * 分支定界法（Branch and Bound） 该方法构造一个搜索树，每个节点表示当前城市的部分路径，通过上下界进行剪枝，减少搜索空间。

        * 线性规划与割平面法（Linear Programming and Cutting Planes） 通过线性规划的松弛问题求解TSP，并使用割平面法来逐步排除不满足TSP约束的解。

    * 近似算法

        * 贪心算法（Greedy Algorithm） 从某一城市开始，每次选择离当前城市最近的未访问城市，直到所有城市被访问完。

        * 最近邻算法（Nearest Neighbor Algorithm） 从某个城市开始，每次选择最近的未访问城市，直到访问所有城市。
        2-Opt与3-Opt算法 通过交换路径中的两个或三个边来优化当前解，逐步改进路径。

        * 模拟退火算法（Simulated Annealing） 通过模拟物理退火过程，从随机解开始，逐步降低搜索空间的“温度”，以跳出局部最优。

        * 遗传算法（Genetic Algorithm） 遗传算法通过模拟生物进化的过程，使用选择、交叉和变异操作生成更优的解。

        * 蚁群算法（Ant Colony Optimization, ACO） 模拟蚂蚁通过信息素引导寻找最短路径的过程，逐步寻找最优解。

        * 粒子群优化（Particle Swarm Optimization, PSO） 该方法通过模拟鸟群的飞行轨迹，群体中的个体根据自身经验和群体经验来更新解。

* 跳表的 search 功能

    整个跳表的功能比较复杂，今天只看了`search()`函数。

    下面是对`search()`函数的解析：

    ```c
    bool search(int target) {
        SkiplistNode *curr = this->head;
        // 这里的 level 应该改成 num_levels 比较好
        for (int i = level - 1; i >= 0; i--) {
            // 如果当前节点的下一个节点存在，那么判断下一个节点是否小于 target
            // 若小于 target，继续往右搜索
            // 这样得到的当前节点 curr 就是在所有小于 target 的数中，最大的那个
            // 即 curr < target <= curr->forward
            while (curr->forward[i] && curr->forward[i]->val < target) {
                curr = curr->forward[i];
            }
            // 当搜索停止后，由于 for 的作用，i 会自然加 1
            // 因此在下一轮循环时，会跳到下一层搜索
        }
        // 此时 curr 已经搜索到了最底层，再向右走一个元素，
        // 这个元素要么等于，要么大于 target，当然也有可能是 NULL
        curr = curr->forward[0];
        if (curr && curr->val == target) {
            return true;
        } 
        return false;
    }
    ```

    自己盲写的一版：

    ```cpp
    bool search(int target)
    {
        SkiplistNode *cur = head;
        for (int cur_layer = level - 1; cur_layer >= 0; --cur_layer)
        {
            while (cur->forward[cur_layer]
                && cur->forward[cur_layer]->val < target)
            {
                cur = cur->forward[cur_layer];
            }
        }
        cur = cur->forward[0];
        if (!cur)
            return false;
        if (cur->val != target)
            return false;
        return true;
    }
    ```

    回头再研究下`add()`。

    对应的 leetcode 题目是`1206. 设计跳表`。

    ref: <https://oi-wiki.org/ds/skiplist/>

* 对跳表（skip list）的理解

    跳表有点像从海底长出来的海草，又有点像马里奥里的管道迷宫。节点有可能纵跨多层，最底层则包含所有元素。

    我们从最顶层开始向右搜索，每次遇到 right bound 就向下走一层（为什么总能保证在遇到 right bound 时，总是有向下的通道？），直到走到最底层。最底层的 right bound，要么正好是我们要找到数字，要么就说明我们要找的元素不存在。（为什么在最底层一定可以做出判断是否遇到？有没有可能在上面几层就遇到 target？）

* 有关元素的递推性质

    将每个元素替换为右侧最大元素

    给你一个数组 arr ，请你将每个元素用它右边最大的元素替换，如果是最后一个元素，用 -1 替换。

    完成所有替换操作后，请你返回这个数组。

    示例 1：

    输入：arr = [17,18,5,4,6,1]
    输出：[18,6,6,6,1,-1]
    解释：
    - 下标 0 的元素 --> 右侧最大元素是下标 1 的元素 (18)
    - 下标 1 的元素 --> 右侧最大元素是下标 4 的元素 (6)
    - 下标 2 的元素 --> 右侧最大元素是下标 4 的元素 (6)
    - 下标 3 的元素 --> 右侧最大元素是下标 4 的元素 (6)
    - 下标 4 的元素 --> 右侧最大元素是下标 5 的元素 (1)
    - 下标 5 的元素 --> 右侧没有其他元素，替换为 -1
    示例 2：

    输入：arr = [400]
    输出：[-1]
    解释：下标 0 的元素右侧没有其他元素。
    

    提示：

    1 <= arr.length <= 104
    1 <= arr[i] <= 105

    代码：

    1. 自己写的

        ```cpp
        class Solution {
        public:
            vector<int> replaceElements(vector<int>& arr) {
                int max_val = -1;
                vector<int> ans(arr.size());
                for (int i = arr.size() - 1; i >= 0; --i)
                {
                    ans[i] = max_val;
                    if (arr[i] > max_val)
                    {
                        max_val = arr[i];
                    }
                }
                return ans;
            }
        };
        ```

    1. 官方答案

        ```cpp
        class Solution {
        public:
            vector<int> replaceElements(vector<int>& arr) {
                int n = arr.size();
                vector<int> ans(n);
                ans[n - 1] = -1;
                for (int i = n - 2; i >= 0; --i) {
                    ans[i] = max(ans[i + 1], arr[i + 1]);
                }
                return ans;
            }
        };
        ```

    题目本身并不难，也很容易做对。但是注意到官方答案并没有使用单独一个变量`max_val`，而是使用了`max(ans[i + 1], arr[i + 1])`这样的递推式。

    是否有可能在其他题目中，某个数组的元素满足某种性质，从而可以使用递推关系得到简便的答案，我们不得而知。现在我们来看当前这道题：`ans[i+1]`本身就表示了`arr[i+2]`及之后的所有数的最大值，让它和`arr[i+1]`比较，便能得到`arr[i+1]`及之后的所有数的最大值。`ans[i]`表示的正好是`arr[i+1]`及之后的所有数的最大值，因此可由`max(ans[i+1], arr[i+1])`得到。

    接下来考虑边界，当`arr[i+1]`为最后一个元素时，`ans[i+1]`为`-1`，`ans[i]`可以正常计算，没有问题。但此时`i`最大取值为`arr.size() - 2`，不能取到最后一个。若`ans[i]`为最后一个元素，后面的`i+1`肯定越界，因此`i`不能取到最后一个。这样`ans[i]`作为最后一个元素时，必须单独计算。幸好题目直接指定其为`-1`。

    可以看出来，整个过程还是比较费脑子的。

* 使用 c + 回溯算法生成不同测试的组合，本意是想减少测试用例中的 for 循环，但是实际上并不会减少很多

    比如下面一段测试代码：

    ```cpp
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 4; ++b)
        {
            for (int c = 0; c < 2; ++c)
            {
                vector<int> my_vec;
                for (int d = 0; d < 4; ++d)
                {
                    vec.push_back(d);
                }
            }
        }
    }
    ```

    如果指定了`a`, `b`的 ranges，那么其实只能消除两层 for：

    ```cpp
    struct CurrentConfig
    {
        int a;
        int b;
    } cur_cfg;

    void test(struct CurrentConfig *cur_cfg)
    {
        int a = cur_cfg->a;
        int b = cur_cfg->b;
        for (int c = 0; c < 2; ++c)
        {
            vector<int> my_vec;
            for (int d = 0; d < 4; ++d)
            {
                vec.push_back(d);
            }
        }
    }
    ```

    因为`c`没有写在 current config 内，所以这层循环没有消除掉；因为`d`是必要的循环操作，所以无法写到 current config 内。

    因此，对于可以写到 current config 中的参数，其实可以直接写个 bash 或 python 生成一个组合，放在 argc 和 argv 里传递给测试程序就可以，这样可以尽量减少对测试程序的改动；对于必要的循环，无论如何也消除不掉。

* 尝试自己实现一个 unordered_map

    里面的`pair<A, B>`其实是 list node，即

    ```cpp
    struct ListNode
    {
        A a;
        B b;
        struct ListNode *next;
    } ;
    ```

* `unordered_map<string, int> m;`和`vector<pair<string, int>> v;`各有什么优劣？

    下面的代，如果将`ConfigSample`改成`unordered_map<string, int>`，会有什么优缺点？

    ```cpp
    #include <vector>
    #include <string>
    #include <iostream>
    using namespace std;

    struct RangeInfo_2
    {
        string name;
        int start;
        int end;
        int step;  // minus means multiplying
    };

    typedef vector<pair<string, int>> ConfigSample;
    typedef vector<RangeInfo_2> ConfigRanges_2;
    ConfigSample path_2;
    vector<ConfigSample> all_path_2;
    void backtrack_2(ConfigRanges_2 &config_ranges, int depth)
    {
        if (path_2.size() == config_ranges.size())
        {
            all_path_2.push_back(path_2);
            return;
        }

        int start = config_ranges[depth].start;
        int end = config_ranges[depth].end;
        string &name = config_ranges[depth].name;
        for (int val = start; val <= end; ++val)
        {
            path_2.push_back({name, val});
            backtrack_2(config_ranges, depth + 1);
            path_2.pop_back();
        }
    }

    void display_path(vector<pair<string, int>> &path)
    {
        for (int i = 0; i < path.size() - 1; ++i)
        {
            cout << path[i].first << ": " << path[i].second << ", ";
        }
        cout << path.back().first << ": " << path.back().second << endl;
    }

    int main()
    {
        vector<RangeInfo_2> config_ranges_2 {
            {"mr_len", 1, 3, 1},
            {"sge_num", 1, 2, 1}
        };

        backtrack_2(config_ranges_2, 0);

        for (ConfigSample &path: all_path_2)
        {
            display_path(path);
        }
    }
    ```

* 多个变量在指定 range 中的不同组合

    比如变量`a`的 range 为 0 到 2，`b`的 range 为 4 到 5，现在需要拿到 a 和 b 的所有可能的组合，如下所示：

    ```
    a: 0, b: 4
    a: 0, b: 5
    a: 1, b: 4
    a: 1, b: 5
    a: 2, b: 4
    a: 2, b: 5
    ```

    一个比较简单的想法是写 2 层 for 循环：

    ```cpp
    int main()
    {
        for (int a = 0; a <= 2; ++a)
        {
            for (int b = 4; b <= 5; ++b)
            {
                printf("a: %d, b: %d\n", a, b);
            }
        }

        return 0;
    }
    ```

    output:

    ```
    a: 0, b: 4
    a: 0, b: 5
    a: 1, b: 4
    a: 1, b: 5
    a: 2, b: 4
    a: 2, b: 5
    ```

    如果变量的数量为 3 个，那就得写 3 层 for 循环；如果变量的数量为 4，那就得写 4 层 for 循环。如果变量的数量不确定有多少个，那就不可能用确定的 for 循环的层数去应对所有情况。

    这时候可以用回溯算法解决这个问题：

    ```cpp
    #include <vector>
    #include <stdio.h>
    using namespace std;

    void display_path(vector<int> &path)
    {
        for (int i = 0; i < path.size() - 1; ++i)
        {
            printf("%d, ", path[i]);
        }
        printf("%d\n", path.back());
    }

    vector<int> path;
    vector<vector<int>> paths;
    void backtrack(vector<pair<int, int>> &ranges, int depth)
    {
        if (path.size() == ranges.size())
        {
            // display_path(path);
            paths.push_back(path);
            return;
        }

        int start = ranges[depth].first;
        int end = ranges[depth].second;
        for (int val = start; val <= end; ++val)
        {
            path.push_back(val);
            backtrack(ranges, depth + 1);
            path.pop_back();
        }
    }

    int main()
    {
        vector<pair<int, int>> ranges {
            {1, 3}, 
            {2, 3},
            {3, 4}
        };

        backtrack(ranges, 0);

        for (vector<int> &path: paths)
        {
            display_path(path);
        }

        return 0;
    }
    ```

    output:

    ```
    1, 2, 3
    1, 2, 4
    1, 3, 3
    1, 3, 4
    2, 2, 3
    2, 2, 4
    2, 3, 3
    2, 3, 4
    3, 2, 3
    3, 2, 4
    3, 3, 3
    3, 3, 4
    ```

    这份代码只需要指定`ranges`，不用改动其他部分，就可以输出正确的结果。

    如果变量的数目太多，那么可能会递归层数太深爆栈，也有可能`paths`占用的内存过大。一个比较好的解决办法是使用 generator，每次只生成一个结果，还不需要改变函数的代码逻辑。

* 有关排列和组合

    全排列：123, 132, 213, 231, 312, 321...

    组合：a, b, c, d, ab, ac, ad, bc, bd, cd, abc, acd, bcd, abcd...

    那么 111, 112, 113, 121, 122, 123, 211, 212, 213, 221, 222, 223 这种算什么问题？它既不是数量可变的，又不是可交换位置的。

* 如果一个数据可以使用数组计数来抽取特征，那么可以使用排序使得数据相等

    可以尽量将数据转换成字符串的形式，以便使用哈希表加速。

* `2589. 完成所有任务的最少时间`

    一个初步的想法：

    1. 这道题可以转化为，对于第`i`个区间，从集合`[start, end]`中选出一个最少含`duration`个元素的子集。假设有`n_i`个选法。

    2. 对于每个区间，选一个子集，然后对所有子集取并集，共有`n_0 * n_1 * ... * n_{N-1}`种取法。

    3. 取这$\Pi_0^{N-1} n_i$个并集中，元素最少的那个并集，其元素个数就是答案。

* 如果在写一个正向的动态规划的时候用到了以前的值，或者在确定以后的值时需要分类讨论，那么它的本质其实还是反向的动态规划。

    因为分类讨论本质还是在确定“未来”的未来值时，用到了“未来”的当前值的状态。那我还不如直接在未来全部用过去值，这样的结构更清晰明了。

    一个写正向动态规划写不下去的案例：`ref_14`。写到条件 5 的时候，想到 5 的两种分解情况，6 对应着能分解和不能分解的两种情况，意识到做不下去了。

    * 假如写 4 = 3 + 1，那么 4 这个位置就应该是 -1

        然而实际上 4 = 2 + 2，所以 4 可以被分解

        但是问题来到了 5，如果我们从 4 跳到 5，一定是从 2 过来的，那么 5 一定被分解成 2 + 3

        现在到了 6，由于 5 最后留了 3，所以 6 必须被分解成 2 + 3 + 1，这样就一定会返回 -1。

        然而，6 其实可以被分解成 3 + 3。

        由此我们可以得出，不可能存在一个正向的方法实现动态规划。

        猜想：是否所有不只依赖当前状态的动态规划都无法写成正向的？

        猜想：如果一个正向的动态规划写法需要依赖之前的状态，那么是否无论如何增加状态也无法解耦？

            解耦：即只依赖当前索引 i 的数据，就可以推导出后面的最优值。

* 动态规划的一些心得

    从前向后推导要求可以更新多个位置，并且可以覆盖原来更新过的位置。

    但是一个位置是否更新不能影响到下一个位置，即不能与下一个位置有相关性，因为这样就引入了记忆状态。

    可以引入多个状态数组来进行相关性解耦。比如同时引入一个`int dp[2][n]`，其中`dp[0][n]`用来记录最优值，`dp[1][n]`用来记录某种状态（一个简易的有限状态机）。

* Hanoi tower, 假设现在有 a, b, c 三个塔，a 上有 n 个盘子，上小下大，要求借助 c 把它们都移动到 b 上，并且要求在移动的过程中，必须保持所有的叠放都是上小下大。

    我们假设这个游戏已经到了最后一步，最后一步的情况是 a 上只剩底部一个最大的盘子，其余的 n - 1 个盘子都在 c 上，并且保持上小下大的特点。

    此时只需要把 a 上的最后一个盘子移动到 b 上，然后把 c 上的所有盘子移动到 b 上就可以了。

    此时我们把问题分解成了两个子问题：

    ```
    original problem: (n, 0, 0) -> (0, n, 0)
    child problem 1: (n, 0, 0) -> (1, 0, n-1) -> (0, 1, n-1)
    child problem 2: (0, 1, n-1) -> (0, n, 0)
    ```

    注意到子问题 1，它其实和原问题一样，只不过把 b 作为了缓冲区，总移动数量变成了 n - 1。

    子问题 2 中，a 作为了缓冲区，总移动数量仍然是 n - 1。

    这样就把一个总移动量为 n 的问题拆成了两个总移动量为 n - 1 的问题。当总移动量为 1 时，就可以直接移动了。

    代码：

    ```cpp
    #include <iostream>
    using namespace std;

    void hanoi(int n, int from, int to)
    {
        if (n == 1)
        {
            printf("move from %d to %d\n", from, to);
            return;
        }
        int another_idx;
        for (int i = 0; i < 3; ++i)
        {
            if (i == from || i == to)
                continue;
            another_idx = i;
            break;
        }
        hanoi(n-1, from, another_idx);
        printf("move from %d to %d\n", from, to);
        hanoi(n-1, another_idx, to);
    }

    int main()
    {
        hanoi(3, 0, 1);
        return 0;
    }
    ```

    output:

    ```
    move from 0 to 1
    move from 0 to 2
    move from 1 to 2
    move from 0 to 1
    move from 2 to 0
    move from 2 to 1
    move from 0 to 1
    ```

    书上给的 example:

    ```cpp
    #include <iostream>
    using namespace std;

    void hanoi(int n, int x, int y, int z)
    {
        if (n >= 1)
        {
            hanoi(n-1, x, z, y);
            printf("move from %d to %d\n", x, y);
            hanoi(n-1, z, y, x);
        }
    }

    int main()
    {
        hanoi(3, 0, 1, 2);
        return 0;
    }
    ```

    output:

    ```
    move from 0 to 1
    move from 0 to 2
    move from 1 to 2
    move from 0 to 1
    move from 2 to 0
    move from 2 to 1
    move from 0 to 1
    ```

    不清楚他是如何做到这么简洁的。

* 冒泡排序

    ```cpp
    #include <iostream>
    #include <random>
    #include <algorithm>
    using namespace std;

    int main()
    {
        int arr[10];
        for (int i = 0; i < 10; ++i)
            arr[i] = i;
        shuffle(arr, arr+10, mt19937(random_device{}()));
        
        for (int num: arr)
            printf("%d, ", num);
        putchar('\n');

        for (int i = 0; i < 9; ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                if (arr[j] > arr[j+1])
                    swap(arr[j], arr[j+1]);
            }
        }

        for (int num: arr)
            printf("%d, ", num);
        putchar('\n');
        return 0;
    }
    ```

    第`0`轮排完，最后一个数一定是最大的，即最后`1`**个**数是排好的。

    第`1`轮排完，最后两个数是最大的，即最后`2`**个**数是排好的，并且排好的区间的左界一定大于左界之外的所有数。（假设所有数都不重复）

    由此可以归纳出猜想：第`i`轮排完，最后`i+1`个数是排好的。

    当剩 2 个数没排时，这 2 个数比排好的数都小，但是这 2 个数的大小关系未知，还得再排一次。

    当只剩 1 个数没排时，它一定是最小的，不需要再排了。

    因此我们要排好`n-1`个数。应用前面归纳出的猜想，当`i+1 = n-1`时，`i = n-2`，那么只需要排完第`n - 2`轮就可以了。

    也就是说，`i`的取值范围为`[0, n-2]`，一共排`n-1`轮。

    这是一个序数与基数之间的归纳猜想，可以看出即使是冒泡排序，也有一定的复杂性。

    我们可以把序数`i`直接換成基数“轮数”。即第 1 轮排完，最后 1 个数是最大的；第 2 轮排完，最后 2 个数是最大的。

    第`n`轮排完，最后`n`个数是最大的。我们只需要排`n-1`个数，因此需要排`n-1`轮。

    当`i`的取值范围为`[0, n-1]`时，共执行`n`轮。要想执行`n-1`轮，只需要让`i`取到`n-2`就可以了。

    这样稍微简单一点。

    大部分的算法题都是这样归纳出猜想以节约时间，是否有更好的分析方法？

* 归并排序

    ```cpp
    #include <iostream>
    #include <random>
    #include <algorithm>
    using namespace std;

    void merge_sort(int arr[10], int left, int right)
    {
        if (right - left + 1 <= 1)
            return;

        int mid_idx = left + (right - left) / 2;
        merge_sort(arr, left, mid_idx);
        merge_sort(arr, mid_idx+1, right);
        int i = left, j = mid_idx + 1, p = 0;
        int *temp = (int*) malloc(right - left + 1);
        while (i <= mid_idx && j <= right)
        {
            if (arr[i] < arr[j])
                temp[p++] = arr[i++];
            else
                temp[p++] = arr[j++];
        }
        while (i <= mid_idx)
            temp[p++] = arr[i++];
        while (j <= right)
            temp[p++] = arr[j++];
        for (p = 0; p < right - left + 1; ++p)
            arr[left + p] = temp[p];
        free(temp);
    }

    int main()
    {
        int arr[10];
        for (int i = 0; i < 10; ++i)
            arr[i] = i;
        shuffle(arr, arr+10, mt19937(random_device{}()));

        for (int num: arr)
            printf("%d, ", num);
        putchar('\n');
        
        merge_sort(arr, 0, 9);

        for (int num: arr)
            printf("%d, ", num);
        putchar('\n');
        return 0;
    }
    ```

    一些边界条件没想清楚：

    * `int mid_idx = left + (right - left) / 2;`

        这一行为什么能保证最后一定是左右两个区间长度分别为`[0, 1], [1, 0], [1, 1]`这三种情况？

    * `while (i <= mid_idx && j <= right)`

        这一行是否可以保证，left 和 right 的长度最多只差 1？

        如果左右两个区间的长度最大只差 1，那么后面的两行

        ```cpp
        while (i <= mid_idx)
        while (j <= right)
        ```

        就可以不这么写了。

    归并排序其实是一个后序遍历的树，因为只有处理完了当前节点的两个子节点，才能去归并当前节点。

## Basic

### 循环的执行次数

1. `for`循环的次数

    对数组的索引进行遍历：

    如果索引从`0`开始，条件是`< n`，比如`for (int i = 0; i < n; ++i)`，那么一共需要执行`n`次。

    如果索引从`n-1`开始，条件是大于`> -1`，比如`for (int i = n-1; i > -1; --i)`，那么也是需要执行`n`次。

    索引模型（一定要两端都能取到）：

    ```
    [0, ..., n-1]

    [i, ..., j]  长度：j - i + 1
    ```

## Sort

### Bubble sort

```c++
void bubble_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - 1 - i; ++j)
        {
            if (nums[j] < nums[j+1])
                swap(nums[j], nums[j+1]);
        }
    }
}
```

1. 外层循环确定了执行的轮数，一共需要执行`n-1`轮。每执行一轮，都能保证当前索引区间`[0, n-1-i]`中最后一个元素是最大的，也就是说整个数组最后的`i+1`个元素是递增顺序。第一个元素就不需要再排了，所以一共需要`n-1`轮。

1. 内层循环的变量`j`确定了比较两个数字时，前一个数字的索引的位置，位置的范围为`[0, n-2-i]`，即每轮需要比较`n-1-i`次。因为需要比较当前元素和下一个元素，所以`j`不能取到当前区间末尾的元素。

### Select sort

```c++
void select_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 0; i < n - 1; ++i)
    {
        int min_idx = i;
        for (int j = i + 1; j < n; ++j)
        {
            if (nums[j] < nums[min_idx])
                min_idx = j;
        }
        swap[nums[i], nums[min_idx]];
    }
}
```

1. 外层循环确定了执行的轮数，一共需要`n-1`轮。每轮结束后，都能保证当前索引区间`[i, n-1]`中第一个元素是最小的，即整个数组的前`i+1`个元素是递增顺序。最后一个元素不需要再排了，所以一共需要`n-1`轮。

1. 内层循环确定了比较的次数，一共需要比较`n-1-i`次。

### Insertion sort

插入排序，灵感来源于我们对扑克牌进行排序。假设现在手里有了一些已经排好序的牌，现在来了一张新牌，我们直接的想法是把比这张牌大的牌都往后移动一个位置，然后把这张牌插入进去。

对于算法，假设目前区间是这样的：`[a ... b, c, d ... e]`，其中`[a ... b]`已经排好序，`c`是要处理的下一个元素，`[d ... e]`是无序的。一个简单的想法是让`b`和`c`进行比较，如果`b`大于`c`，那么交换这两个数的位置。然后再让`c`和前一个元素比较，如果`c`小于前一个元素，那么就交换位置。直到最后`c`和`a`进行比较，并判断是否交换位置。

代码实现如下：

```c++
void insertion_sort(vector<int> &nums)
{
    for (int i = 1; i < nums.size(); ++i)  // 第一个元素组成的区间不需要维护，因此 i 从 1 开始
    {
        for (int j = i; j > 0; --j)  // 从后往前处理，因为后面要用到 j-1，所以 j 最小取到 1
        {
            if (nums[j] < nums[j-1])  // 比较并交换 j 和 j-1 对应的两个元素
                swap(nums[j], nums[j-1]);
            else
                break;  // 维护有序区间结束，退出循环
        }
    }
}
```

上面的代码中，`i`表示的不是处理轮数，而是要 insert 的 number 的索引。`j`表示的是两个相邻 number 的索引。

事实上，`swap()`表示的是三个操作：一次存储临时值，两次赋值。如果我们事先把新来的数字存储起来，然后只是向右移动序列，等合适的位置空出来后，再把新来的数字放进去，就可以避免一直`swap()`了。代码如下：

```c++
void insertion_sort(vector<int> &nums)
{
    for (int i = 1; i < nums.size(); ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1 && nums[j] > num)
        {
            nums[j+1] = nums[j];
            --j;
        }
        nums[j+1] = num;
    }
}
```

### Merge sort

merge sort 的灵感灵源是拿两副已经排好序的扑克牌，每次只比较最上面的一张的大小，每次都只把较小的一张放下来。这样当手里的两副牌都比较完后，我们得到的第三副牌就是已经排好序的。

我们首先要把牌分成两摞，依次分下去，直到每摞都只有 0 张或 1 张。事实上，这个过程是展开了一棵二叉树。

然后对于合并的过程，我们只需要创建一个临时数组，把两摞按大小合并进去就可以了，最后再把这个临时数组复制到原数组上。

```cpp
void merge(vector<int> &nums, int l, int m, int r)
{
    int i = l, j = m + 1;
    int p = 0;
    vector<int> temp(r - l + 1);
    while (i <= m && j <= r)
    {
        if (nums[i] < nums[j])
            temp[p++] = nums[i++];
        else
            temp[p++] = nums[j++];
    }
    while (i <= m)
        temp[p++] = nums[i++];
    while (j <= r)
        temp[p++] = nums[j++];
    p = 0;
    i = l;
    while (i <= r)
        nums[i++] = temp[p++];
}

void partition(vector<int> &nums, int l, int r)
{
    if (r - l + 1 <= 1)
        return;
    int n = nums.size();
    int m = l + (r - l) / 2;
    partition(nums, l, m);
    partition(nums, m+1, r);
    merge(nums, l, m, r);
}

void merge_sort(vector<int> &nums)
{
    partition(nums, 0, nums.size() - 1);
}
```

归并排序 merge sort，其实本质上是树的后序遍历。即我们想象手上有两副牌，分别代表一个父节点的两个子节点，我们必须等两个子节点都被处理好排好序后，才能处理父节点（即 merge 操作）。

`partition()`函数是怎么被构建出来的呢？

首先，我们知道拿到一副牌后，要先把它分成两半，然后对这两半进行递归处理。为了把牌分成两半，我们总得知道中间牌的索引吧。为了知道中间牌的索引，我们总得知道当前手中牌的数量吧。因为我们每次都是简单地从中间切牌，所以实际上拿到的牌相当于一个“连续子数组”。因为它是连续子数组，所以我们只需要给出区间左端点的索引和右端点的索引就可以了。以上这些线索引导我们：必须创建一个变量`m`以记录中间牌索引，必须给函数传递`left, right`参数，以确定牌的范围。我们可以写出如下代码：

```cpp
void partition(vector<int> &nums, int l, int r)
{
    int m = l + (r - l) / 2;
    partition(nums, l, m);
    partition(nums, m+1, r);
}
```

接下来我们判断递归的结束条件：当手里没有牌，或只剩一张牌时，就不需要再分了。这时候我们需要回退到父节点合并手中排序好的两副牌。在合并时，我们总得知道手里都是哪两副牌吧，这个线索告诉我们，`merge()`函数一定接收 3 个索引参数，用来定位两副牌。这些线索引导我们写出如下代码：

```cpp
void merge(vector<int> &nums, int l, int m, int r)
{

}

void partition(vector<int> &nums, int l, int r)
{
    if (r - l + 1 <= 1)
        return;
    int m = l + (r - l) / 2;
    partition(nums, l, m);
    partition(nums, m+1, r);
    merge(nums, l, m, r);
}
```

这个写法是很明显的二叉树的后序遍历，即先处理左节点和右节点，最后再处理本节点。

接下来我们继续完善`merge()`函数。在 merge 函数中，我们已经有了数组`[a, ..., b, c, ..., d]`其中`[a, ..., b]`是第一副牌，`[c, ..., d]`是第二幅牌，我们需要把这两个有序子数组合并成一个大的有序数组。可以原地修改吗？不可以，因为大数组没有额外的空间，不能倒序双指针。我们可以新建个数组用于缓存结果，等排完序后，再把拷贝回去；也可以把`[c, ..., d]`复制到临时数组中，然后用倒序双指针。

方案一：使用完整的临时数组

```cpp
void merge(vector<int> &nums, int l, int m, int r)
{
    vector<int> temp(r - l + 1);
    int i = l, j = m + 1;
    int p = 0;
    while (i <= m && j <= r)  // 双指针一起向前走
    {
        if (nums[i] < nums[j])
            temp[p++] = nums[i++];
        else
            temp[p++] = nums[j++];
    }
    while (i <= m)  // 双指针一个走到头，另一个还没走完，后面跟上双 while 来处理这种情况，是常见的写法
        temp[p++] = nums[i++];
    while (j <= r)
        temp[p++] = nums[j++];
    p = 0;
    i = l;
    while (i <= r)  // 最后再把临时数组中的排好序的结果，放回到原数组中
        nums[i++] = temp[p++];
}
```

方案二：只缓存数组二，然后用倒序双指针

```cpp
void merge(vector<int> &nums, int l, int m, int r)
{
    vector<int> temp(nums.begin()+m+1, nums.end());
    int i = m, j = r - (m+1) + 1 - 1;  // 其中，r - (m+1) + 1 指的是 temp 的长度，-1 表示根据长度计算最后一个元素的索引
    int p = r;
    while (i > -1 && j > -1)
    {
        if (temp[j] >= nums[i])
            nums[p--] = temp[j--];
        else
            nums[p--] = nums[i--];
    }
    while (j > -1)  // 如果 i > -1，那么相当于大数组的指针没走完，这种情况就不需要管了
        nums[p--] = temp[j--];
}
```

可以看到倒序双指针的效率要高于方案一。

### Quicksort

快速排序有一点点像归并排序的前半段，有点像冒泡排序。灵感可能来自于分治。

快速排序基本的想法是先设置一个分界元素，通过冒泡的方法，将数组分成两部分，左半部分都小于等于分界元素，右半部分都大于分界元素。再对两侧进行排序。然后再设置下一个分界元素，再对两侧进行排序：

```cpp
void quick_sort(vector<int> &nums)
{
    int pivot_idx = get_pivot_idx();
    int pivot_num = nums[pivot_idx];
    while ()
    {
        int num_left = find_a_elm_smaller_than_pivot_num();
        int num_right = find_a_elm_greater_tham_pivot_num();
        swap(num_left, num_right);
    }
}
```

但是目前`pivot_num`在数组中间的话，我们可以在轮到它时选择跳过，也可以将它先和最右边元素交换位置，等处理完了再交换回来。对于找到两个适合的元素，一个想法是双重循环，先找到一个大于等于`pivot_num`的元素，记为`i`，再从`i+1`开始，找到一个小于`pivot_num`的元素，记为`j`，然后再交换`i`和`j`对应的元素就可以了；另一个想法是对撞双指针，分别从数组的两端，找一个比`pivot_num`小的和比`pivot_num`大的元素。

根据以上想法，我们可以写出下面四种情况：

1. 跳过`pivot_num` + 双重循环

    ```cpp

    ```

1. 交换`pivot_num` + 双重循环

1. 跳过`pivot_num` + 对撞双指针

1. 交换`pivot_num` + 对撞双指针

从以上可以看出，跳过并不是一个明智的选择。因为最终我们一定要交换`pivot_num`到合适的位置，起码要交换一次，而跳过的话，会增加很多比较的运算。另外，双重循环如果总是遇到`[5, 1, 1, 1, 1]`这样的情况，那么`5`会被一个一个地移动到最右边，交换了很多次，而用对撞双指针就没这个问题。

综上，我们选择交换 + 对撞双指针为最优解。

粗略排完了当前数组，我们得到一个新的索引，这个索引左侧的元素都小于索引处的元素，索引右侧的元素都大于等于索引处的元素。

接下来我们需要细排子数组。显然快速排序的本质是一个先序遍历，当对当前节点对应的区间进行粗排序，再到下一个子区间进行精排序。我们希望`quick_sort()`的函数参数不增多，因此稍微改写下结构：

```cpp
int partition_sort(vector<int> &nums, int left, int right, int pivot_idx)
{
    // 交换 + 对撞双指针
}

void partition(vector<int> &nums, int left, int right)
{
    int pivot_idx = get_pivot_idx();
    int idx = partition_sort(nums, left, right, pivot_idx);  // 先处理当前区间，根据函数返回值得到排好的两个子区间
    partition(nums, left, idx-1);  // 处理左区间
    partition(nums, idx+1, right);  // 处理右区间
}

void quick_sort(vector<int> &nums)
{
    partition(nums, 0, nums.size() - 1);
}
```

由于是树的先序遍历，所以我们想一下终止条件，当当前节点的区间长度只有 1 或 0 的时候，就不需要排序了，因此将此设置为终止条件：

```cpp
void partition(vector<int> &nums, int left, int right)
{
    if (right - left + 1 <= 1)
        return;
    int pivot_idx = get_pivot_idx();
    // ...
}
```

最后我们再考虑一个问题：如何得到分界元素呢？我们可以随机选一个，也可以取中间元素。随机选一个效果可能好一点。

```cpp
int pivot_idx = left + rand() % (right - left + 1);
```

为什么这里要对`right - left + 1`取模，而不是`right - left`？

最终我们可以得到完整版的快速排序算法：

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
using namespace std;

int sort_with_pivot_elm(vector<int> &nums, int left, int right, int idx)
{
    int pivot_elm = nums[idx];
    swap(nums[idx], nums[right]);
    int i = left, j = right - 1;
    while (i < j)
    {
        while (i < j && nums[i] < pivot_elm) ++i;
        while (i < j && nums[j] >= pivot_elm) --j;
        swap(nums[i], nums[j]);
    }
    if (nums[i] > nums[right])  // 没想明白为啥要加这个，背会
        swap(nums[right], nums[i]);
    return i;
}

void partition(vector<int> &nums, int left, int right)
{
    if (right - left + 1 <= 1)
        return;
    int pivot_idx = left + rand() % (right - left + 1);
    int idx = sort_with_pivot_elm(nums, left, right, pivot_idx);
    partition(nums, left, idx-1);
    partition(nums, idx+1, right);
}

void quick_sort(vector<int> &nums)
{
    partition(nums, 0, nums.size() - 1);
}

int main()
{
    srand(time(NULL));
    vector<int> nums({3, 4, 5, 1, 1, 3, 4, 2});
    quick_sort(nums);
    for (int num: nums)
        cout << num << ", ";
    cout << endl;
    return 0;
}
```

#### 有关边界情况的问题

如何解决这些代码里的边界问题呢？

```cpp
int quicksort_helper(vector<int> &nums, int l, int r, int idx_pivot)
{
    int val = nums[idx_pivot];
    swap(nums[idx_pivot], nums[r]);
    int i = l, j = r-1;
    while (i < j)  // 假如对于数组 [3, 4]，i = 0, j = 0，此时不会进入循环
    {
        while (i < j && nums[i] < val) ++i;
        while (i < j && nums[j] >= val) --j;
        swap(nums[i], nums[j]);
    }
    swap(nums[i], nums[r]);  // 这里 r = 1, i = 0，会把 [3, 4] 交换成 [4, 3]
    return i;
}

void partition(vector<int> &nums, int l, int r)  // 先处理当前节点，再处理左节点，再处理右节点
{
    if (r - l + 1 <= 1)
    {
        cout << "partition: [" << l << ", " << r << "], not enough length" << endl;
        return;
    }
    int idx_pivot = l + (r - l) / 2;
    cout << "partition: [" << l << ", " << r <<
     "], idx: " << idx_pivot << ", x: " << nums[idx_pivot] << ", ";
    int delim = quicksort_helper(nums, l, r, idx_pivot);
    
    cout << "delim: " << delim << ", " << "nums: ";
    for (auto &num: nums)
    {
        cout << num << ", ";
    }
    cout << endl;
    partition(nums, l, delim);
    partition(nums, delim + 1, r);
}

void quicksort(vector<int> &nums)
{
    partition(nums, 0, nums.size() - 1);
}

int main()
{
    vector<int> nums({5, 3, 1, 2, 4});
    quicksort(nums);
    for (auto &num: nums)
        cout << num << ", ";
    cout << endl;
    return 0;
}
```

## Binary search

给定一个`n`个元素有序的（升序）整型数组`nums`和一个目标值`target`，写一个函数搜索`nums`中的`target`，如果目标值存在返回下标，否则返回`-1`。

分析：无论数组中的元素是正数，负数，还是零，只要数组**有序**，所以可以用二分查找。二分查找可以判断某个值是否在数组中，找到这个值的某个索引，或者找到这个值的左边界或右边界。

### exact element

代码：

```c++
int binary_search(vector<int> &nums, int target) {
    int left = 0, right = nums.size() - 1, mid;  // 这里 right 取到最后一个元素，而不是 nums.size()，因为下面的 left <= right 也取到等号了。假如 right 在这里取到 nums.size()，那么假如在 [2, 3, 4] 中找 5，left 就会取到 3，mid 也会取到 3，从而数组越界（即，如果搜索一个大于最大值的数，会导致 left 和 mid 越界）
    while (left <= right) {  // 因为下面在更新 right 的时候用的是 mid - 1，所以这里的等号必须带上。否则在找 [2, 3, 4, 5] 中的 5 时，最后两掓是 left = 2, right = 3，mid = 2，然后 left = mid + 1 = 3，因为 left < right，就会跳出循环，从而 mid 无法取到 3，最终无法找到 5
        mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;  // 为什么这里是 left = mid + 1，而不是 left = mid
        else if (nums[mid] > target) right = mid - 1;  // 为什么这里是 right = mid - 1，而不是 right = mid
        else return mid;
    }
    return -1;
}
```

二分查找的要求：

1. `mid`能够取到所有元素的下标，尤其是边界处的。
1. 尽量不要找重复的元素（比如某个区间边界处）


### left bound

### right bound

## Math

### 最大公约数

```java
static int gcd(int p, int q) {
        if (q == 0) return p;
        int r = p % q;
        return gcd(q, r);
    }
```

## Array-processing

* 找最大值

    find the maximum of the array values

    ```java
    static int findMax(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length - 1; ++i)
            if (arr[i] > max) max = arr[i];
        return max;
    }
    ```

* 计算均值

    compute the average of the array values

    ```java
    static double calcMean(double[] arr) {
        double n = arr.length;
        double sum = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            sum += arr[i];
        }
        return sum / n;
    }
    ```

* 复制另一个数组

    copy to another array

    ```java
    int N = a.length;
    double[] b = new double[N];
    for (int i = 0; i < N; i++)
        b[i] = a[i]; 
    ```

* 反转数组

    reverse the elements within an array

    ```java
    static void reverseArray(int[] arr) {
        int left = 0, right = arr.length - 1;
        int n = arr.length;
        n /= 2;
        int temp;
        for (int i = 0; i < n; ++i) {
            temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            ++left;
            --right;
        }
    }
    ```

    ```java
    int N = a.length;
    for (int i = 0; i < N/2; i++)
    {
        double temp = a[i];
        a[i] = a[N-1-i];
        a[N-i-1] = temp;
    }
    ```

* 矩阵乘法

    ```java
    static double[][] matMultiply(double[][] A, double[][] B) {
        int m = A.length, l = A[0].length, n = B[0].length;
        double[][] C = new double[m][n];
        for (int k = 0; k < l; ++k) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
    ```

## 树，递归，与数据保持

计算机的很多算法都是树的变形。树与递归也经常有联系。这两者中间有个重要的问题，即我们如何记录数据？

常见的数据有 4 种：函数的参数，函数的返回值，函数中的变量，独立在函数之外的数据。

## 树的节点与递归

判断一棵树是否为空，可以在`if()`中处理，也可以交给递归处理。这两种有什么区别呢？

看一个例子：假如我们想对一棵树进行先序遍历，第一种方法为

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> ans;
    vector<int> preorderTraversal(TreeNode* root) {
        if (!root)
            return ans;
        ans.push_back(root->val);
        if (root->left)
            preorderTraversal(root->left);
        if (root->right)
            preorderTraversal(root->right);
        return ans;
    }
};
```

这种方式完全以`if`的形式判断当前节点有没有左右子节点。前面的`if (!root)`实际上只是判断外部的输入是否为空树，并不能起到判断空节点的作用。

我们也可以把它写成递归处理的方式：

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> ans;
    void pre(TreeNode *r)
    {
        if (!r)
            return;
        ans.push_back(r->val);
        pre(r->left);
        pre(r->right);
    }

    vector<int> preorderTraversal(TreeNode* root) {
        pre(root);
        return ans;
    }
};
```

这种情况把子节点是否存在作为递归的终止条件来判断了。

问题：什么情况下必须用第一种方式？什么情况下可以用第二种方式？

## 离散与连续

很多问题如果不考虑边界情况，把所有量作为连续量来处理，那么问题会简单很多。如果考虑边界值，情况则会变得复杂。如何处理离散值的边界问题？如何才能保证使用连续量进行思考，使用离散值来完善算法细节？

### 索引与长度

### 整数的一分为二

### 二分搜索

### 前缀和

### 滑动窗口

### 链表的中间节点

## 循环与控制

我们在编写一个循环时，是直接使用`for`，`while`，`if...else...`，`break/continue`编写，还是先写一些特例，找到共同规律后再简写成一个循环？

### insertion sort 中的循环

插入排序虽然说起来原理很简单，但是有一些细节没想清楚：

我们首先要把大于 key 值的牌往后挪，空出来一个位置，等位置空出来后，再把 key 插进去。

假设第 i 张牌是 key 牌，我们选择 key 牌前面的一张牌开始比较，即`j = i - 1`，最后一张要比较的牌为`j = 0`：

```cpp
for (int j = i - 1; j > -1; --j)
```

对于每一张牌，它如果比 key 牌大，那么需要后移：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j]  // 这句执行结束后，nums[j] 这个位置就空出来了
}
```

如果发现某张牌比 key 小，或等于 key，那么就可以结束循环了：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j]
    else
        break;
}
```

此时我们需要把空出来的位置插入 key 值，可以选择在循环内插入，也可以选择在循环外。假如在循环内：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j]
    else
    {
        nums[j+1] = num;  // 如果发现某张牌比 key 小，那么下一张牌一定是属于 key 的位置
        break;
    }
}
```

但是结果并不是正确的，因为当`j = 0`进入循环后，假如发现`j = 1`处的值比`num`大，会执行`nums[1] = nums[0]`。接着`j--`，变成`-1`，不再进入循环。可是这时`nums[0]`处的值已经空出来了，`elxe`语句也并未得到执行。由于我们选择在循环内部处理 key 值，所以需要再一次进入循环。那么就必须把`j > -1`改成`j > -2`。但是这样的话，`nums[j]`就会越界，所以我们需要对这种情况进行单独处理：

```cpp
for (int j = i - 1; j > -2; --j)
{
    if (j == -1)
    {
        nums[0] = num;
        break;
    }

    if (nums[j] > num)
        nums[j+1] = nums[j];
    else
    {
        nums[j+1] = num;
        break;
    }
}
```

这样就没问题了。

如果我们把处理写到外面：

```cpp
for (int j = i - 1; j > -1; --j)
{
    if (nums[j] > num)
        nums[j+1] = nums[j];
    else
        break;
}

nums[j+1] = num;
```

当`j = 0`时，发现`nums[0] > num`，也会移动，并使`j`递减得到`-1`。这样我们在外面使用`nums[j+1]`仍是有效的。可是问题在于，`j`是在`for`里面定义的，不是在`for`外面定义的。我们把`j`的定义也移到外面：

```cpp
for (int i = 1; i < n; ++i)
{
    int num = nums[i];
    int j = i - 1;
    for (; j > -1; --j)
    {
        if (nums[j] > num)
            nums[j+1] = nums[j];
        else
            break;
    }
    nums[j+1] = num;
}
```

这样是没问题的。可是这样的话，`for`就失去了它的功能，不如把改成`while`：

```cpp
void insertion_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 1; i < n; ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1)
        {
            if (nums[j] > num)
            {
                nums[j+1] = nums[j];
                --j;
            }
            else
                break;
        }
        nums[j+1] = num;
    }
}
```

`else`单独一行，有点丑，可以和上面的`if`换下位置：

```cpp
void insertion_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 1; i < n; ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1)
        {
            if (nums[j] <= num)
                break;
            nums[j+1] = nums[j];
            --j;

        }
        nums[j+1] = num;
    }
}
```

`if`单独一行，也有点繁琐，可以和`while`合一下：

```cpp
void insertion_sort(vector<int> &nums)
{
    int n = nums.size();
    for (int i = 1; i < n; ++i)
    {
        int num = nums[i];
        int j = i - 1;
        while (j > -1 && nums[j] > num)
        {
            nums[j+1] = nums[j];
            --j;
        }
        nums[j+1] = num;
    }
}
```

现在我们的问题来了：为什么在循环内部进行插入的代码需要单独处理边界情况，而在循环外部进行插入的代码不需要处理边界？为什么第二种情况的循环变量一定要写在循环外面？对于`for`和`while`，我们该如何才能快速选择正确的那个？有没有可能先`--j`，再执行其他语句？

### 一道滑动窗口题的循环

题目：字符串中的变位词

给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的某个变位词。

换句话说，第一个字符串的排列之一是第二个字符串的 子串 。

分析：

看到这道题，我们很快就能想到滑动窗口+数组计数。首先把滑动窗口扩大到`s1`的长度，然后不断把整个窗口在`s2`中向右移动，每移动一格，都将左侧指针指向的字符的计数减一，让右指针指向的新加进来的字符的计数加一。如果我们可以判断滑动窗口中的字符的计数与`s1`正好相等，那么就可以返回 true 了。

由此思路我们可以写出如下的代码：

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if (s1.size() > s2.size()) return false;
        int left = 0, right = 0;
        vector<int> count(26, 0);
        int n = s1.size();
        for (int i = 0; i < n; ++i)
            --count[s1[i] - 'a'];
        while (right < n)  // 扩充滑动窗口的长度，直到元素够 n 个为止
            ++count[s2[right++] - 'a'];
        bool contains;
        while (right < s2.size())
        {
            contains = true;
            for (int i = 0; i < 26; ++i)
            {
                if (count[i] != 0)
                {
                    contains = false;
                    break;
                }
            }
            if (contains)
                return true;
            ++count[s2[right++] - 'a'];  // 向右移动滑动窗口，
            --count[s2[left++] - 'a'];
        }
        return false;
    }
};
```

但是这段代码是有问题的。首先，假设我们的`s2`是`[a, ..., b, c, d, ..., e, f]`，其中`[a, ..., b]`的长度恰好是`s1`的长度。在执行完

```cpp
while (right < n)
    ++count[s2[right++] - 'a'];
```

这两行代码后，`right`会指向数组中的`c`，而我们统计的滑动窗口，实际上只有`[a, ..., b]`而已。现在我们考虑最后的边界条件，当`right`指向`f`时，滑动窗口为`[..., e]`，进入`while (right < s2.size())`循环，在循环的最后，滑动窗口变成`[..., f]`，`right++`指向数组之外，此时已经不满足循环的条件，因此退出循环。可是这样一来，最后一种情况`[..., f]`我们就没有判断。

我们可以额外对最后一种情况单独处理：

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if (s1.size() > s2.size()) return false;
        int left = 0, right = 0;
        vector<int> count(26, 0);
        int n = s1.size();
        for (int i = 0; i < n; ++i)
            --count[s1[i] - 'a'];
        while (right < n)
            ++count[s2[right++] - 'a'];
        bool contains;
        while (right < s2.size())
        {
            contains = true;
            for (int i = 0; i < 26; ++i)
            {
                if (count[i] != 0)
                {
                    contains = false;
                    break;
                }
            }
            if (contains)
                return true;
            ++count[s2[right++] - 'a'];
            --count[s2[left++] - 'a'];
        }
        contains = true;
        for (int i = 0; i < 26; ++i)
        {
            if (count[i] != 0)
            {
                contains = false;
                break;
            }
        }
        if (contains)
            return true;
        return false;
    }
};
```

这样就没问题了。可是这样略显繁琐，为什么我们没有提前判断出来最后的边界会有问题？有没有什么提前判断的方法？对于现在这种额外处理的办法，有没有什么更简洁的办法呢？

我们首先考虑把`while`的条件改成`while (right <= s2.size())`。这样确实可以进入循环，没问题，但是在循环时，`right`就已经指向数组外面了，因此循环最后的`s2[right++]`一定会数组越界。

为了不让`right`在循环内越界，我们考虑两种方案，一种是把`right`的递增操作放到`while`循环的前一部分，另一种是将`right`定义为滑动窗口的最后一个元素的索引，而不是最后一个元素的下一个位置。

对于方案一，我们可以先写出

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if (s1.size() > s2.size()) return false;
        int left = 0, right = 0;
        vector<int> count(26, 0);
        int n = s1.size();
        for (int i = 0; i < n; ++i)
            --count[s1[i] - 'a'];
        while (right < n)
            ++count[s2[right++] - 'a'];
        bool contains;
        while (right < s2.size())
        {
            ++count[s2[right++] - 'a'];
            --count[s2[left++] - 'a'];
            contains = true;
            for (int i = 0; i < 26; ++i)
            {
                if (count[i] != 0)
                {
                    contains = false;
                    break;
                }
            }
            if (contains)
                return true;
        }
        return false;
    }
};
```

这样也是有问题的。假设我们的`s2`是`[a, ..., b, c, d, ..., e, f]`，其中`[a, ..., b]`的长度恰好是`s1`的长度。在进入循环时，`right`实际指向`c`，所以

```cpp
++count[s2[right++] - 'a'];
```

这行代码实际修改的是`[a, ..., c]`的内容。然而我们对`[a, ..., b]`是否符合情况的判断，在后面才进行。这样就把`[a, ..., b]`这种情况跳过了。

如果我们尝试把`right`定义为最后一个元素的索引，则有

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if (s1.size() > s2.size()) return false;
        int left = 0, right = 0;
        vector<int> count(26, 0);
        int n = s1.size();
        for (int i = 0; i < n; ++i)
            --count[s1[i] - 'a'];
        while (right < n)
            ++count[s2[right++] - 'a'];
        bool contains;
        --right;
        while (right < s2.size())
        {
            contains = true;
            for (int i = 0; i < 26; ++i)
            {
                if (count[i] != 0)
                {
                    contains = false;
                    break;
                }
            }
            if (contains)
                return true;
            ++count[s2[++right] - 'a'];
            --count[s2[left++] - 'a'];
        }
        return false;
    }
};
```

因为在前面使用`--right`变成了索引，所以后面使用

```cpp
++count[s2[++right] - 'a'];
```

先递增，再使用`right`。我们考虑最后一种情况，当循环结束时，`right`指向`f`。由于满足循环条件，再次进入循环，前面的判断没有问题，可是当再次执行到`s2[++right]`时，先递增，再取数，会导致数组越界。

由此可见，无论如何，都无法简单地使用一个 while 处理完所有情况。我们需要重新思考 while 的几大模块：

1. 初始状态。指在进入循环前各个变量，指针，数组的状态。
1. 进入循环的条件。指`while()`中的条件语句。当然这个语句也可以作为退出循环的条件。
1. 状态判定。即`while`中的`if`语句，对当前的数组，变量，指针之类的只做检查，不做改变。
1. 状态改变。改变循环中的数据状态。
1. （可选的）预处理与后处理。

对于这道题，我们的问题是：有没有可能不做预处理和后处理，只使用一个`while`解决所有的情况呢？

假如我们遵循先判定状态，再改变状态的模式，即：

```cpp
// 初始状态
while ()
{
    // 判定状态
    // 改变状态
}
```

那么我们需要保证进入循环时，初始状态即我们要判断的第一组状态，这一点是没问题的。接下来我们需要保证改变状态后，能再次进入循环，并判断最后一组状态。问题就在于，判断完最后一组状态后，我们不再需要改变状态了。

```cpp
// 把 while 展开：

// 初始状态
// 判断状态
// 改变状态
// 判断状态
// 改变状态
// 判断状态
// ----------------
// 改变状态  (最后一次循环中，这个模块已经不再被需要了)
```

在这里，我们可以看到引起冲突的根本原因是，`while`中有两个模块，而我们并不是每次都完整地用到两个模块。由此我们即可得到所有的解决方案：

1. 在改变状态前加上`if`，每次都判断是否需要改变状态

    ```cpp
    class Solution {
    public:
        bool checkInclusion(string s1, string s2) {
            if (s1.size() > s2.size()) return false;
            int left = 0, right = 0;
            vector<int> count(26, 0);
            int n = s1.size();
            for (int i = 0; i < n; ++i)
                --count[s1[i] - 'a'];
            while (right < n)
                ++count[s2[right++] - 'a'];
            bool contains;
            while (right <= s2.size())
            {
                contains = true;
                for (int i = 0; i < 26; ++i)
                {
                    if (count[i] != 0)
                    {
                        contains = false;
                        break;
                    }
                }
                if (contains)
                    return true;
                if (right == s2.size())
                {
                    ++right;
                    continue;
                }
                ++count[s2[right++] - 'a'];
                --count[s2[left++] - 'a'];
            }
            return false;
        }
    };
    ```

1. 不执行最后一个循环，加入后处理进行状态判断

1. 交换“判断状态”和“改变状态”的位置，在设置初始状态的时候，少写一个循环，在`while`中由改变状态来实现初始状态。

满足我们要求的只有方案 3。但是我们可以在循环中实现状态初始化吗？要想实现初始化，我们必须把`left`放到`-1`的位置，这样又会造成数组越界。

由此我们得出，如果`while`中的状态改变不会造成数组下标越界等问题，那么就无所谓。如果会造成下标越界的问题，那么可以用`if`判断是否改变状态，或额外使用后处理进行状态判断。

### 判断状态，更新数据，改变状态

循环的三要素：判断状态，更新数据，改变状态。

我们要保证在更新数据时，状态是有效的。什么时候状态有效？可能在循环外，也可能在循环内。

### while 与 do while

如果一段代码至少需要执行一遍，那么我们就用 do while。

```cpp
wchar_t* not_reach_EOF = (wchar_t*) true;
while (not_reach_EOF)
{
    not_reach_EOF = fgetws(buf_read, BUFFSIZE, file);
    wstr.append(buf_read);
    wcout << wstr << endl;
    memset(buf_read, 0, BUFFSIZE);
}
```

上面这段代码是读文件时的一段代码，由于`not_reach_EOF`必须设置为`true`，所以这段代码可以直接写成 do while 型。

建议：

1. 优先考虑 while，如果一个 while 可能不被执行，那么就继续用 while。如果一个 while 的条件在进入 while 之前必须设置为 true，那么可以考虑把 while 替换成 do while。

### 多重退出条件

如果一个 while 有多重退出条件，那么必须对它们进行排列组合，分析每一种情况。

## 尝试与状态改变

有时候我们会遇到这样的情况：先尝试一些运算看看行不行，如果不行的话就不改变状态。如果没问题，那么实际执行运算。

这种问题有两种处理思路，一种是对逻辑进行模拟，然后计算，如果不满足条件就不进入下一个循环或下一层递归；一种是使用递归，直接实际改变状态，然后在下一个函数的入口处检测是否满足递归的退出条件。

这两种方法哪种更好一点呢？

## if 的连锁效应

考虑一个链表找交点问题：

输入两个链表，找出它们的第一个公共结点。

当不存在公共节点时，返回空节点。

分析：

我们常用的算法是双指针：

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (!headA || !headB) return nullptr;
        ListNode *p1 = headA, *p2 = headB;
        while (p1 && p2)
        {
            p1 = p1->next;
            p2 = p2->next;
        }
        if (!p1) p1 = headB;
        if (!p2) p2 = headA;
        while (p1 && p2)
        {
            p1 = p1->next;
            p2 = p2->next;
        }
        if (!p1) p1 = headB;
        if (!p2) p2 = headA;
        while (p1 && p2 && p1 != p2)
        {
            p1 = p1->next;
            p2 = p2->next;
        }
        return p1;
    }
};
```

其中这两行

```cpp
if (!p1) p1 = headB;
if (!p2) p2 = headA;
```

我们有一个模糊的想法：如果指针`p1`已经走到结尾的`NULL`，那么`p2`一定还在半途中，可不可以用`p2`来判断`p1`的状态呢？比如写成这样：

```cpp
if (p2) p1 = headB;
if (p1) p2 = headA;
```

但是实际上，这样写是有问题的：当`p1`被置为`headB`后，下面一行的`if (p1)`一定是 true，因此`p2 = headA`一定会被执行。

那么稍微改一下是否可以呢：

```cpp
if (p2) p1 = headB;
else if (p1) p2 = headA;
```

这样也不行。首先前面的`while()`保证了`p1`，`p2`至少有一个是`NULL`。如果`p1`是`NULL`，`p2`不是`NULL`，那么`if (p2)`为 true；如果`p1`不是`NULL`，`p2`是`NULL`，那么`else if (p1) `为 true。如果`p1`，`p2`都是`NULL`，那么说明两个指针同时走到结尾，这两个条件都为 false。而我们下一轮循环的理想条件是两个指针至少有一个在链表的开头处，因此如果执意要使用`if (p1), if (p2)`，需要再加一个判断：

```cpp
if (p2) p1 = headB;
else if (p1) p2 = headA;
else  // p1, p2 都为空
{
    p1 = headB;
    p2 = headA;
}
```

但是这样看来，代码麻烦了许多，还不如刚开始的方案。

由此我们可以得出结论：如果在一个`if`语句中，`if ()`中的条件只和单一变量`x`有关，`if ()`中执行的代码也只和单一变量`x`相关，那么就没有问题；如果`if ()`的条件和其它变量相关，那么就说明`if`是“耦合”的，我们需要慎重处理。

[这节的标题改成“if 的耦合与解耦”是不是比较好]

### if 中的多重判断

比较含退格的字符串

给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

 

示例 1：

输入：s = "ab#c", t = "ad#c"
输出：true
解释：s 和 t 都会变成 "ac"。
示例 2：

输入：s = "ab##", t = "c#d#"
输出：true
解释：s 和 t 都会变成 ""。
示例 3：

输入：s = "a#c", t = "b"
输出：false
解释：s 会变成 "c"，但 t 仍然是 "b"。
 

提示：

1 <= s.length, t.length <= 200
s 和 t 只含有小写字母以及字符 '#'
 

进阶：

你可以用 O(n) 的时间复杂度和 O(1) 的空间复杂度解决该问题吗？


错误的代码：

```cpp
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        stack<char> stk1, stk2;
        for (int i = 0; i < s.size(); ++i)
        {
            if (s[i] == '#' && !stk1.empty())
                stk1.pop();
            else
                stk1.push(s[i]);
        }
        for (int i = 0; i < t.size(); ++i)
        {
            if (t[i] == '#' && !stk2.empty())
                stk2.pop();
            else
                stk2.push(t[i]);
        }
        while (!stk1.empty())
        {
            cout << stk1.top() << ", ";
            stk1.pop();
        }
        cout << endl;
        while (!stk2.empty())
        {
            cout << stk2.top() << ", ";
            stk2.pop();
        }
        cout << endl;
        return stk1 == stk2;
    }
};
```

`if`中的两个条件，实际上是四个分支。我们需要对这四个分支都考虑清楚。

正确的版本：

```cpp
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        stack<char> stk1, stk2;
        for (int i = 0; i < s.size(); ++i)
        {
            if (s[i] == '#')
            {
                if (!stk1.empty())
                    stk1.pop();
            }
            else
                stk1.push(s[i]);
        }
        for (int i = 0; i < t.size(); ++i)
        {
            if (t[i] == '#')
            {
                if (!stk2.empty())
                    stk2.pop();
            }
                
            else
                stk2.push(t[i]);
        }
        return stk1 == stk2;
    }
};
```

## 静态枚举的优化

大部分程序可以分成两种，一种是使用有限的几层`for`或`while`循环可以完成任务，另一种无法写成有限的几层`for`循环，或者不知道要写几层。前一种我称其为静态枚举，后一种我称其为动态枚举。

如果一开始就知道可以最大迭代多少次结束程序，那么称这种类型的程序叫有限枚举。常见的`for`循环都是有限枚举。

### 滑动窗口

### 单调栈

## 动态枚举的优化

如果一开始不知道最大迭代多少次结束程序，那么称这种类型的程序为无限枚举。常见的递归，dfs，bfs，树的搜索等都是无限枚举。

或者不知道写多少层`for`才能完成程序。

