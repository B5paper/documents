# Graph Note

## cache

* 如果使用 index 来唯一地标记 vertex 和 edge，那么意味着它们不能被排序。如果只使用 idx 来标记 vertex，那么 edge 可以被排序，但 vertex 不能。

    opengl 中完全使用 idx 来标记 vertex, edge 以及 triange，所以输入数据、处理数据时，都不能对 entity 排序。好在图形处理也不需要排序。

    另外一种方式是使用指针来定位 entity，这种效率最高。

* 使用指针存储大部分数据，使用链表存储边，并且将边存储在顶点数据结构里的一份代码与讨论

    ```cpp
    #include <stdio.h>
    #include <time.h>
    #include <stdlib.h>
    #include <string>
    #include <iostream>
    #include <vector>
    #include <algorithm>
    #include <utility>
    #include <initializer_list>
    using namespace std;

    struct Vertex;

    struct Edge {
        Vertex *nex_vert;
        int dist;
    };

    struct Vertex {
        vector<Edge*> edges;

        ~Vertex() {
            for (Edge *edge : edges) {
                delete edge;
            }
        }
    };

    struct Graph {
        vector<Vertex*> verts;

        ~Graph() {
            for (Vertex *vert : verts) {
                delete vert;
            }
        }
    };

    struct PathSearcher {
        static vector<Vertex*> tmp_path;
        static vector<Vertex*> min_dist_path;
        static int min_dist;

        static Vertex *vert_dst;
        static Graph *graph;

        static int recur_search_path_first_match(Vertex *vert) {
            if (vert == vert_dst) {
                tmp_path.push_back(vert);
                return 0;
            }

            for (int i = 0; i < vert->edges.size(); ++i) {
                Edge *edge = vert->edges[i];
                int ret = recur_search_path_first_match(edge->nex_vert);
                if (ret == 0) {
                    tmp_path.push_back(vert);
                    return 0;
                }
            }

            return -1;
        }

        static vector<Vertex*> get_path_first_match() {
            std::reverse(tmp_path.begin(), tmp_path.end());
            return tmp_path;
        }

        static int recur_search_path_min_dist(Vertex *vert) {
            if (vert == vert_dst) {
                int dist = 0;
                Vertex *cur_vert = tmp_path[0];
                for (int i = 1; i < tmp_path.size(); ++i) {
                    Vertex *nex_vert = tmp_path[i];
                    for (int j = 0; j < cur_vert->edges.size(); ++j) {
                        Edge *edge = cur_vert->edges[j];
                        if (edge->nex_vert == nex_vert) {
                            dist += edge->dist;
                            break;
                        }
                    }
                    cur_vert = nex_vert;
                }
                if (dist < min_dist) {
                    min_dist_path = tmp_path;
                    min_dist = dist;
                }
                return 0;
            }

            for (int i = 0; i < vert->edges.size(); ++i) {
                Edge *edge = vert->edges[i];
                Vertex *nex_vert = edge->nex_vert;
                tmp_path.push_back(nex_vert);
                int ret = recur_search_path_min_dist(nex_vert);
                tmp_path.pop_back();
            }
            return 0;
        }
    };

    vector<Vertex*> PathSearcher::tmp_path;
    vector<Vertex*> PathSearcher::min_dist_path;
    Vertex* PathSearcher::vert_dst;
    Graph* PathSearcher::graph;
    int PathSearcher::min_dist;

    int search_path_first_match(vector<Vertex*> &path, Graph &graph,
        int src_vert, int dst_vert) {
        Vertex *vert_src = graph.verts[src_vert];
        Vertex *vert_dst = graph.verts[dst_vert];

        PathSearcher::tmp_path.clear();
        PathSearcher::graph = &graph;
        PathSearcher::vert_dst = vert_dst;
        int ret = PathSearcher::recur_search_path_first_match(vert_src);
        if (ret != 0) {
            printf("fail to recur search path first match\n");
            return -1;
        }
        path = PathSearcher::get_path_first_match();
        return 0;
    }

    int search_path_min_dist(vector<Vertex*> &path, int &min_dist,
        Graph &graph, int src_vert, int dst_vert) {
        Vertex *vert_src = graph.verts[src_vert];
        Vertex *vert_dst = graph.verts[dst_vert];

        PathSearcher::min_dist = INT32_MAX;
        PathSearcher::tmp_path.clear();
        PathSearcher::tmp_path.push_back(vert_src);
        PathSearcher::graph = &graph;
        PathSearcher::vert_dst = vert_dst;

        int ret = PathSearcher::recur_search_path_min_dist(vert_src);
        if (ret != 0) {
            printf("fail to recur search path min dist\n");
            return -1;
        }
        path = PathSearcher::min_dist_path;
        min_dist = PathSearcher::min_dist;

        return 0;
    }

    Vertex* graph_add_vert(Graph &graph) {
        Vertex *new_vert = new Vertex;
        graph.verts.push_back(new_vert);
        return new_vert;
    }

    Edge* graph_add_edge(Graph &graph, Vertex &vert, int nex_vert) {
        Vertex *vert_nex = graph.verts[nex_vert];
        Edge *new_edge = new Edge;
        new_edge->nex_vert = vert_nex;
        vert.edges.push_back(new_edge);
        return new_edge;
    }

    void print_path(Graph &graph, vector<Vertex*> &path) {
        for (int i = 0; i < path.size(); ++i) {
            Vertex *vert = path[i];
            int vert_idx;
            for (int j = 0; j < graph.verts.size(); ++j) {
                if (graph.verts[j] == vert) {
                    vert_idx = j;
                    break;
                }
            }
            if (i == path.size() - 1)
                printf("%d", vert_idx);
            else
                printf("%d -> ", vert_idx);
        }
        putchar('\n');
    }

    int main() {
        Graph graph;
        int num_verts = 7;
        for (int i = 0; i < num_verts; ++i) {
            graph_add_vert(graph);
        }
        vector<pair<int, int>> input_edges {
            {0, 1}, {0, 2}, {0, 3},
            {1, 4}, {1, 2},
            {2, 5}, {2, 3},
            {3, 5},
            {4, 5},
            {5, 6}
        };

        srand(time(NULL));
        for (int i = 0; i < input_edges.size(); ++i) {
            auto [cur_vert, nex_vert] = input_edges[i];
            Vertex *vert_cur = graph.verts[cur_vert];
            Vertex *vert_nex = graph.verts[nex_vert];
            Edge *new_edge = new Edge;
            new_edge->nex_vert = vert_nex;
            new_edge->dist = rand() % 19 + 1;
            vert_cur->edges.push_back(new_edge);
        }

        vector<Vertex*> path;

        printf("-------- search_path_first_match --------\n");
        int ret = search_path_first_match(path, graph, 0, 6);
        if (ret != 0) {
            printf("fail to serach path first match\n");
            return -1;
        }

        for (int i = 0; i < path.size(); ++i) {
            printf("%p -> ", path[i]);
        }
        putchar('\n');

        print_path(graph, path);
        putchar('\n');

        printf("-------- search_path_min_dist --------\n");
        int min_dist;
        ret = search_path_min_dist(path, min_dist, graph, 0, 6);
        if (ret != 0) {
            printf("fail to search path min dist\n");
            return -1;
        }

        printf("min dist: %d\n", min_dist);
        for (int i = 0; i < path.size(); ++i) {
            printf("%p -> ", path[i]);
        }
        putchar('\n');

        print_path(graph, path);

        return 0;
    }
    ```

    output:

    ```
    -------- search_path_first_match --------
    0x55bcfa310eb0 -> 0x55bcfa310ef0 -> 0x55bcfa310f60 -> 0x55bcfa310fd0 -> 0x55bcfa310ff0 -> 
    0 -> 1 -> 4 -> 5 -> 6

    -------- search_path_min_dist --------
    min dist: 31
    0x55bcfa310eb0 -> 0x55bcfa310ed0 -> 0x55bcfa310fd0 -> 0x55bcfa310ff0 -> 
    0 -> 2 -> 5 -> 6
    ```

    讨论：

    * `struct Vertex;`: 因为`Edge`和`Vertex`都需要对方的指针，所以需要打破循环依赖，这里将`Vertex`的声明前置。

    * `struct Edge {`

        ```cpp
        struct Edge {
            Vertex *nex_vert;
            int dist;
        };
        ```

        `Edge`中保存的不是 index 而是指针，理论上存 index 也可以。指针的话，要求 vertex 必须先创建好，才能创建 edge。index 则没有这个约束。

        `Edge`没有保存 current vertex 的信息，只存了下一跳的顶点，这样导致`Edge`必须依赖一个已经存在的 vertex 而存在，不能独立存在。比如，我只能`vert.add_edge(Vertex *nex_vert);`，而不能`Edge edge{vert_cur, vert_nex};`。

    * `struct Vertex {`:

        ```cpp
        struct Vertex {
            vector<Edge*> edges;

            ~Vertex() {
                for (Edge *edge : edges) {
                    delete edge;
                }
            }
        };
        ```

        这里的`edges`存储的是指针，所以我们在析构时需要 delete。那么是否可以用`vector<Edge> edges;`来存储呢？看起来应该是没什么问题，vector 要求的是连续内存，因为`Edge`占用内存比较少，所以申请连续内存没有压力。

    * `struct Graph {`:

        ```cpp
        struct Graph {
            vector<Vertex*> verts;

            ~Graph() {
                for (Vertex *vert : verts) {
                    delete vert;
                }
            }
        };
        ```

        我们目前方案主打使用指针存储，所以`verts`就不考虑`vector<Vertex>`存储了。

        由于 edge 已经存在了 vert 中，所以 graph 就不再存储 edge 信息了。

    * `struct PathSearcher {`:

        ```cpp
        struct PathSearcher {
            static vector<Vertex*> tmp_path;
            static vector<Vertex*> min_dist_path;
            static int min_dist;

            static Vertex *vert_dst;
            static Graph *graph;
        ```

        一开始在实现`recur_search_path_first_match()`时，发现有些变量在函数参数中一直保持不变，所以只需要全局保存一份就可以了，直接做成全局变量又会和别的命令冲突，所以这里直接用 struct 把这些变量都变成静态类型变量。

        类的静态变量需要在类外定义，做初始化，所以后面

        ```cpp
        vector<Vertex*> PathSearcher::tmp_path;
        vector<Vertex*> PathSearcher::min_dist_path;
        Vertex* PathSearcher::vert_dst;
        Graph* PathSearcher::graph;
        int PathSearcher::min_dist;
        ```

        对类的静态成员做了初始化。

    * `static int recur_search_path_first_match(Vertex *vert) {`:

        这个函数只要找到一条路径就直接返回。由于函数的递归调用本身就相当于 stack，保存了一条 path，所以我们只需要在函数退出时把这条 path 保存下来，再后再做一遍反序就可以了。

        如果需要回溯，显然这个 trick 没法用，因为只有所有函数都返回时，才能拿到完整的 path，这样我们就来不及比较多条 path 了。

    * `static int recur_search_path_min_dist`:

        递归终止条件`if (vert == vert_dst)`会遍历一遍 path，计算 dist 的和，然后与存储的最小值比较。这里有 2 个问题：

        1. 每次都从 0 开始累加 dist，效率较低，其实可以每次往 path 里增减 edge 的时候，dist 也做对应的增减，这样 dist 时刻都是计算好的。

        2. `tmp_path`存储的是节点，不是边，而 dist 是边的属性，所以每次都要根据 nex_vert 找到对应的 edge，再得到 dist，效率低下。根据 edge 找 nex vert 比较方便，而根据 nex vert 反过来找 edge 比较困难。因此 path 还是保存成 edge 比较好。

## note