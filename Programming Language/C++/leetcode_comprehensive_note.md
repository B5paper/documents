# LeetCode Comprehensive Note

这里主要记录力扣的综合题和应用题。

## cache

* 2209. 用地毯覆盖后的最少白色砖块

    tag: 回溯，动态规划

    给你一个下标从 0 开始的 二进制 字符串 floor ，它表示地板上砖块的颜色。

    floor[i] = '0' 表示地板上第 i 块砖块的颜色是 黑色 。
    floor[i] = '1' 表示地板上第 i 块砖块的颜色是 白色 。
    同时给你 numCarpets 和 carpetLen 。你有 numCarpets 条 黑色 的地毯，每一条 黑色 的地毯长度都为 carpetLen 块砖块。请你使用这些地毯去覆盖砖块，使得未被覆盖的剩余 白色 砖块的数目 最小 。地毯相互之间可以覆盖。

    请你返回没被覆盖的白色砖块的 最少 数目。

    

    示例 1：



    输入：floor = "10110101", numCarpets = 2, carpetLen = 2
    输出：2
    解释：
    上图展示了剩余 2 块白色砖块的方案。
    没有其他方案可以使未被覆盖的白色砖块少于 2 块。
    示例 2：



    输入：floor = "11111", numCarpets = 2, carpetLen = 3
    输出：0
    解释：
    上图展示了所有白色砖块都被覆盖的一种方案。
    注意，地毯相互之间可以覆盖。
    

    提示：

    1 <= carpetLen <= floor.length <= 1000
    floor[i] 要么是 '0' ，要么是 '1' 。
    1 <= numCarpets <= 1000


    解法：

    1. 自己想的，超出时间限制

        看起来是用的圆溯法。遍历每一条地毯可以覆盖的所有位置，直到把所有的地毯都放完。

        ```cpp
        class Solution {
        public:
            vector<int> path;
            int min_white_cnt = INT32_MAX;

            void backtrack(string &floor, int numCarpets,int carpetLen)
            {
                if (numCarpets == 0)
                {
                    string floor_copy = floor;
                    for (int i = 0; i < path.size(); ++i)
                    {
                        for (int j = 0; j < carpetLen; ++j)
                        {
                            floor_copy[path[i] + j] = '0';
                        }
                    }
                    int white_cnt = 0;
                    for (int i = 0; i < floor_copy.size(); ++i)
                    {
                        if (floor_copy[i] == '1')
                            white_cnt++;
                    }
                    min_white_cnt = min(min_white_cnt, white_cnt);
                    return;
                }

                for (int i = 0; i < floor.size() - carpetLen + 1; ++i)
                {
                    path.push_back(i);
                    backtrack(floor, numCarpets - 1, carpetLen);
                    path.pop_back();
                }
            }

            int minimumWhiteTiles(string floor, int numCarpets, int carpetLen) {
                backtrack(floor, numCarpets, carpetLen);
                return min_white_cnt;
            }
        };
        ```

    2. 对解法 1 做了些改进，仍然超时

        当发现最后的几个格子放不下一条地毯时，就可以不用搜索了。

        ```cpp
        class Solution {
        public:
            vector<int> path;
            int min_white_cnt = INT32_MAX;

            void backtrack(string &floor, int numCarpets,int carpetLen, int start_pos)
            {
                if (numCarpets == 0)
                {
                    string floor_copy = floor;
                    for (int i = 0; i < path.size(); ++i)
                    {
                        for (int j = 0; j < carpetLen; ++j)
                        {
                            floor_copy[path[i] + j] = '0';
                        }
                    }
                    int white_cnt = 0;
                    for (int i = 0; i < floor_copy.size(); ++i)
                    {
                        if (floor_copy[i] == '1')
                            white_cnt++;
                    }
                    min_white_cnt = min(min_white_cnt, white_cnt);
                    return;
                }

                for (int i = start_pos; i < floor.size() - carpetLen + 1; ++i)
                {
                    path.push_back(i);
                    backtrack(floor, numCarpets - 1, carpetLen, min(i + carpetLen, (int) floor.size() - carpetLen));
                    path.pop_back();
                }
            }

            int minimumWhiteTiles(string floor, int numCarpets, int carpetLen) {
                backtrack(floor, numCarpets, carpetLen, 0);
                return min_white_cnt;
            }
        };
        ```

    3. 官方解答，动态规划

        没来得及看。

        ```cpp
        class Solution {
        public:
            static constexpr int INF = 0x3f3f3f3f;
            int minimumWhiteTiles(string floor, int numCarpets, int carpetLen) {
                int n = floor.size();
                vector<vector<int>> d(n + 1, vector<int>(numCarpets + 1, INF));
                for (int j = 0; j <= numCarpets; j++) {
                    d[0][j] = 0;
                }
                for (int i = 1; i <= n; i++) {
                    d[i][0] = d[i - 1][0] + (floor[i - 1] == '1');
                }
                
                for (int i = 1; i <= n; i++) {
                    for (int j = 1; j <= numCarpets; j++) {
                        d[i][j] = d[i - 1][j] + (floor[i - 1] == '1');
                        d[i][j] = min(d[i][j], d[max(0, i - carpetLen)][j - 1]);
                    }
                }

                return d[n][numCarpets];
            }
        };
        ```

### 设计一个文本编辑器

tag: 模拟，链表，栈

请你设计一个带光标的文本编辑器，它可以实现以下功能：

添加：在光标所在处添加文本。
删除：在光标所在处删除文本（模拟键盘的删除键）。
移动：将光标往左或者往右移动。
当删除文本时，只有光标左边的字符会被删除。光标会留在文本内，也就是说任意时候 0 <= cursor.position <= currentText.length 都成立。

请你实现 TextEditor 类：

TextEditor() 用空文本初始化对象。
void addText(string text) 将 text 添加到光标所在位置。添加完后光标在 text 的右边。
int deleteText(int k) 删除光标左边 k 个字符。返回实际删除的字符数目。
string cursorLeft(int k) 将光标向左移动 k 次。返回移动后光标左边 min(10, len) 个字符，其中 len 是光标左边的字符数目。
string cursorRight(int k) 将光标向右移动 k 次。返回移动后光标左边 min(10, len) 个字符，其中 len 是光标左边的字符数目。


示例 1：

输入：
["TextEditor", "addText", "deleteText", "addText", "cursorRight", "cursorLeft", "deleteText", "cursorLeft", "cursorRight"]
[[], ["leetcode"], [4], ["practice"], [3], [8], [10], [2], [6]]
输出：
[null, null, 4, null, "etpractice", "leet", 4, "", "practi"]

解释：
TextEditor textEditor = new TextEditor(); // 当前 text 为 "|" 。（'|' 字符表示光标）
textEditor.addText("leetcode"); // 当前文本为 "leetcode|" 。
textEditor.deleteText(4); // 返回 4
                        // 当前文本为 "leet|" 。
                        // 删除了 4 个字符。
textEditor.addText("practice"); // 当前文本为 "leetpractice|" 。
textEditor.cursorRight(3); // 返回 "etpractice"
                        // 当前文本为 "leetpractice|". 
                        // 光标无法移动到文本以外，所以无法移动。
                        // "etpractice" 是光标左边的 10 个字符。
textEditor.cursorLeft(8); // 返回 "leet"
                        // 当前文本为 "leet|practice" 。
                        // "leet" 是光标左边的 min(10, 4) = 4 个字符。
textEditor.deleteText(10); // 返回 4
                        // 当前文本为 "|practice" 。
                        // 只有 4 个字符被删除了。
textEditor.cursorLeft(2); // 返回 ""
                        // 当前文本为 "|practice" 。
                        // 光标无法移动到文本以外，所以无法移动。
                        // "" 是光标左边的 min(10, 0) = 0 个字符。
textEditor.cursorRight(6); // 返回 "practi"
                        // 当前文本为 "practi|ce" 。
                        // "practi" 是光标左边的 min(10, 6) = 6 个字符。


提示：

1 <= text.length, k <= 40
text 只含有小写英文字母。
调用 addText ，deleteText ，cursorLeft 和 cursorRight 的 总 次数不超过 2 * 104 次。


进阶：你能设计并实现一个每次调用时间复杂度为 O(k) 的解决方案吗？

代码：

1. 自己想的

    纯模拟，没有什么算法。

    ```cpp
    class TextEditor {
    public:
        vector<int> txt;
        int cur;

        TextEditor() {
            cur = 0;
        }
        
        void addText(string text) {
            txt.insert(txt.begin() + cur, text.begin(), text.end());
            cur += text.size();
        }
        
        int deleteText(int k) {
            if (cur <= k)
            {
                txt.erase(txt.begin(), txt.begin() + cur);
                int del_len = cur;
                cur = 0;
                return del_len;
            }
            txt.erase(txt.begin() + cur - k, txt.begin() + cur);
            cur -= k;
            return k;
        }
        
        string cursorLeft(int k) {
            if (cur <= k)
            {
                cur = 0;
                return "";
            }
            cur -= k;
            return string(txt.begin() + cur - min(10, cur), txt.begin() + cur);
        }
        
        string cursorRight(int k) {
            if (txt.size() - cur <= k)
                cur = txt.size();
            else
                cur += k;
            return string(txt.begin() + cur - min(10, cur),  txt.begin() + cur);
        }
    };

    /**
    * Your TextEditor object will be instantiated and called as such:
    * TextEditor* obj = new TextEditor();
    * obj->addText(text);
    * int param_2 = obj->deleteText(k);
    * string param_3 = obj->cursorLeft(k);
    * string param_4 = obj->cursorRight(k);
    */
    ```

1. 官方答案 1：双向链表

    没看。

    ```cpp
    class TextEditor {
    private:
        list<char> editor;
        list<char>::iterator cursor;

    public:
        TextEditor() {
            cursor = editor.end();
        }

        void addText(string text) {
            for (char c : text) {
                editor.insert(cursor, c);
            }
        }

        int deleteText(int k) {
            int count = 0;
            for (; k > 0 && cursor != editor.begin(); k--) {
                editor.erase(prev(cursor));
                count++;
            }
            return count;
        }

        string cursorLeft(int k) {
            while (k > 0 && cursor != editor.begin()) {
                k--;
                cursor = prev(cursor);
            }
            auto head = cursor;
            for (int i = 0; i < 10 && head != editor.begin(); i++) {
                head = prev(head);
            }
            return string(head, cursor);
        }

        string cursorRight(int k) {
            while (k > 0 && cursor != editor.end()) {
                k--;
                cursor = next(cursor);
            }
            auto head = cursor;
            for (int i = 0; i < 10 && head != editor.begin(); i++) {
                head = prev(head);
            }
            return string(head, cursor);
        }
    };
    ```

1. 官方答案 2：对顶栈

    没看。

    ```cpp
    class TextEditor {
    private:
        vector<char> left;
        vector<char> right;
    public:
        TextEditor() {
            
        }

        void addText(string text) {
            for (char c : text) {
                left.push_back(c);
            }
        }
        
        int deleteText(int k) {
            int n = 0;
            for (; !left.empty() && k > 0; k--) {
                left.pop_back();
                n++;
            }
            return n;
        }
        
        string cursorLeft(int k) {
            while (!left.empty() && k > 0) {
                right.push_back(left.back());
                left.pop_back();
                k--;
            }
            int n = left.size();
            return string(left.begin() + max(0, n - 10), left.end());
        }
        
        string cursorRight(int k) {
            while (!right.empty() && k > 0) {
                left.push_back(right.back());
                right.pop_back();
                k--;
            }
            int n = left.size();
            return string(left.begin() + max(0, n - 10), left.end());
        }
    };
    ```

## note