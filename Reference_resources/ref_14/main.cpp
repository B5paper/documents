#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int get_ans_1(vector<int> &&tasks)
{
    int ans = 0;
    sort(tasks.begin(), tasks.end());
    int task_len = tasks.size();
    vector<int> min_round_along_pos(task_len, INT32_MAX);
    vector<int> appear_cnt(task_len, -1);
    min_round_along_pos[0] = -1;
    appear_cnt[0] = 1;
    for (int i = 0; i < task_len; ++i)
    {
        if (appear_cnt[i] == 1)
        {
            if (i + 1 < task_len && tasks[i+1] == tasks[i] && i - 2 >= 0)
            {
                appear_cnt[i+1] = 2;
                min_round_along_pos[i+1] = min_round_along_pos[i-2] + 1;
            }
            else if (i + 1 < task_len && tasks[i+1] != tasks[i])
            {
                appear_cnt[i+1] = 1;
                min_round_along_pos[i+1] = -1;
            }
        }

        if (appear_cnt[i] == 2)
        {
            if (i + 1 < task_len && tasks[i+1] == tasks[i])
            {
                appear_cnt[i+1] = 3;
                min_round_along_pos[i+1] = min_round_along_pos[i];
            }
            else if (i + 1 < task_len && tasks[i+1] != tasks[i])
            {
                appear_cnt[i+1] = 1;
                min_round_along_pos[i+1] = -1;
            }
        }

        if (appear_cnt[i] == 3)  // 4，只能拆分成 2 + 2
        {
            if (i + 1 < task_len && tasks[i+1] == tasks[i])
            {
                appear_cnt[i+1] = 2;  // 拆分成 2 + 2
                min_round_along_pos[i+1] = min_round_along_pos[i-1] + 1;
            }
        }

        if (appear_cnt[i] == 4)  // 5, 5 = 2 + 3, 5 = 3 + 2
        {
            if (i + 1 < task_len && tasks[i+1] == tasks[i])
            {
                appear_cnt[i+1] = 2;  // 3 + 2
                min_round_along_pos[i+1] = min_round_along_pos[i-1] + 1;
            }
        }
    }
    return ans;
}

int main()
{
    int (*get_ans)(vector<int> &&) = get_ans_1;
    int ans;

    // test 1
    ans = get_ans({2,2,3,3,2,4,4,4,4,4});
    if (ans != 4) {
        printf("fail to pass test 1\n");
        return -1;
    }
    printf("successfully pass test 1\n");
    
    // test 2
    ans = get_ans({2,3,3});
    if (ans != -1) {
        printf("fail to pass test 2\n");
        return -1;
    }
    printf("successfully pass test 2\n");
    return 0;
}