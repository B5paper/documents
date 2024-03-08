#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

float median_3x3(vector<vector<float>> &mat, int row, int col)
{
    float buf[9];
    int p = 0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            buf[p++] = mat[row+i][col+j];
        }
    }
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8 - i; ++j)
        {
            if (buf[j] > buf[j+1])
                swap(buf[j], buf[j+1]);
        }
    }
    cout << "buf: ";
    for (auto elm: buf)
        cout << elm << ", ";
    cout << endl;
    return buf[4];
}

int main()
{
    vector<vector<float>> mat(3, vector<float>(3));
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mat[i][j] = rand() % 30;
        }
    }
    float median = median_3x3(mat, 1, 1);
    for (auto row: mat)
    {
        for (auto elm: row)
            cout << elm << ", ";
        cout << endl;
    }
    printf("%.2f\n", median);
    return 0;
}