#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

using namespace cimg_library;

class Img
{
    public:
    Img(const char *img_path)
    {
        CImg<uint8_t> img(img_path);
        int width, height, spectrum, depth;
        width = img.width();
        height = img.height();
        spectrum = img.spectrum();
        depth = img.depth();

        this->width = width;
        this->height = height;
        this->num_channels = spectrum;
        this->depth = depth;  // depth == 1 (byte)

        img_data.resize(width * height * spectrum * 1);
        for (int x = 0; x < width; ++x)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int c = 0; c < spectrum; ++c)
                {
                    img_data[x * height * spectrum + y * spectrum + c] = img(x, y, 0, c);
                }
            }
        }
    }

    uint8_t xyc(int x, int y, int channel)
    {
        return img_data[x * height + y * num_channels + channel];
    }

    uint8_t row_col_channel(int row, int col, int channel)
    {
        return img_data[col * height * num_channels + row * num_channels + channel];
    }

    public:
    vector<uint8_t> img_data;
    int width, height, num_channels, depth;
};

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