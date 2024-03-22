#include "../ocl_simple/simple_ocl.hpp"
#include "../CImg/CImg.h"
#include <time.h>
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

    uint8_t& xyc(int x, int y, int channel)
    {
        return img_data[x * height + y * num_channels + channel];
    }

    uint8_t row_col_channel(int row, int col, int channel)
    {
        return img_data[col * height * num_channels + row * num_channels + channel];
    }

    void save(const char *path)
    {
        CImg<uint8_t> img(width, height, 1, 1, 0);
        int row, col;  // ij coordinate
        for (size_t x = 0; x < width; ++x)  // xy coordinate
        {
            for (size_t y = 0; y < height; ++y)
            {
                row = y;
                col = x;
                for (int k = 0; k < num_channels; ++k)
                {
                    // img(x, y, 0, k) = img_data[(col + row * width) * num_channels + k];
                    img(x, y, 0, k) = xyc(x, y, k);
                }
            }
        }
        img.save_png(path);
    }

    public:
    vector<uint8_t> img_data;
    int width, height, num_channels, depth;
};


// use float type for storage
class FImg
{
    public:
    FImg(const char *img_path)
    {
        CImg<float> img(img_path);
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
                    img_data[x * height * spectrum + y * spectrum + c] = img(x, y, 0, c) / 255;
                }
            }
        }
    }

    float& xyc(int x, int y, int channel)
    {
        return img_data[x * height + y * num_channels + channel];
    }

    float& row_col_channel(int row, int col, int channel)
    {
        return img_data[col * height * num_channels + row * num_channels + channel];
    }

    void save(const char *path)
    {
        CImg<uint8_t> img(width, height, 1, 1, 0);
        int row, col;  // ij coordinate
        for (size_t x = 0; x < width; ++x)  // xy coordinate
        {
            for (size_t y = 0; y < height; ++y)
            {
                row = y;
                col = x;
                for (int k = 0; k < num_channels; ++k)
                {
                    // img(x, y, 0, k) = img_data[(col + row * width) * num_channels + k];
                    img(x, y, 0, k) = round(xyc(x, y, k) * 255);
                }
            }
        }
        img.save_png(path);
    }

    public:
    vector<float> img_data;
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


float median_3x3_cont(vector<uint8_t> &mat, int x, int y, int width, int height)
{
    float buf[9];
    int p = 0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            // buf[p++] = mat[row+i][col+j];
            buf[p++] = mat[(x+i) * width + y + j];
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
    // cout << "buf: ";
    // for (auto elm: buf)
    //     cout << elm << ", ";
    // cout << endl;
    return buf[4];
}

float median_3x3_cont(vector<float> &mat, int x, int y, int width, int height)
{
    float buf[9];
    int p = 0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            // buf[p++] = mat[row+i][col+j];
            buf[p++] = mat[(x+i) * width + y + j];
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
    return buf[4];
}

void median_filter_3x3_cont(FImg &img, FImg &out)
{
	
    for (int x = 1; x < img.width - 1; ++x)
    {
        for (int y = 1; y < img.height - 1; ++y)
        {
            out.xyc(x, y, 0) = median_3x3_cont(img.img_data, x, y, img.width, img.height);
        }
    }
}

void median_filter_3x3_cont(Img &img, Img &out)
{
	
    for (int x = 1; x < img.width - 1; ++x)
    {
        for (int y = 1; y < img.height - 1; ++y)
        {
            out.xyc(x, y, 0) = median_3x3_cont(img.img_data, x, y, img.width, img.height);
        }
    }
}

int main()
{
    FImg fimg("pic_test.png");
    FImg fout("pic_test.png");
    clock_t tic = clock();
    median_filter_3x3_cont(fimg, fout);
    clock_t toc = clock();
    clock_t dur = toc - tic;
    printf("cpu time consumption: %ld\n", dur);
    fout.save("median_filter.png");

    init_ocl_env("kernels.cl", {"median_filter_3x3"});
    tic = clock();
    add_buf("img", sizeof(float), fimg.img_data.size(), fimg.img_data.data());
    add_buf("out", sizeof(float), fimg.img_data.size(), fimg.img_data.data());
    add_buf("width", sizeof(int), 1, &fimg.width);
    add_buf("height", sizeof(int), 1, &fimg.height);
    run_kern("median_filter_3x3", {(unsigned long) fimg.width, (unsigned long) fimg.height},
        "img", "out", "width", "height");
    read_buf(fout.img_data.data(), "out");
    toc = clock();
    dur = toc - tic;
    printf("gpu time consumption: %ld\n", dur);
    fout.save("opencl_out.png");
    return 0;
}

