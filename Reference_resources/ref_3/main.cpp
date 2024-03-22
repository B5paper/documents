#include <stdio.h>
#include <stdlib.h>
#include "../CImg/CImg.h"
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;
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


int main(int argc, const char** argv)
{
    if (argc <= 1)
    {
        printf("param error.\n");
        exit(-1);
    }
    const char *img_file_path = argv[1];
    Img img(img_file_path);

    return 0;
}
