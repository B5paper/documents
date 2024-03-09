# CImg Note

Official site: <https://cimg.eu/index.html>

**CImg 的 rgb 是分开的三个通道，并不是一个像素一个像素排列的！**

**CImg 的 rgb 是分开的三个通道，并不是一个像素一个像素排列的！**

**CImg 的 rgb 是分开的三个通道，并不是一个像素一个像素排列的！**

## 将 png 图片转换成 array 形式

`main.cpp`:

```cpp
#include "CImg.h"
#include <iostream>
#include <vector>
using namespace cimg_library;
using namespace std;

int main()
{
    CImg<unsigned char> img("./clothes.png");

    // get the basic info
    int width, height, spectrum, depth;
    width = img.width();
    height = img.height();
    spectrum = img.spectrum();
    depth = img.depth();
    cout << "width: " << width << ", height: " << height << endl;
    cout << "spectrum: " << spectrum << ", depth: " << depth << " (bytes)" << endl;

    // index the pixel
    cout << "the rgb value of the first pixel: " 
        << (int) img(0, 0, 0, 0) << ", "  // img(x, y, z, channel)
        << (int) img(0, 0, 0, 1) << ", "
        << (int) img(0, 0, 0, 2) << endl;
    
    // store data into rgb array
    vector<uint8_t> arr(width * height * spectrum);
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < spectrum; ++k)
            {
                arr[i * width + j * height + k] = img(i, j, 0, k);  // 这代码有错
                // 可能应该改成 i * height * spectrum + j * spectrum + k
            }   
        }
    }
    return 0;
}
```

编译：

```bash
g++ -g main.cpp -lX11 -o main
```

运行：

```bash
./main
```

一个可能的输出：

```
width: 2048, height: 2048
spectrum: 3, depth: 1 (bytes)
the rgb value of the first pixel: 48, 36, 33
```

在索引的时候我们使用了`img(0, 0, 0, 0)`这样的索引方式，坐标模式为`xy`（区别于 row, column），即`x`代表列，`y`代表行。

第三个参数`z`表示图片的“厚度”，由于我们的 png 图片是二维的，“厚度”为 1，所以`z`总是填 0。

第四个参数表示颜色通道，排序方式是 rgb（区别于 opencv 的 bgr）。

## 将 array 内存数据转换成 png 图片

`main.cpp`:

```cpp
#include "CImg.h"
#include <vector>
using namespace cimg_library;
using namespace std;

int main()
{
    vector<uint8_t> arr(1024 * 768 * 3);
    for (int i = 0; i < arr.size() / 3; ++i)
    {
        arr[i * 3 + 0] = 0;
        arr[i * 3 + 1] = 0;
        arr[i * 3 + 2] = 255;
    }

    CImg<unsigned char> img_arr(1024, 768, 1, 3, 0);  // 宽为 1024，高为 768，“厚度” 为 1，通道数为 3，初始数据填充为 0
    int row, col;  // ij coordinate
    for (size_t x = 0; x < 1024; ++x)  // xy coordinate
    {
        for (size_t y = 0; y < 768; ++y)
        {
            row = y;
            col = x;
            img_arr(x, y, 0, 0) = arr[(col + row * 1024) * 3 + 0];
            img_arr(x, y, 0, 1) = arr[(col + row * 1024) * 3 + 1];
            img_arr(x, y, 0, 2) = arr[(col + row * 1024) * 3 + 2];
        }
    }

    img_arr.save_png("img_arr.png");
    return 0;
}
```

编译：

```bash
g++ -g main.cpp -lX11 -o main
```

运行：

要求提前安装`imagemagick`。

```bash
./main
```

效果：

在当前文件夹下生成纯蓝色图片`img_arr.png`。

注：

1. 在使用`img(x, y, 0, 0)`进行索引时，左上角为图像的原点，向右为 x 轴正方向，向下为 y 轴正方向。参数中的`x`指的即是 x 坐标，而不是“第`x`行”。

## 显示图片

`main.cpp`:

```cpp
#include "CImg.h"
#include <unistd.h>
using namespace cimg_library;
using namespace std;

int main()
{   
    CImg<unsigned char> img_arr(1024, 768, 1, 3);
    CImgDisplay main_disp(img_arr, "test");
    while (!main_disp.is_closed())
    {
        main_disp.wait();
        usleep(10000);
    }
    return 0;
}
```

编译：

```bash
g++ -g main.cpp -lX11 -o main
```

运行：

```bash
./main
```

效果：

出现一个黑色的窗口。

