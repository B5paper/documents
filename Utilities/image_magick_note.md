# Image magick

`magick image.jpg image.png`

## cache

* 是否有能实时预览的命令行图片编辑工具

    是的，有多个支持实时预览的命令行图片编辑工具：

    1. chafa - 终端图片查看器

        ```bash
        # 安装
        sudo apt install chafa    # Ubuntu
        brew install chafa        # macOS

        # 使用
        chafa image.jpg           # 直接显示图片
        chafa -s 80x40 image.jpg  # 指定终端大小
        chafa --animate gif.gif   # 显示GIF动画
        ```

    2. termimage - Python终端图片库

        ```bash
        # 安装
        pip install termimage

        # 使用
        termimage image.png       # 自适应显示
        termimage -s 60x30 img.jpg # 指定尺寸
        ```

    3. timg - 终端图片和视频查看器

        ```bash
        # 安装
        sudo apt install timg     # Ubuntu
        brew install timg         # macOS

        # 使用
        timg image.jpg            # 显示图片
        timg -g 80x40 image.png   # 指定几何尺寸
        timg -p image.jpg         # 半透明混合显示
        ```

    4. viu - Rust编写的终端图片查看器

        ```bash
        # 安装
        cargo install viu        # 通过cargo
        # 或下载预编译二进制

        # 使用
        viu image.jpg            # 显示图片
        viu -w 80 image.png      # 指定宽度
        viu -t 0.5 image.jpg     # 透明度
        ```

    5. imgcat - iTerm2专用

        ```bash
        # iTerm2内置功能，可直接显示
        imgcat photo.jpg
        ```

    6. fim - Fbi Improved (FrameBuffer)

        ```bash
        # 安装
        sudo apt install fim     # Ubuntu

        # 使用
        fim image.jpg            # 全屏查看
        fim -a *.jpg            # 幻灯片播放
        ```

    7. sxiv - 简单X图像查看器

        ```bash
        # 安装
        sudo apt install sxiv    # Ubuntu
        brew install sxiv        # macOS

        # 使用
        sxiv image.jpg           # 查看单张
        sxiv *.jpg              # 查看多张，支持翻页
        ```

    8. feh - 轻量级图像查看器

        ```bash
        # 安装
        sudo apt install feh     # Ubuntu

        # 使用
        feh image.jpg            # 查看图片
        feh -g 800x600 img.jpg   # 指定窗口大小
        feh -Z *.jpg            # 缩放适应窗口
        ```

    9. mpv - 视频播放器（也支持图片）

        ```bash
        # 安装
        sudo apt install mpv     # Ubuntu

        # 使用
        mpv --image-display-duration=5 *.jpg   # 幻灯片播放
        ```

    10. ranger + w3m - 文件管理器集成

        ```bash
        # 安装
        sudo apt install ranger w3m-img

        # 配置 ~/.config/ranger/rc.conf
        set preview_images true
        set preview_images_method w3m

        # 使用
        ranger                        # 进入ranger，图片自动预览
        ```

    实时编辑预览方案

    方案一：ImageMagick + 终端预览

    ```bash
    #!/bin/bash
    # 创建预览脚本 preview-edit.sh
    input="input.jpg"
    temp="temp_preview.jpg"

    # 1. 复制原图
    cp "$input" "$temp"

    # 2. 应用效果并预览
    convert "$temp" -resize 800x600 - | chafa -

    # 3. 交互式编辑
    while true; do
        echo "选项: 1)调整亮度 2)裁剪 3)保存退出"
        read choice
        case $choice in
            1) convert "$temp" -brightness-contrast 10x0 "$temp" ;;
            2) convert "$temp" -crop 300x300+100+100 "$temp" ;;
            3) cp "$temp" "output.jpg"; break ;;
        esac
        convert "$temp" -resize 800x600 - | chafa -
    done
    ```

    方案二：使用 watch 命令实时监控

    ```bash
    # 监控文件夹变化并预览
    watch -n 1 'convert latest_edit.jpg -resize 80x40 - | base64 | awk "{print \"<img src=\"data:image/png;base64,\"\$0\" />\"}" | lynx -stdin'

    # 或使用更简单的方式
    watch -n 0.5 'timg -g 100x50 current_image.jpg'
    ```

    方案三：终端图片编辑器组合

    ```bash
    # 使用 vimiv + 脚本
    #!/bin/bash
    # edit-with-preview.sh
    INPUT=$1
    PREVIEW="/tmp/preview.jpg"

    # 创建初始预览
    convert "$INPUT" -resize 50% "$PREVIEW"

    # 在一个终端显示预览
    tmux new-window "while true; do timg $PREVIEW; sleep 0.5; done"

    # 在另一个终端编辑
    vim "$INPUT"

    # 在脚本中更新预览
    watch_conversion() {
        while inotifywait -e close_write "$INPUT"; do
            convert "$INPUT" -resize 50% "$PREVIEW"
        done
    }
    ```

    推荐组合

    简单预览：

    ```bash
    # 快速查看图片
    alias view='timg -g $(($(tput cols)-10))x$(($(tput lines)-5))'

    # 使用
    view image.jpg
    ```

    编辑时实时预览：

    ```bash
    #!/bin/bash
    # realtime-edit.sh
    set -e

    input="$1"
    preview="/tmp/img_preview.png"

    # 清理函数
    cleanup() {
        rm -f "$preview"
        kill $watch_pid 2>/dev/null
    }
    trap cleanup EXIT

    # 创建预览窗口
    {
        while true; do
            if [ -f "$input" ]; then
                convert "$input" -resize 60x30 - "$preview" 2>/dev/null
                clear
                chafa "$preview" 2>/dev/null || echo "正在生成预览..."
            fi
            sleep 0.5
        done
    } &
    watch_pid=$!

    # 主编辑循环
    echo "编辑命令: resize WxH, crop WxH+X+Y, rotate DEG, save, quit"
    while read -p "> " cmd; do
        case $cmd in
            resize*)
                size=${cmd#resize }
                convert "$input" -resize "$size" "$input"
                ;;
            save*)
                cp "$input" "edited_${input}"
                echo "已保存为 edited_${input}"
                ;;
            quit) break ;;
            *)
                convert "$input" $cmd "$input" 2>/dev/null || echo "命令错误"
                ;;
        esac
    done
    ```

    使用建议：

    * 终端内预览：timg 或 chafa（纯终端）

    * 弹窗预览：feh 或 sxiv（需要X窗口）

    * 编辑时预览：使用 watch 命令监控文件变化

    * 批量处理：先小图预览，确认效果后再处理原图

* imagemagick

    ImageMagick 是一款强大的命令行图像处理工具集。以下是一些常见用法：

    安装

    ```bash
    # Ubuntu/Debian
    sudo apt install imagemagick

    # macOS
    brew install imagemagick

    # Windows
    # 从官网下载安装包
    ```

    基本命令格式

    ```bash
    convert [输入选项] 输入文件 [输出选项] 输出文件
    ```

    常用功能示例

    1. 格式转换

        ```bash
        convert image.jpg image.png          # JPG转PNG
        convert image.tiff image.pdf         # TIFF转PDF
        convert *.jpg images.pdf             # 多张JPG合并为PDF
        ```

    2. 调整尺寸

        ```bash
        convert input.jpg -resize 800x600 output.jpg      # 调整到指定尺寸
        convert input.jpg -resize 50% output.jpg          # 缩放50%
        convert input.jpg -resize 800x600! output.jpg     # 强制尺寸（忽略比例）
        ```

    3. 裁剪图像

        ```bash
        convert input.jpg -crop 300x300+100+50 output.jpg  # 从(100,50)裁剪300x300区域
        convert input.jpg -crop 2x2@ output.jpg           # 分割为2x2网格
        ```

    4. 旋转和翻转

        ```bash
        convert input.jpg -rotate 90 output.jpg           # 旋转90度
        convert input.jpg -flip output.jpg                # 垂直翻转
        convert input.jpg -flop output.jpg                # 水平翻转
        ```

    5. 添加水印

        ```bash
        convert input.jpg -pointsize 20 -fill white -annotate +10+30 "水印文字" output.jpg
        convert input.jpg watermark.png -gravity southeast -composite output.jpg
        ```

    6. 调整质量

        ```bash
        convert input.jpg -quality 85 output.jpg          # 设置JPEG质量(1-100)
        convert input.jpg -compress lossless output.png   # 无损压缩
        ```

    7. 批量处理

        ```bash
        mogrify -resize 800x600 *.jpg                     # 批量调整尺寸
        convert *.jpg -set filename:f "%t" +adjoin "%[filename:f].png"  # 批量转换格式
        ```

    实用技巧

    创建缩略图

    ```bash
    convert input.jpg -thumbnail 200x200 -background white -alpha remove thumbnail.jpg
    ```

    图片合成

    ```bash
    convert image1.jpg image2.jpg +append output.jpg   # 水平拼接
    convert image1.jpg image2.jpg -append output.jpg   # 垂直拼接
    ```

    添加边框

    ```bash
    convert input.jpg -bordercolor black -border 10x10 output.jpg
    ```

    调整亮度和对比度

    ```bash
    convert input.jpg -brightness-contrast 10x20 output.jpg
    ```

    注意事项

    * ImageMagick 支持 200+ 种图像格式

    * 使用 identify 命令查看图像信息：identify image.jpg

    * mogrify 命令会直接修改原文件，使用时建议备份

    * 可以使用 -verbose 参数查看处理详情

    这些只是基础用法，ImageMagick 功能非常丰富，可通过 man convert 查看完整手册。

* imagemagick 将 pdf 转换成 png

    ```bash
    convert -density 300 -background white -alpha remove -alpha off -define pdf:use-cropbox=true example.pdf output/example.png
    ```

    说明：

    * `-density 300`: 不写这个 dpi 的话，默认值可能是 100 以下，字很糊

    * `-background white -alpha remove -alpha off`: 不设置这个的话，底色是透明色

        不清楚`-alpha off`是什么作用，加上似乎也不影响前两个参数的效果。但是如果单独使用`-alpha off`，那么底色仍是透明色，不清楚原因。

    * `-define pdf:use-cropbox=true`: 不设置这个的话，假如 pdf 有多页，并且 size 不一致，那么输出的 png 有的可能有大面积空白

    * `output/example.png`：imagemagick 没有类似`--output-dir`的选项，输出的目录需要自己配置

    * 如果 pdf 有多页，那么输出的 png 会被命名为`example-0.png`, `example-1.png`...

    * imagemagice 处理 pdf 文件时会报错

        ```
        convert-im6.q16: attempt to perform an operation not allowed by the security policy `PDF' @ error/constitute.c/IsCoderAuthorized/426.
        convert-im6.q16: no images defined `output/31060125_0055105189101995897.png' @ error/convert.c/ConvertImageCommand/3229.
        ```

        解决办法是在`/etc/ImageMagick-6/policy.xml`文件中，将

        `<policy domain="coder" rights="none" pattern="PDF" />`

        这一行的`rights`的`none`改为`read | write`。

* ImageMagick 把 PDF 转换成 PNG

    基本转换方法

    ```bash
    # 将PDF转换为PNG
    convert input.pdf output.png
    ```

    重要参数说明

    1. 设置分辨率（关键参数）

        PDF 是矢量格式，PNG 是位图格式，需要指定输出分辨率：

        ```bash
        convert -density 300 input.pdf output.png        # 300 DPI，高质量
        convert -density 150 input.pdf output.png        # 150 DPI，中等质量
        convert -density 72 input.pdf output.png         # 72 DPI，网页质量
        ```

    2. 处理多页PDF

        ```bash
        # 方法1：每页单独输出为PNG文件
        convert -density 150 input.pdf output-%02d.png
        # 生成：output-01.png, output-02.png, ...

        # 方法2：所有页合并为一个PNG（垂直排列）
        convert -density 150 input.pdf -append output.png

        # 方法3：指定特定页面
        convert -density 150 input.pdf[0] first-page.png    # 第1页
        convert -density 150 input.pdf[0-2] pages.png       # 第1-3页
        ```

    3. 调整输出质量

        ```bash
        # 设置PNG压缩级别（0-9，0=无压缩）
        convert -density 150 input.pdf -quality 95 output.png

        # 去除透明度背景为白色
        convert -density 150 input.pdf -background white -alpha remove -alpha off output.png
        ```

    实用示例

    批量处理

    ```bash
    # 批量转换文件夹内所有PDF
    for pdf in *.pdf; do
        convert -density 150 "$pdf" "${pdf%.pdf}.png"
    done
    ```

    高质量转换

    ```bash
    # 高质量转换单页
    convert -density 300 input.pdf[0] -resize 25% -quality 100 page1.png

    # 转换并优化文件大小
    convert -density 150 input.pdf -colors 256 -strip page.png
    ```

    自定义输出

    ```bash
    # 指定尺寸而不是DPI
    convert -density 150 input.pdf -resize 1024x768 output.png

    # 仅转换前5页
    convert -density 150 input.pdf[0-4] output-%d.png
    ```

    替代命令 pdftoppm

    ImageMagick 内部使用 Ghostscript 处理 PDF，也可以直接使用其他工具：

    ```bash
    # 使用 pdftoppm (poppler-utils 的一部分)
    pdftoppm -png -r 300 input.pdf output-prefix

    # 使用 gs (Ghostscript)
    gs -dNOPAUSE -sDEVICE=png16m -r300 -sOutputFile=output-%d.png input.pdf -c quit
    ```

    常见问题

    安全问题

    新版本 ImageMagick 出于安全考虑，默认禁用 PDF 处理。需要修改策略文件：

    编辑 /etc/ImageMagick-6/policy.xml（路径可能不同）：

    ```xml
    <!-- 将这一行 -->
    <policy domain="coder" rights="none" pattern="PDF" />

    <!-- 改为 -->
    <policy domain="coder" rights="read|write" pattern="PDF" />
    ```

    内存问题

    处理大尺寸PDF时：

    ```bash
    # 限制内存使用
    convert -limit memory 2GiB -limit map 4GiB -density 150 large.pdf output.png

    # 使用临时磁盘缓存
    convert -limit disk 10GiB -density 150 large.pdf output.png
    ```

    推荐的工作流程

    ```bash
    # 1. 高质量转换单页
    convert -density 300 input.pdf[0] -quality 100 page1.png

    # 2. 批量转换多页PDF
    convert -density 150 input.pdf -scene 1 page-%02d.png

    # 3. 转换为适合网页的PNG
    convert -density 150 input.pdf -resize 800x600 -strip -quality 90 web-ready.png
    ```

    注意：PDF 到 PNG 转换的质量主要取决于 -density 参数的值，值越高图像越清晰，但文件也越大。

## topics