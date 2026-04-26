# Ffmpeg Note

## cache

* ffmpeg 如何将一个文件下规律命名的 000.png, 001.png, 002.png 等图片文件转换成 mp4 视频 ？

    使用 FFmpeg 将序列图片合成为视频是一个非常高效的操作。针对你这种 `000.png`, `001.png` 这种**规律命名**的文件，最常用的方法是使用 **通配符（Sequence patterns）**。

    **核心命令**

    打开终端或命令行，进入图片所在文件夹，执行以下命令：

    ```bash
    ffmpeg -framerate 24 -i %03d.png -c:v libx264 -pix_fmt yuv420p output.mp4
    ```

    ---

    **参数详细拆解**

    * **`-framerate 24`**: 设置输出视频的帧率（每秒播放多少张图片）。24 或 30 是常用的数值。**注意：** 这个参数必须放在 `-i` 之前。
    * **`-i %03d.png`**: 这是关键。`%03d` 是一个占位符，表示**三位数字**（如 000, 001, 002）。
        * 如果是 `0000, 0001`，则改用 `%04d`。
        * 如果是 `1, 2, 3`，则改用 `%d`。
    * **`-c:v libx264`**: 指定视频编码器为 H.264，这是目前兼容性最好的格式。
    * **`-pix_fmt yuv420p`**: **非常重要！** 很多图片是 RGB 格式，但大多数播放器（如 QuickTime 或手机端）对 H.264 的要求是 YUV420P 像素格式。如果不加这个参数，生成的视频可能在某些设备上无法播放。
    * **`output.mp4`**: 输出的文件名。

    ---

    **进阶场景处理**

    1. 图片不是从 000 开始的？

        如果你的序列是从 `005.png` 开始的，可以使用 `-start_number` 参数：
        ```bash
        ffmpeg -start_number 5 -i %03d.png -c:v libx264 -pix_fmt yuv420p output.mp4
        ```

    2. 图片尺寸不一或需要缩放？
        
        如果图片分辨率很大，或者你想强制视频为 1080p，可以添加缩放滤镜：
        ```bash
        ffmpeg -framerate 24 -i %03d.png -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" -c:v libx264 -pix_fmt yuv420p output.mp4
        ```
        > **提示：** 上面的复杂滤镜会在保持比例的基础上，不足的地方补黑边。

    3. 命名不完全规律（如：img_1.png, image_2.png）？
        
        如果文件名不完全符合 `%d` 格式，但在文件夹里按名称排序是正确的，可以使用 **Glob 模式**（仅限 Linux/macOS）：
        ```bash
        ffmpeg -framerate 24 -pattern_type glob -i "*.png" -c:v libx264 -pix_fmt yuv420p output.mp4
        ```

    **常见小坑提醒**

    * **分辨率必须是偶数**：H.264 编码要求视频的宽和高必须能被 2 整除。如果你的图片是单数像素（如 1081x1920），会报错。可以使用 `-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"` 来自动修正。
    * **路径问题**：如果路径中包含空格，记得给输入输出路径加上英文双引号。

* ffmpeg如何提取视频中的音频

    使用FFmpeg提取视频中的音频有多种方法，以下是几种常见的方式：

    1. 基本提取音频

        ```bash
        ffmpeg -i input.mp4 output.mp3
        ```

        这将从input.mp4中提取音频并保存为MP3格式。

    2. 指定音频格式

        ```bash
        # 提取为MP3
        ffmpeg -i input.mp4 -vn -c:a mp3 output.mp3

        # 提取为AAC
        ffmpeg -i input.mp4 -vn -c:a aac output.aac

        # 提取为无损WAV
        ffmpeg -i input.mp4 -vn -c:a pcm_s16le output.wav

        # 提取为FLAC
        ffmpeg -i input.mp4 -vn -c:a flac output.flac
        ```

    3. 常用参数说明

        ```bash
        ffmpeg -i input.mp4 [参数] output.音频格式
        ```

        * -vn：禁用视频流（不处理视频）

        * -c:a 或 -acodec：指定音频编码器

        * -b:a：指定音频比特率（如 -b:a 192k）

        * -ar：指定采样率（如 -ar 44100）

        * -ac：指定声道数（如 -ac 2 为立体声）

    4. 常用格式示例

        MP3（高质量）

        ```bash
        ffmpeg -i input.mp4 -vn -c:a libmp3lame -q:a 0 output.mp3
        # -q:a 0 表示最高质量（0-9，0最好）
        ```

        保持原始音频编码

        ```bash
        ffmpeg -i input.mp4 -vn -c:a copy output.m4a
        # 如果原视频音频是AAC，这会直接复制不重新编码
        ```

        WAV无损格式

        ```bash
        ffmpeg -i input.mp4 -vn -c:a pcm_s16le output.wav
        ```

    5. 批量处理

        ```bash
        # Linux/Mac
        for file in *.mp4; do
            ffmpeg -i "$file" -vn -c:a mp3 "${file%.mp4}.mp3"
        done

        # Windows (批处理文件)
        for %%f in (*.mp4) do (
            ffmpeg -i "%%f" -vn -c:a mp3 "%%~nf.mp3"
        )
        ```

    6. 查看音频信息

        ```bash
        # 查看视频中的音频流信息
        ffmpeg -i input.mp4

        # 只查看音频流信息
        ffmpeg -i input.mp4 -hide_banner | findstr Audio
        ```

    7. 提取特定时间段的音频

        ```bash
        # 提取10-30秒的音频
        ffmpeg -i input.mp4 -vn -ss 00:00:10 -t 00:00:20 output.mp3
        # -ss 开始时间，-t 持续时间
        ```

    注意事项：

    * 确保已安装FFmpeg（可运行 ffmpeg -version 检查）

    * MP3格式需要libmp3lame编码器支持

    * 使用 -c:a copy 可以快速提取而不重新编码，但格式可能受限

    * 提取无损音频推荐使用WAV或FLAC格式

    选择哪种方式取决于你对音频质量、文件大小和兼容性的需求。

* ffmpeg 将 mp4 转换为固定码率 128Kbps 的 mp3

    `ffmpeg -i input.mp4 -b:a 128k output.mp3`

    指定编码器和采样率：

    `ffmpeg -i input.mp4 -codec:a libmp3lame -b:a 128k -ar 44100 output.mp3`

    * -codec:a libmp3lame：指定使用 LAME MP3 编码器

        LAME 编码器支持的标准比特率有：

        96k、128k、192k、256k、320k 等

        128k 是常见的平衡质量和文件大小的选择

## note
