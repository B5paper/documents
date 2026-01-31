# Ffmpeg Note

## cache

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
