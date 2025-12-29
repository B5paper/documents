# Ffmpeg Note

## cache

* ffmpeg 将 mp4 转换为固定码率 128Kbps 的 mp3

    `ffmpeg -i input.mp4 -b:a 128k output.mp3`

    指定编码器和采样率：

    `ffmpeg -i input.mp4 -codec:a libmp3lame -b:a 128k -ar 44100 output.mp3`

    * -codec:a libmp3lame：指定使用 LAME MP3 编码器

        LAME 编码器支持的标准比特率有：

        96k、128k、192k、256k、320k 等

        128k 是常见的平衡质量和文件大小的选择

## note
