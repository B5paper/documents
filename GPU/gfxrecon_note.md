# gfxrecon note

官网：

<https://github.com/LunarG/gfxreconstruct/tree/dev>

忘了 gfxreconstruct 是怎么安装的了，好像是直接用 apt 安装就可以了

capture frame:

```bash
gfxrecon-capture-vulkan.py --trigger F5 -f 1 <program>
```

如果不灵，可以在前面加上`python3`，然后把`gfxrecon-capture-vulkan.py`替换成使用`which`找到的绝对路径。

convert captured data to json file:

```bash
gfxrecon-convert --file-per-frame <capture_data>
```

extract shaders:

```bash
gfxrecon-extract --dir <dir> <capture_data>
```

replay:

```bash
gfxrecon-replay --paused --replace-shaders <dir> --validate <capture_data>
```

## Operations in order

* capture

    ```bash
    python3 /usr/bin/gfxrecon-capture-vulkan.py --trigger F3 --trigger-frames 1 <program_path>
    ```

* 测试 replay

    ```bash
    gfxrecon-replay --paused --validate <gfxr_file_path>
    ```
    
    用这个命令测试就可以了，可以使用右键或`n`切换到下一帧。

* 抽取所有的 shaders

    ```bash
    gfxrecon-extract --dir <dir> <gfxr_file_path>
    ```

    将 gfxr 文件中的所有 shader 都放到指定目录下。

    这里的 shader 都是二进制的格式，如果需要反汇编，那么还需要与`spirv-cross`工具搭配使用。

* 抽取 api trace

    ```bash
    gfxrecon-convert --file-per-frame <gfxr_file_path>
    ```

    将指定 gfxr 文件中的 vulkan api 调用部分转换成 json 格式的文件。
    
    `--file-per-frame`表示每一帧生成一个 json 文件，这样可以防止一个文件太大。

    这一步过后可以清晰地看到整个程序的 vk api 调用过程。但是目前还不清楚怎么把里面的非 api 数据，比如图像数据，字体，文字之类的数据拿出来。

    通过观察 pipeline 和 renderpass 的创建过程，以及`vkCmdDraw()`，`vkCmdDrawIndexed()`等绘制函数，可以找到绘制某一帧时对应的 shader。

* shader 的反汇编与重新编译

    可以使用`spirv-cross`对二进制格式的 shader 进行反汇编，得到源代码，修改完代码后，再编译回来。

    反汇编：

    ```bash
    spirv-cross -V <shader_module_path>
    ```

    这行命令可以在 stdout 输出 shader module 的反汇编代码。
    
    其中`-V`表示使用 vulkan 的 GLSL 语法。（经过测试，不加`-V`得到的代码确实不能用，后续会有问题）

    如果需要将反汇编代码写到文件中，可以加上`--output <output_file>`参数，也可以直接重定向：`spirv-cross -V <shader_module> > <source_code_path>`。

    在生成反汇编代码时，会丢失一些资源信息。可以使用

    ```bash
    spirv-cross --reflect --dump-resources <shader_module>
    ```

    得到 json 格式的输入输出资源信息。在后续编译之前，我们需要手动把 set，binding 等信息写入到反汇编代码里。

    `--reflect`必须在前，`--dump-resources`必须在后，不然会报错。

    此时就可以放心对反汇编的代码文件进行修改了。

    修改完成后，可以使用`glslang`将反汇编重新编译成 binary module:

    ```bash
    glslang -H -V -e <new_entry_point_name> --source-entrypoint <old_entry_point_name> -o <output_file> <source_code_file>
    ```

    其中，`-H`表示在 stdout 上输出人类可读的 IR 代码，`-V`表示使用 vulkan 语法。

    `-e`表示修改入口函数的名称，`--source-entrypoint`表示代码中的函数入口名称，通常为`main`。

    注意，`<source_code_file>`需要有扩展名，比如`.vert`或`.frag`，用于区分不同的 stage。

* replace shaders, and replay

    ```bash
    gfxrecon-replay --paused --validate --replace-shaders <dir> <gfxr_file_path>
    ``` 

    可以使用`--replace-shaders`对重编译的 shader 进行替换，并重新渲染，这样就可以看自己修改 shader 之后的效果了。

    如果重编译的 shader 无法正常使用，那么 stdout 中会有报错信息，照着修改就可以了。

    常见的问题：

    1. glsl 语法不是 vulkan 语法。解决方案：记得参数加`-V`

    2. set, binding 有问题。解决方案：使用`spirv-cross`将修改之前的 shader 和修改之后的 shader 都使用`--reflect --dump-resources`将资源打印出来，并做对比，看看前后是否一致。

    3. entry point 的问题。解决方案：程序需要的 entry point name 可以通过 dump resources 获得，也可以通过在程序运行时 validation layer 的报错获得。


## Problems shooting

* 直接运行`gfxrecon-capture-vulkan.py`并配合抓帧快捷键参数，在启动程序后，按下快捷键无反应。

    可以先用`which`找到 py 文件的地址：

    ```bash
    which gfxrecon-capture-vulkan.py
    ```

    输出：

    ```
    /usr/bin/gfxrecon-capture-vulkan.py
    ```
    
    然后在前面加上`python3`：

    ```bash
    python3 /usr/bin/gfxrecon-capture-vulkan.py --trigger F3 --trigger-frames 1 <program_path>
    ```

    这样在被抓帧的程序界面按 F3，就有可能可以抓到帧。由于使用 gfxrecon 启动`<program>`时，被抓帧的程序会变卡，所以按 F3 不一定能成功抓帧。多按几次总有一次会成功的。

    抓帧成功后，生成的 gfxr 文件可能有 200M 到 500M 不等，这些都是正常的。convert 后可以看到里面只有一帧。