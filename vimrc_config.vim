call plug#begin()

" List your plugins here
Plug 'junegunn/seoul256.vim'

Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']}

call plug#end()

" autocmd FileType c,cpp setlocal iskeyword+=#
" autocmd FileType c,cpp setlocal complete+=s

set autoindent
set softtabstop=4
set tabstop=4
set expandtab
set shiftwidth=4

nnoremap <space> i<space><esc>l
nnoremap <BS> X
nnoremap <CR> o<Esc>
nnoremap <leader><CR> dd
" nnoremap <leader><tab> i	<esc>
nnoremap <leader><tab> i<C-v><Tab><Esc>
" nnoremap <Tab> i    <Esc>


" 快速移动
nnoremap H ^                     " H 移动到行首
nnoremap L $                     " L 移动到行尾
nnoremap J 5j                    " J 向下移动5行
nnoremap K 5k                    " K 向上移动5行


" 定义带范围判断的函数，确保仅在Visual选中后有效
function! AddStarPrefixToSelectedLines() abort
    " ==============================================
    " 第一步：校验并获取Visual选中的行范围（增加容错判断）
    " ==============================================
    " 先判断是否存在Visual选中标记（避免无选中时执行报错）
    if !exists("'<") || !exists("'>")
        echom "错误：请先在Visual模式下选中目标行！"
        return
    endif

    " 获取选中区域的起始行和结束行（行号为正整数）
    let s:start_line = line("'<")
    let s:end_line = line("'>")

    " 校验行范围有效性
    if s:start_line > s:end_line
        echom "错误：选中行范围无效！"
        return
    endif

    " ==============================================
    " 第二步：正序/倒序可选遍历（这里保留倒序，彻底避免行偏移）
    " ==============================================
    " 倒序遍历：从结束行到起始行，不受行内容修改的影响
    for s:current_line in range(s:end_line, s:start_line, -1)
        " ==============================================
        " 第三步：获取当前行内容，并严格判断是否为纯空白行
        " ==============================================
        " 获取当前行完整内容（自动过滤末尾换行符，避免干扰判断）
        let s:line_text = getline(s:current_line)
        
        " 严格判断：是否仅包含空格、Tab（纯空白行，无有效文字）
        " 方法：将所有空白符替换为空，若结果为空则是纯空白行
        let s:non_whitespace_text = substitute(s:line_text, '\s', '', 'g')
        if empty(s:non_whitespace_text)
            " 纯空白行：跳过当前循环，不做任何处理
            continue
        endif

        " ==============================================
        " 第四步：精准找到第一个非空白符的位置，插入* 
        " ==============================================
        " 方法1：用searchpos在当前行内搜索第一个非空白符（更稳定）
        " 参数说明：\S（非空白符）, 'cn'（不移动光标、从行首开始搜索）
        let s:pos = searchpos('\S', 'cn', s:current_line)
        let s:first_non_blank_col = s:pos[1]  " 列号（Vim中列从1开始）

        " 拼接新行内容：行首到第一个非空白符前 + *  + 第一个非空白符到行尾
        " strpart：Vim字符串截取（索引从0开始，需注意列号转换）
        let s:prefix = strpart(s:line_text, 0, s:first_non_blank_col - 1)
        let s:suffix = strpart(s:line_text, s:first_non_blank_col - 1)
        let s:new_line_text = s:prefix . "* " . s:suffix

        " 更新当前行内容
        call setline(s:current_line, s:new_line_text)
    endfor

    " 执行成功提示
    echom "成功处理！行范围：" . s:start_line . " - " . s:end_line
endfunction

" 可视化模式下按 \s 一键调用（<leader>默认是\）
vnoremap <leader>s :call AddStarPrefixToSelectedLines()<CR>


function AddAsterisk()
    " let start_line = line("'<")
    " let end_line = line("'>")
    " echo "选中的行范围: " . start_line . " 到 " . end_line
    " for lnum in range(start_line, end_line)
    "     let line = getline(lnum)
    "     if line !~ '\S'
    "         continue
    "     endif
    "     execute lnum . 'normal! ^i* '
    " endfor
    let line = getline('.')
    if line !~ '\S'
        return 1
    endif
    let lnum = line('.')
    execute lnum . 'normal! ^i* '
    return 1
endfunction

vnoremap <leader>a :call AddAsterisk()<CR>

" inoremap <leader>t <Tab>


function ConvertMarkdownTable()
    let lnum = line('.')
    execute lnum . 'normal! ^i| '
    execute lnum . 's/\t/ | /g'
    execute lnum . 'normal! $a |'
endfunction

vnoremap <leader>m :call ConvertMarkdownTable()<cr>


" sroundding macro
let @b = '`<i**`>2la**`<'

" srounding code
let @c = '`<^i````>oi````<jddk'

set relativenumber

nnoremap zT zt3<C-y>
vnoremap zT zt3<C-y>

nnoremap > >>
nnoremap m 2j
vnoremap m 2j
