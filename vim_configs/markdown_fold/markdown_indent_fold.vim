" Markdown headings contain indent folds. Source this file to try it.
" Only ATX headings (# ... through ###### ...) are recognized.
let s:cache_version = 3

function! s:BuildLevels() abort
    let l:levels = []
    let l:headings = []
    let l:fence_char = ''
    let l:fence_width = 0

    for l:lnum in range(1, line('$'))
        let l:text = getline(l:lnum)
        let l:inside_fence = !empty(l:fence_char)
        let l:fence = matchlist(l:text, '^\s*\(`\{3,}\|\~\{3,}\)\(.*\)$')

        if l:inside_fence
            if !empty(l:fence) && strpart(l:fence[1], 0, 1) ==# l:fence_char
                        \ && strlen(l:fence[1]) >= l:fence_width
                        \ && l:fence[2] =~# '^\s*$'
                let l:fence_char = ''
            endif
        elseif !empty(l:fence)
            " Backtick fence info strings cannot themselves contain backticks.
            if strpart(l:fence[1], 0, 1) !=# '`' || l:fence[2] !~# '`'
                let l:fence_char = strpart(l:fence[1], 0, 1)
                let l:fence_width = strlen(l:fence[1])
                let l:inside_fence = 1
            endif
        endif

        let l:heading = l:inside_fence ? '' :
                    \ matchstr(l:text, '^ \{0,3}\zs#\{1,6}\ze\(\s\|$\)')
        if !empty(l:heading)
            let l:rank = strlen(l:heading)
            while !empty(l:headings) && l:headings[-1] >= l:rank
                call remove(l:headings, -1)
            endwhile
            call add(l:headings, l:rank)
            " Keep the heading outside its own fold, but inside its parent.
            " The following content rises one level and starts the body fold.
            call add(l:levels, len(l:headings) - 1)
        elseif l:text =~# '^\s*$'
            " Resolve each blank run explicitly after scanning the document.
            call add(l:levels, -1)
        else
            call add(l:levels, len(l:headings) + indent(l:lnum) / shiftwidth())
        endif
    endfor

    " Give ALL blanks in a run the lower surrounding level. In particular,
    " chapter-ending blanks and EOF blanks must stay outside the body fold.
    " Do not leave these as -1 for Vim to resolve during fold calculation.
    let l:index = 0
    while l:index < len(l:levels)
        if l:levels[l:index] != -1
            let l:index += 1
            continue
        endif
        let l:start = l:index
        let l:before = l:start > 0 ? l:levels[l:start - 1] : 0
        while l:index < len(l:levels) && l:levels[l:index] == -1
            let l:index += 1
        endwhile
        let l:after = get(l:levels, l:index, 0)
        let l:blank_level = min([l:before, l:after])
        for l:blank in range(l:start, l:index - 1)
            let l:levels[l:blank] = l:blank_level
        endfor
    endwhile

    let b:markdown_indent_fold_cache = {
                \ 'key': [s:cache_version, b:changedtick, &l:shiftwidth, &l:tabstop, &l:vartabstop],
                \ 'levels': l:levels,
                \ }
endfunction

function! MarkdownIndentFold(lnum) abort
    let l:key = [s:cache_version, b:changedtick, &l:shiftwidth, &l:tabstop, &l:vartabstop]
    if !exists('b:markdown_indent_fold_cache')
                \ || b:markdown_indent_fold_cache.key !=# l:key
        call s:BuildLevels()
    endif
    return get(b:markdown_indent_fold_cache.levels, a:lnum - 1, 0)
endfunction

function! s:Setup() abort
    setlocal foldexpr=MarkdownIndentFold(v:lnum)
    setlocal foldmethod=expr
    setlocal foldlevel=99
endfunction

augroup MarkdownIndentFolding
    autocmd!
    autocmd FileType markdown call s:Setup()
augroup END

if &l:filetype ==# 'markdown'
    call s:Setup()
endif
