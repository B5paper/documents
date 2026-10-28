" Run: vim -Nu NONE -i NONE -n -es -S test_markdown_indent_fold.vim
set nomore
set shiftwidth=4 tabstop=4 expandtab
let s:root = expand('<sfile>:p:h')
execute 'source ' . fnameescape(s:root . '/markdown_indent_fold.vim')
execute 'edit ' . fnameescape(s:root . '/markdown_indent_fold_demo.md')
setfiletype markdown
normal! zx

function! s:Line(text) abort
    return search('^' . escape(a:text, '\.*$^~[]') . '$', 'nw')
endfunction

call assert_equal('expr', &l:foldmethod)
call assert_equal(0, foldlevel(1))
call assert_equal(1, foldlevel(s:Line('    Preamble detail one')))
call assert_equal(0, foldlevel(s:Line('# Chapter one')))
call assert_equal(1, foldlevel(s:Line('* List A (this line is outside its indent fold)')))
call assert_equal(2, foldlevel(s:Line('    Detail A1')))
call assert_equal(2, foldlevel(s:Line('    * Nested list (outside its own deeper fold)')))
call assert_equal(3, foldlevel(s:Line('        Nested detail one')))
call assert_equal(1, foldlevel(s:Line('## Section one')))
call assert_equal(3, foldlevel(s:Line('    Detail C1')))
call assert_equal(2, foldlevel(s:Line('# This is code, not a chapter')))
call assert_equal(3, foldlevel(s:Line('    return 42')))
call assert_equal(0, foldlevel(s:Line('# Chapter two')))
call assert_equal(1, foldlevel(s:Line('### Skipped heading level')))
call assert_equal(2, foldlevel(s:Line('# Still code')))

" zc first closes indentation, then the containing chapter.
normal! zR
call cursor(s:Line('    Detail A1'), 1)
normal! zc
call assert_equal(s:Line('    Detail A1'), foldclosed('.'))
call assert_equal(s:Line('    Detail A3'), foldclosedend('.'))
call assert_equal(-1, foldclosed(s:Line('* List A (this line is outside its indent fold)')))
normal! zc
call assert_equal(s:Line('Normal paragraph.'), foldclosed('.'))
call assert_equal(-1, foldclosed(s:Line('# Chapter one')))
call assert_equal(prevnonblank(s:Line('# Chapter two') - 1), foldclosedend('.'))

" Adjacent sibling sections remain separate folds.
normal! zR
call cursor(s:Line('Section paragraph.'), 1)
normal! zc
call assert_equal(s:Line('Section paragraph.'), foldclosed('.'))
call assert_equal(-1, foldclosed(s:Line('## Section one')))
call assert_equal(-1, foldclosed(s:Line('## Section two')))
call assert_equal(prevnonblank(s:Line('## Section two') - 1), foldclosedend('.'))

" Edits invalidate the cached fold levels.
normal! zR
call setline(s:Line('    Detail B1'), '        Detail B1')
normal! zx
call assert_equal(3, foldlevel(s:Line('        Detail B1')))
setlocal shiftwidth=2
normal! zx
call assert_equal(5, foldlevel(s:Line('        Detail B1')))

" With no headings this behaves like indent folding, including # code lines.
execute 'edit! ' . fnameescape(tempname() . '.md')
setfiletype markdown
setlocal shiftwidth=4
call setline(1, ['* List', '    first', '    # indented code', '    last', '* Next'])
normal! zx
call assert_equal([0, 1, 1, 1, 0], map(range(1, 5), 'foldlevel(v:val)'))

" Empty headings and shorter fences do not swallow following chapters.
execute 'edit! ' . fnameescape(tempname() . '.md')
setfiletype markdown
call setline(1, ['# Empty', '# Next', '````', '# code', '```',
            \ '# still code', '````', '## Child', 'text', '# End', 'text'])
normal! zx
call assert_equal([0, 0, 1, 1, 1, 1, 1, 1, 2, 0, 1],
            \ map(range(1, 11), 'foldlevel(v:val)'))

" Empty headings, blank lines and deeply indented first body lines.
execute 'edit! ' . fnameescape(tempname() . '.md')
setfiletype markdown
call setline(1, ['# Empty', '# Parent', '', '    detail', '    more',
            \ '## Child', '', 'child body', 'more body', '## Sibling',
            \ 'sibling body', 'more body', '# Last'])
normal! zx
normal! zR
call cursor(8, 1)
normal! zc
call assert_equal(8, foldclosed('.'))
call assert_equal(9, foldclosedend('.'))
call assert_equal(-1, foldclosed(6))
call assert_equal(-1, foldclosed(10))
normal! zc
call assert_equal(4, foldclosed('.'))
call assert_equal(12, foldclosedend('.'))
call assert_equal(-1, foldclosed(1))
call assert_equal(-1, foldclosed(2))
call assert_equal(-1, foldclosed(13))

" Every trailing blank stays visible, including whitespace-only lines and EOF.
execute 'edit! ' . fnameescape(tempname() . '.md')
setfiletype markdown
call setline(1, ['# Parent', 'parent body', '## Child', 'child body',
            \ 'more child body', '', '    ', '', '## Sibling',
            \ 'sibling body', '', '', '', '# Last', 'last body',
            \ 'more last body', '', '  ', ''])
normal! zx
normal! zR
call cursor(4, 1)
normal! zc
call assert_equal(5, foldclosedend(4))
for s:lnum in range(6, 8)
    call assert_equal(-1, foldclosed(s:lnum), 'child trailing blank ' . s:lnum)
endfor
normal! zR
call cursor(2, 1)
normal! zc
call assert_equal(10, foldclosedend(2))
for s:lnum in range(11, 13)
    call assert_equal(-1, foldclosed(s:lnum), 'parent trailing blank ' . s:lnum)
endfor
call cursor(15, 1)
normal! zc
call assert_equal(16, foldclosedend(15))
for s:lnum in range(17, 19)
    call assert_equal(-1, foldclosed(s:lnum), 'EOF trailing blank ' . s:lnum)
endfor

if !empty(v:errors)
    for s:error in v:errors
        echoerr s:error
    endfor
    cquit
endif
echom 'PASS: Markdown heading + indent folding'
qa!
