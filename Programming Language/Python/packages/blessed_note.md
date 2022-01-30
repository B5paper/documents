# Blessed Note

安装：`pip install blessed`

```python
from blessed import Terminal
term = Terminal()

term.height, term.width
term.number_of_colors

str = term.green_reverse('ALL SYSTEMS GO')
print(str)

print(f"{term.yellow}Yellow is brown, {term.bright_yellow}"
          f"Bright yellow is actually yellow!{term.normal}")

print(term.underline_bold_green_on_yellow('They live! In sewers!'))
```

clearing the screen

* `clear`: clear the whole screen
* `clear_eol`: clear to the end of the line
* `clear_bol`: clear backward to the beginning of the line
* `clear_eos`: clear to the end of screen

在使用时建议总是和`home`合起来用：`print(term.home + term.on_blue + term.clear)`

styles:

* `bold`: turn on 'extra bright' mode.
* `reverse`: switch fore and background attribues
* `normal`: reset attribues to default
* `underline`: enable underline mode
* `no_underline`: disable underline mode