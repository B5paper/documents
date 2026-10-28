Preamble: indent folding also works before the first heading.
* Preamble list
    Preamble detail one
    Preamble detail two

# Chapter one
Normal paragraph.

* List A (this line is outside its indent fold)
    Detail A1
    Detail A2
    * Nested list (outside its own deeper fold)
        Nested detail one
        Nested detail two
    Detail A3

* List B
    Detail B1
    Detail B2

## Section one
Section paragraph.
* List C
    Detail C1
    Detail C2

```python
# This is code, not a chapter
def example():
    return 42
```



## Section two
No indentation here, but the section still folds.
More section text.

# Chapter two
* List D
    Detail D1
    Detail D2

### Skipped heading level
This is a child of Chapter two without an artificial intermediate section.

~~~text
# Still code
~~~

# Last chapter
Final paragraph.
