# LeetCode Note (rust version)

### 树的先序遍历

```cpp
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;

struct MyStruc {
    ans: Rc<RefCell<Vec<i32>>>,
}

impl MyStruc {
    fn new() -> MyStruc {
        MyStruc {
            ans: Rc::new(RefCell::new(Vec::<i32>::new())),
        }
    }

    fn dfs(&self, r: &Option<Rc<RefCell<TreeNode>>>) {
        match r {
            None => {
                return;
            },
            Some(root) => {
                self.ans.borrow_mut().push(root.borrow().val);
                self.dfs(&root.borrow().left);
                self.dfs(&root.borrow().right);
            }
        }
    }
}

impl Solution {
    pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let my_struc = MyStruc::new();
        my_struc.dfs(&root);
        return my_struc.ans.borrow().to_owned();
    }
}
```

### 反转链表

1. 我写的

    ```rust
    impl Solution {
        fn new() -> Solution {
            Solution {}
        }

        pub fn reverse_list(&self, head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
            let mut pre = None;
            let mut cur = head.clone();
            let mut next = head.clone().unwrap().next.clone();
            while let Some(mut cur_node) = cur {
                cur_node.next = pre;
                pre = Some(cur_node);
                cur = next;
                if cur.is_some() {
                    next = cur.clone().unwrap().next;
                } else {
                    next = None;
                }
            }
            pre.to_owned()
        }
    }
    ```

    用过一次（在等号右边出现过一次，或者`unwrap()`过一次）的变量就不能再用了。如果需要多次用到某个变量，直接`clone()`就可以了。

    不能在循环内部申请变量，然后把这个变量赋给某个引用。因为第一轮循环结束时，申请的变量会被 drop 掉，然后第二轮循环引用会变成无效，这样的情况编译器是不允许的。

    能用`Option`就别硬拿里面的东西了，因为里面的指针也是指向`Option`对象，最后还是得处理`Option`。

    别用`&mut Option`或`& Option`，因为我们最终还是要把`Box<ListNode>`指向`Option`对象。为了从`&`得到`Option`，我们需要在循环中创建`Option`对象，可是当一轮循环结束，对象会被释放，导致引用失效。

    可以在等号右边`clone()`，但不要在等号左边`clone`，因为等号左边`clone()`修改的就不再是原变量了。补充：右侧的`clone()`也不能乱用。`clone()`是一份拷贝，不是一份引用。

    `clone()`是 deepcopy。要想明白再用，不要随便用。

1. 别人写的

    ```rust
    impl Solution {
        pub fn reverse_list(&self, head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
            let mut pre = None;
            let mut head = head;
            while let Some(mut node) = head {
                head = node.next.take();
                node.next = pre;
                pre = Some(node);
            }
            pre
        }
    }
    ```

