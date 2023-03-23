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

    后来又写的：

    ```cpp
    impl Solution {
        pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
            if head.clone().unwrap().next.is_none() {
                return head;
            }

            let mut pre = None;
            let mut cur = head;
            let mut next = cur.clone().unwrap().next;
            while cur.is_some() {
                cur.as_mut().unwrap().next = pre;
                pre = cur;
                cur = next;
                if cur.is_some() {
                    let a = cur.as_mut();
                    let b = a.unwrap();
                    let c = b.to_owned().next;
                    next = c;
                } else {
                    next = None;
                }
            }
            pre
        }
    }
    ```

    后来又写的（这个效率很低很低，接近个位数了）：

    ```rust
    fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut pre = None;
        let mut cur = head;
        let mut nex = cur.as_mut().unwrap().to_owned().next;
        while cur.is_some() {
            cur.as_mut()?.next = pre;
            pre = cur;
            cur = nex;
            nex = match cur.as_mut() {
                Some(cur_node) => {
                    cur_node.to_owned().next
                },
                None => None
            }
        }
        pre
    }
    ```

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

### 遍历链表

```rust
fn traverse(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut p = &head;
    while let Some(pnode) = p {
        print!("{}, ", pnode.val);
        p = &pnode.next;
    }
    head
}
```

```rust
fn traverse(head: &mut Option<Box<ListNode>>) {
    let mut p = head.clone();
    while let Some(pnode) = p {
        print!("{}, ", pnode.val);
        p = pnode.next;
    }
}
```

后来写的：

```rust
fn traverse(head: &mut Option<Box<ListNode>>) {
    let mut p = head;
    while p.is_some() {
        print!("{}, ", p.as_mut().unwrap().val);
        p = &mut p.as_mut().unwrap().next;
    }
}
```

### 无重复字符的最长子串

给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
 

提示：

0 <= s.length <= 5 * 104
s 由英文字母、数字、符号和空格组成

代码：

1. 哈希表 + 滑动窗口

```rust
use std::collections::*;
use std::cmp::*;

impl Solution {
    pub fn length_of_longest_substring(s: String) -> i32 {
        if s.is_empty() {
            return 0;
        }
        let mut ans: i32 = 1;
        let mut left: i32 = 0;
        let mut right: i32 = 0;
        let mut occ = HashSet::<u8>::new();
        let bs = s.as_bytes();
        while right < s.len() as i32 {
            match occ.get(&bs[right as usize]) {
                Some(_) => {
                    occ.remove(&bs[left as usize]);
                    left += 1;
                },
                None => {
                    occ.insert(bs[right as usize]);
                    right += 1;
                    ans = max(ans, right - left);
                }
            }
        }
        ans
    }
}
```

### 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
 

示例 1：

输入：s = "()"
输出：true
示例 2：

输入：s = "()[]{}"
输出：true
示例 3：

输入：s = "(]"
输出：false
 

提示：

1 <= s.length <= 104
s 仅由括号 '()[]{}' 组成

代码：

1. 栈

    ```rust
    impl Solution {
        pub fn is_valid(s: String) -> bool {
            let bs = s.as_bytes();
            let mut stk: Vec<u8> = Vec::new();
            for c in bs.iter() {
                if c == &b'(' || c == &b'[' || c == &b'{' {
                    stk.push(*c);
                    continue;
                }
                if stk.is_empty() {
                    return false;
                }
                if stk.last().unwrap() == &b'(' && c != &b')' {
                    return false;
                } else if stk.last().unwrap() == &b'[' && c != &b']' {
                    return false;
                } else if stk.last().unwrap() == &b'{' && c != &b'}' {
                    return false;
                }
                stk.pop();
            }
            if !stk.is_empty() {
                return false;
            }
            return true;
        }
    }
    ```

### 快速选择（top-k）

```rust
use rand::{self, Rng};

fn sort(vec: &mut Vec<i32>, left: i32, right: i32, idx: i32) -> i32 {
    let n = right - left + 1;
    let pivot_num = vec[idx as usize];
    vec.swap(idx as usize, right as usize);
    let mut i = left;
    let mut j = right - 1;
    while i < j {
        while i < j && vec[i as usize] < pivot_num {
            i += 1;
        }
        while i < j && vec[j as usize] >= pivot_num {
            j -= 1;
        }
        if vec[i as usize] > vec[j as usize] {
            vec.swap(i as usize, j as usize);
        }
    }
    if vec[i as usize] > pivot_num {
        vec.swap(i as usize, right as usize);
    }
    i
}

fn partition(vec: &mut Vec<i32>, left: i32, right: i32, k: i32) {
    if right - left + 1 <= 1 {
        return;
    }
    let mut rng = rand::thread_rng();
    let mut idx:i32 = rng.gen_range(left..=right);
    let mut idx = sort(vec, left, right, idx);
    println!("{left}, {right}, {idx}, {:?}", vec);
    
    if idx < k - 1 {
        partition(vec, idx, right, k);
    } else if idx > k - 1 {
        partition(vec, left, idx, k);
    } else {
        return;
    }
}

fn get_top_k(vec: &mut Vec<i32>, k: i32) -> Vec<i32> {
    partition(vec, 0, vec.len() as i32 - 1, k);
    let mut ans: Vec<i32> = Vec::<i32>::new();
    ans.resize(k as usize, 0);
    let mut i: i32 = 0;
    while i < k {
        ans[i as usize] = vec[i as usize];
        i += 1;
    }
    ans
}

fn main() {
    let mut vec = vec![5, 3, 2, 1, 4];
    let mut k = 3;
    let ans = get_top_k(&mut vec, k);
    println!("{:?}", ans);
}

```

### 最长公共前缀

代码：

1. 我写的

    ```rust
    use std::cmp::*;
    impl Solution {
        pub fn longest_common_prefix(strs: Vec<String>) -> String {
            let mut strs = strs.clone();
            strs.sort_by(|a, b| if a.len() <= b.len() {
                Ordering::Less
            } else {
                Ordering::Greater
            });
            let mut p = 0;
            let shortest_len = strs[0].len() as i32;
            let n = strs.len();
            let mut i = 0;
            let mut ch: u8;
            let mut ans = String::new();
            while p < shortest_len {
                ch = strs[0].as_bytes()[p as usize];
                i = 1;
                while i < n {
                    let s = strs[i as usize].as_bytes();
                    if s[p as usize] != ch {
                        return ans;
                    }
                    i += 1;
                }
                ans.push(char::from(ch));
                p += 1;
            }
            return ans;
        }
    }
    ```

1. 官方答案

    ```rust
    impl Solution {
        pub fn longest_common_prefix(strs: Vec<String>) -> String {
            if strs.len() == 0 {
                return String::from("");
            } else if strs.len() == 1 {
                return strs[0].clone();
            }
            let mut i = 0;
            let mut is_end = false;
            while !is_end {
                for j in 0..strs.len() {
                    if strs[j].len() <= i {
                        is_end = true;
                        break;
                    }
                    if j == 0 {
                        continue;
                    }
                    if strs[j][i..i + 1] != strs[0][i..i + 1] {
                        is_end = true;
                        break;
                    }
                }
                i += 1;
            }
            return strs[0][0..i-1].to_string();
        }
    }
    ```

### 全排列

```rust
struct GetAns {
    ans: Vec<Vec<i32>>,
    temp: Vec<i32>,
    vis: Vec<bool>
}

impl GetAns {
    fn backtrack(&mut self, nums: &Vec<i32>) {
        let n = nums.len();
        if self.temp.len() == n {
            self.ans.push(self.temp.clone());
            return;
        }

        for i in 0..n {
            if self.vis[i] {
                continue;
            }
            self.vis[i] = true;
            self.temp.push(nums[i]);
            self.backtrack(&nums);
            self.temp.pop();
            self.vis[i] = false;
        }
    }
}

impl Solution {
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut ans_obj = GetAns {
            ans: Vec::new(),
            temp: Vec::new(),
            vis: Vec::new(),
        };
        ans_obj.vis.resize_with(nums.len(), || false);
        ans_obj.backtrack(&nums);
        ans_obj.ans
    }
}
```

### 单词搜索

代码：

```rust
struct GetAns<'a> {
    ans: bool,
    vis: Vec<Vec<bool>>,
    word: &'a [u8]
}

impl<'a> GetAns<'a> {
    fn backtrack(&mut self, board: &Vec<Vec<char>>, x: i32, y: i32, word: &String, pos: i32) {
        if self.ans == true {
            return;
        }

        if x < 0 || y < 0 || x >= board.len() as i32 || y >= board[0].len() as i32 {
            return;
        }

        if self.vis[x as usize][y as usize] {
            return;
        }
        
        if board[x as usize][y as usize] as u8 != self.word[pos as usize] {
            return;
        }

        if pos == word.len() as i32 - 1 {
            self.ans = true;
            return;
        }

        self.vis[x as usize][y as usize] = true;
        self.backtrack(board, x + 1, y, word, pos + 1);
        self.backtrack(board, x - 1, y, word, pos + 1);
        self.backtrack(board, x, y + 1, word, pos + 1);
        self.backtrack(board, x, y - 1, word, pos + 1);
        self.vis[x as usize][y as usize] = false;
    }

    fn get_ans<'b: 'a>(&mut self, board: Vec<Vec<char>>, word: &'b String) -> bool {
        self.word = word.as_bytes();
        let m = board.len();
        let n = board[0].len();
        for i in 0..m {
            let mut v = Vec::new();
            v.resize(n, false);
            self.vis.push(v);
        }
        for i in 0..(m as i32) {
            for j in 0..(n as i32) {
                self.backtrack(&board, i, j, &word, 0);
                if self.ans {
                    break;
                }
            }
            if self.ans {
                break;
            }
        }
        
        self.ans
    }
}

struct Solution {

}

impl Solution {
    pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
        let mut ans_obj = GetAns {
            ans: false,
            vis: Vec::new(),
            word: word.as_bytes(),
        };
        ans_obj.get_ans(board, &word);
        ans_obj.ans
    }
}
```