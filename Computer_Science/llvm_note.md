# LLVM Note

## cache

* llvm hello world

    `main.cpp`:

    ```cpp
    int main()
    {
        return 42;
    }
    ```

    执行`clang++-11 -S -emit-llvm ./main.cpp`，会生成

    `main.ll`:

    ```
    ; ModuleID = './main.cpp'
    source_filename = "./main.cpp"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-pc-linux-gnu"

    ; Function Attrs: noinline norecurse nounwind optnone uwtable
    define dso_local i32 @main() #0 {
      %1 = alloca i32, align 4
      store i32 0, i32* %1, align 4
      ret i32 42
    }

    attributes #0 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

    !llvm.module.flags = !{!0}
    !llvm.ident = !{!1}

    !0 = !{i32 1, !"wchar_size", i32 4}
    !1 = !{!"Ubuntu clang version 11.1.0-6"}

    ```

    执行`clang++-11 -o main main.ll`，可生成可执行程序`main`。

    运行`./main`，再执行`echo $?`，可以看到 main 程序的返回值为 42。

    执行`lli main.ll`，同样也可以生成 binary。

* How to uninstall LLVM?

    `cd` to the LLVM build directory, then

    ```bash
    xargs rm < install_manifest.txt
    ```

## note