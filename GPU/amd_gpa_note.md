# AMD gpa note

## cache

* amd gpa usage

    example: initialize gpa

    `main.cpp`:

    ```cpp
    #include "gpu_performance_api/gpu_perf_api.h"
    #include "dlfcn.h"  // dlopen(), dlclose(), dlerr()
    #include <stdio.h>  // printf()
    #include <stdlib.h>  // exit()

    int main()
    {
        
        void *dl_X11 = dlopen("/usr/lib/x86_64-linux-gnu/libX11.so", RTLD_NOW | RTLD_GLOBAL);
        if (dl_X11 == NULL) {
            printf("fail to open libX11.so\n");
            const char *err_msg = dlerror();
            printf("error message:\n%s\n", err_msg);
            return -1;
        }

        void *dl_GPUPerfAPIGL = dlopen("/home/hlc/Softwares/amdgpa/3_15/Bin/Linx64/libGPUPerfAPIGL.so",
            RTLD_NOW);
        if (dl_GPUPerfAPIGL == NULL) {
            printf("fail to dlopen\n");
            const char *err_msg = dlerror();
            printf("error message:\n%s\n", err_msg);
            return -1;
        }
        dlclose(dl_X11);  // should close libX11 here?
        // guess: if we would not invoke x11 relative functions, closing dl_X11 here is OK.
        // and if we invoke some functions relative to X11 after closing the dl_X11,
        // then the program will be broken.

        GpaGetFuncTablePtrType GpaGetFuncTable = (GpaGetFuncTablePtrType) dlsym(dl_GPUPerfAPIGL, "GpaGetFuncTable");
        if (GpaGetFuncTable == NULL) {
            printf("fail to get func address of GpaGetFuncTable()\n");
            exit(-1);
        }
        printf("get func address: %p\n", GpaGetFuncTable);

        GpaFunctionTable gpa_func_table;
        GpaStatus status = GpaGetFuncTable(&gpa_func_table);
        if (status != GpaStatus::kGpaStatusOk)
        {
            printf("fail to gpa get functable GpaInitialize\n");
            return -1;
        }
        printf("successfully get gpa function table.\n");


        GpaInitializePtrType GpaInitialize = gpa_func_table.GpaInitialize;
        status = GpaInitialize(GpaInitializeBits::kGpaInitializeDefaultBit);
        if (status != kGpaStatusOk)
        {
            printf("fail to initialize gpa\n");
            return -1;
        }
        printf("successfully initialize gpa.\n");

        GpaDestroyPtrType GpaDestroy = gpa_func_table.GpaDestroy;
        dlclose(dl_GPUPerfAPIGL);
        return 0;
    }
    ```

    `Makefile`:

    ```Makefile
    main: main.cpp
        g++ -g main.cpp -I/home/hlc/Softwares/amdgpa/3_15/Include -o main
    ```