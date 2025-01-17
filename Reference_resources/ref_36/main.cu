#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cumem_hlc.h"

#define WARP_SIZE 32

template<int Size>
union BytePack;

template<>
union BytePack<0> {};

template<>
union BytePack<1> {
  uint8_t u8, native;
};

template<>
union BytePack<2> {
  BytePack<1> half[2];
  uint8_t u8[2];
  uint16_t u16, native;
};

template<>
union BytePack<4> {
  BytePack<2> half[2];
  uint8_t u8[4];
  uint16_t u16[2];
  uint32_t u32, native;
};

template<>
union BytePack<8> {
  BytePack<4> half[2];
  uint8_t u8[8];
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, native;
};

template<>
union alignas(16) BytePack<16>
{
    BytePack<8> half[2];
    uint8_t u8[16];
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
    ulong2 ul2, native;
};

template<typename T>
struct BytePackOf
{
    static constexpr int Size = sizeof(T);
    using Pack = BytePack<Size>;
};

template<>
struct BytePackOf<BytePack<0>>
{
    static constexpr int Size = 0;
    using Pack = BytePack<0>;
};

// toPack 和 fromPack 都是模板函数
// 这个看起来是个类型转换
// 跟强制类型转换相比，使用 union 进行类型转换有什么好处？
template<typename T>
__device__ typename BytePackOf<T>::Pack toPack(T value)
{
    union {
        typename BytePackOf<T>::Pack p;
        T v;
    };
    v = value;
    return p;
}

template<typename T>
__device__ T fromPack(typename BytePackOf<T>::Pack pack)
{
    union {
        typename BytePackOf<T>::Pack p;
        T v;
    };
    p = pack;
    return v;
}

// 这个是个空实现，拿来占位吗，还是后面有特化？
// 因为下面的 Apply_Reduce 模板类走的是偏特化，所以这里
// 看起来比较像类型的占位了
// 如果是占位，又为什么需要成员函数 FuncCopy 的实现？
// 完全可以这样写：
// template<typename T> struct FuncCopy;
// template<typename T> struct FuncSum;
// 这样也可以起到占位的作用
template<typename T>
struct FuncCopy
{
    using EltType = T;
    __device__ FuncCopy(uint64_t opArg=0) {}; 
};

template<typename T>
struct FuncSum
{
    using EltType = T;
    __device__ FuncSum(uint64_t opArg=0) {};
};

// 这里是个递归二分处理
// 让 a 的上半和 b 的上半相加，a 的下半和 b 的下半相加
// 再合并起来，返回 a
// 猜想的好处：
// 1. 不用考虑二分元素后剩下的是奇数个还是偶数个
// 2. 可以正常处理 128 位的数据
// 什么情况下会触发 half 的处理方式？
template<typename Fn, int EltPerPack>
struct Apply_Reduce
{
    template<int Size>
    __device__ static BytePack<Size> reduce(Fn fn, BytePack<Size> a, BytePack<Size> b)
    {
        a.half[0] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[0], b.half[0]);
        a.half[1] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[1], b.half[1]);
        return a;
    }
};

template<typename T>
struct Apply_Reduce<FuncCopy<T>, 1>
{
    __device__ static BytePack<sizeof(T)> reduce(FuncCopy<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b)
    {
        return a;
    }
};

template<typename T>
struct Apply_Reduce<FuncSum<T>, 1>
{
    __device__ static BytePack<sizeof(T)> reduce(
        FuncSum<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b)
    {
        return toPack<T>(fromPack<T>(a) + fromPack<T>(b));
    }
};

// 这里的 Pack 应该是 BytePack<size>
template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyReduce(Fn fn, Pack a, Pack b)
{
    return fromPack<Pack>(
        Apply_Reduce<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
            ::reduce(fn, toPack(a), toPack(b))
    );
}


template<typename T>
__device__ uintptr_t cvta_to_global(T* ptr) {
    return (uintptr_t)__cvta_generic_to_global(ptr);
}

template<int Size>
__device__ void st_global(uintptr_t addr, BytePack<Size> value);
template<int Size>
__device__ BytePack<Size> ld_volatile_global(uintptr_t addr);


template<>
__device__ void st_global<0>(uintptr_t addr, BytePack<0> value) {}

template<>
__device__ BytePack<0> ld_volatile_global<0>(uintptr_t addr)
{
    return {};
}

// template<>
// __device__ BytePack<1> ld_volatile_global<1>(uintptr_t addr) {
//     uint32_t tmp;
//     asm("ld.volatile.global.b8 %0, [%1];" : "=r"(tmp) : "l"(addr));
//     BytePack<1> ans;
//     ans.native = tmp;
//     return ans;
// }

// template<>
// __device__ BytePack<2> ld_volatile_global<2>(uintptr_t addr) {
//     uint16_t tmp;
//     asm("ld.volatile.global.b16 %0, [%1];" : "=h"(tmp) : "l"(addr));
//     BytePack<2> ans;
//     ans.native = tmp;
//     return ans;
// }

// template<>
// __device__ BytePack<4> ld_volatile_global<4>(uintptr_t addr) {
//     uint32_t tmp;
//     asm("ld.volatile.global.b32 %0, [%1];" : "=r"(tmp) : "l"(addr));
//     BytePack<4> ans;
//     ans.native = tmp;
//     return ans;
// }

// #define DEFINE_ld_st_16__space(space, addr_cxx_ty, addr_reg_ty) \
//   template<> \
//   __device__ __forceinline__ BytePack<16> ld_##space<16>(addr_cxx_ty addr) { \
//     BytePack<16> ans; \
//     asm("ld." #space ".v2.b64 {%0,%1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : #addr_reg_ty(addr)); \
//     return ans; \
//   } \
//   template<> \
//   __device__ __forceinline__ BytePack<16> ld_volatile_##space<16>(addr_cxx_ty addr) { \
//     BytePack<16> ans; \
//     asm("ld.volatile." #space ".v2.b64 {%0,%1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : #addr_reg_ty(addr)); \
//     return ans; \
//   } \
//   template<> \
//   __device__ __forceinline__ void st_##space<16>(addr_cxx_ty addr, BytePack<16> value) { \
//     asm("st." #space ".v2.b64 [%0], {%1,%2};" :: #addr_reg_ty(addr), "l"(value.u64[0]), "l"(value.u64[1]) : "memory"); \
//   }

#define DEFINE_ld_st_16__space(space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<16> ld_volatile_##space<16>(addr_cxx_ty addr) { \
    BytePack<16> ans; \
    asm("ld.volatile." #space ".v2.b64 {%0,%1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : #addr_reg_ty(addr)); \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<16>(addr_cxx_ty addr, BytePack<16> value) { \
    asm("st." #space ".v2.b64 [%0], {%1,%2};" :: #addr_reg_ty(addr), "l"(value.u64[0]), "l"(value.u64[1]) : "memory"); \
  }
DEFINE_ld_st_16__space(global, uintptr_t, l)
// DEFINE_ld_st_16__space(shared, uint32_t, r)
#undef DEFINE_ld_st_16

__device__ uint64_t ld_volatile_global(uint64_t *ptr) {
    uint64_t ans;
    asm("ld.volatile.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)));
    return ans;
}

#define DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_volatile_##space<bytes>(addr_cxx_ty addr) { \
    data_cxx_ty tmp; \
    asm("ld.volatile." #space "." #data_ptx_ty " %0, [%1];" : "="#data_reg_ty(tmp) : #addr_reg_ty(addr)); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<bytes>(addr_cxx_ty addr, BytePack<bytes> value) { \
    data_cxx_ty tmp = value.native; \
    asm volatile("st." #space "." #data_ptx_ty " [%0], %1;" :: #addr_reg_ty(addr), #data_reg_ty(tmp) : "memory"); \
  }

#define DEFINE_ld_st__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty) \
  DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, global, uintptr_t, l)
//   DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, shared, uint32_t, r) \
//   DEFINE_ld_st_gpu_relaxed__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty)

DEFINE_ld_st__size(1, uint32_t, b8, r)
DEFINE_ld_st__size(2, uint16_t, b16, h)
DEFINE_ld_st__size(4, uint32_t, b32, r)
DEFINE_ld_st__size(8, uint64_t, b64, l)


template<typename RedFn, typename T, int BytePerPack, int PreOpSrcs>
__device__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, T **srcPtrs,
    int nDsts, T **dstPtrs,
    size_t &nBytesBehind, size_t &nBytesAhead)
{
    // static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
    // if (BytePerPack == 0) __trap();

    // A hunk is the amount of contiguous data a warp consumes per loop iteration
    // assuming all threads partake.
    const int Unroll = 1;
    constexpr int BytePerHunk = Unroll * WARP_SIZE * BytePerPack;
    int nWarps = nThreads / WARP_SIZE;
    int warp = thread / WARP_SIZE;
    int lane = thread % WARP_SIZE;

    // This thread's initial position.
    size_t threadBytesBehind = nBytesBehind + (warp * BytePerHunk + lane * BytePerPack);
    size_t threadBytesAhead = nBytesAhead - (warp * BytePerHunk + lane * BytePerPack);
    // Number of hunks to be consumed over all warps.
    size_t nHunksAhead = nBytesAhead / (BytePerHunk + !BytePerHunk);
    // Advance collective position.
    nBytesBehind += nHunksAhead * BytePerHunk;
    nBytesAhead -= nHunksAhead * BytePerHunk;
    if (Unroll == 1 && BytePerPack <= nBytesAhead)
    {
        // Only Unroll=1 can do partial hunks (where not all threads partake).
        nHunksAhead += 1;
        nBytesBehind += nBytesAhead - (nBytesAhead % (BytePerPack + !BytePerPack));
        nBytesAhead = nBytesAhead % (BytePerPack + !BytePerPack);
    }
    nHunksAhead -= warp;

    RedFn redFn(redArg);
    const int MinSrcs = 2, MinDsts = 2;
    uintptr_t minSrcs[MinSrcs + !MinSrcs];
    uintptr_t minDsts[MinDsts + !MinDsts];
    for (int s = 0; s < MinSrcs; s++)
    {
        // 这里只是把地址取出来了，并不是值
        minSrcs[s] = cvta_to_global(srcPtrs[s]) + threadBytesBehind;
        // printf("srcPtrs[s]: %lu\n", srcPtrs[s]);
        // printf("minSrcs[s]: %lu\n", minSrcs[s]);
    }
    for (int d = 0; d < MinDsts; d++)
    {
        minDsts[d] = cvta_to_global(dstPtrs[d]) + threadBytesBehind;
    }

    const int MultimemSrcs = 0;
    // We dictate loop termination condition according to whether partial hunks
    // can be handled or not.
    while (Unroll == 1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead))
    {
        BytePack<BytePerPack> acc[Unroll];

        {
            RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
            #pragma unroll Unroll

            for (int u=0; u < Unroll; u++)
            {
                if (0 < MultimemSrcs)
                {
                    // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
                    // acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]);
                }
                else
                {
                    // Use volatile loads in case credits are polled for with volatile (instead of acquire).
                    acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
                    // if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
                }
                minSrcs[0] += WARP_SIZE * BytePerPack;
            }
        }

        // #pragma unroll (MinSrcs-1 + !(MinSrcs-1))
        // MinSrcs 至少得是 2 才会触发下面的 applyReduce()，非常合理
        for (int s = 1; s < MinSrcs; s++)
        {
            BytePack<BytePerPack> tmp[Unroll];
            RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
            #pragma unroll Unroll
            for (int u = 0; u < Unroll; u++)
            {
                if (s < MultimemSrcs) {
                  // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
                //   acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
                } else {
                    // Use volatile loads in case credits are polled for with volatile (instead of acquire).
                    tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
                }
                minSrcs[s] += WARP_SIZE * BytePerPack;
            }
            #pragma unroll Unroll
            for (int u=0; u < Unroll; u++)
            {
                // if (s < PreOpSrcs)
                //     tmp[u] = applyPreOp(preFn, tmp[u]);
                // 因为指定了特化 FuncSum，又只有一个元素，
                // 所以直接调用到了特化的函数
                // template<typename T>
                // struct Apply_Reduce<FuncSum<T>, 1>
                // 此时并不会触发 half
                // 问题：什么情况下 half 的处理方式会被触发？
                acc[u] = applyReduce(redFn, acc[u], tmp[u]);
            }
        }

        const int MaxSrcs = 1;
        for (int s = MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++)
        {
            uintptr_t src = cvta_to_global(srcPtrs[s]) + threadBytesBehind;
            BytePack<BytePerPack> tmp[Unroll];
            // RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
            #pragma unroll Unroll
            for (int u=0; u < Unroll; u++)
            {
                // Use volatile loads in case credits are polled for with volatile (instead of acquire).
                tmp[u] = ld_volatile_global<BytePerPack>(src);
                src += WARP_SIZE*BytePerPack;
            }
            #pragma unroll Unroll
            for (int u = 0; u < Unroll; u++)
            {
                // if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
                acc[u] = applyReduce(redFn, acc[u], tmp[u]);
            }
        }

        // postOp 应该是后处理的意思
        // if (postOp)
        // {
        //     #pragma unroll Unroll
        //     for (int u=0; u < Unroll; u++)
        //         acc[u] = applyPostOp(redFn, acc[u]);
        // }

        const int MultimemDsts = 0;
        #pragma unroll (MinDsts + !MinDsts)
        for (int d = 0; d < MinDsts; d++)
        {
            #pragma unroll Unroll
            for (int u = 0; u < Unroll; u++)
            {
                if (d < MultimemDsts)
                {
                    // multimem_st_global(minDsts[d], acc[u]);
                }
                else
                {
                    st_global<BytePerPack>(minDsts[d], acc[u]);
                }
                minDsts[d] += WARP_SIZE * BytePerPack;
            }
        }

        const int MaxDsts = 1;
        for (int d = MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++)
        {
            uintptr_t dst = cvta_to_global(dstPtrs[d]) + threadBytesBehind;
            #pragma unroll Unroll
            for (int u=0; u < Unroll; u++)
            {
                st_global<BytePerPack>(dst, acc[u]);
                dst += WARP_SIZE * BytePerPack;
            }
        }

        nWarps = nThreads / WARP_SIZE;
        #pragma unroll
        for (int s = 0; s < MinSrcs; s++)
            minSrcs[s] += (nWarps-1) * BytePerHunk;
        #pragma unroll
        for (int d = 0; d < MinDsts; d++)
            minDsts[d] += (nWarps-1)*BytePerHunk;
        threadBytesBehind += nWarps * BytePerHunk;
        threadBytesAhead -= nWarps * BytePerHunk;
        nHunksAhead -= nWarps;

        break;
    }

    nWarps = nThreads / WARP_SIZE;
    warp = thread / WARP_SIZE;
    lane = thread % WARP_SIZE;
    // The last loop iteration could have been partial, i.e. not taken by all
    // threads. The threads that weren't included need an extra subtraction to
    // make the value warp uniform.
    if (Unroll == 1 && nHunksAhead > 0) nHunksAhead -= nWarps;
    // Rotate warps so the warp which got the least work here will be warp 0.
    // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
    warp = -nHunksAhead;
    thread = warp * WARP_SIZE + lane;
}

template<int Unroll, typename RedFn, typename T, int PreOpSrcs>
__device__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, T **srcPtrs,
    int nDsts, T **dstPtrs,
    size_t nElts)
{
    // static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts, "Multimem pointers cannot exceed respective Min values.");
    //int nWarps = nThreads/WARP_SIZE;
    //int warp = thread/WARP_SIZE;
    int lane = thread % WARP_SIZE;
    // If a multimem src is present then our biggest pack size is limited to what
    // is supported for this redfn/type.
    // const int BigPackSize = 16;
    const int BigPackSize = 4;

    // if (MaxDsts==0) return;
    // if (MinDsts==0 && nDsts==0) return;

    // 看来是从头开始传输数据
    size_t nBytesBehind = 0;
    size_t nBytesAhead = nElts * sizeof(T);

    if (BigPackSize > sizeof(T))
    {
        // Check that all pointers are BigPackSize aligned.
        bool aligned = true;
        // srcPtrFn 是一个 lambda 表达式，返回数组指定索引的元素
        // return srcPtrs[i];
        if (lane < nSrcs)
            aligned &= 0 == cvta_to_global(srcPtrs[lane]) % (BigPackSize + !BigPackSize);
        if (lane < nDsts)
            aligned &= 0 == cvta_to_global(dstPtrs[lane]) % (BigPackSize + !BigPackSize);
        aligned = __all_sync(~0u, aligned);
        if (aligned)
        {
            // 按引用传递的数据：
            // thread, nBytesBehind, nBytesAhead
            // 之前这里传的 Unroll 应该是 4，下面那个 reduceCopyPacks 传的
            // Unroll 是 1
            // 因为目前只考虑功能，不考虑性能，所以我们统一按 1 处理了
            reduceCopyPacks<RedFn, T, BigPackSize, PreOpSrcs>(
                nThreads, thread,
                redArg, preOpArgs, postOp,
                nSrcs, srcPtrs,
                nDsts, dstPtrs,
                nBytesBehind, nBytesAhead
            );

            if (nBytesAhead == 0) return;

            // 这里传的 Unroll 是 1
            reduceCopyPacks<RedFn, T, BigPackSize, PreOpSrcs>(
                nThreads, thread,
                redArg, preOpArgs, postOp,
                nSrcs, srcPtrs,
                nDsts, dstPtrs,
                nBytesBehind, nBytesAhead
            );

            if (nBytesAhead == 0) return;
        }
    }

    // 这里传入的 Unroll 数据是 Unroll*(16/sizeof(T))/2，我们按 1 处理
    reduceCopyPacks<RedFn, T, sizeof(T), PreOpSrcs>(
        nThreads, thread,
        redArg, preOpArgs, postOp,
        nSrcs, srcPtrs,
        nDsts, dstPtrs,
        nBytesBehind, nBytesAhead
    );
    
    if (nBytesAhead == 0) return;

    // 这里传入的 Unroll 是 1
    reduceCopyPacks<RedFn, T, sizeof(T), PreOpSrcs>(
        nThreads, thread, redArg,
        preOpArgs, postOp,
        nSrcs, srcPtrs,
        nDsts, dstPtrs,
        nBytesBehind, nBytesAhead
    );
}

template<typename T>
__global__ void all_reduce_sum(T *cubuf_1, T *cubuf_2, int num_elm)
{
    int thd_id = threadIdx.x;
    int blk_id = blockIdx.x;
    size_t thread_id = blk_id * WARP_SIZE + thd_id;
    size_t num_threads = gridDim.x * WARP_SIZE;
    T* srcPtrs[2] = {
        cubuf_1,
        cubuf_2
    };
    T* dstPtrs[2] = {
        cubuf_1,
        cubuf_2
    };
    reduceCopy<4, FuncSum<T>, T, 0>(
        thread_id, num_threads,
        0, NULL, true,
        2, srcPtrs,
        2, dstPtrs,
        num_elm
    );
}

int main()
{
    using elm_type = float;
    elm_type *buf_1, *buf_2, *buf_3;
    elm_type *cubuf_1, *cubuf_2;
    int num_elm = 1024;
    
    sibling_alloc_buf_assign_rand_int<elm_type>(&buf_1, &cubuf_1, num_elm);
    sibling_alloc_buf_assign_rand_int<elm_type>(&buf_2, &cubuf_2, num_elm);
    buf_3 = (elm_type*) malloc(sizeof(elm_type) * num_elm);

    int num_disp_elm = 8;
    printf("the first %d elms:\n", num_disp_elm);
    printf("cubuf_1: ");
    print_cubuf(cubuf_1, num_disp_elm);
    printf("cubuf_2: ");
    print_cubuf(cubuf_2, num_disp_elm);

    // int nthreads = 256;
    if (num_elm % WARP_SIZE != 0)
    {
        printf("num_elm can't be divided by WARP_SIZE exactly\n");
        return -1;
    }
    int num_blocks = num_elm / WARP_SIZE;
    all_reduce_sum<elm_type><<<num_blocks, WARP_SIZE>>>(cubuf_1, cubuf_2, num_elm);
    cudaDeviceSynchronize();

    printf("after launch, the first %d elms:\n", num_disp_elm);
    printf("cubuf_1: ");
    print_cubuf(cubuf_1, num_disp_elm);
    printf("cubuf_2: ");
    print_cubuf(cubuf_2, num_disp_elm);

    for (int i = 0; i < num_elm; ++i)
        buf_3[i] = buf_1[i] + buf_2[i];

    bool all_equal = false;
    compare_buf_cubuf<elm_type>(buf_3, cubuf_1, num_elm, &all_equal);
    if (!all_equal)
        goto FREE_BUFS;
    compare_buf_cubuf<elm_type>(buf_3, cubuf_2, num_elm, &all_equal);
    if (!all_equal)
        goto FREE_BUFS;
    printf("all results are correct\n");

    FREE_BUFS:
    sibling_free_buf(buf_1, cubuf_1);
    sibling_free_buf(buf_2, cubuf_2);
    free(buf_3);
    return 0;
}
