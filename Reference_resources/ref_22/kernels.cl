kernel void sum_1(global float *arr, global float *out, const int len)
{
    size_t glb_id = get_global_id(0);
    size_t work_span = 4;
    float s = arr[glb_id * work_span];
    for (int i = 1; i < work_span; ++i)
    {
        s += arr[glb_id * work_span + i];
    }
    arr[glb_id * work_span] = s;
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int span_len = work_span * 2; span_len <= len; span_len *= 2)
    {
        if ((glb_id * work_span) % span_len == 0)
        {
            arr[glb_id * work_span] += arr[glb_id * work_span + span_len / 2];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    
    if (glb_id == 0)
        *out = arr[0];
}


kernel void sum_2(global float *arr, global float *out, const int len)
{
    size_t glb_id = get_global_id(0);
    for (int i = 0; i < 4; ++i)
    {
        for (int span_len = 2; span_len <= 256; span_len *= 2)
        {
            if ((glb_id + i * 256) % span_len == 0)
            {
                arr[glb_id + i * 256] += arr[(glb_id + i * 256) + span_len / 2];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }

    int max_work_size = 2;
    for (int span_len = 512; span_len <= len; span_len *= 2)
    {
        if (glb_id < max_work_size)
        {
            arr[glb_id * span_len] += arr[glb_id * span_len + span_len / 2];
        }
        max_work_size /= 2;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (glb_id == 0)
        *out = arr[0];
}


