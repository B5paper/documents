kernel void gsum(global float *arr, global float *output, const int arr_len,
    local float *loc_arr)
{
    size_t glb_size = get_global_size(0);
    size_t loc_id = get_local_id(0);
    loc_arr[loc_id] = arr[loc_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 1; i < glb_size; i *= 2)
    {
        if (loc_id % (2 * i) == 0)
        {
            loc_arr[loc_id] += loc_arr[loc_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (loc_id == 0)
    {
        *output = loc_arr[0];
    }
}