void median_filter_3x3_at_pos(global float *out, global float *img, 
	uint row, uint col, int width, int height)
{
	uint x = col;
	uint y = row;
	float buf[9];
	uint pos = 0;
	for (int dx = -1; dx <= 1; ++dx)
	{
		for (int dy = -1; dy <= 1; ++dy)
		{
			buf[pos++] = img[(x+dx) * height + (y+dy)];
		}
	}

	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8 - i; ++j)
		{
			if (buf[j] > buf[j+1])
			{
				float temp = buf[j];
				buf[j] = buf[j+1];
				buf[j+1] = temp;
			}
		}
	}	

	out[x * height + y] = buf[4];
}

kernel void median_filter_3x3(global float *img, global float *out,
	global int *width, global int *height)
{
	size_t work_width = get_global_size(0);
	size_t work_height = get_global_size(1);
	size_t col = get_global_id(0);
	size_t row = get_global_id(1);
	if (row == 0 || row == work_height - 1 ||
		col == 0 || col == work_width - 1)
	{
		return;
	}
	median_filter_3x3_at_pos(out, img, row, col, *width, *height);
}
