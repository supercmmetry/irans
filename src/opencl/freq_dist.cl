R"(
  
  
  __kernel void run(
	  __global unsigned char *arr,
	  __global unsigned long int *out,
	  const unsigned long int n,
	  const unsigned long int s,
	  const unsigned long int arr_size
) {
	unsigned long int tid = get_global_id(0);
	if (tid >= n) {
		return;
	}
	
	unsigned long int out_offset = 256 * tid;
	unsigned long int start_index = s * tid;
	unsigned long int end_index = start_index + s - 1;
	
	for (unsigned long int i = start_index; (i <= end_index) && (i < arr_size); i++) {
		out[out_offset + arr[i]]++;
	}
} 
  
)"