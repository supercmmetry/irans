R"(
/* Rans64 OpenCL implementation by Vishaal Selvaraj
 * SCALE: 24
 * MODEL SUPPORT: Zero-order only
 */

#define SCALE 24
	
__kernel void encode(
	__global unsigned char *input,
	const unsigned long int input_n,
	__global unsigned long int *ftable,
	__global unsigned long int *ctable,
	__global unsigned int *output,
	__global unsigned long int *output_ns,
	__global unsigned long int *input_residues,
	const unsigned long int output_size,
	const unsigned long int n,
	const unsigned long int stride_size
) {
	unsigned long int tid = get_global_id(0);
	if (tid >= n) {
		return;
	}

	unsigned long int input_start_index = tid * stride_size;
	unsigned long int input_end_index = input_start_index + stride_size - 1;
	
	if (input_end_index >= input_n) {
		input_end_index = input_n - 1;
	}
	
	unsigned long int input_size = input_end_index - input_start_index + 1;
	
	unsigned long int output_start_index = tid * output_size;
	unsigned long int output_end_index = output_start_index + output_size - 1;

	unsigned char *input_ptr = input + input_start_index;
	unsigned int *output_ptr = output + output_start_index;
	unsigned long int *output_ns_ptr = output_ns + tid;
	unsigned long int *input_residue_ptr = input_residues + tid;
	
	unsigned long int input_index = input_end_index;
	unsigned long int counter = 0;
	
	
	const unsigned long int lower_bound = 1 << 31;
	const unsigned long int up_prefix = (lower_bound >> SCALE) << 32;
	const unsigned long int mask = (1 << SCALE) - 1;
	
	unsigned long int state = lower_bound;
	unsigned long int state_counter = 0;
	
	while (true) {
		if (counter == input_size) {
			break;
		}
		
		unsigned char symbol = input_ptr[input_index];
		unsigned long int ls = ftable[symbol];
		unsigned long int bs = ctable[symbol];
		unsigned long int upper_bound = ls * up_prefix;
		
		if (state >= upper_bound) {
			output_ptr[state_counter] = state;
			state >>= 32;
			state_counter++;
		}
		
		state = ((state / ls) << SCALE) + bs + (state % ls);
		counter++;
		input_index--;
		
		if (state_counter == output_size - 2) {
			break;
		}
	}
	
	*input_residue_ptr = input_size - counter;
	
	output_ptr[state_counter] = state;
	output_ptr[state_counter + 1] = state >> 32;
	*output_ns_ptr = state_counter + 2;
}

)"