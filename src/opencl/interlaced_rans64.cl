R"(
/* Rans64 OpenCL implementation by Vishaal Selvaraj
 * SCALE: 24
 * MODEL SUPPORT: Zero-order only
 */

#define SCALE 24
#define u64 unsigned long int
#define u8 unsigned char
#define u32 unsigned int
	
__kernel void encode(
	__global u8 *input,
	const u64 input_n,
	__global u64 *ftable,
	__global u64 *ctable,
	__global u32 *output,
	__global u64 *output_ns,
	__global u64 *input_residues,
	const u64 output_size,
	const u64 n,
	const u64 stride_size
) {
	u64 tid = get_global_id(0);
	if (tid >= n) {
		return;
	}

	u64 input_start_index = tid * stride_size;
	u64 input_end_index = input_start_index + stride_size - 1;
	
	if (input_end_index >= input_n) {
		input_end_index = input_n - 1;
	}
	
	u64 input_size = input_end_index - input_start_index + 1;
	
	u64 output_unit_size = stride_size >> 2;
	u64 output_start_index = tid * output_unit_size;
	u64 output_end_index = output_start_index + output_unit_size - 1;

	u32 *output_ptr = output + output_start_index;
	u64 *output_ns_ptr = output_ns + tid;
	u64 *input_residue_ptr = input_residues + tid;
	
	u64 input_index = input_end_index;
	u64 counter = 0;
	
	
	const u64 lower_bound = 1 << 31;
	const u64 up_prefix = (lower_bound >> SCALE) << 32;
	const u64 mask = (1 << SCALE) - 1;
	
	u64 state = lower_bound;
	u64 state_counter = 0;
	
	while (true) {
		if (counter == input_size) {
			break;
		}
		
		u8 symbol = input[input_index];
		u64 ls = ftable[symbol];
		u64 bs = ctable[symbol];
		u64 upper_bound = ls * up_prefix;
		
		if (state >= upper_bound) {
			output_ptr[state_counter] = state;
			state >>= 32;
			state_counter++;
		}
		
		state = ((state / ls) << SCALE) + bs + (state % ls);
		counter++;
		input_index--;
		
		if (state_counter == output_unit_size - 2) {
			break;
		}
	}
	
	*input_residue_ptr = input_size - counter;
	
	output_ptr[state_counter] = state;
	output_ptr[state_counter + 1] = state >> 32;
	*output_ns_ptr = state_counter + 2;
}



u8 inv_bs(u64 *ctable, u64 bs) {
	u8 symbol = 0xff;
	
	for (int i = 0; i < 0x100; i++) {
		if (ctable[i] > bs) {
			symbol = i - 1;
			break;
		}
	}
	
	return symbol;
}


__kernel void decode(
	__global u8 *input,
	const u64 input_n,
	__global u64 *ftable,
	__global u64 *ctable,
	__global u32 *output,
	__global u64 *output_ns,
	__global u64 *input_residues,
	const u64 output_size,
	const u64 n,
	const u64 stride_size
) {
	u64 tid = get_global_id(0);
	if (tid >= n) {
		return;
	}

	u64 input_start_index = tid * stride_size;
	u64 input_end_index = input_start_index + stride_size - 1;
	
	if (input_end_index >= input_n) {
		input_end_index = input_n - 1;
	}
	
	u64 input_size = input_end_index - input_start_index + 1;
	
	u64 output_unit_size = stride_size >> 2;
	u64 output_start_index = tid * output_unit_size;
	u64 output_end_index = output_start_index + output_ns[tid] - 1;

	u32 *output_ptr = output + output_start_index;
	u64 *output_ns_ptr = output_ns + tid;
	u64 *input_residue_ptr = input_residues + tid;
	
	u64 input_index = input_start_index + *input_residue_ptr;
	input_size = input_size - *input_residue_ptr;
	u64 counter = 0;
	
	
	const u64 lower_bound = 1 << 31;
	const u64 up_prefix = (lower_bound >> SCALE) << 32;
	const u64 mask = (1 << SCALE) - 1;
	
	u64 state = output[output_end_index];
	state = (state << 32) | output[output_end_index - 1];
	u64 state_counter = output_end_index - 2;
	
	while (true) {
		if (counter == input_size) {
			break;
		}
		
		u64 bs = state & mask;
		u8 symbol = inv_bs(ctable, bs);
		
		input[input_index] = symbol;
		u64 ls = ftable[symbol];
		bs = ctable[symbol];
		
		state = (ls * (state >> SCALE)) + (state & mask) - bs;
		
		if (state < lower_bound) {
			state = (state << 32) | output[state_counter];
			state_counter--;
		}
		
		counter++;
		input_index++;
	}
}

)"