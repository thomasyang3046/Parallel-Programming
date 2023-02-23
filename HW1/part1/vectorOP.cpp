#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float* values, float* output, int N)
{
	__pp_vec_float x;
	__pp_vec_float result;
	__pp_vec_float zero = _pp_vset_float(0.f);
	__pp_mask maskAll, maskIsNegative, maskIsNotNegative;

	//  Note: Take a careful look at this loop indexing.  This example
	//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
	//  Why is that the case?
	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{

		// All ones
		maskAll = _pp_init_ones();

		// All zeros
		maskIsNegative = _pp_init_ones(0);

		// Load vector of values from contiguous memory addresses
		_pp_vload_float(x, values + i, maskAll); // x = values[i];

		// Set mask according to predicate
		_pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

		// Execute instruction using mask ("if" clause)
		_pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

		// Inverse maskIsNegative to generate "else" mask
		maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

		// Execute instruction ("else" clause)
		_pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

		// Write results back to memory
		_pp_vstore_float(output + i, result, maskAll);
	}
}

void clampedExpVector(float* values, int* exponents, float* output, int N)
{
	// PP STUDENTS TODO: Implement your vectorized version of
	// clampedExpSerial() here.
	// Your solution should work for any value of
	// N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{
		__pp_mask maskAll, maskIsNegative, maskIsNotNegative, maskislarger;
		__pp_vec_float x;// Declare a floating point vector register with __pp_vec_float
		__pp_vec_int y;// Declare a floating point vector register with __pp_vec_float
		__pp_vec_float result;// Declare a floating point vector register with __pp_vec_float
		__pp_vec_float one = _pp_vset_float(1.f);
		__pp_vec_int ones = _pp_vset_int(1);
		__pp_vec_float maxin = _pp_vset_float(9.999999f);
		__pp_vec_int zero = _pp_vset_int(0);
		__pp_vec_int count = _pp_vset_int(0);
		int compute;
		if (i + VECTOR_WIDTH > N)//(0+4>3   12+4>13(1) 14(1) 15(1) 16(2))
		{
			maskAll = _pp_init_ones(N % VECTOR_WIDTH);
			compute = N % VECTOR_WIDTH;
		}
		else
		{
			maskAll = _pp_init_ones();
			compute = VECTOR_WIDTH;
		}
		maskIsNegative = _pp_init_ones(0);//看y是否等於0
		_pp_vload_float(x, values + i, maskAll); // x = values[i];
		_pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];
		_pp_veq_int(maskIsNegative, zero, y, maskAll); //if (y==0) maskIsNegative=1, else maskIsnegative=0
		_pp_vmove_float(result, one, maskIsNegative);//把1放到result上
		maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {
		_pp_vload_float(result, values + i, maskIsNotNegative); // result = values[i];
		_pp_vsub_int(count, y, ones, maskIsNotNegative);//count=y-1
		__pp_mask maskForNegative, maskForNotNegative;
		maskForNegative = _pp_init_ones(0);
		while (true)
		{
			_pp_veq_int(maskForNegative, zero, count, maskAll);//抓出是否有exp=0 maskForNegative=1代表某個是0
			if (_pp_cntbits(maskForNegative) == compute)//代表都算完了
				break;
			maskForNotNegative = _pp_mask_not(maskForNegative);// if exp!=0 
			_pp_vmult_float(result, result, x, maskForNotNegative);
			_pp_vsub_int(count, count, ones, maskForNotNegative);
		}
		_pp_vlt_float(maskislarger, maxin, result, maskAll); //if result[i]>9.9999f maskislarger=1
		_pp_vmove_float(result, maxin, maskislarger);//result[i]=9.9999f
		_pp_vstore_float(output + i, result, maskAll);
	}
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N)
{
	//
	// PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
	//
	__pp_mask maskAll = _pp_init_ones();
	__pp_vec_float x, result;
	float sum = 0;
	_pp_vset_float(result, 0.f, maskAll);
	for (int i = 0;i < N;i += VECTOR_WIDTH)
	{
		_pp_vload_float(x, values + i, maskAll);
		_pp_vadd_float(result, result, x, maskAll);//result=(1,2,3,4)
	}
	_pp_hadd_float(result, result);//(3 3 7 7)
	__pp_vec_float buffer;
	for (int j = 2;j <= log2(VECTOR_WIDTH);j++)
	{
		_pp_interleave_float(buffer, result);//(3 7 3 7)
		_pp_hadd_float(result, buffer);//(10 10 10 10)
	}
	sum += result.value[0];
	return sum;
}