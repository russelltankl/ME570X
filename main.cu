#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
using namespace std;

#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		}																					\
}

inline int GetBlockSize(int b, int maxSize)
{
	if (b <= maxSize)
		return b;
	else
		return maxSize;
}

inline int GetGridSize(int n, int b)
{
	if (n%b == 0)
		return n / b;
	else
		return int(n*1.0 / (b*1.0)) + 1;
}

#define max(a,b)  (((a) > (b)) ? (a) : (b))

__global__ void FindMax4(float *in, float *out, int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	

	if (i < n && j < n)
	{
		// Output index
		int index = j * n + i;
		int z = ((n+1)*(n+1)) - (n*n);

		// Input index
		// Set up the input index correctly for the 4 inputs

		// Pattern 1
		int index1 = (i * 2) + (j * n * 4);
		int index2 = 1 + (i * 2) + (j * n * 4);
		int index3 = (i * 2) + (n * 2) + (j * n * 4);
		int index4 = 1 + (i * 2) + (n * 2) + (j * n * 4);

		// Pattern 2
		//int index1 = i + (j * n * 2); 
		//int index2 = i + n + (j * n * 2);
		//int index3 = i + (n * n * 2) + (j * n * 2);
		//int index4 = i + n + (n * n * 2) + (j * n * 2);

		// Compute the max of 4 values
		float max1 = max(in[index1], in[index2]);
		float max2 = max(in[index3], in[index4]);
		float max = max(max1, max2);
		out[index] = max;

		for (int h = 0; h < z; h++)
		{
			out[n * n + i*h] = 0;
		}
	}

	//if (i > z && i > z)
	//{
	//	int index = j * n + i;
	//	out[index] = 0;
	//}
}

//__global__ void NoneP2(float *in, float *out, int n)
//{
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	int j = blockIdx.y*blockDim.y + threadIdx.y;
//
//	if (i < n && j < n)
//	{
//		if (i < y && j < y)
//		{
//			int index1 =
//				int index2 =
//				int index3 =
//				int index4 =
//
//				float max1 = max(in[index1], in[index2]);
//			float max2 = max(in[index3], in[index4]);
//			float max = max(max1, max2);
//			out[index] = max;
//		}
//
//		if (i > (x - y) && j < y)
//		{
//			int index5 =
//				int index6 =
//
//				float max = max(in[index5], in[index6]);
//			out[index] = max;
//		}
//
//		if (i < y && j >(x - y))
//		{
//			int index7 =
//				int index8 =
//
//				float max = max(in[index7], in[index8]);
//			out[index] = max;
//		}
//	}
//
//}


void InitMatrix(float* a, int n, int m, int o)
{
	srand((int)time(NULL));
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
		{
			if (a[(o*o)-1] == (o*o)-1)
			{
				a[j*n + i] = 0;
			}
			else
			{
				a[j*n + i] = j*n + i;
			}
		}

	}

	
	// Use the code below for debugging if required
	//a[j*n + i] = j*n + i;
	//a[j*n + i] = float(10.0 * rand() / (RAND_MAX*1.0));

}

//void TrailingZeros(float* out, int p)
//{
//	//z is the difference of the upscaled matrix from the original
//	int z = (p*p) - ((p - 1)*(p - 1));
//	int index = ((p - 1)*(p - 1)) + z;
//	out[index] = 0;
//}

void PrintMatrix(float* a, int n, int m)
{
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
			cout << a[j*n + i] << " ";
		cout << endl;
	}
}
float FindMaxCPU(float* a, int n, int m)
{ 
	clock_t startTime = clock();
	float maxVal = 0;
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
		{
			maxVal = max(maxVal, a[j*n + i]);
		}
	}
	
	clock_t endTime = clock();
	clock_t clockTicksTaken = endTime - startTime;
	long double timeInMilliSeconds = (clockTicksTaken / (double)CLOCKS_PER_SEC)*(10 ^ 3);
	cout << "CPU Time in ms: " << timeInMilliSeconds << '\n';
	return maxVal;
}

void PaddingMatrix(float* a, int n, int m)
{
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
		{
			if (a[j*n + i] == (n - 1) + (n*j))
			{
				a[j*n + i] = 0;
			}
			else if (a[j*n + i] == (n*(n - 1)) + i)
			{
				a[j*n + i] = 0;
			}
		}
	}
}

int main()
{
	// Set size of the matrix
	int n = 35;
	int decoy = n;
	if (n % 2 != 0)
	{
		n = n + 1;
		cout << "Created a " << n << " x " << n << " Matrix from a " << decoy << " x " << decoy << " Matrix originally."<< endl;
	}
	else
	{
		cout << "Created a " << n << " x " << n << " Matrix." << endl;
	}

	// Create CPU Array
	float* matrix = new float[n*n];
	InitMatrix(matrix, n, n, decoy);

	// CPU max computation
	float maxVal = FindMaxCPU(matrix, n, n);
	cout << "Maximum value from CPU computation is : " << maxVal << endl;

	// Use the following code for print debugging
	cout << endl;
	PrintMatrix(matrix, n, n);
	cout << endl;

	// Init cuda event
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate GPU Memory
	float* matrix1CUDA;
	float* matrix2CUDA;
	//float* matrix3CUDA; //Reduction matrix
	cudaMalloc((void**)&(matrix1CUDA), n*n*sizeof(float));
	cudaMalloc((void**)&(matrix2CUDA), n*n*sizeof(float));
	cudaCheckError();

	// Copy GPU Memory
	cudaMemcpy(matrix1CUDA, matrix, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();

	// Setup swap of CUDA device pointers
	float* inputCUDA;
	float* outputCUDA;
	inputCUDA = (matrix1CUDA);
	outputCUDA = (matrix2CUDA);

	//cout << inputCUDA << endl;
	//cout << outputCUDA << endl;
	//cout << endl;

	// Run the Kernel
	cudaEventRecord(start);
	for (int p = n / 2; p >= 1; p = p / 2)
	{
		dim3 block(GetBlockSize(p, 16), GetBlockSize(p, 16), 1);
		dim3 grid(GetGridSize(p, block.x), GetGridSize(p, block.y), 1);
		//cudaEventRecord(start);
		FindMax4 << < grid, block >> >(inputCUDA, outputCUDA, p);
		//cudaThreadSynchronize();
		//cudaEventRecord(stop);

		//If resulting reduction matrix is an odd number, +1 to make it work for FindMax4 function (even numbers only)
		if ((p % 2 != 0) && (p != 1))
		{
			p = p++;
			cout << "Created a " << p << " x " << p << " Matrix from a " << p-1 << " x " << p-1 << " Matrix." <<endl;
		}
		cudaCheckError();

		// Use the following code for print debugging
#ifdef DEBUG
		//float* tempDataIn = new float[2 * p * 2 * p];
		//cudaMemcpy(tempDataIn, inputCUDA, 2 * p * 2 * p * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaCheckError();
		//PrintMatrix(tempDataIn, 2*p, 2*p);
		//cout << endl;
		//delete[] tempDataIn;

		float* tempDataOut = new float[p*p];
		cudaMemcpy(tempDataOut, outputCUDA, p * p * sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckError();
		PrintMatrix(tempDataOut, p, p);
		cout << endl;
		delete[] tempDataOut;
#endif

		// Swap input output pointers
		float* oldInputCUDA = inputCUDA;
		inputCUDA = outputCUDA;
		outputCUDA = oldInputCUDA;

	}
	//cudaThreadSynchronize();
	cudaEventRecord(stop);

	float maxValGPU;
	cudaMemcpy(&maxValGPU, inputCUDA, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Maximum value from GPU computation is : " << maxValGPU << endl;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU Time in ms: %f ms\n", milliseconds);

	cudaDeviceSynchronize();
	cudaCheckError();

	// Free the Memory
	cudaFree(matrix1CUDA);
	cudaFree(matrix2CUDA);
#ifdef DEBUG
	cudaCheckError();
#endif

	return 0;
}
