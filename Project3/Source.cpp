#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <stdlib.h>
#include <string>
#include <CL/cl.h>
#include <CL/cl.hpp>
#include<fstream>
#include <time.h>

#define __CL_ENABLE_EXCEPTIONS

#define row 4096*2// 1024//4096
#define col 4096*2//1024//4096
#define LOCAL_SIZE 512

using namespace std;

int *matrix_A;
int *matrix_B;
int *result;
float runTime = 0;

// initilization of matrix.
void randomMemInit(int* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() % 100 + 1;
}

//cpu execution
void PerformCalculationOnHost() {
	float tmp;
	for (int row_A = 0; row_A < row; row_A++) {
		tmp = 0;
		for (int col_A = 0; col_A < row; col_A++) {
			tmp += matrix_A[row_A * row + col_A] * matrix_B[col_A];
		}
		result[row_A] = tmp;
	}
}

//AMD
void PerformCalculationOnDevice(cl::Device device) {
	clock_t start_t, end_t;
	start_t = clock();
	vector<cl::Device> contextDevices;
	contextDevices.push_back(device);
	cl::Context context(contextDevices);

	cl::CommandQueue queue(context, device);

	std::fill_n(result, row, 0);

	cl::Buffer cl_matrix_A = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, row * col * sizeof(int), matrix_A);
	cl::Buffer cl_vector_B = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, col * sizeof(int), matrix_B);
	cl::Buffer cl_result_vector = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, col * sizeof(int), result);
	
	std::ifstream sourceFile("kernal.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

	start_t = clock();
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	cl::Program program = cl::Program(context, source);
	program.build(contextDevices);
	cl::Kernel kernel(program, "matrixVectorMul");

	int iArg = 0;
	kernel.setArg(iArg++, cl_result_vector);
	kernel.setArg(iArg++, cl_matrix_A);
	kernel.setArg(iArg++, cl_vector_B);
	kernel.setArg(iArg++, col);

	
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(row), cl::NDRange(row));
	queue.finish();

	queue.enqueueReadBuffer(cl_result_vector, CL_TRUE, 0, col * sizeof(int), result);
	end_t = clock();
	runTime += end_t - start_t;
}


//intel gpu
long LoadOpenCLKernel(char const* path, char **buf)
{
	FILE  *fp;
	size_t fsize;
	long   off_end;
	int    rc;

	/* Open the file */
	fp = fopen(path, "r");
	if (NULL == fp) {
		return -1L;
	}

	/* Seek to  */
	rc = fseek(fp, 0L, SEEK_END);
	if (0 != rc) {
		return -1L;
	}

	/* Byte offset to the end of the file (size) */
	if (0 > (off_end = ftell(fp))) { //points end of the file
		return -1L;
	}
	fsize = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	*buf = (char *)malloc(fsize + 1);
	if (NULL == *buf) {
		return -1L;
	}

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Slurp file into buffer */
	if (fsize != fread(*buf, 1, fsize, fp)) {
		free(*buf);
		return -1L;
	}

	/* Close the file */
	if (EOF == fclose(fp)) {
		free(*buf);
		return -1L;
	}


	/* Make sure the buffer is NUL-terminated, just in case */
	(*buf)[fsize] = '\0';

	/* Return the file size */
	return (long)fsize;
}

void mult()
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j <row; j++)
		{
			result[i] = 0;
			for (int x = 0; x < row; x++)
			{
				*((result + i) + j) += *((matrix_A + i) + x) *
					*((matrix_B + x) + j);
			}
		}
	}
}

int main()
{
	clock_t start_t, end_t, start_t1, end_t2;
	int err;                            // error code returned from api calls

	cl_device_id device_id;             // compute device id 
	cl_context context;                 // buffer
	cl_command_queue commands;          // queue of commands
	cl_program program;                 // Create a program from a vector of source strings
	cl_kernel kernel;                   // Returns a reference to the underlying OpenCL kernel object.

	 // OpenCL device memory for matrices
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	// set seed for rand()
	srand(0);

	//Allocate host memory for matrices A and B
	unsigned int size_A = row * col;
	unsigned int mem_size_A = sizeof(int) * size_A;
	matrix_A = (int*)malloc(mem_size_A);

	unsigned int size_B = row * row;
	unsigned int mem_size_B = sizeof(int) * size_B;
	matrix_B = (int*)malloc(mem_size_B);

	//Initialize host memory
	randomMemInit(matrix_A, size_A);
	randomMemInit(matrix_B, size_B);

	//Allocate host memory for the result C
	unsigned int size_C = row * col;
	unsigned int mem_size_C = sizeof(int) * size_C;
	result = (int*)malloc(mem_size_C);

	printf("Initializing OpenCL device...\n");
	start_t1 = clock();
	
	cl_uint dev_cnt = 0;  //clockFrequency
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[2];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	// Connect to a compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU , 1, &device_id, NULL);

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, 0, &err);

	// Create the compute program from the source file
	char *KernelSource;
	long lFileSize;

	lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource);

	program = clCreateProgramWithSource(context, 1, (const char **)& KernelSource, NULL, &err);

	start_t = clock();
	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	//
	kernel = clCreateKernel(program, "matrixMul", &err);

	// Create the input and output arrays in device memory for our calculation
	d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, matrix_A, &err);
	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, matrix_B, &err);

	//Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

	int wA = row;
	int wC = row;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);


	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = 1024;
	globalWorkSize[1] = 1024;

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);   //executes kernel

	//Retrieve result from device
	err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, result, 0, NULL, NULL);
	//print out the results

	end_t = clock();
	end_t2 = clock();

	std::cout << "\nTime Taken By Interl GPU " << (float)(end_t - start_t) / CLOCKS_PER_SEC << " seconds" << endl;
	std::cout << "Device load and link time of Intel GPU: " << (float)(end_t2 - start_t1) / CLOCKS_PER_SEC << " seconds" << endl;

	//Intel done



	
	//AMD START
	//new
	start_t = clock();
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::vector<cl::Device> devices;
	
	for (int iPlatform = 0; iPlatform < platforms.size(); iPlatform++) {
		platforms[iPlatform].getDevices(CL_DEVICE_TYPE_GPU, &devices);
		for (int iDevice = 0; iDevice < devices.size(); iDevice++) {

			PerformCalculationOnDevice(devices[iDevice]);

		}
	}

	end_t = clock();
	std::cout << "\nTime Taken By AMD GPU: " << (float)runTime / CLOCKS_PER_SEC << " seconds" << endl;
	std::cout << "Device load and link time of AMD GPU: " << (float)(end_t - start_t) / CLOCKS_PER_SEC << " seconds" << endl;

	/*
	for (int x = 0; x < row*col; x++)
	{

		cout << result[x]<<" ";
		if(x%row==0)
		cout << endl;
	}
	*/
	cout << endl;

	start_t = clock();
	mult();
	PerformCalculationOnHost();

	end_t = clock();
	std::cout << "\nTime Taken By CPU: " << (float)(end_t-start_t)/CLOCKS_PER_SEC << " seconds" << endl;

	/*
	for (int x = 0; x < row*col; x++)
	{
		
		cout << result[x]<<" ";
		if(x%row==0)
		cout << endl;
	}
	*/

	
	//cleanup
	free(matrix_A);
	free(matrix_B);
	free(result);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_C);
	clReleaseMemObject(d_B);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	system("pause");

	return 0;
}
