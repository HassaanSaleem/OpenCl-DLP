__kernel void matrixVectorMul(__global float* resultVector,
    __global float* matrixA,
    __global float* vectorB, 
    int width_A)
{
    int tx = get_global_id(0); 

    float value = 0;
    for (unsigned int k = 0; k < width_A; ++k) {
        value += matrixA[tx * width_A + k] * vectorB[k];
    }

    resultVector[tx] = value;
}