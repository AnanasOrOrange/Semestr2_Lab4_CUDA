#include "cuda_runtime.h"					
#include "device_launch_parameters.h"		
#include <iostream>
#include <chrono>
#include <math.h>
#include "thrust\reduce.h"
#include "thrust\device_ptr.h"

#define CHECK_ERR()														
void check() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\M", __FILE__, __LINE__, cudaGetErrorString(err));
        system("pause");
        exit(1);
    }
}

#define EPS double(1e-4)
// число разбиений
#define N int(10)
// время, до которого считать
#define T 0.05

#define BLOCK_SIZE 32

// для вывода матриц
#define OUT

// упрощение для удобства
#define node(i,j) ((i) * (N) + (j))
//#define Q(i,j) double((i) == (N) / 2 && (j) == (N) / 2)
#define isBorder(i,j) bool((i) == (0) || (j) == (0) || (i) == (N) - 1 || (j) == (N) - 1)

#define hx (1.0 / double((N) - 1))
#define hy (1.0 / double((N) - 1))

//#define tau (hx * hx * hy * hy / (3.0 * (hx * hx + hy * hy)))

void initMesh(double* mesh) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0) {
                mesh[node(i, j)] = exp(1. - 1. / (N - 1.) * j);
            }
            else if (j == 0) {
                mesh[node(i, j)] = exp(1. - 1. / (N - 1.) * i);
            }
            else if (i == N - 1 || j == N - 1) {
                mesh[node(i, j)] = 1;
            }
            else {
                mesh[node(i, j)] = 0;
            }
        }
    }
}

void CPU_Cross(double*& prev, double*& next) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!isBorder(i, j)) {
                next[node(i, j)] = 0.5 * (hy * hy * (prev[node(i + 1, j)] + prev[node(i - 1, j)]) + hx * hx * (prev[node(i, j + 1)] + prev[node(i, j - 1)])) / (hx * hx + hy * hy);
            }
        }
    }
}

//__global__
//void GPU_Cross(double* prev, double* next) {
//    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//
//    int i = tid / N;
//    int j = tid % N;
//
//    if (tid < N * N && !isBorder(i, j)) {
//        next[node(i, j)] = prev[node(i, j)] + tau * ((prev[node(i + 1, j)] - 2.0 * prev[node(i, j)] + prev[node(i - 1, j)]) / hx / hx + \
//            (prev[node(i, j + 1)] - 2.0 * prev[node(i, j)] + prev[node(i, j - 1)]) / hy / hy);// +Q(i, j));
//    }
//}

__global__
void GPU_Cross_L4(double* prev, double* next, double* arrL4) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int i = tid / N;
    int j = tid % N;

    if (tid < N * N) {
        if (!isBorder(i, j)) {
            next[node(i, j)] =  0.5 * (hy * hy * (prev[node(i + 1, j)] + prev[node(i - 1, j)]) + hx * hx * (prev[node(i, j + 1)] + prev[node(i, j - 1)])) / (hx * hx + hy * hy);
        }
        arrL4[node(i, j)] = pow(next[node(i, j)] - prev[node(i, j)], 4) * hx * hy;
        //arrL4[node(i, j)] = (next[node(i, j)] - prev[node(i, j)]) * (next[node(i, j)] - prev[node(i, j)]) * \
        //                    (next[node(i, j)] - prev[node(i, j)]) * (next[node(i, j)] - prev[node(i, j)]) * hx * hy;
    }

}

#ifdef OUT
void printMesh(double* mesh) {
    std::cout << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mesh[node(i, j)] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}
#endif

bool isEquel(double* mesh1, double* mesh2) {
    for (int i = 0; i < N * N; i++) {
        if (mesh1[i] != mesh2[i]) {
            return false;
        }
    }
    return true;
}


int main()
{
    // CPU
    double L4;

    double* prevCPU = new double[N * N];
    double* nextCPU = new double[N * N];
    initMesh(prevCPU);
    initMesh(nextCPU);

    std::cout << "N\t" << N << std::endl;
    std::cout << "EPS\t" << EPS << std::endl;
    std::cout << std::endl;
    std::cout << "hx\t" << hx << std::endl;
    std::cout << "hy\t" << hy << std::endl;
    std::cout << std::endl;

    auto startCPU = std::chrono::steady_clock::now();

	std::cout << "CPU start" << std::endl;

    do {
        L4 = 0;
        CPU_Cross(prevCPU, nextCPU);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                L4 += pow(nextCPU[node(i, j)] - prevCPU[node(i, j)], 4) * hx * hy;
            }
        }
        L4 /= N * N;
		L4 = pow(L4, 0.25);
		//std::cout << L4 << std::endl;
        std::swap(prevCPU, nextCPU);

	} while (L4 > EPS);

    std::cout << "CPU end" << std::endl;

    auto endCPU = std::chrono::steady_clock::now();
    std::cout << "CPU time\t= " << std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count() << "\t\tmillisec, (1e-3 sec)" << std::endl;

#ifdef OUT
    printMesh(prevCPU);
#endif

    // CUDA

    double* prevGPU = new double[N * N];
    double* nextGPU = new double[N * N];
    initMesh(prevGPU);
    initMesh(nextGPU);

    double* prevDev, * nextDev, * arrL4;

    cudaMalloc((void**)&prevDev, N * N * sizeof(double)); CHECK_ERR();
	cudaMalloc((void**)&nextDev, N * N * sizeof(double)); CHECK_ERR();
	cudaMalloc((void**)&arrL4,   N * N * sizeof(double)); CHECK_ERR();

    cudaEvent_t startGPU, stopGPU;

    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU);

    cudaMemcpy(prevDev, prevGPU, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR();
    cudaMemcpy(nextDev, nextGPU, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR();
 

    do {
        L4 = 0;
        GPU_Cross_L4 << < N * N / BLOCK_SIZE + 1, BLOCK_SIZE >> > (prevDev, nextDev, arrL4); CHECK_ERR();
        std::swap(prevDev, nextDev);

        thrust::device_ptr<double> ptr(arrL4);
        L4 = thrust::reduce(ptr, ptr + N * N);
        L4 /= N * N;
        L4 = pow(L4, 0.25);

    } while (L4 > EPS);


    cudaMemcpy(prevGPU, prevDev, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR();
    //cudaMemcpy(nextGPU, nextDev, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    float timeCUDA = 0;
    cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

    std::cout << "GPU time\t= " << timeCUDA << "\tmillisec, (1e-3 sec)" << std::endl;

#ifdef OUT
    printMesh(prevGPU);
#endif
    std::cout << std::endl;
    std::cout << "Checking results:" << std::endl;
    if (!isEquel(prevCPU, prevGPU)) {
        std::cout << "ERROR" << std::endl;
    }
    else {
        std::cout << "OK" << std::endl;
    }

}

