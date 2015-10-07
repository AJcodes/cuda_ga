#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <iomanip>
#include <iostream>

#define BLOCKSIZE 64
#define GA_POPSIZE		2048		// ga population size
#define GA_MAXITER		16384		// maximum iterations
#define GA_ELITRATE		0.10f		// elitism rate
#define GA_MUTATIONRATE	0.25f		// mutation rate
#define GA_MUTATION		RAND_MAX * GA_MUTATIONRATE
#define GA_TARGET		"Hello world!"
#define GA_TARGETLEN    12

struct ga_struct 
{
	char str[GA_TARGETLEN];			// the string
	unsigned int fitness;			// its fitness
};

cudaError_t gaCuda(struct ga_struct *population, int size);

__device__ int rand(unsigned int *seed, int m) {
	unsigned int a = 32767;  		
	unsigned int x = *seed;

	x = (a * x) % m;
	*seed = x;

 	return ((int)x);
}

__device__ void elitismKernel(struct ga_struct *population, struct ga_struct *buffer, const int esize) {

}

__global__ void initKernel(struct ga_struct *population, const int size) {
	unsigned int id = blockIdx.x * BLOCKSIZE + threadIdx.x;
	unsigned int seed = id + 1;
	if (id < GA_POPSIZE) {
		ga_struct citizen;
		citizen.fitness = 0;
		for (int j = 0; j < size; j++)
			citizen.str[j] = (rand(&seed, 90) + 32);

		population[id] = citizen;
	}
}

__global__ void calcKernel(struct ga_struct *population, const int size) {
	unsigned int id = blockIdx.x * BLOCKSIZE + threadIdx.x;
	if (id < GA_POPSIZE) {
		char *target = GA_TARGET;
		unsigned int fitness = 0;
		ga_struct pop = population[id];
		for (int j = 0; j < size; j++)
			fitness += abs(pop.str[j] - target[j]);

		population[id].fitness = fitness;
	}
}

__global__ void bestKernel(struct ga_struct *population, struct ga_struct *best) {
	__shared__ struct ga_struct temp[BLOCKSIZE];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
	if (i < GA_POPSIZE) {
		temp[tid] = population[i];
		__syncthreads();
		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			int index = 2 * s * tid;
			if (index < blockDim.x) {
				if (temp[tid + s].fitness < temp[tid].fitness)
					temp[tid] = temp[tid + s];
			}
			__syncthreads();
		}
	}

	if (tid == 0) {
		best[blockIdx.x] = temp[0];
	}
}

__global__ void mateKernel(struct ga_struct *population, struct ga_struct *buffer) {

}

inline void print_best(struct ga_struct *gav) { 
	std::cout << "Best: " << " (" << gav[0].fitness << ")" << std::endl; 
}

int main() {
	srand(unsigned(time(NULL)));

	struct ga_struct *population;
	int size = sizeof(GA_TARGET) / sizeof(GA_TARGET[0]);

	population = (struct ga_struct *) malloc(GA_POPSIZE * sizeof(struct ga_struct));
	
    cudaError_t cudaStatus = gaCuda(population, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "gaCuda failed!");
        return 1;
    }
	std::cout << size << std::endl;
	for (int i = 0; i < size; i++) {
		std::cout << population[0].str[i];
	}
	std::cout << std::endl;
	for (int i = 0; i < size; i++) {
		std::cout << population[1].str[i];
	}
	std::cout << std::endl;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t gaCuda(struct ga_struct *population, int size) {
	struct ga_struct pop_alpha, pop_beta;
	struct ga_struct *buffer;
	dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(GA_POPSIZE/BLOCKSIZE);
    struct ga_struct *dev_population = 0;
    struct ga_struct *dev_buffer = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_population, GA_POPSIZE * sizeof(struct ga_struct));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_buffer, GA_POPSIZE * sizeof(struct ga_struct));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    initKernel<<<dimGrid, dimBlock>>>(dev_population, size);

	for (int i=0; i<GA_MAXITER; i++) {
		calcKernel<<<dimGrid, dimBlock>>>(dev_population, size);
	}

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "gaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(population, dev_population, GA_POPSIZE * sizeof(struct ga_struct), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_population);
    cudaFree(dev_buffer);
    
    return cudaStatus;
}
