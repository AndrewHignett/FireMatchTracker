#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "opencv2/cudaimgproc.hpp"
#include "matchTracker.h"
#include "Particle.h"

__global__ void genericErodeKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat dilatedFrame, int x, int y)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < x * y)
	{
		int row = threadId / x;
		int column = threadId % x;
		uint8_t pixelR = dilatedFrame.data[(row*dilatedFrame.step) + column * 3 + 2];

		if (pixelR == 255)
		{
			bool allPixelsRed = true;
			for (int i = -2; i < 3; i++)
			{
				for (int j = -2; j < 3; j++)
				{
					if ((row + i > -1) && (row + i < y) && (column + j > -1) && (column + j < x))
					{
						if (dilatedFrame.data[((row + i)*dilatedFrame.step) + (column + j) * 3 + 2] == 0)
						{
							allPixelsRed = false;
						}
					}
				}
			}
			if (allPixelsRed)
			{
				out.data[(row*out.step) + column * 3] = 0;
				out.data[(row*out.step) + column * 3 + 1] = 255;
				out.data[(row*out.step) + column * 3 + 2] = 0;
			}
		}
	}
}

//out must be a black frame
__global__ void genericDilateKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat flameFrame, int *particleCount)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int x = WINDOW_WIDTH;
	int y = WINDOW_HEIGHT;
	if (threadId < x * y)
	{
		int row = threadId / x;
		int column = threadId % x;
		uint8_t pixelR = flameFrame.data[(row*flameFrame.step) + column * 3 + 2];
		uint8_t pixelG = flameFrame.data[(row*flameFrame.step) + column * 3 + 1];
		uint8_t pixelB = flameFrame.data[(row*flameFrame.step) + column * 3];
		
		//printf("%d %d %d\n", threadId, x, y);
		if ((pixelR + pixelG + pixelB) > 0)
		{
			printf("%d %u %u %u\n", threadId, pixelR, pixelG, pixelB);
			for (int i = -6; i < 7; i++)
			{
				for (int j = -6; j < 7; j++)
				{
					if (!(i == 0 && j == 0))
					{
						if ((row + i > -1) && (row + i < y) && (column + j > -1) && (column + j < x))
						{
							//out.data[((row + i)*out.step) + (column + j) * 3 + 2] = 255;
							out.data[((row + i)*out.step) + (column + j) * 3 + 2] = (out.data[((row + i)*out.step) + (column + j) * 3 + 2] * particleCount[(row + i)*out.step + (column + j)] + flameFrame.data[row*flameFrame.step + column * 3 + 2]) / (particleCount[(row + i)*out.step + (column + j)] + 1);
							out.data[((row + i)*out.step) + (column + j) * 3 + 1] = (out.data[((row + i)*out.step) + (column + j) * 3 + 1] * particleCount[(row + i)*out.step + (column + j)] + flameFrame.data[row*flameFrame.step + column * 3 + 1]) / (particleCount[(row + i)*out.step + (column + j)] + 1);
							out.data[((row + i)*out.step) + (column + j) * 3] = (out.data[((row + i)*out.step) + (column + j) * 3] * particleCount[(row + i)*out.step + (column + j)] + flameFrame.data[row*flameFrame.step + column * 3]) / (particleCount[(row + i)*out.step + (column + j)] + 1);
							particleCount[(row + i)*out.step + (column + j) * 3 + 2] += 1;
							particleCount[(row + i)*out.step + (column + j) * 3 + 1] += 1;
							particleCount[(row + i)*out.step + (column + j) * 3] += 1;
							//printf("%d\n", out.data[((row + i)*out.step) + (column + j) * 3 + 2]);
							//printf("%d %d %d %d\n", threadId, row, column, flameFrame.data[row*flameFrame.step + column * 3 + 2]);
						}
					}
				}
			}
		}

	}
}

__host__ __device__
void Particle::setValues(float pos[3], float vel[3], unsigned char colour[4], float sizeI, float angleI, float weightI, float lifeI) {
	//position = pos;
	//velocity = vel;
	position[0] = pos[0];
	position[1] = pos[1];
	position[2] = pos[2];
	velocity[0] = vel[0];
	velocity[1] = vel[1];
	velocity[2] = vel[2];
	//acceleration = acc;
	r = colour[0];
	g = colour[1];
	b = colour[2];
	a = colour[3];
	//size may be irrelevant, if deeling with sub-pixel particles, however, this may result in a flame with gaps in
	size = sizeI;
	angle = angleI;
	weight = weightI;
	//particle life span must be longer as the particle's initial location approaches the match tracked location
	//life starts as a number (e.g. 4), and the particle is inactive when it's <= 0
	//all particles should be initialised to a life of 0
	life = lifeI;
}

//Update the state parameters of the particle based off it's acceleration, initial velocity and position
__host__ __device__
void Particle::updateParticle(float deltaT) {
	//update the values for this particle
	//postion =
	//velocity =
	//acceleration probably wont change, intiialise as a standard acceleration (in 3d, the acceleration must be for the 3d coordinates and then
	//appropriately converted to the 2D pixel coordinates)
	//acceleration is ignored, and instead just a fixed velocity could be used
	//acceleration =
	//life = 
	//Later, update colour based roughly off life span to emulate smoke or edge colouring of the fire
	//if the life is above the particle lifespan, then remove particle by resetting start inital particle attributes
}

__global__ void flameKernel(cv::cuda::GpuMat frame, Particle *container) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < MaxParticles)
	{
		//printf("%d %d %f\n", MaxParticles, threadId, container[threadId].getLife());
		if ((threadId < MaxParticles) && (container[threadId].getLife() > 0)) {
			//int row = threadId / X;
			//int column = threadId % X;
			float *xyz = container[threadId].getPosition();
			int row = xyz[1];
			int column = xyz[0];
			//BGR pixel values
			frame.data[(row*frame.step) + column * 3] = container[threadId].getBlue();
			frame.data[(row*frame.step) + column * 3 + 1] = container[threadId].getGreen();
			frame.data[(row*frame.step) + column * 3 + 2] = container[threadId].getRed();
		}
		
	}
}

Mat addFlame(Mat frame, Particle *container) {
	int threadCount = 1024;
	int blocks = (MaxParticles - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = MaxParticles;
	}
	Particle *d_container;
	//allocate device memory for the particle containe
	cudaMalloc((void**)&d_container, sizeof(Particle) * MaxParticles);
	//transfer from host to device memory	
	cudaMemcpy(d_container, container, sizeof(Particle) * MaxParticles, cudaMemcpyHostToDevice);

	uint8_t *d_imgPtr;
	cv::cuda::GpuMat d_newFrame;
	d_newFrame.upload(frame);

	//Allocate device memory
	cudaMalloc((void **)&d_imgPtr, d_newFrame.rows*d_newFrame.step);
	cudaMemcpyAsync(d_imgPtr, d_newFrame.ptr<uint8_t>(), d_newFrame.rows*d_newFrame.step, cudaMemcpyDeviceToDevice);
	flameKernel << <blocks, threadCount >> > (d_newFrame, d_container);
	cudaDeviceSynchronize();

	//int xCopy = (int)malloc(sizeof(int));
	//int yCopy = (int)malloc(sizeof(int));
	//xCopy = x;
	//yCopy = y;
	int x = WINDOW_WIDTH;
	int y = WINDOW_HEIGHT;
	int *particleCount;
	particleCount = (int*)malloc(sizeof(int)*x*y*3);
	memset(particleCount, 0, sizeof(int)*x*y*3);
	Mat out(y, x, CV_8UC3, cv::Scalar(0, 0, 0));
	uint8_t *d_outPtr;
	cv::cuda::GpuMat d_out;
	d_out.upload(out);
	//int *d_x, *d_y, *d_particleCount;
	int *d_particleCount;
	//cudaMalloc((void**)&d_x, sizeof(int));
	//cudaMalloc((void**)&d_y, sizeof(int));
	cudaMalloc((void**)&d_particleCount, sizeof(int)*x*y*3);
	cudaMalloc((void**)&d_outPtr, d_out.rows*d_out.step);
	//cudaMemcpy(d_x, &x, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_y, &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_particleCount, particleCount, sizeof(int)*x*y, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_outPtr, d_out.ptr<uint8_t>(), d_out.rows*d_out.step, cudaMemcpyDeviceToDevice);

	threadCount = 1024;
	blocks = (x * y - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = x* y;
	}

	genericDilateKernel << <blocks, threadCount >> > (d_out, d_newFrame, d_particleCount);
	//Downloading frame can crash
	//d_newFrame.download(frame);
	d_out.download(frame);
	//Free original frame pointer device memory
	cudaFree(d_imgPtr);
	d_newFrame.release();
	cudaFree(d_outPtr);
	d_out.release();
	cudaFree(d_particleCount);
	//free the device memory for the particle container
	cudaFree(d_container);

	//free host memory
	free(particleCount);

	//for debug only
	return frame;
}

__global__ void particleKernel(Particle *container, int *matchTip, curandState_t *states){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < MaxParticles)
	{
		if (container[threadId].getLife() > 0) {
			//update all active particles
			//some may be reduced to a life below 0
			float *pos = container[threadId].getPosition();
			//float vel[3] = { 0.0, -300.0, 0.0 };
			float *vel = container[threadId].getVelocity();
			pos[1] += vel[1]*FrameTime;
			float size = 1;
			//angle and weight may be unnessecary for this particle system
			float angle = 0;
			float weight = 1;
			float life = container[threadId].getLife() + FrameTime;
			//unsigned char colour[4] = { container[threadId].getRed(), 85*log10f(32/life), life * 2 * 255, container[threadId].getAlpha() };
			unsigned char colour[4] = { container[threadId].getRed(), 85 * log10f(32 / life), container[threadId].getBlue(), container[threadId].getAlpha() };
			//give the particles a max life time
			if (life < 0.5){
				
				container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
			}
			else if (life < 0.6) {
				unsigned char colour[4] = { 144, 144, 144, container[threadId].getAlpha() };
				container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
			}
			else {
				//unsigned char colour[4] = { container[threadId].getRed(), 85 * log10f(32 / life), container[threadId].getBlue(), container[threadId].getAlpha() };
				unsigned char colour[4] = { 255, 255, 0, container[threadId].getAlpha() };
				container[threadId].setValues(pos, vel, colour, size, angle, weight, 0);
			}
		}
		else if (threadId < EmissionsPerFrame) {
			//update EmissionsPerFrame particles that have a life <= 0
			//inactive particles, all will have a life <= 0
			//update these particles as new particles
			//it's possible this may be less than the number of emmissions per frame and that there still may be remaining inactive particles
			float width = 20;
			float baseVelocity = -200;
			curand_init(0, threadId, 0, &states[threadId]);
			float randomStartPosX = curand_uniform(&states[threadId])*width - (width/2) + float(matchTip[0]);
			float randomStartPosY = curand_uniform(&states[threadId])*(width/2) - (width / 4) + float(matchTip[1]);
			float velY = curand_uniform(&states[threadId])*200 + baseVelocity;
			if (velY > -50) {
				velY = -100;
			}
			float pos[3] = { randomStartPosX, randomStartPosY, 0.0 };
			//float vel[3] = { 0.0, -300.0, 0.0 };
			float vel[3] = { 0.0, velY, 0.0 };
			unsigned char colour[4] = { container[threadId].getRed(), container[threadId].getGreen(), container[threadId].getBlue(), container[threadId].getAlpha() };
			float size = 1;
			//angle and weight may be unnessecary for this particle system
			float angle = 0;
			float weight = 1;
			float life = FrameTime;
			container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
		}
	}
}

__global__ void initialParticleKernel(Particle *container) {
	//printf("%d\n", MaxParticles);
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < MaxParticles)
	{
		float pos[3] = { 0.0, 0.0, 0.0 };
		float vel[3] = { 0.0, -300.0, 0.0 };
		unsigned char colour[4] = { 255, 255, 0, 0 };
		float size = 1;
		//angle and weight may be unnessecary for this particle system
		float angle = 0;
		float weight = 1;
		float life = 0;
		//printf("%f\n", life);
		container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
	}
}

//update the particle postions and return the new positions, before adding the flame to the frame
//the particles are already sorted by their life, low to high
Particle *updateParticles(Particle *container, int matchTip[2]) {
	//add a new number of particles based on the emmissions per frame
	//max out at maxParticles
	//it's possible for particles to be removed, as they time out
	//we need a way to check if a particle is in use quickly so that particles can be overwritten
	//with new particles. The particle's age can act as this.
	//We can update a given number of known particles, so that they can be made visible
	//The particles would need to be sorted by age, or at the very least, guarenteed that the first
	//"emmisions per frame" particles are innactive
	//Alternatively, we could add another variable to the Particle class, a Boolean "Active", to indicate
	//whether the particle is active or not. This adds a little memory useage, but makes the sorting much
	//easier

	//reduced threadCount to fix the "Too many resources requested for launch" error
	int threadCount = 256;
	int blocks = (MaxParticles - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = MaxParticles;
	}

	Particle *d_container;
	int *d_matchTip;
	curandState_t *d_randStates;
	//allocate device memory for the particle container aand match tip
	cudaMalloc((void**)&d_container, sizeof(Particle) * MaxParticles);
	cudaMalloc((void**)&d_matchTip, sizeof(int) * 2);
	cudaMalloc((void**)&d_randStates, sizeof(curandState_t) * MaxParticles);
	//transfer from host to device memory	
	cudaMemcpy(d_container, container, sizeof(Particle) * MaxParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matchTip, matchTip, sizeof(int) * 2, cudaMemcpyHostToDevice);
	particleKernel<<<blocks, threadCount>>>(d_container, d_matchTip, d_randStates);
	//for testing the kernel for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize;
	cudaMemcpy(container, d_container, sizeof(Particle) * MaxParticles, cudaMemcpyDeviceToHost);
	cudaFree(d_container);
	cudaFree(d_matchTip);
	cudaFree(d_randStates);
	//for debug only
	return container;
}

Particle *initialSetValues(Particle *container) {
	//pos, vel, colour, size, angle, weight, life
	//call a cuda kernel and initialise each of the particles simultaneously
	int threadCount = 1024;
	int blocks = (MaxParticles - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = MaxParticles;
	}
	Particle *d_container;
	//allocate device memory for the particle container
	cudaMalloc((void**)&d_container, sizeof(Particle) * MaxParticles);
	//transfer from host to device memory
	//cudaMemcpy(d_container, container, sizeof(Particle) * MaxParticles, cudaMemcpyHostToDevice);
	//allocate device memory for the maxParticle count
	//cudaMalloc((void**)&d_maxParticles, sizeof(int));
	//transfer from host to device memory
	//cudaMemcpy(d_maxParticles, maxParticles, sizeof(int), cudaMemcpyHostToDevice);
	initialParticleKernel << <blocks, threadCount >> > (d_container);
	//cudaError_t error = cudaGetLastError();
	//if (error != cudaSuccess){
	//	printf("%s\n", cudaGetErrorString(error));
	//}
	cudaDeviceSynchronize;
	//copy device memory for the particle container back to host memory
	cudaMemcpy(container, d_container, sizeof(Particle) * MaxParticles, cudaMemcpyDeviceToHost);
	//free(containerCopy);
	//free(maxParticlesCopy);
	cudaFree(d_container);
	//for debug only
	return container;
}