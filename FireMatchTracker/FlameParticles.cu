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

__global__ void genericErodeKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat flameFrame, cv::cuda::GpuMat fullFrame, int *alphas)
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

		bool allPixelsFire = false;
		if ((pixelR + pixelG + pixelB) > 0)
		{
			allPixelsFire = true;
			for (int i = -2; i < 3; i++)
			{
				for (int j = -2; j < 3; j++)
				{
					if ((row + i > -1) && (row + i < y) && (column + j > -1) && (column + j < x))
					{
						if ((flameFrame.data[((row + i)*flameFrame.step) + (column + j) * 3 + 2] == 0)&& (flameFrame.data[((row + i)*flameFrame.step) + (column + j) * 3 + 1] == 0)&& (flameFrame.data[((row + i)*flameFrame.step) + (column + j) * 3] == 0))
						{
							allPixelsFire = false;
						}
					}
				}
			}
		}
		if (allPixelsFire)
		{
			out.data[(row*out.step) + column * 3] = pixelB * float(alphas[row * x + column])/255 + (1 - float(alphas[row * x + column])/255) * fullFrame.data[(row*fullFrame.step) + column * 3];
			out.data[(row*out.step) + column * 3 + 1] = pixelG * float(alphas[row * x + column])/255 + (1 - float(alphas[row * x + column])/255) * fullFrame.data[(row*fullFrame.step) + column * 3 + 1];
			out.data[(row*out.step) + column * 3 + 2] = pixelR * float(alphas[row * x + column])/255 + (1 - float(alphas[row * x + column])/255) * fullFrame.data[(row*fullFrame.step) + column * 3 + 2];
		}
		else {
			out.data[(row*out.step) + column * 3 + 2] = fullFrame.data[(row*fullFrame.step) + column * 3 + 2];
			out.data[(row*out.step) + column * 3 + 1] = fullFrame.data[(row*fullFrame.step) + column * 3 + 1];
			out.data[(row*out.step) + column * 3] = fullFrame.data[(row*fullFrame.step) + column * 3];
		}
	}
}

__global__ void applyDilation(cv::cuda::GpuMat out, int *particleCount, int *alphas) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int x = WINDOW_WIDTH;
	int y = WINDOW_HEIGHT;
	if (threadId < x * y)
	{
		int row = threadId / x;
		int column = threadId % x;
		if (particleCount[row * 5 * x + 5 * column + 4] > 0) {
			int particles = particleCount[row * 5 * x + 5 * column + 4];
			//particle count is sometimes not a valid number and it results in the pixel colours being above 255
			while (((particleCount[row * 5 * x + 5 * column] / particles) > 255) || ((particleCount[row * 5 * x + 5 * column + 1] / particles) > 255)||((particleCount[row * 5 * x + 5 * column + 2] / particles) > 255) || ((particleCount[row * 5 * x + 5 * column + 4] / particles) > 255)) {
				particles += 1;
			}
			for (int k = 0; k < 3; k++) {
				out.data[(row*out.step) + column * 3 + 2 - k] = particleCount[row * 5 * x + 5 * column + k] / particles;
			}
			alphas[row * x + column] = particleCount[row * 5 * x + 5 * column + 3] / particles;
		}	
		else {
			out.data[(row*out.step) + column * 3 + 2] = 0;
			out.data[(row*out.step) + column * 3 + 1] = 0;
			out.data[(row*out.step) + column * 3] = 0;
			alphas[row * x + column] = 0;
		}
	}
}

/*
flameFrame must start as a black frame
alpha transparency needs to be taken into account, and added to the smoke part
*/
__global__ void genericDilateKernel(cv::cuda::GpuMat flameFrame, int *particleCount, int *alphas)
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
		
		if ((pixelR + pixelG + pixelB) > 0)
		{
			for (int i = -6; i < 7; i++)
			//for (int i = -4; i < 5; i++)
			{
				for (int j = -6; j < 7; j++)
				//for (int j = -4; j < 5; j++)
				{
					if ((row + i > -1) && (row + i < y) && (column + j > -1) && (column + j < x))
					{
						particleCount[(row + i) * 5 * x + 5 * (column + j)] += pixelR;
						particleCount[(row + i) * 5 * x + 5 * (column + j) + 1] += pixelG;
						particleCount[(row + i) * 5 * x + 5 * (column + j) + 2] += pixelB;
						particleCount[(row + i) * 5 * x + 5 * (column + j) + 3] += alphas[row * x + column];
						//store count of particles used in this given pixel
						particleCount[(row + i) * 5 * x + 5 * (column + j) + 4] += 1;
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

__global__ void flameKernel(cv::cuda::GpuMat frame, Particle *container, int *alphas) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int x = WINDOW_WIDTH;
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
			if ((row >= 0) && (column >= 0) && (row < WINDOW_HEIGHT) && (column < WINDOW_WIDTH)) {
				frame.data[(row*frame.step) + column * 3] = container[threadId].getBlue();
				frame.data[(row*frame.step) + column * 3 + 1] = container[threadId].getGreen();
				frame.data[(row*frame.step) + column * 3 + 2] = container[threadId].getRed();
				alphas[row * x + column] = container[threadId].getAlpha();
			}
		}
	}
}

void addFlame(Mat frame, Mat fullFrame, Particle *container) {
	int x = WINDOW_WIDTH;
	int y = WINDOW_HEIGHT;
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
	int *alphas;
	alphas = (int*)malloc(sizeof(int)*x*y);
	memset(alphas, 0, sizeof(int)*x*y);
	int *d_alphas;
	Mat newFrame(y, x, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::cuda::GpuMat d_newFrame;
	//d_newFrame.upload(frame);
	d_newFrame.upload(newFrame);

	//Allocate device memory
	cudaMalloc((void **)&d_imgPtr, d_newFrame.rows*d_newFrame.step);
	cudaMalloc((void**)&d_alphas, sizeof(int)*x*y);
	cudaMemcpyAsync(d_imgPtr, d_newFrame.ptr<uint8_t>(), d_newFrame.rows*d_newFrame.step, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_alphas, alphas, sizeof(int)*x*y, cudaMemcpyHostToDevice);
	flameKernel << <blocks, threadCount >> > (d_newFrame, d_container, d_alphas);
	cudaDeviceSynchronize();

	//free the device memory for the particle container
	cudaFree(d_container);
	
	int *particleCount;
	particleCount = (int*)malloc(sizeof(int)*x*y*5);
	memset(particleCount, 0, sizeof(int)*x*y*5);
	uint8_t *d_fullFramePtr;
	cv::cuda::GpuMat d_fullFrame;
	
	d_fullFrame.upload(fullFrame);
	int *d_particleCount;
	cudaMalloc((void**)&d_particleCount, sizeof(int)*x*y*5);
	cudaMalloc((void**)&d_fullFramePtr, d_fullFrame.rows*d_fullFrame.step);
	cudaMemcpy(d_particleCount, particleCount, sizeof(int)*x*y*5, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_fullFramePtr, d_fullFrame.ptr<uint8_t>(), d_fullFrame.rows*d_fullFrame.step, cudaMemcpyDeviceToDevice);
	

	threadCount = 1024;
	blocks = (x * y - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = x * y;
	}

	genericDilateKernel << <blocks, threadCount >> > (d_newFrame, d_particleCount, d_alphas);
	cudaDeviceSynchronize();

	//free new frame from device memory
	cudaFree(d_imgPtr);
	d_newFrame.release();

	Mat out(y, x, CV_8UC3, cv::Scalar(0, 0, 0));
	//allocate flame frame to host and device memory
	uint8_t *d_flameFramePtr;
	cv::cuda::GpuMat d_flameFrame;
	d_flameFrame.upload(out);
	cudaMalloc((void**)&d_flameFramePtr, d_fullFrame.rows*d_fullFrame.step);
	cudaMemcpyAsync(d_flameFramePtr, d_flameFrame.ptr<uint8_t>(), d_flameFrame.rows*d_flameFrame.step, cudaMemcpyDeviceToDevice);

	applyDilation << <blocks, threadCount >> > (d_flameFrame, d_particleCount, d_alphas);
	cudaDeviceSynchronize();

	cv::cuda::GpuMat d_out;
	uint8_t *d_outPtr;
	d_out.upload(out);
	cudaMalloc((void**)&d_outPtr, d_out.rows*d_out.step);
	cudaMemcpyAsync(d_outPtr, d_out.ptr<uint8_t>(), d_out.rows*d_out.step, cudaMemcpyDeviceToDevice);

	genericErodeKernel<<<blocks, threadCount>>>(d_out, d_flameFrame, d_fullFrame, d_alphas);
	cudaDeviceSynchronize();
	d_out.download(frame);
	//Free frame pointers device memory	
	cudaFree(d_outPtr);
	d_out.release();
	cudaFree(d_fullFramePtr);
	d_fullFrame.release();
	cudaFree(d_flameFramePtr);
	d_flameFrame.release();
	cudaFree(d_particleCount);
	cudaFree(d_alphas);

	//free host memory
	free(particleCount);
	free(alphas);


	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err));

		// Possibly: exit(-1) if program cannot continue....
	}
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
			unsigned char colour[4] = { container[threadId].getRed(), 85 * log10f(32 / life), container[threadId].getBlue(), 255};
			//give the particles a max life time
			if (life < 0.5){
				
				container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
			}
			else if (life < 0.6) {
				//life*life*life*life*life may be faster than std::power(life, 5)
				unsigned char colour[4] = { 144, 144, 144, 8/(life*life*life*life*life)};
				container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
			}
			else {
				//unsigned char colour[4] = { container[threadId].getRed(), 85 * log10f(32 / life), container[threadId].getBlue(), container[threadId].getAlpha() };
				unsigned char colour[4] = { 255, 255, 0, 255 };
				container[threadId].setValues(pos, vel, colour, size, angle, weight, 0);
			}
		}
		else if ((threadId < EmissionsPerFrame)&&(matchTip[0] > -1)) {
			//update EmissionsPerFrame particles that have a life <= 0
			//inactive particles, all will have a life <= 0
			//update these particles as new particles
			//it's possible this may be less than the number of emmissions per frame and that there still may be remaining inactive particles
			float width = 20;
			float baseVelocity = -200;
			curand_init(clock(), threadId, 0, &states[threadId]);
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
		unsigned char colour[4] = { 255, 255, 0, 255 };
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