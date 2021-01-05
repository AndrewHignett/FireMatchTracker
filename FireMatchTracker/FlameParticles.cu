#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "matchTracker.h"
#include "Particle.h"

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
			frame.data[(row*frame.step) + column * 3] = 0;
			frame.data[(row*frame.step) + column * 3 + 1] = 255;
			frame.data[(row*frame.step) + column * 3 + 2] = 255;
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
	d_newFrame.download(frame);
	//Free original frame pointer device memory
	cudaFree(d_imgPtr);
	d_newFrame.release();
	//free the device memory for the particle container
	cudaFree(d_container);

	//for debug only
	return frame;
}

__global__ void particleKernel(Particle *container, int *matchTip){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < MaxParticles)
	{
		if (container[threadId].getLife() > 0) {
			//update all active particles
			//some may be reduced to a life below 0
			float *pos = container[threadId].getPosition();
			float vel[3] = { 0.0, 0.0, 0.0 };
			unsigned char colour[4] = { 0, 0, 0, 0 };
			float size = 1;
			//angle and weight may be unnessecary for this particle system
			float angle = 0;
			float weight = 1;
			float life = container[threadId].getLife() + FrameTime;
			//give the particles a max life time
			if (life < 0.5){
				container[threadId].setValues(pos, vel, colour, size, angle, weight, life);
			}
			else {
				container[threadId].setValues(pos, vel, colour, size, angle, weight, 0);
			}
		}
		else if (threadId < EmissionsPerFrame) {
			//update EmissionsPerFrame particles that have a life <= 0
			//inactive particles, all will have a life <= 0
			//update these particles as new particles
			//it's possible this may be less than the number of emmissions per frame and that there still may be remaining inactive particles
			float pos[3] = { float(matchTip[0]), float(matchTip[1]), 0.0 };
			float vel[3] = { 0.0, 0.0, 0.0 };
			unsigned char colour[4] = { 0, 0, 0, 0 };
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
		float vel[3] = { 0.0, 0.0, 0.0 };
		unsigned char colour[4] = { 0, 0, 0, 0 };
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
	float pos[3] = { 0.0, 0.0, 0.0 };
	float vel[3] = { 0.0, 0.0, 0.0 };
	unsigned char colour[4] = { 0, 0, 0, 0 };
	float size = 1;
	//angle and weight may be unnessecary for this particle system
	float angle = 0;
	float weight = 1;

	int threadCount = 1024;
	int blocks = (MaxParticles - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = MaxParticles;
	}

	Particle *d_container;
	int *d_matchTip;
	//allocate device memory for the particle container aand match tip
	cudaMalloc((void**)&d_container, sizeof(Particle) * MaxParticles);
	cudaMalloc((void**)&d_matchTip, sizeof(int) * 2);
	//transfer from host to device memory	
	cudaMemcpy(d_container, container, sizeof(Particle) * MaxParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matchTip, matchTip, sizeof(int) * 2, cudaMemcpyHostToDevice);
	particleKernel<<<blocks, threadCount>>>(d_container, d_matchTip);
	cudaDeviceSynchronize;
	cudaMemcpy(container, d_container, sizeof(Particle) * MaxParticles, cudaMemcpyDeviceToHost);
	cudaFree(d_container);
	cudaFree(d_matchTip);
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