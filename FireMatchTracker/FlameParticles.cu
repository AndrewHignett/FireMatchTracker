#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

Mat addFlame(Mat frame, int matchTip[2], Particle *container, int maxParticles) {
	//for debug only
	return frame;
}

__global__ void particleKernel(Particle *container, int maxParticles, int emissionsPerFrame){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < maxParticles)
	{
		if (container[threadId].getLife() > 0) {
			//update active particles
			//some may be reduced to a life below 0
		}
		else if (threadId < emissionsPerFrame) {
			//inactive particles, all will have a life <= 0
			//update these particles as new particles
			//it's possible this may be less than the number of emmissions per frame and that there still may be remaining inactive particles
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
Particle *updateParticles(float deltaT, Particle *container, int maxParticles, int emissionsPerFrame) {
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
	//printf("%f\n", container[100].getLife());
	float pos[3] = { 0.0, 0.0, 0.0 };
	float vel[3] = { 0.0, 0.0, 0.0 };
	unsigned char colour[4] = { 0, 0, 0, 0 };
	float size = 1;
	//angle and weight may be unnessecary for this particle system
	float angle = 0;
	float weight = 1;
	float life = container[100].getLife() + 1;
	//printf("%f\n", life);
	container[100].setValues(pos, vel, colour, size, angle, weight, life);
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
	//float *d_pos[3], *d_vel[3];
	//unsigned char *d_colour[4];
	//float *d_size, *d_angle, *d_weight, *d_life;
	//Particle *containerCopy;
	//int *maxParticlesCopy;
	//float *posCopy[3], *velCopy[3];
	//unsigned char *colourCopy[4];
	//float *sizeCopy, *angleCopy, *weightCopy, *lifeCopy;
	//allocate host memory for initialiser variables
	//containerCopy = (Particle*)malloc(sizeof(Particle) * maxParticles);
	//maxParticlesCopy = (int*)malloc(sizeof(int));
	//*posCopy = (float*)malloc(sizeof(float) * 3);
	//*velCopy = (float*)malloc(sizeof(float) * 3);
	//*colourCopy = (unsigned char*)malloc(sizeof(unsigned char) * 4);
	//sizeCopy = (float*)malloc(sizeof(float));
	//angleCopy = (float*)malloc(sizeof(float));
	//weightCopy = (float*)malloc(sizeof(float));
	//lifeCopy = (float*)malloc(sizeof(float));
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