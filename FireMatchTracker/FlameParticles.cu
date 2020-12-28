#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"
#include "Particle.h"

__host__ __device__
void Particle::setValues(glm::vec3 pos, glm::vec3 vel, unsigned char colour[4], float sizeI, float angleI, float weightI, float lifeI) {
	position = pos;
	velocity = vel;
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

Mat addFlame(Mat frame, int matchTip[2], Particle container[], int maxParticles) {

}

__global__ void particleKernel(Particle container[], int maxParticles){

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < maxParticles)
	{

	}
}

//update the particle postions and return the new positions, before adding the flame to the frame
Particle *updateParticles(float deltaT, Particle container[], int maxParticles) {
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
}