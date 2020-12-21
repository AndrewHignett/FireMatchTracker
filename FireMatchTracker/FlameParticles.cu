#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"

const int MaxParticles = 10000;
Particle ParticleContainer[MaxParticles];
int LastUsedParticle = 0;


//For the flame setup
//Particle definition
class Particle {
	//glm::vec4 Position;
	//glm::vec4 velocity;
	//glm::vec4 Color;
	glm::vec3 position, velocity, acceleration;
	unsigned char r, g, b, a; //colour and alpha
	float size, angle, weight;
	float life; //remaining life of the particle. Dead and unused if < 0
public:
	void setValues(glm::vec3, glm::vec3, glm::vec3, unsigned char[4], float, float, float, float);
	void updateParticle(float);
};

void Particle::setValues(glm::vec3 pos, glm::vec3 vel, glm::vec3 acc, unsigned char colour[4], float sizeI, float angleI, float weightI, float lifeI) {
	position = pos;
	velocity = vel;
	acceleration = acc;
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
void Particle::updateParticle(float deltaT) {
	//update the values for this particle
	//postion =
	//velocity =
	//acceleration probably wont change, intiialise as a standard acceleration (in 3d, the acceleration must be for the 3d coordinates and then
	//appropriately converted to the 2D pixel coordinates)
	//acceleration =
	//life = 
	//Later, update colour based roughly off life span to emulate smoke or edge colouring of the fire
	//if the life is above the particle lifespan, then remove particle by resetting start inital particle attributes
}

//class container for the particle system, may be unnecessary
class ParticleSystem {

};

//code intended for opengl use
int findUnusedParticle() {
	for (int i = LastUsedParticle; i < MaxParticles; i++) {
		if (ParticleContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	for (int i = 0; i < LastUsedParticle; i++) {
		if (ParticleContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	return 0; //all particles are taken
}