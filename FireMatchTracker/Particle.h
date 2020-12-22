#pragma once
#include "matchTracker.h"

#ifndef _Particle_
#define _Particle_
//For the flame setup
//Particle definition
class Particle {
	//glm::vec4 Position;
	//glm::vec4 velocity;
	//glm::vec4 Color;
	glm::vec3 position, velocity;
	unsigned char r, g, b, a; //colour and alpha
	float size, angle, weight;
	float life; //remaining life of the particle. Dead and unused if < 0
public:
	void setValues(glm::vec3, glm::vec3, unsigned char[4], float, float, float, float);
	void updateParticle(float);
};

//class container for the particle system, may be unnecessary
class ParticleSystem {

};
#endif