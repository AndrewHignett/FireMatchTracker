#pragma once
#include "matchTracker.h"
#include <cuda_runtime.h>

#ifndef _Particle_
#define _Particle_
//For the flame setup
//Particle definition
class Particle {
	//glm::vec4 Position;
	//glm::vec4 velocity;
	//glm::vec4 Color;
	//glm::vec3 position, velocity;
	float position[3], velocity[3];
	unsigned char r, g, b, a; //colour and alpha
	float size, angle, weight;
	float life; //remaining life of the particle. Dead and unused if < 0
public:
	__host__ __device__
	void setValues(float[3], float[3], unsigned char[4], float, float, float, float);
	__host__ __device__
	void updateParticle(float);
	__host__ __device__
	float getLife() { return life; };
};

//class container for the particle system, may be unnecessary
class ParticleSystem {
	int width;
	int particleCount;
	//Particle *flameParticles;
	int emmissionsPerFrame;
};

Particle *updateParticles(float deltaT, Particle *container, int maxParticles, int emissionsPerFrame);
Mat addFlame(Mat frame, int matchTip[2], Particle *container, int maxParticles);
Particle *initialSetValues(Particle *container, int maxParticles, float pos[3], float vel[3], unsigned char colour[4], float size, float angle, float weight, float life);
#endif