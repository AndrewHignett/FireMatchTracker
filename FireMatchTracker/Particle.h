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
	//include an original matchip X for easch particle, in order to taper towards the end of the flame
public:
	__host__ __device__
	void setValues(float[3], float[3], unsigned char[4], float, float, float, float);
	__host__ __device__
	void updateParticle(float);
	__host__ __device__
	float getLife() { return life; };
	__host__ __device__
	float *getPosition() { return position; };
	__host__ __device__
	float *getVelocity() { return velocity; };
	__host__ __device__
	unsigned char getRed() { return r; };
	__host__ __device__
	unsigned char getGreen() { return g; };
	__host__ __device__
	unsigned char getBlue() { return b; };
	__host__ __device__
	unsigned char getAlpha() { return a; };
};

//class container for the particle system, may be unnecessary
class ParticleSystem {
	int width;
	int particleCount;
	//Particle *flameParticles;
	int emmissionsPerFrame;
};

Particle *updateParticles(Particle *container, int matchTip[2]);
void addFlame(Mat frame, Mat fullFrame, Particle *container);
Particle *initialSetValues(Particle *container);

const int MaxParticles = 1000;
__device__
const float FrameTime = 0.033;
const int EmissionsPerFrame = 25;
#endif