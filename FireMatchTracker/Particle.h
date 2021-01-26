#pragma once

#ifndef _Particle_
#define _Particle_
/*
For the flame setup
Particle definition including getter functions for device code too
*/
class Particle {
	float position[3], velocity[3];
	unsigned char r, g, b, a; //colour and alpha
	float life; //remaining life of the particle. Dead and unused if < 0
public:
	__host__ __device__
	void setValues(float[3], float[3], unsigned char[4], float);
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

/*
Declare function templates that relate to the particles
*/
Particle *updateParticles(Particle *container, int matchTip[2]);
void addFlame(Mat frame, Mat fullFrame, Particle *container);
Particle *initialSetValues(Particle *container);

/*
Declare some global variables that are necessary across many functions:
MaxParticles - the maximum number of particles that could be displayed
FrameTime - the amount of time in seconds a single frame is expected to take
EmissionsPerFrame - the number of particles emitted from the match every frame
*/
const int MaxParticles = 1000;
__device__
const float FrameTime = 0.033f;
const int EmissionsPerFrame = 25;
#endif