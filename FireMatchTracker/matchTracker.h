#pragma once
#include <opencv2/core.hpp>
using namespace cv;
#include <glew.h>
#include <glfw3.h>
#include <glm.hpp>
#include <vec2.hpp>
#include <vec3.hpp>
#include <vec4.hpp>
#include <mat4x4.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>
#include <algorithm>

#ifndef _matchTracker_
#define _matchTracker_
/*
Define function template for the track function, for tracking the match tip
*/
void track(Mat frame, int *tip);

/*
Declared dimensions of the window captured through the webcam
*/
#define X 1280
#define Y 720
#endif