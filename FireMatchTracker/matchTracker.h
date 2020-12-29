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
int* track(Mat frame);
Mat averageFrame(Mat buffer[3]);
#endif