#pragma once
#include <opencv2/core.hpp>
using namespace cv;
#include <glfw3.h>

#ifndef _matchTracker_
#define _matchTracker_
Mat track(Mat frame);
Mat averageFrame(Mat buffer[3]);
#endif