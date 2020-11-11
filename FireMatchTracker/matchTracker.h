#include <opencv2/core.hpp>
using namespace cv;
#pragma once
#ifndef _matchTracker_
#define _matchTracker_
Mat track(Mat frame);
Mat averageFrame(Mat buffer[3]);
#endif