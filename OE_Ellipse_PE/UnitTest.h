#pragma once
#include "YXYUtils.h"
#include "EdgeDetection.h"
#include "EllipseDetection.h"
#include "PoseEstimation.h"
#include "MarkerValidator.h"
#include "SceneGenerator.h"

#include <vector>
#include <iostream>
#include <sstream>

#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace yxy;
using namespace std;
using namespace ElliFit;

//void CVCalibTest();
//void UndistortTest();
//void Cv2arTest();
//从单张图像中 检测椭圆，可获得    
void DetectMajorEllipses(cv::Mat & src);
void TestCoarsePose();
void TestImgConvertFunctions();
void TestSyntheticTemplateGeneration();
void RunAllTests();
Mat ReadCVPoseMatFromFile(string & filename);
void WriteCVPoseMatToFile(string & filename, Mat & outMat);
