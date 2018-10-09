#include <windows.h>
#include "commonlibs.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include "EdgeDetection.h"
#include "EllipseDetection.h"
#include <fstream>


using namespace std;
using namespace ElliFit;

/*

int main()
{
	string WindowName = "test";
	cv::Mat src,test,contour,temp,result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	ofstream ofile1;
	ofile1.open("../Data/time.txt");

#pragma region IMAGE_TEST
	float resizescale =1;
	src = imread("../Data/cvundistort0709.bmp", 1);
	cv::Size dsize = cv::Size(src.cols*resizescale, src.rows*resizescale);
	resize(src, test, dsize);
	_EdgeDetector.SetSrcImg(test);
	//_EllipseDetector.SetSrcImg(test);

	double t_begin = cv::getTickCount();
	//contour = _EdgeDetector.IPOLContourDetection();
	//contour = _EdgeDetector.BinContourDetection();
	contour = _EdgeDetector.CannyContourDetection(50,150);
	//imshow("contour", contour);
	double t_end = cv::getTickCount();
	double t_cost = (t_end - t_begin) / cv::getTickFrequency() * 1000;
	cout << "轮廓检测耗时(ms)： " << t_cost << endl;

	//由于canny没有薄边处理，junction, spur, isolated不能和canny连用。
	temp = contour.clone();
//  	temp = _EdgeDetector.FilterJunction(temp);
//  	temp = _EdgeDetector.FilterSpur(temp);
//  	temp = _EdgeDetector.FilterIsolated(temp);
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);
	//imshow("temp", temp);
	edges = _EdgeDetector.GetFinalContours();

 	//cv::Mat colortemp;
 	//cvtColor(temp, colortemp, CV_GRAY2BGR);
 	//_EllipseDetector.SetSrcImg(colortemp);
	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.DetectEllipses(temp, edges);

	double t_refine = cv::getTickCount();
	t_cost = (t_refine - t_end) / cv::getTickFrequency() * 1000;
	cout << "轮廓预处理耗时(ms)： " << t_cost << endl;

	result = _EllipseDetector.GetSrcImg();
	//namedWindow(WindowName, CV_WINDOW_NORMAL);
	namedWindow(WindowName, 1);
	imshow(WindowName, result);
	waitKey(0);
#pragma endregion IMAGE_TEST

#pragma region VIDEO_TEST
	/ * VideoCapture cap;
	VideoWriter writer;
	string VideoFileName = "../Data/cover.avi";
	int stFrmNum = 0;
	cap.open(VideoFileName);
	int capwidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int capheight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	cap.set(CV_CAP_PROP_POS_FRAMES, stFrmNum);
	writer.open("../Data/result.avi",CV_FOURCC('M','J','P','G'),25.0,
		cv::Size(capwidth,capheight),true);
	int frm_contour = 0;
	double avr_preprocesstime = 0;
	double total_preprocesstime = 0;
	while (cap.isOpened())
	{
		cap >> test;
		if (test.empty())
			break;
		frm_contour++;
		_EdgeDetector.SetSrcImg(test);
		_EllipseDetector.SetSrcImg(test);

		double t_begin = cv::getTickCount();
		//contour = _EdgeDetector.IPOLContourDetection();
		//contour = _EdgeDetector.BinContourDetection();
		contour = _EdgeDetector.CannyContourDetection();
 		double t_end = cv::getTickCount();
 		double t_cost = (t_end - t_begin) / cv::getTickFrequency() * 1000;
 		//cout << "轮廓检测耗时(ms)： " << t_cost << endl;
		//由于canny没有薄边处理，junction, spur, isolated不能和canny连用。
		temp = contour.clone();
		//temp = _EdgeDetector.FilterJunction(temp);
		//temp = _EdgeDetector.FilterSpur(temp);
		//temp = _EdgeDetector.FilterIsolated(temp);
		temp = _EdgeDetector.FilterTurning(temp, 10);
		temp = _EdgeDetector.FilterLines(temp);
		temp = _EdgeDetector.FilterLength(temp, 10);
		edges = _EdgeDetector.GetFinalContours();

// 		cv::Mat colortemp;
// 		cvtColor(temp, colortemp, CV_GRAY2BGR);
// 		_EllipseDetector.SetSrcImg(colortemp);
		_EllipseDetector.DetectEllipses(temp, edges);

 		double t_refine = cv::getTickCount();
 		t_cost = (t_refine - t_end) / cv::getTickFrequency() * 1000;
 		//cout << "轮廓预处理耗时(ms)： " << t_cost << endl;
		total_preprocesstime = total_preprocesstime + t_cost;
		avr_preprocesstime = total_preprocesstime / frm_contour;
		result = _EllipseDetector.GetSrcImg();
		namedWindow(WindowName, CV_WINDOW_NORMAL);
		imshow(WindowName, result);
		writer << result;
		//cout << "current frm number is: " << frm_contour << endl;
		//ofile1 << frm_contour << endl;
		waitKey(1);
	}
	cout << "平均处理时间((ms)： " << avr_preprocesstime << endl;
	ofile1 << "视频帧数： "<<frm_contour<<endl
		   <<"平均处理时间((ms)： " << avr_preprocesstime << endl;
	cap.release();
	writer.release();
	ofile1.close(); * /
#pragma endregion VIDEO_TEST
   return 0;
}
*/
