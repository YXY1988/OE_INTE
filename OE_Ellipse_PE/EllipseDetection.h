#pragma once

#include "commonlibs.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <float.h>
#include <math.h>

using namespace std;

#define RAD2DEG 57.2957795
#define DEG2RAD 0.0174532925
#define PI 3.1415926

/*输入vector<Contours>,输出vector<Ellipse>,里面是Candidate Ellipses
难点：
1. 不确定图像中是否含有椭圆，尽量不让算法pre-trained，所以不用SVM了
2. 多个contours对应一个ellipse
3. 检测出multi ellipses后过滤优化，确定candidate ellipses*/

namespace ElliFit {

	typedef struct Point2i
	{
		int x,
			y;
	}
	point2i;

	typedef struct Point2d
	{
		double x,
			y;
	}
	point2d;

	typedef std::vector<cv::Point> pointSet;

	//TODO 问为什么不直接用struct名字，而是又单起了一个名字？很多地方都是这样，typedef 后instance了一个一样的名字
	typedef struct ellipseStruct
	{
		double residue,
			orientationAngle;
		cv::Point ellipseCentroid;
		double majorRadius,
			minorRadius;
	}
	Ellipse;

	class EllipseDetection
	{
	public:
		EllipseDetection();
		~EllipseDetection();
		void SetSrcImg(cv::Mat& srcImage);
		cv::Mat GetSrcImg() { return m_SrcImg; };

		void ellipseFit(int& rows,
			int& cols,
			pointSet& inliers,
			Ellipse& ell);
		void DetectEllipses(cv::Mat& binImg,vector<vector<cv::Point>>& contours);
		void DrawEllipses();//Filter ell
		bool IsContourAnEll(vector<cv::Point> pts, cv::Mat m_ell);
		void DistContourToEll(vector<cv::Point> SingleContour, cv::Mat m_ell, double &avr, double &max);
		double DistPointToEll(cv::Point pt, cv::Mat m_ell);
		bool IsEnoughPtsOnEll(int ContourSize, float a, float b,float thresh);

		//bool EllipseSort(const Ellipse &v1, const Ellipse &v2);

		//计算 Sampson error 需要把椭圆参数转换为椭圆方程矩阵
		cv::Mat EllParam2EllMat(Ellipse ell);
	
	public:
		std::vector<cv::Rect> GetEllRects() const { return m_ellRects; }
		void SetEllRects(std::vector<cv::Rect> val) { m_ellRects = val; }
		double GetFilter_radius() const { return filter_radius; }
		void SetFilter_radius(double val) { filter_radius = val; }
		vector<Ellipse> GetEllDetectionResult() { return m_ellipses; }
		vector<cv::Mat> GetEllMatResult() { return m_ellMats;}

	private:
		double filter_radius;
		vector<cv::Rect> m_ellRects;
		vector<Ellipse> m_ellipses;
		vector<cv::Mat> m_ellMats;
		cv::Mat m_SrcImg;
		cv::Mat m_Original;

	};
}



