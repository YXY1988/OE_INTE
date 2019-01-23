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

/*����vector<Contours>,���vector<Ellipse>,������Candidate Ellipses
�ѵ㣺
1. ��ȷ��ͼ�����Ƿ�����Բ�����������㷨pre-trained�����Բ���SVM��
2. ���contours��Ӧһ��ellipse
3. ����multi ellipses������Ż���ȷ��candidate ellipses*/

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

	//TODO ��Ϊʲô��ֱ����struct���֣������ֵ�����һ�����֣��ܶ�ط�����������typedef ��instance��һ��һ��������
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

		//���� Sampson error ��Ҫ����Բ����ת��Ϊ��Բ���̾���
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



