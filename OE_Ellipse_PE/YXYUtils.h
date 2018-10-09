#ifndef Define_YxyUtils
#define Define_YxyUtils

#include <io.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <osgViewer/Viewer>
#include <AR/ar.h>

using namespace std;

namespace yxy
{
#pragma region FILE_OPER
	//Get all file names
	void GetAllFiles(string& path, vector<string>& files);
	
	//Get all file names with target format
	void GetAllFormatFiles(string& path, string& format, vector<string>& files);

	//Get file Numbers
	void GetFileLines(string& path, int& lineNum);
#pragma endregion FILE_OPER

#pragma region TYPES
//预留做格式转换
	osg::ref_ptr<osg::Image> ConvertCVMat2OsgImg(cv::Mat mat, bool convertToRGB = true);
	cv::Mat ConvertOsgImg2CVMat(osg::ref_ptr<osg::Image> & Img, bool convertToBGR = true);
#pragma endregion TYPES

#pragma region CV_MAT_TOOL
	//Copy 2 image cv::Mat
	void SwapMat(cv::Mat& src, cv::Mat& dst);
	//列数相等的情况下，按照列拼接矩阵
	void MergeMatByCol(cv::Mat& temp, cv::Mat& input, cv::Mat& result);
#pragma endregion CV_MAT_TOOL

#pragma region ROI_SELECTOR
	class RoiSelector
	{
		enum MOUSE_SELECT_STATE
		{
			NOT_SET =0,
			IN_PROCESS=1,
			SET=2
		};

	public:
		RoiSelector(void);
		~RoiSelector(void);
		cv::Mat RoiImg;
		cv::Rect RoiRect;

	private:
		cv::Mat src;
		cv::Mat gray;
		int SrcWidth;
		int SrcHeight;
		double scale;
		string winName;

		cv::Rect rect;
		MOUSE_SELECT_STATE rectState;

		int origin_x;
		int origin_y;
		int rect_width;
		int rect_height;

		int x;
		int y;
		int width;
		int height;

	public:

		void SetSrcFrame(cv::Mat OutSrc);
		cv::Rect GetRoiRect() const { return RoiRect; }

		//清除矩形，重新画
		void reset();
		//显示矩形
		void DrawRect();
		//鼠标事件
		void MouseClick(int event, int x, int y, int flags, void *param);
		static void on_mouse( int event, int x, int y, int flags, void* param );
		//选框
		void setRoi();
	};
#pragma endregion ROI_SELECTOR

#pragma region ALG
	//统计选票
	class MaxCounter
	{
	public:
		void AddCount(int index, int numofCount = 1);
		int GetMaxIndex();
	protected:
		map<int, int> _counterMap;
	};
#pragma endregion ALG

//machine learning 指标计算， 包括 precision, recall, accuracy, true positive rate, false positive rate, F score, accuracy
#pragma region Validator
	class Validator
	{
	public:
		Validator(){fp=tp=fn=tn=0.0f;}
		Validator( cv::Mat & res, cv::Mat & lab, float& thresh);
		~Validator(void);

	private:
		float resLab;
		float resScore;
		float gtLab;
		float CurrentThresh;
		float fp,tp,fn,tn;
		float testSize;
		float sampleSize;
		float getPrecision(int i=1);
		float getRecall(int i=1);
		float getF1(int i=1);
		float getZeroOne();
		float getTPR();
		float getFPR();


	public:

		enum validType{Current,Final};
		validType _validType;
		void display();
		void save(ofstream& ofile);
		void countFinal(cv::Mat& result, 
			cv::Mat& label_gt, 
			float thresh, 
			float& tp, float & fp, float & tn, float & fn);
		void countCurrent(cv::Mat& result, 
			cv::Mat& label_gt, 
			float& thresh, 
			int& iCurrent);
	};
#pragma endregion Validator

//cv3 camera calibration || param read || undistort || get projection param 
#pragma region CALIB
	class CameraParam
	{
	public:
		void ReadCVCalibParam(string& cvCalibFileName);
		void IniValidUndistort();
		void UndistortImg(cv::Mat& src,cv::Mat & result);
		void UndistortImgValid(cv::Mat & src, cv::Mat & result);
		void UndistrotImgCropped(cv::Mat & src, cv::Mat & result);
		void UndistrotImgResized(cv::Mat & src, cv::Mat & result);
		//intrinsic 可以选 cameraMatrix, 也可以选 newIntrinsic
		static void GetOsgVirtualCamParam(cv::Mat intrinsic, int width, int height, float& left, float &right, float& bottom, float& top);
		void GetARTKCamParam(string& arCalibFileName, int dist_version);
		
	public:
		ARParam GetARParam() { return param; }
		cv::Mat GetCameraMatrix() const { return cameraMatrix; }
		void SetCameraMatrix(cv::Mat val) { cameraMatrix = val;}
		cv::Mat GetNewIntrinsic() const { return newIntrinsic; }
		void SetNewIntrinsic(cv::Mat val) { newIntrinsic = val; }
	private:
		cv::Mat cameraMatrix, distCoeffs;
		cv::Mat temp,undistort;
		cv::Size imageSize;
		cv::Mat newIntrinsic, map1, map2;
		cv::Rect validroi;
		ARParam  param;
	private:
 		static ARdouble getSizeFactor(ARdouble dist_factor[], int xsize, int ysize, int dist_function_version);
		static void convParam(float intr[3][4], float dist[4], int xsize, int ysize, ARParam *param, int dist_version);
	};
#pragma endregion CALIB
}

#endif