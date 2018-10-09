/*输入图像，获取自适应边缘
输入 cv::Mat img, 输出 vector<Contours>
难点：随着相机运动，contour连续性变差，参数怎么设置，使得ELLipse特征明显？预处理剔除直线、零星点用不用？*/
#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <float.h>

using namespace std;

struct region
{
	double val;  /* pixel value */
	double w;    /* absolute value of the lateral distance to the arc */
	int reg;     /* number of the lateral region to which it belongs: 1 or 2 */
};

/*-----structure to store a parameterization of an arc of circle---*/
/* structure to store a parameterization of an arc of circle
*/
struct arc_of_circle
{
	int is_line_segment;
	double a, b, c; /* a*x+b*y+c = 0 -> line through the arc endpoints */
	double d;     /* b*x-a*y+d = 0 -> orthogonal to previous through midpoint */
	double len;   /* arc length */
	double xc, yc, radius;     /* center and radius of circle containing the arc */
	double ang_ref, ang_span;
	int dir;
	int bbx0, bby0, bbx1, bby1; /* bounding box */
};

class EdgeDetection
{
public:
	EdgeDetection();
	~EdgeDetection();
	void SetSrcImg(cv::Mat& srcImage);
	cv::Mat GetSrcImg() { return m_SrcImg; };
	
	cv::Mat BinContourDetection();
	cv::Mat CannyContourDetection(int low, int high);
	cv::Mat IPOLContourDetection();
	
	void ContourRefinement();//preprocessing, filter out points, lines,调用PreProcess
	void DrawContourOnSrc(cv::Mat& binImg);//show bin, colored 
	cv::Mat GetMatBinaryContour() { return m_MatBinaryContour; }
	cv::Mat GetMatColoredContour() { return m_MatColoredContour; }
	vector<vector<cv::Point>> GetContoursFromBin(cv::Mat& binImg);

	void PreProcess(cv::Mat& binImg, cv::Mat& dstImg);

	//point level
	cv::Mat FilterSpur(cv::Mat& binImg);
	cv::Mat FilterIsolated(cv::Mat& binImg);
	cv::Mat FilterJunction(cv::Mat& binImg);

	//Single contour level
	cv::Mat FilterLength(cv::Mat& binImg, int length);
	cv::Mat FilterTurning(cv::Mat& binImg,int length);
	
	//contours level
	cv::Mat FilterLines(cv::Mat& binImg);
	bool IsContourALine(vector<cv::Point> pts,float LineThresh);
	double DistOfLine(vector<cv::Point> ContourLine, cv::Point StartPt, cv::Point EndPt);
	double DistPointToLine(cv::Point pt, cv::Point StartPt, cv::Point EndPt);

	vector<vector<cv::Point>> GetFinalContours() {return m_vContours; }

private:
	vector<vector<cv::Point>> m_vContours;
	cv::Mat m_SrcImg;
	cv::Mat m_MatBinaryContour;//binary contour
	cv::Mat m_MatColoredContour;//gray src with green contour

private:
	double * g_x;          /* x[n] y[n] coordinates of result contour point n */
	double * g_y;
	int * g_curve_limits;  /* limits of the curves in the x[] and y[] */
	int g_N;               /* result: N contour points */
	int g_M;	          /* result: forming M curves */
	double dog_rate = 1.6;    /* DoG sigma rate to approx. Laplacian of Gaussian
							  optimal value 1.6 [Marr-Hildreth 1980] */
	double sigma_step = 1.8;  /* sigma to sampling step rate in Gaussian sampling
							  optimal value 0.8 [Morel-Yu 2011] */
	double log_eps = 0.0;     /* log10(epsilon), where epsilon is the mean number
							  of false detections one can afford per image */
	int num_w = 3;            /* number of arc widths to be tested */
	double fac_w = sqrt(2.0); /* arc width factor  */
	double min_w = sqrt(2.0); /* minimal arc width */

							  /*---structure to store an element of the lateral regions of pixels to an arc operator---*/
							  /* structure to store an element of the lateral regions of pixels to an arc operator.
							  for each pixel it stores its value, the lateral distance to the arc defining the operator,
							  and to which of the two lateral regions belongs
							  */

	void smooth_contours(double ** x, double ** y, int * N,
		int ** curve_limits, int * M,
		double * image, int X, int Y, double Q);
};

static bool SortByPtx(const cv::Point &p1, const cv::Point &p2)
{
	return p1.x < p2.x;
}

static double rad2degree(double rad)
{
	double degree;
	degree = rad * 180 / 3.1415;
	return degree;
}