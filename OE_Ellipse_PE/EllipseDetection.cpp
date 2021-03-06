#include "EllipseDetection.h"
#include <Eigen/Dense>
#include <Eigen/LU>
//#define DETAIL
namespace ElliFit {
	typedef Eigen::MatrixXd matXd;
	/*
	辅助函数，移除相对伪椭圆中心的outliers points
	Helper function for the ellipse fit where the outliers
	are somewhat removed by using the median as the pseudo-center
	of the ellipse.
	*/
	static void findMaxMinXY(const pointSet& contour,
		const int& rows,
		const int& cols,
		cv::Point& minpt,
		cv::Point& maxpt)
	{
		int maxX = 0,
			maxY = 0,
			minY = rows,
			minX = cols;

		for (unsigned int k = 0; k < contour.size(); ++k)
		{
			cv::Point currPoint = contour[k];
			int x = currPoint.x;
			int y = currPoint.y;

			if (x < minX)
			{
				minX = x;
			}

			if (y < minY)
			{
				minY = y;
			}

			if (x > maxX)
			{
				maxX = x;
			}

			if (y > maxY)
			{
				maxY = y;
			}
		}

		minpt = cv::Point(minX, minY);
		maxpt = cv::Point(maxX, maxY);
	}

	/*
	生成中间矩阵
	Generating the matrices for the operations:
	output matrices are A &B which are the matrices
	generated for the ellipse fitter.
	*/
	static void generateMats(const pointSet& contour,
		matXd& A,
		matXd& B)
	{
#pragma omp parallel for
		for (unsigned int i = 0; i < contour.size(); ++i)
		{
			A(i, 0) = static_cast<double>(contour[i].x * contour[i].x);
			A(i, 1) = static_cast<double>(2.0 * contour[i].x * contour[i].y);
			A(i, 2) = static_cast<double>(-2.0 * contour[i].x);
			A(i, 3) = static_cast<double>(-2.0 * contour[i].y);
			A(i, 4) = static_cast<double>(-1.0);

			B(i, 0) = static_cast<double>(-1.0 *contour[i].y * contour[i].y);
		}
	}

	/*
	用基于ransac的算法拟合椭圆
	The ellipse fit is a RANSAC based algorithm which will be
	*/
	static void ellipseMatrixOps(matXd& A,
		matXd& B,
		matXd& phi,
		Ellipse& ell)
	{
		/*
		Note that the MATLAB value uses extra large floating points and
		because of truncation the MATLAB and Eigen matrix multiplication
		values vary a little bit. We shall see how much destruction this
		causes in the eventual ellipse fit but I suspect not much since we
		have 8-bit images and any rounding should come to the same values..
		*/
		phi = ((A.transpose() * A).inverse()) * A.transpose() * B;
		ell.residue = ((A * phi) - B).norm() / B.norm();
	}

	/*
	Add constant negative value to each element of input vector.
	Put another way, we subtract the mean to center the ellipse
	better.
	*/
	static inline void addConstantToVector(const cv::Point& meanVal,
		pointSet& outliers)
	{
		for (int p = 0; p < outliers.size(); ++p)
		{
			outliers[p].x -= meanVal.x;
			outliers[p].y -= meanVal.y;
		}
	}


	/*
	From the ellipse matrices that are previously generated, the
	matrix Phi is computed which is then input to this function.
	The output is a set of ellipse parameters that define the ellipse
	which encodes the pupil.
	*/
	static void computeEllipseParams(const matXd& phi,
		Ellipse& ell)
	{
		double a = phi(0, 0),
			b = phi(1, 0),
			c = phi(2, 0),
			d = phi(3, 0),
			e = phi(4, 0),
			x, y;

		x = (c - (d * b)) / (a - (b * b));
		y = ((a * d) - (c * b)) / (a - (b*b));

		double temp1 = sqrt(((1 - a)*(1 - a) + (4 * (b * b)))),
			temp2 = e + (y * y) + (x*x*a) + (2 * b),
			temp3 = 1 + a;

		ell.orientationAngle = -0.5 * atan2(2 * b, 1 - a);

		ell.minorRadius = sqrt(fabs((2 * temp2) / (temp3 + temp1)));
		ell.majorRadius = sqrt(fabs((2 * temp2) / (temp3 - temp1)));

		ell.ellipseCentroid.x = static_cast<int>(x);
		ell.ellipseCentroid.y = static_cast<int>(y);
	}

	/*
	For the input points, we generate an ellipse
	fitting via least squares.
	*/
	void EllipseDetection::ellipseFit(int& rows,
		int& cols,
		pointSet& inliers,
		Ellipse& ell)
	{
		cv::Point minp, maxp, meanp;
		findMaxMinXY(inliers, rows, cols, minp, maxp);

		meanp.x = (minp.x + maxp.x) / 2;
		meanp.y = (minp.y + maxp.y) / 2;
		addConstantToVector(meanp, inliers);

		matXd A(inliers.size(), 5);
		matXd B(inliers.size(), 1);
		generateMats(inliers, A, B);

		{
			matXd Phi;
			ellipseMatrixOps(A, B, Phi, ell);
			computeEllipseParams(Phi, ell);
		}

		ell.ellipseCentroid.x += meanp.x;
		ell.ellipseCentroid.y += meanp.y;
	}

	static bool EllipseSort(const Ellipse &v1, const Ellipse &v2)
	{
		return v1.majorRadius > v2.majorRadius;
	}
	
	EllipseDetection::EllipseDetection()
	{
		filter_radius = 100;
	}
	EllipseDetection::~EllipseDetection()
	{
	}

	void EllipseDetection::SetSrcImg(cv::Mat& srcImage)
	{
		srcImage.copyTo(m_SrcImg);
		srcImage.copyTo(m_Original);
	}

	void EllipseDetection::DetectEllipses(cv::Mat& binImg, vector<vector<cv::Point>>& contours)
	{
		int iRows = binImg.rows;
		int iCols = binImg.cols;
		Ellipse tempell;
		cv::RotatedRect rotrect;
		vector<cv::Point> tempcontour;
		
		//第一轮，从 contours 里面 选出 可能是椭圆的轮廓
		vector<Ellipse> ellParams;
		vector<vector<cv::Point>> ellContours;
		int iContoursNum = contours.size();
		if (iContoursNum == 0)
		{
			cout << "No curves for ell detection" << endl;
			return;
		}

		bool bIsPtEnough = false;

#pragma omp parallel for
		for (int i = 0;i < iContoursNum;i++)
		{
			tempcontour = contours[i];
			CvScalar color = CV_RGB(rand() & 255, rand() & 255, rand()&255);
			ellipseFit(iRows, iCols, tempcontour, tempell);
			double FlatRate = tempell.majorRadius / tempell.minorRadius;
			cout << tempell.residue << ""<< tempell.majorRadius<<endl;
			if(tempell.residue>0.2
				||isnan(tempell.residue)
				||tempell.majorRadius<filter_radius
				||FlatRate>5) 
				continue;
			else
			{
 				//cout << "the initial residue is:" << tempell.residue << endl;
//    				rotrect.center = cv::Point(tempell.ellipseCentroid.x, tempell.ellipseCentroid.y);
//    				rotrect.angle = static_cast<float>(RAD2DEG * tempell.orientationAngle);
//   				rotrect.size.width = static_cast<float>(2 * tempell.majorRadius);
//    				rotrect.size.height = static_cast<float>(2 * tempell.minorRadius); //draw in red
//   				cv::ellipse(m_SrcImg, rotrect, cv::Scalar(0, 0, 255), 1, 8);
 				//cv::ellipse(m_SrcImg, rotrect, color, 2, 8); //draw in random color
				ellParams.push_back(tempell);  
				ellContours.push_back(contours[i]);//只保留成功拟合椭圆的轮廓
			}
		}
#ifdef DETAIL
		cv::imshow("First detection", m_SrcImg);
		cv::waitKey(50);
		cv::imwrite("../Data/ellfigure/elldetect.jpg", m_SrcImg);
		cv::waitKey(50);
#endif
		//第二轮 arc grouping
		//ell grouping 轮廓聚合，将vector<vector<cv::Point>> ellContours内的轮廓两两遍历，符合条件的聚合，不符合仍然保持独立
		vector<vector<cv::Point>>::iterator GroupIt1 = ellContours.begin();
		vector<vector<cv::Point>>::iterator GroupIt2;
		bool bIsEll = false;
		cv::Mat m_tempell;
		while (GroupIt1 != ellContours.end())
		{
			//cout << "遍历" << endl;
			vector<cv::Point> TempGroupedContour;
			GroupIt2 = GroupIt1 + 1;
			while (GroupIt2 != ellContours.end())
			{
				double dGroupedres = 0;
				TempGroupedContour = *GroupIt1;
				TempGroupedContour.insert(TempGroupedContour.end(), GroupIt2->begin(), GroupIt2->end());
				tempcontour = TempGroupedContour;
				ellipseFit(iRows, iCols, tempcontour, tempell);
				dGroupedres = tempell.residue;
				m_tempell = EllParam2EllMat(tempell);
				bIsEll = IsContourAnEll(TempGroupedContour, m_tempell);
	
				if (dGroupedres < 0.2 //0.2已经很好了
					&&!isnan(tempell.residue)
					&&bIsEll==true)//如符合聚合条件
				{
					//cout << "聚合" << endl;
					//cout << "dGroupedres = " << dGroupedres << endl;
					*GroupIt1 = TempGroupedContour;//合并两个轮廓到前一个轮廓里
					GroupIt2=ellContours.erase(GroupIt2);//擦除后一个轮廓
				}
				else
				{
					GroupIt2++;
				}
			}
			GroupIt1++;
		}

		//第三轮 ell filter
		vector<vector<cv::Point>> GroupedContours=ellContours;
		vector<Ellipse> GroupedElls;
		vector<cv::Mat>  GroupedEllMats;
		vector<cv::Rect> GroupedEllRects;
		cv::Rect tempRect;
		double PeriThresh = 0.1;
		float tempMajorRadius = 10.0;

#pragma omp parallel for
		for (int j = 0;j < GroupedContours.size();j++)
		{
			CvScalar color = CV_RGB(rand() & 255, rand() & 255, rand() & 255);
			tempcontour = GroupedContours[j];
			ellipseFit(iRows, iCols, tempcontour, tempell);
			m_tempell = EllParam2EllMat(tempell);
			bIsEll = IsContourAnEll(GroupedContours[j], m_tempell);
			bIsPtEnough = IsEnoughPtsOnEll(tempcontour.size(), tempell.majorRadius, tempell.minorRadius, PeriThresh);

			{
				rotrect.center = cv::Point(tempell.ellipseCentroid.x, tempell.ellipseCentroid.y);
				rotrect.angle = static_cast<float>(RAD2DEG * tempell.orientationAngle);
				rotrect.size.width = static_cast<float>(2 * tempell.majorRadius);
				rotrect.size.height = static_cast<float>(2 * tempell.minorRadius);
				//ellipse(m_SrcImg, rotrect, color, 2, 8);
// 				ellipse(m_SrcImg, rotrect, cv::Scalar(0,255,0), 2, 8);
// 				cv::imshow("Grouped",m_SrcImg);
// 				cv::waitKey(50);
// 				cv::imwrite("../Data/ellfigure/grouped.jpg", m_SrcImg);
// 				cv::waitKey(50);
			}

			
			double FlatRate = tempell.majorRadius / tempell.minorRadius;
			if(tempell.residue>0.2||isnan(tempell.residue)
				||bIsEll==false||bIsPtEnough==false||FlatRate>5
				/*||tempell.majorRadius<100*/)
				continue;
			else
			{
#ifdef DETAIL
				cout << "Groupedres = " << tempell.residue << endl;
#endif
// 				if (tempell.majorRadius < tempMajorRadius)
// 					continue;
// 				else
// 				{
// 					tempMajorRadius = tempell.majorRadius;
					rotrect.center = cv::Point(tempell.ellipseCentroid.x, tempell.ellipseCentroid.y);
					rotrect.angle = static_cast<float>(RAD2DEG * tempell.orientationAngle);
					rotrect.size.width = static_cast<float>(2 * tempell.majorRadius);
					rotrect.size.height = static_cast<float>(2 * tempell.minorRadius);
					//ellipse(m_SrcImg, rotrect, cv::Scalar(0, 255, 0), 2, 8);
					tempRect = rotrect.boundingRect();
					if (tempRect.x < 0)
						tempRect.x = 0;
					if (tempRect.y < 0)
						tempRect.y = 0;
					if (tempRect.x + tempRect.width > m_SrcImg.cols)
						tempRect.width = m_SrcImg.cols - tempRect.x;
					if (tempRect.y + tempRect.height > m_SrcImg.rows)
						tempRect.height = m_SrcImg.rows - tempRect.y;
					GroupedElls.push_back(tempell);
					GroupedEllMats.push_back(m_tempell);
					GroupedEllRects.push_back(tempRect);
#ifdef DETAIL
					cout << "---------------我是第三轮------------------" << endl;
					cout << "ell center is: " << rotrect.center << endl;
					cout << "ell angle is: " << tempell.orientationAngle << endl;
					cout << "ell a b is: " << tempell.majorRadius << "," << tempell.minorRadius << endl;
					cout << "rotrect center is: " << rotrect.center << endl;
					cout << "rotrect angle is: " << rotrect.angle << endl;
					cout << "rotrect size is: " << rotrect.size << endl;
					cout << "Ell mat is: " << endl << m_tempell << endl;
#endif
					//ellipse(m_SrcImg, rotrect, cv::Scalar(0, 255, 0), 2, 8);
					//cv::imshow("GroupedFinal", m_SrcImg);
					//cv::waitKey(50);
					//imwrite("../Data/Out/Ellipse.jpg", m_SrcImg);

					
					/*cv::imwrite("../Data/ellfigure/groupedfinal.jpg", m_SrcImg);
					cv::waitKey(50);*/
				/*}*/
			}
		}
		if (GroupedElls.size() > 3)
		{
			std::sort(GroupedElls.begin(), GroupedElls.end(), EllipseSort);
			
			m_ellipses.push_back(GroupedElls[0]);
			m_ellipses.push_back(GroupedElls[1]);
			m_ellipses.push_back(GroupedElls[2]);
			m_ellRects.clear();
			m_ellMats.clear();
			//m_ellRects.assign(GroupedEllRects.begin(),GroupedEllRects.begin()+3);
			//m_ellMats.assign(GroupedEllMats.begin(),GroupedEllMats.begin()+3);
		}
		else 
		{
			std::sort(GroupedElls.begin(), GroupedElls.end(), EllipseSort);
			m_ellipses = GroupedElls;
			m_ellRects.clear();
			m_ellMats.clear();
		}
		for (int i = 0; i < m_ellipses.size(); ++i)
		{
			tempell = m_ellipses[i];
			rotrect.center = cv::Point(tempell.ellipseCentroid.x, tempell.ellipseCentroid.y);
			rotrect.angle = static_cast<float>(RAD2DEG * tempell.orientationAngle);
			rotrect.size.width = static_cast<float>(2 * tempell.majorRadius);
			rotrect.size.height = static_cast<float>(2 * tempell.minorRadius);
			ellipse(m_SrcImg, rotrect, cv::Scalar(0, 255, 0), 2, 8);
			m_tempell = EllParam2EllMat(tempell);
			tempRect = rotrect.boundingRect();
// 			if (tempRect.x - 0.5*tempRect.width < 0)
// 				tempRect.x = 0;
// 			else
// 				tempRect.x = tempRect.x - 0.5*tempRect.width;
// 			if (tempRect.y - 0.5*tempRect.height < 0)
// 				tempRect.y = 0;
// 			else
// 				tempRect.y = tempRect.y - 0.5*tempRect.height;
// 			if (tempRect.x + 1.5*tempRect.width > m_SrcImg.cols)
// 				tempRect.width = m_SrcImg.cols - tempRect.x;
// 			else
// 				tempRect.width = tempRect.width *2;
// 			if (tempRect.y + 1.5*tempRect.height  > m_SrcImg.rows)
// 				tempRect.height = m_SrcImg.rows - tempRect.y;
// 			else
// 				tempRect.height = tempRect.height*2;
			if (tempRect.x < 0)
				tempRect.x = 0;
			if (tempRect.y < 0)
				tempRect.y = 0;
			if (tempRect.x + tempRect.width > m_SrcImg.cols)
				tempRect.width = m_SrcImg.cols - tempRect.x;
			if (tempRect.y + tempRect.height > m_SrcImg.rows)
				tempRect.height = m_SrcImg.rows - tempRect.y;
			
			if (i == 0)
			{
				cv::rectangle(m_SrcImg,
					cv::Point(tempRect.x, tempRect.y),
					cv::Point(tempRect.x + tempRect.width, tempRect.y + tempRect.height),
					cv::Scalar(0, 0, 255),
					2);
				cv::waitKey(10);
			}
			m_ellMats.push_back(m_tempell);
			m_ellRects.push_back(tempRect);
			//cv::imshow("GroupedFinal", m_SrcImg);
			//cv::waitKey(50);
		}
		return;
	}
	void EllipseDetection::DrawEllipses()
	{
		if (m_Original.empty() || m_ellipses.size() == 0)
		{
			cout << "no image or no ellipse detected" << endl;
			return;
		}
			
		Ellipse tempell;
		cv::RotatedRect rotrect;
		for (int i = 0;i < m_ellipses.size();i++)
		{
			tempell = m_ellipses[i];
			rotrect.center = cv::Point(tempell.ellipseCentroid.x, tempell.ellipseCentroid.y);
			rotrect.angle = static_cast<float>(RAD2DEG * tempell.orientationAngle);
			rotrect.size.width = static_cast<float>(2 * tempell.majorRadius);
			rotrect.size.height = static_cast<float>(2 * tempell.minorRadius);
			ellipse(m_Original, rotrect, cv::Scalar(0, 255, 0), 2, 8);
		}
#ifdef DETAIL
		imshow("EllResult", m_Original);
		cv::waitKey(10);
#endif
		return;
	}

	bool EllipseDetection::IsContourAnEll(vector<cv::Point> pts, cv::Mat m_ell)
	{
		bool IsEll = false;
		double avrdist = 0;
		double maxdist = 0;
		DistContourToEll(pts, m_ell, avrdist, maxdist);
		//if (avrdist < 0.5&&maxdist < 1.5)
		if (avrdist < 2 && maxdist < 5)
		{
			IsEll = true;
 			/*cout << "符合avr和max约束： " << endl;
			cout << "当前轮廓的平均sampson误差为：" << avrdist<< endl;
 			cout << "当前轮廓的单点最远距离为：" << maxdist << endl;*/
		}
		else
		{
			IsEll = false;
			/*cout << "不符合avr和max约束： " << endl;
			cout << "当前轮廓的平均sampson误差为：" << avrdist << endl;
			cout << "当前轮廓的单点最远距离为：" << maxdist << endl;*/
		}
		return IsEll;
	}

	void EllipseDetection::DistContourToEll(vector<cv::Point> SingleContour, cv::Mat m_ell, double &avr, double &max)
	{
		int PtNum = SingleContour.size();
		double dist = 0;
		double distSum = 0;
		double distAvr = 0;
		double maxdist = 0;
		cv::Point Pt_farest;

#pragma omp parallel for
		for (int i = 0;i < PtNum;i++)
		{
			//circle(m_SrcImg, SingleContour[i], 1, Scalar(0, 0, 255), 1, 8);
			dist = DistPointToEll(SingleContour[i], m_ell);
			distSum = distSum + dist;
			if (dist > maxdist)
			{
				maxdist = dist;
				Pt_farest = SingleContour[i];
			}
			//contour tracing 的效果
			//cout << "当前点的 sampson 误差为： " << dist << endl;
			//imshow("single sampson", m_SrcImg);
			//waitKey(5);
		}
// 		circle(m_SrcImg, Pt_farest, 2, Scalar(0, 255, 255), 2, 8);
// 		imshow("single sampson", m_SrcImg);
// 		waitKey(5);
		distAvr = distSum / PtNum;   
		avr = distAvr;
		max = maxdist;
	}

	double EllipseDetection::DistPointToEll(cv::Point pt, cv::Mat m_ell)
	{
		double dist=0;
		cv::Mat m_point = (cv::Mat_<float>(3,1)<<pt.x,pt.y,1);

		// sampson 的分子
		cv::Mat m_numerator = m_point.t()*m_ell*m_point;
		float f_numerator = m_numerator.at<float>(0, 0);
		f_numerator = f_numerator * f_numerator;

		// sampson 的分母
		float e11 = m_ell.at<float>(0, 0);
		float e12 = m_ell.at<float>(0, 1);
		float e13 = m_ell.at<float>(0, 2);
		float e22 = m_ell.at<float>(1, 1);
		float e23 = m_ell.at<float>(1, 2);
		float J_x = 2 * e11*pt.x + 2 * e12*pt.y + 2 * e13;
		float J_y = 2 * e22*pt.y + 2 * e12*pt.x + 2 * e23;
		float f_denominator = 4 * (J_x * J_x + J_y * J_y);

		// sampson error
		dist = f_numerator / f_denominator; 
		return dist;
	}

	bool EllipseDetection::IsEnoughPtsOnEll(int ContourSize, float a, float b, float thresh)
	{
		bool IsEnough = false;
		double d_EllPerimeter = 0;
		d_EllPerimeter = 2 * PI*b + 4 * (a - b);

		double PtPercentage = 0;
		PtPercentage = ContourSize / d_EllPerimeter;
		if (PtPercentage > thresh)
		{
			IsEnough = true;
		/*	cout << "符合周长比约束： " << endl;
			cout << "当前轮廓比例为：" << PtPercentage << endl;*/
		}
		else
		{
			IsEnough = false;
			/*cout << "不符合周长比约束： " << endl;
			cout << "当前轮廓比例为：" << PtPercentage << endl;*/
		}
		return IsEnough;
	}

	cv::Mat EllipseDetection::EllParam2EllMat(Ellipse ell)
	{
		double gamma, a, b, cosr, sinr;
		cv::Point2f ell_center_xy,ell_center_uv;

		gamma = ell.orientationAngle;
		a = ell.majorRadius;
		b = ell.minorRadius;
		ell_center_xy = ell.ellipseCentroid;
		cosr = cos(gamma);
		sinr = sin(gamma);
		
		//求 uv相对xy系的变换
		cv::Mat m_rot_uv2xy = (cv::Mat_<float>(3, 3) <<
			cosr, sinr, 0,
		   -sinr, cosr, 0,
			0, 0, 1);
// 		cout << "the rot cv::Mat from uv coord to xy coord is: " << endl
// 			<< " " << m_rot_uv2xy << endl;

		//uv系下的原点
		cv::Mat m_rot_center = m_rot_uv2xy(cv::Range(0,2), cv::Range(0,2));
		//cout << "the m_rot_center is: " << endl
		//<< "" << m_rot_center << endl;

		//格式一样即可
		cv::Mat m_ell_center_xy = cv::Mat(ell_center_xy);
		//cout << "m_ell_center_xy" << endl << m_ell_center_xy << endl << endl;
		cv::Mat m_ell_center_uv= m_rot_center * m_ell_center_xy;
		//cout << "m_ell_center_uv" << endl << m_ell_center_uv << endl << endl;
		float u0 = m_ell_center_uv.at<float>(0,0);
		float v0 = m_ell_center_uv.at<float>(1,0);
		//cout << "uv coord is : " << u0 << " , " << v0 << endl << endl;
		//ell_center_uv.x = m_ell_center_uv.at()

		float c11, c12, c13, c21, c22, c23, c31, c32, c33;
		c11 = 1 / (a*a);
		c12 = 0;
		c13 = -u0 / (a*a);
		c21 = c12;
		c22 = 1 / (b*b);
		c23 = -v0 / (b*b);
		c31 = c13;
		c32 = c23;
		c33 = u0 * u0 / (a*a) + v0 * v0 / (b*b) - 1;

		cv::Mat mElluv = (cv::Mat_<float>(3, 3) << c11, c12, c13, c21, c22, c23, c31, c32, c33);
		//cout << "ell in uv coord is : " << mElluv << endl << endl;

		cv::Mat mEllxy = m_rot_uv2xy.t()*mElluv*m_rot_uv2xy;
		//cout << "ell in xy coord is : " << mEllxy << endl << endl;
		return mEllxy;
	}
}

	
