#include "PoseEstimation.h"
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/core/eigen.hpp>
//#define DETAIL
//#define SHOWIMG

PoseEstimation::PoseEstimation()
{
	m_ModelRadius = 0;
}

PoseEstimation::~PoseEstimation()
{
}

void PoseEstimation::Initialize(cv::Mat & Intrinsic, string & ModelPath, float & ModelRadius, cv::Mat & ObjectTransform)
{
	surf = SURF::create(); //TODO:替换实验室后来编译的OpenCV342库才能用
	m_Intrinsic = Intrinsic;
	m_ModelName = ModelPath;
	m_ModelRadius = ModelRadius;
	m_ObjectTransform = ObjectTransform;

	m_SyncGenerator.SetUseImgBgFlag(false);
	m_SyncGenerator.SetCameraIntrinsic(Intrinsic);
	m_SyncGenerator.SetModelName(m_ModelName);
	m_SyncGenerator.SetObjectSelfTransform(m_ObjectTransform);

	m_ARGenerator.SetCameraIntrinsic(Intrinsic);
	m_ARGenerator.SetModelName(m_ModelName);
	m_ARGenerator.SetObjectSelfTransform(m_ObjectTransform);
}

cv::Mat PoseEstimation::GenerateTemplateImg(cv::Mat & pose)
{
	cv::Mat result;
	cv::Mat OsgPoseMat = pose.t();
	m_SyncGenerator.SetPoseMat(OsgPoseMat);
	m_SyncGenerator.SetUseImgBgFlag(false);
	m_SyncGenerator.SetUseTransparent(false);
	m_SyncGenerator.SetUseWireframe(false);
	result = m_SyncGenerator.GetSyntheticImg();
	return result;
}

cv::Mat PoseEstimation::GenerateARImg(cv::Mat & pose, cv::Mat & bgImg)
{
	cv::Mat result;
	cv::Mat OsgPoseMat = pose.t();
	m_ARGenerator.SetPoseMat(OsgPoseMat);
	m_ARGenerator.SetUseImgBgFlag(true);
	m_ARGenerator.SetBgImgMat(bgImg);
	m_ARGenerator.SetUseTransparent(true);
	m_ARGenerator.SetUseWireframe(true);
	result = m_ARGenerator.GetSyntheticImg();
	return result;
}

float PoseEstimation::CalImgError(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates)
{
	float err;
	uint nChannels(images.size());
	vector<vector<float> > templatePixelIntensities(nChannels, vector<float>(pixelsOnTemplate.size()));
	vector<Mat> imageDx(nChannels), imageDy(nChannels);

	LucasKanade optimization;

	//Eigen::MatrixXf sdImages(pixelsOnTemplate.size(), nParam);
	vector<float>  errorImage(pixelsOnTemplate.size(), 0.0);
	
	for (int iChannel = 0; iChannel < nChannels; ++iChannel)
	{
		ComputeImageDerivatives(images[iChannel], imageDx[iChannel], imageDy[iChannel]);

		for (int iPoint = 0; iPoint < pixelsOnTemplate.size(); ++iPoint)
		{
			int pos = templates[iChannel].cols* pixelsOnTemplate.y[iPoint] + pixelsOnTemplate.x[iPoint];
			templatePixelIntensities[iChannel][iPoint] = ((float*)templates[iChannel].data)[pos];
		}
	}
	for (int iChannel = 0; iChannel < images.size(); ++iChannel)
	{
		optimization.ComputeResiduals(images[iChannel], templatePixelIntensities[iChannel], pixelsOnTemplate, errorImage); 
	}
	err = optimization.ComputeResidualNorm(errorImage);
	return err;
}

float PoseEstimation::CalImgErrorBySSD(cv::Mat & m_CapImg, cv::Mat & m_TmplImg)
{
	float err_ssd=50.0;

	//UnitTest SSD params are the same of LucaskanadeVideoSSDSpeedTest() of EPFL's UnitTest
	OptimizationParameters optimizationParameters;
	optimizationParameters.resTol = 1e-5;
	optimizationParameters.pTol = 5e-5;
	optimizationParameters.maxIter = 50;
	optimizationParameters.maxIterSingleLevel = 10;
	optimizationParameters.pyramidSmoothingVariance.push_back(7);
	optimizationParameters.presmoothingVariance = 1;
	optimizationParameters.nControlPointsOnEdge = 50;
	optimizationParameters.borderThicknessHorizontal = 100;
	optimizationParameters.borderThicknessVertical = 50;
	optimizationParameters.bAdaptativeChoiceOfPoints = 0;
	optimizationParameters.bNormalizeDescriptors = 1;

	ConvertImageToFloat(m_TmplImg);
	ConvertImageToFloat(m_CapImg);
	StructOfArray2di controlPoints = CreateDenseGridOfControlPoints(m_TmplImg.cols, m_TmplImg.rows);
	
	StructOfArray2di pixelsOnTemplate = controlPoints;
	Mat grayscaleFloatTemplate = m_TmplImg;
	Mat grayscaleFloatImage = m_CapImg;

	//SSDCalibration
	vector<Mat> templateDescriptorFields, imageDescriptorFields;
	templateDescriptorFields.push_back(grayscaleFloatTemplate.clone());
	imageDescriptorFields.push_back(grayscaleFloatImage.clone());
	
	if (optimizationParameters.bNormalizeDescriptors)
	{
		NormalizeImage(templateDescriptorFields[0]);
		NormalizeImage(imageDescriptorFields[0]);
	}
	
	if (optimizationParameters.pyramidSmoothingVariance.empty())
	{
		err_ssd = CalImgError(pixelsOnTemplate, imageDescriptorFields, templateDescriptorFields);
		return err_ssd;
	}

	//PyramidMultilevelCalibration
	vector< vector<Mat> > smoothedImages(optimizationParameters.pyramidSmoothingVariance.size());
	vector< vector<Mat> > smoothedTemplates(optimizationParameters.pyramidSmoothingVariance.size());

#pragma omp parallel for
	for (int i = 0;i < optimizationParameters.pyramidSmoothingVariance.size();i++)
	{
		smoothedImages[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], imageDescriptorFields);
		smoothedTemplates[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], templateDescriptorFields);
	}

	float err_temp=50.0;
	for (int iLevel = 0;iLevel < optimizationParameters.pyramidSmoothingVariance.size();iLevel++)
	{
		if (iLevel == optimizationParameters.pyramidSmoothingVariance.size() - 1)
		{
			optimizationParameters.maxIterSingleLevel = optimizationParameters.maxIter;
		}
		//cout << "Start using pyramid level no." << iLevel + 1 << endl;
		err_temp = CalImgError(pixelsOnTemplate, smoothedImages[iLevel], smoothedTemplates[iLevel]);
		if (err_ssd > err_temp)
			err_ssd = err_temp;
	}
	return err_ssd;
}

float PoseEstimation::CalImgErrorByGF(cv::Mat & m_CapImg, cv::Mat & m_TmplImg)
{
	float err_ssd = 50.0;

	//optimization parameters using GradientMagnitudeVideoTest of EPFL's Unittest
	OptimizationParameters optimizationParameters;
	optimizationParameters.resTol = 1e-5;
	optimizationParameters.pTol = 5e-5;
	optimizationParameters.maxIter = 50;
	optimizationParameters.maxIterSingleLevel = 10;
	optimizationParameters.pyramidSmoothingVariance.push_back(10);
	optimizationParameters.pyramidSmoothingVariance.push_back(5);
	optimizationParameters.presmoothingVariance = 1;
	optimizationParameters.nControlPointsOnEdge = 60;
	optimizationParameters.borderThicknessHorizontal = 100;
	optimizationParameters.borderThicknessVertical = 50;
	optimizationParameters.bAdaptativeChoiceOfPoints = 0;
	optimizationParameters.bNormalizeDescriptors = 1;

	ConvertImageToFloat(m_TmplImg);
	ConvertImageToFloat(m_CapImg);
	StructOfArray2di controlPoints = CreateGridOfControlPoints(m_TmplImg, 30, 0.0f, 0.0f);

	StructOfArray2di pixelsOnTemplate = controlPoints;
	Mat grayscaleFloatTemplate = m_TmplImg;
	Mat grayscaleFloatImage = m_CapImg;

	//GradientMagnitudeCalibration
	vector<Mat> templateDescriptorFields, imageDescriptorFields;

	ComputeGradientMagnitudeDescriptorFields(grayscaleFloatTemplate, templateDescriptorFields);
	ComputeGradientMagnitudeDescriptorFields(grayscaleFloatImage, imageDescriptorFields);
	
	if (optimizationParameters.bNormalizeDescriptors)
	{
		NormalizeImage(templateDescriptorFields[0]);
		NormalizeImage(imageDescriptorFields[0]);
	}

	//PyramidMultilevelCalibration
	vector< vector<Mat> > smoothedImages(optimizationParameters.pyramidSmoothingVariance.size());
	vector< vector<Mat> > smoothedTemplates(optimizationParameters.pyramidSmoothingVariance.size());

#pragma omp parallel for
	for (int i = 0;i < optimizationParameters.pyramidSmoothingVariance.size();i++)
	{
		smoothedImages[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], imageDescriptorFields);
		smoothedTemplates[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], templateDescriptorFields);
	}

	float err_temp = 50.0;
	for (int iLevel = 0;iLevel < optimizationParameters.pyramidSmoothingVariance.size();iLevel++)
	{
		if (iLevel == optimizationParameters.pyramidSmoothingVariance.size() - 1)
		{
			optimizationParameters.maxIterSingleLevel = optimizationParameters.maxIter;
		}
		//cout << "Start using pyramid level no." << iLevel + 1 << endl;
		err_temp = CalImgError(pixelsOnTemplate, smoothedImages[iLevel], smoothedTemplates[iLevel]);
		if (err_ssd > err_temp)
			err_ssd = err_temp;
	}
	return err_ssd;
}

float PoseEstimation::CalImgErrorByDF(cv::Mat & m_CapImg, cv::Mat & m_TmplImg)
{
	float err_ssd = 50.0;
	float err_temp = 50.0;
	
	//optimization parameters using LukasKanadeVideoDescriptorFieldsSpeedTest of EPFL's Unittest
	OptimizationParameters optimizationParameters;
	optimizationParameters.resTol = 1e-5;
	optimizationParameters.pTol = 5e-5;
	optimizationParameters.maxIter = 50;
	optimizationParameters.maxIterSingleLevel = 10;
	optimizationParameters.pyramidSmoothingVariance.push_back(10);
	optimizationParameters.pyramidSmoothingVariance.push_back(5);
	optimizationParameters.presmoothingVariance = 0;
	optimizationParameters.nControlPointsOnEdge = 60;
	optimizationParameters.borderThicknessHorizontal = 100;
	optimizationParameters.borderThicknessVertical = 50;
	optimizationParameters.bAdaptativeChoiceOfPoints = 1;
	optimizationParameters.bNormalizeDescriptors = 1;

	ConvertImageToFloat(m_TmplImg);
	ConvertImageToFloat(m_CapImg);
	StructOfArray2di controlPoints;
	if(optimizationParameters.bNormalizeDescriptors = true)
		controlPoints = CreateAnisotropicGridOfControlPoints(m_TmplImg, 30, 0.0f, 0.0f);
	else
		controlPoints= CreateGridOfControlPoints(m_TmplImg, 30, 0.0f, 0.0f);

	StructOfArray2di pixelsOnTemplate = controlPoints;
	Mat grayscaleFloatTemplate = m_TmplImg;
	Mat grayscaleFloatImage = m_CapImg;

	//DescriptorFieldsCalibration
	vector<Mat> templateDescriptorFields, imageDescriptorFields;

	ComputeGradientBasedDescriptorFields(grayscaleFloatTemplate, templateDescriptorFields);
	ComputeGradientBasedDescriptorFields(grayscaleFloatImage, imageDescriptorFields);

	if (optimizationParameters.bNormalizeDescriptors)
	{
		for (uint i = 0; i < templateDescriptorFields.size(); ++i)
		{
			NormalizeImage(templateDescriptorFields[i]);
			NormalizeImage(imageDescriptorFields[i]);
		}
	}

	//PyramidMultilevelCalibration
	vector< vector<Mat> > smoothedImages(optimizationParameters.pyramidSmoothingVariance.size());
	vector< vector<Mat> > smoothedTemplates(optimizationParameters.pyramidSmoothingVariance.size());

#pragma omp parallel for
	for (int i = 0;i < optimizationParameters.pyramidSmoothingVariance.size();i++)
	{
		smoothedImages[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], imageDescriptorFields);
		smoothedTemplates[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], templateDescriptorFields);
	}

	for (int iLevel = 0;iLevel < optimizationParameters.pyramidSmoothingVariance.size();iLevel++)
	{
		if (iLevel == optimizationParameters.pyramidSmoothingVariance.size() - 1)
		{
			optimizationParameters.maxIterSingleLevel = optimizationParameters.maxIter;
		}
		//cout << "Start using pyramid level no." << iLevel + 1 << endl;
		err_temp = CalImgError(pixelsOnTemplate, smoothedImages[iLevel], smoothedTemplates[iLevel]);
		if (err_ssd > err_temp)
			err_ssd = err_temp;
	}
	return err_ssd;
}

void PoseEstimation::CalCoarsePoses(vector<cv::Mat>& ellMats)
{
	m_coarsePoses.clear();
	if (ellMats.size() < 1)
	{
		cout << "No ellipse param for cal coarse pose" << endl;
		return;
	}
	if (m_ModelRadius == 0)
	{
		cout << "ModelRadius improperly set" << endl;
		return;
	}
	if (m_Intrinsic.empty())
	{
		cout << "No camera intrinsic parameter read" << endl;
		return;
	}

	cv::Mat rot_cv2osg = (cv::Mat_<double>(3, 3) <<
		1, 0, 0,
		0, -1, 0,
		0, 0, -1);
	
	cv::Mat Qone, ellMat, eig_Vector, eig_Value;
	double lmd1, lmd2, lmd3, w;
	cv::Mat Vmax, Vmin;
	cv::Mat pNorm, nCenter, tCenter;
	double p1, p2, p3;
	cv::Mat vec_P1, vec_P2;
	double dist_factor, dist;
	cv::Mat CoarsePose;
	cv::Mat rowmat = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

	for (int i = 0;i < ellMats.size();i++)
	{
		ellMat = ellMats[i];
		ellMat.convertTo(ellMat, CV_64F);
		Qone = rot_cv2osg.t()*m_Intrinsic.t()*ellMat*m_Intrinsic*rot_cv2osg;
		cv::eigen(Qone, eig_Value, eig_Vector);
		lmd1 = eig_Value.at<double>(2, 0);
		lmd2 = eig_Value.at<double>(1, 0);
		lmd3 = eig_Value.at<double>(0, 0);
		Vmax = (eig_Vector.row(0)).t();
		Vmin = (eig_Vector.row(2)).t();
		for (int j = 0; j < 2;j++)
		{
			if (j == 0)
				w = 1;
			else
				w = -1;
			pNorm = sqrt((lmd3 - lmd2) / (lmd3 - lmd1))*Vmax + w * sqrt((lmd2 - lmd1) / (lmd3 - lmd1))*Vmin;
			if (pNorm.at<double>(2, 0) < 0)
				pNorm = -pNorm;
			cv::normalize(pNorm, pNorm);
			nCenter = Qone.inv()*pNorm;
			cv::normalize(nCenter, nCenter);
			p1 = pNorm.at<double>(0, 0);
			p2 = pNorm.at<double>(1, 0);
			p3 = pNorm.at<double>(2, 0);
			vec_P1 = (cv::Mat_<double>(3, 1) << -p2, p1, 0);
			cv::normalize(vec_P1, vec_P1);
			vec_P2 = (cv::Mat_<double>(3, 1) << -p3 * p1, -p2 * p3, p1*p1 + p2 * p2);
			cv::normalize(vec_P2, vec_P2);
			dist_factor = sqrt(-lmd2 / lmd3 - lmd2 / lmd1 + 1);
			dist = dist_factor * m_ModelRadius;
			tCenter = dist * nCenter;
			hconcat(vec_P1, vec_P2, CoarsePose);
			hconcat(CoarsePose, pNorm, CoarsePose);
			hconcat(CoarsePose, tCenter, CoarsePose);
			vconcat(CoarsePose, rowmat, CoarsePose);
#ifdef DETAIL
			cout << "The coarsePose " << j << " is: " << endl << CoarsePose << endl;
#endif
			m_coarsePoses.push_back(CoarsePose);
		}
	}
}

void PoseEstimation::SelectCandidatePose(vector<cv::Mat>& CoarsePoses, vector<cv::Rect> & ellRects, int ErrMode)
{
	/*
	1. 首先，从img中检测到了 vector<Ellipses> 对应的 vector<Rect>
	 矩阵vector<ellMat> 
	2. 从 vector<ellMat>中得到了 vector<Mat> CoarsePoses
	3. 每个 CoarsePose,Rect 都可以 GenerateTemplateImg
	4. 1 rect 1 tmplimg 可得到 1 score
	5. 选最小的score, 记录对应的 pose, rect, pose作为 candidate. p0, rect用于计算后面的score
	*/
	cv::Mat tmpl_full;
	cv::Rect ellRect;
	float fPoseScore = 50.0;
	float fTempScore = 50.0;
	cv::Mat CapRoi;
	cv::Mat TmplRoi;
	int pose_index;
	//m_SyncGenerator.SetReInitialize(true);

	if (CoarsePoses.size() == 0 || ellRects.size() == 0)
	{
		cout << "no poses or no rects" << endl;
		return;
	}

	for (int i = 0;i < CoarsePoses.size();++i)
	{

		tmpl_full = GenerateTemplateImg(CoarsePoses[i]);

#ifdef SHOWIMG
		imshow("current template", tmpl_full);
		waitKey(50);
#endif
		ellRect = ellRects[i / 2];
		CapRoi = m_CapImg(ellRect).clone();
		TmplRoi = tmpl_full(ellRect).clone();

		switch (ErrMode)
		{
		case 1:
			fTempScore = CalImgErrorBySSD(CapRoi, TmplRoi);
			break;
		case 2:
			fTempScore = CalImgErrorByGF(CapRoi, TmplRoi);
			break;
		case 3:
			fTempScore = CalImgErrorByDF(CapRoi,TmplRoi);
			break;
		default: 
			break;
		}
#ifdef DETAIL
		cout << "The candidate score of pose " << i << " is " << fTempScore << endl;
#endif
		if(fTempScore<fPoseScore)
		{	
			fPoseScore = fTempScore;
			pose_index = i;
		}
	}

	cout << "The minimal candidate pose score is: " << fPoseScore << endl;
	m_FinalScore = fPoseScore;
	m_CandidatePose = CoarsePoses[pose_index];
	//m_CandidatePose = CoarsePoses[0];
	m_CandidateRect = ellRects[pose_index/2];
	//m_SyncGenerator.SetReInitialize(true);
	m_TmplImg = GenerateTemplateImg(m_CandidatePose);
	m_iCandidateEllIndex = pose_index / 2;

	//以下两行可以看到 Candidate PE的AR效果，不应用 m_TmplImg
	//m_TmplGenerator.SetReInitialize(true);
	//m_TmplImg = GenerateARImg(m_CandidatePose, m_CapImg);
	
}

void PoseEstimation::SelectFinePoses(vector<cv::Mat>& VecPoses, cv::Rect & rect, Mat & ARImg, int ErrMode)
{
 	//经过实验后发现，fine完全不用select，每一个都是一样的
	cv::Mat tmpl_full;
 	cv::Rect ellRect;
 	float fPoseScore = 50.0;
 	float fTempScore = 50.0;
 	cv::Mat CapRoi;
 	cv::Mat TmplRoi;
 	int pose_index=0;
	m_SyncGenerator.SetReInitialize(true);

	if (VecPoses.size() == 0 || rect.empty())
	{
		cout << "no poses or no rect" << endl;
		return;
	}

	for (int i = 0;i < VecPoses.size();++i)
	{

		tmpl_full = GenerateTemplateImg(VecPoses[i]);

#ifdef SHOWIMG
		imshow("select fine", tmpl_full);
		waitKey(0);
#endif

	//tmpl_full = GenerateTemplateImg(VecPoses[pose_index]);
 		CapRoi = m_CapImg(rect).clone();
 		TmplRoi = tmpl_full(rect).clone();

		switch (ErrMode)
		{
		case 1:
			fTempScore = CalImgErrorBySSD(CapRoi, TmplRoi);
			break;
		case 2:
			fTempScore = CalImgErrorByGF(CapRoi, TmplRoi);
			break;
		case 3:
			fTempScore = CalImgErrorByDF(CapRoi, TmplRoi);
			break;
		default:
			break;
		}
#ifdef DETAIL
		cout << "The candidate score of pose " << i << " is " << fTempScore << endl;
#endif
		if (fTempScore < fPoseScore)
		{
			fPoseScore = fTempScore;
			pose_index = i;
		}
	}

	fPoseScore = CalImgErrorByGF(CapRoi, TmplRoi);
	cout << "The minimal fine pose score is: " << fPoseScore << endl;
	m_FinePose= VecPoses[pose_index];
	cout << "The fine pose is: " << endl<<m_FinePose << endl;
	m_SyncGenerator.SetReInitialize(true);
	ARImg = GenerateARImg(m_FinePose,m_CapImg);
}

void PoseEstimation::CalFinePoseByDFHomography()
{
	//计算出 homo 后分解 R,t, 值传给 m_FinePose
	m_FinePoses.clear();
	if (m_CandidatePose.empty())
	{
		cout << "no candidate pose as initial pose!" << endl;
		return;
	}
	
	OptimizationParameters optimizationParameters;
	optimizationParameters.pTol = 1e-5;//-5
	optimizationParameters.resTol = 1e-5;//-5
	optimizationParameters.maxIter = 50;
	optimizationParameters.maxIterSingleLevel = 10;
	//5,10效果最好
	optimizationParameters.pyramidSmoothingVariance.push_back(10);
	optimizationParameters.pyramidSmoothingVariance.push_back(5);
	optimizationParameters.presmoothingVariance = 0;//0 计算量大且不准
	optimizationParameters.nControlPointsOnEdge = 50;
	optimizationParameters.bAdaptativeChoiceOfPoints = 1;//1
	optimizationParameters.bNormalizeDescriptors = 1;

	Mat Img_Real = m_CapImg.clone();
	Mat Img_Sync = m_TmplImg.clone();
	cvtColor(Img_Real, Img_Real, CV_BGR2GRAY);
	cvtColor(Img_Sync, Img_Sync, CV_BGR2GRAY);

	cv::Rect temproi = m_CandidateRect;
	Mat tmpl = Img_Sync(temproi).clone();
	Mat image = Img_Real;
	ConvertImageToFloat(tmpl);
	ConvertImageToFloat(image);
	vector<float> parametersBaseline(8, 0); 
	parametersBaseline[4] = temproi.x;
	parametersBaseline[5] = temproi.y;

	StructOfArray2di controlPoints = CreateGridOfControlPoints(tmpl, 50, 0.0f, 0.0f);
	//StructOfArray2di controlPoints = CreateAnisotropicGridOfControlPoints(tmpl, 50);
	//StructOfArray2di controlPoints = CreateDenseGridOfControlPoints(tmpl.cols, tmpl.rows);
	vector<float> parametersInitialGuess(parametersBaseline);

	LucasKanade optimization;
	AlignmentResults results = optimization.DescriptorFieldsCalibration(
		controlPoints,/*sparse grid points on template image*/
		tmpl, /*the template image (query patch)*/
		image, /*the captured image */
		parametersInitialGuess, /*search from roi center*/
		optimizationParameters /*initialized parameters*/);
#ifdef DETAIL
	cout << "The residual Norm error is: " << results.residualNorm[results.residualNorm.size() - 1] << endl;
	cout << "The iter times is : " << results.residualNorm.size() << endl;
#endif

	Matx33f matresult = Homography::GetMatrix(results.poseIntermediateGuess[results.poseIntermediateGuess.size() - 1]);

#ifdef DETAIL
	cout << endl << "The GF homo mat result is: " << endl << matresult << endl;
#endif // DETAIL
	
	Mat cvHomographyResult(matresult);
	vector<Mat> rotMat, transVec, normVec;
	Mat norm;
	cv::Mat rowmat = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
	cv::Mat transvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
	decomposeHomographyMat(cvHomographyResult, m_Intrinsic, rotMat, transVec, normVec);

	Mat temp_FinePose,temp_CamRelMat;
	int index = 0;
	float fineScore;
	hconcat(rotMat[index], transVec[index], temp_CamRelMat);
	vconcat(temp_CamRelMat, rowmat, temp_CamRelMat);
	temp_FinePose = m_CandidatePose * (temp_CamRelMat.inv());

	Mat CapRoi = Img_Real(temproi).clone();
	Mat TmplRoi = Img_Sync(temproi).clone();
	fineScore = CalImgErrorByGF(CapRoi, TmplRoi);
	cout << "The minimal fine pose score is: " << fineScore << endl;
	m_FinePose = temp_FinePose;
	cout << "The fine pose is: " << endl << m_FinePose << endl;
	m_SyncGenerator.SetReInitialize(true);
	m_FineImg = GenerateARImg(m_FinePose, m_CapImg);
// 	for (int i = 0; i < rotMat.size(); ++i)
// 	{
// #ifdef DETAIL
// 		cout << "======== testing " << i << " th homo pose ========" << endl;
// 		cout << "rotation " << i << " = " << endl;
// 		cout << rotMat[i] << endl;
// 		cout << "translation " << i << " = " << endl;
// 		cout << transVec[i] << endl;
// 		cout << "normal " << i << " = " << endl;
// 		cv::normalize(normVec[i], norm);
// 		cout <<"homo norm is: "<< norm << endl;
// #endif
// 		hconcat(rotMat[i], transVec[i],temp_CamRelMat);
// 		//hconcat(rotMat[i], transvec, temp_CamRelMat);
// 		vconcat(temp_CamRelMat, rowmat, temp_CamRelMat);
// 		temp_FinePose = m_CandidatePose * (temp_CamRelMat.inv());
// 		m_FinePoses.push_back(temp_FinePose);
// #ifdef DETAIL
// 		cout << "the " << i << " th fine pose is" <<endl<< temp_FinePose << endl;
// #endif
// 	}
	//SelectFinePoses(m_FinePoses, temproi, m_FineImg);
}

void PoseEstimation::CalFinePoseByKpsHomography()
{
	Mat Img_Real = m_CapImg.clone();
	Mat Img_Sync = m_TmplImg.clone();
	cvtColor(Img_Real, Img_Real, CV_BGR2GRAY);
	cvtColor(Img_Sync, Img_Sync, CV_BGR2GRAY);

	cv::Rect temproi = m_CandidateRect;
	Mat tmpl = Img_Sync(temproi).clone();
	Mat image = Img_Real;

	      //创建方式和2中的不一样
	Mat c, d;
	vector<KeyPoint>key1, key2;
	vector<DMatch> matches;
	surf->detectAndCompute(tmpl, Mat(), key1, c);
	surf->detectAndCompute(image, Mat(), key2, d);
	matcher.match(c, d, matches);
	sort(matches.begin(), matches.end());  //筛选匹配点    
	vector< DMatch > good_matches;                 
	int ptsPairs = std::min(50, (int)(matches.size() * 0.15));    
	for (int i = 0; i < ptsPairs; i++)    
	{       
		good_matches.push_back(matches[i]);    
	}
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++) 
	{ 
		obj.push_back(key1[good_matches[i].queryIdx].pt);       
		scene.push_back(key2[good_matches[i].trainIdx].pt);
	}
	vector<uchar> inliers = vector<uchar>(obj.size(), 0);
	Mat cvHomographyResult = cv::findHomography(obj, scene, inliers, CV_FM_RANSAC, 3.0);
	vector<Mat> rotMat, transVec, normVec;
	Mat norm;
	cv::Mat rowmat = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
	decomposeHomographyMat(cvHomographyResult, m_Intrinsic, rotMat, transVec, normVec);

	Mat temp_FinePose, temp_CamRelMat;
 	int index = 2;
 	float fineScore;
	hconcat(rotMat[index], transVec[index], temp_CamRelMat);
	vconcat(temp_CamRelMat, rowmat, temp_CamRelMat);
	temp_FinePose = m_CandidatePose * (temp_CamRelMat.inv());

	Mat CapRoi = Img_Real(temproi).clone();
	Mat TmplRoi = Img_Sync(temproi).clone();
	fineScore = CalImgErrorByGF(CapRoi, TmplRoi);
	cout << "The minimal fine pose score is: " << fineScore << endl;
	m_FinalScore = fineScore;
	m_FinePose = temp_FinePose;
	cout << "The fine pose is: " << endl << m_FinePose << endl;
// 	m_SyncGenerator.SetReInitialize(true);
// 	m_FineImg = GenerateARImg(m_FinePose, m_CapImg);
}

void PoseEstimation::CalFinePoseBy3DIC41DOF()
{
	m_FinePoses.clear();
	if (m_CandidatePose.empty())
	{
		cout << "no candidate pose as initial pose!" << endl;
		return;
	}
	Mat Img_Real = m_CapImg.clone();
	Mat Img_Sync = m_TmplImg.clone();
	cvtColor(Img_Real, Img_Real, CV_BGR2GRAY);
	cvtColor(Img_Sync, Img_Sync, CV_BGR2GRAY);
	cv::Rect temproi = m_CandidateRect;
	Mat tmplroi = Img_Sync(temproi).clone();
	Mat imageroi = Img_Real(temproi).clone();
	Mat temp_FinePose,temp_sync;
	vector<cv::Mat> GenPoses;
#pragma region CAL_BY_1D_ITER
	//Todo: 1. 按照 1D Rotation Rz(theta) 计算 score, 求出 theta 的增量, 合成 temp_CamRelMat
#pragma endregion CAL_BY_1D_ITER
#pragma  region CAL_BY_STEP
	//Todo: 2. 按照 score 收敛的变化量调整 theta 的步长，逐渐 choose best templates，比如 30度生成 12 个 templates，选最接近的，60度再生成 12 个 templates, 之后 10° 生成 10 个 templates,误差控制在1°
	Mat NormVec_Z = (cv::Mat_<double>(1, 3) << 0, 0, 1);
	GenPoses = GenRotPoses(m_CandidatePose,NormVec_Z,PI,PI/6);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi,3);

	Mat NormVec_X = (cv::Mat_<double>(1, 3) << 1, 0, 0);
	GenPoses = GenRotPoses(temp_FinePose, NormVec_X, PI / 6, PI / 36);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi, 3);
	GenPoses = GenRotPoses(temp_FinePose, NormVec_X, PI / 36, PI / 180);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi,3);
	Mat NormVec_Y = (cv::Mat_<double>(1, 3) << 0, 1, 0);
	GenPoses = GenRotPoses(temp_FinePose, NormVec_Y, PI /6, PI / 36);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi, 3);
	GenPoses = GenRotPoses(temp_FinePose, NormVec_Y, PI / 36, PI / 180);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi, 3);
	//z 轴优化放到前面也可以，放到最后也可以
	GenPoses = GenRotPoses(temp_FinePose, NormVec_Z, PI / 3, PI / 36);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi, 2);
	GenPoses = GenRotPoses(temp_FinePose, NormVec_Z, PI / 36, PI / 180);
	temp_FinePose = SelectOptimalPose(GenPoses, temproi, imageroi, 2);

#pragma  endregion CAL_BY_STEP
	//Todo: 3. 配置 Release 加快试验速度
	
//#ifdef DETAIL
	temp_sync = GenerateTemplateImg(temp_FinePose);
	cvtColor(temp_sync, temp_sync, CV_BGR2GRAY);
	Mat CapRoi = imageroi;
	Mat TmplRoi = temp_sync(temproi).clone();
	float fineScore = CalImgErrorByGF(CapRoi, TmplRoi);
	m_FinalScore = fineScore;
	cout << "The minimal fine pose score is: " << fineScore << endl;
//#endif
	m_FinePose = temp_FinePose;
	cout << "The fine pose is: " << endl << m_FinePose << endl;
// 	m_ARGenerator.SetReInitialize(true);
// 	m_FineImg = GenerateARImg(m_FinePose, m_CapImg);
	
#ifdef SHOWIMG
	imshow("Final optimized 1D rot pose", m_FineImg);
	waitKey(0);
#endif
}
 
Mat PoseEstimation::SelectOptimalPose(vector<cv::Mat>& Poses, cv::Rect & rect, cv::Mat & CapRoi, int ErrMode)
{
	if(Poses.size()==0||rect.empty()||CapRoi.empty())
	{
		cout << "No Poses || no rect || no CapRoi " << endl;
	}
	Mat ResultPose,resultMat;
	Mat tmpl_full,tmpl_roi;
	float fPoseScore = 50.0;
	float fTempScore = 50.0;
	int pose_index = 0;
	//m_SyncGenerator.SetReInitialize(true);
	for (int i = 0; i < Poses.size(); ++i)
	{
		tmpl_full = GenerateTemplateImg(Poses[i]);
		tmpl_roi = tmpl_full(rect).clone();
		cvtColor(tmpl_roi, tmpl_roi, CV_BGR2GRAY);
#ifdef SHOWIMG
		imshow("sync tmpl by gened poses", tmpl_full);
		waitKey(50);
#endif
		switch (ErrMode)
		{
		case 1:
			fTempScore = CalImgErrorBySSD(CapRoi, tmpl_roi);
			break;
		case 2:
			fTempScore = CalImgErrorByGF(CapRoi, tmpl_roi);
			break;
		case 3:
			fTempScore = CalImgErrorByDF(CapRoi, tmpl_roi);
			break;
		default:
			break;
		}
#ifdef DETAIL
		cout << "The score of gened pose " << i << " is " << fTempScore << endl;
#endif
		if (fTempScore < fPoseScore)
		{
			fPoseScore = fTempScore;
			pose_index = i;
		}
	}
#ifdef DETAIL
	cout << "The minimal pose score is: " << fPoseScore << endl;
#endif
	ResultPose = Poses[pose_index];
#ifdef SHOWIMG
// 	m_SyncGenerator.SetReInitialize(true);
// 	resultMat = GenerateARImg(ResultPose, m_CapImg);
// 	imshow("Selected gened pose", resultMat);
// 	waitKey(0);
#endif
	return ResultPose;
}

vector<cv::Mat> PoseEstimation::GenRotPoses(cv::Mat & IniPose, cv::Mat & VecNorm, float degree_range, float degree_step)
{
	if (IniPose.empty() || VecNorm.empty())
	{
		cout << "error IniPose or error VecNorm" << endl;
	}
	vector<cv::Mat> GenRotPoses;
	cv::Mat RotMat,RotDegTemp,RotAfterMat;
	cv::Mat TransVec,RowVec;
	cv::Mat GenRotPose;

	RotMat = IniPose(Range(0, 3), Range(0, 3));
	TransVec = IniPose.col(3).clone();
	RowVec = (cv::Mat_<double>(1, 3) << 0, 0, 0);
	int N = ceil( degree_range / degree_step);
	int M = ceil(N / 2);
#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		float alpha = i * degree_step-M*degree_step;
		double cs = cos(alpha);
		double ss = sin(alpha);
		if (VecNorm.at<double>(0, 2) == 1)
		{
			RotDegTemp = (cv::Mat_<double>(3, 3) <<
				cs, -ss, 0,
				ss, cs, 0,
				0, 0, 1);
		}
		if (VecNorm.at<double>(0, 0) == 1)
		{
			RotDegTemp = (cv::Mat_<double>(3, 3) <<
				1, 0, 0,
				0, cs, -ss,
				0, ss, cs);
		}
		if (VecNorm.at<double>(0, 1) == 1)
		{
			RotDegTemp = (cv::Mat_<double>(3, 3) <<
				cs, 0, ss,
				0, 1 , 0,
				-ss, 0, cs);
		}
		RotAfterMat = RotMat * RotDegTemp;
		vconcat(RotAfterMat, RowVec, GenRotPose);
		hconcat(GenRotPose, TransVec, GenRotPose);
		GenRotPoses.push_back(GenRotPose);
	}
	return GenRotPoses;
}

void PoseEstimation::Cal6DPoseError(cv::Mat & Pose_gt, cv::Mat & Pose_est, ofstream & ofile, bool isWrite)
{
	double err_tx, err_ty, err_tz, err_rx, err_ry, err_rz;
	Mat Trans_gt = Pose_gt(Range(0, 3), Range(3, 4));
	Mat Rot_gt = Pose_gt(Range(0, 3), Range(0, 3));
	Mat Trans_est = Pose_est(Range(0, 3), Range(3, 4));
	Mat Rot_est = Pose_est(Range(0, 3), Range(0, 3));
	Mat Rx_est = Rot_est.col(0);
	Mat Ry_est = Rot_est.col(1);
	Mat Rz_est = Rot_est.col(2);
	Mat Rx_gt = Rot_gt.col(0);
	Mat Ry_gt = Rot_gt.col(1);
	Mat Rz_gt = Rot_gt.col(2);
	cv::normalize(Rx_est, Rx_est);
	cv::normalize(Ry_est, Ry_est);
	cv::normalize(Rz_est, Rz_est);
	cv::normalize(Rx_gt, Rx_gt);
	cv::normalize(Ry_gt, Ry_gt);
	cv::normalize(Rz_gt, Rz_gt);
	Mat TransErr = Trans_gt - Trans_est;
	err_tx = TransErr.at<double>(0, 0);
	err_ty = TransErr.at<double>(0, 0);
	err_tz = TransErr.at<double>(0, 0);
	double test = Rx_gt.dot(Rx_est);
	err_rx = acos((Rx_gt.dot(Rx_est)));
	err_ry = acos(Ry_gt.dot(Ry_est));
	err_rz = acos(Rz_gt.dot(Rz_est));
	double err_rx_d = rad2degree(err_rx);
	double err_ry_d = rad2degree(err_ry);
	double err_rz_d = rad2degree(err_rz);
	cout << "the trans error is: " << err_tx << ","<<err_ty << ","
	     	<<err_tz << endl;
	cout << "the rotation error is: " << err_rx_d << "," << err_ry_d << ","
		<< err_rz_d << endl;
	if (isWrite == true && ofile)
	{
		ofile << err_tx << ',' << err_ty << ',' << err_tz << ','
			<< err_rx_d << ',' << err_ry_d << ',' << err_rz_d << endl;
	}
}


