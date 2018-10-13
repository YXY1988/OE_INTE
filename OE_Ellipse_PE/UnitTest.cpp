#include "UnitTest.h"
#include "commonlibs.h"
#define VERBOSE
/*
void CVCalibTest()
{
	CameraParam cvParam;
	string cvCalibFileName = "../../Data/Params/Logitechc270_cv0709.xml";
	cvParam.ReadCVCalibParam(cvCalibFileName);
	bool bOk = true;
	if (cvParam.GetCameraMatrix().empty()) bOk = false;
	if (bOk == false)
		throw "Read cv calibration param fail";
	return;
}
 
void UndistortTest()
{
	CameraParam cvParam;
	string cvCalibFileName = "../../Data/Params/Logitechc270_cv0709.xml";
	string ImageFileName = "../../Data/Temp/SB_test120.bmp";
	cv::Mat src, undistort;
	bool bOk = true;

	cvParam.ReadCVCalibParam(cvCalibFileName);
	cvParam.IniValidUndistort();

	src = imread(ImageFileName);
	imshow("src", src);
	waitKey(10);
	cvParam.UndistortImgValid(src, undistort);
	if (src.empty() || undistort.empty()) bOk = false;
	if(bOk==false)
		throw "undistort test fail";
	imshow("undistort", undistort);
	waitKey(10);
	return;
}

void Cv2arTest()
{
	bool bOk = true;
	CameraParam cvParam;
	ARParam arParam;
	string arCalibFileName = "../../Data/Params/Logitechc270_cv0709.dat";
	string cvCalibFileName = "../../Data/Params/Logitechc270_cv0709.xml";

	cvParam.ReadCVCalibParam(cvCalibFileName);
	cvParam.GetARTKCamParam(arCalibFileName);
	arParam = cvParam.GetARParam();

	if (arParam.mat[0][0] == 0) bOk = false;
	if (bOk == false)
		throw "Fail convert cv calibfile to artk format.";
	return;
}
*/
void DetectMajorEllipses(cv::Mat & src)
{
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	bool bOk = true;

	//float resizescale = 1;
	//cv::Size dsize = cv::Size(src.cols*resizescale, src.rows*resizescale);
	//resize(src, test, dsize);
	test = src.clone();
	_EdgeDetector.SetSrcImg(test);

# ifdef VERBOSE
	double t_begin = cv::getTickCount();
#endif

	contour = _EdgeDetector.CannyContourDetection(50, 150);

#ifdef VERBOSE
	double t_end = cv::getTickCount();
	double t_cost = (t_end - t_begin) / cv::getTickFrequency() * 1000;
	cout << "ÂÖÀª¼ì²âºÄÊ±(ms)£º " << t_cost << endl;
#endif

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);

#ifdef VERBOSE
	double t_refine = cv::getTickCount();
	t_cost = (t_refine - t_end) / cv::getTickFrequency() * 1000;
	cout << "ÂÖÀªÔ¤´¦ÀíºÄÊ±(ms)£º " << t_cost << endl;
#endif

	edges = _EdgeDetector.GetFinalContours();
	cv::Mat rgb_contour;
	cv::cvtColor(temp.clone(), rgb_contour, cv::COLOR_GRAY2BGR);
	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(200.);//100
	_EllipseDetector.DetectEllipses(temp, edges);
	_EllipseDetector.DrawEllipses();

#ifdef VERBOSE
	double t_ell = cv::getTickCount();
	t_cost = (t_ell - t_refine) / cv::getTickFrequency() * 1000;
	cout << "ÍÖÔ²¼ì²âºÄÊ±(ms)£º " << t_cost << endl;
#endif

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	
	if (ellResult.size() == 0) bOk = false;
	if (bOk == false)
		throw "No ellipse detected in img";
	return;
}

void TestCoarsePose()
{
	cv::Mat ellMat = (cv::Mat_<double>(3, 3) << 3.526086e-05, 7.714522e-07, -0.01527425,
		7.714522e-07, 3.595431e-05, -0.0071982,
		-0.01527425, -0.0071982, 6.92751687);
	vector<cv::Mat> ellMats;
	ellMats.push_back(ellMat);
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522,0 ,299.60111699063754,
	0,814.73836342732341,256.75622898129393, 0,0,1);

	PoseEstimation CTestCoarse;
	CTestCoarse.SetIntrinsic(Intrinsic);
	CTestCoarse.SetModelRadius(178);
	CTestCoarse.CalCoarsePoses(ellMats);
	return;
}

void TestImgConvertFunctions()
{
	string imageFileName = "../Data/Temp/SB_test120.bmp";

	cv::Mat testing = cv::imread(imageFileName);
	osg::ref_ptr<osg::Image> testResult = ConvertCVMat2OsgImg(testing);
	osgDB::writeImageFile(*testResult, "../Data/Temp/TestResult/SB_test120_reslut.bmp");

	osg::ref_ptr<osg::Image> testing2 = osgDB::readImageFile(imageFileName);
	cv::Mat testResult2 = ConvertOsgImg2CVMat(testing2);
	cv::imwrite("../Data/Temp/TestResult/SB_test120_reslut2.bmp", testResult2);
}

void TestSyntheticTemplateGeneration()
{

	SceneGenerator sceneGenerator;
	string imageFileName = "../Data/Temp/SB_test120.bmp";
	cv::Mat bgimg = cv::imread(imageFileName);
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	cv::Mat PoseMat = (cv::Mat_<double>(4, 4) << 0.9808679135958881, -0.1929804413015563, 0.02562587272866551, 137.053573311234,
		0.1946744361184826, 0.9723327140347628, -0.129116060735072, 65.60683142202694,
		0, 0.1316345034284271, 0.9912983191285818, -871.1839883388401,
		0, 0, 0, 1);
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	cv::Mat OsgPoseMat = PoseMat.t();
	string IveModelName = "../Data/Temp/cylinder.ive";

	sceneGenerator.SetBgImgMat(bgimg);
	sceneGenerator.SetCameraIntrinsic(Intrinsic);
	sceneGenerator.SetModelName(IveModelName);
	sceneGenerator.SetUseImgBgFlag(true);
	sceneGenerator.SetPoseMat(OsgPoseMat);
	sceneGenerator.SetViewerSize(640, 480);
	sceneGenerator.SetObjectSelfTransform(ObjectTransform);

	cv::Mat result;
	result = sceneGenerator.GetSyntheticImg();
	imshow("SyntheticTest", result);
	cv::waitKey(0);

/*
	cv::Mat PoseMat2 = (Mat_<double>(4, 4) << 0.9808679135958881, -0.1929804413015563, 0.02562587272866551, 137.053573311234,
		0.1946744361184826, 0.9723327140347628, -0.129116060735072, 65.60683142202694,
		0, 0.1316345034284271, 0.9912983191285818, -1071.1839883388401,
		0, 0, 0, 1);
	sceneGenerator.SetPoseMat(PoseMat2.t());
	result = sceneGenerator.GetSyntheticImg();
	imshow("SyntheticTest", result);
	waitKey(1000);


	cv::Mat PoseMat3 = (Mat_<double>(4, 4) << 0.9808679135958881, -0.1929804413015563, 0.02562587272866551, 137.053573311234,
		0.1946744361184826, 0.9723327140347628, -0.129116060735072, 65.60683142202694,
		0, 0.1316345034284271, 0.9912983191285818, -1271.1839883388401,
		0, 0, 0, 1);
	sceneGenerator.SetPoseMat(PoseMat3.t());
	result = sceneGenerator.GetSyntheticImg();
	imshow("SyntheticTest", result);
	waitKey(100);*/

}

void TestTmplGeneration()
{
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	cv::Mat PoseMat = (cv::Mat_<double>(4, 4) << 0.9808679135958881, -0.1929804413015563, 0.02562587272866551, 137.053573311234,
		0.1946744361184826, 0.9723327140347628, -0.129116060735072, 65.60683142202694,
		0, 0.1316345034284271, 0.9912983191285818, -871.1839883388401,
		0, 0, 0, 1);

	PoseEstimation CTestTmplGen;
	CTestTmplGen.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	cv::Mat result;
	result = CTestTmplGen.GenerateTemplateImg(PoseMat);
	imshow("GenTmpl", result);
	cv::waitKey(0);
}

void TestSSDImgErr()
{
	PoseEstimation PoseTester;

	Mat Img_Reprojection = imread("D:\\OECodes\\Data\\INTE\\dipan_reproj.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat Img_Capture = imread("D:\\OECodes\\Data\\INTE\\dipan_cap.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (Img_Reprojection.empty() || Img_Capture.empty())
		throw "test img empty";

	RotatedRect ellparam;
	ellparam.center = cv::Point2f(268.0, 228.0);
	ellparam.angle = -0.798988 / 3.1415926 * 180;
	ellparam.size.width = 143.071 * 2;
	ellparam.size.height = 89.7407 * 2;
	Rect EllRoi = ellparam.boundingRect();

	Mat templ = Img_Reprojection(EllRoi).clone();
	Mat image = Img_Reprojection(EllRoi).clone();

	float result;
	result = PoseTester.CalImgErrorBySSD(image, templ);
	cout << "the SSD image error of images are: " << result << endl;
	result = PoseTester.CalImgErrorBySSD(image, image);
	cout << "the SSD image error of images are: " << result << endl;
}

void TestGFImgErr()
{
	PoseEstimation PoseTester;

	Mat Img_Reprojection = imread("D:\\OECodes\\Data\\INTE\\dipan_reproj.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat Img_Capture = imread("D:\\OECodes\\Data\\INTE\\dipan_cap.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (Img_Reprojection.empty() || Img_Capture.empty())
		throw "test img empty";

	RotatedRect ellparam;
	ellparam.center = cv::Point2f(268.0, 228.0);
	ellparam.angle = -0.798988 / 3.1415926 * 180;
	ellparam.size.width = 143.071 * 2;
	ellparam.size.height = 89.7407 * 2;
	Rect EllRoi = ellparam.boundingRect();

	Mat templ = Img_Reprojection(EllRoi).clone();
	Mat image = Img_Capture(EllRoi).clone();

	float result;
	result = PoseTester.CalImgErrorByGF(image, templ);
	cout << "the GradientField image error of tmpl and image is: " << result << endl;
	result = PoseTester.CalImgErrorByGF(image, image);
	cout << "the GradientField image error of image and image is: " << result << endl;
}

void TestDFImgErr()
{
	PoseEstimation PoseTester;

	Mat Img_Reprojection = imread("D:\\OECodes\\Data\\INTE\\dipan_reproj.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat Img_Capture = imread("D:\\OECodes\\Data\\INTE\\dipan_cap.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (Img_Reprojection.empty() || Img_Capture.empty())
		throw "test img empty";

	RotatedRect ellparam;
	ellparam.center = cv::Point2f(268.0, 228.0);
	ellparam.angle = -0.798988 / 3.1415926 * 180;
	ellparam.size.width = 143.071 * 2;
	ellparam.size.height = 89.7407 * 2;
	Rect EllRoi = ellparam.boundingRect();

	Mat templ = Img_Reprojection(EllRoi).clone();
	Mat image = Img_Capture(EllRoi).clone();

	float result;
	result = PoseTester.CalImgErrorByDF(image, templ);
	cout << "the DescriptorField image error of tmpl and image is: " << result << endl;
	result = PoseTester.CalImgErrorByDF(image, image);
	cout << "the DescriptorField image error of image and image is: " << result << endl;
}

void TestSelectCandidatePose()
{
	//cv::Mat testimg = cv::imread("../../Data/INTE/realsrc.jpg");
	cv::Mat testimg = cv::imread("../../Data/INTE/coarse1.png");
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;

	test = testimg.clone();
	_EdgeDetector.SetSrcImg(test);
	contour = _EdgeDetector.CannyContourDetection(50, 150);

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);
	edges = _EdgeDetector.GetFinalContours();
	
	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(100.);//100
	_EllipseDetector.DetectEllipses(temp, edges);

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	ellRects = _EllipseDetector.GetEllRects();
	
	//Ini Pose Estimator
	PoseEstimation CTestCandidate;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	CTestCandidate.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);
	
	//Get vector<Mat> CoarsePoses
	vector<cv::Mat> CoarsePoses;
	CTestCandidate.CalCoarsePoses(ellMats);
	CoarsePoses = CTestCandidate.GetCoarsePoses();

	//Select CandidatePose using default gradient mode
	CTestCandidate.SetCapImg(testimg);
	CTestCandidate.SelectCandidatePose(CoarsePoses,ellRects);
	cout << "The candidate ell's index is: " << CTestCandidate.GetCandidateEllIndex() << endl;
	cout << "The candidate pose matrix is: " << endl << CTestCandidate.GetCandidatePose() << endl;

	//Show CandidatePose with regard to original image
#ifdef VERBOSE
	cv::Mat CandidateSyntheticImg;
	CandidateSyntheticImg = CTestCandidate.GetTmplImg();
	imshow("CandidateSync", CandidateSyntheticImg);
	cv::waitKey(0);
#endif

}

void TestEPFLHomoFinePose()
{
	//cv::Mat testimg = cv::imread("../../Data/INTE/realsrc.jpg");
	cv::Mat testimg = cv::imread("../../Data/INTE/coarse1.png");
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;

	test = testimg.clone();
	_EdgeDetector.SetSrcImg(test);
	contour = _EdgeDetector.CannyContourDetection(50, 150);

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);
	edges = _EdgeDetector.GetFinalContours();

	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(100.);//100
	_EllipseDetector.DetectEllipses(temp, edges);

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	ellRects = _EllipseDetector.GetEllRects();

	//Ini Pose Estimator
	PoseEstimation CTestFine;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	CTestFine.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	//Get vector<Mat> CoarsePoses
	vector<cv::Mat> CoarsePoses;
	CTestFine.CalCoarsePoses(ellMats);
	CoarsePoses = CTestFine.GetCoarsePoses();

	//Select CandidatePose using default gradient mode
	CTestFine.SetCapImg(testimg);
	CTestFine.SelectCandidatePose(CoarsePoses, ellRects);
	cout << "The candidate ell's index is: " << CTestFine.GetCandidateEllIndex() << endl;
	cout << "The candidate pose matrix is: " << endl << CTestFine.GetCandidatePose() << endl;

	//Cal Fine pose and show AR Registered img
	CTestFine.CalFinePoseByDFHomography();
#ifdef VERBOSE
	cv::Mat FineARImg;
	FineARImg = CTestFine.GetFineImg();
	imshow("fine", FineARImg);
	waitKey(0);
#endif // VERBOSE
}

void TestKpsHomoFinePose()
{
	//cv::Mat testimg = cv::imread("../../Data/INTE/realsrc.jpg");
	cv::Mat testimg = cv::imread("../../Data/INTE/coarse1.png");
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;

	test = testimg.clone();
	_EdgeDetector.SetSrcImg(test);
	contour = _EdgeDetector.CannyContourDetection(50, 150);

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);
	edges = _EdgeDetector.GetFinalContours();

	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(100.);//100
	_EllipseDetector.DetectEllipses(temp, edges);

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	ellRects = _EllipseDetector.GetEllRects();

	//Ini Pose Estimator
	PoseEstimation CTestFine;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	CTestFine.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	//Get vector<Mat> CoarsePoses
	vector<cv::Mat> CoarsePoses;
	CTestFine.CalCoarsePoses(ellMats);
	CoarsePoses = CTestFine.GetCoarsePoses();

	//Select CandidatePose using default gradient mode
	CTestFine.SetCapImg(testimg);
	CTestFine.SelectCandidatePose(CoarsePoses, ellRects);
	cout << "The candidate ell's index is: " << CTestFine.GetCandidateEllIndex() << endl;
	cout << "The candidate pose matrix is: " << endl << CTestFine.GetCandidatePose() << endl;

	//Cal Fine pose and show AR Registered img
	CTestFine.CalFinePoseByKpsHomography();
#ifdef VERBOSE
	cv::Mat FineARImg;
	FineARImg = CTestFine.GetFineImg();
	imshow("fine", FineARImg);
	waitKey(0);
#endif // VERBOSE
}

void TestRotIterFinePose()
{
	cv::Mat testimg = cv::imread("../../Data/INTE/realsrc.jpg");
	//cv::Mat testimg = cv::imread("../../Data/INTE/coarse1.png");
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;

	test = testimg.clone();
	_EdgeDetector.SetSrcImg(test);
	contour = _EdgeDetector.CannyContourDetection(50, 150);

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);
	edges = _EdgeDetector.GetFinalContours();

	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(100.);//100
	_EllipseDetector.DetectEllipses(temp, edges);

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	ellRects = _EllipseDetector.GetEllRects();

	//Ini Pose Estimator
	PoseEstimation CTestFine;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	CTestFine.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	//Get vector<Mat> CoarsePoses
	vector<cv::Mat> CoarsePoses;
	CTestFine.CalCoarsePoses(ellMats);
	CoarsePoses = CTestFine.GetCoarsePoses();

	//Select CandidatePose using default gradient mode
	CTestFine.SetCapImg(testimg);
	CTestFine.SelectCandidatePose(CoarsePoses, ellRects);
	cout << "The candidate ell's index is: " << CTestFine.GetCandidateEllIndex() << endl;
	cout << "The candidate pose matrix is: " << endl << CTestFine.GetCandidatePose() << endl;

	//Cal Fine pose and show AR Registered img
	CTestFine.CalFinePoseBy3DIC41DOF();
#ifdef VERBOSE
	cv::Mat FineARImg;
	FineARImg = CTestFine.GetFineImg();
	imshow("fine", FineARImg);
	waitKey(0);
#endif // VERBOSE
}

void TestRotIterFinePoseVideo()
{
	VideoCapture cap;
	cv::Mat testimg,FineARImg;
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;
	cv::namedWindow("fine", CV_WINDOW_NORMAL);

	//Ini Pose Estimator
	PoseEstimation CTestFine;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	CTestFine.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	cap.open("../Data/vid/SimpleBright.wmv");
	while (cap.isOpened())
	{
		cap >> testimg;
		test = testimg.clone();
		_EdgeDetector.SetSrcImg(test);
		contour = _EdgeDetector.CannyContourDetection(50, 150);

		temp = contour.clone();
		temp = _EdgeDetector.FilterTurning(temp, 5);
		temp = _EdgeDetector.FilterLines(temp);
		temp = _EdgeDetector.FilterLength(temp, 10);
		edges = _EdgeDetector.GetFinalContours();

		_EllipseDetector.SetSrcImg(test);
		_EllipseDetector.SetFilter_radius(100.);//100
		_EllipseDetector.DetectEllipses(temp, edges);

		ellResult = _EllipseDetector.GetEllDetectionResult();
		ellMats = _EllipseDetector.GetEllMatResult();
		ellRects = _EllipseDetector.GetEllRects();
		if(ellResult.size()==0)
			continue;
		//Get vector<Mat> CoarsePoses
		vector<cv::Mat> CoarsePoses;
		CTestFine.CalCoarsePoses(ellMats);
		CoarsePoses = CTestFine.GetCoarsePoses();

		//Select CandidatePose using default gradient mode
		CTestFine.SetCapImg(testimg);
		CTestFine.SelectCandidatePose(CoarsePoses, ellRects);
		CTestFine.CalFinePoseBy3DIC41DOF();

		//CTestFine.GetARGenerator().SetReInitialize(true);
		CTestFine.GetARGenerator().SetBgImgMat(testimg);
		FineARImg = CTestFine.GetFineImg();
		imshow("fine", FineARImg);
		waitKey(10);
	}
#ifdef VERBOSE
	cout << "The candidate ell's index is: " << CTestFine.GetCandidateEllIndex() << endl;
	cout << "The candidate pose matrix is: " << endl << CTestFine.GetCandidatePose() << endl;
#endif // VERBOSE
}

void RunAllTests()
{
	cv::Mat testimg = cv::imread("../Data/ellfigure/test38.jpg");
	try
	{
		/*cout << "CVCalibTest()......" << endl;
		CVCalibTest();
		cout << ".................... ok." << endl<<endl;*/

		/*cout << "UndistortTest()....." << endl;  
		UndistortTest();
		cout << ".................... ok." << endl<<endl;*/

		/*cout << "Cv2arTest()....." << endl;
		Cv2arTest();
		cout << ".................... ok." << endl << endl;*/

		/*cout << "Test osg and opencv image format bidirectional convertion functions" << endl;
		TestImgConvertFunctions();
		cout << ".................. ok." << endl << endl;*/

		/*cout << "DetectMajorEllipses(cv::Mat & src)...." << endl;
		DetectMajorEllipses(testimg);
		cout << ".................... ok." << endl << endl;*/
		
		/*cout << "TestCoarsePoseEstimation(vector<cv::Mat> & ellMats)..." << endl;
		TestCoarsePose();
		cout << "................... ok." << endl << endl;*/

		/*cout << "TestSyntheticTemplateGeneration......" << endl;
		TestSyntheticTemplateGeneration();
		cout << ".....................ok" << endl;*/
		
		/*cout << "TestTmplGeneration()......" << endl;
		TestTmplGeneration();
		cout << ".....................ok" << endl;*/

		/*cout << "TestSSDImgErr()......" << endl;
		TestSSDImgErr();
		cout << ".....................ok" << endl;*/

		/*cout << "TestGFImgErr()......" << endl;
		TestGFImgErr();
		cout << ".....................ok" << endl;*/

		/*cout << "TestDFImgErr()......" << endl;
		TestDFImgErr();
		cout << ".....................ok" << endl;*/

		/*cout << "TestCandidateSelection()......" << endl;
		TestSelectCandidatePose();
		cout << ".....................ok" << endl;*/

		/*cout << "TestEPFLHomoFinePose()......" << endl;
		TestEPFLHomoFinePose();
		cout << ".....................ok" << endl;	*/

		/*cout << "TestKpsHomoFinePose()......" << endl;
		TestKpsHomoFinePose();
		cout << ".....................ok" << endl;*/

		/*cout << "TestRotIterFinePose()......" << endl;
		TestRotIterFinePose();
		cout << ".....................ok" << endl;*/

		cout << "TestRotIterFinePoseVideo()......" << endl;
		TestRotIterFinePoseVideo();
		cout << ".....................ok" << endl;
	}

	catch (char const* message)
	{
		cerr << message << endl;
		getchar();
	}
	return;
}

int main()
{
	RunAllTests();
	getchar();

	cv::destroyAllWindows();

	return 0;
}
 