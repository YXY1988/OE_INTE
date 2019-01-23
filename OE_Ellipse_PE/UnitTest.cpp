#include "UnitTest.h"
#include "commonlibs.h"
#define VERBOSE
#define SAVEDATA
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
	//string cvCalibFileName = "../../Data/Params/Logitechc270_cv0709.xml";
	//string arCalibFileName = "../../Data/Params/Logitechc270_cv0709.dat";

	string cvCalibFileName = "F:/Datasets/EPFL_can/CAN-TestAndInfo/intrinsic_CV.xml";
	string arCalibFileName = "F:/Datasets/EPFL_can/CAN-TestAndInfo/intrinsic_AR_v2.dat";

	cvParam.ReadCVCalibParam(cvCalibFileName);
	cvParam.GetARTKCamParam(arCalibFileName,2);
	arParam = cvParam.GetARParam();

	if (arParam.mat[0][0] == 0) bOk = false;
	if (bOk == false)
		throw "Fail convert cv calibfile to artk format.";
	return;
}

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

	contour = _EdgeDetector.CannyContourDetection(20, 100);

#ifdef VERBOSE
	double t_end = cv::getTickCount();
	double t_cost = (t_end - t_begin) / cv::getTickFrequency() * 1000;
	cout << "轮廓检测耗时(ms)： " << t_cost << endl;
#endif

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);

#ifdef VERBOSE
	double t_refine = cv::getTickCount();
	t_cost = (t_refine - t_end) / cv::getTickFrequency() * 1000;
	cout << "轮廓预处理耗时(ms)： " << t_cost << endl;
#endif

	edges = _EdgeDetector.GetFinalContours();
	cv::Mat rgb_contour;
	cv::cvtColor(temp.clone(), rgb_contour, cv::COLOR_GRAY2BGR);
	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(15.0);//100
	_EllipseDetector.DetectEllipses(temp, edges);
	_EllipseDetector.DrawEllipses();

#ifdef VERBOSE
	double t_ell = cv::getTickCount();
	t_cost = (t_ell - t_refine) / cv::getTickFrequency() * 1000;
	cout << "椭圆检测耗时(ms)： " << t_cost << endl;
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
	
	//2018年10月25日 测试 Mat 读写
	string savefilename = "../Data/Out/testcoarsepose.txt";
	vector<cv::Mat> PoseMats = CTestCoarse.GetCoarsePoses();
	cv::Mat result = PoseMats[0];
	WriteCVPoseMatToFile(savefilename, result);
	cv::Mat read = ReadCVPoseMatFromFile(savefilename);
	cout << "successful R and W: " <<endl << read << endl;
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
	sceneGenerator.SetViewerSize(960, 540);
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
	cv::Mat testimg = cv::imread("F:/Datasets/EPFL_can/video2/frame00080.png");
	//cv::Mat testimg = cv::imread("../Data/coarse2.png");
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;

	//Ini Pose Estimator
	PoseEstimation CTestFine;
// 	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
// 		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) <<
		2666.67, 0, 960,
		0, 2666.67, 540,
		0, 0, 1);
	string IveModelName = "F:/Datasets/EPFL_can/CAN-TestAndInfo/transcan.ive";
	float modelRadius = 0.042;
// 	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) <<
// 		1, 0, 0, 0,
// 		0, 0, 1, 0,
// 		0, -1, 0, 0,
// 		0, 0, -0.086, 1);//模型不变，index为0
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);
	CTestFine.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	double t_begin = cv::getTickCount();
	test = testimg.clone();
	_EdgeDetector.SetSrcImg(test);
	contour = _EdgeDetector.CannyContourDetection(100, 200);
	double t_end = cv::getTickCount();
	double t_cost = (t_end - t_begin) / cv::getTickFrequency() * 1000;
	cout << "轮廓检测耗时(ms)： " << t_cost << endl;

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
 	temp = _EdgeDetector.FilterLength(temp, 10);
	edges = _EdgeDetector.GetFinalContours();

	double t_refine = cv::getTickCount();
	t_cost = (t_refine - t_end) / cv::getTickFrequency() * 1000;
	cout << "轮廓预处理耗时(ms)： " << t_cost << endl;

	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(100.0);//100
	_EllipseDetector.DetectEllipses(temp, edges);

	double t_ell = cv::getTickCount();
	t_cost = (t_ell - t_refine) / cv::getTickFrequency() * 1000;
	cout << "椭圆检测处理耗时(ms)： " << t_cost << endl;

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	ellRects = _EllipseDetector.GetEllRects();

	vector<cv::Mat> CoarsePoses;
	CTestFine.CalCoarsePoses(ellMats);
	CoarsePoses = CTestFine.GetCoarsePoses();
	double t_coarsepose = cv::getTickCount();
	t_cost = (t_coarsepose - t_ell) / cv::getTickFrequency() * 1000;
	cout << "位姿粗估计处理耗时(ms)： " << t_cost << endl;

	if (CoarsePoses.size() == 0)
		return;
	//Select CandidatePose using default gradient mode
	CTestFine.SetCapImg(testimg);
	CTestFine.SelectCandidatePose(CoarsePoses, ellRects,2);
	double t_candipose = cv::getTickCount();
	t_cost = (t_candipose - t_coarsepose) / cv::getTickFrequency() * 1000;
	cout << "迭代起点选择处理耗时(ms)： " << t_cost << endl;
	
	//cout << "The candidate ell's index is: " << CTestFine.GetCandidateEllIndex() << endl;
	//cout << "The candidate pose matrix is: " << endl << CTestFine.GetCandidatePose() << endl;

	//Cal Fine pose and show AR Registered img
	CTestFine.CalFinePoseBy3DIC41DOF();
	double t_finepose = cv::getTickCount();
	t_cost = (t_finepose - t_candipose) / cv::getTickFrequency() * 1000;
	cout << "位姿精估计处理耗时(ms)： " << t_cost << endl;

#ifdef VERBOSE
	cv::Mat FineARImg;
	FineARImg = CTestFine.GetFineImg();
	imshow("fine", FineARImg);
	waitKey(50);
#endif

#ifdef SAVEDATA
// 	string saveImgName = "../Data/Out/FineAR.jpg";
// 	string saveSrcName = "../Data/Out/SrcImg.jpg";
// 	string savePoseMatName = "../Data/Out/FinePose.txt";
	string saveSrcName = "F:/Datasets/EPFL_can/YXYexp/SrcImg.jpg";
	string saveEllName = "F:/Datasets/EPFL_can/YXYexp/EllImg.jpg";
	string saveImgName = "F:/Datasets/EPFL_can/YXYexp/FineAR.jpg";
	string savePoseMatName = "F:/Datasets/EPFL_can/YXYexp/FinePose.txt";
	cv::Mat ARMat,EllImg;
	EllImg = _EllipseDetector.GetSrcImg();
	ARMat = CTestFine.GetFineImg();
	imwrite(saveImgName, ARMat);
	imwrite(saveSrcName, testimg);
	imwrite(saveEllName, EllImg);
	cv::Mat FinePoseMat;
	FinePoseMat = CTestFine.GetFinePose();
	WriteCVPoseMatToFile(savePoseMatName, FinePoseMat);
#endif
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

	cap.open("D:/Datasets/IJAMT/nedplus.mp4");
	int framenumber = 0;
	while (cap.isOpened())
	{
		framenumber++;
		cout << "addressing the " << framenumber << " th frame." << endl;
		cap >> testimg;
		test = testimg.clone();
		imshow("src", testimg);
		waitKey(10);
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

		CTestFine.GetARGenerator().SetReInitialize(true);
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

void TestCal6DPoseError()
{
	ofstream outfile;
	outfile.open("../Data/Out/savedata.csv");
	outfile << "tx" << ',' << "ty" << ',' << "tz" << ','
		<< "rx" << ',' << "ry" << ',' << "rz" << endl;
	Mat test_gt = (cv::Mat_<double>(4, 4) << 0.3929252092290151, 0.8621589373979998, -0.3198308093618978, 127.9172690262129,
		-0.91957043229561, 0.3683937292547413, -0.1366611879556503, 60.04240737299161,
		0, 0.3478045814973347, 0.9375670499166787, -871.9922573477522,
		0, 0, 0, 1);
	Mat test_est = (cv::Mat_<double>(4, 4) << 0.3721778266686779, 0.868786669052228, -0.3266395925573681, 127.9172690262129,
		-0.9281044312317063, 0.352249729216505, -0.1205916595269381, 60.04240737299161,
		0.0102902820154755, 0.3480371846190551, 0.9374242679549672, -871.9922573477522,
		0, 0, 0, 1);
	PoseEstimation validator;
	validator.Cal6DPoseError(test_gt, test_est, outfile, true);
	outfile.close();
}

void SyntheticValidator()
{
	PoseEstimation SynDataGenerator;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "../Data/Temp/cylinder.ive";
	float modelRadius = 178;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << -1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1);
	SynDataGenerator.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	string IniPoseFilePath = "../Data/Temp/testInitialPose.txt";
	Mat InitialPose = ReadCVPoseMatFromFile(IniPoseFilePath);

	Mat NormVec_X = (cv::Mat_<double>(1, 3) << 1, 0, 0);
	Mat NormVec_Y = (cv::Mat_<double>(1, 3) << 0, 1, 0);
	Mat NormVec_Z = (cv::Mat_<double>(1, 3) << 0, 0, 1);
	Mat NormVecTest = NormVec_Z;

	vector<Mat> TestPoses;
	double TestRotRange = PI*2 / 3;
	double TestRotStep = PI / 36;
	double TestTransRange = 200;
	double TestTransStep = 5;
	//TestPoses = SynDataGenerator.GenRotPoses(InitialPose, NormVecTest, TestRotRange, TestRotStep);
	TestPoses = SynDataGenerator.GenTransPoses(InitialPose, NormVecTest, TestTransRange, TestTransStep);
	//Save pose error and Img Score Error
	string SaveCoarsePoseFilePath = "../Data/Out/CoarsePose_TransZ.csv";
	string SaveKpsPoseFilePath = "../Data/Out/KpsPose_TransZ.csv";
	string SaveFinePoseFilePath = "../Data/Out/FinePose_TransZ.csv";
	string SaveScorePoseFilePath = "../Data/Out/ImgScore_TransZ.csv";

	ofstream ofile_coarse;
	ofstream ofile_fine;
	ofstream ofile_kps;
	ofstream ofile_score;

	ofile_coarse.open(SaveCoarsePoseFilePath);
	ofile_kps.open(SaveKpsPoseFilePath);
	ofile_fine.open(SaveFinePoseFilePath);
	ofile_score.open(SaveScorePoseFilePath);
	ofile_coarse << "tx" << ',' << "ty" << ',' << "tz" << ','
		<< "rx" << ',' << "ry" << ',' << "rz" << endl;
	ofile_kps << "tx" << ',' << "ty" << ',' << "tz" << ','
		<< "rx" << ',' << "ry" << ',' << "rz" << endl;
	ofile_fine << "tx" << ',' << "ty" << ',' << "tz" << ','
		<< "rx" << ',' << "ry" << ',' << "rz" << endl;
	ofile_score << "Coarse" << ',' << "Kps" << ',' << "Proposed" << endl;

	for (int i = 0; i < TestPoses.size(); ++i)
	{
		Mat GtPose = TestPoses[i];
		Mat testimg = SynDataGenerator.GenerateTemplateImg(GtPose);
		SynDataGenerator.SetCapImg(testimg);

		//Ellipse Detection
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

		//Test Coarse Pose
		vector<cv::Mat> CoarsePoses;
		SynDataGenerator.CalCoarsePoses(ellMats);
		CoarsePoses = SynDataGenerator.GetCoarsePoses();
		SynDataGenerator.SelectCandidatePose(CoarsePoses, ellRects,1);
		Mat CoarsePose = SynDataGenerator.GetCandidatePose();
		SynDataGenerator.Cal6DPoseError(GtPose, CoarsePose, ofile_coarse);
		float fCoarseScore = SynDataGenerator.GetFinalScore();
		ofile_score << fCoarseScore << ",";

		//Test Kps Pose
		SynDataGenerator.CalFinePoseByKpsHomography();
		Mat KpsPose = SynDataGenerator.GetFinePose();
		SynDataGenerator.Cal6DPoseError(GtPose, KpsPose, ofile_kps);
		float fKpsScore = SynDataGenerator.GetFinalScore();
		ofile_score << fKpsScore << ",";

		//Test Fine Pose
		SynDataGenerator.CalFinePoseBy3DIC41DOF();
		Mat FinePose = SynDataGenerator.GetFinePose();
		SynDataGenerator.Cal6DPoseError(GtPose, FinePose, ofile_fine);
		float fFineScore = SynDataGenerator.GetFinalScore();
		ofile_score << fFineScore << "," << endl;
	}
	ofile_coarse.close();
	ofile_kps.close();
	ofile_fine.close();
	ofile_score.close();
}

void TestGearbox()
{
	cv::Mat testimg = cv::imread("D:/Datasets/GearboxImgs/Picture17.jpg");
	//cv::Mat testimg = cv::imread("../Data/coarse2.png");
	cv::Mat test, contour, temp, result;
	vector<vector<cv::Point>> edges;
	EdgeDetection _EdgeDetector;
	EllipseDetection _EllipseDetector;
	vector<ElliFit::Ellipse> ellResult;
	vector<cv::Mat> ellMats;
	vector<cv::Rect> ellRects;

	//Ini Pose Estimator
	PoseEstimation CTestFine;
	cv::Mat Intrinsic = (cv::Mat_<double>(3, 3) << 827.50897692124522, 0, 299.60111699063754,
		0, 814.73836342732341, 256.75622898129393, 0, 0, 1);
	string IveModelName = "D:/Datasets/GearboxModels/REVERSEGEAR.ive";
	float modelRadius = 15;
	cv::Mat ObjectTransform = (cv::Mat_<double>(4, 4) << 
		0, 1, 0, 0,
		0, 0, 1, 0,
		1, 0, 0, 0,
		0, 0, 0, 1);
	CTestFine.Initialize(Intrinsic, IveModelName, modelRadius, ObjectTransform);

	double t_begin = cv::getTickCount();
	test = testimg.clone();
	_EdgeDetector.SetSrcImg(test);
	contour = _EdgeDetector.CannyContourDetection(50, 150);
	double t_end = cv::getTickCount();
	double t_cost = (t_end - t_begin) / cv::getTickFrequency() * 1000;
	cout << "轮廓检测耗时(ms)： " << t_cost << endl;

	temp = contour.clone();
	temp = _EdgeDetector.FilterTurning(temp, 5);
	temp = _EdgeDetector.FilterLines(temp);
	temp = _EdgeDetector.FilterLength(temp, 10);
	edges = _EdgeDetector.GetFinalContours();

	double t_refine = cv::getTickCount();
	t_cost = (t_refine - t_end) / cv::getTickFrequency() * 1000;
	cout << "轮廓预处理耗时(ms)： " << t_cost << endl;

	_EllipseDetector.SetSrcImg(test);
	_EllipseDetector.SetFilter_radius(15);//100
	_EllipseDetector.DetectEllipses(temp, edges);

	double t_ell = cv::getTickCount();
	t_cost = (t_ell - t_refine) / cv::getTickFrequency() * 1000;
	cout << "椭圆检测处理耗时(ms)： " << t_cost << endl;

	ellResult = _EllipseDetector.GetEllDetectionResult();
	ellMats = _EllipseDetector.GetEllMatResult();
	ellRects = _EllipseDetector.GetEllRects();

	vector<cv::Mat> CoarsePoses;
	CTestFine.CalCoarsePoses(ellMats);
	CoarsePoses = CTestFine.GetCoarsePoses();
	double t_coarsepose = cv::getTickCount();
	t_cost = (t_coarsepose - t_ell) / cv::getTickFrequency() * 1000;
	cout << "位姿粗估计处理耗时(ms)： " << t_cost << endl;

	if (CoarsePoses.size() == 0)
		return;
	//Select CandidatePose using default gradient mode
	CTestFine.SetCapImg(testimg);
	CTestFine.SelectCandidatePose(CoarsePoses, ellRects, 2);
	double t_candipose = cv::getTickCount();
	t_cost = (t_candipose - t_coarsepose) / cv::getTickFrequency() * 1000;
	cout << "迭代起点选择处理耗时(ms)： " << t_cost << endl;

	//cout << "The candidate ell's index is: " << CTestFine.GetCandidateEllIndex() << endl;
	//cout << "The candidate pose matrix is: " << endl << CTestFine.GetCandidatePose() << endl;

	//Cal Fine pose and show AR Registered img
	CTestFine.CalFinePoseBy3DIC41DOF();
	double t_finepose = cv::getTickCount();
	t_cost = (t_finepose - t_candipose) / cv::getTickFrequency() * 1000;
	cout << "位姿精估计处理耗时(ms)： " << t_cost << endl;

#ifdef VERBOSE
	cv::Mat FineARImg;
	FineARImg = CTestFine.GetFineImg();
	imshow("fine", FineARImg);
	waitKey(50);
#endif

#ifdef SAVEDATA
	string saveImgName = "D:/Datasets/GearboxExperiment/FineAR.jpg";
	//ImageFile = D:\Datasets\GearboxImgs\Picture23.jpg
	// 	string saveSrcName = "../Data/Out/SrcImg.jpg";
	// 	string savePoseMatName = "../Data/Out/FinePose.txt";
	//string saveImgName = "D:/Datasets/GearboxExperiment/FineAR.jpg";
	//string saveSrcName = "D:/Datasets/GearboxExperiment/SrcImg.jpg";
	string saveCoarseMatName = "D:/Datasets/GearboxExperiment/CoarsePose.txt";
	string savePoseMatName = "D:/Datasets/GearboxExperiment/FinePose.txt";
	cv::Mat ARMat;
	ARMat = CTestFine.GetFineImg();
	imwrite(saveImgName, ARMat);
	//imwrite(saveSrcName, testimg);
	cv::Mat FinePoseMat,CandidatePose;
	FinePoseMat = CTestFine.GetFinePose();
	CandidatePose = CTestFine.GetCandidatePose();
	WriteCVPoseMatToFile(saveCoarseMatName, CandidatePose);
	WriteCVPoseMatToFile(savePoseMatName, FinePoseMat);
#endif
}

void RunAllTests()
{
	//cv::Mat testimg = cv::imread("D:/Datasets/GearboxImgs/Picture19.jpg");
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

		cout << "TestRotIterFinePose()......" << endl;
		TestRotIterFinePose();
		cout << ".....................ok" << endl;

		/*cout << "TestRotIterFinePoseVideo()......" << endl;
		TestRotIterFinePoseVideo();
		cout << ".....................ok" << endl;*/

		/*cout << "TestCal6DPoseError()......" << endl;
		TestCal6DPoseError();
		cout << ".....................ok" << endl;*/

		/*cout << "SyntheticValidator()......" << endl;
		SyntheticValidator();
		cout << ".....................ok" << endl;*/

		/*cout << "TestGearbox()......" << endl;
		TestGearbox();
		cout << ".....................ok" << endl;*/
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

Mat ReadCVPoseMatFromFile(string & filename)
{
	Mat result = Mat::zeros(4, 4, CV_64FC1);
	if (filename == " ")
	{
		cout << "Error: Read pose file path not defined ! " << endl;
		return result;
	}
	ifstream infile;
	infile.open(filename);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			infile >> result.at<double>(i, j);
		}
	}
	infile.close();
	return result;
}

void WriteCVPoseMatToFile(string & filename, Mat & outMat)
{
	if (filename == " ")
	{
		cout << "Error: Out pose file path not defined ! " << endl;
		return;
	}
	ofstream ofile;
	ofile.open(filename);
	for (int i = 0; i < outMat.rows; i++)
	{
		for (int j = 0; j < outMat.cols; j++)
		{
			ofile << outMat.at<double>(i, j)<<"\t";
		}
		ofile << endl;
	}
	ofile.close();
	return;
}
 