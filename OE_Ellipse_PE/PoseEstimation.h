#pragma once
#include "EllipseDetection.h"
#include "YXYUtils.h" //指数积
#include "SceneGenerator.h"
#include "MarkerValidator.h"
#include "HomographyEstimation.hpp"
//#include "kpshomography.h"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <math.h>

#define PI 3.1415926

using namespace yxy;
using namespace ElliFit;
using namespace std;
using namespace cv::xfeatures2d;

/*从Ellipse/Ellipses计算Pose,连续帧之间pose可以优化，得找一个目标函数*/
class PoseEstimation
{
public:
	PoseEstimation();
	~PoseEstimation();
	
	float GetModelRadius() const { return m_ModelRadius; }
	void SetModelRadius(float val) { m_ModelRadius = val; }
	cv::Mat GetIntrinsic() const { return m_Intrinsic; }
	void SetIntrinsic(cv::Mat val) { m_Intrinsic = val; }
	cv::Mat GetCapImg() const { return m_CapImg; }
	void SetCapImg(cv::Mat val) { m_CapImg = val; }
	std::string GetModelName() const { return m_ModelName; }
	void SetModelName(std::string val) { m_ModelName = val; }
	SceneGenerator GetSyncGenerator() const { return m_SyncGenerator; }
	void SetSyncGenerator(SceneGenerator val) { m_SyncGenerator = val; }
	std::vector<cv::Mat> GetCoarsePoses() const { return m_coarsePoses; }
	void SetCoarsePoses(std::vector<cv::Mat> val) { m_coarsePoses = val; }
	cv::Mat GetTmplImg() const { return m_TmplImg; }
	void SetTmplImg(cv::Mat val) { m_TmplImg = val; }
	cv::Mat GetCandidatePose() const { return m_CandidatePose; }
	void SetCandidatePose(cv::Mat val) { m_CandidatePose = val; }
	cv::Rect GetCandidateRect() const { return m_CandidateRect; }
	void SetCandidateRect(cv::Rect val) { m_CandidateRect = val; }
	int GetCandidateEllIndex() const { return m_iCandidateEllIndex; }
	void SetCandidateEllIndex(int val) { m_iCandidateEllIndex = val; }
	cv::Mat GetFinePose() const { return m_FinePose; }
	void SetFinePose(cv::Mat val) { m_FinePose = val; }
	cv::Mat GetFineImg() const { return m_FineImg; }
	void SetFineImg(cv::Mat val) { m_FineImg = val; }
	SceneGenerator GetARGenerator() const { return m_ARGenerator; }
	void SetARGenerator(SceneGenerator val) { m_ARGenerator = val; }
public:
	//初始化模型和相机内参数信息，不要在计算过程中反复加载。pose 和 bg Img 是变数，不加载。
	void Initialize(cv::Mat & Intrinsic, string & ModelPath, float & ModelRadius, cv::Mat & ObjectTransform);
	cv::Mat GenerateTemplateImg(cv::Mat & pose);
	cv::Mat GenerateARImg(cv::Mat & pose, cv::Mat & bgImg);
	//TODO:由 pose 生成Template Img,不需要bg但是需要Intrinsic和model
	void CalCoarsePoses(vector<cv::Mat> & ellMats);//set radius,K,后用ellMats计算2个coarse pose

    //计算两个图像的 ROI 的 Error,用于SelectCandidate
	float CalImgError(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates);
	float CalImgErrorBySSD(cv::Mat & m_CapImg, cv::Mat & m_TmplImg);
	float CalImgErrorByGF(cv::Mat & m_CapImg, cv::Mat & m_TmplImg);
	float CalImgErrorByDF(cv::Mat & m_CapImg, cv::Mat & m_TmplImg);

	// Mode 1 - SSD, 2 - GF, 3 - DF
	void SelectCandidatePose(vector<cv::Mat> & CoarsePoses, vector<cv::Rect> & ellRects, int ErrMode=2);
	void SelectFinePoses(vector<cv::Mat> & VecPoses, cv::Rect & rect, Mat& ARImg,int ErrMode = 2);

	void CalFinePoseByDFHomography();//用m_SrcImg,m_candidatePose,m_TemplateImg计算homo,把homo分解 r，t
	void CalFinePoseByKpsHomography();//用 HomoMatch中的方法计算 homography 并求 r,t分解
	void CalFinePoseBy3DIC41DOF(); //自己写 3D IC 的方法，git上 只有 2D IC
	void CalFinePoseBy3DIC46DOF();
	Mat SelectOptimalPose(vector<cv::Mat> & Poses, cv::Rect & rect, cv::Mat & CapRoi, int ErrMode = 2);
	//在ini pose 附近 range 角度内以 degree 增量生成若干 Poses
	vector<cv::Mat> GenRotPoses(cv::Mat & IniPose, cv::Mat & VecNorm, float range, float degree);

	//Todo:可视化计算结果（这部分考虑放Validator里去）缺 quantitative data
	void ShowARPoseResults();//Mode 3, 半透明 //这个已经实现了，show一下就好了，同时也可以save
	void ShowMethodDifference();//Mode 4 暂且不整合两个object线框，把以前PPT的当作qualitative results放上去 视频输出 6DOF error，Image GF error

private:
	cv::Mat m_CapImg;//输入源图像,可以是普通图像，也可以是校正后的图像
	cv::Mat m_UndistortImg;//畸变校正后的原图像
	cv::Mat m_Intrinsic;//计算P要用到,可以用校正后的也可以用没校正过的
	string m_ModelName;//3D模型路径
	float m_ModelRadius;//3D模型直径，用于计算Coarse
	cv::Mat m_ObjectTransform;//3D模型绕自身的变换，与建模有关
	SceneGenerator m_SyncGenerator;//图像合成类
	SceneGenerator m_ARGenerator;

	cv::Mat m_TmplImg;//合成的模板图像
	vector<cv::Mat> m_coarsePoses;

	cv::Mat m_CandidatePose;
	cv::Rect m_CandidateRect;
	int m_iCandidateEllIndex;

	vector<cv::Mat> m_FinePoses;
	cv::Mat m_FinePose;
	cv::Mat m_FineImg;

	Ptr<SURF> surf;
	BFMatcher matcher;
};

