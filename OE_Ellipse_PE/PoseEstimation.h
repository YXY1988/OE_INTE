#pragma once
#include "EllipseDetection.h"
#include "YXYUtils.h" //ָ����
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

/*��Ellipse/Ellipses����Pose,����֮֡��pose�����Ż�������һ��Ŀ�꺯��*/
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
	//��ʼ��ģ�ͺ�����ڲ�����Ϣ����Ҫ�ڼ�������з������ء�pose �� bg Img �Ǳ����������ء�
	void Initialize(cv::Mat & Intrinsic, string & ModelPath, float & ModelRadius, cv::Mat & ObjectTransform);
	cv::Mat GenerateTemplateImg(cv::Mat & pose);
	cv::Mat GenerateARImg(cv::Mat & pose, cv::Mat & bgImg);
	//TODO:�� pose ����Template Img,����Ҫbg������ҪIntrinsic��model
	void CalCoarsePoses(vector<cv::Mat> & ellMats);//set radius,K,����ellMats����2��coarse pose

    //��������ͼ��� ROI �� Error,����SelectCandidate
	float CalImgError(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates);
	float CalImgErrorBySSD(cv::Mat & m_CapImg, cv::Mat & m_TmplImg);
	float CalImgErrorByGF(cv::Mat & m_CapImg, cv::Mat & m_TmplImg);
	float CalImgErrorByDF(cv::Mat & m_CapImg, cv::Mat & m_TmplImg);

	// Mode 1 - SSD, 2 - GF, 3 - DF
	void SelectCandidatePose(vector<cv::Mat> & CoarsePoses, vector<cv::Rect> & ellRects, int ErrMode=2);
	void SelectFinePoses(vector<cv::Mat> & VecPoses, cv::Rect & rect, Mat& ARImg,int ErrMode = 2);

	void CalFinePoseByDFHomography();//��m_SrcImg,m_candidatePose,m_TemplateImg����homo,��homo�ֽ� r��t
	void CalFinePoseByKpsHomography();//�� HomoMatch�еķ������� homography ���� r,t�ֽ�
	void CalFinePoseBy3DIC41DOF(); //�Լ�д 3D IC �ķ�����git�� ֻ�� 2D IC
	void CalFinePoseBy3DIC46DOF();
	Mat SelectOptimalPose(vector<cv::Mat> & Poses, cv::Rect & rect, cv::Mat & CapRoi, int ErrMode = 2);
	//��ini pose ���� range �Ƕ����� degree ������������ Poses
	vector<cv::Mat> GenRotPoses(cv::Mat & IniPose, cv::Mat & VecNorm, float range, float degree);

	//Todo:���ӻ����������ⲿ�ֿ��Ƿ�Validator��ȥ��ȱ quantitative data
	void ShowARPoseResults();//Mode 3, ��͸�� //����Ѿ�ʵ���ˣ�showһ�¾ͺ��ˣ�ͬʱҲ����save
	void ShowMethodDifference();//Mode 4 ���Ҳ���������object�߿򣬰���ǰPPT�ĵ���qualitative results����ȥ ��Ƶ��� 6DOF error��Image GF error

private:
	cv::Mat m_CapImg;//����Դͼ��,��������ͨͼ��Ҳ������У�����ͼ��
	cv::Mat m_UndistortImg;//����У�����ԭͼ��
	cv::Mat m_Intrinsic;//����PҪ�õ�,������У�����Ҳ������ûУ������
	string m_ModelName;//3Dģ��·��
	float m_ModelRadius;//3Dģ��ֱ�������ڼ���Coarse
	cv::Mat m_ObjectTransform;//3Dģ��������ı任���뽨ģ�й�
	SceneGenerator m_SyncGenerator;//ͼ��ϳ���
	SceneGenerator m_ARGenerator;

	cv::Mat m_TmplImg;//�ϳɵ�ģ��ͼ��
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

