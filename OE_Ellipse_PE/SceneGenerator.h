#pragma once
#include "YXYUtils.h"
using namespace yxy;

#include <osgViewer/Viewer>
#include <osgViewer/api/win32/GraphicsWindowWin32>
#include <osgGA/TrackballManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgUtil/Optimizer>
#include <string>
#include <list>
#include <osg/MatrixTransform>
#include <osg/Depth>
#include <osg/ShadeModel>

#include <osg/LightModel>
#include <osg/AlphaFunc>
#include <osg/BufferObject>
#include <osg/BlendFunc>
#include <osg/BlendColor>
#include <osg/PolygonMode>
#include <osg/ComputeBoundsVisitor>
#include <osg/LineWidth>


class SceneGenerator
{
public:
	SceneGenerator();
	~SceneGenerator();
	
	std::string GetModelName() const { return m_ModelName; }
	void SetModelName(std::string val) { m_ModelName = val; }
	std::string GetBgImgName() const { return m_BgImgName; }
	void SetBgImgName(std::string val) { m_BgImgName = val; }
	cv::Mat GetBgImgMat() const { return m_BgImgMat; }
	void SetBgImgMat(cv::Mat val); 
	bool GetUseImgBgFlag() const { return m_bUseImgBg; }
	void SetUseImgBgFlag(bool val) { m_bUseImgBg = val; }

	cv::Mat GetPoseMat() const { return m_PoseMat; }
	void SetPoseMat(cv::Mat val) { m_PoseMat = val; }
	cv::Mat GetCameraIntrinsic() const { return m_Intrinsic; }
	void SetCameraIntrinsic(cv::Mat val) { m_Intrinsic = val; }
	void SetViewerSize(int width, int height);
	void SetObjectSelfTransform(cv::Mat & mat);
	bool GetUseTransparent() const { return m_bUseTransparent; }
	void SetUseTransparent(bool val) { m_bUseTransparent = val; }
	bool GetUseWireframe() const { return m_bUseWireframe; }
	void SetUseWireframe(bool val) { m_bUseWireframe = val; }
	bool GetReInitialize() const { return m_bReInitialize; }
	void SetReInitialize(bool val) { m_bReInitialize = val; }
public:
	void Initialize();
	cv::Mat GetSyntheticImg();
	static osg::Matrix toOSGMat(const cv::Mat&);

protected:
	osg::Texture2D* GetOrCreateBKTexture();
	bool AddBoundingBox(osg::ref_ptr<osg::Group> parentNode, osg::ref_ptr<osg::Node> partNode, const osg::Matrix & refMat, const osg::Vec4 & color);
	void RemoveBoundingBoxs();

private:
	string m_ModelName;
	string m_BgImgName;
	cv::Mat m_BgImgMat;
	bool m_bUseImgBg;
	bool m_bUseTransparent;
	bool m_bUseWireframe;
	bool m_bReInitialize;

	cv::Mat m_PoseMat;
	cv::Mat m_Intrinsic; //相机内参数
	int m_iViewerWidth, m_iViewerHeight;

	typedef std::vector<osg::ref_ptr<osg::MatrixTransform>> BoundingBoxs;
	BoundingBoxs m_BoundingBoxs;

private:
	osg::ref_ptr<osgViewer::Viewer> m_MainViewer;		//Viewer

	osg::ref_ptr<osg::Group> m_Root;				//根节点
	osg::ref_ptr<osg::MatrixTransform> m_Model;				//模型
	
	osg::ref_ptr<osg::Camera> m_BGCamera;			//背景相机
	osg::ref_ptr<osg::Texture2D> m_BGTexture;		//背景贴图

	osg::ref_ptr<osg::Image> m_FBOImage;
	osg::ref_ptr<osg::Camera> m_FBOCamera;
	osg::Matrix m_ObjectSelfTransform;
};

