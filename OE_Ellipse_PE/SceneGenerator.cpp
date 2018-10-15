#include "SceneGenerator.h"
SceneGenerator::SceneGenerator()
{
	m_bUseImgBg = false;
	m_iViewerWidth = 640;
	m_iViewerHeight = 480;
}

SceneGenerator::~SceneGenerator()
{
}

void SceneGenerator::SetBgImgMat(cv::Mat val)
{
	m_BgImgMat = val.clone();
}


void SceneGenerator::SetViewerSize(int width, int height)
{
	m_iViewerWidth = width;
	m_iViewerHeight = height;
}

void SceneGenerator::SetObjectSelfTransform(cv::Mat & mat)
{
	m_ObjectSelfTransform = toOSGMat(mat);
}

void SceneGenerator::Initialize()
{
	m_Root = new osg::Group;

	m_MainViewer = new osgViewer::Viewer;
	m_MainViewer->setSceneData(m_Root);

//	m_MainViewer->setCameraManipulator(new osgGA::TrackballManipulator);


	m_Model = new osg::MatrixTransform;
	m_Root->addChild(m_Model);

	osg::Node* pNode = osgDB::readNodeFile(m_ModelName);
	if (pNode)
		m_Model->addChild(pNode);

	//TODO:这里增加 if（b_Compare ==true）{add boundingbox},或把initialize 函数 按照 mode 1 sample，2 tmpl，3 compare，4 AR

	osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
	traits->x = 100;
	traits->y = 100;
	traits->width = 0;
	traits->height = 0;
	traits->windowDecoration = false;
	traits->doubleBuffer = true;
	traits->sharedContext = 0;
	traits->samples = 16;


	osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(traits.get());
	osgViewer::GraphicsWindow* gw = dynamic_cast<osgViewer::GraphicsWindow*>(gc.get());

	// create the view of the scene.
	m_MainViewer->getCamera()->setGraphicsContext(gc.get());
	m_MainViewer->getCamera()->setViewport(0, 0, m_iViewerWidth, m_iViewerHeight);

	float left, right, bottom, top, zNear = 1.0f, zFar = 1000.f;	//zNear, zFar自动计算
	yxy::CameraParam::GetOsgVirtualCamParam(m_Intrinsic, m_iViewerWidth, m_iViewerHeight, left, right, bottom, top);
	m_MainViewer->getCamera()->setProjectionMatrixAsFrustum(left, right, bottom, top, zNear, zFar);
	m_MainViewer->getCamera()->setViewMatrix(osg::Matrix::identity());
	m_MainViewer->getCamera()->setClearColor(osg::Vec4(51.0 / 255.0, 51.0 / 255.0, 102.0 / 255.0, 1.0));

	if (m_bUseTransparent)
	{
		osg::StateSet* state = pNode->getOrCreateStateSet();
		state->setMode(GL_BLEND, osg::StateAttribute::ON);
		state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
		osg::ref_ptr<osg::Depth> depth = new osg::Depth();
		depth->setWriteMask(false);
		state->setAttributeAndModes(depth, osg::StateAttribute::ON);

		osg::BlendFunc *blendFunc = new osg::BlendFunc();
		osg::BlendColor *blendColor = new osg::BlendColor(osg::Vec4(1, 1, 1, 0.4f));
		blendFunc->setSource(osg::BlendFunc::CONSTANT_ALPHA);
		blendFunc->setDestination(osg::BlendFunc::ONE_MINUS_CONSTANT_ALPHA);

		state->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
		state->setAttributeAndModes(blendColor, osg::StateAttribute::ON);

		state->setRenderBinDetails(10, "RenderBin");
	}

	if (m_bUseWireframe)
	{
		RemoveBoundingBoxs();
		AddBoundingBox(m_Model, pNode, osg::Matrix::identity(), osg::Vec4(1, 0, 1, 1));
	}

	if (m_bUseImgBg)
		GetOrCreateBKTexture()->setImage(ConvertCVMat2OsgImg(m_BgImgMat));

	m_MainViewer->getCamera()->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
	m_FBOImage = new osg::Image();
	m_FBOImage->allocateImage(m_iViewerWidth, m_iViewerHeight, 1, GL_RGBA, GL_UNSIGNED_BYTE);
	m_MainViewer->getCamera()->attach(osg::Camera::COLOR_BUFFER, m_FBOImage,6);

	m_MainViewer->realize();
}

cv::Mat SceneGenerator::GetSyntheticImg()
{
	if (!m_Root || m_bReInitialize == true)
	{
		Initialize();
		m_bReInitialize == false;
	}
		
	osg::Matrix PoseMatrix = toOSGMat(m_PoseMat);

	m_Model->setMatrix(m_ObjectSelfTransform* PoseMatrix);

	m_MainViewer->frame();
	Sleep(100);				//等待写入完成

	return ConvertOsgImg2CVMat(m_FBOImage);
}

osg::Matrix SceneGenerator::toOSGMat(const cv::Mat& mat)
{
	osg::Matrix osgMat;
	if (mat.rows == 4 && mat.cols == 4 && mat.elemSize() == sizeof(double))
	{
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				osgMat(i, j) = mat.at<double>(i, j);
	}

	return osgMat;
}

osg::Texture2D* SceneGenerator::GetOrCreateBKTexture()
{
	if (!m_BGTexture)
	{
		m_BGTexture = new osg::Texture2D;
		m_BGTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
		m_BGTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
		m_BGTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP);
		m_BGTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP);
		m_BGTexture->setResizeNonPowerOfTwoHint(false);

		osg::ref_ptr<osg::Drawable> quad = osg::createTexturedQuadGeometry(osg::Vec3(), osg::Vec3(1.0f, 0.0f, 0.0f), osg::Vec3(0.0f, 1.0f, 0.0f));
		quad->getOrCreateStateSet()->setTextureAttributeAndModes(0, m_BGTexture.get());
		quad->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
		quad->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
		quad->getOrCreateStateSet()->setMode(GL_SHADE_MODEL, osg::ShadeModel::FLAT);
		osg::ref_ptr<osg::Geode> geode = new osg::Geode;
		geode->addDrawable(quad.get());

		m_BGCamera = new osg::Camera;
		m_BGCamera->setClearMask(0);
		m_BGCamera->setCullingActive(false);
		m_BGCamera->setAllowEventFocus(false);
		m_BGCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
		m_BGCamera->setRenderOrder(osg::Camera::NESTED_RENDER);
		m_BGCamera->setProjectionMatrix(osg::Matrix::ortho2D(0.0, 1.0, 0.0, 1.0));
		m_BGCamera->addChild(geode.get());
		osg::StateSet* ss = m_BGCamera->getOrCreateStateSet();
		ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
		ss->setAttributeAndModes(new osg::Depth(osg::Depth::LEQUAL, 1.0, 1.0));

		if (!m_BGCamera->getNumParents())
			m_MainViewer->getCamera()->addChild(m_BGCamera.get());
	}


	return m_BGTexture.get();

	return 0;
}

bool SceneGenerator::AddBoundingBox(osg::ref_ptr<osg::Group> parentNode, osg::ref_ptr<osg::Node> partNode, const osg::Matrix& refMat, const osg::Vec4& color)
{
	osg::BoundingBox box;
	osg::ComputeBoundsVisitor cbv;
	partNode->accept(cbv);//要计算几何模型的包围盒，不要计算矩阵的包围盒
	box = cbv.getBoundingBox();

	if (box._min[0] >= box._max[0] || box._min[1] >= box._max[1] || box._min[2] >= box._max[2])
		return false;

	osg::ref_ptr<osg::Geode> geode = new osg::Geode;

	osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;

	osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
	v->push_back(box.corner(0));
	v->push_back(box.corner(1));
	v->push_back(box.corner(0));
	v->push_back(box.corner(2));
	v->push_back(box.corner(2));
	v->push_back(box.corner(3));
	v->push_back(box.corner(3));
	v->push_back(box.corner(1));
	v->push_back(box.corner(1));
	v->push_back(box.corner(5));
	v->push_back(box.corner(5));
	v->push_back(box.corner(7));
	v->push_back(box.corner(7));
	v->push_back(box.corner(3));
	v->push_back(box.corner(7));
	v->push_back(box.corner(6));
	v->push_back(box.corner(6));
	v->push_back(box.corner(2));
	v->push_back(box.corner(6));
	v->push_back(box.corner(4));
	v->push_back(box.corner(4));
	v->push_back(box.corner(0));
	v->push_back(box.corner(4));
	v->push_back(box.corner(5));

	geom->setVertexArray(v);

	osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
	c->push_back(color);

	geom->setColorArray(c);
	geom->setColorBinding(osg::Geometry::BIND_OVERALL);

	geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 24));

	geode->addDrawable(geom.get());

	// 设置包围盒的多边形渲染模式为多边形
	osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
	polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
	osg::ref_ptr<osg::StateSet> state = geom->getOrCreateStateSet();//实例化一个StateSet
	state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);//关闭灯光

														  //设置线宽
	osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(1.0f);
	state->setAttribute(lw.get());
	geode->addDrawable(geom.get());


	osg::ref_ptr<osg::MatrixTransform> matTrans = new osg::MatrixTransform;
	matTrans->addChild(geode);
	matTrans->setMatrix(refMat);
	m_BoundingBoxs.push_back(matTrans);
	parentNode->addChild(matTrans);
	//matTrans->getOrCreateStateSet()->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);


	return true;
}

void SceneGenerator::RemoveBoundingBoxs()
{
	int num = m_BoundingBoxs.size();
	for (int i = 0;i < num;++i)
	{
		if (m_BoundingBoxs[i]->getNumParents() != 0)
		{
			osg::ref_ptr<osg::Group> parent = m_BoundingBoxs[i]->getParent(0);
			parent->removeChild(m_BoundingBoxs[i].get());
		}
	}
	m_BoundingBoxs.clear();
}