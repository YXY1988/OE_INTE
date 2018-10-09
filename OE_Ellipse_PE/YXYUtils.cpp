#include "YXYUtils.h"

#if _MSC_VER>=1900
#include "stdio.h" 
_ACRTIMP_ALT FILE* __cdecl __acrt_iob_func(unsigned); 
#ifdef __cplusplus 
extern "C"
#endif 
FILE* __cdecl __iob_func(unsigned i) {
	return __acrt_iob_func(i);
}
#endif

/*文件操作*/
void yxy::GetAllFiles(string& path, vector<string>& files)
{
	long   hFile   =   0;    

	struct _finddata_t fileinfo;    
	string p;    
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)    
	{    
		do    
		{     
			if((fileinfo.attrib &  _A_SUBDIR))    
			{    
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)    
				{  
					files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
					GetAllFiles( p.assign(path).append("\\").append(fileinfo.name), files );   
				}  
			}    
			else    
			{    
				files.push_back(p.assign(path).append("\\").append(fileinfo.name) );    
			}   

		}while(_findnext(hFile, &fileinfo)  == 0);    

		_findclose(hFile); 
	}

}
void yxy::GetAllFormatFiles(string& path, string& format, vector<string>& files)
{
	//文件句柄    
	long   hFile   =   0;    
	//文件信息    
	struct _finddata_t fileinfo;    
	string p;    
	if((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(),&fileinfo)) !=  -1)    
	{    
		do    
		{      
			if((fileinfo.attrib &  _A_SUBDIR))    
			{    
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)    
				{  
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
					GetAllFormatFiles( p.assign(path).append("\\").append(fileinfo.name), format,files);   
				}  
			}    
			else    
			{    
				files.push_back(p.assign(path).append("\\").append(fileinfo.name) );    
			}    
		}while(_findnext(hFile, &fileinfo)  == 0);    

		_findclose(hFile);   
	}
}
void yxy::GetFileLines(string& path, int& lineNum)
{
	ifstream infile;
	string temp;
	lineNum = 0;
	infile.open(path,ios::in);
	if (infile.fail())
	{
		return;
	}
	else
	{
		while(getline(infile,temp))
		{
			lineNum++;
		}
	}
	infile.close();
	return;
}

osg::ref_ptr<osg::Image> yxy::ConvertCVMat2OsgImg(cv::Mat mat, bool convertToRGB)
{
	if (mat.empty())
	{
		return NULL;
	}

	cv::Mat tmpMat;
	GLenum type = GL_BGR;

	if (convertToRGB)
	{
		cv::cvtColor(mat, tmpMat, cv::COLOR_BGR2RGB);
		cv::flip(tmpMat, tmpMat, 0);
		type = GL_RGB;
	}
	else
	{
		cv::flip(mat, tmpMat, 0);
	}


	int imageSize = tmpMat.cols * tmpMat.rows * tmpMat.channels();
	unsigned char* pBuffer = new unsigned char[imageSize];

	memcpy(pBuffer, tmpMat.data, imageSize);

	osg::ref_ptr<osg::Image> osgframe = new osg::Image;
	osgframe->setImage(tmpMat.cols, tmpMat.rows, 1, type, type, GL_UNSIGNED_BYTE, pBuffer, osg::Image::USE_NEW_DELETE, 1);

	return osgframe;
}

cv::Mat yxy::ConvertOsgImg2CVMat(osg::ref_ptr<osg::Image> & Img, bool convertToBGR)
{
	cv::Mat mat;

	if (!Img.valid() || !(Img->getPixelFormat() == GL_RGB || Img->getPixelFormat() == GL_RGBA))
		return mat;

	int size = 0;

	if (Img->getPixelFormat() == GL_RGB)
	{
		mat.create(Img->t(), Img->s(), CV_8UC3);
		size = mat.cols * mat.rows * mat.channels();
		memcpy(mat.data, Img->data(), size);

		if (convertToBGR)
			cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
	}
	else
	{
		mat.create(Img->t(), Img->s(), CV_8UC4);
		size = mat.cols * mat.rows * mat.channels();
		memcpy(mat.data, Img->data(), size);

		if (convertToBGR)
			cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
	}

	cv::flip(mat, mat, 0);
	
	return mat;
}


/*CV矩阵操作*/
void yxy::SwapMat(cv::Mat& src, cv::Mat& dst)
{
	if(!src.empty())
		src.copyTo(dst);
	else
		return;
}
void yxy::MergeMatByCol(cv::Mat& temp, cv::Mat& input, cv::Mat& result)
{
	if (temp.cols==input.cols)
	{
		vconcat(temp,input,result);
	}
	else
		input.copyTo(result);
}

/*数据结构操作，排序，计算索引等*/
/*1. 计算 label 统计数最多的类别或索引*/
void yxy::MaxCounter::AddCount(int index, int numofCount)
{
	map<int, int>::iterator it = _counterMap.find(index);
	if(it == _counterMap.end())
		_counterMap[index] = numofCount;
	else
		it->second += numofCount;
}
int yxy::MaxCounter::GetMaxIndex()
{
	int maxIndex = -1;
	int maxNum = -1;

	for(map<int, int>::iterator it = _counterMap.begin(); it != _counterMap.end(); ++it)
	{
		if(maxNum < it->second)
		{
			maxIndex = it->first;
			maxNum = it->second;
		}
	}

	return maxIndex;
}

/*
名称：RoiSelector
功能：框选ROI
*/
yxy::RoiSelector::RoiSelector(void)
{
	scale = 1.0;
	winName = "ROI selector";
	rectState = NOT_SET;
}
yxy::RoiSelector::~RoiSelector(void)
{
}
void yxy::RoiSelector::SetSrcFrame(cv::Mat OutSrc)
{
	SwapMat(OutSrc,src);
}
void yxy::RoiSelector::DrawRect()
{
	if(src.empty()||winName.empty())
		return;
	if(rectState==IN_PROCESS||rectState==SET)
		cv::rectangle(src,
		cv::Point(rect.x, rect.y),
		cv::Point(rect.x+rect.width,rect.y+rect.height),
		cv::Scalar(0,0,255),
		3);
	cv::waitKey(10);
	imshow(winName,src);
}
void yxy::RoiSelector::MouseClick(int event, int x, int y, int flags, void *param)
{
	switch(event)
	{
	case CV_EVENT_LBUTTONDOWN:
		{
			if(rectState==NOT_SET)
			{
				rectState=IN_PROCESS;
				rect=cv::Rect(x,y,1,1);
				origin_x = x;
				origin_y = y;
			}
			if(rectState==SET)
				cout<<"rect already exist!"<<endl;
		}
		break;
	case CV_EVENT_LBUTTONUP:
		if(rectState==IN_PROCESS)
		{
			rect=cv::Rect(cv::Point(rect.x,rect.y), cv::Point(x,y));
			rectState=SET;//画矩形结束
			rect_width = x-rect.x;
			rect_height = y-rect.y;
			DrawRect();
			RoiImg=src;
			RoiRect.x = origin_x;
			RoiRect.y = origin_y;
			RoiRect.width = rect_width;
			RoiRect.height = rect_height;
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		if(rectState==IN_PROCESS)
		{
			rect=cv::Rect(cv::Point(rect.x,rect.y), cv::Point(x,y));
			DrawRect();//绘制中，不断更新
		}
		break;
	case CV_EVENT_RBUTTONUP:
		{
			rectState = NOT_SET;
		}
		break;
	}
}
void yxy::RoiSelector::on_mouse( int event, int x, int y, int flags, void* param )
{
	RoiSelector* pThis = (RoiSelector*)param;
	if(pThis)
		pThis->MouseClick(event,x,y,flags,param);
}
void yxy::RoiSelector::setRoi()
{
	if(src.empty())
	{
		cout<<"Couldn't read src Image"<<endl;
		return;
	}
	cvNamedWindow(winName.c_str(),CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(winName.c_str(),on_mouse,this);
	DrawRect();
}

#pragma region ML_VALIDATOR
/*
名称：Validator
功能：计算 PRCurve、ROC曲线各项指标
*/
yxy::Validator::Validator( cv::Mat & res, cv::Mat & lab, float& thresh)
{
	countFinal(res, lab, thresh, tp, fp, tn, fn);
}
yxy::Validator::~Validator(void)
{
}

float yxy::Validator::getZeroOne()
{
	return (tp+tn)/(tp+tn+fp+fn);
}
float yxy::Validator::getF1(int i)
{
	float p = getPrecision(i);
	float r = getRecall(i);
	return 2*p*r/(1e-5f+p+r);
}
float yxy::Validator::getPrecision(int i)
{
	if(i){return tp/(1e-5f+tp+fp);} 
	else {return tn/(1e-5f+tn+fn);}
}
float yxy::Validator::getRecall(int i)
{

	if(i){return tp/(1e-5f+tp+fn);}
	else {return tn/(1e-5f+fp+tn);}
}
float yxy::Validator::getTPR()
{ 
	return tp/(1e-5f+tp+fn); 
}
float yxy::Validator::getFPR()
{
	return fp/(1e-5f+fp+tn);
}
#pragma endregion ML_VALIDATOR

void yxy::Validator::countFinal( cv::Mat & result, cv::Mat & label_gt, float thresh, float & tp, float & fp, float & tn, float & fn)
{
	if( result.rows != label_gt.rows){ cout << " size unmatch while predicting " << endl; return;}

	tp = fp = tn = fn = 0.0f;
	CurrentThresh = thresh;

	float * p_result = (float*) result.data;
	float * p_label = (float*) label_gt.data;

	for(int sz = result.rows * result.cols ; sz>0; sz--, p_result++, p_label++)	
	{
		if( *p_result >= thresh)  
		{
			if( *p_label == 1.0f) tp += 1.0f; 
			else fp += 1.0f;                  
		}
		else //p_result<thresh negative
		{
			if( *p_label == 1.0f) fn += 1.0f;
			else tn += 1.0f;
		}
	}

	{
		float n = float( result.rows*result.cols);
		fp/=n; tp/=n; tn/=n; fn/=n;
	}
}
void yxy::Validator::countCurrent(cv::Mat& result, cv::Mat& label_gt, float& thresh, int& iCurrent)
{
	if( (result.rows != label_gt.rows)||result.cols!=label_gt.cols)
	{ 
		cout << " size unmatch while predicting " << endl; 
		return;
	}

	testSize   = float (iCurrent*result.cols);
	sampleSize = float (label_gt.rows*label_gt.cols);

	tp = fp = tn = fn = 0.0f;
	resLab = 0.0f;
	resScore = 0.0f;
	gtLab = 0.0f;

	for(int i = 0; i<iCurrent;i++)
	{
		resScore = result.at<float>(i,0);
		gtLab = label_gt.at<float>(i,0);
		if(resScore>=thresh)
		{  resLab = 1.0f;}
		else
		{
			resLab = 0.0f;
		}

		if(resLab == 1.0f)
		{
			if(gtLab == 1.0f) 
				tp+=1.0f;
			else 
				fp+=1.0f;
		}
		else
		{
			if( gtLab == 1.0f) 
				fn += 1.0f;
			else 
				tn += 1.0f;
		}
	}
	for(int j = iCurrent; j<label_gt.rows; j++ )
	{
		if(label_gt.at<float>(j,0) == 1.0f)
			fn += 1.0f;
		else
			tn+=1.0f;
	}

	//float n = float( result.rows*result.cols);
	//fp/=n; tp/=n; tn/=n; fn/=n;
}
void yxy::Validator::display()
{
	cout << "Precision: " << getPrecision(1)  ;
	cout << "  Recall: " << getRecall(1)  ;
	cout << " TPR: "<<getTPR() ;
	cout << " FPR: "<<getFPR() ;
	cout << "  F1: " << getF1() << " 0-1:" << getZeroOne() << endl;
}
void yxy::Validator::save(ofstream& ofile)
{
	ofile<< getPrecision(1)<<","<< getRecall(1)<<","
		<<getTPR()<<","<<getFPR()<<","
		<<getF1(1)<<","<<getZeroOne()<<","<<CurrentThresh<<","<<endl;

}


void yxy::CameraParam::ReadCVCalibParam(string & cvCalibFileName)
{
	bool FSflag = false;
	cv::FileStorage readfs;

	FSflag = readfs.open(cvCalibFileName, cv::FileStorage::READ);
	if (FSflag == false) cout << "Cannot open the file" << endl;
	readfs["Camera_Matrix"] >> cameraMatrix;
	readfs["Distortion_Coefficients"] >> distCoeffs;
	readfs["image_Width"] >> imageSize.width;
	readfs["image_Height"] >> imageSize.height;

	cout << "Camera Matrix = "<< endl<<cameraMatrix << endl 
		<< "Distortion Coefficients = "<<endl<<distCoeffs << endl 
		<< "Image cv::Size = "<<imageSize << endl;

	readfs.release();
}

void yxy::CameraParam::UndistortImg(cv::Mat & src, cv::Mat & result)
{
	if (cameraMatrix.empty())
	{
		cout << "Please Read in CV Calibration parameters first! " << endl;
		return;
	}
	cv::undistort(src, result, cameraMatrix, distCoeffs);
}

void yxy::CameraParam::UndistortImgValid(cv::Mat & src, cv::Mat & result)
{

	if (newIntrinsic.empty())
	{
		cout << "Please use 'IniValidUndistort' parameters first! " << endl;
		return;
	}
	remap(src, temp, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	result = temp;
}

void yxy::CameraParam::UndistrotImgResized(cv::Mat & src, cv::Mat & result)
{
	if (newIntrinsic.empty())
	{
		cout << "Please use 'IniValidUndistort' parameters first! " << endl;
		return;
	}
	remap(src, temp, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	undistort = temp(validroi);
	resize(undistort, result, src.size(), 0, 0, cv::INTER_LINEAR);
}

void yxy::CameraParam::UndistrotImgCropped(cv::Mat & src, cv::Mat & result)
{
	if (newIntrinsic.empty())
	{
		cout << "Please use 'IniValidUndistort' parameters first! " << endl;
		return;
	}
	remap(src, temp, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	undistort = temp(validroi);
	result = undistort;
}

void yxy::CameraParam::IniValidUndistort()
{
	if (cameraMatrix.empty())
	{
		cout << "Please Read in CV Calibration parameters first! " << endl;
		return;
	}
	newIntrinsic = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, &validroi, 0);
	initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newIntrinsic, imageSize, CV_16SC2, map1, map2);
}

void yxy::CameraParam::GetOsgVirtualCamParam(cv::Mat intrinsic, int width, int height, float & left, float & right, float & bottom, float & top)
{
	if (intrinsic.empty())
	{
		cout << "Please Read in CV Calibration parameters first! " << endl;
		return;
	}
	float fx, fy, x0, y0;
	fx = (float)intrinsic.at<double>(0, 0);
	fy = (float)intrinsic.at<double>(1, 1);
	x0 = (float)intrinsic.at<double>(0, 2);
	y0 = (float)intrinsic.at<double>(1, 2);

	left = -x0 / fx;
	right = (width - x0) / fx;
	bottom = -(height - y0) / fy;
	top = y0 / fy;
}
void yxy::CameraParam::GetARTKCamParam(string & arCalibFileName, int dist_version)
{
	if (cameraMatrix.empty())
	{
		cout << "Please Read in CV Calibration parameters first! " << endl;
		return;
	}

	float intr[3][4];
	float dist[4];
	int xsize, ysize;

	xsize = imageSize.width;
	ysize = imageSize.height;

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			intr[j][i] = (float)cameraMatrix.at<double>(j, i);
		}
		intr[j][3] = 0.0f;
	}

	for (int i = 0; i < 4; i++) {
		dist[i] = (float)distCoeffs.at<double>(i, 0);
	}

	convParam(intr, dist, xsize, ysize, &param, dist_version);
	cout << endl << "CV2AR calibration parameter convert success!" << endl;
	arParamDisp(&param);

	char *s = (char*)arCalibFileName.data();
	int saveFlag = arParamSave(s, 1, &param);
	if (saveFlag < 0)
		cout << "Parameter write error!!" << endl;
	else
		cout << "Saved Success" << endl;
}

ARdouble yxy::CameraParam::getSizeFactor(ARdouble dist_factor[], int xsize, int ysize, int dist_function_version)
{
	ARdouble  ox, oy, ix, iy;
	ARdouble  olen, ilen;
	ARdouble  sf, sf1;

	sf = 100.0;

	ox = 0.0;
	oy = dist_factor[7];
	olen = dist_factor[6];
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = dist_factor[6] - ix;
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	ox = xsize;
	oy = dist_factor[7];
	olen = xsize - dist_factor[6];
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = ix - dist_factor[6];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	ox = dist_factor[6];
	oy = 0.0;
	olen = dist_factor[7];
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = dist_factor[7] - iy;
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	ox = dist_factor[6];
	oy = ysize;
	olen = ysize - dist_factor[7];
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = iy - dist_factor[7];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}


	ox = 0.0;
	oy = 0.0;
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = dist_factor[6] - ix;
	olen = dist_factor[6];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}
	ilen = dist_factor[7] - iy;
	olen = dist_factor[7];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	ox = xsize;
	oy = 0.0;
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = ix - dist_factor[6];
	olen = xsize - dist_factor[6];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}
	ilen = dist_factor[7] - iy;
	olen = dist_factor[7];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	ox = 0.0;
	oy = ysize;
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = dist_factor[6] - ix;
	olen = dist_factor[6];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}
	ilen = iy - dist_factor[7];
	olen = ysize - dist_factor[7];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	ox = xsize;
	oy = ysize;
	arParamObserv2Ideal(dist_factor, ox, oy, &ix, &iy, dist_function_version);
	ilen = ix - dist_factor[6];
	olen = xsize - dist_factor[6];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}
	ilen = iy - dist_factor[7];
	olen = ysize - dist_factor[7];
	//ARLOG("Olen = %f, Ilen = %f, s = %f\n", olen, ilen, ilen / olen);
	if (ilen > 0) {
		sf1 = ilen / olen;
		if (sf1 < sf) sf = sf1;
	}

	if (sf == 100.0) sf = 1.0;

	return sf;
}
void yxy::CameraParam::convParam(float intr[3][4], float dist[4], int xsize, int ysize, ARParam *param, int dist_version)
{
	ARdouble   s;
	int      i, j;

	param->dist_function_version = dist_version;
	param->xsize = xsize;
	param->ysize = ysize;

	param->dist_factor[0] = (ARdouble)dist[0];     /* k1  */
	param->dist_factor[1] = (ARdouble)dist[1];     /* k2  */
	param->dist_factor[2] = (ARdouble)dist[2];     /* p1  */
	param->dist_factor[3] = (ARdouble)dist[3];     /* p2  */
	param->dist_factor[4] = (ARdouble)intr[0][0];  /* fx  */
	param->dist_factor[5] = (ARdouble)intr[1][1];  /* fy  */
	param->dist_factor[6] = (ARdouble)intr[0][2];  /* x0  */
	param->dist_factor[7] = (ARdouble)intr[1][2];  /* y0  */
	param->dist_factor[8] = (ARdouble)1.0;         /* s   */

	for (j = 0; j < 3; j++) {
		for (i = 0; i < 4; i++) {
			param->mat[j][i] = (ARdouble)intr[j][i];
		}
	}

	s = getSizeFactor(param->dist_factor, xsize, ysize, param->dist_function_version);
	param->mat[0][0] /= s;
	param->mat[0][1] /= s;
	param->mat[1][0] /= s;
	param->mat[1][1] /= s;
	param->dist_factor[8] = s;//实际上是fx,fy 除以了一个比例因子
}