/*
Copyright 2014 Alberto Crivellaro, Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
alberto.crivellaro@epfl.ch

terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include "IterativeOptimization.hpp"
#include "Utilities.hpp"
#include "Homography.hpp"

using namespace std;
using namespace cv;

AlignmentResults IterativeOptimization::SSDCalibration(const StructOfArray2di & pixelsOnTemplate, Mat &grayscaleFloatTemplate, Mat &grayscaleFloatImage, vector<float> & parameters, OptimizationParameters & optimizationParameters)
{
	vector<Mat> templateDescriptorFields, imageDescriptorFields;

	templateDescriptorFields.push_back(grayscaleFloatTemplate.clone());
	imageDescriptorFields.push_back(grayscaleFloatImage.clone());
	if (optimizationParameters.bNormalizeDescriptors)
	{
		NormalizeImage(templateDescriptorFields[0]);
		NormalizeImage(imageDescriptorFields[0]);
	}
	return PyramidMultiLevelCalibration(pixelsOnTemplate, templateDescriptorFields, imageDescriptorFields, parameters, optimizationParameters);
}

AlignmentResults IterativeOptimization::DescriptorFieldsCalibration(const StructOfArray2di & pixelsOnTemplate, Mat &grayscaleFloatTemplate, Mat &grayscaleFloatImage, vector<float> &parameters, OptimizationParameters & optimizationParameters)
{
	vector<Mat> templateDescriptorFields, imageDescriptorFields;
	
	ComputeGradientBasedDescriptorFields(grayscaleFloatTemplate, templateDescriptorFields);
	ComputeGradientBasedDescriptorFields(grayscaleFloatImage, imageDescriptorFields);

	if (optimizationParameters.bNormalizeDescriptors)
	{
		for(uint i=0; i < templateDescriptorFields.size(); ++i)
		{
			NormalizeImage(templateDescriptorFields[i]);
			NormalizeImage(imageDescriptorFields[i]);
		}
	}

	return PyramidMultiLevelCalibration(pixelsOnTemplate, templateDescriptorFields, imageDescriptorFields, parameters, optimizationParameters);
}

//yxy 180601 gradient 改为1D 
AlignmentResults IterativeOptimization::GradientModuleCalibration(const StructOfArray2di & pixelsOnTemplate, Mat &grayscaleFloatTemplate, Mat &grayscaleFloatImage, vector<float> &parameters, OptimizationParameters & optimizationParameters)
{
	vector<Mat> templateDescriptorFields, imageDescriptorFields;
	
	ComputeGradientMagnitudeDescriptorFields(grayscaleFloatTemplate, templateDescriptorFields);
	ComputeGradientMagnitudeDescriptorFields(grayscaleFloatImage, imageDescriptorFields);

	//yxy 180601 鉴于 CAD model 和 real object 直接的区别，normalize这一步就很重要了
	if (optimizationParameters.bNormalizeDescriptors)
	{
		NormalizeImage(templateDescriptorFields[0]);
		NormalizeImage(imageDescriptorFields[0]);
	}

	return PyramidMultiLevelCalibration(pixelsOnTemplate, templateDescriptorFields, imageDescriptorFields, parameters, optimizationParameters);
}


AlignmentResults IterativeOptimization::PyramidMultiLevelCalibration(const StructOfArray2di & pixelsOnTemplate, vector<Mat> &templateDescriptorFields, vector<Mat> &imageDescriptorFields, vector<float> & parameters, OptimizationParameters & optimizationParameters)
{
	//TODO: information in pixelsOnTemplate is redundant, find a better way (compute it inside the function ?)
	AlignmentResults alignmentResults, tempResults;
	alignmentResults.nIter = 0;

	//一般 pyramidSmoothingVariance都是非空的，即这段不工作
	if(optimizationParameters.pyramidSmoothingVariance.empty())
	{
		optimizationParameters.maxIterSingleLevel = max(optimizationParameters.maxIter, optimizationParameters.maxIterSingleLevel);
		alignmentResults = GaussNewtonMinimization(pixelsOnTemplate, imageDescriptorFields, templateDescriptorFields, optimizationParameters, parameters);
		return alignmentResults;
	}

	float originalPTol = optimizationParameters.pTol;
	float originalResTol = optimizationParameters.resTol;
	optimizationParameters.pTol = optimizationParameters.pTol * 10;
	optimizationParameters.resTol = optimizationParameters.resTol * 10;

	Mat smoothedImage, smoothedTemplate;
	vector< vector<Mat> > smoothedImages( optimizationParameters.pyramidSmoothingVariance.size());
	vector< vector<Mat> > smoothedTemplates( optimizationParameters.pyramidSmoothingVariance.size());

#pragma omp parallel for
	for (int i=0;i<optimizationParameters.pyramidSmoothingVariance.size();i++)
	{
		smoothedImages[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], imageDescriptorFields);
		smoothedTemplates[i] = SmoothDescriptorFields(optimizationParameters.pyramidSmoothingVariance[i], templateDescriptorFields);
	}
	for (int iLevel = 0;iLevel < optimizationParameters.pyramidSmoothingVariance.size();iLevel++)
	{
		if (iLevel == optimizationParameters.pyramidSmoothingVariance.size()-1)
		{
			optimizationParameters.maxIterSingleLevel = optimizationParameters.maxIter - alignmentResults.nIter;
			optimizationParameters.pTol = originalPTol;
			optimizationParameters.resTol = originalResTol;
		}

#ifdef VERBOSE_OPT
		cout<<"Start using pyramid level no."<< iLevel + 1<<endl;
#endif
		tempResults = GaussNewtonMinimization(pixelsOnTemplate, smoothedImages[iLevel], smoothedTemplates[iLevel], optimizationParameters, parameters);

		alignmentResults.poseIntermediateGuess.insert(alignmentResults.poseIntermediateGuess.end(), tempResults.poseIntermediateGuess.begin(), tempResults.poseIntermediateGuess.end());
		alignmentResults.residualNorm.insert(alignmentResults.residualNorm.end(), tempResults.residualNorm.begin(), tempResults.residualNorm.end());
		alignmentResults.nIter +=tempResults.nIter;
		alignmentResults.exitFlag = tempResults.exitFlag;

#ifdef VERBOSE_OPT
		cout<<"Ended using pyramid level no."<<iLevel+1<<" after "<< tempResults.nIter<<" iterations with local exit flag  "<<tempResults.exitFlag<<endl;
#endif
		if (alignmentResults.nIter > optimizationParameters.maxIter)
			break;

		if(alignmentResults.nIter > 10 && tempResults.exitFlag <= 0)
		{
			if (iLevel == optimizationParameters.pyramidSmoothingVariance.size()-1)
				break;
			else
				iLevel = optimizationParameters.pyramidSmoothingVariance.size()-1;
		}
	}
	return alignmentResults;
}


float IterativeOptimization::ComputeResidualNorm(vector<float> &errorImage)
{
	float residualNorm(0.);
	for(uint i(0); i<errorImage.size(); ++i)
		residualNorm+=errorImage[i]*errorImage[i];
	residualNorm/=errorImage.size();
	return sqrt(residualNorm);
}


int IterativeOptimization::CheckConvergenceOptimization(float deltaPoseNorm, int nIter, float residualNormIncrement, OptimizationParameters optParam)
{
	if (residualNormIncrement < optParam.resTol)
	{
		cout <<endl<< "满足图像 residue 收敛" << endl;
		return -1;
	}
		
	else if (deltaPoseNorm < optParam.pTol)
	{
		cout << endl<<"满足待优化的 homo 参数收敛" << endl;
		return 0;
	}
		
	else if (nIter >= optParam.maxIterSingleLevel)
	{
		cout << endl<<"已经超过迭代的最多次数" << endl;
		return 1;
	}
	return 1e6;
}

inline float estimatorL1L2(const float res)
{
	//estimator
	//http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
	return res/sqrt((1+res*res/2));
}

inline float estimatorGerman(const float res)
{
	//estimator
	//http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
	return res/((1+res*res)*(1+res*res));
}

inline float estimatorTukey(const float res)
{
	//estimator
	//http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
	const float thrs = 1;
	if (abs(res) > thrs)
	{
		//cout<<"0 virgule qqxchose "<<res<<endl;
		return std::numeric_limits<float>::infinity();// // KMYI: Below needs to be removed and everything would be fine?
	}
	else
		return res*(1-(res/thrs)*(res/thrs))*(1-(res/thrs)*(res/thrs));
}

enum class estimator_t {L2,L1L2,Turkey,German};

float estimator(const float res, const estimator_t e = estimator_t::L1L2)
{
	switch(e)
	{
	case estimator_t::L2:
		return res;//L2
	case estimator_t::L1L2:
		return estimatorL1L2(res);
	case estimator_t::Turkey:
		return estimatorTukey(res);
	case estimator_t::German:
		return estimatorGerman(res);
	}
}

void IterativeOptimization::ComputeResiduals(const Mat &image,  vector<float> & templatePixelIntensities, const StructOfArray2di & warpedPixels, vector<float> & errorImage)
{
#pragma omp parallel for
	for(int iPoint = 0; iPoint < warpedPixels.size(); ++iPoint)
	{
		float res = 0;
		if((warpedPixels).x[iPoint] >= 0 && (warpedPixels).x[iPoint] < image.cols && (warpedPixels).y[iPoint] >= 0 && (warpedPixels).y[iPoint] < image.rows)
		{
			res = templatePixelIntensities[iPoint] - ((float*)image.data)[image.cols * warpedPixels.y[iPoint] + warpedPixels.x[iPoint]];
			errorImage[iPoint] = estimator(res);
		}
	}
}

void IterativeOptimization::ComputeWarpedPixels(const StructOfArray2di & pixelsOnTemplate, const vector<float>&  parameters, StructOfArray2di & warpedPixels)
{
	if(warpedPixels.size()!= pixelsOnTemplate.size())
		warpedPixels.resize(pixelsOnTemplate.size());

#pragma omp parallel for
	for(int iPoint = 0; iPoint < pixelsOnTemplate.size(); ++iPoint)
	{
		int x, y;
		Homography::ComputeWarpedPixels(pixelsOnTemplate.x[iPoint],pixelsOnTemplate.y[iPoint], x,y, parameters);
		warpedPixels.x[iPoint] = x;
		warpedPixels.y[iPoint] = y;
	}
}

void IterativeOptimization::AssembleSDImages(const vector<float>&  parameters, const Mat &imageDx, const Mat &imageDy, const StructOfArray2di & warpedPixels, const vector<Eigen::Matrix<float, 2, N_PARAM> >& warpJacobians,  Eigen::MatrixXf & sdImages)
{

	// KMYI: OMP for parallization
	const int stop = warpedPixels.size();
#pragma omp parallel for
	for(int iPoint = 0; iPoint < stop; ++iPoint)
	{
		float imDx, imDy;//, val;
		//val = 0;
		if((warpedPixels).x[iPoint] >= 0 && (warpedPixels).x[iPoint] < imageDx.cols && (warpedPixels).y[iPoint] >= 0 && (warpedPixels).y[iPoint] < imageDx.rows)
		{
			// KMYI: let's try to calculate only once assuming same size for imageDx and iamge Dy
			int warpedIdx = imageDx.cols * warpedPixels.y[iPoint] + warpedPixels.x[iPoint];
			imDx = ((float*)imageDx.data)[warpedIdx];
			imDy = ((float*)imageDy.data)[warpedIdx];
			for(int iParam = 0; iParam < parameters.size(); ++iParam)
				sdImages(iPoint, iParam) = imDx * (warpJacobians[iPoint])(0, iParam) + imDy * (warpJacobians[iPoint])(1, iParam);
		}
		else
			for(int iParam = 0; iParam < parameters.size(); ++iParam)
				sdImages(iPoint, iParam) = 0;
	}
}

// namespace myMem
// {
// // parameters must contain the initial guess. It is updated during optimization
// uint nChannels;
// std::vector<std::vector<float> > templatePixelIntensities;
// vector<Mat> imageDx, imageDy;
// uint nParam;

// vector<Eigen::Matrix<float, 2, N_PARAM> > warpJacobians;
// //AlignmentResults alignmentResults;
// Eigen::MatrixXf sdImages;
// vector<float>  errorImage;

// StructOfArray2di warpedPixels;
// Eigen::Matrix<float, N_PARAM, N_PARAM> hessian;
// Eigen::Matrix<float, N_PARAM, 1> rhs;
// Eigen::Matrix<float, N_PARAM, 1> deltaParam;

// bool init = 0;
// };

// void setInit(const int n)
// {
// 	myMem::init = n;
// }

// void initmyMem(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> &images, const vector<float> & parameters)
// {
// 	//using namespace myMem;
// 	myMem::nChannels = images.size();
// 	myMem::templatePixelIntensities.resize(myMem::nChannels);
// 	myMem::imageDx.resize(myMem::nChannels);
// 	myMem::imageDy.resize(myMem::nChannels);
// 	myMem::nParam = parameters.size();

// 	myMem::warpJacobians.resize(pixelsOnTemplate.size());
// 	//myMem::alignmentResults.nIter = 0;
// 	//myMem::alignmentResults.exitFlag = 1e6;
// 	myMem::sdImages = Eigen::MatrixXf(pixelsOnTemplate.size(), myMem::nParam);
// 	myMem::errorImage = vector<float>(pixelsOnTemplate.size(), 0.0);

// 	myMem::init = 1;
// };

AlignmentResults LucasKanade::GaussNewtonMinimization(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates, const OptimizationParameters optParam, vector<float> & parameters)
{

	// for (int i=0;i<8;i++)
	// 	cout<<parameters[i]<<"\t";
	// cout << endl;

// #define USE_NAMESPACE 0//0 or 1

 	AlignmentResults alignmentResults;
 	alignmentResults.nIter = 0;
 	alignmentResults.exitFlag = 1e6;

// #if USE_NAMESPACE == 1
// 	using namespace myMem;
// 	if (init == 0)
// 	{
// 		printf("in init%u\n",uint(images.size()));
// 		initmyMem(pixelsOnTemplate,images,parameters);
// 		printf("done %u\n",nChannels);
// 	}
// 	//comment from here
// #else
	//parameters must contain the initial guess. It is updated during optimization
	uint nChannels(images.size());
	vector<vector<float> > templatePixelIntensities(nChannels,vector<float>(pixelsOnTemplate.size()));
	vector<Mat> imageDx(nChannels), imageDy(nChannels);
	//2018-5-25 01:00:24 nParam改为1，代表 alpha
	uint nParam = parameters.size();
	//uint nParam = 1;

	vector<Eigen::Matrix<float, 2, N_PARAM> > warpJacobians(pixelsOnTemplate.size());
	Eigen::MatrixXf sdImages(pixelsOnTemplate.size(), nParam);
	vector<float>  errorImage(pixelsOnTemplate.size(), 0.0);

	StructOfArray2di warpedPixels;
	Eigen::Matrix<float, N_PARAM, N_PARAM> hessian; //NXN
	Eigen::Matrix<float, N_PARAM, N_PARAM> invhessian;
	Eigen::Matrix<float, N_PARAM, 1> rhs;   //NX1
	Eigen::Matrix<float, N_PARAM, 1> deltaParam;  //Nx1
//#endif
	//to here



// #if USE_NAMESPACE == 1
// 	//init val
// 	sdImages = Eigen::MatrixXf(pixelsOnTemplate.size(), nParam);
// 	fill(errorImage.begin(), errorImage.end(), 0);//same time as memset in -O3
// #endif

	//vector <Eigen::MatrixXf> tmpHessian(images.size());
	//vector <Eigen::MatrixXf> tmpsdImages(images.size(),sdImages);
	//vector <vector<float> >  tmperrorImage(images.size(),errorImage);

	//#pragma omp parallel for
	//#pragma unroll
	for(int iChannel = 0; iChannel<nChannels; ++iChannel)
	{
		ComputeImageDerivatives(images[iChannel], imageDx[iChannel], imageDy[iChannel]);

		for(int iPoint = 0; iPoint < pixelsOnTemplate.size(); ++iPoint)
		{
			int pos = templates[iChannel].cols* pixelsOnTemplate.y[iPoint] + pixelsOnTemplate.x[iPoint];
			//if (pos < templates[iChannel].total())
			templatePixelIntensities[iChannel][iPoint] = ((float*)templates[iChannel].data)[pos];
			//else
			//	cout<<"tooo big "<<pos<<" / "<<templates[iChannel].size()<<endl;

		}
	}

	while (alignmentResults.exitFlag == 1e6)
	{
		hessian.setZero();
		rhs.setZero();

		ComputeWarpedPixels(pixelsOnTemplate, parameters, warpedPixels);

#pragma omp parallel for
		//compute Jic(x)
		for (int iPoint = 0; iPoint < pixelsOnTemplate.size(); ++iPoint)
			Homography::ComputeWarpJacobian(pixelsOnTemplate.x[iPoint], pixelsOnTemplate.y[iPoint], parameters, warpJacobians[iPoint]);


		//#pragma omp parallel for //reduction(+ : hessian)
		for(int iChannel = 0; iChannel<images.size(); ++iChannel)
		{
			//SD Images: steepest descent images
			AssembleSDImages(parameters, imageDx[iChannel], imageDy[iChannel], warpedPixels, warpJacobians, sdImages);
			//step 7
			ComputeResiduals(images[iChannel], templatePixelIntensities[iChannel], warpedPixels, errorImage);// no need to put errorImage(outsidePixels) = 0, since corresponding row os sdImages is already 0.

			//tmpsdImages[iChannel] = sdImages;
			//tmperrorImage[iChannel] = errorImage;// no need to put errorImage(outsidePixels) = 0, since corresponding row os sdImages is already 0.

			/////////
			//hessian = hessian + sdImages.transpose() * sdImages;
			//compute H

			//step 8
			hessian += sdImages.transpose() * sdImages;
			//tmpHessian[iChannel] = tmpsdImages[iChannel].transpose() * tmpsdImages[iChannel];

//#pragma unroll
			for (int i = 0; i<nParam; ++i)
			{
				for(uint iPoint(0); iPoint<pixelsOnTemplate.size(); ++iPoint)
				{
					float val =	(errorImage[iPoint] == std::numeric_limits<float>::infinity() ? 0: errorImage[iPoint]); 
					//rhs(i,0) += tmpsdImages[iChannel](iPoint,i) * tmperrorImage[iChannel][iPoint];
					rhs(i,0) += sdImages(iPoint,i) * val;
				}
			}
		}

		//for(int iChannel = 0; iChannel<images.size(); ++iChannel)
		//	hessian += tmpsdImages[iChannel].transpose() * tmpsdImages[iChannel];//tmpHessian[iChannel];
		//invhessian = hessian.inverse();
		//deltaParam = hessian.fullPivLu().solve(rhs);//作者src code里用的，虽然在eigen里评价accuracy高，但是实际效果烂
		deltaParam = hessian.partialPivLu().solve(rhs);//效果坠好
		
		//#pragma omp parallel for
		//#pragma unroll
		//FA 的更新方式
		for(int i = 0; i<nParam; ++i)
			parameters[i] += deltaParam(i,0);
	    
		//FC
// 		vector<float> deltap(8,0);
// 		for (int i = 0; i < nParam; ++i)
// 			deltap[i] = deltaParam(i, 0);
// 		Homography::DisplayParameters(deltap);
// 		Homography::ParametersUpdateCompositional(parameters, deltap);
// 		Homography::DisplayParameters(parameters);

		alignmentResults.poseIntermediateGuess.push_back(parameters);
		alignmentResults.residualNorm.push_back(ComputeResidualNorm(errorImage));
		if(alignmentResults.nIter > 0)
			alignmentResults.exitFlag = CheckConvergenceOptimization(deltaParam.norm(), alignmentResults.nIter, abs(alignmentResults.residualNorm[alignmentResults.nIter] - alignmentResults.residualNorm[alignmentResults.nIter-1]), optParam);
		alignmentResults.nIter++;

#ifdef VERBOSE_OPT
		cout<< "iteration n. " <<alignmentResults.nIter<<endl;
		//		cout<< " hessian: "<<endl<<hessian<<endl;
		cout<<"parameters"<<endl;
		for(uint i(0); i<nParam; ++i)
		{
			cout<<"     ["<< i<<"]-> "<< parameters[i];
		}
		cout<<endl;
#endif

	}
	return alignmentResults;
}
