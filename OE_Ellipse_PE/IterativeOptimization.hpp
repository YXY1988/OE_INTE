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


#ifndef ITERATIVEOPTIMIZATION_HPP_
#define ITERATIVEOPTIMIZATION_HPP_
#include "opencv2/opencv.hpp"
#include "Typedefs.hpp"
#include <Eigen/Dense>
#include <Eigen/LU>

#include <math.h>
#include <vector>
using namespace std;
using namespace cv;

class IterativeOptimization{
public:
	AlignmentResults SSDCalibration(const StructOfArray2di & pixelsOnTemplate, Mat &grayscaleFloatTemplate, Mat &grayscaleFloatImage, vector<float> &parameters, OptimizationParameters & optimizationParameters);
	AlignmentResults DescriptorFieldsCalibration(const StructOfArray2di & pixelsOnTemplate, Mat &grayscaleFloatTemplate, Mat &grayscaleFloatImage, vector<float> &paramters, OptimizationParameters & optimizationParameters);
	AlignmentResults GradientModuleCalibration(const StructOfArray2di & pixelsOnTemplate, Mat &grayscaleFloatTemplate, Mat &grayscaleFloatImage, vector<float> &paramters, OptimizationParameters & optimizationParameters);

	//these should be protected but there are tests on them.
	void ComputeWarpedPixels(const StructOfArray2di & pixelsOnTemplate, const vector<float>&  parameters, StructOfArray2di & warpedPixels);
	void AssembleSDImages(const vector<float>&  parameters, const Mat &imageDx, const Mat &imageDy, const StructOfArray2di & warpedPixels, const vector<Eigen::Matrix<float, 2, N_PARAM> >& warpJacobians,  Eigen::MatrixXf & sdImages);
	void ComputeResiduals(const Mat &image, vector<float> & templatePixelIntensities, const StructOfArray2di & warpedPixels, vector<float> & errorImage);
	float ComputeResidualNorm(vector<float> &errorImage);

protected:
	//父类某函数只声明，未实现；子类继承了父类的函数名，并进行了具体实现；父类其他函数调用这个函数名时，自动使用了子类的实现。那么使用子类的目的是什么？T_T眼镜看花了，是有virtual的
	virtual AlignmentResults GaussNewtonMinimization(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates, const OptimizationParameters optParam, vector<float> & parameters) = 0;
	AlignmentResults PyramidMultiLevelCalibration(const StructOfArray2di & pixelsOnTemplate, vector<Mat> &templateDescriptorFields, vector<Mat> &imageDescriptorFields, vector<float> & parameters, OptimizationParameters & optimizationParameters);
	int CheckConvergenceOptimization(float deltaPoseNorm, int nIter, float residualNormIncrement, OptimizationParameters optParam);

};


class LucasKanade: public IterativeOptimization{
private:
	
	AlignmentResults GaussNewtonMinimization(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates, const OptimizationParameters optParam, vector<float> & parameters);

};

// TODO: implement this
//class ICA: public IterativeOptimization{
//private:
//	AlignmentResults GaussNewtonMinimization(const StructOfArray2di & pixelsOnTemplate, const vector<Mat> & images, const vector<Mat> & templates, const OptimizationParameters optParam, vector<float> & parameters);
//};

#endif /* ITERATIVEOPTIMIZATION_HPP_*/
