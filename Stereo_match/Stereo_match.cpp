//双目匹配并输出深度图
//版本:Version 1.2
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <stdio.h>
#include <iostream>
//#define IMG_B(img,y,x) img.at<Vec3b>(y,x)[0]  
//#define IMG_G(img,y,x) img.at<Vec3b>(y,x)[1]  
//#define IMG_R(img,y,x) img.at<Vec3b>(y,x)[2] 
using namespace cv;
using namespace std;

//同一窗口显示左右摄像头图像函数
void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size = cvSize(400, 280));

int main()
{   
	//读取yml双目匹配文件
	const char* intrinsic_filename = "data/intrinsics.yml";
	const char* extrinsic_filename = "data/extrinsics.yml";
	
	vector<Mat> imgs(2);
	
	Mat left, right;
	//打开左摄像头
	VideoCapture capleft(1);
	//打开右摄像头
	VideoCapture capright(0);

	cout << "Press Q to quit the program" << endl;
	
	for (;;)
	{
		capleft >> left;
		//imshow("Left Capture", left);
		capright >> right;
		//imshow("Right Capture", right);
		imgs[0] = left;
		imgs[1] = right;
		MultiImage_OneWin("Mult_Camera_Calibratin", imgs, cvSize(2, 1), cvSize(400, 280));
		//设置匹配模式:STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3
		int alg = 1;

		//匹配参数
		int SADWindowSize = 3, numberOfDisparities = 128;
		////////////////////5////////////////////////256//////

		bool no_display = false;
		float scale = 1.0;

		Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

		int color_mode = -1;
		Mat img1 = left;
		Mat img2 = right;


		Size img_size = img1.size();

		Rect roi1, roi2;
		Mat Q;


		// 读取标定参数
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		Mat M1, D1, M2, D2;
		fs["cameraMatrixL"] >> M1;
		fs["cameraDistcoeffL"] >> D1;
		fs["cameraMatrixR"] >> M2;
		fs["cameraDistcoeffR"] >> D2;
		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, FileStorage::READ);
		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		Mat img1r, img2r;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;


		numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

		sgbm->setPreFilterCap(63);
		int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
		sgbm->setBlockSize(sgbmWinSize);

		int cn = img1.channels();

		sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
		sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
		sgbm->setMinDisparity(0);
		sgbm->setNumDisparities(numberOfDisparities);
		sgbm->setUniquenessRatio(10);
		sgbm->setSpeckleWindowSize(100);
		sgbm->setSpeckleRange(32);
		sgbm->setDisp12MaxDiff(1);
		sgbm->setMode(alg == 2 ? StereoSGBM::MODE_HH : StereoSGBM::MODE_SGBM);

		Mat disp, disp8;

		//显示运行时间
		int64 t = getTickCount();
		sgbm->compute(img1, img2, disp);
		t = getTickCount() - t;
		cout << "Time elapsed: " << t * 1000 / getTickFrequency() << "ms" << endl;

		//disp = dispp.colRange(numberOfDisparities, img1p.cols);

		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
		imshow("Deepwindow",disp8);

		//以下函数将灰度图显示为伪彩色图

		//Mat img = disp8;
		//Mat img_color(img.rows, img.cols, CV_8UC3);//构造伪RGB图像  

		//uchar tmp2 = 0;
		//for (int y = 0;y < img.rows;y++)//把灰度图对应的0～255的数值分别转换成彩虹色：红、橙、黄、绿、青、蓝。  
		//{
		//	for (int x = 0;x < img.cols;x++)
		//	{
		//		tmp2 = img.at<uchar>(y, x);
		//		if (tmp2 <= 51)
		//		{
		//			IMG_B(img_color, y, x) = 255;
		//			IMG_G(img_color, y, x) = tmp2 * 5;
		//			IMG_R(img_color, y, x) = 0;
		//		}
		//		else if (tmp2 <= 102)
		//		{
		//			tmp2 -= 51;
		//			IMG_B(img_color, y, x) = 255 - tmp2 * 5;
		//			IMG_G(img_color, y, x) = 255;
		//			IMG_R(img_color, y, x) = 0;
		//		}
		//		else if (tmp2 <= 153)
		//		{
		//			tmp2 -= 102;
		//			IMG_B(img_color, y, x) = 0;
		//			IMG_G(img_color, y, x) = 255;
		//			IMG_R(img_color, y, x) = tmp2 * 5;
		//		}
		//		else if (tmp2 <= 204)
		//		{
		//			tmp2 -= 153;
		//			IMG_B(img_color, y, x) = 0;
		//			IMG_G(img_color, y, x) = 255 - uchar(128.0*tmp2 / 51.0 + 0.5);
		//			IMG_R(img_color, y, x) = 255;
		//		}
		//		else
		//		{
		//			tmp2 -= 204;
		//			IMG_B(img_color, y, x) = 0;
		//			IMG_G(img_color, y, x) = 127 - uchar(127.0*tmp2 / 51.0 + 0.5);
		//			IMG_R(img_color, y, x) = 255;
		//		}
		//	}
		//}
		//namedWindow("img_ rainbowcolor");
		//imshow("img_ rainbowcolor", img_color);

		if (cvWaitKey(10) == 'q')
			break;
	}
	return 0;
}

//显示函数
void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size)
{

	Mat Disp_Img;
	//Width of source image  
	CvSize Img_OrigSize = cvSize(SrcImg_V[0].cols, SrcImg_V[0].rows);
	//******************** Set the width for displayed image ********************//  
	//Width vs height ratio of source image  
	float WH_Ratio_Orig = Img_OrigSize.width / (float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(100, 100);
	if (Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)ImgMax_Size.width / WH_Ratio_Orig);
	else if (Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)ImgMax_Size.height*WH_Ratio_Orig, ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//******************** Check Image numbers with Subplot layout ********************//  
	int Img_Num = (int)SrcImg_V.size();
	if (Img_Num > SubPlot.width * SubPlot.height)
	{
		cout << "Your SubPlot Setting is too small !" << endl;
		exit(0);
	}
	//******************** Blank setting ********************//  
	CvSize DispBlank_Edge = cvSize(80, 60);
	CvSize DispBlank_Gap = cvSize(15, 15);
	//******************** Size for Window ********************//  
	Disp_Img.create(Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width,
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_8UC3);
	Disp_Img.setTo(0);//Background  
					  //Left top position for each image  
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height)) / 2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);
	CvPoint LT_Pos = LT_BasePos;

	//Display all images  
	for (int i = 0; i < Img_Num; i++)
	{
		//Obtain the left top position  
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		//Writting each to Window's Image  
		Mat imgROI = Disp_Img(Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		resize(SrcImg_V[i], imgROI, Size(ImgDisp_Size.width, ImgDisp_Size.height));

		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}
	putText(Disp_Img, "Left Capture", cv::Point(200, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	putText(Disp_Img, "Right Capture", cv::Point(600, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	//putText(Disp_Img, "Left Chessboard", cv::Point(175, 342), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	//putText(Disp_Img, "Right Chessboard", cv::Point(575, 342), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

	cvShowImage(MultiShow_WinName.c_str(), &(IplImage(Disp_Img)));
}