//双目匹配程序
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <stdio.h>
#include <iostream>
#define IMG_B(img,y,x) img.at<Vec3b>(y,x)[0]  
#define IMG_G(img,y,x) img.at<Vec3b>(y,x)[1]  
#define IMG_R(img,y,x) img.at<Vec3b>(y,x)[2] 
using namespace cv;
using namespace std;


int main()
{
	const char* intrinsic_filename = "data/intrinsics.yml";
	const char* extrinsic_filename = "data/extrinsics.yml";
	
	Mat left, right;
	VideoCapture capleft(1); // 打开左摄像头
	VideoCapture capright(0);  //打开右摄像头
	cout << "Press Q to quit the program" << endl;
	for (;;)
	{
		capleft >> left;
		imshow("Left Capture", left);

		capright >> right;
		imshow("Right Capture", right);

		if (cvWaitKey(10) == 'q')
			break;
	
		//设置匹配模式STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3
		int alg = 1;

		//匹配参数
		int SADWindowSize = 5, numberOfDisparities = 256;
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
	}
	return 0;
}
