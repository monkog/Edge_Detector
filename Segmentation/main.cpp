#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <algorithm>

using namespace cv;
using namespace std;

int spatialR = 20, colorR = 40, maxLevel = 2;
Mat src, dst;

void changeSpatialR(int newValue, void * object)
{
	spatialR = newValue;
	pyrMeanShiftFiltering(src, dst, spatialR, colorR, maxLevel);
	imshow("Segmentation", dst);
}

void changeColorR(int newValue, void * object)
{
	colorR = newValue;
	pyrMeanShiftFiltering(src, dst, spatialR, colorR, maxLevel);
	imshow("Segmentation", dst);
}

void changeMaxLevel(int newValue, void * object)
{
	maxLevel = newValue;
	pyrMeanShiftFiltering(src, dst, spatialR, colorR, maxLevel);
	imshow("Segmentation", dst);
}

void runWatershed()
{
	Mat src_gray, tresh, opening, background, dist, foreground;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	Mat kernel = Mat::ones(3, 3, CV_8UC1);
	double max;

	threshold(src_gray, tresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	morphologyEx(tresh, opening, MORPH_OPEN, kernel, Point(-1, -1), 2);
	dilate(opening, background, kernel, Point(-1, -1), 3);
	distanceTransform(opening, dist, CV_DIST_L2, 5);
	minMaxLoc(dist, nullptr, &max);
	threshold(dist, foreground, 0.7 * max, 255, 0);
	convertScaleAbs(foreground, foreground);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarching;

	findContours(foreground, contours, hierarching, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	Mat markers = Mat::zeros(foreground.size(), CV_32S);
	int component = 0;
	for (int i = 0; i >= 0; i = hierarching[i][0], ++component)
		drawContours(markers, contours, i, Scalar::all(component + 1), -1, 8, hierarching);

	for (int x = 0; x < background.cols; x++)
		for (int y = 0; y < background.rows; y++)
			if (background.at<uchar>(y, x) == 0)
				markers.at<int>(y, x) = component + 1;

	watershed(src, markers);
	vector<Vec3b> colors;

	for (int i = 0; i < component; i++)
	{
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);

		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	for (int x = 0; x < markers.cols; x++)
		for (int y = 0; y < markers.rows; y++)
		{
			if (markers.at<int>(y, x) == -1) src.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
			else if (markers.at<int>(y, x) <= 0 || markers.at<int>(y, x) > component) src.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
			else src.at<Vec3b>(y, x) = colors[markers.at<int>(y, x) - 1];
		}

	imshow("Watershed", src);
}

int main(int argc, char* argv[])
{
	src = imread("coins.jpg", IMREAD_COLOR);

	if (!src.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Segmentation", CV_WINDOW_AUTOSIZE);

	createTrackbar("spatial radius", "Segmentation", &spatialR, 100, &changeSpatialR);
	createTrackbar("color radius", "Segmentation", &colorR, 100, &changeColorR);
	createTrackbar("maxLevel", "Segmentation", &maxLevel, 5, &changeMaxLevel);

	pyrMeanShiftFiltering(src, dst, spatialR, colorR, maxLevel);
	imshow("Segmentation", dst);

	runWatershed();

	waitKey(0);
	return 0;
}