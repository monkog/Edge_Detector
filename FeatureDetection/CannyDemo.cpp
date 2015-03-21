#include "CannyDemo.h"
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

void CannyDemo::updateThreshold(int value, void* instance)
{
	reinterpret_cast<CannyDemo*>(instance)->updateThreshold(value);
}

CannyDemo::CannyDemo(string img, string srcWnd, string dstWnd)
	: image(img), srcWindow(srcWnd), dstWindow(dstWnd), threshold(200 - minThreshold)
{ }

void CannyDemo::Run()
{
	src = imread(image, IMREAD_COLOR);
	Mat scena = imread("scena.jpg", IMREAD_COLOR);

	Mat src_gray, copy;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	GaussianBlur(src_gray, src_blur, Size(5,5), 0);
	namedWindow(srcWindow, WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", srcWindow, &threshold, thresholdRange, updateThreshold, this);

	SurfFeatureDetector detector( 400 );
	SurfDescriptorExtractor extractor;
	vector<KeyPoint> okladkaKP, scenaKP;
	Mat okladkaDesc, scenaDesc;
	FlannBasedMatcher matcher;

	detector.detect(src/*okladka*/, okladkaKP);
	extractor.compute(src, okladkaKP, okladkaDesc);
	detector.detect(scena/*scena*/, scenaKP);
	extractor.compute(scena, scenaKP, scenaDesc);

	vector<DMatch> matches;
	matcher.match(okladkaDesc, scenaDesc, matches);
	int min_dist = DBL_MAX;
	int max_dist = 0;

	for(int i = 0; i < okladkaDesc.rows; i++)
	{
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}

	vector<DMatch> good_matches;

	for(int i = 0; i < okladkaDesc.rows; i++)
	{
		if(matches[i].distance <= std::max(min_dist * 2., 0.02))
			good_matches.push_back(matches[i]);
	}

	drawMatches(src, okladkaKP, scena, scenaKP, good_matches, copy, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	vector<Point2f> obj;
	vector<Point2f> scene;
	
	for(int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(okladkaKP[good_matches[i].queryIdx].pt);
		scene.push_back(scenaKP[good_matches[i].trainIdx].pt);
	}

	imshow(srcWindow, copy);
	updateThreshold(threshold);
	waitKey(0);
}

void CannyDemo::updateThreshold(int value)
{
	Mat dst;
	int lowT = value + minThreshold;
	int highT = lowT*3;
	Canny(src_blur, dst, lowT, highT, CV_SCHARR, true);

	namedWindow(dstWindow, WINDOW_AUTOSIZE);
	imshow(dstWindow, dst);
}