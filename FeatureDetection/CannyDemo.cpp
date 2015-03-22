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
#include <opencv2\calib3d\calib3d.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

void CannyDemo::updateThreshold(int value, void* instance)
{
	reinterpret_cast<CannyDemo*>(instance)->updateThreshold(value);
}

CannyDemo::CannyDemo(string img, Mat video, string srcWnd, string dstWnd)
: image(img), srcWindow(srcWnd), dstWindow(dstWnd), threshold(200 - minThreshold), scene(video)
{ }

void CannyDemo::Run()
{
	object_color = imread(image, IMREAD_COLOR);

	if (!object_color.data || !scene.data)
	{
		std::cout << "Error reading images " << std::endl; 
		return;
	}

	Mat object;
	cvtColor(object_color, object, CV_BGR2GRAY);
	
	// Detect the keypoints using SURF Detector
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	vector<KeyPoint> objectKeypoints, sceneKeypoints;

	detector.detect(object, objectKeypoints);
	detector.detect(scene, sceneKeypoints);

	// Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	Mat objectDesc, sceneDesc;

	extractor.compute(object, objectKeypoints, objectDesc);
	extractor.compute(scene, sceneKeypoints, sceneDesc);

	// Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(objectDesc, sceneDesc, matches);

	// calculation of max and min distances between keypoints
	int min_dist = DBL_MAX;
	int max_dist = 0;

	for (int i = 0; i < objectDesc.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("Max dist : %f \n", max_dist);
	printf("Min dist : %f \n", min_dist);

	// Draw only good matches, whose distance is less than 3 * min_dist 
	vector<DMatch> good_matches;

	for (int i = 0; i < objectDesc.rows; i++)
	{
		if (matches[i].distance <= min_dist * 3)
			good_matches.push_back(matches[i]);
	}

	Mat img_matches;
	drawMatches(object, objectKeypoints, scene, sceneKeypoints, good_matches
			  , img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>()
			  , DrawMatchesFlags::DEFAULT);

	// Localize the object
	vector<Point2f> obj;
	vector<Point2f> scn;

	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(objectKeypoints[good_matches[i].queryIdx].pt);
		scn.push_back(sceneKeypoints[good_matches[i].trainIdx].pt);
	}

	if (good_matches.size() >= 4)
	{
		Mat H = findHomography(obj, scn, CV_RANSAC);

		// Get the corners from the object (the object to be detected)
		vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(object.cols, 0);
		obj_corners[2] = cvPoint(object.cols, object.rows); obj_corners[3] = cvPoint(0, object.rows);
		vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);

		// Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(object.cols, 0), scene_corners[1] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(object.cols, 0), scene_corners[2] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(object.cols, 0), scene_corners[3] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(object.cols, 0), scene_corners[0] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
	}

	// Show detected matches
	imshow("Good Matches", img_matches);

	updateThreshold(threshold);
}

void CannyDemo::updateThreshold(int value)
{
	Mat dst;
	int lowT = value + minThreshold;
	int highT = lowT * 3;
	Canny(object_blur, dst, lowT, highT, CV_SCHARR, true);

	//namedWindow(dstWindow, WINDOW_AUTOSIZE);
	//imshow(dstWindow, dst);
}