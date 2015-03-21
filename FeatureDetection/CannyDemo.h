#pragma once

#include <string>
#include <opencv2\core\core.hpp>

class CannyDemo
{
private:
	std::string image;
	std::string srcWindow;
	std::string dstWindow;
	cv::Mat src, src_blur;
	int threshold;
	static const int minThreshold = 150;
	static const int maxThreshold = 250;
	static const int thresholdRange = maxThreshold - minThreshold;

	static void updateThreshold(int, void*);
	void updateThreshold(int);

public:
	CannyDemo(std::string image, std::string srcWindow, std::string dstWindow);
	void Run();
};