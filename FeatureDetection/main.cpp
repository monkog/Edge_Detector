#include "CannyDemo.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>
#include <stack>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera

		try
		{
			Mat frame_gray;
			cvtColor(frame, frame_gray, CV_BGR2GRAY);
			CannyDemo demo("black-on-white-on-black.jpg", frame_gray, "Original", "Result");
			demo.Run();
		}
		catch (Exception& e)
		{
			cout << e.err;
		}
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}