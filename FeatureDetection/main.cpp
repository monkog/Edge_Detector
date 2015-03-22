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
			CannyDemo demo("black-on-white-on-black.jpg", "Original", "Result");
			demo.Run(frame);
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