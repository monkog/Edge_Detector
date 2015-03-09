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

const string IMAGE = "building.jpg";

int main()
{
	Mat src;
	src = imread(IMAGE, IMREAD_COLOR); // Read the file

	if(! src.data ) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl ;
		return -1;
	}

	namedWindow( "Original image", WINDOW_AUTOSIZE ); // Create a window for display.
	imshow( "Original image", src ); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}