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

float neighbours(unsigned short angle, Point& n1, Point& n2)
{
	if(angle >=180)
		angle -= 180;
	int quarter = angle / 45;
	int weight = angle - quarter;

	switch(quarter)
	{
	case 0:
		n1 = Point(1, 0);
		n2 = Point(1, 1);
		break;
	case 1:
		n1 = Point(1, 1);
		n2 = Point(0, 1);
		break;
	case 2:
		n1 = Point(0, 1);
		n2 = Point(-1, 1);
		break;
	default:
		n1 = Point(-1, 1);
		n2 = Point(-1, 0);
		break;
	}
	return weight / 45.0f;
}

int main()
{
	Mat src, src_gray, src_blur, grad_x, grad_y;
	src = imread(IMAGE, IMREAD_COLOR); // Read the file
	if(! src.data ) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl ;
		return -1;
	}

	cvtColor(src, src_gray, CV_BGR2GRAY);

	//Blur the original image to reduce the noise
	GaussianBlur(src_gray, src_blur, Size(5, 5), 0);

	//Calculate gradients
	Sobel(src_blur, grad_x, CV_16S, 1, 0, 3);
	Sobel(src_blur, grad_y, CV_16S, 0, 1, 3);

	Mat grad(grad_x.rows, grad_x.cols, CV_16S); 
	Mat theta(grad_x.rows, grad_x.cols, CV_16S); 

	for(int i = 0; i < grad.cols; i++)
	{
		for(int j = 0; j < grad.rows; j++)
		{
			grad.at<short>(j, i) = sqrt((grad_x.at<short>(j, i) * grad_x.at<short>(j, i)) + (grad_y.at<short>(j, i) * grad_y.at<short>(j, i)));
			theta.at<short>(j, i) = fastAtan2(grad_y.at<short>(j, i), grad_x.at<short>(j, i));
		}
	}

	Mat grad_border(grad.rows + 2, grad.cols + 2, grad.type());
	copyMakeBorder(grad, grad_border, 1, 1, 1, 1, BORDER_REPLICATE);

	for(int i = 0; i < grad.cols; i++)
	{
		for(int j = 0; j < grad.rows; j++)
		{
			Point n1;
			Point n2;
			float weight = neighbours(theta.at<short>(j, i), n1, n2);
			float g1 = (grad_border.at<short>(j + n1.y + 1, i + n1.x + 1) * weight) 
				+ (grad_border.at<short>(j + n2.y + 1, i + n2.x + 1) * (1 - weight));
			float g2 = (grad_border.at<short>(j - n1.y + 1, i - n1.x + 1) * weight) 
				+ (grad_border.at<short>(j - n2.y + 1, i - n2.x + 1) * (1 - weight));

			if(grad.at<short>(j, i) < g1 || grad.at<short>(j, i) < g2)
				grad.at<short>(j, i) = 0;
		}
	}

	namedWindow( "Original image", WINDOW_AUTOSIZE ); // Create a window for display.
	imshow( "Original image", src_blur ); // Show our image inside it.

	namedWindow( "Grad", WINDOW_AUTOSIZE ); // Create a window for display.
	imshow( "Grad", grad ); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}