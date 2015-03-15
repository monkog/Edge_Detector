#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>
#include <stack>
#include <list>
#include <tuple>

using namespace cv;
using namespace std;

const string IMAGE = "building.jpg";

/// <summary>
/// Finds the neighbours for the specified angle.
/// </summary>
/// <param name="angle">The angle.</param>
/// <param name="n1">The 1st neighbour.</param>
/// <param name="n2">The 2nd neighbour.</param>
/// <returns>The weight of influence from the neighbours</returns>
float neighbours(unsigned short angle, Point& n1, Point& n2)
{
	if (angle >= 180)
		angle -= 180;
	int quarter = angle / 45;
	int weight = angle - quarter;

	switch (quarter)
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
/// <summary>
/// Calculates the grad and theta matrices.
/// </summary>
/// <param name="grad">The grad matrix.</param>
/// <param name="theta">The theta matrix.</param>
/// <param name="grad_x">The grad_x matrix.</param>
/// <param name="grad_y">The grad_y matrix.</param>
void calculateGradAndTheta(Mat grad, Mat theta, Mat grad_x, Mat grad_y)
{
	for (int i = 0; i < grad.cols; i++)
	{
		for (int j = 0; j < grad.rows; j++)
		{
			grad.at<short>(j, i) = sqrt((grad_x.at<short>(j, i) * grad_x.at<short>(j, i))
				+ (grad_y.at<short>(j, i) * grad_y.at<short>(j, i)));
			theta.at<short>(j, i) = fastAtan2(grad_y.at<short>(j, i), grad_x.at<short>(j, i));
		}
	}
}
/// <summary>
/// Calculates the grad border.
/// </summary>
/// <param name="grad">The grad matrix.</param>
/// <param name="theta">The theta matrix.</param>
/// <param name="grad_border">The grad border matrix.</param>
void calculateGradBorder(Mat grad, Mat theta, Mat grad_border)
{
	for (int i = 0; i < grad.cols; i++)
	{
		for (int j = 0; j < grad.rows; j++)
		{
			Point n1;
			Point n2;
			float weight = neighbours(theta.at<short>(j, i), n1, n2);
			float g1 = (grad_border.at<short>(j + n1.y + 1, i + n1.x + 1) * weight)
				+ (grad_border.at<short>(j + n2.y + 1, i + n2.x + 1) * (1 - weight));
			float g2 = (grad_border.at<short>(j - n1.y + 1, i - n1.x + 1) * weight)
				+ (grad_border.at<short>(j - n2.y + 1, i - n2.x + 1) * (1 - weight));

			if (grad.at<short>(j, i) < g1 || grad.at<short>(j, i) < g2)
				grad.at<short>(j, i) = 0;
		}
	}
}
/// <summary>
/// Finds the high values.
/// </summary>
/// <param name="grad">The grad matrix.</param>
/// <param name="highT">The high value.</param>
/// <returns>List of matrix indices and their values, that are greater than high value</returns>
list<tuple<tuple<int, int>, short>> findHighValues(Mat grad, short highT)
{
	list<tuple<tuple<int, int>, short>> values;

	for (int i = 0; i < grad.cols; i++)
	{
		for (int j = 0; j < grad.rows; j++)
		{
			if (grad.at<short>(j, i) > highT)
			{
				tuple<int, int> pos(j, i);
				tuple<tuple<int, int>, short> tup(pos, grad.at<short>(j, i));

				values.push_back(tup);
			}
		}
	}
	return values;
}
/// <summary>
/// Calculates the white border.
/// </summary>
/// <param name="values">The values.</param>
/// <param name="white_border">The white border matrix.</param>
/// <param name="grad">The grad matrix.</param>
/// <param name="theta">The theta matrix.</param>
/// <param name="lowT">The low value.</param>
void calculateWhiteBorder(list<tuple<tuple<int, int>, short>> values, Mat white_border, Mat grad, Mat theta, short lowT)
{
	while (!values.empty())
	{
		tuple<tuple<int, int>, short> tup = values.front();
		tuple<int, int> pos = get<0>(tup);
		short value = get<1>(tup);
		values.pop_front();

		white_border.at<short>(get<0>(pos), get<1>(pos)) = 255;

		Point n1, n2;
		neighbours(theta.at<short>(get<0>(pos), get<1>(pos)), n1, n2);

		if (grad.at<short>(get<0>(pos) +n1.y, get<1>(pos) +n1.x) > lowT && white_border.at<short>(get<0>(pos) +n1.y, get<1>(pos) +n1.x) != 255)
		{
			tuple<int, int> pos(get<0>(pos) +n1.y, get<1>(pos) +n1.x);
			tuple<tuple<int, int>, short> tup(pos, grad.at<short>(n1.y, n1.x));

			values.push_back(tup);
		}
		if (grad.at<short>(get<0>(pos) -n1.y, get<1>(pos) -n1.x) > lowT && white_border.at<short>(get<0>(pos) -n1.y, get<1>(pos) -n1.x) != 255)
		{
			tuple<int, int> pos(get<0>(pos) -n1.y, get<1>(pos) -n1.x);
			tuple<tuple<int, int>, short> tup(pos, grad.at<short>(n1.y, n1.x));

			values.push_back(tup);
		}
	}
}

int main()
{
	Mat src, src_gray, src_blur, grad_x, grad_y;
	src = imread(IMAGE, IMREAD_COLOR); // Read the file

	if (!src.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//Convert the image to greyscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	//Blur the original image to reduce the noise
	GaussianBlur(src_gray, src_blur, Size(5, 5), 0);

	//Calculate gradients
	Sobel(src_blur, grad_x, CV_16S, 1, 0, 3);
	Sobel(src_blur, grad_y, CV_16S, 0, 1, 3);

	//Matrices keeping the values and the degree of each gradient
	Mat grad(grad_x.rows, grad_x.cols, CV_16S);
	Mat theta(grad_x.rows, grad_x.cols, CV_16S);

	calculateGradAndTheta(grad, theta, grad_x, grad_y);

	//Keep only the most visible lines by comparing values to the neighbours
	Mat grad_border(grad.rows + 2, grad.cols + 2, grad.type());
	copyMakeBorder(grad, grad_border, 1, 1, 1, 1, BORDER_REPLICATE);

	calculateGradBorder(grad, theta, grad_border);

	short highT = 250, lowT = 100;
	Mat white_border(grad.rows + 2, grad.cols + 2, grad.type());
	//copyMakeBorder(grad, white_border, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED, 0);
	list<tuple<tuple<int, int>, short>> values = findHighValues(grad, highT);

	calculateWhiteBorder(values, white_border, grad, theta, lowT);

	//Canny algorythm for result check
	Mat result;
	Canny(src_blur, result, lowT, highT, 3 /*CV_SCHARR*/, true);

	namedWindow("Original image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original image", src_blur); // Show our image inside it.

	namedWindow("Grad", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Grad", grad); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}