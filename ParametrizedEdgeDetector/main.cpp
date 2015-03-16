#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const string IMAGE = "building.jpg";
Mat src_blur;
int lowT = 50;
int sobel = 3;
int cornerT = 100;

/// <summary>
/// Calculates the lines.
/// </summary>
/// <param name="result">The result matrix with lines detected with Canny algorythm.</param>
void calculateLines(Mat result)
{
	vector<Vec2f> lines;
	Mat src_lines;

	//result - matrix to search for circles (grayscale)
	//lines - out collection of found lines
	//rho - The resolution of the parameter r in pixels. We use 1 pixel.
	//theta - The resolution of the parameter theta in radians. We use 1 degree
	//treshold - number of minimum intersections determining the line
	HoughLines(result, lines, 1, CV_PI / 180, 200 - lowT);
	cvtColor(result, src_lines, CV_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++)
	{
		//Hough Line transform returns Polar coordinates. 
		//To display the lines on 2D picture, coordinates have to be converted do Cartesian coordinates.

		//rho – Distance resolution of the accumulator in pixels.
		//theta – Angle resolution of the accumulator in radians.
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;

		//family of lines that goes through (x, y) is defined by:
		//r = x * cos(theta) + y * sin(theta)
		//meaning each pair (r, theta) represents each line that goes through (x, y)
		//pairs for each point can be visualised as a sinusoide.
		//intersecting sinusoids for different points determine the same line
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;

		//Calculat start and end points which are set to fixed position -1000 and +1000 pixels from the converted point
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(src_lines, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}
	imshow("detected lines", src_lines);
}
/// <summary>
/// Calculates the circles.
/// </summary>
/// <param name="result">The result matrix with lines detected with Canny algorythm.</param>
void calculateCircles(Mat result)
{
	vector<Vec3f> circles; 
	Mat src_circles;
	//result - matrix to find the circles on (grayscale)
	//circles - out vector of the (x, y, r) of found circles
	//method - method to find the circles - CV_HOUGH_GRADIENT - currently this is the only one available in OpenCV
	//dp - The inverse ratio of resolution
	//minDist - Minimum distance between detected centers
	//param1 - Upper threshold for the internal Canny edge detector
	//param2 - Threshold for center detection.
	//minRadius - Minimum radio to be detected.If unknown, put zero as default.
	//maxRadius - Maximum radius to be detected.If unknown, put zero as default
	HoughCircles(result, circles, CV_HOUGH_GRADIENT, 1, 1, 100, 30, 1, 30);
	cvtColor(result, src_circles, CV_GRAY2BGR);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		// circle outline
		circle(src_circles, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 1, CV_AA);
		// circle center
		circle(src_circles, Point(c[0], c[1]), 2, Scalar(0, 255, 0), 1, CV_AA);
	}

	imshow("detected circles", src_circles);
}
/// <summary>
/// Detects the corners using Harris algorythm.
/// </summary>
/// <param name="source">Source matrix.</param>
void detectCorners(Mat source)
{
	Mat result(source.rows, source.cols, CV_32FC1);
	Mat result_norm;
	//result - float matrix one canal
	//ksize - sobel size
	cornerHarris(source, result, 2, sobel, 0.04);
	normalize(result, result_norm, 0, 255, NORM_MINMAX);
	convertScaleAbs(result_norm, result);

	/*
	for (int i = 0; i < result.cols; i++)
	{
		for (int j = 0; j < result.rows; j++)
		{
			if(result.at<uchar>(j, i) > cornerT)
				circle(result, Point(j, i), 5, Scalar(0, 0, 255), 1, CV_AA);
		}
	}*/

	vector<Point2f> corners;

	goodFeaturesToTrack(source, corners, cornerT, 0.01, 10, Mat(), 3, false, 0.04);
	
	for (size_t i = 0; i < corners.size(); i++)
	{
		Point2f c = corners[i];
		circle(result, Point(c.x, c.y), 5, Scalar(0, 0, 255), 1, CV_AA);
	}

	imshow("detected corners", result);
}
/// <summary>
/// Calculates the result matrix with lines detected with Canny algorythm.
/// </summary>
/// <param name="treshold">The treshold.</param>
/// <param name="sobol">The sobol parameter.</param>
void calculateResultMatrix(int treshold, int sobol)
{
	Mat result;
	cout << "\n" << treshold << "\t" << sobol;
	Canny(src_blur, result, treshold, 3 * treshold, sobol, true);
	imshow("Parametrized edge detector", result);
	calculateLines(result);
	calculateCircles(result);
	detectCorners(result);
}
/// <summary>
/// Changes the treshold and recalculates the result matrix.
/// </summary>
/// <param name="newValue">The new value.</param>
/// <param name="object">The optional parameter.</param>
void changeTreshold(int newValue, void * object)
{
	lowT = newValue + 50;
	calculateResultMatrix(lowT, sobel);
}

void cornerTreshold(int newValue, void * object)
{
	cornerT = newValue;
	calculateResultMatrix(lowT, sobel);
}
/// <summary>
/// Changes the sobel parameter.
/// </summary>
/// <param name="newValue">The new value.</param>
/// <param name="object">The optional parameter.</param>
void changeSobel(int newValue, void * object)
{
	switch (newValue)
	{
	case 0:
		sobel = CV_SCHARR;
		break;
	default:
		sobel = newValue * 2 + 1;
		break;
	}
	calculateResultMatrix(lowT, sobel);
}

int main(int argc, char* argv[])
{
	Mat src = imread(IMAGE, IMREAD_COLOR);
	if (!src.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat src_gray, result;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	GaussianBlur(src, src_blur, Size(5, 5), 0);
	Canny(src_blur, result, 150, 3 * 150, 3, true);

	namedWindow("Parametrized edge detector", WINDOW_AUTOSIZE);
	createTrackbar("sobol", "Parametrized edge detector", &sobel, 3, &changeSobel);
	createTrackbar("low treshold", "Parametrized edge detector", &lowT, 100, &changeTreshold);
	createTrackbar("corners", "Parametrized edge detector", &cornerT, 200, &cornerTreshold);

	imshow("Parametrized edge detector", result);

	namedWindow("detected lines", CV_WINDOW_AUTOSIZE);
	namedWindow("detected circles", CV_WINDOW_AUTOSIZE);
	namedWindow("detected corners", CV_WINDOW_AUTOSIZE);

	calculateLines(result);
	calculateCircles(result);
	detectCorners(result);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}