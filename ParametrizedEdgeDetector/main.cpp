#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const string IMAGE = "building.jpg";
Mat src_blur;
int lowT = 50;
int sobol = 3;

void calculateLines(Mat result)
{
	vector<Vec2f> lines;
	Mat src_lines;
	HoughLines(result, lines, 1, CV_PI / 180, 200);
	cvtColor(result, src_lines, CV_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(src_lines, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}
	imshow("detected lines", src_lines);
}

void calculateCircles(Mat result)
{
	vector<Vec3f> circles; 
	Mat src_circles;
	cout << "ROWS:" << result.rows << "/ROWS";
	HoughCircles(result, circles, CV_HOUGH_GRADIENT, 1, result.rows / 8);
	cvtColor(result, src_circles, CV_GRAY2BGR);

	for (size_t i = 0; i < circles.size(); i++) 
	{ 
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1])); 
		int radius = cvRound(circles[i][2]); 
		circle(src_circles, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(src_circles, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	} 
	imshow("detected circles", src_circles);
}

void calculateResultMatrix(int treshold, int sobol)
{
	Mat result;
	cout << "\n" << treshold << "\t" << sobol;
	Canny(src_blur, result, treshold, 3 * treshold, sobol, true);
	imshow("Parametrized edge detector", result);
	calculateLines(result);
	calculateCircles(result);
}

void changeTreshold(int newValue, void * object)
{
	lowT = newValue + 50;
	calculateResultMatrix(lowT, sobol);
}

void changeSobol(int newValue, void * object)
{
	switch (newValue)
	{
	case 0:
		sobol = CV_SCHARR;
		break;
	default:
		sobol = newValue * 2 + 1;
		break;
	}
	calculateResultMatrix(lowT, sobol);
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
	createTrackbar("sobol", "Parametrized edge detector", &sobol, 3, &changeSobol);
	createTrackbar("low treshold", "Parametrized edge detector", &lowT, 100, &changeTreshold);

	imshow("Parametrized edge detector", result);
	namedWindow("detected lines", CV_WINDOW_AUTOSIZE);
	namedWindow("detected circles", CV_WINDOW_AUTOSIZE);
	calculateLines(result);
	calculateCircles(result);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}