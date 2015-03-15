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

void calculateResultMatrix(int treshold, int sobol)
{
	Mat result;
	Canny(src_blur, result, treshold, 3 * treshold, sobol, true);
	imshow("Parametrized edge detector", result);
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
		sobol = -1;
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
	Canny(src_blur, result, lowT, 3 * lowT, 3, true);

	namedWindow("Parametrized edge detector", WINDOW_AUTOSIZE);
	createTrackbar("sobol", "Parametrized edge detector", &sobol, 4, &changeSobol);
	createTrackbar("low treshold", "Parametrized edge detector", &lowT, 100, &changeTreshold);

	imshow("Parametrized edge detector", result);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}