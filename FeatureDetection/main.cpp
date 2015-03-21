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
	try
	{
		CannyDemo demo("okladka.jpg", "Original", "Result");
		demo.Run();
	}
	catch(Exception& e)
	{
		cout << e.err;
	}
	return 0;
}