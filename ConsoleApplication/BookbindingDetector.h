#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <windows.h>
#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"

#include <io.h>

using namespace std;
using namespace cv;

#ifndef None
#define None -100
#endif

#ifndef False
#define False 0
#endif

#ifndef True
#define True 1
#endif

#define MISS_COLOR Vec3b(255, 0, 255)
#define ABNORMAL_COLOR Vec3b(0, 0, 255)
#define NORMAL_COLOR Vec3b(0, 255, 0)
#define WHITE_COLOR Vec3b(255, 255, 255)
#define BLACK_COLOR Vec3b(0, 0, 0)


typedef struct Globals
{
	bool wait_key;
	bool show_process;
	String in_folder;
	String out_folder;
	String mask_file;
}Globals;

typedef struct Thresholds
{
	float binary_threshold;
	float max_difference_ratio;
	float ignore_left_right_ratio;
	float ignore_top_bottom_ratio;
	float min_nonzero_pixel_ratio;
}Thresholds;

typedef struct IndexRange
{
	int start;
	int end;
	int length;
}IndexRange;

typedef struct Shape
{
	int width;
	int height;
}Shape;

static Globals _Globals;
static Thresholds _Thresholds;
#define INPUT_FOLDER "image"


void default_globals();
void default_thresholds();

int _bookbingDetector(String filename);
//vector<int> bookbingDetector();
//int bookbingDetector_file(String filename);

Mat to_gray(Mat img);
vector<vector<Point> > extract_contours(Mat binary_img);