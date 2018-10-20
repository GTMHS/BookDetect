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


typedef struct Globals
{
	bool save_result;
	bool show_process;
	bool show_result;
	bool insert_pause;
	String predict_file;
}Globals;

typedef struct Thresholds
{
	float threshold1;
	float threshold2;
	float threshold3;
	float threshold4;
	float threshold5;
	float threshold6;
}Thresholds;

typedef struct IndexRange
{
	int start;
	int end;
	int length;
}IndexRange;

static Globals _Globals;
static Thresholds _Thresholds;
#define INPUT_FOLDER "image"


void default_globals();
void default_thresholds();

int _bookbingDetector(String filename, Globals global, Thresholds thresholds);
vector<int> bookbingDetector();
int bookbingDetector_file(String filename);