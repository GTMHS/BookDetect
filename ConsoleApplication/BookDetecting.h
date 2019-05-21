#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <windows.h>
#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"

#include <io.h>

#include "myexception.h"

using namespace std;
using namespace cv;

#ifndef False
#define False 0
#endif

#ifndef True
#define True 1
#endif

typedef struct Globals
{
	bool wait_key;//是否插入暂停，即show_process为True时，是否等待显示窗口
	bool show_process;//是否显示过程
	String in_folder;//输入文件夹，暂未使用，只为了与Python保持一致
	String out_folder;//输出文件夹，暂未使用，只为了与Python保持一致
	String mask_file;//mask文件路径，暂未使用，只为了与Python保持一致
}Globals;

typedef struct Thresholds
{
	float binary_threshold;//二值化阈值，默认为80
	float max_difference_ratio;//最大差异阈值，默认为0.5
	float ignore_left_right_ratio;//左右噪声比例，默认为0.15
	float ignore_top_bottom_ratio;//上下噪声比例，默认为0.2
	float min_nonzero_pixel_ratio;//最小非零像素比例，默认为0.5
}Thresholds;

// 全局配置
static Globals _Globals;
// 全局阈值
static Thresholds _Thresholds;

// 默认全局配置
void default_globals();

// 默认阈值
void default_thresholds();

//重定向opencv的错误输出
int cvErrorRedirector(int status, const char* func_name, const char* err_msg,
	const char* file_name, int line, void* userdata);

/*
 * @brief 图书检测算法
 * @param[in] filename，待检测的输入文件路径，建议不要包含中文
 * @param[in] maskfile，模板文件路径，建议不要包含中文
 * @param[in] outfile，输出文件，建议不要包含中文，默认为""，表示不输出文件
 * @param[in] verbose, 是否输出算法运行异常信息，True表示如果算法运行异常，
 * 则输出异常信息，默认为False
 * @param[in] timing, 是否计时，默认为False
 * @param[out] time_ms，计时结果指针，单位为ms，如果timing为True，该参数必须指定
 * @return 0表示正常，1表示异常，-1表示算法运行异常
 * @note 在运行该算法之前，必须先设置_Globals和_Thresholds，
 * 若不知道如何设置，可使用default_globals()和default_thresholds()设置
 */
int bookDetecting(string filename, string maskfile, string outfile = "", bool verbose = False, bool timing = False, double *time_ms = 0);