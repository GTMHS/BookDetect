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
	bool wait_key;//�Ƿ������ͣ����show_processΪTrueʱ���Ƿ�ȴ���ʾ����
	bool show_process;//�Ƿ���ʾ����
	String in_folder;//�����ļ��У���δʹ�ã�ֻΪ����Python����һ��
	String out_folder;//����ļ��У���δʹ�ã�ֻΪ����Python����һ��
	String mask_file;//mask�ļ�·������δʹ�ã�ֻΪ����Python����һ��
}Globals;

typedef struct Thresholds
{
	float binary_threshold;//��ֵ����ֵ��Ĭ��Ϊ80
	float max_difference_ratio;//��������ֵ��Ĭ��Ϊ0.5
	float ignore_left_right_ratio;//��������������Ĭ��Ϊ0.15
	float ignore_top_bottom_ratio;//��������������Ĭ��Ϊ0.2
	float min_nonzero_pixel_ratio;//��С�������ر�����Ĭ��Ϊ0.5
}Thresholds;

// ȫ������
static Globals _Globals;
// ȫ����ֵ
static Thresholds _Thresholds;

// Ĭ��ȫ������
void default_globals();

// Ĭ����ֵ
void default_thresholds();

//�ض���opencv�Ĵ������
int cvErrorRedirector(int status, const char* func_name, const char* err_msg,
	const char* file_name, int line, void* userdata);

/*
 * @brief ͼ�����㷨
 * @param[in] filename�������������ļ�·�������鲻Ҫ��������
 * @param[in] maskfile��ģ���ļ�·�������鲻Ҫ��������
 * @param[in] outfile������ļ������鲻Ҫ�������ģ�Ĭ��Ϊ""����ʾ������ļ�
 * @param[in] verbose, �Ƿ�����㷨�����쳣��Ϣ��True��ʾ����㷨�����쳣��
 * ������쳣��Ϣ��Ĭ��ΪFalse
 * @param[in] timing, �Ƿ��ʱ��Ĭ��ΪFalse
 * @param[out] time_ms����ʱ���ָ�룬��λΪms�����timingΪTrue���ò�������ָ��
 * @return 0��ʾ������1��ʾ�쳣��-1��ʾ�㷨�����쳣
 * @note �����и��㷨֮ǰ������������_Globals��_Thresholds��
 * ����֪��������ã���ʹ��default_globals()��default_thresholds()����
 */
int bookDetecting(string filename, string maskfile, string outfile = "", bool verbose = False, bool timing = False, double *time_ms = 0);