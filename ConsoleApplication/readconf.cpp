#include "readconf.h"

float* read_config(string filename)
{
	ifstream ss(filename);
	//float binary_threshold = 80.0;//二值化阈值，默认为80
	//float max_difference_ratio = 0.5;//最大差异阈值，默认为0.5
	//float ignore_left_right_ratio = 0.15;//左右噪声比例，默认为0.15
	//float ignore_top_bottom_ratio = 0.2;//上下噪声比例，默认为0.2
	//float min_nonzero_pixel_ratio = 0.5; //最小非零像素比例，默认为0.5

	float arr[5] = { 80.0, 0.5, 0.15, 0.2, 0.5};
	int i = 0;
	while (i < 5 && !ss.eof())
	{
		string line;
		ss >> line;
		arr[i] = float(atof(line.c_str()));
		i++;
	}
	ss.close();

	return arr;
}

string read_mask(string filename)
{
	ifstream ss(filename);
	string line;
	ss >> line;
	return line;
}