#include "BookbindingDetector.h"

void default_globals()
{
	_Globals.save_result = true;
	_Globals.show_process = true;
	_Globals.show_result = true;
	_Globals.insert_pause = false;
	_Globals.predict_file = "result.csv";
}

void default_thresholds()
{
	_Thresholds.threshold1 = 5;
	_Thresholds.threshold2 = 2.0;
	_Thresholds.threshold3 = 20;
	_Thresholds.threshold4 = 100;
	_Thresholds.threshold5 = 10;
	_Thresholds.threshold6 = 10;
}

void get_files(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					get_files(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

Vector<IndexRange> cal_abnormal_index_range(Vector<bool> flags)
{
	int count = 0;
	bool last_flag = false;
	int start_idx = 0;
	int end_idx = -1;
	Vector<IndexRange> index_range;
	int size = flags.size();
	bool flag = false;

	IndexRange ir;

	for (int idx = 0; idx < size; idx++)
	{
		flag = flags[idx];

		if (flag)
		{
			if (!last_flag)
			{
				count += 1;
				start_idx = idx;
				end_idx = None;
			}
		}
		else
		{
			if (last_flag)
			{
				end_idx = idx;
			}
		}
		if ((end_idx != None) && (end_idx != -1))
		{
			ir.start = start_idx;
			ir.end = end_idx;
			ir.length = end_idx - start_idx;
			index_range.push_back(ir);
		}
		last_flag = flag;
	}
	if ((start_idx != None) && (end_idx == None))
	{
		ir.start = start_idx;
		ir.end = size;
		ir.length = end_idx - start_idx;
		index_range.push_back(ir);
	}

	return index_range;
}

int find_max_contour(vector<vector<Point> > contours)
{
	int index = 0;
	int count = contours.size();
	double area = 0.0;
	double max_area = 0.0;
	double max_rate = 1.3;
	for (int i = 0; i < count; i++)
	{
		Rect rect = boundingRect(contours.at(i));
		double con_area = contourArea(contours.at(i));
		area = rect.width * rect.height;
		if ((area > max_area) && (con_area * max_rate > area))
		{
			max_area = area;
			index = i;
		}
	}

	return index;
}

Mat grabcut(String filename, Rect rect)
{
	Mat img = imread(filename, 1);
	Mat bgModel, fgModel;
	Mat result;
	grabCut(img, result, rect, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
	result = result & 1;
	Mat foreGround(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	img.copyTo(foreGround, result);
	return foreGround;
}

void projection_process_horizontal(Mat shared_img,
	Mat extract_local, int miny, int maxy,
	Globals globals, Thresholds thresholds)
{

}

void projection_process_vertical(Mat shared_img,
	Mat extract_local, int minx, int maxx,
	Globals globals, Thresholds thresholds, bool *warning, bool *abnormal)
{
	if (globals.show_process)
	{
		imshow("extracted", extract_local);
	}

	const int height = extract_local.rows;
	const int width = extract_local.cols;

	Mat projected_v = Mat::zeros(height, width, CV_8U);
	Mat projected = projected_v.clone();

	Mat count_arr = Mat::zeros(1, width, CV_32S);
	Mat gray;
	cvtColor(extract_local, gray, COLOR_BGR2GRAY);

	int count = 0;

	for (int i = 0; i < width; i++)
	{
		count = 0;
		for (int j = 0; j < height; j++)
		{
			if (gray.ptr(j)[i] > 0)
			{
				count += 1;
			}
		}

		count_arr.ptr(0)[i] = count;

		for (int j = height - count; j < height; j++)
		{
			projected_v.ptr(j)[i] = 255;
			projected.ptr(j)[i] = 255;

			extract_local.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
			//extract_local.ptr(j)[i * 3 + 0] = 255;
			//extract_local.ptr(j)[i * 3 + 1] = 255;
			//extract_local.ptr(j)[i * 3 + 2] = 255;
			//shared_img.ptr(j)[i * 3 + 0] = 255;
			//shared_img.ptr(j)[i * 3 + 1] = 255;
			//shared_img.ptr(j)[i * 3 + 2] = 255;

			shared_img.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
		}
	}

	if (globals.show_process)
	{
		imshow("projected vertical", projected_v);
		imshow("projected vertical color", extract_local);
	}

	int sum_count = 0, avg_count = 0;

	for (int i = minx; i < maxx; i++)
	{
		sum_count += count_arr.ptr(0)[i];
	}

	avg_count = sum_count / (maxx - minx);

	for (int i = 0; i < width; i++)
	{
		//extract_local.ptr(height - avg_count)[i * 3 + 0] = 0;
		//extract_local.ptr(height - avg_count)[i * 3 + 1] = 0;
		//extract_local.ptr(height - avg_count)[i * 3 + 2] = 255;

		extract_local.at<Vec3b>(height - avg_count, i) = Vec3b(0, 0, 255);
		//shared_img.ptr(height - avg_count)[i * 3 + 0] = 0;
		//shared_img.ptr(height - avg_count)[i * 3 + 1] = 0;
		//shared_img.ptr(height - avg_count)[i * 3 + 2] = 255;
		shared_img.at<Vec3b>(height - avg_count, i) = Vec3b(0, 0, 255);
	}

	if (globals.show_process)
	{
		imshow("vertical avg line", extract_local);
	}

	int start = minx;
	int end = maxx;
	Mat count_arr_copy = count_arr.clone();
	int ignore_length = int(thresholds.threshold4);

	for (int i = 0; i < height; i++)
	{
		//extract_local.ptr(i)[(start + ignore_length) * 3 + 0] = 0;
		//extract_local.ptr(i)[(start + ignore_length) * 3 + 1] = 255;
		//extract_local.ptr(i)[(start + ignore_length) * 3 + 2] = 255;

		//shared_img.ptr(i)[(start + ignore_length) * 3 + 0] = 0;
		//shared_img.ptr(i)[(start + ignore_length) * 3 + 1] = 255;
		//shared_img.ptr(i)[(start + ignore_length) * 3 + 2] = 255;

		extract_local.at<Vec3b>(i, start + ignore_length) = Vec3b(0, 255, 255);
		shared_img.at<Vec3b>(i, start + ignore_length) = Vec3b(0, 255, 255);

		//extract_local.ptr(i)[(end - ignore_length) * 3 + 0] = 0;
		//extract_local.ptr(i)[(end - ignore_length) * 3 + 1] = 255;
		//extract_local.ptr(i)[(end - ignore_length) * 3 + 2] = 255;

		//shared_img.ptr(i)[(end - ignore_length) * 3 + 0] = 0;
		//shared_img.ptr(i)[(end - ignore_length) * 3 + 1] = 255;
		//shared_img.ptr(i)[(end - ignore_length) * 3 + 2] = 255;

		extract_local.at<Vec3b>(i, end - ignore_length) = Vec3b(0, 255, 255);
		shared_img.at<Vec3b>(i, end - ignore_length) = Vec3b(0, 255, 255);
	}

	int max_idx = 0, count_max = 0;
	int min_idx = 0, count_min = height;

	for (int i = 0; i < width; i++)
	{
		if (count_arr.ptr(0)[i] > count_max)
		{
			max_idx = i;
			count_max = count_arr.ptr(0)[i];
		}
	}

	for (int i = start; i < end; i++)
	{
		if (count_arr.ptr(0)[i] < count_min)
		{
			min_idx = i;
			count_min = count_arr.ptr(0)[i];
		}
	}

	int left_range_span = int(thresholds.threshold1);
	for (int i = 0; i < 100; i++)
	{
		for (int j = max_idx - left_range_span; j < max_idx + left_range_span; j++)
		{
			//extract_local.ptr(i)[j * 3 + 0] = 0;
			//extract_local.ptr(i)[j * 3 + 1] = 255;
			//extract_local.ptr(i)[j * 3 + 2] = 0;

			//shared_img.ptr(i)[j * 3 + 0] = 0;
			//shared_img.ptr(i)[j * 3 + 1] = 255;
			//shared_img.ptr(i)[j * 3 + 2] = 0;

			extract_local.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			shared_img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
		}

		for (int j = min_idx - left_range_span; j < min_idx + left_range_span; j++)
		{
			//extract_local.ptr(i)[j * 3 + 0] = 0;
			//extract_local.ptr(i)[j * 3 + 1] = 255;
			//extract_local.ptr(i)[j * 3 + 2] = 0;

			//shared_img.ptr(i)[j * 3 + 0] = 0;
			//shared_img.ptr(i)[j * 3 + 1] = 255;
			//shared_img.ptr(i)[j * 3 + 2] = 0;
			extract_local.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			shared_img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
		}
	}
	if (globals.show_process)
	{
		imshow("vertical avg line range", extract_local);
	}

	int max_range_sum = 0, max_avg = 0;
	int min_range_sum = 0, min_avg = 0;

	for (int i = max_idx - left_range_span; i < max_idx + left_range_span; i++)
	{
		max_range_sum += count_arr.ptr(0)[i];
	}
	for (int i = min_idx - left_range_span; i < min_idx + left_range_span; i++)
	{
		min_range_sum += count_arr.ptr(0)[i];
	}

	max_avg = max_range_sum / (left_range_span * 2);
	min_avg = min_range_sum / (left_range_span * 2);

	for (int i = 0; i < width; i++)
	{
		//extract_local.ptr(height - max_avg)[i * 3 + 0] = 0;
		//extract_local.ptr(height - max_avg)[i * 3 + 1] = 255;
		//extract_local.ptr(height - max_avg)[i * 3 + 2] = 0;

		//shared_img.ptr(height - max_avg)[i * 3 + 0] = 0;
		//shared_img.ptr(height - max_avg)[i * 3 + 1] = 255;
		//shared_img.ptr(height - max_avg)[i * 3 + 2] = 0;

		extract_local.at<Vec3b>(height - max_avg, i) = Vec3b(0, 255, 0);
		shared_img.at<Vec3b>(height - max_avg, i) = Vec3b(0, 255, 0);
	}

	if (globals.show_process)
	{
		imshow("vertical avg line range max and min", extract_local);
	}

	float max_diff_rate = float(thresholds.threshold2);

	bool abnormal_flag = false;
	bool warning_flag = false;

	if (max_avg / (min_avg + 0.1) > max_diff_rate)
	{
		if ((start + ignore_length <= max_idx) && (max_idx <= end - ignore_length))
		{
			abnormal_flag = true;
		}
		else
		{
			warning_flag = true;
		}
	}
	for (int i = 0; i < width; i++)
	{
		if (count_arr.ptr(0)[i] > 0)
		{
			start = i;
			break;
		}
	}

	for (int i = width - 1; i > 0; i--)
	{
		if (count_arr.ptr(0)[i] > 0)
		{
			end = i;
			break;
		}
	}

	Vector<bool> flags;
	for (int i = start; i < end + 1; i++)
	{
		if (count_arr.ptr(0)[i] > 0)
		{
			flags.push_back(false);
		}
		else
		{
			flags.push_back(true);
		}
	}

	Vector<IndexRange> range_ = cal_abnormal_index_range(flags);
	count = range_.size();

	if (count)
	{
		int max_range_length = 0;
		int max_range_idx = 0;
		for (int i = 0; i < count; i++)
		{
			if (max_range_length < range_[i].length)
			{
				max_range_length = range_[i].length;
				max_range_idx = i;
			}
		}

		float max_gap_threshold = float(thresholds.threshold3);
		if (max_range_length > max_gap_threshold)
		{
			if ((range_[max_range_idx].end < minx + ignore_length) ||
				(range_[max_range_idx].start > maxx - ignore_length))
			{
				warning_flag = true;
			}
			else
			{
				abnormal_flag = true;
			}
		}
	}

	*warning = warning_flag;
	*abnormal = abnormal_flag;
}

void projection_process_horizontal(Mat shared_img,
	Mat extract_local, int miny, int maxy,
	Globals globals, Thresholds thresholds, bool *warning, bool *abnormal)
{
	if (globals.show_process)
	{
		imshow("extracted", extract_local);
	}
	const int height = extract_local.rows;
	const int width = extract_local.cols;

	Mat projected_h = Mat::zeros(height, width, CV_8U);
	Mat projected = projected_h.clone();
	Mat count_arr = Mat::zeros(height, 1, CV_32S);
	Mat gray;
	cvtColor(extract_local, gray, COLOR_BGR2GRAY);

	int count = 0;
	for (int j = 0; j < height; j++)
	{
		count = 0;
		for (int i = 0; i < width; i++)
		{
			if (gray.ptr(j)[i] > 0)
			{
				count += 1;
			}
		}
		count_arr.ptr(j)[0] = count;

		for (int i = width - count; i < width; i++)
		{
			projected_h.ptr(j)[i] = 255;
			projected.ptr(j)[i] = 255;
			extract_local.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
			shared_img.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
		}
	}
	if (globals.show_process)
	{
		imshow("projected horizontal", projected_h);
	}

	int avg_count_sum = 0, avg_count_h = 0;

	for (int j = miny; j < maxy; j++)
	{
		avg_count_sum += count_arr.ptr(j)[0];
	}
	avg_count_h = avg_count_sum / (maxy - miny);

	for (int j = 0; j < height; j++)
	{
		extract_local.at<Vec3b>(j, width - avg_count_h) = Vec3b(0, 0, 255);
		shared_img.at<Vec3b>(j, width - avg_count_h) = Vec3b(0, 0, 255);
	}

	if (globals.show_process)
	{
		imshow("horizontal avg line", extract_local);
	}

	int start_h = miny;
	int end_h = maxy;

	int ignore_length_h = int(thresholds.threshold5);

	for (int j = start_h; j < start_h + ignore_length_h; j++)
	{
		projected_h.ptr(j)[0] = 0;
		count_arr.ptr(j)[0] = 0;
	}
	for (int j = end_h - ignore_length_h; j < end_h; j++)
	{
		projected_h.ptr(j)[0] = 0;
		count_arr.ptr(j)[0] = 0;
	}

	if (globals.show_process)
	{
		imshow("project h", projected_h);
	}

	for (int i = 0; i < width; i++)
	{
		extract_local.at<Vec3b>(start_h + ignore_length_h, i) = Vec3b(0, 255, 255);
		shared_img.at<Vec3b>(start_h + ignore_length_h, i) = Vec3b(0, 255, 255);
		extract_local.at<Vec3b>(end_h - ignore_length_h, i) = Vec3b(0, 255, 255);
		shared_img.at<Vec3b>(end_h - ignore_length_h, i) = Vec3b(0, 255, 255);
	}

	if (globals.show_process)
	{
		imshow("extract_local_copy", extract_local);
	}

	int max_idx_h = 0;
	int max_count = 0;
	for (int j = 0; j < height; j++)
	{
		if (max_count < count_arr.ptr(j)[0])
		{
			max_idx_h = j;
			max_count = count_arr.ptr(j)[0];
		}
	}

	int left_range_span = int(thresholds.threshold5);
	for (int j = max_idx_h - left_range_span; j < max_idx_h + left_range_span; j++)
	{
		for (int i = 0; i < 100; i++)
		{
			extract_local.at<Vec3b>(j, i) = Vec3b(0, 255, 0);
			shared_img.at<Vec3b>(j, i) = Vec3b(0, 255, 0);
		}
	}

	if (globals.show_process)
	{
		imshow("horizontal avg line range ", extract_local);
	}

	int middle_count_sum = 0, middle_avg = 0;
	int max_range_count_sum = 0, max_range_avg = 0;

	for (int j = start_h + left_range_span; j < end_h - left_range_span; j++)
	{
		middle_count_sum += count_arr.ptr(j)[0];
	}

	middle_avg = middle_count_sum / (end_h - start_h - 2 * left_range_span);

	for (int j = max_idx_h - left_range_span; j < max_idx_h + left_range_span; j++)
	{
		max_range_count_sum += count_arr.ptr(j)[0];
	}

	max_range_avg = max_range_count_sum / (2 * left_range_span);

	for (int j = 0; j < height; j++)
	{
		extract_local.at<Vec3b>(j, width - max_range_avg) = Vec3b(0, 255, 0);
		shared_img.at<Vec3b>(j, width - max_range_avg) = Vec3b(0, 255, 0);
	}

	if (globals.show_process)
	{
		imshow("extract_local_horizontal", extract_local);
	}

	int max_diff = thresholds.threshold6;
	bool abnormal_flag = false;

	if (middle_avg + max_diff < max_range_avg)
	{
		abnormal_flag = true;
	}

	*warning = false;
	*abnormal = abnormal_flag;
}

int _bookbingDetector(String filename, Globals global, Thresholds thresholds)
{
	int result = -1;
	try
	{

		//初始化计时
		LARGE_INTEGER  freq_num;
		long long start_time, end_time, freq;
		QueryPerformanceFrequency(&freq_num);
		freq = freq_num.QuadPart;
		QueryPerformanceCounter(&freq_num);
		start_time = freq_num.QuadPart;

		//String filename = String(argv[1]);
		//global.save_result = strcmp(argv[2], "true") == 0 ? true : false;
		//global.show_process = strcmp(argv[3], "true") == 0 ? true : false;
		//global.show_result = strcmp(argv[4], "true") == 0 ? true : false;
		//global.insert_pause = strcmp(argv[5], "true") == 0 ? true : false;
		//global.predict_file = String(argv[6]);

		//thresholds.threshold1 = strcmp(argv[7], "default") == 0 ? 5 : atof(argv[7]);
		//thresholds.threshold2 = strcmp(argv[8], "default") == 0 ? 2.0 : atof(argv[8]);
		//thresholds.threshold3 = strcmp(argv[9], "default") == 0 ? 20 : atof(argv[9]);
		//thresholds.threshold4 = strcmp(argv[10], "default") == 0 ? 100 : atof(argv[10]);
		//thresholds.threshold5 = strcmp(argv[11], "default") == 0 ? 10 : atof(argv[11]);
		//thresholds.threshold6 = strcmp(argv[12], "default") == 0 ? 10 : atof(argv[12]);



		//std::cout << "Hello world\n";
		//String filename = "demo.jpg";
		Mat img = imread(filename, 0);
		if (!img.data)
		{
			cout << " failed to load image";
			exit(1);
		}
		if (global.show_process) imshow("origin", img);
		Mat thresh3;
		threshold(img, thresh3, 127, 255, THRESH_TRUNC);
		//cvResizeWindow("demo", 600, 800);
		if (global.show_process) imshow("thresh3", thresh3);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat dilated;
		dilate(thresh3, dilated, kernel, cv::Point(-1, -1), 3);

		if (global.show_process)imshow("dilated", dilated);

		Mat binary;

		threshold(dilated, binary, 100, 255, THRESH_BINARY);

		if (global.show_process)imshow("binary", binary);

		// 噪声处理
		int bin_w = binary.cols;
		int bin_h = binary.rows;
		for (int i = 0; i < bin_h; i++)
		{
			int sum = 0;
			uchar *ptr = binary.ptr(i);
			for (int j = 0; j < bin_w; j++)
			{
				if (ptr[j] > 0)
				{
					sum += 1;
				}
			}
			if (sum * 10 < bin_w)
			{
				for (int j = 0; j < bin_w; j++)
				{
					ptr[j] = 0;
				}
			}
		}
		if (global.show_process)imshow("binary2", binary);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		int idx = 0;
		Mat dst = Mat::zeros(binary.rows, binary.cols, CV_8UC3);
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color(rand() & 255, rand() & 255, rand() & 255);
			//drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
		if (global.show_process)imshow("Contours", dst);

		int max_contour_idx = find_max_contour(contours);
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, max_contour_idx, color, CV_FILLED, 8, hierarchy);
		if (global.show_process)imshow("Contours2", dst);

		Rect clip_rect2 = boundingRect(contours[max_contour_idx]);

		Mat color_img2 = grabcut(filename, clip_rect2);

		if (global.show_process)imshow("color_img2", color_img2);

		Mat gray_img2;
		cvtColor(color_img2, gray_img2, COLOR_RGB2GRAY);

		if (global.show_process)imshow("binary", gray_img2);

		threshold(gray_img2, binary, 100, 255, THRESH_BINARY);

		//Mat smoothed;
		//medianBlur(gray_img2, smoothed, 3);

		//imshow("smoothed", smoothed);

		Mat binary3;

		dilate(binary, binary3, kernel);

		if (global.show_process)imshow("binary3", binary3);

		medianBlur(binary3, binary3, 3);

		if (global.show_process)imshow("binary3_2", binary3);

		Mat binary4;

		erode(binary3, binary4, kernel, Point(-1, -1), 4);

		if (global.show_process)imshow("binary4", binary4);

		Mat binary5;

		medianBlur(binary4, binary5, 5);

		if (global.show_process)imshow("binary5", binary5);

		// 中间的查找轮廓可以不要
		Mat binary5_color;
		cvtColor(binary5, binary5_color, COLOR_GRAY2RGB);
		if (global.show_process)imshow("binary5 color", binary5_color);

		//
		Mat binary6 = binary5.clone();
		int minx = clip_rect2.x;
		int maxx = clip_rect2.x + clip_rect2.width;
		int miny = clip_rect2.y;
		int maxy = clip_rect2.y + clip_rect2.height;

		int start_pos = 0;
		int end_pos = 0;
		for (int i = miny; i < maxy; i++)
		{
			start_pos = minx;
			end_pos = start_pos + 1;
			uchar *ptr = binary6.ptr(i);
			while (ptr[end_pos] < 255 && end_pos < maxx)
			{
				ptr[end_pos] = 255;
				end_pos++;
			}
			// 合并为一次循环
			start_pos = maxx;
			end_pos = start_pos - 1;
			while (ptr[end_pos] < 255 && end_pos > minx)
			{
				ptr[end_pos] = 255;
				end_pos--;
			}
		}
		if (global.show_process)imshow("after fill left and right", binary6);

		Mat binary7;
		Mat mask = Mat::zeros(binary6.rows, binary6.cols, CV_8UC1);
		for (int i = miny; i < maxy; i++)
		{
			uchar *ptr = mask.ptr(i);
			// 没明白为什么要+1
			// 不+1，会有一条竖线
			for (int j = minx + 1; j < maxx; j++)
			{
				ptr[j] = 1;
			}
		}
		bitwise_not(binary6, binary7, mask);
		if (global.show_process)imshow("binary 7", binary7);

		Mat extracted = binary7.clone();
		// 在findContours执行结束之后，binary7内容会发生变化。
		findContours(binary7, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
		//binary7.copyTo(extracted);
		//imshow("extracted 0", extracted);
		idx = 0;
		dst = Mat::zeros(binary7.rows, binary7.cols, CV_8UC3);
		Rect rect;
		minx = 0, maxx = 0, miny = 0, maxy = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			rect = boundingRect(contours[idx]);
			if (rect.height < 10)
			{
				minx = rect.x;
				maxx = rect.x + rect.width;
				miny = rect.y;
				maxy = rect.y + rect.height;
				//zero
				for (int i = miny; i <= maxy; i++)
				{
					uchar *ptr = extracted.ptr(i);
					for (int j = minx; j <= maxx; j++)
					{
						ptr[j] = 0;
					}
				}
				continue;
			}
			Scalar color(rand() & 255, rand() & 255, rand() & 255);

			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
		if (global.show_process)imshow("Contours 2", dst);
		if (global.show_process)imshow("extracted", extracted);

		Mat origin_color_img = imread(filename, 1);
		Mat bg = Mat::zeros(extracted.rows, extracted.cols, CV_8UC3);
		Mat extracted_local;
		add(bg, origin_color_img, extracted_local, extracted);
		if (global.show_process)imshow("extracted local", extracted_local);
		//TODO 水平和竖直投影
		minx = clip_rect2.x;
		maxx = clip_rect2.x + clip_rect2.width;
		miny = clip_rect2.y;
		maxy = clip_rect2.y + clip_rect2.height;

		Mat shared_img = extracted_local.clone();
		Mat extract_local_v = extracted_local.clone();
		bool warning_v = false, abnormal_v = false;
		projection_process_vertical(shared_img, extract_local_v,
			minx, maxx, global, thresholds, &warning_v, &abnormal_v);
		Mat extract_local_h = extracted_local.clone();
		bool warning_h = false, abnormal_h = false;
		projection_process_horizontal(shared_img, extract_local_h,
			miny, maxy, global, thresholds, &warning_h, &abnormal_h);

		// draw text
		char v_str[20];
		if (warning_v)
		{
			if (abnormal_v)
			{
				sprintf(v_str, "V: %s %s", "Warning", "Abnormal");
			}
			else
			{
				sprintf(v_str, "V: %s %s", "Warning", "Normal");
			}
		}
		else
		{
			if (abnormal_v)
			{
				sprintf(v_str, "V: %s %s", "", "Abnormal");
			}
			else
			{
				sprintf(v_str, "V: %s %s", "", "Normal");
			}
		}

		char h_str[20];
		if (abnormal_h)
		{
			sprintf(h_str, "H: %s", "Abnormal");
		}
		else
		{
			sprintf(h_str, "H: %s", "Normal");
		}

		char final_str[20];

		if (abnormal_v || abnormal_h)
		{
			result = 1;
			sprintf(final_str, "Result: %s", "Abnormal");
			std::cout << filename << " : Abnormal\n";
		}
		else
		{
			result = 0;
			sprintf(final_str, "Result: %s", "Normal");
			std::cout << filename << " : Normal\n";
		}

		putText(shared_img, String(v_str), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2);
		putText(shared_img, String(h_str), Point(10, 60), FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2);
		putText(shared_img, String(final_str), Point(10, 90), FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2);

		if (global.show_result)
		{
			imshow("Final", shared_img);
		}
		if (global.save_result)
		{
			String out_filename = filename + "_ret.jpg";
			//out_filename = os.path.join(outfolder, os.path.splitext(os.path.basename(filename))[0]) + '_ret.jpg';
			//std::vector<uchar> data_encode;
			//imencode(".jpg", shared_img, data_encode);
			//std::string str_encode(data_encode.begin(), data_encode.end());
			//ofstream ofs(out_filename);
			//ofs << str_encode;
			//ofs.flush();
			//ofs.close();

			imwrite(out_filename, shared_img);
		}

		if (global.insert_pause || global.show_result)
		{
			waitKey(0);
		}

		if (global.show_process || global.show_result)
		{
			destroyAllWindows();
		}
		QueryPerformanceCounter(&freq_num);
		end_time = freq_num.QuadPart;
		cout << "Used Time: " << (end_time - start_time) * 1000 / (freq *  1.0) << " ms" << std::endl;
	}
	catch (std::exception& e)
	{
		std::cout << e.what();
		result = -1;
	}
	return result;
}

vector<int> bookbingDetector()
{
	vector<string> files;
	get_files(INPUT_FOLDER, files);

	int count = files.size();
	vector<int> result;
	for (int i = 0; i < count; i++)
	{
		result.push_back(bookbingDetector_file(files.at(i)));
	}
	return result;
}

int bookbingDetector_file(String filename)
{
	return _bookbingDetector(filename, _Globals, _Thresholds);
}