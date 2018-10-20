#include "BookbindingDetector.h"

void default_globals()
{
	_Globals.wait_key = true;
	_Globals.show_process = true;
	_Globals.in_folder = "";
	_Globals.out_folder = "";
	_Globals.mask_file = "";
}

void default_thresholds()
{
	_Thresholds.binary_threshold = 80;
	_Thresholds.max_difference_ratio = 0.5;
	_Thresholds.ignore_left_right_ratio = 0.15;
	_Thresholds.ignore_top_bottom_ratio = 0.2;
	_Thresholds.min_nonzero_pixel_ratio = 0.5;
}

Mat read_mask(String maskfile, int channels = 3)
{
	Mat mask = imread(maskfile);

	Mat mask2;

	const int height = mask.rows;
	const int width = mask.cols;

	if (channels == 3)
	{
		mask2 = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				if (mask.ptr(j)[i * 3 + 3] > 0){
					mask2.at<Vec3b>(j, i) = WHITE_COLOR;
				}
			}
		}
	}
	else if (channels == 1)
	{
		mask2 = Mat::zeros(mask.rows, mask.cols, CV_8U);

		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				if (mask.ptr(j)[i * 3 + 3] > 0){
					mask2.ptr(j)[i] = 255;
				}
			}
		}
	}
	return mask2;
}

Mat process_mask(Mat mask)
{
	Rect rect = boundingRect(mask);
	Mat ret = Mat::zeros(rect.height, rect.width, mask.type());

	if (mask.channels == 1){
		for (int i = 0; i < rect.height; i++)
		{
			for (int j = 0; j < rect.width; j++)
			{
				ret.ptr(i)[j] = mask.ptr(i)[j];
			}
		}

	}
	else if (mask.channels == 3)
	{
		for (int i = 0; i < rect.height; i++)
		{
			for (int j = 0; j < rect.width; j++)
			{
				ret.at<Vec3b>(i, j) = mask.at<Vec3b>(i + rect.y, j + rect.x);
			}
		}
	}
	return ret;
}

vector<vector<Point> > extract_contours(Mat binary_img)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(binary_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	return contours;
}

void draw_text(Mat image, String text, Point pos)
{
	putText(image, String(text), pos, FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2);
}

void __draw_text(Mat color_image, String text, vector<Point> contour)
{
	Rect rect = boundingRect(contour);
	Point pos = Point(rect.x, rect.y + rect.height + 20);
	draw_text(color_image, text, pos);
}

void draw_contour(Mat color_img, vector<Point> contour, Vec3b color, String text = "", bool filled = True)
{
	int flag = filled ? 3 : 1;
	drawContours(color_img, contour, -1, cvScalar(color[0], color[1], color[2]), flag);
	if (text != ""){
		Rect rect = boundingRect(contour);
		Point pos = Point(rect.x, rect.y + rect.height + 50);
		draw_text(color_img, text, pos);
	}
}

void draw_contours(Mat color_img, vector<vector<Point>> contours, Vec3b color = Vec3b(0, 0, 255),
	bool random_color = False, bool show_index = False)
{
	const int count = contours.size();
	for (int i = 0; i < count; i++)
	{
		vector<Point> contour = contours[i];
		if (random_color)
		{
			color[0] = rand() % 256;
			color[1] = rand() % 256;
			color[2] = rand() % 256;
		}
		draw_contour(color_img, contour, color);
		if (show_index)
		{
			char s[100];
			itoa(i, s, 10);
			__draw_text(color_img, s, contour);
		}
	}
}

Mat make_draw_img(Mat img)
{
	Mat cloned = img.clone();
	Mat draw_img;
	if (cloned.channels() == 1)
	{
		cvtColor(cloned, draw_img, COLOR_GRAY2BGR);
	}
	else
	{
		draw_img = img.clone();
	}
	return draw_img;
}

void show_image(Mat img, string name, bool wait_key = False)
{
	if (_Globals.show_process){
		imshow(name, img);
		if (_Globals.wait_key && wait_key){
			waitKey(0);
		}
	}
}

void show_mask(string filename)
{
	Mat mask = read_mask(filename, 1);
	mask = process_mask(mask);
	vector<vector<Point>> contours = extract_contours(mask);
	Mat draw_img = make_draw_img(mask);
	draw_contours(draw_img, contours, Vec3b(0, 0, 255), True, True);
	show_image(draw_img, filename, True);
}

Mat read_image(string infile, int color_mode = 0)
{
	Mat img = imread(infile, color_mode);
	return img;
}

void write_image(Mat img, string outfile)
{
	imwrite(outfile, img);
}

Mat denoise_2d(Mat img, int color = 255)
{
	int h = img.rows, w = img.cols;
	for (int i = 0; i < h; i++)
	{
		int sum = 0;
		for (int j = 0; j < w; j++)
		{
			if (img.ptr(i)[j] > 0) sum += 1;
		}
		if (sum * 10 < w){
			for (int j = 0; j < w; j++)
			{
				img.ptr(i)[j] = color;
			}
		}
	}
	return img;
}

double __match_contour(vector<Point> contour, Shape mask_shape)
{
	Rect rect = boundingRect(contour);
	double x = rect.x * 1.0, y = rect.y * 1.0, w = rect.width * 1.0, h = rect.height * 1.0;
	double tw = mask_shape.width * 1.0, th = mask_shape.height * 1.0;
	double value = 0.0;
	value = (pow((w - tw), 2) + pow((h - th), 2) + pow((w * h - tw * th), 2)) /
		(pow(tw, 2) + pow(th, 2) + pow((tw * th), 2));
	return value;
}

vector<Point> find_max_contour(vector<vector<Point>> contours, Shape mask_shape, int *idx)
{
	const int size = contours.size();
	double min_value = 0.0, value = 0.0;
	int index = 0;
	for (int i = 0; i < size; i++){
		vector<Point> con = contours[i];
		value = __match_contour(con, mask_shape);
		if (i == 0){
			min_value = value;
		}
		else
		{
			if (min_value > value){
				min_value = value;
				index = i;
			}
		}
	}

	*idx = index;

	return contours[index];
}

vector<Point> extract_best_match_contour(Mat img, Shape mask_shape)
{
	Mat thresh3;
	threshold(img, thresh3, 127, 255, THRESH_TRUNC);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilated;
	dilate(thresh3, dilated, kernel, cv::Point(-1, -1), 1);
	Mat binary;
	int th = int(_Thresholds.binary_threshold);
	threshold(dilated, binary, th, 255, THRESH_BINARY);
	show_image(binary, "dilate", True);
	binary = denoise_2d(binary, 0);
	show_image(binary, "denoised binary", True);
	vector<vector<Point>> contours = extract_contours(binary);
	vector<Point> max_contour = find_max_contour(contours, mask_shape, &th);

	return max_contour;
}

Rect get_contour_rect(vector<Point> contour)
{
	return boundingRect(contour);
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

Mat to_gray(Mat img)
{
	Mat ret;
	if (img.channels == 3){
		cvtColor(img, ret, COLOR_BGR2GRAY);
	}
	else{
		ret = img.clone();
	}
	return ret;
}

Mat extract_image_by_rect(Mat img, Rect rect, int left_offset = 0, int right_offset = 0,
	int top_offset = 0, int bottom_offset = 0)
{
	int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
	int new_h = h + bottom_offset - top_offset, new_w = w + right_offset - left_offset;
	Mat ret = Mat::zeros(new_h, new_w, img.type());
	if (img.channels == 1){
		for (int i = 0; i < new_h; i++){
			for (int j = 0; j < new_w; j++){
				ret.ptr(i)[j] = img.ptr(i + y + top_offset)[x + left_offset + j];
			}
		}

	}
	else if (img.channels == 3)
	{
		for (int i = 0; i < new_h; i++){
			for (int j = 0; j < new_w; j++){
				ret.at<Vec3b>(i, j) = img.at<Vec3b>(i + y + top_offset, x + left_offset + j);
			}
		}
	}

	Mat ret2 = to_gray(ret);
	return ret2;
}

Mat fill_seed(Mat img, Point seed, CvScalar color, Mat *_mask, int threshold = 10)
{
	int h = img.rows, w = img.cols;
	Mat mask = Mat::zeros(h + 2, w + 2, CV_8U);
	Rect rect;
	int floodflags = 4;
	floodflags |= FLOODFILL_FIXED_RANGE;
	floodflags |= (255 << 8);
	floodFill(img, seed, color, &rect, Scalar(0, 0, 0), Scalar(threshold, threshold, threshold), floodflags);
	*_mask = mask;
	return img;
}

Mat fill_4corners(Mat img, int top_offset = 0, int bottom_offset = 0, 
	CvScalar color = cvScalar(255, 255, 255), int threshold = 10)
{
	Mat img2 = img.clone();
	Mat mask;
	int h = img.rows, w = img.cols;
	fill_seed(img2, Point(1, 1 + top_offset), color, &mask, threshold);
	fill_seed(img2, Point(1, h - 2 + bottom_offset), color, &mask, threshold);
	fill_seed(img2, Point(w - 2, h - 2 + bottom_offset), color, &mask, threshold);
	fill_seed(img2, Point(w - 2, 1), color, &mask, threshold);

	return img2;
}

Mat extract_target(Mat img)
{
	Mat gray_img = to_gray(img);
	Mat thresh;
	threshold(gray_img, thresh, 1, 255, THRESH_BINARY_INV);
	return thresh;
}

void draw_rect(Mat img, int x, int y, int w, int h, CvScalar color = cvScalar(255, 0, 0),
	int line_width = 1, string text = "")
{
	rectangle(img, Point(x, y), Point(x + w - 1, y + h - 1), color, line_width);
	if (text != ""){
		Point pos = Point(x, y + h + 30);
		putText(img, text, pos, FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2);
	}
}

void draw_contour_rect(Mat img, vector<Point> contour, CvScalar color, int line_width = 1, string text = "")
{
	Rect rect = boundingRect(contour);
	int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
	draw_rect(img, x, y, w, h, color, line_width, text);
}

Mat match_with_template(Mat img, Mat _template, int method = TM_SQDIFF)
{
	Mat result;
	matchTemplate(img, _template, result, method);
	return result;
}

bool compare_by_width(const Rect &a, const Rect &b)
{
	return a.width < b.width;
}

bool compare_by_height(const Rect &a, const Rect &b)
{
	return a.height < b.height;
}

bool compare_by_area(const Rect &a, const Rect &b)
{
	return a.width * a.height < b.width * b.height;
}

void find_minmax_width_and_height(vector<Rect> rects, int *minw, int *maxw,
	int *minh, int *maxh, int *min_area, int *max_area)
{
	const int size = rects.size();
	std::sort(rects.begin(), rects.end(), compare_by_width);
	*minw = rects[0].width, *maxw = rects[size - 1].width;

	std::sort(rects.begin(), rects.end(), compare_by_height);
	*minh = rects[0].height, *maxh = rects[size - 1].height;

	std::sort(rects.begin(), rects.end(), compare_by_area);
	*min_area = rects[0].width * rects[0].height;
	*max_area = rects[size - 1].width * rects[size - 1].height;
}

Mat process_left(Mat left_target, vector<Rect> matched_rects,
	Mat *_draw_img,
	CvScalar best_match_color = cvScalar(0, 255, 0),
	CvScalar worst_match_color = cvScalar(0, 0, 255),
	double min_nonzero_pixel_ratio = 0.5)
{
	medianBlur(left_target, left_target, 3);
	Mat draw_img = make_draw_img(left_target);
	Mat cloned_draw_img = draw_img.clone();
	Mat gray_img = to_gray(left_target);
	Mat mask = Mat::zeros(left_target.rows, left_target.cols, CV_8U);

	int minw, maxw,
		minh, maxh,
		min_area_int, max_area_int;

	find_minmax_width_and_height(matched_rects, &minw, &maxw,
		&minh, &maxh, &min_area_int, &max_area_int);
	double min_area = min_area_int * min_nonzero_pixel_ratio;

	Rect rect;
	int x, y, w, h;

	const int size = matched_rects.size();
	for (int i = 0; i < size; i++){
		rect = matched_rects[i];
		x = rect.x, y = rect.y, w = rect.width, h = rect.height;
		Mat template_ = Mat::zeros(h, w, CV_8U);
		for (int j = 0; j < h; j++){
			for (int k = 0; k < w; k++){
				template_.ptr(j)[k] = 255;
			}
		}
		Mat result = match_with_template(gray_img, template_);
		double min_val, max_val;
		CvPoint min_loc, max_loc;
		cvMinMaxLoc(&result, &min_val, &max_val, &min_loc, &max_loc);

		int sum = 0;

		for (int j = min_loc.y; j < min_loc.y + h; j++){
			for (int k = min_loc.x; k < min_loc.x + w; k++){
				if (gray_img.ptr(j)[k] > 0) sum += 1;
			}
		}

		if (sum < min_area)
			continue;

		for (int j = min_loc.y; j < min_loc.y + h; j++){
			for (int k = min_loc.x; k < min_loc.x + w; k++){
				cloned_draw_img.at<Vec3b>(j, k) = WHITE_COLOR;
			}
		}
		Mat fill_mask;
		fill_seed(cloned_draw_img, min_loc, ABNORMAL_COLOR, &fill_mask);
		show_image(fill_mask, "fill mask", True);

		Mat same_size_mask = Mat::zeros(cloned_draw_img.rows, cloned_draw_img.cols, fill_mask.type());

		for (int j = 0; j < cloned_draw_img.rows; j++){
			for (int k = 0; k < cloned_draw_img.cols; k++){
				same_size_mask.ptr(j)[k] = fill_mask.ptr(j + 1)[k + 1];
			}
		}
		//where = np.where(same_size_mask > 0)
		//width = where[1].max() - where[1].min() + 1
		//height = where[0].max() - where[0].min() + 1

		int j, k;
		int start_w, end_w;
		int start_h, end_h;

		for (j = 0; j < same_size_mask.rows; j++){
			// top, start_h
			bool flag = False;
			for (k = 0; k < same_size_mask.cols; k++){
				if (same_size_mask.ptr(j)[k] > 0){
					flag = True;
					break;
				}
			}
			if (flag) break;
		}
		start_h = j;
		for (j = same_size_mask.rows - 1; j > start_h; j--){
			// bottom, end_h
			bool flag = False;
			for (k = 0; k < same_size_mask.cols; k++){
				if (same_size_mask.ptr(j)[k] > 0){
					flag = True;
					break;
				}
			}
			if (flag) break;
		}
		end_h = j;

		for (j = 0; j < same_size_mask.cols; j++){
			// left, start_w
			bool flag = False;
			for (k = 0; k < same_size_mask.rows; k++){
				if (same_size_mask.ptr(k)[j] > 0){
					flag = True;
					break;
				}
			}
			if (flag) break;
		}
		start_w = j;

		for (j = same_size_mask.cols - 1; j > start_w; j--){
			// left, start_w
			bool flag = False;
			for (k = 0; k < same_size_mask.rows; k++){
				if (same_size_mask.ptr(k)[j] > 0){
					flag = True;
					break;
				}
			}
			if (flag) break;
		}
		end_w = j;

		int width = end_w - start_w + 1;
		int height = end_h - start_h + 1;

		if ((width > maxw) || (height > maxh)){
			for (int j = 0; j < same_size_mask.rows; j++){
				for (int k = 0; k < same_size_mask.cols; k++){
					if (same_size_mask.ptr(j)[k] > 0)
						gray_img.ptr(j)[k] = 0;
				}
			}
			show_image(gray_img, "update gray image", True);
		}
		else
		{
			draw_rect(draw_img, min_loc.x, min_loc.y, w, h, best_match_color, 1);
			draw_rect(draw_img, max_loc.x, max_loc.y, w, h, worst_match_color, 1);
			for (int j = min_loc.y; j < min_loc.y + h; j++){
				for (int k = min_loc.x; k < min_loc.x + w; k++){
					mask.ptr(j)[k] = 1;
				}
			}
		}
	}
	for (int j = 0; j < mask.rows; j++){
		for (int k = 0; k < mask.cols; k++){
			if (mask.ptr(j)[k] < 1)
				left_target.ptr(j)[k] = 0;
		}
	}

	*_draw_img = draw_img;
	return left_target;
}

vector<Point> rect_2_vector(Rect rect)
{
	vector<Point> ret;
	ret.push_back(Point(rect.x, rect.y));
	ret.push_back(Point(rect.x, rect.y + rect.height));
	ret.push_back(Point(rect.x + rect.width, rect.y + rect.height));
	ret.push_back(Point(rect.x + rect.width, rect.y));
	ret.push_back(Point(rect.x, rect.y));
	return ret;
}

double match_shapes(Rect rect, Rect template_rect, int method = CV_CONTOURS_MATCH_I3)
{
	double value = 0.0;
	value = matchShapes(rect_2_vector(template_rect), rect_2_vector(rect), method, 0);
	return value;
}

bool __is_rect_center(Rect rect, int top, int bottom, int left, int right)
{
	int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
	if ((y < top) || (y > bottom)) return False;
	if ((x < left) || (x > right)) return False;
	return True;
}

vector<double> __match_rects(Rect rect, vector<Rect> template_rects, int method = CV_CONTOURS_MATCH_I3)
{
	vector<double> ret;
	const int size = template_rects.size();
	for (int i = 0; i < size; i++){
		Rect trect = template_rects[i];
		double value = match_shapes(rect, trect, method);
		ret.push_back(value);
	}
	return ret;
}

vector<int> __detect_algorithm(vector<vector<Point>> contours, vector<Rect> matched_rects,
	double threshold, int top, int bottom, int left, int right)
{
	const int size = contours.size();
	vector<int> indexes;
	for (int i = 0; i < size; i++){
		Rect bounding_rect = boundingRect(contours[i]);
		if (!__is_rect_center(bounding_rect, top, bottom, left, right)) continue;
		vector<double> ret = __match_rects(bounding_rect, matched_rects);
		vector<double>::iterator minimum = min_element(std::begin(ret), std::end(ret));
		if (*minimum <= threshold) indexes.push_back(i);
	}
	return indexes;
}

vector<Rect> detect_if_abnormal_of_left_image(Mat left_image, vector<Rect> matched_rects,
	double max_difference_ratio = 0.1,
	double ignore_left_right_ratio = 0.1,
	double ignore_top_bottom_ratio = 0.1)
{
	Mat left_img_gray = to_gray(left_image);
	vector<vector<Point>> contours = extract_contours(left_img_gray);
	Mat draw_image = make_draw_img(left_image);

	draw_contours(draw_image, contours, Vec3b(0, 0, 255), True, True);
	show_image(draw_image, "left contours", True);

	Mat abnormal_draw_image = make_draw_img(left_image);
	vector<Rect> abnormal_rect;

	int image_h = left_image.rows, image_w = left_image.cols;

	int left = int(image_w * ignore_left_right_ratio);
	int right = int(image_w * (1 - ignore_left_right_ratio));
	int	top = int(image_h * ignore_top_bottom_ratio);
	int bottom = int(image_h * (1 - ignore_top_bottom_ratio));

	vector<int> indexes = __detect_algorithm(contours, matched_rects,
		max_difference_ratio, top, bottom, left, right);

	const int size = indexes.size();
	for (int i = 0; i < size; i++){
		draw_contour_rect(abnormal_draw_image, 
			contours[indexes[i]], cvScalar(0, 0, 255), 1, "Abnormal");
		abnormal_rect.push_back(boundingRect(contours[indexes[i]]));
	}

	show_image(abnormal_draw_image, "abnormal draw image", True);
	return abnormal_rect;
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