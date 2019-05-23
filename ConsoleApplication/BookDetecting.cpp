#include "BookDetecting.h"
#include "readconf.h"

#define MISS_COLOR Vec3b(255, 0, 255)
#define ABNORMAL_COLOR Vec3b(0, 0, 255)
#define NORMAL_COLOR Vec3b(0, 255, 0)
#define WHITE_COLOR Vec3b(255, 255, 255)
#define BLACK_COLOR Vec3b(0, 0, 0)

// 全局配置
static Globals _Globals;
// 全局阈值
static Thresholds _Thresholds;

typedef struct Shape
{
	int width;
	int height;
}Shape;

void default_globals()
{
	_Globals.wait_key = True;
	_Globals.show_process = True;
	_Globals.in_folder = "";
	_Globals.out_folder = "";
	_Globals.mask_file = "";
}

//void default_thresholds()
//{
//	_Thresholds.binary_threshold = 80;
//	_Thresholds.max_difference_ratio = 0.5;
//	_Thresholds.ignore_left_right_ratio = 0.15;
//	_Thresholds.ignore_top_bottom_ratio = 0.2;
//	_Thresholds.min_nonzero_pixel_ratio = 0.5;
//}

void read_thresholds(string filename)
{
	float *arr = read_config(filename);
	_Thresholds.binary_threshold = arr[0];
	_Thresholds.max_difference_ratio = arr[1];
	_Thresholds.ignore_left_right_ratio = arr[2];
	_Thresholds.ignore_top_bottom_ratio = arr[3];
	_Thresholds.min_nonzero_pixel_ratio = arr[4];
}

int cvErrorRedirector(int status, const char * func_name, const char * err_msg, const char * file_name, int line, void * userdata)
{
	return 1;
}

Mat to_gray(Mat img)
{
	Mat ret;
	if (img.channels() == 3) {
		cvtColor(img, ret, COLOR_BGR2GRAY);
	}
	else {
		ret = img.clone();
	}
	return ret;
}

vector<vector<Point> > extract_contours(Mat binary_img)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	if (binary_img.empty() || binary_img.channels() != 1)
	{
		throw MyException("Image is empty.", 72);
	}
	try
	{
		findContours(binary_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	}
	catch (cv::Exception &e)
	{
		throw MyException("findContours", 72);
	}
	return contours;
}

Mat read_mask(String maskfile, int channels = 3)
{
	Mat mask = imread(maskfile);
	if (mask.empty())
	{
		throw MaskException();
	}
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
				if (mask.ptr(j)[i * 3 + 0] > 0){
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
				if (mask.ptr(j)[i * 3 + 0] > 0){
					mask2.ptr(j)[i] = 255;
				}
			}
		}
	}
	return mask2;
}

Rect mask_boundingRect(Mat mask)
{
	int j, k;
	int start_w, end_w;
	int start_h, end_h;

	for (j = 0; j < mask.rows; j++){
		// top, start_h
		bool flag = False;
		for (k = 0; k < mask.cols; k++){
			if (mask.ptr(j)[k] > 0){
				flag = True;
				break;
			}
		}
		if (flag) break;
	}
	start_h = j;

	for (j = mask.rows - 1; j > start_h; j--){
		// bottom, end_h
		bool flag = False;
		for (k = 0; k < mask.cols; k++){
			if (mask.ptr(j)[k] > 0){
				flag = True;
				break;
			}
		}
		if (flag) break;
	}
	end_h = j;

	for (j = 0; j < mask.cols; j++){
		// left, start_w
		bool flag = False;
		for (k = 0; k < mask.rows; k++){
			if (mask.ptr(k)[j] > 0){
				flag = True;
				break;
			}
		}
		if (flag) break;
	}
	start_w = j;

	for (j = mask.cols - 1; j > start_w; j--){
		// left, start_w
		bool flag = False;
		for (k = 0; k < mask.rows; k++){
			if (mask.ptr(k)[j] > 0){
				flag = True;
				break;
			}
		}
		if (flag) break;
	}

	end_w = j;
	Rect rect;
	rect.x = start_w;
	rect.y = start_h;
	rect.width = end_w - start_w + 1;
	rect.height = end_h - start_h + 1;

	return rect;
}

Mat process_mask(Mat mask)
{
	//Mat gray_mask = to_gray(mask);
	//Rect rect = boundingRect(mask);
	Rect rect = mask_boundingRect(mask);
	Mat ret = Mat::zeros(rect.height, rect.width, mask.type());

	if (mask.channels() == 1){
		for (int i = 0; i < rect.height; i++)
		{
			for (int j = 0; j < rect.width; j++)
			{
				ret.ptr(i)[j] = mask.ptr(i + rect.y)[j + rect.x];
			}
		}

	}
	else if (mask.channels() == 3)
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

void draw_text(Mat image, String text, Point pos)
{
	try
	{
		putText(image, String(text), pos, FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2);
	}
	catch (cv::Exception &e)
	{
		throw MyException("putText", 224);
	}
}

void __draw_text(Mat color_image, String text, vector<Point> contour)
{
	Rect rect = boundingRect(contour);
	Point pos = Point(rect.x, rect.y + rect.height + 20);
	draw_text(color_image, text, pos);
}

void draw_contour(Mat color_img, vector<Point> contour, CvScalar color, String text = "", bool filled = True)
{
	int flag = filled == True ? 3 : 1;
	vector<vector<Point>> contours;
	contours.push_back(contour);
	if (color_img.empty())
	{
		throw MyException("Image is empty", 250);
	}
	try
	{
		drawContours(color_img, contours, 0, color, flag);
	}
	catch (cv::Exception &e)
	{
		throw MyException("drawContours", 258);
	}
	if (text != "") {
		Rect rect = boundingRect(contour);
		Point pos = Point(rect.x, rect.y + rect.height + 50);
		draw_text(color_img, text, pos);
	}
}

void draw_contours(Mat color_img, vector<vector<Point>> contours, Vec3b color = Vec3b(0, 0, 255), bool random_color = False, bool show_index = False)
{
	if (contours.empty())
	{
		throw MyException("Contours are empty", 268);
	}
	const int count = contours.size();
	for (int i = 0; i < count; i++)
	{
		vector<Point> contour = contours.at(i);
		if (random_color == True)
		{
			color[0] = rand() % 256;
			color[1] = rand() % 256;
			color[2] = rand() % 256;
		}
		draw_contour(color_img, contour, color);
		if (show_index == True)
		{
			char s[100];
			_itoa_s(i, s, 10);
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
	Mat img;
	img = imread(infile, color_mode);
	if (img.empty())
	{
		throw MyException("read_image", 320, "Failed to read image");
	}
	return img;
}

void write_image(Mat img, string outfile)
{
	try
	{
		imwrite(outfile, img);
	}
	catch (cv::Exception &e)
	{
		throw MyException("imwrite", 332);
	}
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
	try
	{
		value = (pow((w - tw), 2) + pow((h - th), 2) + pow((w * h - tw * th), 2)) /
			(pow(tw, 2) + pow(th, 2) + pow((tw * th), 2));
	}
	catch (Exception &e)
	{
		throw MyException("__match_contour", 360);
	}
	return value;
}

Mat fill_seed(Mat img, Point seed, CvScalar color, Mat *_mask, int threshold = 10)
{
	int h = img.rows, w = img.cols;
	Mat mask = Mat::zeros(h + 2, w + 2, CV_8U);
	Rect rect;
	int floodflags = 4;
	floodflags |= FLOODFILL_FIXED_RANGE;
	floodflags |= (255 << 8);
	try
	{
		floodFill(img, seed, color, &rect, Scalar(0, 0, 0), Scalar(threshold, threshold, threshold), floodflags);
	}
	catch (cv::Exception &e)
	{
		throw MyException("floodFill", 522);
	}
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
	try
	{
		threshold(gray_img, thresh, 1, 255, THRESH_BINARY_INV);
	}
	catch (cv::Exception &e)
	{
		throw MyException("threshold", 552);
	}
	return thresh;
}

void draw_rect(Mat img, int x, int y, int w, int h, CvScalar color = cvScalar(255, 0, 0),
	int line_width = 1, string text = "")
{
	try
	{
		rectangle(img, Point(x, y), Point(x + w - 1, y + h - 1), color, line_width);
	}
	catch (cv::Exception &e)
	{
		throw MyException("rectangle", 566);
	}
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
	try
	{
		matchTemplate(img, _template, result, method);
	}
	catch (cv::Exception &e)
	{
		throw MyException("matchTemplate", 590);
	}
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
	if (size == 0)
	{
		throw MyException("find_minmax_width_and_height", 614, "Size of rects equals 0.");
	}
	std::sort(rects.begin(), rects.end(), compare_by_width);
	*minw = rects.at(0).width, *maxw = rects.at(size - 1).width;

	std::sort(rects.begin(), rects.end(), compare_by_height);
	*minh = rects.at(0).height, *maxh = rects.at(size - 1).height;

	std::sort(rects.begin(), rects.end(), compare_by_area);
	*min_area = rects.at(0).width * rects.at(0).height;
	*max_area = rects.at(size - 1).width * rects.at(size - 1).height;
}

Mat process_left(Mat left_target, vector<Rect> matched_rects,
	Mat *_draw_img,
	CvScalar best_match_color = cvScalar(0, 255, 0),
	CvScalar worst_match_color = cvScalar(0, 0, 255),
	double min_nonzero_pixel_ratio = 0.5)
{
	try
	{
		medianBlur(left_target, left_target, 3);
	}
	catch (cv::Exception &e)
	{
		throw MyException("medianBlur", 641);
	}
	Mat draw_img = make_draw_img(left_target);
	Mat cloned_draw_img = draw_img.clone();
	Mat gray_img = to_gray(left_target);
	Mat mask = Mat::zeros(left_target.rows, left_target.cols, CV_8U);

	int minw = 0, maxw = 0,
		minh = 0, maxh = 0,
		min_area_int = 0, max_area_int = 0;

	find_minmax_width_and_height(matched_rects, &minw, &maxw,
		&minh, &maxh, &min_area_int, &max_area_int);
	double min_area = min_area_int * min_nonzero_pixel_ratio;

	Rect rect;
	int x, y, w, h;

	const int size = matched_rects.size();
	for (int i = 0; i < size; i++){
		rect = matched_rects.at(i);
		x = rect.x, y = rect.y, w = rect.width, h = rect.height;
		Mat template_ = Mat::zeros(h, w, CV_8U);
		for (int j = 0; j < h; j++){
			for (int k = 0; k < w; k++){
				template_.ptr(j)[k] = 255;
			}
		}
		Mat result = match_with_template(gray_img, template_);
		double min_val, max_val;
		Point min_loc, max_loc;
		try
		{
			minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
		}
		catch (cv::Exception &e)
		{
			throw MyException("minMaxLoc", 678);
		}
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

		//TODO replace
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
	try
	{
		value = matchShapes(rect_2_vector(template_rect), rect_2_vector(rect), method, 0);
	}
	catch (cv::Exception &e)
	{
		throw MyException("matchShapes", 821);
	}
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
		Rect trect = template_rects.at(i);
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
		Rect bounding_rect = boundingRect(contours.at(i));
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
	vector<Rect> abnormal_rect;
	Mat left_img_gray = to_gray(left_image);
	vector<vector<Point>> contours = extract_contours(left_img_gray);
	if (contours.empty()) {
		return abnormal_rect;
	}

	Mat draw_image = make_draw_img(left_image);

	draw_contours(draw_image, contours, Vec3b(0, 0, 255), True, True);
	show_image(draw_image, "left contours", True);

	Mat abnormal_draw_image = make_draw_img(left_image);

	int image_h = left_image.rows, image_w = left_image.cols;

	int left = int(image_w * ignore_left_right_ratio);
	int right = int(image_w * (1 - ignore_left_right_ratio));
	int	top = int(image_h * ignore_top_bottom_ratio);
	int bottom = int(image_h * (1 - ignore_top_bottom_ratio));

	vector<int> indexes = __detect_algorithm(contours, matched_rects,
		max_difference_ratio, top, bottom, left, right);

	const int size = indexes.size();
	for (int i = 0; i < size; i++) {
		draw_contour_rect(abnormal_draw_image,
			contours.at(indexes.at(i)), cvScalar(0, 0, 255), 1, "Abnormal");
		abnormal_rect.push_back(boundingRect(contours.at(indexes.at(i))));
	}

	show_image(abnormal_draw_image, "abnormal draw image", True);
	return abnormal_rect;
}

vector<Rect> find_mask_rects(Mat mask)
{
	Mat cloned;
	if (cloned.channels() == 3){
		cvtColor(mask, cloned, COLOR_BGR2GRAY);
	}
	else{
		cloned = mask.clone();
	}
	vector<vector<Point>> contours = extract_contours(cloned);

	vector<Rect> ret;
	if (contours.empty())
	{
		throw MyException("Contours are empty.", 792);
	}
	const int size = contours.size();
	for (int i = 0; i < size; i++){
		Rect rect = boundingRect(contours.at(i));
		ret.push_back(rect);
	}
	return ret;
}

//match with target
Rect mask_matches_target(Mat mask, Mat target) {
	int h = mask.rows, w = mask.cols;
	Mat draw_img = make_draw_img(target);
	CvScalar best_match_color = cvScalar(0, 255, 0);
	CvScalar worst_match_color = cvScalar(0, 0, 255);
	Mat result = match_with_template(target, mask);
	double min_val, max_val;
	Point min_loc, max_loc;
	try
	{
		minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
	}
	catch (cv::Exception &e)
	{
		throw MyException("minMaxLoc", 932);
	}
	// print(min_val, max_val, min_loc, max_loc)
	// best match, green
	CvScalar color = best_match_color;
	int line_width = 1;
	draw_rect(draw_img, min_loc.x, min_loc.y, w, h, color, line_width);
	// worst match, red
	color = worst_match_color;
	draw_rect(draw_img, max_loc.x, max_loc.y, w, h, color, line_width);
	// show image
	show_image(draw_img, "match mask with target", True);

	Rect rect;
	rect.x = min_loc.x;
	rect.y = min_loc.y;
	rect.width = w;
	rect.height = h;
	return rect;
}

Mat fill_rect_bound(Mat gray, Rect rect, int color = 255){								////////////////////////////////
	int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
	int gray_h = gray.rows, gray_w = gray.cols;
	int mx = x + w;
	int my = y + h;
	mx = mx > gray_w ? gray_w : mx;
	my = my > gray_h ? gray_h : my;

	for (int i = x; i < mx; i++)
	{
		gray.ptr(y)[i] = color;
	}

	for (int i = y; i < my; i++)
	{
		gray.ptr(i)[x] = color;
	}

	if (mx == gray_w)
	{
		for (int i = y; i < my; i++)
		{
			gray.ptr(i)[mx - 1] = color;
		}
	}
	else
	{
		for (int i = y; i < my; i++)
		{
			gray.ptr(i)[mx] = color;
		}
	}

	if (my == gray_h)
	{
		for (int i = x; i < mx; i++)
		{
			gray.ptr(my - 1)[i] = color;
		}
	}
	else
	{
		for (int i = x; i < mx; i++)
		{
			gray.ptr(my)[i] = color;
		}
	}
	return gray;
}

Mat clip_Mat_by_rect(Mat mat, Rect rect)
{
	IplImage* p_img = &IplImage(mat);
	IplImage *newImg = cvCreateImage(cvSize(rect.width, rect.height), p_img->depth, p_img->nChannels);
	cvSetImageROI(p_img, rect);
	cvCopy(p_img, newImg);
	cvResetImageROI(p_img);
	Mat temp_img = cvarrToMat(newImg);
	Mat ret = temp_img.clone();
	cvReleaseImage(&newImg);
	return ret;
}

int count_pixels(Mat img, int threshold = 0)
{
	int w = img.cols, h = img.rows;

	int sum = 0;
	for (int j = 0; j < h; j++) {
		for (int k = 0; k < w; k++)
		{
			if (img.ptr(j)[k] > threshold) {
				sum += 1;
			}
		}
	}
	return sum;
}

void process_main_part(string infile, string maskfile,
	int left_offset = -5, int right_offset = 5,
	int top_offset = -5, int bottom_offset = 5, 
	vector<Rect> *_template_rects = 0,
	vector<Rect> *_matched_rects = 0,
	int *_miss_flag = 0,
	vector<Rect> *_miss_rects = 0,
	Mat *_left_target = 0,
	Mat *_for_draw_result = 0,
	Rect *_best_match_rect = 0,
	Rect *_target_rect = 0)
{
	//read mask
	Mat mask = read_mask(maskfile, 1);
	show_image(mask, "mask", True);
	Shape mask_shape = { mask.cols, mask.rows };
	Mat processed_mask = process_mask(mask);
	show_image(processed_mask, "processed mask", True);

	//read color image
	Mat color_img = read_image(infile, 1);
	show_image(color_img, "origin", True);
	Mat gray_img;
	cvtColor(color_img, gray_img, CV_BGR2GRAY);

	Mat mask_255 = Mat::zeros(mask.rows, mask.cols, CV_8U);
	Mat temp_mask;
	fill_seed(mask_255, cvPoint(1, 1), cvScalar(255, 255, 255), &temp_mask);
	show_image(mask_255, "mask 255");

	Mat result = match_with_template(gray_img, mask_255);
	double min_val, max_val;
	Point min_loc, max_loc;
	try
	{
		minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
	}
	catch (cv::Exception &e)
	{
		throw MyException("minMaxLoc", 656);
	}

	Rect target_rect = cvRect(min_loc.x, min_loc.y, mask.cols, mask.rows);

	//long min_value = result.ptr(min_loc.y)[min_loc.x];

	//if (min_value > 215.0)
	//{
	//	throw MyException("Invalid Image", 952);
	//}

	Mat clip_grabcut = clip_Mat_by_rect(gray_img, target_rect);

	//// 判断匹配像素是否有效
	//int pixel_count = count_pixels(clip_grabcut);


	show_image(clip_grabcut, "clip result", True);
	Mat for_draw_result = clip_grabcut.clone();
	try
	{
		threshold(clip_grabcut, clip_grabcut, int(_Thresholds.binary_threshold), 255, THRESH_TOZERO);
	}
	catch (cv::Exception &e)
	{
		throw MyException("threshold", 1040);
	}
	show_image(clip_grabcut, "clip result 2", True);


	for (int j = 0; j < clip_grabcut.rows; j++) {
		for (int k = 0; k < clip_grabcut.cols; k++) {
			if (clip_grabcut.ptr(j)[k] > 0)
				clip_grabcut.ptr(j)[k] = 255;
		}
	}

	//clip_grabcut[clip_grabcut > 0] = 255;	

	//clone for draw rect
	Mat clone_clip_grabcut = clip_grabcut.clone();

	show_image(clip_grabcut, "clip result 3", True);

	threshold(clip_grabcut, clip_grabcut, 1, 255, THRESH_BINARY_INV);

	show_image(clip_grabcut, "clip result 31", True);

	Mat filled_img;
	////fill 4 corners
	//CvScalar color = cvScalar(255, 255, 255);
	//Mat filled_img = fill_4corners(clip_grabcut, top_offset = 0, bottom_offset = 0, color, 10);
	//show_image(filled_img, "filled 4 corners", True);

	medianBlur(clip_grabcut, filled_img, 3);

	////extract target
	//Mat target = extract_target(filled_img);
	//show_image(target, "target", True);

	Rect best_match_rect = mask_matches_target(processed_mask, filled_img);

	clone_clip_grabcut = fill_rect_bound(clone_clip_grabcut, best_match_rect);
	show_image(clone_clip_grabcut, "clone clip grabcut", True);

	clone_clip_grabcut = fill_4corners(clone_clip_grabcut, 0, 0, cvScalar(255, 255, 255));
	show_image(clone_clip_grabcut, "clone clip grabcut 2", True);

	Mat target2 = extract_target(clone_clip_grabcut);
	show_image(target2, "target2", True);

	Mat clip_target2 = clip_Mat_by_rect(target2, best_match_rect);
	show_image(clip_target2, "clip target2", True);

	Mat draw_img = make_draw_img(clip_target2);
	int bx = best_match_rect.x, by = best_match_rect.y, bw = best_match_rect.width, bh = best_match_rect.height;
	int miss_flag = 0;
	vector<Rect> template_rects, matched_rects, miss_rects;
	vector<Rect> rects = find_mask_rects(processed_mask);
	const int size = rects.size();
	int x, y, w, h;
	CvScalar draw_color;
	Mat _mask;
	for (int i = 0; i < size; i++) {
		Rect rect = rects.at(i);
		template_rects.push_back(rect);
		x = rect.x, y = rect.y, w = rect.width, h = rect.height;
		//Mat roi = Mat::zeros(h, w, clip_target2.type());

		int sum = 0;
		for (int j = 0; j < h; j++) {
			for (int k = 0; k < w; k++)
			{
				if (clip_target2.ptr(j + y)[k + x] > 0) {
					sum += 1;
				}
			}
		}
		if (sum > 0) {
			matched_rects.push_back(rect);
			draw_color = NORMAL_COLOR;

			for (int j = y + by; j < y + by + h; j++)
			{
				for (int k = x + bx; k < x + bx + w; k++)
				{
					target2.ptr(j)[k] = 255;
				}
			}
			target2 = fill_seed(target2, Point(x + bx, y + by), cvScalar(0, 0, 0), &_mask);
			show_image(target2, "target2", True);
		}
		else {
			miss_rects.push_back(rect);
			miss_flag += 1;
			draw_color = MISS_COLOR;
		}
		draw_rect(draw_img, x, y, w, h, draw_color, 1);
		show_image(draw_img, "draw_img", True);
	}
	Mat left_target = target2.clone();

	*_template_rects = template_rects;
	*_matched_rects = matched_rects;
	*_miss_flag = miss_flag;
	*_miss_rects = miss_rects;
	*_left_target = left_target;
	*_for_draw_result = for_draw_result;
	*_best_match_rect = best_match_rect;
	*_target_rect = target_rect;
}

bool make_decision(bool miss_flag, vector<Rect> left_rect)
{
	if (miss_flag || left_rect.size() > 0) return True;
	return False;
}

void draw_decision_text(Mat image, bool decision)
{
	String text;
	text = decision ? "Abnormal" : "Normal";
	try
	{
		putText(image, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(0, 0, 255), 2);
	}
	catch (cv::Exception &e)
	{
		throw MyException("putText", 1176);
	}
}

void draw_result(Rect best_match_rect, vector<Rect> matched_rects, vector<Rect> miss_rects,
	vector<Rect> abnormal_rects, Mat draw_image, CvScalar normal_color = NORMAL_COLOR,
	CvScalar abnormal_color = ABNORMAL_COLOR, CvScalar miss_color = MISS_COLOR, int line_width = 1)
{
	int bx = best_match_rect.x, by = best_match_rect.y, bw = best_match_rect.width, bh = best_match_rect.height;

	Rect rect;
	//matched
	int size = matched_rects.size();
	for (int i = 0; i < size; i++){
		rect = matched_rects.at(i);
		draw_rect(draw_image, bx + rect.x,
			by + rect.y, rect.width, rect.height,
			normal_color, line_width);
	}
	//miss
	size = miss_rects.size();
	for (int i = 0; i < size; i++){
		rect = miss_rects.at(i);
		draw_rect(draw_image, bx + rect.x,
			by + rect.y, rect.width, rect.height,
			miss_color, line_width);
	}
	//abnormal
	size = abnormal_rects.size();
	for (int i = 0; i < size; i++){
		rect = abnormal_rects.at(i);
		draw_rect(draw_image, bx + rect.x,
			by + rect.y, rect.width, rect.height,
			abnormal_color, line_width);
	}
}

void draw_on_origin(Mat image, vector<Rect> matched_rects, vector<Rect> miss_rects,
	vector<Rect> abnormal_rects, Rect best_match_rect, Rect target_rect, bool decision,
	int left_offset, int top_offset, CvScalar normal_color=NORMAL_COLOR,
	CvScalar abnormal_color=ABNORMAL_COLOR, CvScalar miss_color=MISS_COLOR, int line_width=1)
{
	int bx = best_match_rect.x, by = best_match_rect.y, bw = best_match_rect.width, bh = best_match_rect.height;
	int tx = target_rect.x, ty = target_rect.y, tw = target_rect.width, th = target_rect.height;

	Rect rect;
	//matched
	int size = matched_rects.size();
	for (int i = 0; i < size; i++){
		rect = matched_rects.at(i);
		draw_rect(image, bx + rect.x + tx + left_offset,
			by + rect.y + ty + top_offset, rect.width, rect.height,
			normal_color, line_width);
	}
	//miss
	size = miss_rects.size();
	for (int i = 0; i < size; i++){
		rect = miss_rects.at(i);
		draw_rect(image, bx + rect.x + tx + left_offset,
			by + rect.y + ty + top_offset, rect.width, rect.height,
			miss_color, line_width);
	}
	//abnormal
	size = abnormal_rects.size();
	for (int i = 0; i < size; i++){
		rect = abnormal_rects.at(i);
		draw_rect(image, bx + rect.x + tx + left_offset,
			by + rect.y + ty + top_offset, rect.width, rect.height,
			abnormal_color, line_width);
	}
	draw_decision_text(image, decision);
}

bool run(string infile, string mask,string outfile = "")
{
	int left_offset = -5,
		right_offset = 5,
		top_offset = -5,
		bottom_offset = 5;

	vector<Rect> template_rects;
	vector<Rect> matched_rects;
	int miss_flag;
	vector<Rect> miss_rects;
	Mat left_target;
	Mat for_draw_result;
	Rect best_match_rect;
	Rect target_rect;

	process_main_part(infile, mask, left_offset, right_offset, top_offset, bottom_offset,
		&template_rects, &matched_rects, &miss_flag, &miss_rects, &left_target,
		&for_draw_result, &best_match_rect, &target_rect);

	Mat left_img, left_draw_img;

	left_img = process_left(left_target, matched_rects, &left_draw_img, 
		cvScalar(0, 255, 0), cvScalar(0, 0, 255),_Thresholds.min_nonzero_pixel_ratio);

	show_image(left_draw_img, "match in left image", True);
	show_image(left_img, "left image", True);

	vector<Rect> abnormal_rects = detect_if_abnormal_of_left_image(left_img, matched_rects, _Thresholds.max_difference_ratio,
		_Thresholds.ignore_left_right_ratio, _Thresholds.ignore_top_bottom_ratio);

	bool decision = make_decision(miss_flag, abnormal_rects);

	Mat for_draw = make_draw_img(for_draw_result);
	
	draw_result(best_match_rect, matched_rects, miss_rects, abnormal_rects, for_draw);

	draw_decision_text(for_draw, decision);

	show_image(for_draw, "Final", True);

	Mat origin_image = read_image(infile, 1);
	draw_on_origin(origin_image, matched_rects, miss_rects,
		abnormal_rects, best_match_rect, target_rect,
		decision, left_offset, top_offset, cvScalar(0, 255, 0),
		cvScalar(0, 0, 255), cvScalar(255, 0, 255), 1);

	show_image(origin_image, "Result", True);
	if (outfile != ""){
		write_image(origin_image, outfile);
	}

	if (_Globals.show_process){
		destroyAllWindows();
	}

	return decision;
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

int bookDetecting(string filename, string maskfile, string outfile, bool verbose, bool timing, double *time_ms)
{
	//初始化计时
	LARGE_INTEGER  freq_num;
	long long start_time, end_time, freq;
	if (timing){
		QueryPerformanceFrequency(&freq_num);
		freq = freq_num.QuadPart;
		QueryPerformanceCounter(&freq_num);
		start_time = freq_num.QuadPart;
	}

	int ret = -1;
	try
	{
		ret = run(filename, maskfile, outfile);
	}
	catch (MyException &e)
	{
		if (verbose) {
			std::cout << e.what() << "\t Line " << e.lineno();
		}
	}

	catch (MaskException &e)
	{
		if (verbose)
		{
			std::cout << "Failed to read mask file." << std::endl;
		}
		ret = -2;
	}
	
	catch (cv::Exception &e)
	{
		if (verbose)
		{
			std::cout << e.what();
		}
	}
	catch(...)
	{
		;
	}

	if (timing){
		QueryPerformanceCounter(&freq_num);
		end_time = freq_num.QuadPart;
		if (time_ms != 0) {
			*time_ms = (end_time - start_time) * 1000 / (freq *  1.0);
		}
	}
	return ret;
}
