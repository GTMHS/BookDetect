#include "BookbindingDetector.h"

void default_globals()
{
	_Globals.wait_key = true;
	_Globals.show_process = true;
	_Globals.in_folder = "";
	_Globals.out_folder = "";
	_Globals.mask_file = "ws.png";
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

	if (mask.channels() == 1){
		for (int i = 0; i < rect.height; i++)
		{
			for (int j = 0; j < rect.width; j++)
			{
				ret.ptr(i)[j] = mask.ptr(i)[j];
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
			_itoa(i, s, 10);
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
	if (img.channels() == 3){
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
	if (img.channels() == 1){
		for (int i = 0; i < new_h; i++){
			for (int j = 0; j < new_w; j++){
				ret.ptr(i)[j] = img.ptr(i + y + top_offset)[x + left_offset + j];
			}
		}

	}
	else if (img.channels() == 3)
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
	const int size = contours.size();
	for (int i = 0; i < size; i++){
		Rect rect = boundingRect(cloned);
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
	CvPoint min_loc, max_loc;
	cvMinMaxLoc(&result, &min_val, &max_val, &min_loc, &max_loc);
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

Mat fill_rect_bound(Mat gray, Rect rect, int color = 255){									////////////////////////////////
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

Mat clip_target(Mat target, Rect rect) {
	int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
	Mat ret = Mat::zeros(h, w, target.type());
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			ret.ptr(i)[j] = target.ptr(i + y)[j + x];
		}
	}
	return ret;
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
	Shape mask_shape = { mask.cols, mask.rows };														//				

	//process mask
	Mat processed_mask = process_mask(mask);
	show_image(processed_mask, "processed mask", True);

	//read color image
	Mat color_img = read_image(infile, 1);
	show_image(color_img, "origin", True);
	Mat gray_img;
	cvtColor(color_img, gray_img, CV_BGR2GRAY);
	//find contour
	vector<Point> contour = extract_best_match_contour(gray_img, mask_shape);
	//get rect
	Rect target_rect = get_contour_rect(contour);
	//grabcut
	Mat grabcut_result = grabcut(infile, target_rect);
	show_image(grabcut_result, "grabcut result", True);
	// clip
	Mat clip_grabcut = extract_image_by_rect(grabcut_result, target_rect,
		left_offset = left_offset, right_offset = right_offset,
		top_offset = top_offset, bottom_offset = bottom_offset);
	show_image(clip_grabcut, "clip result", True);
	Mat for_draw_result = clip_grabcut.clone();											//1607

	threshold(clip_grabcut, clip_grabcut,
		100, 255, THRESH_TOZERO);
	show_image(clip_grabcut, "clip result 2", True);


	for (int j = 0; j < clip_grabcut.rows; j++){
		for (int k = 0; k < clip_grabcut.cols; k++){
			if (clip_grabcut.ptr(j)[k] > 0)
				clip_grabcut.ptr(j)[k] = 255;
		}
	}

	//clip_grabcut[clip_grabcut > 0] = 255;												//

	//clone for draw rect
	Mat clone_clip_grabcut = clip_grabcut.clone();

	show_image(clip_grabcut, "clip result 3", True);

	//fill 4 corners
	CvScalar color = cvScalar(255, 255, 255);
	Mat filled_img = fill_4corners(clip_grabcut, top_offset = 0, bottom_offset = 0, color, 10);
	show_image(filled_img, "filled 4 corners", True);

	//extract target
	Mat target = extract_target(filled_img);
	show_image(target, "target", True);

	Rect best_match_rect = mask_matches_target(processed_mask, target);

	clone_clip_grabcut = fill_rect_bound(clone_clip_grabcut, best_match_rect);
	show_image(clone_clip_grabcut, "clone clip grabcut", True);

	clone_clip_grabcut = fill_4corners(clone_clip_grabcut, 0, 0, cvScalar(255, 255, 255));
	show_image(clone_clip_grabcut, "clone clip grabcut 2", True);

	Mat target2 = extract_target(clone_clip_grabcut);
	show_image(target2, "target2", True);

	Mat clip_target2 = clip_target(target2, best_match_rect);
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
	for (int i = 0; i < size; i++){
		Rect rect = rects[i];
		template_rects.push_back(rect);
		x = rect.x, y = rect.y, w = rect.width, h = rect.height;
		//Mat roi = Mat::zeros(h, w, clip_target2.type());

		int sum = 0;
		for (int j = 0; j < h; j++){
			for (int k = 0; k < w; k++)
			{
				if (clip_target2.ptr(j + y)[k + x] > 0){
					sum += 1;
				}
			}
		}
		if (sum > 0){
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
		else{
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
	putText(image, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(0, 0, 255), 2);
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
		rect = matched_rects[i];
		draw_rect(draw_image, bx + rect.x,
			by + rect.y, rect.width, rect.height,
			normal_color, line_width);
	}
	//miss
	size = miss_rects.size();
	for (int i = 0; i < size; i++){
		rect = miss_rects[i];
		draw_rect(draw_image, bx + rect.x,
			by + rect.y, rect.width, rect.height,
			miss_color, line_width);
	}
	//abnormal
	size = abnormal_rects.size();
	for (int i = 0; i < size; i++){
		rect = abnormal_rects[i];
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
		rect = matched_rects[i];
		draw_rect(image, bx + rect.x + tx + left_offset,
			by + rect.y + ty + top_offset, rect.width, rect.height,
			normal_color, line_width);
	}
	//miss
	size = miss_rects.size();
	for (int i = 0; i < size; i++){
		rect = miss_rects[i];
		draw_rect(image, bx + rect.x + tx + left_offset,
			by + rect.y + ty + top_offset, rect.width, rect.height,
			miss_color, line_width);
	}
	//abnormal
	size = abnormal_rects.size();
	for (int i = 0; i < size; i++){
		rect = abnormal_rects[i];
		draw_rect(image, bx + rect.x + tx + left_offset,
			by + rect.y + ty + top_offset, rect.width, rect.height,
			abnormal_color, line_width);
	}
	draw_decision_text(image, decision);
}

bool run(string infile, string outfile="", string mask = "")
{
	int left_offset = -5,
		right_offset = 5,
		top_offset = -5,
		bottom_offset = 5;

	if (mask == "")
	{
		mask = _Globals.mask_file;
	}
	
	//TODO
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

int _bookbingDetector(String filename)
{
	int ret = -1;
	ret = run(filename);
	//try
	//{
	//	ret = run(filename);
	//}
	//catch (Exception e)
	//{
	//	std::cout << e.msg;
	//	;
	//}
	return ret;
}

//vector<int> bookbingDetector()
//{
//	vector<string> files;
//	get_files(INPUT_FOLDER, files);
//
//	int count = files.size();
//	vector<int> result;
//	for (int i = 0; i < count; i++)
//	{
//		result.push_back(bookbingDetector_file(files.at(i)));
//	}
//	return result;
//}
//
//int bookbingDetector_file(String filename)
//{
//	return _bookbingDetector(filename, _Globals, _Thresholds);
//}