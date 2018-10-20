#include "BookbindingDetector.h"

int main(int argc, char* argv[])
{
	std::cout << "  Usage 1: ConsoleApplication.exe" << std::endl;
	std::cout << "  Usage 2: ConsoleApplication.exe filename" << std::endl;
	std::cout << "  Usage 3: ConsoleApplication.exe filename "
		"save_result show_process show_result insert_pause predict_file "
		"threshold1 threshold2 threshold3 threshold4 threshold5 threshold6" << std::endl;
	std::cout << "  Example 1: ConsoleApplication.exe demo.jpg "
		"true true true false result.csv "
		"5 2.0 20 100 10 10" << std::endl;
	std::cout << "  Example 2: ConsoleApplication.exe demo.jpg "
		"true true true false result.csv "
		"default default default default default default" << std::endl;
	
	/*Globals global;
	global.save_result = true;
	global.show_process = true;
	global.show_result = true;
	global.insert_pause = false;
	global.predict_file = "result.csv";

	Thresholds thresholds;
	thresholds.threshold1 = 5;
	thresholds.threshold2 = 2.0;
	thresholds.threshold3 = 20;
	thresholds.threshold4 = 100;
	thresholds.threshold5 = 10;
	thresholds.threshold6 = 10;*/

	if (argc == 1)
	{
		//必须首先使用下面的方式初始化参数
		default_globals();
		default_thresholds();
		vector<int> result = bookbingDetector();
		int size = result.size();
		for (int i = 0; i < size; i++)
		{
			std::cout << result.at(i) << std::endl;
		}
	}
	else if (argc == 2)
	{
		String filename = String(argv[1]);
		//必须首先使用下面的方式初始化参数
		default_globals();
		default_thresholds();
		std::cout << bookbingDetector_file(filename);
	}

	if (argc == 13)
	{
		std::cout << "Error parameters." << std::endl;
		String filename = String(argv[1]);

		//通过下面的方式修改参数
		_Globals.save_result = strcmp(argv[2], "true") == 0 ? true : false;
		_Globals.show_process = strcmp(argv[3], "true") == 0 ? true : false;
		_Globals.show_result = strcmp(argv[4], "true") == 0 ? true : false;
		_Globals.insert_pause = strcmp(argv[5], "true") == 0 ? true : false;
		_Globals.predict_file = String(argv[6]);

		_Thresholds.threshold1 = strcmp(argv[7], "default") == 0 ? 5 : atof(argv[7]);
		_Thresholds.threshold2 = strcmp(argv[8], "default") == 0 ? 2.0 : atof(argv[8]);
		_Thresholds.threshold3 = strcmp(argv[9], "default") == 0 ? 20 : atof(argv[9]);
		_Thresholds.threshold4 = strcmp(argv[10], "default") == 0 ? 100 : atof(argv[10]);
		_Thresholds.threshold5 = strcmp(argv[11], "default") == 0 ? 10 : atof(argv[11]);
		_Thresholds.threshold6 = strcmp(argv[12], "default") == 0 ? 10 : atof(argv[12]);
		
		bookbingDetector_file(filename);
	}

	return 0;
}