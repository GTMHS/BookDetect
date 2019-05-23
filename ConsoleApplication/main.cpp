#include <Windows.h>
#include <tchar.h>
#include <sstream>
#include <string>
#include <atlstr.h>
#include "BookDetecting.h"

extern Globals _Globals;

std::wstring string2tchar(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

void run_exe(string path, string params)
{
	std::wstring t_path = string2tchar(path);
	std::wstring t_params = string2tchar(params);

	SHELLEXECUTEINFO ShellInfo;

	memset(&ShellInfo, 0, sizeof(ShellInfo));

	ShellInfo.cbSize = sizeof(ShellInfo);

	ShellInfo.hwnd = NULL;

	ShellInfo.lpVerb = _T("open");

	ShellInfo.lpFile = t_path.c_str();
	ShellInfo.lpParameters = t_params.c_str();

	ShellInfo.nShow = SW_SHOWNORMAL;

	ShellInfo.fMask = SEE_MASK_NOCLOSEPROCESS;

	BOOL bResult = ShellExecuteEx(&ShellInfo);
}

void run_shower()
{
	string filename = "shower.exe";
	// 默认图像路径， 刷新频率
	string params = "";
	
	run_exe(filename, params);
}

void run_database(string time, string result)
{
	// database.exe 2016-01-22 08:45:50 异常
	string filename = "database.exe";
	//string params = "2016-01-22 08:45:50 异常"
	std::stringstream ss;
	ss << " " << time << " " << result;
	std::string params = ss.str();
	run_exe(filename, params);
}

void getAllFiles(string path, vector<string> &files)
{
	//文件句柄 
	long  hFile = 0;
	//文件信息 
	struct _finddata_t fileinfo; //很少用的文件信息读取结构
	string p; //string类很有意思的一个赋值函数:assign()，有很多重载版本
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR)) //判断是否为文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(p.assign(path).append("/").append(fileinfo.name));//保存文件夹名字
					getAllFiles(p.assign(path).append("/").append(fileinfo.name), files);//递归当前文件夹
				}
			}
			else  //文件处理
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));//文件名
			}
		} while (_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
}

int main(int argc, char* argv[])
{
	//忽略opencv的异常输出
	//必须在程序的入口添加下面两行
	cvSetErrMode(2);
	cvRedirectError(cvErrorRedirector);
	//run_shower();
	//run_database("2016-01-22 08:45:50", "正常");
	//
	string config_file = "conf.ini";
	read_thresholds(config_file);
	string imgs_dir = "imgs";
	//imgs_dir = "E:\\My Projects\\Python\\bookdetecting\\09-27\\往事";
	imgs_dir = "wrong";
	//imgs_dir = "G:\\20181203\\ConsoleApplication\\ConsoleApplication\\12-06";
	vector<string> files;

	getAllFiles(imgs_dir, files);
	cout << files[0] << endl << files.size() << endl;
	flush(std::cout);
	//输入文件名
	string filename = "Pic_2018_09_27_121823_blockId#25593.bmp";

	//使用默认全局配置
	default_globals();
	//使用默认全局阈值
	//default_thresholds();
	//mask
	string mask_file = "moban.png";

	//outfile,""表示不输出
	string outfile = "";
	//string outfile = "out.jpg";

	const int count = files.size();
	for (int i = 0; i < count; i++) {
		filename = files[i];
		//filename = "wrong/2019-05-23-15-42-37_266.jpg";
		//outfile = filename + "_ret.jpg";
		//计时结果
		double timems = 0;
		//运行检测算法
		int ret = bookDetecting(filename, mask_file, outfile, True, True, &timems);
		//打印结果
		std::cout << filename << " : ";// << ret << std::endl;
		switch (ret)
		{
		case 1:
			std::cout << "Abnormal" << std::endl;
			break;
		case 0:
			std::cout << "Normal" << std::endl;
			break;
		case -1:
			std::cout << "Exception" << std::endl;
			break;
		default:
			break;
		}
		//打印耗时
		std::cout << "Used time: " << timems << " ms" << std::endl;
	}
	system("pause");

	return 0;
}
