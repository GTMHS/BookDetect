#include "BookDetecting.h"

extern Globals _Globals;

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
	//

	string imgs_dir = "G:/20181203/imgs";
	imgs_dir = "E:\\My Projects\\Python\\bookdetecting\\09-27\\往事";
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
	default_thresholds();
	//mask
	string mask_file = "G:\\20181203\\ConsoleApplication\\ConsoleApplication\\ws.png";

	//outfile,""表示不输出
	string outfile = "";
	//string outfile = "out.jpg";

	const int count = files.size();
	for (int i = 0; i < count; i++) {
		filename = files[i];
		outfile = filename + "_ret.jpg";
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