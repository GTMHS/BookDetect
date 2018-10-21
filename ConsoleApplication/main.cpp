#include "BookDetecting.h"

extern Globals _Globals;

int main(int argc, char* argv[])
{
	//输入文件名
	string filename = "Pic_2018_09_27_121823_blockId#25593.bmp";

	//使用默认全局配置
	default_globals();
	//使用默认全局阈值
	default_thresholds();
	//mask
	string mask_file = "ws.png";

	//outfile,""表示不输出
	string outfile = "";
	//string outfile = "out.jpg";

	//计时结果
	double timems = 0;

	//运行检测算法
	int ret = bookDetecting(filename, mask_file, outfile, True, True, &timems);
	//打印结果
	std::cout << filename << " : " << ret << std::endl;
	//打印耗时
	std::cout << "Used time: " << timems << " ms" << std::endl;
	
	system("pause");
	return 0;
}