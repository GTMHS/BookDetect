#include "BookDetecting.h"

extern Globals _Globals;

int main(int argc, char* argv[])
{
	//�����ļ���
	string filename = "Pic_2018_09_27_121823_blockId#25593.bmp";

	//ʹ��Ĭ��ȫ������
	default_globals();
	//ʹ��Ĭ��ȫ����ֵ
	default_thresholds();
	//mask
	string mask_file = "ws.png";

	//outfile,""��ʾ�����
	string outfile = "";
	//string outfile = "out.jpg";

	//��ʱ���
	double timems = 0;

	//���м���㷨
	int ret = bookDetecting(filename, mask_file, outfile, True, True, &timems);
	//��ӡ���
	std::cout << filename << " : " << ret << std::endl;
	//��ӡ��ʱ
	std::cout << "Used time: " << timems << " ms" << std::endl;
	
	system("pause");
	return 0;
}