#include "BookDetecting.h"

extern Globals _Globals;

void getAllFiles(string path, vector<string> &files)
{
	//�ļ���� 
	long  hFile = 0;
	//�ļ���Ϣ 
	struct _finddata_t fileinfo; //�����õ��ļ���Ϣ��ȡ�ṹ
	string p; //string�������˼��һ����ֵ����:assign()���кܶ����ذ汾
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR)) //�ж��Ƿ�Ϊ�ļ���
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(p.assign(path).append("/").append(fileinfo.name));//�����ļ�������
					getAllFiles(p.assign(path).append("/").append(fileinfo.name), files);//�ݹ鵱ǰ�ļ���
				}
			}
			else  //�ļ�����
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));//�ļ���
			}
		} while (_findnext(hFile, &fileinfo) == 0); //Ѱ����һ�����ɹ�����0������-1
		_findclose(hFile);
	}
}

int main(int argc, char* argv[])
{
	//����opencv���쳣���
	//�����ڳ������������������
	cvSetErrMode(2);
	cvRedirectError(cvErrorRedirector);
	//

	string imgs_dir = "G:/20181203/imgs";
	imgs_dir = "E:\\My Projects\\Python\\bookdetecting\\09-27\\����";
	//imgs_dir = "G:\\20181203\\ConsoleApplication\\ConsoleApplication\\12-06";
	vector<string> files;
	getAllFiles(imgs_dir, files);
	cout << files[0] << endl << files.size() << endl;
	flush(std::cout);
	//�����ļ���
	string filename = "Pic_2018_09_27_121823_blockId#25593.bmp";

	//ʹ��Ĭ��ȫ������
	default_globals();
	//ʹ��Ĭ��ȫ����ֵ
	default_thresholds();
	//mask
	string mask_file = "G:\\20181203\\ConsoleApplication\\ConsoleApplication\\ws.png";

	//outfile,""��ʾ�����
	string outfile = "";
	//string outfile = "out.jpg";

	const int count = files.size();
	for (int i = 0; i < count; i++) {
		filename = files[i];
		outfile = filename + "_ret.jpg";
		//��ʱ���
		double timems = 0;
		//���м���㷨
		int ret = bookDetecting(filename, mask_file, outfile, True, True, &timems);
		//��ӡ���
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
		//��ӡ��ʱ
		std::cout << "Used time: " << timems << " ms" << std::endl;
	}
	system("pause");

	return 0;
}