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
	// Ĭ��ͼ��·���� ˢ��Ƶ��
	string params = "";
	
	run_exe(filename, params);
}

void run_database(string time, string result)
{
	// database.exe 2016-01-22 08:45:50 �쳣
	string filename = "database.exe";
	//string params = "2016-01-22 08:45:50 �쳣"
	std::stringstream ss;
	ss << " " << time << " " << result;
	std::string params = ss.str();
	run_exe(filename, params);
}

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
	//run_shower();
	//run_database("2016-01-22 08:45:50", "����");
	//
	string config_file = "conf.ini";
	read_thresholds(config_file);
	string imgs_dir = "imgs";
	//imgs_dir = "E:\\My Projects\\Python\\bookdetecting\\09-27\\����";
	imgs_dir = "wrong";
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
	//default_thresholds();
	//mask
	string mask_file = "moban.png";

	//outfile,""��ʾ�����
	string outfile = "";
	//string outfile = "out.jpg";

	const int count = files.size();
	for (int i = 0; i < count; i++) {
		filename = files[i];
		//filename = "wrong/2019-05-23-15-42-37_266.jpg";
		//outfile = filename + "_ret.jpg";
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
