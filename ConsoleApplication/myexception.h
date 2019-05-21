#pragma once

#include <iostream>
#include <sstream>
#include <exception>
using namespace std;

class MyException : public exception
{
public:
	MyException(char const* funcname, int lineno, char const* message="") :
		_lineno(lineno), _funcname(funcname), _message(message) {
		stringstream ss;
		ss << "Exception: " << _funcname << " in line " << _lineno << ". " << _message << "\n";
		_exception = ss.str();
	}

	const char* what() const
	{
		return _exception.c_str();
	}
private:
	string _funcname;
	int _lineno;
	string _message;
	string _exception;
};