// コマンドライン引数のパーサー（pythonのargparse風）
// Keita Yamada
// 2015.08.29

#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

enum ARG_TYPE
{
	INT, FLOAT, STRING, BOOL
};

class ArgParser
{
private:

	struct Argument
	{
		ARG_TYPE type;
		std::string arg_name;
		std::string value_name;
		std::string description;
		bool parsed;
		bool bool_default;
	};

	std::vector<Argument> requests;
	static std::map<std::string, int> int_values;
	static std::map<std::string, float> float_values;
	static std::map<std::string, std::string> string_values;
	static std::map<std::string, bool> bool_values;

	int argc;
	char** argv;

public:

	ArgParser(int argc, char** argv);
	void request(ARG_TYPE type, std::string arg_name, std::string value_name, std::string description, float default);
	void request(ARG_TYPE type, std::string arg_name, std::string value_name, std::string description, std::string default);
	void parse();
	void print_all();

	struct get_value{
	private:
		std::string key;
	public:
		operator int(){ return int_values[this->key]; }
		operator float(){ return float_values[this->key]; }
		operator std::string(){ return string_values[this->key]; }
		operator bool(){ return bool_values[this->key]; }
		get_value& operator()(std::string key){ this->key = key; return *this; }
	} get;
};
