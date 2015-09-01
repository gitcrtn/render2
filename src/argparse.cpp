// コマンドライン引数のパーサー（pythonのargparse風）
// Keita Yamada
// 2015.08.29

#include "argparse.h"

ArgParser::ArgParser(int argc, char** argv) :argc(argc), argv(argv){}

void ArgParser::request(ARG_TYPE type, std::string arg_name, std::string value_name, std::string description, float default)
{
	switch (type)
	{

	case ARG_TYPE::INT:
		this->int_values.insert(std::map<std::string, int>::value_type(value_name, static_cast<int>(default)));
		break;

	case ARG_TYPE::FLOAT:
		this->float_values.insert(std::map<std::string, float>::value_type(value_name, default));
		break;

	case ARG_TYPE::BOOL:
		this->bool_values.insert(std::map<std::string, bool>::value_type(value_name, static_cast<bool>(default)));
		break;

	default:
		break;

	}

	this->requests.push_back({ type, arg_name, value_name, description, false, false });
}

void ArgParser::request(ARG_TYPE type, std::string arg_name, std::string value_name, std::string description, std::string default)
{
	this->string_values.insert(std::map<std::string, std::string>::value_type(value_name, default));
	this->requests.push_back({ type, arg_name, value_name, description, false, false });
}

void ArgParser::parse()
{
	int i = 1;
	bool success = true;

	while (true)
	{
		if (i >= argc) break;

		for (Argument& a : this->requests)
		{
			if (std::string(argv[i]).compare(a.arg_name) == 0)
			{
				switch (a.type)
				{

				case ARG_TYPE::INT:
					i++;
					this->int_values[a.value_name] = atoi(argv[i]);

				case ARG_TYPE::FLOAT:
					i++;
					this->float_values[a.value_name] = atof(argv[i]);

				case ARG_TYPE::STRING:
					i++;
					this->string_values[a.value_name] = std::string(argv[i]);

				case ARG_TYPE::BOOL:
					this->bool_values[a.value_name] = true;
					break;

				default:
					break;

				}

				a.parsed = true;
				i++;
				continue;
			}
		}
	}
}

void ArgParser::print_all()
{
	for (Argument& a : this->requests)
	{
		switch (a.type)
		{

		case ARG_TYPE::INT:
			std::cout << a.value_name << ": " << this->int_values[a.value_name] << " (" << a.description << ")" << std::endl;
			break;

		case ARG_TYPE::FLOAT:
			std::cout << a.value_name << ": " << this->float_values[a.value_name] << " (" << a.description << ")" << std::endl;
			break;

		case ARG_TYPE::STRING:
			std::cout << a.value_name << ": " << this->string_values[a.value_name] << " (" << a.description << ")" << std::endl;
			break;

		case ARG_TYPE::BOOL:
			if (this->bool_values[a.value_name]) std::cout << a.value_name << ": true (" << a.description << ")" << std::endl;
			else std::cout << a.value_name << ": false (" << a.description << ")" << std::endl;
			break;

		default:
			break;

		}
	}
}

std::map<std::string, int> ArgParser::int_values;
std::map<std::string, float> ArgParser::float_values;
std::map<std::string, std::string> ArgParser::string_values;
std::map<std::string, bool> ArgParser::bool_values;