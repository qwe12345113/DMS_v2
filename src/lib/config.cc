// File: config.cc
// Date: Sat May 04 22:22:20 2013 +0800
// Author: Yuxin Wu

#include "config.hh"
#include "debugutils.hh"
#include "utils.hh"
using namespace std;

namespace config
{

	// TODO allow different types for a value. using template
	ConfigParser::ConfigParser(const char *fname)
	{
		if (!exists_file(fname))
			error_exit("Cannot find config file!");
		const static size_t BUFSIZE = 4096; // TODO overflow
		ifstream fin(fname);
		string s;
		s.resize(BUFSIZE);
		float val;
		while (fin >> s)
		{
			if (s[0] == '#')
			{
				fin.getline(&s[0], BUFSIZE, '\n');
				continue;
			}
			fin >> val;
			data[s] = val;
			fin.getline(&s[0], BUFSIZE, '\n');
		}
	}

	float ConfigParser::get(const std::string &s)
	{
		if (data.count(s) == 0)
			error_exit(ssprintf("Option %s not found in config file!\n", s.c_str()));
		return data[s];
	}

	extern float EYE_AR_THRESH;
	// extern float MOUTH_AR_THRESH;
	// extern int HEAD_X_THRESH;
	// extern int HEAD_Y_THRESH;

	// extern int EYE_AR_CONSEC_FRAME;
	// extern int EYE_AR_SLEEP_FRAME;
	// extern int LOWER_HEAD_FRAME;
	// extern int TURN_AROUND_FRAME;
}
