// File: config.hh
// Date: Sat May 04 22:22:25 2013 +0800
// Author: Yuxin Wu

#pragma once
#define _USE_MATH_DEFINES
#include <cmath>

#include <map>
#include <cstring>
#include <fstream>

namespace config
{

	class ConfigParser
	{
	public:
		std::map<std::string, float> data;

		ConfigParser(const char *fname);

		float get(const std::string &s);
	};

	extern float EYE_AR_THRESH;
	// extern float MOUTH_AR_THRESH;
	// extern int HEAD_X_THRESH;
	// extern int HEAD_Y_THRESH;

	// extern int EYE_AR_CONSEC_FRAME;
	// extern int EYE_AR_SLEEP_FRAME;
	// extern int LOWER_HEAD_FRAME;
	// extern int TURN_AROUND_FRAME;

}
