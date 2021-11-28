#include<iostream>
#include<vector>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include <sstream>  
#include <string> 
#include <algorithm>
#include<numeric>
#include<sys/time.h>
#include<limits.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h> 
#include<string.h>
#include <sys/stat.h>
#include <omp.h>
#include<unordered_map>
#include<stdio.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<assert.h>
#include<unistd.h>
#include<stdlib.h>
#include<errno.h>
#include<sys/types.h>
#include<fcntl.h>
#include<unordered_set>
#include<queue>
using namespace std;
vector<vector<float>>queryTimes;//query-pall time
float threshold = 1000;
vector<unsigned>transForm = { 1, 2, 4, 8 };
vector<unsigned>Pall;

void readTimes(string filepath)
{
	queryTimes.resize(79542);//////////////
	for (unsigned i = 0; i < queryTimes.size(); i++)
		queryTimes[i].resize(transForm.size());

	for (unsigned i = 0; i < transForm.size(); i++)
	{
		string filename = filepath + to_string(transForm[i]) + "_timeT.txt";
		ifstream fin(filename);
		for (unsigned j = 0; j < queryTimes.size(); j++)
			fin >> queryTimes[j][i];
		fin.close();
	}
}

void cal_Pall()
{
	Pall.resize(queryTimes.size());
	for (unsigned i = 0; i < queryTimes.size(); i++)
	{
		Pall[i] = 8;
		for (unsigned j = 0; j < transForm.size(); j++)
		{
			if (queryTimes[i][j] <= threshold)
			{
				Pall[i] = transForm[j];
				break;
			}
		}
	}
}
void writePall(string filename)
{
	ofstream fout(filename);
	for (unsigned i = 0; i < Pall.size(); i++)
		fout << Pall[i] << endl;
	fout.close();
}
int main(int argc, char *argv[])
{
	threshold *= atof(argv[1]);
	cout << "thresh=" << threshold << endl;
	readTimes("/home/lxy/NVM_code/RawData/ClueWeb/Feature/Regression/Label/ClueWeb_Test_BMW_");
	cal_Pall();
	writePall("/home/lxy/NVM_code/RawData/ClueWeb/Feature/Regression/Label/ClueWebPall.txt");
}

