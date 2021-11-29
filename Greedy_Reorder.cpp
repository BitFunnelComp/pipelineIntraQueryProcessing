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
#include<limits.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h> 
#include<string.h>
#include <omp.h>
#include<unordered_map>
#include<set>
#include"LRU_Cache_Batch.h"
using namespace std;
struct QueryInfo
{
	unsigned oldqueryid;
	int64_t score;
	int64_t qlength;
};


vector<vector<unsigned>>queries;
vector<QueryInfo>queryinfo;
vector<unsigned>queryOrder;
LRUCache global_LRUCache;

unsigned QSIZE = 100000;
unsigned batchSize = 10000;

void read_query(string filename, unsigned turn)
{
	queries.clear();
	ifstream fin(filename);
	string str = "";
	unsigned startqid = turn*batchSize, endqid = (turn + 1)*batchSize;
	unsigned qid = 0;
	while (getline(fin, str) && qid<endqid)
	{
		if (qid >= startqid)
		{
			istringstream sin(str);
			string field = "";
			vector<unsigned> tmpq;
			while (getline(sin, field, '\t'))
				tmpq.push_back(atoi(field.c_str()));
			queries.push_back(tmpq);
		}
		qid++;
	}
}
void read_List_Length(string filename)
{
	FILE *lengthfile = fopen((filename + ".list-l").c_str(), "rb");
	unsigned tmplength = 0;
	while (fread(&tmplength, sizeof(unsigned), 1, lengthfile))
	{
		listLength.push_back(tmplength);
	}
	fclose(lengthfile);
	int64_t sumsize = 0;
	for (auto i : listLength)
		sumsize += i;
}
void LRUGet_Query(unsigned qid)
{
	for (unsigned i = 0; i < queries[qid].size(); i++)
		global_LRUCache.Get(queries[qid][i]);
}
int64_t LRUGet_Score(unsigned qid)
{
	vector<unsigned>terms = queries[qid];
	std::sort(terms.begin(), terms.end());
	terms.erase(std::unique(terms.begin(), terms.end()), terms.end());

	int64_t misssize = 0;;
	for (unsigned i = 0; i < terms.size(); i++)
	{
		if (global_LRUCache.hashmap_.find(terms[i]) == global_LRUCache.hashmap_.end())
		{
			misssize += listLength[terms[i]];
		}
	}
	return misssize;
}
bool cmpScore(QueryInfo const&a, QueryInfo const&b)
{
	if ((a.score < b.score) || ((a.score == b.score) && (a.qlength < b.qlength)))
		return true;
	return false;
}
bool cmpqlength(QueryInfo const&a, QueryInfo const&b)
{
	return a.qlength>b.qlength;
}
void greedyLongtail()
{
	sort(queryinfo.begin(), queryinfo.end(), cmpqlength);
	unsigned pos = 0.05*(double)queries.size(); 
	for (unsigned t = 0; t < pos; t++)
	{
		for (unsigned i = 0; i<pos - t; i++)
		{
			queryinfo[i].score = LRUGet_Score(queryinfo[i].oldqueryid);
		}
		sort(queryinfo.begin(), queryinfo.begin() + pos - t, cmpScore);
		unsigned qid = queryinfo[0].oldqueryid;
		LRUGet_Query(qid);
		queryOrder.push_back(qid);
		queryinfo.erase(queryinfo.begin());
	}
}
void greedyReorder()
{
	greedyLongtail();
	unsigned pos = queryinfo.size(); 
	for (unsigned t = 0; t < pos; t++)
	{
		for (unsigned i = 0; i<queryinfo.size(); i++)
		{
			queryinfo[i].score = LRUGet_Score(queryinfo[i].oldqueryid);
		}
		sort(queryinfo.begin(), queryinfo.end(), cmpScore);
		unsigned qid = queryinfo[0].oldqueryid;
		LRUGet_Query(qid);
		queryOrder.push_back(qid);
		queryinfo.erase(queryinfo.begin());
	}
	cout << "After reorder queryOrdersize=" << queryOrder.size() << endl;
	cout << "We eistimate the miss size=" << global_LRUCache.miss_size << endl;
}
void writeQuery(string filename)
{
	ofstream fout(filename, ios::app);
	for (unsigned i = 0; i < queryOrder.size(); i++)
	{
		unsigned qid = queryOrder[i];
		for (unsigned j = 0; j < queries[qid].size() - 1; j++)
		{
			fout << queries[qid][j] << "\t";
		}
		fout << queries[qid][queries[qid].size() - 1] << endl;
	}
	fout.close();
}
void initData()
{
	read_List_Length("");
}
int64_t cal_qlength(unsigned qid)
{
	vector<unsigned>terms = queries[qid];
	std::sort(terms.begin(), terms.end());
	terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
	int64_t length = 0;
	for (auto tid : terms)
		length += listLength[tid];
	return length;
}
bool cmpListLength(QueryInfo const&a, QueryInfo const&b)
{
	return a.score > b.score;
}
void cacheWarmup()
{
	vector<QueryInfo>listInfo(listLength.size());
	for (unsigned i = 0; i < listLength.size(); i++)
	{
		listInfo[i].oldqueryid = i;
		listInfo[i].score = listLength[i];
	}
	sort(listInfo.begin(), listInfo.end(), cmpListLength);
	unsigned warmuplrucount = 0;
	int64_t tmpsize = 0;
	for (unsigned i = 0; i < listLength.size() && tmpsize<CACHE_SIZE; i++, warmuplrucount++)
		tmpsize += listLength[listInfo[i].oldqueryid];
	for (unsigned i = 0; i < warmuplrucount; i++)
		global_LRUCache.Get(listInfo[i].oldqueryid);
}
void batchProcess()
{
	unsigned turn = ceil((double)QSIZE / (double)batchSize);
	global_LRUCache.cacheClear(); cacheWarmup();
	for (unsigned i = 0; i < turn; i++)
	{
		cout << "***********************turn=" << i << "*****************************" << endl;
		read_query("", i);
		queryinfo.clear(); queryOrder.clear();
		queryinfo.resize(queries.size());
		for (unsigned i = 0; i < queryinfo.size(); i++)
		{
			queryinfo[i].oldqueryid = i;
			queryinfo[i].qlength = cal_qlength(i);
		}
		greedyReorder();
		writeQuery("");
	}
}
int main(int argc, char *argv[])
{
	CACHE_SIZE *= atoi(argv[1]);
	batchSize = atoi(argv[2]);
	cout << "Cache size=" << CACHE_SIZE << endl;
	initData();
	cout << "Initdata over" << endl;
	batchProcess();
	cout << "Greedy Reorder Over" << endl;
}
