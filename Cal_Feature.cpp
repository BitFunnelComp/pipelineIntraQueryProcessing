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
#include<queue>

#include"bm25.hpp"
#include"Mycodec.hpp"
using namespace std;

const unsigned constShardcount = 8;

vector<uint64_t>Block_Start;
vector<uint32_t>Block_Docid;
vector<float>Block_Max_Term_Weight;

typedef uint32_t term_id_type;
typedef std::vector<term_id_type> term_id_vec;
vector<term_id_vec> queries;

vector<float>termTopkThresh;
vector<vector<float>>TermTopkThreshShard;
const unsigned topK = 10;
vector<unsigned>ListLength;
unsigned num_docs;

struct SplitInfo
{
	unsigned split_blockid;
	SplitInfo(){ split_blockid = 0; }
};
vector<vector<SplitInfo>>splitInfo(constShardcount);

struct QueryFeature
{
	//query信息
	float term_count;
	float query_topK_score;
	float query_sample_time;

	//term信息（sum,max,var）
	//posting个数
	float sum_no_positings;
	float max_no_positings;
	float var_no_positings;

	//每个term的块最大分数 最大值
	float sum_max_score_term;
	float max_max_score_term;
	float var_max_score_term;

	//每个term的块最大分数 均值
	float sum_avg_score_term;
	float max_avg_score_term;
	float var_avg_score_term;

	//每个term的块最大分数 topk为每条链的topk
	float sum_topk_score_term;
	float max_topk_score_term;
	float var_topk_score_term;

	//大于块最大分数均值的块数
	float sum_morethan_avg_score;
	float max_morethan_avg_score;
	float var_morethan_avg_score;

	//大于块最大分数topk的块数 topk分数为查询的topk分数
	float sum_morethan_topk_score;
	float max_morethan_topk_score;
	float var_morethan_topk_score;

	//块最大值均方误差
	float sum_var_max_score_term;
	float max_var_max_score_term;
	float var_var_max_score_term;

	//块df倒数
	float sum_inverse_df;
	float max_inverse_df;
	float var_inverse_df;

	//大于最大分数95%的posting数量
	float sum_morethan_maxscore5_score;
	float max_morethan_maxscore5_score;
	float var_morethan_maxscore5_score;

	//大于链中topk分数95%的posting数量
	float sum_morethan_topkscore5_score;
	float max_morethan_topkscore5_score;
	float var_morethan_topkscore5_score;

	//shard级别
	//每个shard中topk阈值的均方差 topk为每个shard的topk
	float sum_var_topk_score_shard;
	float max_var_topk_score_shard;
	float var_var_topk_score_shard;

	//每个shard中avg分数的均方差
	float sum_var_avg_score_shard;
	float max_var_avg_score_shard;
	float var_var_avg_score_shard;

	//每个shard中大于topk块数的均方差 topk分数为查询的topk分数
	float sum_var_morethan_topk_score_shard;
	float max_var_morethan_topk_score_shard;
	float var_var_morethan_topk_score_shard;

	//每个shard中大于avg块数的均方差
	float sum_var_morethan_avg_score_shard;
	float max_var_morethan_avg_score_shard;
	float var_var_morethan_avg_score_shard;

	//每个shard中最大分数的均方差
	float sum_var_max_score_shard;
	float max_var_max_score_shard;
	float var_var_max_score_shard;

	//每个shard中大于最大分数95%的posting数量
	float sum_var_morethan_maxscore5_score_shard;
	float max_var_morethan_maxscore5_score_shard;
	float var_var_morethan_maxscore5_score_shard;

	//每个shard中大于链topk分数95%的posting数量
	float sum_var_morethan_topkscore5_score_shard;
	float max_var_morethan_topkscore5_score_shard;
	float var_var_morethan_topkscore5_score_shard;

};
vector<QueryFeature>queryFeatures;

vector<int64_t>Head_offset;
vector<uint8_t>Head_Data;
class TermInfo
{
public:
	float df = 0;
	float topkscore = 0;
	float maxscore = 0;
	TermInfo(){ df = 0; topkscore = 0; maxscore = 0; }
};
vector<TermInfo>termInfo;
vector<vector<float>>preCalScore;
vector<vector<unsigned>>docList;

void read_Head_Length(string filename)
{
	FILE *lengthfile = fopen((filename + ".head-l").c_str(), "rb");
	unsigned tmplength = 0;
	int64_t prelength = 0;
	fread(&num_docs, sizeof(unsigned), 1, lengthfile);
	cout << "num_docs=" << num_docs << endl;
	while (fread(&tmplength, sizeof(unsigned), 1, lengthfile))
	{
		Head_offset.push_back(prelength);
		prelength += tmplength;
	}
	Head_offset.push_back(prelength);
	cout << "All we have " << Head_offset.size() - 1 << " heads" << endl;
	fclose(lengthfile);
}
void read_Head_Data(string filename)
{
	filename = filename + ".head";
	FILE *file = fopen(filename.c_str(), "rb");
	cout << "Head Data has " << Head_offset[Head_offset.size() - 1] << " Bytes" << endl;
	int64_t length = Head_offset[Head_offset.size() - 1];
	Head_Data.resize(length);
	fread(Head_Data.data(), sizeof(uint8_t), length, file);
	fclose(file);
}
void read_list()
{
	FILE *file = fopen("xxx.docs", "rb");
	unsigned length = 0, docid = 0;
	fread(&length, sizeof(unsigned), 1, file);
	unsigned maxdocID = 0;
	fread(&maxdocID, sizeof(unsigned), 1, file);
	docList.resize(Head_offset.size() - 1);
	for (unsigned i = 0; i < docList.size(); i++)
	{
		fread(&length, sizeof(unsigned), 1, file);
		docList[i].resize(length);
		fread(docList[i].data(), sizeof(unsigned), length, file);
	}

	cout << "Max docID=" << maxdocID << endl;
}
void read_ListDF()
{
	ListLength.resize(Head_offset.size() - 1);
	for (unsigned i = 0; i < Head_offset.size() - 1; i++)
	{
		uint8_t const* headdata = Head_Data.data() + Head_offset[i];
		unsigned m_n = 0;
		TightVariableByte::decode(headdata, &m_n, 1);
		ListLength[i] = m_n;
		if (m_n != docList[i].size())cout << "m_n wrong" << endl;
	}
}
void read_BlockWand_Data(string filename)
{
	FILE *file = fopen((filename + ".BMWwand").c_str(), "rb");
	uint64_t length = 0;
	fread(&length, sizeof(uint64_t), 1, file);
	fread(&length, sizeof(uint64_t), 1, file);
	Block_Start.resize(length); cout << "All we have " << length << " block_start" << endl;
	fread(Block_Start.data(), sizeof(uint64_t), length, file);
	fread(&length, sizeof(uint64_t), 1, file); cout << "All we have " << length << " blocks" << endl;
	Block_Max_Term_Weight.resize(length);
	fread(Block_Max_Term_Weight.data(), sizeof(float), length, file);
	fread(&length, sizeof(uint64_t), 1, file);
	Block_Docid.resize(length);
	fread(Block_Docid.data(), sizeof(uint32_t), length, file);
	fclose(file);
}
void read_termInfo(string filename)
{
	termInfo.resize(Head_offset.size() - 1);
	ifstream fin(filename);
	string str = "";
	for (unsigned i = 0; i < termInfo.size(); i++)
	{
		getline(fin, str);
		stringstream ss(str);
		string field = "";
		unsigned id = 0;
		while (getline(ss, field, ','))
		{
			if (id == 0)termInfo[i].df = atof(field.c_str());
			else if (id == 1)termInfo[i].topkscore = atof(field.c_str());
			else if (id == 2)termInfo[i].maxscore = atof(field.c_str());
			id++;
		}
	}
}
void read_scoreInfo(string filename)
{
	FILE *file = fopen((filename + ".score").c_str(), "rb");
	unsigned length = 0;
	preCalScore.resize(Head_offset.size() - 1);
	for (unsigned i = 0; i < preCalScore.size(); i++)
	{
		fread(&length, sizeof(unsigned), 1, file);
		preCalScore[i].resize(length);
		fread(preCalScore[i].data(), sizeof(float), length, file);
	}
	fclose(file);
}

typedef std::pair<uint64_t, uint64_t> term_freq_pair;
typedef std::vector<term_freq_pair> term_freq_vec;
term_freq_vec query_freqs(term_id_vec terms)
{
	term_freq_vec query_term_freqs;
	std::sort(terms.begin(), terms.end());
	for (size_t i = 0; i < terms.size(); ++i) {
		if (i == 0 || terms[i] != terms[i - 1]) {
			query_term_freqs.emplace_back(terms[i], 1);
		}
		else {
			query_term_freqs.back().second += 1;
		}
	}

	return query_term_freqs;
}

void read_SplitInfo(string filename)
{
	FILE *file = fopen((filename + ".splitinfo").c_str(), "rb");
	SplitInfo tmps;
	int64_t tmpoffset;
	for (unsigned i = 0; i < constShardcount; i++)
	{
		splitInfo[i].resize(ListLength.size());
		for (unsigned j = 0; j < ListLength.size(); j++)
		{
			fread(&tmps.split_blockid, sizeof(unsigned), 1, file);
			fread(&tmpoffset, sizeof(int64_t), 1, file);
			splitInfo[i][j] = tmps;
		}
	}
	fclose(file);
	cout << "read split over" << endl;
}
void read_query(string filename)
{
	queries.clear();
	ifstream fin(filename);
	string str = "";
	while (getline(fin, str))
	{
		istringstream sin(str);
		string field = "";
		term_id_vec tmpq;
		while (getline(sin, field, '\t'))
			tmpq.push_back(atoi(field.c_str()));
		queries.push_back(tmpq);
	}
	cout << "All we have " << queries.size() << " queries" << endl;
	queryFeatures.resize(queries.size());
}

void read_TermtopKThresh(string filename)
{
	ifstream fin(filename);
	string str = "";
	while (getline(fin, str))
	{
		float score = atof(str.c_str());
		termTopkThresh.push_back(score);
	}
	fin.close();
	for (int i = 0; i<queries.size(); i++)
	{
		float score = 0;
		auto terms = query_freqs(queries[i]);
		for (auto t : terms)
		{
			score = max(score, termTopkThresh[t.first] * t.second);
		}
		queryFeatures[i].query_topK_score = score;
		queryFeatures[i].term_count = terms.size();
	}
}
void read_TermTopkThreshShard(string filename)
{
	ifstream fin(filename);
	string str = "";
	TermTopkThreshShard.resize(ListLength.size());
	unsigned linecount = 0;
	while (getline(fin, str))
	{
		stringstream ss(str);
		string field = "";
		while (getline(ss, field, '\t'))
		{
			TermTopkThreshShard[linecount].push_back(atof(field.c_str()));
		}
		linecount++;
	}
	fin.close();
}

void init(string filename)
{
	read_Head_Length(filename);
	read_Head_Data(filename);
	read_list();
	read_ListDF();
	read_BlockWand_Data(filename);

	read_termInfo("");
	read_scoreInfo("");

	read_SplitInfo(filename);
	read_query("");
	read_TermtopKThresh("");
	read_TermTopkThreshShard("");
}

float Sum(vector<float>&v)
{
	float ret = 0;
	for (auto n : v)
		ret += n;
	return ret;
}
float Max(vector<float>&v)
{
	float ret = 0;
	for (auto n : v)
		ret = max(ret, n);
	return ret;
}
float Var(vector<float>&v)
{
	float avg = 0, ret = 0;
	for (auto n : v)
		avg += n;
	if (v.size() != 0)
		avg /= v.size();

	for (auto n : v)
	{
		ret += (n - avg)*(n - avg);
	}
	if (v.size() != 0)
		ret = sqrt(ret / v.size());
	return ret;
}

float morethan(unsigned termid, unsigned begin, unsigned end, float scorethresh)
{
	float ret = 0;
	for (unsigned i = 0; i < docList[termid].size(); i++)
	{
		unsigned docid = docList[termid][i];
		if (docid < begin)continue;
		if (docid >= end)break;
		if (docList[termid][i] >= scorethresh)ret++;
	}
	return ret;
}

typedef quasi_succinct::bm25 scorer_type;
void cal_ShardScoreFeature(unsigned queryid)
{
	vector<float>morethan_Topk, morethan_Avg, morethan_Max5, morethan_Topk5, var_Maxscore, topkScore, avgScore;
	auto terms = query_freqs(queries[queryid]);
	vector<unsigned>SharddocIDThresh(constShardcount + 1);

	SharddocIDThresh[0] = 0; SharddocIDThresh[1] = 5741824; SharddocIDThresh[2] = 9980428; SharddocIDThresh[3] = 15681198; SharddocIDThresh[4] = 21977502;
	SharddocIDThresh[5] = 28980684; SharddocIDThresh[6] = 36028534; SharddocIDThresh[7] = 43169356; SharddocIDThresh[8] = 50220423;
	for (auto term : terms)
	{
		auto q_weight = scorer_type::query_term_weight(term.second, ListLength[term.first], num_docs);

		unsigned termStartBlock = Block_Start[term.first], termEndBlock = termStartBlock + Block_Start[term.first + 1] - Block_Start[term.first];
		float topkscore = queryFeatures[queryid].query_topK_score, avgscore = 0;
		for (unsigned b = termStartBlock; b < termEndBlock; b++)
		{
			avgscore += Block_Max_Term_Weight[b] * q_weight;
		}
		if ((termEndBlock - termStartBlock) != 0)
			avgscore /= (termEndBlock - termStartBlock);


		vector<float>morethan_topk(constShardcount), morethan_avg(constShardcount), morethan_max5(constShardcount), morethan_topk5(constShardcount), var_maxscore(constShardcount), topkscore_shard(constShardcount), avgscore_shard(constShardcount);
		for (unsigned f = 0; f < constShardcount; f++)
		{
			unsigned endblock = f == constShardcount - 1 ? Block_Start[term.first + 1] - Block_Start[term.first] : splitInfo[f + 1][term.first].split_blockid;
			unsigned startblockpos = Block_Start[term.first];
			for (unsigned b = splitInfo[f][term.first].split_blockid; b < endblock; b++)
			{
				float score = Block_Max_Term_Weight[startblockpos + b] * q_weight;
				if (score >= topkscore)
					morethan_topk[f]++;
				if (score >= avgscore)
					morethan_avg[f]++;
				var_maxscore[f] = max(var_maxscore[f], score);
				avgscore_shard[f] += score;
			}
			topkscore_shard[f] = TermTopkThreshShard[term.first][f] * term.second;
			if ((endblock - splitInfo[f][term.first].split_blockid) != 0)
				avgscore_shard[f] /= (endblock - splitInfo[f][term.first].split_blockid);
			morethan_max5[f] = morethan(term.first, SharddocIDThresh[f], SharddocIDThresh[f + 1], termInfo[term.first].maxscore*0.95);
			morethan_topk5[f] = morethan(term.first, SharddocIDThresh[f], SharddocIDThresh[f + 1], termInfo[term.first].topkscore*0.95);
		}
		morethan_Topk.push_back(Var(morethan_topk));
		morethan_Avg.push_back(Var(morethan_avg));
		var_Maxscore.push_back(Var(var_maxscore));
		topkScore.push_back(Var(topkscore_shard));
		avgScore.push_back(Var(avgscore_shard));
		morethan_Max5.push_back(Var(morethan_max5));
		morethan_Topk5.push_back(Var(morethan_topk5));
	}

	//每个shard中topk阈值的均方差
	queryFeatures[queryid].sum_var_topk_score_shard = Sum(topkScore);
	queryFeatures[queryid].max_var_topk_score_shard = Max(topkScore);
	queryFeatures[queryid].var_var_topk_score_shard = Var(topkScore);

	//每个shard中avg分数的均方差
	queryFeatures[queryid].sum_var_avg_score_shard = Sum(avgScore);
	queryFeatures[queryid].max_var_avg_score_shard = Max(avgScore);
	queryFeatures[queryid].var_var_avg_score_shard = Var(avgScore);

	//每个shard中大于topk块数的均方差
	queryFeatures[queryid].sum_var_morethan_topk_score_shard = Sum(morethan_Topk);
	queryFeatures[queryid].max_var_morethan_topk_score_shard = Max(morethan_Topk);
	queryFeatures[queryid].var_var_morethan_topk_score_shard = Var(morethan_Topk);

	//每个shard中大于avg块数的均方差
	queryFeatures[queryid].sum_var_morethan_avg_score_shard = Sum(morethan_Avg);
	queryFeatures[queryid].max_var_morethan_avg_score_shard = Max(morethan_Avg);
	queryFeatures[queryid].var_var_morethan_avg_score_shard = Var(morethan_Avg);

	//每个shard中最大分数的均方差
	queryFeatures[queryid].sum_var_max_score_shard = Sum(var_Maxscore);
	queryFeatures[queryid].max_var_max_score_shard = Max(var_Maxscore);
	queryFeatures[queryid].var_var_max_score_shard = Var(var_Maxscore);

	//每个shard中大于最大分数95%的posting数量
	queryFeatures[queryid].sum_var_morethan_maxscore5_score_shard = Sum(morethan_Max5);
	queryFeatures[queryid].max_var_morethan_maxscore5_score_shard = Max(morethan_Max5);
	queryFeatures[queryid].var_var_morethan_maxscore5_score_shard = Var(morethan_Max5);

	//每个shard中大于链topk分数95%的posting数量
	queryFeatures[queryid].sum_var_morethan_topkscore5_score_shard = Sum(morethan_Topk5);
	queryFeatures[queryid].max_var_morethan_topkscore5_score_shard = Max(morethan_Topk5);
	queryFeatures[queryid].var_var_morethan_topkscore5_score_shard = Var(morethan_Topk5);
}


void cal_TermScoreFeature(unsigned queryid)
{
	auto terms = query_freqs(queries[queryid]);

	vector<float>postingV, maxscoreV, avgscoreV, topkscoreV;
	vector<float>morethan_topkV, morethan_avgV, var_maxV, morethan_max5V, morethan_topk5V;
	vector<float>inverse_dfV;

	float checktopkscore = 0;
	for (auto term : terms)
	{
		postingV.push_back(ListLength[term.first]);
		topkscoreV.push_back(termTopkThresh[term.first] * term.second);
		auto q_weight = scorer_type::query_term_weight(term.second, ListLength[term.first], num_docs);
		float checkmaxscore = termInfo[term.first].maxscore*term.second;
		checktopkscore = max(checktopkscore, termInfo[term.first].topkscore*term.second);


		unsigned termStartBlock = Block_Start[term.first], termEndBlock = termStartBlock + Block_Start[term.first + 1] - Block_Start[term.first];
		float maxscore = 0, avgscore = 0;
		for (unsigned b = termStartBlock; b < termEndBlock; b++)
		{
			float score = Block_Max_Term_Weight[b] * q_weight;
			avgscore += score;
			maxscore = max(maxscore, score);
		}
		if (termEndBlock - termStartBlock != 0)
			avgscore /= (termEndBlock - termStartBlock);

		if (abs(checkmaxscore - maxscore) > 0.0001)
			cout << "query " << queryid << " checkmaxscore=" << checkmaxscore << " calmaxscore=" << maxscore << endl;

		avgscoreV.push_back(avgscore);
		maxscoreV.push_back(maxscore);

		float morethan_avg = 0, morethan_topk = 0, var_max = 0;
		for (unsigned b = termStartBlock; b < termEndBlock; b++)
		{
			float score = Block_Max_Term_Weight[b] * q_weight;
			var_max += (score - avgscore)*(score - avgscore);
			if (score >= avgscore)morethan_avg++;
			if (score >= queryFeatures[queryid].query_topK_score)morethan_topk++;
		}
		if (termEndBlock - termStartBlock != 0)
			var_max = sqrt(var_max / (termEndBlock - termStartBlock));

		morethan_topkV.push_back(morethan_topk);
		morethan_avgV.push_back(morethan_avg);
		var_maxV.push_back(var_max);
		inverse_dfV.push_back(1.0 / ListLength[term.first]);

		float morethan_max5 = morethan(term.first, 0, num_docs, termInfo[term.first].maxscore*0.95);
		morethan_max5V.push_back(morethan_max5);

		float morethan_topk5 = morethan(term.first, 0, num_docs, termInfo[term.first].topkscore*0.95);
		morethan_topk5V.push_back(morethan_topk5);
	}
	if (abs(checktopkscore - queryFeatures[queryid].query_topK_score) > 0.0001)
		cout << "query " << queryid << " checktopkscore=" << checktopkscore << " caltopkscore=" << queryFeatures[queryid].query_topK_score << endl;
	//term信息（sum,max,var）

	//posting个数
	queryFeatures[queryid].sum_no_positings = Sum(postingV);
	queryFeatures[queryid].max_no_positings = Max(postingV);
	queryFeatures[queryid].var_no_positings = Var(postingV);

	//每个term的块最大分数 最大值
	queryFeatures[queryid].sum_max_score_term = Sum(maxscoreV);
	queryFeatures[queryid].max_max_score_term = Max(maxscoreV);
	queryFeatures[queryid].var_max_score_term = Var(maxscoreV);

	//每个term的块最大分数 均值
	queryFeatures[queryid].sum_avg_score_term = Sum(avgscoreV);
	queryFeatures[queryid].max_avg_score_term = Max(avgscoreV);
	queryFeatures[queryid].var_avg_score_term = Var(avgscoreV);

	//每个term的块最大分数 topK
	queryFeatures[queryid].sum_topk_score_term = Sum(topkscoreV);
	queryFeatures[queryid].max_topk_score_term = Max(topkscoreV);
	queryFeatures[queryid].var_topk_score_term = Var(topkscoreV);

	//大于块最大分数均值的块数
	queryFeatures[queryid].sum_morethan_avg_score = Sum(morethan_avgV);
	queryFeatures[queryid].max_morethan_avg_score = Max(morethan_avgV);
	queryFeatures[queryid].var_morethan_avg_score = Var(morethan_avgV);

	//大于块最大分数topk的块数
	queryFeatures[queryid].sum_morethan_topk_score = Sum(morethan_topkV);
	queryFeatures[queryid].max_morethan_topk_score = Max(morethan_topkV);
	queryFeatures[queryid].var_morethan_topk_score = Var(morethan_topkV);

	//块最大值均方误差
	queryFeatures[queryid].sum_var_max_score_term = Sum(var_maxV);
	queryFeatures[queryid].max_var_max_score_term = Max(var_maxV);
	queryFeatures[queryid].var_var_max_score_term = Var(var_maxV);

	//块df倒数
	queryFeatures[queryid].sum_inverse_df = Sum(inverse_dfV);
	queryFeatures[queryid].max_inverse_df = Max(inverse_dfV);
	queryFeatures[queryid].var_inverse_df = Var(inverse_dfV);

	//大于最大分数95%的posting数量
	queryFeatures[queryid].sum_morethan_maxscore5_score = Sum(morethan_max5V);
	queryFeatures[queryid].max_morethan_maxscore5_score = Max(morethan_max5V);
	queryFeatures[queryid].var_morethan_maxscore5_score = Var(morethan_max5V);

	//大于链中topk分数95%的posting数量
	queryFeatures[queryid].sum_morethan_topkscore5_score = Sum(morethan_topk5V);
	queryFeatures[queryid].max_morethan_topkscore5_score = Max(morethan_topk5V);
	queryFeatures[queryid].var_morethan_topkscore5_score = Var(morethan_topk5V);
}
void calFeature()
{
	for (unsigned i = 0; i < queries.size(); i++)
	{
		cal_ShardScoreFeature(i);
		cal_TermScoreFeature(i);
	}
}
vector<float>pall;
void readLabel(string filename)
{
	ifstream fin(filename);
	string str = "";
	vector<int>transform = { -1, 0, 1, -1, 2, -1, -1, -1, 3 };//1 2 4 8并行度
	while (getline(fin, str))
	{
		unsigned label = transform[atoi(str.c_str())];
		pall.push_back(label);
	}
	fin.close();
	if (pall.size() != queryFeatures.size())cout << "pall size wrong" << endl;
}
void readSampleTime(string filename)
{
	ifstream fin(filename);
	string str = "";
	for (unsigned i = 0; i < queryFeatures.size(); i++)
	{
		getline(fin, str);
		float label = atof(str.c_str());
		queryFeatures[i].query_sample_time = label;
	}
	fin.close();
}
void writeFeature(string filename)
{
	cout << "queryFeature.size()=" << queryFeatures.size() << endl;
	ofstream fout(filename);
	for (unsigned i = 0; i < queryFeatures.size(); i++)
	{
		fout << queryFeatures[i].term_count << ",";
		fout << queryFeatures[i].query_topK_score << ",";

		fout << queryFeatures[i].sum_no_positings << ",";
		fout << queryFeatures[i].max_no_positings << ",";
		fout << queryFeatures[i].var_no_positings << ",";

		fout << queryFeatures[i].sum_max_score_term << ",";
		fout << queryFeatures[i].max_max_score_term << ",";
		fout << queryFeatures[i].var_max_score_term << ",";

		fout << queryFeatures[i].sum_avg_score_term << ",";
		fout << queryFeatures[i].max_avg_score_term << ",";
		fout << queryFeatures[i].var_avg_score_term << ",";

		fout << queryFeatures[i].sum_topk_score_term << ",";
		fout << queryFeatures[i].max_topk_score_term << ",";
		fout << queryFeatures[i].var_topk_score_term << ",";

		fout << queryFeatures[i].sum_morethan_avg_score << ",";
		fout << queryFeatures[i].max_morethan_avg_score << ",";
		fout << queryFeatures[i].var_morethan_avg_score << ",";

		fout << queryFeatures[i].sum_morethan_topk_score << ",";
		fout << queryFeatures[i].max_morethan_topk_score << ",";
		fout << queryFeatures[i].var_morethan_topk_score << ",";

		fout << queryFeatures[i].sum_var_max_score_term << ",";
		fout << queryFeatures[i].max_var_max_score_term << ",";
		fout << queryFeatures[i].var_var_max_score_term << ",";

		fout << queryFeatures[i].sum_inverse_df << ",";
		fout << queryFeatures[i].max_inverse_df << ",";
		fout << queryFeatures[i].var_inverse_df << ",";

		fout << queryFeatures[i].sum_var_topk_score_shard << ",";
		fout << queryFeatures[i].max_var_topk_score_shard << ",";
		fout << queryFeatures[i].var_var_topk_score_shard << ",";

		fout << queryFeatures[i].sum_var_avg_score_shard << ",";
		fout << queryFeatures[i].max_var_avg_score_shard << ",";
		fout << queryFeatures[i].var_var_avg_score_shard << ",";

		fout << queryFeatures[i].sum_var_morethan_topk_score_shard << ",";
		fout << queryFeatures[i].max_var_morethan_topk_score_shard << ",";
		fout << queryFeatures[i].var_var_morethan_topk_score_shard << ",";

		fout << queryFeatures[i].sum_var_morethan_avg_score_shard << ",";
		fout << queryFeatures[i].max_var_morethan_avg_score_shard << ",";
		fout << queryFeatures[i].var_var_morethan_avg_score_shard << ",";

		fout << queryFeatures[i].sum_var_max_score_shard << ",";
		fout << queryFeatures[i].max_var_max_score_shard << ",";
		fout << queryFeatures[i].var_var_max_score_shard << ",";


		fout << queryFeatures[i].sum_morethan_maxscore5_score << ",";
		fout << queryFeatures[i].max_morethan_maxscore5_score << ",";
		fout << queryFeatures[i].var_morethan_maxscore5_score << ",";

		fout << queryFeatures[i].sum_morethan_topkscore5_score << ",";
		fout << queryFeatures[i].max_morethan_topkscore5_score << ",";
		fout << queryFeatures[i].var_morethan_topkscore5_score << ",";

		fout << queryFeatures[i].sum_var_morethan_maxscore5_score_shard << ",";
		fout << queryFeatures[i].max_var_morethan_maxscore5_score_shard << ",";
		fout << queryFeatures[i].var_var_morethan_maxscore5_score_shard << ",";

		fout << queryFeatures[i].sum_var_morethan_topkscore5_score_shard << ",";
		fout << queryFeatures[i].max_var_morethan_topkscore5_score_shard << ",";
		fout << queryFeatures[i].var_var_morethan_topkscore5_score_shard << ",";

		fout << queryFeatures[i].query_sample_time << endl;
	}
	fout.close();
}
int main()
{
	init("");
	cout << "init over" << endl;
	calFeature();
	cout << "cal feature over" << endl;
	readSampleTime("");
	writeFeature("");
	cout << "write over" << endl;
}
