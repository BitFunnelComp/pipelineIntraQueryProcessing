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
#include <libaio.h>
#include<unordered_set>
#include <unistd.h>
#include<queue>

#include <succinct/mapper.hpp>
#include"bm25.hpp"
#include"wand_data.hpp"
#include"Mycodec.hpp"
#include"LRUCache_PrefetchIntra.h"
using namespace std;

struct QueryInfo
{
	unsigned threadcount;
	double time;
	unsigned resultcount;
	vector<float>scoreHeap;
	QueryInfo(){ threadcount = 0; time = 0; resultcount = 0; }
};

struct Task
{
	vector<unsigned>shardids;
	unsigned queryid;
};


const unsigned topK = 10;
vector<float>globalScoreThresh;
vector<omp_lock_t>queryLock;
struct topk_queue {
	topk_queue()
	: m_k(topK)
	{
	}
	void updateScore()
	{
		omp_set_lock(&queryLock[qid]);
		if (globalScoreThresh[qid] < m_q.front())
			globalScoreThresh[qid] = m_q.front();
		omp_unset_lock(&queryLock[qid]);
	}
	bool insert(float score)
	{
		if (m_q.size() < m_k) {
			if (score>globalScoreThresh[qid])
			{
				m_q.push_back(score);
				std::push_heap(m_q.begin(), m_q.end(), std::greater<float>());
				if (m_q.size() == m_k)updateScore();
				return true;
			}
		}
		else {
			if (score > globalScoreThresh[qid] && score > m_q.front()) {
				std::pop_heap(m_q.begin(), m_q.end(), std::greater<float>());
				m_q.back() = score;
				std::push_heap(m_q.begin(), m_q.end(), std::greater<float>());
				updateScore();
				return true;
			}
		}
		return false;
	}

	bool would_enter(float score) const
	{
		return (score > globalScoreThresh[qid] && (m_q.size() < m_k || score > m_q.front()));
	}

	void finalize()
	{
		std::sort_heap(m_q.begin(), m_q.end(), std::greater<float>());
	}

	void clear(unsigned queryid)
	{
		m_q.clear();
		qid = queryid;
	}
	uint64_t m_k;
	std::vector<float> m_q;
	unsigned qid;
};

vector<unsigned>SharddocIDThresh(constShardcount + 1);
queue<Task>queryTasks;
vector<QueryInfo>queryInfo;
vector<topk_queue>scoreQueue;


vector<int64_t>Head_offset;
string indexFileName = "";
vector<vector<double>>query_Times;
vector<uint8_t>Head_Data;
unsigned threadCount = 4;

vector<uint64_t>Block_Start;
vector<uint32_t>Block_Docid;
vector<float>Block_Max_Term_Weight;

typedef uint32_t term_id_type;
typedef std::vector<term_id_type> term_id_vec;
vector<term_id_vec> queries;
unsigned num_docs = 0;
unsigned curQID = 0;


const int AIOLIST_SIZE = 16;
const unsigned MAXREQUEST = 128;

vector<io_context_t> ctx;
vector<vector<struct io_event>> events;
vector<vector<struct iocb>> readrequest;
vector<vector<struct iocb*>> listrequest;
vector<int>curAIOLIST_SIZE;
vector<int>curReadID;
vector<vector<Node*>>Nodeforthread;
vector<vector<Node*>>NodeforQuery;
int IndexFile;

LRUCache global_LRUCache;


vector<Node*>prefetchList(AIOLIST_SIZE*constShardcount);
int64_t curBandwidth = 0;

string queryFilename = "";

vector<vector<pair<unsigned, float>>>ShardScore;
vector<vector<float>>TermTopkThreshShard;
vector<unsigned>assignedThreads;

int queryFileNo=0;
void initThreadStruct()
{
	Nodeforthread.resize(threadCount);
	NodeforQuery.resize(threadCount);
	ctx.resize(threadCount);
	events.resize(threadCount, vector<struct io_event>(MAXREQUEST));
	readrequest.resize(threadCount, vector<struct iocb>(MAXREQUEST));
	listrequest.resize(threadCount, vector<struct iocb*>(MAXREQUEST));
	curReadID.resize(threadCount);
	curAIOLIST_SIZE.resize(threadCount);

	for (unsigned i = 0; i < threadCount; i++)
	{
		ctx[i] = 0;
		if (io_setup(MAXREQUEST, &ctx[i])) {
			perror("io_setup");
			return;
		}
	}
}

bool hasReadFinish()
{
	int threadid = omp_get_thread_num();
	unsigned count = 0;
	for (auto l : Nodeforthread[threadid])
	{
		if (l->aiodata.listlength <= l->aiodata.curSendpos)
			count++;
	}
	return count == Nodeforthread[threadid].size();
}
void aioReadBlock()
{
	int threadid = omp_get_thread_num();
	while (io_getevents(ctx[threadid], curAIOLIST_SIZE[threadid], MAXREQUEST, events[threadid].data(), NULL) != curAIOLIST_SIZE[threadid])
	{
		;
	}
	for (unsigned i = 0; i < Nodeforthread[threadid].size(); i++)
	{
		Nodeforthread[threadid][i]->aiodata.curReadpos = Nodeforthread[threadid][i]->aiodata.curSendpos;
#pragma omp flush(Nodeforthread)
	}

	int64_t cur = 0, i = 0;
	for (i = 0, cur = 0; cur < AIOLIST_SIZE&&Nodeforthread[threadid].size(); curReadID[threadid]++, i++)
	{
		unsigned lid = curReadID[threadid] % Nodeforthread[threadid].size();
		if (Nodeforthread[threadid][lid]->aiodata.listlength <= Nodeforthread[threadid][lid]->aiodata.curSendpos)
		{
			if (hasReadFinish()){ break; }
			else { continue; }
		}
		io_prep_pread(&readrequest[threadid][cur], IndexFile, Nodeforthread[threadid][lid]->aiodata.list_data + Nodeforthread[threadid][lid]->aiodata.memoffset, Nodeforthread[threadid][lid]->aiodata.readblocksize, Nodeforthread[threadid][lid]->aiodata.readoffset);
		listrequest[threadid][cur] = &readrequest[threadid][cur];
		Nodeforthread[threadid][lid]->aiodata.memoffset += Nodeforthread[threadid][lid]->aiodata.readblocksize;
		Nodeforthread[threadid][lid]->aiodata.readoffset += Nodeforthread[threadid][lid]->aiodata.readblocksize;
		Nodeforthread[threadid][lid]->aiodata.curSendpos += Nodeforthread[threadid][lid]->aiodata.readblocksize;
#pragma omp critical(bandwidth)
		{
			curBandwidth += Nodeforthread[threadid][lid]->aiodata.readblocksize;
#pragma omp flush(curBandwidth)
		}
		cur++;
	}
	curAIOLIST_SIZE[threadid] = cur; 
	if (cur == 0){ return; }
	if (io_submit(ctx[threadid], curAIOLIST_SIZE[threadid], listrequest[threadid].data()) != curAIOLIST_SIZE[threadid]) {
		perror("io_submit"); cout << "submit 1 in aioReadBlock" << endl;
	}
}

bool hasReadFinishIOWork()
{
#pragma omp flush (prefetchList)
	for (auto l : prefetchList)
	{
		if (l != NULL&&l->aiodata.listlength > l->aiodata.curSendpos)
			return false;
	}
	return true;
}
void readIOWork()
{
	io_context_t ctxPref;
	struct io_event eventsPref[MAXREQUEST];
	struct iocb readrequestPref[MAXREQUEST];
	struct iocb* listrequestPref[MAXREQUEST];
	if (io_setup(MAXREQUEST, &ctxPref)) {
		perror("io_setup");
		return;
	}

	int64_t curReadID = 0;
	while (curQID < queries.size())
	{
#pragma omp flush(curQID)
		int cur = 0;
		for (; cur < AIOLIST_SIZE; curReadID++)
		{
			if (curReadID>INT32_MAX)curReadID = 0;
			int i = curReadID%prefetchList.size();
#pragma omp flush (prefetchList)
			if (prefetchList[i] == NULL || prefetchList[i]->aiodata.listlength <= prefetchList[i]->aiodata.curSendpos)
			{
				if (hasReadFinishIOWork()){ break; }
				else { continue; }
			}
			io_prep_pread(&readrequestPref[cur], IndexFile, prefetchList[i]->aiodata.list_data + prefetchList[i]->aiodata.memoffset, prefetchList[i]->aiodata.readblocksize, prefetchList[i]->aiodata.readoffset);
			listrequestPref[cur] = &readrequestPref[cur];
			prefetchList[i]->aiodata.memoffset += prefetchList[i]->aiodata.readblocksize;
			prefetchList[i]->aiodata.readoffset += prefetchList[i]->aiodata.readblocksize;
			prefetchList[i]->aiodata.curSendpos += prefetchList[i]->aiodata.readblocksize;
#pragma omp critical(bandwidth)
			{
				curBandwidth += prefetchList[i]->aiodata.readblocksize;
#pragma omp flush(curBandwidth)
			}
			cur++;
		}
		if (io_submit(ctxPref, cur, listrequestPref) != cur) {
			perror("io_submit"); cout << "submit 1 in aioReadBlock" << endl;
		}
		while (io_getevents(ctxPref, cur, MAXREQUEST, eventsPref, NULL) != cur)
		{
			;
		}
		for (unsigned i = 0; i< prefetchList.size(); i++)
		{
			if (prefetchList[i] != NULL)
			{
				prefetchList[i]->aiodata.curReadpos = prefetchList[i]->aiodata.curSendpos;
#pragma omp flush(prefetchList)
				if (prefetchList[i]->aiodata.curSendpos >= prefetchList[i]->aiodata.listlength)
				{
#pragma omp atomic
					prefetchList[i]->aiodata.usedfreq--;

					prefetchList[i] = NULL;
				}
#pragma omp flush (prefetchList)
			}
		}
	}
	cout << "Out Pref" << endl;
}
bool cmpFreq(const pair<unsigned, unsigned>&a, const pair<unsigned, unsigned>&b)
{
	return a.second > b.second;
}
void detectIOWork()
{
	while (curQID < queries.size())
	{
#pragma omp flush(curQID)
		usleep(1000);
#pragma omp flush(curBandwidth)
		int64_t tmpbandwidth = curBandwidth / 1024;

#pragma omp critical(bandwidth)
		{
			curBandwidth = 0;
#pragma omp flush(curBandwidth)
		}
		if (tmpbandwidth > 2000)
		{
			continue;
		}
#pragma omp flush(curQID)
		unordered_map<unsigned, unsigned>querytermfreq;
		for (unsigned i = curQID; i < curQID + threadCount && i < queries.size(); i++)
		{
			for (auto tid : queries[i])
				querytermfreq[tid]++;
		}
		vector<pair<unsigned, unsigned>>querytermset(querytermfreq.begin(), querytermfreq.end());
		sort(querytermset.begin(), querytermset.end(), cmpFreq);

		unsigned prefid = 0;
		for (unsigned i = 0; i < querytermset.size(); i++)
		{
			for (unsigned j = 0; j < constShardcount; j++)
			{
#pragma omp flush (prefetchList)
				while (prefid < prefetchList.size() && prefetchList[prefid] != NULL){ prefid++; }
				if (prefid == prefetchList.size())break;
#pragma omp critical(VisitsLRU)
				{
					bool flag = 0;
					Node *node = global_LRUCache.Get_Prefetch(j, querytermset[i].first, flag);
					if (flag == 0 && node != NULL)
					{
#pragma omp atomic
						node->aiodata.usedfreq++;
						prefetchList[prefid] = node;
						prefid++;
					}
				}
			}
		}
	}
	cout << "Out detect" << endl;
}

struct block_posting_list {


	static unsigned write(std::vector<uint8_t>& out, uint32_t n,
	vector<uint32_t> docs_begin, vector<uint32_t> freqs_begin) {
		unsigned outsizeold = out.size();
		TightVariableByte::encode_single(n, out);

		uint64_t block_size = Codec::block_size;
		uint64_t blocks = ceil((double)n / (double)block_size);
		size_t begin_block_maxs = out.size();
		size_t begin_block_endpoints = begin_block_maxs + 4 * blocks;
		size_t begin_blocks = begin_block_endpoints + 4 * (blocks - 1);
		out.resize(begin_blocks);

		uint32_t* docs_it = docs_begin.data();
		uint32_t* freqs_it = freqs_begin.data();
		std::vector<uint32_t> docs_buf(block_size);
		std::vector<uint32_t> freqs_buf(block_size);
		uint32_t last_doc(-1);
		uint32_t block_base = 0;
		for (size_t b = 0; b < blocks; ++b) {
			uint32_t cur_block_size =
				((b + 1) * block_size <= n)
				? block_size : (n % block_size);

			for (size_t i = 0; i < cur_block_size; ++i) {
				uint32_t doc(*docs_it++);
				docs_buf[i] = doc - last_doc - 1;
				last_doc = doc;

				freqs_buf[i] = *freqs_it++ - 1;
			}
			*((uint32_t*)&out[begin_block_maxs + 4 * b]) = last_doc;

			NoComp::encode(docs_buf.data(), last_doc - block_base - (cur_block_size - 1),
				cur_block_size, out);
			NoComp::encode(freqs_buf.data(), uint32_t(-1), cur_block_size, out);
			if (b != blocks - 1) {
				*((uint32_t*)&out[begin_block_endpoints + 4 * b]) = out.size() - begin_blocks;
			}
			block_base = last_doc + 1;
		}
		return (out.size() - outsizeold);
	}

	class document_enumerator {
	public:
		uint32_t m_n;
		uint8_t const* m_base;
		uint32_t m_blocks;
		uint8_t const* m_block_maxs;
		uint8_t const* m_block_endpoints;
		uint8_t const* m_blocks_data;
		uint64_t m_universe;

		uint32_t m_cur_block;
		uint32_t m_pos_in_block;
		uint32_t m_cur_block_max;
		uint32_t m_cur_block_size;
		uint32_t m_cur_docid;
		SIMDNewPfor m_codec;

		uint8_t const* m_freqs_block_data;
		bool m_freqs_decoded;

		std::vector<uint32_t> m_docs_buf;
		std::vector<uint32_t> m_freqs_buf;

		unsigned m_shardid;
		unsigned m_termid;
		unsigned m_baseblockid;

		unsigned m_lid;
		int m_threadid;

		uint32_t block_max(uint32_t block) const
		{
			return ((uint32_t const*)m_block_maxs)[block];
		}
		void QS_NOINLINE decode_docs_block(uint64_t block)
		{
			uint32_t base_endpoint = m_baseblockid
				? ((uint32_t const*)m_block_endpoints)[m_baseblockid - 1]
				: 0;

			int64_t tmpendpoint = 0;
			if (block == m_blocks - 1)
			{
				tmpendpoint = List_offset[m_termid + 1] - List_offset[m_termid];
			}
			else
			{
				tmpendpoint = ((uint32_t const*)m_block_endpoints)[block];
			}
			tmpendpoint -= base_endpoint;
#pragma omp flush(NodeforQuery)
			while (NodeforQuery[m_threadid][m_lid]->aiodata.curReadpos < tmpendpoint)
			{
#pragma omp flush(NodeforQuery)
				
				aioReadBlock();
			}
			static const uint64_t block_size = Codec::block_size;
			uint32_t endpoint = block
				? ((uint32_t const*)m_block_endpoints)[block - 1]
				: 0;
			uint8_t const* block_data = m_blocks_data + endpoint - base_endpoint;

			m_cur_block_size =
				((block + 1) * block_size <= size())
				? block_size : (size() % block_size);
			uint32_t cur_base = (block ? block_max(block - 1) : uint32_t(-1)) + 1;
			m_cur_block_max = block_max(block);
			m_freqs_block_data =
				m_codec.decode(block_data, m_docs_buf.data(),
				m_cur_block_max - cur_base - (m_cur_block_size - 1),
				m_cur_block_size);
			m_docs_buf[0] += cur_base;

			m_cur_block = block;
			m_pos_in_block = 0;
			m_cur_docid = m_docs_buf[0] >= SharddocIDThresh[m_shardid + 1] ? m_universe : m_docs_buf[0];
			m_freqs_decoded = false;
		}
		void pprint(uint8_t const* in)
		{
			cout << "print" << (int)*in << " " << (int)*(in + 1) << " " << (int)*(in + 2) << " " << (int)*(in + 3) << endl;
		}
		void QS_NOINLINE decode_freqs_block()
		{
			m_codec.decode(m_freqs_block_data, m_freqs_buf.data(),
				uint32_t(-1), m_cur_block_size);
			m_freqs_decoded = true;
		}
		void reset()
		{
			decode_docs_block(0 + m_baseblockid); 
			next_geq(SharddocIDThresh[m_shardid]);
		}
		document_enumerator(uint8_t const* headdata, uint8_t const* listdata, unsigned universe, unsigned shardid, unsigned termid, unsigned lid)
			: m_n(0)
			, m_base(TightVariableByte::decode(headdata, &m_n, 1))
			, m_blocks(ceil((double)m_n / (double)Codec::block_size))
			, m_block_maxs(m_base)
			, m_block_endpoints(m_block_maxs + 4 * m_blocks)
			, m_blocks_data(listdata)
			, m_universe(universe)
			, m_shardid(shardid)
			, m_termid(termid)
			, m_lid(lid)
		{
			m_baseblockid = splitInfo[m_shardid][m_termid].split_blockid;
			m_docs_buf.resize(Codec::block_size);
			m_freqs_buf.resize(Codec::block_size);
			m_threadid = omp_get_thread_num();
		}
		void inline next()
		{
			++m_pos_in_block;
			if (QS_UNLIKELY(m_pos_in_block == m_cur_block_size)) {
				if (m_cur_block + 1 == m_blocks || ((m_shardid<constShardcount - 1) && (m_cur_block + 1 == splitInfo[m_shardid + 1][m_termid].split_blockid + 1))) {
					m_cur_docid = m_universe;
					return;
				}
				decode_docs_block(m_cur_block + 1);
			}
			else {
				m_cur_docid += m_docs_buf[m_pos_in_block] + 1;
				m_cur_docid = m_cur_docid >= SharddocIDThresh[m_shardid + 1] ? m_universe : m_cur_docid;
			}
		}
		uint64_t docid() const
		{
			return m_cur_docid; 
		}
		uint64_t inline freq()
		{
			if (!m_freqs_decoded) {
				decode_freqs_block();
			}
			return m_freqs_buf[m_pos_in_block] + 1;
		}

		uint64_t position() const
		{
			return m_cur_block * Codec::block_size + m_pos_in_block;
		}

		uint64_t size() const
		{
			return m_n;
		}
		void  inline next_geq(uint64_t lower_bound)
		{
			if (QS_UNLIKELY(lower_bound > m_cur_block_max)) {
				if (lower_bound >= SharddocIDThresh[m_shardid + 1] || lower_bound > block_max(m_blocks - 1)) {
					m_cur_docid = m_universe;
					return;
				}

				uint64_t block = m_cur_block + 1;
				while (block_max(block) < lower_bound) {
					++block;
				}

				decode_docs_block(block);
			}

			while (docid() < lower_bound) {
				m_cur_docid += m_docs_buf[++m_pos_in_block] + 1;
				assert(m_pos_in_block < m_cur_block_size);
			}
			m_cur_docid = m_cur_docid >= SharddocIDThresh[m_shardid + 1] ? m_universe : m_cur_docid;
		}

	};

};

class wand_data_enumerator{
public:

	wand_data_enumerator(uint64_t _block_start, uint64_t _block_number, vector<float> const & max_term_weight,
		vector<uint32_t> const & block_docid) :
		cur_pos(0),
		block_start(_block_start),
		block_number(_block_number),
		m_block_max_term_weight(max_term_weight),
		m_block_docid(block_docid)
	{}


	void  next_geq(uint64_t lower_bound) {
		while (cur_pos + 1 < block_number &&
			m_block_docid[block_start + cur_pos] <
			lower_bound) {
			cur_pos++;
		}
	}


	float score() const {
		return m_block_max_term_weight[block_start + cur_pos];
	}

	uint64_t docid() const {
		return m_block_docid[block_start + cur_pos];
	}


	uint64_t find_next_skip() {
		return m_block_docid[cur_pos + block_start];
	}

private:
	uint64_t cur_pos;
	uint64_t block_start;
	uint64_t block_number;
	vector<float> const &m_block_max_term_weight;
	vector<uint32_t> const &m_block_docid;

};

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
void read_List_Length(string filename)
{
	FILE *lengthfile = fopen((filename + ".list-l").c_str(), "rb");
	unsigned tmplength = 0;
	int64_t prelength = 0;
	while (fread(&tmplength, sizeof(unsigned), 1, lengthfile))
	{
		List_offset.push_back(prelength);
		prelength += tmplength;
	}
	List_offset.push_back(prelength);
	cout << "All we have " << List_offset.size() - 1 << " lists" << endl;
	fclose(lengthfile);
}
void read_BlockWand_Data(string filename)
{
	FILE *file = fopen((filename + ".BMWwand").c_str(), "rb");
	uint64_t length = 0;
	fread(&length, sizeof(uint64_t), 1, file);
	fread(&length, sizeof(uint64_t), 1, file);
	Block_Start.resize(length);
	fread(Block_Start.data(), sizeof(uint64_t), length, file);
	fread(&length, sizeof(uint64_t), 1, file); cout << "All we have " << length << " blocks" << endl;
	Block_Max_Term_Weight.resize(length);
	fread(Block_Max_Term_Weight.data(), sizeof(float), length, file);
	fread(&length, sizeof(uint64_t), 1, file);
	Block_Docid.resize(length);
	fread(Block_Docid.data(), sizeof(uint32_t), length, file);
	fclose(file);
}
void read_SplitInfo(string filename)
{
	FILE *file = fopen((filename + ".splitinfo").c_str(), "rb");
	SplitInfo tmps;
	int64_t tmpoffset;
	for (unsigned i = 0; i < constShardcount; i++)
	{
		splitInfo[i].resize(List_offset.size() - 1);
		for (unsigned j = 0; j < List_offset.size() - 1; j++)
		{
			fread(&tmps.split_blockid, sizeof(unsigned), 1, file);
			fread(&tmpoffset, sizeof(int64_t), 1, file);
			splitInfo[i][j] = tmps;
		}
	}
	fclose(file);
	cout << "read split over" << endl;
	for (unsigned j = 0; j < List_offset.size() - 1; j++)
	{
		for (unsigned i = 1; i < constShardcount; i++)
		{
			if (splitInfo[i][j].split_blockid < splitInfo[i - 1][j].split_blockid)
				cout << "SplitInfo blockid Wrong!" << endl;
		}
	}
	unsigned docIDinterval = ceil((double)num_docs / constShardcount);
	SharddocIDThresh[0] = 0; SharddocIDThresh[1] = 5741824; SharddocIDThresh[2] = 9980428; SharddocIDThresh[3] = 15681198; SharddocIDThresh[4] = 21977502;
	SharddocIDThresh[5] = 28980684; SharddocIDThresh[6] = 36028534; SharddocIDThresh[7] = 43169356; SharddocIDThresh[8] = 50220423;
	cout << "init thresh over" << endl;
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
	queryInfo.resize(queries.size());
	cout << "All we have " << queries.size() << " queries" << endl;
}
void read_Endpoint(string filename)
{
	FILE *file = fopen((filename + ".blockendpoint").c_str(), "rb");
	blockEndpoint.resize(List_offset.size() - 1);
	for (unsigned i = 0; i < List_offset.size() - 1; i++)
	{
		unsigned length = 0;
		fread(&length, sizeof(unsigned), 1, file);
		blockEndpoint[i].resize(length);
		fread(blockEndpoint[i].data(), sizeof(unsigned), length, file);
	}
	fclose(file);
}
void read_AssignedThreads()
{
	assignedThreads.resize(queries.size());
	vector<int>transform = { 1, 2, 4, 8 };
	string filename = "";
	ifstream fin(filename);
	int threadsnum;
	for (unsigned i = 0; i < queries.size(); i++)
	{
		fin >> threadsnum;
		assignedThreads[i] = threadsnum;
	}
	fin.close();

	for (int i = 0; i<10; i++)
		cout << assignedThreads[i] << " ";
	cout << endl;
}
void read_ShardScore()
{
	ShardScore.resize(queries.size());
	string filename = "";
	ifstream fin(filename);
	for (unsigned i = 0; i < queries.size(); i++)
	{
		ShardScore[i].resize(constShardcount);
		for (unsigned j = 0; j < constShardcount; j++)
		{
			ShardScore[i][j].first = j;
			fin >> ShardScore[i][j].second;
		}
	}
	fin.close();
	for (unsigned i = queries.size() - 3; i < queries.size(); i++)
	{
		for (unsigned j = 0; j < constShardcount; j++)
		{
			cout << ShardScore[i][j].second << "\t";
		}
		cout << endl;
	}
}

void remove_duplicate_terms(term_id_vec& terms)
{
	std::sort(terms.begin(), terms.end());
	terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
}
typedef quasi_succinct::bm25 scorer_type;

typedef typename block_posting_list::document_enumerator enum_type;
typedef wand_data_enumerator wdata_enum;

struct scored_enum {
	enum_type docs_enum;
	wdata_enum w;
	float q_weight;
	float max_weight;
};

void aioReadFirstBlock()
{
	int threadid = omp_get_thread_num();
	curReadID[threadid] = 0;
	unsigned cur = 0, i = 0;
	for (i = 0, cur = 0; i < Nodeforthread[threadid].size(); i++)
	{
		unsigned lid = i;
		io_prep_pread(&readrequest[threadid][cur], IndexFile, Nodeforthread[threadid][lid]->aiodata.list_data + Nodeforthread[threadid][lid]->aiodata.memoffset, Nodeforthread[threadid][lid]->aiodata.readblocksize, Nodeforthread[threadid][lid]->aiodata.readoffset);
		listrequest[threadid][cur] = &readrequest[threadid][cur];
		Nodeforthread[threadid][lid]->aiodata.memoffset += Nodeforthread[threadid][lid]->aiodata.readblocksize;
		Nodeforthread[threadid][lid]->aiodata.readoffset += Nodeforthread[threadid][lid]->aiodata.readblocksize;
		Nodeforthread[threadid][lid]->aiodata.curSendpos += Nodeforthread[threadid][lid]->aiodata.readblocksize;
#pragma omp critical(bandwidth)
		{
			curBandwidth += Nodeforthread[threadid][lid]->aiodata.readblocksize;
#pragma omp flush(curBandwidth)
		}
		cur++;
	}
	curAIOLIST_SIZE[threadid] = cur;
	int ret = io_submit(ctx[threadid], curAIOLIST_SIZE[threadid], listrequest[threadid].data());
	if (ret != curAIOLIST_SIZE[threadid]) {
		perror("io_submit"); cout << "submit 2" << endl; cout << "cur " << cur << " ret=" << ret << endl;
	}
}


struct and_query {
	uint64_t operator()(unsigned shardid, unsigned queryid)const
	{
		int threadid = omp_get_thread_num();
		vector<unsigned>terms = queries[queryid];
		if (terms.empty()) return 0;
		remove_duplicate_terms(terms);
		typedef typename block_posting_list::document_enumerator enum_type;
		std::vector<enum_type> enums;
		enums.reserve(terms.size());
		Nodeforthread[threadid].clear();
		NodeforQuery[threadid].clear();
		int64_t tmpcount = 0;
#pragma omp critical(VisitsLRU)
		{
			Node*node;
			for (auto term : terms)
			{
				bool flag = 0;
				node = global_LRUCache.Get(shardid, term, flag); 
#pragma omp atomic
				node->aiodata.usedfreq++;
				NodeforQuery[threadid].push_back(node);
				if (flag == 0)
				{
					Nodeforthread[threadid].push_back(node); 
				}
			}
		}
		aioReadFirstBlock(); 

		for (auto term : terms)
		{
			AIOReadInfo tmpaio = NodeforQuery[threadid][tmpcount]->aiodata;
			enum_type tmplist(Head_Data.data() + Head_offset[term], tmpaio.list_data + tmpaio.offsetForenums, num_docs, shardid, term, tmpcount);
			enums.push_back(tmplist);
			tmpcount++;
		}
		for (unsigned i = 0; i < enums.size(); i++)
		{
			enums[i].reset();
		}
		std::sort(enums.begin(), enums.end(),
			[](enum_type const& lhs, enum_type const& rhs) {
			return lhs.size() < rhs.size();
		});

		uint64_t results = 0;
		uint64_t candidate = enums[0].docid();
		size_t i = 1;
		while (candidate < num_docs) {
			for (; i < enums.size(); ++i) {
				enums[i].next_geq(candidate);
				if (enums[i].docid() != candidate) {
					candidate = enums[i].docid();
					i = 0;
					break;
				}
			}

			if (i == enums.size()) {
				results += 1;
				enums[0].next();
				candidate = enums[0].docid();
				i = 1;
			}
		}
		return results;
	}

};
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
void read_TermtopKThresh(string filename)
{
	vector<float>termTopkThresh;
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
		globalScoreThresh[i] = max(0.0, score - 0.0001);
	}
}
void read_TermTopkThreshShard(string filename)
{
	ifstream fin(filename);
	string str = "";
	TermTopkThreshShard.resize(List_offset.size() - 1);
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

struct block_max_wand_query {

	block_max_wand_query(quasi_succinct::wand_data<scorer_type> const& wdata)
	: m_wdata(wdata) {
	}



	uint64_t operator()(unsigned shardid, unsigned queryid) {

		unsigned threadid = omp_get_thread_num();
		vector<unsigned>terms = queries[queryid];
		if (terms.empty()) return 0;
		auto query_term_freqs = query_freqs(terms);

		std::vector<scored_enum> enums;
		enums.reserve(query_term_freqs.size());
		uint64_t tmpnum_docs = num_docs;

		Nodeforthread[threadid].clear();
		NodeforQuery[threadid].clear();
		int64_t tmpcount = 0;


#pragma omp critical(VisitsLRU)
		{
			for (auto term : query_term_freqs)
			{
				Node*node;
				bool flag = 0;
				node = global_LRUCache.Get(shardid, term.first, flag);
#pragma omp atomic
				node->aiodata.usedfreq++;
				NodeforQuery[threadid].push_back(node);
				if (flag == 0)
					Nodeforthread[threadid].push_back(node);
			}
		}

		aioReadFirstBlock();

		for (auto term : query_term_freqs) {
			AIOReadInfo tmpaio = NodeforQuery[threadid][tmpcount]->aiodata;
			enum_type list(Head_Data.data() + Head_offset[term.first], tmpaio.list_data + tmpaio.offsetForenums, num_docs, shardid, term.first, tmpcount);
			auto q_weight = scorer_type::query_term_weight
				(term.second, list.size(), tmpnum_docs);
			auto max_weight = q_weight * m_wdata.max_term_weight(term.first);
			wdata_enum w_enum(Block_Start[term.first], Block_Start[term.first + 1] - Block_Start[term.first], Block_Max_Term_Weight, Block_Docid);
			enums.push_back(scored_enum{ std::move(list), w_enum, q_weight, max_weight });
			tmpcount++;
		}

		for (unsigned i = 0; i < enums.size(); i++)
			enums[i].docs_enum.reset();

		std::vector<scored_enum *> ordered_enums;
		ordered_enums.reserve(enums.size());
		for (auto &en : enums) {
			ordered_enums.push_back(&en);
		}


		auto sort_enums = [&]() {
			std::sort(ordered_enums.begin(), ordered_enums.end(),
				[](scored_enum *lhs, scored_enum *rhs) {
				return lhs->docs_enum.docid() < rhs->docs_enum.docid();
			});
		};


		sort_enums();

		while (true) {

			float upper_bound = 0.f;
			size_t pivot;
			bool found_pivot = false;
			uint64_t pivot_id = num_docs;

			for (pivot = 0; pivot < ordered_enums.size(); ++pivot) {
				if (ordered_enums[pivot]->docs_enum.docid() == num_docs) {
					break;
				}

				upper_bound += ordered_enums[pivot]->max_weight;
				if (scoreQueue[threadid].would_enter(upper_bound)) {
					found_pivot = true;
					pivot_id = ordered_enums[pivot]->docs_enum.docid();
					for (; pivot + 1 < ordered_enums.size() &&
						ordered_enums[pivot + 1]->docs_enum.docid() == pivot_id; ++pivot);
						break;
				}
			}

			if (!found_pivot) {
				break;
			}

			double block_upper_bound = 0;

			for (size_t i = 0; i < pivot + 1; ++i) {
				if (ordered_enums[i]->w.docid() < pivot_id) {
					ordered_enums[i]->w.next_geq(pivot_id);
				}

				block_upper_bound += ordered_enums[i]->w.score() * ordered_enums[i]->q_weight;
			}


			if (scoreQueue[threadid].would_enter(block_upper_bound)) {


				if (pivot_id == ordered_enums[0]->docs_enum.docid()) {
					float score = 0;
					float norm_len = m_wdata.norm_len(pivot_id);

					for (scored_enum *en : ordered_enums) {
						if (en->docs_enum.docid() != pivot_id) {
							break;
						}
						float part_score = en->q_weight * scorer_type::doc_term_weight
							(en->docs_enum.freq(), norm_len);
						score += part_score;
						block_upper_bound -= en->w.score() * en->q_weight - part_score;
						if (!scoreQueue[threadid].would_enter(block_upper_bound)) {
							break;
						}

					}
					for (scored_enum *en : ordered_enums) {
						if (en->docs_enum.docid() != pivot_id) {
							break;
						}
						en->docs_enum.next();
					}

					scoreQueue[threadid].insert(score);
					sort_enums();

				}
				else {

					uint64_t next_list = pivot;
					for (; ordered_enums[next_list]->docs_enum.docid() == pivot_id;
						--next_list);
						ordered_enums[next_list]->docs_enum.next_geq(pivot_id);

					for (size_t i = next_list + 1; i < ordered_enums.size(); ++i) {
						if (ordered_enums[i]->docs_enum.docid() <=
							ordered_enums[i - 1]->docs_enum.docid()) {
							std::swap(ordered_enums[i], ordered_enums[i - 1]);
						}
						else {
							break;
						}
					}
				}

			}
			else {

				uint64_t next;
				uint64_t next_list = pivot;

				float q_weight = ordered_enums[next_list]->q_weight;


				for (uint64_t i = 0; i < pivot; i++){
					if (ordered_enums[i]->q_weight > q_weight){
						next_list = i;
						q_weight = ordered_enums[i]->q_weight;
					}
				}
				uint64_t next_jump = uint64_t(-2);

				if (pivot + 1 < ordered_enums.size()) {
					next_jump = ordered_enums[pivot + 1]->docs_enum.docid();
				}


				for (size_t i = 0; i <= pivot; ++i){
					if (ordered_enums[i]->w.docid() < next_jump)
						next_jump = std::min(ordered_enums[i]->w.docid(), next_jump);
				}

				next = next_jump + 1;
				if (pivot + 1 < ordered_enums.size()) {
					if (next > ordered_enums[pivot + 1]->docs_enum.docid()) {
						next = ordered_enums[pivot + 1]->docs_enum.docid();
					}
				}


				if (next <= ordered_enums[pivot]->docs_enum.docid()) {
					next = ordered_enums[pivot]->docs_enum.docid() + 1;
				}
				ordered_enums[next_list]->docs_enum.next_geq(next);


				for (size_t i = next_list + 1; i < ordered_enums.size(); ++i) {
					if (ordered_enums[i]->docs_enum.docid() <
						ordered_enums[i - 1]->docs_enum.docid()) {
						std::swap(ordered_enums[i], ordered_enums[i - 1]);
					}
					else {
						break;
					}
				}
			}
		}
		return scoreQueue[threadid].m_q.size();
	}

private:
	quasi_succinct::wand_data<scorer_type> const& m_wdata;
};

struct wand_query {

	wand_query(quasi_succinct::wand_data<scorer_type> const& wdata)
	: m_wdata(wdata)
	{}

	uint64_t operator()(unsigned shardid, unsigned queryid)
	{
		unsigned threadid = omp_get_thread_num();
		vector<unsigned>terms = queries[queryid];
		if (terms.empty()) return 0;

		auto query_term_freqs = query_freqs(terms);

		uint64_t tmpnum_docs = num_docs;

		std::vector<scored_enum> enums;
		enums.reserve(query_term_freqs.size());

		Nodeforthread[threadid].clear();
		NodeforQuery[threadid].clear();
		int64_t tmpcount = 0;

#pragma omp critical(VisitsLRU)
		{
			for (auto term : query_term_freqs)
			{
				Node*node;
				bool flag = 0;
				node = global_LRUCache.Get(shardid, term.first, flag);
#pragma omp atomic
				node->aiodata.usedfreq++;
				NodeforQuery[threadid].push_back(node);
				if (flag == 0)
					Nodeforthread[threadid].push_back(node);
			}
		}
		aioReadFirstBlock();

		for (auto term : query_term_freqs) {
			AIOReadInfo tmpaio = NodeforQuery[threadid][tmpcount]->aiodata;
			enum_type list(Head_Data.data() + Head_offset[term.first], tmpaio.list_data + tmpaio.offsetForenums, num_docs, shardid, term.first, tmpcount);
			auto q_weight = scorer_type::query_term_weight
				(term.second, list.size(), tmpnum_docs);
			auto max_weight = q_weight * m_wdata.max_term_weight(term.first);
			wdata_enum w_enum(Block_Start[term.first], Block_Start[term.first + 1] - Block_Start[term.first], Block_Max_Term_Weight, Block_Docid);
			enums.push_back(scored_enum{ std::move(list), w_enum, q_weight, max_weight });
			tmpcount++;
		}

		for (unsigned i = 0; i < enums.size(); i++)
			enums[i].docs_enum.reset();

		std::vector<scored_enum*> ordered_enums;
		ordered_enums.reserve(enums.size());
		for (auto& en : enums) {
			ordered_enums.push_back(&en);
		}

		auto sort_enums = [&]() {
			std::sort(ordered_enums.begin(), ordered_enums.end(),
				[](scored_enum* lhs, scored_enum* rhs) {
				return lhs->docs_enum.docid() < rhs->docs_enum.docid();
			});
		};
		sort_enums();
		while (true) {
			float upper_bound = 0;
			size_t pivot;
			bool found_pivot = false;
			for (pivot = 0; pivot < ordered_enums.size(); ++pivot) {
				if (ordered_enums[pivot]->docs_enum.docid() == tmpnum_docs) {
					break;
				}
				upper_bound += ordered_enums[pivot]->max_weight;
				if (scoreQueue[threadid].would_enter(upper_bound)) {
					found_pivot = true;
					break;
				}
			}
			if (!found_pivot) {
				break;
			}

			uint64_t pivot_id = ordered_enums[pivot]->docs_enum.docid();
			if (pivot_id == ordered_enums[0]->docs_enum.docid()) {
				float score = 0;
				float norm_len = m_wdata.norm_len(pivot_id);
				for (scored_enum* en : ordered_enums) {
					if (en->docs_enum.docid() != pivot_id) {
						break;
					}
					score += en->q_weight * scorer_type::doc_term_weight
						(en->docs_enum.freq(), norm_len);
					en->docs_enum.next();
				}

				scoreQueue[threadid].insert(score);
				sort_enums();
			}
			else {
				uint64_t next_list = pivot;
				for (; ordered_enums[next_list]->docs_enum.docid() == pivot_id;
					--next_list);
					ordered_enums[next_list]->docs_enum.next_geq(pivot_id);
				for (size_t i = next_list + 1; i < ordered_enums.size(); ++i) {
					if (ordered_enums[i]->docs_enum.docid() <
						ordered_enums[i - 1]->docs_enum.docid()) {
						std::swap(ordered_enums[i], ordered_enums[i - 1]);
					}
					else {
						break;
					}
				}
			}
		}
		return scoreQueue[threadid].m_q.size();
	}



private:
	quasi_succinct::wand_data<scorer_type> const& m_wdata;
};
struct maxscore_query {

	maxscore_query(quasi_succinct::wand_data<scorer_type> const& wdata)
	: m_wdata(wdata)
	{}

	uint64_t operator()(unsigned shardid, unsigned queryid)
	{
		unsigned threadid = omp_get_thread_num();
		vector<unsigned>terms = queries[queryid];
		if (terms.empty()) return 0;

		auto query_term_freqs = query_freqs(terms);

		uint64_t tmpnum_docs = num_docs;

		std::vector<scored_enum> enums;
		enums.reserve(query_term_freqs.size());

		Nodeforthread[threadid].clear();
		NodeforQuery[threadid].clear();
		int64_t tmpcount = 0;

#pragma omp critical(VisitsLRU)
		{
			for (auto term : query_term_freqs)
			{
				Node*node;
				bool flag = 0;
				node = global_LRUCache.Get(shardid, term.first, flag);
#pragma omp atomic
				node->aiodata.usedfreq++;
				NodeforQuery[threadid].push_back(node);
				if (flag == 0)
					Nodeforthread[threadid].push_back(node);
			}
		}
		aioReadFirstBlock();

		for (auto term : query_term_freqs) {
			AIOReadInfo tmpaio = NodeforQuery[threadid][tmpcount]->aiodata;
			enum_type list(Head_Data.data() + Head_offset[term.first], tmpaio.list_data + tmpaio.offsetForenums, num_docs, shardid, term.first, tmpcount);
			auto q_weight = scorer_type::query_term_weight
				(term.second, list.size(), tmpnum_docs);
			auto max_weight = q_weight * m_wdata.max_term_weight(term.first);
			wdata_enum w_enum(Block_Start[term.first], Block_Start[term.first + 1] - Block_Start[term.first], Block_Max_Term_Weight, Block_Docid);
			enums.push_back(scored_enum{ std::move(list), w_enum, q_weight, max_weight });
			tmpcount++;
		}

		for (unsigned i = 0; i < enums.size(); i++)
			enums[i].docs_enum.reset();

		std::vector<scored_enum*> ordered_enums;
		ordered_enums.reserve(enums.size());
		for (auto& en : enums) {
			ordered_enums.push_back(&en);
		}

		std::sort(ordered_enums.begin(), ordered_enums.end(),
			[](scored_enum* lhs, scored_enum* rhs) {
			return lhs->max_weight < rhs->max_weight;
		});

		std::vector<float> upper_bounds(ordered_enums.size());
		upper_bounds[0] = ordered_enums[0]->max_weight;
		for (size_t i = 1; i < ordered_enums.size(); ++i) {
			upper_bounds[i] = upper_bounds[i - 1] + ordered_enums[i]->max_weight;
		}

		uint64_t non_essential_lists = 0;
		uint64_t cur_doc =
			std::min_element(enums.begin(), enums.end(),
			[](scored_enum const& lhs, scored_enum const& rhs) {
			return lhs.docs_enum.docid() < rhs.docs_enum.docid();
		})
			->docs_enum.docid();

		while (non_essential_lists < ordered_enums.size() &&
			cur_doc < tmpnum_docs) {
			float score = 0;
			float norm_len = m_wdata.norm_len(cur_doc);
			uint64_t next_doc = tmpnum_docs;
			for (size_t i = non_essential_lists; i < ordered_enums.size(); ++i) {
				if (ordered_enums[i]->docs_enum.docid() == cur_doc) {
					score += ordered_enums[i]->q_weight * scorer_type::doc_term_weight
						(ordered_enums[i]->docs_enum.freq(), norm_len);
					ordered_enums[i]->docs_enum.next();
				}
				if (ordered_enums[i]->docs_enum.docid() < next_doc) {
					next_doc = ordered_enums[i]->docs_enum.docid();
				}
			}

			for (size_t i = non_essential_lists - 1; i + 1 > 0; --i) {
				if (!scoreQueue[threadid].would_enter(score + upper_bounds[i])) {
					break;
				}
				ordered_enums[i]->docs_enum.next_geq(cur_doc);
				if (ordered_enums[i]->docs_enum.docid() == cur_doc) {
					score += ordered_enums[i]->q_weight * scorer_type::doc_term_weight
						(ordered_enums[i]->docs_enum.freq(), norm_len);
				}
			}

			if (scoreQueue[threadid].insert(score)) {
				while (non_essential_lists < ordered_enums.size() &&
					!scoreQueue[threadid].would_enter(upper_bounds[non_essential_lists])) {
					non_essential_lists += 1;
				}
			}

			cur_doc = next_doc;
		}

		return scoreQueue[threadid].m_q.size();
	}


private:
	quasi_succinct::wand_data<scorer_type> const& m_wdata;
};
struct ranked_and_query {

	ranked_and_query(quasi_succinct::wand_data<scorer_type> const& wdata)
	: m_wdata(wdata)
	{}

	uint64_t operator()(unsigned shardid, unsigned queryid)
	{
		unsigned threadid = omp_get_thread_num();
		vector<unsigned>terms = queries[queryid];
		if (terms.empty()) return 0;
		auto query_term_freqs = query_freqs(terms);

		uint64_t tmpnum_docs = num_docs;

		std::vector<scored_enum> enums;
		enums.reserve(query_term_freqs.size());

		Nodeforthread[threadid].clear();
		NodeforQuery[threadid].clear();
		int64_t tmpcount = 0;

#pragma omp critical(VisitsLRU)
		{
			for (auto term : query_term_freqs)
			{
				Node*node;
				bool flag = 0;
				node = global_LRUCache.Get(shardid, term.first, flag);
#pragma omp atomic
				node->aiodata.usedfreq++;
				NodeforQuery[threadid].push_back(node);
				if (flag == 0)
					Nodeforthread[threadid].push_back(node);
			}
		}
		aioReadFirstBlock();

		for (auto term : query_term_freqs) {
			AIOReadInfo tmpaio = NodeforQuery[threadid][tmpcount]->aiodata;
			enum_type list(Head_Data.data() + Head_offset[term.first], tmpaio.list_data + tmpaio.offsetForenums, num_docs, shardid, term.first, tmpcount);
			auto q_weight = scorer_type::query_term_weight
				(term.second, list.size(), tmpnum_docs);
			auto max_weight = q_weight * m_wdata.max_term_weight(term.first);
			wdata_enum w_enum(Block_Start[term.first], Block_Start[term.first + 1] - Block_Start[term.first], Block_Max_Term_Weight, Block_Docid);
			enums.push_back(scored_enum{ std::move(list), w_enum, q_weight, max_weight });
			tmpcount++;
		}

		for (unsigned i = 0; i < enums.size(); i++)
			enums[i].docs_enum.reset();

		std::vector<scored_enum*> ordered_enums;
		ordered_enums.reserve(enums.size());
		for (auto& en : enums) {
			ordered_enums.push_back(&en);
		}

		std::sort(ordered_enums.begin(), ordered_enums.end(),
			[](scored_enum* lhs, scored_enum* rhs) {
			return lhs->docs_enum.size() < rhs->docs_enum.size();
		});

		uint64_t candidate = ordered_enums[0]->docs_enum.docid();
		size_t i = 1;
		while (candidate < tmpnum_docs) {
			for (; i < ordered_enums.size(); ++i) {
				ordered_enums[i]->docs_enum.next_geq(candidate);
				if (ordered_enums[i]->docs_enum.docid() != candidate) {
					candidate = ordered_enums[i]->docs_enum.docid();
					i = 0;
					break;
				}
			}

			if (i == ordered_enums.size()) {
				float norm_len = m_wdata.norm_len(candidate);
				float score = 0;
				for (i = 0; i < ordered_enums.size(); ++i) {
					score += ordered_enums[i]->q_weight * scorer_type::doc_term_weight
						(ordered_enums[i]->docs_enum.freq(), norm_len);
				}

				scoreQueue[threadid].insert(score);
				ordered_enums[0]->docs_enum.next();
				candidate = ordered_enums[0]->docs_enum.docid();
				i = 1;
			}
		}

		return scoreQueue[threadid].m_q.size();
	}

private:
	quasi_succinct::wand_data<scorer_type> const& m_wdata;
};

void initial_data()
{
	string filename = indexFileName;
	read_Head_Length(filename);
	read_Head_Data(filename);
	read_List_Length(filename);
	read_BlockWand_Data(filename);
	read_SplitInfo(filename);
	read_Endpoint(filename);
	read_query("");
	scoreQueue.resize(threadCount + 1);
	queryLock.resize(queries.size());
	globalScoreThresh.resize(queries.size());
	for (unsigned i = 0; i < queries.size(); i++)
		omp_init_lock(&queryLock[i]);
	initThreadStruct();

	read_AssignedThreads();
	read_ShardScore();

	read_TermtopKThresh("");
	read_TermTopkThreshShard("");
}
void print_statistics(string querytype)
{
	vector<double>query_times;
	for (unsigned i = 0; i < queryInfo.size(); i++)
		query_times.push_back(queryInfo[i].time);

	cout << "query times count=" << query_times.size() << endl;
	std::sort(query_times.begin(), query_times.end());
	double avg = std::accumulate(query_times.begin(), query_times.end(), double()) / query_times.size();
	double q25 = query_times[25 * query_times.size() / 100];
	double q50 = query_times[query_times.size() / 2];
	double q75 = query_times[75 * query_times.size() / 100];
	double q95 = query_times[95 * query_times.size() / 100];
	double q99 = query_times[99 * query_times.size() / 100];
	double q999 = query_times[999 * query_times.size() / 1000];
	cout << "----------" + querytype + "----------" << endl;
	cout << "Mean: " << avg << std::endl;
	cout << "25% quantile: " << q25 << std::endl;
	cout << "50% quantile: " << q50 << std::endl;
	cout << "75% quantile: " << q75 << std::endl;
	cout << "95% quantile: " << q95 << std::endl;
	cout << "99% quantile: " << q99 << std::endl;
	cout << "99.9% quantile: " << q999 << std::endl;
}
inline double get_time_usecs() {
	timeval tv;
	gettimeofday(&tv, NULL);
	return double(tv.tv_sec) * 1000000 + double(tv.tv_usec);
}

template <class T>
inline void do_not_optimize_away(T&& datum) {
	asm volatile("" : "+r" (datum));
}

unsigned assignedThreadNum(unsigned qid)
{
	return assignedThreads[qid];
}
bool cmpShardScore(const pair<unsigned, float>&a, const pair<unsigned, float>&b)
{
	return a.second > b.second;
}
void cal_FragScore(unsigned queryid)
{
	vector<uint8_t>fake_list_data(1024);
	for (unsigned f = 0; f < constShardcount; f++)
	{
		float avgmaxScore = 0, avgminScore = 0, avgScore = 0, topKScore = 0;

		auto terms = query_freqs(queries[queryid]);
		for (auto term : terms)
		{
			enum_type list(Head_Data.data() + Head_offset[term.first], fake_list_data.data(), num_docs, 0, term.first, 0);
			auto q_weight = scorer_type::query_term_weight(term.second, list.size(), num_docs);

			float sumscore = 0, blockcount = 0, maxscore = 0, minscore = INT_MAX;
			unsigned endblock = f == constShardcount - 1 ? Block_Start[term.first + 1] - Block_Start[term.first] : splitInfo[f + 1][term.first].split_blockid;
			unsigned startblockpos = Block_Start[term.first];
			for (unsigned b = splitInfo[f][term.first].split_blockid; b < endblock; b++)
			{
				float score = Block_Max_Term_Weight[startblockpos + b] * q_weight;
				sumscore += score;
				blockcount++;
				maxscore = max(maxscore, score);
				minscore = min(minscore, score);
			}
			avgmaxScore += maxscore;
			avgminScore += minscore;
			avgScore += (sumscore / blockcount);
		}

		for (auto t : terms)
		{
			topKScore = max(topKScore, TermTopkThreshShard[t.first][f] * t.second);
		}

		ShardScore[queryid][f].first = f;
		ShardScore[queryid][f].second = topKScore;
	}
}
void assignedQuery()
{
	for (unsigned i = 0; i < queries.size(); i++)
	{
		unsigned tasknum = assignedThreadNum(i);
		unsigned tasksize = ceil((double)constShardcount / tasknum);
#pragma omp critical(QueryInfo)
		{queryInfo[i].threadcount = tasknum; }
		vector<vector<unsigned>>shards(tasknum);

		cal_FragScore(i);
		sort(ShardScore[i].begin(), ShardScore[i].end(), cmpShardScore);
		for (unsigned t = 0; t < constShardcount; t++)
		{
			shards[t%tasknum].push_back(ShardScore[i][t].first);
			
		}

		Task task;
		task.queryid = i;
		for (unsigned t = 0; t < tasknum; t++)
		{
			task.shardids = shards[t];
#pragma omp critical(TaskQueue)
			{queryTasks.push(task); }
		}
	}
	cout << "assigned over" << endl;
}

void receiveLastRequest(unsigned qid)
{
	vector<unsigned>query = queries[qid];
	int threadid = omp_get_thread_num();
	while (io_getevents(ctx[threadid], curAIOLIST_SIZE[threadid], MAXREQUEST, events[threadid].data(), NULL) != curAIOLIST_SIZE[threadid])
	{
		;
	}
	for (unsigned i = 0; i < Nodeforthread[threadid].size(); i++)
	{
		Nodeforthread[threadid][i]->aiodata.curReadpos = Nodeforthread[threadid][i]->aiodata.curSendpos;
#pragma omp flush(Nodeforthread)
	}
	while (1)
	{
		int64_t cur = 0, i = 0;
		for (i = 0, cur = 0; cur < MAXREQUEST&& Nodeforthread[threadid].size(); curReadID[threadid]++, i++)
		{
			unsigned lid = curReadID[threadid] % Nodeforthread[threadid].size();
			if (Nodeforthread[threadid][lid]->aiodata.listlength <= Nodeforthread[threadid][lid]->aiodata.curSendpos)
			{
				if (hasReadFinish())break;
				else continue;
			}
			io_prep_pread(&readrequest[threadid][cur], IndexFile, Nodeforthread[threadid][lid]->aiodata.list_data + Nodeforthread[threadid][lid]->aiodata.memoffset, Nodeforthread[threadid][lid]->aiodata.readblocksize, Nodeforthread[threadid][lid]->aiodata.readoffset);
			listrequest[threadid][cur] = &readrequest[threadid][cur];
			Nodeforthread[threadid][lid]->aiodata.memoffset += Nodeforthread[threadid][lid]->aiodata.readblocksize;
			Nodeforthread[threadid][lid]->aiodata.readoffset += Nodeforthread[threadid][lid]->aiodata.readblocksize;
			Nodeforthread[threadid][lid]->aiodata.curSendpos += Nodeforthread[threadid][lid]->aiodata.readblocksize;
#pragma omp critical(bandwidth)
			{
				curBandwidth += Nodeforthread[threadid][lid]->aiodata.readblocksize;
#pragma omp flush(curBandwidth)
			}
			cur++;
		}
		curAIOLIST_SIZE[threadid] = cur;
		if (cur == 0)break;
		if (io_submit(ctx[threadid], curAIOLIST_SIZE[threadid], listrequest[threadid].data()) != curAIOLIST_SIZE[threadid]) {
			perror("io_submit"); cout << "submit 1 inreceiveLastRequest" << endl;
		}
		while (io_getevents(ctx[threadid], curAIOLIST_SIZE[threadid], MAXREQUEST, events[threadid].data(), NULL) != curAIOLIST_SIZE[threadid])
		{
			;
		}
		for (unsigned i = 0; i < Nodeforthread[threadid].size(); i++)
		{
			Nodeforthread[threadid][i]->aiodata.curReadpos = Nodeforthread[threadid][i]->aiodata.curSendpos;
#pragma omp flush(Nodeforthread)
		}
	}
	remove_duplicate_terms(query);
	for (unsigned i = 0; i < NodeforQuery[threadid].size(); i++)
	{
#pragma omp atomic
		NodeforQuery[threadid][i]->aiodata.usedfreq--;
#pragma omp flush(Nodeforthread)
	}
}

struct ListScore
{
	double score;
	unsigned termid;
};
bool cmpScore(ListScore const &a, ListScore const &b)
{
	return a.score > b.score;
}
void warmUpLRUCache()
{
	vector<ListScore>listscore(List_offset.size() - 1);
	for (unsigned i = 0; i < listscore.size(); i++)
	{
		listscore[i].score = List_offset[i + 1] - List_offset[i];
		listscore[i].termid = i;
	}

	sort(listscore.begin(), listscore.end(), cmpScore);

	int64_t off = 0, i;
	for (i = 0; i < listscore.size(); i++)
	{
		if (off >= CACHE_SIZE)break;
		for (unsigned j = 0; j < constShardcount; j++)
		{
			unsigned term = listscore[i].termid;
			bool flag = 0;
			Node *node = global_LRUCache.Get(j, term, flag);
			if (flag == 1)cout << "global_LRUCache.Get wrong" << endl;
			pread(IndexFile, node->aiodata.list_data, node->aiodata.readlength, node->aiodata.readoffset);
			node->aiodata.curSendpos = node->aiodata.listlength;
			node->aiodata.memoffset = node->aiodata.readlength;
			node->aiodata.readoffset = node->aiodata.readlength;
			node->aiodata.curReadpos = node->aiodata.listlength;
			off += node->aiodata.readlength;
			if (off >= CACHE_SIZE)break;
		}
	}
	cout << "All we cache " << i << " lists, account for " << CACHE_SIZE / 1024 / 1024 << " MB" << endl;
}

template <typename QueryOperator>
void perform_query(QueryOperator&& query_op, Task task)
{
	auto tick = get_time_usecs();
	unsigned result = 0;
	unsigned threadid = omp_get_thread_num();
	scoreQueue[threadid].clear(task.queryid);
	for (unsigned t = 0; t < task.shardids.size(); t++)
	{
		result += query_op(task.shardids[t], task.queryid);
		receiveLastRequest(task.queryid);
	}
#pragma omp critical(QueryInfo)
	{
#pragma omp flush(queryInfo)
		queryInfo[task.queryid].resultcount += result;
		queryInfo[task.queryid].scoreHeap.insert(queryInfo[task.queryid].scoreHeap.end(), scoreQueue[threadid].m_q.begin(), scoreQueue[threadid].m_q.end());
		if (queryInfo[task.queryid].time == 0 || queryInfo[task.queryid].time>tick)queryInfo[task.queryid].time = tick;
		queryInfo[task.queryid].threadcount--;
#pragma omp flush(queryInfo)
		if (queryInfo[task.queryid].threadcount == 0){
			sort(queryInfo[task.queryid].scoreHeap.rbegin(), queryInfo[task.queryid].scoreHeap.rend());
			queryInfo[task.queryid].scoreHeap.resize(min(topK, (unsigned)queryInfo[task.queryid].scoreHeap.size()));
			queryInfo[task.queryid].scoreHeap.shrink_to_fit();
			auto end = get_time_usecs();
			queryInfo[task.queryid].time = end - queryInfo[task.queryid].time;
			curQID++;
#pragma omp flush(curQID)
		}
	}
}

template <typename QueryOperator>
void ProcessQuery(vector<QueryOperator>&queryop)
{
	while (curQID < queries.size())
	{
		Task task;
		task.queryid = INT_MAX;
#pragma omp critical(TaskQueue)
		{
			if (!queryTasks.empty())
			{
				task = queryTasks.front();
				queryTasks.pop();
			}
		}
		if (task.queryid != INT_MAX)
		{
			perform_query(queryop[task.queryid], task);
		}
#pragma omp flush(curQID)
	}
}


void writeResult(string querytype, vector<and_query>&queryop)
{
	ofstream fout(indexFileName + querytype + "_result" + ".txt");
	for (unsigned i = 0; i < queryInfo.size(); i++)
		fout << queryInfo[i].resultcount << endl;
	fout.close();
}
void writeResult(string querytype)
{
	ofstream fout(indexFileName + querytype + "_result" + ".txt");
	for (unsigned i = 0; i < queryInfo.size(); i++)
	{
		for (unsigned j = 0; j < queryInfo[i].scoreHeap.size(); j++)
			fout << queryInfo[i].scoreHeap[j] << " ";
		fout << endl;
	}
	fout.close();
}

int main(int argc, char *argv[])
{
	indexFileName = argv[1];
	string querytype = argv[2];
	threadCount = atoi(argv[3]);
	CACHE_SIZE *= atoi(argv[4]);
  queryFileNo=atoi(argv[4]);
	threadCount += 3;
	IndexFile = open((indexFileName + ".list").c_str(), O_RDONLY | O_DIRECT);
	initial_data();
	cout << "initial data over" << endl;
	warmUpLRUCache();
	cout << "Warm Up over" << endl;

	if (querytype == "AND")
	{
		vector<and_query>queryop(queries.size());
		auto tick = get_time_usecs();
#pragma omp parallel for num_threads(threadCount) schedule(dynamic)
		for (unsigned i = 0; i < threadCount; i++)
		{
			if (i == 0)assignedQuery();
			else if (i == 1)detectIOWork();
			else if (i == 2)readIOWork();
			else ProcessQuery(queryop);
		}
		double elapsed = double(get_time_usecs() - tick);
		cout << "Performed AND query, sum time=" << elapsed << "throught put=" << queries.size() / elapsed * 1000000 << endl;
		writeResult("AND", queryop);
	}
	else
	{
		quasi_succinct::wand_data<> wdata;
		boost::iostreams::mapped_file_source md(indexFileName + ".wand");
		succinct::mapper::map(wdata, md, succinct::mapper::map_flags::warmup);
		if (querytype == "MAXSCORE")
		{
			vector<maxscore_query>queryop;
			for (unsigned i = 0; i < queries.size(); i++)
			{
				queryop.push_back(maxscore_query(wdata));
			}

			auto tick = get_time_usecs();
#pragma omp parallel for num_threads(threadCount) schedule(dynamic)
			for (unsigned i = 0; i < threadCount; i++)
			{
				if (i == 0)assignedQuery();
				else if (i == 1)detectIOWork();
				else if (i == 2)readIOWork();
				else ProcessQuery(queryop);
			}
			double elapsed = double(get_time_usecs() - tick);
			cout << "Performed MAXSCORE query, sum time=" << elapsed << "throught put=" << queries.size() / elapsed * 1000000 << endl;
			writeResult("MAXSCORE");
		}
		else if (querytype == "WAND")
		{
			vector<wand_query>queryop;
			for (unsigned i = 0; i < queries.size(); i++)
			{
				queryop.push_back(wand_query(wdata));
			}

			auto tick = get_time_usecs();
#pragma omp parallel for num_threads(threadCount) schedule(dynamic)
			for (unsigned i = 0; i < threadCount; i++)
			{
				if (i == 0)assignedQuery();
				else if (i == 1)detectIOWork();
				else if (i == 2)readIOWork();
				else ProcessQuery(queryop);
			}
			double elapsed = double(get_time_usecs() - tick);
			cout << "Performed WAND query, sum time=" << elapsed << "throught put=" << queries.size() / elapsed * 1000000 << endl;
			writeResult("WAND");
		}
		else if (querytype == "RANKAND")
		{
			vector<ranked_and_query>queryop;
			for (unsigned i = 0; i < queries.size(); i++)
			{
				queryop.push_back(ranked_and_query(wdata));
			}

			auto tick = get_time_usecs();
#pragma omp parallel for num_threads(threadCount) schedule(dynamic)
			for (unsigned i = 0; i < threadCount; i++)
			{
				if (i == 0)assignedQuery();
				else if (i == 1)detectIOWork();
				else if (i == 2)readIOWork();
				else ProcessQuery(queryop);
			}
			double elapsed = double(get_time_usecs() - tick);
			cout << "Performed RANKAND query, sum time=" << elapsed << "throught put=" << queries.size() / elapsed * 1000000 << endl;
			writeResult("RANKAND");
		}
		else if (querytype == "BMW")
		{
			vector<block_max_wand_query>queryop;
			for (unsigned i = 0; i < queries.size(); i++)
			{
				queryop.push_back(block_max_wand_query(wdata));
			}

			auto tick = get_time_usecs();
#pragma omp parallel for num_threads(threadCount) schedule(dynamic)
			for (unsigned i = 0; i < threadCount; i++)
			{
				if (i == 0)assignedQuery();
				else if (i == 1)detectIOWork();
				else if (i == 2)readIOWork();
				else ProcessQuery(queryop);
			}
			double elapsed = double(get_time_usecs() - tick);
			cout << "Performed BMW query, sum time=" << elapsed << "throught put=" << queries.size() / elapsed * 1000000 << endl;
		}
	}

	print_statistics(querytype);
	cout << "miss size=" << global_LRUCache.miss_size << endl;
	for (unsigned i = 0; i < threadCount; i++)
		io_destroy(ctx[i]);
}
