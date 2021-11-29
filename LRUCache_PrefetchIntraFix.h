#include <iostream>
#include<stdint.h>
#include <unordered_map>
#include <vector>
using namespace std;

struct SplitInfo
{
	unsigned split_blockid;
	SplitInfo(){ split_blockid = 0; }
};

const unsigned constShardcount = 8;
vector<int64_t>List_offset;
vector<vector<SplitInfo>>splitInfo(constShardcount);
vector<vector<unsigned>>blockEndpoint;
struct AIOReadInfo
{
	int64_t readlength;
	int64_t readoffset;
	int64_t listlength;
	int64_t offsetForenums;
	int64_t memoffset;
	int64_t curSendpos;
	int64_t curReadpos;
	uint8_t usedfreq;
	uint8_t *list_data;
	uint32_t termid;
	uint32_t shardid;
	uint32_t readblocksize;
};
vector<int64_t>READ_BLOCK_options = { 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024 };
const uint64_t DISK_BLOCK = 4096;

struct Node{
	AIOReadInfo aiodata;
	Node*prev, *next;
};
int64_t CACHE_SIZE = 1024 * 1024;

class LRUCache{
public:
	LRUCache();
	~LRUCache();
	Node* Put(unsigned shard, unsigned key);
	Node* Get(unsigned shard, unsigned key, bool& flag);

	Node* Put_Prefetch(unsigned shard, unsigned key);
	Node* Get_Prefetch(unsigned shard, unsigned key, bool& flag);


	void print(unsigned shardid);
	uint64_t hit_size;
	uint64_t miss_size;
	uint64_t hit_count;
	uint64_t miss_count;

	void attach(Node *node);
	void detach(Node *node);
	AIOReadInfo calAioreadinfo(unsigned shard, unsigned term);

	vector<unordered_map<unsigned, Node*>>hashmap_;
	Node*head_, *tail_;
	int64_t sumBytes;/
};

LRUCache::LRUCache()
{
	hashmap_.resize(constShardcount);
	miss_size = 0; hit_size = 0;
	miss_count = 0; hit_count = 0;
	head_ = new Node;
	tail_ = new Node;
	head_->prev = NULL;
	head_->next = tail_;
	tail_->prev = head_;
	tail_->next = NULL;
	sumBytes = 0;
}

LRUCache::~LRUCache()
{
	delete head_;
	delete tail_;
}
int64_t cal_properReadBlockSize(int64_t length)
{
	return READ_BLOCK_options[READ_BLOCK_options.size() - 1];
}
AIOReadInfo LRUCache::calAioreadinfo(unsigned shard, unsigned term)
{
	AIOReadInfo tmpaio;
	tmpaio.termid = term;
	tmpaio.shardid = shard;
	int64_t startpos = splitInfo[shard][term].split_blockid == 0 ? 0 : blockEndpoint[term][splitInfo[shard][term].split_blockid - 1];
	int64_t listlength = shard == constShardcount - 1 ? List_offset[term + 1] - List_offset[term] - startpos : blockEndpoint[term][splitInfo[shard + 1][term].split_blockid] - startpos;
	tmpaio.listlength = listlength;
	tmpaio.memoffset = 0;
	int64_t offset = startpos + List_offset[term]; 
	tmpaio.readoffset = ((int64_t)(offset / DISK_BLOCK))*DISK_BLOCK;
	tmpaio.offsetForenums = offset - tmpaio.readoffset;
	tmpaio.readblocksize = cal_properReadBlockSize(tmpaio.listlength);
	int64_t readlength = ((int64_t)(ceil((double)(listlength + tmpaio.offsetForenums) / tmpaio.readblocksize)))*tmpaio.readblocksize;
	tmpaio.readlength = readlength;
	tmpaio.curSendpos = -tmpaio.offsetForenums;
	tmpaio.usedfreq = 0;
	tmpaio.curReadpos = -tmpaio.offsetForenums;
	miss_size += tmpaio.listlength;
	return tmpaio;
}

Node* LRUCache::Put(unsigned shard, unsigned key)
{
	AIOReadInfo tmpaio = calAioreadinfo(shard, key);
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << tmpaio.readlength << " " << CACHE_SIZE << " listlength=" << List_offset[key + 1] - List_offset[key] << endl;
		cout << "That block overflow!!" << endl;
		return NULL;
	}
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE)
	{
		if (node == head_){ node = tail_->prev; }
		if (node->aiodata.usedfreq > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);
		node->aiodata.curReadpos = -node->aiodata.offsetForenums; 
		sumBytes -= node->aiodata.readlength;
		hashmap_[node->aiodata.shardid].erase(node->aiodata.termid);
		Node *tmp = node->prev;
		delete node;
		node = tmp;
	}
	node = new Node();
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, tmpaio.readlength);
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node); 
	hashmap_[shard][key] = node;
	return node;
}

Node* LRUCache::Put_Prefetch(unsigned shard, unsigned key)
{
	AIOReadInfo tmpaio = calAioreadinfo(shard, key);
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << "That block overflow!!" << endl;
		return NULL;
	}
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE&&node != head_)
	{
		if (node->aiodata.usedfreq > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);
		node->aiodata.curReadpos = -node->aiodata.offsetForenums; 
		sumBytes -= node->aiodata.readlength;
		hashmap_[node->aiodata.shardid].erase(node->aiodata.termid);

		Node *tmp = node->prev;
		delete node;
		node = tmp;
	}
	if (node == head_)
	{
		return NULL;
	}

	node = new Node();
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, tmpaio.readlength);
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node); 
	hashmap_[shard][key] = node;
	return node;
}
Node* LRUCache::Get(unsigned shard, unsigned key, bool &flag)
{
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_[shard].find(key);
	if (it != hashmap_[shard].end())
	{
		node = it->second;
		flag = true;
		hit_count++;
		detach(node);
		attach(node);
	}
	else
	{
		flag = false;
		miss_count++;
		node = Put(shard, key);
	}
	return node;
}
Node* LRUCache::Get_Prefetch(unsigned shard, unsigned key, bool &flag)
{
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_[shard].find(key);
	if (it != hashmap_[shard].end())
	{
		node = it->second;
		flag = true;
		detach(node);
		attach(node);
	}
	else
	{
		flag = false;
		miss_count++;
		node = Put_Prefetch(shard, key);
	}
	return node;
}

void LRUCache::attach(Node *node)
{
	node->next = head_->next;
	head_->next = node;
	node->next->prev = node;
	node->prev = head_;
}


void LRUCache::detach(Node *node)
{
	node->prev->next = node->next;
	node->next->prev = node->prev;
}

void LRUCache::print(unsigned shard)
{
	unordered_map<unsigned, Node* >::iterator iter;
	int64_t mysumsize = 0;
	for (iter = hashmap_[shard].begin(); iter != hashmap_[shard].end(); iter++)
	{
		mysumsize += iter->second->aiodata.listlength;
	}
	cout << "sumsize=" << mysumsize << endl;
}
