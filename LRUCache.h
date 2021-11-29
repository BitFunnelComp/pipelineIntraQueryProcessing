#include <iostream>
#include<stdint.h>
#include <unordered_map>
#include <vector>
using namespace std;

vector<int64_t>List_offset;

struct AIOReadInfo
{
	int64_t readlength;
	int64_t readoffset;
	int64_t listlength;
	int64_t offsetForenums;
	int64_t memoffset;
	int64_t curSendpos;
	uint8_t *list_data;
	uint32_t termid;
};
vector<int64_t>curReadpos;
vector<int64_t>usedFreq;
const uint64_t DISK_BLOCK = 4096;
const int64_t READ_BLOCK = 64 * 1024; 

struct Node{
	AIOReadInfo aiodata;
	Node*prev, *next;
};
int64_t CACHE_SIZE = 1024 * 1024;

class LRUCache{
public:
	LRUCache();
	~LRUCache();
	Node* Put(unsigned key);
	Node* Get(unsigned key, bool& flag);
	void print();
	uint64_t hit_size;
	uint64_t miss_size;
	uint64_t hit_count;
	uint64_t miss_count;

	void attach(Node *node);
	void detach(Node *node);
	AIOReadInfo calAioreadinfo(unsigned term);

	unordered_map<unsigned, Node*>hashmap_;
	Node*head_, *tail_;
	int64_t sumBytes;
};

LRUCache::LRUCache()
{
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
AIOReadInfo LRUCache::calAioreadinfo(unsigned term)
{
	AIOReadInfo tmpaio;
	tmpaio.termid = term; 
	int64_t listlength = List_offset[term + 1] - List_offset[term];
	tmpaio.listlength = listlength;
	tmpaio.memoffset = 0;
	int64_t offset = List_offset[term]; 
	tmpaio.readoffset = ((int64_t)(offset / DISK_BLOCK))*DISK_BLOCK;
	tmpaio.offsetForenums = offset - tmpaio.readoffset;
	int64_t readlength = ((int64_t)(ceil((double)(listlength + tmpaio.offsetForenums) / READ_BLOCK)))*READ_BLOCK;
	tmpaio.readlength = readlength;
	tmpaio.curSendpos = -tmpaio.offsetForenums;
	curReadpos[term] = -tmpaio.offsetForenums;
#pragma omp flush(curReadpos)
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, readlength);
	miss_size += tmpaio.listlength;
	return tmpaio;
}

Node* LRUCache::Put(unsigned key)
{
	AIOReadInfo tmpaio = calAioreadinfo(key); 
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << "That block overflow!!" << endl;
		return NULL;
	}
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE)
	{
		if (node == head_){ node = tail_->prev; }
#pragma omp flush(usedFreq)
		if (usedFreq[node->aiodata.termid] > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);
		curReadpos[node->aiodata.termid] = node->aiodata.offsetForenums;

		sumBytes -= node->aiodata.readlength;
		hashmap_.erase(node->aiodata.termid);

		Node *tmp = node->prev;
		delete node;
		node = tmp;
	}
	node = new Node();
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node);
	hashmap_[key] = node;
	return node;
}

Node* LRUCache::Get(unsigned key, bool &flag)
{
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_.find(key);
	if (it != hashmap_.end())
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
		node = Put(key);
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

void LRUCache::print()
{
	unordered_map<unsigned, Node* >::iterator iter;
	int64_t mysumsize = 0;
	for (iter = hashmap_.begin(); iter != hashmap_.end(); iter++)
	{
		mysumsize += iter->second->aiodata.listlength;
	}
	cout << "sumsize=" << mysumsize << endl;
}
