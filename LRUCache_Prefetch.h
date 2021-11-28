#include <iostream>
#include<stdint.h>
#include <unordered_map>
#include <vector>
using namespace std;

vector<int64_t>List_offset;

struct AIOReadInfo
{
	int64_t readlength;//读长度（4K对齐）
	int64_t readoffset;//读偏移（4K对齐）
	//int64_t listoffset;//实际偏移
	int64_t listlength;//实际长度
	int64_t offsetForenums;
	int64_t memoffset;
	int64_t curSendpos;
	//int64_t usedfreq;
	uint8_t *list_data;//数据部分
	uint32_t termid;
};
vector<int64_t>curReadpos;
vector<int64_t>usedFreq;
//vector<AIOReadInfo>AIOreadinfo;
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

	Node* Put_Prefetch(unsigned key);
	Node* Get_Prefetch(unsigned key, bool& flag);


	void print();
	uint64_t hit_size;
	uint64_t miss_size;
	uint64_t hit_count;
	uint64_t miss_count;

	void attach(Node *node);//插入到链表头
	void detach(Node *node);//链表中删除节点
	AIOReadInfo calAioreadinfo(unsigned term);//计算Aio结构数据

	unordered_map<unsigned, Node*>hashmap_;//hash表<termid,链表节点>
	Node*head_, *tail_;//头节点尾节点
	int64_t sumBytes;//当前内存占用字节数
};

LRUCache::LRUCache()//construct
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

LRUCache::~LRUCache()//destruct
{
	delete head_;
	delete tail_;
}
AIOReadInfo LRUCache::calAioreadinfo(unsigned term)
{
	AIOReadInfo tmpaio;
	tmpaio.termid = term; //cout << "LRU term=" << term << endl;
	int64_t listlength = List_offset[term + 1] - List_offset[term];
	tmpaio.listlength = listlength;
	tmpaio.memoffset = 0;
	int64_t offset = List_offset[term]; //cout << "term " << term << " offset=" << offset << endl;
	tmpaio.readoffset = ((int64_t)(offset / DISK_BLOCK))*DISK_BLOCK;
	tmpaio.offsetForenums = offset - tmpaio.readoffset;
	int64_t readlength = ((int64_t)(ceil((double)(listlength + tmpaio.offsetForenums) / READ_BLOCK)))*READ_BLOCK;//4K对齐
	tmpaio.readlength = readlength;
	tmpaio.curSendpos = -tmpaio.offsetForenums;
	//tmpaio.usedfreq++;//usedFreq怎么处理？
	curReadpos[term] = -tmpaio.offsetForenums;
#pragma omp flush(curReadpos)
	//posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, readlength);
	miss_size += tmpaio.listlength;
	return tmpaio;
}

Node* LRUCache::Put(unsigned key)//压入的链一定不在cache中
{
	//cout << "In put" << endl;
	AIOReadInfo tmpaio = calAioreadinfo(key); //cout << "put 0" << endl;
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << "That block overflow!!" << endl;
		return NULL;
	}//cout << "put1" << endl;
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE)//删到内存有一定空间装入当前数据
	{//cout<<"In"<<endl;
		if (node == head_){ node = tail_->prev; }//所有链都不可释放，重新一轮查找
#pragma omp flush(usedFreq)
		//如果有链正在使用(CPU用或者IO在读)，换下一个节点
		//if (usedFreq[node->aiodata.termid] > 0 || curReadpos[node->aiodata.termid] < node->aiodata.listlength){ node = node->prev; continue; }
		if (usedFreq[node->aiodata.termid] > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);//////////////////////////////////////////
		curReadpos[node->aiodata.termid] = node->aiodata.offsetForenums;

		sumBytes -= node->aiodata.readlength;
		hashmap_.erase(node->aiodata.termid);////节点删除

		Node *tmp = node->prev;
		delete node;
		node = tmp;
	}
	//cout << "put2" << endl;
	node = new Node();
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, tmpaio.readlength);
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node); //cout << "put3" << endl;
	hashmap_[key] = node;
	//cout << "put 4" << endl;
	return node;
}

Node* LRUCache::Put_Prefetch(unsigned key)
{
	AIOReadInfo tmpaio = calAioreadinfo(key); //cout << "put 0" << endl;
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << "That block overflow!!" << endl;
		return NULL;
	}//cout << "put1" << endl;
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE&&node != head_)//删到内存有一定空间装入当前数据,或所有链都不可释放
	{
#pragma omp flush(usedFreq)
		//如果有链正在使用(CPU用或者IO在读)，换下一个节点
		//if (usedFreq[node->aiodata.termid] > 0 || curReadpos[node->aiodata.termid] < node->aiodata.listlength){ node = node->prev; continue; }
		if (usedFreq[node->aiodata.termid] > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);//////////////////////////////////////////
		curReadpos[node->aiodata.termid] = node->aiodata.offsetForenums;

		sumBytes -= node->aiodata.readlength;
		hashmap_.erase(node->aiodata.termid);////节点删除

		Node *tmp = node->prev;
		delete node;
		node = tmp;
	}
	if (node == head_)//未成功插入数据
	{
		return NULL;
	}

	node = new Node();
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, tmpaio.readlength);
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node); //cout << "put3" << endl;
	hashmap_[key] = node;
	//cout << "put 4" << endl;
	return node;
}
Node* LRUCache::Get(unsigned key, bool &flag)//flag=false ——miss的数据由当前线程负责读完
{
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_.find(key);
	if (it != hashmap_.end())//cache hit
	{
		node = it->second;
		flag = true;
		hit_count++;
		detach(node);
		attach(node);
		//node = NULL;
	}
	else//cache miss
	{
		flag = false;
		miss_count++;
		node = Put(key);
	}
	return node;
	//cout << "get over" << endl;
}
Node* LRUCache::Get_Prefetch(unsigned key, bool &flag)//在预取中hit并不会放到链表头
{
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_.find(key);
	if (it != hashmap_.end())//cache hit
	{
		node = it->second;
		flag = true;
		//hit_count++;
		detach(node);
		attach(node);
	}
	else//cache miss
	{
		flag = false;
		miss_count++;
		node = Put_Prefetch(key);
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
		//cout << iter->first << " ";
		mysumsize += iter->second->aiodata.listlength;
	}
	cout << "sumsize=" << mysumsize << endl;
}
