#include <iostream>
#include<stdint.h>
#include <unordered_map>
#include <vector>
using namespace std;

vector<int64_t>List_offset;

struct AIOReadInfo
{
	int64_t readlength;//�����ȣ�4K���룩
	int64_t readoffset;//��ƫ�ƣ�4K���룩
	//int64_t listoffset;//ʵ��ƫ��
	int64_t listlength;//ʵ�ʳ���
	int64_t offsetForenums;
	int64_t memoffset;
	int64_t curSendpos;
	//int64_t usedfreq;
	uint8_t *list_data;//���ݲ���
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
	void print();
	uint64_t hit_size;
	uint64_t miss_size;
	uint64_t hit_count;
	uint64_t miss_count;

	void attach(Node *node);//���뵽����ͷ
	void detach(Node *node);//������ɾ���ڵ�
	AIOReadInfo calAioreadinfo(unsigned term);//����Aio�ṹ����

	unordered_map<unsigned, Node*>hashmap_;//hash��<termid,����ڵ�>
	Node*head_, *tail_;//ͷ�ڵ�β�ڵ�
	int64_t sumBytes;//��ǰ�ڴ�ռ���ֽ���
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
	int64_t readlength = ((int64_t)(ceil((double)(listlength + tmpaio.offsetForenums) / READ_BLOCK)))*READ_BLOCK;//4K����
	tmpaio.readlength = readlength;
	tmpaio.curSendpos = -tmpaio.offsetForenums;
	//tmpaio.usedfreq++;//usedFreq��ô����
	curReadpos[term] = -tmpaio.offsetForenums;
#pragma omp flush(curReadpos)
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, readlength);
	miss_size += tmpaio.listlength;
	return tmpaio;
}

Node* LRUCache::Put(unsigned key)//ѹ�����һ������cache��
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
	while (sumBytes + tmpaio.readlength>CACHE_SIZE)//ɾ���ڴ���һ���ռ�װ�뵱ǰ����
	{
		if (node == head_){ node = tail_->prev; }//�������������ͷţ�����һ�ֲ���
#pragma omp flush(usedFreq)
		//�����������ʹ��(CPU�û���IO�ڶ�)������һ���ڵ�
		//if (usedFreq[node->aiodata.termid] > 0 || curReadpos[node->aiodata.termid] < node->aiodata.listlength){ node = node->prev; continue; }
		if (usedFreq[node->aiodata.termid] > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);//////////////////////////////////////////
		curReadpos[node->aiodata.termid] = node->aiodata.offsetForenums;

		sumBytes -= node->aiodata.readlength;
		hashmap_.erase(node->aiodata.termid);////�ڵ�ɾ��

		Node *tmp = node->prev;
		delete node;
		node = tmp;
	}
	//cout << "put2" << endl;
	node = new Node();
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node); //cout << "put3" << endl;
	hashmap_[key] = node;
	//cout << "put 4" << endl;
	return node;
}

Node* LRUCache::Get(unsigned key, bool &flag)//flag=false ����miss�������ɵ�ǰ�̸߳������
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
