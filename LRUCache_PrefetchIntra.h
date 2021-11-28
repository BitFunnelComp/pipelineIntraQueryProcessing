//每个数据读适配大小的块
#include <iostream>
#include<stdint.h>
#include <unordered_map>
#include <vector>
using namespace std;

struct SplitInfo
{
	unsigned split_blockid;
	//int64_t split_offset;
	//unsigned doc_count;
	SplitInfo(){ split_blockid = 0; }
};

const unsigned constShardcount = 8;//总共shard个数
vector<int64_t>List_offset;
vector<vector<SplitInfo>>splitInfo(constShardcount);//shard,term
vector<vector<unsigned>>blockEndpoint;//term,block
struct AIOReadInfo
{
	int64_t readlength;//读长度（4K对齐）
	int64_t readoffset;//读偏移（4K对齐）
	//int64_t listoffset;//实际偏移
	int64_t listlength;//实际长度
	int64_t offsetForenums;
	int64_t memoffset;
	int64_t curSendpos;
	int64_t curReadpos;//相对于当前读的shard数据起始位置的pos
	uint8_t usedfreq;
	uint8_t *list_data;//数据部分
	uint32_t termid;
	uint32_t shardid;
	uint32_t readblocksize;
};
//vector<vector<int64_t>>curReadpos(constShardcount);
//vector<vector<uint8_t>>usedFreq(constShardcount);
//vector<AIOReadInfo>AIOreadinfo;
vector<int64_t>READ_BLOCK_options = { 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024 };
const uint64_t DISK_BLOCK = 4096;
//const int64_t READ_BLOCK = 64 * 1024;

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

	void attach(Node *node);//插入到链表头
	void detach(Node *node);//链表中删除节点
	AIOReadInfo calAioreadinfo(unsigned shard, unsigned term);//计算Aio结构数据

	vector<unordered_map<unsigned, Node*>>hashmap_;//hash表<termid,链表节点>
	Node*head_, *tail_;//头节点尾节点
	int64_t sumBytes;//当前内存占用字节数
};

LRUCache::LRUCache()//construct
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

LRUCache::~LRUCache()//destruct
{
	delete head_;
	delete tail_;
}
int64_t cal_properReadBlockSize(int64_t length)
{
	for (unsigned i = 0; i < READ_BLOCK_options.size(); i++)
	{
		if (length <= READ_BLOCK_options[i])
			return READ_BLOCK_options[i];
	}
	return READ_BLOCK_options[READ_BLOCK_options.size() - 1];
}
AIOReadInfo LRUCache::calAioreadinfo(unsigned shard, unsigned term)//默认放进来的链长>0
{
	AIOReadInfo tmpaio;
	tmpaio.termid = term;
	tmpaio.shardid = shard;//cout << "LRU term=" << term << endl;
	int64_t startpos = splitInfo[shard][term].split_blockid == 0 ? 0 : blockEndpoint[term][splitInfo[shard][term].split_blockid - 1];
	int64_t listlength = shard == constShardcount - 1 ? List_offset[term + 1] - List_offset[term] - startpos : blockEndpoint[term][splitInfo[shard + 1][term].split_blockid] - startpos;
	tmpaio.listlength = listlength;
	tmpaio.memoffset = 0;
	int64_t offset = startpos + List_offset[term]; //cout << "term " << term << " offset=" << offset << endl;
	tmpaio.readoffset = ((int64_t)(offset / DISK_BLOCK))*DISK_BLOCK;
	tmpaio.offsetForenums = offset - tmpaio.readoffset;
	tmpaio.readblocksize = cal_properReadBlockSize(tmpaio.listlength);
	int64_t readlength = ((int64_t)(ceil((double)(listlength + tmpaio.offsetForenums) / tmpaio.readblocksize)))*tmpaio.readblocksize;//4K对齐
	tmpaio.readlength = readlength;
	tmpaio.curSendpos = -tmpaio.offsetForenums;
	tmpaio.usedfreq = 0;//usedFreq怎么处理？
	tmpaio.curReadpos = -tmpaio.offsetForenums;
	//#pragma omp flush(curReadpos)
	//posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, readlength);
	miss_size += tmpaio.listlength;////////////////tmpaio.readlength

	//if (shard == 4 && term == 31726210)cout << "listlength=" << tmpaio.listlength << " curReadpos=" << tmpaio.curReadpos << endl;
	return tmpaio;
}

Node* LRUCache::Put(unsigned shard, unsigned key)//压入的链一定不在cache中
{
	//cout << "In put" << endl;
	AIOReadInfo tmpaio = calAioreadinfo(shard, key); //cout << "put 0" << endl;
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << tmpaio.readlength << " " << CACHE_SIZE << " listlength=" << List_offset[key + 1] - List_offset[key] << endl;
		cout << "That block overflow!!" << endl;
		return NULL;
	}//cout << "put1" << endl;
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE)//删到内存有一定空间装入当前数据
	{
		if (node == head_){ node = tail_->prev; }//所有链都不可释放，重新一轮查找
		//如果有链正在使用(CPU用或者IO在读)，换下一个节点
		//if (usedFreq[node->aiodata.termid] > 0 || curReadpos[node->aiodata.termid] < node->aiodata.listlength){ node = node->prev; continue; }
		if (node->aiodata.usedfreq > 0){ node = node->prev; continue; }
		detach(node);//cout<<"AA"<<endl;
		free(node->aiodata.list_data);//////////////////////////////////////////
		node->aiodata.curReadpos = -node->aiodata.offsetForenums; //if (node->aiodata.curReadpos == 64173)cout << "Put" << endl;
		//cout << "*****We will delete term=" << node->aiodata.termid << " shard=" << node->aiodata.shardid << endl;
		sumBytes -= node->aiodata.readlength;
		hashmap_[node->aiodata.shardid].erase(node->aiodata.termid);////节点删除
		//cout<<"DD"<<endl;
		Node *tmp = node->prev;
		delete node;
		node = tmp;//cout<<"EE"<<endl;
	}
	//cout << "put2" << endl;
	node = new Node();
	posix_memalign((void**)&tmpaio.list_data, DISK_BLOCK, tmpaio.readlength);
	node->aiodata = tmpaio;
	sumBytes += tmpaio.readlength;
	attach(node); //cout << "put3" << endl;
	hashmap_[shard][key] = node;
	//cout << "put 4" << endl;
	return node;
}

Node* LRUCache::Put_Prefetch(unsigned shard, unsigned key)
{
	AIOReadInfo tmpaio = calAioreadinfo(shard, key); //cout << "put 0" << endl;
	Node *node;
	if (tmpaio.readlength> CACHE_SIZE)
	{
		cout << "That block overflow!!" << endl;
		return NULL;
	}//cout << "put1" << endl;
	node = tail_->prev;
	while (sumBytes + tmpaio.readlength>CACHE_SIZE&&node != head_)//删到内存有一定空间装入当前数据,或所有链都不可释放
	{
		//如果有链正在使用(CPU用或者IO在读)，换下一个节点
		//if (usedFreq[node->aiodata.termid] > 0 || curReadpos[node->aiodata.termid] < node->aiodata.listlength){ node = node->prev; continue; }
		if (node->aiodata.usedfreq > 0){ node = node->prev; continue; }
		detach(node);
		free(node->aiodata.list_data);//////////////////////////////////////////
		node->aiodata.curReadpos = -node->aiodata.offsetForenums; //if (node->aiodata.curReadpos == 64173)cout << "Put" << endl;
		sumBytes -= node->aiodata.readlength;
		hashmap_[node->aiodata.shardid].erase(node->aiodata.termid);////节点删除

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
	hashmap_[shard][key] = node;
	//cout << "put 4" << endl;
	return node;
}
Node* LRUCache::Get(unsigned shard, unsigned key, bool &flag)//flag=false ——miss的数据由当前线程负责读完
{//cout<<"In get"<<endl;
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_[shard].find(key);
	if (it != hashmap_[shard].end())//cache hit
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
		node = Put(shard, key);
	}
	return node;
	//cout << "get over" << endl;
}
Node* LRUCache::Get_Prefetch(unsigned shard, unsigned key, bool &flag)//在预取中hit会放到链表头
{
	Node *node;
	unordered_map<unsigned, Node* >::iterator it = hashmap_[shard].find(key);
	if (it != hashmap_[shard].end())//cache hit
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
		//cout << iter->first << " ";
		mysumsize += iter->second->aiodata.listlength;
	}
	cout << "sumsize=" << mysumsize << endl;
}
