#pragma once

#include<iostream>
#include<vector>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include <algorithm>
#include "FastPFor/headers/optpfor.h"
#include "FastPFor/headers/variablebyte.h"
#include "FastPFor/headers/VarIntG8IU.h"
#include "FastPFor/headers/simdnewpfor.h"
#include "FastPFor/headers/simple16.h"
#include"FastPFor/headers/simdoptpfor.h"
#include "streamvbyte.h"
#undef ASSERT // XXX WHERE IS THIS DEFINED??
#include "io/BitsReader.hpp"
#include "io/BitsWriter.hpp"

using namespace std;
using namespace FastPForLib;




class TightVariableByte {
public:
	template<uint32_t i>
	static uint8_t extract7bits(const uint32_t val) {
		return static_cast<uint8_t>((val >> (7 * i)) & ((1U << 7) - 1));
	}

	template<uint32_t i>
	static uint8_t extract7bitsmaskless(const uint32_t val) {
		return static_cast<uint8_t>((val >> (7 * i)));
	}

	static void encode(const uint32_t *in, const size_t length,
		uint8_t *out, size_t& nvalue)
	{
		uint8_t * bout = out;
		for (size_t k = 0; k < length; ++k) {
			const uint32_t val(in[k]);
			/**
			* Code below could be shorter. Whether it could be faster
			* depends on your compiler and machine.
			*/
			if (val < (1U << 7)) {
				*bout = static_cast<uint8_t>(val | (1U << 7));
				++bout;
			}
			else if (val < (1U << 14)) {
				*bout = extract7bits<0>(val);
				++bout;
				*bout = extract7bitsmaskless<1>(val) | (1U << 7);
				++bout;
			}
			else if (val < (1U << 21)) {
				*bout = extract7bits<0>(val);
				++bout;
				*bout = extract7bits<1>(val);
				++bout;
				*bout = extract7bitsmaskless<2>(val) | (1U << 7);
				++bout;
			}
			else if (val < (1U << 28)) {
				*bout = extract7bits<0>(val);
				++bout;
				*bout = extract7bits<1>(val);
				++bout;
				*bout = extract7bits<2>(val);
				++bout;
				*bout = extract7bitsmaskless<3>(val) | (1U << 7);
				++bout;
			}
			else {
				*bout = extract7bits<0>(val);
				++bout;
				*bout = extract7bits<1>(val);
				++bout;
				*bout = extract7bits<2>(val);
				++bout;
				*bout = extract7bits<3>(val);
				++bout;
				*bout = extract7bitsmaskless<4>(val) | (1U << 7);
				++bout;
			}
		}
		nvalue = bout - out;
	}

	static void encode_single(uint32_t val, std::vector<uint8_t>& out)
	{
		uint8_t buf[5];
		size_t nvalue;
		encode(&val, 1, buf, nvalue);
		out.insert(out.end(), buf, buf + nvalue);
	}

	static uint8_t const* decode(const uint8_t *in, uint32_t *out, size_t n)
	{
		const uint8_t * inbyte = in;
		for (size_t i = 0; i < n; ++i) {
			unsigned int shift = 0;
			for (uint32_t v = 0;; shift += 7) {
				uint8_t c = *inbyte++;
				v += ((c & 127) << shift);
				if ((c & 128)) {
					*out++ = v;
					break;
				}
			}
		}
		return inbyte;
	}
};
struct Codec//压缩方法的父类
{
	static const unsigned block_size = 128;//需要改！
	//virtual void encode(const uint32_t *in, const size_t length, uint8_t *out, size_t& nvalue){ ; }
	//virtual const uint8_t* decode(uint8_t const* /*in*/, uint32_t* /*out*/, uint32_t  /*sum_of_values*/, size_t /*n*/){ return NULL; }
	//virtual ~Codec(){ }
};
struct Interpolative
{
	//virtual void encode(const uint32_t *in, const size_t length, uint8_t *out, size_t& nvalue){ ; }
	~Interpolative(){ }
	const uint8_t* decode(uint8_t const* in, uint32_t* out, uint32_t  sum_of_values, size_t n)
	{
		assert(n <= block_size);
		uint8_t const* inbuf = in;
		if (sum_of_values == uint32_t(-1)) {
			inbuf = TightVariableByte::decode(inbuf, &sum_of_values, 1);
		}
		uint32_t high = sum_of_values + n - 1;
		out[n - 1] = high;
		if (n > 1) {
			integer_encoding::internals::BitsReader br((uint32_t const*)inbuf, 2 * n);//cout<<"1"<<endl;
			br.intrpolatvArray(out, n - 1, 0, 0, high);//cout<<"2"<<endl;
			for (size_t i = n - 1; i > 0; --i) {
				out[i] -= out[i - 1] + 1;
			}//cout<<"3"<<endl;
			return (uint8_t const*)(br.pos() + 1);
		}
		else {
			return inbuf;
		}
	}//如果n==128调用declodeBlock
};
struct NoComp
{
	Interpolative interpo_codec;
	~NoComp(){}
	const uint8_t* decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
	{
		uint8_t const* ret;
		if (n == Codec::block_size)
		{
			uint8_t *tmp = const_cast<uint8_t*>(in);
			uint32_t *p = reinterpret_cast<uint32_t*>(tmp);
			memcpy(out, p, n*sizeof(uint32_t));
			ret = in + 4 * n;
		}
		else
		{
			ret = interpo_codec.decode(in, out, sum_of_values, n);
		}
		return ret;
	}
	static void encode(uint32_t const* in, uint32_t /*sum_of_values*/, size_t n, std::vector<uint8_t>& out)
	{
		uint32_t *tmp = const_cast<uint32_t*>(in);
		uint8_t* p = reinterpret_cast<uint8_t*>(tmp);
		size_t i = 0;
		while (i < n * 4)
		{
			out.push_back(p[i]);
			i++;
		}
		//memcpy(out.data(), p, n*sizeof(uint8_t));
	}
};

struct Varint_G8IU
{
	//virtual void encode(const uint32_t *in, const size_t length, uint8_t *out, size_t& nvalue){ ; }
	VarIntG8IU varint_codec;
	Interpolative interpo_codec;
	~Varint_G8IU(){}
	const uint8_t* decode(uint8_t const* in, uint32_t* out, uint32_t  sum_of_values, size_t n)
	{
		//assert(n = block_size);
		size_t out_len = Codec::block_size;
		uint8_t const* ret;

		if (n == Codec::block_size) {
			const uint8_t * src = in;
			uint32_t* dst = out;
			size_t srclen = 2 * out_len * 4; // upper bound
			size_t dstlen = out_len * 4;
			out_len = 0;
			while (out_len <= (n - 8)) {
				out_len += varint_codec.decodeBlock(src, srclen, dst, dstlen);
			}

			// decodeBlock can overshoot, so we decode the last blocks in a
			// local buffer
			while (out_len < n) {
				uint32_t buf[8];
				uint32_t* bufptr = buf;
				size_t buflen = 8 * 4;
				size_t read = varint_codec.decodeBlock(src, srclen, bufptr, buflen);
				size_t needed = std::min(read, n - out_len);
				memcpy(dst, buf, needed * 4);
				dst += needed;
				out_len += needed;
			}
			assert(out_len == n);
			ret = src;
		}
		else {
			ret = interpo_codec.decode(in, out, sum_of_values, n);
			//exit(0);
		}
		return ret;
	}//如果n==128调用declodeBlock
};
struct OptPforBlock : OPTPFor<4, Simple16<false>> {
	// workaround: OPTPFor does not define decodeBlock, so we cut&paste
	// the code
	uint32_t const* decodeBlock(const uint32_t *in, uint32_t *out, size_t& nvalue)
	{
		const uint32_t * const initout(out);
		const uint32_t b = *in >> (32 - PFORDELTA_B);
		const size_t nExceptions = (*in >> (32 - (PFORDELTA_B
			+ PFORDELTA_NEXCEPT))) & ((1 << PFORDELTA_NEXCEPT) - 1);
		const uint32_t encodedExceptionsSize = *in & ((1 << PFORDELTA_EXCEPTSZ)
			- 1);

		size_t twonexceptions = 2 * nExceptions;
		++in;
		if (encodedExceptionsSize > 0)
			ecoder.decodeArray(in, encodedExceptionsSize, &exceptions[0],
			twonexceptions);
		assert(twonexceptions >= 2 * nExceptions);
		in += encodedExceptionsSize;

		uint32_t * beginout(out);// we use this later

		for (uint32_t j = 0; j < BlockSize; j += 32) {
			fastunpack(in, out, b);
			in += b;
			out += 32;
		}

		for (uint32_t e = 0, lpos = -1; e < nExceptions; e++) {
			lpos += exceptions[e] + 1;
			beginout[lpos] |= (exceptions[e + nExceptions] + 1) << b;
		}

		nvalue = out - initout;
		return in;
	}
};
struct OptPfor
{
	OptPforBlock optpfor_codec;
	Interpolative interpo_codec;
	//virtual void encode(const uint32_t *in, const size_t length, uint8_t *out, size_t& nvalue){ ; }
	~OptPfor(){}
	const uint8_t* decode(uint8_t const* in, uint32_t* out, uint32_t  sum_of_values, size_t n)
	{
		//assert(n = Codec::block_size);
		size_t out_len = Codec::block_size;
		uint8_t const* ret;

		if (n == Codec::block_size) {
			ret = reinterpret_cast<uint8_t const*>
				(optpfor_codec.decodeBlock(reinterpret_cast<uint32_t const*>(in),
				out, out_len));
			assert(out_len == n);
		}
		else {
			ret = interpo_codec.decode(in, out, sum_of_values, n);
		}
		return ret;
	}//如果n==128调用declodeBlock
};
struct SIMDNewPforBlock :SIMDNewPFor<4, Simple16<false>> {
	uint32_t const* decodeBlock(const uint32_t *in, uint32_t *out, size_t& nvalue)
	{
		const uint32_t * const initout(out);
		const uint32_t b = *in >> (32 - PFORDELTA_B);
		const size_t nExceptions = (*in >> (32 - (PFORDELTA_B
			+ PFORDELTA_NEXCEPT))) & ((1 << PFORDELTA_NEXCEPT) - 1);
		const uint32_t encodedExceptionsSize = *in & ((1 << PFORDELTA_EXCEPTSZ)
			- 1);

		size_t twonexceptions = 2 * nExceptions;
		++in;
		if (encodedExceptionsSize > 0)
			ecoder.decodeArray(in, encodedExceptionsSize, &exceptions[0],
			twonexceptions);
		assert(twonexceptions >= 2 * nExceptions);
		in += encodedExceptionsSize;

		uint32_t * beginout(out);// we use this later

		usimdunpack(reinterpret_cast<const __m128i *>(in), out, b);
		in += 4 * b;
		out += 128;

		for (uint32_t e = 0, lpos = -1; e < nExceptions; e++) {
			lpos += exceptions[e] + 1;
			beginout[lpos] |= (exceptions[e + nExceptions] + 1) << b;
		}

		nvalue = out - initout;
		return in;
	}
};
struct SIMDNewPfor 
{
	//virtual void encode(const uint32_t *in, const size_t length, uint8_t *out, size_t& nvalue){ ; }
	SIMDNewPforBlock simdnewpfor_codec;
	Interpolative interpo_codec;
	~SIMDNewPfor(){}
	const uint8_t* decode(uint8_t const* in, uint32_t* out, uint32_t  sum_of_values, size_t n)
	{//cout<<"in"<<(int)*(in)<<" "<<(int)*(in+1)<<" "<<(int)*(in+2)<<" "<<(int)*(in+3)<<endl;
		//assert(n = Codec::block_size);
		size_t out_len = Codec::block_size;
		uint8_t const* ret;
		if (n == Codec::block_size) {
			ret = reinterpret_cast<uint8_t const*>
				(simdnewpfor_codec.decodeBlock(reinterpret_cast<uint32_t const*>(in),
				out, out_len));
			assert(out_len == n);
		}
		else {
			ret = interpo_codec.decode(in, out, sum_of_values, n);
		}//cout<<"ret"<<(int)*(ret)<<" "<<(int)*(ret+1)<<" "<<(int)*(ret+2)<<" "<<(int)*(ret+3)<<endl;
		return ret;
	}
};
struct SIMDOptPforBlock :SIMDOPTPFor<4, Simple16<false>> {
	// workaround: OPTPFor does not define decodeBlock, so we cut&paste
	// the code
	uint32_t const* decodeBlock(const uint32_t *in, uint32_t *out, size_t& nvalue)
	{
		const uint32_t * const initout(out);
		const uint32_t b = *in >> (32 - PFORDELTA_B);
		const size_t nExceptions = (*in >> (32 - (PFORDELTA_B
			+ PFORDELTA_NEXCEPT))) & ((1 << PFORDELTA_NEXCEPT) - 1);
		const uint32_t encodedExceptionsSize = *in & ((1 << PFORDELTA_EXCEPTSZ)
			- 1);

		size_t twonexceptions = 2 * nExceptions;
		++in;
		if (encodedExceptionsSize > 0)
			ecoder.decodeArray(in, encodedExceptionsSize, &exceptions[0],
			twonexceptions);
		assert(twonexceptions >= 2 * nExceptions);
		in += encodedExceptionsSize;

		uint32_t * beginout(out);// we use this later

		usimdunpack(reinterpret_cast<const __m128i *>(in), out, b);
		in += 4 * b;
		out += 128;

		for (uint32_t e = 0, lpos = -1; e < nExceptions; e++) {
			lpos += exceptions[e] + 1;
			beginout[lpos] |= (exceptions[e + nExceptions] + 1) << b;
		}

		nvalue = out - initout;
		return in;
	}
};
struct SIMDOptPfor
{
	SIMDOptPforBlock simdoptpfor_codec;
	TightVariableByte vbyte_codec;
	Interpolative interpolative_codec;
	~SIMDOptPfor(){}
	uint8_t const* decode(uint8_t const* in, uint32_t* out,
		uint32_t  sum_of_values, size_t n)
	{
		//cout<<"hello"<<" ";
		//assert(n = Codec::block_size);
		size_t out_len = Codec::block_size;
		uint8_t const* ret;

		if (n == Codec::block_size) {
			ret = reinterpret_cast<uint8_t const*>
				(simdoptpfor_codec.decodeBlock(reinterpret_cast<uint32_t const*>(in),
				out, out_len));
			assert(out_len == n);
		}
		else {
			ret = interpolative_codec.decode(in, out, sum_of_values, n);

		}
		return ret;
	}
};
struct StreamVByte
{
	TightVariableByte vbyte_codec;
	Interpolative interpolative_codec;
	~StreamVByte(){}
	uint8_t const* decode(uint8_t const* in, uint32_t* out,
		uint32_t  sum_of_values, size_t n){//n 有没有必要引用
		uint8_t const* inbuf = in;
		size_t compsize2;//解压前读取的数据大小（字节）
		if (n == Codec::block_size) {
			compsize2 = streamvbyte_decode(in, out, n); // decoding (fast)
			out += n;//每个块解压后要往后跳out，跳n个docid的长度
			return inbuf + compsize2;//每条链压缩后是块最大值+块偏移+docid压缩后数据+freq压缩后数据，inbuf是docid+freq压缩后数据的头指针，加comsize2是为了调到freq压缩后数据的头部，因为在block_posting_list.hpp的document_enumerator中的stats_freqs_size()会计算压缩后freq的长度（为了在creat_freq_index.cpp中报告每个整数占的bit数。
		}
		else{
			uint8_t const* ret;
			ret = interpolative_codec.decode(in, out, sum_of_values, n);
			return ret;
		}
	}
};

struct Simple16_block
{
	TightVariableByte vbyte_codec;
	Interpolative interpolative_codec;
	~Simple16_block(){}
	uint8_t const* decode(uint8_t const* in, uint32_t* out,
		uint32_t  sum_of_values, size_t n){//n 有没有必要引用
       Simple16<false> s16;
		   size_t compsize=n;//解压前读取的数据大小（字节）
       if (n == Codec::block_size) {
    			uint8_t const* ret=reinterpret_cast<uint8_t const*>(s16.decodeArray(reinterpret_cast<uint32_t const*>(in), n, out, compsize));
          assert(compsize == n);
          return ret;
    		}
    		else{
    			uint8_t const* ret;
    			ret = interpolative_codec.decode(in, out, sum_of_values, n);
    			return ret;
    		}
	}
};

//压缩首信息
