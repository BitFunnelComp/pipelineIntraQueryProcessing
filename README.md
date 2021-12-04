# pipelineQueryProcessing
The code is rewritten from https://github.com/ot/partitioned_elias_fano

Collection input format is same as partitioned_elias_fano

A binary sequence is a sequence of integers prefixed by its length, where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.

A collection consists of 3 files, basename.docs, basename.freqs, basename.sizes.

(1)basename.docs starts with a singleton binary sequence where its only integer is the number of documents in the collection. It is then followed by one binary sequence for each posting list, in order of term-ids. Each posting list contains the sequence of document-ids containing the term.

(2)basename.freqs is composed of a one binary sequence per posting list, where each sequence contains the occurrence counts of the postings, aligned with the previous file (note however that this file does not have an additional singleton list at its beginning).

(3)basename.sizes is composed of a single binary sequence whose length is the same as the number of documents in the collection, and the i-th element of the sequence is the size (number of terms) of the i-th document.

To execute greedy reordering, use the command:
./Greedy_Reorder [query_file_name] [cache_size_in_MB] [batch_size]

To execute query, use the command:
./Query_Intra [index_file_name_prefix] [query_type] [thread_count] [cache_size_in_MB] [query_file_name]




