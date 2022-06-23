# Read-and-Write-tfrecords-in-Tensorflow
Tensorflow v1.3

How to write serialized examples into tfrecords, and how to read and decode tfrecords file, espacially for variable length data and label,
eg:ASR, NMT task.

怎样将样本【序列化】写入tfrecords，并读取解码tfrecords文件，针对序列化任务，如语音识别、NMT等输入输出变长的数据。
Tensorflow 的tfrecords文件的介绍，官方例子，可以参考以下链接：
* [参考1](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py)
* [参考2](https://zhuanlan.zhihu.com/p/33223782     )
* [参考3](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
* [参考4](https://blog.csdn.net/u010223750/article/details/70482498)
* [参考5](https://lc222.github.io/2017/06/23/Tensorflow%E4%B8%AD%E4%BD%BF%E7%94%A8TFRecords%E9%AB%98%E6%95%88%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE-%E7%BB%93%E5%90%88NLP%E6%95%B0%E6%8D%AE%E5%AE%9E%E8%B7%B5/)
