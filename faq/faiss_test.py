import faiss

# 传入特征维度
dim = 2048

# IndexFlatIP表示利用内积来比较特征的相似度
# 这里一般会让提取的特征进行L2归一化，那么内积就等于余弦相似度
index_ip = faiss.IndexFlatIP(dim)

# IndexFlatL2表示利用L2距离来比较特征的相似度
index_l2 = faiss.IndexFlatL2(dim)

import numpy as np

# 新建一个特征，维度为2048，shape为(1, 2048)
feature = np.random.random((1, 2048)).astype('float32')
index_ip.add(feature)

# 当然，也可以一次性添加多个特征
features = np.random.random((10, 2048)).astype('float32')
index_ip.add(features)

# 打印index_ip包含的特征数量
print(index_ip.ntotal) 

index_ids = faiss.IndexFlatIP(2048)
index_ids = faiss.IndexIDMap(index_ids)

# 添加特征，并指定id，注意添加的id类型为int64
ids = 20
feature_ids = np.random.random((1, 2048)).astype('float32')
index_ids.add_with_ids(feature_ids, np.array((ids,)).astype('int64'))


