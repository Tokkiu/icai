import pickle
import random


def dataset(config):
    f_count = 1
    with open(config.data, 'rb') as f:
        dataset = pickle.load(f)
        train_data = list(dataset['train'].values())
        val_data = list(dataset['val'].values())
        test_data = list(dataset['test'].values())
        neg_data = list(dataset['neg'].values())
        umap = dataset['umap']
        smap = dataset['smap']
        num_user = len(umap)
        num_items = 0
        for i in smap:
            num_items = max(int(i), num_items)
        num_items += 2  # 一个0，一个mask
        len_all = 0
        for td in train_data:
            len_all += len(td)
        ave_len = len_all / num_user
        print("user数量{}，items数量{}，平均交互长度{}".format(num_user, num_items, ave_len))

    feature_num = [0]*f_count
    if config.use_feature:
        train_feature = dataset['train_fea']
        val_feature = dataset['val_fea']
        test_feature = dataset['test_fea']
        for n in range(f_count):
            for fea in list(train_feature[n].values()):
                for f in fea:
                    for sf in f:
                        feature_num[n] = max(feature_num[n], sf)
            for fea in list(val_feature[n].values()):
                for f in fea:
                    for sf in f:
                        feature_num[n] = max(feature_num[n], sf)
            for fea in list(test_feature[n].values()):
                for f in fea:
                    for sf in f:
                        feature_num[n] = max(feature_num[n], sf)

    sequence_length = config.L

    train_data = dataset['train']
    val_target = dataset['val']
    test_target = dataset['test']
    neg_data = dataset['neg']
    feature_data = dataset['train_fea']
    val_feature_target = dataset['val_fea']
    test_feature_target = dataset['test_fea']

    avg_fea_len = dataset['avg_fea_len']

    umap = dataset['umap']
    smap = dataset['smap']

    num_items = 0
    for i in smap:
        num_items = max(int(i), num_items)

    last_token = [num_items + 1]
    last_feature_token = []
    for i in feature_num:
        last_feature_token.append([i + 1])

    data_dic = {}
    data_dic["train_id"] = []
    data_dic["train_data"] = []
    data_dic["train_label"] = []
    data_dic["train_feature"] = [[],[],[]]
    data_dic["train_feature_label"] = [[],[],[]]
    data_dic["train_neg"] = []
    data_dic["train_feature_neg"] = [[],[],[]]

    data_dic["val_id"] = []
    data_dic["val_data"] = []
    data_dic["val_label"] = []
    data_dic["val_feature"] = [[],[],[]]
    data_dic["val_feature_label"] = [[],[],[]]

    data_dic["test_id"] = []
    data_dic["test_data"] = []
    data_dic["test_label"] = []
    data_dic["test_feature"] = [[],[],[]]
    data_dic["test_feature_label"] = [[],[],[]]

    data_dic["neg_data"] = []

    seed = config.dataloader_random_seed
    rng = random.Random(seed)
    for user, items in train_data.items():
        train_d = train_data[user]
        val_t = val_target[user]
        test_d = train_d + val_t
        test_t = test_target[user]
        neg_d = neg_data[user]

        data_dic["val_id"].append(user)
        data_dic["val_data"].append([0]*(sequence_length-len(train_d[-sequence_length:]))
                                    + train_d[-sequence_length:] + last_token)
        data_dic["val_label"].append(val_t)
        for n in range(f_count):
            feature_val_list = [[0]*avg_fea_len[n]] * (sequence_length - len(train_d[-sequence_length:])) \
                                 + feature_data[n][user][-sequence_length:] + [last_feature_token[n]*avg_fea_len[n]]
            feature_val_target_list = val_feature_target[n][user]
            data_dic["val_feature"][n].append(feature_val_list)
            data_dic["val_feature_label"][n].append(feature_val_target_list)

        data_dic["test_id"].append(user)
        data_dic["test_data"].append([0]*(sequence_length-len(test_d[-sequence_length:]))
                                     + test_d[-sequence_length:] + last_token)
        data_dic["test_label"].append(test_t)
        for n in range(f_count):
            feature_test_list = [[0]*avg_fea_len[n]] * (sequence_length - len(feature_data[n][user][-(sequence_length-1):] + val_feature_target[n][user])) \
                                 + feature_data[n][user][-(sequence_length-1):] + val_feature_target[n][user] + [last_feature_token[n]*avg_fea_len[n]]
            feature_test_target_list = test_feature_target[n][user]
            data_dic["test_feature"][n].append(feature_test_list)
            data_dic["test_feature_label"][n].append(feature_test_target_list)

        data_dic["neg_data"].append(neg_d)

        for i in range(len(train_d)-1, 1, -1):
            target_list = train_d[i]
            train_list = [0]*(sequence_length-len(train_d[-(i+sequence_length):-i])) \
                            + train_d[-(i+sequence_length):-i] + last_token

            data_dic["train_id"].append(user)
            data_dic["train_data"].append(train_list)
            data_dic["train_label"].append(target_list)

            temp_train_neg = rng.sample(neg_d, config.neg_samples)
            data_dic["train_neg"].append(temp_train_neg)

            for n in range(f_count):
                feature_train_list = [[0]*avg_fea_len[n]]*(sequence_length-len(feature_data[n][user][-(i + sequence_length):-i])) \
                            + feature_data[n][user][-(i + sequence_length):-i] + [last_feature_token[n]*avg_fea_len[n]]
                feature_train_target_list = feature_data[n][user][i]

                data_dic["train_feature"][n].append(feature_train_list)
                data_dic["train_feature_label"][n].append(feature_train_target_list)
                data_dic["train_feature_neg"].append(rng.sample(neg_d, config.neg_samples))

    return data_dic, num_items, num_user, [f+2 for f in feature_num]
