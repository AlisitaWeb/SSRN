class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'D:\PT\测试\DSRL-main\DSRL-main\VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/newdata/why/cityscapes'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'rockdataset':
            return 'D:\PT\测试\DSRL-main\DSRL-main\VOC2007/'
        elif dataset == 'rockdataset_10':
            return r'D:\PT\测试\DSRL-main\DSRL-main\rockdataset_10/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
