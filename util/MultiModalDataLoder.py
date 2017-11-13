import os
import torch


class MultiModalDataLoader(torch.utils.data.dataset):

    def __init__(self, data_dir, nclass, img_dim, doc_length, batch_size,
                 randomize_pair, ids_file, num_caption, image_dir, flip):
        super(MultiModalDataLoader, self).__init__()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        self.dict = {}
        for i in range(len(self.alphabet)):
            self.dict[self.alphabet[i]] = i

        self.alphabet_size = len(self.alphabet)
        # load manifest file
        self.files = []
        with open(os.path.join(data_dir, 'manifest.txt'), 'r') as file:
            for line in file:
                self.files.append(line)
        # load train / val / test splits
        self.trainids = {}
        with open(os.path.join(data_dir, ids_file), 'r') as file:
            for line in file:
                self.trainids.append(line)
        self.nclass_train = len(self.trainids)
        self.trainids_tensor = torch.zeros(len(self.trainids))

        for i in range(len(self.trainids)):
            self.trainids_tensor[i] = self.trainids[i]

        self.nclass = nclass
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.img_dim = img_dim
        self.doc_length = doc_length
        self.ntrain = self.nclass_train
        self.randomize_pair = randomize_pair
        self.num_caption = num_caption
        self.image_dir = image_dir if image_dir else ''
        self.flip = flip if flip else 0

    def __getitem__(self, index):

        idx = self.trainids_tensor[index]
        fname = self.files[idx]

        if self.image_dir == '':
            cls_imgs = torch.load(os.path.join(self.data_dir, 'img', fname))
        else:
            cls_imgs = torch.load(os.path.join(self.data_dir, self.image_dir. fname))

        cls_sens = torch.load(os.path.join(self.data_dir, 'text_c{}'.format(self.num_caption), fname))
        sen_idx =


    def __len__(self):
        return self.data_tensor.size(0)