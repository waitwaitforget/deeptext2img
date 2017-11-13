import torch
import torch.nn as nn
import torch.optim as optim
from modules.DocumentCNN import DocumentCNN
from modules.ImageEncoder import ImageEncoder
import argparse


parser = argparse.ArgumentParser(description="Structured joint embedding")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size of training(default: 100')
parser.add_argument('--emb-dim', type=int, default=1536, metavar='N',
                    help='embedding dimension')
parser.add_argument('--image-dim', type=int, default=1024, metavar='N',
                    help='image feature dimension')
parser.add_argument('--image-noop', type=bool, default=False, metavar='True or False',
                    help='no operation on image or not')
parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                    help='dropout rate')
parser.add_argument('--avg', type=bool, default=False, metavar='N',
                    help='whether to time-average hidden units')

args = parser.parse_args()


class MultiModalEmbedding(nn.Module):

    def __init__(self, alphasize, emb_dim, img_dim, cnn_dim, dropout, avg, image_noop):
        super(MultiModalEmbedding, self).__init__()
        self.enc_doc = DocumentCNN(alphasize)
        self.enc_img = ImageEncoder(img_dim, emb_dim, image_noop)

    def forward(self, fea_txt, fea_img):
        fea_txt = self.enc_doc(fea_txt)
        fea_img = self.enc_img(fea_img)
        return fea_txt, fea_img


def JointEmbeddingLoss(fea_txt, fea_img, labels):
    batchsize = fea_img.size(0)
    num_class = fea_txt.size(1)
    score = torch.zero(batchsize, num_class)

    loss = 0
    #acc_batch = 0
    for i in range(batchsize):

        for j in range(num_class):
            score[i][j] = torch.dot(fea_img[i], fea_txt[:,j])

        label_score = score[i, labels[i]]
        for j in range(num_class):
            if j != labels[i]:
                cur_score = score[i][j]
                thresh = cur_score - label_score + 1
                if thresh > 0:
                    loss += thresh
                    txt_diff = fea_txt[:,j] - fea_txt[:, labels[i]]

        max_score, max_ix = score[i].max()
        if max_ix[1][1] == labels[i]:
            acc_batch += 1

        #acc_batch = 100 * (acc_batch / batchsize)
        denom = batchsize * num_class

        return loss / denom


def train(model, optimizer, dataloader, epoch):
    optimizer.zero_grad()

    losses = 0.0
    for (txt, img, labels) in enumerate(dataloader):
        fea_txt, fea_img = model(txt, img)

        loss = JointEmbeddingLoss(fea_txt, fea_img, labels)

        if args.symmetric:
            loss += JointEmbeddingLoss(fea_img, fea_txt, labels)
        losses += loss.data[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('Epcoh {}: joint embedding loss {.4f}'.format(epoch, losses * args.batch_size/len(dataloader)))
    return model


def main():
    dataloader = load_data()

    model = MultiModalEmbedding(args.alphasize, args.emb_dim, args.img_dim, args.cnn_dim, args.dropout, args.avg,
                                args.image_noop)

    optimizer = optim.RMSprop(model, lr=1e-3)

    for epoch in range(args.epochs):
        model = train(model, optimizer, dataloader, epoch)

        # save checkpoint
        if (epoch+1) % 20 == 0:
            torch.save(model, 'checkpoint{}.pt'.format(epoch + 1))
