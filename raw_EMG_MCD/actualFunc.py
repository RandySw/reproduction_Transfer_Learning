import torch
import torch.nn as nn
import paramsTrans
from utilsTrans import discrepancy

criterion = nn.CrossEntropyLoss()


def train(src_data, tgt_data, extractor, classifier1, classifier2):
    extractor.train()
    classifier1.train()
    classifier2.train()
    extractor.to(paramsTrans.device)
    classifier1.to(paramsTrans.device)
    classifier2.to(paramsTrans.device)

    data = enumerate(zip(src_data, tgt_data))
    opt_e = torch.optim.Adam(extractor.parameters(), lr=paramsTrans.lr)
    opt_c1 = torch.optim.Adam(classifier1.parameters(), lr=paramsTrans.lr)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=paramsTrans.lr)
    # 跳出循环
    len_dataloader = min(len(src_data), len(tgt_data))
    for idx, ((src_img, src_labels), (tgt_img, _)) in data:
        if idx > len_dataloader:
            break

        src_img = src_img.to(paramsTrans.device)
        tgt_img = tgt_img.to(paramsTrans.device)
        src_labels = src_labels.to(paramsTrans.device)
        '''
        STEP A
        '''
        opt_e.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        src_feat = extractor(src_img)
        preds_s1 = classifier1(src_feat)
        preds_s2 = classifier2(src_feat)

        loss_A = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels)
        loss_A.backward()

        opt_e.step()  # updating weights
        opt_c1.step()
        opt_c2.step()

        '''
        STEP B # fix feature extractor
        '''
        opt_e.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        src_feat = extractor(src_img)
        preds_s1 = classifier1(src_feat)
        preds_s2 = classifier2(src_feat)

        src_tgt = extractor(tgt_img)
        preds_t1 = classifier1(src_tgt)
        preds_t2 = classifier2(src_tgt)

        loss_B = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels) - discrepancy(preds_t1, preds_t2)
        loss_B.backward()

        opt_c1.step()
        opt_c2.step()

        opt_e.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        '''
        STEP C # fix (c1 与 c2)
        '''
        for i in range(paramsTrans.N):
            feat_tgt = extractor(tgt_img)
            preds_t1 = classifier1(feat_tgt)
            preds_t2 = classifier1(feat_tgt)
            loss_C = discrepancy(preds_t1, preds_t2)
            loss_C.backward()
            opt_e.step()

            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

        if (idx+1) % 50 == 0:
            print("loss_A = {:.2f}, loss_B = {:.2f}, loss_C = {:.2f}".format(loss_A.item(),
                                                                             loss_B.item(), loss_C.item()))
    return extractor, classifier1, classifier2


def test(data, extractor, classifier1, classifier2):
    acc1 = 0
    acc2 = 0

    for img, labels in data:
        img = img.to(paramsTrans.device)
        labels = labels.to(paramsTrans.device)
        preds1 = classifier1(extractor(img))
        preds2 = classifier2(extractor(img))

        acc1 += (preds1.argmax(dim=1) == labels).sum().item()
        acc2 += (preds2.argmax(dim=1) == labels).sum().item()

    print("acc1={:.2%}, acc2={:.2%}".format(acc1/(len(data) * 128), acc2/(len(data) * 128)))
