def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs, max_gradient_norm):
    for epoch in range(num_epochs):
        net.train()
        batch_idx = 1
        for index, batch in enumerate(train_iter):
            optimizer.zero_grad()
            X = batch['x']
            y = batch['y']
            # CRF
            loss = model.log_likelihood(X, y)
            loss.backward()
            # CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

            optimizer.step()
            if index % 200 == 0:
                print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(valid_dataloader):
            model.eval()
            val_x, val_y = batch['x'], batch['y']
            predict = model(val_x)
            # CRF
            loss = model.log_likelihood(val_x, val_y)
            aver_loss += loss.item()
            # 统计非0的，也就是真实标签的长度
            leng = []
            for i in val_y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)

            for index, i in enumerate(predict):
                preds += i[:len(leng[index])]

            for index, i in enumerate(val_y.tolist()):
                labels += i[:len(leng[index])]
        aver_loss /= (len(valid_dataloader) * 64)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds)
        print(report)
        torch.save(model.state_dict(), './model/params.pkl')


def predict(self, tag, input_str=""):
    model.load_state_dict(torch.load("./model/params.pkl"))
    if not input_str:
        input_str = input("请输入文本: ")
    input_vec = [word2id.get(i, 0) for i in input_str]
    # convert to tensor
    sentences = torch.tensor(input_vec).view(1, -1)
    paths = model(sentences)

    entities = []
    tags = get_tags(paths[0], tag, tag2id)
    entities += format_result(tags, input_str, tag)
    print(entities)
