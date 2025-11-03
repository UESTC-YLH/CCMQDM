from tools.Tools import read_density_matrices, read_label
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from constants import (batch_train, Epochs)


file_path_train_feature = fr'data\Encode\Encode_train_feature.csv'
file_path_train_label = fr'data/Encode/Encode_train_label.csv'
file_path_test_feature = fr'data/Encode/Encode_test_feature.csv'
file_path_test_label = fr'data/Encode/Encode_test_label.csv'


def train(trainmodel):
    # $\rho' = \sum_i E_i \rho E_i^\dagger$$$ H ( p , q ) = - [ p \log ( q ) + ( 1 - p ) \log ( 1 - q ) ]$$
    """
    训练模型
    :param trainmodel: 加载模型
    """
    batch_num_train = batch_train
    train_feature = read_density_matrices(file_path_train_feature)
    train_label = read_label(file_path_train_label)

    train_feature_batch = trainmodel.reshape_batch(train_feature, batch_num_train)
    train_label_batch = trainmodel.reshape_batch(train_label, batch_num_train)
    avg_train_loss = []
    acc_train_loss = []

    for epoch in range(Epochs):
        train_loss = 0
        train_acc = 0

        for batch_features, batch_labels_tensor in zip(train_feature_batch, train_label_batch):
            predicted_logits = trainmodel.forward(batch_features)

            loss = F.cross_entropy(predicted_logits, batch_labels_tensor.squeeze(1).long())
            gradients = torch.autograd.grad(loss, list(trainmodel.parameters()), retain_graph=True)
            with torch.no_grad():
                trainmodel.update_kraus_operators(gradients)
            train_loss += loss.item()
            predicted_probs = torch.softmax(predicted_logits, dim=1)  # 计算概率
            predictions = torch.argmax(predicted_probs, dim=1)  # 选择最大概率的索引作为预测类别

            correct_predictions = (predictions == batch_labels_tensor.flatten()).sum().item()
            acc = correct_predictions / batch_labels_tensor.size(0)

            train_acc += acc

        print(f"Epoch [{epoch + 1}/{Epochs}] Train Loss: {train_loss / len(train_label_batch):.10f} "
              f"Accuracy: {train_acc / len(train_label_batch):.10f}")

        # 存储平均损失和准确率
        avg_train_loss.append(train_loss / len(train_feature_batch))
        acc_train_loss.append(train_acc / len(train_feature_batch))

    train_steps = list(range(1, len(avg_train_loss) + 1))
    import matplotlib.pyplot as plt
    plt.plot(train_steps, avg_train_loss, label='Train Loss', color='blue')
    plt.plot(train_steps, acc_train_loss, label='Accumulated Train Loss', color='purple')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Visualization')
    plt.legend()
    plt.show()


def test(model):
    batch = 64
    test_feature = read_density_matrices(file_path_test_feature)
    test_label = read_label(file_path_test_label)

    test_feature_batch = model.reshape_batch(test_feature, batch)
    test_feature_batch = model.reshape_batch(test_label, batch)
    # prediction
    total = 0
    correct = 0
    for images, labels in zip(test_feature_batch, test_feature_batch):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        outputs = model.forward(images)

        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()
    print("Accuracy = %.2f" % (100 * correct / total))

