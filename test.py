from sklearn.metrics import confusion_matrix

import torch

def get_outfit_dictionary(outfit):
    return json.loads(outfit.replace("'", "\""))

def test(model, test_loader, threshold):
    cuda_available = torch.cuda.is_available()

    confusion_matrix = {'TN': [], 'FP': [],
                        'FN': [], 'TP': []}
    with torch.no_grad():
        test_loss = 0.0
        test_accuracy = 0
        for i, test_data in enumerate(test_loader):
            outfits, outfit_lens, labels, outfit_strs = test_data

            if cuda_available:
                outfits = outfits.cuda(device)
                outfit_lens = outfit_lens.tolist()
                labels = labels.cuda(device)

            output = model(outfits, outfit_lens)

            loss = criterion(output, labels)
            test_loss += loss

            predicted = (output > threshold).int()
            test_accuracy += (labels == predicted).type(torch.FloatTensor).mean()
            
            for j in range(len(predicted)):
                if labels[j] == 0 and predicted[j] == 0:
                    confusion_matrix['TN'].append(get_outfit_dictionary(outfit_strs[j]))

                if labels[j] == 0 and predicted[j] == 1:
                    confusion_matrix['FP'].append(get_outfit_dictionary(outfit_strs[j]))

                if labels[j] == 1 and predicted[j] == 0:
                    confusion_matrix['FN'].append(get_outfit_dictionary(outfit_strs[j]))

                if labels[j] == 1 and predicted[j] == 1:
                    confusion_matrix['TP'].append(get_outfit_dictionary(outfit_strs[j]))

            print('-' * 80)
            print('Batch: {}/{} | Test loss: {:.4f} | Test accuracy: {:.4f}'.format(
                i+1, len(test_loader), test_loss / len(test_loader), test_accuracy / len(test_loader)))
            print('-' * 80)

    return confusion_matrix