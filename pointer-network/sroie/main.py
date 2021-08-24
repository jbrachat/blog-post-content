from sroie.preprocess_data import generate_documents
from utils import display_doc, text_pre_processing, make_tensors, loss_function, get_loaders, train_model, \
    get_threshold_data, get_metrics, find_threshold
from model import Model
import random
import torch
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import TensorDataset, DataLoader

field_key = "company"
display = True

batch_size = 16
embedding_dim = 64
position_dim = 10
learning_rate = 5e-3
n_epochs = 180

max_seq_len = 10

target_accuracy = 0.99

do_train = False
best_step = 179

if __name__ == "__main__":

    documents = generate_documents(field_key,
                                   r"C:\Users\brachj\PycharmProjects\pointer-network\sroie\ocr",
                                   r"C:\Users\brachj\PycharmProjects\pointer-network\sroie\labels",
                                   r"C:\Users\brachj\PycharmProjects\pointer-network\sroie\images")

    # The ground truth is a rectangle on the page that delimites where the ground is located. On the following example
    # it is displayed in red, all the tokens transcribed by pytesseract are in blue.
    if display:
        image = display_doc(documents[3])
        image.show()

    # Checking label length distribution:
    lengths = [
        len(doc['labels'])
        for doc in documents.values()
    ]
    if max(lengths) > max_seq_len:
        warnings.warn(
            f"The maximum ground truth length: {max(lengths)} is greater than the model max_seq_len: {max_seq_len}")

    if display:
        plt.figure()
        plt.hist(lengths)
        plt.show()

    # We put everything in lowercase and only keep alphabet letters and numbers.
    data = [
        (
            key,
            [
                (text_pre_processing(token['text']), token['position']) for token in doc['OCR']
            ],
            doc['labels']
        )
        for key, doc in documents.items()
    ]

    # We will train the model on train, find a threshold on validation and make sure it works as expected on test
    N_DOCS = len(data)
    split = 60, 20, 20  # train / validation / test

    random.seed(42)
    random.shuffle(data)
    n_train = int(split[0] / 100 * N_DOCS)
    n_val = n_train + int(split[1] / 100 * N_DOCS)

    dataset_split = {
        'train': [doc for doc in data[:n_train]],
        'validation': [doc for doc in data[n_train:n_val]],
        'test': [doc for doc in data[n_val:]],
    }

    # We decided on a simple architecture with character level embedding that we feed to an encoder to get word
    # level embedding. Here we map each character to a number.
    characters = set()
    for _, doc_input, _ in dataset_split['train']:
        for word, _ in doc_input:
            characters |= set([x for x in word])
    characters = sorted(list(characters))
    characters_mapping = {char: i + 1 for i, char in enumerate(characters)}  # + 1 to account for the stop token
    len(characters_mapping)

    dataset_split = {
        mode: [
            (
                key,
                (
                    [
                        ([characters_mapping[c] for c in word], position)
                        for word, position in input_data
                    ]
                ),
                target
            )
            for key, input_data, target in dataset_split[mode]
        ]
        for mode in dataset_split
    }

    # Tensorification
    tensors_data = {}
    for mode in dataset_split:
        tensors_data[mode] = make_tensors(dataset_split[mode], max_seq_len)

    if display:
        for mode in tensors_data:
            print('-' * 40)
            print('mode', mode)
            print('words', tensors_data[mode].words.shape)
            print('positions', tensors_data[mode].positions.shape)
            print('target', tensors_data[mode].target.shape)

        from torch.utils.data import TensorDataset, DataLoader

    # Wrapping it up in a TensorDataset
    datasets = {
        mode: TensorDataset(
            tensors_data[mode].keys,
            tensors_data[mode].words.type(torch.LongTensor),
            tensors_data[mode].positions,
            tensors_data[mode].target.type(torch.LongTensor)
        )
        for mode in tensors_data
    }

    # Training loop
    train_loader, val_loader, test_loader = get_loaders(datasets, batch_size)
    model = Model(
        len(characters_mapping) + 1,  # + 1 to account for the stop token with index 0
        embedding_dim,
        position_dim,
        max_seq_len
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if do_train:
        train_losses, val_losses, val_metrics = train_model(n_epochs, model, optimizer, train_loader, val_loader)
        if display:
            plt.figure()
            plt.plot(train_losses, color='red', label="train")
            plt.plot(val_losses, color='blue', label="validation")
            plt.xlabel('n_epoch')
            plt.ylabel('cross_entropy_loss')
            plt.legend()
            plt.show()

    # Best validation:
    if do_train:
        best_step = np.argmin(val_losses[:n_epochs])
        if display:
            print(
                f"Best validation: best_step={best_step}, best_loss={min(val_losses)}, best_metric={val_metrics[best_step]}")

    model = torch.load(f'models/model_{best_step}.torch')

    # Test:
    if display:
        test_threshold_data = get_threshold_data(model, test_loader)
        print(get_metrics(test_threshold_data))

        correct = test_threshold_data.loc[test_threshold_data.is_correct].confidence.values
        incorrect = test_threshold_data.loc[~test_threshold_data.is_correct].confidence.values

        plt.figure(figsize=(10, 10))
        plt.hist(correct, bins=50, alpha=0.5, color='green', label='correct')
        plt.hist(incorrect, bins=50, alpha=0.5, color='red', label='incorrect')
        plt.xlabel('Confidence score')
        plt.ylabel('Number of documents per bucket')
        plt.legend()
        plt.show()

    # Automation and accuracy
    val_threshold_data = get_threshold_data(model, val_loader)
    accuracies, automations, threshold_acc = find_threshold(target_accuracy, val_threshold_data)

    val_above_threshold = val_threshold_data.loc[val_threshold_data.confidence > threshold_acc]
    val_accuracy = val_above_threshold.is_correct.mean()
    val_automation = len(val_above_threshold) / len(val_threshold_data)
    if display:
        print(f"val_accuracy: {val_accuracy}, val_automation: {val_automation}")

        plt.figure(figsize=(10, 5))
        thresholds = np.linspace(val_threshold_data.confidence.min(), 1, 100)
        plt.plot(thresholds, automations, color='blue', label='automation')
        plt.plot(thresholds, accuracies, color='green', label='accuracy')
        plt.axvline(x=threshold_acc, color='red', linestyle='--', label=f'{threshold_acc} threshold')
        plt.xlabel('Confidence score')
        plt.ylabel('Number of documents per bucket')
        plt.ylim(ymin=0.6)
        plt.legend()
        plt.show()

    # Get test automation and test accuracy at the threshold
    test_above_threshold = test_threshold_data.loc[test_threshold_data.confidence > threshold_acc]
    test_accuracy = test_above_threshold.is_correct.mean()
    test_automation = len(test_above_threshold) / len(test_threshold_data)
    if display:
        print(f"test_accuracy: {test_accuracy}, test_automation: {test_automation}")

    print("Toto")
