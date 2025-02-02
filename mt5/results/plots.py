import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# model_name = 'distilbert'
# models = [
#     model_name + '-multilingual-roitd_aug.json',
#     model_name + '-multilingual-roitd_noaug.json',
#     model_name + '-multilingual-xquad.json',
#     model_name + '-romanian-roitd_aug.json',
#     model_name + '-romanian-roitd_noaug.json',
#     model_name + '-romanian-xquad.json'
# ]

models = ['mt5-base-roitd_aug.json',
          'mt5-base-roitd_noaug.json',
          'mt5-base-xquad.json']

for model in models:
    json_file = model
    ablation_name = json_file.split('.')[0]
    if not os.path.exists(f'./images/{ablation_name}'):
        os.makedirs(f'./images/{ablation_name}')
    with open(json_file, 'r') as f:
        data = json.load(f)
    data = data['test']
    data_list = []
    # print(data.keys())
    for lr, lr_value in data.items():
        for epoch, epoch_value in lr_value.items():
            data_list.append(
                {
                    'lr': lr,
                    'epoch': epoch,
                    'exact_match': epoch_value['squad']['exact_match'],
                    'f1': epoch_value['squad']['f1'],
                    'bleu': epoch_value['bleu']['bleu'],
                    'rouge1': epoch_value['rouge']['rouge1'],
                    'rouge2': epoch_value['rouge']['rouge2'],
                    'rougeL': epoch_value['rouge']['rougeL'],
                    'rougeLsum': epoch_value['rouge']['rougeLsum'],
                    'semantic_similarity': epoch_value['semantic_similarity']
                }
            )
    df_data = pd.DataFrame(data_list)
    df_data.head(2)

    for lr in df_data['lr'].unique():
        fig, ax = plt.subplots()
        fig.suptitle(f'{lr} Exact Match and F1')

        ax1 = df_data[df_data['lr'] == lr].plot(x='epoch', y='exact_match', ax=ax,
                                                label='Exact Match')
        ax2 = df_data[df_data['lr'] == lr].plot(x='epoch', y='f1', ax=ax, label='F1')

        ax1.set_ylabel('Exact Match and F1')
        ax1.set_xlabel('Epoch')

        # plt.show()

        fig.savefig(f'images/{ablation_name}/{lr}_exact_match_f1.png')
        plt.close(fig)

    for lr in df_data['lr'].unique():
        fig, ax = plt.subplots()
        fig.suptitle(f'{lr} Semantic Similarity')

        ax1 = df_data[df_data['lr'] == lr].plot(x='epoch', y='semantic_similarity', ax=ax,
                                                label='semantic_similarity')

        ax1.set_ylabel('Semantic Similarity')
        ax1.set_xlabel('Epoch')

        # plt.show()

        fig.savefig(f'images/{ablation_name}/{lr}_semantic_similarity.png')
        plt.close(fig)

    for lr in df_data['lr'].unique():
        fig, ax = plt.subplots()
        fig.suptitle(f'{lr} BLEU, ROUGE1, ROUGE2, ROUGEL, ROUGELsum')

        ax1 = df_data[df_data['lr'] == lr].plot(x='epoch', y='bleu', ax=ax, label='Bleu')
        ax2 = df_data[df_data['lr'] == lr].plot(x='epoch', y='rouge1', ax=ax, label='rouge1')
        ax2 = df_data[df_data['lr'] == lr].plot(x='epoch', y='rouge2', ax=ax, label='rouge2')
        ax2 = df_data[df_data['lr'] == lr].plot(x='epoch', y='rougeL', ax=ax, label='rougeL')
        ax2 = df_data[df_data['lr'] == lr].plot(x='epoch', y='rougeLsum', ax=ax, label='rougeLsum')

        ax1.set_ylabel('Bleu and Rouge')
        ax1.set_xlabel('Epoch')

        # plt.show()

        fig.savefig(f'images/{ablation_name}/{lr}_bleu_rouge.png')
        plt.close(fig)