import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np

from qa_baselines import QA_baseline, QA_lm, QA_embedkgqa, QA_cronkgqa
from qa_tempoqr import QA_TempoQR
from qa_datasets import QA_Dataset, QA_Dataset_TempoQR, QA_Dataset_Baseline
from torch.utils.data import Dataset, DataLoader
import utils
from tqdm import tqdm
from utils import loadTkbcModel, loadTkbcModel_complex, print_info
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description="Temporal KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='tcomplex.ckpt', type=str,
    help="Pretrained tkbc model checkpoint"
)
parser.add_argument(
    '--tkg_file', default='full.txt', type=str,
    help="TKG to use for hard-supervision"
)

parser.add_argument(
    '--model', default='tempoqr', type=str,
    help="Which model to use."
)
parser.add_argument(
    '--supervision', default='soft', type=str,
    help="Which supervision to use."
)

parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)

parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)

parser.add_argument(
    '--max_epochs', default=20, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--eval_k', default=1, type=int,
    help="Hits@k used for eval. Default 10."
)

parser.add_argument(
    '--valid_freq', default=1, type=int,
    help="Number of epochs between each valid."
)


parser.add_argument(
    '--batch_size', default=150, type=int,
    help="Batch size."
)

parser.add_argument(
    '--valid_batch_size', default=50, type=int,
    help="Valid batch size."
)

parser.add_argument(
    '--frozen', default=1, type=int,
    help="Whether entity/time embeddings are frozen or not. Default frozen."
)

parser.add_argument(
    '--lm_frozen', default=1, type=int,
    help="Whether language model params are frozen or not. Default frozen."
)

parser.add_argument(
    '--lr', default=2e-4, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--mode', default='train', type=str,
    help="Whether train or eval."
)

parser.add_argument(
    '--eval_split', default='valid', type=str,
    help="Which split to validate on"
)

parser.add_argument(
    '--dataset_name', default='wikidata_big', type=str,
    help="Which dataset."
)
parser.add_argument(
    '--lm', default='distill_bert', type=str,
    help="Lm to use."
)
parser.add_argument(
    '--fuse', default='add', type=str,
    help="For fusing time embeddings."
)
parser.add_argument(
    '--extra_entities', default=False, type=bool,
    help="For some question types."
)
parser.add_argument(
    '--corrupt_hard', default=0., type=float,
    help="For some question types."
)

parser.add_argument(
    '--test', default="test", type=str,
    help="Test data."
)



args = parser.parse_args()
print_info(args)


def eval(qa_model, dataset, batch_size = 128, split='valid', k=10):
    num_workers = 4
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k # not change name in fn signature since named param used in places
    k_list = [1,10]
    #k_list = [1,2,5, 10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        # todo: is there a more elegant way?
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-1] # last one assumed to be target
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot.cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    # do eval for each k in k_list
    # want multiple hit@k
    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)

        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['type']
            if 'simple' in question_type:
                simple_complex_type = 'simple'
            else:
                simple_complex_type = 'complex'
            entity_time_type = question['answer_type']
            # question_type = question['template']
            predicted = topk_answers[i][:k]
            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0
            question_types_count[question_type].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1

        eval_accuracy = hits_at_k/total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))


        question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
        simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
        entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
        # for dictionary in [question_types_count]:
        for dictionary in [question_types_count, simple_complex_count, entity_time_count]:
        # for dictionary in [simple_complex_count, entity_time_count]:
            for key, value in dictionary.items():
                hits_at_k = sum(value)/len(value)
                s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
                    q_type = key,
                    hits_at_k = round(hits_at_k, 3),
                    num_questions = len(value)
                )
                if print_numbers_only:
                    s = str(round(hits_at_k, 3))
                eval_log.append(s)
            eval_log.append('')

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log


def append_log_to_file(eval_log, epoch, filename):
    f = open(filename, 'a+')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()

def train(qa_model, dataset, valid_dataset, args,result_filename=None):
    num_workers = 5
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=dataset._collate_fn)
    max_eval_score = 0
    if args.save_to == '':
        args.save_to = 'temp'
    if result_filename is None:
        result_filename = 'results/{dataset_name}/{model_file}.log'.format(
            dataset_name = args.dataset_name,
            model_file = args.save_to
        )
    checkpoint_file_name = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name = args.dataset_name,
        model_file = args.save_to
    )

    # if not loading from any previous file
    # we want to make new log file
    # also log the config ie. args to the file
    if args.load_from == '':
        print('Creating new log file')
        f = open(result_filename, 'a+')
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('Log time: %s\n' % dt_string)
        f.write('Config: \n')
        for key, value in vars(args).items():
            key = str(key)
            value = str(value)
            f.write('%s:\t%s\n' % (key, value))
        f.write('\n')
        f.close()

    max_eval_score= 0.

    print('Starting training')
    for epoch in range(args.max_epochs):
        qa_model.train()
        epoch_loss = 0
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0
        for i_batch, a in enumerate(loader):
            qa_model.zero_grad()
            # so that don't need 'if condition' here
                        # scores = qa_model.forward(question_tokenized.cuda(), 
            #             question_attention_mask.cuda(), entities_times_padded.cuda(), 
            #             entities_times_padded_mask.cuda(), question_text)

            answers_khot = a[-1] # last one assumed to be target
            scores = qa_model.forward(a)

            loss = qa_model.loss(scores, answers_khot.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, args.max_epochs))
            loader.update()

        print('Epoch loss = ', epoch_loss)
        if (epoch + 1) % args.valid_freq == 0:
            print('Starting eval')
            eval_score, eval_log = eval(qa_model, valid_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k = args.eval_k)
            if eval_score > max_eval_score:
                print('Valid score increased') 
                save_model(qa_model, checkpoint_file_name)
                max_eval_score = eval_score
            # log each time, not max
            # can interpret max score from logs later
            append_log_to_file(eval_log, epoch, result_filename)


def save_model(qa_model, filename):
    print('Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to ', filename)
    return

if args.model != 'embedkgqa': #TODO this is a hack
    tkbc_model = loadTkbcModel('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name = args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))
else:
    tkbc_model = loadTkbcModel_complex('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name = args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))

if args.mode == 'test_kge':
    utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
    exit(0)

  
train_split = 'train'
test = args.test
if args.model == 'bert' or args.model == 'roberta':
    qa_model = QA_lm(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    #valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
elif args.model == 'embedkgqa':
    qa_model = QA_embedkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    #valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
elif args.model == 'cronkgqa' and args.supervision != 'hard':
    qa_model = QA_cronkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
    #valid_dataset = QA_Dataset_baseline(split=args.eval_split, dataset_name=args.dataset_name)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name)
elif args.model in ['tempoqr', 'entityqr', 'cronkgqa']:  #supervised models
    qa_model = QA_TempoQR(tkbc_model, args)
    if args.mode == 'train':
        dataset = QA_Dataset_TempoQR(split=train_split, dataset_name=args.dataset_name, args=args)
    #valid_dataset = QA_Dataset_TempoQR(split=args.eval_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_TempoQR(split=test, dataset_name=args.dataset_name, args=args)
else:
    print('Model %s not implemented!' % args.model)
    exit(0)

print('Model is', args.model)


if args.load_from != '':
    filename = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.load_from
    )
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from ', filename)
else:
    print('Not loading from checkpoint. Starting fresh!')

qa_model = qa_model.cuda()

if args.mode == 'eval':
    score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k = args.eval_k)
    exit(0)

result_filename = 'results/{dataset_name}/{model_file}.log'.format(
    dataset_name=args.dataset_name,
    model_file=args.save_to
)


train(qa_model, dataset, test_dataset, args,result_filename=result_filename)

# score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split="test", k=args.eval_k)
# log=["######## TEST EVALUATION FINAL (BEST) #########"]+log
# append_log_to_file(log,0,result_filename)

print('Training finished')
