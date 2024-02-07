from genericpath import isfile
from unittest import result
import torch
from sklearn.model_selection import train_test_split

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments
from transformers import BertForTokenClassification
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support
import evaluate
from datasets import Dataset, load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

import pandas as pd

import json
import os, sys
sys.path.append("../")
sys.path.append("../finetuning/")
# from utils import eval_datareader, get_tokenizers
from utils import get_tokenizer
import argparse

# from scripts import finetuning_hfbert
# PTLM_NAME = "bert-base-multilingual-cased"



def evaluate_pos(EXP_ID, eval_datapath, tokenizer_inpath,\
                  model_inpath, SEED = 42, save_results = True, \
                    HF_MODEL_NAME = None, EP_MODEL_NAME = None):
    '''Runs evaluation, also saves the results automatically in experiment records'''

    _, train_labels, _, dev_labels, test_data, test_labels = \
     eval_datareader.get_data(eval_datapath, format = "conllu", SEED = SEED)

    if not HF_MODEL_NAME and not EP_MODEL_NAME:
        assert os.path.isfile(tokenizer_inpath)
        print("Loading from path! ")
        tokenizer = get_tokenizer.train_or_load_tokenizer(tokenizer_inpath)
    else:
        print("Loading from HF! ")
        TOK_PATH = HF_MODEL_NAME if HF_MODEL_NAME else EP_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)


    pipe = pipeline("token-classification", model = model_inpath, tokenizer = tokenizer, device=0)

    all_labels = sorted(list({label for sent in train_labels+dev_labels+test_labels for label in sent.split()}))
    label2idx = {label:idx for idx, label in enumerate(all_labels)}
    test_labels_ids = [[label2idx[label] for label in sent.split()] for sent in test_labels]

    pred_labels = list()
    for sent in test_data:
        pred_labels.append([int(word["entity"].split("_")[1]) for word in pipe(sent) if not word["word"].startswith("#")])  # type: ignore

    pred_labels_ids = list()
    for idx, sent in enumerate(test_data):
        pred_labels_ids += [[int(word["entity"].split("_")[1]) for word in pipe(sent) if not word["word"].startswith("#")]]  # type: ignore

    true_labels = list()
    pred_labels = list()

    for idx, sent in enumerate(test_data):
        if len(pred_labels_ids[idx]) == len(test_labels_ids[idx]):
            true_labels += test_labels_ids[idx]
            pred_labels += pred_labels_ids[idx]

    scores_all = precision_recall_fscore_support(true_labels, pred_labels, average = "macro")
    micro_scores_all = precision_recall_fscore_support(true_labels, pred_labels, average = "micro")
    print(scores_all)
    scores = {"prec":scores_all[0], "rec":scores_all[1], "f1":scores_all[2],\
         "micro_f1":micro_scores_all[2]}

    if save_results:
        RECORD_FILE = "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/experiments_results_pos.json"
        save_results_to_file(EXP_ID, eval_datapath, scores, RECORD_FILE= RECORD_FILE)

    return scores



def evaluate_mt_bleu(DATAFILE_L1, DATAFILE_L2, tokenizer_inpath, model_inpath, SEED = 42,  \
                    charbleu = False, \
                    save_results = True, output_dir = None, \
                    EXP_ID = None):
    '''Runs evaluation for MT, also saves the results automatically in experiment records
    eval_datapath: eval DIR to the MT parallel data'''

    print("Evaluating MT! ")
    if charbleu:
        max_order = 18
    else:
        max_order = 4
    print(f"Char-level: {charbleu}, max order: {max_order}")
    #seq2seq Pipeline with model_inpath and tokenizer
    if os.path.isfile(tokenizer_inpath):
        print("Loading from path! ")
        tokenizer = get_tokenizer.train_or_load_tokenizer(tokenizer_inpath)
        # Set the max length of the tokenizer
        tokenizer.model_max_length = 512
    else:
        print("Loading from HF! ")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_inpath, model_max_length = 512)
    
    # tokenizer.decoder = decoders.WordPiece()
    # tokenizer = get_tokenizers.add_dialectid_tokens(tokenizer)
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(f"Device: {device}")
    pipe = pipeline("translation", model = model_inpath, tokenizer = tokenizer, max_length = 512, truncation = True, device=device)

    # Save MT outputs
    # scores = {"bho":{}, "mag":{}}
    # for lang in ["bho", "mag"]:

    print(f"Evaluating...")
    # hrl_sents, lrl_sents = \
        # eval_datareader.get_data_mt(eval_datapath, lang, SEED = SEED)
    dataset = load_dataset("text", data_files={"source": [DATAFILE_L1], \
            "target": [DATAFILE_L2]})

    # Create a new dataset that has source and target as columns
    dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
    # Take only 1000 examples
    dataset = dataset.select(range(500))
    
    true_sents = dataset["target"]
    pred_sents = list()
    for output in tqdm(pipe(KeyDataset(dataset, "source"), batch_size = 32, max_length = 512, truncation = True)):        
        for sent in output:
            token_ids = tokenizer.convert_tokens_to_ids(sent["translation_text"].split())
            pred = tokenizer.decode(token_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True)

            # Process the sentence to combine subwords
            pred = pred.replace(" ##", "")

            print(f"Pred: {pred}")
            pred_sents.append(pred)

    print(f"True sentences: {len(true_sents)}")
    print(f"Predicted sentences: {len(pred_sents)}")
    print(f"Sample true sentence: {true_sents[:3]}")
    print(f"Sample predicted sentence: {pred_sents[:3]}")
    # Get references
    
    
    output_path = os.path.join(output_dir, "preds.txt") #if not charbleu else os.path.join(output_dir, lang+"_preds_charbleu.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(pred_sents))

    if charbleu:
        # Put space after each character
        pred_sents = [" ".join(list(pred_sent)) for pred_sent in pred_sents]
        true_sents = [[" ".join(list(true_sent[0]))] for true_sent in true_sents]
    
    # Find BLEU score
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=pred_sents, references=true_sents, max_order=max_order)
    print(f"BLEU: {bleu_score}")
    # scores[lang] = bleu_score

    
    # Save inputs
    # with open(os.path.join(output_dir, "hin_sents.txt"), "w") as f:
    #     f.write("\n".join(hrl_sents))

    # if save_results:
    #     RECORD_FILE = "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/experiments_results_mt.json" \
    #         if not charbleu else "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/experiments_results_mt_charbleu.json"
    #     save_results_to_file(EXP_ID, DATAFILE_L1, scores, RECORD_FILE= RECORD_FILE)


def evaluate_mt_bleu_from_file(EXP_ID, eval_datapath, translations_file, save_results = True, charbleu = False, SEED = 42):
    '''Runs evaluation for MT if predictions (and true sents) are already saved in some file
    Also saves the results automatically in experiment records
    eval_datapath: eval DIR to the MT parallel data'''

    print("Evaluating MT! ")

    if charbleu:
        max_order = 10
    else:
        max_order = 4
    print("Char-level: {}, max order: {}".format(charbleu, max_order))
    #seq2seq Pipeline with model_inpath and tokenizer
    
    with open(eval_datapath, "r") as f:
        true_sents = f.read().split("\n")
    with open(translations_file, "r") as f:
        pred_sents = f.read().split("\n")
    
    true_sents = [[true_sent] for true_sent in true_sents]
    assert len(true_sents) == len(pred_sents)
    print("Sentences: ", len(true_sents))
        
    if charbleu:
        # Put space after each character
        pred_sents = [" ".join(list(pred_sent)) for pred_sent in pred_sents]
        true_sents = [[" ".join(list(true_sent[0]))] for true_sent in true_sents]
    
    # Find BLEU score
    bleu = evaluate.load("bleu")
    scores = bleu.compute(predictions=pred_sents, references=true_sents, max_order=max_order)
    print("Score: ", scores)

    # if save_results:
    #     RECORD_FILE = "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/experiments_results_mt.json" \
    #         if not charbleu else "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/experiments_results_mt_charbleu.json"
    #     save_results_to_file(EXP_ID, eval_datapath, scores, RECORD_FILE= RECORD_FILE)



def save_results_to_file(EXP_ID, eval_datapath, scores, RECORD_FILE = None):
    '''
    Hard code the outputs file path
    '''
    if not RECORD_FILE:
        RECORD_FILE = "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/experiments_results_pos.json"

    if os.path.isfile(RECORD_FILE):
        with open(RECORD_FILE, "r") as f:
            results = json.load(f)
    else:
        results = dict()

    # _, train_labels, _, _, test_data, test_labels = \
    #  eval_datareader.get_data(eval_datapath, SEED = 42)
    # all_labels = sorted(list({label for sent in train_labels for label in sent.split()}))
    # label2idx = {label:idx for idx, label in enumerate(all_labels)}

    # RECORD_FILE_CONF_MATRIX = "/home/nbafna/scratch/repos/large-language-models-for-related-dialects/evaluation/outputs/conf_matrix_hin_chb_finetuning.json"
    if "confusion_matrix" in scores:
    #     df_conf_matrix = pd.DataFrame(scores["confusion_matrix"], columns=all_labels,\
    #                                 index=all_labels)
    #     df_conf_matrix.to_json(RECORD_FILE_CONF_MATRIX, orient="index", indent=2)
        del scores["confusion_matrix"]

    idx = len(results)
    results[idx] = {
        "MODEL_ID": EXP_ID, 
        "results": scores,
        "dataset": eval_datapath}
    
    with open(RECORD_FILE, "w") as f:
        json.dump(results, f, indent = 2)


if __name__=="__main__":
    # print(PTLM_NAME)
    # eval_datapath = "../data/eval_POS/bho.nsurl.bis.conllu"
    # tokenizer_inpath="../finetuning_outputs/models/bho/ft_single_transfer_eqsub_pos.bho.ptlm_mono_eqsub.mag.batchsize_16.vocabsize_30522.epochs_5/tokenizer.json"
    # model_inpath = "../finetuning_outputs/models/bho/ft_single_transfer_eqsub_pos.bho.ptlm_mono_eqsub.mag.batchsize_16.vocabsize_30522.epochs_5/"
    # EXP_ID = "ft_single_transfer_eqsub_pos.bho.ptlm_mono_eqsub.mag.batchsize_16.vocabsize_30522.epochs_5"
    # evaluate_pos(EXP_ID, eval_datapath, tokenizer_inpath, model_inpath, SEED = 42, save_results=False)

    # DATAPATH = "../data/raw_parallel/loresmt/"
    # TOKENIZER_INPATH="../training_outputs/tokenizers/awa_bho_bra_mag_mai/lm_multi_abbmm.awa_bho_bra_mag_mai.batchsize_16.vocabsize_30522.epochs_20.json"
    # MODEL_INPATH = "../training_outputs_tali/models/abbhmm/tali_mt.abbhmm.lidpret_abbmm.encdecpret_abbmm.alpha_0.5.hardtau_0.1.maxlen_512.hrlsents_50000.epochs_10/checkpoint-375000/"

    # EXP_ID = "tali_mt.abbhmm.lidpret_abbmm.encdecpret_abbmm.alpha_0.5.hardtau_0.1.maxlen_512.hrlsents_50000.epochs_10"

    # Take the above arguments from argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATAFILE_L1", type=str, required=True, help="Path to the eval data directory")
    parser.add_argument("--DATAFILE_L2", type=str, required=True, help="Path to the eval data directory")
    parser.add_argument("--TOKENIZER_INPATH", type=str, default=None, help="Path to the tokenizer")
    parser.add_argument("--MODEL_INPATH", type=str, default=None, help="Path to the model")
    parser.add_argument("--EXP_ID", type=str, required=True, help="Experiment ID")
    parser.add_argument("--save_results", action="store_true", help="Save results to file")
    parser.add_argument("--task", type=str, default="translation", help="Task to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save the results")
    # Accept translated data from_file,
    parser.add_argument("--from_file", action="store_true", help="Evaluate from file")
    parser.add_argument("--translations_file", type=str, default=None, help="Path to the transformed data")
    


    
    args = parser.parse_args()
    task = args.task
    MODEL_INPATH = args.MODEL_INPATH
    task = "translation"
    MODEL_INPATH = None

    if task == "translation":
        if not args.from_file:
            print("Evaluating pipeline")
            evaluate_mt_bleu(args.DATAFILE_L1, args.DATAFILE_L2, args.TOKENIZER_INPATH, args.MODEL_INPATH, SEED = 42,  \
                    charbleu = False, \
                        save_results = args.save_results, output_dir = args.output_dir, \
                        EXP_ID = args.EXP_ID)
        else:
            # Assume we evaluate from file
            # n = 3
            # temperature = 0.4
            # alpha_clm = 0.75
            # for lang in ["mag"]: #, "mag"]:
            #     for temperature in [0.1, 0.2, 0.4, 0.8, 1]:
                
            #         EXP_ID = "hin_to_{}.n_{}.temperature_{}.alphaclm_{}".format(lang, n, temperature, alpha_clm)
            #         eval_datapath = "../data/raw_parallel/loresmt/hi2{0}.test.{0}".format(lang)
            #         transformed_datapath = "../data_transformation/outputs/{0}/hi2{0}.test.hi/hin_to_{0}.n_{1}.temperature_{2}.alphaclm_{3}.txt".format(lang, n, temperature, alpha_clm)
                    # transformed_datapath =  "../data/raw_parallel/loresmt/hi2{0}.test.hi".format(lang)
            print("Evaluating from file")
            # evaluate_mt_bleu_from_file(args.EXP_ID, args.DATAPATH, args.transformed_datapath, save_results = args.save_results, charbleu = True)
            evaluate_mt_bleu_from_file(args.DATAFILE_L1, args.DATAFILE_L2, args.TOKENIZER_INPATH, args.MODEL_INPATH, SEED = 42,  \
                    charbleu = False, \
                    save_results = args.save_results, translations_file = args.translations_file, \
                        EXP_ID = args.EXP_ID)
    # see_results()