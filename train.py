debug = True
SEED = 42


from typing import Optional, Tuple, Union, List
import torch
from torch import nn
from torch.nn import Embedding, CrossEntropyLoss
# from numpy import ndarray

from transformers import AutoTokenizer, EncoderDecoderModel, EncoderDecoderConfig, PretrainedConfig, \
    PreTrainedModel, BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, \
    BertEncoder, BertPooler, BertForSequenceClassification,BertPreTrainedModel, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right, DEPRECATION_WARNING
# from transformers.generation_utils import LogitsProcessorList, StoppingCriteriaList

from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset, load_dataset
from math import inf as INF
import numpy as np

import evaluate


import argparse
import os, sys

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
sys.path.append("../")
# from utils.variables import LANGS
from utils import get_tokenizer
import logging
# LANGS = sorted(LANGS)
# print(LANGS)

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations import TensorBoardCallback, is_tensorboard_available

class EncoderDecoderModelNew(EncoderDecoderModel):

    # init should initialize self.model_lid
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        # model_lid: Optional[BertForSequenceClassificationSmoothEmbeddings] = None,
    ):
        self.log = {}
        self.increment = 0
        super().__init__(config, encoder, decoder)

    
    def print_grads(self):
        for param in self.model_lid.parameters():
            print(param.requires_grad)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ooga: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import EncoderDecoderModel, BertTokenizer
        >>> import torch
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "bert-base-uncased", "bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints
        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size
        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = EncoderDecoderModel.from_pretrained("bert2bert")
        >>> # generation
        >>> generated = model.generate(input_ids)
        ```"""
        if self.training:
            self.log["something_to_visualize"] = self.increment
            self.increment += 1
        return super().forward(input_ids, attention_mask, decoder_input_ids, \
                               decoder_attention_mask, encoder_outputs, past_key_values, \
                                inputs_embeds, decoder_inputs_embeds, labels, use_cache, \
                                    output_attentions, output_hidden_states, return_dict, **kwargs)


        # global increment_for_tb, tb_writer
        ## Write both losses to tensorboard summary writer
        
        # tb_writer.add_scalar("loss/xe_loss", token_loss, increment_for_tb)
        # tb_writer.add_scalar("loss/lid_loss", lid_loss, increment_for_tb)
        # increment_for_tb += 1


        
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # print("In derived class for generate!")
        return super().generate(inputs, **kwargs)

class Seq2SeqTrainerTali(Seq2SeqTrainer):
    def _compute_loss(self, model, inputs, labels):
        print("In derived class for compute loss!")
        if self.args.label_smoothing == 0:
            if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
                # force training to ignore pad token
                logits = model(**inputs, use_cache=False)[0]
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                # compute usual loss via models
                loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        else:
            # compute label smoothed loss
            logits = model(**inputs, use_cache=False)[0]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
        return loss, logits

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loss, _ = self._compute_loss(model, inputs, labels)
        return loss
    
    # def _compute_loss(self, model, inputs, labels, labels_lid):
    #     if self.args.label_smoothing == 0:
    #         if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
    #             # force training to ignore pad token
    #             logits = model(**inputs, use_cache=False)[0]
    #             loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    #         else:
    #             # compute usual loss via models
    #             loss, logits = model(**inputs, labels=labels, labels_lid = labels_lid, use_cache=False)[:2]
    #     else:
    #         raise NotImplementedError("Label smoothing not implemented for TALI seq2seqtrainer!")
    #         # compute label smoothed loss
    #         logits = model(**inputs, use_cache=False)[0]
    #         lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    #         loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
    #     return loss, logits

    # def compute_loss(self, model, inputs):
    #     labels = inputs.pop("labels")
        # labels_lid = inputs.pop("labels_lid")
        # loss, _ = self._compute_loss(model, inputs, labels, labels_lid)
        # return loss
        # # Save past state if it exists


def init_tokenizer(TOKENIZER_INPATH, FILES):
    '''Note that if we are using a pretrained tokenizer,
    we should simply pass the HF key of the model as TOKENIZER_INPATH'''

    logging.info("Loading src tokenizer from {}".format(TOKENIZER_INPATH))
    tokenizer = get_tokenizer.train_or_load_tokenizer(TOKENIZER_INPATH,  \
        FILES = FILES)
    # Looks like HF MT doesn't support separate source and target tokenizers
    # logging.info("Loading tgt tokenizer from {}".format(TGT_TOKENIZER_INPATH))
    # tgt_tokenizer = get_tokenizer.train_or_load_tokenizer(TGT_TOKENIZER_INPATH, tokenizer)

    ### Optionally add language ID tokens
    # tokenizer = get_tokenizer.add_langid_tokens(tokenizer, LANGS)

    
    return tokenizer

def init_models(ENC_DEC_MODELPATH, tokenizer, PT_CKPT = None):
    '''Get seq2seq model'''

    # Initialize Seq2Seq model, input and output tokenizer, special hyperparameters
    if PT_CKPT:
        # First we check if there is some enc-dec checkpoint (along with cross-attention weights)
        logging.info("Loading encoder-decoder model from {}".format(PT_CKPT))
        model_enc_dec = EncoderDecoderModelNew.from_pretrained(PT_CKPT)
    elif ENC_DEC_MODELPATH:
        # If not, we check if the encoder and decoder can be initalized separately from some model
        logging.info("Loading encoder-decoder model from {}".format(ENC_DEC_MODELPATH))
        model_enc_dec = EncoderDecoderModelNew.from_encoder_decoder_pretrained(ENC_DEC_MODELPATH, ENC_DEC_MODELPATH)
    else:
        # If not, we initialize the encoder and decoder from scratch
        logging.info("Initializing encoder-decoder model from scratch")
        encoder_config = BertConfig(vocab_size=len(tokenizer), num_hidden_layers=6, num_attention_heads=4, hidden_size=512, intermediate_size=1024)
        decoder_config = BertConfig(vocab_size=len(tokenizer), num_hidden_layers=6, num_attention_heads=4, hidden_size=512, intermediate_size=1024)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        model_enc_dec = EncoderDecoderModelNew(config)

    ## Set model parameters
    # model_enc_dec.model_lid = model_lid
    # model_enc_dec.alpha = alpha
    # model_enc_dec.tau = tau
    # model_enc_dec.istauhard = istauhard
    if model_enc_dec.encoder.config.vocab_size !=len(tokenizer):
        model_enc_dec.encoder.resize_token_embeddings(len(tokenizer))
    if model_enc_dec.decoder.config.vocab_size != len(tokenizer):
        model_enc_dec.decoder.resize_token_embeddings(len(tokenizer))

    # print("In init: model lid print info 1")
    # model_lid.print_info()
    # print("In init: model lid print info 2")
    # model_enc_dec.model_lid.print_info()
    
    model_enc_dec.config.decoder_start_token_id = tokenizer.cls_token_id
    model_enc_dec.config.pad_token_id = tokenizer.pad_token_id

    model_size = sum(t.numel() for t in model_enc_dec.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    return model_enc_dec

def get_dataset_split(DATAFILE_L1, DATAFILE_L2, max_lines, tokenizer, max_length = 512):

    def preprocess_function(examples):
        # HF doesn't support separate source and target tokenizers, 
        # so we assume it's the same tokenizer for now.
        inputs = [ex for ex in examples["source"]]
        targets = [ex for ex in examples["target"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs
    
    logging.info("Loading Datasets...")
    dataset = load_dataset("text", data_files={"source": [DATAFILE_L1], \
        "target": [DATAFILE_L2]})
    
    if debug:
        print(dataset)
        # print(dataset["source"])
        # print(dataset["target"])
        print("Example:")
        print(dataset["source"][0])
        print(dataset["target"][0])
    
    # Create a new dataset that has source and target as columns
    dataset = Dataset.from_dict({"source": dataset["source"]["text"], "target": dataset["target"]["text"]})
    
    dataset = dataset.select(range(min(max_lines, len(dataset))))

    if debug:
        print("Examples:")
        print(dataset[0])
    
    logging.info("STARTING TOKENIZING")
    dataset = dataset.map(preprocess_function, batched = True, \
                                        remove_columns=["source", "target"])
    
    logging.info("DONE TOKENIZING!")
    dataset = dataset.with_format("torch")
    return dataset

def get_mt_dataset(DATADIR_L1, DATADIR_L2, max_lines, tokenizer, max_length = 512):

    '''
    This function assumes that the datadir contains train, dev, and test splits, 
    called "train", "dev", and "test"
    '''
    # splits = {"train", "test", "dev"}

    def get_split(split):
        DATAFILE_L1 = os.path.join(DATADIR_L1, split)
        DATAFILE_L2 = os.path.join(DATADIR_L2, split)
        return get_dataset_split(DATAFILE_L1, DATAFILE_L2, max_lines, tokenizer, max_length = max_length)

    train_dataset = get_split("train")
    dev_dataset = get_split("dev")
    test_dataset = get_split("test")

    ## Split dataset into train, dev, and test if no splits are provided
    # dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    # train_dataset = dataset["train"]
    # devtest_dataset = dataset["test"]
    # devtest_dataset = devtest_dataset.train_test_split(test_size=0.5, seed=SEED)
    # dev_dataset = devtest_dataset["train"]
    # test_dataset = devtest_dataset["test"]

    # Log all sizes
    # logging.info("Length of dataset: {}".format(len(dataset)))
    logging.info("Length of train dataset: {}".format(len(train_dataset)))
    logging.info("Length of dev dataset: {}".format(len(dev_dataset)))
    logging.info("Length of test dataset: {}".format(len(test_dataset)))
    
    return train_dataset, dev_dataset, test_dataset

def compute_metrics(pred):
    '''Compute BLEU score'''
    global tokenizer, bleu

    # Get predictions
    predictions = pred.predictions
    labels = pred.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    if sum([len(pred.split()) for pred in predictions]) == 0:
        return {'bleu': 0.0, \
                'brevity_penalty': 0, \
                'length_ratio': 0, 'translation_length': 0, 'reference_length': 0}
    bleu_metric = bleu.compute(predictions = predictions, references = labels)

    # Remove precisions
    bleu_metric = {k: v for k, v in bleu_metric.items() if k != "precisions"}

    return bleu_metric

class CustomTensorboardCallback(TensorBoardCallback):
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''We visualize things here'''
        model = kwargs.pop("model")
        something_to_visualize = model.log["something_to_visualize"]

        print(f"In custom callback, something to visualize: {something_to_visualize}")

        # Log to tensorboard
        self.tb_writer.add_scalar("something_to_visualize", something_to_visualize, state.global_step)


def main(args):
# TRAIN_FILES, DEV_FILES, 
# LID_MODELPATH, NUM_LID_LABELS, ENC_DEC_MODELPATH, alpha = 0.5, max_length = 512
# OUTPUT_DIR, LOG_DIR, epochs, batch_size

    global tokenizer
    global bleu

    # if "spanish" in args.ENC_DEC_MODELPATH:
    #     script = "lat"
    # elif "hindi" in args.ENC_DEC_MODELPATH:
    #     script = "dev"

    # Get seq2seq model and tokenizer
    logging.info("Initializing tokenizer...")
    FILES = [os.path.join(args.DATADIR_L1, "train"),\
             os.path.join(args.DATADIR_L2, "train"),\
                os.path.join(args.DATADIR_L1, "dev"),\
                    os.path.join(args.DATADIR_L2, "dev")]
    tokenizer = init_tokenizer(args.TOKENIZER_INPATH, \
                               FILES)

    logging.info("Initializing models...")
    model_enc_dec = init_models(args.ENC_DEC_MODELPATH, tokenizer, args.PT_CKPT)
    
    logging.info("Getting datasets...")
    # Get dataset splits, and preprocess them
    train_dataset, dev_dataset, test_dataset = \
    get_mt_dataset(args.DATADIR_L1, args.DATADIR_L2, max_lines=args.max_lines, tokenizer= tokenizer, max_length= args.max_length)
    
    # Instead of that, download some MT dataset from HF
    # tokenizer = AutoTokenizer.from_pretrained(args.ENC_DEC_MODELPATH)
    # train_dataset = load_dataset("wmt16", "de-en", split="train[:1%]").select(range(args.max_lines))
    # dev_dataset = load_dataset("wmt16", "de-en", split="test[:1%]").select(range(int(args.max_lines/9))) # This gives us a 9:1 ratio
    # train_dataset, dev_dataset = preprocess_wmt(train_dataset, dev_dataset, tokenizer, max_length = args.max_length)
    # test_dataset = dev_dataset
    # Print example
    logging.info(f"EXAMPLE: {train_dataset[0]} ")   

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_enc_dec)
    
    # Metric
    bleu = evaluate.load("bleu")

    # Initialize Seq2SeqTrainer
    logging.info("Initializing trainer...")

    # if args.resume_from_checkpoint: train_steps = 1
    # else: train_steps = len(train_dataset) * args.epochs // args.batch_size

    training_args = Seq2SeqTrainingArguments(
    output_dir=args.OUTPUT_DIR,
    resume_from_checkpoint=args.resume_from_checkpoint,
    overwrite_output_dir=False,
    num_train_epochs=args.epochs,
    # max_steps=train_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=args.LOG_DIR,
    predict_with_generate=True,
    report_to="tensorboard",
    logging_steps=100,
    save_strategy="steps",
    save_steps=2000, # For 15000 examples, this will save roughly every epoch with batch size 8
    evaluation_strategy="steps",
    eval_steps=100, 
    load_best_model_at_end=True,
    save_total_limit=2
    )

    trainer = Seq2SeqTrainer(
        model=model_enc_dec,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator= data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomTensorboardCallback],
    )   

    
    logging.info("STARTING TRAINING")
    logging.info(f"CUDA: {torch.cuda.is_available()}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logging.info("SAVING MODEL")
    model_enc_dec.save_pretrained(args.OUTPUT_DIR)

    # # Get performance and labels on test set
    if test_dataset:
        logging.info("STARTING EVALUATION")
        test_results = trainer.predict(test_dataset)
        test_metrics = test_results.metrics
        predictions = test_results.predictions
        labels = test_results.label_ids 
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode into text
        inputs = tokenizer.batch_decode(test_dataset["input_ids"], skip_special_tokens=True)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log examples
        logging.info("Logging examples...")
        for i in range(len(predictions[:10])):
            logging.info("Example {}: ".format(i))
            logging.info("Input: {}".format(inputs[i]))
            logging.info("Prediction: {}".format(predictions[i]))
            logging.info("Label: {}".format(labels[i]))
        # Log metrics
        logging.info("Logging metrics...")
        logging.info("Test metrics: {}".format(test_metrics))


        logging.info("DONE EVALUATION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATADIR_L1", type=str, default=None)
    parser.add_argument("--DATADIR_L2", type=str, default=None)
    parser.add_argument("--ENC_DEC_MODELPATH", type=str, default=None, help="Path to encoder model to initalize encoder/decoder (separately)")
    parser.add_argument("--TOKENIZER_INPATH", type=str, default=None, help="Path to tokenizer - if self-trained, put path. If None, \
                        the tokenizer from the encoder model will be used")
    parser.add_argument("--PT_CKPT", type=str, default=None, help="Path to PGN checkpoint")
    parser.add_argument("--max_length", type=int, default = 512)
    parser.add_argument("--OUTPUT_DIR", type=str, default="output_dir", help="Path to save model")
    parser.add_argument("--LOG_DIR", type=str, default="logs", help="Path to save tensorboard logs")
    parser.add_argument("--epochs", type=int, default = 20)
    parser.add_argument("--batch_size", type=int, default = 16)
    parser.add_argument("--max_lines", type=int, default = INF)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="Resume training from args.OUTPUT_DIR")
    # Take any additional approach-related parameters

    args = parser.parse_args()

    logging.basicConfig(filename=f"{args.LOG_DIR}/log.txt", filemode="w", format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)


    main(args)