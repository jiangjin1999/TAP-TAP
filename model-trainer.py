# from accelerate import Accelerator
import copy
from genericpath import exists
from modulefinder import Module
import random
from re import L
# from MeCab import Model
from datasets import load_metric, Metric
import json
from loguru import logger
from sklearn.feature_selection import SelectFdr
from sqlalchemy import false
from sympy import true
from tap import Tap
import numpy as np
from torch.optim import AdamW
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from model.phoneme_encoder import *
from model.audio_encoder import *
from model.modeling_bart import BartForConditionalGeneration
from transformers import (
    BertConfig,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    set_seed,
)
from transformers.models import bart
from typing import Optional, Tuple  # 将wav2vec processor 和 model 合并
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch, torchaudio
import h5py
from typing import Dict, List
import os
import shutil
from utils import EarlyStopping, CustomSchedule

from processor import DataProcessor, TextDataProcessor, TextInputExample
# from model.models import (BartForConditionalGeneration, )

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class Config(Tap):
    seed: int = 2022

    pwd: str = '/home/users/jiangjin/jiangjin_bupt/ASR_CORRECTION/Cross_modal/TAP/'

    # 需修改参数配置
    mode: str = 'train'    
    is_use_DDP = True

    current_dataset: str = 'AISHELL-1' #['AISHELL-1', 'AIDATATANG', 'thchs'][0]
    is_pretrained: bool = True
    is_phoneme: bool = True
    is_audio: bool = True

    is_jointly_train: bool = False
    is_multi_task_parameters: bool = True

    batch_size: int = 35
    #AISHELL-1:50
    #AIDATATANG: 35
    
    lambda_text: int = 1
    lambda_phoneme: int = 1
    lambda_audio: int = 1
    
    
    # 文件路径 参数配置
    model_type: str = 'nopretrained-' # default
    if is_pretrained is True:
        model_type = 'pretrained-'
    if is_phoneme is True and is_audio is True:
        if is_jointly_train is True:
            model_type = model_type + 'jointly-TAP-model'
        else:
            model_type = model_type + 'TAP-model'
    elif is_phoneme is True:
        if is_jointly_train is True:
            model_type = model_type + 'jointly-TP-model'
        else:
            model_type = model_type + 'TP-model'
    elif is_audio is True:
        if is_jointly_train is True:
            model_type = model_type + 'jointly-TA-model'
        else:
            model_type = model_type + 'TA-model'
    mode_mode_path: str = pwd + model_type
    mode_mode_path_dataset: str = mode_mode_path + '/' + current_dataset
    
    best_model_dir: str = mode_mode_path_dataset + '/model-checkpoint/'
    test_result_dir: str = mode_mode_path_dataset + '/result/'
    log_path: str =mode_mode_path_dataset + '/log/'
    tensorboard_path: str =mode_mode_path_dataset + '/tensorboard/' 

    is_zh: bool = False
    language: str = 'en'
    if current_dataset in ['AISHELL-1', 'AIDATATANG', 'thchs']:
        is_zh = True
        language = 'zh'

    text_data_dir: str = pwd +'data/'+ language 
    
    audio_feature_path: str = text_data_dir +'/' +current_dataset +'/audio-feature/wav2vec_feature.h5'

    pretrained_model: str = pwd + 'pretrained-model/'+language+'/BART'
    phoneme_model_path: str = pwd + 'pretrained-model/'+language+'/phoneme_model'
    Model_config = AutoConfig.from_pretrained(pretrained_model)

    shuffle: bool = True
    max_seq_length: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.02
    lr_scheduler_type: str = 'linear'
    num_warmup_steps: int = 500
    max_train_steps: int = 2000
    gradient_accumulation_steps: int = 1
    epochs: int = 100
    num_batch_per_evaluation: int = 10
    audio_encoder_input_dim: int = 1024
    audio_encoder_output_dim: int = 768

    # 模型相关 参数配置
    early_stop = EarlyStopping(patience=5)
    device: str = 'cuda'
    metric: str = 'cer'
    if language == 'en': metric = 'wer'
    early_stop_flag: str = False

    # arg for ddp
    local_rank = '0'

    # for CustomSchedule
    d_model = 768

    def get_device(self):
        """return the device"""
        if config.is_use_DDP is True:
            return torch.device(self.device, int(local_rank))
        else:
            return torch.device(self.device)


class ContextContainer:
    """Context data container for training
    """

    def __init__(self) -> None:
        """init the variables"""
        self.train_step: int = 0
        self.dev_step: int = 0
        self.epoch: int = 0

        self.train_cer: float = 1000
        self.dev_cer: float = 1000
        self.best_dev_cer: float = 1000
        self.test_cer: float = 1000

        self.loss = 0
        self.audio_loss = 0
        self.phoneme_loss = 0
        self.total_loss = 0
        self.dev_loss = 0
        self.output_loss = 0
        self.logits = 0
        self.labels = 0



class Trainer:
    """Trainer which can handle the train/eval/test/predict stage of the model
    """

    def __init__(
        self, config: Config,
        text_processor: DataProcessor,
        text_tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        metric: Metric,
        audio_processor:  Optional[DataProcessor]=None,
        audio_encoder:  Optional[Module]=None,
        # phoneme_processor: Optional[DataProcessor]=None,
        phoneme_encoder: Optional[Module]=None,
    ) -> None:
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_processor


        model.resize_token_embeddings(len(text_tokenizer))

        if self.config.is_use_DDP is True:
            model = model.to(self.config.get_device())
            self.model = DDP(model, device_ids=[int(self.config.local_rank)], output_device=[int(self.config.local_rank)], find_unused_parameters=True)
            if self.config.is_audio is True:
                audio_encoder = audio_encoder.to(self.config.get_device())
                self.audio_encoder = DDP(audio_encoder, device_ids=[int(self.config.local_rank)], output_device=[int(self.config.local_rank)], find_unused_parameters=True)
            if self.config.is_phoneme is True:
                phoneme_encoder = phoneme_encoder.to(self.config.get_device())
                self.phoneme_encoder = DDP(phoneme_encoder, device_ids=[int(self.config.local_rank)], output_device=[int(self.config.local_rank)], find_unused_parameters=True)
        else:
            self.model = model.to(self.config.get_device())
            if self.config.is_audio is True:
                self.audio_encoder = audio_encoder.to(self.config.get_device())
            if self.config.is_phoneme is True:
                self.phoneme_encoder = phoneme_encoder.to(self.config.get_device())

        self.metric = metric

        # 2. build text & audio dataloader
        if self.config.local_rank=='0':
            logger.info('init text  dataloaders ...')
            if self.config.is_audio is True:
                logger.info('init audio  dataloaders ...')
            if self.config.is_phoneme is True:
                logger.info('init phoneme  dataloaders ...')

        if self.config.is_use_DDP is True:
            self.train_dataloader = self.create_DDP_dataloader(
                dataset=text_processor.get_train_dataset(),
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
            )
            self.dev_dataloader = self.create_dataloader(
                dataset=text_processor.get_dev_dataset(),
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
            )
            self.test_dataloader = self.create_dataloader(
                dataset=text_processor.get_test_dataset(),
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
                )
            if self.config.is_phoneme is True:
                self.phoneme_train_dataloader = self.create_dataloader(
                    dataset=text_processor.get_train_dataset(),
                    shuffle=False,
                    collate_fn=self.conver_text_to_phoneme_feature,
                )
            else:
                self.phoneme_train_dataloader = self.train_dataloader
            if self.config.is_audio is True:
                self.audio_train_dataloader = self.create_dataloader(
                    dataset=text_processor.get_train_dataset(),
                    shuffle=False,
                    collate_fn=self.convert_audio_examples_to_features,
                )
            else:
                self.audio_train_dataloader = self.train_dataloader
        else:   
            self.train_dataloader = self.create_dataloader(
                dataset=text_processor.get_train_dataset(),
                shuffle=self.config.shuffle,
                collate_fn=self.convert_examples_to_features,
            )
            self.dev_dataloader = self.create_dataloader(
                dataset=text_processor.get_dev_dataset(),
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
            )
            self.test_dataloader = self.create_dataloader(
                dataset=text_processor.get_test_dataset(),
                shuffle=False,
                collate_fn=self.convert_examples_to_features,
            )
            if self.config.is_phoneme is True:
                self.phoneme_train_dataloader = self.create_dataloader(
                    dataset=text_processor.get_train_dataset(),
                    shuffle=self.config.shuffle,
                    collate_fn=self.conver_text_to_phoneme_feature,
                )
            else:
                self.phoneme_train_dataloader = self.train_dataloader
            if self.config.is_audio is True:
                self.audio_train_dataloader = self.create_dataloader(
                    dataset=text_processor.get_train_dataset(),
                    shuffle=self.config.shuffle,
                    collate_fn=self.convert_audio_examples_to_features,
                )
            else:
                self.audio_train_dataloader = self.train_dataloader

        # 3. init model related
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            # {
            #     "params": [p for n, p in self.audio_encoder.named_parameters()],
            #     "weight_decay": 0.0,
            # },
            {
                "params": [p for n, p in self.phoneme_encoder.named_parameters()],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=config.learning_rate
                               )

        self.config.max_train_steps = len(self.train_dataloader) * self.config.epochs

        self.lr_scheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=config.num_warmup_steps*2, # 前 * step 进行warm up（即让lr 从0-设定的lr）
            num_training_steps=config.max_train_steps*2, # 最大的step
        )
        # self.lr_scheduler = CustomSchedule(
        #     d_model=config.d_model,
        #     optimizer=self.optimizer,
        #     warmup_steps=config.num_warmup_steps, # 前 * step 进行warm up（即让lr 从0-设定的lr）
        # )

        self.context_data = ContextContainer()
        self._init_output_dir()
        self.writer: SummaryWriter = SummaryWriter(self.config.tensorboard_path)
        self.train_bar: tqdm = None


    def create_DDP_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        if self.config.is_use_DDP is True:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle, # self.config.shuffle,
                collate_fn=collate_fn,
                sampler=torch.utils.data.distributed.DistributedSampler(dataset)
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle, # self.config.shuffle,
                collate_fn=collate_fn,
                # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
            )

    def create_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle, # self.config.shuffle,
            collate_fn=collate_fn,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        )

    def convert_examples_to_features(self, examples: List[TextInputExample]):
        """convert the examples to features"""
        texts = [example.rec for example in examples]
        encoded_features = self.text_tokenizer.batch_encode_plus(
            texts,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        labels = [example.lab for example in examples]
        label_features = self.text_tokenizer.batch_encode_plus(
            labels,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return encoded_features['input_ids'], label_features['input_ids']

    def conver_text_to_phoneme_feature(self, examples: List[TextInputExample]):
        texts = [example.rec for example in examples]
        encoded_features = self.text_tokenizer.batch_encode_plus(
            texts,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        labels = [example.lab for example in examples]
        label_features = self.text_tokenizer.batch_encode_plus(
            labels,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        if self.config.language == 'zh':
            # logger.info('language is zh: using pinyin convertor')
            src_idx = label_features['input_ids'].flatten().tolist()
            chars = self.text_tokenizer.convert_ids_to_tokens(src_idx)
            pho_idx, pho_lens = pinyin_convertor.convert(chars)
        else:
            # logger.info('language is en: using phoneme convertor')
            src_idx = label_features['input_ids'].flatten().tolist()
            tokens = self.text_tokenizer.convert_ids_to_tokens(src_idx)
            chars = [self.text_tokenizer.convert_tokens_to_string(token).replace(' ','') for token in tokens]
            pho_idx, pho_lens = phoneme_convertor.convert(chars)
        return encoded_features['input_ids'], label_features['input_ids'], pho_idx, pho_lens

    def convert_audio_examples_to_features(self, audio_examples: List[TextInputExample]):
        "load audio from disk"
        f_wav2vec = h5py.File(self.config.audio_feature_path, 'r')
        speechs = []
        for i in range(len(audio_examples)):
            key = audio_examples[i].utt
            speechs.append(f_wav2vec[key][()])
        texts = [example.lab for example in audio_examples]
        encoded_features = self.text_tokenizer.batch_encode_plus(
            texts,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return torch.tensor(np.array(speechs)), encoded_features['input_ids']

    def _init_output_dir(self):
        if self.config.local_rank=='0':
            logger.info(f'init the output dir: {self.config.log_path}')
        if os.path.exists(self.config.log_path):
            pass
        else:
            os.makedirs(self.config.log_path)

    def _update_train_bar(self):
        infos = [f'epoch: {self.context_data.epoch}/{self.config.epochs}']

        loss = self.context_data.loss
        if torch.is_tensor(loss):
            loss = loss.detach().clone().cpu().numpy().item()
        infos.append(f'loss: <{loss}>')

        self.train_bar.update()
        self.train_bar.set_description('\t'.join(infos))

    def on_batch_start(self):
        '''handle the on batch start logits
        '''
        self.model.train()

    def on_batch_end(self):
        """handle the on batch training is ending logits
        """
        # 1. update global step
        self.context_data.train_step += 1

        self._update_train_bar()

        self.writer.add_scalar(
            'train/text-loss',
            scalar_value=self.context_data.loss,
            global_step=self.context_data.train_step,
        )
        
        self.writer.add_scalar(
            'train/audi-loss',
            scalar_value=self.context_data.audio_loss,
            global_step=self.context_data.train_step,
        )
        self.writer.add_scalar(
            'train/phoneme-loss',
            scalar_value=self.context_data.phoneme_loss,
            global_step=self.context_data.train_step,
        )
        self.writer.add_scalar(
            'train/total-loss',
            scalar_value=self.context_data.total_loss,
            global_step=self.context_data.train_step,
        )

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        # """
        if self.config.local_rank=='0':
            logger.info('\n')
            logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        for text_batch, phoneme_batch, audio_batch in zip(self.train_dataloader, self.phoneme_train_dataloader, self.audio_train_dataloader):
            
            self.on_batch_start()

            self.train_epoch_text(text_batch)
            
            if self.config.is_phoneme:
                self.train_epoch_phoneme(phoneme_batch)

            if self.config.is_audio:
                self.train_epoch_audio(audio_batch)

            # self.optimizer.zero_grad()    
            self.context_data.total_loss = self.context_data.loss + self.context_data.audio_loss + self.context_data.phoneme_loss

            if self.config.is_jointly_train:
                self.train_jointly()

            if self.config.early_stop_flag:
                if self.config.local_rank=='0':
                    logger.info('early stopping')
                    break

            self.on_batch_end()
    
    def train_epoch_text(self, text_batch):
        input_ids, labels = text_batch
        input_ids, labels = input_ids.to(
            self.config.get_device()), labels.to(self.config.get_device())

        self.on_batch_start()

        self.optimizer.zero_grad()    
        # forward on text data
        output: Seq2SeqLMOutput = self.model(
            input_ids=input_ids, labels=labels)

        self.context_data.loss = output.loss.sum().detach().cpu().numpy().item()
        self.context_data.output_loss = output.loss* self.config.lambda_text
        self.context_data.output_loss.backward()

        # output.loss.sum().backward() #calculate the gradient
        self.optimizer.step() # update the model para with gradient & for nn.DP 只有GPU_0上的模型参数得到了更新
        self.lr_scheduler.step()       

    def train_epoch_audio(self, audio_batch):
        # forward on the audio data
        self.audio_encoder.train()

        speech_values, labels = audio_batch
        speech_values, labels = speech_values.to(
            self.config.get_device()), labels.to(self.config.get_device())

        self.optimizer.zero_grad()
        speech_input_embeddings = self.audio_encoder(speech_values)
        output: Seq2SeqLMOutput = self.model(
            inputs_embeds=speech_input_embeddings,
            labels=labels
        )
        self.context_data.audio_loss = output.loss.sum().detach().cpu().numpy().item()
        self.context_data.output_loss = output.loss * self.config.lambda_audio
        self.context_data.output_loss.backward(retain_graph=True)
        # output.loss.sum().backward()
        self.optimizer.step()
        self.lr_scheduler.step()        


    def train_epoch_phoneme(self, phoneme_batch):
        input_ids, labels, pho_idx, pho_lens = phoneme_batch
        input_ids, labels, pho_idx = input_ids.to(
            self.config.get_device()), labels.to(
                self.config.get_device()), pho_idx.to(
                    self.config.get_device())
        pho_lens = torch.tensor(pho_lens).to(self.config.get_device())
        # input_shape = input_ids.size()
        self.phoneme_encoder.train()
        self.optimizer.zero_grad()

        phoneme_embedding = self.phoneme_encoder.forward(pho_idx, pho_lens, input_ids)
        output: Seq2SeqLMOutput = self.model(
            inputs_embeds=phoneme_embedding,
            labels=labels
        )
        self.context_data.phoneme_loss = output.loss.sum().detach().cpu().numpy().item()
        self.context_data.output_loss = output.loss* self.config.lambda_phoneme
        self.context_data.output_loss.backward()
        # output.loss.sum().backward()
        self.optimizer.step()
        self.lr_scheduler.step()


    def train_jointly(self,):
        # self.optimizer.zero_grad()
        # self.context_data.total_loss.sum().backward() #calculate the gradient
        self.context_data.output_loss  = self.context_data.output_loss/self.context_data.output_loss * self.context_data.total_loss
        self.context_data.output_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()



        


    def evaluate(self, dataloader,):
        """handle the logit of evaluating

        Args:
            epoch (int): _description_
        """
        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []
        # 这里因为tqdm 中包含 tqdm 所以，暂时采用logger方式
        # for text_batch in tqdm(dataloader, desc='evaluation stage ...'):
        for text_batch in dataloader:
            with torch.no_grad():
                input_ids, labels = text_batch
                input_ids, labels = input_ids.to(
                    self.config.get_device()), labels.to(self.config.get_device())

                # forward on dev/test data
                # add .module for multi-GPU
                Output: Seq2SeqLMOutput = self.model(
                    input_ids=input_ids)

                generated_tokens = torch.argmax(Output.logits, dim=2)
                generated_tokens = generated_tokens.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                decoded_preds = self.text_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                decoded_labels = self.text_tokenizer.batch_decode(
                    labels, skip_special_tokens=True)

                decoded_preds = [decoded_pred.replace(' ','') for decoded_pred in decoded_preds]
                decoded_labels = [decoded_label.replace(' ','') for decoded_label in decoded_labels]

                all_decoded_preds = all_decoded_preds + decoded_preds
                all_decoded_labels = all_decoded_labels + decoded_labels

        metric_score = self.metric.compute(
            predictions=all_decoded_preds, references=all_decoded_labels)

        self.model.train()
        return metric_score

    def on_evaluation_end(self, metric_score):
        '''always save the best model'''
        if self.context_data.best_dev_cer > metric_score:
            self.save_model(self.config.best_model_dir)
            self.context_data.best_dev_cer = metric_score
            if self.config.local_rank=='0':
                logger.info('\n')
                logger.info(f'dev/best_cer is {self.context_data.dev_cer}')
                self.writer.add_scalar(
                    tag='dev/best_cer',
                    scalar_value=self.context_data.best_dev_cer,
                    global_step=self.context_data.train_step
                )
            self.context_data.test_cer = self.predict('test')
            self.writer.add_scalar(
                tag='test/cer',
                scalar_value=self.context_data.test_cer,
                global_step=self.context_data.train_step
            )


    def save_model(self, path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        if self.config.is_use_DDP is True:
            torch.save(self.model.module.state_dict(), path+'/checkpoint_best.pt')
        else:
            torch.save(self.model.state_dict(), path+'/checkpoint_best.pt')

    def train(self):
        """the main train epoch"""
        if self.config.local_rank=='0':
            logger.info('start training ...')
            logger.info(f'  num example = {len(self.train_dataloader)}')
            logger.info(f'  num epochs = {self.config.epochs}')
            logger.info(f'  total train batch size (parallel) = {self.config.batch_size}' )
            logger.info(f'  total optimization step = {self.config.max_train_steps}')

        self.on_train_start()
        for _ in range(self.config.epochs):            
            self.context_data.epoch += 1
            self.train_epoch()

            self.on_epoch_end()
            if self.config.early_stop_flag:
                if self.config.local_rank=='0':
                    logger.info('early stopping on train epoch')
                break

    def on_epoch_end(self):
        self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        self.config.early_stop_flag = self.config.early_stop.step(self.context_data.dev_cer)
        if self.config.local_rank=='0':
            logger.info('\n')
            logger.info(f'dev/cer is {self.context_data.dev_cer}')
        self.writer.add_scalar(
            tag='dev/cer',
            scalar_value=self.context_data.dev_cer,
            global_step=self.context_data.train_step
        )


        self.on_evaluation_end(self.context_data.dev_cer)
        
    def on_train_start(self):
        '''inite the dev and test cer'''
        # self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        # self.context_data.test_cer = self.evaluate(self.test_dataloader)
        # self.writer.add_scalar(
        #     tag='dev/cer',
        #     # scalar_value=self.context_data.dev_cer,
        #     scalar_value=0.2701,
        #     global_step=self.context_data.dev_step
        # )
        # self.writer.add_scalar(
        #     tag='test/cer',
        #     # scalar_value=self.context_data.test_cer,
        #     scalar_value=0.2431,
        #     global_step=self.context_data.dev_step
        # )

    def predict(self, FLAG: Optional[str] = None,):
        """ predict the example
            test_dataset = ['test_aidatatang', 'test_magicdata', 'test_thchs']
        """
        # self.load_model(self.config.best_model_dir)
        dataloader = self.test_dataloader

        logger.info('start predicting ...')
        if FLAG is not None:
            pass
        else:
            self.load_model(self.config.best_model_dir + 'checkpoint_best.pt')

        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []

        for text_batch in dataloader:
            with torch.no_grad():
                input_ids, labels = text_batch
                input_ids, labels = input_ids.to(
                    self.config.get_device()), labels.to(self.config.get_device())

                # forward on dev/test data
                # add .module for multi-GPU
                Output: Seq2SeqLMOutput = self.model(
                    input_ids=input_ids)

                generated_tokens = torch.argmax(Output.logits, dim=2)
                generated_tokens = generated_tokens.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                decoded_preds = self.text_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                decoded_labels = self.text_tokenizer.batch_decode(
                    labels, skip_special_tokens=True)

                decoded_preds = [decoded_pred.replace(' ','') for decoded_pred in decoded_preds]
                decoded_labels = [decoded_label.replace(' ','') for decoded_label in decoded_labels]

                all_decoded_preds = all_decoded_preds + decoded_preds
                all_decoded_labels = all_decoded_labels + decoded_labels

        metric_score = self.metric.compute(
            predictions=all_decoded_preds, references=all_decoded_labels)

        self.save_test_result(all_decoded_preds, all_decoded_labels, self.config.current_dataset)

        self.context_data.test_cer = metric_score
        # self.writer.add_scalar(
        #     tag='test/'+self.config.current_dataset+'_cer',
        #     scalar_value=self.context_data.test_cer,
        #     global_step=self.context_data.dev_step
        # )
        if self.config.local_rank=='0':
            logger.info(f'test/cer is {self.context_data.test_cer}')
        # add test cer every time evaluate test data
        self.model.train()
        return metric_score

    def load_model(self, path):
        if self.config.local_rank=='0':
            logger.info('load model ...')
        self.model.load_state_dict(torch.load(path))

    def save_test_result(self, all_decoded_preds, all_decoded_labels, test_data_name):
        # for text_modal: add additional 'text_modal_' to distinguish
        # ['cross_modal', 'text_modal']
        if os.path.exists(self.config.test_result_dir):
            pass
        else:
            os.makedirs(self.config.test_result_dir) 
        with open(config.test_result_dir+'T_modal_'+test_data_name+'.txt', 'w') as f_result:
            data_output_list = []
            for item_pred, item_label in zip(all_decoded_preds, all_decoded_labels):
                data_output_list.append(item_pred + ' ' + item_label + '\n') 
            f_result.writelines(data_output_list)


def set_my_seed(seed):
    '''random:
        python
        Numpy'''
    set_seed(seed)
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True



if __name__ == "__main__":
    config: Config = Config().parse_args(known_only=True)

    set_my_seed(config.seed)
    if os.path.exists(config.mode_mode_path_dataset):
        pass
    else:
        os.makedirs(config.mode_mode_path_dataset)
        

    if config.is_use_DDP is True:
        # 新增1:依赖
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        # # 新增：从外面得到local_rank参数
        # import argparse
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--local_rank", default=-1)
        # FLAGS = parser.parse_args()
        # local_rank = FLAGS.local_rank
        local_rank = config.local_rank

        # print(local_rank)

        # 新增：DDP backend初始化
        torch.cuda.set_device('cuda:'+str(local_rank))
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    if config.is_pretrained==True:
        MODEL_TYPE = BartForConditionalGeneration.from_pretrained(config.pretrained_model)
    else:
        MODEL_TYPE = AutoModelForSeq2SeqLM.from_config(config.Model_config)

    if config.is_phoneme is True:
        phoneme_encoder_path = config.phoneme_model_path
        phoneme_config = BertConfig.from_pretrained(phoneme_encoder_path)
        if config.language == 'en':
            Phoneme_encoder: phoneme_encoder = phoneme_encoder(phoneme_config)
        elif config.language == 'zh':
            Phoneme_encoder: pinyin_encoder = pinyin_encoder.from_pretrained(phoneme_encoder_path, config=phoneme_config)
    else:
        Phoneme_encoder = None
    
    if config.is_audio is True:
        Audio_encoder: audio_encoder = audio_encoder(
        mlp_dim=config.audio_encoder_input_dim, fc_output_dim=config.audio_encoder_output_dim)
    else:
        Audio_encoder = None


    trainer = Trainer(
        config,
        text_processor=TextDataProcessor(
            config.text_data_dir, config),
        text_tokenizer=AutoTokenizer.from_pretrained(config.pretrained_model),
        model=MODEL_TYPE,
        phoneme_encoder=Phoneme_encoder,
        audio_encoder=Audio_encoder,
        metric=load_metric(config.metric)
    )
    if config.mode == 'train':
        logger.add(os.path.join(config.log_path, 'train.'+config.current_dataset+'.T-model-log.txt'))
        if config.local_rank=='0':
            logger.info(config)
        trainer.train()
    elif config.mode == 'test':
        logger.add(os.path.join(config.log_path, 'test.'+config.current_dataset+'.T-model-log.txt'))
        if config.local_rank=='0':
            logger.info(config)
        trainer.predict()



