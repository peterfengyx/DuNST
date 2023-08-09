# -*- coding: utf-8 -*-
from transformers import RobertaForSequenceClassification,RobertaTokenizer,GPT2Tokenizer,GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset

train_args = TrainingArguments(
    #generation_max_length=extra_args.max_body_len + extra_args.max_title_len + 2,
    output_dir="../ckpts/",
    seed=506,
    do_eval=True,
    per_device_eval_batch_size=32,
    local_rank=-1,
    xpu_backend='ncll',
    no_cuda=False,
    remove_unused_columns=False,
)

def count_ngram(text_samples, n, tokenizer=None):
    """
    Count the number of unique n-grams
    :param text_samples: list, a list of samples
    :param n: int, n-gram
    :return: the number of unique n-grams in text_samples
    """
    if len(text_samples) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    ngram = set()
    for sample in text_samples:
        if len(sample) < n:
            continue

        sample = list(map(str, sample))
        for i in range(len(sample) - n + 1):
            ng = ' '.join(sample[i: i + n])

            ngram.add(' '.join(ng))
    return len(ngram)

def evaluate_dist_scores(batch):#, model, tokenizer, prefix=""):
    '''
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, text_json_key=args.text_json_key, prepended_text_to_remove=args.prepended_text_to_remove)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    '''
    eval_loss = 0.0
    nb_eval_steps = 0
    #model.eval()

    dist_eval_samples = []
    num_tokens = 0
    #for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #sample_flattened = batch.reshape(-1)
    #dist_eval_samples.append(sample_flattened.tolist())
    #num_tokens += len(sample_flattened)
    for text in batch:
        dist_eval_samples.append(text)
        num_tokens+=len(text)
    
    dist1_score = count_ngram(dist_eval_samples, 1) / float(num_tokens)
    dist2_score = count_ngram(dist_eval_samples, 2) / float(num_tokens-len(batch))
    dist3_score = count_ngram(dist_eval_samples, 3) / float(num_tokens-len(batch)-len(batch))

    result = {"Dist-1": dist1_score, "Dist-2": dist2_score, "Dist-3": dist3_score}
    '''
    output_filename = "distK_" + args.eval_output_filename
    output_eval_file = os.path.join(eval_output_dir, prefix, output_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("***** Dist-1,2,3 Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    '''

    return result

def evaluate_cls(ds, model_path="../roberta-large_imdb"):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    testing_set = ds.map(tokenize_function, batched=True,remove_columns=["text"])

    #train_dataset=dataset_train.map(tokenize_function, batched=True)

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #data_collator = DataCollator(tokenizer=tokenizer, padding='longest')
    trainer = Trainer(
        model=model, args=train_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
        
    results = trainer.predict(testing_set, metric_key_prefix="predict")
    
    raw_preds = torch.tensor(results.predictions, device=model.device)
    probs = F.softmax(raw_preds, dim=-1)
    
    preds = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    pred_probs = probs[:, -1].detach().cpu().numpy()
    labels = results.label_ids
    
    accu = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, pred_probs)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    return  {"Accuracy":accu, "F1":f1, "AUC":auc, "Precision":prec, "Recall":recall}




def evaluate_ppl(ds):
    device="cuda:0"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    tokenizer.pad_token=tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    testing_set = ds.map(tokenize_function, batched=True,remove_columns=["text", "labels"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader=DataLoader(testing_set,batch_size=16,collate_fn=data_collator)
    
    model.to(device)
    model.eval()
    global_step=0
    loss=0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = (torch.ones(batch["input_ids"].shape)*(-100)).to(batch["input_ids"])
        last_non_masked_idx = torch.sum(batch["attention_mask"], dim=1) - 1
        for i in range(len(labels)):
            labels[i][:last_non_masked_idx[i]]=batch["input_ids"][i][:last_non_masked_idx[i]]
        with torch.no_grad():
            outputs = model(**batch, labels=labels)
        loss+=float(outputs.loss)
        global_step+=1
    return np.exp(loss/global_step)

def main():
    ds=Dataset.from_json("../data/generation.json")
    cls_results=evaluate_cls(ds)
    ppl_scores=evaluate_ppl(ds)
    print(cls_results)
    print("PPL:",ppl_scores)


if __name__ == "__main__":
    main()