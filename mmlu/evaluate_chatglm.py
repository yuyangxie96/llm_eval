import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForCausalLM , AutoTokenizer
#from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
#from hqq.core.quantize import *
import time
from typing import Optional

choices = ["A", "B", "C", "D"]

def get_choices():
    return ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # code from TensorRT-LLM
        output_len = 2
        top_k = 1
        top_p = 0.0
        batch_input_ids = [tokenizer.encode(prompt, return_tensors="pt").squeeze(0)]#.input_ids]#.cuda()]
        input_lengths = [x.size(0) for x in batch_input_ids]
        max_length = max(input_lengths)
        paddings = [
            torch.ones(max_length - l, dtype=torch.int32) * pad_id
            for l in input_lengths
        ]
        batch_input_ids = [
            torch.cat([pad, x])
            for x, pad in zip(batch_input_ids, paddings)
        ]
        batch_input_ids = torch.stack(batch_input_ids)
        batch_input_ids = batch_input_ids.cuda()
        with torch.no_grad():
            # Use default temperature and top_k
            outputs = model.generate(batch_input_ids,
                                    max_new_tokens=output_len,
                                    top_k=top_k,
                                    pad_token_id=tokenizer.eos_token_id
                                    )
            output_ids = outputs[0, input_lengths[0]:]
        pred = tokenizer.decode(output_ids, skip_special_tokens=True)
        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)


        # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        # decoder_input_ids = model._shift_right(decoder_input_ids)
        # logits = model(
        #     input_ids=input_ids, decoder_input_ids=decoder_input_ids
        # ).logits.flatten()
        # probs = (
        #     torch.nn.functional.softmax(
        #         torch.tensor(
        #             [
        #                 logits[tokenizer("A").input_ids[0]],
        #                 logits[tokenizer("B").input_ids[0]],
        #                 logits[tokenizer("C").input_ids[0]],
        #                 logits[tokenizer("D").input_ids[0]],
        #             ]
        #         ),
        #         dim=0,
        #     )
        #     .detach()
        #     .cpu()
        #     .numpy()
        # )
        # pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        # cor = pred == label
        # cors.append(cor)
        # all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model,padding_side="left",truncation_side='left',trust_remote_code=True) 
    model = AutoModelForCausalLM.from_pretrained(args.model,do_sample=True,trust_remote_code=True,torch_dtype=torch.bfloat16, device_map='auto')#.bfloat16().cuda()#.cuda()#.quantize(4).cuda()
    #model     = HQQModelForCausalLM.from_pretrained(args.model,do_sample=True,trust_remote_code=True)
    #quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
    #quant_config = BaseQuantizeConfig(nbits=3, group_size=64)
    #quant_config = BaseQuantizeConfig(nbits=2, group_size=16)
    #quant_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_scale=True)
    #HQQModelForCausalLM.quantize_model_(model, quant_config=quant_config)
    #model.quantize_model(quant_config=quant_config)
    #heads_per_gpu = len(model.encoder.block) // args.ngpu
    # device_map = {
    #     gpu: list(
    #         range(
    #             0 + (gpu * heads_per_gpu),
    #             (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
    #         )
    #     )
    #     for gpu in range(args.ngpu)
    # }
    #model.parallelize(device_map)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model.rsplit('/',1)[1]))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model.rsplit('/',1)[1])))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model.rsplit('/',1)[1]), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--gpu", type = str, default = "7")
    parser.add_argument("--data_dir", "-d", type=str, default="/workspace/data/mmlu_data")
    parser.add_argument("--save_dir", "-s", type=str, default="./output")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="/workspace/chatglm2-6b",
    )
    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #device = torch.device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    main(args)
