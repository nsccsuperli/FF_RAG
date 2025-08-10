import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, \
    GPT2Model, GPT2LMHeadModel, AutoConfig
import mypeft
from mypeft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
    AdaLoraConfig,
    AdaLoraModel,
)
print(mypeft.__file__)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
import numpy as np
import random
import copy
# import os
os.environ["HF_HUB_OFFLINE"] = "1"


def fl_finetune(
        # model/data params
        global_model: str = 'huggyllama/llama-7b',
        data_path: str = './data',
        output_dir: str = './fedgpt-llama7b-5-2/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 1,
        num_clients: int = 2,
        # Local training hyperparams
        local_batch_size: int = 128,  # 64,
        local_micro_batch_size: int = 16,
        local_num_epochs: int = 2,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        # aggregation mode
        stacking: bool = True,
        # evaluation
        # dev_data_path: str = './mmlu_test_100.jsonl',
        dev_data_path: str = './mmlu_test_1444.jsonl',
        # heterogeneous
        heter: bool = True,
        # local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4],
        local_ranks: List[int] = [24,8],
        # local_ranks: List[int] = [4, 8, 16, 16, 8, 8, 4, 4, 4, 4],
        zero_padding: bool = False,
        Adalora: bool = False,
        full: bool = False
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"stacking: {stacking}\n"
            f"heter: {heter}\n"
            f"full: {full}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    pretrained_model_name_or_path = "/fd/lct/skyline2006/llama-7b"
    print("ddp:", ddp)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if global_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token='your token',
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            # global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token="your token",
        )

    if global_model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        tokenizer = AutoTokenizer.from_pretrained(global_model, token='your_token', )
    else:
        tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path)
        # tokenizer = LlamaTokenizer.from_pretrained(global_model, token="your_token", )

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"


    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        # full_prompt = prompter.generate_prompt(
        #     data_point["instruction"],
        #     data_point["input"],
        #     data_point["output"],
        # )
        if data_path == './data/10':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["context"],
                data_point["response"],
            )
        elif data_path == './data_wiz/2' or data_path == './data_mix/20':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["output"],
            )
        else:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt



    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []

    # 训练ABCD
    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:

            print(f"Heterogeneous LoRA, Client {client_id} LoRA rank (r):", local_ranks[client_id])
            config = LoraConfig(
                r=local_ranks[client_id],
                lora_alpha=2 * local_ranks[client_id],
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                base_model_name_or_path=global_model,
            )
            model_client = copy.deepcopy(model)
            # 返回一个支持 LoRA 的模型副本，具有额外可训练参数
            model_client = get_peft_model(model_client, config)


            client = GeneralClient(client_id, model_client, data_path, output_dir)

   

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)


            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()


            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client # 删除了客户端，为了下次重新初始化启动


        '''
    
        print("Collecting the weights of clients and performing aggregation")
        # 聚合 A1 A2... =>C;  B1 B2... =>D; 
        # 聚合后的 保存在 output_dir/epoch/local_output_client_id
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       stacking,
                       lora_r,
                       heter,
                       local_ranks,
                       zero_padding,
                       full
                       )

        # print("FedAvg done: ", model_avg)


        for client_id in selected_clients_set:

            adapter_path = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id))


            config_ori = LoraConfig(
                base_model_name_or_path=global_model,
                # r=lora_r * num_clients,
                r=local_ranks[client_id],
                lora_alpha=lora_alpha * num_clients,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("Loading adapter from:", adapter_path)

            config_ori.save_pretrained(
                os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id)),
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map=device_map,
            )

            print("===000===",model)
            model = PeftModel.from_pretrained(model, adapter_path)
            print("===111===",model)
            print('aggregated model.............done')

            print('=====client_id:', client_id, 'model:', model)
        '''
        acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        print('Acc of Epoch', str(epoch), 'is:', acc)
        acc_list.append(acc)
        '''x_dir = os.path.join(output_dir, str(epoch))
        current_dir = x_dir # + "/temp/"
        print(current_dir)'''

        if stacking:
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(output_dir, str(epoch) + '/final'),
                                  load_in_8bit=False,
                                  torch_dtype=torch.float32,
                                  device_map=device_map, )
            print('model2',model)
            print('save final model')

        if epoch < (num_communication_rounds - 1):
            rm_dir = os.path.join(output_dir, str(epoch))
            os.system("rm -rf {xxxxx}".format(xxxxx=rm_dir))
            print("Remove the directory: ")

    print(acc_list)
    # os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = output_dir + 'llama-Flora-6-18log.txt'
    file = open(filename, 'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Log Saved")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
