import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, \
    GPT2Model, GPT2LMHeadModel, AutoConfig

from mypeft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
    AdaLoraConfig,
    AdaLoraModel,
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
import numpy as np
import random
import copy
import sys
# Freeze C and D，


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        param.requires_grad = True
        # if "lora_C" in name or "lora_D" in name:
        #     param.requires_grad = False
        if param.requires_grad:
            trainable_params += param.numel()
            print(name)

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable/ all params : {trainable_params / all_param} || "
    )


def fl_finetune(
        # model/data params
        global_model: str = 'huggyllama/llama-7b',
        data_path: str = './data',
        output_dir: str = './fedgpt-llama7b-5-2/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 1,
        num_clients: int = 10,
        # Local training hyperparams
        # local_batch_size: int = 1,  # 64,
        local_batch_size: int = 128,  # 64,
        local_micro_batch_size: int = 16,
        # local_micro_batch_size: int = 1,
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
        stacking: bool = False,
        # evaluation
        dev_data_path: str = './mmlu_test_1444.jsonl',
        # heterogeneous
        heter: bool = False,
        # local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4],
        local_ranks: List[int] = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
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
    # device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    pretrained_model_name_or_path = "/fd/lct/skyline2006/llama-7b"



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
        if data_path == './data/10':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["context"],
                data_point["response"],
            )
        elif data_path == './data_wiz/10' or data_path == './data_mix/20':
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

    # model = prepare_model_for_int8_training(model)
    if full == False:
        if stacking == False:
            if zero_padding:
                config_ori = LoraConfig(
                    base_model_name_or_path=global_model,
                    r=max(local_ranks),
                    lora_alpha=lora_alpha * max(local_ranks),
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            else:
                config = LoraConfig(
                    base_model_name_or_path=global_model,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)

        else:
            config_ori = LoraConfig(
                base_model_name_or_path=global_model,
                r=lora_r * num_clients,
                lora_alpha=lora_alpha * num_clients,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # 1 临时设置
            config_ori = LoraConfig(
                base_model_name_or_path=global_model,
                r=16,
                lora_alpha=lora_alpha * num_clients,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []
    paramrequeir_grad=[]
    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        # Org 
        model_org = copy.deepcopy(model)
        # 

        for client_id in selected_clients_set:
            if full == False:
                if Adalora:
                    config = AdaLoraConfig(
                        r=local_ranks[client_id],
                        lora_alpha=2 * local_ranks[client_id],
                        target_modules=lora_target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        base_model_name_or_path=global_model,
                    )
                    model_client = copy.deepcopy(model)
                    model_client = get_peft_model(model_client, config)
                else:
                    if stacking:
                        if heter:
                            config = LoraConfig(
                                r=local_ranks[client_id],
                                lora_alpha=2 * local_ranks[client_id],
                                target_modules=lora_target_modules,
                                lora_dropout=lora_dropout,
                                bias="none",
                                task_type="CAUSAL_LM",
                                base_model_name_or_path=global_model,
                            )
                            config_clint = LoraConfig(
                                r=local_ranks[client_id],
                                lora_alpha=2 * local_ranks[client_id],
                                target_modules=lora_target_modules,
                                lora_dropout=lora_dropout,
                                bias="none",
                                task_type="CAUSAL_LM",
                                base_model_name_or_path=global_model,
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                            for name, param in model_client.named_parameters():
                                if  param.requires_grad == True and name not in paramrequeir_grad:

                                    if "lora_C" in name or "lora_D" in name:
                                        # param.requires_grad = False
                                        print("CD")
                                    else:
                                        paramrequeir_grad.append(name)
                        else:
                            config = LoraConfig(
                                r=lora_r,
                                lora_alpha=lora_alpha,
                                target_modules=lora_target_modules,
                                lora_dropout=lora_dropout,
                                bias="none",
                                task_type="CAUSAL_LM",
                                base_model_name_or_path=global_model,
                            )
                            config_clint = LoraConfig(
                                r=lora_r,
                                lora_alpha=lora_alpha,
                                target_modules=lora_target_modules,
                                lora_dropout=lora_dropout,
                                bias="none",
                                task_type="CAUSAL_LM",
                                base_model_name_or_path=global_model,
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                            for name, param in model_client.named_parameters():
                                if  param.requires_grad == True and name not in paramrequeir_grad:

                                    if "lora_C" in name or "lora_D" in name:
                                        param.requires_grad = False
                                    else:
                                        paramrequeir_grad.append(name)
                    else:
                        if heter:
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
                            
                            model_client = get_peft_model(model_client, config)
                        else:
                            model_client = model

            else:
                model_client = model
            #     
            if epoch>0:
                model_client = PeftModel.from_pretrained(model_org, os.path.join(output_dir, str(epoch-1), "local_output_{}".format(client_id)))
            
            print(f'before:', print_trainable_parameters(model_client))
            for name, param in model_client.named_parameters():
                if epoch==0 and param.requires_grad==True and name not in paramrequeir_grad:
                    # print(name)
                    if "lora_C" in name or "lora_D" in name:
                        param.requires_grad = False
                        # print("name")
                    else:
                        paramrequeir_grad.append(name)
                    # paramrequeir_grad.append(name)
                if epoch >0 and name in paramrequeir_grad:
                    param.requires_grad = True
            print(f'after:',print_trainable_parameters(model_client))

                    # temp_a = name
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
            if epoch==0:
                print("continue")
                client.train()
                acc = global_evaluation(model_client, tokenizer, prompter, dev_data_path)
                print(acc)
                # model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training1(
                    # epoch, local_dataset_len_dict, previously_selected_clients_set)
            else:
                client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)

            # c=0
            # if c==0:
            #     print("\nTerminating the local training of Client_{}".format(client_id))
            #     model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
            #         epoch, local_dataset_len_dict, previously_selected_clients_set)
            #     del client
            # else:
            config_clint.save_pretrained(
                os.path.join(output_dir, str(epoch),"local_output_{}".format(client_id)),
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map=device_map,
            )

            del client



        print("Collecting the weights of clients and performing aggregation")
        # local_dataset_len_dict = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

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

        if full == False:
            if stacking:
                config_ori.save_pretrained(
                    os.path.join(output_dir, str(epoch),"local_output_{}".format(0)),
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
                print(os.path.join(output_dir, str(epoch)))

                model = PeftModel.from_pretrained(model, os.path.join(output_dir, str(epoch),"local_output_{}".format(0)))
                # model = PeftModel.from_pretrained(model, os.path.join(output_dir, str(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
                config.save_pretrained(
                    os.path.join(output_dir, str(epoch)),
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
        else:
            config = AutoConfig.from_pretrained(global_model)
            tokenizer.save_pretrained(os.path.join(output_dir, str(epoch)),
                                      load_in_8bit=False,
                                      torch_dtype=torch.float32,
                                      device_map=device_map, )
            config.save_pretrained(os.path.join(output_dir, str(epoch)),
                                   load_in_8bit=False,
                                   torch_dtype=torch.float32,
                                   device_map=device_map, )

            print('save model')
        if epoch == 0 :
            acc = 0.29244428051981697

        else:
            acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        print('Acc of Epoch', str(epoch), 'is:', acc)
        acc_list.append(acc)
        # ========== Accuracy ==========
        # Acc of Epoch 0 is: 0.29244428051981697

        # ========== Accuracy ==========
        # 0.29483970157369155
        # Acc of Epoch 1 is: 0.29483970157369155
        #
        # ========== Accuracy ==========
        # 0.2955706956952802
        # Acc of Epoch 2 is: 0.2955706956952802
        '''x_dir = os.path.join(output_dir, str(epoch))
        current_dir = x_dir # + "/temp/"
        print(current_dir)'''
        # arc_easy,hellaswag,mmlu,truthfulqa
        # os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_easy,hellaswag,mmlu,truthfulqa --device cuda --output_path {current_dir}".format(current_dir = current_dir))
        # os.system("lm_eval --model_args pretrained={current_dir},parallelize=True,load_in_4bit=False --tasks arc_easy,hellaswag,mmlu,truthfulqa --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
        # if stacking:
        #     
        #     model = model.merge_and_unload()
        #     model.save_pretrained(os.path.join(output_dir, str(epoch) + '/final'),
        #                           load_in_8bit=False,
        #                           torch_dtype=torch.float32,
        #                           device_map=device_map, )

        # if epoch < (num_communication_rounds - 1):
        #     rm_dir = os.path.join(output_dir, str(epoch))
        #     os.system("rm -rf {xxxxx}".format(xxxxx=rm_dir))

    print(acc_list)
    # os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = output_dir + 'llama-7b-frezon-DFLora-623log.txt'
    file = open(filename, 'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Log Saved")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
