from mypeft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d
import copy

def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, stacking, lora_r, heter, local_ranks, zero_padding, full):
    # 根据每个客户端的数据集大小计算聚合权重
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    print("Weights:", weights_array)
    all_single_weights = []
    # 加载每个客户端在当前epoch的模型参数
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),"local_abcd",
                                         "pytorch_model.bin")
        print("Loading single output dir:", single_output_dir)
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        all_single_weights.append(copy.deepcopy(single_weights))

        x = 0
        if full:
            # 第一个客户端：直接用其权重乘以聚合系数作为初始值, 后续客户端：将加权后的参数累加到结果中
            if k == 0:
                weighted_single_weights = single_weights
                for key in weighted_single_weights.keys():
                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
            else:
                for key in single_weights.keys():
                    weighted_single_weights[key] += single_weights[key] * (weights_array[k])
            
        else:
            if stacking:
                if zero_padding:
                    # 需要零填充，与其他客户端的local_rank对齐
                    max_lora = max(local_ranks)
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            # 第0维，pad行
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                            # 第一维，pad列
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                    else:
                        for key in single_weights.keys():
                            #print(single_weights[key].shape)
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                weighted_single_weights[key] += single_weights[key]
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                #print(single_weights[key][255,32])
                                weighted_single_weights[key] += single_weights[key]
                        
                else:
                    # 参数堆叠
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                            #print(weighted_single_weights[key].shape)
                            if heter:
                                x += 1
                                if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                            else:
                                if weighted_single_weights[key].shape[0] == lora_r:
                                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

                    else:
                        for key in single_weights.keys():
                            if heter:# 选择矩阵A
                                x += 1
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                    weighted_single_weights[key] = torch.cat(new, dim=0)#将已有的权重和新的进行相加，从64一直加到最后一个客户端的rank=4
                            else:
                                if single_weights[key].shape[0] == lora_r:
                                    new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                    weighted_single_weights[key] = torch.cat(new, dim=0)
                            
                            if heter:#选择矩阵B
                                if single_weights[key].shape[1] == local_ranks[client_id]:
                                    new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                    weighted_single_weights[key] = torch.cat(new, dim=1)
                            else:
                                if single_weights[key].shape[1] == lora_r:
                                    new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                    weighted_single_weights[key] = torch.cat(new, dim=1)

            else:
                if zero_padding:
                    max_lora = max(local_ranks)
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                    else:
                        for key in single_weights.keys():
                            #print(single_weights[key].shape)
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                weighted_single_weights[key] += single_weights[key]
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                #print(single_weights[key][255,32])
                                weighted_single_weights[key] += single_weights[key]
                else:
                    if k == 0:
                        weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                            single_weights.keys()}
                    else:
                        weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                            for key in
                                            single_weights.keys()}

    # merged_state 事实上应该处理的是将weighted_single_weights中的A和B赋值给C和D
    # merged_state = {}



    # 从本地字典取 lora_A 和 lora_B
    k=0
    global_site_model = {}
    for i in range(10):
        temp_a = ""
        temp_b = ""
        merged_state = {}

        for name, param in all_single_weights[k].items():
            if name.endswith("lora_A.weight"):
                merged_state[name] = param
                temp_a = name
            elif  name.endswith("lora_B.weight"):
                merged_state[name] = param
                temp_b = name
            # 从全局字典取 LoRA矩阵A赋值给lora_C 和 LoRA矩阵B赋值给lora_D
            elif name.endswith("lora_C.weight"):
                param = weighted_single_weights[temp_a]
                merged_state[name] = param
                global_site_model[temp_a] = param
            #
            else:
                param = weighted_single_weights[temp_b]
                merged_state[name] = param
                global_site_model[temp_b] = param
            # print(f"***merged-{k}:{name}: {param.shape}")
        agg_output_dir=  os.path.join(output_dir, str(epoch), "local_output_{}".format(k))

        # agg_output_dir=  os.path.join("local_output_{}".format(client_id))
        if not os.path.exists(agg_output_dir):
            os.makedirs(agg_output_dir)
        print("Saving aggregated abcd_adapter model at:", agg_output_dir)

        torch.save(merged_state, os.path.join(agg_output_dir, "adapter_model.bin"))

        k = k + 1
    global_output_dir = os.path.join(output_dir, str(epoch))
    print("Saving aggregated Global A&B model at:", global_output_dir)
    torch.save(global_site_model, os.path.join(global_output_dir, "adapter_model.bin"))

    return model



