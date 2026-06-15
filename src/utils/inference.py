import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import os 
import vllm
from openai import OpenAI
from anthropic import Anthropic, types
from xai_sdk import Client as GrokClient
from xai_sdk.chat import system, user
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from types import MethodType
import hashlib
import json
import time
import os
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from typing import List

MAX_VLLM_BATCH_SIZE = 600_000
MAX_OPENAI_BATCH_SIZE = 45_000
MAX_GROK_BATCH_SIZE = 500
MAX_GROK_REST_BATCH_SIZE = 200
ERR_TOKEN = "ERR"

def is_mistral_chat_model(model_name: str) -> bool:
    return model_name in ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "mistralai/Mistral-Large-Instruct-2407")

def hash_batch(model: str, inputs: list[str], gen_kwargs: dict) -> str:
    """
    Create a stable hash so the same batch is not resubmitted.
    """
    h = hashlib.sha256()
    h.update(model.encode())
    for k in sorted(gen_kwargs):
        h.update(f"{k}={gen_kwargs[k]}".encode())
    for text in inputs:
        h.update(text.encode())
    return h.hexdigest()


def _get_tensor_parallel_size():
    """
    Infer tensor parallel size from CUDA_VISIBLE_DEVICES.
    If unset, default to 1 (single GPU).
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return 1
    num_gpus = len(cuda_visible.split(","))
    return num_gpus


def _get_tokenizer(model:str)->AutoTokenizer:
    tokenizer_args = {}

    if "gemma" in model:
        print(f"Gemma model detected, enabling right padding.") 
        # Gemma models were trained with right padding hence padding_side needs to be set to 'right' 
        # https://github.com/huggingface/transformers/issues/30004

        tokenizer_args["padding_side"] = "right"
    else:
        tokenizer_args["padding_side"] = "left"
    
    tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_args)

    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    
    return tokenizer

def _get_model_forEval(model:str)->AutoModelForCausalLM:

    _model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype="bfloat16")

    _model.eval()

    return _model


def make_prompt(user_prompt:str, sys_prompt:str=None, user_only:bool = False)->list[dict]:
    if not user_only:
        return [
            {"role": 'system', 'content': sys_prompt},
            {'role': 'user', 'content':user_prompt}
        ]
    # Some gemma models do not have a 'system' role: 
    # https://ai.google.dev/gemma/docs/core/prompt-structure#system-instructions
    # https://github.com/abetlen/llama-cpp-python/issues/1580
    # For such models, set user_only to True
    else:
        return [
            {'role': 'user', 'content': user_prompt}
        ]

def run_inference(backend:str, model:str, inputs:list[str], engine_kwargs, sys_prompt=None, **gen_kwargs):

    if backend == "vllm":
        return _run_vllm(model, inputs, engine_kwargs, sys_prompt, **gen_kwargs)
    elif backend == "openai":
        return _run_openai(model,inputs,**gen_kwargs)
    elif backend == "claude":
        return _run_claude(model, inputs, **gen_kwargs)
    elif backend == "grok":
        return _run_grok(model,inputs, **gen_kwargs)
    elif backend == "grok_REST":
        return _run_grok_REST(model,inputs, **gen_kwargs)
    elif backend == "transformers":
        return _run_transformers(model, inputs, engine_kwargs, sys_prompt, **gen_kwargs)
    else:
        raise(f"Unsupported backend: {backend}")
    

def _run_vllm(model:str,inputs:list[str], engine_kwargs, sys_prompt, **gen_kwargs)->list[str]:

    num_chunks = (len(inputs) + MAX_VLLM_BATCH_SIZE - 1) // MAX_VLLM_BATCH_SIZE
    outputs = []

    os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'

    config = AutoConfig.from_pretrained(model).get_text_config()

    tokenizer = _get_tokenizer(model)

    if 'tensor_parallel_size' not in engine_kwargs:
        engine_kwargs['tensor_parallel_size'] = _get_tensor_parallel_size()
    
    engine_kwargs['model'] = model

    chunks = [
        inputs[i * MAX_VLLM_BATCH_SIZE : min((i + 1) * MAX_VLLM_BATCH_SIZE, len(inputs))]
        for i in range(num_chunks)
    ]

    if is_mistral_chat_model(model):
        engine_kwargs["tokenizer_mode"] = "mistral"
        engine_kwargs["config_format"] = "mistral"
        engine_kwargs["load_format"] = "mistral"


    _model = vllm.LLM(**engine_kwargs)
    
    if 'do_sample' in gen_kwargs and gen_kwargs['do_sample'] == False:
        _gen_kwargs = vllm.SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens = gen_kwargs['max_new_tokens'],
            **{k:v for k,v in gen_kwargs.items() if k not in ('temperature', 'top_p', 'do_sample', 'max_new_tokens', 'use_cache')}
        )
    
    else:
        _gen_kwargs = vllm.SamplingParams(
            max_tokens = gen_kwargs['max_new_tokens'],
            **{k:v for k,v in gen_kwargs.items() if k not in ('max_new_tokens', 'use_cache')}
        )
    

    for chunk_id in range(num_chunks):
        chunk = inputs[chunk_id*MAX_VLLM_BATCH_SIZE:min(((chunk_id+1)*MAX_VLLM_BATCH_SIZE), len(inputs))]

        outputs.extend(_run_vllm_chunk(_model, tokenizer,model,chunk,sys_prompt, _gen_kwargs))
    
    return outputs


def _run_vllm_chunk(model:vllm.LLM, tokenizer:AutoTokenizer, model_name:str, chunk:list[str], sys_prompt, gen_kwargs)->list[str]:

    if "gemma" in model_name:
        prompts = [make_prompt(user_prompt=x+(sys_prompt if sys_prompt else ""), user_only=True) for x in chunk]
    else:
        prompts = [make_prompt(user_prompt=x, sys_prompt=(sys_prompt if sys_prompt else ""), user_only=False) for x in chunk]
        
    if not is_mistral_chat_model(model_name):
        strings = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        tokens = [tokenizer(string, add_special_tokens=False).input_ids for string in strings]
            
        tokens  = [vllm.TokensPrompt(prompt_token_ids=tok_ids) for tok_ids in tokens]

        outputs = model.generate(tokens, sampling_params=gen_kwargs, use_tqdm=True)
        out_strings = [output.outputs[0].text for output in outputs]

    else:
        out_strings = model.chat(prompts, sampling_params=gen_kwargs, use_tqdm=True)

    return out_strings

def _run_openai(model:str, inputs:list[str], **gen_kwargs):

    if len(inputs) <= MAX_OPENAI_BATCH_SIZE:
        return _run_openai_batch(model, inputs, 0, **gen_kwargs)
    
    outputs = [None] * len(inputs)
    any_pending = False

    num_batches = (len(inputs) + MAX_OPENAI_BATCH_SIZE - 1) // MAX_OPENAI_BATCH_SIZE

    for batch_id in range(num_batches):
        start = batch_id * MAX_OPENAI_BATCH_SIZE
        end = min((batch_id + 1) * MAX_OPENAI_BATCH_SIZE, len(inputs))
        chunk = inputs[start:end]

        
        result = _run_openai_batch(
            model,
            chunk,
            batch_index=batch_id,
            **gen_kwargs
        )

        if result is None:
            any_pending = True
        else:
            outputs[start:end] = result

    if any_pending:
        return None

    return outputs

def _run_openai_batch(model:str, inputs:list[str], batch_index:int, **gen_kwargs):
    client = OpenAI()

    for k in gen_kwargs:
        if isinstance(gen_kwargs[k], DictConfig):
            gen_kwargs[k] = OmegaConf.to_container(gen_kwargs[k], resolve=True)

    if len(inputs) > 45_000:
        raise ValueError("Too many inputs for OpenAI batch API")

    batch_hash = hash_batch(model, inputs, {**gen_kwargs, "batch_index": batch_index})
    batch_dir = os.path.join("data", "cache", "openai_batches", batch_hash)
    os.makedirs(batch_dir, exist_ok=True)

    meta_path = os.path.join(batch_dir, "batch_meta.json")
    input_path = os.path.join(batch_dir, "batch_input.jsonl")
    output_path = os.path.join(batch_dir, "batch_output.jsonl")
    error_path = os.path.join(batch_dir, "batch_error.jsonl")

    if os.path.exists(output_path):
        outputs = {}
        with open(output_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj["custom_id"]

                resp_body = obj.get("response", {}).get("body", {})
                output = resp_body.get("output", [])
                text=""
                for item in output:
                    if "text" in item and isinstance(item["text"], str):
                        text = item["text"]
                        break
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                text = block.get("text", "")
                                break
                            if text:
                                break
                outputs[cid] = text
        if len(outputs)!= len(inputs):
            raise RuntimeError(
                f"Expected {len(inputs)} outputs, got {len(outputs)}"
            )

        return [outputs[f"row_{i}"] for i in range(len(inputs))]
    
    if os.path.exists(error_path):
        raise RuntimeError(f"Batch previously completed with errors; see {error_path}")
    
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))

        primary_id = meta["batch_id"]
        retry_attempts = meta["retry_attempts"]

        primary_batch = client.batches.retrieve(primary_id)

        if len(retry_attempts) == 0:
            print(
                f"[OpenAI Batch] primary={primary_id} "
                f"(completed={primary_batch.request_counts.completed}/"
                f"{primary_batch.request_counts.total})"
            )

            if primary_batch.status in ("failed", "expired", "cancelled"):
                os.remove(meta_path)
                raise RuntimeError(
                    f"Primary batch ended with status = {primary_batch.status}"
                )

            if primary_batch.status!="completed":
                return None
            
            outputs = {}
            if primary_batch.output_file_id:
                content = client.files.content(
                    primary_batch.output_file_id
                ).text

                for line in content.splitlines():
                    obj = json.loads(line)
                    outputs[obj["custom_id"]] = obj
            
            if primary_batch.error_file_id:
                error_content = client.files.content(
                    primary_batch.error_file_id
                ).text

                failed_ids = [
                    json.loads(line)["custom_id"]
                    for line in error_content.splitlines()
                ]

                print(f"[OpenAI Batch] {len(failed_ids)} rows failed.")

                answer = input(
                    f"Resubmit {len(failed_ids)} failed rows? (y/n): "
                ).strip().lower()

                if answer != "y":
                    tmp_path = output_path + ".tmp"
                    with open(tmp_path, "w") as f:
                        for i in range(len(inputs)):
                            cid = f"row_{i}"
                            f.write(json.dumps(outputs.get(cid, ERR_TOKEN)) + "\n")
                    
                    os.replace(tmp_path, output_path)
                    return _run_openai_batch(model, inputs, batch_index, **gen_kwargs)

                retry_input_path = os.path.join(batch_dir, "retry_input_0.jsonl")

                with open(retry_input_path, "w") as f:
                    for cid in failed_ids:
                        idx = int(cid.split("_")[1])
                        record = {
                            "custom_id": cid,
                            "method": "POST",
                            "url": "/v1/responses",
                            "body": {
                                "model": model,
                                "input": inputs[idx],
                                **gen_kwargs,
                            },
                        }
                        f.write(json.dumps(record)+"\n")
                
                with open(retry_input_path, "rb") as f:
                    file_obj = client.files.create(file=f, purpose="batch")
                
                retry_batch = client.batches.create(
                    input_file_id=file_obj.id,
                    endpoint="/v1/responses",
                    completion_window="24h"
                )

                meta["retry_attempts"].append(retry_batch.id)
                json.dump(meta, open(meta_path, "w"), indent=2)

                print(f"[OpenAI Batch] Retry Batch 0 created: {retry_batch.id}")
                return None
            
            tmp_path = output_path + ".tmp"
            with open(tmp_path, "w") as f:
                for i in range(len(inputs)):
                    cid = f"row_{i}"
                    f.write(json.dumps(outputs[cid])+"\n")
            
            os.replace(tmp_path, output_path)
            return _run_openai_batch(model, inputs, batch_index, **gen_kwargs)
        
        else:
            outputs = {}
            failed_ids = set()

            for rt_idx, retry_id in enumerate(retry_attempts):

                retry_batch = client.batches.retrieve(retry_id)

                total = retry_batch.request_counts.total
                completed = retry_batch.request_counts.completed

                print(
                    f"[OpenAI Batch] retry batch {rt_idx} progress: {completed}/{total}"
                )

                if retry_batch.status!="completed":
                    return None
                
                if retry_batch.output_file_id:
                    content = client.files.content(
                        retry_batch.output_file_id
                    ).text
                    for line in content.splitlines():
                        obj = json.loads(line)
                        outputs[obj["custom_id"]] = obj
                        if obj["custom_id"] in failed_ids:
                            failed_ids.remove(obj["custom_id"])
                
                if retry_batch.error_file_id:
                    error_content = client.files.content(
                        retry_batch.error_file_id
                    ).text

                    for line in error_content.splitlines():
                        failed_ids.add(json.loads(line)["custom_id"])

            if primary_batch.output_file_id:
                content = client.files.content(
                    primary_batch.output_file_id
                ).text

                for line in content.splitlines():
                    obj = json.loads(line)
                    outputs[obj["custom_id"]] = obj
                    if obj["custom_id"] in failed_ids:
                        failed_ids.remove(obj["custom_id"])
            
            if primary_batch.error_file_id:
                error_content = client.files.content(
                    primary_batch.error_file_id
                ).text

                for line in error_content.splitlines():
                    failed_ids.add(json.loads(line)["custom_id"])
            
            for cid in outputs:
                if cid in failed_ids:
                    failed_ids.remove(cid)
            
            if failed_ids:
                
                print(f"[OpenAI Batch] {len(failed_ids)} rows still failed afer {len(retry_attempts)} retries.")

                answer = input(f"Retry again? (y/n): ").strip().lower()

                if answer!="y":
                    tmp_path = output_path + ".tmp"
                    with open(tmp_path, "w") as f:
                        for i in range(len(inputs)):
                            cid = f"row_{i}"
                            f.write(json.dumps(outputs.get(cid, ERR_TOKEN)) + "\n")
                    os.replace(tmp_path, output_path)
                    return _run_openai_batch(model, inputs, batch_index, **gen_kwargs)
                else:

                    retry_input_path = os.path.join(batch_dir, f"retry_input_{len(retry_attempts)}.jsonl")

                    with open(retry_input_path, "w") as f:
                        for cid in failed_ids:
                            idx = int(cid.split("_")[1])
                            record = {
                                "custom_id": cid,
                                "method": "POST",
                                "url": "/v1/responses",
                                "body": {
                                    "model": model, 
                                    "input": inputs[idx],
                                    **gen_kwargs
                                },
                            }
                            f.write(json.dumps(record)+ "\n")
                    
                    with open(retry_input_path, "rb") as f:
                        file_obj = client.files.create(file=f, purpose="batch")
                    
                    retry_batch = client.batches.create(
                        input_file_id=file_obj.id,
                        endpoint="/v1/responses",
                        completion_window="24h"
                    )

                    meta["retry_attempts"].append(retry_batch.id)
                    json.dump(meta, open(meta_path, "w"), indent=2)

                    print(f"[OpenAI Batch] Retry Batch {len(meta['retry_attempts'])} created: {retry_batch.id}")

                    return None
            
            tmp_path = output_path + ".tmp"
            with open(tmp_path, "w") as f:
                for i in range(len(inputs)):
                    cid = f"row_{i}"
                    f.write(json.dumps(outputs[cid])+"\n")
            
            os.replace(tmp_path, output_path)
            return _run_openai_batch(model, inputs, batch_index, **gen_kwargs)
                

    with open(input_path, "w") as f:
        for i, prompt in enumerate(inputs):
            record = {
                "custom_id": f"row_{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model, 
                    "input": prompt,
                    **gen_kwargs
                }
            }
            f.write(json.dumps(record) + "\n")
    
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )

    meta = {
        "batch_id": batch.id,
        "retry_attempts": [],
        "batch_hash": batch_hash,
        "num_inputs": len(inputs),
        "created_at": time.time()
    }

    json.dump(meta, open(meta_path, "w"), indent=2)

    return None

def _run_grok(model: str, inputs: list[str], **gen_kwargs):

    if len(inputs) <= MAX_GROK_BATCH_SIZE:
        return _run_grok_batch(model, inputs, 0, **gen_kwargs)

    outputs = [None] * len(inputs)
    any_pending = False

    num_batches = (len(inputs) + MAX_GROK_BATCH_SIZE - 1) // MAX_GROK_BATCH_SIZE

    for batch_id in range(num_batches):
        start = batch_id * MAX_GROK_BATCH_SIZE
        end = min((batch_id + 1) * MAX_GROK_BATCH_SIZE, len(inputs))
        chunk = inputs[start:end]

        result = _run_grok_batch(
            model,
            chunk,
            batch_index=batch_id,
            **gen_kwargs
        )

        if result is None:
            any_pending = True
        else:
            outputs[start:end] = result

    if any_pending:
        return None

    return outputs


def _run_grok_batch(model:str, inputs:list[str], batch_index:int, sys_prompt=None, **gen_kwargs):

    client = GrokClient()

    batch_hash = hash_batch(model, inputs, {**gen_kwargs, "batch_index":batch_index})
    batch_dir = os.path.join("data", "cache", "grok_batches", batch_hash)
    os.makedirs(batch_dir, exist_ok=True)

    meta_path = os.path.join(batch_dir, "batch_meta.json")
    output_path = os.path.join(batch_dir, "batch_output.json")

    if os.path.exists(output_path):
        return json.load(open(output_path))
    
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))

        batch_id = meta["batch_id"]
        retry_attempts = meta["retry_attempts"]

        primary_batch = client.batch.get(batch_id=batch_id)
        completed = primary_batch.state.num_success + primary_batch.state.num_error
        total = primary_batch.state.num_requests

        print(f"[Grok Batch] progress: {completed}/{total}")

        if primary_batch.state.num_pending > 0:
            return None
        
        if len(retry_attempts) == 0:
            results = client.batch.list_batch_results(batch_id=batch_id, limit=len(inputs))

            outputs = {}
            failed_ids = []

            for r in results.succeeded:
                outputs[r.batch_request_id] = r.response.content
            
            for r in results.failed:
                failed_ids.append(r.batch_request_id)

            if failed_ids:
                print(f"[Grok Batch] {len(failed_ids)} rows failed.")

                answer = input(f"Resubmit {len(failed_ids)} failed rows? (y/n): ").strip().lower()

                if answer!='y':
                    final = []
                    for i in range(len(inputs)):
                        cid = f"row_{i}"
                        final.append(outputs.get(cid, ERR_TOKEN))
                    json.dump(final, open(output_path, "w"),indent=2)
                    return final
                
                retry_batch = client.batch.create(
                    batch_name = f"grok_retry_{batch_hash}_0"
                )

                retry_requests = []
                for cid in failed_ids:
                    idx = int(cid.split('_')[1])
                    prompt = inputs[idx]

                    chat = client.chat.create(
                        model=model,
                        batch_request_id=cid,
                        **{k:v for k,v in gen_kwargs.items() if k in ('temperature', 'max_tokens', 'tool_choice','reasoning_effort', 'presence_penalty', 'frequency_penalty')}
                    )

                    if sys_prompt:
                        chat.append(system(sys_prompt))
                    
                    chat.append(user(prompt))

                    retry_requests.append(chat)
                
                client.batch.add(
                    batch_id=retry_batch.batch_id,
                    batch_requests=retry_requests
                )

                meta["retry_attempts"].append(retry_batch.batch_id)
                json.dump(meta, open(meta_path, "w"), indent=2)

                return None

            ordered = [outputs[f"row_{i}"] for i in range(len(inputs))]

            json.dump(ordered, open(output_path, "w"), indent=2)
            
            return ordered
        else:
            rt_outputs = {}
            rt_failed_ids = set()
            for idx,retry_id in enumerate(retry_attempts):
                retry_batch = client.batch.get(batch_id=retry_id)
                rt_completed = retry_batch.state.num_success + retry_batch.state.num_error

                rt_total = retry_batch.state.num_requests

                print(f"[Grok Batch] retry progress {idx+1}: {rt_completed}/{rt_total}")

                if retry_batch.state.num_pending > 0:
                    return None

                rt_results = client.batch.list_batch_results(batch_id=retry_id, limit=len(inputs))

                for r in rt_results.succeeded:
                    rt_outputs[r.batch_request_id] = r.response.content
                    if r.batch_request_id in rt_failed_ids:
                        rt_failed_ids.remove(r.batch_request_id)

                for r in rt_results.failed:
                    rt_failed_ids.add(r.batch_request_id)

            outputs = {}
            _failed_ids = set()
            results = client.batch.list_batch_results(batch_id=batch_id, limit=len(inputs))

            for r in results.succeeded:
                outputs[r.batch_request_id] = r.response.content
            
            for r in results.failed:
                _failed_ids.add(r.batch_request_id)
            
            outputs.update(rt_outputs)
            _failed_ids.update(rt_failed_ids)
            failed_ids = {cid for cid in _failed_ids if cid not in outputs}
            
            if failed_ids:
                print(f"[Grok Batch] {len(failed_ids)} rows still failed after {len(retry_attempts)} retries.")

                answer = input(f"Retry again? (y/n): ").strip().lower()

                if answer!='y':
                    final = []
                    for i in range(len(inputs)):
                        cid = f"row_{i}"
                        final.append(outputs.get(cid, ERR_TOKEN))

                    json.dump(final, open(output_path, "w"), indent=2)
                    return final
                else:
                    retry_batch = client.batch.create(
                        batch_name=f"grok_retry_{batch_hash}_{len(retry_attempts)}"
                    )

                    retry_requests = []

                    for cid in failed_ids:
                        idx = int(cid.split("_")[1])
                        prompt = inputs[idx]

                        chat = client.chat.create(
                            model=model,
                            batch_request_id=cid,
                            **{k:v for k,v in gen_kwargs.items() if k in ('temperature', 'max_tokens', 'tool_choice','reasoning_effort', 'presence_penalty', 'frequency_penalty')}
                        )

                        if sys_prompt:
                            chat.append(system(sys_prompt))
                        
                        chat.append(user(prompt))

                        retry_requests.append(chat)
                    
                    client.batch.add(
                        batch_id=retry_batch.batch_id,
                        batch_requests=retry_requests
                    )

                    meta["retry_attempts"].append(retry_batch.batch_id)
                    json.dump(meta, open(meta_path,"w"), indent=2)

                    print(f"[Grok Batch] Retry batch number {len(retry_attempts)} submitted: {retry_batch.batch_id}")

                    return None
                
            ordered = [outputs[f"row_{i}"] for i in range(len(inputs))]

            json.dump(ordered, open(output_path, "w"), indent=2)
            return ordered         

    batch = client.batch.create(
        batch_name = f"grok_batch_{batch_hash}"
    )

    batch_requests = []

    for i,prompt in enumerate(inputs):
        chat = client.chat.create(
            model=model,
            batch_request_id=f"row_{i}",
            **{k:v for k,v in gen_kwargs.items() if k in ('temperature', 'max_tokens', 'tool_choice','reasoning_effort', 'presence_penalty', 'frequency_penalty')}
        )

        if sys_prompt:
            chat.append(system(sys_prompt))
        
        chat.append(user(prompt))

        batch_requests.append(chat)
    
    client.batch.add(batch_id=batch.batch_id, batch_requests=batch_requests)

    meta = {
        "batch_id": batch.batch_id,
        "retry_attempts":[],
        "batch_hash": batch_hash,
        "num_inputs": len(inputs),
        "created_at": time.time()
    }

    json.dump(meta, open(meta_path, "w"), indent=2)

    return None

def _run_transformers(model: str, inputs: list[str], engine_kwargs, sys_prompt, **gen_kwargs):

    original_cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']

    if engine_kwargs.get('tensor_parallel_size', None):
        avl_gpus = list(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(avl_gpus[:engine_kwargs['tensor_parallel_size']])

    tokenizer = _get_tokenizer(model)
    prompts = []
    for x in inputs:
        if "gemma" in model:
            prompt = make_prompt(user_prompt=x + (sys_prompt if sys_prompt else ""), user_only=True)
        else:
            prompt = make_prompt(user_prompt=x, sys_prompt=(sys_prompt if sys_prompt else ""), user_only=False)
        _prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(_prompt)

    def collate(batch):
        tokens = tokenizer(batch, add_special_tokens=False, padding="longest", truncation=True, return_tensors="pt")
        return tokens

    dataloader = torch.utils.data.DataLoader(prompts, batch_size=engine_kwargs["batch_size"], collate_fn=collate)
    _model = _get_model_forEval(model)
    device = _model.device
    output_texts = []

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = _model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        decoded_outputs = [
            tokenizer.decode(output[batch["input_ids"].shape[-1]:], skip_special_tokens=True)
            for output in outputs
        ]
        output_texts.extend(decoded_outputs)
    
    if engine_kwargs.get('tensor_parallel_size', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices

    return output_texts

def get_bert_embeddings(inputs:list[str],model_name:str, **gen_kwargs)->torch.Tensor:
    import logging
    logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)


    original_cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']

    # tensor_parallel_size should always be 1
    avl_gpus = list(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(avl_gpus[:1])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained(model_name)

    def collate(batch):
        raw_tokens = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
        for i, input_ids in enumerate(raw_tokens['input_ids']):
            if len(input_ids) > config.max_position_embeddings:
                print(f"Warning: Input index {i} is {len(input_ids)} tokens long. "
                    f"It will be truncated to {tokenizer.model_max_length}.")
        tokens = tokenizer(batch, add_special_tokens=True, padding="longest", truncation=True, return_tensors="pt")

        return tokens
    
    dataloader = torch.utils.data.DataLoader(inputs, batch_size=gen_kwargs['batch_size'], collate_fn=collate)

    model.eval()
    device = model.device

    cls_embeddings = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader) ,unit="Batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        batch_cls_embeddings = outputs.last_hidden_state[:,0,:] # batch_size, hidden_dim

        cls_embeddings.append(batch_cls_embeddings.cpu()) 
    
    cls_embeddings = torch.cat(cls_embeddings, dim=0)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices

    return cls_embeddings

def get_sentence_embeddings(inputs:list[str], model:str, **gen_kwargs)->torch.Tensor:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine_kwargs = {'device': device}
    for k,v in gen_kwargs.items():
        if k in ['truncate_dim']:
            engine_kwargs[k] = v
    
    model = SentenceTransformer(model, **engine_kwargs)

    embeddings = model.encode(
        inputs,
        batch_size=gen_kwargs['batch_size'],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.cpu()


def get_sentence_embeddings_vllm(inputs: list[str], model: str, **gen_kwargs) -> torch.Tensor:


    llm = vllm.LLM(
        model=model,
        task="embed",
        trust_remote_code=True,
        enable_chunked_prefill=False,   # IMPORTANT
        gpu_memory_utilization=0.9
    )

    truncate_dim = gen_kwargs.get("truncate_dim")

    request_chunk = gen_kwargs.get("request_chunk_size", 20000)

    all_embeddings = []

    for start in tqdm(range(0, len(inputs), request_chunk), desc="Submitting to vLLM"):

        chunk = inputs[start:start + request_chunk]

        outputs = llm.embed(chunk)

        chunk_embeddings = [o.outputs.embedding for o in outputs]

        all_embeddings.extend(chunk_embeddings)

    embeddings = torch.tensor(all_embeddings, dtype=torch.float32)

    if truncate_dim is not None:
        embeddings = embeddings[:, :truncate_dim]

    embeddings = F.normalize(embeddings, dim=1)

    return embeddings.cpu()