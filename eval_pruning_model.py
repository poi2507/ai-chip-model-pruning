
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import random
import time
import gc
import argparse
import logging

logging.getLogger("transformers").setLevel(logging.WARNING)

# ── 1. PPL Evaluation for Wikitext-2 ──────────────────────────────────
@torch.no_grad()
def eval_ppl_wikitext(model: nn.Module, tokenizer: AutoTokenizer, device: str = "cuda", seq_len: int = 2048, stride: int = 512) -> float:
    """
    Wikitext2 테스트셋을 사용하여 모델의 Perplexity(PPL)를 측정합니다.
    """
    print("\n" + "="*80)
    print("Running Perplexity (PPL) evaluation on wikitext-2...")
    print("="*80)
    model.eval() # 평가 모드 설정
    
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    loss_fct = nn.CrossEntropyLoss(reduction="mean")
    nlls = []
    
    for i in tqdm(range(0, input_ids.size(1), stride), desc="Evaluating PPL"):
        begin_loc = max(i + stride - seq_len, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i
        
        input_window = input_ids[:, begin_loc:end_loc]
        target_window = input_window.clone()
        target_window[:, :-trg_len] = -100

        outputs = model(input_window, labels=target_window)
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

    mean_nll = torch.stack(nlls).mean()
    ppl = torch.exp(mean_nll)
    
    print(f"\n=> Wikitext-2 Perplexity: {ppl.item():.4f}")
    return ppl.item()

# ── 2. MMLU Accuracy & Latency Evaluation ───────────────────────────────
def format_mmlu_messages(example):
    """
    MMLU 데이터셋을 Chat-style 프롬프트로 변환합니다.
    """
    system_msg = {
        "role": "system",
        "content": "You are a factual, high‐performance benchmarking assistant for the MMLU dataset. When given a multiple‐choice question, respond with exactly one of the provided choice values—no letters, no explanations, no extra text."
    }
    question = example["question"].strip()
    choices = example["choices"]
    user_msg = {"role": "user", "content": f"Question : {question}\n\n choices : {choices}\nAnswer:"}
    return [system_msg, user_msg]

def calculate_mmlu_performance(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    mmlu_ds,
    num_samples: int = 1000,
    warmup_iters: int = 5,
    model_name: str = "model"
) -> None:
    """
    Zero-shot MMLU 정확도 및 Latency 벤치마킹을 수행합니다.
    """
    print("\n" + "="*80)
    print(f"Running MMLU & Latency evaluation for {model_name}...")
    print("="*80)
    
    chat_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
        return_full_text=False,
    )

    # 워밍업
    print("Warming up...")
    for _ in tqdm(range(warmup_iters), desc="Warm-up"):
        _ = chat_pipe(
            format_mmlu_messages(mmlu_ds[random.randrange(len(mmlu_ds))]),
            max_new_tokens=10,
            do_sample=False
        )

    # 재현성을 위한 샘플 인덱싱
    rng = random.Random(42)
    sampled_indices = rng.sample(range(len(mmlu_ds)), num_samples)

    correct_cnt = 0
    sum_latency = 0.0
    
    for idx in tqdm(sampled_indices, desc=f"Evaluating MMLU"):
        example = mmlu_ds[idx]
        messages = format_mmlu_messages(example)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        out = chat_pipe(
            messages,
            max_new_tokens=10, # 정답은 짧으므로 길게 생성할 필요 없음
            do_sample=False,
            pad_token_id=chat_pipe.tokenizer.eos_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        sum_latency += (end_time - start_time)
        pred = out[0]["generated_text"].strip().replace("'", "").replace('"', '')
        correct = example["choices"][example["answer"]].strip()
        
        if pred.lower() == correct.lower():
            correct_cnt += 1

    avg_acc = (correct_cnt / num_samples) * 100
    avg_lat = sum_latency / num_samples

    print("\n" + "="*60)
    print(f"MMLU Evaluation Results for: {model_name:^20}")
    print("="*60)
    print(f"{'Average Accuracy:':<30}{avg_acc:6.2f}%")
    print(f"{'Average Latency:':<30}{avg_lat:8.4f} sec")
    print(f"{'Samples Evaluated:':<30}{num_samples}")
    print("="*60)

    del chat_pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_acc, avg_lat # 계산된 정확도와 지연시간을 반환
# --- 최종 실행 블록 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM PPL 및 MMLU 평가 스크립트")
    parser.add_argument(
        '--model_id', 
        type=str, 
        required=True, 
        help='평가할 Hugging Face 모델 ID 또는 로컬 경로'
    )
    parser.add_argument(
        '--mmlu_samples', 
        type=int, 
        default=50, 
        help='MMLU 평가에 사용할 샘플 수'
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model and tokenizer: {args.model_id} on device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 1. PPL 평가 실행 및 결과 저장 ---
    wikitext_ppl = eval_ppl_wikitext(model, tokenizer, device=device)
    
    model.to('cpu')
    torch.cuda.empty_cache()

    # --- 2. MMLU 평가 실행 및 결과 저장 ---
    print("\nLoading MMLU dataset...")
    mmlu_dataset = load_dataset("cais/mmlu", "all", split="validation")
    mmlu_accuracy, mmlu_latency = calculate_mmlu_performance(
        model, 
        tokenizer, 
        mmlu_ds=mmlu_dataset, 
        num_samples=args.mmlu_samples, 
        model_name=args.model_id
    )
    
    # --- 3. 최종 결과 요약 출력 ---
    print("\n" + "#"*80)
    print(f" 종합 평가 결과 (Final Evaluation Summary) : {args.model_id} ".center(80, "#"))
    print("#"*80)
    print(f"\n[Wikitext-2 Performance]")
    print(f"  - Perplexity (PPL): {wikitext_ppl:.4f}")
    print(f"\n[MMLU Performance]")
    print(f"  - Zero-shot Accuracy: {mmlu_accuracy:.2f}%")
    print(f"  - Average Latency:    {mmlu_latency:.4f} sec/sample")
    print("\n" + "#"*80)
