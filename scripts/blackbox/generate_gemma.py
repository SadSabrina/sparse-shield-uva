#!/usr/bin/env python3
"""
SAE Evaluation Script for Jailbreak Benchmarks
Supports Gemma-2-2b, Gemma-2-9b and Gemma-2-27b with configurable SAE layers
Evaluates on: safe-eval, prompt-injections, and salad benchmarks
Supports both SAE intervention and baseline (no SAE) modes
"""

import json
import os
import argparse
from datetime import datetime

import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


class GemmaSAEModel:
    """SAE intervention wrapper for Gemma"""
    
    def __init__(
        self, 
        model_name: str, 
        sae_release: str = None, 
        sae_id: str = None, 
        device: str = "cuda",
        cache_dir: str = None,
        use_sae: bool = True
    ):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.use_sae = use_sae
        self.sae = None
        self.sae_layer = None
        
        if use_sae:
            if not sae_release or not sae_id:
                raise ValueError("SAE release and ID must be provided when use_sae=True")
            
            print(f"Loading SAE: {sae_release}/{sae_id}")
            self.sae, _, _ = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device
            )
            
            self.sae_layer = int(sae_id.split('/')[0].split('_')[1])
            print(f"✓ SAE layer: {self.sae_layer}")
        else:
            print("✓ Running in baseline mode (no SAE)")
        
        self.device = device
        self.model_name = model_name
        
        print(f"✓ Model loaded on {device}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        
        """Generate with optional SAE intervention"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        hook = None
        
        if self.use_sae:
            # SAE intervention hook
            def sae_hook(module, input, output):
                if isinstance(output, tuple):
                    acts = output[0]
                else:
                    acts = output

                orig_dtype = acts.dtype
                sae_dtype = next(self.sae.parameters()).dtype
               
                # acts to SAE dtype
                acts = acts.to(sae_dtype)
                
                with torch.no_grad():
                    b, s, h = acts.shape
                    acts_flat = acts.view(-1, h)
                    sae_out = self.sae(acts_flat)
                    reconstructed = sae_out.view(b, s, h)

                reconstructed = reconstructed.to(orig_dtype)
                
                return (reconstructed,) + output[1:] if isinstance(output, tuple) else reconstructed
            
            layer = self.model.model.layers[self.sae_layer]
            hook = layer.register_forward_hook(sae_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        finally:
            if hook is not None:
                hook.remove()
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response


def evaluate_prompt_injections(
    model: GemmaSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Evaluate on prompt injections benchmark"""
    print("\n=== Evaluating on prompt-injections ===")
    
    dataset = load_from_disk("sae_evaluation/datasets/prompt_injections")
    dataset = dataset.filter(lambda x: x['label'] == 'jailbreak')
    
    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} jailbreak prompts")
    
    # Check for existing results
    completed_keys = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        completed_keys.add(entry['prompt_id'])
            print(f"Found {len(completed_keys)} already completed entries")
        except Exception as e:
            print(f"Error loading existing file: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        prompt_id = f'injection_{idx}'
        
        if prompt_id in completed_keys:
            continue
        
        prompt = example['text']
        
        try:
            response = model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            result = {
                'prompt_id': prompt_id,
                'eval_model': model.model_name,
                'timestamp': datetime.now().isoformat(),
                'use_sae': model.use_sae,
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'do_sample': do_sample
                },
                'prompt': prompt,
                'response': response,
                'label': example['label']
            }
            
            # Add SAE config only if using SAE
            if model.use_sae:
                result['sae_config'] = {
                    'layer': model.sae_layer,
                    'width': model.sae.cfg.d_sae
                }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def evaluate_salad(
    model: GemmaSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Evaluate on SALAD dataset"""
    print("\n=== Evaluating on SALAD ===")
    
    dataset = load_from_disk("sae_evaluation/datasets/salad")

    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} prompts")
    
    # Check for existing results
    completed_keys = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        completed_keys.add(entry['prompt_id'])
            print(f"Found {len(completed_keys)} already completed entries")
        except Exception as e:
            print(f"Error loading existing file: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        prompt_id = f'salad_{idx}'
        
        if prompt_id in completed_keys:
            continue
        
        prompt = example['augq']
        
        try:
            response = model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            result = {
                'prompt_id': prompt_id,
                'eval_model': model.model_name,
                'timestamp': datetime.now().isoformat(),
                'use_sae': model.use_sae,
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'do_sample': do_sample
                },
                'prompt': prompt,
                'response': response
            }
            
            # Add SAE config only if using SAE
            if model.use_sae:
                result['sae_config'] = {
                    'layer': model.sae_layer,
                    'width': model.sae.cfg.d_sae
                }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def evaluate_safe_eval(
    model: GemmaSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Evaluate on safe-eval dataset"""
    print("\n=== Evaluating on safe-eval ===")
    
    dataset = load_from_disk("sae_evaluation/datasets/safe_eval")
    
    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} prompts")
    
    # Check for existing results
    completed_keys = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        completed_keys.add(entry['prompt_id'])
            print(f"Found {len(completed_keys)} already completed entries")
        except Exception as e:
            print(f"Error loading existing file: {e}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        prompt_id = f'safe_eval_{idx}'
        
        if prompt_id in completed_keys:
            continue
        
        prompt = example['prompt']
        
        try:
            response = model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            result = {
                'prompt_id': prompt_id,
                'eval_model': model.model_name,
                'timestamp': datetime.now().isoformat(),
                'use_sae': model.use_sae,
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'do_sample': do_sample
                },
                'prompt': prompt,
                'response': response,
                'violation': example['violation'],
                'source': example['source_dataset']
            }
            
            # Add SAE config only if using SAE
            if model.use_sae:
                result['sae_config'] = {
                    'layer': model.sae_layer,
                    'width': model.sae.cfg.d_sae
                }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAE evaluation for jailbreak benchmarks"
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name (e.g., google/gemma-2-2b, google/gemma-2-9b or google/gemma-2-27b)'
    )
    
    # SAE mode toggle
    parser.add_argument(
        '--no-sae',
        action='store_true',
        help='Run baseline mode without SAE intervention'
    )
    
    parser.add_argument(
        '--sae-release',
        type=str,
        default=None,
        help='SAE release (e.g., gemma-scope-9b-pt-res-canonical) - required if not using --no-sae'
    )
    parser.add_argument(
        '--sae-id',
        type=str,
        default=None,
        help='SAE ID (e.g., layer_19/width_16k/canonical) - required if not using --no-sae'
    )
    
    # Benchmark selection
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        choices=['safe_eval', 'injections', 'salad', 'all'],
        default=['all'],
        help='Benchmarks to run'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per benchmark (None for all)'
    )
    
    # Paths
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='~/cache',
        help='Model cache directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/sae_responses',
        help='Output directory'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1000,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--do-sample',
        action='store_true',
        help='Enable sampling'
    )
    
    # Device and auth
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace token'
    )
    
    args = parser.parse_args()
    
    # Validate SAE arguments
    use_sae = not args.no_sae
    if use_sae and (not args.sae_release or not args.sae_id):
        parser.error("--sae-release and --sae-id are required when not using --no-sae")
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login(args.hf_token)
    
    print(f"\n{'='*60}")
    print(f"SAE Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'SAE Intervention' if use_sae else 'Baseline (No SAE)'}")
    if use_sae:
        print(f"SAE Release: {args.sae_release}")
        print(f"SAE ID: {args.sae_id}")
    print(f"Device: {args.device}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    # Load model
    model = GemmaSAEModel(
        model_name=args.model_name,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        cache_dir=os.path.expanduser(args.cache_dir),
        use_sae=use_sae
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp and model name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_clean = args.model_name.replace('/', '_')
    
    # Add mode suffix to filename
    mode_suffix = "SAE" if use_sae else "baseline"
    
    # Determine which benchmarks to run
    benchmarks = args.benchmarks
    if 'all' in benchmarks:
        benchmarks = ['safe_eval', 'injections', 'salad']
    
    # Run benchmarks
    if 'safe_eval' in benchmarks:
        output_file = f"{args.output_dir}/{model_clean}_{mode_suffix}_safe_eval_{timestamp}.jsonl"
        evaluate_safe_eval(
            model=model,
            output_path=output_file,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
    
    if 'injections' in benchmarks:
        output_file = f"{args.output_dir}/{model_clean}_{mode_suffix}_injections_{timestamp}.jsonl"
        evaluate_prompt_injections(
            model=model,
            output_path=output_file,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
    
    if 'salad' in benchmarks:
        output_file = f"{args.output_dir}/{model_clean}_{mode_suffix}_salad_{timestamp}.jsonl"
        evaluate_salad(
            model=model,
            output_path=output_file,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
    
    print("\n=== Complete! ===")


if __name__ == "__main__":
    main()