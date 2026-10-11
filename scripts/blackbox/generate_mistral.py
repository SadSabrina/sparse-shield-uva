#!/usr/bin/env python3
"""
SAE Evaluation Script for Jailbreak Benchmarks - Mistral Version
Supports mistralai/Mistral-7B-v0.1 with JoshEngels SAE
Evaluates on: safe-eval, prompt-injections, and salad benchmarks
Supports both SAE intervention and baseline (no SAE) modes
"""

import json
import os
import argparse
from datetime import datetime

import torch
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import safetensors.torch


class MistralSAEModel:
    """SAE intervention wrapper for Mistral models"""
    
    def __init__(
        self, 
        model_name: str,
        sae_repo: str = None,
        sae_layer: int = None,
        device: str = "cuda",
        dtype = torch.bfloat16,
        cache_dir: str = None,
        use_sae: bool = True
    ):
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype,
            device_map=device,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_name = model_name
        self.use_sae = use_sae
        
        # Initialize SAE components
        self.W_enc = None
        self.W_dec = None
        self.b_enc = None
        self.b_dec = None
        self.sae_layer_idx = None

        # Load SAE if enabled
        if use_sae:
            if not sae_repo or sae_layer is None:
                raise ValueError("SAE repo and layer must be provided when use_sae=True")
            
            self.sae_layer_idx = sae_layer
            
            # Load SAE weights
            print(f"Loading SAE from layer {sae_layer}...")
            sae_filename = f'mistral_7b_layer_{sae_layer}/sae_weights.safetensors'
            
            file_path = hf_hub_download(
                repo_id=sae_repo,
                filename=sae_filename,
                repo_type="model"
            )
            
            # Load SAE weights
            sae_state_dict = safetensors.torch.load_file(file_path)
            
            # Check for key names
            if 'W_enc' in sae_state_dict:
                self.W_enc = sae_state_dict['W_enc'].to(dtype)
                self.W_dec = sae_state_dict['W_dec'].to(dtype)
                self.b_enc = sae_state_dict['b_enc'].to(dtype)
                self.b_dec = sae_state_dict['b_dec'].to(dtype)
            else:
                # Alternative key names
                self.W_enc = sae_state_dict['encoder.weight'].to(dtype)
                self.W_dec = sae_state_dict['decoder.weight'].to(dtype)
                self.b_enc = sae_state_dict['encoder.bias'].to(dtype)
                self.b_dec = sae_state_dict['decoder.bias'].to(dtype)
            
            print(f"✓ SAE layer: {self.sae_layer_idx}")
            print(f"✓ SAE weights loaded")
        else:
            print("✓ Running in baseline mode (no SAE)")
        
        print(f"✓ Model loaded on {device}")
    
    def apply_sae(self, x):
        """Apply SAE with normalization"""
        norm_constant = 64.0
        original_norm = torch.norm(x, dim=-1, keepdim=True)
        x_normalized = x * (norm_constant / (original_norm + 1e-8))
        f = torch.nn.functional.relu(torch.matmul(x_normalized, self.W_enc) + self.b_enc)
        x_reconstructed_norm = torch.matmul(f, self.W_dec) + self.b_dec
        x_reconstructed = x_reconstructed_norm * (original_norm / norm_constant)
        
        return x_reconstructed
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.0,
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
            layer_module = self.model.model.layers[self.sae_layer_idx]
            
            def sae_hook(module, input, output):
                if isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output

                orig_dtype = activations.dtype
                activations = activations.to(self.dtype)
                
                with torch.no_grad():
                    batch_size, seq_len, hidden_size = activations.shape
                    acts_reshaped = activations.view(-1, hidden_size)
                    
                    # Move SAE weights if needed
                    if self.W_enc.device != acts_reshaped.device:
                        self.W_enc = self.W_enc.to(acts_reshaped.device)
                        self.W_dec = self.W_dec.to(acts_reshaped.device)
                        self.b_enc = self.b_enc.to(acts_reshaped.device)
                        self.b_dec = self.b_dec.to(acts_reshaped.device)
                    
                    # Apply SAE and edit dtype
                    reconstructed = self.apply_sae(acts_reshaped)
                    reconstructed = reconstructed.view(batch_size, seq_len, hidden_size)
                    reconstructed = reconstructed.to(orig_dtype)
                
                if isinstance(output, tuple):
                    return (reconstructed,) + output[1:]
                else:
                    return reconstructed
            
            hook = layer_module.register_forward_hook(sae_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        finally:
            if hook is not None:
                hook.remove()
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove the prompt part)
        response = generated_text[len(prompt):].strip()
        
        return response


def evaluate_safe_eval(
    model: MistralSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Evaluate on safe-eval dataset"""
    print("\n=== Evaluating on safe-eval ===")
    
    dataset = load_dataset("dynamoai/safe_eval", split="train")
    
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
        
        prompt = example['prompt'] + " "
        
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
                    'layer': model.sae_layer_idx,
                    'sae_name': 'JoshEngels Mistral SAE'
                }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def evaluate_prompt_injections(
    model: MistralSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Evaluate on prompt injections benchmark"""
    print("\n=== Evaluating on prompt-injections ===")
    
    dataset = load_dataset("qualifire/prompt-injections-benchmark", split="test")
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
        
        prompt = example['text'] + " "
        
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
                    'layer': model.sae_layer_idx,
                    'sae_name': 'JoshEngels Mistral SAE'
                }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def evaluate_salad(
    model: MistralSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    do_sample: bool = False
):
    """Evaluate on SALAD dataset"""
    print("\n=== Evaluating on SALAD ===")
    
    dataset = load_dataset("OpenSafetyLab/Salad-Data", name="attack_enhanced_set", split="train")
    
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
        
        prompt = example['augq'] + " "
        
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
                'level': example['level']
            }
            
            # Add SAE config only if using SAE
            if model.use_sae:
                result['sae_config'] = {
                    'layer': model.sae_layer_idx,
                    'sae_name': 'JoshEngels Mistral SAE'
                }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAE evaluation for Mistral on jailbreak benchmarks"
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Model name'
    )
    
    # SAE mode toggle
    parser.add_argument(
        '--no-sae',
        action='store_true',
        help='Run baseline mode without SAE intervention'
    )
    
    parser.add_argument(
        '--sae-repo',
        type=str,
        default='JoshEngels/Mistral-7B-Residual-Stream-SAEs',
        help='SAE repository on HuggingFace'
    )
    parser.add_argument(
        '--sae-layer',
        type=int,
        default=16,
        help='Layer to apply SAE (options: 8, 16, 24)'
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
        '--output-dir',
        type=str,
        default='data/sae_responses',
        help='Output directory'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for models'
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
        default='cuda',
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
    if use_sae and args.sae_layer is None:
        parser.error("--sae-layer is required when not using --no-sae")
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login(args.hf_token)
    
    print(f"\n{'='*60}")
    print(f"SAE Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'SAE Intervention' if use_sae else 'Baseline (No SAE)'}")
    if use_sae:
        print(f"SAE Repo: {args.sae_repo}")
        print(f"SAE Layer: {args.sae_layer}")
    print(f"Device: {args.device}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    # Load model with optional SAE
    model = MistralSAEModel(
        model_name=args.model_name,
        sae_repo=args.sae_repo if use_sae else None,
        sae_layer=args.sae_layer if use_sae else None,
        device=args.device,
        cache_dir=args.cache_dir,
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


# srun python3 sae_eval_mistral.py --model-name mistralai/Mistral-7B-v0.1 --sae-repo JoshEngels/Mistral-7B-Residual-Stream-SAEs --sae-layer 16 --benchmarks all --max-samples 500 --max-tokens 256
# srun python3 sae_eval_mistral.py --model-name mistralai/Mistral-7B-v0.1 --no-sae --benchmarks all --max-samples 500 --max-tokens 256