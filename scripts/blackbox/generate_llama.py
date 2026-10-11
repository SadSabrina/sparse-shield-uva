#!/usr/bin/env python3
"""
SAE Evaluation Script for Jailbreak Benchmarks - LLAMA Version
Supports Meta-Llama-3.1-8B-Instruct and Meta-Llama-3.3-70B-Instruct with Goodfire SAE
Also supports baseline run without SAE (--no-sae).
Evaluates on: safe-eval, prompt-injections, and salad benchmarks
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


class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device,
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)
    return sae


class LlamaSAEModel:
    """SAE intervention wrapper for LLAMA models (optionally no-SAE baseline)"""
    
    def __init__(
        self, 
        model_name: str,
        sae,                 # may be None (baseline run)
        sae_layer: str,
        device: str = "cuda",
        dtype = torch.bfloat16
    ):
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.sae = sae
        self.sae_layer = sae_layer
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_name = model_name
        
        # SAE can be off
        if self.sae is not None and self.sae_layer:
            self.sae_layer_idx = int(sae_layer.split('.')[-1])
        else:
            self.sae_layer_idx = None
        
        print(f"✓ Model loaded on {device}")
        if self.sae is not None:
            print(f"✓ SAE layer: {self.sae_layer}")
        else:
            print(f"✓ NO-SAE baseline (no intervention)")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """Generate with optional SAE intervention"""
        
        # Format with chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        hook = None

        # SAE intervention hook — if sae not None
        if self.sae is not None:
            def sae_hook(module, input, output):

                if isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output

                orig_dtype = activations.dtype
                sae_dtype = next(self.sae.parameters()).dtype

                activations = activations.to(sae_dtype)

                batch_size, seq_len, hidden_size = activations.shape
                acts_reshaped = activations.view(-1, hidden_size)

                with torch.no_grad():
                    features = self.sae.encode(acts_reshaped)
                    reconstructed = self.sae.decode(features).view(
                        batch_size, seq_len, hidden_size
                    )

                reconstructed = reconstructed.to(orig_dtype)
                
                if isinstance(output, tuple):
                    return (reconstructed,) + output[1:]
                else:
                    return reconstructed
            
            # Find the module to hook
            module = self.model
            for name in self.sae_layer.split('.'):
                module = getattr(module, name)
            
            hook = module.register_forward_hook(sae_hook)
        
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
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove the formatted prompt part)
        response = generated_text.replace(formatted_prompt, "").strip()
        
        return response


def evaluate_safe_eval(
    model: LlamaSAEModel,
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
                'sae_config': {
                    'layer': model.sae_layer_idx,
                    'sae_name': 'Goodfire SAE' if model.sae is not None else 'no-sae'
                },
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
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def evaluate_prompt_injections(
    model: LlamaSAEModel,
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
                for line in f.readlines():
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
                'sae_config': {
                    'layer': model.sae_layer_idx,
                    'sae_name': 'Goodfire SAE' if model.sae is not None else 'no-sae'
                },
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'do_sample': do_sample
                },
                'prompt': prompt,
                'response': response,
                'label': example['label']
            }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def evaluate_salad(
    model: LlamaSAEModel,
    output_path: str,
    max_samples: int = None,
    max_tokens: int = 512,
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
                'sae_config': {
                    'layer': model.sae_layer_idx,
                    'sae_name': 'Goodfire SAE' if model.sae is not None else 'no-sae'
                },
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'do_sample': do_sample
                },
                'prompt': prompt,
                'response': response,
            }
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
                
        except Exception as e:
            print(f"\nError processing {prompt_id}: {e}")
            continue
    
    print(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAE evaluation for LLAMA on jailbreak benchmarks"
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
        help='Model name (meta-llama/Meta-Llama-3.1-8B-Instruct or meta-llama/Llama-3.3-70B-Instruct)'
    )
    parser.add_argument(
        '--sae-name',
        type=str,
        default='Llama-3.1-8B-Instruct-SAE-l19',
        help='SAE name from Goodfire (e.g., Llama-3.1-8B-Instruct-SAE-l19 or Llama-3.3-70B-Instruct-SAE-l50)'
    )
    parser.add_argument(
        '--sae-layer',
        type=str,
        default='model.layers.19',
        help='Layer to apply SAE (e.g., model.layers.19 for 8B, model.layers.50 for 70B)'
    )
    parser.add_argument(
        '--expansion-factor',
        type=int,
        default=16,
        help='SAE expansion factor (16 for l19, 8 for others)'
    )
    parser.add_argument(
        '--no-sae',
        action='store_true',
        help='Run baseline model without SAE intervention'
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
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login(args.hf_token)
    
    print(f"\n{'='*60}")
    print(f"SAE Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    if args.no_sae:
        print(f"SAE: NO-SAE (baseline)")
        print(f"SAE Layer: N/A")
        print(f"Expansion Factor: N/A")
    else:
        print(f"SAE: {args.sae_name}")
        print(f"SAE Layer: {args.sae_layer}")
        print(f"Expansion Factor: {args.expansion_factor}")
    print(f"Device: {args.device}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    # Load SAE (optional)
    if not args.no_sae:
        print("Loading SAE...")
        file_path = hf_hub_download(
            repo_id=f"Goodfire/{args.sae_name}",
            filename=f"{args.sae_name}.pth",
            repo_type="model"
        )
        
        # Get d_model from a temporary model
        temp_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        d_model = temp_model.config.hidden_size
        del temp_model
        
        sae = load_sae(
            file_path,
            d_model=d_model,
            expansion_factor=args.expansion_factor,
            device=torch.device(args.device),
        )
        print("✓ SAE loaded")
    else:
        sae = None
        print("Skipping SAE loading: running baseline model without intervention.")
    
    # Load model with (or without) SAE
    model = LlamaSAEModel(
        model_name=args.model_name,
        sae=sae,
        sae_layer=args.sae_layer,
        device=args.device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp and model name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_clean = args.model_name.replace('/', '_')
    
    # Determine which benchmarks to run
    benchmarks = args.benchmarks
    if 'all' in benchmarks:
        benchmarks = ['safe_eval', 'injections', 'salad']
    
    # Run benchmarks
    if 'safe_eval' in benchmarks:
        output_file = f"{args.output_dir}/{model_clean}_SAE_safe_eval_{timestamp}.jsonl"
        evaluate_safe_eval(
            model=model,
            output_path=output_file,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
    
    if 'injections' in benchmarks:
        output_file = f"{args.output_dir}/{model_clean}_SAE_injections_{timestamp}.jsonl"
        evaluate_prompt_injections(
            model=model,
            output_path=output_file,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
    
    if 'salad' in benchmarks:
        output_file = f"{args.output_dir}/{model_clean}_SAE_salad_{timestamp}.jsonl"
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
    