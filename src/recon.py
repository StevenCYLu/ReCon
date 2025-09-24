"""
RECON Generator - Object-Oriented Implementation

This module provides a clean, class-based interface for the RECON method,
supporting both FAISS-based concept retrieval and fallback NLP extraction.
"""

import os
import sys
import pickle
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, DDPMScheduler

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from .pipeline import ReconStableDiffusionXLPipeline


class ReconGenerator:
    """
    RECON method implementation with concept-based image reconstruction.
    
    This class handles the complete RECON pipeline including:
    - Model loading and initialization
    - Concept extraction (FAISS-based or NLP fallback)
    - Multi-step generation with latent caching
    - Progressive reconstruction with different k values
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir: str = "./models",
        device: Optional[str] = None,
        torch_dtype=torch.float16,
        use_safetensors: bool = True,
    ):
        """
        Initialize the RECON generator.
        
        Args:
            model_id: HuggingFace model identifier for SDXL
            cache_dir: Directory to cache downloaded models
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            torch_dtype: Torch data type for models
            use_safetensors: Whether to use safetensors format
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.use_safetensors = use_safetensors
        
        # Initialize components
        self.pipeline = None
        self.clip_tokenizer = None
        self.clip_model = None
        self.nlp = None
        self.faiss_index = None
        self.metadata_dict = None
        
        # Configuration
        self.default_negative_prompt = "anime, vector art, art, painting, illustration, ugly, abstract, black and white"
        self.default_seed = 1337
        
    def load_models(self, load_clip: bool = True, load_spacy: bool = True):
        """
        Load all required models.
        
        Args:
            load_clip: Whether to load CLIP models for embedding
            load_spacy: Whether to load spaCy for NLP processing
        """
        print(f"Loading models on device: {self.device}")
        
        # Load SDXL pipeline
        print("Loading Stable Diffusion XL pipeline...")
        self.pipeline = ReconStableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors
        )
        scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = scheduler
        
        self.pipeline.to(self.device)
        
        # Load CLIP models if requested
        if load_clip:
            print("Loading CLIP models...")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", 
                cache_dir=self.cache_dir,
                # local_files_only=True
            )
            self.clip_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14", 
                cache_dir=self.cache_dir,
                # local_files_only=True
            )
            self.clip_model.to(self.device)
        
        # Load spaCy if available and requested
        if load_spacy and SPACY_AVAILABLE:
            try:
                print("Loading spaCy model...")
                self.nlp = spacy.load("en_core_web_trf")
            except OSError:
                print("Warning: spaCy model 'en_core_web_trf' not found. Install with: python -m spacy download en_core_web_trf")
                self.nlp = None
        
        print("✓ Models loaded successfully")

    def load_faiss_index(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata for concept retrieval.
        
        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata pickle file
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")
            
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
            
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(index_path)
        
        print("Loading metadata...")
        with open(metadata_path, 'rb') as f:
            self.metadata_dict = pickle.load(f)
            
        print(f"✓ FAISS index loaded with {self.faiss_index.ntotal} vectors")

    def extract_concepts_nlp(self, prompt: str) -> List[str]:
        """
        Extract concepts using NLP (noun chunks).
        
        Args:
            prompt: Input prompt text
            
        Returns:
            List of extracted concept strings
        """
        if self.nlp is None:
            print("Warning: spaCy not available, using full prompt as concept")
            return [prompt]
            
        doc = self.nlp(prompt)
        concepts = [chunk.text for chunk in doc.noun_chunks]
        
        if not concepts:
            print("No noun chunks found, using full prompt")
            concepts = [prompt]
            
        print(f"Extracted concepts via NLP: {concepts}")
        return concepts

    def retrieve_concepts_faiss(self, prompt: str, top_k: int = 1) -> List[str]:
        """
        Retrieve concepts using FAISS similarity search.
        
        Args:
            prompt: Input prompt text
            top_k: Number of similar concepts to retrieve per extracted concept
            
        Returns:
            List of retrieved concept prompts
        """
        if self.faiss_index is None or self.metadata_dict is None:
            raise ValueError("FAISS index not loaded. Call load_faiss_index() first.")
            
        if self.clip_model is None or self.clip_tokenizer is None:
            raise ValueError("CLIP models not loaded. Call load_models() with load_clip=True first.")
        
        # Extract concepts using NLP
        concepts = self.extract_concepts_nlp(prompt)
        
        # Get embeddings for concepts
        inputs = self.clip_tokenizer(
            concepts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Search FAISS index
        query = text_features.cpu().numpy().astype('float32')
        D, I = self.faiss_index.search(query, top_k)
        
        # Retrieve matched prompts
        retrieved_prompts = []
        for i in range(len(I)):
            for j in range(top_k):
                idx = I[i][j]
                if idx in self.metadata_dict:
                    retrieved_prompts.append(self.metadata_dict[idx]['caption'])
                else:
                    print(f"Warning: Index {idx} not found in metadata")
        
        # Remove duplicates while preserving order
        unique_prompts = []
        seen = set()
        for prompt in retrieved_prompts:
            if prompt not in seen:
                unique_prompts.append(prompt)
                seen.add(prompt)
        
        # Remove last prompt if we have multiple (often less relevant)
        if len(unique_prompts) > 1:
            unique_prompts = unique_prompts[:-1]
            
        print(f"Retrieved concept prompts via FAISS: {unique_prompts}")
        return unique_prompts

    def get_concept_prompts(self, prompt: str, use_faiss: bool = True) -> List[str]:
        """
        Get concept prompts using either FAISS or NLP fallback.
        
        Args:
            prompt: Input prompt text
            use_faiss: Whether to use FAISS retrieval (if available)
            
        Returns:
            List of concept prompts
        """
        if use_faiss and self.faiss_index is not None:
            return self.retrieve_concepts_faiss(prompt)

        # Fallback to NLP extraction
        return self.extract_concepts_nlp(prompt)

    def generate_concept_image(
        self, 
        concept_prompt: str, 
        width: int = 768, 
        height: int = 768,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> tuple[Image.Image, List[torch.Tensor]]:
        """
        Generate an image for a concept and collect latents during generation.
        
        Args:
            concept_prompt: The concept prompt to generate
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            seed: Random seed for generation
            
        Returns:
            Tuple of (generated image, list of latent tensors)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_models() first.")
            
        seed = seed or self.default_seed
        latents = []

        def callback(step, t, latent, pred_original):
            if step % 5 == 0:
                latents.append(latent.clone())

        generator = torch.Generator(self.device).manual_seed(seed)
        
        result = self.pipeline(
            concept_prompt,
            callback=callback,
            negative_prompt=self.default_negative_prompt,
            width=width,
            height=height,
            generator=generator,
            num_inference_steps=num_inference_steps,
            cross_attention_kwargs={},
        )
        
        return result.images[0], latents

    def generate_with_recon(
        self,
        prompt: str,
        concept_prompts: List[str],
        k_values: List[int] = None,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[int, Image.Image]:
        """
        Generate images using the RECON method with different k values.
        
        Args:
            prompt: Target prompt for reconstruction
            concept_prompts: List of concept prompts for caching
            k_values: List of k values (caching steps) to try
            width: Image width
            height: Image height  
            num_inference_steps: Number of denoising steps
            seed: Random seed for generation
            output_dir: Directory to save intermediate results
            
        Returns:
            Dictionary mapping k values to generated images
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_models() first.")
            
        k_values = k_values or [5, 10, 15]
        seed = seed or self.default_seed
        results = {}
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print("Starting RECON generation process...")
        
        for k in k_values:
            print(f"\nProcessing k={k} (caching at step {k})...")
            
            cache_latents = []

            # Generate concept images and cache latents
            for ind, concept_prompt in enumerate(concept_prompts):
                print(f"  Generating concept {ind+1}/{len(concept_prompts)}: '{concept_prompt}'")

                concept_img, latents = self.generate_concept_image(
                    concept_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    seed=seed
                )

                # Save concept image if output directory specified
                if output_dir:
                    safe_name = "".join(c for c in concept_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    concept_img.save(f"{output_dir}/concept_{ind}_{safe_name[:30]}_k{k}.png")

                # Cache the latent at step k
                if len(latents) > k // 5:
                    cache_latents.append(latents[k // 5])
                else:
                    print(f"    Warning: Not enough latents cached for k={k}")
                    if latents:
                        cache_latents.append(latents[-1])

            if not cache_latents:
                print(f"  No latents cached for k={k}, skipping...")
                continue

            # Calculate weighted average based on CLIP similarity to original prompt
                # Get embedding for original prompt
            prompt_inputs = self.clip_tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                prompt_outputs = self.clip_model(**prompt_inputs)
                prompt_features = prompt_outputs.last_hidden_state.mean(dim=1)
                prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

            # Get embeddings for concept prompts
            concept_inputs = self.clip_tokenizer(
                concept_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                concept_outputs = self.clip_model(**concept_inputs)
                concept_features = concept_outputs.last_hidden_state.mean(dim=1)
                concept_features = concept_features / concept_features.norm(dim=-1, keepdim=True)

            # Calculate similarities
            similarities = torch.cosine_similarity(prompt_features, concept_features, dim=1)
            similarities = similarities[:len(cache_latents)]  # Ensure same length as cached latents

            # Normalize similarities to get weights
            weights = torch.softmax(similarities, dim=0)

            # Calculate weighted average
            stacked_latents = torch.stack(cache_latents)
            weighted_latents = stacked_latents * weights.view(-1, 1, 1, 1, 1)
            averaged_latents = torch.sum(weighted_latents, dim=0)

            print(f"  Weighted averaged {len(cache_latents)} latent tensors (weights: {weights.cpu().numpy()})")

            averaged_latents = averaged_latents.to(self.device, dtype=self.torch_dtype)
            # Generate final reconstruction
            print(f"  Generating final reconstruction with start_ratio={(k + 1) / num_inference_steps:.2f}")
            generator = torch.Generator(self.device).manual_seed(seed)

            output = self.pipeline(
                prompt,
                latents=averaged_latents,
                width=width,
                height=height,
                negative_prompt=self.default_negative_prompt,
                cross_attention_kwargs={},
                start_ratio=(k + 1) / num_inference_steps,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

            results[k] = output

            if output_dir:
                output.save(f"{output_dir}/recon_k={k}.png")

            print(f"  ✓ Generated reconstruction for k={k}")
        
        print(f"\nRECON generation complete! Generated {len(results)} images.")
        return results

    def demo_generation(
        self,
        prompt: str = None,
        use_faiss: bool = True,
        output_dir: str = "./demo_output"
    ) -> Dict[int, Image.Image]:
        """
        Run a demo generation with a sample prompt.
        
        Args:
            prompt: Prompt to use (default: sample prompt)
            use_faiss: Whether to use FAISS for concept retrieval
            output_dir: Output directory for results
            
        Returns:
            Dictionary of generated images by k value
        """
        # Default demo prompt
        if prompt is None:
            prompt = 'close-up photography of an old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux'
        
        print(f"Running RECON demo with prompt: '{prompt}'")
        print(f"FAISS available: {FAISS_AVAILABLE and self.faiss_index is not None}")
        print(f"Using FAISS: {use_faiss and FAISS_AVAILABLE and self.faiss_index is not None}")
        
        # Get concept prompts
        concept_prompts = self.get_concept_prompts(prompt, use_faiss=use_faiss)
        
        # Generate using RECON
        results = self.generate_with_recon(
            prompt=prompt,
            concept_prompts=concept_prompts,
            output_dir=output_dir
        )
        
        print(f"\nDemo complete! Results saved to: {output_dir}")
        return results

