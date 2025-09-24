#!/usr/bin/env python3
"""
FAISS Index Creation Tool for RECON

Creates FAISS indices from DiffusionDB for concept retrieval.
Compatible with the loading mechanism in src/recon.py.

Usage:
    python create_faiss.py --subset 2k
    python create_faiss.py --subset 10k --max-samples 5000
"""

import argparse
import os
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("ERROR: FAISS is not installed. Install with: pip install faiss-cpu")
    exit(1)


class FAISSIndexCreator:
    """Creates FAISS indices for RECON from DiffusionDB."""

    def __init__(self, cache_dir: str = "/scratch/gilbreth/lu842/models", batch_size: int = 32):
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.tokenizer = None
        self.model = None

    def load_clip_models(self):
        """Load CLIP models - same as used in src/recon.py"""
        print("Loading CLIP models...")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=self.cache_dir
        )
        self.model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=self.cache_dir
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ CLIP models loaded on {self.device}")

    def create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Create CLIP embeddings exactly as done in src/recon.py retrieve_concepts_faiss"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype('float32')

    def load_diffusiondb(self, subset: str = "2k", max_samples: int = None) -> List[str]:
        """Load DiffusionDB and extract prompts"""
        print(f"Loading DiffusionDB subset: {subset}")

        dataset = load_dataset("poloclub/diffusiondb", subset, cache_dir=self.cache_dir)

        prompts = []
        total_items = len(dataset['train'])
        if max_samples:
            total_items = min(total_items, max_samples)

        for i, item in enumerate(tqdm(dataset['train'], desc="Extracting prompts", total=total_items)):
            if max_samples and i >= max_samples:
                break

            if 'prompt' in item and item['prompt'].strip():
                prompts.append(item['prompt'].strip())

        print(f"✓ Extracted {len(prompts)} prompts from DiffusionDB")
        return prompts

    def create_index(
        self,
        prompts: List[str],
        index_path: str = "faiss_index_concept_diffdb.index",
        metadata_path: str = "metadata_dict_concept_diffdb.pkl"
    ):
        """Create FAISS index compatible with src/recon.py loading"""
        print("Creating FAISS index...")

        total_prompts = len(prompts)
        all_embeddings = []

        # Generate embeddings in batches
        for i in tqdm(range(0, total_prompts, self.batch_size), desc="Creating embeddings"):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_embeddings = self.create_embeddings_batch(batch_prompts)
            all_embeddings.append(batch_embeddings)

        # Stack all embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"✓ Created embeddings with shape: {embeddings.shape}")

        # Create FAISS index - using IndexFlatIP for inner product (cosine similarity)
        dimension = embeddings.shape[1]  # 768 for CLIP
        index = faiss.IndexFlatIP(dimension)

        # Add embeddings (already normalized)
        index.add(embeddings)

        # Save FAISS index
        faiss.write_index(index, index_path)
        print(f"✓ Saved FAISS index to: {index_path}")

        # Create metadata dict compatible with src/recon.py
        metadata_dict = {}
        for i, prompt in enumerate(prompts):
            metadata_dict[i] = {'caption': prompt}

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)

        print(f"✓ Saved metadata to: {metadata_path}")
        print(f"✓ Index contains {index.ntotal} vectors")

    def verify_index(
        self,
        index_path: str = "faiss_index_concept_diffdb.index",
        metadata_path: str = "metadata_dict_concept_diffdb.pkl"
    ):
        """Verify the index works with the same loading mechanism as src/recon.py"""
        print("Verifying index compatibility...")

        # Load exactly as done in src/recon.py
        index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            metadata_dict = pickle.load(f)

        # Test search with a sample query
        test_query = "a red car in the city"
        query_embedding = self.create_embeddings_batch([test_query])

        D, I = index.search(query_embedding, 3)

        print(f"Test query: '{test_query}'")
        print("Top 3 similar prompts:")

        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx in metadata_dict:
                caption = metadata_dict[idx]['caption']
                print(f"  {i+1}. (similarity: {distance:.4f}) {caption[:80]}...")
            else:
                print(f"  ERROR: Index {idx} not found in metadata")

        print("✓ Index verification complete")


def main():
    parser = argparse.ArgumentParser(
        description="Create FAISS index from DiffusionDB for RECON",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="2k",
        help="DiffusionDB subset (2k, 10k, large_first_1k, etc.)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use (optional)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/scratch/gilbreth/lu842/models",
        help="Model cache directory"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the created index"
    )

    args = parser.parse_args()

    # Create index creator
    creator = FAISSIndexCreator(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size
    )

    # Load CLIP models
    creator.load_clip_models()

    # Load DiffusionDB
    prompts = creator.load_diffusiondb(
        subset=args.subset,
        max_samples=args.max_samples
    )

    # Create FAISS index
    creator.create_index(prompts)

    # Verify if requested
    if args.verify:
        creator.verify_index()

    print("\n" + "="*50)
    print("✓ FAISS index creation complete!")
    print("\nFiles created:")
    print("  - faiss_index_concept_diffdb.index")
    print("  - metadata_dict_concept_diffdb.pkl")
    print("\nYou can now run RECON with FAISS support:")
    print("  python main.py --demo")


if __name__ == "__main__":
    main()