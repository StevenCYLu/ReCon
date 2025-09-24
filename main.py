#!/usr/bin/env python3
"""
RECON Demo - Main Entry Point

This script provides two main modes:
1. --demo: Run RECON generation demo
2. --create-index: Show instructions for creating FAISS indices

Usage:
    python main.py --demo                    # Run demo with default prompt
    python main.py --demo --prompt "..."     # Run demo with custom prompt
    python main.py --create-index            # Show FAISS index creation guide
"""

import argparse
import os
import sys
from pathlib import Path

from src.recon import ReconGenerator


def run_demo(
    prompt: str = None,
    cache_dir: str = "./models",
    output_dir: str = "./demo_output",
    use_faiss: bool = True,
    faiss_index_path: str = "faiss_index_concept_diffdb.index",
    metadata_path: str = "metadata_dict_concept_diffdb.pkl"
):
    """
    Run the RECON demo.
    
    Args:
        prompt: Custom prompt to use
        cache_dir: Directory to cache models
        output_dir: Output directory for results
        use_faiss: Whether to attempt FAISS loading
        faiss_index_path: Path to FAISS index file
        metadata_path: Path to metadata file
    """
    print("=== RECON Demo ===")
    print()
    
    # Initialize generator
    generator = ReconGenerator(cache_dir=cache_dir)
    
    # Load models
    # try:
    generator.load_models(load_clip=True, load_spacy=True)
    
    
    # Try to load FAISS index if available
    faiss_loaded = False
    if use_faiss and os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        generator.load_faiss_index(faiss_index_path, metadata_path)
        faiss_loaded = True
        print("✓ FAISS index loaded successfully")
    else:
        if use_faiss:
            print("FAISS index files not found. Using NLP-based concept extraction.")
            print(f"Expected files: {faiss_index_path}, {metadata_path}")
            print("Run 'python main.py --create-index' for setup instructions.")
    
    # Run demo
    results = generator.demo_generation(
        prompt=prompt,
        use_faiss=faiss_loaded,
        output_dir=output_dir
    )

    print(f"\n✓ Demo completed successfully!")
    print(f"Generated {len(results)} images with different k values")
    print(f"Results saved to: {output_dir}")

    # List generated files
    if os.path.exists(output_dir):
        files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        if files:
            print(f"\nGenerated files:")
            for file in files:
                print(f"  - {file}")


def show_create_index_guide():
    """Show instructions for creating FAISS indices."""
    print("\n=== FAISS Index Creation ===")
    print("To create FAISS indices for concept retrieval, run:")
    print("  python create_faiss.py")
    print("\nThis will create the required index files for enhanced concept retrieval.")



def main():
    parser = argparse.ArgumentParser(
        description="RECON: REconstructive CONcept Diffusion Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --demo --prompt "a red car in the city"
  python main.py --create-index
  python main.py --demo --cache-dir ./models --output-dir ./results
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run RECON generation demo"
    )
    
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Show instructions for creating FAISS indices"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt for demo (default: sample prompt)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Directory to cache downloaded models (default: ./models)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./demo_output",
        help="Output directory for generated images (default: ./demo_output)"
    )
    
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Skip FAISS loading and use only NLP concept extraction"
    )
    
    parser.add_argument(
        "--faiss-index",
        type=str,
        default="faiss_index_concept_diffdb.index",
        help="Path to FAISS index file"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata_dict_concept_diffdb.pkl",
        help="Path to metadata pickle file"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.demo, args.create_index, args.check_deps]):
        parser.print_help()
        return
    
    if args.create_index:
        show_create_index_guide()
        return
    
    if args.demo:
        run_demo(
            prompt=args.prompt,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            use_faiss=not args.no_faiss,
            faiss_index_path=args.faiss_index,
            metadata_path=args.metadata
        )
        return


if __name__ == "__main__":
    main()