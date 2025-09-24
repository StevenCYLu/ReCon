"""
Create a visualization showing the RECON method process
This creates a diagram without needing to run the full pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_recon_diagram():
    """Create a visual diagram of the RECON process"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Colors
    input_color = '#E8F4FD'
    concept_color = '#FFF2CC'
    latent_color = '#F8CECC'
    output_color = '#D5E8D4'
    arrow_color = '#666666'
    
    # Step 1: Input Prompt
    input_box = FancyBboxPatch((1, 8), 12, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=input_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 8.5, 'Input Prompt:\n"close-up photography of an old man standing in the rain at night"', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrow down
    ax.arrow(7, 7.8, 0, -0.6, head_width=0.2, head_length=0.1, fc=arrow_color, ec=arrow_color)
    
    # Step 2: Concept Extraction
    ax.text(7, 6.8, 'Concept Extraction (spaCy NLP)', ha='center', va='center', 
            fontsize=11, weight='bold', style='italic')
    
    # Arrow down
    ax.arrow(7, 6.5, 0, -0.3, head_width=0.2, head_length=0.1, fc=arrow_color, ec=arrow_color)
    
    # Step 3: Concepts
    concepts = ['an old man', 'the rain at night', 'street lamps']
    concept_boxes = []
    for i, concept in enumerate(concepts):
        x_pos = 2 + i * 4
        box = FancyBboxPatch((x_pos, 5), 3, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=concept_color, 
                             edgecolor='black')
        ax.add_patch(box)
        ax.text(x_pos + 1.5, 5.4, f'Concept {i+1}:\n"{concept}"', 
                ha='center', va='center', fontsize=10)
        concept_boxes.append((x_pos + 1.5, 5))
    
    # Arrows down from concepts
    for x_pos, _ in concept_boxes:
        ax.arrow(x_pos, 4.8, 0, -0.5, head_width=0.15, head_length=0.1, fc=arrow_color, ec=arrow_color)
    
    # Step 4: Image Generation + Latent Caching
    ax.text(7, 4.0, 'Generate Images + Cache Latents at Step 25/50', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Image generation boxes
    for i, (x_pos, _) in enumerate(concept_boxes):
        # Image box
        img_box = FancyBboxPatch((x_pos - 1, 2.5), 2, 1, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='black')
        ax.add_patch(img_box)
        ax.text(x_pos, 3, f'Image {i+1}', ha='center', va='center', fontsize=9)
        
        # Latent cache box
        latent_box = FancyBboxPatch((x_pos - 0.75, 1.2), 1.5, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=latent_color, 
                                    edgecolor='black')
        ax.add_patch(latent_box)
        ax.text(x_pos, 1.5, f'Cached\nLatent {i+1}', ha='center', va='center', fontsize=8)
        
        # Arrow from image to latent
        ax.arrow(x_pos, 2.4, 0, -0.5, head_width=0.1, head_length=0.05, fc=arrow_color, ec=arrow_color)
    
    # Arrows converging to averaging
    for x_pos, _ in concept_boxes:
        ax.arrow(x_pos, 1.1, (7 - x_pos) * 0.7, -0.4, 
                head_width=0.1, head_length=0.05, fc=arrow_color, ec=arrow_color)
    
    # Step 5: Latent Averaging
    avg_box = FancyBboxPatch((5.5, 0.2), 3, 0.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=latent_color, 
                             edgecolor='black', linewidth=2)
    ax.add_patch(avg_box)
    ax.text(7, 0.5, 'Averaged Latents', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrow down
    ax.arrow(7, 0.1, 0, -0.3, head_width=0.2, head_length=0.1, fc=arrow_color, ec=arrow_color)
    
    # Step 6: Final Reconstruction
    recon_box = FancyBboxPatch((4, -1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=output_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(recon_box)
    ax.text(7, -1, 'RECON Reconstruction\n(25 steps starting from cached latents)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Side annotation
    ax.text(12.5, 1.5, 'Key Benefits:\n\n• Semantic decomposition\n• Reduced final steps\n• Concept interpretability\n• Quality preservation', 
            ha='left', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Title
    ax.text(7, 9.5, 'RECON: Concept-based Reconstruction Method', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Set limits and hide axes
    ax.set_xlim(0, 14)
    ax.set_ylim(-2.5, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_comparison_chart():
    """Create a comparison chart showing computational savings"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Step comparison
    methods = ['Standard\nGeneration', 'RECON\nMethod']
    steps = [50, 25]  # Final generation steps
    colors = ['#ff7f7f', '#7fbf7f']
    
    bars1 = ax1.bar(methods, steps, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Final Generation Steps')
    ax1.set_title('Computational Steps Comparison')
    ax1.set_ylim(0, 60)
    
    # Add value labels on bars
    for bar, step in zip(bars1, steps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(step), ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Process breakdown
    labels = ['Concept\nGeneration', 'Final\nReconstruction']
    standard_steps = [0, 50]
    recon_steps = [150, 25]  # 3 concepts × 50 steps + 25 final steps
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax2.bar(x - width/2, standard_steps, width, label='Standard', color='#ff7f7f', alpha=0.7)
    ax2.bar(x + width/2, recon_steps, width, label='RECON', color='#7fbf7f', alpha=0.7)
    
    ax2.set_ylabel('Total Steps')
    ax2.set_title('Detailed Process Breakdown')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    # Add annotations
    ax2.text(1, 175, f'Total: {sum(recon_steps)} steps\nvs {sum(standard_steps)} steps', 
             ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def main():
    """Create and save visualization diagrams"""
    
    print("Creating RECON method visualization...")
    
    # Create method diagram
    fig1 = create_recon_diagram()
    fig1.savefig('/scratch/gilbreth/lu842/Recon/examples/recon_demo/method_diagram.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Method diagram saved to examples/recon_demo/method_diagram.png")
    
    # Create comparison chart
    fig2 = create_comparison_chart()
    fig2.savefig('/scratch/gilbreth/lu842/Recon/examples/recon_demo/performance_comparison.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Performance comparison saved to examples/recon_demo/performance_comparison.png")
    
    plt.close('all')
    print("Visualization creation complete!")

if __name__ == "__main__":
    main()