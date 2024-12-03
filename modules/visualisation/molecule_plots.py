# Standard library imports
from typing import List, Tuple

# Third-party imports
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

def plot_best_molecule(smiles: str, title: str = "Best Molecule", image_size=(1200, 1200)):
    """
    Plots the best molecule given a SMILES string with high quality.

    Parameters:
    - smiles (str): The SMILES string of the molecule to be plotted.
    - title (str): The title of the plot.
    - image_size (tuple): The size of the image as a tuple (width, height).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return
    
    # Generate a high-quality image of the molecule
    img = Draw.MolToImage(mol, size=image_size)

    # Create a figure and axis to display the image
    dpi = 300  # Adjust DPI for the figure
    fig, ax = plt.subplots(figsize=(image_size[0]/dpi, image_size[1]/dpi), dpi=dpi)
    ax.imshow(img)
    ax.axis('off')  # Hide the axes
    ax.set_title(title)

    plt.show()  # Display the plot


def plot_top_molecules(top_molecules: List[Tuple[str, float]], title: str = "Top Molecules", 
                       image_size=(1200, 1200), title_fontsize=4):
    """
    Plots the top molecules given a list of SMILES strings with their LogP values, 
    displaying both the SMILES string and LogP value in the title of each plot 
    with a specified font size.

    Parameters:
    - top_molecules (List[Tuple[str, float]]): List of tuples containing SMILES 
                                               strings and their LogP values.
    - title (str): The title for the entire plot.
    - image_size (tuple): The size of each individual molecule image as a tuple (width, height).
    - title_fontsize (int): The font size for the titles.
    """
    # Set the DPI for the figure
    dpi = 300
    fig, axes = plt.subplots(1, 3, figsize=(image_size[0] * 3 / dpi, image_size[1] / dpi), dpi=dpi)

    # Plot each molecule
    for ax, (smiles, logp) in zip(axes, top_molecules):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            ax.axis('off')  # Hide axes if the molecule is invalid
            continue
        
        img = Draw.MolToImage(mol, size=image_size)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        # Set title with specified font size
        ax.set_title(f"{smiles}\nLogP: {logp:.4f}", fontsize=3)

    plt.suptitle(title, fontsize=title_fontsize + 2)  # Slightly larger font for the overall title
    plt.tight_layout()
    plt.show()  # Display the plot


def main():
    # Example usage
    smiles = "CCO"
    plot_best_molecule(smiles, title="Best Molecule")

    top_molecules = [("CCO", -0.32), ("CCN", 0.12), ("CCC", 0.45)]
    plot_top_molecules(top_molecules, title="Top Molecules")


if __name__ == "__main__":
    main()
    