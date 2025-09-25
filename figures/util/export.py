import matplotlib.pyplot as plt
import os

def savefig(path, bbox_inches='tight', pad_inches=0.012, dpi=300, **kwargs):
    print(f'Figure saved to {os.path.abspath(path)}')
    plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)