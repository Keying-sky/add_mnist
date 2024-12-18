import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Tsne:
    def __init__(self, random=100, perplexity=30, figsize=(20, 16), dpi=300):
       """
       Initialise Tsne visualisation class.
       
       Params:
           random: random seed for reproducibility
           perplexity: perplexity param for t-SNE
           figsize: figure size for plots
           dpi: dpi for saved figures
       """
       self.random = random
       self.perplexity = perplexity
       self.figsize = figsize
       self.dpi = dpi
       self.x_test = None
       self.y_test = None  
       self.digits_test = None
       self.embeddings = None
       self.tsne_embed = None
       self.tsne_raw = None
       
       np.random.seed(random)
       
    def load_data(self, paths):
        """Load test data and model."""
        
        data = np.load(paths.DATA_DIR/'combined_mnist.npz')
        self.x_test = data['x_test']
        self.y_test = data['y_test']
        self.digits_test = data['digits_test']
        
        model = keras.models.load_model(paths.MODEL_DIR/'best_model.h5')
        
        # get embedding
        embedding_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        self.embeddings = embedding_model.predict(self.x_test)
       
    def compute_tsne(self):
        """Compute t-SNE for both embeddings and raw input."""
        # raw input
        x_test_flat = self.x_test.reshape(len(self.x_test), -1)

        # compute t-SNE
        self.tsne_embed = TSNE(n_components=2, perplexity=self.perplexity, random_state=self.random).fit_transform(self.embeddings) 
        self.tsne_raw = TSNE(n_components=2, perplexity=self.perplexity, random_state=self.random).fit_transform(x_test_flat)
                      
    def visualise_tsne(self, tsne_results, title, path):
        """Basic t-SNE visualisation without labels."""

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                            c=self.y_test, cmap='tab20',
                            alpha=0.8, s=50)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sum Label', fontsize=20)
        
        plt.title(title, fontsize=24, pad=20)
        plt.xlabel('t-SNE dimension 1', fontsize=20)
        plt.ylabel('t-SNE dimension 2', fontsize=20)
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
       
    def with_labels(self, title, path):
        """t-SNE visualisation with annotations showing original label pairs."""

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        scatter = plt.scatter(self.tsne_embed[:, 0], self.tsne_embed[:, 1], c=self.y_test, cmap='tab20', alpha=0.8, s=50)
                            
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sum Label', fontsize=20)
        
        plt.title(title, fontsize=24, pad=20)
        plt.xlabel('t-SNE dimension 1', fontsize=20)
        plt.ylabel('t-SNE dimension 2', fontsize=20)
        
        # compute and add annotations
        unique_pairs = {}
        for i, (pair, point) in enumerate(zip(self.digits_test, self.tsne_embed)):
            pair = tuple(pair)
            if pair not in unique_pairs:
                unique_pairs[pair] = {'points': [], 'sum': self.y_test[i]}
            unique_pairs[pair]['points'].append(point)
        
        for pair, data in unique_pairs.items():
            points = np.array(data['points'])
            center = points.mean(axis=0)
            
            plt.annotate(f'{pair[0]}+{pair[1]}',xy=(center[0], center[1]),
                        xytext=(4, 4), textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                        fontsize=15)
        
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
       
    def perplexities(self, perplex_list, path):
        """ Visualise t-SNE results with different perplexity values in subplots."""
        
        perpls = perplex_list
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes_flat = axes.flatten()
        
        # iterate through perplexities
        for idx, perp in enumerate(perpls):
            # t-SNE
            tsne = TSNE(n_components=2, perplexity=perp, random_state=self.random)
            tsne_results = tsne.fit_transform(self.embeddings)
            
            scatter = axes_flat[idx].scatter(tsne_results[:, 0], tsne_results[:, 1], c=self.y_test, cmap='tab20', alpha=0.8, s=10)
                                        
            axes_flat[idx].set_title(f'Perplexity: {perp}', fontsize=14, pad=10)
            axes_flat[idx].set_xlabel('t-SNE dimension 1', fontsize=12)
            axes_flat[idx].set_ylabel('t-SNE dimension 2', fontsize=12)
        
        plt.subplots_adjust(right=0.9) 
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
        fig.colorbar(scatter, cax=cbar_ax, label='Sum Value')

        plt.savefig(path/'perplexities.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()

