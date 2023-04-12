import numpy as np
import matplotlib.pyplot as plt
import argparse

from viewformer.data.loaders import DatasetLoader
from viewformer.utils.tensorflow import load_model
from viewformer.utils.visualization import np_imgrid
from viewformer.evaluate.evaluate_transformer_multictx import generate_batch_predictions
import viewformer.models.utils
import os

def create_parser():
    """Creates a parser from command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='/home/ubuntu/novel-pov/viewformer/dataset', help='Path for getting dataset')

    parser.add_argument('--codebook_path', type=str, default='interiornet-codebook-th', help='Path for getting checkbook weights')
    parser.add_argument('--transformer_path', type=str, default='interiornet-transformer-tf', help='Path for getting transformer weights')

    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_name', type=str, default="foo", help='The name of the output')

    parser.add_argument('--batch_size', type=int, default=1, help='The number of images in a batch.')

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    setattr(viewformer.models.utils, 'load_lpips_model', lambda *args, **kwargs: None)

    plt.rcParams['figure.figsize'] = [12, 8]
    test_loader = DatasetLoader(path=args.dataset_path, split='test')

    codebook = load_model(args.codebook_path)
    transformer = load_model(args.transformer_path)

    input_batch = test_loader[0]['frames'].astype('float32') / 255.

    plt.imshow(np_imgrid(input_batch)[0])
    plt.savefig('{}/{}_context.png'.format(args.output_dir, args.output_name))

    output_batch = np.clip(codebook(input_batch * 2 - 1)[0] / 2 + 0.5, 0, 1)
    plt.imshow(np_imgrid(output_batch)[0])
    plt.savefig('{}/{}_codebook.png'.format(args.output_dir, args.output_name))

    images, cameras = test_loader[0]['frames'][np.newaxis, ...], test_loader[0]['cameras'][np.newaxis, ...]
    output = generate_batch_predictions(transformer, codebook, images, cameras)
    plt.imshow(np_imgrid(output['generated_images'][0])[0])
    plt.savefig('{}/{}_generated.png'.format(args.output_dir, args.output_name))