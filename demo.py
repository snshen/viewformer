import numpy as np
import matplotlib.pyplot as plt

from viewformer.data.loaders import DatasetLoader
from viewformer.utils.tensorflow import load_model
from viewformer.utils.visualization import np_imgrid
from viewformer.evaluate.evaluate_transformer_multictx import generate_batch_predictions
import viewformer.models.utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
setattr(viewformer.models.utils, 'load_lpips_model', lambda *args, **kwargs: None)

plt.rcParams['figure.figsize'] = [12, 8]

test_loader = DatasetLoader(path='/home/ubuntu/viewformer/dataset', split='test')

codebook = load_model('interiornet-codebook-th')
transformer = load_model('interiornet-transformer-tf')

input_batch = test_loader[0]['frames'].astype('float32') / 255.
plt.imshow(np_imgrid(input_batch)[0])
plt.savefig('foo.png')

output_batch = np.clip(codebook(input_batch * 2 - 1)[0] / 2 + 0.5, 0, 1)

plt.imshow(np_imgrid(output_batch)[0])
plt.savefig('foo2.png')

images, cameras = test_loader[0]['frames'][np.newaxis, ...], test_loader[0]['cameras'][np.newaxis, ...]
print(cameras)
output = generate_batch_predictions(transformer, codebook, images, cameras)
plt.imshow(np_imgrid(output['generated_images'][0])[0])
plt.savefig('output.png')

print("FUCK ME")