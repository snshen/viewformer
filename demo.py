import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
from PIL import Image
import argparse
import tensorflow as tf

from viewformer.data.loaders import DatasetLoader
from colmap.get_poses import ColmapDataLoader
from viewformer.utils.tensorflow import load_model
from viewformer.utils.visualization import np_imgrid
from viewformer.evaluate.evaluate_transformer import to_relative_cameras, normalize_cameras, resize_tf, to_relative_cameras2
from generate_query_trajectories import QueryTrajectories
import viewformer.models.utils

def generate_batch_predictions(transformer_model, codebook_model, images,
                               cameras, query_cameras):
    ground_truth_cameras = cameras
    if transformer_model.config.augment_poses == 'relative':
        # Rotate poses for relative augment
        query_cameras, _ = to_relative_cameras2(query_cameras, cameras)
        cameras, transform = to_relative_cameras(cameras)
    cameras = normalize_cameras(cameras)

    with tf.name_scope('encode'):

        def encode(images):
            fimages = resize_tf(images, codebook_model.config.image_size)
            fimages = tf.image.convert_image_dtype(fimages, tf.float32)
            fimages = fimages * 2 - 1
            codes = codebook_model.encode(fimages)[-1]  # [N, H', W']
            codes = tf.cast(codes, dtype=tf.int32)
            return codes

        # codes = tf.ragged.map_flat_values(encode, codes)
        batch_size, seq_len, *im_dim = tf.unstack(tf.shape(images), 5)
        code_len = transformer_model.config.token_image_size
        codes = tf.reshape(
            encode(tf.reshape(images, [batch_size * seq_len] + list(im_dim))),
            [batch_size, seq_len, code_len, code_len])

    # Generate image tokens
    with tf.name_scope('transformer_generate_images'):
        n = query_cameras.shape[1]
        image_generation_input_ids = tf.concat([
            codes,
            tf.fill(
                tf.shape(codes[:, :1]),
                tf.constant(transformer_model.mask_token, dtype=codes.dtype))
        ], 1)
        generated_codes = tf.zeros((1, 0, 8, 8), dtype=tf.int64)
        for i in range(n):
            tot_cameras = tf.concat([cameras, [query_cameras[:, i, :]]], 1)
            output = transformer_model(dict(
                input_ids=image_generation_input_ids, poses=tot_cameras),
                                       training=False)
            generated_code_i = tf.argmax(output['logits'], -1)[:, -1]
            generated_codes = tf.concat([generated_codes, [generated_code_i]],
                                        1)

    # Decode images
    with tf.name_scope('decode_images'):
        batch_size, seq_len, token_image_shape = tf.split(
            tf.shape(generated_codes), (1, 1, 2), 0)
        generated_images = codebook_model.decode_code(
            tf.reshape(generated_codes,
                       tf.concat((batch_size * seq_len, token_image_shape),
                                 0)))
        generated_images = tf.clip_by_value(generated_images, -1, 1)
        generated_images = tf.image.convert_image_dtype(
            generated_images / 2 + 0.5, tf.uint8)
        generated_images = tf.reshape(
            generated_images,
            tf.concat((batch_size, seq_len, tf.shape(generated_images)[-3:]),
                      0))

    return dict(ground_truth_images=images[:, -1],
                generated_images=generated_images,
                ground_truth_cameras=ground_truth_cameras)

def get_query_trajectory(env_path, start_frame, num_frames, skip_frames):
    qt = QueryTrajectories(env_path)
    output = qt.get_traj(start_frame, num_frames, skip_frames)
    return output

def get_straight_trajectory(start_camera, num_frames):
    query_cameras = np.zeros((1, num_frames, 7), dtype=np.float32)
    xyz = start_camera[:3]
    quat = start_camera[3:]
    r = R.from_quat(quat)
    orig_euler = r.as_euler('zyx', degrees=True)
    for i in range(num_frames):
        new_euler = orig_euler + np.array([i*2, 0, 0])
        new_xyz = xyz + np.array([-i*0.05, 0, 0])
        new_quat = R.from_euler('zyx', new_euler, degrees=True).as_quat()
        query_cameras[0][i] = np.concatenate((new_xyz, quat))
    return query_cameras

def create_parser():
    """Creates a parser from command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='/home/ec2-user/novel-cross-view-generation/viewformer/dataset/interiornet', help='Path for getting dataset')
    parser.add_argument('--query_env_path', type=str, default='/home/ec2-user/novel-cross-view-generation/viewformer/data/HD6/3FO4JTI0VMJS', help='Path to environment folder that you want to extract queries from.')
    parser.add_argument('--codebook_path', type=str, default='interiornet-codebook-th', help='Path for getting checkbook weights')
    parser.add_argument('--transformer_path', type=str, default='interiornet-transformer-tf', help='Path for getting transformer weights')

    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_name', type=str, default="foo", help='The name of the output')

    parser.add_argument('--batch_size', type=int, default=1, help='The number of images in a batch.')
    parser.add_argument('--seq_num', type=int, default=0, help='The batch of context images to use.')
    parser.add_argument('--start_query_frame', type=int, default=0, help='The start query frame.')
    parser.add_argument('--num_frames', type=int, default=50, help='Number of frames to evaluate.')

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    setattr(viewformer.models.utils, 'load_lpips_model', lambda *args, **kwargs: None)

    plt.rcParams['figure.figsize'] = [12, 8]
    # test_loader = DatasetLoader(path=args.dataset_path, split='test', sequence_size=30)
    test_loader = ColmapDataLoader()

    seq_num = args.seq_num
    print("Number of sequence batches: ", len(test_loader))
    input_batch = test_loader[seq_num]['frames'].astype('float32') / 255.

    plt.imshow(np_imgrid(input_batch)[0])
    plt.savefig('{}/{}_context.png'.format(args.output_dir, args.output_name))

    # images, cameras = test_loader[seq_num]['frames'][np.newaxis, ...], test_loader[seq_num]['cameras'][np.newaxis, ...]
    i = args.seq_num
    images = []
    cameras = []
    while i < args.seq_num+args.batch_size and i < len(input_batch):
        frame, camera = test_loader[i]
        images.append(frame)
        cameras.append(camera)
    images = np.array(images)[np.newaxis, ...]
    cameras = np.array(cameras)[np.newaxis, ...]

    # Build query traj
<<<<<<< HEAD
    # env_path = args.query_env_path
    # output = get_query_trajectory(env_path, args.start_query_frame, args.num_frames, 5)
    # query_cameras = output['cameras']
    # gt_images = output['frames']
    query_cameras = get_straight_trajectory(cameras[0][-1], 10)
=======
    env_path = args.query_env_path
    output = get_query_trajectory(env_path, args.start_query_frame, args.num_frames, 2)
    query_cameras = output['cameras']
    gt_images = output['frames']
>>>>>>> 8bb5449 (make frame rate smaller)

    codebook = load_model(args.codebook_path)
    transformer = load_model(args.transformer_path)

    output = generate_batch_predictions(transformer, codebook, images, cameras, query_cameras)
    img_arr = output['generated_images'][0]
    plt.imshow(np_imgrid(img_arr)[0])
    plt.savefig('{}/{}_generated.png'.format(args.output_dir, args.output_name))
    # plt.imshow(np_imgrid(gt_images)[0])
    # plt.savefig('{}/{}_gt.png'.format(args.output_dir, args.output_name))

    imgs = [Image.fromarray(img.numpy()) for img in img_arr]
    imgs[0].save('{}/generated_video.gif'.format(args.output_dir), save_all=True, append_images=imgs[1:], duration=200, loop=0)

    # imgs = [Image.fromarray(img) for img in gt_images]
    # imgs[0].save('{}/gt_video.gif'.format(args.output_dir), save_all=True, append_images=imgs[1:], duration=200, loop=0)
