# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import glob
import os
import sys
import argparse
import h5py
import fnmatch

import PIL.Image
import numpy as np

from collections import defaultdict

size_stats = defaultdict(int)
format_stats = defaultdict(int)


def load_image(fname):
    global format_stats, size_stats
    im = PIL.Image.open(fname)
    format_stats[im.mode] += 1
    if (im.width < 256 or im.height < 256):
        size_stats['< 256x256'] += 1
    else:
        size_stats['>= 256x256'] += 1
    arr = np.array(im.convert('RGB'), dtype=np.uint8)
    assert len(arr.shape) == 3
    return arr.transpose([2, 0, 1])


def filter_image_sizes(images):
    filtered = []
    for idx, fname in enumerate(images):
        if (idx % 100) == 0:
            print ('loading images', idx, '/', len(images))
        try:
            with PIL.Image.open(fname) as img:
                w = img.size[0]
                h = img.size[1]
                if (w > 512 or h > 512) or (w < 256 or h < 256):
                    continue
                filtered.append((fname, w, h))
        except:
            print ('Could not load image', fname, 'skipping file..')
    return filtered


examples='''examples:

  python %(prog)s --input-dir=./ILSVRC2012_img_val --out=imagenet_val_raw.h5
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a HDF5 dataset file.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", help="Directory containing ImageNet images (can be glob pattern for subdirs)")
    parser.add_argument("--out", help="Filename of the output file")
    parser.add_argument("--max-files", help="Convert up to max-files images.  Process all if unspecified.")
    args = parser.parse_args()

    if args.input_dir is None:
        print ('Must specify input file directory with --input-dir')
        sys.exit(1)
    if args.out is None:
        print ('Must specify output filename with --out')
        sys.exit(1)

    print ('Loading image list from %s' % args.input_dir)
    images = []
    pattern = os.path.join(args.input_dir, '**/*')
    all_fnames = glob.glob(pattern, recursive=True)
    for fname in all_fnames:
        # include only JPEG/jpg/png
        if fnmatch.fnmatch(fname, '*.JPEG') or fnmatch.fnmatch(fname, '*.jpg') or fnmatch.fnmatch(fname, '*.png'):
            images.append(fname)
    images = sorted(images)
    np.random.RandomState(0xbadf00d).shuffle(images)

    filtered = filter_image_sizes(images)
    if args.max_files:
        filtered = filtered[0:int(args.max_files)]

    # ----------------------------------------------------------
    outdir = os.path.dirname(args.out)
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)
    num_images = len(filtered)
    num_pixels_total = 0
    with h5py.File(args.out, 'w') as h5file:
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        dset_shapes = h5file.create_dataset('shapes', (num_images, 3), dtype=np.int32)
        dset_images = h5file.create_dataset('images', (num_images,), dtype=dt)
        for (idx, (imgname, w, h)) in enumerate(filtered):
            print ("%d/%d: %s" % (idx+1, len(filtered), imgname))
            dset_images[idx] = load_image(imgname).flatten()
            dset_shapes[idx] = (3, h, w)
            num_pixels_total += h*w

    print ('Dataset statistics:')
    print ('  Total pixels', num_pixels_total)
    print ('  Formats:')
    for key in format_stats:
        print ('    %s: %d images' % (key, format_stats[key]))
    print ('  width,height buckets:')
    for key in size_stats:
        print ('    %s: %d images' % (key, size_stats[key]))


if __name__ == "__main__":
    main()
