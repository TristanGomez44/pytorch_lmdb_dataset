# Copyright 2019 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npuichigo@gmail.com (zhangyuchao)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import lmdb
import numpy as np

from proto import utils
from proto import tensor_pb2

import glob,os,cv2

def create_db(dataPath,output_file,classNb,imgPerClass,logInterval):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 160000000000   # MODIFY
    print(LMDB_MAP_SIZE)
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
    #env = lmdb.open(output_file)

    checksum = 0
    with env.begin(write=True) as txn:
        classFolds = sorted(glob.glob(os.path.join(dataPath,"*/")))
        for i in range(classNb):
            imgPaths = sorted(glob.glob(os.path.join(classFolds[i],"*.JPEG")))
            imgToRead = len(imgPaths) if imgPerClass is None else imgPerClass
            for j in range(imgToRead):
                # MODIFY: add your own data reader / creator
                width = 64
                height = 32
                img_data = cv2.imread(imgPaths[j])
                label = np.asarray(i)

                # Create TensorProtos
                tensor_protos = tensor_pb2.TensorProtos()
                img_tensor = utils.numpy_array_to_tensor(img_data)
                tensor_protos.protos.extend([img_tensor])

                label_tensor = utils.numpy_array_to_tensor(label)
                tensor_protos.protos.extend([label_tensor])
                txn.put(
                    '{}'.format((j+1)+i*imgToRead).encode('ascii'),
                    tensor_protos.SerializeToString()
                )

                if (j % logInterval == 0):
                    print("Inserted {} rows".format((j+1)+i*imgToRead))


def main():
    parser = argparse.ArgumentParser(
        description="LMDB creation"
    )
    parser.add_argument("--data_path", type=str, default=None,help="Data path",required=True)
    parser.add_argument("--output_file", type=str, default=None,help="Path to write the database to",required=True)
    parser.add_argument("--class_nb", type=int, default=1000,help="The number of class")
    parser.add_argument("--img_per_class", type=int, default=None,help="The number of image per class. Do not set this arg\
                            if you want all to use all the images available.")
    parser.add_argument("--log_interval", type=int, default=20,help="The number of images to insert before to print.")

    args = parser.parse_args()

    create_db(args.data_path,args.output_file,args.class_nb,args.img_per_class,args.log_interval)


if __name__ == '__main__':
    main()
