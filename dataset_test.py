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

import unittest

from torch.utils.data import DataLoader

from dataset import LmdbDataset
from torchvision import transforms

class LmdbDatasetTest(unittest.TestCase):
    def setUp(self):
        normalizeFunc = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalizeFunc])
        self.dataset = LmdbDataset("val.lmdb",transform=transf)
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True,num_workers=4)

    def testRead(self):
        for i, data in enumerate(self.dataloader):
            img, label = data
            print(i, img.shape, label)

if __name__ == '__main__':
    unittest.main()
