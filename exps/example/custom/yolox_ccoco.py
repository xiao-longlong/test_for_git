#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = "/workspace/datasets/CustomCOCO"
        self.train_ann = "instances_traincustomcoco.json"
        self.val_ann = "instances_valcustomcoco.json"
        self.test_ann = "instances_testcustomcoco.json"
        self.max_epoch = 10
