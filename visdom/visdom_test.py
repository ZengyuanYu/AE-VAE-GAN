#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-1-18
import visdom
import numpy as np

vis = visdom.Visdom()
vis.text('Hello,World!')
vis.image(np.ones((3, 100, 100)))