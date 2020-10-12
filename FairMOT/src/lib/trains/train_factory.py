from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .fcos_mot import MotTrainer
# from .mot import MotTrainer


train_factory = {
  'mot': MotTrainer,
}
