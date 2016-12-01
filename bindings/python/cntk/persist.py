# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import cntk_py
from .utils.swig_helper import typemap

def save_as_legacy_model(root_op, filename):
    '''
    Save the network of ``root_op`` in ``filename``.

    Args:
        root_op (:class:`~cntk.functions.Function`): op of the graph to save
        filename (str): filename to store the model in
        use_legacy_format (str): if 'True', model is stored using legacy format.
             Otherwise, it's stored using protobuf-based protocol serialization.
    '''
    cntk_py.save_as_legacy_model(root_op, filename)
