# Data Preprocessing in ONNX

Suppose there is a standardized input for an inference
(as in the OG Modelthon
spot forecast tracks) but the model that has been developed requires
some changes to that standardized input such as feature
engineering or just selection of a subset of features. This can
be done by manually building a data preprocessing flow in ONNX
which can then be added to an already built ONNX ML model.

Suppose we have the standard input of 81 features below that we are
required to use as our input, however our model requires only
a subset of them, and they need to be log-transformed. Note that
the input is actually a 2d array as opposed to 1d.

```python
[[1825.26, 1823.47, 1823.07, 1825.58, 1826.91, 1825.99, 1829.26,
1833.43, 1822.85, 1821.52, 1828.32, 1828.44, 1826.23, 1825.64,
1827.31, 1829.67, 1829.68, 1842.83, 1834.32, 1827.73, 1824.48,
1821.07, 1818.83, 1812.12, 1820.19, 1824.4, 1823.7, 1824.07,
1821.2, 1819.61, 1824.48, 1825.26, 1823.46, 1823.06, 1825.58,
1826.92, 1825.99, 1829.25, 1833.43, 1822.85, 28440.56, 28424.98,
28482.71, 28571.08, 28580.01, 28541.1, 28583.36, 28610.62,
28438.26, 28465.36, 28464.65, 28485.11, 28482.72, 28572.93,
28589.13, 28625.1, 28603.78, 28819.71, 28625.61, 28547.2, 28420.54,
28389.37, 28351.35, 28353.78, 28530.59, 28526.59, 28503.29,
28467.31, 28410.35, 28408.57, 28425.49, 28440.56, 28424.99,
28482.72, 28571.08, 28580.0, 28541.1, 28583.36, 28610.63, 28438.27,
10.0]]
```

We can cleverly use [ONNX Operators](https://onnx.ai/onnx/operators/) 
to perform necessary preprocessing. In the described case, if our ML
model requires features at indices 5 to 16 (note that ONNX
begins with index `0` like Python) we can subset the standard input by using
the `Slice` operation and log-transform with the `Log` operation, then
finally merge with our ML model using `compose.merge_models` as follows.

In the following case we assume we have built an ML model
which is the object `ml_model` that has
an input called `X` (which is standard in some ONNX conversion
packages).

```python
import onnx
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx import numpy_helper, TensorProto
import numpy as np

# Build ONNX from defined Nodes
node_list = []

# for set values we require initialized values
initializer_list = []

# define input name
input_data = make_tensor_value_info('candles', 
                                    TensorProto.FLOAT, [1,81])

# build start and end values for slice
# note index shape 2 for the 2d array
startval = np.array([0,5], dtype=np.int32)
initializer_list.append(
    numpy_helper.from_array(startval, name='startval'))
endval = np.array([0,16], dtype=np.int32)
initializer_list.append(
    numpy_helper.from_array(endval, name='endval'))

# create slice node
node_list.append(make_node('Slice', 
          inputs=['candles',
           'startval',
           'endval'
           ], 
          outputs=['selected_features']))


# log-transform features
node_list.append(make_node(
    "Log",
    inputs=['selected_features'],
    outputs=['log_features'],
))

output = make_tensor_value_info('log_features', 
                                TensorProto.FLOAT, [None,None])

# build graph and ONNX
graph = make_graph(node_list,  # nodes
                    'preprocess',  # a name
                    [input_data],  # inputs
                    [output],
                    initializer_list
                    )

preprocessing_onnx_model = make_model(graph)

# merge models
preproc_plus_reg = onnx.compose.merge_models(
                        preprocessing_onnx_model,
                        ml_model,
                        [('log_features','X')])
```

We have now created a new ONNX model as the object
`preproc_plus_reg` composed of our preprocessing
plus our ML model which required the preprocessing. This
can then be saved and exported normally. Postprocessing
of the ML model output can be performed in a similar matter.

