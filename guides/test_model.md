# Testing Your Model
- After completing the training & upload process outlined in `/og-example-model/og-example.ipynb`, you can use the following input to test that the model will work during the evaluation phase. Note that the the input name *must* be `"candles"`. 

```python
candles = np.array(
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
       10.0]], dtype=float)

tx_hash, output = og.infer(
    model_cid=model_cid,
    model_input={
        "candles": candles  
    },
    inference_mode=og.InferenceMode.VANILLA
)
```
