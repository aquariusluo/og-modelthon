# [Model Name]

## Overview

[Description of the model, its purpose, and key features]

## Model Details

- **Architecture:** [e.g., Transformer, LSTM, etc.]
- [Any other descriptive features that you may find helpful to include]

### Inference

Using the CLI:

```bash
opengradient infer -m [Model CID] --input 'input.json'

```

Using Python:

```python
import opengradient as og

og.init(...)

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

## Performance

[Key performance metrics and benchmarks]

*any of the following:*

- **Accuracy**: Ratio of correct predictions to total predictions
- **Precision**: Proportion of true positive predictions among all positive predictions
- **Recall (Sensitivity)**: Proportion of true positive predictions among all actual positive instances
- **F1 Score**: Harmonic mean of precision and recall
- **Mean Absolute Error (MAE)**: Average magnitude of errors between predicted and actual values
- **Mean Squared Error (MSE)**: Average of squared differences between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Area Under the Curve (AUC-ROC)**: Performance measure for binary classification problems
- **Confusion Matrix**: Table showing true positives, true negatives, false positives, and false negatives
- **R-squared (R2) Score**: Proportion of variance in the dependent variable predictable from the independent variable(s)

## Limitations and Biases

[Discuss any known limitations or biases of the model]

## Citation

If you use this model in your research, please cite:

```
@article{[author_last_name][year][model_name],
  title={[Model Name]: [Brief Title]},
  author={[Author list]},
  year={[Year]}
}

```
