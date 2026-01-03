# HyBINN
Research project to create an attentive hybrid biologically-informed neural network for breast cancer survival prediction.

### Neural Network Architecture

The network will use 3 input streams:
1. Sparse BINN <—— Genes mapped to pathways in Reactome
2. Dense MLP <—— All other genes (not mapped in Reactome)
3. Clinical MLP <—— Clinical variables

The three streams will be merged into a single output node using an attention mechanism, so the network learns which inputs are more important based on the patient.
The final output node provides a CoxPH survival prediction.
