from dowhy import CausalModel
import dowhy.datasets
import networkx as nx, numpy as np, pandas as pd
from dowhy import gcm

# Generate some sample data
data = dowhy.datasets.linear_dataset(
    beta=10,
    num_common_causes=5,
    num_instruments=2,
    num_samples=10000)

# Step 1: Create a causal model from the data and given graph.
model = CausalModel(
    data=data["df"],
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["gml_graph"])

# Step 2: Identify causal effect and return target estimands
identified_estimand = model.identify_effect()

# Step 3: Estimate the target estimand using a statistical method.
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")

# Step 4: Refute the obtained estimate using multiple robustness checks.
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")

# Let's generate some "normal" data we assume we're given from our problem domain:
X = np.random.normal(loc=0, scale=1, size=1000)
Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
data = pd.DataFrame(dict(X=X, Y=Y, Z=Z))

# Step 1: Model our system:
causal_model = gcm.StructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')]))
gcm.auto.assign_causal_mechanisms(causal_model, data)

# Step 2: Train our causal model with the data from above:
gcm.fit(causal_model, data)

# Step 3: Perform a causal analysis. For instance, root cause analysis, where we observe
anomalous_sample = pd.DataFrame(dict(X=[0.1], Y=[6.2], Z=[19]))  # Here, Y is the root cause.
# ... and would like to answer the question:
# "Which node is the root cause of the anomaly in Z?":
anomaly_attribution = gcm.attribute_anomalies(causal_model, "Z", anomalous_sample)