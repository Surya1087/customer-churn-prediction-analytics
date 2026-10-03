from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churn_analytics.data import load_data
from churn_analytics.modeling import train
root = Path(__file__).parents[1]
df = load_data(root / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(train(df, root / "models"))
