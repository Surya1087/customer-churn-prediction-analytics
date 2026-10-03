from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churn_analytics.data import load_data, validate_data
from churn_analytics.database import write_database

root = Path(__file__).parents[1]
source = root / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not source.exists():
    raise SystemExit(f"Dataset not found: {source}. See data/README.md")
df = load_data(source)
print(validate_data(df))
write_database(df, root / "database/churn.db")
print("Wrote database/churn.db")
