from pathlib import Path
import sqlite3
root = Path(__file__).parents[1]
sql = (root / "sql/analytical_queries.sql").read_text()
with sqlite3.connect(root / "database/churn.db") as con:
    for statement in sql.split(";"):
        if statement.strip() and not statement.strip().startswith("--"):
            try:
                print(con.execute(statement).fetchall())
            except sqlite3.Error as exc:
                print(f"Skipped statement: {exc}")
