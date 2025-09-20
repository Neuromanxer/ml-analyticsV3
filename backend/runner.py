# migrate_actions.py
from sqlalchemy import create_engine, text

URL = "postgresql+psycopg://ethanhong:jXGQTW5siqxNNK5YXztrEovd1cJTDMWa@dpg-d1n0o56uk2gs739cigq0-a.virginia-postgres.render.com/ml_insights_db"

engine = create_engine(URL, future=True)

def scalar(conn, sql, **params):
    return conn.execute(text(sql), params).scalar()

def table_exists(conn, table):
    return scalar(conn, "SELECT to_regclass(:t)", t=f"public.{table}") is not None

def col_exists(conn, table, col):
    return scalar(conn, """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t AND column_name=:c
    """, t=table, c=col) is not None

def constraint_exists(conn, name):
    return scalar(conn, "SELECT 1 FROM pg_constraint WHERE conname=:n", n=name) is not None

def index_exists(conn, name):
    return scalar(conn, "SELECT 1 FROM pg_class WHERE relkind='i' AND relname=:n", n=name) is not None

def data_type(conn, table, col):
    return scalar(conn, """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t AND column_name=:c
    """, t=table, c=col)

def migrate():
    with engine.begin() as conn:
        # 1) Ensure plans table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS plans (
          id           BIGSERIAL PRIMARY KEY,
          user_id      BIGINT NOT NULL,
          dataset_id   BIGINT,
          goal_amount  NUMERIC(12,2) NOT NULL,
          period       VARCHAR(32) NOT NULL,
          start_date   DATE NOT NULL,
          end_date     DATE NOT NULL,
          risk         VARCHAR(16) NOT NULL,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """))

        # 2) Create actions if it doesn't exist (with the RIGHT schema)
        if not table_exists(conn, "actions"):
            conn.execute(text("""
            CREATE TABLE actions (
              id          BIGSERIAL PRIMARY KEY,
              plan_id     BIGINT REFERENCES plans(id) ON DELETE CASCADE,
              name        VARCHAR(128) NOT NULL,
              channel     VARCHAR(64) NOT NULL,
              cost        NUMERIC(10,4) NOT NULL DEFAULT 0,
              cooldown    INT,
              daily_cap   INT,
              provider    VARCHAR(128),
              active      BOOLEAN NOT NULL DEFAULT TRUE,
              sort_order  INT NOT NULL DEFAULT 0,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """))
        else:
            # 3) Upgrade existing actions table in-place
            if not col_exists(conn, "actions", "plan_id"):
                conn.execute(text("ALTER TABLE actions ADD COLUMN plan_id BIGINT"))
            if not col_exists(conn, "actions", "sort_order"):
                conn.execute(text("ALTER TABLE actions ADD COLUMN sort_order INT NOT NULL DEFAULT 0"))
            if not col_exists(conn, "actions", "active"):
                conn.execute(text("ALTER TABLE actions ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE"))

            # Ensure cost is NUMERIC(10,4)
            dt = data_type(conn, "actions", "cost")
            if dt and dt.lower() != "numeric":
                conn.execute(text("ALTER TABLE actions ALTER COLUMN cost TYPE numeric(10,4) USING cost::numeric"))

            # FK to plans(id)
            if not constraint_exists(conn, "actions_plan_fk"):
                conn.execute(text("""
                    ALTER TABLE actions
                    ADD CONSTRAINT actions_plan_fk
                    FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
                """))

        # 4) Unique + index (safe to run even if table was just created)
        if not constraint_exists(conn, "uq_actions_plan_name_channel"):
            # create only if columns exist
            if col_exists(conn, "actions", "plan_id") and col_exists(conn, "actions", "name") and col_exists(conn, "actions", "channel"):
                conn.execute(text("""
                    ALTER TABLE actions
                    ADD CONSTRAINT uq_actions_plan_name_channel
                    UNIQUE (plan_id, name, channel)
                """))

        if not index_exists(conn, "ix_actions_plan_id") and col_exists(conn, "actions", "plan_id"):
            conn.execute(text("CREATE INDEX ix_actions_plan_id ON actions (plan_id)"))

        # Helpful indexes on plans
        if not index_exists(conn, "ix_plans_user_id"):
            conn.execute(text("CREATE INDEX ix_plans_user_id ON plans (user_id)"))
        if not index_exists(conn, "ix_plans_dataset_id"):
            conn.execute(text("CREATE INDEX ix_plans_dataset_id ON plans (dataset_id)"))

        # 5) Print a quick schema summary
        rows = conn.execute(text("""
          SELECT table_name, column_name, data_type
          FROM information_schema.columns
          WHERE table_schema='public' AND table_name IN ('plans','actions')
          ORDER BY table_name, ordinal_position
        """)).all()
        print("\n== Schema ==")
        for t, c, d in rows:
            print(f"{t:8s} | {c:15s} | {d}")

if __name__ == "__main__":
    migrate()
    print("\n✅ Migration complete.")
