import sqlite3
from pathlib import Path
from .schema import create_schema

class SQLiteManager:
    def __init__(self, db_path="logiedge.db"):
        self.db_path = Path(db_path)
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        create_schema(self.conn)
        return self.conn

    def execute(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def executemany(self, sql, seq):
        cur = self.conn.executemany(sql, seq)
        self.conn.commit()
        return cur

    def close(self):
        if self.conn:
            self.conn.close()
