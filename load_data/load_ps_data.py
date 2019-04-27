import sqlalchemy
import pandas as pd


def create_engine(
        user_name: str,
        password: str,
        host: str,
        port: int,
        db: str
        ) -> sqlalchemy.engine.Engine:
    """Create a PostgreSQL engine."""
    url = "postgresql://{}:{}@{}:{}/{}".format(user_name, password, host, port, db)
    engine = sqlalchemy.create_engine(url)
    return engine


def load_query(
        con,
        dataid,
        res,
        col=None,
        date_limits=None
        ):
    assert res in [1, 15, 60]
    if res == 1:
        table_name = "electricity_egauge_minutes"
        time_name = "localminute"
    elif res == 15:
        table_name = "electricity_egauge_15min"
        time_name = "local_15min"
    elif res == 60:
        table_name = "electricity_egauge_hours"
        time_name = "localhour"

    if col is None or not col:
        selection = "*"
    else:
        selection = ','.join([time_name] + col)

    query = [
        "SELECT", selection,
        "FROM", "university.{}".format(table_name),
        "WHERE",
        "dataid={}".format(str(dataid))
    ]
    if date_limits:
        query += [
            "AND", time_name,
            "BETWEEN", "'{}'".format(date_limits[0]), "and", "'{}'".format(date_limits[1])
        ]

    query = " ".join(query)
    df = pd.read_sql_query(query, con=con)
    df.rename(index=str, columns={time_name: "localtime"}, inplace=True)
    return df

