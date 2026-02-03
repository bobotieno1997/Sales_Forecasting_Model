from datetime import datetime, timedelta
from airflow.decorators import dag, task


import pandas as pd
import sys


sys.path.append("/usr/local/airflow/include")


default_args = {
    'owner':'Bob Okech O.',
    'depends_on_past': False,
    'start_date' : datetime(year=2025, month=8, day=9),
    'retires':1,
    'retry_delay':timedelta(minutes=1),
    'catchup':False,
    'schedule':'@daily'
}


@dag(
    default_args=default_args,
    description = 'Sales Forect Training DAG',
    tags=['ml','training','sales_forecast','sales'],
)

def sales_forecast_training():
    @task()
    def extract_data_task():
        from utils.data_generate import RealisticSalesDataGenerator

        data_output_dir = 'tmp/sales_data'
        generator = RealisticSalesDataGenerator(
            start_date='2011-01-01',
            end_date='2026-02-03'
        )

        print("Generating realistic sales data")