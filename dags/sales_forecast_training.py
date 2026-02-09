from datetime import datetime, timedelta
from airflow.decorators import dag, task
import logging


import pandas as pd
import sys


sys.path.append("/usr/local/airflow/include")
from utils.data_generate import RealisticSalesDataGenerator

logger = logging.getLogger(__name__)

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

        data_output_dir = 'tmp/sales_data'
        generator = RealisticSalesDataGenerator(
            start_date='2011-01-01',
            end_date='2026-02-03'
        )

        print("Generating realistic sales data")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files}")

        for data_type, paths in file_paths.items():
            print(f"{data_type}:{len(paths)} files")

        return {
            'data_output_dir': data_output_dir,
            'file_paths': file_paths,
            "total_files": total_files
        }
    

    @task()
    def validate_data_tast(extact_results):
        file_paths = extract_results['file_paths']
        total_rows = 0
        issues_found = []

        logger.info(f'Validating {len(file_paths["Sales"])} sales files ....')
        for i, sales_file in enumerate(file_paths['sales_file'][:10]):
            df = pd.read_parquet(sales_file)
            if i == 0:
                logger.info(f"sales data columns: {df.columns.tolist()}")
            
            if df.empty:
                issues_found.append(f"Sales file {sales_file} is empty.")
                continue

            required_col = [
                'date',
                'store_id',
                'prouct_id',
                'quantity_sold',
                'revenue'
            ]

            missing_cols = set(required_col) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Sales file {sales_file} is missing columns: {missing_cols}")
                continue

            total_rows += len(df)
            if df['quantity_sold'].min() < 0:
                issues_found.append(f"Sales file {sales_file} has negative quantity_sold.")

            if df['revenue'].min() < 0:
                issues_found.append(f"Sales file {sales_file} has negative revenue.")

            for data_type in ['promotions',
                              'customer_traffic',
                              'store_events']:
                if data_type in file_paths and file_paths[data_type]:
                    sample_file = file_paths[data_type][0]
                    df = pd.read_parquet(sample_file)
                    logger.info(f"{data_type} data shape: {df.shape}")
                    logger.info(f"{data_type} data columns: {df.columns.tolist()}")

            validation_sumamary = {
                "total_files_validated": len(file_paths['sales_file']),
                "total_rows_validated": total_rows, 
                "issues_found": issues_found,
                "issued_count": len(issues_found),
                'file_paths': file_paths
            }

            if issues_found:
                logger.error(f"Validation summary: {validation_sumamary}")
                for issues in issues_found:
                    logger.error(issues)
                raise Exception("Validation issues found.")
            else:
                logger.info(f"Validation summary: {validation_sumamary}")
                
            return validation_sumamary
    @task()
    def train_models_task(validation_summary):
        file_paths = validation_summary['file_paths']
        logger.info(f"Training models...")
        sales_df = []
        max_files = 50 
        for i, sales_file in enumerate(file_paths['sales'][:max_files]):
            df = pd.read_parquet(sales_file)
            sales_df.append(df)
            if (i+1)%10 ==0:
                logger.info(f"Loaded {i+1} sales files")

        sales_df = pd.concat(sales_df, ignore_index=True) 
        print(f"Comnined sales data shape : {sales_df.shape}")

        daily_sales = (
            sales_df.groupby(['date','store_id','product_id','category'])
            .agg({'quantity_sold':'sum',
                  'revenue':'sum',
                  'cost':'sum',
                  'profit':'sum',
                  'discount_percent':'mean',
                  'unit_price':'mean'})
            .reset_index()
        )

        daily_sales = daily_sales.rename(columns={'revenue':'sales'})
        if file_paths.get('promotions'):
            promo_df = pd.read_parquet(file_paths['promotions'][0])
            promo_summary = (
                promo_df.groupby(
                    ['date','product_id'])['discounte_percent']
                    .max()
                    .reset_index()
                )
            promo_summary['has_promotion'] = 1  

            daily_sales = daily_sales.merge(
                promo_summary[['date','product_id','has_promotion']],
                on=['date','product_id'],
                how='left'
            )   
            daily_sales['has_promotion'] = daily_sales['has_promotion'].fillna(0).astype(int)  


        if file_paths.get('customer_traffic'):
            traffic_dfs =[]
            for traffic_file in file_paths['customer_traffic'][:10]:
                traffic_dfs.append(pd.read_parquet(traffic_file))
            traffic_df = pd.concat(traffic_dfs,ignore_index=True)
            traffic_summary = (
                traffic_df.groupby(['date','store_id'])
                .agg({'customer_traffic':'sum',
                      'is_holiday':'max'})
                .reset_index()
            )
            daily_sales = daily_sales.merge(
                traffic_summary,
                on = ['date','store_id'],
                how = 'left'
            )

        logger.info(f'Final training data: {daily_sales.shape}')
        logger.info(f'Colums: {daily_sales.colums.tolist()}')

        trainer = ModelTrainer()

        store_daily_sales = (
            daily_sales.group(['store_id','store_id'])
            .agg({'sales':'sum',
                  'has_promotion':'mean',
                  'quantity_sold':'sum',
                  'profit':'sum',
                  'customer_traffic':'first',
                  'is_holiday':'first'})
        )
        store_daily_sales['date'] = pd.to_datetime(store_daily_sales['date'])

        

    extract_results = extract_data_task()
    validation_summary = validate_data_tast(extract_results)

sales_forecast_training_dag = sales_forecast_training()


