#!/usr/bin/env python3
"""
Pipeline Scheduler
==================

A cron-like scheduler for running data pipeline tasks.

Usage:
    python scheduler.py

Configuration:
    Edit the SCHEDULED_JOBS list below to add/modify scheduled tasks.
"""

from data_validation import DataValidator
from data_ingestion import DataIngestionPipeline
from raw_data_storage import RawDataStorage
import sys
import os
import schedule
import time
import logging
from datetime import datetime

# Add src directory to path
sys.path.append('src')

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
# Import pipeline modules

class PipelineScheduler:
    def __init__(self):
        self.jobs = []

    def add_job(self, job_name, job_function, schedule_type, schedule_time=None):
        """Add a job to the scheduler"""
        job_info = {
            'name': job_name,
            'function': job_function,
            'type': schedule_type,
            'time': schedule_time
        }
        self.jobs.append(job_info)

    def schedule_jobs(self):
        """Schedule all configured jobs"""
        for job in self.jobs:
            if job['type'] == 'hourly':
                schedule.every().hour.do(self._run_job, job)
                logging.info(f"Scheduled '{job['name']}' to run every hour")

            elif job['type'] == 'daily' and job['time']:
                schedule.every().day.at(job['time']).do(self._run_job, job)
                logging.info(
                    f"Scheduled '{job['name']}' to run daily at {job['time']}")

            elif job['type'] == 'weekly' and job['time']:
                schedule.every().week.do(self._run_job, job)
                logging.info(f"Scheduled '{job['name']}' to run weekly")

            elif job['type'] == 'minutes' and job['time']:
                schedule.every(job['time']).minutes.do(self._run_job, job)
                logging.info(
                    f"Scheduled '{job['name']}' to run every {job['time']} minutes")

    def _run_job(self, job):
        """Execute a scheduled job"""
        try:
            logging.info(f"Starting job: {job['name']}")
            result = job['function']()
            logging.info(f"Job '{job['name']}' completed successfully")
            return result
        except Exception as e:
            logging.error(f"Job '{job['name']}' failed: {str(e)}")

    def start(self):
        """Start the scheduler"""
        logging.info("Pipeline Scheduler starting...")
        logging.info(f"Total jobs scheduled: {len(self.jobs)}")

        # Run initial execution of all jobs
        logging.info("Running initial execution of all jobs...")
        for job in self.jobs:
            self._run_job(job)

        # Start scheduler loop
        logging.info("Scheduler loop started. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")

# Job Functions
def run_data_ingestion():
    """Run data ingestion pipeline"""
    pipeline = DataIngestionPipeline()
    return pipeline.run_ingestion()

def run_raw_data_storage():
    """Run raw data storage pipeline"""
    storage = RawDataStorage()
    return storage.create_data_catalog()

def run_data_validation():
    """Run data validation pipeline"""
    validator = DataValidator()
    return validator.run_validation()

def run_data_preparation():
    """Run data preparation (placeholder)"""
    logging.info("Data preparation job executed")
    return {'status': 'success', 'message': 'Data preparation completed'}


# SCHEDULED JOBS CONFIGURATION
# Add your jobs here with their schedule
SCHEDULED_JOBS = [
    {
        'name': 'Data Ingestion',
        'function': run_data_ingestion,
        'type': 'hourly',  # Options: 'hourly', 'daily', 'weekly', 'minutes'
        'time': None       # For daily: "09:00", for minutes: 30, for hourly/weekly: None
    },
    {
        'name': 'Raw Data Storage',
        'function': run_raw_data_storage,
        'type': 'hourly',
        'time': None
    },
    {
        'name': 'Data Validation',
        'function': run_data_validation,
        'type': 'hourly',
        'time': None # '10:00'
    },
    {
        'name': 'Data Preparation',
        'function': run_data_preparation,
        'type': 'minutes',
        'time': 30  # Every 30 minutes
    }
]


def main():
    """Main scheduler function"""
    print("Pipeline Scheduler")
    print("=" * 30)
    print("Configured Jobs:")

    scheduler = PipelineScheduler()

    # Add all configured jobs
    for job_config in SCHEDULED_JOBS:
        scheduler.add_job(
            job_config['name'],
            job_config['function'],
            job_config['type'],
            job_config['time']
        )

        # Display job info
        if job_config['type'] == 'hourly':
            print(f"  - {job_config['name']}: Every hour")
        elif job_config['type'] == 'daily':
            print(f"  - {job_config['name']}: Daily at {job_config['time']}")
        elif job_config['type'] == 'weekly':
            print(f"  - {job_config['name']}: Weekly")
        elif job_config['type'] == 'minutes':
            print(
                f"  - {job_config['name']}: Every {job_config['time']} minutes")

    print("=" * 30)

    # Schedule and start
    scheduler.schedule_jobs()
    scheduler.start()

if __name__ == "__main__":
    main()
