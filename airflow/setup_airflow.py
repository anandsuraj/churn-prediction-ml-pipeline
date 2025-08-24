#!/usr/bin/env python3
"""
Airflow setup script for initializing the database and creating admin user
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and handle errors"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def main():
    """Initialize Airflow"""
    print("Setting up Airflow...")

    # Set environment variables
    os.environ['AIRFLOW_HOME'] = '/app/airflow'
    os.environ['PYTHONPATH'] = '/app'

    # Initialize Airflow database
    print("Initializing Airflow database...")
    run_command("airflow db init")

    # Create admin user
    print("Creating admin user...")
    username = os.getenv('AIRFLOW_WWW_USER_USERNAME', 'admin')
    password = os.getenv('AIRFLOW_WWW_USER_PASSWORD', 'admin')

    # Check if user already exists
    result = run_command(f"airflow users list | grep {username}", check=False)
    if result.returncode != 0:
        run_command(f"""airflow users create \
            --username {username} \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email admin@example.com \
            --password {password}""")
        print(f"Admin user '{username}' created successfully!")
    else:
        print(f"Admin user '{username}' already exists.")

    print("Airflow setup completed successfully!")


if __name__ == "__main__":
    main()
