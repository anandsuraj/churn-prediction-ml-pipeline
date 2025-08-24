============================================================================
Streamlining Data Workflow with Apache Airflow on AWS EC2
============================================================================
1) create EC2  Ubuntu instance

2) connect to EC2 instance
ssh -i "airflow-key-pair.pem" ubuntu@ec2-54-86-92-182.compute-1.amazonaws.com

3) On AWS instance run following setup commands
	sudo yum update -y
	sudo yum install -y python3 python3-pip
	sudo pip3 install virtualenv
	python3 -m virtualenv ~/airflow_venv
	source ~/airflow_venv/bin/activate
	pip install pandas s3fs apache-airflow requests

4) On EC2 instance, enable inbound and outbound rules for ssh / http / https
	
5) Start the Airflow
	airflow standaloneSimple auth manager | Password for user 'admin': DAGFnyUUCQAFaPN3

	Login with username provided at time of starting airflow
        #admin  password: FRwDReZ7zrXEtYyx
		Simple auth manager | Password for user 'admin': DAGFnyUUCQAFaPN3


	airflow webserver create-user \
	--username admin \
	--firstname Suraj \
	--lastname Anand \
	--role Admin \
	--email surajanand.work@gmail.com \
	--password mysecurepassword

		airflow users create \
    --username admin \
    --firstname Suraj \
    --lastname Anand \
    --role Admin \
    --email surajanand.work@gmail.com \
    --password mysecurepassword


6) Run scheduler and webserver
	airflow scheduler
	airflow webserver --port 8080

7) Open Airflow web browser interface 
	<IP of AWS instance>:8080

8) Connect to Remote Host through VSCode 
	install the VSCode on local system
	install the Remote – SSH extension
	configure SSH on EC2 Instance
	ssh -i "C:\Users\BITS\Desktop\airflow\airflow-key-pair.pem" ubuntu@ec2-54-86-92-182.compute-1.amazonaws.com

9) create "dags" folder under "airflow" setup directory
10) Create dag file such as "my_dag.py" under dags directory 
11) Add the DAG snippet into this file

12) Run the DAG from the airflow web interface 
	look at the logs on the web page
	look at the logs on the VSCode

===========================================================
Demo : Read data using Polygon API and write to S3 bucket
===========================================================
13) Create free account on https://polygon.io/ and note the API key 
14) Create a S3 bucket "churn-data-lake"

15) Setting up permissions for EC2 to connect to S3 
	on EC2 instance ---> Actions ---> Security ---> Modify IAM role ---> Create A New IAM role
16) On IAM service, create a new IAM role by granting 
AmazonS3FullAccess 
AmazonEC2FullAccess 
permissions on EC2 instance
17) On EC2 instance, select new role and save
18) Access Token Generation for IAM user
	AWS Management Console ---> Profile --> My Security Credential ---> Access keys ---> Create Access Keys
	Note down the details
19) Write the new Dag file "market_etl.dag" 
	Modify the details like Polygon API key, AWS credentials and S3 bucket name
20) Run the DAG from the airflow web interface 
	look at the logs on the web page
	look at the logs on the VSCode
21) Explore the file added into the S3 bucket

===============================================================================================
References
===============================================================================================
Streamlining Data Workflow with Apache Airflow on AWS EC2
https://www.analyticsvidhya.com/blog/2024/04/streamlining-data-workflow-with-apache-airflow-on-aws-ec2/

Getting Started with Apache Airflow (dummy etl)
https://www.datacamp.com/tutorial/getting-started-with-apache-airflow

Building an ETL Pipeline with Airflow (stock data)
https://www.datacamp.com/tutorial/building-an-etl-pipeline-with-airflow?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720824&utm_adgroupid=157098107015&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=684592141145&utm_targetid=dsa-2264919292029&utm_loc_interest_ms=&utm_loc_physical_ms=9146371&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-row-p2_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&gad_source=1&gclid=Cj0KCQiA4L67BhDUARIsADWrl7G3wKy_XrZ1oQsTP4mhwp7RJ_03XCR2wRQVkhqrekfpyKQfFSsumAoaArmfEALw_wcB