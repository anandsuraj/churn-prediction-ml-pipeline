# Docker Setup

## Prerequisites
- Docker (Desktop or Engine)
- Docker Compose v2

## 1) Prepare environment (optional for local S3)
Create `.env` in project root if you want to use S3:

```
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=your_bucket
STORAGE_TYPE=cloud   # or leave unset for local
```

If you just want local storage, skip credentials and either omit STORAGE_TYPE or set `STORAGE_TYPE=local`.

## 2) Build image
```
docker compose build
```

## 3) Run pipeline once
```
docker compose up --remove-orphans --abort-on-container-exit
```
Logs will appear in your terminal. Outputs are mounted to host:
- `data/` (raw data and catalog)
- `logs/`
- `reports/`

## Common commands
- Rebuild after code changes:
```
docker compose build --no-cache
```
- Run container in background:
```
docker compose up -d
```
- View logs:
```
docker compose logs -f
```
- Stop and remove:
```
docker compose down -v
```

## Notes
- The container runs `python main_pipeline.py` by default.
- For S3, ensure the bucket exists and your IAM user has write access.
- Data, logs, and reports are persisted to your local folders via bind mounts.
