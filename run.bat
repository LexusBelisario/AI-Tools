start cmd /k "cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001 --env-file .env"
start cmd /k "cd frontend && npm run dev"

