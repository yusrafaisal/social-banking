uvicorn app:app --host 0.0.0.0 --port $PORT &
uvicorn chat_api:app --host 0.0.0.0 --port 8080 &
wait