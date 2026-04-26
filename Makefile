.PHONY: build start serve stop health

# Docker 
build:
	@docker build -t emergency:latest -f Dockerfile .

start:
	@docker run -p 8000:8000 emergency:latest

stop:
	@docker ps -q --filter ancestor=emergency:latest | xargs -r docker stop

# Local dev (uv) 
serve:
	@uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool
