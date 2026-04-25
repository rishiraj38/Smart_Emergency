build:
	@docker build -t emergency:latest -f Dockerfile .

start:
	@docker run -p 8000:8000 emergency:latest

venv: 
	@source .venv/bin/activate
