build:
	@docker build -t emergency:latest -f Dockerfile .

start:
	@docker run -p 7860:7860 emergency:latest
