SHELL := /bin/zsh

.PHONY: help up down restart logs ps build rebuild pull config init buckets

help:
	@echo "make up       Start services"
	@echo "make down     Stop services"
	@echo "make restart  Restart services"
	@echo "make logs     Tail logs"
	@echo "make ps       Show status"
	@echo "make build    Build custom services"
	@echo "make rebuild  Rebuild without cache"
	@echo "make pull     Pull latest images"
	@echo "make config   Show docker compose config"
	@echo "make init     Start services and initialize MinIO buckets"
	@echo "make buckets  Create MinIO buckets"

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose down && docker compose up -d

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

build:
	docker compose build

rebuild:
	docker compose build --no-cache

pull:
	docker compose pull

config:
	docker compose config

init:
	cp -n .env.example .env || true
	docker compose up -d --build
	@sleep 10
	@bash ./scripts/init-minio.sh

buckets:
	@bash ./scripts/init-minio.sh
