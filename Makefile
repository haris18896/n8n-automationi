SHELL := /bin/zsh

.PHONY: up down restart logs ps pull build config

up:
	docker compose up -d --build

down:
	docker compose down

restart:
	docker compose down && docker compose up -d --build

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

pull:
	docker compose pull

build:
	docker compose build --no-cache

config:
	docker compose config
