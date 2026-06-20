.PHONY: help install count evaluate plot clean

CONFIG ?=
RUN ?=

help:
	@echo "Targets disponibles:"
	@echo "  make install                   Instala dependencias con uv"
	@echo "  make count    CONFIG=ruta.yaml Cuenta usando el YAML indicado"
	@echo "  make evaluate CONFIG=ruta.yaml Cuenta y compara contra ground truth"
	@echo "  make plot     RUN=ruta/run     Genera la imagen por trial de un run"
	@echo "  make clean                     Borra el directorio runs/"
	@echo ""
	@echo "Ejemplos:"
	@echo "  make count    CONFIG=experiments/example.yaml"
	@echo "  make evaluate CONFIG=experiments/example.yaml"
	@echo "  make plot     RUN=runs/example/20260520_201945"

install:
	uv sync

count:
ifndef CONFIG
	$(error CONFIG es obligatorio. Uso: make count CONFIG=experiments/<archivo>.yaml)
endif
	uv run python -m ops run $(CONFIG)

evaluate:
ifndef CONFIG
	$(error CONFIG es obligatorio. Uso: make evaluate CONFIG=experiments/<archivo>.yaml)
endif
	uv run python -m ops run $(CONFIG)

plot:
ifndef RUN
	$(error RUN es obligatorio. Uso: make plot RUN=runs/<name>/<timestamp>)
endif
	uv run python -m ops plot $(RUN)

clean:
	rm -rf runs/
