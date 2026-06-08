# nca-digits — convenience targets.
# Override any variable: `make train STEPS=30000`, `make gen DIGIT=7`,
# `make train-robust RESUME=1`, `make eval WEIGHTS=weights_robust.bin`.

CARGO   ?= cargo
RUN     := $(CARGO) run --release --
STEPS   ?= 10000
ITERS   ?= 8000
DIGIT   ?= 3
WEIGHTS ?= weights.bin
RESUME  ?=
RESUME_ARG := $(if $(RESUME),resume,)

.DEFAULT_GOAL := help

.PHONY: help build test check train train-robust overfit-all overfit \
        overfit-solid gen eval viz clean

help: ## Show this help
	@echo "nca-digits — targets (override vars, e.g. make train STEPS=30000):"
	@grep -E '^[a-zA-Z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "vars: STEPS=$(STEPS) ITERS=$(ITERS) DIGIT=$(DIGIT) WEIGHTS=$(WEIGHTS) RESUME=$(RESUME)"

build: ## Compile (release)
	$(CARGO) build --release

test: ## Gradient-check test (verifies the hand-written BPTT)
	$(CARGO) test --release

check: ## Run the gradient check directly (prints worst rel error)
	$(RUN) check

train: ## Pool trainer, truncated BPTT -> weights.bin  (robust/hazy)
	$(RUN) train $(STEPS) $(RESUME_ARG)

train-robust: ## Full BPTT + pool + damage aug -> weights_robust.bin  (crisp + heals)
	$(RUN) train-robust $(STEPS) $(RESUME_ARG)

overfit-all: ## All 10 digits, full BPTT, no pool -> weights_all.bin  (sharpest)
	$(RUN) overfit-all $(ITERS)

overfit: ## Single-digit overfit (debug): make overfit DIGIT=3 ITERS=3000
	$(RUN) overfit $(ITERS) $(DIGIT)

overfit-solid: ## Solid-target overfit (loop-correctness discriminator)
	$(RUN) overfit-solid $(ITERS)

gen: ## Grow one digit and render: make gen DIGIT=7 WEIGHTS=weights.bin
	$(RUN) gen $(DIGIT) $(WEIGHTS)

eval: ## Score a checkpoint: make eval WEIGHTS=weights_robust.bin
	$(RUN) eval $(WEIGHTS)

viz: ## Visualizer window (macroquad)
	$(CARGO) run --release --features viz --bin viz

clean: ## cargo clean + remove all weights*.bin
	$(CARGO) clean
	rm -f weights*.bin
