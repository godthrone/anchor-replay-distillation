# Anchor Replay Distillation Agent Notes

## Strict API Requirement

Do not set `max_tokens` or provider-equivalent token caps on large-model API calls. Let the model/API use its default output budget unless the user explicitly overrides this requirement.

## Repository Scope

This repository contains the standalone Anchor Replay Distillation (ARD) data generation and ARD-SFT utilities extracted from GRASPO. Do not copy local API env files, generated logs, model outputs, `.venv`, or cache directories into commits.