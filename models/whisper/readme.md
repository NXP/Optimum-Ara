# Whisper Model for Ara Hardware

This directory contains the configuration files for the `Whisper` model optimized for Ara hardware.

## Model Information

- **Model**: openai/whisper-medium
- **Architecture**: encoder–decoder
- **Hardware**: Ara DVM

## Files

- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `readme.md` - This file

## Usage

An example Whisper usage script is provided at [here](../../examples/whisper.py).

## Configuration

The model uses the same single-model architecture as llama2 & llama3:

## Hardware Requirements

- Ara hardware with DVM support
- Whisper model DVM files

## Input Requirements

- Audio inputs allowed be upto 30 seconds long.
