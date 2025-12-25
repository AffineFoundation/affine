# Deterministic Profile-Based Configuration System

## Problem Description
The `affine-cortex` SDK previously lacked a centralized, robust configuration management system. Users faced several challenges:
1.  **Manual Configuration**: To change API endpoints or timeouts, users often had to modify code directly or pass arguments to every function call.
2.  **Lack of Environment Switching**: There was no standard way to switch between "development", "staging", and "production" environments locally or on servers.
3.  **Inconsistent Configuration**: Different tools (CLI vs SDK) might resolve configuration differently or rely on scattered environment variables.
4.  **No Persistence**: Setting a preference (like a preferred profile) wasn't persistent across sessions.

## Thought Process

To address these issues, we needed a system that is **deterministic**, **hierarchical**, and **user-friendly**.

### Design Decisions:
1.  **Singleton Pattern**: Since configuration is global to the process, a Singleton `ConfigManager` ensures that configuration is loaded once and accessible everywhere (`affine.config.config`).
2.  **Hierarchical Configuration**:
    *   **Project Level** (`./affine.json`): Highest priority (except CLI flags). Allows per-project settings.
    *   **Global Level** (`~/.affine/config.json`): User-level defaults.
    *   This follows the standard precedence rule: CLI > Project > Global > Defaults.
3.  **Profiles**: Instead of just flat keys, we group settings into "profiles". This allows a user to define a "local" profile pointing to `localhost:8000` and a "prod" profile pointing to the live API, and switch between them instantly.
4.  **Schema Validation**: We used `pydantic` models (`ProfileConfig`) to enforce type safety. If a user provides a string for a timeout, the system fails fast with a clear error rather than causing cryptic runtime bugs.
5.  **CLI & SDK Unification**:
    *   The CLI (`af`) was updated to expose configuration commands (`list`, `use`, `set`) and a global `--profile` flag.
    *   The SDK (`APIClient`) automatically reads from the active profile, so user code becomes cleaner (no more manually passing `base_url`).

## The Fix

### 1. New Configuration Module (`affine/config.py`)
We introduced `affine/config.py` which contains:
*   `ProfileConfig`: Pydantic model for profile validation.
*   `ConfigManager`: Singleton class that handles:
    *   Loading/merging JSON files.
    *   Managing the active profile state.
    *   Persisting changes (like `set_active_profile`).

### 2. SDK Integration (`affine/utils/api_client.py`)
The `APIClient` and `CLIAPIClient` were refactored to:
*   Import the global `config` object.
*   Automatically initialize `base_url`, `timeout`, and `retry` settings from the `active_profile`.
*   Implement retry logic that respects the profile's retry configuration.

### 3. CLI Enhancements (`affine/cli/main.py`)
We added a new `config` command group:
*   `af config list`: Shows available profiles.
*   `af config use <name>`: Switches the active profile globally.
*   `af config set <name> <key>=<value>`: Allows easy modification of profile settings.
*   Added global `--profile` flag to override the active profile for a single command.

### 4. Testing (`tests/test_profiles.py`)
A comprehensive test suite was added to verify:
*   Precedence rules (Project > Global).
*   Profile switching functionality.
*   Error handling for missing/invalid profiles.
