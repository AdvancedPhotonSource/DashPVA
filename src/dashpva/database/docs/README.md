MODELS
detectors: detector name, pv, desc
profiles: the manager of your program
profile_config: pvs
system: system variables, filepath, etc

# Profile
 a profile is a configuration setting


# Profile config
plkace to put your pv'ds and pv values
can nest

# Detectors

# System Variables
place to put system variables
store paths
required: LOGS, CONFIG, OUTPUT_PATH,  

## Database Tables

The DashPVA database consists of several tables that store configuration, system information, and user profiles. Below is a comprehensive list of all tables and their columns:

# PROFILE & PV CONFIGURATION MANAGEMENT
### profiles
Stores user configuration profiles for different experimental setups.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier for the profile (auto-increment) |
| name | VARCHAR(255) NOT NULL UNIQUE | Human-readable name of the profile |
| description | TEXT | Optional description of the profile's purpose |
| selected | INTEGER DEFAULT 0 | Whether this profile is currently selected (0=no, 1=yes) |
| default | INTEGER DEFAULT 0 | Whether this is the default profile (0=no, 1=yes) |
| created_at | DATETIME | When the profile was created (UTC timestamp) |
| updated_at | DATETIME | When the profile was last modified (UTC timestamp) |

### profile_configs
Stores TOML configuration data associated with profiles.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier for the config entry (auto-increment) |
| profile_id | INTEGER NOT NULL | Foreign key to profiles table |
| config_type | VARCHAR(50) NOT NULL | Type of config (e.g., 'root', 'CACHE_OPTIONS', 'METADATA') |
| config_section | VARCHAR(100) | Config section name (e.g., 'ALIGNMENT', 'CA', 'ROI1') |
| config_key | VARCHAR(100) NOT NULL | Configuration key (e.g., 'DETECTOR_PREFIX', 'MAX_CACHE_SIZE') |
| config_value | TEXT | The actual configuration value as text |
| created_at | DATETIME | When the config was created (UTC timestamp) |
| updated_at | DATETIME | When the config was last modified (UTC timestamp) |


# SETTINGS RELATED TABLES
### settings
Stores hierarchical configuration settings with support for nested organization.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier for the setting (auto-increment) |
| name | VARCHAR(255) NOT NULL | The setting name/identifier |
| type | VARCHAR(100) NOT NULL | The type of setting (e.g., 'int', 'str', 'list') |
| desc | TEXT | Optional description of what this setting controls, only applies to sections
| parent_id | INTEGER | Foreign key to parent setting for hierarchical organization |

### setting_values
Stores key-value pairs associated with settings.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier for the setting value (auto-increment) |
| setting_id | INTEGER NOT NULL | Foreign key to the settings table |
| key | VARCHAR(255) NOT NULL | The key name for this value |
| value | TEXT NOT NULL | The actual value stored as text (converted from original type) |
| value_type | VARCHAR(20) NOT NULL | Type indicator for the value ('string' or 'int') |

### system (removed)
System table has been removed. Use the hierarchical `settings` and `setting_values` tables to store system-wide configuration and paths.

# DETECTORS RELATED TABLES
### detectors
Stores detector configuration and metadata.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Unique identifier for the detector (auto-increment) |
| name | VARCHAR(255) NOT NULL UNIQUE | Name of the detector |
| description | TEXT | Description of the detector |

### Foreign Key Relationships

- `profile_configs.profile_id` → `profiles.id`
- `settings.parent_id` → `settings.id` (self-referencing for hierarchical settings)
- `setting_values.setting_id` → `settings.id`

### Indexes

The database includes indexes on frequently queried columns:
- `profiles.name` for fast profile lookups (unique constraint)
- `profile_configs.profile_id` for profile-based config queries
- `settings.name` for fast setting lookups
- `settings.parent_id` for hierarchical queries
- `setting_values.setting_id` for value retrieval
- `setting_values.key` for key-based lookups
- `system.name` for system setting lookups (unique constraint)
- `detectors.name` for detector lookups (unique constraint)

# Paths
paths setting has required paths structure
APP_META(root)-> Application Metadata
PATHS(root) -> LOG -> logs/
            -> CONFIGS -> configs/
            -> OUTPUTS -> outputs/
                -> SCAN -> scans/
                -> SLICES -> slices/
            -> CONSUMERS -> consumers/
                -> IOC -> ioc/
                
                


# Getting Started
Want to stop using your toml file and use something simpler great

It's super easy just Run the PVA Workflow Setup -> Configuration -> Import Toml and that's it

