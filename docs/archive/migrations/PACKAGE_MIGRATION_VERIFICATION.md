# Package Migration Verification

## Root `neuros/` Folder Content Distribution

### Mapping of Root `neuros/` to Packages

| Root neuros/ Folder | Target Package | Status |
|---------------------|----------------|--------|
| `agents/` | neuros-core | ✓ Migrated |
| `alignment.py` | neuros-core | ✓ Migrated |
| `annotation/` | neuros-cloud | ✓ Migrated |
| `api/` | neuros-ui | ✓ Migrated |
| `augmentation.py` | neuros-core | ✓ Migrated |
| `autoconfig.py` | neuros-core | ✓ Migrated |
| `benchmarks/` | neuros-core | ✓ Migrated |
| `cli.py` | neuros (CLI package) | ✓ Migrated |
| `cloud/` | neuros-cloud | ✓ Migrated |
| `dashboard.py` | neuros-ui | ✓ Migrated |
| `datasets/` | neuros-core | ✓ Migrated |
| `db/` | neuros-cloud | ✓ Migrated |
| `drivers/` | neuros-drivers | ✓ Migrated |
| `etl/` | neuros-cloud | ✓ Migrated |
| `evaluation.py` | neuros-core | ✓ Migrated |
| `export/` | neuros-cloud | ✓ Migrated |
| `federated/` | neuros-cloud | ✓ Migrated |
| `foundation_models/` | neuros-foundation | ✓ Migrated |
| `ingest/` | neuros-cloud | ✓ Migrated |
| `io/` | neuros-core | ✓ Migrated |
| `models/` | neuros-models | ✓ Migrated |
| `pipeline.py` | neuros-core | ✓ Migrated |
| `plugins/` | neuros-core | ✓ Migrated |
| `processing/` | neuros-core | ✓ Migrated |
| `security.py` | neuros-core | ✓ Migrated |
| `serve/` | neuros-ui | ✓ Migrated |
| `sync/` | neuros-cloud | ✓ Migrated |
| `training/` | neuros-foundation | ✓ Migrated |
| `utils/` | neuros-core | ✓ Migrated |
| `visualization/` | neuros-ui | ✓ Migrated |

---

## Verification Commands

### Check neuros-core
```bash
ls packages/neuros-core/src/neuros/
# Should contain: agents/, alignment.py, augmentation.py, autoconfig.py,
#                 benchmarks/, core/, evaluation.py, pipeline.py,
#                 plugins/, processing/, security.py
```

### Check neuros-cloud
```bash
ls packages/neuros-cloud/src/neuros/
# Should contain: annotation/, cloud/, db/, etl/, export/, federated/,
#                 ingest/, sync/
```

### Check neuros-drivers
```bash
ls packages/neuros-drivers/src/neuros/
# Should contain: drivers/
```

### Check neuros-foundation
```bash
ls packages/neuros-foundation/src/neuros/
# Should contain: foundation_models/, training/
```

### Check neuros-models
```bash
ls packages/neuros-models/src/neuros/
# Should contain: models/
```

### Check neuros-ui
```bash
ls packages/neuros-ui/src/neuros/
# Should contain: api/, dashboard.py, serve/, visualization/
```

---

## Safe to Delete

Once verified that all content exists in packages, the following can be safely deleted:

1. **Root `neuros/` folder** - All content migrated to packages
2. **`neuros.egg-info/`** - Old egg-info, no longer needed

---

## Migration Complete Checklist

- [x] neuros-core contains: agents, alignment, augmentation, autoconfig, benchmarks, evaluation, pipeline, plugins, processing, security
- [x] neuros-cloud contains: annotation, cloud, db, etl, export, federated, ingest, sync
- [x] neuros-drivers contains: drivers
- [x] neuros-foundation contains: foundation_models, training
- [x] neuros-models contains: models
- [x] neuros-ui contains: api, dashboard, serve, visualization
- [ ] All imports tested and working
- [ ] All tests passing
- [ ] Root neuros/ folder can be deleted
